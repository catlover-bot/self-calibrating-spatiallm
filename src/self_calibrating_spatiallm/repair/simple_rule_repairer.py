"""Simple rule-based scene repairer."""

from __future__ import annotations

from self_calibrating_spatiallm.artifacts import Point3D, RepairResult, ScenePrediction, SceneRelation
from self_calibrating_spatiallm.geometry import horizontal_distance, point_in_bounds
from self_calibrating_spatiallm.repair.base import SceneRepairer


class SimpleRuleRepairer(SceneRepairer):
    """Repairs common structural violations with lightweight geometric rules."""

    name = "simple_rule_repairer"

    def repair(self, scene: ScenePrediction) -> RepairResult:
        repaired = ScenePrediction.from_dict(scene.to_dict())
        issues: list[str] = []
        fixes: list[str] = []

        object_ids = {obj.object_id for obj in repaired.objects}
        floor_id = self._floor_id(repaired)
        wall_ids = [obj.object_id for obj in repaired.objects if obj.label == "wall"]

        self._fix_negative_and_out_of_bounds_positions(repaired, issues, fixes)
        self._fix_implausible_sizes(repaired, issues, fixes)

        valid_relations: list[SceneRelation] = []
        for rel in repaired.relations:
            if rel.subject_id not in object_ids or rel.object_id not in object_ids:
                issues.append(
                    "invalid relation: "
                    f"{rel.subject_id} {rel.predicate} {rel.object_id} references unknown object"
                )
                fixes.append(
                    "dropped invalid relation: "
                    f"{rel.subject_id} {rel.predicate} {rel.object_id}"
                )
                continue
            valid_relations.append(rel)
        repaired.relations = valid_relations

        if floor_id is not None:
            self._fix_floating_objects(repaired, floor_id, issues, fixes)

        if wall_ids:
            self._fix_door_window_attachment(repaired, wall_ids, issues, fixes)
        else:
            for obj in repaired.objects:
                if obj.label in {"door", "window"}:
                    issues.append(f"{obj.object_id}: no wall candidates for attachment consistency check")

        return RepairResult(
            sample_id=scene.sample_id,
            repairer_name=self.name,
            issues=issues,
            fixes_applied=fixes,
            repaired_scene=repaired,
            metadata={"is_placeholder": False},
        )

    def _fix_negative_and_out_of_bounds_positions(
        self,
        scene: ScenePrediction,
        issues: list[str],
        fixes: list[str],
    ) -> None:
        bounds = _parse_room_bounds(scene.metadata.get("room_bounds"))

        for obj in scene.objects:
            if obj.position.z < 0.0:
                issues.append(f"{obj.object_id}: negative height ({obj.position.z:.3f})")
                obj.position = Point3D(x=obj.position.x, y=obj.position.y, z=0.0)
                fixes.append(f"clamped {obj.object_id} to floor level")

            if bounds is None:
                continue

            bounds_min, bounds_max = bounds
            if point_in_bounds(obj.position, bounds_min, bounds_max):
                continue

            clamped = Point3D(
                x=min(max(obj.position.x, bounds_min.x), bounds_max.x),
                y=min(max(obj.position.y, bounds_min.y), bounds_max.y),
                z=min(max(obj.position.z, bounds_min.z), bounds_max.z),
            )
            issues.append(f"{obj.object_id}: outside room bounds")
            obj.position = clamped
            fixes.append(f"clamped {obj.object_id} into room bounds")

    def _fix_implausible_sizes(
        self,
        scene: ScenePrediction,
        issues: list[str],
        fixes: list[str],
    ) -> None:
        for obj in scene.objects:
            min_size = 0.05
            max_size = 10.0 if obj.label in {"floor", "wall", "ceiling"} else 3.5

            size_x = _clamp_size(obj.size.x, min_size, max_size)
            size_y = _clamp_size(obj.size.y, min_size, max_size)
            size_z = _clamp_size(obj.size.z, min_size, max_size)

            if (size_x, size_y, size_z) != (obj.size.x, obj.size.y, obj.size.z):
                issues.append(
                    f"{obj.object_id}: implausible size "
                    f"({obj.size.x:.2f}, {obj.size.y:.2f}, {obj.size.z:.2f})"
                )
                obj.size = Point3D(x=size_x, y=size_y, z=size_z)
                fixes.append(f"clamped {obj.object_id} size to plausible range")

    def _fix_floating_objects(
        self,
        scene: ScenePrediction,
        floor_id: str,
        issues: list[str],
        fixes: list[str],
    ) -> None:
        support_candidates = [obj for obj in scene.objects if obj.label in {"floor", "table", "counter", "desk"}]
        if not support_candidates:
            support_candidates = list(scene.objects)
        by_id = {obj.object_id: obj for obj in scene.objects}

        for obj in scene.objects:
            if obj.label in {"floor", "wall", "ceiling"}:
                continue

            support_rels = [
                rel
                for rel in scene.relations
                if rel.subject_id == obj.object_id and rel.predicate == "supported-by"
            ]

            has_valid_support = any(rel.object_id in by_id for rel in support_rels)
            if has_valid_support:
                continue

            if obj.position.z <= 0.25:
                target_id = floor_id
            else:
                nearest = min(
                    support_candidates,
                    key=lambda candidate: horizontal_distance(obj.position, candidate.position),
                )
                target_id = nearest.object_id
                issues.append(f"{obj.object_id}: floating object without support")

            scene.relations.append(
                SceneRelation(
                    subject_id=obj.object_id,
                    predicate="supported-by",
                    object_id=target_id,
                    score=0.55,
                )
            )
            fixes.append(f"added fallback support {obj.object_id} -> {target_id}")

    def _fix_door_window_attachment(
        self,
        scene: ScenePrediction,
        wall_ids: list[str],
        issues: list[str],
        fixes: list[str],
    ) -> None:
        objects_by_id = {obj.object_id: obj for obj in scene.objects}

        for obj in scene.objects:
            if obj.label not in {"door", "window"}:
                continue

            attached = [
                rel
                for rel in scene.relations
                if rel.subject_id == obj.object_id and rel.predicate == "attached-to" and rel.object_id in wall_ids
            ]
            if attached:
                continue

            nearest_wall = min(
                (objects_by_id[wall_id] for wall_id in wall_ids),
                key=lambda wall: horizontal_distance(obj.position, wall.position),
            )
            wall_distance = horizontal_distance(obj.position, nearest_wall.position)

            if wall_distance > 1.5:
                issues.append(f"{obj.object_id}: far from nearest wall (distance={wall_distance:.2f})")

            scene.relations.append(
                SceneRelation(
                    subject_id=obj.object_id,
                    predicate="attached-to",
                    object_id=nearest_wall.object_id,
                    score=0.5,
                )
            )
            fixes.append(f"added attachment relation {obj.object_id} -> {nearest_wall.object_id}")

    @staticmethod
    def _floor_id(scene: ScenePrediction) -> str | None:
        for obj in scene.objects:
            if obj.label == "floor":
                return obj.object_id
        return scene.objects[0].object_id if scene.objects else None


def _clamp_size(value: float, min_size: float, max_size: float) -> float:
    return float(min(max(value, min_size), max_size))


def _parse_room_bounds(raw: object) -> tuple[Point3D, Point3D] | None:
    if not isinstance(raw, dict):
        return None

    if "min" in raw and "max" in raw:
        raw_min = raw.get("min")
        raw_max = raw.get("max")
        if isinstance(raw_min, list) and isinstance(raw_max, list) and len(raw_min) == 3 and len(raw_max) == 3:
            return (
                Point3D(x=float(raw_min[0]), y=float(raw_min[1]), z=float(raw_min[2])),
                Point3D(x=float(raw_max[0]), y=float(raw_max[1]), z=float(raw_max[2])),
            )

    return None
