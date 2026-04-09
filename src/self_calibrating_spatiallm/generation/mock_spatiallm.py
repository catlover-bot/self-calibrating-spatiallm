"""Mock SpatialLM wrapper for early pipeline integration."""

from __future__ import annotations

import math
from typing import Any

from self_calibrating_spatiallm.artifacts import (
    CalibratedPointCloud,
    Point3D,
    SceneObject,
    ScenePrediction,
    SceneRelation,
)
from self_calibrating_spatiallm.generation.base import SceneGenerator
from self_calibrating_spatiallm.geometry import centroid


class MockSpatialLMGenerator(SceneGenerator):
    """Deterministic placeholder with calibration-sensitive scene emission."""

    name = "mock_spatiallm"

    def __init__(self) -> None:
        self._last_execution_info: dict[str, Any] | None = None

    def generate(self, calibrated: CalibratedPointCloud) -> ScenePrediction:
        center = centroid(calibrated.points)
        stats = calibrated.metadata.get("transformed_point_cloud", {})
        ranges = stats.get("ranges", [5.0, 5.0, 3.0]) if isinstance(stats, dict) else [5.0, 5.0, 3.0]

        room_x = max(float(ranges[0]), 3.0)
        room_y = max(float(ranges[1]), 3.0)
        room_z = max(float(ranges[2]) if len(ranges) > 2 else 2.4, 2.2)

        propagation = _build_propagation_profile(calibrated, room_x=room_x, room_y=room_y, room_z=room_z)
        mode = propagation["generator_mode"]
        include_box = bool(propagation["include_box"])
        layout_changes = list(propagation["layout_changes"])
        horizontal_x = float(propagation["horizontal_basis"]["x"])
        horizontal_y = float(propagation["horizontal_basis"]["y"])
        side_x = float(propagation["side_basis"]["x"])
        side_y = float(propagation["side_basis"]["y"])

        floor = SceneObject(
            object_id="obj_floor",
            label="floor",
            position=Point3D(x=center.x, y=center.y, z=0.0),
            size=Point3D(x=room_x, y=room_y, z=0.1),
            confidence=0.99,
        )
        wall = SceneObject(
            object_id="obj_wall_north",
            label="wall",
            position=Point3D(
                x=center.x + (side_x * (room_y / 2.0)),
                y=center.y + (side_y * (room_y / 2.0)),
                z=(room_z / 2.0),
            ),
            size=Point3D(x=room_x, y=0.1, z=2.4),
            confidence=0.90 if mode == "calibration_informed" else 0.80,
        )
        table = SceneObject(
            object_id="obj_table",
            label="table",
            position=Point3D(x=center.x, y=center.y, z=0.8),
            size=Point3D(x=1.2, y=0.8, z=0.75),
            confidence=0.90 if mode == "calibration_informed" else 0.82,
        )
        mug = SceneObject(
            object_id="obj_mug",
            label="mug",
            position=Point3D(
                x=center.x + (0.22 * horizontal_x) + (0.08 * side_x),
                y=center.y + (0.22 * horizontal_y) + (0.08 * side_y),
                z=1.02,
            ),
            size=Point3D(x=0.09, y=0.09, z=0.11),
            confidence=0.73,
            attributes={"movable": True, "facing_target_id": "obj_wall_north"},
        )
        door_x = center.x + (side_x * ((room_y / 2.0) - 0.15)) + (horizontal_x * ((room_x / 2.0) - 0.35))
        door_y = center.y + (side_y * ((room_y / 2.0) - 0.15)) + (horizontal_y * ((room_x / 2.0) - 0.35))
        door = SceneObject(
            object_id="obj_door",
            label="door",
            position=Point3D(
                x=door_x if mode == "calibration_informed" else (center.x + (room_x / 2.0) - 0.1),
                y=door_y if mode == "calibration_informed" else (center.y + (room_y / 2.0) - 0.2),
                z=1.0,
            ),
            size=Point3D(x=0.9, y=0.2, z=2.0),
            confidence=0.78 if mode == "calibration_informed" else 0.60,
        )
        objects: list[SceneObject] = [floor, wall, table, mug, door]
        relations: list[SceneRelation] = [
            SceneRelation(subject_id="obj_mug", predicate="supported-by", object_id="obj_table", score=0.91),
        ]

        if mode == "calibration_informed":
            relations.extend(
                [
                    SceneRelation(subject_id="obj_table", predicate="supported-by", object_id="obj_floor", score=0.85),
                    SceneRelation(subject_id="obj_door", predicate="attached-to", object_id="obj_wall_north", score=0.81),
                    SceneRelation(subject_id="obj_door", predicate="supported-by", object_id="obj_floor", score=0.78),
                ]
            )
            layout_changes.append("added_structural_support_relations")

        if include_box:
            if mode == "calibration_informed":
                box = SceneObject(
                    object_id="obj_box",
                    label="box",
                    position=Point3D(
                        x=center.x - (0.30 * horizontal_x) - (0.22 * side_x),
                        y=center.y - (0.30 * horizontal_y) - (0.22 * side_y),
                        z=0.10,
                    ),
                    size=Point3D(x=0.35, y=0.25, z=0.20),
                    confidence=0.70,
                )
                relations.append(
                    SceneRelation(subject_id="obj_box", predicate="supported-by", object_id="obj_floor", score=0.68)
                )
                layout_changes.append("included_plausible_box_from_scale_hint")
            else:
                box = SceneObject(
                    object_id="obj_box",
                    label="box",
                    position=Point3D(x=center.x - 0.3, y=center.y - 0.2, z=-0.05),
                    size=Point3D(x=0.02, y=5.5, z=0.01),
                    confidence=0.64,
                )
                relations.append(
                    SceneRelation(
                        subject_id="obj_box",
                        predicate="supported-by",
                        object_id="missing_surface",
                        score=0.60,
                    )
                )
                layout_changes.append("retained_noisy_box_due_to_weak_calibration")
            objects.append(box)
        else:
            layout_changes.append("suppressed_uncertain_box_candidate")

        prediction = ScenePrediction(
            sample_id=calibrated.sample_id,
            generator_name=self.name,
            objects=objects,
            relations=relations,
            metadata={
                "calibration_method": calibrated.calibration.method,
                "calibration_confidence": calibrated.calibration.metadata.get("diagnostics", {}).get(
                    "confidence", {}
                ),
                "room_bounds": calibrated.metadata.get("room_bounds"),
                "is_placeholder": True,
                "propagation_diagnostics": propagation,
            },
        )
        self._last_execution_info = {
            "generator_mode": "mock",
            "generator_name": self.name,
            "success": True,
            "object_count": len(prediction.objects),
            "relation_count": len(prediction.relations),
            "propagation_mode": mode,
            "propagation_signal_strength": propagation["calibration_signal"]["signal_strength"],
            "layout_changes": layout_changes,
            "include_box": include_box,
        }
        return prediction

    def get_last_execution_info(self) -> dict[str, Any] | None:
        return self._last_execution_info


def _build_propagation_profile(
    calibrated: CalibratedPointCloud,
    *,
    room_x: float,
    room_y: float,
    room_z: float,
) -> dict[str, Any]:
    diagnostics = calibrated.calibration.metadata.get("diagnostics", {})
    if not isinstance(diagnostics, dict):
        diagnostics = {}
    confidence = diagnostics.get("confidence", {})
    if not isinstance(confidence, dict):
        confidence = {}
    execution = calibrated.calibration.metadata.get("execution", {})
    if not isinstance(execution, dict):
        execution = {}

    up_confidence = _safe_float(confidence.get("up_axis"), default=0.0)
    horizontal_confidence = _safe_float(confidence.get("horizontal_orientation"), default=0.0)
    reliability = _safe_float(
        confidence.get("overall_reliability"),
        default=((0.6 * up_confidence) + (0.4 * horizontal_confidence)),
    )
    manhattan_ambiguity = _safe_float(diagnostics.get("manhattan_ambiguity"), default=1.0)
    candidate_plane_count = int(
        _safe_float(
            diagnostics.get("candidate_plane_count"),
            default=(
                len(diagnostics.get("plane_candidates", []))
                if isinstance(diagnostics.get("plane_candidates"), list)
                else 0
            ),
        )
    )
    partial_calibration = bool(execution.get("partial_calibration_applied", False))
    fallback_used = bool(execution.get("fallback_used", False))

    signal_strength = max(
        0.0,
        min(
            1.0,
            (0.30 * up_confidence)
            + (0.35 * horizontal_confidence)
            + (0.25 * reliability)
            + (0.10 * (1.0 - manhattan_ambiguity)),
        ),
    )
    if partial_calibration:
        signal_strength *= 0.95
    if fallback_used:
        signal_strength *= 0.80

    generator_mode = "calibration_informed"
    if horizontal_confidence < 0.50 or manhattan_ambiguity > 0.82 or signal_strength < 0.60:
        generator_mode = "degraded_template"
    if fallback_used:
        generator_mode = "degraded_template"

    inferred_scale_hint = str(calibrated.metadata.get("inferred_scale_hint") or "")
    include_box = (inferred_scale_hint == "meter_scale_likely") or (generator_mode != "calibration_informed")

    horizontal_basis, side_basis = _horizontal_basis(calibrated)
    layout_changes: list[str] = []
    if generator_mode == "calibration_informed":
        layout_changes.append("enabled_calibration_informed_layout")
    else:
        layout_changes.append("using_degraded_template_layout")

    return {
        "generator_mode": generator_mode,
        "include_box": include_box,
        "layout_changes": layout_changes,
        "calibration_signal": {
            "up_confidence": up_confidence,
            "horizontal_confidence": horizontal_confidence,
            "overall_reliability": reliability,
            "manhattan_ambiguity": manhattan_ambiguity,
            "candidate_plane_count": candidate_plane_count,
            "signal_strength": signal_strength,
            "partial_calibration_applied": partial_calibration,
            "fallback_used": fallback_used,
        },
        "calibrated_input_summary": {
            "frame": calibrated.metadata.get("frame"),
            "num_points": calibrated.num_points,
            "ranges_xyz": [room_x, room_y, room_z],
            "normalization_applied": bool(calibrated.calibration.metadata.get("normalization_applied", False)),
            "inferred_scale_hint": inferred_scale_hint,
            "axis_convention": {
                "up_vector": calibrated.calibration.up_vector.to_dict(),
                "horizontal_axis": calibrated.calibration.horizontal_axis.to_dict(),
            },
        },
        "horizontal_basis": horizontal_basis,
        "side_basis": side_basis,
    }


def _horizontal_basis(calibrated: CalibratedPointCloud) -> tuple[dict[str, float], dict[str, float]]:
    axis = calibrated.calibration.horizontal_axis
    norm = math.hypot(axis.x, axis.y)
    if norm <= 1e-8:
        hx, hy = 1.0, 0.0
    else:
        hx, hy = axis.x / norm, axis.y / norm
    sx, sy = -hy, hx
    return {"x": float(hx), "y": float(hy)}, {"x": float(sx), "y": float(sy)}


def _safe_float(value: Any, *, default: float) -> float:
    try:
        parsed = float(value)
        if math.isfinite(parsed):
            return parsed
    except (TypeError, ValueError):
        pass
    return float(default)
