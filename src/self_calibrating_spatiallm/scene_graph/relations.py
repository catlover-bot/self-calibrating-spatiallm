"""Spatial relation derivation for actionable scene construction."""

from __future__ import annotations

from typing import Sequence

from self_calibrating_spatiallm.artifacts import SceneObject, SceneRelation
from self_calibrating_spatiallm.geometry import euclidean_distance, horizontal_distance, inside, intersects, supports


def derive_basic_relations(objects: Sequence[SceneObject], near_threshold: float = 1.0) -> list[SceneRelation]:
    """Derive geometric scene relations used by actionable scene construction."""
    derived: list[SceneRelation] = []
    seen: set[tuple[str, str, str]] = set()

    by_id = {obj.object_id: obj for obj in objects}
    walls = [obj for obj in objects if obj.label == "wall"]

    for index, first in enumerate(objects):
        for second in objects[index + 1 :]:
            distance = euclidean_distance(first.position, second.position)
            if distance <= near_threshold:
                _append_relation(derived, seen, first.object_id, "near", second.object_id, 0.6)
                _append_relation(derived, seen, second.object_id, "near", first.object_id, 0.6)

            if supports(first, second):
                _append_relation(derived, seen, first.object_id, "supported-by", second.object_id, 0.65)
            elif supports(second, first):
                _append_relation(derived, seen, second.object_id, "supported-by", first.object_id, 0.65)

            if inside(first, second):
                _append_relation(derived, seen, first.object_id, "inside", second.object_id, 0.55)
            if inside(second, first):
                _append_relation(derived, seen, second.object_id, "inside", first.object_id, 0.55)

            if intersects(first, second):
                _append_relation(derived, seen, first.object_id, "intersects", second.object_id, 0.5)
                _append_relation(derived, seen, second.object_id, "intersects", first.object_id, 0.5)

    for obj in objects:
        if _is_structural(obj):
            continue

        if 0.2 <= obj.position.z <= 1.7:
            _append_relation(derived, seen, obj.object_id, "accessible", "agent", 0.4)

        explicit_facing = obj.attributes.get("facing_target_id")
        if isinstance(explicit_facing, str) and explicit_facing in by_id:
            _append_relation(derived, seen, obj.object_id, "facing", explicit_facing, 0.45)
            continue

        if walls:
            nearest_wall = min(walls, key=lambda wall: horizontal_distance(obj.position, wall.position))
            _append_relation(derived, seen, obj.object_id, "facing", nearest_wall.object_id, 0.3)

    return derived


def _is_structural(obj: SceneObject) -> bool:
    return obj.label in {"floor", "wall", "ceiling", "room"}


def _append_relation(
    collection: list[SceneRelation],
    seen: set[tuple[str, str, str]],
    subject_id: str,
    predicate: str,
    object_id: str,
    score: float,
) -> None:
    key = (subject_id, predicate, object_id)
    if key in seen:
        return
    seen.add(key)
    collection.append(
        SceneRelation(
            subject_id=subject_id,
            predicate=predicate,
            object_id=object_id,
            score=score,
        )
    )
