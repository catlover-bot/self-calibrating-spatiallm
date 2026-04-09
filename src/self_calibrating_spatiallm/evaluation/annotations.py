"""Lightweight per-scene annotation models and loaders."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


def _read_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object at {path}")
    return payload


@dataclass
class ExpectedRelation:
    """Lightweight expected relation label for qualitative/quantitative checks."""

    subject_category: str
    predicate: str
    object_category: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExpectedRelation":
        return cls(
            subject_category=str(data["subject_category"]),
            predicate=str(data["predicate"]),
            object_category=str(data["object_category"]),
        )


@dataclass
class TraversabilityLabel:
    """Expected accessibility/traversability label for a target category."""

    target_category: str
    should_be_accessible: bool
    question: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TraversabilityLabel":
        return cls(
            target_category=str(data["target_category"]),
            should_be_accessible=bool(data["should_be_accessible"]),
            question=(str(data["question"]) if data.get("question") else None),
        )


@dataclass
class SceneAnnotation:
    """Lightweight annotation schema for small-scale evaluation packs."""

    scene_id: str
    expected_up_axis: str | None = None
    expected_horizontal_axis: str | None = None
    room_orientation_hint: str | None = None
    expected_scale_hint: str | None = None
    expected_door_count: int | None = None
    expected_window_count: int | None = None
    expected_object_categories: list[str] = field(default_factory=list)
    expected_relations: list[ExpectedRelation] = field(default_factory=list)
    traversability_labels: list[TraversabilityLabel] = field(default_factory=list)
    notes: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save_json(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        return path

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SceneAnnotation":
        rel_raw = data.get("expected_relations", [])
        trav_raw = data.get("traversability_labels", [])
        if not isinstance(rel_raw, list):
            raise TypeError("expected_relations must be a list")
        if not isinstance(trav_raw, list):
            raise TypeError("traversability_labels must be a list")

        expected_categories_raw = data.get("expected_object_categories", [])
        if not isinstance(expected_categories_raw, list):
            raise TypeError("expected_object_categories must be a list")

        return cls(
            scene_id=str(data["scene_id"]),
            expected_up_axis=(str(data["expected_up_axis"]) if data.get("expected_up_axis") else None),
            expected_horizontal_axis=(
                str(data["expected_horizontal_axis"]) if data.get("expected_horizontal_axis") else None
            ),
            room_orientation_hint=(
                str(data["room_orientation_hint"]) if data.get("room_orientation_hint") else None
            ),
            expected_scale_hint=(
                str(data["expected_scale_hint"]) if data.get("expected_scale_hint") else None
            ),
            expected_door_count=(
                int(data["expected_door_count"]) if data.get("expected_door_count") is not None else None
            ),
            expected_window_count=(
                int(data["expected_window_count"]) if data.get("expected_window_count") is not None else None
            ),
            expected_object_categories=[str(value) for value in expected_categories_raw],
            expected_relations=[ExpectedRelation.from_dict(item) for item in rel_raw],
            traversability_labels=[TraversabilityLabel.from_dict(item) for item in trav_raw],
            notes=(str(data["notes"]) if data.get("notes") else None),
            metadata=dict(data.get("metadata", {})),
        )

    @classmethod
    def load_json(cls, path: Path) -> "SceneAnnotation":
        return cls.from_dict(_read_json_object(path))


def load_scene_annotation(path: Path | None) -> SceneAnnotation | None:
    if path is None:
        return None
    return SceneAnnotation.load_json(path)
