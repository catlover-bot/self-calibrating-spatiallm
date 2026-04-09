"""Evaluation pack manifest models for small multi-scene research runs."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object at {path}")
    return payload


@dataclass
class EvaluationSceneEntry:
    """One scene entry in an evaluation pack manifest."""

    sample_config_path: str
    annotation_path: str | None = None
    source_type: str | None = None
    tags: list[str] = field(default_factory=list)
    notes: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvaluationSceneEntry":
        tags = data.get("tags", [])
        if not isinstance(tags, list):
            raise TypeError("tags must be a list")
        return cls(
            sample_config_path=str(data["sample_config_path"]),
            annotation_path=(str(data["annotation_path"]) if data.get("annotation_path") else None),
            source_type=(str(data["source_type"]) if data.get("source_type") else None),
            tags=[str(tag) for tag in tags],
            notes=(str(data["notes"]) if data.get("notes") else None),
        )


@dataclass
class EvaluationPackManifest:
    """Manifest for evaluating a small collection of scenes."""

    name: str
    entries: list[EvaluationSceneEntry]
    notes: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save_json(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        return path

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvaluationPackManifest":
        entries_raw = data.get("entries", [])
        if not isinstance(entries_raw, list):
            raise TypeError("entries must be a list")
        return cls(
            name=str(data["name"]),
            entries=[EvaluationSceneEntry.from_dict(item) for item in entries_raw],
            notes=(str(data["notes"]) if data.get("notes") else None),
            metadata=dict(data.get("metadata", {})),
        )

    @classmethod
    def load_json(cls, path: Path) -> "EvaluationPackManifest":
        return cls.from_dict(_read_json(path))

    def resolve_paths(self, manifest_path: Path) -> list[tuple[Path, Path | None, EvaluationSceneEntry]]:
        base = manifest_path.parent
        resolved: list[tuple[Path, Path | None, EvaluationSceneEntry]] = []
        for entry in self.entries:
            config_path = Path(entry.sample_config_path)
            annotation_path = Path(entry.annotation_path) if entry.annotation_path else None

            resolved_config = config_path if config_path.is_absolute() else (base / config_path).resolve()
            resolved_annotation = (
                annotation_path if (annotation_path and annotation_path.is_absolute()) else None
            )
            if annotation_path is not None and resolved_annotation is None:
                resolved_annotation = (base / annotation_path).resolve()

            resolved.append((resolved_config, resolved_annotation, entry))
        return resolved
