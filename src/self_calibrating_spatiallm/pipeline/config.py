"""Configuration models for single-scene pipeline runs."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class SceneInputConfig:
    """Config describing one real point-cloud scene input."""

    scene_id: str
    file_path: str
    source_type: str = "auto"
    metadata_path: str | None = None
    expected_unit: str | None = None
    scale_hint: str | None = None

    output_dir: str | None = "outputs/runs/single_scene"
    normalize_scene: bool = True
    calibration_mode: str = "v1"

    generator_mode: str = "mock"
    spatiallm_command: str | None = None
    spatiallm_output_json: str | None = None
    external_timeout_sec: int = 120
    spatiallm_command_env_var: str = "SCSLM_SPATIALLM_COMMAND"
    spatiallm_export_format: str = "json"
    compare_with_external_generator: bool = False

    _base_dir: Path = field(default_factory=Path.cwd, repr=False, compare=False)

    @classmethod
    def from_dict(cls, data: dict[str, Any], base_dir: Path | None = None) -> "SceneInputConfig":
        config = cls(
            scene_id=str(data["scene_id"]),
            file_path=str(data["file_path"]),
            source_type=str(data.get("source_type", "auto")),
            metadata_path=(str(data["metadata_path"]) if data.get("metadata_path") else None),
            expected_unit=(str(data["expected_unit"]) if data.get("expected_unit") else None),
            scale_hint=(str(data["scale_hint"]) if data.get("scale_hint") else None),
            output_dir=(str(data["output_dir"]) if data.get("output_dir") else None),
            normalize_scene=bool(data.get("normalize_scene", True)),
            calibration_mode=str(data.get("calibration_mode", "v1")),
            generator_mode=str(data.get("generator_mode", "mock")),
            spatiallm_command=(str(data["spatiallm_command"]) if data.get("spatiallm_command") else None),
            spatiallm_output_json=(
                str(data["spatiallm_output_json"]) if data.get("spatiallm_output_json") else None
            ),
            external_timeout_sec=int(data.get("external_timeout_sec", 120)),
            spatiallm_command_env_var=str(
                data.get("spatiallm_command_env_var", "SCSLM_SPATIALLM_COMMAND")
            ),
            spatiallm_export_format=str(data.get("spatiallm_export_format", "json")),
            compare_with_external_generator=bool(data.get("compare_with_external_generator", False)),
        )
        if base_dir is not None:
            config._base_dir = base_dir
        return config

    @classmethod
    def load_json(cls, path: Path) -> "SceneInputConfig":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError(f"Expected JSON object at {path}")
        return cls.from_dict(payload, base_dir=path.parent)

    def resolve_file_path(self) -> Path:
        return _resolve_path(self.file_path, self._base_dir)

    def resolve_metadata_path(self) -> Path | None:
        if self.metadata_path is None:
            return None
        return _resolve_path(self.metadata_path, self._base_dir)

    def resolve_output_dir(self, override: Path | None = None) -> Path:
        if override is not None:
            return override
        if self.output_dir is None:
            return self._base_dir / "outputs" / "runs" / self.scene_id
        return _resolve_path(self.output_dir, self._base_dir)

    def resolve_external_output_json(self) -> Path | None:
        if self.spatiallm_output_json is None:
            return None
        return _resolve_path(self.spatiallm_output_json, self._base_dir)

    def has_external_generator_config(self) -> bool:
        if self.spatiallm_command:
            return True
        if self.spatiallm_output_json:
            return True
        return bool(os.environ.get(self.spatiallm_command_env_var))


def _resolve_path(raw_path: str, base_dir: Path) -> Path:
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return candidate
    return (base_dir / candidate).resolve()
