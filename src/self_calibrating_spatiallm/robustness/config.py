"""Config models for robustness-boundary experiments."""

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
class RobustnessSplitConfig:
    """One split definition for boundary experiments."""

    split_name: str
    role: str
    manifest_path: str
    enabled: bool = True
    allow_missing_manifest: bool = False
    tags: list[str] = field(default_factory=list)
    notes: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RobustnessSplitConfig":
        tags_raw = data.get("tags", [])
        if not isinstance(tags_raw, list):
            raise TypeError("split.tags must be a list")
        return cls(
            split_name=str(data["split_name"]),
            role=str(data.get("role", "generalization")),
            manifest_path=str(data["manifest_path"]),
            enabled=bool(data.get("enabled", True)),
            allow_missing_manifest=bool(data.get("allow_missing_manifest", False)),
            tags=[str(item) for item in tags_raw],
            notes=(str(data["notes"]) if data.get("notes") else None),
        )


@dataclass
class PerturbationFamilyConfig:
    """One perturbation family and its severity schedule."""

    name: str
    severities: list[float]
    enabled: bool = True
    params: dict[str, Any] = field(default_factory=dict)
    notes: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PerturbationFamilyConfig":
        raw_severities = data.get("severities", [])
        if not isinstance(raw_severities, list):
            raise TypeError("perturbation.severities must be a list")
        severities: list[float] = []
        for value in raw_severities:
            parsed = float(value)
            if parsed < 0.0:
                raise ValueError(f"severity must be >= 0, got {parsed}")
            severities.append(parsed)
        if not severities:
            raise ValueError(f"Perturbation {data.get('name')} must define severities")

        params = data.get("params", {})
        if not isinstance(params, dict):
            raise TypeError("perturbation.params must be an object")

        return cls(
            name=str(data["name"]),
            severities=sorted(set(severities)),
            enabled=bool(data.get("enabled", True)),
            params=dict(params),
            notes=(str(data["notes"]) if data.get("notes") else None),
        )


@dataclass
class RobustnessBoundaryConfig:
    """Top-level config for perturbation-driven robustness boundary studies."""

    name: str
    seed: int
    output_dir: str
    include_unperturbed_baseline: bool = True
    export_language_outputs: bool = True
    splits: list[RobustnessSplitConfig] = field(default_factory=list)
    perturbations: list[PerturbationFamilyConfig] = field(default_factory=list)
    notes: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    _base_dir: Path = field(default_factory=Path.cwd, repr=False, compare=False)

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        *,
        base_dir: Path | None = None,
    ) -> "RobustnessBoundaryConfig":
        split_rows = data.get("splits", [])
        if not isinstance(split_rows, list):
            raise TypeError("splits must be a list")
        perturbation_rows = data.get("perturbations", [])
        if not isinstance(perturbation_rows, list):
            raise TypeError("perturbations must be a list")
        metadata = data.get("metadata", {})
        if not isinstance(metadata, dict):
            raise TypeError("metadata must be an object")

        config = cls(
            name=str(data["name"]),
            seed=int(data.get("seed", 13)),
            output_dir=str(data.get("output_dir", "outputs/eval_pack/robustness_boundary/latest")),
            include_unperturbed_baseline=bool(data.get("include_unperturbed_baseline", True)),
            export_language_outputs=bool(data.get("export_language_outputs", True)),
            splits=[RobustnessSplitConfig.from_dict(item) for item in split_rows],
            perturbations=[PerturbationFamilyConfig.from_dict(item) for item in perturbation_rows],
            notes=(str(data["notes"]) if data.get("notes") else None),
            metadata=dict(metadata),
        )
        if base_dir is not None:
            config._base_dir = base_dir
        return config

    @classmethod
    def load_json(cls, path: Path) -> "RobustnessBoundaryConfig":
        return cls.from_dict(_read_json(path), base_dir=path.parent)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("_base_dir", None)
        return payload

    def resolve_output_dir(self, override: Path | None = None) -> Path:
        if override is not None:
            return override
        return _resolve_path(self.output_dir, self._base_dir)

    def resolve_manifest_path(self, split: RobustnessSplitConfig) -> Path:
        return _resolve_path(split.manifest_path, self._base_dir)


def _resolve_path(raw_path: str, base_dir: Path) -> Path:
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return candidate
    return (base_dir / candidate).resolve()

