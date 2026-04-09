"""Artifact datamodels for the single-scene pipeline."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

JsonDict = dict[str, Any]


def _read_json(path: Path) -> JsonDict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object at {path}")
    return payload


class JsonArtifact:
    """Mixin that provides JSON persistence helpers."""

    def to_dict(self) -> JsonDict:
        return asdict(self)

    def save_json(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        return path


@dataclass
class Point3D(JsonArtifact):
    x: float
    y: float
    z: float

    @classmethod
    def from_dict(cls, data: JsonDict) -> "Point3D":
        return cls(x=float(data["x"]), y=float(data["y"]), z=float(data["z"]))


@dataclass
class SceneObject(JsonArtifact):
    object_id: str
    label: str
    position: Point3D
    size: Point3D
    confidence: float = 1.0
    attributes: JsonDict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: JsonDict) -> "SceneObject":
        return cls(
            object_id=str(data["object_id"]),
            label=str(data["label"]),
            position=Point3D.from_dict(data["position"]),
            size=Point3D.from_dict(data["size"]),
            confidence=float(data.get("confidence", 1.0)),
            attributes=dict(data.get("attributes", {})),
        )


@dataclass
class SceneRelation(JsonArtifact):
    subject_id: str
    predicate: str
    object_id: str
    score: float = 1.0
    metadata: JsonDict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: JsonDict) -> "SceneRelation":
        return cls(
            subject_id=str(data["subject_id"]),
            predicate=str(data["predicate"]),
            object_id=str(data["object_id"]),
            score=float(data.get("score", 1.0)),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class ActionDirective(JsonArtifact):
    action: str
    target_object_id: str
    rationale: str

    @classmethod
    def from_dict(cls, data: JsonDict) -> "ActionDirective":
        return cls(
            action=str(data["action"]),
            target_object_id=str(data["target_object_id"]),
            rationale=str(data["rationale"]),
        )


@dataclass
class PointCloudSample(JsonArtifact):
    sample_id: str
    points: list[Point3D]
    sensor_frame: str = "sensor"
    metadata: JsonDict = field(default_factory=dict)

    @property
    def num_points(self) -> int:
        return len(self.points)

    @classmethod
    def from_dict(cls, data: JsonDict) -> "PointCloudSample":
        points_raw = data.get("points", [])
        if not isinstance(points_raw, list):
            raise TypeError("PointCloudSample.points must be a list")
        return cls(
            sample_id=str(data["sample_id"]),
            points=[Point3D.from_dict(point) for point in points_raw],
            sensor_frame=str(data.get("sensor_frame", "sensor")),
            metadata=dict(data.get("metadata", {})),
        )

    @classmethod
    def load_json(cls, path: Path) -> "PointCloudSample":
        return cls.from_dict(_read_json(path))


@dataclass
class PointCloudMetadata(JsonArtifact):
    sample_id: str
    source_path: str
    source_type: str
    num_points: int
    bbox_min: Point3D
    bbox_max: Point3D
    centroid: Point3D
    coordinate_ranges: dict[str, float]
    has_rgb: bool
    expected_unit: str | None = None
    inferred_scale_hint: str | None = None
    metadata: JsonDict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: JsonDict) -> "PointCloudMetadata":
        ranges_raw = data.get("coordinate_ranges", {})
        if not isinstance(ranges_raw, dict):
            raise TypeError("PointCloudMetadata.coordinate_ranges must be an object")
        return cls(
            sample_id=str(data["sample_id"]),
            source_path=str(data["source_path"]),
            source_type=str(data["source_type"]),
            num_points=int(data["num_points"]),
            bbox_min=Point3D.from_dict(data["bbox_min"]),
            bbox_max=Point3D.from_dict(data["bbox_max"]),
            centroid=Point3D.from_dict(data["centroid"]),
            coordinate_ranges={str(key): float(value) for key, value in ranges_raw.items()},
            has_rgb=bool(data["has_rgb"]),
            expected_unit=(str(data["expected_unit"]) if data.get("expected_unit") else None),
            inferred_scale_hint=(
                str(data["inferred_scale_hint"]) if data.get("inferred_scale_hint") else None
            ),
            metadata=dict(data.get("metadata", {})),
        )

    @classmethod
    def load_json(cls, path: Path) -> "PointCloudMetadata":
        return cls.from_dict(_read_json(path))


@dataclass
class CalibrationResult(JsonArtifact):
    sample_id: str
    method: str
    up_vector: Point3D
    horizontal_axis: Point3D
    origin_offset: Point3D
    metadata: JsonDict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: JsonDict) -> "CalibrationResult":
        horizontal = data.get("horizontal_axis", {"x": 1.0, "y": 0.0, "z": 0.0})
        return cls(
            sample_id=str(data["sample_id"]),
            method=str(data["method"]),
            up_vector=Point3D.from_dict(data["up_vector"]),
            horizontal_axis=Point3D.from_dict(horizontal),
            origin_offset=Point3D.from_dict(data["origin_offset"]),
            metadata=dict(data.get("metadata", {})),
        )

    @classmethod
    def load_json(cls, path: Path) -> "CalibrationResult":
        return cls.from_dict(_read_json(path))


@dataclass
class CalibratedPointCloud(JsonArtifact):
    sample_id: str
    points: list[Point3D]
    calibration: CalibrationResult
    metadata: JsonDict = field(default_factory=dict)

    @property
    def num_points(self) -> int:
        return len(self.points)

    @classmethod
    def from_dict(cls, data: JsonDict) -> "CalibratedPointCloud":
        points_raw = data.get("points", [])
        if not isinstance(points_raw, list):
            raise TypeError("CalibratedPointCloud.points must be a list")
        return cls(
            sample_id=str(data["sample_id"]),
            points=[Point3D.from_dict(point) for point in points_raw],
            calibration=CalibrationResult.from_dict(data["calibration"]),
            metadata=dict(data.get("metadata", {})),
        )

    @classmethod
    def load_json(cls, path: Path) -> "CalibratedPointCloud":
        return cls.from_dict(_read_json(path))


@dataclass
class ScenePrediction(JsonArtifact):
    sample_id: str
    generator_name: str
    objects: list[SceneObject]
    relations: list[SceneRelation] = field(default_factory=list)
    metadata: JsonDict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: JsonDict) -> "ScenePrediction":
        objects_raw = data.get("objects", [])
        relations_raw = data.get("relations", [])
        if not isinstance(objects_raw, list) or not isinstance(relations_raw, list):
            raise TypeError("ScenePrediction.objects and relations must be lists")
        return cls(
            sample_id=str(data["sample_id"]),
            generator_name=str(data["generator_name"]),
            objects=[SceneObject.from_dict(item) for item in objects_raw],
            relations=[SceneRelation.from_dict(item) for item in relations_raw],
            metadata=dict(data.get("metadata", {})),
        )

    @classmethod
    def load_json(cls, path: Path) -> "ScenePrediction":
        return cls.from_dict(_read_json(path))


@dataclass
class RepairResult(JsonArtifact):
    sample_id: str
    repairer_name: str
    issues: list[str]
    fixes_applied: list[str]
    repaired_scene: ScenePrediction
    metadata: JsonDict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: JsonDict) -> "RepairResult":
        return cls(
            sample_id=str(data["sample_id"]),
            repairer_name=str(data["repairer_name"]),
            issues=[str(v) for v in data.get("issues", [])],
            fixes_applied=[str(v) for v in data.get("fixes_applied", [])],
            repaired_scene=ScenePrediction.from_dict(data["repaired_scene"]),
            metadata=dict(data.get("metadata", {})),
        )

    @classmethod
    def load_json(cls, path: Path) -> "RepairResult":
        return cls.from_dict(_read_json(path))


@dataclass
class ActionableScene(JsonArtifact):
    sample_id: str
    builder_name: str
    anchor_object_id: str | None
    relations: list[SceneRelation]
    actions: list[ActionDirective]
    metadata: JsonDict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: JsonDict) -> "ActionableScene":
        return cls(
            sample_id=str(data["sample_id"]),
            builder_name=str(data["builder_name"]),
            anchor_object_id=(str(data["anchor_object_id"]) if data.get("anchor_object_id") else None),
            relations=[SceneRelation.from_dict(item) for item in data.get("relations", [])],
            actions=[ActionDirective.from_dict(item) for item in data.get("actions", [])],
            metadata=dict(data.get("metadata", {})),
        )

    @classmethod
    def load_json(cls, path: Path) -> "ActionableScene":
        return cls.from_dict(_read_json(path))


@dataclass
class EvaluationResult(JsonArtifact):
    sample_id: str
    evaluator_name: str
    metrics: dict[str, float]
    passed: bool
    notes: list[str] = field(default_factory=list)
    metadata: JsonDict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: JsonDict) -> "EvaluationResult":
        metrics_raw = data.get("metrics", {})
        if not isinstance(metrics_raw, dict):
            raise TypeError("EvaluationResult.metrics must be an object")
        return cls(
            sample_id=str(data["sample_id"]),
            evaluator_name=str(data["evaluator_name"]),
            metrics={str(k): float(v) for k, v in metrics_raw.items()},
            passed=bool(data["passed"]),
            notes=[str(v) for v in data.get("notes", [])],
            metadata=dict(data.get("metadata", {})),
        )

    @classmethod
    def load_json(cls, path: Path) -> "EvaluationResult":
        return cls.from_dict(_read_json(path))


@dataclass
class AblationSettingResult(JsonArtifact):
    setting_name: str
    calibration_enabled: bool
    repair_enabled: bool
    calibration_method: str
    repairer_name: str
    evaluation: EvaluationResult
    metadata: JsonDict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: JsonDict) -> "AblationSettingResult":
        return cls(
            setting_name=str(data["setting_name"]),
            calibration_enabled=bool(data["calibration_enabled"]),
            repair_enabled=bool(data["repair_enabled"]),
            calibration_method=str(data["calibration_method"]),
            repairer_name=str(data["repairer_name"]),
            evaluation=EvaluationResult.from_dict(data["evaluation"]),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class AblationReport(JsonArtifact):
    sample_id: str
    settings: list[AblationSettingResult]
    generator_settings: list[AblationSettingResult] = field(default_factory=list)
    metadata: JsonDict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: JsonDict) -> "AblationReport":
        settings_raw = data.get("settings", [])
        generator_raw = data.get("generator_settings", [])
        if not isinstance(settings_raw, list):
            raise TypeError("AblationReport.settings must be a list")
        if not isinstance(generator_raw, list):
            raise TypeError("AblationReport.generator_settings must be a list")
        return cls(
            sample_id=str(data["sample_id"]),
            settings=[AblationSettingResult.from_dict(item) for item in settings_raw],
            generator_settings=[AblationSettingResult.from_dict(item) for item in generator_raw],
            metadata=dict(data.get("metadata", {})),
        )

    @classmethod
    def load_json(cls, path: Path) -> "AblationReport":
        return cls.from_dict(_read_json(path))
