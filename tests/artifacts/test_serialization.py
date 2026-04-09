from pathlib import Path

from self_calibrating_spatiallm.artifacts import (
    AblationReport,
    AblationSettingResult,
    ActionDirective,
    ActionableScene,
    CalibratedPointCloud,
    CalibrationResult,
    EvaluationResult,
    Point3D,
    PointCloudMetadata,
    PointCloudSample,
    RepairResult,
    SceneObject,
    ScenePrediction,
    SceneRelation,
)


def test_artifact_json_roundtrip(tmp_path: Path) -> None:
    sample = PointCloudSample(
        sample_id="sample-1",
        points=[Point3D(0.0, 0.0, 0.0), Point3D(1.0, 0.5, 0.2)],
        metadata={"source": "test"},
    )
    metadata = PointCloudMetadata(
        sample_id="sample-1",
        source_path="/tmp/sample.ply",
        source_type="ply",
        num_points=2,
        bbox_min=Point3D(0.0, 0.0, 0.0),
        bbox_max=Point3D(1.0, 0.5, 0.2),
        centroid=Point3D(0.5, 0.25, 0.1),
        coordinate_ranges={"x": 1.0, "y": 0.5, "z": 0.2},
        has_rgb=True,
    )
    calibration = CalibrationResult(
        sample_id="sample-1",
        method="geometric_v0",
        up_vector=Point3D(0.0, 0.0, 1.0),
        horizontal_axis=Point3D(1.0, 0.0, 0.0),
        origin_offset=Point3D(0.0, 0.0, 0.0),
    )
    calibrated = CalibratedPointCloud(
        sample_id="sample-1",
        points=sample.points,
        calibration=calibration,
    )
    prediction = ScenePrediction(
        sample_id="sample-1",
        generator_name="mock",
        objects=[
            SceneObject(
                object_id="obj_floor",
                label="floor",
                position=Point3D(0.0, 0.0, 0.0),
                size=Point3D(4.0, 4.0, 0.1),
            )
        ],
        relations=[],
    )
    repair = RepairResult(
        sample_id="sample-1",
        repairer_name="rule",
        issues=[],
        fixes_applied=[],
        repaired_scene=prediction,
    )
    actionable = ActionableScene(
        sample_id="sample-1",
        builder_name="builder",
        anchor_object_id="obj_floor",
        relations=[SceneRelation("obj_floor", "near", "obj_floor", 1.0)],
        actions=[ActionDirective("inspect:floor", "obj_floor", "smoke")],
    )
    evaluation = EvaluationResult(
        sample_id="sample-1",
        evaluator_name="simple",
        metrics={"num_objects": 1.0},
        passed=True,
    )
    ablation = AblationReport(
        sample_id="sample-1",
        settings=[
            AblationSettingResult(
                setting_name="no_calibration",
                calibration_enabled=False,
                repair_enabled=False,
                calibration_method="none",
                repairer_name="none",
                evaluation=evaluation,
            )
        ],
    )

    pairs = [
        (sample, PointCloudSample, "sample.json"),
        (metadata, PointCloudMetadata, "point_cloud_metadata.json"),
        (calibration, CalibrationResult, "calibration.json"),
        (calibrated, CalibratedPointCloud, "calibrated.json"),
        (prediction, ScenePrediction, "prediction.json"),
        (repair, RepairResult, "repair.json"),
        (actionable, ActionableScene, "actionable.json"),
        (evaluation, EvaluationResult, "evaluation.json"),
        (ablation, AblationReport, "ablation.json"),
    ]

    for artifact, cls, filename in pairs:
        path = artifact.save_json(tmp_path / filename)
        loaded = cls.load_json(path)
        assert loaded == artifact
