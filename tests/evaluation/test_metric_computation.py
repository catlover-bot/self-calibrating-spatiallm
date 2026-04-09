from self_calibrating_spatiallm.artifacts import (
    ActionableScene,
    CalibrationResult,
    Point3D,
    PointCloudMetadata,
    RepairResult,
    SceneObject,
    ScenePrediction,
    SceneRelation,
)
from self_calibrating_spatiallm.evaluation import (
    ExpectedRelation,
    SceneAnnotation,
    TraversabilityLabel,
    compute_scene_metrics,
)


def test_metric_computation_smoke() -> None:
    annotation = SceneAnnotation(
        scene_id="scene-metric",
        expected_up_axis="z",
        expected_horizontal_axis="x",
        expected_scale_hint="expected_unit:meter",
        expected_door_count=1,
        expected_window_count=0,
        expected_object_categories=["floor", "wall", "table", "door", "mug"],
        expected_relations=[ExpectedRelation("mug", "supported-by", "table")],
        traversability_labels=[TraversabilityLabel("door", True)],
    )

    metadata = PointCloudMetadata(
        sample_id="scene-metric",
        source_path="/tmp/scene.ply",
        source_type="ply",
        num_points=100,
        bbox_min=Point3D(-1.0, -1.0, 0.0),
        bbox_max=Point3D(1.0, 1.0, 2.0),
        centroid=Point3D(0.0, 0.0, 1.0),
        coordinate_ranges={"x": 2.0, "y": 2.0, "z": 2.0},
        has_rgb=False,
        expected_unit="meter",
        inferred_scale_hint="expected_unit:meter",
    )
    calibration = CalibrationResult(
        sample_id="scene-metric",
        method="geometric_v0",
        up_vector=Point3D(0.0, 0.0, 1.0),
        horizontal_axis=Point3D(1.0, 0.0, 0.0),
        origin_offset=Point3D(0.0, 0.0, 0.0),
    )

    prediction = ScenePrediction(
        sample_id="scene-metric",
        generator_name="mock",
        objects=[
            SceneObject("obj_floor", "floor", Point3D(0.0, 0.0, 0.0), Point3D(4.0, 4.0, 0.1)),
            SceneObject("obj_wall", "wall", Point3D(0.0, 2.0, 1.2), Point3D(4.0, 0.1, 2.4)),
            SceneObject("obj_table", "table", Point3D(0.0, 0.0, 0.8), Point3D(1.0, 1.0, 0.8)),
            SceneObject("obj_door", "door", Point3D(1.5, 1.9, 1.0), Point3D(0.9, 0.2, 2.0)),
            SceneObject("obj_mug", "mug", Point3D(0.1, 0.1, 1.0), Point3D(0.1, 0.1, 0.1)),
        ],
        relations=[SceneRelation("obj_mug", "supported-by", "obj_table", 0.9)],
    )
    repair_result = RepairResult(
        sample_id="scene-metric",
        repairer_name="simple_rule_repairer",
        issues=[],
        fixes_applied=["added support relation"],
        repaired_scene=prediction,
    )
    actionable = ActionableScene(
        sample_id="scene-metric",
        builder_name="rule_based_actionable_builder",
        anchor_object_id="obj_table",
        relations=[
            SceneRelation("obj_mug", "supported-by", "obj_table", 0.9),
            SceneRelation("obj_door", "accessible", "agent", 0.7),
        ],
        actions=[],
    )

    metrics = compute_scene_metrics(
        annotation=annotation,
        point_cloud_metadata=metadata,
        calibration=calibration,
        prediction_before_repair=prediction,
        repair_result=repair_result,
        violations_before=1,
        violations_after=0,
        actionable_scene=actionable,
    )

    assert metrics["calibration_up_axis_error_deg"] == 0.0
    assert metrics["calibration_horizontal_error_deg"] == 0.0
    assert metrics["calibration_scale_match"] == 1.0
    assert metrics["structured_door_count_error"] == 0.0
    assert metrics["repair_violation_reduction"] == 1.0
    assert metrics["actionable_relation_f1"] > 0.0


def _build_metric_inputs(
    *,
    expected_horizontal_axis: str,
    predicted_horizontal_axis: Point3D,
):
    annotation = SceneAnnotation(
        scene_id="scene-metric",
        expected_up_axis="z",
        expected_horizontal_axis=expected_horizontal_axis,
        expected_scale_hint="expected_unit:meter",
        expected_door_count=1,
        expected_window_count=0,
        expected_object_categories=["floor", "wall", "table", "door", "mug"],
        expected_relations=[ExpectedRelation("mug", "supported-by", "table")],
        traversability_labels=[TraversabilityLabel("door", True)],
    )

    metadata = PointCloudMetadata(
        sample_id="scene-metric",
        source_path="/tmp/scene.ply",
        source_type="ply",
        num_points=100,
        bbox_min=Point3D(-1.0, -1.0, 0.0),
        bbox_max=Point3D(1.0, 1.0, 2.0),
        centroid=Point3D(0.0, 0.0, 1.0),
        coordinate_ranges={"x": 2.0, "y": 2.0, "z": 2.0},
        has_rgb=False,
        expected_unit="meter",
        inferred_scale_hint="expected_unit:meter",
    )
    calibration = CalibrationResult(
        sample_id="scene-metric",
        method="geometric_v0",
        up_vector=Point3D(0.0, 0.0, 1.0),
        horizontal_axis=predicted_horizontal_axis,
        origin_offset=Point3D(0.0, 0.0, 0.0),
    )

    prediction = ScenePrediction(
        sample_id="scene-metric",
        generator_name="mock",
        objects=[
            SceneObject("obj_floor", "floor", Point3D(0.0, 0.0, 0.0), Point3D(4.0, 4.0, 0.1)),
            SceneObject("obj_wall", "wall", Point3D(0.0, 2.0, 1.2), Point3D(4.0, 0.1, 2.4)),
            SceneObject("obj_table", "table", Point3D(0.0, 0.0, 0.8), Point3D(1.0, 1.0, 0.8)),
            SceneObject("obj_door", "door", Point3D(1.5, 1.9, 1.0), Point3D(0.9, 0.2, 2.0)),
            SceneObject("obj_mug", "mug", Point3D(0.1, 0.1, 1.0), Point3D(0.1, 0.1, 0.1)),
        ],
        relations=[SceneRelation("obj_mug", "supported-by", "obj_table", 0.9)],
    )
    repair_result = RepairResult(
        sample_id="scene-metric",
        repairer_name="simple_rule_repairer",
        issues=[],
        fixes_applied=[],
        repaired_scene=prediction,
    )
    actionable = ActionableScene(
        sample_id="scene-metric",
        builder_name="rule_based_actionable_builder",
        anchor_object_id="obj_table",
        relations=[
            SceneRelation("obj_mug", "supported-by", "obj_table", 0.9),
            SceneRelation("obj_door", "accessible", "agent", 0.7),
        ],
        actions=[],
    )
    return annotation, metadata, calibration, prediction, repair_result, actionable


def test_horizontal_axis_metric_treats_unsigned_axis_as_undirected() -> None:
    annotation, metadata, calibration, prediction, repair_result, actionable = _build_metric_inputs(
        expected_horizontal_axis="x",
        predicted_horizontal_axis=Point3D(-1.0, 0.0, 0.0),
    )

    metrics = compute_scene_metrics(
        annotation=annotation,
        point_cloud_metadata=metadata,
        calibration=calibration,
        prediction_before_repair=prediction,
        repair_result=repair_result,
        violations_before=0,
        violations_after=0,
        actionable_scene=actionable,
    )

    assert metrics["calibration_horizontal_error_deg"] == 0.0


def test_horizontal_axis_metric_respects_explicit_signed_axis() -> None:
    annotation, metadata, calibration, prediction, repair_result, actionable = _build_metric_inputs(
        expected_horizontal_axis="-x",
        predicted_horizontal_axis=Point3D(1.0, 0.0, 0.0),
    )

    metrics = compute_scene_metrics(
        annotation=annotation,
        point_cloud_metadata=metadata,
        calibration=calibration,
        prediction_before_repair=prediction,
        repair_result=repair_result,
        violations_before=0,
        violations_after=0,
        actionable_scene=actionable,
    )

    assert metrics["calibration_horizontal_error_deg"] == 180.0
