from self_calibrating_spatiallm.artifacts import Point3D, PointCloudSample
from self_calibrating_spatiallm.calibration import GeometricCalibratorV0, PlaneAwareCalibratorV1


def test_geometric_calibration_outputs_expected_diagnostics() -> None:
    sample = PointCloudSample(
        sample_id="scene-1",
        points=[
            Point3D(-1.0, -1.0, 0.0),
            Point3D(1.0, -1.0, 0.0),
            Point3D(-1.0, 1.0, 0.0),
            Point3D(1.0, 1.0, 0.0),
            Point3D(0.0, 0.0, 1.0),
        ],
    )

    calibrated = GeometricCalibratorV0(normalize_scene=True).calibrate(sample)
    calibration = calibrated.calibration

    assert calibration.method == "geometric_v0"
    assert calibration.metadata["normalization_applied"] is True
    assert "rotation_matrix" in calibration.metadata
    assert "diagnostics" in calibration.metadata
    assert "transformed_point_cloud" in calibration.metadata

    diagnostics = calibration.metadata["diagnostics"]
    assert "up_axis_candidate" in diagnostics
    assert "confidence" in diagnostics
    assert "up_axis" in diagnostics["confidence"]
    assert "horizontal_orientation" in diagnostics["confidence"]


def test_plane_aware_v1_outputs_expected_diagnostics() -> None:
    sample = PointCloudSample(
        sample_id="scene-v1",
        points=[
            Point3D(-2.0, -2.0, 0.0),
            Point3D(2.0, -2.0, 0.0),
            Point3D(-2.0, 2.0, 0.0),
            Point3D(2.0, 2.0, 0.0),
            Point3D(-2.0, -2.0, 2.6),
            Point3D(2.0, -2.0, 2.6),
            Point3D(-2.0, 2.0, 2.6),
            Point3D(2.0, 2.0, 2.6),
            Point3D(-2.0, -2.0, 1.2),
            Point3D(-2.0, 2.0, 1.2),
            Point3D(2.0, -2.0, 1.2),
            Point3D(2.0, 2.0, 1.2),
            Point3D(0.0, -2.0, 1.4),
            Point3D(0.0, 2.0, 1.4),
            Point3D(-2.0, 0.0, 1.0),
            Point3D(2.0, 0.0, 1.0),
            Point3D(-1.0, -1.0, 0.75),
            Point3D(1.0, -1.0, 0.75),
            Point3D(-1.0, 1.0, 0.75),
            Point3D(1.0, 1.0, 0.75),
            Point3D(-1.4, -1.1, 0.05),
            Point3D(1.3, -1.2, 0.06),
            Point3D(-1.2, 1.3, 0.04),
            Point3D(1.1, 1.1, 0.05),
            Point3D(-1.8, -0.8, 2.4),
            Point3D(1.8, -0.9, 2.45),
            Point3D(-1.7, 0.9, 2.48),
            Point3D(1.7, 0.8, 2.44),
            Point3D(-0.4, -0.2, 1.0),
            Point3D(0.4, -0.2, 1.0),
            Point3D(-0.4, 0.2, 1.0),
            Point3D(0.4, 0.2, 1.0),
            Point3D(-0.2, -0.1, 1.05),
            Point3D(0.2, 0.1, 1.05),
            Point3D(0.0, 0.0, 1.12),
            Point3D(0.3, 0.0, 1.12),
        ],
    )

    calibrated = PlaneAwareCalibratorV1(normalize_scene=True).calibrate(sample)
    calibration = calibrated.calibration

    assert calibration.method == "plane_aware_v1"
    assert "diagnostics" in calibration.metadata
    diagnostics = calibration.metadata["diagnostics"]
    assert "plane_candidates" in diagnostics
    assert "selected_plane_roles" in diagnostics
    assert "axis_candidates" in diagnostics
    assert "confidence" in diagnostics
    assert "overall_reliability" in diagnostics["confidence"]
    assert "scale_reasoning" in diagnostics


def test_plane_aware_v1_fallback_behavior_for_sparse_cloud() -> None:
    sample = PointCloudSample(
        sample_id="scene-v1-fallback",
        points=[
            Point3D(-1.0, -1.0, 0.0),
            Point3D(1.0, -1.0, 0.0),
            Point3D(-1.0, 1.0, 0.0),
            Point3D(1.0, 1.0, 0.0),
            Point3D(0.0, 0.0, 1.0),
        ],
    )
    calibrated = PlaneAwareCalibratorV1(normalize_scene=True).calibrate(sample)
    diagnostics = calibrated.calibration.metadata.get("diagnostics", {})
    assert calibrated.calibration.method == "plane_aware_v1"
    assert diagnostics.get("fallback_reason")
    assert calibrated.calibration.metadata.get("fallback_to") == "geometric_v0"


def test_plane_aware_v1_confidence_fields_present() -> None:
    sample = PointCloudSample(
        sample_id="scene-v1-confidence",
        points=[
            Point3D(-2.0, -2.0, 0.0),
            Point3D(2.0, -2.0, 0.0),
            Point3D(-2.0, 2.0, 0.0),
            Point3D(2.0, 2.0, 0.0),
            Point3D(-2.0, -2.0, 2.8),
            Point3D(2.0, -2.0, 2.8),
            Point3D(-2.0, 2.0, 2.8),
            Point3D(2.0, 2.0, 2.8),
            Point3D(-2.0, -2.0, 1.4),
            Point3D(-2.0, 2.0, 1.4),
            Point3D(2.0, -2.0, 1.4),
            Point3D(2.0, 2.0, 1.4),
            Point3D(0.0, -2.0, 1.2),
            Point3D(0.0, 2.0, 1.2),
            Point3D(-2.0, 0.0, 1.1),
            Point3D(2.0, 0.0, 1.1),
            Point3D(-1.0, -1.0, 0.8),
            Point3D(1.0, -1.0, 0.8),
            Point3D(-1.0, 1.0, 0.8),
            Point3D(1.0, 1.0, 0.8),
            Point3D(-1.8, -0.7, 2.5),
            Point3D(1.8, -0.6, 2.5),
            Point3D(-1.7, 0.7, 2.5),
            Point3D(1.7, 0.6, 2.5),
            Point3D(-0.4, -0.2, 1.0),
            Point3D(0.4, -0.2, 1.0),
            Point3D(-0.4, 0.2, 1.0),
            Point3D(0.4, 0.2, 1.0),
            Point3D(0.0, 0.0, 1.08),
            Point3D(0.2, 0.0, 1.08),
            Point3D(-0.2, 0.0, 1.08),
            Point3D(0.0, 0.2, 1.08),
        ],
    )
    calibrated = PlaneAwareCalibratorV1(normalize_scene=True).calibrate(sample)
    diagnostics = calibrated.calibration.metadata.get("diagnostics", {})
    confidence = diagnostics.get("confidence", {})
    assert "up_axis" in confidence
    assert "horizontal_orientation" in confidence
    assert "overall_reliability" in confidence
    assert "reliability_breakdown" in diagnostics
