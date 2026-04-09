from __future__ import annotations

from self_calibrating_spatiallm.artifacts import (
    CalibratedPointCloud,
    CalibrationResult,
    Point3D,
)
from self_calibrating_spatiallm.generation.mock_spatiallm import MockSpatialLMGenerator


def _calibrated_input(
    *,
    horizontal_confidence: float,
    reliability: float,
    ambiguity: float,
    inferred_scale_hint: str = "expected_unit:meter",
) -> CalibratedPointCloud:
    points = [
        Point3D(-1.0, -1.0, 0.0),
        Point3D(1.0, -1.0, 0.0),
        Point3D(-1.0, 1.0, 0.0),
        Point3D(1.0, 1.0, 0.0),
        Point3D(0.0, 0.0, 1.0),
    ]
    calibration = CalibrationResult(
        sample_id="scene-propagation",
        method="plane_aware_v1",
        up_vector=Point3D(0.0, 0.0, 1.0),
        horizontal_axis=Point3D(1.0, 0.0, 0.0),
        origin_offset=Point3D(0.0, 0.0, 0.0),
        metadata={
            "normalization_applied": True,
            "diagnostics": {
                "confidence": {
                    "up_axis": 0.75,
                    "horizontal_orientation": horizontal_confidence,
                    "overall_reliability": reliability,
                },
                "manhattan_ambiguity": ambiguity,
                "candidate_plane_count": 6,
            },
            "execution": {
                "true_v1_execution": True,
                "fallback_used": False,
                "partial_calibration_applied": False,
            },
        },
    )
    return CalibratedPointCloud(
        sample_id="scene-propagation",
        points=points,
        calibration=calibration,
        metadata={
            "frame": "canonical",
            "transformed_point_cloud": {
                "ranges": [6.0, 5.0, 2.4],
                "center": [0.0, 0.0, 1.2],
            },
            "room_bounds": {"min": [-3.0, -2.5, -0.1], "max": [3.0, 2.5, 2.7]},
            "inferred_scale_hint": inferred_scale_hint,
        },
    )


def test_mock_generator_propagates_calibration_signal_into_layout_mode() -> None:
    generator = MockSpatialLMGenerator()
    degraded = generator.generate(
        _calibrated_input(
            horizontal_confidence=0.20,
            reliability=0.55,
            ambiguity=0.96,
        )
    )
    informed = generator.generate(
        _calibrated_input(
            horizontal_confidence=0.80,
            reliability=0.76,
            ambiguity=0.20,
        )
    )

    degraded_diag = degraded.metadata.get("propagation_diagnostics", {})
    informed_diag = informed.metadata.get("propagation_diagnostics", {})

    assert degraded_diag.get("generator_mode") == "degraded_template"
    assert informed_diag.get("generator_mode") == "calibration_informed"

    degraded_labels = {obj.label for obj in degraded.objects}
    informed_labels = {obj.label for obj in informed.objects}
    assert "box" in degraded_labels
    assert "box" not in informed_labels

    degraded_rel_targets = {rel.object_id for rel in degraded.relations}
    assert "missing_surface" in degraded_rel_targets
    assert any(rel.predicate == "attached-to" for rel in informed.relations)


def test_mock_generator_keeps_box_when_scale_hint_requests_it() -> None:
    generator = MockSpatialLMGenerator()
    prediction = generator.generate(
        _calibrated_input(
            horizontal_confidence=0.82,
            reliability=0.78,
            ambiguity=0.18,
            inferred_scale_hint="meter_scale_likely",
        )
    )

    box = next(obj for obj in prediction.objects if obj.label == "box")
    assert box.size.x >= 0.2
    assert box.size.y >= 0.2
    assert box.position.z >= 0.0
    assert any(
        rel.subject_id == box.object_id and rel.predicate == "supported-by" and rel.object_id == "obj_floor"
        for rel in prediction.relations
    )
