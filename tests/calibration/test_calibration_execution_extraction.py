from self_calibrating_spatiallm.artifacts import CalibrationResult, Point3D
from self_calibrating_spatiallm.calibration import extract_calibration_execution


def test_extract_calibration_execution_true_v1() -> None:
    calibration = CalibrationResult(
        sample_id="scene-true",
        method="plane_aware_v1",
        up_vector=Point3D(0.0, 0.0, 1.0),
        horizontal_axis=Point3D(1.0, 0.0, 0.0),
        origin_offset=Point3D(0.0, 0.0, 0.0),
        metadata={
            "execution": {
                "plane_aware_logic_ran": True,
                "true_v1_execution": True,
                "fallback_used": False,
                "fallback_reason": None,
                "candidate_plane_count": 4,
                "weak_scale_reasoning_active": True,
                "partial_calibration_applied": True,
                "partial_calibration_reasons": ["guardrail:up_axis_disagreement_with_v0"],
            },
            "diagnostics": {
                "confidence": {"up_axis": 0.7, "horizontal_orientation": 0.6, "overall_reliability": 0.66},
                "manhattan_ambiguity": 0.2,
            },
            "scale_reasoning": {"scale_drift": 0.1},
        },
    )
    execution = extract_calibration_execution(calibration)
    assert execution["true_v1_execution"] is True
    assert execution["fallback_used"] is False
    assert execution["candidate_plane_count"] == 4
    assert execution["weak_scale_reasoning_active"] is True
    assert execution["partial_calibration_applied"] is True
    assert execution["partial_calibration_reasons"] == ["guardrail:up_axis_disagreement_with_v0"]


def test_extract_calibration_execution_fallback() -> None:
    calibration = CalibrationResult(
        sample_id="scene-fallback",
        method="plane_aware_v1",
        up_vector=Point3D(0.0, 0.0, 1.0),
        horizontal_axis=Point3D(1.0, 0.0, 0.0),
        origin_offset=Point3D(0.0, 0.0, 0.0),
        metadata={
            "diagnostics": {
                "backend": "fallback_v0",
                "fallback_reason": "numpy_unavailable",
                "plane_candidates": [],
                "confidence": {"up_axis": 0.0, "horizontal_orientation": 0.0},
            },
        },
    )
    execution = extract_calibration_execution(calibration)
    assert execution["true_v1_execution"] is False
    assert execution["fallback_used"] is True
    assert execution["fallback_reason"] == "numpy_unavailable"
    assert execution["candidate_plane_count"] == 0
    assert execution["partial_calibration_applied"] is False
