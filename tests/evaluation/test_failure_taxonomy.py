from self_calibrating_spatiallm.evaluation import classify_failures, summarize_failure_taxonomy


def test_failure_taxonomy_classification() -> None:
    metrics = {
        "calibration_up_axis_error_deg": 45.0,
        "calibration_up_axis_confidence": 0.2,
        "calibration_horizontal_error_deg": 35.0,
        "calibration_manhattan_ambiguity": 0.7,
        "calibration_scale_error": 1.0,
        "calibration_scale_drift": 0.9,
        "calibration_insufficient_plane_evidence_flag": 1.0,
        "structured_door_count_error": 1.0,
        "structured_window_count_error": 0.0,
        "structured_violation_count_before_repair": 6.0,
        "repair_overcorrection_flag": 1.0,
        "actionable_relation_f1": 0.2,
    }
    failures = classify_failures(metrics)
    assert "wrong_up_axis" in failures
    assert "unstable_up_axis_confidence" in failures
    assert "wrong_horizontal_axis" in failures
    assert "manhattan_ambiguity" in failures
    assert "scale_drift" in failures
    assert "insufficient_plane_evidence" in failures
    assert "missing_structural_elements" in failures
    assert "repair_overcorrection" in failures
    assert "relation_derivation_failure" in failures
    assert "clutter_dominated_failure" in failures
    assert "non_manhattan_scene_failure" in failures


def test_failure_taxonomy_summary() -> None:
    summary = summarize_failure_taxonomy(
        [
            {"setting_name": "no_calibration", "failures": ["wrong_up_axis", "scale_drift"]},
            {"setting_name": "calibration_v1", "failures": ["scale_drift"]},
        ]
    )
    assert summary["overall_counts"]["scale_drift"] == 2
    assert summary["counts_by_setting"]["no_calibration"]["wrong_up_axis"] == 1
    assert summary["calibration_failure_counts"]["scale_drift"] == 2
