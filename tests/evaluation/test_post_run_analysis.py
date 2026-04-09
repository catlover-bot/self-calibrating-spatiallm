from __future__ import annotations

from self_calibrating_spatiallm.evaluation.post_run_analysis import (
    ALLOWED_IMPROVEMENT_TARGETS,
    build_next_improvement_decision,
    build_partition_comparison_summaries,
    build_post_true_v1_analysis_bundle,
    build_researcher_facing_summary,
    build_scene_level_delta_report,
    build_stratified_comparison_summaries,
    build_trustworthy_comparison_status,
    recommend_next_calibration_improvement_target,
)


def _scene_payload(
    *,
    scene_id: str,
    source_type: str,
    tags: list[str],
    true_v1: bool,
    fallback_reason: str | None,
    partial_calibration: bool = False,
    failure_labels: list[str] | None = None,
    v0_relation_f1: float,
    v1_relation_f1: float,
    v1_repair_relation_f1: float | None = None,
) -> dict:
    if failure_labels is None:
        failure_labels = []
    execution = {
        "true_v1_execution": true_v1,
        "fallback_used": not true_v1,
        "fallback_reason": fallback_reason,
        "candidate_plane_count": 12 if true_v1 else 0,
        "up_axis_confidence": 0.7 if true_v1 else 0.4,
        "horizontal_confidence": 0.65 if true_v1 else 0.5,
        "overall_reliability": 0.68 if true_v1 else 0.45,
        "manhattan_ambiguity": 0.21 if true_v1 else 0.5,
        "scale_drift": 0.06,
    }
    if partial_calibration:
        execution["partial_calibration_applied"] = True
        execution["fallback_used"] = False

    if v1_repair_relation_f1 is None:
        v1_repair_relation_f1 = v1_relation_f1

    return {
        "scene_id": scene_id,
        "source_type": source_type,
        "tags": tags,
        "settings": [
            {
                "setting_name": "calibration_v0",
                "status": "success",
                "metrics": {
                    "calibration_up_axis_error_deg": 5.0,
                    "calibration_horizontal_error_deg": 10.0,
                    "calibration_reliability": 0.5,
                    "calibration_scale_drift": 0.05,
                    "structured_violation_count_before_repair": 4.0,
                    "actionable_relation_f1": v0_relation_f1,
                },
                "failures": [],
            },
            {
                "setting_name": "calibration_v1",
                "status": "success",
                "metrics": {
                    "calibration_up_axis_error_deg": 3.0,
                    "calibration_horizontal_error_deg": 8.0,
                    "calibration_reliability": 0.7,
                    "calibration_scale_drift": 0.02,
                    "structured_violation_count_before_repair": 3.0,
                    "actionable_relation_f1": v1_relation_f1,
                },
                "failures": failure_labels,
                "metadata": {
                    "calibration_execution": execution,
                },
            },
            {
                "setting_name": "calibration_v1_plus_repair",
                "status": "success",
                "metrics": {
                    "calibration_up_axis_error_deg": 3.0,
                    "calibration_horizontal_error_deg": 8.0,
                    "calibration_reliability": 0.71,
                    "calibration_scale_drift": 0.02,
                    "structured_violation_count_before_repair": 2.0,
                    "repair_violations_after": 0.0,
                    "actionable_relation_f1": v1_repair_relation_f1,
                },
                "failures": failure_labels,
            },
        ],
    }


def test_scene_level_delta_report_generation() -> None:
    scenes = [
        _scene_payload(
            scene_id="scene-1",
            source_type="ply",
            tags=["office"],
            true_v1=True,
            fallback_reason=None,
            failure_labels=["wrong_up_axis"],
            v0_relation_f1=0.20,
            v1_relation_f1=0.35,
            v1_repair_relation_f1=0.40,
        )
    ]

    rows = build_scene_level_delta_report(scenes)

    assert len(rows) == 1
    row = rows[0]
    assert row["scene_id"] == "scene-1"
    assert row["true_v1_execution"] is True
    assert row["fallback_used"] is False
    assert row["fallback_reason"] is None
    assert row["candidate_plane_count"] == 12
    assert row["major_failure_category"] == "wrong_up_axis"
    assert abs(float(row["delta_v1_minus_v0"]["actionable_relation_f1"]) - 0.15) < 1e-9
    assert abs(float(row["delta_v1_plus_repair_minus_v1"]["actionable_relation_f1"]) - 0.05) < 1e-9


def test_scene_level_delta_report_includes_prediction_and_propagation_diagnostics() -> None:
    scene = _scene_payload(
        scene_id="scene-propagation",
        source_type="ply",
        tags=["office"],
        true_v1=True,
        fallback_reason=None,
        failure_labels=["manhattan_ambiguity"],
        v0_relation_f1=0.20,
        v1_relation_f1=0.28,
        v1_repair_relation_f1=0.33,
    )

    for setting in scene["settings"]:
        if setting["setting_name"] == "calibration_v0":
            setting["metadata"] = {
                "prediction_summary_pre_repair": {
                    "object_count": 6,
                    "relation_count": 2,
                    "object_labels": ["floor", "wall", "table", "mug", "door", "box"],
                    "relation_predicates": ["supported-by"],
                },
                "calibrated_input_summary": {
                    "frame": "canonical",
                    "num_points": 500,
                    "ranges": [6.0, 5.0, 2.4],
                },
                "propagation_diagnostics": {
                    "generator_mode": "degraded_template",
                    "layout_changes": ["retained_noisy_box_due_to_weak_calibration"],
                    "calibration_signal": {"signal_strength": 0.40},
                },
            }
        if setting["setting_name"] == "calibration_v1":
            setting["metadata"]["prediction_summary_pre_repair"] = {
                "object_count": 5,
                "relation_count": 4,
                "object_labels": ["floor", "wall", "table", "mug", "door"],
                "relation_predicates": ["supported-by", "attached-to"],
            }
            setting["metadata"]["calibrated_input_summary"] = {
                "frame": "canonical",
                "num_points": 500,
                "ranges": [6.0, 5.0, 2.4],
            }
            setting["metadata"]["propagation_diagnostics"] = {
                "generator_mode": "calibration_informed",
                "layout_changes": ["suppressed_uncertain_box_candidate"],
                "calibration_signal": {"signal_strength": 0.73},
            }
        if setting["setting_name"] == "calibration_v1_plus_repair":
            setting["metadata"] = {
                "prediction_summary_pre_repair": {
                    "object_count": 5,
                    "relation_count": 4,
                    "object_labels": ["floor", "wall", "table", "mug", "door"],
                    "relation_predicates": ["supported-by", "attached-to"],
                },
                "propagation_diagnostics": {
                    "generator_mode": "calibration_informed",
                    "layout_changes": ["suppressed_uncertain_box_candidate"],
                    "calibration_signal": {"signal_strength": 0.73},
                },
            }

    rows = build_scene_level_delta_report([scene])
    row = rows[0]

    assert row["prediction_summary_v0"]["object_count"] == 6
    assert row["prediction_summary_v1"]["object_count"] == 5
    assert row["prediction_delta_v1_minus_v0"]["object_count_delta"] == -1
    assert "box" in row["prediction_delta_v1_minus_v0"]["removed_labels"]
    assert "attached-to" in row["prediction_delta_v1_minus_v0"]["added_predicates"]
    assert row["propagation_diagnostics_v1"]["generator_mode"] == "calibration_informed"
    assert row["calibrated_input_summary_v1"]["frame"] == "canonical"


def test_scene_level_delta_report_includes_external_generator_deltas() -> None:
    scene = _scene_payload(
        scene_id="scene-external-delta",
        source_type="ply",
        tags=["office"],
        true_v1=True,
        fallback_reason=None,
        failure_labels=["manhattan_ambiguity"],
        v0_relation_f1=0.20,
        v1_relation_f1=0.28,
        v1_repair_relation_f1=0.30,
    )

    scene["settings"].append(
        {
            "setting_name": "mock_generator",
            "status": "success",
            "metrics": {
                "structured_violation_count_before_repair": 4.0,
                "actionable_relation_f1": 0.25,
            },
            "failures": [],
            "metadata": {
                "prediction_summary_pre_repair": {
                    "object_count": 6,
                    "relation_count": 2,
                    "object_labels": ["floor", "wall", "table", "mug", "door", "box"],
                    "relation_predicates": ["supported-by"],
                },
                "calibrated_input_summary": {
                    "frame": "canonical",
                    "num_points": 500,
                    "ranges": [6.0, 5.0, 2.4],
                },
                "propagation_diagnostics": {
                    "generator_mode": "degraded_template",
                    "layout_changes": ["using_degraded_template_layout"],
                },
            },
        }
    )
    scene["settings"].append(
        {
            "setting_name": "external_generator",
            "status": "success",
            "metrics": {
                "structured_violation_count_before_repair": 3.0,
                "actionable_relation_f1": 0.30,
            },
            "failures": [],
            "metadata": {
                "prediction_summary_pre_repair": {
                    "object_count": 5,
                    "relation_count": 3,
                    "object_labels": ["floor", "wall", "table", "door", "mug"],
                    "relation_predicates": ["supported-by", "attached-to"],
                },
                "calibrated_input_summary": {
                    "frame": "canonical",
                    "num_points": 500,
                    "ranges": [6.0, 5.0, 2.4],
                },
                "propagation_diagnostics": {
                    "generator_mode": "external",
                    "command_source": "config",
                    "payload_source": "stdout_json",
                    "parse_mode": "full_schema",
                    "parse_warning_count": 0,
                },
                "generator_execution_summary": {
                    "generator_mode": "external",
                    "success": True,
                },
            },
            "notes": [],
        }
    )

    rows = build_scene_level_delta_report([scene])
    row = rows[0]

    assert row["external_generator_status"] == "success"
    assert row["prediction_summary_external_generator"]["object_count"] == 5
    assert row["prediction_delta_external_minus_mock"]["object_count_delta"] == -1
    assert row["prediction_delta_external_minus_mock"]["relation_count_delta"] == 1
    assert row["delta_external_minus_mock"]["structured_violation_count_before_repair"] == -1.0
    assert row["propagation_diagnostics_external_generator"]["command_source"] == "config"


def test_true_v1_only_fallback_only_and_partial_partitioning() -> None:
    scenes = [
        _scene_payload(
            scene_id="scene-true",
            source_type="ply",
            tags=["office"],
            true_v1=True,
            fallback_reason=None,
            failure_labels=["wrong_up_axis"],
            v0_relation_f1=0.30,
            v1_relation_f1=0.35,
        ),
        _scene_payload(
            scene_id="scene-fallback",
            source_type="pcd",
            tags=["lab"],
            true_v1=False,
            fallback_reason="insufficient_plane_evidence",
            failure_labels=["insufficient_plane_evidence"],
            v0_relation_f1=0.40,
            v1_relation_f1=0.40,
        ),
        _scene_payload(
            scene_id="scene-partial",
            source_type="ply",
            tags=["lab"],
            true_v1=False,
            fallback_reason=None,
            partial_calibration=True,
            failure_labels=["manhattan_ambiguity"],
            v0_relation_f1=0.25,
            v1_relation_f1=0.27,
        ),
    ]

    rows = build_scene_level_delta_report(scenes)
    partitions = build_partition_comparison_summaries(rows)

    assert partitions["all_scenes"]["num_scenes"] == 3
    assert partitions["true_v1_execution_scenes"]["num_scenes"] == 1
    assert partitions["fallback_used_scenes"]["num_scenes"] == 1
    assert partitions["partial_calibration_scenes"]["num_scenes"] == 1


def test_trustworthy_comparison_status_generation() -> None:
    partitions = {
        "all_scenes": {"num_scenes": 3},
        "true_v1_execution_scenes": {"num_scenes": 2},
        "fallback_used_scenes": {"num_scenes": 1},
        "partial_calibration_scenes": {"num_scenes": 0},
    }
    status = build_trustworthy_comparison_status(
        comparison_warning=None,
        v1_execution_summary={"num_scene_setting_results": 6, "num_fallback_used": 1},
        partition_summaries=partitions,
    )

    assert status["is_trustworthy_v0_v1_comparison"] is True
    assert status["conclusions_provisional"] is False
    assert status["fallback_only_warning_active"] is False


def test_stratified_summary_generation() -> None:
    scenes = [
        _scene_payload(
            scene_id="scene-a",
            source_type="ply",
            tags=["office", "easy"],
            true_v1=False,
            fallback_reason="insufficient_plane_evidence",
            failure_labels=["insufficient_plane_evidence"],
            v0_relation_f1=0.20,
            v1_relation_f1=0.20,
        ),
        _scene_payload(
            scene_id="scene-b",
            source_type="pcd",
            tags=["office", "hard"],
            true_v1=True,
            fallback_reason=None,
            failure_labels=["manhattan_ambiguity"],
            v0_relation_f1=0.20,
            v1_relation_f1=0.30,
        ),
    ]

    rows = build_scene_level_delta_report(scenes)
    stratified = build_stratified_comparison_summaries(rows)

    source_keys = {item["group_key"] for item in stratified["by_source_type"]}
    tag_keys = {item["group_key"] for item in stratified["by_scene_tag"]}
    fallback_keys = {item["group_key"] for item in stratified["by_fallback_reason"]}
    failure_keys = {item["group_key"] for item in stratified["by_major_failure_category"]}

    assert {"ply", "pcd"}.issubset(source_keys)
    assert {"office", "easy", "hard"}.issubset(tag_keys)
    assert {"none", "insufficient_plane_evidence"}.issubset(fallback_keys)
    assert {"insufficient_plane_evidence", "manhattan_ambiguity"}.issubset(failure_keys)
    assert "by_source_type_true_v1_only" in stratified
    assert "by_source_type_fallback_only" in stratified


def test_next_improvement_decision_artifact_generation() -> None:
    scenes = [
        _scene_payload(
            scene_id="scene-a",
            source_type="ply",
            tags=["office"],
            true_v1=False,
            fallback_reason="insufficient_plane_evidence",
            failure_labels=["insufficient_plane_evidence"],
            v0_relation_f1=0.2,
            v1_relation_f1=0.2,
        ),
        _scene_payload(
            scene_id="scene-b",
            source_type="pcd",
            tags=["office"],
            true_v1=False,
            fallback_reason="insufficient_plane_evidence",
            failure_labels=["insufficient_plane_evidence"],
            v0_relation_f1=0.3,
            v1_relation_f1=0.3,
        ),
    ]

    rows = build_scene_level_delta_report(scenes)
    partitions = build_partition_comparison_summaries(rows)
    recommended = recommend_next_calibration_improvement_target(
        v1_execution_summary={
            "num_scene_setting_results": 4,
            "num_true_v1_execution": 0,
            "num_fallback_used": 4,
        },
        failure_summary={"calibration_failure_counts": {"insufficient_plane_evidence": 2}},
        partition_summaries=partitions,
    )
    status = build_trustworthy_comparison_status(
        comparison_warning="fallback-only warning",
        v1_execution_summary={"num_scene_setting_results": 4, "num_fallback_used": 4},
        partition_summaries=partitions,
    )

    decision = build_next_improvement_decision(
        trustworthy_status=status,
        improvement_target=recommended,
        partition_summaries=partitions,
        failure_summary={"calibration_failure_counts": {"insufficient_plane_evidence": 2}},
    )

    assert decision["target"] in ALLOWED_IMPROVEMENT_TARGETS
    assert decision["target"] == "improve fallback triggering policy"
    assert isinstance(decision["v1_1_success_evidence_checklist"], list)
    assert decision["v1_1_success_evidence_checklist"]


def test_researcher_facing_summary_generation() -> None:
    report_payload = {
        "comparison_warning": None,
        "v1_execution_summary": {
            "num_scene_setting_results": 4,
            "num_true_v1_execution": 2,
            "num_fallback_used": 2,
        },
        "failure_summary": {
            "calibration_failure_counts": {
                "manhattan_ambiguity": 1,
                "insufficient_plane_evidence": 1,
            }
        },
        "v0_v1_comparison_summary": {},
        "scenes": [
            _scene_payload(
                scene_id="scene-a",
                source_type="ply",
                tags=["office", "easy"],
                true_v1=True,
                fallback_reason=None,
                failure_labels=["manhattan_ambiguity"],
                v0_relation_f1=0.20,
                v1_relation_f1=0.30,
                v1_repair_relation_f1=0.35,
            ),
            _scene_payload(
                scene_id="scene-b",
                source_type="pcd",
                tags=["office", "hard"],
                true_v1=False,
                fallback_reason="insufficient_plane_evidence",
                failure_labels=["insufficient_plane_evidence"],
                v0_relation_f1=0.25,
                v1_relation_f1=0.24,
                v1_repair_relation_f1=0.30,
            ),
        ],
    }

    bundle = build_post_true_v1_analysis_bundle(evaluation_report=report_payload)
    summary = build_researcher_facing_summary(
        trustworthy_status=bundle["trustworthy_comparison_status"],
        first_research_result=bundle["first_research_result"],
        partition_summaries=bundle["partition_summaries"],
        stratified_summary=bundle["stratified_summary"],
        scene_rows=bundle["scene_level_deltas"],
        next_improvement_decision=bundle["next_improvement_decision"],
    )

    assert "trustworthy_comparison_status" in summary
    assert "where_v1_improved" in summary
    assert "where_v1_regressed" in summary
    assert "single_highest_priority_v1_1_target" in summary
    assert isinstance(summary["evidence_needed_to_validate_v1_1"], list)
