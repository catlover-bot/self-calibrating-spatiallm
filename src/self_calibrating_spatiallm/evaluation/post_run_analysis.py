"""Post-run v0-v1 analysis helpers for trustworthy calibration comparison."""

from __future__ import annotations

import math
from typing import Any


METRIC_DIRECTIONS: dict[str, str] = {
    "calibration_up_axis_error_deg": "lower_better",
    "calibration_horizontal_error_deg": "lower_better",
    "calibration_reliability": "higher_better",
    "calibration_scale_drift": "lower_better",
    "structured_violation_count_before_repair": "lower_better",
    "actionable_relation_f1": "higher_better",
}

DEFAULT_SCENE_METRICS = [
    "calibration_up_axis_error_deg",
    "calibration_horizontal_error_deg",
    "calibration_up_axis_confidence",
    "calibration_horizontal_confidence",
    "calibration_reliability",
    "calibration_manhattan_ambiguity",
    "calibration_scale_drift",
    "structured_violation_count_before_repair",
    "repair_violations_after",
    "actionable_relation_f1",
]

ALLOWED_IMPROVEMENT_TARGETS = {
    "improve plane candidate scoring",
    "improve plane role assignment",
    "improve confidence thresholding",
    "improve weak scale reasoning",
    "improve fallback triggering policy",
}

_CALIBRATION_FAILURE_PRIORITY = [
    "wrong_up_axis",
    "wrong_horizontal_axis",
    "unstable_up_axis_confidence",
    "manhattan_ambiguity",
    "scale_drift",
    "insufficient_plane_evidence",
    "clutter_dominated_failure",
    "non_manhattan_scene_failure",
]


def build_post_true_v1_analysis_bundle(
    *,
    evaluation_report: dict[str, Any],
) -> dict[str, Any]:
    """Build full post-true-v1 analysis outputs from an evaluation report payload."""
    scenes = evaluation_report.get("scenes", [])
    if not isinstance(scenes, list):
        scenes = []

    scene_rows = build_scene_level_delta_report(scenes)
    partition_summaries = build_partition_comparison_summaries(scene_rows)
    stratified_summary = build_stratified_comparison_summaries(scene_rows)

    comparison_warning = _none_if_empty_str(evaluation_report.get("comparison_warning"))
    v1_execution_summary = _ensure_dict(evaluation_report.get("v1_execution_summary"))
    failure_summary = _ensure_dict(evaluation_report.get("failure_summary"))

    trustworthy_status = build_trustworthy_comparison_status(
        comparison_warning=comparison_warning,
        v1_execution_summary=v1_execution_summary,
        partition_summaries=partition_summaries,
    )

    recommended = _extract_recommended_target(evaluation_report)
    if recommended is None:
        recommended = recommend_next_calibration_improvement_target(
            v1_execution_summary=v1_execution_summary,
            failure_summary=failure_summary,
            partition_summaries=partition_summaries,
        )

    next_improvement_decision = build_next_improvement_decision(
        trustworthy_status=trustworthy_status,
        improvement_target=recommended,
        partition_summaries=partition_summaries,
        failure_summary=failure_summary,
    )

    first_research_result = build_first_research_result_summary(
        comparison_warning=comparison_warning,
        v1_execution_summary=v1_execution_summary,
        partition_summaries=partition_summaries,
        failure_summary=failure_summary,
        improvement_target=next_improvement_decision,
        trustworthy_status=trustworthy_status,
    )

    researcher_summary = build_researcher_facing_summary(
        trustworthy_status=trustworthy_status,
        first_research_result=first_research_result,
        partition_summaries=partition_summaries,
        stratified_summary=stratified_summary,
        scene_rows=scene_rows,
        next_improvement_decision=next_improvement_decision,
    )

    return {
        "scene_level_deltas": scene_rows,
        "partition_summaries": partition_summaries,
        "stratified_summary": stratified_summary,
        "trustworthy_comparison_status": trustworthy_status,
        "next_improvement_decision": next_improvement_decision,
        "first_research_result": first_research_result,
        "researcher_summary": researcher_summary,
    }


def build_scene_level_delta_report(
    scenes: list[dict[str, Any]],
    *,
    metric_directions: dict[str, str] | None = None,
    scene_metric_names: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Build per-scene v0-v1/v1+repair delta rows with calibration execution context."""
    if metric_directions is None:
        metric_directions = METRIC_DIRECTIONS
    if scene_metric_names is None:
        scene_metric_names = list(DEFAULT_SCENE_METRICS)

    rows: list[dict[str, Any]] = []
    for scene in scenes:
        scene_id = str(scene.get("scene_id", "unknown"))
        settings_list = scene.get("settings", [])
        settings = _settings_by_name(settings_list)

        v0_setting = settings.get("calibration_v0")
        v1_setting = settings.get("calibration_v1")
        v1_repair_setting = settings.get("calibration_v1_plus_repair")
        mock_setting = settings.get("mock_generator")
        external_setting = settings.get("external_generator")

        v0_metrics = _ensure_dict(v0_setting.get("metrics") if isinstance(v0_setting, dict) else None)
        v1_metrics = _ensure_dict(v1_setting.get("metrics") if isinstance(v1_setting, dict) else None)
        v1_repair_metrics = _ensure_dict(
            v1_repair_setting.get("metrics") if isinstance(v1_repair_setting, dict) else None
        )
        mock_metrics = _ensure_dict(mock_setting.get("metrics") if isinstance(mock_setting, dict) else None)
        external_metrics = _ensure_dict(
            external_setting.get("metrics") if isinstance(external_setting, dict) else None
        )

        v0_prediction_summary = _extract_prediction_summary(v0_setting)
        v1_prediction_summary = _extract_prediction_summary(v1_setting)
        v1_repair_prediction_summary = _extract_prediction_summary(v1_repair_setting)
        mock_prediction_summary = _extract_prediction_summary(mock_setting)
        external_prediction_summary = _extract_prediction_summary(external_setting)
        v0_propagation = _extract_propagation_diagnostics(v0_setting)
        v1_propagation = _extract_propagation_diagnostics(v1_setting)
        v1_repair_propagation = _extract_propagation_diagnostics(v1_repair_setting)
        mock_propagation = _extract_propagation_diagnostics(mock_setting)
        external_propagation = _extract_propagation_diagnostics(external_setting)
        v0_calibrated_input_summary = _extract_calibrated_input_summary(v0_setting)
        v1_calibrated_input_summary = _extract_calibrated_input_summary(v1_setting)
        v1_repair_calibrated_input_summary = _extract_calibrated_input_summary(v1_repair_setting)
        mock_calibrated_input_summary = _extract_calibrated_input_summary(mock_setting)
        external_calibrated_input_summary = _extract_calibrated_input_summary(external_setting)

        v0_status = str(v0_setting.get("status", "missing")) if isinstance(v0_setting, dict) else "missing"
        v1_status = str(v1_setting.get("status", "missing")) if isinstance(v1_setting, dict) else "missing"
        v1_repair_status = (
            str(v1_repair_setting.get("status", "missing")) if isinstance(v1_repair_setting, dict) else "missing"
        )
        mock_status = str(mock_setting.get("status", "missing")) if isinstance(mock_setting, dict) else "missing"
        external_status = (
            str(external_setting.get("status", "missing")) if isinstance(external_setting, dict) else "missing"
        )
        external_comparable = mock_status == "success" and external_status == "success"

        comparison_valid = v0_status == "success" and v1_status == "success"
        comparison_valid_with_repair = comparison_valid and v1_repair_status == "success"

        execution = _extract_execution(v1_setting)
        fallback_reason = _none_if_empty_str(execution.get("fallback_reason"))
        partial_calibration_applied = _partial_calibration_applied(execution)

        failures = []
        if isinstance(v1_setting, dict):
            raw_failures = v1_setting.get("failures", [])
            if isinstance(raw_failures, list):
                failures = [str(item) for item in raw_failures]
        external_failures = []
        external_notes = []
        if isinstance(external_setting, dict):
            raw_external_failures = external_setting.get("failures", [])
            if isinstance(raw_external_failures, list):
                external_failures = [str(item) for item in raw_external_failures]
            raw_external_notes = external_setting.get("notes", [])
            if isinstance(raw_external_notes, list):
                external_notes = [str(item) for item in raw_external_notes]

        row = {
            "scene_id": scene_id,
            "source_type": scene.get("source_type"),
            "tags": _string_list(scene.get("tags", [])),
            "v0_status": v0_status,
            "v1_status": v1_status,
            "v1_plus_repair_status": v1_repair_status,
            "mock_generator_status": mock_status,
            "external_generator_status": external_status,
            "external_generator_failures": external_failures,
            "external_generator_notes": external_notes,
            "comparison_valid": comparison_valid,
            "comparison_valid_with_repair": comparison_valid_with_repair,
            "true_v1_execution": bool(execution.get("true_v1_execution")),
            "fallback_used": bool(execution.get("fallback_used")),
            "partial_calibration_applied": partial_calibration_applied,
            "fallback_reason": fallback_reason,
            "candidate_plane_count": _int_or_none(execution.get("candidate_plane_count")),
            "calibration_summary": {
                "up_axis_confidence": _float_or_none(
                    execution.get("up_axis_confidence", v1_metrics.get("calibration_up_axis_confidence"))
                ),
                "horizontal_confidence": _float_or_none(
                    execution.get("horizontal_confidence", v1_metrics.get("calibration_horizontal_confidence"))
                ),
                "overall_reliability": _float_or_none(
                    execution.get("overall_reliability", v1_metrics.get("calibration_reliability"))
                ),
                "manhattan_ambiguity": _float_or_none(
                    execution.get("manhattan_ambiguity", v1_metrics.get("calibration_manhattan_ambiguity"))
                ),
                "scale_drift": _float_or_none(
                    execution.get("scale_drift", v1_metrics.get("calibration_scale_drift"))
                ),
            },
            "major_failure_labels": failures,
            "major_failure_category": _major_failure_category(failures),
            "v0_metrics": {key: v0_metrics.get(key) for key in scene_metric_names},
            "v1_metrics": {key: v1_metrics.get(key) for key in scene_metric_names},
            "v1_plus_repair_metrics": {key: v1_repair_metrics.get(key) for key in scene_metric_names},
            "prediction_summary_v0": v0_prediction_summary,
            "prediction_summary_v1": v1_prediction_summary,
            "prediction_summary_v1_plus_repair": v1_repair_prediction_summary,
            "prediction_summary_mock_generator": mock_prediction_summary,
            "prediction_summary_external_generator": external_prediction_summary,
            "prediction_delta_v1_minus_v0": _prediction_summary_delta(v1_prediction_summary, v0_prediction_summary),
            "prediction_delta_v1_plus_repair_minus_v0": _prediction_summary_delta(
                v1_repair_prediction_summary,
                v0_prediction_summary,
            ),
            "prediction_delta_v1_plus_repair_minus_v1": _prediction_summary_delta(
                v1_repair_prediction_summary,
                v1_prediction_summary,
            ),
            "prediction_delta_external_minus_mock": (
                _prediction_summary_delta(
                    external_prediction_summary,
                    mock_prediction_summary,
                )
                if external_comparable
                else None
            ),
            "calibrated_input_summary_v0": v0_calibrated_input_summary,
            "calibrated_input_summary_v1": v1_calibrated_input_summary,
            "calibrated_input_summary_v1_plus_repair": v1_repair_calibrated_input_summary,
            "calibrated_input_summary_mock_generator": mock_calibrated_input_summary,
            "calibrated_input_summary_external_generator": external_calibrated_input_summary,
            "propagation_diagnostics_v0": v0_propagation,
            "propagation_diagnostics_v1": v1_propagation,
            "propagation_diagnostics_v1_plus_repair": v1_repair_propagation,
            "propagation_diagnostics_mock_generator": mock_propagation,
            "propagation_diagnostics_external_generator": external_propagation,
            "external_generator_execution_summary": _extract_generator_execution_summary(external_setting),
            "delta_v1_minus_v0": {},
            "delta_v1_plus_repair_minus_v0": {},
            "delta_v1_plus_repair_minus_v1": {},
            "delta_external_minus_mock": {},
        }

        metric_set = set(scene_metric_names)
        metric_set.update(metric_directions.keys())

        for metric_name in sorted(metric_set):
            row["delta_v1_minus_v0"][metric_name] = _delta(
                v1_metrics.get(metric_name),
                v0_metrics.get(metric_name),
            )
            row["delta_v1_plus_repair_minus_v0"][metric_name] = _delta(
                v1_repair_metrics.get(metric_name),
                v0_metrics.get(metric_name),
            )
            row["delta_v1_plus_repair_minus_v1"][metric_name] = _delta(
                v1_repair_metrics.get(metric_name),
                v1_metrics.get(metric_name),
            )
            row["delta_external_minus_mock"][metric_name] = _delta(
                external_metrics.get(metric_name),
                mock_metrics.get(metric_name),
            )

        rows.append(row)

    return rows


def build_partition_comparison_summaries(
    scene_rows: list[dict[str, Any]],
    *,
    metric_directions: dict[str, str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Summarize all/true-v1/fallback/partial partitions for v0-v1 deltas."""
    if metric_directions is None:
        metric_directions = METRIC_DIRECTIONS

    partitions: dict[str, list[dict[str, Any]]] = {
        "all_scenes": list(scene_rows),
        "true_v1_execution_scenes": [row for row in scene_rows if bool(row.get("true_v1_execution"))],
        "fallback_used_scenes": [row for row in scene_rows if bool(row.get("fallback_used"))],
        "partial_calibration_scenes": [
            row for row in scene_rows if bool(row.get("partial_calibration_applied"))
        ],
    }

    return {
        partition_name: _summarize_rows(rows, metric_directions)
        for partition_name, rows in partitions.items()
    }


def build_stratified_comparison_summaries(
    scene_rows: list[dict[str, Any]],
    *,
    metric_directions: dict[str, str] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Build grouped v0-v1 summaries by source/tags/fallback/failure and partition subsets."""
    if metric_directions is None:
        metric_directions = METRIC_DIRECTIONS

    true_rows = [row for row in scene_rows if bool(row.get("true_v1_execution"))]
    fallback_rows = [row for row in scene_rows if bool(row.get("fallback_used"))]

    return {
        "by_source_type": _group_summary(
            scene_rows,
            metric_directions,
            lambda row: str(row.get("source_type") or "unknown"),
        ),
        "by_scene_tag": _tag_group_summary(scene_rows, metric_directions),
        "by_fallback_reason": _group_summary(
            scene_rows,
            metric_directions,
            lambda row: str(row.get("fallback_reason") or "none"),
        ),
        "by_major_failure_category": _group_summary(
            scene_rows,
            metric_directions,
            lambda row: str(row.get("major_failure_category") or "unknown"),
        ),
        "by_source_type_true_v1_only": _group_summary(
            true_rows,
            metric_directions,
            lambda row: str(row.get("source_type") or "unknown"),
        ),
        "by_source_type_fallback_only": _group_summary(
            fallback_rows,
            metric_directions,
            lambda row: str(row.get("source_type") or "unknown"),
        ),
        "by_scene_tag_true_v1_only": _tag_group_summary(true_rows, metric_directions),
        "by_scene_tag_fallback_only": _tag_group_summary(fallback_rows, metric_directions),
    }


def build_trustworthy_comparison_status(
    *,
    comparison_warning: str | None,
    v1_execution_summary: dict[str, Any],
    partition_summaries: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Build explicit trustworthiness status for v0-v1 comparison interpretation."""
    true_summary = _ensure_dict(partition_summaries.get("true_v1_execution_scenes"))
    fallback_summary = _ensure_dict(partition_summaries.get("fallback_used_scenes"))
    partial_summary = _ensure_dict(partition_summaries.get("partial_calibration_scenes"))

    true_scene_count = _as_int(true_summary.get("num_scenes"))
    fallback_scene_count = _as_int(fallback_summary.get("num_scenes"))
    partial_scene_count = _as_int(partial_summary.get("num_scenes"))

    warning_text = comparison_warning or ""
    fallback_only_warning_active = (
        (true_scene_count == 0 and fallback_scene_count > 0)
        or "fallback-only" in warning_text.lower()
        or "not a valid v0-v1" in warning_text.lower()
    )

    scene_setting_total = _as_int(v1_execution_summary.get("num_scene_setting_results"))
    scene_setting_fallback = _as_int(v1_execution_summary.get("num_fallback_used"))
    fallback_ratio = (
        float(scene_setting_fallback) / float(scene_setting_total)
        if scene_setting_total > 0
        else 0.0
    )

    trustworthy = true_scene_count > 0 and comparison_warning is None and not fallback_only_warning_active
    provisional = not trustworthy

    if trustworthy:
        status_note = "Trustworthy: true-v1 scenes are available and no fallback-heavy warning is active."
    elif fallback_only_warning_active:
        status_note = "Provisional: fallback-only or invalid-comparison warning is active."
    else:
        status_note = "Provisional: trustworthiness criteria are not fully satisfied."

    return {
        "is_trustworthy_v0_v1_comparison": trustworthy,
        "num_true_v1_execution_scenes": true_scene_count,
        "num_fallback_scenes": fallback_scene_count,
        "num_partial_calibration_scenes": partial_scene_count,
        "fallback_only_warning_active": fallback_only_warning_active,
        "conclusions_provisional": provisional,
        "comparison_warning": comparison_warning,
        "scene_setting_fallback_ratio": fallback_ratio,
        "status_note": status_note,
    }


def recommend_next_calibration_improvement_target(
    *,
    v1_execution_summary: dict[str, Any],
    failure_summary: dict[str, Any],
    partition_summaries: dict[str, dict[str, Any]],
) -> dict[str, str]:
    """Recommend one prioritized v1.1 improvement target from observed run behavior."""
    total = _as_int(v1_execution_summary.get("num_scene_setting_results"))
    true_count = _as_int(v1_execution_summary.get("num_true_v1_execution"))
    fallback_count = _as_int(v1_execution_summary.get("num_fallback_used"))

    fallback_ratio = (fallback_count / total) if total > 0 else 0.0

    calibration_failures = _ensure_dict(failure_summary.get("calibration_failure_counts"))
    wrong_axis = _as_int(calibration_failures.get("wrong_up_axis")) + _as_int(
        calibration_failures.get("wrong_horizontal_axis")
    )
    ambiguity = _as_int(calibration_failures.get("unstable_up_axis_confidence")) + _as_int(
        calibration_failures.get("manhattan_ambiguity")
    )
    insufficient_plane = _as_int(calibration_failures.get("insufficient_plane_evidence")) + _as_int(
        calibration_failures.get("clutter_dominated_failure")
    )
    scale_failures = _as_int(calibration_failures.get("scale_drift"))

    true_summary = _ensure_dict(partition_summaries.get("true_v1_execution_scenes"))
    true_regressions = _string_list(true_summary.get("regressed_metrics", []))

    if total > 0 and true_count == 0:
        return {
            "target": "improve fallback triggering policy",
            "reason": "All v1 scene-setting results used fallback, so true plane-aware behavior was not evaluated.",
        }
    if fallback_ratio >= 0.4:
        return {
            "target": "improve fallback triggering policy",
            "reason": "Fallback ratio is high, limiting trustworthy evidence from true-v1 execution.",
        }
    if scale_failures > 0:
        return {
            "target": "improve weak scale reasoning",
            "reason": "Scale-related failures are present in calibration failure taxonomy.",
        }
    if wrong_axis > 0:
        return {
            "target": "improve plane role assignment",
            "reason": "Axis-estimation failures suggest floor/wall role inference errors.",
        }
    if ambiguity > 0 or any(
        name in true_regressions for name in ["calibration_reliability", "actionable_relation_f1"]
    ):
        return {
            "target": "improve confidence thresholding",
            "reason": "Confidence/ambiguity signals suggest unreliable normalization decisions.",
        }
    if insufficient_plane > 0:
        return {
            "target": "improve plane candidate scoring",
            "reason": "Insufficient or cluttered plane evidence indicates weaker candidate selection.",
        }

    return {
        "target": "improve plane candidate scoring",
        "reason": "Default priority: strengthen plane candidate quality before other refinements.",
    }


def build_v11_success_evidence_checklist(
    *,
    target: str,
    trustworthy_status: dict[str, Any],
) -> list[str]:
    """Return compact evidence checklist for validating chosen v1.1 target."""
    fallback_note = "fewer fallback scenes with stable true-v1 execution coverage"
    base = [
        "improved actionable_relation_f1 on true-v1 scenes",
        "no regression in structured_violation_count_before_repair",
    ]

    if target == "improve fallback triggering policy":
        return [
            fallback_note,
            "fallback_only_warning_active becomes false",
            "num_true_v1_execution_scenes increases",
            "v0-v1 trustworthiness becomes non-provisional",
        ]
    if target == "improve plane candidate scoring":
        return [
            "higher candidate_plane_count quality with fewer clutter-driven failures",
            "lower calibration_up_axis_error_deg and calibration_horizontal_error_deg",
            "reduced insufficient_plane_evidence and clutter_dominated_failure counts",
        ] + base
    if target == "improve plane role assignment":
        return [
            "lower wrong_up_axis and wrong_horizontal_axis failure counts",
            "lower calibration_up_axis_error_deg and calibration_horizontal_error_deg",
            "higher calibration_reliability on true-v1 scenes",
        ] + base
    if target == "improve confidence thresholding":
        return [
            "lower unstable_up_axis_confidence and manhattan_ambiguity failure counts",
            "higher calibration_reliability without increasing fallback ratio",
            "fewer relation_derivation_failure labels on true-v1 scenes",
        ] + base
    if target == "improve weak scale reasoning":
        return [
            "lower calibration_scale_drift",
            "lower implausible_object_sizes failure counts",
            "fewer scale-related regressions in scene-level deltas",
        ] + base

    if not trustworthy_status.get("is_trustworthy_v0_v1_comparison"):
        return [fallback_note] + base
    return base


def build_next_improvement_decision(
    *,
    trustworthy_status: dict[str, Any],
    improvement_target: dict[str, str],
    partition_summaries: dict[str, dict[str, Any]],
    failure_summary: dict[str, Any],
) -> dict[str, Any]:
    """Build explicit single-target v1.1 decision artifact from run outputs."""
    target = str(improvement_target.get("target", "")).strip().lower()
    reason = str(improvement_target.get("reason", "")).strip()

    if target not in ALLOWED_IMPROVEMENT_TARGETS:
        target = "improve plane candidate scoring"
        if not reason:
            reason = "Fallback to default improvement target due to invalid recommendation payload."

    true_summary = _ensure_dict(partition_summaries.get("true_v1_execution_scenes"))
    fallback_summary = _ensure_dict(partition_summaries.get("fallback_used_scenes"))

    top_failures = _top_categories(failure_summary, "calibration_failure_counts", limit=5)
    checklist = build_v11_success_evidence_checklist(
        target=target,
        trustworthy_status=trustworthy_status,
    )

    decision_basis = (
        "true_v1_execution_scenes"
        if trustworthy_status.get("is_trustworthy_v0_v1_comparison")
        else "provisional_all_scenes"
    )

    return {
        "target": target,
        "reason": reason,
        "decision_basis": decision_basis,
        "is_provisional": bool(trustworthy_status.get("conclusions_provisional", True)),
        "trustworthy_comparison_status": trustworthy_status,
        "supporting_signals": {
            "true_v1_summary": true_summary,
            "fallback_summary": fallback_summary,
            "top_calibration_failure_categories": top_failures,
        },
        "v1_1_success_evidence_checklist": checklist,
    }


def build_first_research_result_summary(
    *,
    comparison_warning: str | None,
    v1_execution_summary: dict[str, Any],
    partition_summaries: dict[str, dict[str, Any]],
    failure_summary: dict[str, Any],
    improvement_target: dict[str, Any],
    trustworthy_status: dict[str, Any],
) -> dict[str, Any]:
    """Compose a concise first-result summary for v0-v1 interpretation."""
    true_summary = _ensure_dict(partition_summaries.get("true_v1_execution_scenes"))
    all_summary = _ensure_dict(partition_summaries.get("all_scenes"))

    trustworthy = bool(trustworthy_status.get("is_trustworthy_v0_v1_comparison"))

    if trustworthy and _as_int(true_summary.get("num_scenes")) > 0:
        basis = "true_v1_execution_scenes"
        selected = true_summary
    else:
        basis = "all_scenes"
        selected = all_summary

    top_failures = _top_categories(failure_summary, "calibration_failure_counts", limit=5)

    return {
        "comparison_trustworthy": trustworthy,
        "trustworthiness_note": trustworthy_status.get("status_note")
        or comparison_warning
        or "Trustworthy v0-v1 comparison: true-v1 scenes are available.",
        "comparison_basis": basis,
        "num_true_v1_scenes": _as_int(true_summary.get("num_scenes")),
        "num_fallback_scenes": _as_int(
            _ensure_dict(partition_summaries.get("fallback_used_scenes")).get("num_scenes")
        ),
        "num_partial_calibration_scenes": _as_int(
            _ensure_dict(partition_summaries.get("partial_calibration_scenes")).get("num_scenes")
        ),
        "fallback_only_warning_active": bool(trustworthy_status.get("fallback_only_warning_active")),
        "conclusions_provisional": bool(trustworthy_status.get("conclusions_provisional")),
        "improved_metrics": _string_list(selected.get("improved_metrics", [])),
        "regressed_metrics": _string_list(selected.get("regressed_metrics", [])),
        "top_failure_categories": top_failures,
        "recommended_next_calibration_improvement_target": improvement_target.get("target"),
        "recommended_next_calibration_improvement_reason": improvement_target.get("reason"),
        "v1_execution_summary": v1_execution_summary,
    }


def build_researcher_facing_summary(
    *,
    trustworthy_status: dict[str, Any],
    first_research_result: dict[str, Any],
    partition_summaries: dict[str, dict[str, Any]],
    stratified_summary: dict[str, list[dict[str, Any]]],
    scene_rows: list[dict[str, Any]],
    next_improvement_decision: dict[str, Any],
) -> dict[str, Any]:
    """Build compact researcher-first summary artifact for post-run interpretation."""
    improved_metrics = _string_list(first_research_result.get("improved_metrics", []))
    regressed_metrics = _string_list(first_research_result.get("regressed_metrics", []))

    benefited_scenes, failed_scenes = _rank_scene_deltas(scene_rows)
    strata_benefit, strata_fail = _rank_strata(stratified_summary)

    return {
        "trustworthy_comparison_status": trustworthy_status,
        "where_v1_improved": improved_metrics,
        "where_v1_regressed": regressed_metrics,
        "scenes_benefited_most": benefited_scenes,
        "scenes_failed_most": failed_scenes,
        "strata_benefited_most": strata_benefit,
        "strata_failed_most": strata_fail,
        "single_highest_priority_v1_1_target": next_improvement_decision.get("target"),
        "target_selection_reason": next_improvement_decision.get("reason"),
        "evidence_needed_to_validate_v1_1": _string_list(
            next_improvement_decision.get("v1_1_success_evidence_checklist", [])
        ),
        "partition_overview": partition_summaries,
    }


def render_scene_level_delta_markdown(scene_rows: list[dict[str, Any]]) -> str:
    lines = ["# Scene-Level v0-v1 Delta Report", ""]
    if not scene_rows:
        lines.append("- no scenes available")
        return "\n".join(lines)

    for row in scene_rows:
        lines.append(f"## {row.get('scene_id')}")
        lines.append(f"- source_type: `{row.get('source_type')}`")
        lines.append(f"- tags: `{row.get('tags')}`")
        lines.append(f"- true_v1_execution: `{row.get('true_v1_execution')}`")
        lines.append(f"- fallback_used: `{row.get('fallback_used')}`")
        lines.append(f"- partial_calibration_applied: `{row.get('partial_calibration_applied')}`")
        lines.append(f"- mock_generator_status: `{row.get('mock_generator_status')}`")
        lines.append(f"- external_generator_status: `{row.get('external_generator_status')}`")
        lines.append(f"- external_generator_failures: `{row.get('external_generator_failures')}`")
        lines.append(f"- external_generator_notes: `{row.get('external_generator_notes')}`")
        lines.append(f"- fallback_reason: `{row.get('fallback_reason')}`")
        lines.append(f"- candidate_plane_count: `{row.get('candidate_plane_count')}`")
        lines.append(f"- calibration_summary: `{row.get('calibration_summary')}`")
        lines.append(f"- major_failure_labels: `{row.get('major_failure_labels')}`")
        lines.append(f"- major_failure_category: `{row.get('major_failure_category')}`")
        lines.append(f"- comparison_valid: `{row.get('comparison_valid')}`")
        lines.append(f"- prediction_summary_v0: `{row.get('prediction_summary_v0')}`")
        lines.append(f"- prediction_summary_v1: `{row.get('prediction_summary_v1')}`")
        lines.append(
            f"- prediction_summary_v1_plus_repair: `{row.get('prediction_summary_v1_plus_repair')}`"
        )
        lines.append(f"- prediction_delta_v1_minus_v0: `{row.get('prediction_delta_v1_minus_v0')}`")
        lines.append(
            f"- prediction_delta_v1_plus_repair_minus_v0: `{row.get('prediction_delta_v1_plus_repair_minus_v0')}`"
        )
        lines.append(
            f"- prediction_delta_v1_plus_repair_minus_v1: `{row.get('prediction_delta_v1_plus_repair_minus_v1')}`"
        )
        lines.append(f"- prediction_summary_mock_generator: `{row.get('prediction_summary_mock_generator')}`")
        lines.append(
            f"- prediction_summary_external_generator: `{row.get('prediction_summary_external_generator')}`"
        )
        lines.append(
            f"- prediction_delta_external_minus_mock: `{row.get('prediction_delta_external_minus_mock')}`"
        )
        lines.append(f"- calibrated_input_summary_v0: `{row.get('calibrated_input_summary_v0')}`")
        lines.append(f"- calibrated_input_summary_v1: `{row.get('calibrated_input_summary_v1')}`")
        lines.append(
            f"- calibrated_input_summary_mock_generator: `{row.get('calibrated_input_summary_mock_generator')}`"
        )
        lines.append(
            f"- calibrated_input_summary_external_generator: `{row.get('calibrated_input_summary_external_generator')}`"
        )
        lines.append(
            f"- propagation_diagnostics_v0: `{row.get('propagation_diagnostics_v0')}`"
        )
        lines.append(
            f"- propagation_diagnostics_v1: `{row.get('propagation_diagnostics_v1')}`"
        )
        lines.append(
            f"- propagation_diagnostics_mock_generator: `{row.get('propagation_diagnostics_mock_generator')}`"
        )
        lines.append(
            f"- propagation_diagnostics_external_generator: `{row.get('propagation_diagnostics_external_generator')}`"
        )
        lines.append(
            f"- external_generator_execution_summary: `{row.get('external_generator_execution_summary')}`"
        )

        lines.append("- calibration_v0_metrics:")
        for key, value in sorted(_ensure_dict(row.get("v0_metrics")).items()):
            lines.append(f"- v0 {key}: `{value}`")

        lines.append("- calibration_v1_metrics:")
        for key, value in sorted(_ensure_dict(row.get("v1_metrics")).items()):
            lines.append(f"- v1 {key}: `{value}`")

        lines.append("- calibration_v1_plus_repair_metrics:")
        for key, value in sorted(_ensure_dict(row.get("v1_plus_repair_metrics")).items()):
            lines.append(f"- v1_plus_repair {key}: `{value}`")

        lines.append("- delta_v1_minus_v0:")
        for key, value in sorted(_ensure_dict(row.get("delta_v1_minus_v0")).items()):
            lines.append(f"- delta_v1_minus_v0 {key}: `{value}`")

        lines.append("- delta_v1_plus_repair_minus_v0:")
        for key, value in sorted(_ensure_dict(row.get("delta_v1_plus_repair_minus_v0")).items()):
            lines.append(f"- delta_v1_plus_repair_minus_v0 {key}: `{value}`")

        lines.append("- delta_v1_plus_repair_minus_v1:")
        for key, value in sorted(_ensure_dict(row.get("delta_v1_plus_repair_minus_v1")).items()):
            lines.append(f"- delta_v1_plus_repair_minus_v1 {key}: `{value}`")
        lines.append("- delta_external_minus_mock:")
        for key, value in sorted(_ensure_dict(row.get("delta_external_minus_mock")).items()):
            lines.append(f"- delta_external_minus_mock {key}: `{value}`")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def render_stratified_summary_markdown(stratified: dict[str, list[dict[str, Any]]]) -> str:
    lines = ["# Stratified v0-v1 Comparison Summary", ""]
    if not stratified:
        lines.append("- no stratified summaries available")
        return "\n".join(lines)

    for section, rows in stratified.items():
        lines.append(f"## {section}")
        if not rows:
            lines.append("- none")
            lines.append("")
            continue
        for item in rows:
            lines.append(
                f"- group `{item.get('group_key')}` num_scenes={item.get('num_scenes')} "
                f"num_comparable={item.get('num_comparable')}"
            )
            lines.append(
                f"- group `{item.get('group_key')}` num_true_v1_execution={item.get('num_true_v1_execution')} "
                f"num_fallback_used={item.get('num_fallback_used')} "
                f"num_partial_calibration={item.get('num_partial_calibration')}"
            )
            lines.append(
                f"- group `{item.get('group_key')}` improved_metrics={item.get('improved_metrics')}"
            )
            lines.append(
                f"- group `{item.get('group_key')}` regressed_metrics={item.get('regressed_metrics')}"
            )
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def render_first_research_result_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# First Research Result Summary",
        "",
        f"- comparison_trustworthy: `{summary.get('comparison_trustworthy')}`",
        f"- trustworthiness_note: `{summary.get('trustworthiness_note')}`",
        f"- comparison_basis: `{summary.get('comparison_basis')}`",
        f"- num_true_v1_scenes: `{summary.get('num_true_v1_scenes')}`",
        f"- num_fallback_scenes: `{summary.get('num_fallback_scenes')}`",
        f"- num_partial_calibration_scenes: `{summary.get('num_partial_calibration_scenes')}`",
        f"- fallback_only_warning_active: `{summary.get('fallback_only_warning_active')}`",
        f"- conclusions_provisional: `{summary.get('conclusions_provisional')}`",
        f"- improved_metrics: `{summary.get('improved_metrics')}`",
        f"- regressed_metrics: `{summary.get('regressed_metrics')}`",
        f"- top_failure_categories: `{summary.get('top_failure_categories')}`",
        f"- recommended_next_calibration_improvement_target: "
        f"`{summary.get('recommended_next_calibration_improvement_target')}`",
        f"- recommended_next_calibration_improvement_reason: "
        f"`{summary.get('recommended_next_calibration_improvement_reason')}`",
    ]
    return "\n".join(lines) + "\n"


def render_trustworthy_comparison_status_markdown(status: dict[str, Any]) -> str:
    lines = [
        "# Trustworthy Comparison Status",
        "",
        f"- is_trustworthy_v0_v1_comparison: `{status.get('is_trustworthy_v0_v1_comparison')}`",
        f"- num_true_v1_execution_scenes: `{status.get('num_true_v1_execution_scenes')}`",
        f"- num_fallback_scenes: `{status.get('num_fallback_scenes')}`",
        f"- num_partial_calibration_scenes: `{status.get('num_partial_calibration_scenes')}`",
        f"- fallback_only_warning_active: `{status.get('fallback_only_warning_active')}`",
        f"- conclusions_provisional: `{status.get('conclusions_provisional')}`",
        f"- scene_setting_fallback_ratio: `{status.get('scene_setting_fallback_ratio')}`",
        f"- status_note: `{status.get('status_note')}`",
    ]
    warning = _none_if_empty_str(status.get("comparison_warning"))
    if warning:
        lines.extend(["", "## Warning", f"- {warning}"])
    return "\n".join(lines) + "\n"


def render_next_improvement_decision_markdown(decision: dict[str, Any]) -> str:
    lines = [
        "# Next Improvement Decision",
        "",
        f"- target: `{decision.get('target')}`",
        f"- reason: `{decision.get('reason')}`",
        f"- decision_basis: `{decision.get('decision_basis')}`",
        f"- is_provisional: `{decision.get('is_provisional')}`",
        f"- supporting_signals: `{decision.get('supporting_signals')}`",
        "",
        "## v1.1 Success Evidence Checklist",
    ]

    checklist = decision.get("v1_1_success_evidence_checklist", [])
    if isinstance(checklist, list) and checklist:
        for item in checklist:
            lines.append(f"- {item}")
    else:
        lines.append("- none")

    return "\n".join(lines) + "\n"


def render_researcher_summary_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Researcher Summary",
        "",
        "## Trustworthiness",
        f"- status: `{summary.get('trustworthy_comparison_status')}`",
        "",
        "## Improvements",
        f"- where_v1_improved: `{summary.get('where_v1_improved')}`",
        "",
        "## Regressions",
        f"- where_v1_regressed: `{summary.get('where_v1_regressed')}`",
        "",
        "## Benefited Scenes",
        f"- scenes_benefited_most: `{summary.get('scenes_benefited_most')}`",
        "",
        "## Failed Scenes",
        f"- scenes_failed_most: `{summary.get('scenes_failed_most')}`",
        "",
        "## Benefited Strata",
        f"- strata_benefited_most: `{summary.get('strata_benefited_most')}`",
        "",
        "## Failed Strata",
        f"- strata_failed_most: `{summary.get('strata_failed_most')}`",
        "",
        "## Next v1.1 Target",
        f"- single_highest_priority_v1_1_target: `{summary.get('single_highest_priority_v1_1_target')}`",
        f"- target_selection_reason: `{summary.get('target_selection_reason')}`",
        "",
        "## Evidence Needed",
    ]

    checklist = summary.get("evidence_needed_to_validate_v1_1", [])
    if isinstance(checklist, list) and checklist:
        for item in checklist:
            lines.append(f"- {item}")
    else:
        lines.append("- none")

    return "\n".join(lines) + "\n"


def _summarize_rows(rows: list[dict[str, Any]], metric_directions: dict[str, str]) -> dict[str, Any]:
    comparable_rows = [row for row in rows if bool(row.get("comparison_valid"))]
    avg_deltas: dict[str, float | None] = {}

    improved_metrics: list[str] = []
    regressed_metrics: list[str] = []

    for metric_name, direction in metric_directions.items():
        values: list[float] = []
        for row in comparable_rows:
            delta_map = _ensure_dict(row.get("delta_v1_minus_v0"))
            delta = delta_map.get(metric_name)
            if isinstance(delta, (int, float)) and math.isfinite(float(delta)):
                values.append(float(delta))

        avg_delta = (sum(values) / len(values)) if values else None
        avg_deltas[metric_name] = avg_delta
        if avg_delta is None:
            continue

        if direction == "higher_better":
            if avg_delta > 1e-8:
                improved_metrics.append(metric_name)
            elif avg_delta < -1e-8:
                regressed_metrics.append(metric_name)
        else:
            if avg_delta < -1e-8:
                improved_metrics.append(metric_name)
            elif avg_delta > 1e-8:
                regressed_metrics.append(metric_name)

    fallback_reasons: dict[str, int] = {}
    for row in rows:
        reason = str(row.get("fallback_reason") or "none")
        fallback_reasons[reason] = fallback_reasons.get(reason, 0) + 1

    return {
        "num_scenes": len(rows),
        "num_comparable": len(comparable_rows),
        "scene_ids": sorted({str(row.get("scene_id", "unknown")) for row in rows}),
        "num_true_v1_execution": sum(1 for row in rows if bool(row.get("true_v1_execution"))),
        "num_fallback_used": sum(1 for row in rows if bool(row.get("fallback_used"))),
        "num_partial_calibration": sum(
            1 for row in rows if bool(row.get("partial_calibration_applied"))
        ),
        "avg_metric_deltas_v1_minus_v0": avg_deltas,
        "improved_metrics": sorted(improved_metrics),
        "regressed_metrics": sorted(regressed_metrics),
        "fallback_reason_counts": fallback_reasons,
    }


def _group_summary(
    rows: list[dict[str, Any]],
    metric_directions: dict[str, str],
    key_fn: Any,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        key = str(key_fn(row))
        grouped.setdefault(key, []).append(row)

    output: list[dict[str, Any]] = []
    for group_key in sorted(grouped.keys()):
        summary = _summarize_rows(grouped[group_key], metric_directions)
        summary["group_key"] = group_key
        output.append(summary)
    return output


def _tag_group_summary(
    rows: list[dict[str, Any]],
    metric_directions: dict[str, str],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        tags = _string_list(row.get("tags", []))
        if not tags:
            grouped.setdefault("untagged", []).append(row)
            continue
        for tag in tags:
            grouped.setdefault(tag, []).append(row)

    output: list[dict[str, Any]] = []
    for group_key in sorted(grouped.keys()):
        summary = _summarize_rows(grouped[group_key], metric_directions)
        summary["group_key"] = group_key
        output.append(summary)
    return output


def _settings_by_name(settings_list: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(settings_list, list):
        return {}

    out: dict[str, dict[str, Any]] = {}
    for setting in settings_list:
        if not isinstance(setting, dict):
            continue
        name = setting.get("setting_name")
        if isinstance(name, str):
            out[name] = setting
    return out


def _extract_execution(v1_setting: Any) -> dict[str, Any]:
    if not isinstance(v1_setting, dict):
        return {}
    metadata = _ensure_dict(v1_setting.get("metadata"))
    execution = _ensure_dict(metadata.get("calibration_execution"))
    return execution


def _extract_prediction_summary(setting: Any) -> dict[str, Any]:
    if not isinstance(setting, dict):
        return {}
    metadata = _ensure_dict(setting.get("metadata"))
    summary = _ensure_dict(metadata.get("prediction_summary_pre_repair"))
    return {
        "object_count": _as_int(summary.get("object_count")),
        "relation_count": _as_int(summary.get("relation_count")),
        "object_labels": sorted(_string_list(summary.get("object_labels", []))),
        "relation_predicates": sorted(_string_list(summary.get("relation_predicates", []))),
    }


def _extract_calibrated_input_summary(setting: Any) -> dict[str, Any]:
    if not isinstance(setting, dict):
        return {}
    metadata = _ensure_dict(setting.get("metadata"))
    summary = _ensure_dict(metadata.get("calibrated_input_summary"))
    return {
        "frame": summary.get("frame"),
        "num_points": _as_int(summary.get("num_points")),
        "ranges": summary.get("ranges"),
        "center": summary.get("center"),
        "up_vector": summary.get("up_vector"),
        "horizontal_axis": summary.get("horizontal_axis"),
    }


def _extract_propagation_diagnostics(setting: Any) -> dict[str, Any]:
    if not isinstance(setting, dict):
        return {}
    metadata = _ensure_dict(setting.get("metadata"))
    diagnostics = _ensure_dict(metadata.get("propagation_diagnostics"))
    calibration_signal = _ensure_dict(diagnostics.get("calibration_signal"))
    return {
        "generator_mode": diagnostics.get("generator_mode"),
        "include_box": diagnostics.get("include_box"),
        "layout_changes": _string_list(diagnostics.get("layout_changes", [])),
        "command_source": diagnostics.get("command_source"),
        "payload_source": diagnostics.get("payload_source"),
        "parse_mode": diagnostics.get("parse_mode"),
        "partial_parse": diagnostics.get("partial_parse"),
        "parse_warning_count": _as_int(diagnostics.get("parse_warning_count")),
        "spatiallm_export_summary": _ensure_dict(diagnostics.get("spatiallm_export_summary")),
        "prediction_summary": _ensure_dict(diagnostics.get("prediction_summary")),
        "calibration_signal": {
            "up_confidence": _float_or_none(calibration_signal.get("up_confidence")),
            "horizontal_confidence": _float_or_none(calibration_signal.get("horizontal_confidence")),
            "overall_reliability": _float_or_none(calibration_signal.get("overall_reliability")),
            "manhattan_ambiguity": _float_or_none(calibration_signal.get("manhattan_ambiguity")),
            "signal_strength": _float_or_none(calibration_signal.get("signal_strength")),
        },
    }


def _extract_generator_execution_summary(setting: Any) -> dict[str, Any]:
    if not isinstance(setting, dict):
        return {}
    metadata = _ensure_dict(setting.get("metadata"))
    summary = _ensure_dict(metadata.get("generator_execution_summary"))
    return summary


def _prediction_summary_delta(current: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    current_object_count = _as_int(current.get("object_count"))
    baseline_object_count = _as_int(baseline.get("object_count"))
    current_relation_count = _as_int(current.get("relation_count"))
    baseline_relation_count = _as_int(baseline.get("relation_count"))

    current_labels = set(_string_list(current.get("object_labels", [])))
    baseline_labels = set(_string_list(baseline.get("object_labels", [])))
    current_predicates = set(_string_list(current.get("relation_predicates", [])))
    baseline_predicates = set(_string_list(baseline.get("relation_predicates", [])))

    return {
        "object_count_delta": current_object_count - baseline_object_count,
        "relation_count_delta": current_relation_count - baseline_relation_count,
        "added_labels": sorted(current_labels - baseline_labels),
        "removed_labels": sorted(baseline_labels - current_labels),
        "added_predicates": sorted(current_predicates - baseline_predicates),
        "removed_predicates": sorted(baseline_predicates - current_predicates),
    }


def _extract_recommended_target(evaluation_report: dict[str, Any]) -> dict[str, str] | None:
    comparison = _ensure_dict(evaluation_report.get("v0_v1_comparison_summary"))
    recommendation = _ensure_dict(comparison.get("recommended_next_calibration_improvement"))
    target = _none_if_empty_str(recommendation.get("target"))
    reason = _none_if_empty_str(recommendation.get("reason"))
    if target:
        return {"target": target, "reason": reason or ""}
    return None


def _partial_calibration_applied(execution: dict[str, Any]) -> bool:
    if isinstance(execution.get("partial_calibration_applied"), bool):
        return bool(execution.get("partial_calibration_applied"))
    if isinstance(execution.get("partial_calibration"), bool):
        return bool(execution.get("partial_calibration"))

    true_v1 = bool(execution.get("true_v1_execution"))
    fallback_used = bool(execution.get("fallback_used"))
    return bool(execution) and (not true_v1) and (not fallback_used)


def _major_failure_category(failures: list[str]) -> str:
    if not failures:
        return "no_major_failure_detected"

    for label in _CALIBRATION_FAILURE_PRIORITY:
        if label in failures:
            return label
    return failures[0]


def _delta(current: Any, baseline: Any) -> float | None:
    if not isinstance(current, (int, float)) or not isinstance(baseline, (int, float)):
        return None
    current_f = float(current)
    baseline_f = float(baseline)
    if not (math.isfinite(current_f) and math.isfinite(baseline_f)):
        return None
    return current_f - baseline_f


def _rank_scene_deltas(scene_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    scored: list[tuple[float, dict[str, Any]]] = []
    for row in scene_rows:
        if not bool(row.get("comparison_valid")):
            continue
        deltas = _ensure_dict(row.get("delta_v1_minus_v0"))
        relation = _float_or_none(deltas.get("actionable_relation_f1")) or 0.0
        violation = _float_or_none(deltas.get("structured_violation_count_before_repair")) or 0.0
        reliability = _float_or_none(deltas.get("calibration_reliability")) or 0.0
        score = relation + (0.5 * reliability) - violation
        scored.append(
            (
                score,
                {
                    "scene_id": row.get("scene_id"),
                    "score": score,
                    "true_v1_execution": row.get("true_v1_execution"),
                    "fallback_used": row.get("fallback_used"),
                    "delta_relation_f1": relation,
                    "delta_violation_before": violation,
                    "major_failure_category": row.get("major_failure_category"),
                },
            )
        )

    scored_sorted = sorted(scored, key=lambda item: item[0], reverse=True)
    benefited = [item[1] for item in scored_sorted[:3]]
    failed = [item[1] for item in list(reversed(scored_sorted))[:3]]
    return benefited, failed


def _rank_strata(
    stratified_summary: dict[str, list[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    candidates = stratified_summary.get("by_scene_tag")
    if not isinstance(candidates, list) or not candidates:
        candidates = stratified_summary.get("by_source_type")
    if not isinstance(candidates, list):
        candidates = []

    scored: list[tuple[float, dict[str, Any]]] = []
    for item in candidates:
        if not isinstance(item, dict):
            continue
        avg = _ensure_dict(item.get("avg_metric_deltas_v1_minus_v0"))
        relation = _float_or_none(avg.get("actionable_relation_f1")) or 0.0
        reliability = _float_or_none(avg.get("calibration_reliability")) or 0.0
        up_err = _float_or_none(avg.get("calibration_up_axis_error_deg")) or 0.0
        horiz_err = _float_or_none(avg.get("calibration_horizontal_error_deg")) or 0.0
        violations = _float_or_none(avg.get("structured_violation_count_before_repair")) or 0.0

        score = relation + (0.5 * reliability) - (0.5 * up_err) - (0.5 * horiz_err) - violations
        scored.append(
            (
                score,
                {
                    "group_key": item.get("group_key"),
                    "score": score,
                    "num_scenes": item.get("num_scenes"),
                    "num_true_v1_execution": item.get("num_true_v1_execution"),
                    "num_fallback_used": item.get("num_fallback_used"),
                    "improved_metrics": item.get("improved_metrics"),
                    "regressed_metrics": item.get("regressed_metrics"),
                },
            )
        )

    scored_sorted = sorted(scored, key=lambda item: item[0], reverse=True)
    benefited = [item[1] for item in scored_sorted[:3]]
    failed = [item[1] for item in list(reversed(scored_sorted))[:3]]
    return benefited, failed


def _top_categories(summary: dict[str, Any], key: str, limit: int = 5) -> list[list[Any]]:
    values = _ensure_dict(summary.get(key))
    items = sorted(
        ((str(name), _as_int(count)) for name, count in values.items()),
        key=lambda item: item[1],
        reverse=True,
    )
    return [[name, count] for name, count in items[:limit]]


def _string_list(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    return [str(value) for value in values]


def _ensure_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _none_if_empty_str(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value
    return None


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _float_or_none(value: Any) -> float | None:
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        return None
    return value_f if math.isfinite(value_f) else None
