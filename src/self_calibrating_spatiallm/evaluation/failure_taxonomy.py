"""Failure taxonomy classification for lightweight evaluation outputs."""

from __future__ import annotations

from typing import Any


def classify_failures(metrics: dict[str, float | None]) -> list[str]:
    """Assign failure categories from scene-level metric values."""
    failures: list[str] = []

    up_axis_error = metrics.get("calibration_up_axis_error_deg")
    if _is_number(up_axis_error) and float(up_axis_error) > 20.0:
        failures.append("wrong_up_axis")

    up_confidence = metrics.get("calibration_up_axis_confidence")
    if _is_number(up_confidence) and float(up_confidence) < 0.35:
        failures.append("unstable_up_axis_confidence")

    horizontal_error = metrics.get("calibration_horizontal_error_deg")
    if _is_number(horizontal_error) and float(horizontal_error) > 30.0:
        failures.append("wrong_horizontal_axis")

    manhattan_ambiguity = metrics.get("calibration_manhattan_ambiguity")
    if _is_number(manhattan_ambiguity) and float(manhattan_ambiguity) > 0.45:
        failures.append("manhattan_ambiguity")

    scale_error = metrics.get("calibration_scale_error")
    scale_drift = metrics.get("calibration_scale_drift")
    if (_is_number(scale_error) and float(scale_error) > 0.5) or (
        _is_number(scale_drift) and float(scale_drift) > 0.35
    ):
        failures.append("scale_drift")

    insufficient_plane_evidence = metrics.get("calibration_insufficient_plane_evidence_flag")
    if _is_number(insufficient_plane_evidence) and float(insufficient_plane_evidence) > 0.5:
        failures.append("insufficient_plane_evidence")

    door_error = metrics.get("structured_door_count_error")
    window_error = metrics.get("structured_window_count_error")
    if (_is_number(door_error) and float(door_error) > 0.0) or (
        _is_number(window_error) and float(window_error) > 0.0
    ):
        failures.append("missing_structural_elements")

    violations_before = metrics.get("structured_violation_count_before_repair")
    if _is_number(violations_before) and float(violations_before) > 0.0:
        failures.append("implausible_object_sizes")
    if _is_number(violations_before) and float(violations_before) > 4.0:
        failures.append("clutter_dominated_failure")

    overcorrection = metrics.get("repair_overcorrection_flag")
    if _is_number(overcorrection) and float(overcorrection) > 0.0:
        failures.append("repair_overcorrection")

    relation_f1 = metrics.get("actionable_relation_f1")
    if _is_number(relation_f1) and float(relation_f1) < 0.5:
        failures.append("relation_derivation_failure")
    if (_is_number(relation_f1) and float(relation_f1) < 0.45) and (
        _is_number(manhattan_ambiguity) and float(manhattan_ambiguity) > 0.30
    ):
        failures.append("non_manhattan_scene_failure")

    if not failures:
        failures.append("no_major_failure_detected")

    return failures


def summarize_failure_taxonomy(
    per_scene_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate failure category frequencies for report generation."""
    counts: dict[str, int] = {}
    per_setting: dict[str, dict[str, int]] = {}
    calibration_counts: dict[str, int] = {}
    calibration_by_setting: dict[str, dict[str, int]] = {}
    calibration_labels = {
        "wrong_up_axis",
        "unstable_up_axis_confidence",
        "wrong_horizontal_axis",
        "manhattan_ambiguity",
        "scale_drift",
        "insufficient_plane_evidence",
        "clutter_dominated_failure",
        "non_manhattan_scene_failure",
    }

    for result in per_scene_results:
        setting_name = str(result.get("setting_name", "unknown"))
        failures = result.get("failures", [])
        if not isinstance(failures, list):
            continue

        for failure in failures:
            label = str(failure)
            counts[label] = counts.get(label, 0) + 1
            setting_counts = per_setting.setdefault(setting_name, {})
            setting_counts[label] = setting_counts.get(label, 0) + 1
            if label in calibration_labels:
                calibration_counts[label] = calibration_counts.get(label, 0) + 1
                calibration_setting_counts = calibration_by_setting.setdefault(setting_name, {})
                calibration_setting_counts[label] = calibration_setting_counts.get(label, 0) + 1

    return {
        "overall_counts": counts,
        "counts_by_setting": per_setting,
        "calibration_failure_counts": calibration_counts,
        "calibration_failure_counts_by_setting": calibration_by_setting,
    }


def render_failure_summary_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Failure Taxonomy Summary",
        "",
        "## Overall Counts",
    ]
    overall = summary.get("overall_counts", {})
    if isinstance(overall, dict) and overall:
        for key in sorted(overall.keys()):
            lines.append(f"- {key}: {overall[key]}")
    else:
        lines.append("- no failures recorded")

    lines.extend(
        [
            "",
            "## Counts By Setting",
        ]
    )
    by_setting = summary.get("counts_by_setting", {})
    if isinstance(by_setting, dict) and by_setting:
        for setting_name in sorted(by_setting.keys()):
            lines.append(f"- {setting_name}:")
            counts = by_setting[setting_name]
            if isinstance(counts, dict):
                for key in sorted(counts.keys()):
                    lines.append(f"- {setting_name} / {key}: {counts[key]}")
    else:
        lines.append("- no setting-level failures recorded")

    lines.extend(
        [
            "",
            "## Calibration Failure Counts",
        ]
    )
    calibration = summary.get("calibration_failure_counts", {})
    if isinstance(calibration, dict) and calibration:
        for key in sorted(calibration.keys()):
            lines.append(f"- {key}: {calibration[key]}")
    else:
        lines.append("- no calibration-specific failures recorded")

    return "\n".join(lines)


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float))
