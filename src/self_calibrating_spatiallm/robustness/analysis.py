"""Boundary-oriented aggregation for perturbation robustness studies."""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from self_calibrating_spatiallm.artifacts import ScenePrediction
from self_calibrating_spatiallm.language import (
    build_grounding_examples,
    build_qa_examples,
    export_scene_prediction_dict_to_language,
)
from self_calibrating_spatiallm.robustness.perturbations import severity_bucket

LOWER_BETTER_METRICS = {
    "calibration_up_axis_error_deg",
    "calibration_horizontal_error_deg",
    "structured_violation_count_before_repair",
    "repair_violations_after",
}
HIGHER_BETTER_METRICS = {
    "calibration_reliability",
    "actionable_relation_f1",
}
TRACKED_METRICS = sorted(LOWER_BETTER_METRICS.union(HIGHER_BETTER_METRICS))


def build_boundary_rows(
    *,
    split_reports: dict[str, dict[str, Any]],
    perturbation_inventory: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build one scene-level boundary row per perturbed scene."""
    inventory_index: dict[tuple[str, str], dict[str, Any]] = {}
    for row in perturbation_inventory:
        split_name = str(row.get("split_name", "unknown"))
        scene_id = str(row.get("scene_id", "unknown"))
        inventory_index[(split_name, scene_id)] = row

    rows: list[dict[str, Any]] = []
    for split_name, payload in sorted(split_reports.items()):
        report = payload.get("report")
        if not isinstance(report, dict):
            continue
        split_role = str(payload.get("split_role", "generalization"))
        delta_rows = report.get("scene_level_delta_report", [])
        delta_by_scene: dict[str, dict[str, Any]] = {}
        if isinstance(delta_rows, list):
            for item in delta_rows:
                if isinstance(item, dict):
                    delta_by_scene[str(item.get("scene_id", "unknown"))] = item

        for scene in report.get("scenes", []):
            if not isinstance(scene, dict):
                continue
            scene_id = str(scene.get("scene_id", "unknown"))
            settings = scene.get("settings", [])
            if not isinstance(settings, list):
                settings = []
            by_setting = {
                str(setting.get("setting_name")): setting
                for setting in settings
                if isinstance(setting, dict)
            }
            delta_row = delta_by_scene.get(scene_id, {})
            inventory_row = inventory_index.get((split_name, scene_id), {})
            perturbation_info = _extract_perturbation_info(
                scene=scene,
                inventory_row=inventory_row,
                split_name=split_name,
                split_role=split_role,
            )

            row = {
                **perturbation_info,
                "scene_id": scene_id,
                "source_type": scene.get("source_type"),
                "scene_tags": list(scene.get("tags", [])) if isinstance(scene.get("tags"), list) else [],
                "comparison_valid": delta_row.get("comparison_valid"),
                "comparison_valid_with_repair": delta_row.get("comparison_valid_with_repair"),
                "partial_calibration_applied": delta_row.get("partial_calibration_applied"),
                "true_v1_execution": delta_row.get("true_v1_execution"),
                "fallback_used": delta_row.get("fallback_used"),
                "fallback_reason": delta_row.get("fallback_reason"),
                "candidate_plane_count": delta_row.get("candidate_plane_count"),
                "major_failure_category": delta_row.get("major_failure_category"),
                "major_failure_labels": delta_row.get("major_failure_labels", []),
                "calibration_summary": _ensure_dict(delta_row.get("calibration_summary")),
                "delta_v1_minus_v0": _ensure_dict(delta_row.get("delta_v1_minus_v0")),
                "delta_v1_plus_repair_minus_v1": _ensure_dict(
                    delta_row.get("delta_v1_plus_repair_minus_v1")
                ),
                "prediction_delta_v1_minus_v0": _ensure_dict(
                    delta_row.get("prediction_delta_v1_minus_v0")
                ),
                "prediction_delta_external_minus_mock": _ensure_dict(
                    delta_row.get("prediction_delta_external_minus_mock")
                ),
                "violation_delta_external_minus_mock": delta_row.get(
                    "violation_delta_external_minus_mock"
                ),
                "v0_metrics": _setting_metrics(by_setting.get("calibration_v0")),
                "v1_metrics": _setting_metrics(by_setting.get("calibration_v1")),
                "v1_plus_repair_metrics": _setting_metrics(by_setting.get("calibration_v1_plus_repair")),
                "mock_metrics": _setting_metrics(by_setting.get("mock_generator")),
                "external_metrics": _setting_metrics(by_setting.get("external_generator")),
                "mock_status": _setting_status(by_setting.get("mock_generator")),
                "external_status": _setting_status(by_setting.get("external_generator")),
            }
            rows.append(row)
    rows.sort(
        key=lambda item: (
            str(item.get("split_name", "")),
            str(item.get("base_scene_id", "")),
            str(item.get("perturbation_type", "")),
            float(item.get("severity", 0.0)),
            str(item.get("scene_id", "")),
        )
    )
    return rows


def build_boundary_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate boundary rows into perturbation-severity summaries."""
    summary_by_group: dict[tuple[str, str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            str(row.get("split_name", "unknown")),
            str(row.get("perturbation_type", "unknown")),
            float(row.get("severity", 0.0)),
        )
        summary_by_group[key].append(row)

    grouped_rows: list[dict[str, Any]] = []
    for (split_name, perturbation_type, severity), group in sorted(summary_by_group.items()):
        grouped_rows.append(
            {
                "split_name": split_name,
                "perturbation_type": perturbation_type,
                "severity": severity,
                "severity_bucket": severity_bucket(severity),
                "num_scenes": len(group),
                "comparison_valid_rate": _mean_bool(group, "comparison_valid"),
                "partial_calibration_rate": _mean_bool(group, "partial_calibration_applied"),
                "fallback_rate": _mean_bool(group, "fallback_used"),
                "avg_candidate_plane_count": _mean_numeric(group, "candidate_plane_count"),
                "avg_manhattan_ambiguity": _mean_nested_numeric(
                    group, "calibration_summary", "manhattan_ambiguity"
                ),
                "avg_horizontal_confidence": _mean_nested_numeric(
                    group, "calibration_summary", "horizontal_confidence"
                ),
                "avg_calibration_reliability": _mean_nested_numeric(
                    group, "calibration_summary", "overall_reliability"
                ),
                "v1_vs_v0_metric_deltas": _mean_metric_deltas(group),
                "external_vs_mock": _summarize_external_vs_mock(group),
                "major_failure_category_counts": dict(_major_failure_counts(group)),
            }
        )

    by_split = _summarize_by_split(rows)
    by_base_scene = _summarize_by_base_scene(rows)
    by_ambiguity_strata = _summarize_by_ambiguity_strata(rows)
    by_major_failure_category = _summarize_by_major_failure_category(rows)
    by_fallback_reason = _summarize_by_fallback_reason(rows)
    boundary_findings = _build_boundary_findings(grouped_rows)

    return {
        "num_rows": len(rows),
        "grouped_by_split_perturbation_severity": grouped_rows,
        "by_split": by_split,
        "by_base_scene": by_base_scene,
        "by_ambiguity_strata": by_ambiguity_strata,
        "by_major_failure_category": by_major_failure_category,
        "by_fallback_reason": by_fallback_reason,
        "boundary_findings": boundary_findings,
    }


def render_boundary_summary_markdown(summary: dict[str, Any]) -> str:
    """Render human-readable robustness-boundary summary."""
    lines = [
        "# Robustness Boundary Summary",
        "",
        f"- num_rows: `{summary.get('num_rows')}`",
        "",
        "## Split Overview",
    ]
    by_split = summary.get("by_split", {})
    if isinstance(by_split, dict) and by_split:
        for split_name, payload in sorted(by_split.items()):
            if not isinstance(payload, dict):
                continue
            lines.append(
                f"- `{split_name}`: rows=`{payload.get('num_rows')}`, "
                f"comparison_valid_rate=`{_fmt(payload.get('comparison_valid_rate'))}`, "
                f"partial_calibration_rate=`{_fmt(payload.get('partial_calibration_rate'))}`"
            )
    else:
        lines.append("- none")

    lines.extend(["", "## Boundary Findings"])
    findings = summary.get("boundary_findings", [])
    if isinstance(findings, list) and findings:
        for finding in findings:
            lines.append(f"- {finding}")
    else:
        lines.append("- none")

    lines.extend(["", "## Grouped Severity Table"])
    grouped = summary.get("grouped_by_split_perturbation_severity", [])
    if isinstance(grouped, list) and grouped:
        lines.append(
            "| split | perturbation | severity | n | valid_rate | partial_rate | "
            "reliability_delta(v1-v0) | violation_delta(v1-v0) | actionable_f1_delta(v1-v0) | "
            "external_object_delta | external_relation_delta |"
        )
        lines.append(
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
        )
        for row in grouped:
            if not isinstance(row, dict):
                continue
            deltas = row.get("v1_vs_v0_metric_deltas", {})
            if not isinstance(deltas, dict):
                deltas = {}
            external_delta = row.get("external_vs_mock", {})
            if not isinstance(external_delta, dict):
                external_delta = {}
            lines.append(
                "| "
                f"{row.get('split_name')} | "
                f"{row.get('perturbation_type')} | "
                f"{_fmt(row.get('severity'))} | "
                f"{row.get('num_scenes')} | "
                f"{_fmt(row.get('comparison_valid_rate'))} | "
                f"{_fmt(row.get('partial_calibration_rate'))} | "
                f"{_fmt(deltas.get('calibration_reliability'))} | "
                f"{_fmt(deltas.get('structured_violation_count_before_repair'))} | "
                f"{_fmt(deltas.get('actionable_relation_f1'))} | "
                f"{_fmt(external_delta.get('avg_object_count_delta_external_minus_mock'))} | "
                f"{_fmt(external_delta.get('avg_relation_count_delta_external_minus_mock'))} |"
            )
    else:
        lines.append("- none")

    lines.extend(["", "## Ambiguity Strata"])
    ambiguity = summary.get("by_ambiguity_strata", {})
    if isinstance(ambiguity, dict) and ambiguity:
        for name, payload in sorted(ambiguity.items()):
            if not isinstance(payload, dict):
                continue
            lines.append(
                f"- `{name}`: n=`{payload.get('num_rows')}`, "
                f"mean_ambiguity=`{_fmt(payload.get('avg_manhattan_ambiguity'))}`, "
                f"mean_v1_reliability_delta=`{_fmt(payload.get('avg_v1_vs_v0_reliability_delta'))}`"
            )
    else:
        lines.append("- none")

    lines.extend(["", "## Major Failure Categories"])
    failure_rows = summary.get("by_major_failure_category", {})
    if isinstance(failure_rows, dict) and failure_rows:
        for category, payload in sorted(failure_rows.items()):
            if not isinstance(payload, dict):
                continue
            lines.append(
                f"- `{category}`: n=`{payload.get('num_rows')}`, "
                f"partial_rate=`{_fmt(payload.get('partial_calibration_rate'))}`, "
                f"mean_reliability_delta=`{_fmt(payload.get('avg_v1_vs_v0_reliability_delta'))}`"
            )
    else:
        lines.append("- none")

    lines.extend(["", "## Fallback Reasons"])
    fallback_rows = summary.get("by_fallback_reason", {})
    if isinstance(fallback_rows, dict) and fallback_rows:
        for reason, payload in sorted(fallback_rows.items()):
            if not isinstance(payload, dict):
                continue
            lines.append(
                f"- `{reason}`: n=`{payload.get('num_rows')}`, "
                f"fallback_rate=`{_fmt(payload.get('fallback_rate'))}`, "
                f"partial_rate=`{_fmt(payload.get('partial_calibration_rate'))}`"
            )
    else:
        lines.append("- none")

    return "\n".join(lines) + "\n"


def export_language_boundary_artifacts(
    *,
    split_reports: dict[str, dict[str, Any]],
    perturbation_inventory: list[dict[str, Any]],
    output_dir: Path,
) -> dict[str, Path]:
    """Export language-facing rows grouped by perturbation severity."""
    output_dir.mkdir(parents=True, exist_ok=True)
    inventory_index: dict[tuple[str, str], dict[str, Any]] = {}
    for row in perturbation_inventory:
        inventory_index[(str(row.get("split_name")), str(row.get("scene_id")))] = row

    scene_rows: list[dict[str, Any]] = []
    qa_rows: list[dict[str, Any]] = []
    grounding_rows: list[dict[str, Any]] = []

    for split_name, payload in sorted(split_reports.items()):
        report = payload.get("report")
        if not isinstance(report, dict):
            continue
        split_role = str(payload.get("split_role", "generalization"))
        for scene in report.get("scenes", []):
            if not isinstance(scene, dict):
                continue
            scene_id = str(scene.get("scene_id", "unknown"))
            inventory_row = inventory_index.get((split_name, scene_id), {})
            perturbation_info = _extract_perturbation_info(
                scene=scene,
                inventory_row=inventory_row,
                split_name=split_name,
                split_role=split_role,
            )
            settings = scene.get("settings", [])
            if not isinstance(settings, list):
                continue
            for setting in settings:
                if not isinstance(setting, dict):
                    continue
                if str(setting.get("status", "")) != "success":
                    continue
                setting_name = str(setting.get("setting_name", "unknown"))
                metadata = setting.get("metadata", {})
                if not isinstance(metadata, dict):
                    metadata = {}

                structured = metadata.get("structured_prediction_pre_repair")
                if not isinstance(structured, dict):
                    structured = None

                language_export = metadata.get("language_export_pre_repair")
                if not isinstance(language_export, dict) and structured is not None:
                    try:
                        language_export = export_scene_prediction_dict_to_language(structured)
                    except Exception:
                        language_export = {}
                if not isinstance(language_export, dict):
                    language_export = {}

                qa_examples: list[dict[str, Any]] = []
                grounding_examples: list[dict[str, Any]] = []
                if structured is not None:
                    try:
                        prediction = ScenePrediction.from_dict(structured)
                        qa_examples = build_qa_examples(prediction)
                        grounding_examples = build_grounding_examples(prediction)
                    except Exception:
                        qa_examples = []
                        grounding_examples = []

                scene_row = {
                    **perturbation_info,
                    "scene_id": scene_id,
                    "setting_name": setting_name,
                    "source_type": scene.get("source_type"),
                    "scene_summary_text": language_export.get("scene_summary_text"),
                    "relation_text": language_export.get("relation_text"),
                    "scene_paragraph_text": language_export.get("scene_paragraph_text"),
                    "relation_statements": language_export.get("relation_statements", []),
                    "relation_evidence_level": language_export.get("relation_evidence_level"),
                    "relation_hint_count": language_export.get("relation_hint_count"),
                    "relation_hint_predicates": language_export.get("relation_hint_predicates", []),
                    "object_count": language_export.get("object_count"),
                    "relation_count": language_export.get("relation_count"),
                    "object_labels": language_export.get("object_labels", []),
                    "relation_predicates": language_export.get("relation_predicates", []),
                    "prediction_source_class": language_export.get("prediction_source_class"),
                    "structured_prediction_available": structured is not None,
                    "structured_prediction": structured,
                }
                scene_rows.append(scene_row)

                for index, qa in enumerate(qa_examples):
                    qa_rows.append(
                        {
                            **perturbation_info,
                            "scene_id": scene_id,
                            "setting_name": setting_name,
                            "qa_id": f"{split_name}:{scene_id}:{setting_name}:qa:{index}",
                            "task_type": qa.get("task_type"),
                            "question": qa.get("question"),
                            "answer": qa.get("answer"),
                            "metadata": qa.get("metadata", {}),
                        }
                    )

                for index, grounding in enumerate(grounding_examples):
                    grounding_rows.append(
                        {
                            **perturbation_info,
                            "scene_id": scene_id,
                            "setting_name": setting_name,
                            "grounding_id": f"{split_name}:{scene_id}:{setting_name}:grounding:{index}",
                            "task_type": grounding.get("task_type"),
                            "text": grounding.get("text"),
                            "target_object_id": grounding.get("target_object_id"),
                            "metadata": grounding.get("metadata", {}),
                        }
                    )

    severity_alignment_rows = _build_language_severity_alignment(scene_rows)
    summary = {
        "num_scene_rows": len(scene_rows),
        "num_qa_rows": len(qa_rows),
        "num_grounding_rows": len(grounding_rows),
        "num_severity_alignment_rows": len(severity_alignment_rows),
        "num_rows_with_structured_prediction": sum(
            1 for row in scene_rows if bool(row.get("structured_prediction_available"))
        ),
        "num_rows_with_explicit_relations": sum(
            1 for row in scene_rows if int(row.get("relation_count") or 0) > 0
        ),
    }

    scene_jsonl_path = output_dir / "robustness_language_scene_examples.jsonl"
    qa_jsonl_path = output_dir / "robustness_language_qa_examples.jsonl"
    grounding_jsonl_path = output_dir / "robustness_language_grounding_examples.jsonl"
    severity_jsonl_path = output_dir / "robustness_language_severity_deltas.jsonl"
    summary_json_path = output_dir / "robustness_language_summary.json"
    summary_md_path = output_dir / "robustness_language_summary.md"

    _write_jsonl(scene_jsonl_path, scene_rows)
    _write_jsonl(qa_jsonl_path, qa_rows)
    _write_jsonl(grounding_jsonl_path, grounding_rows)
    _write_jsonl(severity_jsonl_path, severity_alignment_rows)
    summary_json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    summary_md_path.write_text(_render_language_summary_markdown(summary), encoding="utf-8")

    return {
        "language_scene_examples_jsonl": scene_jsonl_path,
        "language_qa_examples_jsonl": qa_jsonl_path,
        "language_grounding_examples_jsonl": grounding_jsonl_path,
        "language_severity_deltas_jsonl": severity_jsonl_path,
        "language_summary_json": summary_json_path,
        "language_summary_markdown": summary_md_path,
    }


def _build_language_severity_alignment(scene_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in scene_rows:
        grouped[
            (
                str(row.get("split_name")),
                str(row.get("base_scene_id")),
                str(row.get("perturbation_type")),
                str(row.get("setting_name")),
            )
        ].append(row)

    alignment_rows: list[dict[str, Any]] = []
    for key, rows in grouped.items():
        rows_sorted = sorted(rows, key=lambda item: float(item.get("severity", 0.0)))
        for prev, curr in zip(rows_sorted, rows_sorted[1:]):
            alignment_rows.append(
                {
                    "split_name": key[0],
                    "base_scene_id": key[1],
                    "perturbation_type": key[2],
                    "setting_name": key[3],
                    "severity_from": prev.get("severity"),
                    "severity_to": curr.get("severity"),
                    "object_count_delta": _delta(curr.get("object_count"), prev.get("object_count")),
                    "relation_count_delta": _delta(curr.get("relation_count"), prev.get("relation_count")),
                    "relation_evidence_transition": (
                        f"{prev.get('relation_evidence_level')}->{curr.get('relation_evidence_level')}"
                    ),
                    "summary_from": prev.get("scene_summary_text"),
                    "summary_to": curr.get("scene_summary_text"),
                }
            )
    return alignment_rows


def _extract_perturbation_info(
    *,
    scene: dict[str, Any],
    inventory_row: dict[str, Any],
    split_name: str,
    split_role: str,
) -> dict[str, Any]:
    tags = scene.get("tags", [])
    if not isinstance(tags, list):
        tags = []
    tag_map: dict[str, str] = {}
    for tag in tags:
        if not isinstance(tag, str):
            continue
        if tag.startswith("rb:") and "=" in tag:
            _, payload = tag.split("rb:", 1)
            key, value = payload.split("=", 1)
            tag_map[key] = value

    base_scene_id = str(
        inventory_row.get("base_scene_id")
        or tag_map.get("base_scene")
        or scene.get("scene_id")
        or "unknown"
    )
    perturbation_type = str(
        inventory_row.get("perturbation_type")
        or tag_map.get("perturbation")
        or "unknown"
    )
    severity_raw = inventory_row.get("severity")
    if severity_raw is None:
        severity_raw = tag_map.get("severity")
    try:
        severity_value = float(severity_raw) if severity_raw is not None else 0.0
    except Exception:
        severity_value = 0.0

    severity_bucket_value = str(
        inventory_row.get("severity_bucket")
        or tag_map.get("severity_bucket")
        or severity_bucket(severity_value)
    )

    return {
        "split_name": split_name,
        "split_role": split_role,
        "base_scene_id": base_scene_id,
        "perturbation_type": perturbation_type,
        "severity": severity_value,
        "severity_bucket": severity_bucket_value,
        "perturbation_parameters": _ensure_dict(inventory_row.get("parameters")),
    }


def _summarize_external_vs_mock(rows: list[dict[str, Any]]) -> dict[str, Any]:
    object_deltas: list[float] = []
    relation_deltas: list[float] = []
    violation_deltas: list[float] = []
    for row in rows:
        prediction_delta = _ensure_dict(row.get("prediction_delta_external_minus_mock"))
        object_delta = prediction_delta.get("object_count_delta")
        relation_delta = prediction_delta.get("relation_count_delta")
        if isinstance(object_delta, (int, float)) and math.isfinite(float(object_delta)):
            object_deltas.append(float(object_delta))
        if isinstance(relation_delta, (int, float)) and math.isfinite(float(relation_delta)):
            relation_deltas.append(float(relation_delta))
        violation_delta = row.get("violation_delta_external_minus_mock")
        if isinstance(violation_delta, (int, float)) and math.isfinite(float(violation_delta)):
            violation_deltas.append(float(violation_delta))

    return {
        "avg_object_count_delta_external_minus_mock": _mean(object_deltas),
        "avg_relation_count_delta_external_minus_mock": _mean(relation_deltas),
        "avg_pre_repair_violation_delta_external_minus_mock": _mean(violation_deltas),
    }


def _mean_metric_deltas(rows: list[dict[str, Any]]) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for metric in TRACKED_METRICS:
        values: list[float] = []
        for row in rows:
            deltas = _ensure_dict(row.get("delta_v1_minus_v0"))
            value = deltas.get(metric)
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                values.append(float(value))
        out[metric] = _mean(values)
    return out


def _summarize_by_split(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("split_name", "unknown"))].append(row)

    summary: dict[str, dict[str, Any]] = {}
    for split_name, group in sorted(grouped.items()):
        summary[split_name] = {
            "num_rows": len(group),
            "comparison_valid_rate": _mean_bool(group, "comparison_valid"),
            "partial_calibration_rate": _mean_bool(group, "partial_calibration_applied"),
            "fallback_rate": _mean_bool(group, "fallback_used"),
            "avg_manhattan_ambiguity": _mean_nested_numeric(
                group, "calibration_summary", "manhattan_ambiguity"
            ),
            "avg_v1_vs_v0_reliability_delta": _mean_metric_deltas(group).get(
                "calibration_reliability"
            ),
        }
    return summary


def _summarize_by_base_scene(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = str(row.get("base_scene_id", "unknown"))
        grouped[key].append(row)

    output: dict[str, dict[str, Any]] = {}
    for base_scene_id, group in sorted(grouped.items()):
        output[base_scene_id] = {
            "num_rows": len(group),
            "partial_calibration_rate": _mean_bool(group, "partial_calibration_applied"),
            "comparison_valid_rate": _mean_bool(group, "comparison_valid"),
            "top_failure_categories": dict(_major_failure_counts(group).most_common(5)),
            "avg_reliability_delta_v1_vs_v0": _mean_metric_deltas(group).get(
                "calibration_reliability"
            ),
        }
    return output


def _summarize_by_ambiguity_strata(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        ambiguity = _nested_numeric(row, "calibration_summary", "manhattan_ambiguity")
        if ambiguity is None:
            stratum = "unknown"
        elif ambiguity < 0.4:
            stratum = "low"
        elif ambiguity < 0.7:
            stratum = "mid"
        else:
            stratum = "high"
        grouped[stratum].append(row)

    summary: dict[str, dict[str, Any]] = {}
    for stratum, group in sorted(grouped.items()):
        summary[stratum] = {
            "num_rows": len(group),
            "avg_manhattan_ambiguity": _mean(
                [
                    value
                    for value in (
                        _nested_numeric(item, "calibration_summary", "manhattan_ambiguity")
                        for item in group
                    )
                    if value is not None
                ]
            ),
            "avg_v1_vs_v0_reliability_delta": _mean_metric_deltas(group).get(
                "calibration_reliability"
            ),
            "avg_v1_vs_v0_violation_delta": _mean_metric_deltas(group).get(
                "structured_violation_count_before_repair"
            ),
            "partial_calibration_rate": _mean_bool(group, "partial_calibration_applied"),
        }
    return summary


def _summarize_by_major_failure_category(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        category = row.get("major_failure_category")
        if isinstance(category, str) and category:
            grouped[category].append(row)

    summary: dict[str, dict[str, Any]] = {}
    for category, group in sorted(grouped.items()):
        summary[category] = {
            "num_rows": len(group),
            "partial_calibration_rate": _mean_bool(group, "partial_calibration_applied"),
            "comparison_valid_rate": _mean_bool(group, "comparison_valid"),
            "avg_manhattan_ambiguity": _mean_nested_numeric(
                group, "calibration_summary", "manhattan_ambiguity"
            ),
            "avg_v1_vs_v0_reliability_delta": _mean_metric_deltas(group).get(
                "calibration_reliability"
            ),
            "avg_v1_vs_v0_violation_delta": _mean_metric_deltas(group).get(
                "structured_violation_count_before_repair"
            ),
        }
    return summary


def _summarize_by_fallback_reason(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        raw_reason = row.get("fallback_reason")
        if isinstance(raw_reason, str) and raw_reason.strip():
            reason = raw_reason.strip()
        elif bool(row.get("fallback_used")):
            reason = "unknown"
        else:
            reason = "none"
        grouped[reason].append(row)

    summary: dict[str, dict[str, Any]] = {}
    for reason, group in sorted(grouped.items()):
        summary[reason] = {
            "num_rows": len(group),
            "fallback_rate": _mean_bool(group, "fallback_used"),
            "partial_calibration_rate": _mean_bool(group, "partial_calibration_applied"),
            "avg_v1_vs_v0_reliability_delta": _mean_metric_deltas(group).get(
                "calibration_reliability"
            ),
        }
    return summary


def _build_boundary_findings(grouped_rows: list[dict[str, Any]]) -> list[str]:
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in grouped_rows:
        by_type[str(row.get("perturbation_type", "unknown"))].append(row)

    findings: list[str] = []
    for perturbation_type, rows in sorted(by_type.items()):
        rows_sorted = sorted(rows, key=lambda item: float(item.get("severity", 0.0)))
        help_until = None
        reliability_drop_at = None
        partial_trigger_at = None
        relation_drop_at = None
        for row in rows_sorted:
            severity = float(row.get("severity", 0.0))
            deltas = _ensure_dict(row.get("v1_vs_v0_metric_deltas"))
            reliability_delta = _safe_float(deltas.get("calibration_reliability"))
            violation_delta = _safe_float(deltas.get("structured_violation_count_before_repair"))
            relation_delta = _safe_float(deltas.get("actionable_relation_f1"))
            partial_rate = _safe_float(row.get("partial_calibration_rate"))

            if reliability_delta is not None and violation_delta is not None:
                if reliability_delta >= 0.0 and violation_delta <= 0.0:
                    help_until = severity
                elif reliability_drop_at is None and reliability_delta < 0.0:
                    reliability_drop_at = severity

            if partial_trigger_at is None and partial_rate is not None and partial_rate >= 0.5:
                partial_trigger_at = severity

            if relation_drop_at is None and relation_delta is not None and relation_delta < -0.02:
                relation_drop_at = severity

        if help_until is not None:
            findings.append(
                f"{perturbation_type}: v1 remains jointly helpful "
                f"(reliability delta >= 0 and violation delta <= 0) up to severity {help_until:.3f}."
            )
        else:
            findings.append(
                f"{perturbation_type}: no severity showed jointly positive reliability and violation behavior."
            )

        if reliability_drop_at is not None:
            findings.append(
                f"{perturbation_type}: reliability degradation begins around severity {reliability_drop_at:.3f}."
            )
        if partial_trigger_at is not None:
            findings.append(
                f"{perturbation_type}: partial calibration becomes common (>=50%) beyond severity "
                f"{partial_trigger_at:.3f}."
            )
        if relation_drop_at is not None:
            findings.append(
                f"{perturbation_type}: actionable relation outputs regress beyond severity "
                f"{relation_drop_at:.3f}."
            )
    return findings


def _setting_metrics(setting: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(setting, dict):
        return {}
    metrics = setting.get("metrics")
    return dict(metrics) if isinstance(metrics, dict) else {}


def _setting_status(setting: dict[str, Any] | None) -> str | None:
    if not isinstance(setting, dict):
        return None
    status = setting.get("status")
    return str(status) if status is not None else None


def _major_failure_counts(rows: list[dict[str, Any]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for row in rows:
        label = row.get("major_failure_category")
        if isinstance(label, str) and label:
            counts[label] += 1
    return counts


def _mean_bool(rows: list[dict[str, Any]], key: str) -> float | None:
    values: list[float] = []
    for row in rows:
        value = row.get(key)
        if isinstance(value, bool):
            values.append(1.0 if value else 0.0)
    return _mean(values)


def _mean_numeric(rows: list[dict[str, Any]], key: str) -> float | None:
    values: list[float] = []
    for row in rows:
        value = row.get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            values.append(float(value))
    return _mean(values)


def _mean_nested_numeric(rows: list[dict[str, Any]], outer: str, inner: str) -> float | None:
    values: list[float] = []
    for row in rows:
        parsed = _nested_numeric(row, outer, inner)
        if parsed is not None:
            values.append(parsed)
    return _mean(values)


def _nested_numeric(row: dict[str, Any], outer: str, inner: str) -> float | None:
    outer_payload = _ensure_dict(row.get(outer))
    value = outer_payload.get(inner)
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def _delta(lhs: Any, rhs: Any) -> float | None:
    lhs_value = _safe_float(lhs)
    rhs_value = _safe_float(rhs)
    if lhs_value is None or rhs_value is None:
        return None
    return lhs_value - rhs_value


def _safe_float(value: Any) -> float | None:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def _ensure_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True))
            handle.write("\n")


def _render_language_summary_markdown(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Robustness Language Export Summary",
            "",
            f"- num_scene_rows: `{summary.get('num_scene_rows')}`",
            f"- num_qa_rows: `{summary.get('num_qa_rows')}`",
            f"- num_grounding_rows: `{summary.get('num_grounding_rows')}`",
            f"- num_severity_alignment_rows: `{summary.get('num_severity_alignment_rows')}`",
            f"- num_rows_with_structured_prediction: "
            f"`{summary.get('num_rows_with_structured_prediction')}`",
            f"- num_rows_with_explicit_relations: "
            f"`{summary.get('num_rows_with_explicit_relations')}`",
            "",
        ]
    )


def _fmt(value: Any) -> str:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return f"{float(value):.4f}"
    return "n/a"
