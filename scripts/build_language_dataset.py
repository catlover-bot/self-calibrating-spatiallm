"""Build NLP-facing JSONL artifacts from existing evaluation report outputs."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from self_calibrating_spatiallm.artifacts import Point3D, SceneObject, ScenePrediction
from self_calibrating_spatiallm.language import (
    build_grounding_examples,
    build_qa_examples,
    export_scene_prediction_to_language,
)

CORE_SETTINGS = [
    "no_calibration",
    "calibration_v0",
    "calibration_v1",
    "calibration_v1_plus_repair",
    "mock_generator",
    "external_generator",
]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build language-facing dataset artifacts from evaluation report")
    parser.add_argument(
        "--evaluation-report",
        type=Path,
        required=True,
        help="Path to evaluation_report.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for language JSONL and summaries (default: <report_dir>/language)",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    report_path = args.evaluation_report.expanduser().resolve()
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object at {report_path}")

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else (report_path.parent / "language").resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    scene_setting_rows: list[dict[str, Any]] = []
    qa_rows: list[dict[str, Any]] = []
    grounding_rows: list[dict[str, Any]] = []
    alignment_map: dict[str, dict[str, dict[str, Any]]] = {}
    setting_export_counts: dict[str, dict[str, int]] = {}

    scenes = payload.get("scenes", [])
    if not isinstance(scenes, list):
        scenes = []

    for scene in scenes:
        if not isinstance(scene, dict):
            continue
        scene_id = str(scene.get("scene_id", "unknown_scene"))
        source_type = _none_if_empty(scene.get("source_type"))
        tags = scene.get("tags", [])
        settings = scene.get("settings", [])
        if not isinstance(settings, list):
            settings = []

        scene_alignment = alignment_map.setdefault(scene_id, {})
        for setting in settings:
            if not isinstance(setting, dict):
                continue

            setting_name = str(setting.get("setting_name", "unknown_setting"))
            setting_status = str(setting.get("status", "unknown"))
            if setting_status != "success":
                continue

            metadata = setting.get("metadata", {})
            if not isinstance(metadata, dict):
                metadata = {}

            prediction_payload = metadata.get("structured_prediction_pre_repair")
            prediction_source = "structured_prediction_pre_repair"
            if not isinstance(prediction_payload, dict):
                prediction_payload = metadata.get("structured_prediction")
                prediction_source = "structured_prediction"

            prediction = _resolve_prediction(
                scene_id=scene_id,
                setting_name=setting_name,
                prediction_payload=prediction_payload,
                prediction_summary=metadata.get("prediction_summary_pre_repair"),
            )
            prediction = _enrich_prediction_with_summary_hints(
                prediction=prediction,
                prediction_summary=metadata.get("prediction_summary_pre_repair"),
            )

            # Always recompute to ensure relation hints and reconstruction flags are faithfully reflected.
            language_export = export_scene_prediction_to_language(prediction)
            language_export_source = "computed_from_prediction"

            qa_examples = build_qa_examples(prediction)
            grounding_examples = build_grounding_examples(prediction)
            reconstructed_from_summary = bool(
                language_export.get("reconstructed_from_prediction_summary", False)
            )

            scene_setting_row = {
                "scene_id": scene_id,
                "setting": setting_name,
                "source_type": source_type,
                "tags": tags if isinstance(tags, list) else [],
                "structured_prediction": prediction.to_dict(),
                "scene_summary_text": language_export.get("scene_summary_text"),
                "object_list_text": language_export.get("object_list_text"),
                "relation_text": language_export.get("relation_text"),
                "scene_paragraph_text": language_export.get("scene_paragraph_text"),
                "relation_statements": language_export.get("relation_statements", []),
                "relation_evidence_level": language_export.get("relation_evidence_level"),
                "relation_hint_count": language_export.get("relation_hint_count"),
                "relation_hint_predicates": language_export.get("relation_hint_predicates", []),
                "qa_examples": qa_examples,
                "grounding_examples": grounding_examples,
                "metadata": {
                    "prediction_source": prediction_source,
                    "language_export_source": language_export_source,
                    "generator_mode": metadata.get("generator_mode"),
                    "generator_name": metadata.get("generator_name"),
                    "calibration_method": metadata.get("calibration_method"),
                    "calibration_execution": metadata.get("calibration_execution"),
                    "prediction_summary_pre_repair": metadata.get("prediction_summary_pre_repair"),
                    "reconstructed_from_prediction_summary": reconstructed_from_summary,
                    "object_geometry_mode": language_export.get("object_geometry_mode"),
                },
            }
            scene_setting_rows.append(scene_setting_row)

            for idx, qa in enumerate(qa_examples):
                qa_rows.append(
                    {
                        "scene_id": scene_id,
                        "setting": setting_name,
                        "qa_id": f"{scene_id}:{setting_name}:qa:{idx}",
                        "question": qa.get("question"),
                        "answer": qa.get("answer"),
                        "task_type": qa.get("task_type"),
                        "metadata": qa.get("metadata", {}),
                    }
                )

            for idx, grounding in enumerate(grounding_examples):
                grounding_rows.append(
                    {
                        "scene_id": scene_id,
                        "setting": setting_name,
                        "grounding_id": f"{scene_id}:{setting_name}:grounding:{idx}",
                        "text": grounding.get("text"),
                        "task_type": grounding.get("task_type"),
                        "target_object_id": grounding.get("target_object_id"),
                        "metadata": grounding.get("metadata", {}),
                    }
                )

            scene_alignment[setting_name] = {
                "scene_summary_text": language_export.get("scene_summary_text"),
                "relation_text": language_export.get("relation_text"),
                "object_count": language_export.get("object_count"),
                "relation_count": language_export.get("relation_count"),
                "object_labels": language_export.get("object_labels", []),
                "relation_predicates": language_export.get("relation_predicates", []),
                "relation_hint_predicates": language_export.get("relation_hint_predicates", []),
                "relation_hint_count": language_export.get("relation_hint_count"),
                "relation_evidence_level": language_export.get("relation_evidence_level"),
                "reconstructed_from_prediction_summary": reconstructed_from_summary,
                "calibration_method": metadata.get("calibration_method"),
                "generator_mode": metadata.get("generator_mode"),
            }

            counts = setting_export_counts.setdefault(
                setting_name,
                {
                    "num_success": 0,
                    "num_with_structured_prediction": 0,
                    "num_with_language_export": 0,
                    "num_reconstructed_from_summary": 0,
                },
            )
            counts["num_success"] += 1
            if isinstance(prediction_payload, dict):
                counts["num_with_structured_prediction"] += 1
            if isinstance(language_export, dict):
                counts["num_with_language_export"] += 1
            if reconstructed_from_summary:
                counts["num_reconstructed_from_summary"] += 1

    alignment_rows = _build_alignment_rows(alignment_map)
    summary = _build_summary(
        report_path=report_path,
        output_dir=output_dir,
        scene_setting_rows=scene_setting_rows,
        qa_rows=qa_rows,
        grounding_rows=grounding_rows,
        alignment_rows=alignment_rows,
        setting_export_counts=setting_export_counts,
    )

    scene_jsonl_path = output_dir / "language_scene_examples.jsonl"
    qa_jsonl_path = output_dir / "language_qa_examples.jsonl"
    grounding_jsonl_path = output_dir / "language_grounding_examples.jsonl"
    alignment_jsonl_path = output_dir / "language_alignment_examples.jsonl"
    summary_json_path = output_dir / "language_export_summary.json"
    summary_md_path = output_dir / "language_export_summary.md"

    _write_jsonl(scene_jsonl_path, scene_setting_rows)
    _write_jsonl(qa_jsonl_path, qa_rows)
    _write_jsonl(grounding_jsonl_path, grounding_rows)
    _write_jsonl(alignment_jsonl_path, alignment_rows)
    summary_json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    summary_md_path.write_text(_render_summary_markdown(summary), encoding="utf-8")

    print(
        json.dumps(
            {
                "language_scene_examples_jsonl": str(scene_jsonl_path),
                "language_qa_examples_jsonl": str(qa_jsonl_path),
                "language_grounding_examples_jsonl": str(grounding_jsonl_path),
                "language_alignment_examples_jsonl": str(alignment_jsonl_path),
                "language_export_summary_json": str(summary_json_path),
                "language_export_summary_markdown": str(summary_md_path),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _resolve_prediction(
    *,
    scene_id: str,
    setting_name: str,
    prediction_payload: Any,
    prediction_summary: Any,
) -> ScenePrediction:
    if isinstance(prediction_payload, dict):
        try:
            return ScenePrediction.from_dict(prediction_payload)
        except Exception:
            pass

    # Graceful fallback for older reports without structured predictions.
    object_labels: list[str] = []
    relation_predicates: list[str] = []
    object_count = 0
    relation_count = 0
    if isinstance(prediction_summary, dict):
        labels = prediction_summary.get("object_labels", [])
        predicates = prediction_summary.get("relation_predicates", [])
        if isinstance(labels, list):
            object_labels = [str(label) for label in labels]
        if isinstance(predicates, list):
            relation_predicates = [str(item) for item in predicates]
        object_count = int(prediction_summary.get("object_count", len(object_labels)) or 0)
        relation_count = int(prediction_summary.get("relation_count", len(relation_predicates)) or 0)

    objects: list[SceneObject] = []
    for index in range(max(object_count, len(object_labels))):
        label = object_labels[index] if index < len(object_labels) else "object"
        objects.append(
            SceneObject(
                object_id=f"{label}_{index}",
                label=label,
                position=Point3D(0.0, 0.0, 0.0),
                size=Point3D(1.0, 1.0, 1.0),
                confidence=0.5,
                attributes={
                    "reconstructed_from_prediction_summary": True,
                    "geometry_unavailable": True,
                },
            )
        )

    # Predicates are preserved only as metadata when full relation structure is missing.
    metadata = {
        "reconstructed_from_prediction_summary": True,
        "relation_predicates": relation_predicates,
        "relation_count_hint": relation_count,
    }
    return ScenePrediction(
        sample_id=scene_id,
        generator_name=f"{setting_name}_summary_fallback",
        objects=objects,
        relations=[],
        metadata=metadata,
    )


def _enrich_prediction_with_summary_hints(
    *,
    prediction: ScenePrediction,
    prediction_summary: Any,
) -> ScenePrediction:
    metadata = dict(prediction.metadata) if isinstance(prediction.metadata, dict) else {}

    summary_object_count = None
    summary_relation_count = None
    summary_relation_predicates: list[str] = []
    if isinstance(prediction_summary, dict):
        summary_object_count = _safe_int(prediction_summary.get("object_count"))
        summary_relation_count = _safe_int(prediction_summary.get("relation_count"))
        predicates = prediction_summary.get("relation_predicates", [])
        if isinstance(predicates, list):
            summary_relation_predicates = sorted(
                {str(item).strip() for item in predicates if str(item).strip()}
            )

    if "relation_count_hint" not in metadata and summary_relation_count is not None:
        metadata["relation_count_hint"] = summary_relation_count
    if "relation_predicates" not in metadata and summary_relation_predicates:
        metadata["relation_predicates"] = summary_relation_predicates
    if "object_count_hint" not in metadata and summary_object_count is not None:
        metadata["object_count_hint"] = summary_object_count

    prediction.metadata = metadata
    return prediction


def _build_alignment_rows(alignment_map: dict[str, dict[str, dict[str, Any]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scene_id in sorted(alignment_map.keys()):
        by_setting = alignment_map[scene_id]
        v0_v1_diff = _build_pairwise_comparison(by_setting, "calibration_v0", "calibration_v1")
        v1_repair_diff = _build_pairwise_comparison(
            by_setting,
            "calibration_v1",
            "calibration_v1_plus_repair",
        )
        mock_external_diff = _build_pairwise_comparison(
            by_setting,
            "mock_generator",
            "external_generator",
        )
        comparison_examples = _build_alignment_comparison_examples(
            scene_id=scene_id,
            v0_v1_diff=v0_v1_diff,
            mock_external_diff=mock_external_diff,
        )
        row = {
            "scene_id": scene_id,
            "settings": by_setting,
            "available_settings": sorted(by_setting.keys()),
            "has_v0_v1_alignment": all(name in by_setting for name in ["calibration_v0", "calibration_v1"]),
            "has_repair_alignment": all(
                name in by_setting for name in ["calibration_v1", "calibration_v1_plus_repair"]
            ),
            "has_mock_external_alignment": all(
                name in by_setting for name in ["mock_generator", "external_generator"]
            ),
            "pairwise_differences": {
                "calibration_v0_vs_calibration_v1": v0_v1_diff,
                "calibration_v1_vs_calibration_v1_plus_repair": v1_repair_diff,
                "mock_generator_vs_external_generator": mock_external_diff,
            },
            "comparison_examples": comparison_examples,
        }
        rows.append(row)
    return rows


def _build_pairwise_comparison(
    by_setting: dict[str, dict[str, Any]],
    base_setting: str,
    target_setting: str,
) -> dict[str, Any] | None:
    base = by_setting.get(base_setting)
    target = by_setting.get(target_setting)
    if not isinstance(base, dict) or not isinstance(target, dict):
        return None

    base_labels = _to_str_set(base.get("object_labels"))
    target_labels = _to_str_set(target.get("object_labels"))
    base_predicates = _to_str_set(base.get("relation_predicates"))
    target_predicates = _to_str_set(target.get("relation_predicates"))
    base_hint_predicates = _to_str_set(base.get("relation_hint_predicates"))
    target_hint_predicates = _to_str_set(target.get("relation_hint_predicates"))

    return {
        "base_setting": base_setting,
        "target_setting": target_setting,
        "object_count_delta": _safe_int(target.get("object_count"), 0)
        - _safe_int(base.get("object_count"), 0),
        "relation_count_delta": _safe_int(target.get("relation_count"), 0)
        - _safe_int(base.get("relation_count"), 0),
        "object_labels_added": sorted(target_labels - base_labels),
        "object_labels_removed": sorted(base_labels - target_labels),
        "relation_predicates_added": sorted(target_predicates - base_predicates),
        "relation_predicates_removed": sorted(base_predicates - target_predicates),
        "relation_hint_predicates_added": sorted(target_hint_predicates - base_hint_predicates),
        "relation_hint_predicates_removed": sorted(base_hint_predicates - target_hint_predicates),
        "base_relation_evidence_level": str(base.get("relation_evidence_level", "unknown")),
        "target_relation_evidence_level": str(target.get("relation_evidence_level", "unknown")),
        "base_reconstructed_from_summary": bool(base.get("reconstructed_from_prediction_summary", False)),
        "target_reconstructed_from_summary": bool(target.get("reconstructed_from_prediction_summary", False)),
    }


def _build_alignment_comparison_examples(
    *,
    scene_id: str,
    v0_v1_diff: dict[str, Any] | None,
    mock_external_diff: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if isinstance(v0_v1_diff, dict):
        rows.append(
            {
                "scene_id": scene_id,
                "task_type": "setting_delta_qa",
                "question": "How does calibration_v1 differ from calibration_v0 for this scene?",
                "answer": _render_pairwise_delta_answer(v0_v1_diff),
                "metadata": v0_v1_diff,
            }
        )

    if isinstance(mock_external_diff, dict):
        labels_added = mock_external_diff.get("object_labels_added", [])
        if not isinstance(labels_added, list):
            labels_added = []
        rows.append(
            {
                "scene_id": scene_id,
                "task_type": "setting_delta_qa",
                "question": "Which labels appear in external_generator but not mock_generator?",
                "answer": ", ".join(labels_added) if labels_added else "none",
                "metadata": {
                    "base_setting": "mock_generator",
                    "target_setting": "external_generator",
                    "object_labels_added": labels_added,
                },
            }
        )
    return rows


def _render_pairwise_delta_answer(diff: dict[str, Any]) -> str:
    labels_added = diff.get("object_labels_added", [])
    labels_removed = diff.get("object_labels_removed", [])
    predicates_added = diff.get("relation_predicates_added", [])
    predicates_removed = diff.get("relation_predicates_removed", [])

    if not isinstance(labels_added, list):
        labels_added = []
    if not isinstance(labels_removed, list):
        labels_removed = []
    if not isinstance(predicates_added, list):
        predicates_added = []
    if not isinstance(predicates_removed, list):
        predicates_removed = []

    parts = [
        f"object_count_delta={diff.get('object_count_delta', 0)}",
        f"relation_count_delta={diff.get('relation_count_delta', 0)}",
        f"labels_added={','.join(labels_added) if labels_added else 'none'}",
        f"labels_removed={','.join(labels_removed) if labels_removed else 'none'}",
        f"predicates_added={','.join(predicates_added) if predicates_added else 'none'}",
        f"predicates_removed={','.join(predicates_removed) if predicates_removed else 'none'}",
        f"relation_evidence={diff.get('base_relation_evidence_level')}->{diff.get('target_relation_evidence_level')}",
    ]
    return "; ".join(parts)


def _build_summary(
    *,
    report_path: Path,
    output_dir: Path,
    scene_setting_rows: list[dict[str, Any]],
    qa_rows: list[dict[str, Any]],
    grounding_rows: list[dict[str, Any]],
    alignment_rows: list[dict[str, Any]],
    setting_export_counts: dict[str, dict[str, int]],
) -> dict[str, Any]:
    setting_counts = {
        setting: sum(1 for row in scene_setting_rows if row.get("setting") == setting)
        for setting in sorted({row.get("setting") for row in scene_setting_rows})
    }
    required_coverage = {setting: setting_export_counts.get(setting, {}).get("num_success", 0) for setting in CORE_SETTINGS}
    num_reconstructed_examples = sum(
        1
        for row in scene_setting_rows
        if bool(
            isinstance(row.get("metadata"), dict)
            and row["metadata"].get("reconstructed_from_prediction_summary", False)
        )
    )
    num_hinted_relation_only_examples = sum(
        1 for row in scene_setting_rows if row.get("relation_evidence_level") == "hinted"
    )
    num_explicit_relation_examples = sum(
        1 for row in scene_setting_rows if row.get("relation_evidence_level") == "explicit"
    )

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "evaluation_report_path": str(report_path),
        "output_dir": str(output_dir),
        "num_scene_setting_examples": len(scene_setting_rows),
        "num_unique_scenes": len({str(row.get("scene_id")) for row in scene_setting_rows}),
        "num_qa_examples": len(qa_rows),
        "num_grounding_examples": len(grounding_rows),
        "num_alignment_examples": len(alignment_rows),
        "num_scene_v0_v1_aligned": sum(1 for row in alignment_rows if bool(row.get("has_v0_v1_alignment"))),
        "num_scene_v1_repair_aligned": sum(
            1 for row in alignment_rows if bool(row.get("has_repair_alignment"))
        ),
        "num_scene_mock_external_aligned": sum(
            1 for row in alignment_rows if bool(row.get("has_mock_external_alignment"))
        ),
        "num_reconstructed_scene_setting_examples": num_reconstructed_examples,
        "num_hinted_relation_only_examples": num_hinted_relation_only_examples,
        "num_explicit_relation_examples": num_explicit_relation_examples,
        "setting_counts": setting_counts,
        "setting_export_counts": setting_export_counts,
        "required_setting_coverage": required_coverage,
        "all_required_settings_export_cleanly": all(value > 0 for value in required_coverage.values()),
    }


def _render_summary_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Language Export Summary",
        "",
        f"- evaluation_report_path: `{summary.get('evaluation_report_path')}`",
        f"- output_dir: `{summary.get('output_dir')}`",
        f"- num_scene_setting_examples: `{summary.get('num_scene_setting_examples')}`",
        f"- num_unique_scenes: `{summary.get('num_unique_scenes')}`",
        f"- num_qa_examples: `{summary.get('num_qa_examples')}`",
        f"- num_grounding_examples: `{summary.get('num_grounding_examples')}`",
        f"- num_alignment_examples: `{summary.get('num_alignment_examples')}`",
        f"- num_reconstructed_scene_setting_examples: `{summary.get('num_reconstructed_scene_setting_examples')}`",
        f"- num_hinted_relation_only_examples: `{summary.get('num_hinted_relation_only_examples')}`",
        f"- num_explicit_relation_examples: `{summary.get('num_explicit_relation_examples')}`",
        "",
        "## Alignment Coverage",
        f"- num_scene_v0_v1_aligned: `{summary.get('num_scene_v0_v1_aligned')}`",
        f"- num_scene_v1_repair_aligned: `{summary.get('num_scene_v1_repair_aligned')}`",
        f"- num_scene_mock_external_aligned: `{summary.get('num_scene_mock_external_aligned')}`",
        "",
        "## Required Settings",
        f"- all_required_settings_export_cleanly: `{summary.get('all_required_settings_export_cleanly')}`",
    ]

    required = summary.get("required_setting_coverage", {})
    if isinstance(required, dict):
        for setting in CORE_SETTINGS:
            lines.append(f"- {setting}: `{required.get(setting, 0)}`")

    lines.append("")
    lines.append("## Setting Counts")
    setting_counts = summary.get("setting_counts", {})
    if isinstance(setting_counts, dict) and setting_counts:
        for setting in sorted(setting_counts.keys()):
            lines.append(f"- {setting}: `{setting_counts[setting]}`")
    else:
        lines.append("- none")
    return "\n".join(lines)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True))
            handle.write("\n")


def _to_str_set(value: Any) -> set[str]:
    if not isinstance(value, list):
        return set()
    return {str(item).strip() for item in value if str(item).strip()}


def _safe_int(value: Any, default: int | None = None) -> int | None:
    try:
        return int(value)
    except Exception:
        return default


def _none_if_empty(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


if __name__ == "__main__":
    raise SystemExit(main())
