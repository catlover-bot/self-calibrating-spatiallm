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

            prediction, source_info = _resolve_prediction_with_provenance(
                scene_id=scene_id,
                setting_name=setting_name,
                setting_metadata=metadata,
                prediction_summary=metadata.get("prediction_summary_pre_repair"),
                report_dir=report_path.parent,
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
                "prediction_source_class": source_info.get("source_class"),
                "qa_examples": qa_examples,
                "grounding_examples": grounding_examples,
                "metadata": {
                    "prediction_source": source_info.get("selected_from"),
                    "prediction_source_class": source_info.get("source_class"),
                    "prediction_source_selected_from": source_info.get("selected_from"),
                    "prediction_source_trace": source_info.get("trace", []),
                    "prediction_evidence_level": source_info.get("evidence_level"),
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
                "relation_statement_count": len(language_export.get("relation_statements", []))
                if isinstance(language_export.get("relation_statements"), list)
                else 0,
                "reconstructed_from_prediction_summary": reconstructed_from_summary,
                "calibration_method": metadata.get("calibration_method"),
                "generator_mode": metadata.get("generator_mode"),
                "prediction_source_class": source_info.get("source_class"),
                "prediction_source_selected_from": source_info.get("selected_from"),
            }

            counts = setting_export_counts.setdefault(
                setting_name,
                {
                    "num_success": 0,
                    "num_with_structured_prediction": 0,
                    "num_with_language_export": 0,
                    "num_reconstructed_from_summary": 0,
                    "num_source_explicit_structured_prediction": 0,
                    "num_source_structured_prediction_with_hint_only": 0,
                    "num_source_summary_reconstructed": 0,
                },
            )
            counts["num_success"] += 1
            if bool(source_info.get("has_structured_prediction", False)):
                counts["num_with_structured_prediction"] += 1
            if isinstance(language_export, dict):
                counts["num_with_language_export"] += 1
            if reconstructed_from_summary:
                counts["num_reconstructed_from_summary"] += 1
            source_class = str(source_info.get("source_class", "unknown"))
            if source_class == "explicit_structured_prediction":
                counts["num_source_explicit_structured_prediction"] += 1
            elif source_class == "structured_prediction_with_hint_only":
                counts["num_source_structured_prediction_with_hint_only"] += 1
            elif source_class == "summary_reconstructed":
                counts["num_source_summary_reconstructed"] += 1

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


def _resolve_prediction_with_provenance(
    *,
    scene_id: str,
    setting_name: str,
    setting_metadata: dict[str, Any],
    prediction_summary: Any,
    report_dir: Path | None,
) -> tuple[ScenePrediction, dict[str, Any]]:
    trace: list[str] = []

    parsed_candidates: list[dict[str, Any]] = []
    for candidate in _candidate_prediction_payloads(setting_metadata, report_dir=report_dir):
        source_name = str(candidate.get("source", "unknown"))
        payload = candidate.get("payload")
        if not isinstance(payload, dict):
            if payload is not None:
                trace.append(f"{source_name}:not_object")
            continue

        parsed = _parse_prediction_payload_variants(
            payload=payload,
            scene_id=scene_id,
            setting_name=setting_name,
            source_name=source_name,
            source_priority=int(candidate.get("priority", _prediction_source_priority(source_name))),
            trace=trace,
        )
        if parsed is not None:
            parsed_candidates.append(parsed)

    if parsed_candidates:
        best = sorted(
            parsed_candidates,
            key=lambda item: (
                -int(item["rank"]),
                int(item["priority"]),
            ),
        )[0]
        prediction = best["prediction"]
        source_class = str(best["source_class"])
        source_name = str(best["source_name"])
        prediction_metadata = dict(prediction.metadata) if isinstance(prediction.metadata, dict) else {}
        prediction_metadata.setdefault("prediction_source_class", source_class)
        prediction_metadata.setdefault("prediction_source_selected_from", source_name)
        prediction.metadata = prediction_metadata
        return prediction, {
            "selected_from": source_name,
            "source_class": source_class,
            "evidence_level": _relation_evidence_level(prediction),
            "has_structured_prediction": True,
            "trace": trace,
        }

    recovered_prediction = _recover_prediction_from_metadata_artifacts(
        scene_id=scene_id,
        setting_name=setting_name,
        setting_metadata=setting_metadata,
        prediction_summary=prediction_summary,
        trace=trace,
    )
    if recovered_prediction is not None:
        recovered_prediction = _normalize_prediction(recovered_prediction, fallback_scene_id=scene_id)
        source_class = _classify_source_class(recovered_prediction)
        return recovered_prediction, {
            "selected_from": "metadata_relation_recovery",
            "source_class": source_class,
            "evidence_level": _relation_evidence_level(recovered_prediction),
            "has_structured_prediction": False,
            "trace": trace,
        }

    prediction = _build_summary_fallback_prediction(
        scene_id=scene_id,
        setting_name=setting_name,
        prediction_summary=prediction_summary,
    )
    trace.append("summary_fallback:used")
    return prediction, {
        "selected_from": "prediction_summary_pre_repair",
        "source_class": "summary_reconstructed",
        "evidence_level": _relation_evidence_level(prediction),
        "has_structured_prediction": False,
        "trace": trace,
    }


def _candidate_prediction_payloads(
    setting_metadata: dict[str, Any],
    *,
    report_dir: Path | None,
) -> list[dict[str, Any]]:
    keys = [
        "structured_prediction_pre_repair",
        "structured_prediction",
        "scene_prediction_pre_repair",
        "scene_prediction",
        "structured_prediction_post_repair",
        "scene_prediction_post_repair",
    ]
    candidates: list[dict[str, Any]] = []
    seen_sources: set[str] = set()

    contract = setting_metadata.get("prediction_source_contract")
    if isinstance(contract, dict):
        precedence = contract.get("source_precedence", [])
        if isinstance(precedence, list):
            priority_index = 0
            for class_name in precedence:
                entry = contract.get(str(class_name))
                if not isinstance(entry, dict):
                    continue
                inline_key = entry.get("inline_key")
                if isinstance(inline_key, str) and inline_key in keys and inline_key not in seen_sources:
                    candidates.append(
                        {
                            "source": inline_key,
                            "payload": setting_metadata.get(inline_key),
                            "priority": priority_index,
                        }
                    )
                    seen_sources.add(inline_key)
                    priority_index += 1

                artifact_key = entry.get("artifact_key")
                source_name = f"artifact:{artifact_key}" if isinstance(artifact_key, str) else None
                if source_name and source_name not in seen_sources:
                    artifact_payload = _load_single_prediction_artifact_payload(
                        setting_metadata,
                        report_dir=report_dir,
                        artifact_key=artifact_key,
                    )
                    candidates.append(
                        {
                            "source": source_name,
                            "payload": artifact_payload,
                            "priority": priority_index,
                        }
                    )
                    seen_sources.add(source_name)
                    priority_index += 1

    for index, key in enumerate(keys):
        if key in seen_sources:
            continue
        candidates.append(
            {
                "source": key,
                "payload": setting_metadata.get(key),
                "priority": index,
            }
        )
        seen_sources.add(key)

    for index, artifact in enumerate(_load_prediction_artifact_payloads(setting_metadata, report_dir=report_dir)):
        if str(artifact["source"]) in seen_sources:
            continue
        candidates.append(
            {
                "source": artifact["source"],
                "payload": artifact["payload"],
                "priority": len(keys) + index,
            }
        )
        seen_sources.add(str(artifact["source"]))
    return candidates


def _load_prediction_artifact_payloads(
    setting_metadata: dict[str, Any],
    *,
    report_dir: Path | None,
) -> list[dict[str, Any]]:
    if report_dir is None:
        return []

    artifact_paths = setting_metadata.get("prediction_artifact_paths")
    if not isinstance(artifact_paths, dict):
        return []

    keys = [
        "structured_prediction_pre_repair",
        "structured_prediction",
        "scene_prediction_pre_repair",
        "scene_prediction",
        "structured_prediction_post_repair",
        "scene_prediction_post_repair",
    ]
    results: list[dict[str, Any]] = []
    for key in keys:
        rel = artifact_paths.get(key)
        if not isinstance(rel, str) or not rel.strip():
            continue
        path = (report_dir / rel).resolve() if not Path(rel).is_absolute() else Path(rel)
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        results.append(
            {
                "source": f"artifact:{key}",
                "payload": payload,
                "path": str(path),
            }
        )
    return results


def _load_single_prediction_artifact_payload(
    setting_metadata: dict[str, Any],
    *,
    report_dir: Path | None,
    artifact_key: str,
) -> dict[str, Any] | None:
    if report_dir is None:
        return None
    artifact_paths = setting_metadata.get("prediction_artifact_paths")
    if not isinstance(artifact_paths, dict):
        return None
    rel = artifact_paths.get(artifact_key)
    if not isinstance(rel, str) or not rel.strip():
        return None
    path = (report_dir / rel).resolve() if not Path(rel).is_absolute() else Path(rel)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _parse_prediction_payload_variants(
    *,
    payload: dict[str, Any],
    scene_id: str,
    setting_name: str,
    source_name: str,
    source_priority: int,
    trace: list[str],
) -> dict[str, Any] | None:
    variants = _prediction_payload_variants(payload)
    for variant_name, variant_payload in variants:
        if not isinstance(variant_payload, dict):
            continue
        normalized_payload = dict(variant_payload)
        normalized_payload.setdefault("sample_id", scene_id)
        normalized_payload.setdefault("generator_name", f"{setting_name}_from_{source_name}")
        try:
            prediction = ScenePrediction.from_dict(normalized_payload)
            prediction = _normalize_prediction(prediction, fallback_scene_id=scene_id)
        except Exception as error:
            trace.append(f"{source_name}:{variant_name}:parse_failed:{type(error).__name__}")
            continue

        source_class = _classify_source_class(prediction)
        rank = _source_class_rank(source_class)
        trace.append(f"{source_name}:{variant_name}:parsed:{source_class}")
        return {
            "prediction": prediction,
            "source_name": source_name,
            "source_class": source_class,
            "rank": rank,
            "priority": source_priority,
        }
    trace.append(f"{source_name}:no_valid_variant")
    return None


def _prediction_payload_variants(payload: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    variants: list[tuple[str, dict[str, Any]]] = []
    if isinstance(payload.get("scene_prediction"), dict):
        variants.append(("scene_prediction", dict(payload["scene_prediction"])))
    if isinstance(payload.get("prediction"), dict):
        variants.append(("prediction", dict(payload["prediction"])))
    if isinstance(payload.get("data"), dict):
        variants.append(("data", dict(payload["data"])))
    variants.append(("payload", dict(payload)))
    return variants


def _source_class_rank(source_class: str) -> int:
    order = {
        "explicit_structured_prediction": 3,
        "structured_prediction_with_hint_only": 2,
        "summary_reconstructed": 1,
    }
    return order.get(source_class, 0)


def _prediction_source_priority(source_name: str) -> int:
    priority_order = [
        "structured_prediction_pre_repair",
        "artifact:structured_prediction_pre_repair",
        "structured_prediction",
        "artifact:structured_prediction",
        "scene_prediction_pre_repair",
        "artifact:scene_prediction_pre_repair",
        "scene_prediction",
        "artifact:scene_prediction",
        "structured_prediction_post_repair",
        "artifact:structured_prediction_post_repair",
        "scene_prediction_post_repair",
        "artifact:scene_prediction_post_repair",
    ]
    try:
        return priority_order.index(source_name)
    except ValueError:
        return len(priority_order) + 1


def _recover_prediction_from_metadata_artifacts(
    *,
    scene_id: str,
    setting_name: str,
    setting_metadata: dict[str, Any],
    prediction_summary: Any,
    trace: list[str],
) -> ScenePrediction | None:
    relation_rows = _extract_explicit_relation_rows(setting_metadata)
    object_rows = _extract_object_rows_from_metadata(setting_metadata)

    if relation_rows:
        trace.append(f"relation_rows:recovered:{len(relation_rows)}")
        objects = _build_objects_from_rows(
            object_rows,
            fallback_relation_rows=relation_rows,
        )
        relations = _build_relations_from_rows(relation_rows)
        metadata = _base_recovered_metadata(
            prediction_summary=prediction_summary,
            source_key="metadata_relation_rows",
            relation_count=len(relations),
            relation_predicates=sorted({rel.predicate for rel in relations}),
        )
        return ScenePrediction(
            sample_id=scene_id,
            generator_name=f"{setting_name}_metadata_relation_recovery",
            objects=objects,
            relations=relations,
            metadata=metadata,
        )

    language_export = setting_metadata.get("language_export_pre_repair")
    if isinstance(language_export, dict):
        relation_statements = language_export.get("relation_statements", [])
        if isinstance(relation_statements, list):
            parsed_relations, parsed_objects = _parse_relation_statements(relation_statements)
            if parsed_relations:
                trace.append(f"language_export_relation_statements:recovered:{len(parsed_relations)}")
                objects = _build_objects_from_rows(
                    object_rows,
                    fallback_relation_rows=parsed_relations,
                    parsed_objects=parsed_objects,
                )
                relations = _build_relations_from_rows(parsed_relations)
                metadata = _base_recovered_metadata(
                    prediction_summary=prediction_summary,
                    source_key="language_export_pre_repair.relation_statements",
                    relation_count=len(relations),
                    relation_predicates=sorted({rel.predicate for rel in relations}),
                )
                return ScenePrediction(
                    sample_id=scene_id,
                    generator_name=f"{setting_name}_language_relation_recovery",
                    objects=objects,
                    relations=relations,
                    metadata=metadata,
                )

    return None


def _build_summary_fallback_prediction(
    *,
    scene_id: str,
    setting_name: str,
    prediction_summary: Any,
) -> ScenePrediction:
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
        objects.append(_placeholder_object(object_id=f"{label}_{index}", label=label))

    metadata = {
        "reconstructed_from_prediction_summary": True,
        "relation_predicates": relation_predicates,
        "relation_count_hint": relation_count,
        "prediction_source_class": "summary_reconstructed",
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


def _normalize_prediction(prediction: ScenePrediction, *, fallback_scene_id: str) -> ScenePrediction:
    sample_id = str(prediction.sample_id or fallback_scene_id)
    generator_name = str(prediction.generator_name or "unknown_generator")
    metadata = dict(prediction.metadata) if isinstance(prediction.metadata, dict) else {}

    normalized_objects: list[SceneObject] = []
    for index, obj in enumerate(prediction.objects):
        obj_attributes = dict(obj.attributes) if isinstance(obj.attributes, dict) else {}
        normalized_objects.append(
            SceneObject(
                object_id=str(obj.object_id or f"object_{index}"),
                label=str(obj.label or "object"),
                position=Point3D(float(obj.position.x), float(obj.position.y), float(obj.position.z)),
                size=Point3D(
                    max(float(obj.size.x), 0.0),
                    max(float(obj.size.y), 0.0),
                    max(float(obj.size.z), 0.0),
                ),
                confidence=float(obj.confidence),
                attributes=obj_attributes,
            )
        )

    normalized_relations = []
    for rel in prediction.relations:
        subject_id = str(rel.subject_id).strip()
        object_id = str(rel.object_id).strip()
        predicate = str(rel.predicate).strip()
        if not subject_id or not object_id or not predicate:
            continue
        normalized_relations.append(rel)

    return ScenePrediction(
        sample_id=sample_id,
        generator_name=generator_name,
        objects=normalized_objects,
        relations=normalized_relations,
        metadata=metadata,
    )


def _classify_source_class(prediction: ScenePrediction) -> str:
    if prediction.relations:
        return "explicit_structured_prediction"
    metadata = prediction.metadata if isinstance(prediction.metadata, dict) else {}
    hint_count = _safe_int(metadata.get("relation_count_hint"), 0) or 0
    hint_predicates = metadata.get("relation_predicates", [])
    if hint_count > 0 or (isinstance(hint_predicates, list) and bool(hint_predicates)):
        return "structured_prediction_with_hint_only"
    if bool(metadata.get("reconstructed_from_prediction_summary", False)):
        return "summary_reconstructed"
    return "structured_prediction_with_hint_only"


def _relation_evidence_level(prediction: ScenePrediction) -> str:
    if prediction.relations:
        return "explicit"
    metadata = prediction.metadata if isinstance(prediction.metadata, dict) else {}
    hint_count = _safe_int(metadata.get("relation_count_hint"), 0) or 0
    hint_predicates = metadata.get("relation_predicates", [])
    if hint_count > 0 or (isinstance(hint_predicates, list) and bool(hint_predicates)):
        return "hinted"
    return "none"


def _extract_explicit_relation_rows(setting_metadata: dict[str, Any]) -> list[dict[str, Any]]:
    keys = [
        "relations_pre_repair",
        "relation_tuples_pre_repair",
        "relation_tuples",
        "explicit_relations",
        "scene_relations",
        "prediction_relations",
    ]
    for key in keys:
        rows = setting_metadata.get(key)
        if not isinstance(rows, list):
            continue
        normalized = _normalize_relation_rows(rows)
        if normalized:
            return normalized
    return []


def _extract_object_rows_from_metadata(setting_metadata: dict[str, Any]) -> list[dict[str, Any]]:
    keys = [
        "objects_pre_repair",
        "scene_objects",
        "prediction_objects",
        "objects",
    ]
    for key in keys:
        rows = setting_metadata.get(key)
        if not isinstance(rows, list):
            continue
        normalized = _normalize_object_rows(rows)
        if normalized:
            return normalized
    return []


def _normalize_relation_rows(rows: list[Any]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        subject_id = _first_nonempty_str(row.get("subject_id"), row.get("subject"), row.get("source_id"))
        object_id = _first_nonempty_str(row.get("object_id"), row.get("object"), row.get("target_id"))
        predicate = _first_nonempty_str(row.get("predicate"), row.get("relation"), row.get("type"))
        if not subject_id or not object_id or not predicate:
            continue
        score = _safe_float(row.get("score"), 1.0) or 1.0
        metadata = row.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}
        normalized.append(
            {
                "subject_id": subject_id,
                "predicate": predicate,
                "object_id": object_id,
                "score": score,
                "metadata": metadata,
            }
        )
    return normalized


def _normalize_object_rows(rows: list[Any]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        object_id = _first_nonempty_str(row.get("object_id"), row.get("id"))
        label = _first_nonempty_str(row.get("label"), row.get("category"), row.get("class_name"), default="object")
        if not object_id:
            continue
        normalized.append({"object_id": object_id, "label": label, "raw": row})
    return normalized


def _build_objects_from_rows(
    object_rows: list[dict[str, Any]],
    *,
    fallback_relation_rows: list[dict[str, Any]],
    parsed_objects: dict[str, str] | None = None,
) -> list[SceneObject]:
    by_id: dict[str, SceneObject] = {}
    for row in object_rows:
        object_id = str(row.get("object_id", "")).strip()
        label = str(row.get("label", "object")).strip() or "object"
        if not object_id:
            continue
        by_id[object_id] = _placeholder_object(object_id=object_id, label=label)

    if parsed_objects:
        for object_id, label in parsed_objects.items():
            if object_id not in by_id:
                by_id[object_id] = _placeholder_object(object_id=object_id, label=label)

    for rel in fallback_relation_rows:
        for object_id_key in ("subject_id", "object_id"):
            object_id = str(rel.get(object_id_key, "")).strip()
            if not object_id or object_id in by_id:
                continue
            label_from_meta = None
            rel_meta = rel.get("metadata", {})
            if isinstance(rel_meta, dict):
                label_from_meta = rel_meta.get(
                    "subject_label" if object_id_key == "subject_id" else "object_label"
                )
            label = _guess_label_from_object_id(object_id, default=_safe_str(label_from_meta, "object"))
            by_id[object_id] = _placeholder_object(object_id=object_id, label=label)
    return sorted(by_id.values(), key=lambda obj: (obj.label, obj.object_id))


def _build_relations_from_rows(rows: list[dict[str, Any]]) -> list[Any]:
    from self_calibrating_spatiallm.artifacts import SceneRelation

    relations: list[Any] = []
    for row in rows:
        metadata = row.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}
        relations.append(
            SceneRelation(
                subject_id=str(row["subject_id"]),
                predicate=str(row["predicate"]),
                object_id=str(row["object_id"]),
                score=float(row.get("score", 1.0)),
                metadata=metadata,
            )
        )
    return relations


def _parse_relation_statements(
    relation_statements: list[Any],
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    import re

    pattern = re.compile(
        r"^(?P<subject_label>[^\[]+)\[(?P<subject_id>[^\]]+)\]\s+"
        r"(?P<predicate>\S+)\s+"
        r"(?P<object_label>[^\[]+)\[(?P<object_id>[^\]]+)\]"
        r"(?:\s+\(score=(?P<score>[-+]?[0-9]*\.?[0-9]+)\))?$"
    )
    relations: list[dict[str, Any]] = []
    objects: dict[str, str] = {}
    for raw in relation_statements:
        text = str(raw).strip()
        if not text:
            continue
        match = pattern.match(text)
        if not match:
            continue
        subject_id = match.group("subject_id").strip()
        object_id = match.group("object_id").strip()
        predicate = match.group("predicate").strip()
        if not subject_id or not object_id or not predicate:
            continue
        subject_label = match.group("subject_label").strip() or _guess_label_from_object_id(subject_id, default="object")
        object_label = match.group("object_label").strip() or _guess_label_from_object_id(object_id, default="object")
        score = _safe_float(match.group("score"), 1.0) or 1.0
        relations.append(
            {
                "subject_id": subject_id,
                "predicate": predicate,
                "object_id": object_id,
                "score": score,
                "metadata": {
                    "subject_label": subject_label,
                    "object_label": object_label,
                    "source": "relation_statements",
                },
            }
        )
        objects[subject_id] = subject_label
        objects[object_id] = object_label
    return relations, objects


def _base_recovered_metadata(
    *,
    prediction_summary: Any,
    source_key: str,
    relation_count: int,
    relation_predicates: list[str],
) -> dict[str, Any]:
    metadata = {
        "reconstructed_from_prediction_summary": False,
        "recovered_from_metadata": True,
        "recovery_source": source_key,
        "relation_count_hint": relation_count,
        "relation_predicates": relation_predicates,
        "prediction_source_class": "explicit_structured_prediction" if relation_count > 0 else "structured_prediction_with_hint_only",
        "prediction_source_selected_from": source_key,
    }
    if isinstance(prediction_summary, dict):
        object_count = _safe_int(prediction_summary.get("object_count"))
        if object_count is not None:
            metadata["object_count_hint"] = object_count
    return metadata


def _placeholder_object(*, object_id: str, label: str) -> SceneObject:
    return SceneObject(
        object_id=object_id,
        label=label,
        position=Point3D(0.0, 0.0, 0.0),
        size=Point3D(1.0, 1.0, 1.0),
        confidence=0.5,
        attributes={
            "geometry_unavailable": True,
            "approximate_from_metadata": True,
        },
    )


def _guess_label_from_object_id(object_id: str, default: str = "object") -> str:
    object_id = str(object_id).strip()
    if not object_id:
        return default
    token = object_id.split("_", 1)[0].strip().lower()
    if token:
        return token
    return default


def _first_nonempty_str(*values: Any, default: str | None = None) -> str | None:
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return default


def _safe_str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


def _safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        return float(value)
    except Exception:
        return default


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
        "relation_tuple_count_delta": _safe_int(target.get("relation_count"), 0)
        - _safe_int(base.get("relation_count"), 0),
        "relation_statement_count_delta": _safe_int(target.get("relation_statement_count"), 0)
        - _safe_int(base.get("relation_statement_count"), 0),
        "object_labels_added": sorted(target_labels - base_labels),
        "object_labels_removed": sorted(base_labels - target_labels),
        "relation_predicates_added": sorted(target_predicates - base_predicates),
        "relation_predicates_removed": sorted(base_predicates - target_predicates),
        "relation_hint_predicates_added": sorted(target_hint_predicates - base_hint_predicates),
        "relation_hint_predicates_removed": sorted(base_hint_predicates - target_hint_predicates),
        "base_relation_evidence_level": str(base.get("relation_evidence_level", "unknown")),
        "target_relation_evidence_level": str(target.get("relation_evidence_level", "unknown")),
        "relation_evidence_transition": (
            f"{str(base.get('relation_evidence_level', 'unknown'))}"
            f"->{str(target.get('relation_evidence_level', 'unknown'))}"
        ),
        "base_reconstructed_from_summary": bool(base.get("reconstructed_from_prediction_summary", False)),
        "target_reconstructed_from_summary": bool(target.get("reconstructed_from_prediction_summary", False)),
        "base_prediction_source_class": str(base.get("prediction_source_class", "unknown")),
        "target_prediction_source_class": str(target.get("prediction_source_class", "unknown")),
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
        transition = str(v0_v1_diff.get("relation_evidence_transition", "unknown->unknown"))
        if transition != "unknown->unknown":
            rows.append(
                {
                    "scene_id": scene_id,
                    "task_type": "evidence_transition_qa",
                    "question": "How did relation evidence level change from calibration_v0 to calibration_v1?",
                    "answer": transition,
                    "metadata": {
                        "base_setting": "calibration_v0",
                        "target_setting": "calibration_v1",
                        "relation_evidence_transition": transition,
                    },
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
        transition = str(mock_external_diff.get("relation_evidence_transition", "unknown->unknown"))
        if transition != "unknown->unknown":
            rows.append(
                {
                    "scene_id": scene_id,
                    "task_type": "evidence_transition_qa",
                    "question": "How did relation evidence level change from mock_generator to external_generator?",
                    "answer": transition,
                    "metadata": {
                        "base_setting": "mock_generator",
                        "target_setting": "external_generator",
                        "relation_evidence_transition": transition,
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
        f"relation_tuple_count_delta={diff.get('relation_tuple_count_delta', 0)}",
        f"relation_statement_count_delta={diff.get('relation_statement_count_delta', 0)}",
        f"labels_added={','.join(labels_added) if labels_added else 'none'}",
        f"labels_removed={','.join(labels_removed) if labels_removed else 'none'}",
        f"predicates_added={','.join(predicates_added) if predicates_added else 'none'}",
        f"predicates_removed={','.join(predicates_removed) if predicates_removed else 'none'}",
        f"relation_evidence={diff.get('relation_evidence_transition')}",
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
    source_class_counts: dict[str, int] = {}
    for row in scene_setting_rows:
        metadata = row.get("metadata", {})
        source_class = None
        if isinstance(metadata, dict):
            source_class = metadata.get("prediction_source_class")
        key = str(source_class or "unknown")
        source_class_counts[key] = source_class_counts.get(key, 0) + 1

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
        "prediction_source_class_counts": source_class_counts,
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
    lines.append("## Prediction Source Class Counts")
    source_class_counts = summary.get("prediction_source_class_counts", {})
    if isinstance(source_class_counts, dict) and source_class_counts:
        for name in sorted(source_class_counts.keys()):
            lines.append(f"- {name}: `{source_class_counts[name]}`")
    else:
        lines.append("- unknown: `0`")

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
