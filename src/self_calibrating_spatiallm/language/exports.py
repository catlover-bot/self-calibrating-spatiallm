"""Deterministic scene-to-language export helpers."""

from __future__ import annotations

from collections import Counter
from typing import Any

from self_calibrating_spatiallm.artifacts import ScenePrediction


def export_scene_prediction_to_language(prediction: ScenePrediction) -> dict[str, Any]:
    """Convert a structured scene prediction into deterministic language-facing forms."""
    prediction_metadata = _prediction_metadata(prediction)
    objects = sorted(
        prediction.objects,
        key=lambda obj: (str(obj.label).lower(), str(obj.object_id)),
    )
    relations = sorted(
        prediction.relations,
        key=lambda rel: (str(rel.predicate).lower(), str(rel.subject_id), str(rel.object_id)),
    )
    by_id = {obj.object_id: obj for obj in objects}
    label_counts = Counter(obj.label for obj in objects)
    predicate_counts = Counter(rel.predicate for rel in relations)
    relation_hint_predicates, relation_hint_count = _extract_relation_hints(
        prediction_metadata,
        fallback_count=len(relations),
    )
    reconstructed_from_summary = bool(
        prediction_metadata.get("reconstructed_from_prediction_summary", False)
    )
    prediction_source_class = _prediction_source_class(
        metadata=prediction_metadata,
        has_explicit_relations=bool(relations),
        has_hints=bool(relation_hint_count > 0 or relation_hint_predicates),
    )

    object_lines: list[str] = []
    num_reconstructed_objects = 0
    for obj in objects:
        is_reconstructed_object = _is_summary_reconstructed_object(
            obj_attributes=getattr(obj, "attributes", {}),
            prediction_reconstructed=reconstructed_from_summary,
        )
        if is_reconstructed_object:
            num_reconstructed_objects += 1
            object_lines.append(
                f"- {obj.object_id} ({obj.label}) [summary-reconstructed; geometry unavailable]"
            )
            continue

        object_lines.append(
            (
                f"- {obj.object_id} ({obj.label}) "
                f"pos=({_fmt(obj.position.x)},{_fmt(obj.position.y)},{_fmt(obj.position.z)}) "
                f"size=({_fmt(obj.size.x)},{_fmt(obj.size.y)},{_fmt(obj.size.z)}) "
                f"conf={_fmt(obj.confidence)}"
            )
        )

    relation_statements: list[str] = []
    for rel in relations:
        subject_label = by_id[rel.subject_id].label if rel.subject_id in by_id else rel.subject_id
        object_label = by_id[rel.object_id].label if rel.object_id in by_id else rel.object_id
        relation_statements.append(
            f"{subject_label}[{rel.subject_id}] {rel.predicate} {object_label}[{rel.object_id}] "
            f"(score={_fmt(rel.score)})"
        )

    label_summary = ", ".join(f"{label}:{count}" for label, count in sorted(label_counts.items()))
    predicate_summary = ", ".join(
        f"{predicate}:{count}" for predicate, count in sorted(predicate_counts.items())
    )
    if not label_summary:
        label_summary = "none"

    relation_evidence_level = "none"
    if relations:
        relation_evidence_level = "explicit"
    elif relation_hint_count > 0 or relation_hint_predicates:
        relation_evidence_level = "hinted"

    if not predicate_summary:
        if relation_hint_predicates:
            predicate_summary = ", ".join(f"{predicate}:hint" for predicate in relation_hint_predicates)
        elif relation_hint_count > 0:
            predicate_summary = "hinted"
        else:
            predicate_summary = "none"

    relation_summary_text = (
        f"{len(relations)} relations ({predicate_summary})"
        if relation_evidence_level == "explicit"
        else f"0 explicit relations; hinted relation evidence ({predicate_summary})"
        if relation_evidence_level == "hinted"
        else "0 relations (none)"
    )
    scene_summary_text = (
        f"Scene {prediction.sample_id} contains {len(objects)} objects "
        f"({label_summary}) and {relation_summary_text}."
    )

    if relation_evidence_level == "explicit":
        relation_text = "Relations: " + "; ".join(relation_statements)
    elif relation_evidence_level == "hinted":
        relation_text = _build_hinted_relation_text(
            relation_hint_count=relation_hint_count,
            relation_hint_predicates=relation_hint_predicates,
        )
    else:
        relation_text = "No explicit or hinted relation evidence was exported."

    reconstruction_note = ""
    if num_reconstructed_objects > 0:
        reconstruction_note = (
            f" Geometry is summary-reconstructed for {num_reconstructed_objects}/{len(objects)} objects."
        )

    scene_paragraph_text = (
        f"{scene_summary_text} "
        f"{relation_text} "
        "This text is deterministically generated from structured scene outputs."
        f"{reconstruction_note}"
    )

    object_geometry_mode = "grounded"
    if objects and num_reconstructed_objects == len(objects):
        object_geometry_mode = "summary_reconstructed"
    elif num_reconstructed_objects > 0:
        object_geometry_mode = "mixed"

    return {
        "scene_summary_text": scene_summary_text,
        "object_list_text": "\n".join(object_lines) if object_lines else "No objects were predicted.",
        "relation_statements": relation_statements,
        "relation_text": relation_text,
        "scene_paragraph_text": scene_paragraph_text,
        "object_count": len(objects),
        "relation_count": len(relations),
        "object_labels": sorted(label_counts.keys()),
        "relation_predicates": sorted(predicate_counts.keys()),
        "relation_evidence_level": relation_evidence_level,
        "relation_hint_count": int(relation_hint_count),
        "relation_hint_predicates": relation_hint_predicates,
        "reconstructed_from_prediction_summary": reconstructed_from_summary,
        "object_geometry_mode": object_geometry_mode,
        "prediction_source_class": prediction_source_class,
    }


def export_scene_prediction_dict_to_language(prediction_payload: dict[str, Any]) -> dict[str, Any]:
    """Convert serialized ScenePrediction payload to deterministic language forms."""
    prediction = ScenePrediction.from_dict(prediction_payload)
    return export_scene_prediction_to_language(prediction)


def _prediction_metadata(prediction: ScenePrediction) -> dict[str, Any]:
    metadata = getattr(prediction, "metadata", {})
    if not isinstance(metadata, dict):
        return {}
    return dict(metadata)


def _extract_relation_hints(metadata: dict[str, Any], fallback_count: int) -> tuple[list[str], int]:
    hint_predicates_raw = metadata.get("relation_predicates", [])
    hint_predicates: list[str] = []
    if isinstance(hint_predicates_raw, list):
        hint_predicates = sorted({str(item).strip() for item in hint_predicates_raw if str(item).strip()})

    hint_count = fallback_count
    hint_count_raw = metadata.get("relation_count_hint")
    if hint_count_raw is not None:
        try:
            hint_count = max(int(hint_count_raw), 0)
        except Exception:
            hint_count = fallback_count
    elif hint_predicates:
        hint_count = len(hint_predicates)
    return hint_predicates, hint_count


def _is_summary_reconstructed_object(*, obj_attributes: Any, prediction_reconstructed: bool) -> bool:
    if prediction_reconstructed:
        return True
    if not isinstance(obj_attributes, dict):
        return False
    return bool(
        obj_attributes.get("reconstructed_from_prediction_summary", False)
        or obj_attributes.get("geometry_unavailable", False)
    )


def _build_hinted_relation_text(*, relation_hint_count: int, relation_hint_predicates: list[str]) -> str:
    if relation_hint_predicates:
        predicates_text = ", ".join(relation_hint_predicates)
        return (
            "No explicit relation tuples were exported. "
            f"The scene includes {relation_hint_count} hinted relations with predicates: {predicates_text}."
        )
    if relation_hint_count > 0:
        return (
            "No explicit relation tuples were exported. "
            f"The scene includes {relation_hint_count} hinted relations with unknown predicates."
        )
    return "No explicit or hinted relation evidence was exported."


def _prediction_source_class(*, metadata: dict[str, Any], has_explicit_relations: bool, has_hints: bool) -> str:
    raw = metadata.get("prediction_source_class")
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    if has_explicit_relations:
        return "explicit_structured_prediction"
    if has_hints:
        return "structured_prediction_with_hint_only"
    if bool(metadata.get("reconstructed_from_prediction_summary", False)):
        return "summary_reconstructed"
    return "structured_prediction_with_hint_only"


def _fmt(value: float) -> str:
    return f"{float(value):.3f}"
