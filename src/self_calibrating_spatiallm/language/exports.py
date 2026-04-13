"""Deterministic scene-to-language export helpers."""

from __future__ import annotations

from collections import Counter
from typing import Any

from self_calibrating_spatiallm.artifacts import ScenePrediction


def export_scene_prediction_to_language(prediction: ScenePrediction) -> dict[str, Any]:
    """Convert a structured scene prediction into deterministic language-facing forms."""
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

    object_lines: list[str] = []
    for obj in objects:
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
    if not predicate_summary:
        predicate_summary = "none"

    scene_summary_text = (
        f"Scene {prediction.sample_id} contains {len(objects)} objects "
        f"({label_summary}) and {len(relations)} relations ({predicate_summary})."
    )
    relation_text = (
        "No explicit relations were predicted."
        if not relation_statements
        else "Relations: " + "; ".join(relation_statements)
    )
    scene_paragraph_text = (
        f"{scene_summary_text} "
        f"{relation_text} "
        "This text is deterministically generated from structured scene outputs."
    )

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
    }


def export_scene_prediction_dict_to_language(prediction_payload: dict[str, Any]) -> dict[str, Any]:
    """Convert serialized ScenePrediction payload to deterministic language forms."""
    prediction = ScenePrediction.from_dict(prediction_payload)
    return export_scene_prediction_to_language(prediction)


def _fmt(value: float) -> str:
    return f"{float(value):.3f}"

