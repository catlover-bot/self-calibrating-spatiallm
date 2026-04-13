"""Deterministic NLP-style task builders from structured scene predictions."""

from __future__ import annotations

from collections import Counter
from typing import Any

from self_calibrating_spatiallm.artifacts import ScenePrediction
from self_calibrating_spatiallm.language.exports import export_scene_prediction_to_language


def build_qa_examples(prediction: ScenePrediction, max_labels: int = 6, max_relations: int = 6) -> list[dict[str, Any]]:
    """Build lightweight, deterministic scene-grounded QA examples."""
    by_id = {obj.object_id: obj for obj in prediction.objects}
    label_counts = Counter(obj.label for obj in prediction.objects)

    qa_examples: list[dict[str, Any]] = []
    for label in sorted(label_counts.keys())[:max_labels]:
        qa_examples.append(
            {
                "task_type": "count_objects",
                "question": f"How many {label} objects are in the scene?",
                "answer": str(int(label_counts[label])),
                "metadata": {"label": label},
            }
        )

    seen_relation_pairs: set[tuple[str, str, str]] = set()
    for rel in sorted(
        prediction.relations,
        key=lambda item: (str(item.predicate), str(item.subject_id), str(item.object_id)),
    ):
        subject_label = by_id[rel.subject_id].label if rel.subject_id in by_id else rel.subject_id
        object_label = by_id[rel.object_id].label if rel.object_id in by_id else rel.object_id
        triple = (subject_label, rel.predicate, object_label)
        if triple in seen_relation_pairs:
            continue
        seen_relation_pairs.add(triple)
        qa_examples.append(
            {
                "task_type": "relation_exists",
                "question": f"Is there a {subject_label} that {rel.predicate} a {object_label}?",
                "answer": "yes",
                "metadata": {
                    "subject_label": subject_label,
                    "predicate": rel.predicate,
                    "object_label": object_label,
                },
            }
        )
        if len(seen_relation_pairs) >= max_relations:
            break

    return qa_examples


def build_grounding_examples(
    prediction: ScenePrediction,
    max_object_statements: int = 8,
    max_relation_statements: int = 8,
) -> list[dict[str, Any]]:
    """Build simple deterministic grounding/instruction examples."""
    by_id = {obj.object_id: obj for obj in prediction.objects}

    examples: list[dict[str, Any]] = []
    sorted_objects = sorted(
        prediction.objects,
        key=lambda obj: (str(obj.label).lower(), str(obj.object_id)),
    )
    for obj in sorted_objects[:max_object_statements]:
        examples.append(
            {
                "task_type": "referential_statement",
                "text": (
                    f"The object {obj.object_id} is a {obj.label} located at "
                    f"({_fmt(obj.position.x)}, {_fmt(obj.position.y)}, {_fmt(obj.position.z)})."
                ),
                "target_object_id": obj.object_id,
                "metadata": {
                    "label": obj.label,
                    "position": {
                        "x": float(obj.position.x),
                        "y": float(obj.position.y),
                        "z": float(obj.position.z),
                    },
                },
            }
        )

    relation_count = 0
    for rel in sorted(
        prediction.relations,
        key=lambda item: (str(item.predicate), str(item.subject_id), str(item.object_id)),
    ):
        if relation_count >= max_relation_statements:
            break
        subject_label = by_id[rel.subject_id].label if rel.subject_id in by_id else rel.subject_id
        object_label = by_id[rel.object_id].label if rel.object_id in by_id else rel.object_id
        examples.append(
            {
                "task_type": "instruction_grounding",
                "text": f"Identify the {subject_label} that is {rel.predicate} the {object_label}.",
                "target_object_id": rel.subject_id,
                "metadata": {
                    "subject_id": rel.subject_id,
                    "predicate": rel.predicate,
                    "object_id": rel.object_id,
                    "subject_label": subject_label,
                    "object_label": object_label,
                },
            }
        )
        relation_count += 1

    return examples


def build_language_scene_record(
    *,
    scene_id: str,
    setting: str,
    source_type: str | None,
    prediction: ScenePrediction,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one NLP-facing scene-setting record."""
    language_export = export_scene_prediction_to_language(prediction)
    qa_examples = build_qa_examples(prediction)
    grounding_examples = build_grounding_examples(prediction)

    return {
        "scene_id": scene_id,
        "setting": setting,
        "source_type": source_type,
        "structured_prediction": _serialize_prediction(prediction),
        "scene_summary_text": language_export["scene_summary_text"],
        "object_list_text": language_export["object_list_text"],
        "relation_text": language_export["relation_text"],
        "scene_paragraph_text": language_export["scene_paragraph_text"],
        "qa_examples": qa_examples,
        "grounding_examples": grounding_examples,
        "metadata": dict(metadata or {}),
    }


def _serialize_prediction(prediction: ScenePrediction) -> dict[str, Any]:
    return {
        "sample_id": prediction.sample_id,
        "generator_name": prediction.generator_name,
        "objects": [
            {
                "object_id": obj.object_id,
                "label": obj.label,
                "position": {
                    "x": float(obj.position.x),
                    "y": float(obj.position.y),
                    "z": float(obj.position.z),
                },
                "size": {
                    "x": float(obj.size.x),
                    "y": float(obj.size.y),
                    "z": float(obj.size.z),
                },
                "confidence": float(obj.confidence),
                "attributes": dict(obj.attributes),
            }
            for obj in prediction.objects
        ],
        "relations": [
            {
                "subject_id": rel.subject_id,
                "predicate": rel.predicate,
                "object_id": rel.object_id,
                "score": float(rel.score),
                "metadata": dict(rel.metadata),
            }
            for rel in prediction.relations
        ],
        "metadata": {},
    }


def _fmt(value: float) -> str:
    return f"{float(value):.3f}"

