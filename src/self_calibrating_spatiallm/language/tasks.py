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
    sorted_labels = sorted(label_counts.keys())
    relation_predicates = sorted({str(rel.predicate) for rel in prediction.relations})
    hinted_relation_predicates = _extract_relation_hints(prediction)
    combined_predicates = sorted(set(relation_predicates).union(hinted_relation_predicates))

    qa_examples: list[dict[str, Any]] = []
    qa_examples.append(
        {
            "task_type": "list_object_labels",
            "question": "Which object labels appear in the scene?",
            "answer": ", ".join(sorted_labels) if sorted_labels else "none",
            "metadata": {"labels": sorted_labels},
        }
    )

    for label in sorted_labels[:max_labels]:
        qa_examples.append(
            {
                "task_type": "count_objects",
                "question": f"How many {label} objects are in the scene?",
                "answer": str(int(label_counts[label])),
                "metadata": {"label": label},
            }
        )
        qa_examples.append(
            {
                "task_type": "contains_label",
                "question": f"Does the scene contain a {label}?",
                "answer": "yes",
                "metadata": {"label": label},
            }
        )

    key_labels = ["door", "window", "table", "bed", "chair"]
    for label in key_labels:
        if label in sorted_labels:
            continue
        qa_examples.append(
            {
                "task_type": "contains_label",
                "question": f"Does the scene contain a {label}?",
                "answer": "no",
                "metadata": {"label": label, "evidence": "label_set_check"},
            }
        )
        if len(qa_examples) >= 2 * max_labels + 3:
            break

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

    qa_examples.append(
        {
            "task_type": "relation_predicates_list",
            "question": "Which relation predicates appear in the scene evidence?",
            "answer": ", ".join(combined_predicates) if combined_predicates else "none",
            "metadata": {
                "explicit_predicates": relation_predicates,
                "hinted_predicates": hinted_relation_predicates,
            },
        }
    )
    for predicate in combined_predicates[:max_relations]:
        qa_examples.append(
            {
                "task_type": "relation_predicate_evidence",
                "question": f"Is there evidence for relation predicate {predicate}?",
                "answer": "yes",
                "metadata": {
                    "predicate": predicate,
                    "evidence_level": "explicit" if predicate in relation_predicates else "hinted",
                },
            }
        )

    for predicate in ["supported-by", "attached-to"]:
        if predicate in combined_predicates:
            continue
        qa_examples.append(
            {
                "task_type": "relation_predicate_evidence",
                "question": f"Is there evidence for relation predicate {predicate}?",
                "answer": "no",
                "metadata": {"predicate": predicate, "evidence_level": "none"},
            }
        )

    return qa_examples


def build_grounding_examples(
    prediction: ScenePrediction,
    max_object_statements: int = 8,
    max_relation_statements: int = 8,
) -> list[dict[str, Any]]:
    """Build simple deterministic grounding/instruction examples."""
    by_id = {obj.object_id: obj for obj in prediction.objects}
    hinted_relation_predicates = _extract_relation_hints(prediction)
    prediction_is_reconstructed = bool(
        isinstance(prediction.metadata, dict)
        and prediction.metadata.get("reconstructed_from_prediction_summary", False)
    )

    examples: list[dict[str, Any]] = []
    sorted_objects = sorted(
        prediction.objects,
        key=lambda obj: (str(obj.label).lower(), str(obj.object_id)),
    )
    for obj in sorted_objects[:max_object_statements]:
        obj_attributes = getattr(obj, "attributes", {})
        has_geometry = not (
            prediction_is_reconstructed
            or (
                isinstance(obj_attributes, dict)
                and (
                    obj_attributes.get("reconstructed_from_prediction_summary", False)
                    or obj_attributes.get("geometry_unavailable", False)
                )
            )
        )

        if has_geometry:
            examples.append(
                {
                    "task_type": "referential_statement_geometric",
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
                        "grounding_strength": "strong",
                        "geometry_available": True,
                    },
                }
            )
            continue

        examples.append(
            {
                "task_type": "referential_statement_presence",
                "text": f"An object with label {obj.label} is present in the exported structure.",
                "target_object_id": obj.object_id,
                "metadata": {
                    "label": obj.label,
                    "grounding_strength": "weak",
                    "geometry_available": False,
                    "reason": "summary_reconstructed_or_geometry_unavailable",
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
                    "grounding_strength": "strong",
                },
            }
        )
        relation_count += 1

    if relation_count == 0:
        for predicate in hinted_relation_predicates[:max_relation_statements]:
            examples.append(
                {
                    "task_type": "relation_evidence_statement",
                    "text": (
                        f"The exported scene includes hinted relation evidence for predicate {predicate}."
                    ),
                    "target_object_id": None,
                    "metadata": {
                        "predicate": predicate,
                        "evidence_level": "hinted",
                        "grounding_strength": "weak",
                    },
                }
            )

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
        "relation_statements": language_export["relation_statements"],
        "relation_evidence_level": language_export["relation_evidence_level"],
        "relation_hint_count": language_export["relation_hint_count"],
        "relation_hint_predicates": language_export["relation_hint_predicates"],
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
        "metadata": dict(prediction.metadata),
    }


def _extract_relation_hints(prediction: ScenePrediction) -> list[str]:
    metadata = prediction.metadata if isinstance(prediction.metadata, dict) else {}
    raw = metadata.get("relation_predicates", [])
    if not isinstance(raw, list):
        return []
    return sorted({str(item).strip() for item in raw if str(item).strip()})


def _fmt(value: float) -> str:
    return f"{float(value):.3f}"
