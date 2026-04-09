"""Quantitative metrics for lightweight small-scene evaluation."""

from __future__ import annotations

import math
from typing import Any

from self_calibrating_spatiallm.artifacts import (
    ActionableScene,
    CalibrationResult,
    PointCloudMetadata,
    RepairResult,
    ScenePrediction,
)
from self_calibrating_spatiallm.evaluation.annotations import SceneAnnotation


def compute_scene_metrics(
    *,
    annotation: SceneAnnotation | None,
    point_cloud_metadata: PointCloudMetadata,
    calibration: CalibrationResult,
    prediction_before_repair: ScenePrediction,
    repair_result: RepairResult,
    violations_before: int,
    violations_after: int,
    actionable_scene: ActionableScene,
) -> dict[str, float | None]:
    """Compute lightweight research metrics for one setting on one scene."""
    metrics: dict[str, float | None] = {}

    metrics.update(
        _calibration_metrics(
            annotation=annotation,
            point_cloud_metadata=point_cloud_metadata,
            calibration=calibration,
        )
    )
    metrics.update(
        _structured_scene_metrics(
            annotation=annotation,
            prediction=prediction_before_repair,
            violations_before=violations_before,
        )
    )
    metrics.update(
        _repair_metrics(
            repair_result=repair_result,
            violations_before=violations_before,
            violations_after=violations_after,
        )
    )
    metrics.update(
        _actionable_scene_metrics(
            annotation=annotation,
            repaired_scene=repair_result.repaired_scene,
            actionable_scene=actionable_scene,
        )
    )

    return metrics


def _calibration_metrics(
    *,
    annotation: SceneAnnotation | None,
    point_cloud_metadata: PointCloudMetadata,
    calibration: CalibrationResult,
) -> dict[str, float | None]:
    up_error = math.nan
    horizontal_error = math.nan
    scale_error = math.nan
    scale_match = math.nan
    up_confidence = math.nan
    horizontal_confidence = math.nan
    reliability = math.nan
    manhattan_ambiguity = math.nan
    insufficient_plane_evidence = math.nan
    scale_plausibility = math.nan
    scale_drift = math.nan
    horizontal_error_undirected_flag = math.nan

    diagnostics = calibration.metadata.get("diagnostics", {})
    confidence = diagnostics.get("confidence", {}) if isinstance(diagnostics, dict) else {}
    scale_reasoning = calibration.metadata.get("scale_reasoning", {})
    if isinstance(diagnostics, dict) and isinstance(diagnostics.get("scale_reasoning"), dict):
        scale_reasoning = diagnostics.get("scale_reasoning")

    if isinstance(confidence, dict):
        if isinstance(confidence.get("up_axis"), (int, float)):
            up_confidence = float(confidence["up_axis"])
        if isinstance(confidence.get("horizontal_orientation"), (int, float)):
            horizontal_confidence = float(confidence["horizontal_orientation"])
        if isinstance(confidence.get("overall_reliability"), (int, float)):
            reliability = float(confidence["overall_reliability"])

    if not math.isfinite(reliability):
        if math.isfinite(up_confidence) and math.isfinite(horizontal_confidence):
            reliability = (float(up_confidence) * 0.6) + (float(horizontal_confidence) * 0.4)

    if isinstance(diagnostics, dict) and isinstance(diagnostics.get("manhattan_ambiguity"), (int, float)):
        manhattan_ambiguity = float(diagnostics["manhattan_ambiguity"])

    fallback_reason = ""
    if isinstance(diagnostics, dict):
        raw_reason = diagnostics.get("fallback_reason")
        if raw_reason:
            fallback_reason = str(raw_reason)
    if not fallback_reason and calibration.metadata.get("fallback_reason"):
        fallback_reason = str(calibration.metadata.get("fallback_reason"))
    if fallback_reason:
        reason_key = fallback_reason.lower()
        insufficient_plane_evidence = 1.0 if "insufficient_plane_evidence" in reason_key else 0.0

    if isinstance(scale_reasoning, dict):
        if isinstance(scale_reasoning.get("plausibility_score"), (int, float)):
            scale_plausibility = float(scale_reasoning["plausibility_score"])
        if isinstance(scale_reasoning.get("scale_drift"), (int, float)):
            scale_drift = float(scale_reasoning["scale_drift"])

    if annotation and annotation.expected_up_axis:
        expected = _axis_to_vector(annotation.expected_up_axis)
        up_error = _angle_deg(expected, [calibration.up_vector.x, calibration.up_vector.y, calibration.up_vector.z])

    if annotation and annotation.expected_horizontal_axis:
        expected = _axis_to_vector(annotation.expected_horizontal_axis)
        horizontal_is_undirected = not _axis_has_explicit_sign(annotation.expected_horizontal_axis)
        horizontal_error_undirected_flag = 1.0 if horizontal_is_undirected else 0.0
        horizontal_error = _angle_deg(
            expected,
            [
                calibration.horizontal_axis.x,
                calibration.horizontal_axis.y,
                calibration.horizontal_axis.z,
            ],
            undirected=horizontal_is_undirected,
        )

    if annotation and annotation.expected_scale_hint:
        inferred = point_cloud_metadata.inferred_scale_hint or ""
        scale_match = 1.0 if inferred == annotation.expected_scale_hint else 0.0
        scale_error = 0.0 if scale_match == 1.0 else 1.0

    return {
        "calibration_up_axis_error_deg": _finite_or_nan(up_error),
        "calibration_horizontal_error_deg": _finite_or_nan(horizontal_error),
        "calibration_scale_error": _finite_or_nan(scale_error),
        "calibration_scale_match": _finite_or_nan(scale_match),
        "calibration_up_axis_confidence": _finite_or_nan(up_confidence),
        "calibration_horizontal_confidence": _finite_or_nan(horizontal_confidence),
        "calibration_reliability": _finite_or_nan(reliability),
        "calibration_manhattan_ambiguity": _finite_or_nan(manhattan_ambiguity),
        "calibration_insufficient_plane_evidence_flag": _finite_or_nan(insufficient_plane_evidence),
        "calibration_horizontal_error_undirected_flag": _finite_or_nan(horizontal_error_undirected_flag),
        "calibration_scale_plausibility": _finite_or_nan(scale_plausibility),
        "calibration_scale_drift": _finite_or_nan(scale_drift),
    }


def _structured_scene_metrics(
    *,
    annotation: SceneAnnotation | None,
    prediction: ScenePrediction,
    violations_before: int,
) -> dict[str, float | None]:
    precision = math.nan
    recall = math.nan
    f1 = math.nan
    door_error = math.nan
    window_error = math.nan

    predicted_labels = {obj.label for obj in prediction.objects}

    if annotation and annotation.expected_object_categories:
        expected_labels = {label for label in annotation.expected_object_categories}
        tp = len(expected_labels & predicted_labels)
        fp = len(predicted_labels - expected_labels)
        fn = len(expected_labels - predicted_labels)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = _f1(precision, recall)

    if annotation and annotation.expected_door_count is not None:
        predicted_count = sum(1 for obj in prediction.objects if obj.label == "door")
        door_error = abs(predicted_count - annotation.expected_door_count)

    if annotation and annotation.expected_window_count is not None:
        predicted_count = sum(1 for obj in prediction.objects if obj.label == "window")
        window_error = abs(predicted_count - annotation.expected_window_count)

    return {
        "structured_category_presence_precision": _finite_or_nan(precision),
        "structured_category_presence_recall": _finite_or_nan(recall),
        "structured_category_presence_f1": _finite_or_nan(f1),
        "structured_door_count_error": _finite_or_nan(door_error),
        "structured_window_count_error": _finite_or_nan(window_error),
        "structured_violation_count_before_repair": float(violations_before),
    }


def _repair_metrics(
    *,
    repair_result: RepairResult,
    violations_before: int,
    violations_after: int,
) -> dict[str, float | None]:
    reduction = float(violations_before - violations_after)
    overcorrection = 1.0 if violations_after > violations_before else 0.0

    return {
        "repair_violations_before": float(violations_before),
        "repair_violations_after": float(violations_after),
        "repair_violation_reduction": reduction,
        "repair_fixes_applied": float(len(repair_result.fixes_applied)),
        "repair_overcorrection_flag": overcorrection,
    }


def _actionable_scene_metrics(
    *,
    annotation: SceneAnnotation | None,
    repaired_scene: ScenePrediction,
    actionable_scene: ActionableScene,
) -> dict[str, float | None]:
    relation_precision = math.nan
    relation_recall = math.nan
    relation_f1 = math.nan
    traversability_accuracy = math.nan

    if annotation and annotation.expected_relations:
        predicted = _relation_label_set(repaired_scene, actionable_scene)
        expected = {
            (rel.subject_category, rel.predicate, rel.object_category)
            for rel in annotation.expected_relations
        }
        tp = len(expected & predicted)
        fp = len(predicted - expected)
        fn = len(expected - predicted)
        relation_precision = tp / max(tp + fp, 1)
        relation_recall = tp / max(tp + fn, 1)
        relation_f1 = _f1(relation_precision, relation_recall)

    if annotation and annotation.traversability_labels:
        predicted_accessible = _accessible_category_map(repaired_scene, actionable_scene)
        correct = 0
        total = 0
        for label in annotation.traversability_labels:
            predicted = predicted_accessible.get(label.target_category, False)
            total += 1
            if predicted == label.should_be_accessible:
                correct += 1
        traversability_accuracy = correct / max(total, 1)

    return {
        "actionable_relation_precision": _finite_or_nan(relation_precision),
        "actionable_relation_recall": _finite_or_nan(relation_recall),
        "actionable_relation_f1": _finite_or_nan(relation_f1),
        "actionable_traversability_accuracy": _finite_or_nan(traversability_accuracy),
    }


def _relation_label_set(
    scene: ScenePrediction,
    actionable_scene: ActionableScene,
) -> set[tuple[str, str, str]]:
    by_id = {obj.object_id: obj.label for obj in scene.objects}
    labels: set[tuple[str, str, str]] = set()
    for relation in actionable_scene.relations:
        subject_label = by_id.get(relation.subject_id)
        object_label = by_id.get(relation.object_id)
        if subject_label is None:
            subject_label = relation.subject_id
        if object_label is None:
            object_label = relation.object_id
        labels.add((subject_label, relation.predicate, object_label))
    return labels


def _accessible_category_map(
    scene: ScenePrediction,
    actionable_scene: ActionableScene,
) -> dict[str, bool]:
    by_id = {obj.object_id: obj.label for obj in scene.objects}
    accessible: dict[str, bool] = {}
    for relation in actionable_scene.relations:
        if relation.predicate != "accessible" or relation.object_id != "agent":
            continue
        label = by_id.get(relation.subject_id)
        if label is None:
            continue
        accessible[label] = True
    return accessible


def _axis_to_vector(axis: str) -> list[float]:
    normalized = axis.strip().lower()
    sign = -1.0 if normalized.startswith("-") else 1.0
    key = normalized[1:] if normalized.startswith("-") else normalized

    if key not in {"x", "y", "z"}:
        raise ValueError(f"Unsupported axis label: {axis}")

    if key == "x":
        return [sign, 0.0, 0.0]
    if key == "y":
        return [0.0, sign, 0.0]
    return [0.0, 0.0, sign]


def _angle_deg(a: list[float], b: list[float], *, undirected: bool = False) -> float:
    vec_a = normalize_vector_from_list(a)
    vec_b = normalize_vector_from_list(b)
    dot = max(min((vec_a[0] * vec_b[0]) + (vec_a[1] * vec_b[1]) + (vec_a[2] * vec_b[2]), 1.0), -1.0)
    angle = float(math.degrees(math.acos(dot)))
    if undirected:
        return float(min(angle, abs(180.0 - angle)))
    return angle


def normalize_vector_from_list(values: list[float]) -> list[float]:
    norm = math.sqrt((values[0] ** 2) + (values[1] ** 2) + (values[2] ** 2))
    if norm <= 1e-12:
        return [0.0, 0.0, 1.0]
    return [float(values[0] / norm), float(values[1] / norm), float(values[2] / norm)]


def _f1(precision: float, recall: float) -> float:
    return (2.0 * precision * recall) / max(precision + recall, 1e-8)


def _axis_has_explicit_sign(axis: str) -> bool:
    stripped = axis.strip()
    return stripped.startswith("+") or stripped.startswith("-")


def _finite_or_nan(value: float) -> float:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None
