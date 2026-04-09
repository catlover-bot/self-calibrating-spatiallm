"""Calibration diagnostics extraction helpers."""

from __future__ import annotations

from typing import Any

from self_calibrating_spatiallm.artifacts import CalibrationResult


def extract_calibration_execution(calibration: CalibrationResult) -> dict[str, Any]:
    """Extract normalized calibration execution metadata for manifests/reports."""
    metadata = calibration.metadata if isinstance(calibration.metadata, dict) else {}
    diagnostics = metadata.get("diagnostics", {})
    if not isinstance(diagnostics, dict):
        diagnostics = {}

    execution = metadata.get("execution", {})
    if not isinstance(execution, dict):
        execution = {}

    fallback_reason = _first_non_empty_str(
        execution.get("fallback_reason"),
        diagnostics.get("fallback_reason"),
        metadata.get("fallback_reason"),
    )
    candidate_plane_count = _int_or_default(
        execution.get("candidate_plane_count"),
        _int_or_default(
            diagnostics.get("candidate_plane_count"),
            len(diagnostics.get("plane_candidates", []))
            if isinstance(diagnostics.get("plane_candidates"), list)
            else 0,
        ),
    )
    weak_scale_reasoning_active = _bool_or_default(
        execution.get("weak_scale_reasoning_active"),
        _bool_or_default(diagnostics.get("weak_scale_reasoning_active"), False),
    )
    partial_calibration_applied = _bool_or_default(
        execution.get("partial_calibration_applied"),
        _bool_or_default(execution.get("partial_calibration"), False),
    )

    backend = diagnostics.get("backend")
    plane_aware_logic_ran = _bool_or_default(
        execution.get("plane_aware_logic_ran"),
        calibration.method == "plane_aware_v1" and str(backend) != "fallback_v0",
    )
    true_v1_execution = _bool_or_default(
        execution.get("true_v1_execution"),
        calibration.method == "plane_aware_v1" and plane_aware_logic_ran,
    )
    fallback_used = _bool_or_default(execution.get("fallback_used"), bool(fallback_reason))

    confidence = diagnostics.get("confidence", {})
    if not isinstance(confidence, dict):
        confidence = {}

    partial_reasons = execution.get("partial_calibration_reasons", [])
    if not isinstance(partial_reasons, list):
        partial_reasons = []

    return {
        "method": calibration.method,
        "plane_aware_logic_ran": bool(plane_aware_logic_ran),
        "true_v1_execution": bool(true_v1_execution),
        "fallback_used": bool(fallback_used),
        "fallback_reason": fallback_reason,
        "candidate_plane_count": int(max(candidate_plane_count, 0)),
        "weak_scale_reasoning_active": bool(weak_scale_reasoning_active),
        "partial_calibration_applied": bool(partial_calibration_applied),
        "partial_calibration_reasons": partial_reasons,
        "up_axis_confidence": _float_or_none(confidence.get("up_axis")),
        "horizontal_confidence": _float_or_none(confidence.get("horizontal_orientation")),
        "overall_reliability": _float_or_none(confidence.get("overall_reliability")),
        "manhattan_ambiguity": _float_or_none(diagnostics.get("manhattan_ambiguity")),
        "scale_drift": _extract_scale_drift(metadata=metadata, diagnostics=diagnostics),
    }


def _extract_scale_drift(*, metadata: dict[str, Any], diagnostics: dict[str, Any]) -> float | None:
    scale_reasoning = metadata.get("scale_reasoning", {})
    if not isinstance(scale_reasoning, dict):
        scale_reasoning = {}
    if not scale_reasoning and isinstance(diagnostics.get("scale_reasoning"), dict):
        scale_reasoning = diagnostics["scale_reasoning"]
    return _float_or_none(scale_reasoning.get("scale_drift"))


def _first_non_empty_str(*values: Any) -> str | None:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value
    return None


def _int_or_default(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _bool_or_default(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    return bool(default)


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
