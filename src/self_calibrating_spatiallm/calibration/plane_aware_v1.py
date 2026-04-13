"""Plane-aware calibration baseline (v1) with explicit diagnostics and fallback paths."""

from __future__ import annotations

from typing import Any

try:
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover - exercised in minimal envs
    np = None

from self_calibrating_spatiallm.artifacts import CalibratedPointCloud, CalibrationResult, Point3D, PointCloudSample
from self_calibrating_spatiallm.calibration.base import Calibrator
from self_calibrating_spatiallm.calibration.geometric_v0 import GeometricCalibratorV0
from self_calibrating_spatiallm.geometry import array_to_points, points_to_array


class PlaneAwareCalibratorV1(Calibrator):
    """Plane-aware geometric calibrator with inspectable diagnostics."""

    name = "plane_aware_v1"

    def __init__(
        self,
        normalize_scene: bool = True,
        min_reliability_for_normalization: float = 0.45,
        min_up_confidence: float = 0.20,
        min_horizontal_confidence: float = 0.15,
        min_horizontal_confidence_for_acceptance: float = 0.35,
        max_manhattan_ambiguity_for_acceptance: float = 0.85,
        strong_horizontal_confidence_for_ambiguous_acceptance: float = 0.70,
        max_manhattan_ambiguity_for_strong_acceptance: float = 0.95,
        min_up_score: float = 0.45,
        min_up_confidence_for_override: float = 0.80,
        max_up_axis_disagreement_deg: float = 45.0,
        room_height_prior_m: tuple[float, float] = (2.2, 3.8),
        room_height_tolerance_m: float = 4.0,
    ) -> None:
        self.normalize_scene = normalize_scene
        self.min_reliability_for_normalization = min_reliability_for_normalization
        self.min_up_confidence = min_up_confidence
        self.min_horizontal_confidence = min_horizontal_confidence
        self.min_horizontal_confidence_for_acceptance = min_horizontal_confidence_for_acceptance
        self.max_manhattan_ambiguity_for_acceptance = max_manhattan_ambiguity_for_acceptance
        self.strong_horizontal_confidence_for_ambiguous_acceptance = (
            strong_horizontal_confidence_for_ambiguous_acceptance
        )
        self.max_manhattan_ambiguity_for_strong_acceptance = max_manhattan_ambiguity_for_strong_acceptance
        self.min_up_score = min_up_score
        self.min_up_confidence_for_override = min_up_confidence_for_override
        self.max_up_axis_disagreement_deg = max_up_axis_disagreement_deg
        self.room_height_prior_m = room_height_prior_m
        self.room_height_tolerance_m = room_height_tolerance_m
        self._fallback_v0 = GeometricCalibratorV0(normalize_scene=normalize_scene)

    def calibrate(self, sample: PointCloudSample) -> CalibratedPointCloud:
        if np is None:
            return self._fallback(sample, reason="numpy_unavailable", analysis={})

        rows = points_to_array(sample.points)
        xyz = np.asarray(rows, dtype=float)
        if xyz.shape[0] < 32:
            return self._fallback(sample, reason="insufficient_plane_evidence:few_points", analysis={})

        center = np.mean(xyz, axis=0)
        centered = xyz - center

        analysis = _analyze_plane_structure(
            centered,
            room_height_prior=self.room_height_prior_m,
            room_height_tolerance=self.room_height_tolerance_m,
        )
        confidence = analysis["confidence"]
        up_confidence = float(confidence.get("up_axis", 0.0))
        horizontal_confidence = float(confidence.get("horizontal_orientation", 0.0))
        up_axis_selection = analysis.get("up_axis_selection", {})
        selected_up_score = float(up_axis_selection.get("selected_score", 0.0))

        if up_confidence < self.min_up_confidence or selected_up_score < self.min_up_score:
            return self._fallback(
                sample,
                reason="insufficient_plane_evidence:low_up_confidence_or_score",
                analysis=analysis,
            )

        up_vector = np.asarray(analysis["up_vector"], dtype=float)
        horizontal_axis = np.asarray(analysis["horizontal_axis"], dtype=float)
        up_axis_source = "analysis"
        horizontal_axis_source = "analysis"
        fallback_reason: str | None = None
        partial_calibration_applied = False
        partial_reasons: list[str] = []
        guardrail_details: dict[str, Any] = {
            "applied": False,
            "reason": None,
            "up_axis_source": up_axis_source,
            "horizontal_axis_source": horizontal_axis_source,
        }
        horizontal_guardrail_details: dict[str, Any] = {
            "attempted": True,
            "triggered": False,
            "accepted": True,
            "downgraded_to_partial_calibration": False,
            "reasons": [],
            "horizontal_confidence_raw": horizontal_confidence,
            "manhattan_ambiguity": float(analysis.get("manhattan_ambiguity", 0.0)),
            "min_horizontal_confidence_for_acceptance": self.min_horizontal_confidence_for_acceptance,
            "max_manhattan_ambiguity_for_acceptance": self.max_manhattan_ambiguity_for_acceptance,
            "strong_horizontal_confidence_for_ambiguous_acceptance": (
                self.strong_horizontal_confidence_for_ambiguous_acceptance
            ),
            "max_manhattan_ambiguity_for_strong_acceptance": (
                self.max_manhattan_ambiguity_for_strong_acceptance
            ),
            "primary_wall_score": 0.0,
            "secondary_wall_score": 0.0,
            "unique_orientation_count": 0,
        }

        # v1.1 guardrail: avoid accepting a high-disagreement up-axis when evidence is not strong.
        v0_reference = self._fallback_v0.calibrate(sample)
        v0_up_vector = _point_to_array(v0_reference.calibration.up_vector)
        v0_horizontal_axis = _point_to_array(v0_reference.calibration.horizontal_axis)
        up_axis_disagreement_deg = _angle_between_deg(up_vector, v0_up_vector)
        selected_span_plausibility = float(up_axis_selection.get("selected_span_plausibility", 0.0))
        selected_margin = float(up_axis_selection.get("score_margin", 0.0))
        override_guardrail = (
            up_axis_disagreement_deg > self.max_up_axis_disagreement_deg
            and (
                up_confidence < self.min_up_confidence_for_override
                or selected_span_plausibility < 0.65
                or selected_margin < 0.10
            )
        )
        if override_guardrail:
            up_vector = _normalize(v0_up_vector)
            projected_horizontal = _project_to_plane(horizontal_axis, up_vector)
            if np.linalg.norm(projected_horizontal) <= 1e-8:
                horizontal_axis = _normalize(v0_horizontal_axis)
                horizontal_axis_source = "v0_guardrail"
            else:
                horizontal_axis = _normalize(projected_horizontal)
                horizontal_axis_source = "analysis_projected_after_guardrail"
            up_axis_source = "v0_guardrail"
            partial_calibration_applied = True
            partial_reason = "guardrail:up_axis_disagreement_with_v0"
            partial_reasons.append(partial_reason)
            guardrail_details = {
                "applied": True,
                "reason": partial_reason,
                "up_axis_disagreement_deg": up_axis_disagreement_deg,
                "up_axis_source": up_axis_source,
                "horizontal_axis_source": horizontal_axis_source,
                "selected_up_score": selected_up_score,
                "selected_span_plausibility": selected_span_plausibility,
                "selected_score_margin": selected_margin,
            }
            # Cap confidence to avoid overconfident reporting after guarded override.
            confidence["up_axis"] = float(min(float(confidence.get("up_axis", 0.0)), 0.70))
            up_confidence = float(confidence["up_axis"])

        horizontal_ambiguity = float(analysis.get("manhattan_ambiguity", 0.0))
        horizontal_evidence = analysis.get("horizontal_evidence", {})
        primary_wall_score = 0.0
        secondary_wall_score = 0.0
        unique_orientation_count = 0
        if isinstance(horizontal_evidence, dict):
            primary_wall_score = float(horizontal_evidence.get("primary_final_score", 0.0) or 0.0)
            secondary_wall_score = float(horizontal_evidence.get("secondary_final_score", 0.0) or 0.0)
            unique_orientation_count = int(horizontal_evidence.get("unique_orientation_count", 0) or 0)
        horizontal_decision = _decide_horizontal_strategy(
            horizontal_confidence=horizontal_confidence,
            horizontal_ambiguity=horizontal_ambiguity,
            min_horizontal_confidence=self.min_horizontal_confidence,
            min_horizontal_confidence_for_acceptance=self.min_horizontal_confidence_for_acceptance,
            max_manhattan_ambiguity_for_acceptance=self.max_manhattan_ambiguity_for_acceptance,
            strong_horizontal_confidence_for_ambiguous_acceptance=(
                self.strong_horizontal_confidence_for_ambiguous_acceptance
            ),
            max_manhattan_ambiguity_for_strong_acceptance=self.max_manhattan_ambiguity_for_strong_acceptance,
            primary_wall_score=primary_wall_score,
            secondary_wall_score=secondary_wall_score,
            unique_orientation_count=unique_orientation_count,
        )
        horizontal_guardrail_details.update(
            {
                "primary_wall_score": float(primary_wall_score),
                "secondary_wall_score": float(secondary_wall_score),
                "unique_orientation_count": int(unique_orientation_count),
                "dynamic_min_horizontal_confidence_for_acceptance": float(
                    horizontal_decision.get("dynamic_min_horizontal_confidence_for_acceptance", 0.0)
                ),
                "evidence_strength": float(horizontal_decision.get("evidence_strength", 0.0)),
                "structural_strength": float(horizontal_decision.get("structural_strength", 0.0)),
                "structural_separation": float(horizontal_decision.get("structural_separation", 0.0)),
                "effective_manhattan_ambiguity": float(
                    horizontal_decision.get("effective_manhattan_ambiguity", horizontal_ambiguity)
                ),
                "commitment_quality": float(horizontal_decision.get("commitment_quality", 0.0)),
                "decision_mode": horizontal_decision.get("mode", "partial"),
                "accepted_by": horizontal_decision.get("accepted_by"),
            }
        )

        decision_mode = str(horizontal_decision.get("mode", "partial"))
        decision_reasons = [str(reason) for reason in horizontal_decision.get("reasons", [])]

        if decision_mode == "fallback":
            fallback_reason = "insufficient_plane_evidence:weak_horizontal_axis"
            horizontal_axis = _project_to_plane(v0_horizontal_axis, up_vector)
            if np.linalg.norm(horizontal_axis) <= 1e-8:
                horizontal_axis = _fallback_horizontal(up_vector)
            horizontal_axis = _normalize(horizontal_axis)
            horizontal_axis_source = "v0_horizontal_fallback"
            partial_calibration_applied = True
            partial_reasons.append(fallback_reason)
            confidence["horizontal_orientation"] = float(min(max(horizontal_confidence, 0.12), 1.0))
            horizontal_guardrail_details.update(
                {
                    "triggered": True,
                    "accepted": False,
                    "downgraded_to_partial_calibration": True,
                    "reasons": decision_reasons,
                    "horizontal_axis_source": horizontal_axis_source,
                    "action": "fallback_to_v0_horizontal_axis",
                }
            )
        elif decision_mode == "partial":
            partial_axis_strategy = str(horizontal_decision.get("partial_axis_strategy", "up_only"))
            if partial_axis_strategy == "analysis_projected":
                horizontal_axis = _project_to_plane(horizontal_axis, up_vector)
                if np.linalg.norm(horizontal_axis) <= 1e-8:
                    horizontal_axis = _project_to_plane(v0_horizontal_axis, up_vector)
                if np.linalg.norm(horizontal_axis) <= 1e-8:
                    horizontal_axis = _fallback_horizontal(up_vector)
                horizontal_axis_source = "analysis_projected_partial_guardrail"
            else:
                horizontal_axis = _project_to_plane(_fallback_horizontal(up_vector), up_vector)
                if np.linalg.norm(horizontal_axis) <= 1e-8:
                    horizontal_axis = _project_to_plane(v0_horizontal_axis, up_vector)
                if np.linalg.norm(horizontal_axis) <= 1e-8:
                    horizontal_axis = _fallback_horizontal(up_vector)
                horizontal_axis_source = "up_only_horizontal_guardrail"
            horizontal_axis = _normalize(horizontal_axis)
            partial_calibration_applied = True
            partial_reasons.append("partial_calibration:horizontal_evidence_ambiguous")
            partial_confidence_cap = float(horizontal_decision.get("partial_confidence_cap", 0.55))
            confidence["horizontal_orientation"] = float(
                min(float(confidence.get("horizontal_orientation", 0.0)), partial_confidence_cap)
            )
            horizontal_confidence = float(confidence["horizontal_orientation"])
            horizontal_guardrail_details.update(
                {
                    "triggered": True,
                    "accepted": False,
                    "downgraded_to_partial_calibration": True,
                    "reasons": decision_reasons,
                    "horizontal_axis_source": horizontal_axis_source,
                    "action": "analysis_projected_horizontal_axis"
                    if partial_axis_strategy == "analysis_projected"
                    else "up_only_horizontal_axis",
                    "partial_axis_strategy": partial_axis_strategy,
                    "partial_confidence_cap": partial_confidence_cap,
                }
            )
        else:
            horizontal_guardrail_details.update(
                {
                    "reasons": decision_reasons,
                    "horizontal_axis_source": horizontal_axis_source,
                    "action": "analysis_axis_accepted",
                }
            )

        horizontal_axis = _project_to_plane(horizontal_axis, up_vector)
        if np.linalg.norm(horizontal_axis) <= 1e-8:
            horizontal_axis = _fallback_horizontal(up_vector)
        horizontal_axis = _normalize(horizontal_axis)
        horizontal_before_orientation = horizontal_axis.copy()
        oriented_horizontal = _orient_axis_positive(horizontal_axis)
        horizontal_sign_correction_applied = bool(np.dot(horizontal_before_orientation, oriented_horizontal) < 0.0)
        horizontal_axis = oriented_horizontal

        initial_reliability_mode = (
            "degraded_fallback"
            if fallback_reason
            else ("safe_partial_calibration" if partial_calibration_applied else "full_calibration")
        )
        overall_reliability, reliability_breakdown = _compute_reliability_v1_3(
            up_confidence=float(confidence.get("up_axis", 0.0)),
            horizontal_confidence=float(confidence.get("horizontal_orientation", 0.0)),
            manhattan_ambiguity=horizontal_ambiguity,
            reliability_mode=initial_reliability_mode,
            up_guardrail_applied=bool(guardrail_details.get("applied")),
            horizontal_decision_mode=decision_mode,
            horizontal_evidence_strength=float(horizontal_decision.get("evidence_strength", 0.0)),
            effective_manhattan_ambiguity=float(
                horizontal_decision.get("effective_manhattan_ambiguity", horizontal_ambiguity)
            ),
            partial_axis_strategy=str(horizontal_decision.get("partial_axis_strategy", "")),
            accepted_by=str(horizontal_decision.get("accepted_by") or ""),
            commitment_quality=float(horizontal_decision.get("commitment_quality", 0.0)),
        )
        confidence["overall_reliability"] = float(overall_reliability)

        rotation_matrix = _build_rotation_matrix(horizontal_axis=horizontal_axis, up_vector=up_vector)
        overall_reliability = float(confidence.get("overall_reliability", 0.0))
        should_normalize = bool(self.normalize_scene and overall_reliability >= self.min_reliability_for_normalization)

        if should_normalize:
            transformed = (rotation_matrix @ centered.T).T
            origin_offset = Point3D(x=float(-center[0]), y=float(-center[1]), z=float(-center[2]))
        else:
            transformed = xyz.copy()
            origin_offset = Point3D(x=0.0, y=0.0, z=0.0)
            if self.normalize_scene and fallback_reason is None:
                fallback_reason = "insufficient_plane_evidence:normalization_skipped_low_reliability"
                partial_calibration_applied = True
                partial_reasons.append(fallback_reason)

        final_reliability_mode = (
            "degraded_fallback"
            if fallback_reason
            else ("safe_partial_calibration" if partial_calibration_applied else "full_calibration")
        )
        if final_reliability_mode != initial_reliability_mode:
            overall_reliability, reliability_breakdown = _compute_reliability_v1_3(
                up_confidence=float(confidence.get("up_axis", 0.0)),
                horizontal_confidence=float(confidence.get("horizontal_orientation", 0.0)),
                manhattan_ambiguity=horizontal_ambiguity,
                reliability_mode=final_reliability_mode,
                up_guardrail_applied=bool(guardrail_details.get("applied")),
                horizontal_decision_mode=decision_mode,
                horizontal_evidence_strength=float(horizontal_decision.get("evidence_strength", 0.0)),
                effective_manhattan_ambiguity=float(
                    horizontal_decision.get("effective_manhattan_ambiguity", horizontal_ambiguity)
                ),
                partial_axis_strategy=str(horizontal_decision.get("partial_axis_strategy", "")),
                accepted_by=str(horizontal_decision.get("accepted_by") or ""),
                commitment_quality=float(horizontal_decision.get("commitment_quality", 0.0)),
            )
            confidence["overall_reliability"] = float(overall_reliability)
        reliability_breakdown["normalization_requested"] = bool(self.normalize_scene)
        reliability_breakdown["normalization_applied"] = bool(should_normalize)
        reliability_breakdown["normalization_threshold"] = float(self.min_reliability_for_normalization)
        reliability_breakdown["normalization_blocked_reason"] = (
            None if should_normalize else (fallback_reason or "normalization_disabled")
        )

        transformed_rows = transformed.tolist()
        transformed_stats = _compute_transformed_stats_from_rows(transformed_rows)
        scale_reasoning = _assess_scale_plausibility(
            transformed_stats=transformed_stats,
            selected_plane_roles=analysis.get("selected_plane_roles", {}),
        )
        candidate_plane_count = int(len(analysis.get("plane_candidates", [])))
        fallback_used = bool(fallback_reason and str(fallback_reason).startswith("insufficient_plane_evidence"))
        execution = {
            "plane_aware_logic_ran": True,
            "true_v1_execution": True,
            "fallback_used": fallback_used,
            "fallback_reason": fallback_reason,
            "candidate_plane_count": candidate_plane_count,
            "weak_scale_reasoning_active": True,
            "partial_calibration_applied": partial_calibration_applied,
            "partial_calibration_reasons": partial_reasons,
            "reliability_mode": final_reliability_mode,
            "horizontal_decision_mode": decision_mode,
        }

        diagnostics: dict[str, Any] = {
            "backend": "numpy",
            "plane_candidates": analysis.get("plane_candidates", []),
            "selected_plane_roles": analysis.get("selected_plane_roles", {}),
            "wall_candidate_ranking": analysis.get("wall_candidate_ranking", []),
            "horizontal_evidence": analysis.get("horizontal_evidence", {}),
            "axis_candidates": analysis.get("axis_candidates", []),
            "intermediate_axis_candidates": analysis.get("axis_candidates", []),
            "up_axis_selection": analysis.get("up_axis_selection", {}),
            "up_axis_source": up_axis_source,
            "horizontal_axis_source": horizontal_axis_source,
            "up_axis_guardrail": guardrail_details,
            "horizontal_guardrail": horizontal_guardrail_details,
            "horizontal_sign_correction_applied": horizontal_sign_correction_applied,
            "candidate_plane_count": candidate_plane_count,
            "manhattan_ambiguity": float(analysis.get("manhattan_ambiguity", 0.0)),
            "confidence": {
                "up_axis": float(confidence.get("up_axis", 0.0)),
                "horizontal_orientation": float(confidence.get("horizontal_orientation", 0.0)),
                "overall_reliability": float(confidence.get("overall_reliability", 0.0)),
            },
            "reliability_breakdown": reliability_breakdown,
            "scale_reasoning": scale_reasoning,
            "weak_scale_reasoning_active": True,
            "normalization_decision": {
                "requested": bool(self.normalize_scene),
                "applied": bool(should_normalize),
                "overall_reliability": overall_reliability,
                "threshold": self.min_reliability_for_normalization,
                "blocked_reason": None
                if should_normalize
                else (fallback_reason or "normalization_disabled"),
            },
        }
        if fallback_reason:
            diagnostics["fallback_reason"] = fallback_reason

        calibration = CalibrationResult(
            sample_id=sample.sample_id,
            method=self.name,
            up_vector=Point3D(x=float(up_vector[0]), y=float(up_vector[1]), z=float(up_vector[2])),
            horizontal_axis=Point3D(
                x=float(rotation_matrix[0][0]),
                y=float(rotation_matrix[0][1]),
                z=float(rotation_matrix[0][2]),
            ),
            origin_offset=origin_offset,
            metadata={
                "normalization_applied": should_normalize,
                "rotation_matrix": rotation_matrix.tolist(),
                "diagnostics": diagnostics,
                "transformed_point_cloud": transformed_stats,
                "scale_reasoning": scale_reasoning,
                "fallback_reason": fallback_reason,
                "partial_calibration_reasons": partial_reasons,
                "algorithm_version": "v1.4_plane_interpretation_scored",
                "execution": execution,
            },
        )

        calibrated_metadata = {
            "frame": "canonical" if should_normalize else sample.sensor_frame,
            "original_sensor_frame": sample.sensor_frame,
            "num_points": int(transformed.shape[0]),
            "transformed_point_cloud": transformed_stats,
            "room_bounds": sample.metadata.get("room_bounds"),
            "expected_unit": sample.metadata.get("expected_unit"),
            "inferred_scale_hint": sample.metadata.get("inferred_scale_hint"),
            "scale_reasoning": scale_reasoning,
            "fallback_reason": fallback_reason,
            "partial_calibration_reasons": partial_reasons,
            "algorithm_version": "v1.4_plane_interpretation_scored",
            "execution": execution,
        }

        return CalibratedPointCloud(
            sample_id=sample.sample_id,
            points=array_to_points(transformed_rows),
            calibration=calibration,
            metadata=calibrated_metadata,
        )

    def _fallback(
        self,
        sample: PointCloudSample,
        reason: str,
        analysis: dict[str, Any],
    ) -> CalibratedPointCloud:
        v0_result = self._fallback_v0.calibrate(sample)
        prior_diagnostics = v0_result.calibration.metadata.get("diagnostics", {})
        confidence = prior_diagnostics.get("confidence", {}) if isinstance(prior_diagnostics, dict) else {}
        fallback_diagnostics = dict(prior_diagnostics) if isinstance(prior_diagnostics, dict) else {}
        fallback_diagnostics.update(
            {
                "backend": "fallback_v0",
                "fallback_reason": reason,
                "plane_candidates": analysis.get("plane_candidates", []),
                "selected_plane_roles": analysis.get("selected_plane_roles", {}),
                "axis_candidates": analysis.get("axis_candidates", []),
                "intermediate_axis_candidates": analysis.get("axis_candidates", []),
                "wall_candidate_ranking": analysis.get("wall_candidate_ranking", []),
                "horizontal_evidence": analysis.get("horizontal_evidence", {}),
                "candidate_plane_count": int(len(analysis.get("plane_candidates", []))),
                "confidence": {
                    "up_axis": float(confidence.get("up_axis", 0.0)),
                    "horizontal_orientation": float(confidence.get("horizontal_orientation", 0.0)),
                    "overall_reliability": float(
                        max(
                            0.0,
                            min(
                                (float(confidence.get("up_axis", 0.0)) * 0.6)
                                + (float(confidence.get("horizontal_orientation", 0.0)) * 0.4),
                                1.0,
                            ),
                        )
                    ),
                },
                "reliability_breakdown": {
                    "reliability_mode": "degraded_fallback",
                    "horizontal_decision_mode": "fallback",
                    "reasons": ["fallback_to_v0"],
                },
                "scale_reasoning": {
                    "estimated_room_height": None,
                    "estimated_horizontal_extent": None,
                    "plausibility_score": None,
                    "suggested_scale_factor_to_meters": 1.0,
                    "scale_drift": 0.0,
                    "applied_scale_adjustment": False,
                    "reason": "fallback_to_v0",
                },
                "weak_scale_reasoning_active": False,
            }
        )

        execution = {
            "plane_aware_logic_ran": False,
            "true_v1_execution": False,
            "fallback_used": True,
            "fallback_reason": reason,
            "candidate_plane_count": int(len(analysis.get("plane_candidates", []))),
            "weak_scale_reasoning_active": False,
            "partial_calibration_applied": False,
            "partial_calibration_reasons": [],
            "reliability_mode": "degraded_fallback",
            "horizontal_decision_mode": "fallback",
        }

        updated_metadata = dict(v0_result.calibration.metadata)
        updated_metadata["diagnostics"] = fallback_diagnostics
        updated_metadata["fallback_reason"] = reason
        updated_metadata["fallback_to"] = "geometric_v0"
        updated_metadata["algorithm_version"] = "v1.4_plane_interpretation_scored"
        updated_metadata["execution"] = execution

        calibration = CalibrationResult(
            sample_id=v0_result.calibration.sample_id,
            method=self.name,
            up_vector=v0_result.calibration.up_vector,
            horizontal_axis=v0_result.calibration.horizontal_axis,
            origin_offset=v0_result.calibration.origin_offset,
            metadata=updated_metadata,
        )

        calibrated_metadata = dict(v0_result.metadata)
        calibrated_metadata["fallback_reason"] = reason
        calibrated_metadata["calibration_fallback_to"] = "geometric_v0"
        calibrated_metadata["algorithm_version"] = "v1.4_plane_interpretation_scored"
        calibrated_metadata["execution"] = execution

        return CalibratedPointCloud(
            sample_id=v0_result.sample_id,
            points=v0_result.points,
            calibration=calibration,
            metadata=calibrated_metadata,
        )


def _analyze_plane_structure(
    centered: np.ndarray,
    *,
    room_height_prior: tuple[float, float],
    room_height_tolerance: float,
) -> dict[str, Any]:
    eigenvalues, eigenvectors = _pca(centered)
    max_eigen = float(max(float(value) for value in eigenvalues.tolist()))
    if max_eigen <= 1e-12:
        max_eigen = 1.0

    axis_candidates: list[dict[str, Any]] = []
    for idx in range(3):
        axis = _normalize(eigenvectors[:, idx])
        summary = _evaluate_axis_as_up(centered, axis)
        summary["candidate_name"] = f"pca_axis_{idx}"
        summary["variance"] = float(eigenvalues[idx])
        summary["variance_ratio"] = float(eigenvalues[idx] / max_eigen)
        variance_weight = float(max(0.0, min(1.0, 1.0 - summary["variance_ratio"])))
        summary["base_up_score"] = float(
            (summary["plane_balance"] * 0.45) + (summary["plane_support"] * 0.40) + (variance_weight * 0.15)
        )
        axis_candidates.append(summary)

    up_selection = _score_up_axis_candidates(
        axis_candidates,
        room_height_prior=room_height_prior,
        room_height_tolerance=room_height_tolerance,
    )
    best_up = up_selection["selected_candidate"]
    up_axis = np.asarray(best_up["axis"], dtype=float)
    sign_flip_from_support = False
    if float(best_up["low_support"]) < float(best_up["high_support"]):
        up_axis = -1.0 * up_axis
        sign_flip_from_support = True
    up_before_convention = up_axis.copy()
    up_axis = _orient_axis_positive(up_axis)
    sign_flip_from_convention = bool(np.dot(up_before_convention, up_axis) < 0.0)
    oriented = _evaluate_axis_as_up(centered, up_axis)

    horizontal = _estimate_horizontal_from_walls(centered, up_axis)
    up_confidence = float(max(0.0, min(1.0, up_selection["selected_confidence"])))
    horizontal_confidence = float(max(0.0, min(1.0, horizontal["confidence"])))
    overall = float(max(0.0, min((up_confidence * 0.6) + (horizontal_confidence * 0.4), 1.0)))

    plane_candidates = _build_plane_candidates(oriented=oriented, horizontal=horizontal)
    selected_roles = _build_selected_plane_roles(oriented=oriented, horizontal=horizontal)

    return {
        "up_vector": up_axis.tolist(),
        "horizontal_axis": horizontal["horizontal_axis"].tolist(),
        "plane_candidates": plane_candidates,
        "selected_plane_roles": selected_roles,
        "wall_candidate_ranking": horizontal.get("wall_candidate_ranking", []),
        "horizontal_evidence": horizontal.get("horizontal_evidence", {}),
        "axis_candidates": axis_candidates,
        "up_axis_selection": {
            "selected_candidate_name": str(best_up.get("candidate_name")),
            "selected_score": float(up_selection.get("selected_score", 0.0)),
            "selected_base_up_score": float(best_up.get("base_up_score", 0.0)),
            "selected_span": float(best_up.get("span", 0.0)),
            "selected_span_plausibility": float(best_up.get("span_plausibility", 0.0)),
            "selected_compactness": float(best_up.get("compactness", 0.0)),
            "score_margin": float(up_selection.get("score_margin", 0.0)),
            "runner_up_candidate_name": up_selection.get("runner_up_candidate_name"),
            "runner_up_score": float(up_selection.get("runner_up_score", 0.0)),
            "sign_flip_from_support_balance": sign_flip_from_support,
            "sign_flip_from_axis_convention": sign_flip_from_convention,
            "sign_correction_applied": bool(sign_flip_from_support or sign_flip_from_convention),
            "selection_strategy": "v1.4_plane_role_scored_confidence_guarded",
            "room_height_prior": [float(room_height_prior[0]), float(room_height_prior[1])],
            "room_height_tolerance": float(room_height_tolerance),
        },
        "manhattan_ambiguity": float(horizontal["ambiguity"]),
        "confidence": {
            "up_axis": up_confidence,
            "horizontal_orientation": horizontal_confidence,
            "overall_reliability": overall,
        },
    }


def _score_up_axis_candidates(
    axis_candidates: list[dict[str, Any]],
    *,
    room_height_prior: tuple[float, float],
    room_height_tolerance: float,
) -> dict[str, Any]:
    if not axis_candidates:
        return {
            "selected_candidate": {
                "candidate_name": "none",
                "axis": [0.0, 0.0, 1.0],
                "span": 0.0,
                "base_up_score": 0.0,
            },
            "selected_score": 0.0,
            "runner_up_candidate_name": None,
            "runner_up_score": 0.0,
            "score_margin": 0.0,
            "selected_confidence": 0.0,
        }

    spans = [float(candidate.get("span", 0.0)) for candidate in axis_candidates]
    max_span = float(max(max(spans), 1e-8))
    lower, upper = room_height_prior

    for candidate in axis_candidates:
        span = float(candidate.get("span", 0.0))
        plane_support = float(candidate.get("plane_support", 0.0))
        plane_balance = float(candidate.get("plane_balance", 0.0))
        base_up_score = float(candidate.get("base_up_score", 0.0))

        compactness = float(max(0.0, min(1.0 - (span / max_span), 1.0)))
        span_plausibility = float(
            _band_score(
                span,
                lower=float(lower),
                upper=float(upper),
                tolerance=float(max(room_height_tolerance, 1e-6)),
            )
        )

        candidate["compactness"] = compactness
        candidate["span_plausibility"] = span_plausibility
        candidate["up_score_components"] = {
            "plane_support": plane_support,
            "plane_balance": plane_balance,
            "base_up_score": base_up_score,
            "compactness": compactness,
            "span_plausibility": span_plausibility,
        }
        candidate["up_score"] = float(
            (plane_support * 0.22)
            + (plane_balance * 0.13)
            + (base_up_score * 0.10)
            + (compactness * 0.30)
            + (span_plausibility * 0.25)
        )

    ranked = sorted(axis_candidates, key=lambda item: float(item.get("up_score", 0.0)), reverse=True)
    selected = ranked[0]
    runner_up = ranked[1] if len(ranked) > 1 else None

    selected_score = float(selected.get("up_score", 0.0))
    runner_up_score = float(runner_up.get("up_score", 0.0)) if runner_up else 0.0
    margin = float(max(selected_score - runner_up_score, 0.0))
    margin_score = float(max(0.0, min(margin / 0.20, 1.0)))
    selected_confidence = float(max(0.0, min((selected_score * 0.75) + (margin_score * 0.25), 1.0)))

    return {
        "selected_candidate": selected,
        "selected_score": selected_score,
        "runner_up_candidate_name": runner_up.get("candidate_name") if isinstance(runner_up, dict) else None,
        "runner_up_score": runner_up_score,
        "score_margin": margin,
        "selected_confidence": selected_confidence,
    }


def _pca(centered: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    covariance = np.cov(centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)
    return eigenvalues[order], eigenvectors[:, order]


def _evaluate_axis_as_up(centered: np.ndarray, axis: np.ndarray) -> dict[str, Any]:
    axis_vec = _normalize(axis)
    projected = centered @ axis_vec

    low_offset = float(np.quantile(projected, 0.05))
    high_offset = float(np.quantile(projected, 0.95))
    span = float(max(high_offset - low_offset, 1e-8))
    tolerance = float(max(span * 0.02, 0.04))

    low_support = float(np.mean(np.abs(projected - low_offset) <= tolerance))
    high_support = float(np.mean(np.abs(projected - high_offset) <= tolerance))
    plane_support = float(max(0.0, min((low_support + high_support) / 0.35, 1.0)))
    plane_balance = float(
        max(
            0.0,
            min(1.0 - (abs(low_support - high_support) / max(low_support + high_support, 1e-8)), 1.0),
        )
    )
    up_confidence = float(max(0.0, min((plane_support * 0.6) + (plane_balance * 0.4), 1.0)))

    return {
        "axis": axis_vec.tolist(),
        "low_offset": low_offset,
        "high_offset": high_offset,
        "span": span,
        "tolerance": tolerance,
        "low_support": low_support,
        "high_support": high_support,
        "plane_support": plane_support,
        "plane_balance": plane_balance,
        "up_confidence": up_confidence,
    }


def _estimate_horizontal_from_walls(centered: np.ndarray, up_axis: np.ndarray) -> dict[str, Any]:
    basis = _horizontal_basis(up_axis)
    first = basis[0]
    second = basis[1]

    horizontal_coords = centered @ basis.T
    covariance = np.cov(horizontal_coords, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(covariance)
    principal_2d = eigvecs[:, int(np.argmax(eigvals))]
    pca_axis = _normalize((principal_2d[0] * first) + (principal_2d[1] * second))
    pca_axis = _project_to_plane(pca_axis, up_axis)
    pca_axis = _normalize(pca_axis)

    orth_axis = _normalize(np.cross(up_axis, pca_axis))
    candidates = [first, second, pca_axis, orth_axis]

    wall_candidates_raw: list[dict[str, Any]] = []
    for idx, candidate in enumerate(candidates):
        normal = _project_to_plane(candidate, up_axis)
        if np.linalg.norm(normal) <= 1e-8:
            continue
        normal = _normalize(normal)
        wall = _evaluate_wall_pair(centered, normal)
        wall["candidate_name"] = f"horizontal_candidate_{idx}"
        wall_candidates_raw.append(wall)

    if not wall_candidates_raw:
        fallback = _fallback_horizontal(up_axis)
        return {
            "horizontal_axis": _normalize(fallback),
            "wall_candidates": [],
            "wall_candidates_raw": [],
            "wall_candidate_ranking": [],
            "confidence": 0.0,
            "ambiguity": 1.0,
            "primary_wall": None,
            "secondary_wall": None,
            "horizontal_evidence": {
                "raw_candidate_count": 0,
                "unique_orientation_count": 0,
                "orientation_dedup_applied": True,
                "primary_final_score": 0.0,
                "secondary_final_score": 0.0,
            },
        }

    wall_candidates = _dedupe_wall_candidates_by_orientation(wall_candidates_raw)
    scored_candidates = [
        _score_wall_candidate(candidate=candidate, pca_axis=pca_axis, up_axis=up_axis) for candidate in wall_candidates
    ]
    scored_candidates.sort(key=lambda item: float(item.get("final_score", 0.0)), reverse=True)

    primary = scored_candidates[0]
    secondary = scored_candidates[1] if len(scored_candidates) > 1 else None

    horizontal_axis = np.asarray(primary["horizontal_axis_candidate"], dtype=float)
    if np.linalg.norm(horizontal_axis) <= 1e-8:
        horizontal_axis = _normalize(pca_axis)

    ambiguity = 0.25
    if secondary is not None:
        primary_score = float(primary.get("final_score", 0.0))
        secondary_score = float(secondary.get("final_score", 0.0))
        score_gap = float(max(primary_score - secondary_score, 0.0))
        ambiguity = float(
            max(
                0.0,
                min(
                    1.0 - (
                        score_gap / max(primary_score, 1e-8)
                    ),
                    1.0,
                ),
            )
        )
    primary_score = float(primary.get("final_score", 0.0))
    primary_alignment = float(primary.get("pca_alignment", 0.0))
    confidence = float(
        max(
            0.0,
            min(
                ((primary_score * 0.75) + (primary_alignment * 0.25)) * (1.0 - 0.35 * ambiguity),
                1.0,
            ),
        )
    )

    return {
        "horizontal_axis": horizontal_axis,
        "wall_candidates": scored_candidates,
        "wall_candidates_raw": wall_candidates_raw,
        "wall_candidate_ranking": scored_candidates,
        "confidence": confidence,
        "ambiguity": ambiguity,
        "primary_wall": primary,
        "secondary_wall": secondary,
        "horizontal_evidence": {
            "raw_candidate_count": int(len(wall_candidates_raw)),
            "unique_orientation_count": int(len(scored_candidates)),
            "orientation_dedup_applied": True,
            "primary_final_score": float(primary.get("final_score", 0.0)),
            "secondary_final_score": float(secondary.get("final_score", 0.0)) if secondary else 0.0,
        },
    }


def _evaluate_wall_pair(centered: np.ndarray, normal: np.ndarray) -> dict[str, Any]:
    projected = centered @ normal
    low_offset = float(np.quantile(projected, 0.05))
    high_offset = float(np.quantile(projected, 0.95))
    span = float(max(high_offset - low_offset, 1e-8))
    tolerance = float(max(span * 0.025, 0.05))
    low_support = float(np.mean(np.abs(projected - low_offset) <= tolerance))
    high_support = float(np.mean(np.abs(projected - high_offset) <= tolerance))
    support = float(low_support + high_support)
    support_balance = float(
        max(
            0.0,
            min(1.0 - (abs(low_support - high_support) / max(low_support + high_support, 1e-8)), 1.0),
        )
    )
    span_plausibility = float(_band_score(span, lower=2.0, upper=12.0, tolerance=8.0))
    return {
        "normal": _normalize(normal).tolist(),
        "low_offset": low_offset,
        "high_offset": high_offset,
        "span": span,
        "tolerance": tolerance,
        "low_support": low_support,
        "high_support": high_support,
        "support": support,
        "support_balance": support_balance,
        "span_plausibility": span_plausibility,
    }


def _dedupe_wall_candidates_by_orientation(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    for candidate in candidates:
        normal = _normalize(np.asarray(candidate.get("normal", [0.0, 0.0, 1.0]), dtype=float))
        if np.linalg.norm(normal) <= 1e-8:
            continue

        matched_group: dict[str, Any] | None = None
        for group in groups:
            rep = group["representative_normal"]
            if abs(float(np.dot(normal, rep))) >= 0.995:
                matched_group = group
                break

        if matched_group is None:
            groups.append(
                {
                    "representative_normal": normal,
                    "best_candidate": dict(candidate),
                    "count": 1,
                }
            )
            continue

        matched_group["count"] = int(matched_group.get("count", 1)) + 1
        best_candidate = matched_group["best_candidate"]
        if float(candidate.get("support", 0.0)) > float(best_candidate.get("support", 0.0)):
            matched_group["best_candidate"] = dict(candidate)

    deduped: list[dict[str, Any]] = []
    for group in groups:
        representative = _orient_axis_positive(_normalize(np.asarray(group["representative_normal"], dtype=float)))
        best = dict(group["best_candidate"])
        best["orientation_group_size"] = int(group.get("count", 1))
        best["orientation_key"] = [
            int(round(float(representative[0]) * 100)),
            int(round(float(representative[1]) * 100)),
            int(round(float(representative[2]) * 100)),
        ]
        deduped.append(best)

    deduped.sort(key=lambda item: float(item.get("support", 0.0)), reverse=True)
    return deduped


def _score_wall_candidate(
    *,
    candidate: dict[str, Any],
    pca_axis: np.ndarray,
    up_axis: np.ndarray,
) -> dict[str, Any]:
    normal = _normalize(np.asarray(candidate.get("normal", [0.0, 0.0, 1.0]), dtype=float))
    horizontal_axis_candidate = _normalize(np.cross(up_axis, normal))
    if np.linalg.norm(horizontal_axis_candidate) <= 1e-8:
        horizontal_axis_candidate = _normalize(pca_axis)

    support_score = _clamp01(float(candidate.get("support", 0.0)) / 0.35)
    support_balance = _clamp01(float(candidate.get("support_balance", 0.0)))
    span_plausibility = _clamp01(float(candidate.get("span_plausibility", 0.0)))
    pca_alignment = _clamp01(abs(float(np.dot(horizontal_axis_candidate, _normalize(pca_axis)))))
    structural_score = _clamp01((support_score * 0.55) + (support_balance * 0.20) + (span_plausibility * 0.25))
    clutter_penalty = _clamp01(1.0 - support_balance)
    final_score = _clamp01((structural_score * 0.70) + (pca_alignment * 0.30) - (0.05 * clutter_penalty))

    scored = dict(candidate)
    scored.update(
        {
            "horizontal_axis_candidate": horizontal_axis_candidate.tolist(),
            "support_score": support_score,
            "pca_alignment": pca_alignment,
            "structural_score": structural_score,
            "clutter_penalty": clutter_penalty,
            "final_score": final_score,
            "scoring_reason": {
                "support_score": support_score,
                "support_balance": support_balance,
                "span_plausibility": span_plausibility,
                "pca_alignment": pca_alignment,
                "clutter_penalty": clutter_penalty,
            },
            "role_hint": "wall_like",
        }
    )
    return scored


def _build_plane_candidates(
    *,
    oriented: dict[str, Any],
    horizontal: dict[str, Any],
) -> list[dict[str, Any]]:
    up = np.asarray(oriented["axis"], dtype=float)
    planes = [
        {
            "role_candidate": "floor",
            "normal": up.tolist(),
            "offset": float(oriented["low_offset"]),
            "support_ratio": float(oriented["low_support"]),
        },
        {
            "role_candidate": "ceiling",
            "normal": up.tolist(),
            "offset": float(oriented["high_offset"]),
            "support_ratio": float(oriented["high_support"]),
        },
    ]
    for wall in horizontal.get("wall_candidates", []):
        normal = np.asarray(wall["normal"], dtype=float)
        planes.append(
            {
                "role_candidate": "wall_low",
                "normal": normal.tolist(),
                "offset": float(wall["low_offset"]),
                "support_ratio": float(wall["low_support"]),
            }
        )
        planes.append(
            {
                "role_candidate": "wall_high",
                "normal": normal.tolist(),
                "offset": float(wall["high_offset"]),
                "support_ratio": float(wall["high_support"]),
            }
        )
    return planes


def _build_selected_plane_roles(
    *,
    oriented: dict[str, Any],
    horizontal: dict[str, Any],
) -> dict[str, Any]:
    up = np.asarray(oriented["axis"], dtype=float)
    roles: dict[str, Any] = {
        "floor": {
            "normal": up.tolist(),
            "offset": float(oriented["low_offset"]),
            "support_ratio": float(oriented["low_support"]),
            "role_reason": "lower_extreme_along_selected_up_axis",
        },
        "ceiling": {
            "normal": up.tolist(),
            "offset": float(oriented["high_offset"]),
            "support_ratio": float(oriented["high_support"]),
            "role_reason": "upper_extreme_along_selected_up_axis",
        },
    }

    primary = horizontal.get("primary_wall")
    secondary = horizontal.get("secondary_wall")
    if isinstance(primary, dict):
        roles["wall_primary"] = {
            "normal": primary["normal"],
            "low_offset": float(primary["low_offset"]),
            "high_offset": float(primary["high_offset"]),
            "support_ratio": float(primary["support"]),
            "score": float(primary.get("final_score", 0.0)),
            "role_reason": "highest_scored_structural_wall_candidate",
        }
    if isinstance(secondary, dict):
        roles["wall_secondary"] = {
            "normal": secondary["normal"],
            "low_offset": float(secondary["low_offset"]),
            "high_offset": float(secondary["high_offset"]),
            "support_ratio": float(secondary["support"]),
            "score": float(secondary.get("final_score", 0.0)),
            "role_reason": "second_best_structural_wall_candidate",
        }
    return roles


def _assess_scale_plausibility(
    *,
    transformed_stats: dict[str, float | list[float]],
    selected_plane_roles: dict[str, Any],
) -> dict[str, float | list[float] | str | bool]:
    ranges = transformed_stats.get("ranges", [0.0, 0.0, 0.0])
    if not isinstance(ranges, list) or len(ranges) != 3:
        ranges = [0.0, 0.0, 0.0]

    floor_offset = 0.0
    ceiling_offset = 0.0
    floor_role = selected_plane_roles.get("floor", {})
    ceiling_role = selected_plane_roles.get("ceiling", {})
    if isinstance(floor_role, dict):
        floor_offset = float(floor_role.get("offset", 0.0))
    if isinstance(ceiling_role, dict):
        ceiling_offset = float(ceiling_role.get("offset", float(ranges[2])))

    room_height = float(abs(ceiling_offset - floor_offset))
    if room_height <= 1e-6:
        room_height = float(abs(ranges[2]))
    horizontal_extent = float(max(float(ranges[0]), float(ranges[1])))

    room_height_score = _band_score(room_height, lower=2.2, upper=3.8, tolerance=2.0)
    horizontal_extent_score = _band_score(horizontal_extent, lower=2.5, upper=15.0, tolerance=15.0)

    plausibility = float(max(0.0, min((room_height_score * 0.7) + (horizontal_extent_score * 0.3), 1.0)))

    suggested_scale = 1.0
    reason = "within_priors"
    if room_height > 0.0:
        target_height = 2.8
        drift = abs(room_height - target_height)
        if drift > 1.4:
            suggested_scale = float(target_height / max(room_height, 1e-8))
            reason = "room_height_out_of_prior"

    scale_drift = float(abs(suggested_scale - 1.0))
    return {
        "estimated_room_height": room_height,
        "estimated_horizontal_extent": horizontal_extent,
        "priors": {
            "door_height_m": [1.9, 2.2],
            "room_height_m": [2.2, 3.8],
            "furniture_size_m": [0.2, 2.2],
        },
        "plausibility_score": plausibility,
        "suggested_scale_factor_to_meters": float(suggested_scale),
        "scale_drift": scale_drift,
        "applied_scale_adjustment": False,
        "reason": reason,
    }


def _decide_horizontal_strategy(
    *,
    horizontal_confidence: float,
    horizontal_ambiguity: float,
    min_horizontal_confidence: float,
    min_horizontal_confidence_for_acceptance: float,
    max_manhattan_ambiguity_for_acceptance: float,
    strong_horizontal_confidence_for_ambiguous_acceptance: float,
    max_manhattan_ambiguity_for_strong_acceptance: float,
    primary_wall_score: float = 0.0,
    secondary_wall_score: float = 0.0,
    unique_orientation_count: int = 0,
) -> dict[str, Any]:
    conf = _clamp01(horizontal_confidence)
    ambiguity = _clamp01(horizontal_ambiguity)
    primary_score = _clamp01(primary_wall_score)
    secondary_score = _clamp01(secondary_wall_score)
    score_gap = max(primary_score - secondary_score, 0.0)
    separation = _clamp01(score_gap / max(primary_score, 1e-8))
    orientation_bonus = 0.05 if unique_orientation_count >= 2 else 0.0
    structural_strength = _clamp01((primary_score * 0.70) + (separation * 0.30) + orientation_bonus)
    commitment_quality = _clamp01(
        (conf * 0.55) + (structural_strength * 0.35) + ((1.0 - ambiguity) * 0.10)
    )

    # Discount apparent ambiguity when strong structural wall evidence is available.
    effective_ambiguity = _clamp01(ambiguity - (0.25 * structural_strength))

    ambiguity_excess = max(0.0, effective_ambiguity - max_manhattan_ambiguity_for_acceptance)
    dynamic_min_confidence = float(
        max(
            min_horizontal_confidence_for_acceptance,
            min(
                1.0,
                min_horizontal_confidence_for_acceptance + (ambiguity_excess * 0.65),
            ),
        )
    )
    evidence_strength = _clamp01(
        (conf * (1.0 - (0.35 * effective_ambiguity)))
        + (0.15 * (1.0 - effective_ambiguity))
        + (0.10 * structural_strength)
    )

    reasons: list[str] = []
    accepted_by: str | None = None
    partial_axis_strategy = "up_only"
    partial_confidence_cap = 0.55
    if conf < min_horizontal_confidence:
        mode = "fallback"
        reasons.append("hard_low_horizontal_confidence")
    else:
        standard_accept = (
            conf >= dynamic_min_confidence and effective_ambiguity <= max_manhattan_ambiguity_for_acceptance
        )
        strong_accept = (
            conf >= strong_horizontal_confidence_for_ambiguous_acceptance
            and ambiguity <= max_manhattan_ambiguity_for_strong_acceptance
        )
        structural_accept = (
            conf >= max(dynamic_min_confidence + 0.03, min_horizontal_confidence_for_acceptance)
            and structural_strength >= 0.78
            and separation >= 0.45
            and commitment_quality >= 0.62
            and effective_ambiguity <= max_manhattan_ambiguity_for_acceptance
        )
        if standard_accept:
            mode = "accept"
            accepted_by = "standard_threshold"
            reasons.append("sufficient_horizontal_evidence")
        elif strong_accept:
            mode = "accept"
            accepted_by = "strong_confidence_override"
            reasons.append("strong_confidence_override")
        elif structural_accept:
            mode = "accept"
            accepted_by = "structural_consensus"
            reasons.append("structural_consensus_override")
        else:
            mode = "partial"
            if conf < dynamic_min_confidence:
                reasons.append("below_dynamic_confidence_threshold")
            if effective_ambiguity > max_manhattan_ambiguity_for_acceptance:
                reasons.append("high_manhattan_ambiguity")
            if structural_strength < 0.60:
                reasons.append("weak_structural_wall_evidence")
            if (
                structural_strength >= 0.64
                and separation >= 0.35
                and conf >= max(min_horizontal_confidence_for_acceptance - 0.10, min_horizontal_confidence)
            ):
                partial_axis_strategy = "analysis_projected"
                partial_confidence_cap = float(
                    _clamp01(0.56 + (0.06 * structural_strength) + (0.04 * separation))
                )
                reasons.append("conservative_structural_partial")
            else:
                partial_axis_strategy = "up_only"
                partial_confidence_cap = 0.55
            if not reasons:
                reasons.append("uncertain_horizontal_evidence")

    return {
        "mode": mode,
        "accepted_by": accepted_by,
        "dynamic_min_horizontal_confidence_for_acceptance": dynamic_min_confidence,
        "evidence_strength": evidence_strength,
        "structural_strength": structural_strength,
        "structural_separation": separation,
        "effective_manhattan_ambiguity": effective_ambiguity,
        "commitment_quality": commitment_quality,
        "partial_axis_strategy": partial_axis_strategy,
        "partial_confidence_cap": float(partial_confidence_cap),
        "reasons": reasons,
    }


def _compute_reliability_v1_3(
    *,
    up_confidence: float,
    horizontal_confidence: float,
    manhattan_ambiguity: float,
    reliability_mode: str,
    up_guardrail_applied: bool,
    horizontal_decision_mode: str,
    horizontal_evidence_strength: float = 0.0,
    effective_manhattan_ambiguity: float | None = None,
    partial_axis_strategy: str = "",
    accepted_by: str = "",
    commitment_quality: float = 0.0,
) -> tuple[float, dict[str, Any]]:
    up = _clamp01(up_confidence)
    horizontal = _clamp01(horizontal_confidence)
    ambiguity = _clamp01(manhattan_ambiguity)
    effective_ambiguity = _clamp01(
        effective_manhattan_ambiguity if effective_manhattan_ambiguity is not None else ambiguity
    )
    evidence_strength = _clamp01(horizontal_evidence_strength)
    commit_quality = _clamp01(commitment_quality)

    effective_horizontal = _clamp01(horizontal * (1.0 - (0.30 * effective_ambiguity)))
    up_component = 0.65 * up
    horizontal_component = 0.35 * effective_horizontal
    base_score = up_component + horizontal_component

    mode_bonus = 0.0
    mode_penalty = 0.0
    reasons: list[str] = []

    if reliability_mode == "full_calibration":
        mode_bonus += 0.04 + (0.08 * commit_quality)
        reasons.append("full_calibration_bonus")
        if accepted_by == "structural_consensus":
            if commit_quality >= 0.70:
                mode_bonus += 0.01
                reasons.append("high_quality_structural_consensus_bonus")
            else:
                mode_penalty += 0.05
                reasons.append("low_quality_structural_consensus_penalty")
        if effective_ambiguity > 0.80 and commit_quality < 0.60:
            mode_penalty += 0.03
            reasons.append("high_ambiguity_full_commit_penalty")
    elif reliability_mode == "safe_partial_calibration":
        mode_bonus += 0.06 + (0.08 * up) + (0.06 * evidence_strength) + (0.02 * commit_quality)
        mode_penalty += 0.015 * effective_ambiguity
        reasons.append("safe_partial_bonus")
        if partial_axis_strategy == "analysis_projected":
            mode_bonus += 0.015
            reasons.append("analysis_projected_partial_bonus")
        if effective_ambiguity > 0.0:
            reasons.append("ambiguity_penalty")
    else:
        mode_penalty += 0.08
        reasons.append("degraded_fallback_penalty")

    if up_guardrail_applied:
        mode_penalty += 0.02
        reasons.append("up_guardrail_penalty")

    if horizontal_decision_mode == "fallback":
        mode_penalty += 0.02
        reasons.append("horizontal_fallback_penalty")
    elif horizontal_decision_mode == "partial":
        mode_penalty += 0.01
        reasons.append("horizontal_partial_penalty")

    score = _clamp01(base_score + mode_bonus - mode_penalty)
    if reliability_mode == "degraded_fallback":
        score = float(min(score, 0.45))

    return score, {
        "reliability_mode": reliability_mode,
        "horizontal_decision_mode": horizontal_decision_mode,
        "up_confidence": up,
        "horizontal_confidence_raw": horizontal,
        "manhattan_ambiguity": ambiguity,
        "effective_manhattan_ambiguity": effective_ambiguity,
        "horizontal_evidence_strength": evidence_strength,
        "accepted_by": accepted_by,
        "commitment_quality": commit_quality,
        "partial_axis_strategy": partial_axis_strategy,
        "up_component": up_component,
        "horizontal_component_effective": horizontal_component,
        "horizontal_effective_confidence": effective_horizontal,
        "base_score": base_score,
        "mode_bonus": mode_bonus,
        "mode_penalty": mode_penalty,
        "final_reliability": score,
        "reasons": reasons,
    }


def _band_score(value: float, lower: float, upper: float, tolerance: float) -> float:
    if lower <= value <= upper:
        return 1.0
    if value < lower:
        return max(0.0, 1.0 - ((lower - value) / max(tolerance, 1e-8)))
    return max(0.0, 1.0 - ((value - upper) / max(tolerance, 1e-8)))


def _clamp01(value: float) -> float:
    return float(max(0.0, min(float(value), 1.0)))


def _build_rotation_matrix(horizontal_axis: np.ndarray, up_vector: np.ndarray) -> np.ndarray:
    up = _normalize(up_vector)
    horizontal = _project_to_plane(horizontal_axis, up)
    if np.linalg.norm(horizontal) <= 1e-8:
        horizontal = _fallback_horizontal(up)
    horizontal = _normalize(horizontal)

    side = np.cross(up, horizontal)
    if np.linalg.norm(side) <= 1e-8:
        side = _fallback_horizontal(up)
    side = _normalize(side)

    horizontal = _normalize(np.cross(side, up))
    return np.vstack([horizontal, side, up])


def _compute_transformed_stats_from_rows(rows: list[list[float]]) -> dict[str, float | list[float]]:
    xs = [row[0] for row in rows]
    ys = [row[1] for row in rows]
    zs = [row[2] for row in rows]

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    min_z, max_z = min(zs), max(zs)

    center_x = sum(xs) / len(xs)
    center_y = sum(ys) / len(ys)
    center_z = sum(zs) / len(zs)

    return {
        "bbox_min": [float(min_x), float(min_y), float(min_z)],
        "bbox_max": [float(max_x), float(max_y), float(max_z)],
        "centroid": [float(center_x), float(center_y), float(center_z)],
        "ranges": [float(max_x - min_x), float(max_y - min_y), float(max_z - min_z)],
    }


def _fallback_horizontal(up_vector: np.ndarray) -> np.ndarray:
    candidates = [
        np.array([1.0, 0.0, 0.0], dtype=float),
        np.array([0.0, 1.0, 0.0], dtype=float),
    ]
    for candidate in candidates:
        projection = _project_to_plane(candidate, up_vector)
        if np.linalg.norm(projection) > 1e-8:
            return projection
    return np.array([1.0, 0.0, 0.0], dtype=float)


def _horizontal_basis(up_vector: np.ndarray) -> np.ndarray:
    first = _normalize(_fallback_horizontal(up_vector))
    second = np.cross(up_vector, first)
    second = _normalize(second)
    return np.vstack([first, second])


def _project_to_plane(vector: np.ndarray, normal: np.ndarray) -> np.ndarray:
    return vector - (np.dot(vector, normal) * normal)


def _orient_axis_positive(axis: np.ndarray) -> np.ndarray:
    largest_idx = int(np.argmax(np.abs(axis)))
    if axis[largest_idx] < 0:
        return -1.0 * axis
    return axis


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        return np.array([0.0, 0.0, 1.0], dtype=float)
    return vector / norm


def _point_to_array(point: Point3D) -> np.ndarray:
    return np.array([float(point.x), float(point.y), float(point.z)], dtype=float)


def _angle_between_deg(a: np.ndarray, b: np.ndarray) -> float:
    na = _normalize(a)
    nb = _normalize(b)
    dot = float(max(min(np.dot(na, nb), 1.0), -1.0))
    return float(np.degrees(np.arccos(dot)))
