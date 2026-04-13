from pathlib import Path

import numpy as np

from self_calibrating_spatiallm.calibration import PlaneAwareCalibratorV1
from self_calibrating_spatiallm.calibration.plane_aware_v1 import (
    _compute_reliability_v1_3,
    _dedupe_wall_candidates_by_orientation,
    _decide_horizontal_strategy,
    _score_wall_candidate,
)
from self_calibrating_spatiallm.io import PointCloudLoadOptions, load_point_cloud_sample


def _load_indoor_sample():
    root = Path(__file__).resolve().parents[2]
    sample_path = root / "configs" / "samples" / "indoor_room_real.ply"
    metadata_path = root / "configs" / "samples" / "real_scene_metadata.json"
    options = PointCloudLoadOptions(
        scene_id="indoor_room_real_001",
        source_type="ply",
        metadata_path=metadata_path,
        expected_unit="meter",
    )
    sample, _ = load_point_cloud_sample(sample_path, options)
    return sample


def test_plane_aware_v1_prefers_z_up_for_room_sample() -> None:
    sample = _load_indoor_sample()
    calibrated = PlaneAwareCalibratorV1(normalize_scene=True).calibrate(sample)

    up = calibrated.calibration.up_vector
    assert abs(up.z) > 0.90
    assert abs(up.z) > abs(up.x)
    assert abs(up.z) > abs(up.y)

    diagnostics = calibrated.calibration.metadata.get("diagnostics", {})
    selection = diagnostics.get("up_axis_selection", {})
    execution = calibrated.calibration.metadata.get("execution", {})
    assert selection.get("selection_strategy") == "v1.4_plane_role_scored_confidence_guarded"
    assert isinstance(selection.get("selected_score"), float)
    assert "selected_span_plausibility" in selection
    assert "sign_correction_applied" in selection
    assert calibrated.calibration.metadata.get("algorithm_version") == "v1.4_plane_interpretation_scored"
    assert execution.get("fallback_used") is False
    assert execution.get("partial_calibration_applied") is False


def test_plane_aware_v1_records_partial_guardrail_execution() -> None:
    sample = _load_indoor_sample()
    calibrated = PlaneAwareCalibratorV1(
        normalize_scene=True,
        max_up_axis_disagreement_deg=-1.0,
        min_up_confidence_for_override=1.01,
    ).calibrate(sample)

    execution = calibrated.calibration.metadata.get("execution", {})
    diagnostics = calibrated.calibration.metadata.get("diagnostics", {})
    guardrail = diagnostics.get("up_axis_guardrail", {})

    assert execution.get("partial_calibration_applied") is True
    assert diagnostics.get("up_axis_source") == "v0_guardrail"
    assert guardrail.get("applied") is True
    assert isinstance(execution.get("partial_calibration_reasons"), list)


def test_plane_aware_v1_horizontal_ambiguity_guardrail() -> None:
    sample = _load_indoor_sample()
    calibrated = PlaneAwareCalibratorV1(
        normalize_scene=True,
        max_manhattan_ambiguity_for_acceptance=0.2,
        strong_horizontal_confidence_for_ambiguous_acceptance=1.01,
    ).calibrate(sample)

    execution = calibrated.calibration.metadata.get("execution", {})
    diagnostics = calibrated.calibration.metadata.get("diagnostics", {})
    horizontal_guardrail = diagnostics.get("horizontal_guardrail", {})

    assert execution.get("partial_calibration_applied") is True
    assert horizontal_guardrail.get("triggered") is True
    assert diagnostics.get("horizontal_axis_source") in {
        "up_only_horizontal_guardrail",
        "analysis_projected_partial_guardrail",
    }
    assert horizontal_guardrail.get("decision_mode") == "partial"
    assert isinstance(horizontal_guardrail.get("dynamic_min_horizontal_confidence_for_acceptance"), float)

    up = calibrated.calibration.up_vector
    assert abs(up.z) > 0.90


def test_plane_aware_v1_hard_horizontal_confidence_fallback() -> None:
    sample = _load_indoor_sample()
    calibrated = PlaneAwareCalibratorV1(
        normalize_scene=True,
        min_horizontal_confidence=0.99,
    ).calibrate(sample)

    execution = calibrated.calibration.metadata.get("execution", {})
    diagnostics = calibrated.calibration.metadata.get("diagnostics", {})
    horizontal_guardrail = diagnostics.get("horizontal_guardrail", {})

    assert execution.get("fallback_used") is True
    assert execution.get("fallback_reason") == "insufficient_plane_evidence:weak_horizontal_axis"
    assert execution.get("partial_calibration_applied") is True
    assert horizontal_guardrail.get("triggered") is True
    assert horizontal_guardrail.get("action") == "fallback_to_v0_horizontal_axis"
    assert horizontal_guardrail.get("decision_mode") == "fallback"


def test_horizontal_decision_dynamic_thresholding() -> None:
    decision = _decide_horizontal_strategy(
        horizontal_confidence=0.45,
        horizontal_ambiguity=1.0,
        min_horizontal_confidence=0.15,
        min_horizontal_confidence_for_acceptance=0.35,
        max_manhattan_ambiguity_for_acceptance=0.85,
        strong_horizontal_confidence_for_ambiguous_acceptance=0.70,
        max_manhattan_ambiguity_for_strong_acceptance=0.95,
    )
    assert decision["mode"] == "partial"
    assert float(decision["dynamic_min_horizontal_confidence_for_acceptance"]) > 0.44

    strong = _decide_horizontal_strategy(
        horizontal_confidence=0.80,
        horizontal_ambiguity=0.92,
        min_horizontal_confidence=0.15,
        min_horizontal_confidence_for_acceptance=0.35,
        max_manhattan_ambiguity_for_acceptance=0.85,
        strong_horizontal_confidence_for_ambiguous_acceptance=0.70,
        max_manhattan_ambiguity_for_strong_acceptance=0.95,
    )
    assert strong["mode"] == "accept"
    assert strong["accepted_by"] == "strong_confidence_override"


def test_horizontal_decision_structural_consensus_override() -> None:
    decision = _decide_horizontal_strategy(
        horizontal_confidence=0.33,
        horizontal_ambiguity=0.90,
        min_horizontal_confidence=0.15,
        min_horizontal_confidence_for_acceptance=0.35,
        max_manhattan_ambiguity_for_acceptance=0.85,
        strong_horizontal_confidence_for_ambiguous_acceptance=0.70,
        max_manhattan_ambiguity_for_strong_acceptance=0.95,
        primary_wall_score=0.95,
        secondary_wall_score=0.15,
        unique_orientation_count=2,
    )
    assert decision["mode"] == "accept"
    assert decision["accepted_by"] == "structural_consensus"
    assert float(decision["structural_strength"]) > 0.70


def test_reliability_scoring_distinguishes_modes() -> None:
    full_rel, _ = _compute_reliability_v1_3(
        up_confidence=0.72,
        horizontal_confidence=0.65,
        manhattan_ambiguity=0.20,
        reliability_mode="full_calibration",
        up_guardrail_applied=False,
        horizontal_decision_mode="accept",
    )
    partial_rel, _ = _compute_reliability_v1_3(
        up_confidence=0.72,
        horizontal_confidence=0.45,
        manhattan_ambiguity=1.0,
        reliability_mode="safe_partial_calibration",
        up_guardrail_applied=False,
        horizontal_decision_mode="partial",
    )
    fallback_rel, breakdown = _compute_reliability_v1_3(
        up_confidence=0.40,
        horizontal_confidence=0.10,
        manhattan_ambiguity=1.0,
        reliability_mode="degraded_fallback",
        up_guardrail_applied=False,
        horizontal_decision_mode="fallback",
    )

    assert full_rel > partial_rel > fallback_rel
    assert partial_rel > 0.55
    assert breakdown["reliability_mode"] == "degraded_fallback"


def test_reliability_rewards_analysis_projected_partial_strategy() -> None:
    rel_up_only, _ = _compute_reliability_v1_3(
        up_confidence=0.72,
        horizontal_confidence=0.46,
        manhattan_ambiguity=0.90,
        reliability_mode="safe_partial_calibration",
        up_guardrail_applied=False,
        horizontal_decision_mode="partial",
        horizontal_evidence_strength=0.55,
        effective_manhattan_ambiguity=0.82,
        partial_axis_strategy="up_only",
    )
    rel_analysis, breakdown = _compute_reliability_v1_3(
        up_confidence=0.72,
        horizontal_confidence=0.46,
        manhattan_ambiguity=0.90,
        reliability_mode="safe_partial_calibration",
        up_guardrail_applied=False,
        horizontal_decision_mode="partial",
        horizontal_evidence_strength=0.55,
        effective_manhattan_ambiguity=0.82,
        partial_axis_strategy="analysis_projected",
    )
    assert rel_analysis > rel_up_only
    assert "analysis_projected_partial_bonus" in breakdown["reasons"]


def test_plane_role_assignment_and_ranking_diagnostics_present() -> None:
    sample = _load_indoor_sample()
    calibrated = PlaneAwareCalibratorV1(normalize_scene=True).calibrate(sample)

    diagnostics = calibrated.calibration.metadata.get("diagnostics", {})
    ranking = diagnostics.get("wall_candidate_ranking", [])
    roles = diagnostics.get("selected_plane_roles", {})
    evidence = diagnostics.get("horizontal_evidence", {})

    assert isinstance(ranking, list)
    assert ranking
    assert "wall_primary" in roles
    assert "score" in roles["wall_primary"]
    assert "role_reason" in roles["wall_primary"]
    assert isinstance(evidence.get("unique_orientation_count"), int)
    assert evidence.get("unique_orientation_count", 0) >= 2


def test_wall_candidate_orientation_dedup_and_scoring() -> None:
    candidates = [
        {
            "normal": [0.0, 1.0, 0.0],
            "support": 0.42,
            "support_balance": 0.9,
            "span_plausibility": 0.9,
        },
        {
            "normal": [0.0, -1.0, 0.0],
            "support": 0.41,
            "support_balance": 0.9,
            "span_plausibility": 0.9,
        },
        {
            "normal": [1.0, 0.0, 0.0],
            "support": 0.20,
            "support_balance": 0.5,
            "span_plausibility": 0.6,
        },
    ]

    deduped = _dedupe_wall_candidates_by_orientation(candidates)
    assert len(deduped) == 2
    assert max(float(item.get("orientation_group_size", 0)) for item in deduped) >= 2

    scored = [
        _score_wall_candidate(
            candidate=item,
            pca_axis=np.array([1.0, 0.0, 0.0], dtype=float),
            up_axis=np.array([0.0, 0.0, 1.0], dtype=float),
        )
        for item in deduped
    ]
    scores = sorted((float(item["final_score"]) for item in scored), reverse=True)
    assert scores[0] > scores[1]
