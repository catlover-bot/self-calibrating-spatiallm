from __future__ import annotations

import json
from pathlib import Path

from self_calibrating_spatiallm.robustness.analysis import (
    build_boundary_rows,
    build_boundary_summary,
    export_language_boundary_artifacts,
    render_boundary_summary_markdown,
)


def _scene_setting(setting_name: str, *, metrics: dict[str, float], metadata: dict | None = None) -> dict:
    return {
        "setting_name": setting_name,
        "status": "success",
        "metrics": metrics,
        "failures": [],
        "metadata": metadata or {},
    }


def _report_scene(*, scene_id: str, severity: float, reliability_delta: float, partial: bool) -> tuple[dict, dict]:
    delta_row = {
        "scene_id": scene_id,
        "comparison_valid": True,
        "comparison_valid_with_repair": True,
        "partial_calibration_applied": partial,
        "true_v1_execution": True,
        "fallback_used": False,
        "fallback_reason": None,
        "candidate_plane_count": 8,
        "major_failure_category": "manhattan_ambiguity",
        "major_failure_labels": ["manhattan_ambiguity"],
        "calibration_summary": {
            "manhattan_ambiguity": 0.45 + severity * 0.4,
            "horizontal_confidence": 0.75 - severity * 0.2,
            "overall_reliability": 0.70 + reliability_delta,
        },
        "delta_v1_minus_v0": {
            "calibration_reliability": reliability_delta,
            "structured_violation_count_before_repair": -0.2,
            "actionable_relation_f1": 0.03,
        },
        "delta_v1_plus_repair_minus_v1": {
            "repair_violations_after": -0.5,
        },
        "prediction_delta_v1_minus_v0": {
            "object_count_delta": 1,
            "relation_count_delta": 1,
        },
        "prediction_delta_external_minus_mock": {
            "object_count_delta": 2,
            "relation_count_delta": -1,
        },
        "violation_delta_external_minus_mock": 1.0,
    }

    scene = {
        "scene_id": scene_id,
        "source_type": "ply",
        "tags": [
            "rb:split=small",
            "rb:split_role=clean_validation",
            "rb:base_scene=scene_base_1",
            "rb:perturbation=rotation_yaw",
            f"rb:severity={severity:.3f}",
            "rb:severity_bucket=mid",
        ],
        "settings": [
            _scene_setting(
                "calibration_v0",
                metrics={
                    "calibration_reliability": 0.70,
                    "calibration_horizontal_error_deg": 15.0,
                    "calibration_up_axis_error_deg": 2.0,
                    "structured_violation_count_before_repair": 3.0,
                    "repair_violations_after": 1.0,
                    "actionable_relation_f1": 0.3,
                },
            ),
            _scene_setting(
                "calibration_v1",
                metrics={
                    "calibration_reliability": 0.70 + reliability_delta,
                    "calibration_horizontal_error_deg": 12.0,
                    "calibration_up_axis_error_deg": 2.5,
                    "structured_violation_count_before_repair": 2.8,
                    "repair_violations_after": 0.7,
                    "actionable_relation_f1": 0.33,
                },
            ),
            _scene_setting(
                "calibration_v1_plus_repair",
                metrics={
                    "calibration_reliability": 0.70 + reliability_delta,
                    "calibration_horizontal_error_deg": 12.0,
                    "calibration_up_axis_error_deg": 2.5,
                    "structured_violation_count_before_repair": 2.8,
                    "repair_violations_after": 0.3,
                    "actionable_relation_f1": 0.36,
                },
            ),
            _scene_setting(
                "mock_generator",
                metrics={
                    "structured_violation_count_before_repair": 3.0,
                    "actionable_relation_f1": 0.31,
                },
            ),
            _scene_setting(
                "external_generator",
                metrics={
                    "structured_violation_count_before_repair": 4.0,
                    "actionable_relation_f1": 0.28,
                },
            ),
        ],
    }
    return scene, delta_row


def test_boundary_summary_includes_failure_and_fallback_slices() -> None:
    scene_low, delta_low = _report_scene(
        scene_id="scene_low",
        severity=0.2,
        reliability_delta=0.06,
        partial=False,
    )
    scene_high, delta_high = _report_scene(
        scene_id="scene_high",
        severity=0.8,
        reliability_delta=-0.08,
        partial=True,
    )

    split_reports = {
        "small": {
            "split_role": "clean_validation",
            "report": {
                "scenes": [scene_low, scene_high],
                "scene_level_delta_report": [delta_low, delta_high],
            },
        }
    }
    inventory = [
        {
            "split_name": "small",
            "base_scene_id": "scene_base_1",
            "scene_id": "scene_low",
            "perturbation_type": "rotation_yaw",
            "severity": 0.2,
            "severity_bucket": "low",
            "parameters": {"yaw_deg": 14.0},
        },
        {
            "split_name": "small",
            "base_scene_id": "scene_base_1",
            "scene_id": "scene_high",
            "perturbation_type": "rotation_yaw",
            "severity": 0.8,
            "severity_bucket": "high",
            "parameters": {"yaw_deg": 56.0},
        },
    ]

    rows = build_boundary_rows(split_reports=split_reports, perturbation_inventory=inventory)
    assert len(rows) == 2
    assert rows[0]["perturbation_type"] == "rotation_yaw"

    summary = build_boundary_summary(rows)
    assert "by_major_failure_category" in summary
    assert "manhattan_ambiguity" in summary["by_major_failure_category"]
    assert "by_fallback_reason" in summary
    assert "none" in summary["by_fallback_reason"]

    findings = summary.get("boundary_findings", [])
    assert isinstance(findings, list)
    assert any("reliability degradation begins around severity" in item for item in findings)

    markdown = render_boundary_summary_markdown(summary)
    assert "## Major Failure Categories" in markdown
    assert "## Fallback Reasons" in markdown


def test_language_boundary_artifacts_preserve_explicit_relations(tmp_path: Path) -> None:
    split_reports = {
        "small": {
            "split_role": "clean_validation",
            "report": {
                "scenes": [
                    {
                        "scene_id": "scene_lang",
                        "source_type": "ply",
                        "tags": [
                            "rb:split=small",
                            "rb:base_scene=scene_lang_base",
                            "rb:perturbation=none",
                            "rb:severity=0.000",
                            "rb:severity_bucket=low",
                        ],
                        "settings": [
                            {
                                "setting_name": "calibration_v1",
                                "status": "success",
                                "metadata": {
                                    "structured_prediction_pre_repair": {
                                        "sample_id": "scene_lang",
                                        "generator_name": "mock",
                                        "objects": [
                                            {
                                                "object_id": "table_0",
                                                "label": "table",
                                                "position": {"x": 0.0, "y": 0.0, "z": 0.8},
                                                "size": {"x": 1.0, "y": 0.7, "z": 0.7},
                                                "confidence": 0.9,
                                                "attributes": {},
                                            },
                                            {
                                                "object_id": "mug_0",
                                                "label": "mug",
                                                "position": {"x": 0.1, "y": 0.0, "z": 0.95},
                                                "size": {"x": 0.1, "y": 0.1, "z": 0.12},
                                                "confidence": 0.85,
                                                "attributes": {},
                                            },
                                        ],
                                        "relations": [
                                            {
                                                "subject_id": "mug_0",
                                                "predicate": "supported-by",
                                                "object_id": "table_0",
                                                "score": 0.9,
                                                "metadata": {},
                                            }
                                        ],
                                        "metadata": {},
                                    }
                                },
                            }
                        ],
                    }
                ]
            },
        }
    }

    inventory = [
        {
            "split_name": "small",
            "scene_id": "scene_lang",
            "base_scene_id": "scene_lang_base",
            "perturbation_type": "none",
            "severity": 0.0,
            "severity_bucket": "low",
            "parameters": {},
        }
    ]

    outputs = export_language_boundary_artifacts(
        split_reports=split_reports,
        perturbation_inventory=inventory,
        output_dir=tmp_path,
    )

    summary_payload = json.loads(outputs["language_summary_json"].read_text(encoding="utf-8"))
    assert summary_payload["num_rows_with_structured_prediction"] == 1
    assert summary_payload["num_rows_with_explicit_relations"] == 1

    scene_lines = outputs["language_scene_examples_jsonl"].read_text(encoding="utf-8").splitlines()
    first_row = json.loads(scene_lines[0])
    assert first_row["relation_count"] == 1
    assert first_row["relation_evidence_level"] == "explicit"
