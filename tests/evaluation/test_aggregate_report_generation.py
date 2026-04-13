import json
from pathlib import Path

import numpy as np

from self_calibrating_spatiallm.evaluation import run_evaluation_pack


def test_aggregate_report_generation(tmp_path: Path) -> None:
    points = np.array(
        [
            [-1.0, -1.0, 0.0],
            [1.0, -1.0, 0.0],
            [-1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    point_path = tmp_path / "scene.npy"
    np.save(point_path, points)

    annotation_path = tmp_path / "scene.annotation.json"
    annotation_path.write_text(
        json.dumps(
            {
                "scene_id": "scene-eval",
                "expected_up_axis": "z",
                "expected_horizontal_axis": "x",
                "expected_door_count": 1,
                "expected_window_count": 0,
                "expected_object_categories": ["floor", "wall", "table", "door", "mug"],
            }
        ),
        encoding="utf-8",
    )

    config_path = tmp_path / "scene_config.json"
    config_path.write_text(
        json.dumps(
            {
                "scene_id": "scene-eval",
                "file_path": str(point_path),
                "source_type": "npy",
                "expected_unit": "meter",
                "generator_mode": "mock",
                "normalize_scene": True,
                "output_dir": str(tmp_path / "outputs" / "scene"),
            }
        ),
        encoding="utf-8",
    )

    manifest_path = tmp_path / "eval_pack.json"
    manifest_path.write_text(
        json.dumps(
            {
                "name": "test_pack",
                "entries": [
                    {
                        "sample_config_path": str(config_path),
                        "annotation_path": str(annotation_path),
                        "source_type": "npy",
                        "tags": ["test"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    outputs = run_evaluation_pack(manifest_path=manifest_path, output_dir=tmp_path / "eval_out")
    report_payload = json.loads(outputs["evaluation_report_json"].read_text(encoding="utf-8"))

    assert report_payload["manifest_name"] == "test_pack"
    assert len(report_payload["scenes"]) == 1
    assert "no_calibration" in report_payload["aggregate_by_setting"]
    assert "calibration_v0" in report_payload["aggregate_by_setting"]
    assert "calibration_v1" in report_payload["aggregate_by_setting"]
    assert "calibration_v1_plus_repair" in report_payload["aggregate_by_setting"]
    assert "mock_generator" in report_payload["aggregate_by_setting"]
    assert "v1_execution_summary" in report_payload
    assert "v0_v1_comparison_summary" in report_payload
    assert "scene_level_delta_report" in report_payload
    assert "stratified_v0_v1_summary" in report_payload
    assert "first_research_result_summary" in report_payload
    assert "trustworthy_comparison_status" in report_payload
    assert "next_improvement_decision" in report_payload
    assert "researcher_summary" in report_payload
    assert "external_propagation_summary" in report_payload
    assert "num_true_v1_execution" in report_payload["v1_execution_summary"]
    assert "num_fallback_used" in report_payload["v1_execution_summary"]
    assert "improvements_over_v0" in report_payload["v0_v1_comparison_summary"]
    assert isinstance(report_payload["scene_level_delta_report"], list)
    assert isinstance(report_payload["stratified_v0_v1_summary"], dict)
    assert isinstance(report_payload["first_research_result_summary"], dict)
    assert isinstance(report_payload["trustworthy_comparison_status"], dict)
    assert isinstance(report_payload["next_improvement_decision"], dict)
    assert isinstance(report_payload["researcher_summary"], dict)
    assert isinstance(report_payload["external_propagation_summary"], dict)
    v1_summary = report_payload["v1_execution_summary"]
    assert isinstance(v1_summary["num_scene_setting_results"], int)
    assert isinstance(v1_summary["num_true_v1_execution"], int)
    assert isinstance(v1_summary["num_fallback_used"], int)
    assert (
        v1_summary["num_true_v1_execution"] + v1_summary["num_fallback_used"]
        == v1_summary["num_scene_setting_results"]
    )
    scene_settings = report_payload["scenes"][0]["settings"]
    v1_setting = next(item for item in scene_settings if item["setting_name"] == "calibration_v1")
    metadata = v1_setting.get("metadata", {})
    assert "prediction_summary_pre_repair" in metadata
    assert "structured_prediction_pre_repair" in metadata
    assert "structured_prediction_post_repair" in metadata
    assert "calibrated_input_summary" in metadata
    assert "propagation_diagnostics" in metadata
    assert "prediction_artifact_paths" in metadata
    assert "prediction_source_contract" in metadata
    assert "prediction_source_contract_version" in metadata
    assert isinstance(metadata["prediction_summary_pre_repair"], dict)
    assert isinstance(metadata["structured_prediction_pre_repair"], dict)
    assert isinstance(metadata["structured_prediction_post_repair"], dict)
    assert isinstance(metadata["calibrated_input_summary"], dict)
    assert isinstance(metadata["propagation_diagnostics"], dict)
    assert isinstance(metadata["prediction_artifact_paths"], dict)
    assert isinstance(metadata["prediction_source_contract"], dict)
    assert metadata["prediction_source_contract_version"] == "v1"
    assert "structured_prediction_pre_repair" in metadata["prediction_artifact_paths"]
    rel_prediction_path = metadata["prediction_artifact_paths"]["structured_prediction_pre_repair"]
    prediction_artifact_path = outputs["evaluation_report_json"].parent / rel_prediction_path
    assert prediction_artifact_path.exists()

    scene_delta = report_payload["scene_level_delta_report"][0]
    assert "prediction_delta_v1_minus_v0" in scene_delta
    assert "prediction_summary_v0" in scene_delta
    assert "prediction_summary_v1" in scene_delta

    comparison_settings = {row["setting_name"] for row in report_payload["comparison_table"]}
    assert "calibration_v0" in comparison_settings
    assert "calibration_v1" in comparison_settings

    assert "first_research_result_json" in outputs
    assert "scene_level_delta_report_json" in outputs
    assert "stratified_v0_v1_summary_json" in outputs
    assert "trustworthy_comparison_status_json" in outputs
    assert "next_improvement_decision_json" in outputs
    assert "researcher_summary_json" in outputs
    assert "external_propagation_summary_json" in outputs
