import json
from pathlib import Path

import numpy as np

from self_calibrating_spatiallm.pipeline import run_single_scene_pipeline_from_config_path


def test_run_manifest_generation(tmp_path: Path) -> None:
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

    config_path = tmp_path / "scene_config.json"
    config_path.write_text(
        json.dumps(
            {
                "scene_id": "manifest-scene",
                "file_path": str(point_path),
                "source_type": "npy",
                "expected_unit": "meter",
                "generator_mode": "mock",
                "normalize_scene": True,
                "output_dir": str(tmp_path / "outputs"),
            }
        ),
        encoding="utf-8",
    )

    paths = run_single_scene_pipeline_from_config_path(config_path)
    payload = json.loads(paths["run_manifest"].read_text(encoding="utf-8"))

    assert payload["scene_id"] == "manifest-scene"
    assert payload["status"] == "success"
    assert payload["generator_mode"] == "mock"
    assert payload["calibration_mode"] == "plane_aware_v1"
    assert payload["repair_mode"] == "simple_rule_repairer"
    assert "calibration_execution" in payload
    execution = payload["calibration_execution"]
    assert isinstance(execution, dict)
    assert "plane_aware_logic_ran" in execution
    assert "true_v1_execution" in execution
    assert "fallback_used" in execution
    assert "fallback_reason" in execution
    assert "candidate_plane_count" in execution
    assert "weak_scale_reasoning_active" in execution
    assert isinstance(execution["candidate_plane_count"], int)
    assert isinstance(execution["plane_aware_logic_ran"], bool)
    assert isinstance(execution["true_v1_execution"], bool)
    assert isinstance(execution["fallback_used"], bool)
    if execution["true_v1_execution"]:
        assert execution["plane_aware_logic_ran"] is True
    else:
        assert execution["fallback_used"] is True
    assert payload["stages"]["loading"]["status"] == "success"
    assert payload["stages"]["generation"]["status"] == "success"
    assert "qualitative_report" in payload["artifacts"]
