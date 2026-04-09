import json
from pathlib import Path

import numpy as np

from self_calibrating_spatiallm.artifacts import AblationReport
from self_calibrating_spatiallm.pipeline import run_single_scene_pipeline_from_config_path


def test_single_scene_ablation_smoke(tmp_path: Path) -> None:
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

    metadata_path = tmp_path / "scene_meta.json"
    metadata_path.write_text(
        json.dumps(
            {
                "sensor_frame": "test_sensor",
                "room_bounds": {"min": [-2.0, -2.0, -0.2], "max": [2.0, 2.0, 2.5]},
            }
        ),
        encoding="utf-8",
    )

    config_path = tmp_path / "scene_config.json"
    config_path.write_text(
        json.dumps(
            {
                "scene_id": "scene-ablation",
                "file_path": str(point_path),
                "source_type": "npy",
                "metadata_path": str(metadata_path),
                "expected_unit": "meter",
                "generator_mode": "mock",
                "normalize_scene": True,
                "output_dir": str(tmp_path / "outputs"),
            }
        ),
        encoding="utf-8",
    )

    paths = run_single_scene_pipeline_from_config_path(config_path)
    ablation = AblationReport.load_json(paths["ablation_report"])

    names = {setting.setting_name for setting in ablation.settings}
    assert names == {
        "no_calibration",
        "calibration_v0",
        "calibration_v1",
        "calibration_v1_plus_repair",
    }
