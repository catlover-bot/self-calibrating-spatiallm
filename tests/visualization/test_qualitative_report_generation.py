import json
from pathlib import Path

import numpy as np

from self_calibrating_spatiallm.pipeline import run_single_scene_pipeline_from_config_path


def test_qualitative_report_generation(tmp_path: Path) -> None:
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
                "scene_id": "report-scene",
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
    report_text = paths["qualitative_report"].read_text(encoding="utf-8")

    assert "# Qualitative Run Report: report-scene" in report_text
    assert "## Point Cloud Summary" in report_text
    assert "## Calibration Diagnostics" in report_text
    assert "## Ablation Summary" in report_text
