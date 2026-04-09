import json
from pathlib import Path

import numpy as np

from self_calibrating_spatiallm.pipeline import run_multi_scene_pipeline


def test_multi_scene_smoke(tmp_path: Path) -> None:
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

    config_paths: list[Path] = []
    for index in range(2):
        config_path = tmp_path / f"scene_{index}.json"
        config_path.write_text(
            json.dumps(
                {
                    "scene_id": f"scene-{index}",
                    "file_path": str(point_path),
                    "source_type": "npy",
                    "expected_unit": "meter",
                    "generator_mode": "mock",
                    "normalize_scene": True,
                    "output_dir": str(tmp_path / "outputs" / f"scene-{index}"),
                }
            ),
            encoding="utf-8",
        )
        config_paths.append(config_path)

    output_dir = tmp_path / "multi"
    result = run_multi_scene_pipeline(config_paths=config_paths, output_dir=output_dir)
    manifest = json.loads(result["multi_scene_manifest"].read_text(encoding="utf-8"))

    assert manifest["num_scenes"] == 2
    assert manifest["success_count"] == 2
    assert manifest["failure_count"] == 0
