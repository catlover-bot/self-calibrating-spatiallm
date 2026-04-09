from pathlib import Path

import numpy as np

from self_calibrating_spatiallm.io import PointCloudLoadOptions, load_point_cloud_sample


def test_loader_supports_npy_npz_ply_pcd(tmp_path: Path) -> None:
    points = np.array(
        [
            [0.0, 0.0, 0.0, 255.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 255.0, 0.0],
            [0.0, 1.0, 0.5, 0.0, 0.0, 255.0],
        ],
        dtype=float,
    )

    npy_path = tmp_path / "scene.npy"
    np.save(npy_path, points)

    npz_path = tmp_path / "scene.npz"
    np.savez(npz_path, points=points)

    ply_path = tmp_path / "scene.ply"
    ply_path.write_text(
        "\n".join(
            [
                "ply",
                "format ascii 1.0",
                "element vertex 3",
                "property float x",
                "property float y",
                "property float z",
                "property uchar red",
                "property uchar green",
                "property uchar blue",
                "end_header",
                "0.0 0.0 0.0 255 0 0",
                "1.0 0.0 0.0 0 255 0",
                "0.0 1.0 0.5 0 0 255",
            ]
        ),
        encoding="utf-8",
    )

    pcd_path = tmp_path / "scene.pcd"
    pcd_path.write_text(
        "\n".join(
            [
                "# .PCD v0.7 - Point Cloud Data file format",
                "VERSION 0.7",
                "FIELDS x y z r g b",
                "SIZE 4 4 4 4 4 4",
                "TYPE F F F F F F",
                "COUNT 1 1 1 1 1 1",
                "WIDTH 3",
                "HEIGHT 1",
                "POINTS 3",
                "DATA ascii",
                "0.0 0.0 0.0 255 0 0",
                "1.0 0.0 0.0 0 255 0",
                "0.0 1.0 0.5 0 0 255",
            ]
        ),
        encoding="utf-8",
    )

    for source_type, path in [
        ("npy", npy_path),
        ("npz", npz_path),
        ("ply", ply_path),
        ("pcd", pcd_path),
    ]:
        sample, metadata = load_point_cloud_sample(
            path,
            PointCloudLoadOptions(scene_id=f"scene-{source_type}", source_type=source_type),
        )
        assert sample.num_points == 3
        assert metadata.num_points == 3
        assert metadata.coordinate_ranges["x"] == 1.0
        assert metadata.has_rgb is True
