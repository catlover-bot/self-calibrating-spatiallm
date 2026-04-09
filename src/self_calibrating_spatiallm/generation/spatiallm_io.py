"""SpatialLM input conversion and export utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from self_calibrating_spatiallm.artifacts import CalibratedPointCloud


def build_spatiallm_input_payload(calibrated: CalibratedPointCloud) -> dict[str, Any]:
    """Build an explicit payload contract for external SpatialLM adapters."""
    calibration = calibrated.calibration
    frame = str(calibrated.metadata.get("frame", "unknown"))
    normalization_applied = bool(calibration.metadata.get("normalization_applied", False))

    return {
        "scene_id": calibrated.sample_id,
        "contract_version": "spatiallm_input_v1",
        "axis_convention": {
            "coordinate_frame": frame,
            "right_handed": True,
            "x_axis": "dominant horizontal direction",
            "y_axis": "horizontal side direction",
            "z_axis": "up",
            "up_vector": calibration.up_vector.to_dict(),
            "horizontal_axis": calibration.horizontal_axis.to_dict(),
        },
        "scale_assumptions": {
            "expected_unit": calibrated.metadata.get("expected_unit"),
            "inferred_scale_hint": calibrated.metadata.get("inferred_scale_hint"),
        },
        "normalization": {
            "applied": normalization_applied,
            "origin_offset": calibration.origin_offset.to_dict(),
            "rotation_matrix": calibration.metadata.get("rotation_matrix"),
        },
        "num_points": calibrated.num_points,
        "points_xyz": [point.to_dict() for point in calibrated.points],
        "metadata": {
            "room_bounds": calibrated.metadata.get("room_bounds"),
            "original_sensor_frame": calibrated.metadata.get("original_sensor_frame"),
        },
    }


def export_spatiallm_input(
    calibrated: CalibratedPointCloud,
    output_path: Path,
    export_format: str = "json",
) -> dict[str, Any]:
    """Export calibrated point cloud to the configured SpatialLM input format."""
    normalized_format = export_format.strip().lower()
    if normalized_format != "json":
        raise ValueError(f"Unsupported SpatialLM export format: {export_format}")

    payload = build_spatiallm_input_payload(calibrated)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    return {
        "format": normalized_format,
        "path": str(output_path),
        "num_points": calibrated.num_points,
        "payload": payload,
    }
