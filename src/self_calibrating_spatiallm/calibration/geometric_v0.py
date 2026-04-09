"""Geometry-backed calibration baseline (v0)."""

from __future__ import annotations

import math

try:
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover - exercised in minimal envs
    np = None

from self_calibrating_spatiallm.artifacts import CalibratedPointCloud, CalibrationResult, Point3D, PointCloudSample
from self_calibrating_spatiallm.calibration.base import Calibrator
from self_calibrating_spatiallm.geometry import array_to_points, points_to_array


class GeometricCalibratorV0(Calibrator):
    """Heuristic calibrator estimating up-axis and Manhattan-frame candidate."""

    name = "geometric_v0"

    def __init__(self, normalize_scene: bool = True) -> None:
        self.normalize_scene = normalize_scene

    def calibrate(self, sample: PointCloudSample) -> CalibratedPointCloud:
        if np is None:
            return self._calibrate_without_numpy(sample)
        return self._calibrate_with_numpy(sample)

    def _calibrate_with_numpy(self, sample: PointCloudSample) -> CalibratedPointCloud:
        xyz_list = points_to_array(sample.points)
        xyz = np.asarray(xyz_list, dtype=float)
        if xyz.shape[0] < 3:
            raise ValueError("Need at least 3 points for geometric calibration")

        center = np.mean(xyz, axis=0)
        centered = xyz - center

        covariance = np.cov(centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)

        raw_up = eigenvectors[:, 0]
        snapped_up, up_axis_name, up_confidence = _snap_up_axis(raw_up)

        horizontal_axis, horizontal_confidence = _estimate_horizontal_axis(
            centered=centered,
            up_vector=snapped_up,
            principal_direction=eigenvectors[:, 2],
        )
        yaw_degrees = float(math.degrees(math.atan2(horizontal_axis[1], horizontal_axis[0])))

        side_axis = np.cross(snapped_up, horizontal_axis)
        side_norm = np.linalg.norm(side_axis)
        if side_norm > 1e-8:
            side_axis = side_axis / side_norm
        else:
            side_axis = np.array([0.0, 1.0, 0.0], dtype=float)

        rotation_matrix = np.vstack([horizontal_axis, side_axis, snapped_up])

        if self.normalize_scene:
            transformed = (rotation_matrix @ centered.T).T
            origin_offset = Point3D(x=float(-center[0]), y=float(-center[1]), z=float(-center[2]))
        else:
            transformed = xyz.copy()
            origin_offset = Point3D(x=0.0, y=0.0, z=0.0)

        transformed_stats = _compute_transformed_stats_from_rows(transformed.tolist())

        calibration = CalibrationResult(
            sample_id=sample.sample_id,
            method=self.name,
            up_vector=Point3D(x=float(snapped_up[0]), y=float(snapped_up[1]), z=float(snapped_up[2])),
            horizontal_axis=Point3D(
                x=float(horizontal_axis[0]),
                y=float(horizontal_axis[1]),
                z=float(horizontal_axis[2]),
            ),
            origin_offset=origin_offset,
            metadata={
                "normalization_applied": self.normalize_scene,
                "rotation_matrix": rotation_matrix.tolist(),
                "diagnostics": {
                    "backend": "numpy",
                    "up_axis_candidate": up_axis_name,
                    "raw_up_vector": raw_up.tolist(),
                    "eigenvalues": eigenvalues.tolist(),
                    "yaw_degrees": yaw_degrees,
                    "confidence": {
                        "up_axis": float(up_confidence),
                        "horizontal_orientation": float(horizontal_confidence),
                    },
                },
                "transformed_point_cloud": transformed_stats,
            },
        )

        return CalibratedPointCloud(
            sample_id=sample.sample_id,
            points=array_to_points(transformed.tolist()),
            calibration=calibration,
            metadata={
                "frame": "canonical" if self.normalize_scene else sample.sensor_frame,
                "original_sensor_frame": sample.sensor_frame,
                "num_points": int(transformed.shape[0]),
                "transformed_point_cloud": transformed_stats,
                "room_bounds": sample.metadata.get("room_bounds"),
                "expected_unit": sample.metadata.get("expected_unit"),
                "inferred_scale_hint": sample.metadata.get("inferred_scale_hint"),
            },
        )

    def _calibrate_without_numpy(self, sample: PointCloudSample) -> CalibratedPointCloud:
        rows = points_to_array(sample.points)
        if len(rows) < 3:
            raise ValueError("Need at least 3 points for geometric calibration")

        xs = [row[0] for row in rows]
        ys = [row[1] for row in rows]
        zs = [row[2] for row in rows]

        center = [sum(xs) / len(xs), sum(ys) / len(ys), sum(zs) / len(zs)]
        ranges = [max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs)]

        up_idx = int(min(range(3), key=lambda idx: ranges[idx]))
        horiz_candidates = [idx for idx in range(3) if idx != up_idx]
        horiz_idx = int(max(horiz_candidates, key=lambda idx: ranges[idx]))
        side_idx = int(next(idx for idx in range(3) if idx not in {up_idx, horiz_idx}))

        up_vector = [0.0, 0.0, 0.0]
        up_vector[up_idx] = 1.0
        horizontal_axis = [0.0, 0.0, 0.0]
        horizontal_axis[horiz_idx] = 1.0

        transformed_rows: list[list[float]] = []
        for row in rows:
            centered_row = [row[idx] - center[idx] for idx in range(3)]
            if self.normalize_scene:
                transformed_rows.append(
                    [
                        centered_row[horiz_idx],
                        centered_row[side_idx],
                        centered_row[up_idx],
                    ]
                )
            else:
                transformed_rows.append(centered_row)

        transformed_stats = _compute_transformed_stats_from_rows(transformed_rows)
        max_range = max(ranges) if ranges else 1.0
        up_confidence = 1.0 - (ranges[up_idx] / max(max_range, 1e-8))
        horizontal_confidence = ranges[horiz_idx] / max((ranges[horiz_idx] + ranges[side_idx]), 1e-8)

        rotation_matrix = [
            _axis_row(horiz_idx),
            _axis_row(side_idx),
            _axis_row(up_idx),
        ]

        calibration = CalibrationResult(
            sample_id=sample.sample_id,
            method=self.name,
            up_vector=Point3D(x=up_vector[0], y=up_vector[1], z=up_vector[2]),
            horizontal_axis=Point3D(
                x=horizontal_axis[0],
                y=horizontal_axis[1],
                z=horizontal_axis[2],
            ),
            origin_offset=Point3D(x=-center[0], y=-center[1], z=-center[2]),
            metadata={
                "normalization_applied": self.normalize_scene,
                "rotation_matrix": rotation_matrix,
                "diagnostics": {
                    "backend": "fallback_no_numpy",
                    "up_axis_candidate": ["x", "y", "z"][up_idx],
                    "confidence": {
                        "up_axis": float(max(min(up_confidence, 1.0), 0.0)),
                        "horizontal_orientation": float(max(min(horizontal_confidence, 1.0), 0.0)),
                    },
                },
                "transformed_point_cloud": transformed_stats,
            },
        )

        return CalibratedPointCloud(
            sample_id=sample.sample_id,
            points=array_to_points(transformed_rows),
            calibration=calibration,
            metadata={
                "frame": "canonical" if self.normalize_scene else sample.sensor_frame,
                "original_sensor_frame": sample.sensor_frame,
                "num_points": len(transformed_rows),
                "transformed_point_cloud": transformed_stats,
                "room_bounds": sample.metadata.get("room_bounds"),
                "expected_unit": sample.metadata.get("expected_unit"),
                "inferred_scale_hint": sample.metadata.get("inferred_scale_hint"),
            },
        )


def _axis_row(index: int) -> list[float]:
    row = [0.0, 0.0, 0.0]
    row[index] = 1.0
    return row


def _snap_up_axis(raw_up: np.ndarray) -> tuple[np.ndarray, str, float]:
    axis_names = ["x", "y", "z"]
    abs_components = np.abs(raw_up)
    axis_idx = int(np.argmax(abs_components))
    sign = 1.0 if raw_up[axis_idx] >= 0 else -1.0

    snapped = np.zeros(3, dtype=float)
    snapped[axis_idx] = sign

    confidence = float(abs_components[axis_idx] / max(np.sum(abs_components), 1e-8))
    return snapped, axis_names[axis_idx], confidence


def _estimate_horizontal_axis(
    centered: np.ndarray,
    up_vector: np.ndarray,
    principal_direction: np.ndarray,
) -> tuple[np.ndarray, float]:
    projected = principal_direction - (np.dot(principal_direction, up_vector) * up_vector)
    norm = np.linalg.norm(projected)

    if norm < 1e-8:
        projected = _fallback_horizontal(up_vector)
        norm = np.linalg.norm(projected)

    horizontal = projected / max(norm, 1e-8)

    if centered.shape[0] >= 3:
        plane_basis = _horizontal_basis(up_vector)
        plane_coords = centered @ plane_basis.T
        cov2d = np.cov(plane_coords, rowvar=False)
        eigvals2d, _ = np.linalg.eigh(cov2d)
        eigvals2d = np.sort(eigvals2d)
        numerator = float(max(eigvals2d[-1] - eigvals2d[0], 0.0))
        denominator = float(max(eigvals2d[-1], 1e-8))
        confidence = numerator / denominator
    else:
        confidence = 0.0

    return horizontal, confidence


def _horizontal_basis(up_vector: np.ndarray) -> np.ndarray:
    fallback = _fallback_horizontal(up_vector)
    first = fallback / max(np.linalg.norm(fallback), 1e-8)
    second = np.cross(up_vector, first)
    second = second / max(np.linalg.norm(second), 1e-8)
    return np.vstack([first, second])


def _fallback_horizontal(up_vector: np.ndarray) -> np.ndarray:
    candidates = [
        np.array([1.0, 0.0, 0.0], dtype=float),
        np.array([0.0, 1.0, 0.0], dtype=float),
    ]
    for candidate in candidates:
        projection = candidate - (np.dot(candidate, up_vector) * up_vector)
        if np.linalg.norm(projection) > 1e-8:
            return projection
    return np.array([1.0, 0.0, 0.0], dtype=float)


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
        "ranges": [
            float(max_x - min_x),
            float(max_y - min_y),
            float(max_z - min_z),
        ],
    }
