"""Deterministic point-cloud perturbations for robustness-boundary studies."""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from self_calibrating_spatiallm.artifacts import Point3D, PointCloudSample


SUPPORTED_PERTURBATIONS = {
    "rotation_yaw",
    "tilt_up_axis",
    "manhattan_degradation",
    "clutter_injection",
    "structural_dropout",
    "density_sparsity",
}


@dataclass
class PerturbationResult:
    """Perturbation output and diagnostics."""

    sample: PointCloudSample
    metadata: dict[str, Any]


def apply_perturbation(
    *,
    base_sample: PointCloudSample,
    perturbation_type: str,
    severity: float,
    seed: int,
    params: dict[str, Any] | None = None,
) -> PerturbationResult:
    """Apply one deterministic perturbation to a sample."""
    if perturbation_type not in SUPPORTED_PERTURBATIONS:
        raise ValueError(f"Unsupported perturbation_type: {perturbation_type}")
    if severity < 0.0:
        raise ValueError("severity must be >= 0")

    params = dict(params or {})
    points = [(point.x, point.y, point.z) for point in base_sample.points]
    rng = random.Random(seed)
    if not points:
        sample = PointCloudSample(
            sample_id=base_sample.sample_id,
            points=[],
            sensor_frame=base_sample.sensor_frame,
            metadata={
                **dict(base_sample.metadata),
                "perturbation": {
                    "perturbation_type": perturbation_type,
                    "severity": float(severity),
                    "seed": int(seed),
                    "parameters": {"empty_input": True},
                },
            },
        )
        return PerturbationResult(
            sample=sample,
            metadata={
                "perturbation_type": perturbation_type,
                "severity": float(severity),
                "seed": int(seed),
                "num_points_before": 0,
                "num_points_after": 0,
                "parameters": {"empty_input": True},
            },
        )

    if perturbation_type == "rotation_yaw":
        transformed, details = _apply_rotation_yaw(points, severity=severity, params=params)
    elif perturbation_type == "tilt_up_axis":
        transformed, details = _apply_tilt(points, severity=severity, rng=rng, params=params)
    elif perturbation_type == "manhattan_degradation":
        transformed, details = _apply_manhattan_degradation(points, severity=severity, params=params)
    elif perturbation_type == "clutter_injection":
        transformed, details = _apply_clutter(points, severity=severity, rng=rng, params=params)
    elif perturbation_type == "structural_dropout":
        transformed, details = _apply_structural_dropout(
            points,
            severity=severity,
            rng=rng,
            params=params,
        )
    else:
        transformed, details = _apply_density_sparsity(points, severity=severity, rng=rng, params=params)

    if not transformed:
        transformed = list(points)
        details.setdefault("safety_override", "empty_result_prevented")

    sample = PointCloudSample(
        sample_id=base_sample.sample_id,
        points=[Point3D(x=x, y=y, z=z) for x, y, z in transformed],
        sensor_frame=base_sample.sensor_frame,
        metadata={
            **dict(base_sample.metadata),
            "perturbation": {
                "perturbation_type": perturbation_type,
                "severity": float(severity),
                "seed": int(seed),
                "parameters": details,
            },
        },
    )

    metadata = {
        "perturbation_type": perturbation_type,
        "severity": float(severity),
        "seed": int(seed),
        "num_points_before": len(points),
        "num_points_after": len(transformed),
        "parameters": details,
    }
    return PerturbationResult(sample=sample, metadata=metadata)


def derive_variant_seed(*, base_seed: int, components: list[str]) -> int:
    """Derive a stable integer seed from base seed + string components."""
    joined = "|".join([str(base_seed), *components]).encode("utf-8")
    digest = hashlib.sha256(joined).hexdigest()
    return int(digest[:16], 16)


def severity_bucket(value: float) -> str:
    """Map severity to low/mid/high buckets for boundary summaries."""
    if value < 0.34:
        return "low"
    if value < 0.67:
        return "mid"
    return "high"


def write_ascii_ply(path: Path, points: list[tuple[float, float, float]]) -> Path:
    """Write XYZ point cloud as ASCII PLY (deterministic textual format)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "ply",
        "format ascii 1.0",
        f"element vertex {len(points)}",
        "property float x",
        "property float y",
        "property float z",
        "end_header",
    ]
    body = [f"{x:.6f} {y:.6f} {z:.6f}" for x, y, z in points]
    path.write_text("\n".join([*header, *body]), encoding="utf-8")
    return path


def _apply_rotation_yaw(
    points: list[tuple[float, float, float]],
    *,
    severity: float,
    params: dict[str, Any],
) -> tuple[list[tuple[float, float, float]], dict[str, Any]]:
    max_yaw_deg = float(params.get("max_yaw_deg", 60.0))
    yaw_deg = max_yaw_deg * severity
    yaw_rad = math.radians(yaw_deg)
    cos_v = math.cos(yaw_rad)
    sin_v = math.sin(yaw_rad)
    rotated = [
        (
            x * cos_v - y * sin_v,
            x * sin_v + y * cos_v,
            z,
        )
        for x, y, z in points
    ]
    return rotated, {"yaw_deg": yaw_deg, "max_yaw_deg": max_yaw_deg}


def _apply_tilt(
    points: list[tuple[float, float, float]],
    *,
    severity: float,
    rng: random.Random,
    params: dict[str, Any],
) -> tuple[list[tuple[float, float, float]], dict[str, Any]]:
    max_tilt_deg = float(params.get("max_tilt_deg", 18.0))
    tilt_deg = max_tilt_deg * severity
    signed_tilt_deg = tilt_deg if rng.random() >= 0.5 else -tilt_deg
    tilt_rad = math.radians(signed_tilt_deg)
    cos_v = math.cos(tilt_rad)
    sin_v = math.sin(tilt_rad)
    tilted = [
        (
            x,
            y * cos_v - z * sin_v,
            y * sin_v + z * cos_v,
        )
        for x, y, z in points
    ]
    return tilted, {
        "tilt_axis": "x",
        "tilt_deg": signed_tilt_deg,
        "max_tilt_deg": max_tilt_deg,
    }


def _apply_manhattan_degradation(
    points: list[tuple[float, float, float]],
    *,
    severity: float,
    params: dict[str, Any],
) -> tuple[list[tuple[float, float, float]], dict[str, Any]]:
    max_shear = float(params.get("max_shear", 0.28))
    shear = max_shear * severity
    shear_xy = shear
    shear_yx = -0.45 * shear
    degraded = [
        (
            x + shear_xy * y,
            y + shear_yx * x,
            z,
        )
        for x, y, z in points
    ]
    return degraded, {
        "shear_xy": shear_xy,
        "shear_yx": shear_yx,
        "max_shear": max_shear,
    }


def _apply_clutter(
    points: list[tuple[float, float, float]],
    *,
    severity: float,
    rng: random.Random,
    params: dict[str, Any],
) -> tuple[list[tuple[float, float, float]], dict[str, Any]]:
    max_ratio = float(params.get("max_injection_ratio", 0.35))
    ratio = max_ratio * severity
    num_injected = max(0, int(round(len(points) * ratio)))
    if num_injected == 0:
        return list(points), {"injection_ratio": ratio, "num_injected_points": 0}

    mins, maxs = _bbox(points)
    padding_ratio = float(params.get("bbox_padding_ratio", 0.05))
    x_span = max(maxs[0] - mins[0], 1e-6)
    y_span = max(maxs[1] - mins[1], 1e-6)
    z_span = max(maxs[2] - mins[2], 1e-6)
    padded_mins = (
        mins[0] - x_span * padding_ratio,
        mins[1] - y_span * padding_ratio,
        mins[2] - z_span * padding_ratio,
    )
    padded_maxs = (
        maxs[0] + x_span * padding_ratio,
        maxs[1] + y_span * padding_ratio,
        maxs[2] + z_span * padding_ratio,
    )

    injected: list[tuple[float, float, float]] = []
    for _ in range(num_injected):
        injected.append(
            (
                rng.uniform(padded_mins[0], padded_maxs[0]),
                rng.uniform(padded_mins[1], padded_maxs[1]),
                rng.uniform(padded_mins[2], padded_maxs[2]),
            )
        )
    return [*points, *injected], {
        "injection_ratio": ratio,
        "max_injection_ratio": max_ratio,
        "num_injected_points": num_injected,
        "bbox_padding_ratio": padding_ratio,
    }


def _apply_structural_dropout(
    points: list[tuple[float, float, float]],
    *,
    severity: float,
    rng: random.Random,
    params: dict[str, Any],
) -> tuple[list[tuple[float, float, float]], dict[str, Any]]:
    max_dropout = float(params.get("max_dropout_ratio", 0.55))
    dropout_ratio = max_dropout * severity
    if dropout_ratio <= 0.0:
        return list(points), {"dropout_ratio": 0.0, "dropout_mode": "none"}

    mins, maxs = _bbox(points)
    x_span = max(maxs[0] - mins[0], 1e-6)
    y_span = max(maxs[1] - mins[1], 1e-6)
    z_span = max(maxs[2] - mins[2], 1e-6)
    mode = rng.choice(["wall_x_max", "wall_x_min", "floor", "ceiling", "corner"])

    kept: list[tuple[float, float, float]] = []
    x_threshold = x_span * dropout_ratio
    y_threshold = y_span * dropout_ratio
    z_threshold = z_span * dropout_ratio
    for x, y, z in points:
        remove = False
        if mode == "wall_x_max":
            remove = x >= (maxs[0] - x_threshold)
        elif mode == "wall_x_min":
            remove = x <= (mins[0] + x_threshold)
        elif mode == "floor":
            remove = z <= (mins[2] + z_threshold)
        elif mode == "ceiling":
            remove = z >= (maxs[2] - z_threshold)
        else:
            remove = (x >= (maxs[0] - x_threshold)) and (y >= (maxs[1] - y_threshold))
        if not remove:
            kept.append((x, y, z))

    min_keep_fraction = float(params.get("min_keep_fraction", 0.15))
    min_keep = max(16, int(len(points) * min_keep_fraction))
    if len(kept) < min_keep:
        # Keep deterministic random subset to avoid total structure collapse.
        indices = list(range(len(points)))
        rng.shuffle(indices)
        selected = sorted(indices[:min_keep])
        kept = [points[index] for index in selected]
        safety_mode = "min_keep_resample"
    else:
        safety_mode = "none"

    return kept, {
        "dropout_ratio": dropout_ratio,
        "max_dropout_ratio": max_dropout,
        "dropout_mode": mode,
        "num_removed_points": len(points) - len(kept),
        "safety_mode": safety_mode,
    }


def _apply_density_sparsity(
    points: list[tuple[float, float, float]],
    *,
    severity: float,
    rng: random.Random,
    params: dict[str, Any],
) -> tuple[list[tuple[float, float, float]], dict[str, Any]]:
    max_drop = float(params.get("max_drop_ratio", 0.82))
    drop_ratio = max_drop * severity
    keep_ratio = max(0.08, 1.0 - drop_ratio)
    keep_count = max(16, int(round(len(points) * keep_ratio)))
    indices = list(range(len(points)))
    rng.shuffle(indices)
    selected = sorted(indices[:keep_count])
    kept = [points[index] for index in selected]
    return kept, {
        "drop_ratio": drop_ratio,
        "max_drop_ratio": max_drop,
        "keep_ratio": keep_ratio,
        "num_kept_points": len(kept),
    }


def _bbox(points: list[tuple[float, float, float]]) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    zs = [point[2] for point in points]
    mins = (min(xs), min(ys), min(zs))
    maxs = (max(xs), max(ys), max(zs))
    return mins, maxs
