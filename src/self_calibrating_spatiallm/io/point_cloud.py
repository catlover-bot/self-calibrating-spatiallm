"""Point-cloud I/O utilities for real single-scene inputs."""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover - exercised in minimal envs
    np = None

from self_calibrating_spatiallm.artifacts import Point3D, PointCloudMetadata, PointCloudSample


@dataclass
class PointCloudLoadOptions:
    """Options controlling point-cloud loading and metadata generation."""

    scene_id: str
    source_type: str = "auto"
    metadata_path: Path | None = None
    expected_unit: str | None = None
    scale_hint: str | None = None


def load_point_cloud_sample(
    file_path: Path,
    options: PointCloudLoadOptions,
) -> tuple[PointCloudSample, PointCloudMetadata]:
    """Load point cloud from supported formats and derive basic scene metadata."""
    source_type = _resolve_source_type(file_path, options.source_type)

    if source_type in {"npy", "npz"}:
        xyz, rgb, loader_details = _load_numpy_point_cloud(file_path)
    elif source_type == "ply":
        xyz, rgb, loader_details = _load_ply(file_path)
    elif source_type == "pcd":
        xyz, rgb, loader_details = _load_ascii_pcd(file_path)
    else:
        raise ValueError(f"Unsupported point cloud source type: {source_type}")

    if not xyz:
        raise ValueError(f"Point cloud at {file_path} is empty")

    extra_metadata = _load_extra_metadata(options.metadata_path)
    stats = _compute_stats(xyz)
    inferred_scale = options.scale_hint or _infer_scale_hint(stats["coordinate_ranges"], options.expected_unit)

    point_cloud_metadata = PointCloudMetadata(
        sample_id=options.scene_id,
        source_path=str(file_path),
        source_type=source_type,
        num_points=len(xyz),
        bbox_min=Point3D.from_dict(stats["bbox_min"]),
        bbox_max=Point3D.from_dict(stats["bbox_max"]),
        centroid=Point3D.from_dict(stats["centroid"]),
        coordinate_ranges={
            "x": float(stats["coordinate_ranges"]["x"]),
            "y": float(stats["coordinate_ranges"]["y"]),
            "z": float(stats["coordinate_ranges"]["z"]),
        },
        has_rgb=rgb is not None,
        expected_unit=options.expected_unit,
        inferred_scale_hint=inferred_scale,
        metadata={
            "loader_details": loader_details,
            "extra_metadata": extra_metadata,
        },
    )

    sample_metadata: dict[str, Any] = {
        "source_path": str(file_path),
        "source_type": source_type,
        "has_rgb": rgb is not None,
        "num_points": len(xyz),
        "bbox_min": stats["bbox_min"],
        "bbox_max": stats["bbox_max"],
        "centroid": stats["centroid"],
        "coordinate_ranges": stats["coordinate_ranges"],
        "expected_unit": options.expected_unit,
        "inferred_scale_hint": inferred_scale,
        "loader_details": loader_details,
    }
    if extra_metadata:
        sample_metadata.update(extra_metadata)

    sample = PointCloudSample(
        sample_id=options.scene_id,
        points=[Point3D(x=float(row[0]), y=float(row[1]), z=float(row[2])) for row in xyz],
        sensor_frame=str(extra_metadata.get("sensor_frame", "sensor")) if extra_metadata else "sensor",
        metadata=sample_metadata,
    )

    return sample, point_cloud_metadata


def _resolve_source_type(file_path: Path, source_type: str) -> str:
    normalized = source_type.strip().lower()
    if normalized != "auto":
        return normalized

    suffix = file_path.suffix.lower().lstrip(".")
    if suffix in {"npy", "npz", "ply", "pcd"}:
        return suffix
    raise ValueError(f"Could not infer source type from extension: {file_path.suffix}")


def _load_numpy_point_cloud(path: Path) -> tuple[list[list[float]], list[list[float]] | None, dict[str, Any]]:
    if np is None:
        raise RuntimeError(
            "NumPy is required to load .npy/.npz point clouds. "
            "Install dependencies with `pip install -e .` or use .ply/.pcd input."
        )

    if path.suffix.lower() == ".npy":
        array = np.load(path)
        xyz, rgb = _split_xyz_rgb(array)
        return xyz, rgb, {"format": "npy", "array_shape": list(array.shape)}

    loaded = np.load(path)
    if not isinstance(loaded, np.lib.npyio.NpzFile):
        raise TypeError(f"Expected NPZ payload at {path}")

    keys = list(loaded.files)
    points_key = _select_points_key(keys)
    if points_key is None:
        raise ValueError(f"Could not find points array in NPZ file: {path}")

    points_array = loaded[points_key]
    xyz, rgb = _split_xyz_rgb(points_array)

    if rgb is None:
        for candidate in ("rgb", "colors", "colour", "colours"):
            if candidate in loaded.files:
                rgb_candidate = loaded[candidate]
                rgb_rows = _to_rows(rgb_candidate)
                if rgb_rows and len(rgb_rows[0]) >= 3:
                    rgb = [row[:3] for row in rgb_rows]
                    break

    shape = list(points_array.shape) if hasattr(points_array, "shape") else [len(xyz), 3]
    return xyz, rgb, {
        "format": "npz",
        "keys": keys,
        "points_key": points_key,
        "array_shape": shape,
    }


def _split_xyz_rgb(array: Any) -> tuple[list[list[float]], list[list[float]] | None]:
    rows = _to_rows(array)
    if not rows or len(rows[0]) < 3:
        raise ValueError("Expected (N, >=3) point array")

    xyz = [row[:3] for row in rows]
    rgb = [row[3:6] for row in rows] if len(rows[0]) >= 6 else None
    return xyz, rgb


def _to_rows(array: Any) -> list[list[float]]:
    if np is not None:
        arr = np.asarray(array)
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D point array, got shape {arr.shape}")
        return [[float(value) for value in row.tolist()] for row in arr]

    if not isinstance(array, list):
        raise ValueError("Array-like object must be a list when NumPy is unavailable")
    if not array or not isinstance(array[0], list):
        raise ValueError("Expected list-of-lists point representation")
    return [[float(value) for value in row] for row in array]


def _select_points_key(keys: list[str]) -> str | None:
    preferred = ["points", "xyz", "point_cloud", "arr_0"]
    for key in preferred:
        if key in keys:
            return key

    return keys[0] if keys else None


_PLY_STRUCT_FORMATS = {
    "char": "b",
    "int8": "b",
    "uchar": "B",
    "uint8": "B",
    "short": "h",
    "int16": "h",
    "ushort": "H",
    "uint16": "H",
    "int": "i",
    "int32": "i",
    "uint": "I",
    "uint32": "I",
    "float": "f",
    "float32": "f",
    "double": "d",
    "float64": "d",
}


def _load_ply(path: Path) -> tuple[list[list[float]], list[list[float]] | None, dict[str, Any]]:
    raw = path.read_bytes()
    header = _parse_ply_header(raw, path)
    data = raw[header["data_offset"] :]

    fmt = header["format"]
    if fmt == "ascii":
        return _load_ply_ascii_data(data, header)
    if fmt in {"binary_little_endian", "binary_big_endian"}:
        return _load_ply_binary_data(data, header)
    raise ValueError(f"Unsupported PLY format {fmt} in {path}")


def _parse_ply_header(raw: bytes, path: Path) -> dict[str, Any]:
    marker = b"end_header"
    marker_index = raw.find(marker)
    if marker_index < 0:
        raise ValueError(f"PLY header missing end_header: {path}")

    header_line_end = raw.find(b"\n", marker_index)
    if header_line_end < 0:
        header_line_end = marker_index + len(marker)
    else:
        header_line_end += 1

    header_text = raw[:header_line_end].decode("ascii", errors="replace")
    lines = [line.strip() for line in header_text.splitlines() if line.strip()]
    if not lines or lines[0] != "ply":
        raise ValueError(f"Not a PLY file: {path}")

    fmt = "ascii"
    elements: dict[str, dict[str, Any]] = {}
    current_element: str | None = None

    for line in lines[1:]:
        if line.startswith("comment ") or line.startswith("obj_info "):
            continue

        if line.startswith("format "):
            parts = line.split()
            if len(parts) < 2:
                raise ValueError(f"Malformed PLY format row: {line}")
            fmt = parts[1]
            continue

        if line.startswith("element "):
            parts = line.split()
            if len(parts) != 3:
                raise ValueError(f"Malformed PLY element row: {line}")
            element_name = parts[1]
            element_count = int(parts[2])
            elements[element_name] = {"count": element_count, "properties": []}
            current_element = element_name
            continue

        if line.startswith("property "):
            if current_element is None:
                raise ValueError(f"PLY property declared before element block: {line}")
            parts = line.split()
            if len(parts) >= 3 and parts[1] == "list":
                elements[current_element]["properties"].append(
                    {
                        "kind": "list",
                        "count_type": parts[2],
                        "value_type": parts[3] if len(parts) > 3 else "",
                        "name": parts[4] if len(parts) > 4 else "",
                    }
                )
            elif len(parts) == 3:
                elements[current_element]["properties"].append(
                    {"kind": "scalar", "type": parts[1], "name": parts[2]}
                )
            else:
                raise ValueError(f"Malformed PLY property row: {line}")
            continue

        if line == "end_header":
            break

    vertex_element = elements.get("vertex")
    if vertex_element is None:
        raise ValueError(f"PLY file has no vertex element: {path}")

    vertex_count = int(vertex_element["count"])
    if vertex_count <= 0:
        raise ValueError(f"PLY vertex count invalid: {vertex_count}")

    vertex_properties = list(vertex_element.get("properties", []))
    if not vertex_properties:
        raise ValueError("PLY vertex element has no properties")

    property_names: list[str] = []
    for prop in vertex_properties:
        if prop.get("kind") != "scalar":
            raise ValueError("PLY vertex list properties are not supported")
        property_names.append(str(prop.get("name", "")))

    return {
        "format": fmt,
        "vertex_count": vertex_count,
        "vertex_properties": vertex_properties,
        "property_names": property_names,
        "data_offset": header_line_end,
    }


def _load_ply_ascii_data(
    data: bytes,
    header: dict[str, Any],
) -> tuple[list[list[float]], list[list[float]] | None, dict[str, Any]]:
    vertex_count = int(header["vertex_count"])
    properties = list(header["property_names"])
    indices = _find_xyz_indices(properties)
    rgb_indices = _find_rgb_indices(properties)

    text = data.decode("utf-8")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    data_lines = lines[:vertex_count]
    if len(data_lines) < vertex_count:
        raise ValueError(f"PLY vertex section shorter than expected ({len(data_lines)} < {vertex_count})")

    xyz_rows: list[list[float]] = []
    rgb_rows: list[list[float]] = []

    for line in data_lines:
        parts = line.split()
        xyz_rows.append([float(parts[indices[0]]), float(parts[indices[1]]), float(parts[indices[2]])])

        if rgb_indices is not None:
            rgb_rows.append(
                [
                    float(parts[rgb_indices[0]]),
                    float(parts[rgb_indices[1]]),
                    float(parts[rgb_indices[2]]),
                ]
            )

    rgb = rgb_rows if rgb_rows else None
    return xyz_rows, rgb, {
        "format": "ply_ascii",
        "vertex_count": vertex_count,
        "properties": properties,
    }


def _load_ply_binary_data(
    data: bytes,
    header: dict[str, Any],
) -> tuple[list[list[float]], list[list[float]] | None, dict[str, Any]]:
    fmt = str(header["format"])
    vertex_count = int(header["vertex_count"])
    vertex_properties = list(header["vertex_properties"])
    properties = list(header["property_names"])
    indices = _find_xyz_indices(properties)
    rgb_indices = _find_rgb_indices(properties)

    endian = "<" if fmt == "binary_little_endian" else ">"
    scalar_formats: list[str] = []
    for prop in vertex_properties:
        type_name = str(prop["type"]).lower()
        struct_fmt = _PLY_STRUCT_FORMATS.get(type_name)
        if struct_fmt is None:
            raise ValueError(f"Unsupported PLY scalar property type: {type_name}")
        scalar_formats.append(struct_fmt)

    record_struct = struct.Struct(endian + "".join(scalar_formats))
    expected_bytes = vertex_count * record_struct.size
    if len(data) < expected_bytes:
        raise ValueError(
            f"PLY binary vertex section shorter than expected ({len(data)} < {expected_bytes})"
        )

    xyz_rows: list[list[float]] = []
    rgb_rows: list[list[float]] = []
    offset = 0
    for _ in range(vertex_count):
        values = record_struct.unpack_from(data, offset)
        offset += record_struct.size
        xyz_rows.append(
            [
                float(values[indices[0]]),
                float(values[indices[1]]),
                float(values[indices[2]]),
            ]
        )

        if rgb_indices is not None:
            rgb_rows.append(
                [
                    float(values[rgb_indices[0]]),
                    float(values[rgb_indices[1]]),
                    float(values[rgb_indices[2]]),
                ]
            )

    rgb = rgb_rows if rgb_rows else None
    return xyz_rows, rgb, {
        "format": f"ply_{fmt}",
        "vertex_count": vertex_count,
        "properties": properties,
    }


def _load_ascii_pcd(path: Path) -> tuple[list[list[float]], list[list[float]] | None, dict[str, Any]]:
    lines = path.read_text(encoding="utf-8").splitlines()

    header: dict[str, str] = {}
    data_start = -1
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        parts = stripped.split(maxsplit=1)
        key = parts[0].upper()
        value = parts[1] if len(parts) > 1 else ""
        header[key] = value

        if key == "DATA":
            data_start = idx + 1
            break

    if data_start < 0:
        raise ValueError(f"PCD header missing DATA row: {path}")
    if header.get("DATA", "").lower() != "ascii":
        raise ValueError("Only ASCII PCD is supported in this baseline loader")

    fields = header.get("FIELDS", "").split()
    if not fields:
        raise ValueError("PCD FIELDS header is required")

    xyz_indices = _find_xyz_indices(fields)
    rgb_indices = _find_rgb_indices(fields)
    packed_rgb_index = fields.index("rgb") if "rgb" in fields else None

    xyz_rows: list[list[float]] = []
    rgb_rows: list[list[float]] = []

    for line in lines[data_start:]:
        stripped = line.strip()
        if not stripped:
            continue

        values = stripped.split()
        xyz_rows.append(
            [
                float(values[xyz_indices[0]]),
                float(values[xyz_indices[1]]),
                float(values[xyz_indices[2]]),
            ]
        )

        if rgb_indices is not None:
            rgb_rows.append(
                [
                    float(values[rgb_indices[0]]),
                    float(values[rgb_indices[1]]),
                    float(values[rgb_indices[2]]),
                ]
            )
        elif packed_rgb_index is not None:
            r, g, b = _unpack_packed_rgb(float(values[packed_rgb_index]))
            rgb_rows.append([float(r), float(g), float(b)])

    rgb = rgb_rows if rgb_rows else None

    return xyz_rows, rgb, {
        "format": "pcd_ascii",
        "fields": fields,
        "point_count": len(xyz_rows),
    }


def _find_xyz_indices(fields: list[str]) -> tuple[int, int, int]:
    try:
        return fields.index("x"), fields.index("y"), fields.index("z")
    except ValueError as error:
        raise ValueError(f"Could not find x/y/z fields in {fields}") from error


def _find_rgb_indices(fields: list[str]) -> tuple[int, int, int] | None:
    for candidates in (("red", "green", "blue"), ("r", "g", "b")):
        if all(name in fields for name in candidates):
            return (fields.index(candidates[0]), fields.index(candidates[1]), fields.index(candidates[2]))
    return None


def _unpack_packed_rgb(value: float) -> tuple[int, int, int]:
    packed = struct.unpack("I", struct.pack("f", float(value)))[0]
    return (packed >> 16) & 0xFF, (packed >> 8) & 0xFF, packed & 0xFF


def _load_extra_metadata(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected metadata JSON object at {path}")
    return payload


def _compute_stats(xyz: list[list[float]]) -> dict[str, Any]:
    xs = [row[0] for row in xyz]
    ys = [row[1] for row in xyz]
    zs = [row[2] for row in xyz]

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    min_z, max_z = min(zs), max(zs)

    center_x = sum(xs) / len(xs)
    center_y = sum(ys) / len(ys)
    center_z = sum(zs) / len(zs)

    return {
        "bbox_min": {"x": min_x, "y": min_y, "z": min_z},
        "bbox_max": {"x": max_x, "y": max_y, "z": max_z},
        "centroid": {"x": center_x, "y": center_y, "z": center_z},
        "coordinate_ranges": {
            "x": max_x - min_x,
            "y": max_y - min_y,
            "z": max_z - min_z,
        },
    }


def _infer_scale_hint(ranges: dict[str, float], expected_unit: str | None) -> str:
    if expected_unit:
        return f"expected_unit:{expected_unit}"

    max_range = max(ranges.values()) if ranges else 0.0
    if max_range > 100.0:
        return "large_scale_mm_or_cm"
    if max_range > 20.0:
        return "room_scale_cm_or_large_m"
    if max_range > 2.0:
        return "meter_scale_likely"
    if max_range > 0.2:
        return "sub_meter_or_normalized"
    return "normalized_or_unknown"
