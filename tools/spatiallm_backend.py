#!/usr/bin/env python3
"""Thin backend shim for SpatialLM inference command execution."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SpatialLM backend shim")
    parser.add_argument("--input", required=True, help="Path to SpatialLM input payload")
    parser.add_argument(
        "--input-json",
        default=None,
        help="Optional explicit JSON manifest path for {input_json} placeholder",
    )
    parser.add_argument("--output", required=True, help="Path to backend output JSON")
    parser.add_argument("--scene", required=True, help="Scene id")
    parser.add_argument(
        "--real-command",
        default=None,
        help=(
            "Real SpatialLM command template. If omitted, SPATIALLM_REAL_COMMAND is used. "
            "Supported placeholders: {spatiallm_input}, {input_json}, {output_json}, {scene_id}"
        ),
    )
    parser.add_argument("--timeout-sec", type=int, default=300, help="Backend timeout")
    return parser


def _resolve_real_command(args: argparse.Namespace) -> str | None:
    if args.real_command:
        return str(args.real_command).strip()
    return os.environ.get("SPATIALLM_REAL_COMMAND", "").strip() or None


def _is_json_object(path: Path) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return isinstance(payload, dict)


def _try_parse_json_object_text(text: str) -> dict[str, Any] | None:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    if isinstance(payload, dict):
        return payload
    return None


def _load_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON payload at {path} is not an object")
    return payload


def _collect_points_xyz(payload: dict[str, Any]) -> list[tuple[float, float, float]]:
    raw_points = payload.get("points_xyz")
    if not isinstance(raw_points, list):
        raise ValueError("JSON payload does not contain list field `points_xyz`")

    points: list[tuple[float, float, float]] = []
    for index, item in enumerate(raw_points):
        if isinstance(item, dict):
            try:
                x = float(item.get("x", 0.0))
                y = float(item.get("y", 0.0))
                z = float(item.get("z", 0.0))
            except (TypeError, ValueError) as error:
                raise ValueError(f"points_xyz[{index}] has non-numeric xyz fields") from error
            points.append((x, y, z))
            continue

        if isinstance(item, list) and len(item) >= 3:
            try:
                x = float(item[0])
                y = float(item[1])
                z = float(item[2])
            except (TypeError, ValueError) as error:
                raise ValueError(f"points_xyz[{index}] list has non-numeric xyz values") from error
            points.append((x, y, z))
            continue

        raise ValueError(f"points_xyz[{index}] must be object or xyz list")

    if not points:
        raise ValueError("points_xyz is empty")
    return points


def _write_ascii_ply(points: list[tuple[float, float, float]], output_path: Path) -> None:
    header = [
        "ply",
        "format ascii 1.0",
        f"element vertex {len(points)}",
        "property float x",
        "property float y",
        "property float z",
        "end_header",
    ]
    rows = [f"{x:.9f} {y:.9f} {z:.9f}" for x, y, z in points]
    output_path.write_text("\n".join(header + rows) + "\n", encoding="utf-8")


_LAYOUT_LINE_RE = re.compile(r"^\s*([A-Za-z0-9_]+)\s*=\s*([A-Za-z][A-Za-z0-9_]*)\((.*)\)\s*$")
_NUMBER_RE = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


def _split_top_level_csv(text: str) -> list[str]:
    items: list[str] = []
    current: list[str] = []
    depth = 0
    quote: str | None = None

    for ch in text:
        if quote is not None:
            current.append(ch)
            if ch == quote:
                quote = None
            continue

        if ch in ("'", '"'):
            quote = ch
            current.append(ch)
            continue

        if ch in "([{":
            depth += 1
            current.append(ch)
            continue
        if ch in ")]}":
            depth = max(0, depth - 1)
            current.append(ch)
            continue

        if ch == "," and depth == 0:
            token = "".join(current).strip()
            if token:
                items.append(token)
            current = []
            continue

        current.append(ch)

    tail = "".join(current).strip()
    if tail:
        items.append(tail)
    return items


def _extract_vector3(raw_value: str) -> tuple[float, float, float] | None:
    numbers = [float(match) for match in _NUMBER_RE.findall(raw_value)]
    if len(numbers) < 3:
        return None
    return numbers[0], numbers[1], numbers[2]


def _parse_key_values(raw_args: str) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for token in _split_top_level_csv(raw_args):
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        key = key.strip().lower()
        value = value.strip()
        if key:
            mapping[key] = value
    return mapping


def _parse_float(raw_value: str | None) -> float | None:
    if raw_value is None:
        return None
    match = _NUMBER_RE.search(raw_value)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def _default_size_for_label(label: str) -> tuple[float, float, float]:
    normalized = label.lower()
    if normalized == "wall":
        return (3.0, 0.15, 2.6)
    if normalized == "door":
        return (0.9, 0.15, 2.0)
    if normalized == "window":
        return (1.2, 0.15, 1.2)
    if normalized in {"floor", "ceiling"}:
        return (4.0, 4.0, 0.1)
    return (0.5, 0.5, 0.5)


def _build_object_from_layout_line(
    object_id: str,
    class_name: str,
    raw_args: str,
) -> dict[str, Any]:
    label = class_name.lower()
    args = _parse_key_values(raw_args)

    default_sx, default_sy, default_sz = _default_size_for_label(label)

    center = None
    for key in ("position", "center", "centroid", "location"):
        if key in args:
            center = _extract_vector3(args[key])
            if center is not None:
                break

    x = _parse_float(args.get("x")) if "x" in args else None
    y = _parse_float(args.get("y")) if "y" in args else None
    z = _parse_float(args.get("z")) if "z" in args else None

    if center is None:
        center = (0.0, 0.0, 0.0)
    cx, cy, cz = center
    if x is not None:
        cx = x
    if y is not None:
        cy = y
    if z is not None:
        cz = z

    size_vector = None
    for key in ("size", "dimensions", "extent", "bbox"):
        if key in args:
            size_vector = _extract_vector3(args[key])
            if size_vector is not None:
                break

    sx = _parse_float(args.get("size_x")) or _parse_float(args.get("dx")) or _parse_float(args.get("width"))
    sy = _parse_float(args.get("size_y")) or _parse_float(args.get("dy")) or _parse_float(args.get("depth"))
    sz = (
        _parse_float(args.get("size_z"))
        or _parse_float(args.get("dz"))
        or _parse_float(args.get("height"))
        or _parse_float(args.get("thickness"))
    )

    if size_vector is not None:
        vx, vy, vz = size_vector
        sx = sx if sx is not None else vx
        sy = sy if sy is not None else vy
        sz = sz if sz is not None else vz

    if sx is None:
        sx = default_sx
    if sy is None:
        sy = default_sy
    if sz is None:
        sz = default_sz

    confidence = _parse_float(args.get("confidence")) or _parse_float(args.get("score")) or 0.5

    return {
        "object_id": object_id,
        "label": label,
        "position": {"x": cx, "y": cy, "z": cz},
        "size": {"x": sx, "y": sy, "z": sz},
        "confidence": confidence,
        "attributes": {
            "source": "spatiallm_layout_text",
            "class_name": class_name,
            "raw_args": raw_args,
            "parsed_args": args,
        },
    }


def _parse_layout_text_to_scene_prediction(text: str, scene_id: str) -> dict[str, Any] | None:
    objects: list[dict[str, Any]] = []
    parsed_lines: list[str] = []
    unparsed_lines: list[str] = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = _LAYOUT_LINE_RE.match(line)
        if not match:
            if "=" in line and "(" in line and ")" in line:
                unparsed_lines.append(line)
            continue

        object_id = match.group(1)
        class_name = match.group(2)
        raw_args = match.group(3)
        objects.append(_build_object_from_layout_line(object_id=object_id, class_name=class_name, raw_args=raw_args))
        parsed_lines.append(line)

    if not objects:
        return None

    return {
        "scene_prediction": {
            "sample_id": scene_id,
            "generator_name": "spatiallm_external_backend_text_parser",
            "objects": objects,
            "relations": [],
            "metadata": {
                "parser": "spatiallm_layout_text_v1",
                "parsed_line_count": len(parsed_lines),
                "unparsed_line_count": len(unparsed_lines),
                "raw_text_preview": text[:4000],
            },
        }
    }


def _resolve_input_json_path(input_path: Path, explicit_input_json: str | None) -> Path:
    if explicit_input_json:
        candidate = Path(explicit_input_json)
        if candidate.exists():
            return candidate
        print(
            f"warning: --input-json path does not exist, falling back to inferred path: {candidate}",
            file=sys.stderr,
        )

    if input_path.suffix.lower() == ".json":
        return input_path

    candidate = input_path.with_suffix(".json")
    if candidate.exists():
        return candidate

    return input_path


def _resolve_spatiallm_input_path(input_path: Path, tmp_dir: Path) -> tuple[Path, str | None]:
    if input_path.suffix.lower() != ".json":
        return input_path, None

    payload = _load_json_object(input_path)
    points = _collect_points_xyz(payload)
    ply_path = tmp_dir / "spatiallm_point_cloud.ply"
    _write_ascii_ply(points, ply_path)
    return ply_path, f"converted_json_manifest_to_ply(num_points={len(points)})"


def main() -> int:
    args = _build_parser().parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    scene_id = str(args.scene)

    if not input_path.exists():
        print(f"backend input file not found: {input_path}", file=sys.stderr)
        return 2

    real_template = _resolve_real_command(args)
    if not real_template:
        print(
            "Real SpatialLM command is not configured. "
            "Set SPATIALLM_REAL_COMMAND or pass --real-command.",
            file=sys.stderr,
        )
        return 2

    input_json_path = _resolve_input_json_path(input_path=input_path, explicit_input_json=args.input_json)

    try:
        with tempfile.TemporaryDirectory(prefix="scslm_backend_") as tmp_dir_name:
            tmp_dir = Path(tmp_dir_name)
            spatiallm_input_path, conversion_note = _resolve_spatiallm_input_path(
                input_path=input_path,
                tmp_dir=tmp_dir,
            )
            if conversion_note:
                print(
                    f"info: {conversion_note}; spatiallm_input={spatiallm_input_path}; input_json={input_json_path}",
                    file=sys.stderr,
                )

            command = [
                token.format(
                    spatiallm_input=str(spatiallm_input_path),
                    input_json=str(input_json_path),
                    output_json=str(output_path),
                    scene_id=scene_id,
                )
                for token in shlex.split(real_template)
            ]

            try:
                completed = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    timeout=int(args.timeout_sec),
                    check=False,
                )
            except FileNotFoundError as error:
                print(f"real command not found: {error}", file=sys.stderr)
                return 127
            except subprocess.TimeoutExpired:
                print(f"real command timed out after {args.timeout_sec}s", file=sys.stderr)
                return 124
    except Exception as error:
        print(
            "failed to prepare SpatialLM point cloud input: "
            f"{error}. Expected JSON with `points_xyz` or direct point-cloud file (.ply/.pcd/.npy/.npz).",
            file=sys.stderr,
        )
        return 2

    if completed.stdout:
        sys.stdout.write(completed.stdout)
    if completed.stderr:
        sys.stderr.write(completed.stderr)

    if completed.returncode != 0:
        return int(completed.returncode)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        if _is_json_object(output_path):
            return 0
        output_text = output_path.read_text(encoding="utf-8")
        parsed_from_text = _parse_layout_text_to_scene_prediction(text=output_text, scene_id=scene_id)
        if parsed_from_text is not None:
            output_path.write_text(json.dumps(parsed_from_text, indent=2, sort_keys=True), encoding="utf-8")
            print(
                "info: converted non-JSON SpatialLM output file to scene_prediction JSON",
                file=sys.stderr,
            )
            return 0
        print(
            f"backend output exists but is not a valid JSON object and text parsing failed: {output_path}",
            file=sys.stderr,
        )
        return 3

    stdout_text = completed.stdout.strip()
    if not stdout_text:
        print(
            "real command succeeded but produced no output file and no stdout JSON",
            file=sys.stderr,
        )
        return 4

    payload = _try_parse_json_object_text(stdout_text)
    if payload is None:
        parsed_from_text = _parse_layout_text_to_scene_prediction(text=stdout_text, scene_id=scene_id)
        if parsed_from_text is None:
            print("real command stdout is not valid JSON and text parsing failed", file=sys.stderr)
            return 5
        payload = parsed_from_text
        print(
            "info: converted SpatialLM stdout layout text to scene_prediction JSON",
            file=sys.stderr,
        )

    if not isinstance(payload, dict):
        print("real command stdout JSON is not an object", file=sys.stderr)
        return 6

    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
