#!/usr/bin/env python3
"""Thin backend shim for SpatialLM inference command execution."""

from __future__ import annotations

import argparse
import json
import os
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
    parser.add_argument("--timeout-sec", type=int, default=120, help="Backend timeout")
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

    if output_path.exists():
        if _is_json_object(output_path):
            return 0
        print(f"backend output is not a valid JSON object: {output_path}", file=sys.stderr)
        return 3

    stdout_text = completed.stdout.strip()
    if not stdout_text:
        print(
            "real command succeeded but produced no output file and no stdout JSON",
            file=sys.stderr,
        )
        return 4

    try:
        payload = json.loads(stdout_text)
    except json.JSONDecodeError:
        print("real command stdout is not valid JSON", file=sys.stderr)
        return 5

    if not isinstance(payload, dict):
        print("real command stdout JSON is not an object", file=sys.stderr)
        return 6

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
