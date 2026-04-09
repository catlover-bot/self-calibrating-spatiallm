#!/usr/bin/env python3
"""Minimal wrapper for invoking external SpatialLM inference commands."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="External SpatialLM wrapper")
    parser.add_argument("--input", required=True, help="Path to exported SpatialLM input payload")
    parser.add_argument("--output", required=True, help="Path to output JSON prediction file")
    parser.add_argument("--scene", required=True, help="Scene id")
    parser.add_argument(
        "--backend-command",
        default=None,
        help=(
            "Backend command template. If omitted, SPATIALLM_BACKEND_COMMAND is used. "
            "Supported placeholders: {spatiallm_input}, {input_json}, {output_json}, {scene_id}"
        ),
    )
    parser.add_argument(
        "--timeout-sec",
        type=int,
        default=120,
        help="Timeout for backend command execution",
    )
    return parser


def _resolve_backend_command(args: argparse.Namespace) -> str | None:
    if args.backend_command:
        return str(args.backend_command).strip()
    return os.environ.get("SPATIALLM_BACKEND_COMMAND", "").strip() or None


def _validate_json_output(path: Path) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return isinstance(payload, dict)


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    scene_id = str(args.scene)

    if not input_path.exists():
        print(f"input file not found: {input_path}", file=sys.stderr)
        return 2

    backend_template = _resolve_backend_command(args)
    if not backend_template:
        print(
            "SPATIALLM backend is not configured. "
            "Set SPATIALLM_BACKEND_COMMAND or pass --backend-command.",
            file=sys.stderr,
        )
        return 2

    command = [
        token.format(
            spatiallm_input=str(input_path),
            input_json=str(input_path),
            output_json=str(output_path),
            scene_id=scene_id,
        )
        for token in shlex.split(backend_template)
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
        print(f"backend command not found: {error}", file=sys.stderr)
        return 127
    except subprocess.TimeoutExpired:
        print(f"backend command timed out after {args.timeout_sec}s", file=sys.stderr)
        return 124

    if completed.stdout:
        sys.stdout.write(completed.stdout)
    if completed.stderr:
        sys.stderr.write(completed.stderr)

    if completed.returncode != 0:
        return int(completed.returncode)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        if _validate_json_output(output_path):
            return 0
        print(f"output exists but is not a valid JSON object: {output_path}", file=sys.stderr)
        return 3

    stdout_text = completed.stdout.strip()
    if not stdout_text:
        print(
            "backend command succeeded but produced neither output file nor stdout JSON",
            file=sys.stderr,
        )
        return 4

    try:
        payload = json.loads(stdout_text)
    except json.JSONDecodeError:
        print("backend stdout is not valid JSON", file=sys.stderr)
        return 5

    if not isinstance(payload, dict):
        print("backend stdout JSON is not an object", file=sys.stderr)
        return 6

    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
