"""Run qualitative multi-scene pipeline over a small list of scene configs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from self_calibrating_spatiallm.pipeline import run_multi_scene_pipeline


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run multi-scene qualitative pipeline")
    parser.add_argument(
        "--config-list",
        type=Path,
        default=Path("configs/samples/multi_scene_small.json"),
        help="JSON path containing config_paths list",
    )
    parser.add_argument(
        "--sample-config",
        type=Path,
        action="append",
        default=[],
        help="Additional sample config path (can be repeated)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/runs/multi_scene_real"),
        help="Output directory for multi-scene artifacts",
    )
    return parser


def _load_config_paths(config_list: Path) -> list[Path]:
    payload = json.loads(config_list.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        values = payload
    elif isinstance(payload, dict):
        values = payload.get("config_paths", [])
    else:
        raise TypeError(f"Unsupported config list payload at {config_list}")

    if not isinstance(values, list):
        raise TypeError("config_paths must be a list")

    base = config_list.parent
    paths: list[Path] = []
    for value in values:
        if not isinstance(value, str):
            raise TypeError(f"Config path must be string, got {type(value)}")
        candidate = Path(value)
        paths.append(candidate if candidate.is_absolute() else (base / candidate).resolve())
    return paths


def main() -> int:
    args = _build_parser().parse_args()

    config_paths = _load_config_paths(args.config_list)
    config_paths.extend(path.resolve() for path in args.sample_config)

    results = run_multi_scene_pipeline(config_paths=config_paths, output_dir=args.output_dir.resolve())
    print(json.dumps({key: str(path) for key, path in results.items()}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
