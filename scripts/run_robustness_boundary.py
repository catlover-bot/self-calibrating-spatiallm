"""Run perturbation-driven robustness-boundary experiments."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from self_calibrating_spatiallm.robustness import run_robustness_boundary_experiment


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run robustness-boundary experiment workflow")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/robustness_boundary.default.json"),
        help="Path to robustness-boundary config JSON",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override output directory (defaults to config.output_dir)",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    outputs = run_robustness_boundary_experiment(
        config_path=args.config.resolve(),
        output_dir=(args.output_dir.resolve() if args.output_dir else None),
    )
    print(json.dumps({name: str(path) for name, path in outputs.items()}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

