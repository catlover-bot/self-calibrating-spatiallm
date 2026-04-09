"""Run lightweight evaluation pack across 5-10 scene configs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from self_calibrating_spatiallm.evaluation import run_evaluation_pack


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run evaluation pack")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("configs/eval_pack/small_eval_pack.json"),
        help="Path to evaluation pack manifest JSON",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/eval_pack/latest"),
        help="Output directory for evaluation reports",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    outputs = run_evaluation_pack(manifest_path=args.manifest.resolve(), output_dir=args.output_dir.resolve())
    print(json.dumps({key: str(path) for key, path in outputs.items()}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
