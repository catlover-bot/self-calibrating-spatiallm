"""Report runtime environment capabilities for calibration/generation paths."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from self_calibrating_spatiallm.environment import (
    collect_environment_report,
    render_environment_report_text,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check environment capability support")
    parser.add_argument(
        "--format",
        type=str,
        choices=["text", "json"],
        default="text",
        help="Output format",
    )
    parser.add_argument(
        "--spatiallm-command-env-var",
        type=str,
        default="SCSLM_SPATIALLM_COMMAND",
        help="Environment variable used to resolve external SpatialLM command",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    report = collect_environment_report(
        spatiallm_command_env_var=args.spatiallm_command_env_var,
    )
    if args.format == "json":
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(render_environment_report_text(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

