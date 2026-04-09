"""Run the single-scene pipeline and persist all intermediate artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from self_calibrating_spatiallm.generation import ExternalGeneratorError
from self_calibrating_spatiallm.pipeline import SceneInputConfig, run_single_scene_pipeline


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run single-scene self-calibrating spatial pipeline")
    parser.add_argument(
        "--sample-config",
        type=Path,
        default=Path("configs/samples/real_scene_config.json"),
        help="Path to scene input config JSON",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override output directory for generated artifacts",
    )
    parser.add_argument(
        "--calibration-mode",
        type=str,
        choices=["none", "v0", "v1"],
        default=None,
        help="Override calibration mode from config",
    )
    parser.add_argument(
        "--generator-mode",
        type=str,
        choices=["mock", "external"],
        default=None,
        help="Override generator mode from config",
    )
    parser.add_argument(
        "--spatiallm-command",
        type=str,
        default=None,
        help="Override external SpatialLM command (used when generator_mode=external)",
    )
    parser.add_argument(
        "--spatiallm-command-env-var",
        type=str,
        default=None,
        help="Override environment variable used to resolve external SpatialLM command",
    )
    parser.add_argument(
        "--spatiallm-export-format",
        type=str,
        choices=["json"],
        default=None,
        help="Override export format for SpatialLM input payload",
    )
    parser.add_argument(
        "--compare-with-external",
        action="store_true",
        help="Enable external generator comparison in ablation report when configured",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()

    config = SceneInputConfig.load_json(args.sample_config)
    if args.calibration_mode is not None:
        config.calibration_mode = args.calibration_mode
    if args.generator_mode is not None:
        config.generator_mode = args.generator_mode
    if args.spatiallm_command is not None:
        config.spatiallm_command = args.spatiallm_command
    if args.spatiallm_command_env_var is not None:
        config.spatiallm_command_env_var = args.spatiallm_command_env_var
    if args.spatiallm_export_format is not None:
        config.spatiallm_export_format = args.spatiallm_export_format
    if args.compare_with_external:
        config.compare_with_external_generator = True

    try:
        paths = run_single_scene_pipeline(config=config, output_dir=args.output_dir)
        serializable = {key: str(path) for key, path in paths.items()}
        print(json.dumps(serializable, indent=2, sort_keys=True))
        return 0
    except ExternalGeneratorError as error:
        payload = {
            "error": str(error),
            "details": error.details,
        }
        print(json.dumps(payload, indent=2, sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
