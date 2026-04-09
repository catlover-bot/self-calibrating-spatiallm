"""Command-line interface for pipeline-centric workflows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from self_calibrating_spatiallm.environment import (
    collect_environment_report,
    render_environment_report_text,
)
from self_calibrating_spatiallm.generation import ExternalGeneratorError
from self_calibrating_spatiallm.pipeline import (
    SceneInputConfig,
    run_multi_scene_pipeline,
    run_single_scene_pipeline,
)
from self_calibrating_spatiallm.evaluation import run_evaluation_pack


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="self-calibrating-spatiallm CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run-pipeline", help="Run single-scene end-to-end pipeline")
    run_parser.add_argument(
        "--sample-config",
        type=Path,
        default=Path("configs/samples/real_scene_config.json"),
        help="Path to scene input config JSON",
    )
    run_parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override output directory for saved artifacts",
    )
    run_parser.add_argument(
        "--calibration-mode",
        type=str,
        choices=["none", "v0", "v1"],
        default=None,
        help="Override calibration mode in sample config",
    )
    run_parser.add_argument(
        "--generator-mode",
        type=str,
        choices=["mock", "external"],
        default=None,
        help="Override generator mode in sample config",
    )
    run_parser.add_argument(
        "--spatiallm-command",
        type=str,
        default=None,
        help="Override external SpatialLM command",
    )

    multi_parser = subparsers.add_parser("run-multi", help="Run qualitative multi-scene pipeline")
    multi_parser.add_argument(
        "--config-list",
        type=Path,
        default=Path("configs/samples/multi_scene_small.json"),
        help="JSON file containing config_paths list",
    )
    multi_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/runs/multi_scene_real"),
        help="Output directory for multi-scene artifacts",
    )

    eval_parser = subparsers.add_parser("run-eval-pack", help="Run small evaluation pack")
    eval_parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("configs/eval_pack/small_eval_pack.json"),
        help="Path to evaluation pack manifest JSON",
    )
    eval_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/eval_pack/latest"),
        help="Output directory for evaluation outputs",
    )

    env_parser = subparsers.add_parser("check-env", help="Report runtime environment capabilities")
    env_parser.add_argument(
        "--format",
        type=str,
        choices=["text", "json"],
        default="text",
        help="Output format",
    )
    env_parser.add_argument(
        "--spatiallm-command-env-var",
        type=str,
        default="SCSLM_SPATIALLM_COMMAND",
        help="Environment variable used to resolve external SpatialLM command",
    )

    return parser


def _load_config_paths(config_list_path: Path) -> list[Path]:
    payload = json.loads(config_list_path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        entries = payload
    elif isinstance(payload, dict):
        entries = payload.get("config_paths", [])
    else:
        raise TypeError(f"Unsupported config list payload at {config_list_path}")
    if not isinstance(entries, list):
        raise TypeError("config_paths must be a list")

    base = config_list_path.parent
    paths: list[Path] = []
    for value in entries:
        if not isinstance(value, str):
            raise TypeError(f"Config path must be a string, got: {type(value)}")
        candidate = Path(value)
        paths.append(candidate if candidate.is_absolute() else (base / candidate).resolve())
    return paths


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.command == "run-pipeline":
        config = SceneInputConfig.load_json(args.sample_config)
        if args.calibration_mode is not None:
            config.calibration_mode = args.calibration_mode
        if args.generator_mode is not None:
            config.generator_mode = args.generator_mode
        if args.spatiallm_command is not None:
            config.spatiallm_command = args.spatiallm_command

        try:
            artifact_paths = run_single_scene_pipeline(config=config, output_dir=args.output_dir)
            payload = {name: str(path) for name, path in artifact_paths.items()}
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0
        except ExternalGeneratorError as error:
            payload = {"error": str(error), "details": error.details}
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 2

    if args.command == "run-multi":
        config_paths = _load_config_paths(args.config_list)
        artifact_paths = run_multi_scene_pipeline(config_paths=config_paths, output_dir=args.output_dir)
        payload = {name: str(path) for name, path in artifact_paths.items()}
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    if args.command == "check-env":
        report = collect_environment_report(
            spatiallm_command_env_var=args.spatiallm_command_env_var,
        )
        if args.format == "json":
            print(json.dumps(report, indent=2, sort_keys=True))
        else:
            print(render_environment_report_text(report))
        return 0

    artifact_paths = run_evaluation_pack(manifest_path=args.manifest.resolve(), output_dir=args.output_dir.resolve())
    payload = {name: str(path) for name, path in artifact_paths.items()}
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
