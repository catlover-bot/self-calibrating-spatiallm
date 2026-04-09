"""Run failure taxonomy aggregation from an evaluation report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from self_calibrating_spatiallm.evaluation import (
    classify_failures,
    render_failure_summary_markdown,
    summarize_failure_taxonomy,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run failure taxonomy analysis")
    parser.add_argument(
        "--evaluation-report",
        type=Path,
        default=Path("outputs/eval_pack/latest/evaluation_report.json"),
        help="Path to evaluation report JSON",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/eval_pack/latest"),
        help="Output directory for failure summary files",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    payload = json.loads(args.evaluation_report.read_text(encoding="utf-8"))

    flattened = _flatten_results(payload)
    summary = summarize_failure_taxonomy(flattened)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "failure_taxonomy_summary.json"
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    md_path = output_dir / "failure_taxonomy_summary.md"
    md_path.write_text(render_failure_summary_markdown(summary), encoding="utf-8")

    print(
        json.dumps(
            {
                "failure_taxonomy_summary_json": str(json_path),
                "failure_taxonomy_summary_markdown": str(md_path),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _flatten_results(payload: dict[str, Any]) -> list[dict[str, Any]]:
    scenes = payload.get("scenes", [])
    if not isinstance(scenes, list):
        return []

    rows: list[dict[str, Any]] = []
    for scene in scenes:
        if not isinstance(scene, dict):
            continue
        scene_id = str(scene.get("scene_id", "unknown"))
        settings = scene.get("settings", [])
        if not isinstance(settings, list):
            continue
        for setting in settings:
            if not isinstance(setting, dict):
                continue
            metrics = setting.get("metrics", {})
            failures = setting.get("failures")
            if not isinstance(failures, list):
                failures = classify_failures(metrics if isinstance(metrics, dict) else {})
            rows.append(
                {
                    "scene_id": scene_id,
                    "setting_name": str(setting.get("setting_name", "unknown")),
                    "failures": failures,
                }
            )
    return rows


if __name__ == "__main__":
    raise SystemExit(main())
