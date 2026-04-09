"""Run dedicated post-true-v1 interpretation flow from an evaluation report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from self_calibrating_spatiallm.evaluation.post_run_analysis import (
    build_post_true_v1_analysis_bundle,
    render_first_research_result_markdown,
    render_next_improvement_decision_markdown,
    render_researcher_summary_markdown,
    render_scene_level_delta_markdown,
    render_stratified_summary_markdown,
    render_trustworthy_comparison_status_markdown,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run post-true-v1 analysis from evaluation report")
    parser.add_argument(
        "--evaluation-report",
        type=Path,
        default=Path("outputs/eval_pack/latest/evaluation_report.json"),
        help="Path to evaluation report JSON",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for analysis artifacts (defaults to evaluation report directory)",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()

    report_path = args.evaluation_report.resolve()
    payload = json.loads(report_path.read_text(encoding="utf-8"))

    output_dir = args.output_dir.resolve() if args.output_dir else report_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    bundle = build_post_true_v1_analysis_bundle(evaluation_report=payload)

    scene_rows = bundle["scene_level_deltas"]
    partition_summaries = bundle["partition_summaries"]
    stratified_summary = bundle["stratified_summary"]
    trustworthy_status = bundle["trustworthy_comparison_status"]
    next_decision = bundle["next_improvement_decision"]
    first_result = bundle["first_research_result"]
    researcher_summary = bundle["researcher_summary"]

    scene_json = output_dir / "scene_level_delta_report.json"
    scene_json.write_text(json.dumps(scene_rows, indent=2, sort_keys=True), encoding="utf-8")
    scene_md = output_dir / "scene_level_delta_report.md"
    scene_md.write_text(render_scene_level_delta_markdown(scene_rows), encoding="utf-8")

    strat_json = output_dir / "stratified_v0_v1_summary.json"
    strat_json.write_text(
        json.dumps(
            {
                "partitions": partition_summaries,
                "stratified": stratified_summary,
                "trustworthy_comparison_status": trustworthy_status,
                "next_improvement_decision": next_decision,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    strat_md = output_dir / "stratified_v0_v1_summary.md"
    strat_md.write_text(render_stratified_summary_markdown(stratified_summary), encoding="utf-8")

    trust_json = output_dir / "trustworthy_comparison_status.json"
    trust_json.write_text(json.dumps(trustworthy_status, indent=2, sort_keys=True), encoding="utf-8")
    trust_md = output_dir / "trustworthy_comparison_status.md"
    trust_md.write_text(
        render_trustworthy_comparison_status_markdown(trustworthy_status),
        encoding="utf-8",
    )

    first_json = output_dir / "first_research_result.json"
    first_json.write_text(json.dumps(first_result, indent=2, sort_keys=True), encoding="utf-8")
    first_md = output_dir / "first_research_result.md"
    first_md.write_text(render_first_research_result_markdown(first_result), encoding="utf-8")

    decision_json = output_dir / "next_improvement_decision.json"
    decision_json.write_text(json.dumps(next_decision, indent=2, sort_keys=True), encoding="utf-8")
    decision_md = output_dir / "next_improvement_decision.md"
    decision_md.write_text(render_next_improvement_decision_markdown(next_decision), encoding="utf-8")

    researcher_json = output_dir / "researcher_summary.json"
    researcher_json.write_text(json.dumps(researcher_summary, indent=2, sort_keys=True), encoding="utf-8")
    researcher_md = output_dir / "researcher_summary.md"
    researcher_md.write_text(render_researcher_summary_markdown(researcher_summary), encoding="utf-8")

    bundle_json = output_dir / "post_true_v1_analysis.json"
    bundle_json.write_text(json.dumps(bundle, indent=2, sort_keys=True), encoding="utf-8")

    print(
        json.dumps(
            {
                "scene_level_delta_report_json": str(scene_json),
                "scene_level_delta_report_markdown": str(scene_md),
                "stratified_v0_v1_summary_json": str(strat_json),
                "stratified_v0_v1_summary_markdown": str(strat_md),
                "trustworthy_comparison_status_json": str(trust_json),
                "trustworthy_comparison_status_markdown": str(trust_md),
                "first_research_result_json": str(first_json),
                "first_research_result_markdown": str(first_md),
                "next_improvement_decision_json": str(decision_json),
                "next_improvement_decision_markdown": str(decision_md),
                "researcher_summary_json": str(researcher_json),
                "researcher_summary_markdown": str(researcher_md),
                "post_true_v1_analysis_json": str(bundle_json),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
