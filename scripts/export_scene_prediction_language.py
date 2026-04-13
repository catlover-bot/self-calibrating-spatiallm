"""Export one ScenePrediction artifact into deterministic language-facing forms."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from self_calibrating_spatiallm.artifacts import ScenePrediction
from self_calibrating_spatiallm.language import build_grounding_examples, build_qa_examples
from self_calibrating_spatiallm.language.exports import export_scene_prediction_to_language


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export ScenePrediction JSON into language-facing artifacts")
    parser.add_argument(
        "--scene-prediction",
        type=Path,
        required=True,
        help="Path to ScenePrediction JSON artifact",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Output JSON path (default: <scene_prediction_stem>.language.json)",
    )
    parser.add_argument(
        "--output-markdown",
        type=Path,
        default=None,
        help="Optional markdown summary output path",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    scene_prediction_path = args.scene_prediction.expanduser().resolve()
    prediction = ScenePrediction.load_json(scene_prediction_path)

    language_export = export_scene_prediction_to_language(prediction)
    qa_examples = build_qa_examples(prediction)
    grounding_examples = build_grounding_examples(prediction)

    payload: dict[str, Any] = {
        "scene_id": prediction.sample_id,
        "generator_name": prediction.generator_name,
        "structured_prediction": prediction.to_dict(),
        "scene_summary_text": language_export.get("scene_summary_text"),
        "object_list_text": language_export.get("object_list_text"),
        "relation_text": language_export.get("relation_text"),
        "scene_paragraph_text": language_export.get("scene_paragraph_text"),
        "relation_statements": language_export.get("relation_statements"),
        "qa_examples": qa_examples,
        "grounding_examples": grounding_examples,
        "metadata": {
            "source_artifact": str(scene_prediction_path),
            "export_mode": "deterministic_template",
        },
    }

    output_json_path = (
        args.output_json.expanduser().resolve()
        if args.output_json is not None
        else scene_prediction_path.with_suffix(".language.json")
    )
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    output_json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    markdown_path = args.output_markdown.expanduser().resolve() if args.output_markdown else None
    if markdown_path is not None:
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(_render_markdown(payload), encoding="utf-8")

    print(
        json.dumps(
            {
                "output_json": str(output_json_path),
                "output_markdown": str(markdown_path) if markdown_path else None,
                "scene_id": prediction.sample_id,
                "num_qa_examples": len(qa_examples),
                "num_grounding_examples": len(grounding_examples),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Scene Language Export",
        "",
        f"- scene_id: `{payload.get('scene_id')}`",
        f"- generator_name: `{payload.get('generator_name')}`",
        f"- num_qa_examples: `{len(payload.get('qa_examples', []))}`",
        f"- num_grounding_examples: `{len(payload.get('grounding_examples', []))}`",
        "",
        "## Scene Summary",
        str(payload.get("scene_summary_text", "")),
        "",
        "## Relation Text",
        str(payload.get("relation_text", "")),
        "",
        "## QA Examples",
    ]
    for qa in payload.get("qa_examples", [])[:10]:
        lines.append(f"- Q: {qa.get('question')} | A: {qa.get('answer')} ({qa.get('task_type')})")
    if not payload.get("qa_examples"):
        lines.append("- none")

    lines.append("")
    lines.append("## Grounding Examples")
    for item in payload.get("grounding_examples", [])[:10]:
        lines.append(f"- {item.get('text')} [{item.get('task_type')}]")
    if not payload.get("grounding_examples"):
        lines.append("- none")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())

