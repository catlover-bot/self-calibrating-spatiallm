"""Small-scale multi-scene runner for qualitative inspection."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from self_calibrating_spatiallm.pipeline.config import SceneInputConfig
from self_calibrating_spatiallm.pipeline.single_scene import run_single_scene_pipeline


def run_multi_scene_pipeline(
    config_paths: list[Path],
    output_dir: Path,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    started_at = _timestamp()
    scenes: list[dict[str, Any]] = []

    for config_path in config_paths:
        config = SceneInputConfig.load_json(config_path)
        scene_output_dir = output_dir / config.scene_id
        try:
            artifact_paths = run_single_scene_pipeline(config=config, output_dir=scene_output_dir)
            comparison_summary = _extract_comparison_summary(artifact_paths.get("ablation_report"))
            scenes.append(
                {
                    "scene_id": config.scene_id,
                    "config_path": str(config_path),
                    "status": "success",
                    "artifacts": {name: str(path) for name, path in artifact_paths.items()},
                    "comparison_summary": comparison_summary,
                }
            )
        except Exception as error:
            scenes.append(
                {
                    "scene_id": config.scene_id,
                    "config_path": str(config_path),
                    "status": "failed",
                    "error": str(error),
                }
            )

    manifest = {
        "mode": "multi_scene_qualitative_v0",
        "started_at": started_at,
        "finished_at": _timestamp(),
        "num_scenes": len(config_paths),
        "success_count": sum(1 for scene in scenes if scene["status"] == "success"),
        "failure_count": sum(1 for scene in scenes if scene["status"] == "failed"),
        "scenes": scenes,
    }

    manifest_path = output_dir / "multi_scene_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    report_path = output_dir / "multi_scene_report.md"
    report_path.write_text(_render_multi_scene_report(manifest), encoding="utf-8")

    return {
        "multi_scene_manifest": manifest_path,
        "multi_scene_report": report_path,
    }


def _render_multi_scene_report(manifest: dict[str, Any]) -> str:
    lines = [
        "# Multi-Scene Qualitative Report",
        "",
        f"- started_at: `{manifest.get('started_at')}`",
        f"- finished_at: `{manifest.get('finished_at')}`",
        f"- num_scenes: `{manifest.get('num_scenes')}`",
        f"- success_count: `{manifest.get('success_count')}`",
        f"- failure_count: `{manifest.get('failure_count')}`",
        "",
        "## Scene Results",
    ]

    for scene in manifest.get("scenes", []):
        scene_id = scene.get("scene_id")
        status = scene.get("status")
        lines.append(f"- `{scene_id}`: status=`{status}`")
        if status == "success":
            artifacts = scene.get("artifacts", {})
            run_manifest = artifacts.get("run_manifest", "n/a")
            report = artifacts.get("qualitative_report", "n/a")
            lines.append(f"- `{scene_id}` run_manifest: `{run_manifest}`")
            lines.append(f"- `{scene_id}` qualitative_report: `{report}`")
            comparison_summary = scene.get("comparison_summary", {})
            if isinstance(comparison_summary, dict):
                for setting_name, passed in sorted(comparison_summary.items()):
                    lines.append(f"- `{scene_id}` `{setting_name}` passed={passed}")
        else:
            lines.append(f"- `{scene_id}` error: `{scene.get('error')}`")

    return "\n".join(lines)


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _extract_comparison_summary(ablation_report_path: Path | None) -> dict[str, bool]:
    if ablation_report_path is None or not ablation_report_path.exists():
        return {}

    payload = json.loads(ablation_report_path.read_text(encoding="utf-8"))
    summary: dict[str, bool] = {}

    for section_name in ("settings", "generator_settings"):
        section = payload.get(section_name, [])
        if not isinstance(section, list):
            continue
        for item in section:
            if not isinstance(item, dict):
                continue
            setting_name = item.get("setting_name")
            evaluation = item.get("evaluation", {})
            passed = evaluation.get("passed") if isinstance(evaluation, dict) else None
            if isinstance(setting_name, str) and isinstance(passed, bool):
                summary[setting_name] = passed

    return summary
