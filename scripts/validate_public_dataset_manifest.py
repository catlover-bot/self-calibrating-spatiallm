"""Validate a public-dataset eval-pack manifest and report scene usability."""

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

from self_calibrating_spatiallm.evaluation.pack_manifest import EvaluationPackManifest
from self_calibrating_spatiallm.io import PointCloudLoadOptions, load_point_cloud_sample
from self_calibrating_spatiallm.pipeline import SceneInputConfig


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate public-dataset manifest scene usability")
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Path to eval-pack manifest JSON",
    )
    parser.add_argument(
        "--max-scenes",
        type=int,
        default=None,
        help="Optional cap for quick validation",
    )
    parser.add_argument(
        "--check-load",
        dest="check_load",
        action="store_true",
        help="Attempt point-cloud loading for each scene",
    )
    parser.add_argument(
        "--no-check-load",
        dest="check_load",
        action="store_false",
        help="Only check config/path existence without loading point clouds",
    )
    parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Console output format",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=None,
        help="Optional path to save JSON summary",
    )
    parser.add_argument(
        "--fail-on-invalid",
        action="store_true",
        help="Exit non-zero if any scene is invalid",
    )
    parser.set_defaults(check_load=True)
    return parser


def _summarize_num_points(values: list[int]) -> dict[str, float | int | None]:
    if not values:
        return {"min": None, "max": None, "mean": None}
    return {
        "min": min(values),
        "max": max(values),
        "mean": sum(values) / len(values),
    }


def _render_text(summary: dict[str, Any]) -> str:
    lines = [
        "# Public Manifest Validation",
        "",
        f"- manifest: `{summary['manifest_path']}`",
        f"- manifest_name: `{summary['manifest_name']}`",
        f"- total_entries: `{summary['total_entries']}`",
        f"- checked_entries: `{summary['checked_entries']}`",
        f"- check_load: `{summary['check_load']}`",
        f"- usable_scene_count: `{summary['usable_scene_count']}`",
        f"- invalid_scene_count: `{summary['invalid_scene_count']}`",
        "",
        "## Failures",
        f"- missing_config_count: `{summary['missing_config_count']}`",
        f"- invalid_config_count: `{summary['invalid_config_count']}`",
        f"- missing_point_cloud_count: `{summary['missing_point_cloud_count']}`",
        f"- load_failure_count: `{summary['load_failure_count']}`",
        "",
        "## Point Cloud Stats",
        f"- num_points.min: `{summary['num_points_stats']['min']}`",
        f"- num_points.max: `{summary['num_points_stats']['max']}`",
        f"- num_points.mean: `{summary['num_points_stats']['mean']}`",
        "",
        "## Source Types",
    ]

    source_counts = summary.get("source_type_counts", {})
    if isinstance(source_counts, dict) and source_counts:
        for key in sorted(source_counts.keys()):
            lines.append(f"- {key}: `{source_counts[key]}`")
    else:
        lines.append("- none")

    invalid_rows = summary.get("invalid_rows", [])
    lines.extend(["", "## Invalid Scenes (first 10)"])
    if isinstance(invalid_rows, list) and invalid_rows:
        for row in invalid_rows[:10]:
            lines.append(
                f"- `{row.get('scene_id', 'unknown')}` status=`{row.get('status')}` reason=`{row.get('reason')}`"
            )
    else:
        lines.append("- none")

    return "\n".join(lines)


def main() -> int:
    args = _build_parser().parse_args()
    manifest_path = args.manifest.expanduser().resolve()
    manifest = EvaluationPackManifest.load_json(manifest_path)

    resolved_entries = manifest.resolve_paths(manifest_path)
    if args.max_scenes is not None and args.max_scenes > 0:
        resolved_entries = resolved_entries[: args.max_scenes]

    rows: list[dict[str, Any]] = []
    source_type_counts: dict[str, int] = {}
    num_points_values: list[int] = []

    missing_config_count = 0
    invalid_config_count = 0
    missing_point_cloud_count = 0
    load_failure_count = 0

    for config_path, _annotation_path, entry in resolved_entries:
        row: dict[str, Any] = {
            "sample_config_path": str(config_path),
            "scene_id": None,
            "status": "unknown",
            "reason": None,
            "point_cloud_path": None,
            "source_type": entry.source_type,
            "num_points": None,
        }

        if not config_path.exists():
            missing_config_count += 1
            row.update({"status": "invalid", "reason": "missing_sample_config"})
            rows.append(row)
            continue

        try:
            config = SceneInputConfig.load_json(config_path)
        except Exception as error:  # pragma: no cover - defensive path
            invalid_config_count += 1
            row.update({"status": "invalid", "reason": f"invalid_sample_config: {error}"})
            rows.append(row)
            continue

        scene_id = config.scene_id
        point_cloud_path = config.resolve_file_path()
        row.update(
            {
                "scene_id": scene_id,
                "source_type": config.source_type,
                "point_cloud_path": str(point_cloud_path),
            }
        )

        if not point_cloud_path.exists():
            missing_point_cloud_count += 1
            row.update({"status": "invalid", "reason": "missing_point_cloud_file"})
            rows.append(row)
            continue

        if not args.check_load:
            row.update({"status": "usable", "reason": "path_checks_only"})
            source_type_counts[config.source_type] = source_type_counts.get(config.source_type, 0) + 1
            rows.append(row)
            continue

        try:
            _sample, metadata = load_point_cloud_sample(
                file_path=point_cloud_path,
                options=PointCloudLoadOptions(
                    scene_id=config.scene_id,
                    source_type=config.source_type,
                    metadata_path=config.resolve_metadata_path(),
                    expected_unit=config.expected_unit,
                    scale_hint=config.scale_hint,
                ),
            )
        except Exception as error:
            load_failure_count += 1
            row.update({"status": "invalid", "reason": f"load_failure: {error}"})
            rows.append(row)
            continue

        source_type_counts[metadata.source_type] = source_type_counts.get(metadata.source_type, 0) + 1
        row.update(
            {
                "status": "usable",
                "reason": "load_ok",
                "num_points": metadata.num_points,
                "has_rgb": metadata.has_rgb,
            }
        )
        num_points_values.append(int(metadata.num_points))
        rows.append(row)

    invalid_rows = [row for row in rows if row.get("status") != "usable"]
    usable_scene_count = len(rows) - len(invalid_rows)

    summary = {
        "manifest_path": str(manifest_path),
        "manifest_name": manifest.name,
        "total_entries": len(manifest.entries),
        "checked_entries": len(rows),
        "check_load": bool(args.check_load),
        "usable_scene_count": usable_scene_count,
        "invalid_scene_count": len(invalid_rows),
        "missing_config_count": missing_config_count,
        "invalid_config_count": invalid_config_count,
        "missing_point_cloud_count": missing_point_cloud_count,
        "load_failure_count": load_failure_count,
        "source_type_counts": source_type_counts,
        "num_points_stats": _summarize_num_points(num_points_values),
        "rows": rows,
        "invalid_rows": invalid_rows,
    }

    if args.summary_output is not None:
        output_path = args.summary_output.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    if args.format == "json":
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(_render_text(summary))

    if args.fail_on_invalid and invalid_rows:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
