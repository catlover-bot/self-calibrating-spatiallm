"""Build ARKitScenes subset manifests compatible with the existing eval-pack pipeline."""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build ARKitScenes subset manifest")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Path to local ARKitScenes root directory",
    )
    parser.add_argument(
        "--dataset-config",
        type=Path,
        default=None,
        help="Optional dataset config JSON (see configs/public_datasets/arkitscenes/dataset_config.template.json)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where generated sample configs + manifest are written",
    )
    parser.add_argument(
        "--subset-size",
        type=int,
        default=None,
        help="Number of scenes to sample (default: 25)",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed for subset sampling")
    parser.add_argument(
        "--extensions",
        type=str,
        default=None,
        help="Comma-separated point-cloud extensions (default: .ply,.pcd,.npy,.npz)",
    )
    parser.add_argument(
        "--expected-unit",
        type=str,
        default=None,
        help="Expected unit written to generated sample configs (default: meter)",
    )
    parser.add_argument(
        "--scale-hint",
        type=str,
        default=None,
        help="Optional scale hint written to generated sample configs",
    )
    parser.add_argument(
        "--calibration-mode",
        type=str,
        choices=["none", "v0", "v1"],
        default=None,
        help="Calibration mode for generated sample configs (default: v1)",
    )
    parser.add_argument(
        "--generator-mode",
        type=str,
        choices=["mock", "external"],
        default=None,
        help="Generator mode for generated sample configs (default: mock)",
    )
    parser.add_argument(
        "--external-timeout-sec",
        type=int,
        default=None,
        help="Timeout for external generator in generated sample configs (default: 300)",
    )
    parser.add_argument(
        "--manifest-name",
        type=str,
        default=None,
        help="Override eval-pack manifest name",
    )
    parser.add_argument(
        "--normalize-scene",
        dest="normalize_scene",
        action="store_true",
        help="Enable scene normalization in generated sample configs",
    )
    parser.add_argument(
        "--no-normalize-scene",
        dest="normalize_scene",
        action="store_false",
        help="Disable scene normalization in generated sample configs",
    )
    parser.add_argument(
        "--compare-with-external-generator",
        dest="compare_with_external_generator",
        action="store_true",
        help="Set compare_with_external_generator=true in generated sample configs",
    )
    parser.add_argument(
        "--no-compare-with-external-generator",
        dest="compare_with_external_generator",
        action="store_false",
        help="Set compare_with_external_generator=false in generated sample configs",
    )
    parser.set_defaults(normalize_scene=None, compare_with_external_generator=None)
    return parser


def _load_dataset_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object at {path}")
    return payload


def _coalesce(cli_value: Any, config: dict[str, Any], key: str, default: Any) -> Any:
    if cli_value is not None:
        return cli_value
    if key in config:
        return config[key]
    return default


def _parse_extensions(value: Any) -> list[str]:
    if isinstance(value, list):
        parsed = [str(item).strip().lower() for item in value if str(item).strip()]
    else:
        parsed = [part.strip().lower() for part in str(value).split(",") if part.strip()]

    normalized: list[str] = []
    for ext in parsed:
        if not ext.startswith("."):
            ext = f".{ext}"
        if ext not in normalized:
            normalized.append(ext)
    return normalized


def _discover_point_clouds(dataset_root: Path, extensions: list[str]) -> list[Path]:
    discovered: list[Path] = []
    seen: set[str] = set()
    for ext in extensions:
        for candidate in dataset_root.rglob(f"*{ext}"):
            if not candidate.is_file():
                continue
            key = str(candidate.resolve())
            if key in seen:
                continue
            seen.add(key)
            discovered.append(candidate.resolve())
    discovered.sort(key=lambda path: str(path))
    return discovered


def _sanitize_scene_token(raw: str) -> str:
    token = re.sub(r"[^a-zA-Z0-9_]+", "_", raw).strip("_").lower()
    if not token:
        token = "scene"
    if len(token) > 96:
        token = token[-96:]
    return token


def _build_scene_id(dataset_root: Path, point_cloud_path: Path, used: set[str]) -> str:
    relative = point_cloud_path.relative_to(dataset_root)
    token = _sanitize_scene_token(relative.with_suffix("").as_posix())
    scene_id = f"arkitscenes_{token}"

    dedupe_index = 1
    while scene_id in used:
        scene_id = f"arkitscenes_{token}_{dedupe_index:02d}"
        dedupe_index += 1
    used.add(scene_id)
    return scene_id


def _source_type_from_path(path: Path) -> str:
    suffix = path.suffix.lower().lstrip(".")
    return suffix if suffix else "auto"


def _select_subset(candidates: list[Path], subset_size: int, seed: int) -> list[Path]:
    if subset_size <= 0 or subset_size >= len(candidates):
        return list(candidates)

    sampled = list(candidates)
    random.Random(seed).shuffle(sampled)
    selected = sampled[:subset_size]
    selected.sort(key=lambda path: str(path))
    return selected


def main() -> int:
    args = _build_parser().parse_args()
    dataset_config = _load_dataset_config(args.dataset_config.resolve() if args.dataset_config else None)

    dataset_root_raw = _coalesce(args.dataset_root, dataset_config, "dataset_root", None)
    if dataset_root_raw is None:
        raise SystemExit("dataset root is required. Pass --dataset-root or provide dataset_root in --dataset-config.")
    dataset_root = Path(dataset_root_raw).expanduser().resolve()
    if not dataset_root.exists() or not dataset_root.is_dir():
        raise SystemExit(f"dataset root not found or not a directory: {dataset_root}")

    output_dir_raw = _coalesce(
        args.output_dir,
        dataset_config,
        "output_dir",
        ROOT / "configs" / "public_datasets" / "arkitscenes" / "generated",
    )
    output_dir = Path(output_dir_raw).expanduser().resolve()

    subset_size = int(_coalesce(args.subset_size, dataset_config, "subset_size", 25))
    seed = int(_coalesce(args.seed, dataset_config, "seed", 13))
    extensions = _parse_extensions(
        _coalesce(args.extensions, dataset_config, "point_cloud_extensions", ".ply,.pcd,.npy,.npz")
    )

    calibration_mode = str(_coalesce(args.calibration_mode, dataset_config, "calibration_mode", "v1"))
    generator_mode = str(_coalesce(args.generator_mode, dataset_config, "generator_mode", "mock"))
    normalize_scene = bool(_coalesce(args.normalize_scene, dataset_config, "normalize_scene", True))
    compare_with_external_generator = bool(
        _coalesce(
            args.compare_with_external_generator,
            dataset_config,
            "compare_with_external_generator",
            False,
        )
    )

    expected_unit_raw = _coalesce(args.expected_unit, dataset_config, "expected_unit", "meter")
    scale_hint_raw = _coalesce(args.scale_hint, dataset_config, "scale_hint", None)
    external_timeout_sec = int(
        _coalesce(args.external_timeout_sec, dataset_config, "external_timeout_sec", 300)
    )

    discovered = _discover_point_clouds(dataset_root, extensions)
    if not discovered:
        raise SystemExit(
            f"No point-cloud files discovered under {dataset_root} for extensions: {extensions}"
        )

    selected = _select_subset(discovered, subset_size=subset_size, seed=seed)

    sample_config_dir = output_dir / "sample_configs"
    sample_config_dir.mkdir(parents=True, exist_ok=True)

    run_output_root = (ROOT / "outputs" / "runs" / "public_arkitscenes").resolve()

    entries: list[dict[str, Any]] = []
    inventory_rows: list[dict[str, Any]] = []
    used_scene_ids: set[str] = set()

    for index, point_cloud_path in enumerate(selected, start=1):
        scene_id = _build_scene_id(dataset_root, point_cloud_path, used_scene_ids)
        source_type = _source_type_from_path(point_cloud_path)
        scene_output_dir = run_output_root / scene_id

        sample_config = {
            "scene_id": scene_id,
            "file_path": str(point_cloud_path),
            "source_type": source_type,
            "metadata_path": None,
            "expected_unit": expected_unit_raw,
            "scale_hint": scale_hint_raw,
            "output_dir": str(scene_output_dir),
            "normalize_scene": normalize_scene,
            "calibration_mode": calibration_mode,
            "generator_mode": generator_mode,
            "spatiallm_command": None,
            "spatiallm_output_json": None,
            "external_timeout_sec": external_timeout_sec,
            "spatiallm_command_env_var": "SCSLM_SPATIALLM_COMMAND",
            "spatiallm_export_format": "json",
            "compare_with_external_generator": compare_with_external_generator,
        }

        sample_config_path = sample_config_dir / f"{scene_id}.json"
        sample_config_path.write_text(
            json.dumps(sample_config, indent=2, sort_keys=True),
            encoding="utf-8",
        )

        entries.append(
            {
                "sample_config_path": str(Path("sample_configs") / sample_config_path.name),
                "annotation_path": None,
                "source_type": source_type,
                "tags": [
                    "public_dataset",
                    "arkitscenes",
                    "medium_subset" if subset_size >= 20 else "small_subset",
                ],
                "notes": "auto-generated from local ARKitScenes assets",
            }
        )
        inventory_rows.append(
            {
                "index": index,
                "scene_id": scene_id,
                "point_cloud_path": str(point_cloud_path),
                "source_type": source_type,
                "sample_config_path": str(sample_config_path),
            }
        )

    manifest_name = str(
        _coalesce(
            args.manifest_name,
            dataset_config,
            "manifest_name",
            f"arkitscenes_subset_{len(entries)}",
        )
    )

    manifest = {
        "name": manifest_name,
        "notes": "Auto-generated ARKitScenes public-dataset subset manifest.",
        "entries": entries,
        "metadata": {
            "dataset_name": "arkitscenes",
            "dataset_root": str(dataset_root),
            "point_cloud_extensions": extensions,
            "num_candidates_discovered": len(discovered),
            "num_entries_selected": len(entries),
            "subset_size_requested": subset_size,
            "seed": seed,
            "calibration_mode": calibration_mode,
            "generator_mode": generator_mode,
            "normalize_scene": normalize_scene,
            "compare_with_external_generator": compare_with_external_generator,
            "external_timeout_sec": external_timeout_sec,
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "arkitscenes_subset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    inventory_payload = {
        "dataset_root": str(dataset_root),
        "num_candidates_discovered": len(discovered),
        "num_entries_selected": len(entries),
        "scene_rows": inventory_rows,
    }
    inventory_path = output_dir / "arkitscenes_scene_inventory.json"
    inventory_path.write_text(json.dumps(inventory_payload, indent=2, sort_keys=True), encoding="utf-8")

    print(
        json.dumps(
            {
                "dataset_root": str(dataset_root),
                "manifest_path": str(manifest_path),
                "sample_config_dir": str(sample_config_dir),
                "inventory_path": str(inventory_path),
                "num_candidates_discovered": len(discovered),
                "num_entries_selected": len(entries),
                "subset_size_requested": subset_size,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
