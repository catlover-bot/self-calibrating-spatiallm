"""Runner for perturbation-driven robustness-boundary experiments."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from self_calibrating_spatiallm.evaluation import EvaluationPackManifest, EvaluationSceneEntry, run_evaluation_pack
from self_calibrating_spatiallm.io import PointCloudLoadOptions, load_point_cloud_sample
from self_calibrating_spatiallm.pipeline import SceneInputConfig
from self_calibrating_spatiallm.robustness.analysis import (
    build_boundary_rows,
    build_boundary_summary,
    export_language_boundary_artifacts,
    render_boundary_summary_markdown,
)
from self_calibrating_spatiallm.robustness.config import (
    RobustnessBoundaryConfig,
    RobustnessSplitConfig,
)
from self_calibrating_spatiallm.robustness.perturbations import (
    apply_perturbation,
    derive_variant_seed,
    severity_bucket,
    write_ascii_ply,
)


def run_robustness_boundary_experiment(
    *,
    config_path: Path,
    output_dir: Path | None = None,
) -> dict[str, Path]:
    """Run perturbation-grid evaluation and boundary analysis."""
    config = RobustnessBoundaryConfig.load_json(config_path)
    root_output_dir = config.resolve_output_dir(override=output_dir)
    root_output_dir.mkdir(parents=True, exist_ok=True)
    resolved_config_path = root_output_dir / "robustness_boundary_config.resolved.json"
    resolved_config_path.write_text(
        json.dumps(config.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )

    generated_root = root_output_dir / "generated"
    generated_root.mkdir(parents=True, exist_ok=True)

    split_reports: dict[str, dict[str, Any]] = {}
    inventory_rows: list[dict[str, Any]] = []
    generated_manifest_paths: dict[str, str] = {}
    split_report_paths: dict[str, str] = {}
    skipped_splits: list[dict[str, Any]] = []

    for split in config.splits:
        if not split.enabled:
            skipped_splits.append(
                {
                    "split_name": split.split_name,
                    "reason": "disabled",
                }
            )
            continue

        manifest_path = config.resolve_manifest_path(split)
        if not manifest_path.exists():
            if split.allow_missing_manifest:
                skipped_splits.append(
                    {
                        "split_name": split.split_name,
                        "reason": "manifest_missing_but_allowed",
                        "manifest_path": str(manifest_path),
                    }
                )
                continue
            raise FileNotFoundError(
                f"manifest_path for split {split.split_name} does not exist: {manifest_path}"
            )

        split_generated_root = generated_root / split.split_name
        split_generated_root.mkdir(parents=True, exist_ok=True)
        generated_manifest_path, split_inventory = _build_perturbed_manifest_for_split(
            config=config,
            split=split,
            base_manifest_path=manifest_path,
            output_root=split_generated_root,
        )
        generated_manifest_paths[split.split_name] = str(generated_manifest_path)
        inventory_rows.extend(split_inventory)

        split_output_dir = root_output_dir / "splits" / split.split_name
        split_output_dir.mkdir(parents=True, exist_ok=True)
        run_evaluation_pack(manifest_path=generated_manifest_path, output_dir=split_output_dir)
        split_report_paths[split.split_name] = str(split_output_dir / "evaluation_report.json")
        report_payload = json.loads((split_output_dir / "evaluation_report.json").read_text(encoding="utf-8"))
        split_reports[split.split_name] = {
            "report": report_payload,
            "split_role": split.role,
            "output_dir": str(split_output_dir),
        }

    inventory_json_path = root_output_dir / "perturbation_inventory.json"
    inventory_json_path.write_text(
        json.dumps(inventory_rows, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    boundary_rows = build_boundary_rows(
        split_reports=split_reports,
        perturbation_inventory=inventory_rows,
    )
    boundary_summary = build_boundary_summary(boundary_rows)
    boundary_summary["metadata"] = {
        "config_name": config.name,
        "seed": config.seed,
        "generated_at": _timestamp(),
        "generated_manifest_paths": generated_manifest_paths,
        "skipped_splits": skipped_splits,
    }

    boundary_rows_jsonl_path = root_output_dir / "robustness_boundary_rows.jsonl"
    _write_jsonl(boundary_rows_jsonl_path, boundary_rows)

    boundary_summary_json_path = root_output_dir / "robustness_boundary_summary.json"
    boundary_summary_json_path.write_text(
        json.dumps(boundary_summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    boundary_summary_md_path = root_output_dir / "robustness_boundary_summary.md"
    boundary_summary_md_path.write_text(
        render_boundary_summary_markdown(boundary_summary),
        encoding="utf-8",
    )

    language_outputs: dict[str, str] = {}
    if config.export_language_outputs:
        language_artifacts = export_language_boundary_artifacts(
            split_reports=split_reports,
            perturbation_inventory=inventory_rows,
            output_dir=root_output_dir / "language",
        )
        language_outputs = {name: str(path) for name, path in language_artifacts.items()}

    run_manifest = {
        "name": config.name,
        "seed": config.seed,
        "generated_at": _timestamp(),
        "config_path": str(config_path.resolve()),
        "output_dir": str(root_output_dir),
        "num_inventory_rows": len(inventory_rows),
        "num_boundary_rows": len(boundary_rows),
        "generated_manifest_paths": generated_manifest_paths,
        "split_report_paths": split_report_paths,
        "skipped_splits": skipped_splits,
        "language_outputs": language_outputs,
        "resolved_config_path": str(resolved_config_path),
    }
    run_manifest_path = root_output_dir / "robustness_boundary_run_manifest.json"
    run_manifest_path.write_text(json.dumps(run_manifest, indent=2, sort_keys=True), encoding="utf-8")

    outputs: dict[str, Path] = {
        "perturbation_inventory_json": inventory_json_path,
        "robustness_boundary_rows_jsonl": boundary_rows_jsonl_path,
        "robustness_boundary_summary_json": boundary_summary_json_path,
        "robustness_boundary_summary_markdown": boundary_summary_md_path,
        "robustness_boundary_run_manifest_json": run_manifest_path,
        "robustness_boundary_resolved_config_json": resolved_config_path,
    }
    for split_name, manifest_value in generated_manifest_paths.items():
        outputs[f"generated_manifest_{split_name}"] = Path(manifest_value)
    for key, path in language_outputs.items():
        outputs[key] = Path(path)
    return outputs


def _build_perturbed_manifest_for_split(
    *,
    config: RobustnessBoundaryConfig,
    split: RobustnessSplitConfig,
    base_manifest_path: Path,
    output_root: Path,
) -> tuple[Path, list[dict[str, Any]]]:
    base_manifest = EvaluationPackManifest.load_json(base_manifest_path)
    resolved_entries = base_manifest.resolve_paths(base_manifest_path)

    sample_config_dir = output_root / "sample_configs"
    sample_config_dir.mkdir(parents=True, exist_ok=True)
    point_cloud_dir = output_root / "point_clouds"
    point_cloud_dir.mkdir(parents=True, exist_ok=True)

    generated_entries: list[EvaluationSceneEntry] = []
    inventory_rows: list[dict[str, Any]] = []
    used_scene_ids: set[str] = set()

    for config_path, annotation_path, entry in resolved_entries:
        scene_config = SceneInputConfig.load_json(config_path)
        base_scene_id = scene_config.scene_id
        sample, _ = _load_sample_for_config(scene_config)

        if config.include_unperturbed_baseline:
            scene_id = _build_variant_scene_id(
                base_scene_id=base_scene_id,
                perturbation_type="none",
                severity=0.0,
                used_scene_ids=used_scene_ids,
            )
            generated_config_path = _write_variant_config(
                scene_config=scene_config,
                output_dir=sample_config_dir,
                scene_id=scene_id,
                point_cloud_path=scene_config.resolve_file_path(),
                source_type=scene_config.source_type,
                perturbation_metadata={
                    "base_scene_id": base_scene_id,
                    "perturbation_type": "none",
                    "severity": 0.0,
                    "severity_bucket": "low",
                    "seed": config.seed,
                    "parameters": {},
                },
            )
            generated_entries.append(
                _build_manifest_entry(
                    generated_config_path=generated_config_path,
                    annotation_path=annotation_path,
                    entry=entry,
                    split=split,
                    base_scene_id=base_scene_id,
                    perturbation_type="none",
                    severity=0.0,
                    parameter_summary={},
                )
            )
            inventory_rows.append(
                _build_inventory_row(
                    split=split,
                    base_scene_id=base_scene_id,
                    scene_id=scene_id,
                    perturbation_type="none",
                    severity=0.0,
                    seed=config.seed,
                    parameters={},
                    generated_config_path=generated_config_path,
                    generated_point_cloud_path=scene_config.resolve_file_path(),
                    source_sample_config_path=config_path,
                )
            )

        for perturbation in config.perturbations:
            if not perturbation.enabled:
                continue
            for severity in perturbation.severities:
                if severity <= 0.0:
                    continue
                scene_id = _build_variant_scene_id(
                    base_scene_id=base_scene_id,
                    perturbation_type=perturbation.name,
                    severity=severity,
                    used_scene_ids=used_scene_ids,
                )
                variant_seed = derive_variant_seed(
                    base_seed=config.seed,
                    components=[
                        split.split_name,
                        base_scene_id,
                        perturbation.name,
                        f"{severity:.4f}",
                    ],
                )
                perturbation_result = apply_perturbation(
                    base_sample=sample,
                    perturbation_type=perturbation.name,
                    severity=severity,
                    seed=variant_seed,
                    params=perturbation.params,
                )
                point_cloud_path = point_cloud_dir / f"{scene_id}.ply"
                write_ascii_ply(
                    point_cloud_path,
                    [(point.x, point.y, point.z) for point in perturbation_result.sample.points],
                )
                generated_config_path = _write_variant_config(
                    scene_config=scene_config,
                    output_dir=sample_config_dir,
                    scene_id=scene_id,
                    point_cloud_path=point_cloud_path,
                    source_type="ply",
                    perturbation_metadata={
                        "base_scene_id": base_scene_id,
                        "perturbation_type": perturbation.name,
                        "severity": float(severity),
                        "severity_bucket": severity_bucket(float(severity)),
                        "seed": variant_seed,
                        "parameters": perturbation_result.metadata.get("parameters", {}),
                    },
                )
                generated_entries.append(
                    _build_manifest_entry(
                        generated_config_path=generated_config_path,
                        annotation_path=annotation_path,
                        entry=entry,
                        split=split,
                        base_scene_id=base_scene_id,
                        perturbation_type=perturbation.name,
                        severity=float(severity),
                        parameter_summary=perturbation_result.metadata.get("parameters", {}),
                    )
                )
                inventory_rows.append(
                    _build_inventory_row(
                        split=split,
                        base_scene_id=base_scene_id,
                        scene_id=scene_id,
                        perturbation_type=perturbation.name,
                        severity=float(severity),
                        seed=variant_seed,
                        parameters=perturbation_result.metadata.get("parameters", {}),
                        generated_config_path=generated_config_path,
                        generated_point_cloud_path=point_cloud_path,
                        source_sample_config_path=config_path,
                    )
                )

    generated_manifest = EvaluationPackManifest(
        name=f"{config.name}_{split.split_name}",
        notes=(
            f"Robustness-boundary perturbation manifest for split={split.split_name} "
            f"derived from {base_manifest_path}"
        ),
        entries=generated_entries,
        metadata={
            "split_name": split.split_name,
            "split_role": split.role,
            "base_manifest_path": str(base_manifest_path),
            "num_entries": len(generated_entries),
            "num_base_entries": len(resolved_entries),
            "seed": config.seed,
            "perturbations": [
                {
                    "name": perturbation.name,
                    "severities": perturbation.severities,
                    "enabled": perturbation.enabled,
                    "params": perturbation.params,
                }
                for perturbation in config.perturbations
            ],
        },
    )
    generated_manifest_path = output_root / f"{split.split_name}_robustness_manifest.json"
    generated_manifest.save_json(generated_manifest_path)
    return generated_manifest_path, inventory_rows


def _write_variant_config(
    *,
    scene_config: SceneInputConfig,
    output_dir: Path,
    scene_id: str,
    point_cloud_path: Path,
    source_type: str,
    perturbation_metadata: dict[str, Any],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_config_path = output_dir / f"{scene_id}.json"
    payload = {
        "scene_id": scene_id,
        "file_path": str(point_cloud_path),
        "source_type": source_type,
        "metadata_path": (
            str(scene_config.resolve_metadata_path())
            if scene_config.metadata_path is not None
            else None
        ),
        "expected_unit": scene_config.expected_unit,
        "scale_hint": scene_config.scale_hint,
        "output_dir": str(scene_config.resolve_output_dir() / "robustness" / scene_id),
        "normalize_scene": scene_config.normalize_scene,
        "calibration_mode": scene_config.calibration_mode,
        "generator_mode": scene_config.generator_mode,
        "spatiallm_command": scene_config.spatiallm_command,
        "spatiallm_output_json": (
            str(scene_config.resolve_external_output_json())
            if scene_config.spatiallm_output_json is not None
            else None
        ),
        "external_timeout_sec": scene_config.external_timeout_sec,
        "spatiallm_command_env_var": scene_config.spatiallm_command_env_var,
        "spatiallm_export_format": scene_config.spatiallm_export_format,
        "compare_with_external_generator": scene_config.compare_with_external_generator,
        "perturbation_metadata": perturbation_metadata,
    }
    output_config_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return output_config_path


def _build_manifest_entry(
    *,
    generated_config_path: Path,
    annotation_path: Path | None,
    entry: EvaluationSceneEntry,
    split: RobustnessSplitConfig,
    base_scene_id: str,
    perturbation_type: str,
    severity: float,
    parameter_summary: dict[str, Any],
) -> EvaluationSceneEntry:
    base_tags = list(entry.tags)
    severity_value = float(severity)
    rb_tags = [
        f"rb:split={split.split_name}",
        f"rb:split_role={split.role}",
        f"rb:base_scene={base_scene_id}",
        f"rb:perturbation={perturbation_type}",
        f"rb:severity={severity_value:.3f}",
        f"rb:severity_bucket={severity_bucket(severity_value)}",
    ]
    combined_tags = [*base_tags, *split.tags, *rb_tags]
    notes_parts = [
        entry.notes or "",
        (
            f"robustness_boundary variant base={base_scene_id}, "
            f"perturbation={perturbation_type}, severity={severity_value:.3f}, "
            f"params={json.dumps(parameter_summary, sort_keys=True)}"
        ),
    ]
    notes = " | ".join(part for part in notes_parts if part)
    return EvaluationSceneEntry(
        sample_config_path=str(generated_config_path),
        annotation_path=(str(annotation_path) if annotation_path else None),
        source_type="ply" if perturbation_type != "none" else entry.source_type,
        tags=combined_tags,
        notes=notes,
    )


def _build_inventory_row(
    *,
    split: RobustnessSplitConfig,
    base_scene_id: str,
    scene_id: str,
    perturbation_type: str,
    severity: float,
    seed: int,
    parameters: dict[str, Any],
    generated_config_path: Path,
    generated_point_cloud_path: Path,
    source_sample_config_path: Path,
) -> dict[str, Any]:
    return {
        "split_name": split.split_name,
        "split_role": split.role,
        "base_scene_id": base_scene_id,
        "scene_id": scene_id,
        "perturbation_type": perturbation_type,
        "severity": float(severity),
        "severity_bucket": severity_bucket(float(severity)),
        "seed": int(seed),
        "parameters": dict(parameters),
        "generated_config_path": str(generated_config_path),
        "generated_point_cloud_path": str(generated_point_cloud_path),
        "source_sample_config_path": str(source_sample_config_path),
    }


def _load_sample_for_config(scene_config: SceneInputConfig) -> tuple[Any, Any]:
    return load_point_cloud_sample(
        scene_config.resolve_file_path(),
        PointCloudLoadOptions(
            scene_id=scene_config.scene_id,
            source_type=scene_config.source_type,
            metadata_path=scene_config.resolve_metadata_path(),
            expected_unit=scene_config.expected_unit,
            scale_hint=scene_config.scale_hint,
        ),
    )


def _build_variant_scene_id(
    *,
    base_scene_id: str,
    perturbation_type: str,
    severity: float,
    used_scene_ids: set[str],
) -> str:
    severity_token = f"s{int(round(severity * 1000)):04d}"
    candidate = f"{base_scene_id}__rb__{perturbation_type}__{severity_token}"
    candidate = _safe_component(candidate)
    dedupe_index = 1
    final_candidate = candidate
    while final_candidate in used_scene_ids:
        final_candidate = f"{candidate}_{dedupe_index:02d}"
        dedupe_index += 1
    used_scene_ids.add(final_candidate)
    return final_candidate


def _safe_component(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(value))
    return cleaned or "scene"


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True))
            handle.write("\n")


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()
