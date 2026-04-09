"""Markdown report rendering for qualitative run inspection."""

from __future__ import annotations

from typing import Any

from self_calibrating_spatiallm.artifacts import (
    AblationReport,
    ActionableScene,
    CalibratedPointCloud,
    CalibrationResult,
    EvaluationResult,
    PointCloudMetadata,
    RepairResult,
    ScenePrediction,
)


def render_qualitative_report(
    point_cloud_metadata: PointCloudMetadata,
    calibrated: CalibratedPointCloud,
    calibration: CalibrationResult,
    prediction: ScenePrediction,
    repair: RepairResult,
    actionable_scene: ActionableScene,
    evaluation: EvaluationResult,
    ablation_report: AblationReport,
    run_manifest: dict[str, Any],
) -> str:
    diagnostics = calibration.metadata.get("diagnostics", {})
    confidence = diagnostics.get("confidence", {})
    propagation = prediction.metadata.get("propagation_diagnostics", {})
    if not isinstance(propagation, dict):
        propagation = {}

    lines = [
        f"# Qualitative Run Report: {point_cloud_metadata.sample_id}",
        "",
        "## Point Cloud Summary",
        f"- source: `{point_cloud_metadata.source_path}` ({point_cloud_metadata.source_type})",
        f"- num_points: `{point_cloud_metadata.num_points}`",
        f"- has_rgb: `{point_cloud_metadata.has_rgb}`",
        f"- coordinate_ranges: `{point_cloud_metadata.coordinate_ranges}`",
        f"- inferred_scale_hint: `{point_cloud_metadata.inferred_scale_hint}`",
        "",
        "## Calibration Diagnostics",
        f"- method: `{calibration.method}`",
        f"- up_vector: `{calibration.up_vector.to_dict()}`",
        f"- horizontal_axis: `{calibration.horizontal_axis.to_dict()}`",
        f"- confidence: `{confidence}`",
        f"- normalized_frame: `{calibrated.metadata.get('frame')}`",
        "",
        "## Scene Prediction Summary",
        f"- generator: `{prediction.generator_name}`",
        f"- predicted_objects: `{len(prediction.objects)}`",
        f"- predicted_relations: `{len(prediction.relations)}`",
        f"- metadata_keys: `{sorted(prediction.metadata.keys())}`",
        f"- propagation_mode: `{propagation.get('generator_mode')}`",
        f"- propagation_signal: `{propagation.get('calibration_signal')}`",
        f"- propagation_layout_changes: `{propagation.get('layout_changes')}`",
        "",
        "## Violations Found",
    ]

    if repair.issues:
        for issue in repair.issues:
            lines.append(f"- {issue}")
    else:
        lines.append("- none")

    lines.extend(
        [
            "",
            "## Repairs Applied",
        ]
    )
    if repair.fixes_applied:
        for fix in repair.fixes_applied:
            lines.append(f"- {fix}")
    else:
        lines.append("- none")

    lines.extend(
        [
            "",
            "## Derived Relations",
        ]
    )
    if actionable_scene.relations:
        for relation in actionable_scene.relations[:40]:
            lines.append(
                f"- `{relation.subject_id} --{relation.predicate}--> {relation.object_id}` "
                f"(score={relation.score:.2f})"
            )
        if len(actionable_scene.relations) > 40:
            lines.append(f"- ... ({len(actionable_scene.relations) - 40} more)")
    else:
        lines.append("- none")

    lines.extend(
        [
            "",
            "## Ablation Summary",
        ]
    )
    for setting in ablation_report.settings:
        lines.append(
            f"- `{setting.setting_name}`: passed={setting.evaluation.passed}, "
            f"metrics={setting.evaluation.metrics}"
        )

    if ablation_report.generator_settings:
        lines.append("")
        lines.append("## Generator Comparison")
        for setting in ablation_report.generator_settings:
            lines.append(
                f"- `{setting.setting_name}`: passed={setting.evaluation.passed}, "
                f"calibration=`{setting.calibration_method}`, repair=`{setting.repairer_name}`, "
                f"metrics={setting.evaluation.metrics}"
            )

    lines.extend(
        [
            "",
            "## Run Manifest Snapshot",
            f"- status: `{run_manifest.get('status')}`",
            f"- started_at: `{run_manifest.get('started_at')}`",
            f"- finished_at: `{run_manifest.get('finished_at')}`",
            f"- generator_mode: `{run_manifest.get('generator_mode')}`",
            f"- calibration_mode: `{run_manifest.get('calibration_mode')}`",
            f"- repair_mode: `{run_manifest.get('repair_mode')}`",
        ]
    )

    return "\n".join(lines)
