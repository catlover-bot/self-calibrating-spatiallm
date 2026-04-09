"""Text-first visualization for debugging pipeline outputs."""

from __future__ import annotations

from self_calibrating_spatiallm.artifacts import (
    AblationReport,
    ActionableScene,
    CalibratedPointCloud,
    CalibrationResult,
    EvaluationResult,
    PointCloudMetadata,
    PointCloudSample,
    RepairResult,
    ScenePrediction,
)


def render_pipeline_summary(
    point_cloud_metadata: PointCloudMetadata,
    sample: PointCloudSample,
    calibrated: CalibratedPointCloud,
    calibration: CalibrationResult,
    scene: ScenePrediction,
    repair: RepairResult,
    actionable: ActionableScene,
    evaluation: EvaluationResult,
    ablation_report: AblationReport,
) -> str:
    confidence = calibration.metadata.get("diagnostics", {}).get("confidence", {})
    propagation = scene.metadata.get("propagation_diagnostics", {})
    if not isinstance(propagation, dict):
        propagation = {}

    lines = [
        f"sample_id: {sample.sample_id}",
        f"source_path: {point_cloud_metadata.source_path}",
        f"source_type: {point_cloud_metadata.source_type}",
        f"num_points: {point_cloud_metadata.num_points}",
        f"has_rgb: {point_cloud_metadata.has_rgb}",
        f"coordinate_ranges: {point_cloud_metadata.coordinate_ranges}",
        f"scale_hint: {point_cloud_metadata.inferred_scale_hint}",
        "",
        f"calibration_method: {calibration.method}",
        f"up_vector: ({calibration.up_vector.x:.3f}, {calibration.up_vector.y:.3f}, {calibration.up_vector.z:.3f})",
        (
            f"horizontal_axis: ({calibration.horizontal_axis.x:.3f}, "
            f"{calibration.horizontal_axis.y:.3f}, {calibration.horizontal_axis.z:.3f})"
        ),
        f"calibration_confidence: {confidence}",
        f"calibrated_num_points: {calibrated.num_points}",
        "",
        f"objects: {len(scene.objects)}",
        f"propagation_mode: {propagation.get('generator_mode')}",
        f"propagation_signal: {propagation.get('calibration_signal')}",
        f"relations_after_repair: {len(repair.repaired_scene.relations)}",
        f"repair_issues: {len(repair.issues)}",
        f"repair_fixes: {len(repair.fixes_applied)}",
        f"actionable_relations: {len(actionable.relations)}",
        f"actions: {len(actionable.actions)}",
        f"evaluation_passed: {evaluation.passed}",
        "",
        "objects:",
    ]

    for obj in scene.objects:
        lines.append(
            f"- {obj.object_id} ({obj.label}) @ ({obj.position.x:.2f}, {obj.position.y:.2f}, {obj.position.z:.2f})"
        )

    lines.append("")
    lines.append("ablations:")
    for setting in ablation_report.settings:
        lines.append(
            f"- {setting.setting_name}: passed={setting.evaluation.passed}, "
            f"calibration={setting.calibration_method}, repair={setting.repairer_name}, "
            f"metrics={setting.evaluation.metrics}"
        )

    if ablation_report.generator_settings:
        lines.append("")
        lines.append("generator_comparisons:")
        for setting in ablation_report.generator_settings:
            lines.append(
                f"- {setting.setting_name}: passed={setting.evaluation.passed}, "
                f"calibration={setting.calibration_method}, repair={setting.repairer_name}, "
                f"metrics={setting.evaluation.metrics}"
            )

    return "\n".join(lines)
