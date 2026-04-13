"""Evaluation pack runner for evidence-building on small scene collections."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from self_calibrating_spatiallm.artifacts import PointCloudMetadata, PointCloudSample
from self_calibrating_spatiallm.calibration import (
    GeometricCalibratorV0,
    NoCalibrationCalibrator,
    PlaneAwareCalibratorV1,
    extract_calibration_execution,
)
from self_calibrating_spatiallm.evaluation.annotations import SceneAnnotation, load_scene_annotation
from self_calibrating_spatiallm.evaluation.failure_taxonomy import (
    classify_failures,
    render_failure_summary_markdown,
    summarize_failure_taxonomy,
)
from self_calibrating_spatiallm.evaluation.metrics import compute_scene_metrics
from self_calibrating_spatiallm.evaluation.pack_manifest import EvaluationPackManifest, EvaluationSceneEntry
from self_calibrating_spatiallm.evaluation.post_run_analysis import (
    build_post_true_v1_analysis_bundle,
    render_next_improvement_decision_markdown,
    render_first_research_result_markdown,
    render_researcher_summary_markdown,
    render_scene_level_delta_markdown,
    render_stratified_summary_markdown,
    render_trustworthy_comparison_status_markdown,
)
from self_calibrating_spatiallm.evaluation.recommendations import (
    build_next_action_recommendations,
    build_v0_v1_comparison_warning,
)
from self_calibrating_spatiallm.generation import MockSpatialLMGenerator, SpatialLMExternalGenerator
from self_calibrating_spatiallm.io import PointCloudLoadOptions, load_point_cloud_sample
from self_calibrating_spatiallm.language.exports import export_scene_prediction_to_language
from self_calibrating_spatiallm.environment import collect_environment_report
from self_calibrating_spatiallm.pipeline import SceneInputConfig
from self_calibrating_spatiallm.repair import PassThroughRepairer, SimpleRuleRepairer
from self_calibrating_spatiallm.scene_graph import RuleBasedActionableSceneBuilder


@dataclass
class SettingEvaluation:
    setting_name: str
    status: str
    metrics: dict[str, float | None]
    failures: list[str]
    notes: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SceneEvaluation:
    scene_id: str
    sample_config_path: str
    annotation_path: str | None
    source_type: str | None
    tags: list[str]
    notes: str | None
    settings: list[SettingEvaluation]


@dataclass
class EvaluationPackReport:
    manifest_name: str
    generated_at: str
    scenes: list[SceneEvaluation]
    aggregate_by_setting: dict[str, dict[str, float]]
    comparison_table: list[dict[str, Any]]
    failure_summary: dict[str, Any]
    v1_execution_summary: dict[str, Any] = field(default_factory=dict)
    v0_v1_comparison_summary: dict[str, Any] = field(default_factory=dict)
    environment_readiness: dict[str, Any] = field(default_factory=dict)
    comparison_warning: str | None = None
    next_actions: list[str] = field(default_factory=list)
    scene_level_delta_report: list[dict[str, Any]] = field(default_factory=list)
    stratified_v0_v1_summary: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    first_research_result_summary: dict[str, Any] = field(default_factory=dict)
    trustworthy_comparison_status: dict[str, Any] = field(default_factory=dict)
    next_improvement_decision: dict[str, Any] = field(default_factory=dict)
    researcher_summary: dict[str, Any] = field(default_factory=dict)
    external_propagation_summary: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save_json(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        return path


def run_evaluation_pack(
    *,
    manifest_path: Path,
    output_dir: Path,
) -> dict[str, Path]:
    manifest = EvaluationPackManifest.load_json(manifest_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    scenes: list[SceneEvaluation] = []
    flattened_results: list[dict[str, Any]] = []

    for config_path, annotation_path, entry in manifest.resolve_paths(manifest_path):
        scene_result = _evaluate_scene_entry(
            config_path=config_path,
            annotation_path=annotation_path,
            entry=entry,
        )
        scenes.append(scene_result)
        for setting in scene_result.settings:
            flattened_results.append(
                {
                    "scene_id": scene_result.scene_id,
                    "setting_name": setting.setting_name,
                    "status": setting.status,
                    "metrics": setting.metrics,
                    "failures": setting.failures,
                    "metadata": setting.metadata,
                }
            )

    aggregate = _aggregate_by_setting(flattened_results)
    comparison_table = _build_comparison_table(aggregate)
    failure_summary = summarize_failure_taxonomy(flattened_results)
    v1_execution_summary = _build_v1_execution_summary(flattened_results)
    v0_v1_comparison_summary = _build_v0_v1_comparison_summary(
        scenes=scenes,
        aggregate=aggregate,
        failure_summary=failure_summary,
        v1_execution_summary=v1_execution_summary,
    )
    external_propagation_summary = _build_external_propagation_summary(scenes)
    environment_report = collect_environment_report()
    readiness_summary = environment_report.get("readiness", {})
    if not isinstance(readiness_summary, dict):
        readiness_summary = {}

    comparison_warning = build_v0_v1_comparison_warning(v1_execution_summary)
    next_actions = build_next_action_recommendations(
        environment_report=environment_report,
        v1_execution_summary=v1_execution_summary,
        comparison_warning=comparison_warning,
    )
    if comparison_warning:
        v0_v1_comparison_summary["comparison_warning"] = comparison_warning
    v0_v1_comparison_summary["next_actions"] = next_actions
    if comparison_warning:
        failure_summary["comparison_warning"] = comparison_warning

    analysis_bundle = build_post_true_v1_analysis_bundle(
        evaluation_report={
            "scenes": [asdict(scene) for scene in scenes],
            "comparison_warning": comparison_warning,
            "v1_execution_summary": v1_execution_summary,
            "failure_summary": failure_summary,
            "v0_v1_comparison_summary": v0_v1_comparison_summary,
        }
    )
    scene_rows = analysis_bundle["scene_level_deltas"]
    partition_summaries = analysis_bundle["partition_summaries"]
    stratified_summary = analysis_bundle["stratified_summary"]
    trustworthy_status = analysis_bundle["trustworthy_comparison_status"]
    next_improvement_decision = analysis_bundle["next_improvement_decision"]
    first_research_result = analysis_bundle["first_research_result"]
    researcher_summary = analysis_bundle["researcher_summary"]

    post_run_analysis = {
        "partitions": partition_summaries,
        "stratified": stratified_summary,
        "scene_level_deltas": scene_rows,
        "trustworthy_comparison_status": trustworthy_status,
        "next_improvement_decision": next_improvement_decision,
        "first_research_result": first_research_result,
        "researcher_summary": researcher_summary,
        "external_propagation_summary": external_propagation_summary,
    }
    v0_v1_comparison_summary["partition_summaries"] = partition_summaries
    v0_v1_comparison_summary["recommended_next_calibration_improvement"] = {
        "target": next_improvement_decision.get("target"),
        "reason": next_improvement_decision.get("reason"),
    }
    v0_v1_comparison_summary["trustworthy_comparison_status"] = trustworthy_status

    report = EvaluationPackReport(
        manifest_name=manifest.name,
        generated_at=_timestamp(),
        scenes=scenes,
        aggregate_by_setting=aggregate,
        comparison_table=comparison_table,
        failure_summary=failure_summary,
        v1_execution_summary=v1_execution_summary,
        v0_v1_comparison_summary=v0_v1_comparison_summary,
        environment_readiness=readiness_summary,
        comparison_warning=comparison_warning,
        next_actions=next_actions,
        scene_level_delta_report=scene_rows,
        stratified_v0_v1_summary=stratified_summary,
        first_research_result_summary=first_research_result,
        trustworthy_comparison_status=trustworthy_status,
        next_improvement_decision=next_improvement_decision,
        researcher_summary=researcher_summary,
        external_propagation_summary=external_propagation_summary,
        metadata={"num_scenes": len(scenes), "post_run_analysis": post_run_analysis},
    )

    report_json_path = report.save_json(output_dir / "evaluation_report.json")
    report_md_path = (output_dir / "evaluation_report.md")
    report_md_path.write_text(_render_report_markdown(report), encoding="utf-8")

    failure_json_path = output_dir / "failure_summary.json"
    failure_json_path.write_text(json.dumps(failure_summary, indent=2, sort_keys=True), encoding="utf-8")

    failure_md_path = output_dir / "failure_summary.md"
    failure_md_path.write_text(
        _with_warning_banner(
            render_failure_summary_markdown(failure_summary),
            comparison_warning,
        ),
        encoding="utf-8",
    )

    comparison_json_path = output_dir / "v0_v1_comparison_summary.json"
    comparison_json_path.write_text(
        json.dumps(v0_v1_comparison_summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    comparison_md_path = output_dir / "v0_v1_comparison_summary.md"
    comparison_md_path.write_text(
        _render_v0_v1_comparison_markdown(
            v0_v1_comparison_summary,
            comparison_warning=comparison_warning,
            next_actions=next_actions,
        ),
        encoding="utf-8",
    )

    environment_report_path = output_dir / "environment_report.json"
    environment_report_path.write_text(
        json.dumps(environment_report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    readiness_json_path = output_dir / "readiness_summary.json"
    readiness_json_path.write_text(
        json.dumps(
            {
                "generated_at": _timestamp(),
                "readiness": readiness_summary,
                "comparison_warning": comparison_warning,
                "next_actions": next_actions,
                "trustworthy_comparison_status": trustworthy_status,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    readiness_md_path = output_dir / "readiness_summary.md"
    readiness_md_path.write_text(
        _render_readiness_summary_markdown(
            readiness_summary=readiness_summary,
            comparison_warning=comparison_warning,
            next_actions=next_actions,
        ),
        encoding="utf-8",
    )

    scene_delta_json_path = output_dir / "scene_level_delta_report.json"
    scene_delta_json_path.write_text(
        json.dumps(scene_rows, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    scene_delta_md_path = output_dir / "scene_level_delta_report.md"
    scene_delta_md_path.write_text(
        render_scene_level_delta_markdown(scene_rows),
        encoding="utf-8",
    )

    stratified_json_path = output_dir / "stratified_v0_v1_summary.json"
    stratified_json_path.write_text(
        json.dumps(
            {
                "partitions": partition_summaries,
                "stratified": stratified_summary,
                "trustworthy_comparison_status": trustworthy_status,
                "next_improvement_decision": next_improvement_decision,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    stratified_md_path = output_dir / "stratified_v0_v1_summary.md"
    stratified_md_path.write_text(
        render_stratified_summary_markdown(stratified_summary),
        encoding="utf-8",
    )

    first_result_json_path = output_dir / "first_research_result.json"
    first_result_json_path.write_text(
        json.dumps(first_research_result, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    first_result_md_path = output_dir / "first_research_result.md"
    first_result_md_path.write_text(
        render_first_research_result_markdown(first_research_result),
        encoding="utf-8",
    )

    trustworthy_status_json_path = output_dir / "trustworthy_comparison_status.json"
    trustworthy_status_json_path.write_text(
        json.dumps(trustworthy_status, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    trustworthy_status_md_path = output_dir / "trustworthy_comparison_status.md"
    trustworthy_status_md_path.write_text(
        render_trustworthy_comparison_status_markdown(trustworthy_status),
        encoding="utf-8",
    )

    next_improvement_json_path = output_dir / "next_improvement_decision.json"
    next_improvement_json_path.write_text(
        json.dumps(next_improvement_decision, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    next_improvement_md_path = output_dir / "next_improvement_decision.md"
    next_improvement_md_path.write_text(
        render_next_improvement_decision_markdown(next_improvement_decision),
        encoding="utf-8",
    )

    researcher_summary_json_path = output_dir / "researcher_summary.json"
    researcher_summary_json_path.write_text(
        json.dumps(researcher_summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    researcher_summary_md_path = output_dir / "researcher_summary.md"
    researcher_summary_md_path.write_text(
        render_researcher_summary_markdown(researcher_summary),
        encoding="utf-8",
    )

    post_run_analysis_json_path = output_dir / "post_run_analysis.json"
    post_run_analysis_json_path.write_text(
        json.dumps(post_run_analysis, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    external_propagation_json_path = output_dir / "external_propagation_summary.json"
    external_propagation_json_path.write_text(
        json.dumps(external_propagation_summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    external_propagation_md_path = output_dir / "external_propagation_summary.md"
    external_propagation_md_path.write_text(
        _render_external_propagation_summary_markdown(external_propagation_summary),
        encoding="utf-8",
    )

    return {
        "evaluation_report_json": report_json_path,
        "evaluation_report_markdown": report_md_path,
        "failure_summary_json": failure_json_path,
        "failure_summary_markdown": failure_md_path,
        "v0_v1_comparison_summary_json": comparison_json_path,
        "v0_v1_comparison_summary_markdown": comparison_md_path,
        "environment_report_json": environment_report_path,
        "readiness_summary_json": readiness_json_path,
        "readiness_summary_markdown": readiness_md_path,
        "scene_level_delta_report_json": scene_delta_json_path,
        "scene_level_delta_report_markdown": scene_delta_md_path,
        "stratified_v0_v1_summary_json": stratified_json_path,
        "stratified_v0_v1_summary_markdown": stratified_md_path,
        "first_research_result_json": first_result_json_path,
        "first_research_result_markdown": first_result_md_path,
        "trustworthy_comparison_status_json": trustworthy_status_json_path,
        "trustworthy_comparison_status_markdown": trustworthy_status_md_path,
        "next_improvement_decision_json": next_improvement_json_path,
        "next_improvement_decision_markdown": next_improvement_md_path,
        "researcher_summary_json": researcher_summary_json_path,
        "researcher_summary_markdown": researcher_summary_md_path,
        "post_run_analysis_json": post_run_analysis_json_path,
        "external_propagation_summary_json": external_propagation_json_path,
        "external_propagation_summary_markdown": external_propagation_md_path,
    }


def _evaluate_scene_entry(
    *,
    config_path: Path,
    annotation_path: Path | None,
    entry: EvaluationSceneEntry,
) -> SceneEvaluation:
    config = SceneInputConfig.load_json(config_path)
    annotation = load_scene_annotation(annotation_path)

    sample, metadata = _load_scene(config)

    settings: list[SettingEvaluation] = []
    settings.append(
        _run_setting(
            setting_name="no_calibration",
            sample=sample,
            point_cloud_metadata=metadata,
            annotation=annotation,
            calibrator=NoCalibrationCalibrator(),
            use_repair=False,
            generator_mode="mock",
            config=config,
        )
    )
    settings.append(
        _run_setting(
            setting_name="calibration_v0",
            sample=sample,
            point_cloud_metadata=metadata,
            annotation=annotation,
            calibrator=GeometricCalibratorV0(normalize_scene=config.normalize_scene),
            use_repair=False,
            generator_mode="mock",
            config=config,
        )
    )
    settings.append(
        _run_setting(
            setting_name="calibration_v1",
            sample=sample,
            point_cloud_metadata=metadata,
            annotation=annotation,
            calibrator=PlaneAwareCalibratorV1(normalize_scene=config.normalize_scene),
            use_repair=False,
            generator_mode="mock",
            config=config,
        )
    )
    settings.append(
        _run_setting(
            setting_name="calibration_v1_plus_repair",
            sample=sample,
            point_cloud_metadata=metadata,
            annotation=annotation,
            calibrator=PlaneAwareCalibratorV1(normalize_scene=config.normalize_scene),
            use_repair=True,
            generator_mode="mock",
            config=config,
        )
    )
    settings.append(
        _run_setting(
            setting_name="mock_generator",
            sample=sample,
            point_cloud_metadata=metadata,
            annotation=annotation,
            calibrator=PlaneAwareCalibratorV1(normalize_scene=config.normalize_scene),
            use_repair=True,
            generator_mode="mock",
            config=config,
        )
    )
    settings.append(
        _run_setting(
            setting_name="external_generator",
            sample=sample,
            point_cloud_metadata=metadata,
            annotation=annotation,
            calibrator=PlaneAwareCalibratorV1(normalize_scene=config.normalize_scene),
            use_repair=True,
            generator_mode="external",
            config=config,
        )
    )

    scene_id = annotation.scene_id if annotation else config.scene_id
    return SceneEvaluation(
        scene_id=scene_id,
        sample_config_path=str(config_path),
        annotation_path=(str(annotation_path) if annotation_path else None),
        source_type=entry.source_type,
        tags=list(entry.tags),
        notes=entry.notes,
        settings=settings,
    )


def _load_scene(config: SceneInputConfig) -> tuple[PointCloudSample, PointCloudMetadata]:
    return load_point_cloud_sample(
        file_path=config.resolve_file_path(),
        options=PointCloudLoadOptions(
            scene_id=config.scene_id,
            source_type=config.source_type,
            metadata_path=config.resolve_metadata_path(),
            expected_unit=config.expected_unit,
            scale_hint=config.scale_hint,
        ),
    )


def _run_setting(
    *,
    setting_name: str,
    sample: PointCloudSample,
    point_cloud_metadata: PointCloudMetadata,
    annotation: SceneAnnotation | None,
    calibrator: NoCalibrationCalibrator | GeometricCalibratorV0 | PlaneAwareCalibratorV1,
    use_repair: bool,
    generator_mode: str,
    config: SceneInputConfig,
) -> SettingEvaluation:
    if generator_mode == "external" and not config.has_external_generator_config():
        return SettingEvaluation(
            setting_name=setting_name,
            status="skipped",
            metrics={},
            failures=["external-generator-unavailable"],
            notes=["external generator not configured for this scene"],
            metadata={
                "generator_mode": generator_mode,
                "skip_reason": "external-generator-unavailable",
            },
        )

    generator = _build_generator_for_mode(generator_mode, config)
    repairer = SimpleRuleRepairer() if use_repair else PassThroughRepairer()
    analysis_repairer = SimpleRuleRepairer()
    builder = RuleBasedActionableSceneBuilder()

    try:
        calibrated = calibrator.calibrate(sample)
        prediction = generator.generate(calibrated)
        if sample.metadata.get("room_bounds") and not prediction.metadata.get("room_bounds"):
            prediction.metadata["room_bounds"] = sample.metadata.get("room_bounds")

        precheck = analysis_repairer.repair(prediction)
        violations_before = len(precheck.issues)

        repair_result = repairer.repair(prediction)

        postcheck = analysis_repairer.repair(repair_result.repaired_scene)
        violations_after = len(postcheck.issues)

        actionable_scene = builder.build(repair_result.repaired_scene)
        calibration_execution = extract_calibration_execution(calibrated.calibration)
        metrics = compute_scene_metrics(
            annotation=annotation,
            point_cloud_metadata=point_cloud_metadata,
            calibration=calibrated.calibration,
            prediction_before_repair=prediction,
            repair_result=repair_result,
            violations_before=violations_before,
            violations_after=violations_after,
            actionable_scene=actionable_scene,
        )
        failures = classify_failures(metrics)
        propagation_diagnostics = prediction.metadata.get("propagation_diagnostics", {})
        if not isinstance(propagation_diagnostics, dict):
            propagation_diagnostics = {}
        prediction_summary = _summarize_prediction(prediction)
        prediction_serialized = _serialize_scene_prediction(prediction)
        repaired_prediction_serialized = _serialize_scene_prediction(repair_result.repaired_scene)
        language_export_pre_repair = export_scene_prediction_to_language(prediction)
        language_export_post_repair = export_scene_prediction_to_language(repair_result.repaired_scene)
        calibrated_input_summary = _summarize_calibrated_input(calibrated)
        generator_execution_summary = _summarize_generator_execution(generator.get_last_execution_info())

        return SettingEvaluation(
            setting_name=setting_name,
            status="success",
            metrics=metrics,
            failures=failures,
            notes=[],
            metadata={
                "generator_name": prediction.generator_name,
                "generator_mode": generator_mode,
                "calibration_method": calibrated.calibration.method,
                "repairer_name": repairer.name,
                "calibration_execution": calibration_execution,
                "prediction_summary_pre_repair": prediction_summary,
                "structured_prediction_pre_repair": prediction_serialized,
                "structured_prediction_post_repair": repaired_prediction_serialized,
                "language_export_pre_repair": language_export_pre_repair,
                "language_export_post_repair": language_export_post_repair,
                "calibrated_input_summary": calibrated_input_summary,
                "propagation_diagnostics": propagation_diagnostics,
                "generator_execution_summary": generator_execution_summary,
            },
        )
    except Exception as error:
        return SettingEvaluation(
            setting_name=setting_name,
            status="failed",
            metrics={},
            failures=["runtime-failure"],
            notes=[str(error)],
            metadata={"generator_mode": generator_mode},
        )


def _build_generator_for_mode(mode: str, config: SceneInputConfig) -> MockSpatialLMGenerator | SpatialLMExternalGenerator:
    if mode == "mock":
        return MockSpatialLMGenerator()
    if mode == "external":
        return SpatialLMExternalGenerator(
            command=config.spatiallm_command,
            timeout_sec=config.external_timeout_sec,
            output_json_path=config.resolve_external_output_json(),
            command_env_var=config.spatiallm_command_env_var,
            export_format=config.spatiallm_export_format,
        )
    raise ValueError(f"Unsupported generator mode: {mode}")


def _summarize_prediction(prediction: Any) -> dict[str, Any]:
    objects = getattr(prediction, "objects", [])
    relations = getattr(prediction, "relations", [])

    object_labels = []
    if isinstance(objects, list):
        object_labels = sorted({str(getattr(obj, "label", "unknown")) for obj in objects})

    relation_predicates = []
    if isinstance(relations, list):
        relation_predicates = sorted({str(getattr(rel, "predicate", "unknown")) for rel in relations})

    return {
        "object_count": len(objects) if isinstance(objects, list) else 0,
        "relation_count": len(relations) if isinstance(relations, list) else 0,
        "object_labels": object_labels,
        "relation_predicates": relation_predicates,
    }


def _serialize_scene_prediction(prediction: Any) -> dict[str, Any]:
    objects = getattr(prediction, "objects", [])
    relations = getattr(prediction, "relations", [])
    if not isinstance(objects, list):
        objects = []
    if not isinstance(relations, list):
        relations = []

    object_rows: list[dict[str, Any]] = []
    for obj in objects:
        position = getattr(obj, "position", None)
        size = getattr(obj, "size", None)
        object_rows.append(
            {
                "object_id": str(getattr(obj, "object_id", "unknown")),
                "label": str(getattr(obj, "label", "unknown")),
                "position": {
                    "x": float(getattr(position, "x", 0.0)) if position is not None else 0.0,
                    "y": float(getattr(position, "y", 0.0)) if position is not None else 0.0,
                    "z": float(getattr(position, "z", 0.0)) if position is not None else 0.0,
                },
                "size": {
                    "x": float(getattr(size, "x", 0.0)) if size is not None else 0.0,
                    "y": float(getattr(size, "y", 0.0)) if size is not None else 0.0,
                    "z": float(getattr(size, "z", 0.0)) if size is not None else 0.0,
                },
                "confidence": float(getattr(obj, "confidence", 1.0)),
                "attributes": dict(getattr(obj, "attributes", {}) or {}),
            }
        )

    relation_rows: list[dict[str, Any]] = []
    for rel in relations:
        relation_rows.append(
            {
                "subject_id": str(getattr(rel, "subject_id", "unknown")),
                "predicate": str(getattr(rel, "predicate", "unknown")),
                "object_id": str(getattr(rel, "object_id", "unknown")),
                "score": float(getattr(rel, "score", 1.0)),
                "metadata": dict(getattr(rel, "metadata", {}) or {}),
            }
        )

    prediction_metadata = getattr(prediction, "metadata", {})
    if not isinstance(prediction_metadata, dict):
        prediction_metadata = {}

    return {
        "sample_id": str(getattr(prediction, "sample_id", "unknown_scene")),
        "generator_name": str(getattr(prediction, "generator_name", "unknown_generator")),
        "objects": object_rows,
        "relations": relation_rows,
        "metadata": dict(prediction_metadata),
    }


def _summarize_calibrated_input(calibrated: Any) -> dict[str, Any]:
    metadata = getattr(calibrated, "metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}

    stats = metadata.get("transformed_point_cloud", {})
    if not isinstance(stats, dict):
        stats = {}

    calibration = getattr(calibrated, "calibration", None)
    up_vector = None
    horizontal_axis = None
    if calibration is not None:
        if hasattr(calibration, "up_vector") and calibration.up_vector is not None:
            up_vector = calibration.up_vector.to_dict()
        if hasattr(calibration, "horizontal_axis") and calibration.horizontal_axis is not None:
            horizontal_axis = calibration.horizontal_axis.to_dict()

    return {
        "frame": metadata.get("frame"),
        "num_points": getattr(calibrated, "num_points", 0),
        "ranges": stats.get("ranges"),
        "center": stats.get("center"),
        "up_vector": up_vector,
        "horizontal_axis": horizontal_axis,
    }


def _summarize_generator_execution(execution_info: Any) -> dict[str, Any]:
    if not isinstance(execution_info, dict):
        return {}

    summary = {
        "generator_mode": execution_info.get("generator_mode"),
        "generator_name": execution_info.get("generator_name"),
        "success": execution_info.get("success"),
        "command_source": execution_info.get("command_source"),
        "return_code": execution_info.get("return_code"),
        "payload_source": execution_info.get("payload_source"),
        "parse_mode": execution_info.get("parse_mode"),
        "partial_parse": execution_info.get("partial_parse"),
        "parse_warning_count": len(execution_info.get("parse_warnings", []))
        if isinstance(execution_info.get("parse_warnings"), list)
        else 0,
    }

    spatiallm_export = execution_info.get("spatiallm_export", {})
    if isinstance(spatiallm_export, dict):
        summary["spatiallm_export"] = {
            "format": spatiallm_export.get("format"),
            "num_points": spatiallm_export.get("num_points"),
            "payload_hash_sha256": spatiallm_export.get("payload_hash_sha256"),
            "payload_summary": spatiallm_export.get("payload_summary"),
        }

    prediction_summary = execution_info.get("prediction_summary")
    if isinstance(prediction_summary, dict):
        summary["prediction_summary"] = prediction_summary

    raw_files = execution_info.get("raw_files", {})
    if isinstance(raw_files, dict):
        summary["raw_files"] = {
            key: {
                "present": isinstance(value, str) and bool(value.strip()),
                "char_count": len(value) if isinstance(value, str) else 0,
            }
            for key, value in raw_files.items()
        }

    if isinstance(execution_info.get("invoked_command"), list):
        summary["invoked_command"] = [str(token) for token in execution_info["invoked_command"]]

    return summary


def _aggregate_by_setting(
    flattened_results: list[dict[str, Any]],
) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in flattened_results:
        grouped.setdefault(str(row["setting_name"]), []).append(row)

    aggregate: dict[str, dict[str, float]] = {}
    for setting_name, rows in grouped.items():
        metric_values: dict[str, list[float]] = {}
        success_count = 0
        for row in rows:
            if row.get("status") == "success":
                success_count += 1
            metrics = row.get("metrics", {})
            if not isinstance(metrics, dict):
                continue
            for key, value in metrics.items():
                if isinstance(value, (int, float)) and math.isfinite(float(value)):
                    metric_values.setdefault(str(key), []).append(float(value))

        summary: dict[str, float] = {"num_scenes": float(len(rows)), "num_success": float(success_count)}
        for key, values in metric_values.items():
            if values:
                summary[key] = sum(values) / len(values)
        aggregate[setting_name] = summary
    return aggregate


def _build_comparison_table(aggregate: dict[str, dict[str, float]]) -> list[dict[str, Any]]:
    keys_of_interest = [
        "calibration_up_axis_error_deg",
        "calibration_horizontal_error_deg",
        "calibration_up_axis_confidence",
        "calibration_horizontal_confidence",
        "calibration_reliability",
        "structured_category_presence_f1",
        "structured_violation_count_before_repair",
        "repair_violation_reduction",
        "actionable_relation_f1",
        "actionable_traversability_accuracy",
    ]
    rows: list[dict[str, Any]] = []
    for setting_name in sorted(aggregate.keys()):
        summary = aggregate[setting_name]
        row: dict[str, Any] = {"setting_name": setting_name}
        for key in keys_of_interest:
            row[key] = summary.get(key)
        rows.append(row)
    return rows


def _build_v1_execution_summary(flattened_results: list[dict[str, Any]]) -> dict[str, Any]:
    target_settings = {"calibration_v1", "calibration_v1_plus_repair"}
    rows = [
        row
        for row in flattened_results
        if row.get("setting_name") in target_settings and row.get("status") == "success"
    ]

    fallback_scenes: list[str] = []
    true_v1_count = 0
    fallback_count = 0

    reliability_values: list[float] = []
    ambiguity_values: list[float] = []
    scale_drift_values: list[float] = []

    violation_by_setting: dict[str, dict[str, list[float]]] = {}

    for row in rows:
        scene_id = str(row.get("scene_id", "unknown"))
        setting_name = str(row.get("setting_name", "unknown"))
        metadata = row.get("metadata", {})
        metrics = row.get("metrics", {})
        if not isinstance(metadata, dict):
            metadata = {}
        if not isinstance(metrics, dict):
            metrics = {}

        execution = metadata.get("calibration_execution", {})
        if not isinstance(execution, dict):
            execution = {}

        if bool(execution.get("true_v1_execution")):
            true_v1_count += 1
        if bool(execution.get("fallback_used")):
            fallback_count += 1
            fallback_scenes.append(scene_id)

        for key, values in [
            ("calibration_reliability", reliability_values),
            ("calibration_manhattan_ambiguity", ambiguity_values),
            ("calibration_scale_drift", scale_drift_values),
        ]:
            value = metrics.get(key)
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                values.append(float(value))

        before_value = metrics.get("structured_violation_count_before_repair")
        after_value = metrics.get("repair_violations_after")
        per_setting = violation_by_setting.setdefault(
            setting_name,
            {"before": [], "after": []},
        )
        if isinstance(before_value, (int, float)) and math.isfinite(float(before_value)):
            per_setting["before"].append(float(before_value))
        if isinstance(after_value, (int, float)) and math.isfinite(float(after_value)):
            per_setting["after"].append(float(after_value))

    downstream_violation_counts_by_setting: dict[str, dict[str, float | None]] = {}
    for setting_name, values in sorted(violation_by_setting.items()):
        before = values.get("before", [])
        after = values.get("after", [])
        downstream_violation_counts_by_setting[setting_name] = {
            "avg_before_repair": (sum(before) / len(before)) if before else None,
            "avg_after_repair": (sum(after) / len(after)) if after else None,
        }

    return {
        "target_settings": sorted(target_settings),
        "num_scene_setting_results": len(rows),
        "num_true_v1_execution": true_v1_count,
        "num_fallback_used": fallback_count,
        "fallback_scene_ids": sorted(set(fallback_scenes)),
        "avg_calibration_reliability": _mean_or_none(reliability_values),
        "avg_manhattan_ambiguity": _mean_or_none(ambiguity_values),
        "avg_scale_drift": _mean_or_none(scale_drift_values),
        "downstream_violation_counts_by_setting": downstream_violation_counts_by_setting,
    }


def _build_v0_v1_comparison_summary(
    *,
    scenes: list[SceneEvaluation],
    aggregate: dict[str, dict[str, float]],
    failure_summary: dict[str, Any],
    v1_execution_summary: dict[str, Any],
) -> dict[str, Any]:
    improvements: list[str] = []
    regressions: list[str] = []

    v0 = aggregate.get("calibration_v0", {})
    v1 = aggregate.get("calibration_v1", {})
    v1_repair = aggregate.get("calibration_v1_plus_repair", {})

    metric_defs = [
        ("calibration_up_axis_error_deg", "lower_better"),
        ("calibration_horizontal_error_deg", "lower_better"),
        ("calibration_reliability", "higher_better"),
        ("structured_violation_count_before_repair", "lower_better"),
        ("actionable_relation_f1", "higher_better"),
    ]
    metric_deltas: dict[str, float] = {}
    for key, direction in metric_defs:
        v0_value = v0.get(key)
        v1_value = v1.get(key)
        if not isinstance(v0_value, (int, float)) or not isinstance(v1_value, (int, float)):
            continue
        delta = float(v1_value) - float(v0_value)
        metric_deltas[key] = delta
        if direction == "higher_better":
            if delta > 1e-8:
                improvements.append(f"{key}: +{delta:.4f}")
            elif delta < -1e-8:
                regressions.append(f"{key}: {delta:.4f}")
        else:
            if delta < -1e-8:
                improvements.append(f"{key}: {delta:.4f}")
            elif delta > 1e-8:
                regressions.append(f"{key}: +{delta:.4f}")

    scene_deltas: list[dict[str, Any]] = []
    improved_scenes: list[str] = []
    regressed_scenes: list[str] = []
    for scene in scenes:
        by_setting = {setting.setting_name: setting for setting in scene.settings}
        v0_setting = by_setting.get("calibration_v0")
        v1_setting = by_setting.get("calibration_v1")
        if v0_setting is None or v1_setting is None:
            continue
        if v0_setting.status != "success" or v1_setting.status != "success":
            continue

        v0_metrics = v0_setting.metrics
        v1_metrics = v1_setting.metrics
        relation_delta = _delta(v1_metrics.get("actionable_relation_f1"), v0_metrics.get("actionable_relation_f1"))
        violation_delta = _delta(
            v1_metrics.get("structured_violation_count_before_repair"),
            v0_metrics.get("structured_violation_count_before_repair"),
        )
        scene_deltas.append(
            {
                "scene_id": scene.scene_id,
                "relation_f1_delta_v1_minus_v0": relation_delta,
                "violation_before_delta_v1_minus_v0": violation_delta,
            }
        )
        if (relation_delta is not None and relation_delta > 1e-8) or (
            violation_delta is not None and violation_delta < -1e-8
        ):
            improved_scenes.append(scene.scene_id)
        if (relation_delta is not None and relation_delta < -1e-8) or (
            violation_delta is not None and violation_delta > 1e-8
        ):
            regressed_scenes.append(scene.scene_id)

    calibration_failures = failure_summary.get("calibration_failure_counts", {})
    if not isinstance(calibration_failures, dict):
        calibration_failures = {}
    top_calibration_failures = sorted(
        ((str(label), int(count)) for label, count in calibration_failures.items()),
        key=lambda item: item[1],
        reverse=True,
    )[:5]

    return {
        "v0_vs_v1_metric_deltas": metric_deltas,
        "v1_plus_repair_relation_f1": v1_repair.get("actionable_relation_f1"),
        "improvements_over_v0": improvements,
        "regressions_vs_v0": regressions,
        "scene_level_deltas": scene_deltas,
        "improved_scene_ids": sorted(set(improved_scenes)),
        "regressed_scene_ids": sorted(set(regressed_scenes)),
        "fallback_scene_ids": list(v1_execution_summary.get("fallback_scene_ids", [])),
        "top_calibration_failure_categories": top_calibration_failures,
    }


def _build_external_propagation_summary(scenes: list[SceneEvaluation]) -> dict[str, Any]:
    status_counts: dict[str, int] = {"success": 0, "failed": 0, "skipped": 0, "missing": 0}
    skip_reason_counts: dict[str, int] = {}
    failure_counts: dict[str, int] = {}
    scene_rows: list[dict[str, Any]] = []

    propagation_detected_count = 0
    export_difference_count = 0
    violation_deltas: list[float] = []
    comparable_external_vs_mock_count = 0

    for scene in scenes:
        by_name = {setting.setting_name: setting for setting in scene.settings}
        external = by_name.get("external_generator")
        mock = by_name.get("mock_generator")
        v0 = by_name.get("calibration_v0")
        v1 = by_name.get("calibration_v1")

        if external is None:
            status = "missing"
            status_counts[status] += 1
            scene_rows.append(
                {
                    "scene_id": scene.scene_id,
                    "external_status": status,
                    "notes": ["external setting missing"],
                }
            )
            continue

        status = external.status
        status_counts[status] = status_counts.get(status, 0) + 1
        for failure in external.failures:
            failure_key = str(failure)
            failure_counts[failure_key] = failure_counts.get(failure_key, 0) + 1
        if status == "skipped":
            skip_reason = str(
                external.metadata.get("skip_reason")
                if isinstance(external.metadata, dict)
                else "unknown"
            )
            skip_reason_counts[skip_reason] = skip_reason_counts.get(skip_reason, 0) + 1

        external_prediction = _ensure_dict(
            external.metadata.get("prediction_summary_pre_repair")
            if isinstance(external.metadata, dict)
            else None
        )
        mock_prediction = _ensure_dict(
            mock.metadata.get("prediction_summary_pre_repair")
            if isinstance(mock, SettingEvaluation) and isinstance(mock.metadata, dict)
            else None
        )
        prediction_delta_external_minus_mock: dict[str, Any] | None = None
        violation_delta: float | None = None

        external_comparable = (
            status == "success"
            and isinstance(mock, SettingEvaluation)
            and mock.status == "success"
        )
        if external_comparable:
            comparable_external_vs_mock_count += 1
            prediction_delta_external_minus_mock = _prediction_summary_delta(external_prediction, mock_prediction)

            external_violations = external.metrics.get("structured_violation_count_before_repair")
            mock_violations = mock.metrics.get("structured_violation_count_before_repair")
            violation_delta = _delta(external_violations, mock_violations)
            if isinstance(violation_delta, float):
                violation_deltas.append(violation_delta)

            if (
                prediction_delta_external_minus_mock.get("object_count_delta") != 0
                or prediction_delta_external_minus_mock.get("relation_count_delta") != 0
                or prediction_delta_external_minus_mock.get("added_labels")
                or prediction_delta_external_minus_mock.get("removed_labels")
                or prediction_delta_external_minus_mock.get("added_predicates")
                or prediction_delta_external_minus_mock.get("removed_predicates")
            ):
                propagation_detected_count += 1

        v0_input = _ensure_dict(
            v0.metadata.get("calibrated_input_summary")
            if isinstance(v0, SettingEvaluation) and isinstance(v0.metadata, dict)
            else None
        )
        v1_input = _ensure_dict(
            v1.metadata.get("calibrated_input_summary")
            if isinstance(v1, SettingEvaluation) and isinstance(v1.metadata, dict)
            else None
        )
        calibrated_input_delta = _calibrated_input_summary_delta(v1_input, v0_input)
        if calibrated_input_delta.get("has_difference"):
            export_difference_count += 1

        scene_rows.append(
            {
                "scene_id": scene.scene_id,
                "external_status": status,
                "external_failures": [str(item) for item in external.failures],
                "external_notes": [str(item) for item in external.notes],
                "external_generator_execution_summary": _ensure_dict(
                    external.metadata.get("generator_execution_summary")
                    if isinstance(external.metadata, dict)
                    else None
                ),
                "external_prediction_summary_pre_repair": external_prediction,
                "mock_prediction_summary_pre_repair": mock_prediction,
                "prediction_delta_external_minus_mock": prediction_delta_external_minus_mock,
                "violation_delta_external_minus_mock": violation_delta,
                "calibrated_input_delta_v1_minus_v0": calibrated_input_delta,
            }
        )

    num_scenes = len(scenes)
    success_count = status_counts.get("success", 0)
    summary = {
        "num_scenes": num_scenes,
        "status_counts": status_counts,
        "external_path_executed": success_count > 0,
        "external_path_fully_executed": success_count == num_scenes and num_scenes > 0,
        "num_comparable_external_vs_mock_scenes": comparable_external_vs_mock_count,
        "skip_reason_counts": skip_reason_counts,
        "failure_counts": failure_counts,
        "num_scenes_with_calibrated_input_difference_v1_vs_v0": export_difference_count,
        "num_scenes_with_detectable_external_prediction_delta_vs_mock": propagation_detected_count,
        "avg_pre_repair_violation_delta_external_minus_mock": _mean_or_none(violation_deltas),
        "scene_rows": scene_rows,
    }

    if success_count == 0:
        summary["status_note"] = (
            "External SpatialLM path was not executed successfully in this eval-pack run; "
            "external propagation claims are not yet validated."
        )
    elif comparable_external_vs_mock_count == 0:
        summary["status_note"] = (
            "External SpatialLM path executed, but no comparable external-vs-mock scene pairs were available."
        )
    elif propagation_detected_count == 0:
        summary["status_note"] = (
            "External SpatialLM path executed, but no clear pre-repair prediction delta versus mock was detected."
        )
    else:
        summary["status_note"] = (
            "External SpatialLM path executed with detectable pre-repair prediction deltas."
        )

    return summary


def _render_report_markdown(report: EvaluationPackReport) -> str:
    lines = [
        f"# Evaluation Pack Report: {report.manifest_name}",
        "",
        f"- generated_at: `{report.generated_at}`",
        f"- num_scenes: `{len(report.scenes)}`",
    ]
    if report.comparison_warning:
        lines.extend(
            [
                "",
                "## Trustworthiness Warning",
                f"- {report.comparison_warning}",
            ]
        )
    status = report.trustworthy_comparison_status
    if isinstance(status, dict) and status:
        lines.extend(
            [
                "",
                "## Trustworthy Comparison Status",
                f"- is_trustworthy_v0_v1_comparison: `{status.get('is_trustworthy_v0_v1_comparison')}`",
                f"- num_true_v1_execution_scenes: `{status.get('num_true_v1_execution_scenes')}`",
                f"- num_fallback_scenes: `{status.get('num_fallback_scenes')}`",
                f"- num_partial_calibration_scenes: `{status.get('num_partial_calibration_scenes')}`",
                f"- fallback_only_warning_active: `{status.get('fallback_only_warning_active')}`",
                f"- conclusions_provisional: `{status.get('conclusions_provisional')}`",
                f"- status_note: `{status.get('status_note')}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Readiness Summary",
            f"- point_cloud_loading_ready: `{report.environment_readiness.get('point_cloud_loading_ready')}`",
            f"- true_calibration_v1_ready: `{report.environment_readiness.get('true_calibration_v1_ready')}`",
            f"- tests_ready: `{report.environment_readiness.get('tests_ready')}`",
            f"- external_spatiallm_ready: `{report.environment_readiness.get('external_spatiallm_ready')}`",
            f"- v0_v1_method_comparison_ready: `{report.environment_readiness.get('v0_v1_method_comparison_ready')}`",
            "",
            "## Next Actions",
        ]
    )
    if report.next_actions:
        for action in report.next_actions:
            lines.append(f"- {action}")
    else:
        lines.append("- none")

    first_result = report.first_research_result_summary
    if isinstance(first_result, dict) and first_result:
        lines.extend(
            [
                "",
                "## First Research Result",
                f"- comparison_trustworthy: `{first_result.get('comparison_trustworthy')}`",
                f"- trustworthiness_note: `{first_result.get('trustworthiness_note')}`",
                f"- comparison_basis: `{first_result.get('comparison_basis')}`",
                f"- num_true_v1_scenes: `{first_result.get('num_true_v1_scenes')}`",
                f"- num_fallback_scenes: `{first_result.get('num_fallback_scenes')}`",
                f"- improved_metrics: `{first_result.get('improved_metrics')}`",
                f"- regressed_metrics: `{first_result.get('regressed_metrics')}`",
                f"- recommended_next_calibration_improvement_target: "
                f"`{first_result.get('recommended_next_calibration_improvement_target')}`",
            ]
        )

    lines.extend(
        [
            "",
            "## Per-Setting Aggregate Metrics",
        ]
    )

    for setting_name in sorted(report.aggregate_by_setting.keys()):
        summary = report.aggregate_by_setting[setting_name]
        lines.append(f"- `{setting_name}`:")
        for key in sorted(summary.keys()):
            lines.append(f"- `{setting_name}` {key}: {summary[key]}")

    lines.extend(
        [
            "",
            "## Comparison Table",
        ]
    )
    for row in report.comparison_table:
        setting_name = row.get("setting_name")
        lines.append(
            f"- `{setting_name}` up_err={row.get('calibration_up_axis_error_deg')} "
            f"h_err={row.get('calibration_horizontal_error_deg')} "
            f"up_conf={row.get('calibration_up_axis_confidence')} "
            f"h_conf={row.get('calibration_horizontal_confidence')} "
            f"rel={row.get('calibration_reliability')} "
            f"cat_f1={row.get('structured_category_presence_f1')} "
            f"relation_f1={row.get('actionable_relation_f1')}"
        )

    lines.extend(
        [
            "",
            "## V1 Execution Summary",
            f"- num_true_v1_execution: `{report.v1_execution_summary.get('num_true_v1_execution')}`",
            f"- num_fallback_used: `{report.v1_execution_summary.get('num_fallback_used')}`",
            f"- avg_calibration_reliability: `{report.v1_execution_summary.get('avg_calibration_reliability')}`",
            f"- avg_manhattan_ambiguity: `{report.v1_execution_summary.get('avg_manhattan_ambiguity')}`",
            f"- avg_scale_drift: `{report.v1_execution_summary.get('avg_scale_drift')}`",
            f"- fallback_scene_ids: `{report.v1_execution_summary.get('fallback_scene_ids')}`",
            f"- downstream_violation_counts_by_setting: `{report.v1_execution_summary.get('downstream_violation_counts_by_setting')}`",
        ]
    )

    external = report.external_propagation_summary
    if isinstance(external, dict) and external:
        lines.extend(
            [
                "",
                "## External Propagation Summary",
                f"- external_path_executed: `{external.get('external_path_executed')}`",
                f"- external_path_fully_executed: `{external.get('external_path_fully_executed')}`",
                f"- status_counts: `{external.get('status_counts')}`",
                f"- skip_reason_counts: `{external.get('skip_reason_counts')}`",
                f"- num_comparable_external_vs_mock_scenes: "
                f"`{external.get('num_comparable_external_vs_mock_scenes')}`",
                f"- num_scenes_with_calibrated_input_difference_v1_vs_v0: "
                f"`{external.get('num_scenes_with_calibrated_input_difference_v1_vs_v0')}`",
                f"- num_scenes_with_detectable_external_prediction_delta_vs_mock: "
                f"`{external.get('num_scenes_with_detectable_external_prediction_delta_vs_mock')}`",
                f"- avg_pre_repair_violation_delta_external_minus_mock: "
                f"`{external.get('avg_pre_repair_violation_delta_external_minus_mock')}`",
                f"- status_note: `{external.get('status_note')}`",
            ]
        )

    lines.extend(["", "## Per-Scene Setting Status"])
    for scene in report.scenes:
        lines.append(f"- scene `{scene.scene_id}`:")
        for setting in scene.settings:
            metadata = setting.metadata if isinstance(setting.metadata, dict) else {}
            prediction_summary = metadata.get("prediction_summary_pre_repair", {})
            propagation = metadata.get("propagation_diagnostics", {})
            if not isinstance(prediction_summary, dict):
                prediction_summary = {}
            if not isinstance(propagation, dict):
                propagation = {}
            lines.append(
                f"- scene `{scene.scene_id}` `{setting.setting_name}` "
                f"status={setting.status} failures={setting.failures} "
                f"pre_objects={prediction_summary.get('object_count')} "
                f"pre_relations={prediction_summary.get('relation_count')} "
                f"propagation_mode={propagation.get('generator_mode')}"
            )

    return "\n".join(lines)


def _render_v0_v1_comparison_markdown(
    summary: dict[str, Any],
    *,
    comparison_warning: str | None = None,
    next_actions: list[str] | None = None,
) -> str:
    improvements = summary.get("improvements_over_v0", [])
    regressions = summary.get("regressions_vs_v0", [])
    fallback_scene_ids = summary.get("fallback_scene_ids", [])
    top_failures = summary.get("top_calibration_failure_categories", [])

    lines = [
        "# v0 vs v1 Comparison Summary",
    ]
    if comparison_warning:
        lines.extend(["", "## Trustworthiness Warning", f"- {comparison_warning}"])
    trust_status = summary.get("trustworthy_comparison_status", {})
    if isinstance(trust_status, dict) and trust_status:
        lines.extend(
            [
                "",
                "## Trustworthy Comparison Status",
                f"- is_trustworthy_v0_v1_comparison: {trust_status.get('is_trustworthy_v0_v1_comparison')}",
                f"- num_true_v1_execution_scenes: {trust_status.get('num_true_v1_execution_scenes')}",
                f"- num_fallback_scenes: {trust_status.get('num_fallback_scenes')}",
                f"- num_partial_calibration_scenes: {trust_status.get('num_partial_calibration_scenes')}",
                f"- fallback_only_warning_active: {trust_status.get('fallback_only_warning_active')}",
                f"- conclusions_provisional: {trust_status.get('conclusions_provisional')}",
                f"- status_note: {trust_status.get('status_note')}",
            ]
        )

    lines.extend(["", "## Improvements Over v0"])
    if isinstance(improvements, list) and improvements:
        for item in improvements:
            lines.append(f"- {item}")
    else:
        lines.append("- no aggregate improvements detected")

    lines.extend(["", "## Regressions vs v0"])
    if isinstance(regressions, list) and regressions:
        for item in regressions:
            lines.append(f"- {item}")
    else:
        lines.append("- no aggregate regressions detected")

    lines.extend(["", "## Fallback Scenes"])
    if isinstance(fallback_scene_ids, list) and fallback_scene_ids:
        for scene_id in fallback_scene_ids:
            lines.append(f"- {scene_id}")
    else:
        lines.append("- none")

    lines.extend(["", "## Top Calibration Failure Categories"])
    if isinstance(top_failures, list) and top_failures:
        for item in top_failures:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                lines.append(f"- {item[0]}: {item[1]}")
    else:
        lines.append("- none")

    lines.extend(["", "## Scene-Level Deltas"])
    scene_deltas = summary.get("scene_level_deltas", [])
    if isinstance(scene_deltas, list) and scene_deltas:
        for row in scene_deltas:
            if not isinstance(row, dict):
                continue
            lines.append(
                f"- {row.get('scene_id')}: relation_f1_delta={row.get('relation_f1_delta_v1_minus_v0')}, "
                f"violation_before_delta={row.get('violation_before_delta_v1_minus_v0')}"
            )
    else:
        lines.append("- no scene-level deltas available")

    recommendation = summary.get("recommended_next_calibration_improvement", {})
    if isinstance(recommendation, dict) and recommendation:
        lines.extend(
            [
                "",
                "## Next Calibration Target",
                f"- target: {recommendation.get('target')}",
                f"- reason: {recommendation.get('reason')}",
            ]
        )

    lines.extend(["", "## Recommended Next Actions"])
    if next_actions:
        for action in next_actions:
            lines.append(f"- {action}")
    else:
        lines.append("- none")

    return "\n".join(lines)


def _with_warning_banner(content: str, warning: str | None) -> str:
    if not warning:
        return content
    banner_lines = [
        "# Trustworthiness Warning",
        "",
        f"- {warning}",
        "",
    ]
    return "\n".join(banner_lines) + content


def _render_readiness_summary_markdown(
    *,
    readiness_summary: dict[str, Any],
    comparison_warning: str | None,
    next_actions: list[str],
) -> str:
    lines = [
        "# Readiness Summary",
        "",
        f"- point_cloud_loading_ready: `{readiness_summary.get('point_cloud_loading_ready')}`",
        f"- true_calibration_v1_ready: `{readiness_summary.get('true_calibration_v1_ready')}`",
        f"- tests_ready: `{readiness_summary.get('tests_ready')}`",
        f"- external_spatiallm_ready: `{readiness_summary.get('external_spatiallm_ready')}`",
        f"- v0_v1_method_comparison_ready: `{readiness_summary.get('v0_v1_method_comparison_ready')}`",
    ]
    blocking = readiness_summary.get("blocking_reasons", [])
    if isinstance(blocking, list) and blocking:
        lines.extend(["", "## Blocking Reasons"])
        for reason in blocking:
            lines.append(f"- {reason}")

    if comparison_warning:
        lines.extend(["", "## Trustworthiness Warning", f"- {comparison_warning}"])

    lines.extend(["", "## Next Actions"])
    if next_actions:
        for action in next_actions:
            lines.append(f"- {action}")
    else:
        lines.append("- none")

    return "\n".join(lines)


def _render_external_propagation_summary_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# External Propagation Summary",
        "",
        f"- num_scenes: `{summary.get('num_scenes')}`",
        f"- external_path_executed: `{summary.get('external_path_executed')}`",
        f"- external_path_fully_executed: `{summary.get('external_path_fully_executed')}`",
        f"- status_counts: `{summary.get('status_counts')}`",
        f"- skip_reason_counts: `{summary.get('skip_reason_counts')}`",
        f"- failure_counts: `{summary.get('failure_counts')}`",
        f"- num_comparable_external_vs_mock_scenes: `{summary.get('num_comparable_external_vs_mock_scenes')}`",
        f"- num_scenes_with_calibrated_input_difference_v1_vs_v0: "
        f"`{summary.get('num_scenes_with_calibrated_input_difference_v1_vs_v0')}`",
        f"- num_scenes_with_detectable_external_prediction_delta_vs_mock: "
        f"`{summary.get('num_scenes_with_detectable_external_prediction_delta_vs_mock')}`",
        f"- avg_pre_repair_violation_delta_external_minus_mock: "
        f"`{summary.get('avg_pre_repair_violation_delta_external_minus_mock')}`",
        f"- status_note: `{summary.get('status_note')}`",
        "",
        "## Per-Scene External Status",
    ]

    scene_rows = summary.get("scene_rows", [])
    if isinstance(scene_rows, list) and scene_rows:
        for row in scene_rows:
            if not isinstance(row, dict):
                continue
            lines.append(
                f"- scene `{row.get('scene_id')}` status={row.get('external_status')} "
                f"failures={row.get('external_failures')} "
                f"violation_delta_external_minus_mock={row.get('violation_delta_external_minus_mock')}"
            )
            lines.append(
                f"- scene `{row.get('scene_id')}` prediction_delta_external_minus_mock="
                f"{row.get('prediction_delta_external_minus_mock')}"
            )
            lines.append(
                f"- scene `{row.get('scene_id')}` calibrated_input_delta_v1_minus_v0="
                f"{row.get('calibrated_input_delta_v1_minus_v0')}"
            )
    else:
        lines.append("- none")

    return "\n".join(lines) + "\n"


def _ensure_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _prediction_summary_delta(current: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    current_labels = set(_string_list(current.get("object_labels", [])))
    baseline_labels = set(_string_list(baseline.get("object_labels", [])))
    current_predicates = set(_string_list(current.get("relation_predicates", [])))
    baseline_predicates = set(_string_list(baseline.get("relation_predicates", [])))

    return {
        "object_count_delta": _as_int(current.get("object_count")) - _as_int(baseline.get("object_count")),
        "relation_count_delta": _as_int(current.get("relation_count")) - _as_int(baseline.get("relation_count")),
        "added_labels": sorted(current_labels - baseline_labels),
        "removed_labels": sorted(baseline_labels - current_labels),
        "added_predicates": sorted(current_predicates - baseline_predicates),
        "removed_predicates": sorted(baseline_predicates - current_predicates),
    }


def _calibrated_input_summary_delta(current: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    frame_current = current.get("frame")
    frame_baseline = baseline.get("frame")
    num_points_current = _as_int(current.get("num_points"))
    num_points_baseline = _as_int(baseline.get("num_points"))
    up_current = _ensure_dict(current.get("up_vector"))
    up_baseline = _ensure_dict(baseline.get("up_vector"))
    horizontal_current = _ensure_dict(current.get("horizontal_axis"))
    horizontal_baseline = _ensure_dict(baseline.get("horizontal_axis"))

    return {
        "frame_changed": frame_current != frame_baseline,
        "num_points_delta": num_points_current - num_points_baseline,
        "up_vector_changed": up_current != up_baseline,
        "horizontal_axis_changed": horizontal_current != horizontal_baseline,
        "ranges_changed": current.get("ranges") != baseline.get("ranges"),
        "has_difference": any(
            [
                frame_current != frame_baseline,
                num_points_current != num_points_baseline,
                up_current != up_baseline,
                horizontal_current != horizontal_baseline,
                current.get("ranges") != baseline.get("ranges"),
            ]
        ),
    }


def _string_list(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    return [str(value) for value in values]


def _as_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _delta(current: Any, baseline: Any) -> float | None:
    if not isinstance(current, (int, float)) or not isinstance(baseline, (int, float)):
        return None
    return float(current) - float(baseline)
