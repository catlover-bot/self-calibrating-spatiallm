"""Single-scene end-to-end pipeline orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from self_calibrating_spatiallm.artifacts import (
    AblationReport,
    AblationSettingResult,
    ActionableScene,
    ArtifactStore,
    CalibratedPointCloud,
    CalibrationResult,
    EvaluationResult,
    PointCloudMetadata,
    PointCloudSample,
    RepairResult,
    ScenePrediction,
)
from self_calibrating_spatiallm.calibration import (
    Calibrator,
    GeometricCalibratorV0,
    NoCalibrationCalibrator,
    PlaneAwareCalibratorV1,
    extract_calibration_execution,
)
from self_calibrating_spatiallm.evaluation.simple_evaluator import SimpleSceneEvaluator
from self_calibrating_spatiallm.generation import (
    MockSpatialLMGenerator,
    SceneGenerator,
    SpatialLMExternalGenerator,
)
from self_calibrating_spatiallm.io import PointCloudLoadOptions, load_point_cloud_sample
from self_calibrating_spatiallm.pipeline.config import SceneInputConfig
from self_calibrating_spatiallm.repair import PassThroughRepairer, SceneRepairer, SimpleRuleRepairer
from self_calibrating_spatiallm.scene_graph import ActionableSceneBuilder, RuleBasedActionableSceneBuilder
from self_calibrating_spatiallm.visualization import render_pipeline_summary, render_qualitative_report


@dataclass
class SingleSceneRun:
    point_cloud_metadata: PointCloudMetadata
    sample: PointCloudSample
    calibrated_point_cloud: CalibratedPointCloud
    calibration: CalibrationResult
    prediction: ScenePrediction
    repair: RepairResult
    actionable_scene: ActionableScene
    evaluation: EvaluationResult
    ablation_report: AblationReport
    generator_raw_output: str | None = None
    generator_execution_info: dict[str, Any] | None = None
    stage_status: dict[str, dict[str, Any]] = field(default_factory=dict)
    started_at: str | None = None
    finished_at: str | None = None
    pipeline_status: str = "success"
    generator_mode: str = "mock"
    calibration_mode: str = "unknown"
    repair_mode: str = "unknown"

    def save(self, store: ArtifactStore) -> dict[str, Path]:
        paths = {
            "point_cloud_metadata": store.save_artifact(
                "00_point_cloud_metadata.json", self.point_cloud_metadata
            ),
            "point_cloud_sample": store.save_artifact("01_point_cloud_sample.json", self.sample),
            "calibrated_point_cloud": store.save_artifact(
                "02_calibrated_point_cloud.json", self.calibrated_point_cloud
            ),
            "calibration_result": store.save_artifact("03_calibration_result.json", self.calibration),
            "scene_prediction": store.save_artifact("04_scene_prediction.json", self.prediction),
            "repair_result": store.save_artifact("05_repair_result.json", self.repair),
            "actionable_scene": store.save_artifact("06_actionable_scene.json", self.actionable_scene),
            "evaluation_result": store.save_artifact("07_evaluation_result.json", self.evaluation),
            "ablation_report": store.save_artifact("08_ablation_report.json", self.ablation_report),
        }

        propagation_diagnostics = self.prediction.metadata.get("propagation_diagnostics", {})
        if isinstance(propagation_diagnostics, dict) and propagation_diagnostics:
            paths["propagation_diagnostics"] = store.save_json(
                "04a_propagation_diagnostics.json",
                propagation_diagnostics,
            )

        calibrated_metadata = {
            "sample_id": self.calibrated_point_cloud.sample_id,
            "num_points": self.calibrated_point_cloud.num_points,
            "frame": self.calibrated_point_cloud.metadata.get("frame"),
            "original_sensor_frame": self.calibrated_point_cloud.metadata.get("original_sensor_frame"),
            "transformed_point_cloud": self.calibrated_point_cloud.metadata.get("transformed_point_cloud"),
            "axis_convention": {
                "x_axis": "dominant horizontal direction",
                "y_axis": "horizontal side direction",
                "z_axis": "up",
                "right_handed": True,
            },
            "scale_assumptions": {
                "expected_unit": self.calibrated_point_cloud.metadata.get("expected_unit"),
                "inferred_scale_hint": self.calibrated_point_cloud.metadata.get("inferred_scale_hint"),
            },
            "normalization": {
                "applied": bool(self.calibration.metadata.get("normalization_applied", False)),
                "origin_offset": self.calibration.origin_offset.to_dict(),
                "rotation_matrix": self.calibration.metadata.get("rotation_matrix"),
            },
        }
        paths["calibrated_point_cloud_metadata"] = store.save_json(
            "02b_calibrated_point_cloud_metadata.json",
            calibrated_metadata,
        )

        if self.generator_raw_output:
            paths["generator_raw_output"] = store.save_text(
                "04b_generator_raw_output.txt", self.generator_raw_output
            )

        if self.generator_execution_info:
            execution_payload = dict(self.generator_execution_info)
            execution_raw_files = execution_payload.get("raw_files", {})
            if isinstance(execution_raw_files, dict):
                execution_payload["raw_files"] = {
                    key: {
                        "present": isinstance(value, str) and bool(value.strip()),
                        "char_count": len(value) if isinstance(value, str) else 0,
                    }
                    for key, value in execution_raw_files.items()
                }
            paths["generator_execution"] = store.save_json(
                "04c_generator_execution.json", execution_payload
            )
            stdout = str(self.generator_execution_info.get("stdout", "")).strip()
            stderr = str(self.generator_execution_info.get("stderr", "")).strip()
            if stdout:
                paths["generator_stdout"] = store.save_text("04d_generator_stdout.txt", stdout)
            if stderr:
                paths["generator_stderr"] = store.save_text("04e_generator_stderr.txt", stderr)

            raw_files = self.generator_execution_info.get("raw_files", {})
            if isinstance(raw_files, dict):
                export_text = raw_files.get("exported_spatiallm_input_text")
                output_json_text = raw_files.get("output_json_text")
                if isinstance(export_text, str) and export_text.strip():
                    paths["spatiallm_export_point_cloud"] = store.save_text(
                        "02c_spatiallm_export_point_cloud.json",
                        export_text,
                    )
                if isinstance(output_json_text, str) and output_json_text.strip():
                    paths["generator_output_raw"] = store.save_text(
                        "04f_generator_output_raw.json",
                        output_json_text,
                    )

        summary = render_pipeline_summary(
            point_cloud_metadata=self.point_cloud_metadata,
            sample=self.sample,
            calibrated=self.calibrated_point_cloud,
            calibration=self.calibration,
            scene=self.prediction,
            repair=self.repair,
            actionable=self.actionable_scene,
            evaluation=self.evaluation,
            ablation_report=self.ablation_report,
        )
        paths["summary"] = store.save_text("09_summary.txt", summary)

        run_manifest_snapshot = self._build_run_manifest(store=store, artifact_paths=paths)
        qualitative_report = render_qualitative_report(
            point_cloud_metadata=self.point_cloud_metadata,
            calibrated=self.calibrated_point_cloud,
            calibration=self.calibration,
            prediction=self.prediction,
            repair=self.repair,
            actionable_scene=self.actionable_scene,
            evaluation=self.evaluation,
            ablation_report=self.ablation_report,
            run_manifest=run_manifest_snapshot,
        )
        paths["qualitative_report"] = store.save_text("10_qualitative_report.md", qualitative_report)

        paths["manifest"] = store.save_manifest(paths)
        paths["run_manifest"] = store.root_dir / "run_manifest.json"
        final_run_manifest = self._build_run_manifest(store=store, artifact_paths=paths)
        paths["run_manifest"] = store.save_json("run_manifest.json", final_run_manifest)
        paths["manifest"] = store.save_manifest(paths)
        return paths

    def _build_run_manifest(self, store: ArtifactStore, artifact_paths: dict[str, Path]) -> dict[str, Any]:
        return {
            "scene_id": self.sample.sample_id,
            "status": self.pipeline_status,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "generator_mode": self.generator_mode,
            "calibration_mode": self.calibration_mode,
            "repair_mode": self.repair_mode,
            "calibration_execution": extract_calibration_execution(self.calibration),
            "stages": self.stage_status,
            "artifacts": {name: _relative_to_store(store, path) for name, path in artifact_paths.items()},
        }


class SingleScenePipeline:
    """Composable pipeline object for one-scene experiments."""

    def __init__(
        self,
        calibrator: Calibrator | None = None,
        generator: SceneGenerator | None = None,
        repairer: SceneRepairer | None = None,
        builder: ActionableSceneBuilder | None = None,
        evaluator: SimpleSceneEvaluator | None = None,
        comparison_generators: dict[str, SceneGenerator] | None = None,
    ) -> None:
        self.calibrator = calibrator or PlaneAwareCalibratorV1(normalize_scene=True)
        self.generator = generator or MockSpatialLMGenerator()
        self.repairer = repairer or SimpleRuleRepairer()
        self.builder = builder or RuleBasedActionableSceneBuilder()
        self.evaluator = evaluator or SimpleSceneEvaluator()
        self.comparison_generators = comparison_generators or {"mock_generator": MockSpatialLMGenerator()}

    def run(
        self,
        sample: PointCloudSample,
        metadata: PointCloudMetadata,
        started_at: str | None = None,
        stage_status: dict[str, dict[str, Any]] | None = None,
        generator_mode: str | None = None,
    ) -> SingleSceneRun:
        run_started_at = started_at or _timestamp()
        statuses = stage_status if stage_status is not None else {}

        calibrated = self._run_stage(statuses, "calibration", lambda: self.calibrator.calibrate(sample))
        prediction = self._run_stage(statuses, "generation", lambda: self.generator.generate(calibrated))
        if sample.metadata.get("room_bounds") and not prediction.metadata.get("room_bounds"):
            prediction.metadata["room_bounds"] = sample.metadata.get("room_bounds")

        repair = self._run_stage(statuses, "repair", lambda: self.repairer.repair(prediction))
        actionable_scene = self._run_stage(
            statuses,
            "scene_graph",
            lambda: self.builder.build(repair.repaired_scene),
        )
        evaluation = self._run_stage(
            statuses,
            "evaluation",
            lambda: self.evaluator.evaluate(
                repair.repaired_scene,
                repair,
                actionable_scene,
                calibration_method=calibrated.calibration.method,
                setting_name="primary",
            ),
        )
        ablations = self._run_stage(statuses, "ablation", lambda: self._run_ablations(sample))

        return SingleSceneRun(
            point_cloud_metadata=metadata,
            sample=sample,
            calibrated_point_cloud=calibrated,
            calibration=calibrated.calibration,
            prediction=prediction,
            repair=repair,
            actionable_scene=actionable_scene,
            evaluation=evaluation,
            ablation_report=ablations,
            generator_raw_output=self.generator.get_last_raw_output(),
            generator_execution_info=self.generator.get_last_execution_info(),
            stage_status=statuses,
            started_at=run_started_at,
            finished_at=_timestamp(),
            pipeline_status="success",
            generator_mode=generator_mode or _generator_mode_name(self.generator),
            calibration_mode=self.calibrator.name,
            repair_mode=self.repairer.name,
        )

    def _run_ablations(self, sample: PointCloudSample) -> AblationReport:
        normalize_scene = _resolve_normalize_scene(self.calibrator)
        settings = [
            ("no_calibration", NoCalibrationCalibrator(), PassThroughRepairer(), False, False),
            (
                "calibration_v0",
                GeometricCalibratorV0(normalize_scene=normalize_scene),
                PassThroughRepairer(),
                True,
                False,
            ),
            (
                "calibration_v1",
                PlaneAwareCalibratorV1(normalize_scene=normalize_scene),
                PassThroughRepairer(),
                True,
                False,
            ),
            (
                "calibration_v1_plus_repair",
                PlaneAwareCalibratorV1(normalize_scene=normalize_scene),
                SimpleRuleRepairer(),
                True,
                True,
            ),
        ]

        results: list[AblationSettingResult] = []
        for setting_name, calibrator, repairer, calibration_enabled, repair_enabled in settings:
            generator = _clone_generator(self.generator)

            try:
                calibrated = calibrator.calibrate(sample)
                prediction = generator.generate(calibrated)
                if sample.metadata.get("room_bounds") and not prediction.metadata.get("room_bounds"):
                    prediction.metadata["room_bounds"] = sample.metadata.get("room_bounds")
                repair = repairer.repair(prediction)
                actionable = self.builder.build(repair.repaired_scene)
                evaluation = self.evaluator.evaluate(
                    repair.repaired_scene,
                    repair,
                    actionable,
                    calibration_method=calibrated.calibration.method,
                    setting_name=setting_name,
                )
                metadata = {
                    "generator_name": prediction.generator_name,
                    "generator_execution": generator.get_last_execution_info(),
                }
            except Exception as error:
                evaluation = EvaluationResult(
                    sample_id=sample.sample_id,
                    evaluator_name=self.evaluator.name,
                    metrics={},
                    passed=False,
                    notes=[f"ablation_error: {error}"],
                    metadata={"setting_name": setting_name},
                )
                metadata = {"error": str(error)}

            results.append(
                AblationSettingResult(
                    setting_name=setting_name,
                    calibration_enabled=calibration_enabled,
                    repair_enabled=repair_enabled,
                    calibration_method=calibrator.name,
                    repairer_name=repairer.name,
                    evaluation=evaluation,
                    metadata=metadata,
                )
            )

        generator_results: list[AblationSettingResult] = []
        for setting_name, configured_generator in self.comparison_generators.items():
            generator = _clone_generator(configured_generator)
            calibrator = PlaneAwareCalibratorV1(normalize_scene=normalize_scene)
            repairer = SimpleRuleRepairer()
            try:
                calibrated = calibrator.calibrate(sample)
                prediction = generator.generate(calibrated)
                if sample.metadata.get("room_bounds") and not prediction.metadata.get("room_bounds"):
                    prediction.metadata["room_bounds"] = sample.metadata.get("room_bounds")
                repair = repairer.repair(prediction)
                actionable = self.builder.build(repair.repaired_scene)
                evaluation = self.evaluator.evaluate(
                    repair.repaired_scene,
                    repair,
                    actionable,
                    calibration_method=calibrated.calibration.method,
                    setting_name=setting_name,
                )
                metadata = {
                    "generator_name": prediction.generator_name,
                    "generator_execution": generator.get_last_execution_info(),
                }
            except Exception as error:
                evaluation = EvaluationResult(
                    sample_id=sample.sample_id,
                    evaluator_name=self.evaluator.name,
                    metrics={},
                    passed=False,
                    notes=[f"generator_ablation_error: {error}"],
                    metadata={"setting_name": setting_name},
                )
                metadata = {"error": str(error)}

            generator_results.append(
                AblationSettingResult(
                    setting_name=setting_name,
                    calibration_enabled=True,
                    repair_enabled=True,
                    calibration_method=calibrator.name,
                    repairer_name=repairer.name,
                    evaluation=evaluation,
                    metadata=metadata,
                )
            )

        return AblationReport(
            sample_id=sample.sample_id,
            settings=results,
            generator_settings=generator_results,
            metadata={"baseline": "single-scene-ablation-v1"},
        )

    @staticmethod
    def _run_stage(
        statuses: dict[str, dict[str, Any]],
        stage_name: str,
        fn: Callable[[], Any],
    ) -> Any:
        started_at = _timestamp()
        try:
            value = fn()
            statuses[stage_name] = {
                "status": "success",
                "started_at": started_at,
                "finished_at": _timestamp(),
                "error": None,
            }
            return value
        except Exception as error:
            statuses[stage_name] = {
                "status": "failed",
                "started_at": started_at,
                "finished_at": _timestamp(),
                "error": str(error),
            }
            raise


def run_single_scene_pipeline(config: SceneInputConfig, output_dir: Path | None = None) -> dict[str, Path]:
    started_at = _timestamp()
    stage_status: dict[str, dict[str, Any]] = {}

    sample, point_cloud_metadata = _run_loading_stage(config, stage_status)

    generator = _build_generator(config)
    comparison_generators = _build_comparison_generators(config)
    calibrator = _build_calibrator(config)

    pipeline = SingleScenePipeline(
        calibrator=calibrator,
        generator=generator,
        repairer=SimpleRuleRepairer(),
        builder=RuleBasedActionableSceneBuilder(),
        evaluator=SimpleSceneEvaluator(),
        comparison_generators=comparison_generators,
    )

    store = ArtifactStore(config.resolve_output_dir(output_dir))
    try:
        run = pipeline.run(
            sample=sample,
            metadata=point_cloud_metadata,
            started_at=started_at,
            stage_status=stage_status,
            generator_mode=config.generator_mode,
        )
        return run.save(store)
    except Exception as error:
        store.save_json(
            "run_manifest.json",
            {
                "scene_id": config.scene_id,
                "status": "failed",
                "started_at": started_at,
                "finished_at": _timestamp(),
                "generator_mode": config.generator_mode,
                "calibration_mode": pipeline.calibrator.name,
                "repair_mode": pipeline.repairer.name,
                "calibration_execution": None,
                "stages": stage_status,
                "error": str(error),
                "artifacts": {},
            },
        )
        raise


def run_single_scene_pipeline_from_config_path(
    config_path: Path,
    output_dir: Path | None = None,
) -> dict[str, Path]:
    config = SceneInputConfig.load_json(config_path)
    return run_single_scene_pipeline(config=config, output_dir=output_dir)


def _run_loading_stage(
    config: SceneInputConfig,
    stage_status: dict[str, dict[str, Any]],
) -> tuple[PointCloudSample, PointCloudMetadata]:
    started_at = _timestamp()
    try:
        sample, point_cloud_metadata = load_point_cloud_sample(
            file_path=config.resolve_file_path(),
            options=PointCloudLoadOptions(
                scene_id=config.scene_id,
                source_type=config.source_type,
                metadata_path=config.resolve_metadata_path(),
                expected_unit=config.expected_unit,
                scale_hint=config.scale_hint,
            ),
        )
        stage_status["loading"] = {
            "status": "success",
            "started_at": started_at,
            "finished_at": _timestamp(),
            "error": None,
        }
        return sample, point_cloud_metadata
    except Exception as error:
        stage_status["loading"] = {
            "status": "failed",
            "started_at": started_at,
            "finished_at": _timestamp(),
            "error": str(error),
        }
        raise


def _build_generator(config: SceneInputConfig) -> SceneGenerator:
    mode = config.generator_mode.lower().strip()
    if mode == "mock":
        return MockSpatialLMGenerator()
    if mode == "external":
        return _build_external_generator(config)

    raise ValueError(f"Unsupported generator_mode: {config.generator_mode}")


def _build_external_generator(config: SceneInputConfig) -> SpatialLMExternalGenerator:
    return SpatialLMExternalGenerator(
        command=config.spatiallm_command,
        timeout_sec=config.external_timeout_sec,
        output_json_path=config.resolve_external_output_json(),
        command_env_var=config.spatiallm_command_env_var,
        export_format=config.spatiallm_export_format,
    )


def _build_calibrator(config: SceneInputConfig) -> Calibrator:
    mode = config.calibration_mode.strip().lower()
    if mode == "none":
        return NoCalibrationCalibrator()
    if mode == "v0":
        return GeometricCalibratorV0(normalize_scene=config.normalize_scene)
    if mode == "v1":
        return PlaneAwareCalibratorV1(normalize_scene=config.normalize_scene)
    raise ValueError(f"Unsupported calibration_mode: {config.calibration_mode}")


def _build_comparison_generators(config: SceneInputConfig) -> dict[str, SceneGenerator]:
    generators: dict[str, SceneGenerator] = {"mock_generator": MockSpatialLMGenerator()}

    should_add_external = (
        config.generator_mode.lower().strip() == "external"
        or config.compare_with_external_generator
        or config.has_external_generator_config()
    )
    if should_add_external:
        generators["external_generator"] = _build_external_generator(config)

    return generators


def _clone_generator(generator: SceneGenerator) -> SceneGenerator:
    if isinstance(generator, MockSpatialLMGenerator):
        return MockSpatialLMGenerator()

    if isinstance(generator, SpatialLMExternalGenerator):
        return SpatialLMExternalGenerator(
            command=generator.command,
            timeout_sec=generator.timeout_sec,
            output_json_path=generator.output_json_path,
            command_env_var=generator.command_env_var,
            export_format=generator.export_format,
        )

    return generator


def _resolve_normalize_scene(calibrator: Calibrator) -> bool:
    normalize = getattr(calibrator, "normalize_scene", True)
    return bool(normalize)


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _relative_to_store(store: ArtifactStore, path: Path) -> str:
    try:
        return str(path.relative_to(store.root_dir))
    except ValueError:
        return str(path)


def _generator_mode_name(generator: SceneGenerator) -> str:
    if isinstance(generator, SpatialLMExternalGenerator):
        return "external"
    if isinstance(generator, MockSpatialLMGenerator):
        return "mock"
    return generator.name
