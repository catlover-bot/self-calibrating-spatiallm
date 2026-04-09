import json
import subprocess
from pathlib import Path

from self_calibrating_spatiallm.artifacts import (
    CalibratedPointCloud,
    CalibrationResult,
    Point3D,
    PointCloudSample,
)
from self_calibrating_spatiallm.calibration import NoCalibrationCalibrator
from self_calibrating_spatiallm.generation import SpatialLMExternalGenerator


def _build_calibrated(
    *,
    sample_id: str,
    horizontal_axis: Point3D,
    normalization_applied: bool,
) -> CalibratedPointCloud:
    points = [Point3D(0.0, 0.0, 0.0), Point3D(1.0, 0.0, 0.0), Point3D(0.0, 2.0, 0.5)]
    calibration = CalibrationResult(
        sample_id=sample_id,
        method="unit_test",
        up_vector=Point3D(0.0, 0.0, 1.0),
        horizontal_axis=horizontal_axis,
        origin_offset=Point3D(0.0, 0.0, 0.0),
        metadata={"normalization_applied": normalization_applied},
    )
    return CalibratedPointCloud(
        sample_id=sample_id,
        points=points,
        calibration=calibration,
        metadata={"frame": "canonical"},
    )


def test_external_export_differences_preserved_across_calibration_settings(monkeypatch) -> None:
    calibrated_v0 = _build_calibrated(
        sample_id="scene-export",
        horizontal_axis=Point3D(1.0, 0.0, 0.0),
        normalization_applied=False,
    )
    calibrated_v1 = _build_calibrated(
        sample_id="scene-export",
        horizontal_axis=Point3D(0.0, 1.0, 0.0),
        normalization_applied=True,
    )

    captured_exports: list[dict] = []

    def fake_run_command(self: SpatialLMExternalGenerator, command: list[str]) -> subprocess.CompletedProcess[str]:
        input_index = command.index("--input") + 1
        input_path = Path(command[input_index])
        captured_exports.append(json.loads(input_path.read_text(encoding="utf-8")))
        stdout = json.dumps({"scene_prediction": {"objects": [], "relations": []}})
        return subprocess.CompletedProcess(args=command, returncode=0, stdout=stdout, stderr="")

    monkeypatch.setattr(SpatialLMExternalGenerator, "_run_command", fake_run_command)

    generator = SpatialLMExternalGenerator(command=["spatiallm-cli", "--input", "{spatiallm_input}"])
    generator.generate(calibrated_v0)
    generator.generate(calibrated_v1)
    info = generator.get_last_execution_info()

    assert len(captured_exports) == 2
    assert (
        captured_exports[0]["axis_convention"]["horizontal_axis"]
        != captured_exports[1]["axis_convention"]["horizontal_axis"]
    )
    assert (
        captured_exports[0]["normalization"]["applied"]
        != captured_exports[1]["normalization"]["applied"]
    )
    assert info is not None
    assert info["spatiallm_export"]["payload_summary"]["horizontal_axis"] == captured_exports[1]["axis_convention"][
        "horizontal_axis"
    ]
    assert info["spatiallm_export"]["payload_summary"]["normalization_applied"] is True


def test_external_propagation_diagnostics_generation(monkeypatch) -> None:
    sample = PointCloudSample(
        sample_id="scene-external-diagnostics",
        points=[Point3D(0.0, 0.0, 0.0), Point3D(1.0, 0.0, 0.0), Point3D(0.0, 1.0, 0.0)],
    )
    calibrated = NoCalibrationCalibrator().calibrate(sample)

    payload = {
        "scene_prediction": {
            "sample_id": sample.sample_id,
            "generator_name": "external-test",
            "objects": [
                {
                    "object_id": "obj_floor",
                    "label": "floor",
                    "position": {"x": 0.0, "y": 0.0, "z": 0.0},
                    "size": {"x": 3.0, "y": 3.0, "z": 0.1},
                }
            ],
            "relations": [],
        }
    }

    def fake_run_command(self: SpatialLMExternalGenerator, command: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args=command, returncode=0, stdout=json.dumps(payload), stderr="")

    monkeypatch.setattr(SpatialLMExternalGenerator, "_run_command", fake_run_command)

    generator = SpatialLMExternalGenerator(command=["spatiallm-cli", "--input", "{spatiallm_input}"])
    prediction = generator.generate(calibrated)
    info = generator.get_last_execution_info()

    diagnostics = prediction.metadata.get("propagation_diagnostics", {})
    assert diagnostics["generator_mode"] == "external"
    assert diagnostics["parse_mode"] == "full_schema"
    assert diagnostics["parse_warning_count"] == 0
    assert diagnostics["prediction_summary"]["object_count"] == 1
    assert isinstance(diagnostics["spatiallm_export_summary"], dict)

    assert info is not None
    assert info["prediction_summary"]["object_count"] == 1
    assert isinstance(info["spatiallm_export"]["payload_summary"], dict)
