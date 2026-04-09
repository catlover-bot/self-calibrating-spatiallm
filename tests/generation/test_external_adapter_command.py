import json
import subprocess

from self_calibrating_spatiallm.artifacts import Point3D, PointCloudSample
from self_calibrating_spatiallm.calibration import NoCalibrationCalibrator
from self_calibrating_spatiallm.generation import SpatialLMExternalGenerator


def test_external_adapter_command_construction(monkeypatch) -> None:
    sample = PointCloudSample(
        sample_id="scene-command",
        points=[Point3D(0.0, 0.0, 0.0), Point3D(1.0, 0.0, 0.0), Point3D(0.0, 1.0, 0.0)],
    )
    calibrated = NoCalibrationCalibrator().calibrate(sample)

    captured: dict[str, list[str]] = {}

    def fake_run_command(self: SpatialLMExternalGenerator, command: list[str]) -> subprocess.CompletedProcess[str]:
        captured["command"] = list(command)
        stdout = json.dumps(
            {
                "scene_prediction": {
                    "sample_id": sample.sample_id,
                    "generator_name": "external_test",
                    "objects": [],
                    "relations": [],
                }
            }
        )
        return subprocess.CompletedProcess(args=command, returncode=0, stdout=stdout, stderr="")

    monkeypatch.setattr(SpatialLMExternalGenerator, "_run_command", fake_run_command)

    generator = SpatialLMExternalGenerator(
        command=[
            "spatiallm-cli",
            "--input",
            "{spatiallm_input}",
            "--output",
            "{output_json}",
            "--scene",
            "{scene_id}",
        ]
    )
    prediction = generator.generate(calibrated)
    info = generator.get_last_execution_info()

    assert prediction.sample_id == sample.sample_id
    assert info is not None
    assert info["command_source"] == "config"
    assert info["invoked_command"] == captured["command"]
    assert info["return_code"] == 0
    assert info["parse_mode"] == "full_schema"
    assert info["partial_parse"] is False
    assert "payload_hash_sha256" in info["spatiallm_export"]
    assert isinstance(info["prediction_summary"], dict)
    assert "{spatiallm_input}" not in " ".join(captured["command"])
    assert sample.sample_id in captured["command"]


def test_external_adapter_command_from_env(monkeypatch) -> None:
    sample = PointCloudSample(
        sample_id="scene-env",
        points=[Point3D(0.0, 0.0, 0.0), Point3D(1.0, 0.0, 0.0), Point3D(0.0, 1.0, 0.0)],
    )
    calibrated = NoCalibrationCalibrator().calibrate(sample)

    captured: dict[str, list[str]] = {}

    def fake_run_command(self: SpatialLMExternalGenerator, command: list[str]) -> subprocess.CompletedProcess[str]:
        captured["command"] = list(command)
        stdout = json.dumps({"scene_prediction": {"objects": [], "relations": []}})
        return subprocess.CompletedProcess(args=command, returncode=0, stdout=stdout, stderr="")

    monkeypatch.setattr(SpatialLMExternalGenerator, "_run_command", fake_run_command)
    monkeypatch.setenv("TEST_SPATIALLM_COMMAND", "spatiallm-env --input {spatiallm_input}")

    generator = SpatialLMExternalGenerator(command=None, command_env_var="TEST_SPATIALLM_COMMAND")
    generator.generate(calibrated)
    info = generator.get_last_execution_info()

    assert info is not None
    assert info["command_source"] == "env:TEST_SPATIALLM_COMMAND"
    assert captured["command"][0] == "spatiallm-env"
