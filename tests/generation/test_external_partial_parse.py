import json
import subprocess

from self_calibrating_spatiallm.artifacts import Point3D, PointCloudSample
from self_calibrating_spatiallm.calibration import NoCalibrationCalibrator
from self_calibrating_spatiallm.generation import SpatialLMExternalGenerator


def test_external_partial_parse_behavior(monkeypatch) -> None:
    sample = PointCloudSample(
        sample_id="scene-partial",
        points=[Point3D(0.0, 0.0, 0.0), Point3D(1.0, 0.0, 0.0), Point3D(0.0, 1.0, 0.0)],
    )
    calibrated = NoCalibrationCalibrator().calibrate(sample)

    partial_payload = {
        "scene_prediction": {
            "instances": [
                {
                    "id": "obj_1",
                    "category": "chair",
                    "center": [0.2, 0.1, 0.8],
                    "extent": [0.5, 0.5, 1.0],
                    "score": 0.77,
                }
            ],
            "relations": [{"subject": "obj_1", "relation": "near", "object": "obj_2", "score": 0.5}],
        }
    }

    def fake_run_command(self: SpatialLMExternalGenerator, command: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=command,
            returncode=0,
            stdout=json.dumps(partial_payload),
            stderr="",
        )

    monkeypatch.setattr(SpatialLMExternalGenerator, "_run_command", fake_run_command)

    generator = SpatialLMExternalGenerator(command=["spatiallm-cli", "--input", "{spatiallm_input}"])
    prediction = generator.generate(calibrated)
    info = generator.get_last_execution_info()

    assert prediction.sample_id == sample.sample_id
    assert len(prediction.objects) == 1
    assert prediction.objects[0].label == "chair"
    assert len(prediction.relations) == 1
    assert prediction.metadata["partial_parse"] is True
    propagation = prediction.metadata.get("propagation_diagnostics", {})
    assert propagation.get("parse_mode") == "partial_schema"
    assert propagation.get("partial_parse") is True
    assert info is not None
    assert info["parse_mode"] == "partial_schema"
    assert info["partial_parse"] is True
    assert info["parse_warnings"]
