import pytest

from self_calibrating_spatiallm.artifacts import Point3D, PointCloudSample
from self_calibrating_spatiallm.calibration import NoCalibrationCalibrator
from self_calibrating_spatiallm.generation import ExternalGeneratorError, SpatialLMExternalGenerator


def test_external_generator_fails_gracefully_when_command_missing() -> None:
    sample = PointCloudSample(
        sample_id="scene-1",
        points=[Point3D(0.0, 0.0, 0.0), Point3D(1.0, 0.0, 0.0), Point3D(0.0, 1.0, 0.0)],
    )
    calibrated = NoCalibrationCalibrator().calibrate(sample)

    generator = SpatialLMExternalGenerator(command=["this_binary_should_not_exist_12345"])

    with pytest.raises(ExternalGeneratorError) as error:
        generator.generate(calibrated)

    assert "not found" in str(error.value).lower()
