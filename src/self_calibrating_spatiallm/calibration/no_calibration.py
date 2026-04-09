"""No-op calibration variant for ablations."""

from __future__ import annotations

from self_calibrating_spatiallm.artifacts import CalibratedPointCloud, CalibrationResult, Point3D, PointCloudSample
from self_calibrating_spatiallm.calibration.base import Calibrator


class NoCalibrationCalibrator(Calibrator):
    """Passes through original coordinates without frame normalization."""

    name = "none"

    def calibrate(self, sample: PointCloudSample) -> CalibratedPointCloud:
        calibration = CalibrationResult(
            sample_id=sample.sample_id,
            method=self.name,
            up_vector=Point3D(x=0.0, y=0.0, z=1.0),
            horizontal_axis=Point3D(x=1.0, y=0.0, z=0.0),
            origin_offset=Point3D(x=0.0, y=0.0, z=0.0),
            metadata={
                "normalization_applied": False,
                "confidence": {"up_axis": 0.0, "horizontal_orientation": 0.0},
            },
        )
        return CalibratedPointCloud(
            sample_id=sample.sample_id,
            points=list(sample.points),
            calibration=calibration,
            metadata={
                "frame": sample.sensor_frame,
                "source": "no_calibration",
                "is_placeholder": False,
                "room_bounds": sample.metadata.get("room_bounds"),
                "expected_unit": sample.metadata.get("expected_unit"),
                "inferred_scale_hint": sample.metadata.get("inferred_scale_hint"),
            },
        )
