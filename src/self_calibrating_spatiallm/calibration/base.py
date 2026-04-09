"""Calibration interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod

from self_calibrating_spatiallm.artifacts import CalibratedPointCloud, PointCloudSample


class Calibrator(ABC):
    """Base interface for point-cloud calibration modules."""

    name: str

    @abstractmethod
    def calibrate(self, sample: PointCloudSample) -> CalibratedPointCloud:
        """Estimate calibration metadata and return calibrated point cloud artifact."""
        raise NotImplementedError
