"""Structured scene generation interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from self_calibrating_spatiallm.artifacts import CalibratedPointCloud, ScenePrediction


class SceneGenerator(ABC):
    """Base interface for calibrated-point-cloud -> scene prediction generators."""

    name: str

    @abstractmethod
    def generate(self, calibrated: CalibratedPointCloud) -> ScenePrediction:
        """Generate a structured scene prediction from calibrated point cloud artifact."""
        raise NotImplementedError

    def get_last_raw_output(self) -> str | None:
        """Optional raw output from external tools used by this generator."""
        return None

    def get_last_execution_info(self) -> dict[str, Any] | None:
        """Optional structured execution diagnostics from the most recent run."""
        return None
