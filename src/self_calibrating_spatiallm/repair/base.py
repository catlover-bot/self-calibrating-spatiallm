"""Scene repair interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod

from self_calibrating_spatiallm.artifacts import RepairResult, ScenePrediction


class SceneRepairer(ABC):
    """Base interface for structural scene repair."""

    name: str

    @abstractmethod
    def repair(self, scene: ScenePrediction) -> RepairResult:
        """Detect and repair structural violations in a predicted scene."""
        raise NotImplementedError
