"""Actionable scene construction interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod

from self_calibrating_spatiallm.artifacts import ActionableScene, ScenePrediction


class ActionableSceneBuilder(ABC):
    """Base interface for converting repaired scenes into actionable structures."""

    name: str

    @abstractmethod
    def build(self, scene: ScenePrediction) -> ActionableScene:
        """Build actionable scene representation from repaired structured scene."""
        raise NotImplementedError
