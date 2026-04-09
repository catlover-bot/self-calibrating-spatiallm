"""Pass-through repairer for ablations."""

from __future__ import annotations

from self_calibrating_spatiallm.artifacts import RepairResult, ScenePrediction
from self_calibrating_spatiallm.repair.base import SceneRepairer


class PassThroughRepairer(SceneRepairer):
    """Returns scene unchanged without applying repairs."""

    name = "none"

    def repair(self, scene: ScenePrediction) -> RepairResult:
        return RepairResult(
            sample_id=scene.sample_id,
            repairer_name=self.name,
            issues=[],
            fixes_applied=[],
            repaired_scene=ScenePrediction.from_dict(scene.to_dict()),
            metadata={"mode": "passthrough"},
        )
