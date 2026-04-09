"""Placeholder evaluation logic for pipeline smoke runs."""

from __future__ import annotations

from self_calibrating_spatiallm.artifacts import ActionableScene, EvaluationResult, RepairResult, ScenePrediction


class SimpleSceneEvaluator:
    """Evaluates artifact consistency with lightweight summary metrics."""

    name = "simple_scene_evaluator"

    def evaluate(
        self,
        scene: ScenePrediction,
        repair: RepairResult,
        actionable_scene: ActionableScene,
        calibration_method: str | None = None,
        setting_name: str | None = None,
    ) -> EvaluationResult:
        support_relations = sum(1 for rel in actionable_scene.relations if rel.predicate == "supported-by")
        accessibility_relations = sum(
            1
            for rel in actionable_scene.relations
            if rel.predicate == "accessible" and rel.object_id == "agent"
        )

        metrics = {
            "num_objects": float(len(scene.objects)),
            "num_relations": float(len(actionable_scene.relations)),
            "num_actions": float(len(actionable_scene.actions)),
            "supported_by_relations": float(support_relations),
            "accessible_relations": float(accessibility_relations),
            "repair_issues_detected": float(len(repair.issues)),
            "repair_fixes_applied": float(len(repair.fixes_applied)),
        }

        passed = (len(actionable_scene.actions) > 0) and (
            len(repair.fixes_applied) >= min(1, len(repair.issues))
        )

        notes = [
            "Placeholder evaluator: replace with task-specific metrics in future ablations.",
        ]
        if setting_name:
            notes.append(f"setting={setting_name}")
        if calibration_method:
            notes.append(f"calibration={calibration_method}")

        return EvaluationResult(
            sample_id=scene.sample_id,
            evaluator_name=self.name,
            metrics=metrics,
            passed=passed,
            notes=notes,
            metadata={
                "is_placeholder": False,
                "setting_name": setting_name,
                "calibration_method": calibration_method,
            },
        )
