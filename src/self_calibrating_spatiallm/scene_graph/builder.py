"""Rule-based actionable scene builder."""

from __future__ import annotations

from self_calibrating_spatiallm.artifacts import ActionDirective, ActionableScene, ScenePrediction, SceneRelation
from self_calibrating_spatiallm.scene_graph.base import ActionableSceneBuilder
from self_calibrating_spatiallm.scene_graph.relations import derive_basic_relations


class RuleBasedActionableSceneBuilder(ActionableSceneBuilder):
    """Creates actionable scene graph with derived relations and simple tasks."""

    name = "rule_based_actionable_builder"

    def build(self, scene: ScenePrediction) -> ActionableScene:
        derived_relations = derive_basic_relations(scene.objects)
        merged_relations = self._merge_relations(scene.relations, derived_relations)
        anchor_id = self._select_anchor(scene)

        accessible_ids = {
            relation.subject_id
            for relation in merged_relations
            if relation.predicate == "accessible" and relation.object_id == "agent"
        }

        actions: list[ActionDirective] = []
        for obj in scene.objects:
            if obj.label in {"floor", "wall", "ceiling", "table"}:
                continue

            action_prefix = "approach" if obj.object_id in accessible_ids else "inspect"
            actions.append(
                ActionDirective(
                    action=f"{action_prefix}:{obj.label}",
                    target_object_id=obj.object_id,
                    rationale="derived from actionable relation set",
                )
            )

        if not actions and anchor_id is not None:
            actions.append(
                ActionDirective(
                    action="inspect:anchor",
                    target_object_id=anchor_id,
                    rationale="fallback action when no movable objects are predicted",
                )
            )

        return ActionableScene(
            sample_id=scene.sample_id,
            builder_name=self.name,
            anchor_object_id=anchor_id,
            relations=merged_relations,
            actions=actions,
            metadata={"is_placeholder": False},
        )

    @staticmethod
    def _select_anchor(scene: ScenePrediction) -> str | None:
        for obj in scene.objects:
            if obj.label == "table":
                return obj.object_id
        return scene.objects[0].object_id if scene.objects else None

    @staticmethod
    def _merge_relations(
        primary: list[SceneRelation],
        secondary: list[SceneRelation],
    ) -> list[SceneRelation]:
        merged: list[SceneRelation] = []
        seen: set[tuple[str, str, str]] = set()

        for relation in [*primary, *secondary]:
            key = (relation.subject_id, relation.predicate, relation.object_id)
            if key in seen:
                continue
            seen.add(key)
            merged.append(relation)

        return merged
