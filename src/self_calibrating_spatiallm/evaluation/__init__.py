"""Evaluation package."""

from self_calibrating_spatiallm.evaluation.annotations import (
    ExpectedRelation,
    SceneAnnotation,
    TraversabilityLabel,
    load_scene_annotation,
)
from self_calibrating_spatiallm.evaluation.eval_pack import (
    EvaluationPackReport,
    SettingEvaluation,
    run_evaluation_pack,
)
from self_calibrating_spatiallm.evaluation.failure_taxonomy import (
    classify_failures,
    render_failure_summary_markdown,
    summarize_failure_taxonomy,
)
from self_calibrating_spatiallm.evaluation.metrics import compute_scene_metrics
from self_calibrating_spatiallm.evaluation.pack_manifest import EvaluationPackManifest, EvaluationSceneEntry
from self_calibrating_spatiallm.evaluation.post_run_analysis import (
    ALLOWED_IMPROVEMENT_TARGETS,
    build_post_true_v1_analysis_bundle,
    build_first_research_result_summary,
    build_next_improvement_decision,
    build_partition_comparison_summaries,
    build_scene_level_delta_report,
    build_trustworthy_comparison_status,
    build_researcher_facing_summary,
    build_stratified_comparison_summaries,
    render_next_improvement_decision_markdown,
    render_researcher_summary_markdown,
    render_trustworthy_comparison_status_markdown,
    recommend_next_calibration_improvement_target,
)
from self_calibrating_spatiallm.evaluation.recommendations import (
    build_next_action_recommendations,
    build_v0_v1_comparison_warning,
)
from self_calibrating_spatiallm.evaluation.simple_evaluator import SimpleSceneEvaluator

__all__ = [
    "ExpectedRelation",
    "EvaluationPackManifest",
    "EvaluationPackReport",
    "EvaluationSceneEntry",
    "SceneAnnotation",
    "SettingEvaluation",
    "SimpleSceneEvaluator",
    "TraversabilityLabel",
    "classify_failures",
    "compute_scene_metrics",
    "build_scene_level_delta_report",
    "build_partition_comparison_summaries",
    "build_stratified_comparison_summaries",
    "build_trustworthy_comparison_status",
    "build_next_improvement_decision",
    "build_researcher_facing_summary",
    "build_post_true_v1_analysis_bundle",
    "recommend_next_calibration_improvement_target",
    "build_first_research_result_summary",
    "render_trustworthy_comparison_status_markdown",
    "render_next_improvement_decision_markdown",
    "render_researcher_summary_markdown",
    "ALLOWED_IMPROVEMENT_TARGETS",
    "build_v0_v1_comparison_warning",
    "build_next_action_recommendations",
    "load_scene_annotation",
    "render_failure_summary_markdown",
    "run_evaluation_pack",
    "summarize_failure_taxonomy",
]
