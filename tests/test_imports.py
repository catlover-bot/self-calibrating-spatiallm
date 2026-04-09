from self_calibrating_spatiallm import __version__
from self_calibrating_spatiallm.calibration import PlaneAwareCalibratorV1, extract_calibration_execution
from self_calibrating_spatiallm.environment import build_readiness_summary, collect_environment_report
from self_calibrating_spatiallm.evaluation import (
    SceneAnnotation,
    build_next_improvement_decision,
    build_post_true_v1_analysis_bundle,
    build_partition_comparison_summaries,
    build_researcher_facing_summary,
    build_scene_level_delta_report,
    build_stratified_comparison_summaries,
    build_trustworthy_comparison_status,
    build_next_action_recommendations,
    build_v0_v1_comparison_warning,
    recommend_next_calibration_improvement_target,
    render_next_improvement_decision_markdown,
    render_researcher_summary_markdown,
    render_trustworthy_comparison_status_markdown,
    run_evaluation_pack,
)
from self_calibrating_spatiallm.io import load_point_cloud_sample
from self_calibrating_spatiallm.pipeline import SceneInputConfig, SingleScenePipeline, run_multi_scene_pipeline


def test_imports() -> None:
    assert __version__
    assert callable(load_point_cloud_sample)
    assert callable(SceneInputConfig)
    assert callable(SingleScenePipeline)
    assert callable(run_multi_scene_pipeline)
    assert callable(run_evaluation_pack)
    assert callable(SceneAnnotation)
    assert callable(PlaneAwareCalibratorV1)
    assert callable(extract_calibration_execution)
    assert callable(collect_environment_report)
    assert callable(build_readiness_summary)
    assert callable(build_v0_v1_comparison_warning)
    assert callable(build_next_action_recommendations)
    assert callable(build_scene_level_delta_report)
    assert callable(build_partition_comparison_summaries)
    assert callable(build_stratified_comparison_summaries)
    assert callable(build_trustworthy_comparison_status)
    assert callable(build_next_improvement_decision)
    assert callable(build_researcher_facing_summary)
    assert callable(build_post_true_v1_analysis_bundle)
    assert callable(render_trustworthy_comparison_status_markdown)
    assert callable(render_next_improvement_decision_markdown)
    assert callable(render_researcher_summary_markdown)
    assert callable(recommend_next_calibration_improvement_target)
