from self_calibrating_spatiallm.evaluation.recommendations import (
    build_next_action_recommendations,
    build_v0_v1_comparison_warning,
)


def test_warning_generation_when_all_scenes_fallback() -> None:
    warning = build_v0_v1_comparison_warning(
        {
            "num_scene_setting_results": 10,
            "num_true_v1_execution": 0,
            "num_fallback_used": 10,
        }
    )
    assert warning is not None
    assert "NOT a valid v0-v1 method comparison" in warning


def test_next_action_recommendation_generation() -> None:
    environment_report = {
        "readiness": {
            "point_cloud_loading_ready": True,
            "true_calibration_v1_ready": False,
            "tests_ready": False,
            "external_spatiallm_ready": False,
            "v0_v1_method_comparison_ready": False,
        }
    }
    v1_execution_summary = {
        "num_scene_setting_results": 10,
        "num_true_v1_execution": 0,
        "num_fallback_used": 10,
    }
    warning = build_v0_v1_comparison_warning(v1_execution_summary)
    actions = build_next_action_recommendations(
        environment_report=environment_report,
        v1_execution_summary=v1_execution_summary,
        comparison_warning=warning,
    )
    assert any("calibration_v1" in action for action in actions)
    assert any("rerun environment check" in action.lower() for action in actions)
    assert any("fallback-only eval detected" in action.lower() for action in actions)
