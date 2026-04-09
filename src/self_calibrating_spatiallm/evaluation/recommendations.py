"""Trustworthiness guardrails and next-step recommendations for evaluation runs."""

from __future__ import annotations

from typing import Any


def build_v0_v1_comparison_warning(v1_execution_summary: dict[str, Any]) -> str | None:
    """Return a prominent warning when v0-v1 method comparison is not trustworthy."""
    total = _int(v1_execution_summary.get("num_scene_setting_results"))
    true_count = _int(v1_execution_summary.get("num_true_v1_execution"))
    fallback_count = _int(v1_execution_summary.get("num_fallback_used"))

    if total > 0 and true_count == 0:
        return (
            "WARNING: This is NOT a valid v0-v1 method comparison because calibration_v1 never ran "
            "the true plane-aware path (`num_true_v1_execution=0`). Results are fallback-only."
        )

    if total > 0 and fallback_count > 0 and (fallback_count / total) >= 0.5:
        return (
            "WARNING: Calibration_v1 fallback rate is high. Interpret v0-v1 gains cautiously and inspect "
            "fallback reasons before making method claims."
        )

    return None


def build_next_action_recommendations(
    *,
    environment_report: dict[str, Any],
    v1_execution_summary: dict[str, Any],
    comparison_warning: str | None = None,
) -> list[str]:
    readiness = environment_report.get("readiness", {})
    if not isinstance(readiness, dict):
        readiness = {}

    true_ready = bool(readiness.get("true_calibration_v1_ready"))
    tests_ready = bool(readiness.get("tests_ready"))
    external_ready = bool(readiness.get("external_spatiallm_ready"))

    total = _int(v1_execution_summary.get("num_scene_setting_results"))
    true_count = _int(v1_execution_summary.get("num_true_v1_execution"))
    fallback_count = _int(v1_execution_summary.get("num_fallback_used"))

    actions: list[str] = []

    if not true_ready:
        actions.append("Install calibration_v1 dependencies: `pip install -e \".[calibration_v1]\"`.")
        actions.append("Rerun environment check: `PYTHONPATH=src python scripts/check_environment.py --format text`.")

    if not tests_ready:
        actions.append("Install test dependencies: `pip install -e \".[test]\"`.")

    if true_ready and total == 0:
        actions.append(
            "Environment is ready for true v1. Run eval pack: "
            "`PYTHONPATH=src python scripts/run_eval_pack.py --manifest configs/eval_pack/small_eval_pack.json --output-dir outputs/eval_pack/latest`."
        )

    if total > 0 and true_count == 0:
        actions.append(
            "Fallback-only eval detected. Inspect `fallback_reason` in `03_calibration_result.json` and "
            "`run_manifest.json`, then fix environment and rerun eval pack."
        )
    elif total > 0 and fallback_count > 0:
        actions.append("Fallback-heavy results detected. Inspect fallback reasons scene-by-scene before claims.")
    elif total > 0 and true_count > 0:
        actions.append(
            "True v1 execution detected. Inspect `v0_v1_comparison_summary.md` for improvements/regressions."
        )

    if not external_ready:
        actions.append("External generator unavailable; continue with mock mode unless configuring SpatialLM command.")

    if comparison_warning is None and not actions:
        actions.append("Run is ready and interpretable. Proceed to inspect v0-v1 comparison summary.")

    return actions


def _int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0

