"""Runtime environment capability checks for operational pipeline paths."""

from __future__ import annotations

import importlib
import os
import platform
import shutil
from datetime import datetime, timezone
from typing import Any


def collect_environment_report(
    *,
    spatiallm_command_env_var: str = "SCSLM_SPATIALLM_COMMAND",
) -> dict[str, Any]:
    numpy_status = _module_status("numpy")
    pytest_status = _module_status("pytest")

    env_command = os.environ.get(spatiallm_command_env_var, "").strip()
    spatiallm_binary = shutil.which("spatiallm")

    calibration_v1_true_execution = bool(numpy_status["available"])
    external_adapter_available = bool(env_command) or bool(spatiallm_binary)

    point_cloud_backends = {
        "ply_ascii": {"supported": True, "notes": "built-in loader"},
        "pcd_ascii": {"supported": True, "notes": "built-in loader"},
        "npy_npz": {
            "supported": bool(numpy_status["available"]),
            "notes": "requires numpy",
        },
    }

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "platform": {
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "system": platform.system(),
            "machine": platform.machine(),
        },
        "dependencies": {
            "numpy": numpy_status,
            "pytest": pytest_status,
        },
        "capabilities": {
            "calibration_v1_true_execution": calibration_v1_true_execution,
            "calibration_v1_true_execution_reason": (
                "numpy available"
                if calibration_v1_true_execution
                else "numpy missing: calibration_v1 will use fallback path"
            ),
            "point_cloud_loading_backends": point_cloud_backends,
            "external_spatiallm_adapter_execution": external_adapter_available,
            "external_spatiallm_adapter_reason": _external_reason(
                env_command=env_command,
                spatiallm_binary=spatiallm_binary,
                spatiallm_command_env_var=spatiallm_command_env_var,
            ),
        },
    }
    report["readiness"] = build_readiness_summary(report)
    report["next_actions"] = build_environment_next_actions(report)
    return report


def build_readiness_summary(report: dict[str, Any]) -> dict[str, Any]:
    deps = report.get("dependencies", {})
    caps = report.get("capabilities", {})
    backends = caps.get("point_cloud_loading_backends", {}) if isinstance(caps, dict) else {}

    ply_ready = bool(_safe_bool(backends, "ply_ascii"))
    pcd_ready = bool(_safe_bool(backends, "pcd_ascii"))
    npy_ready = bool(_safe_bool(backends, "npy_npz"))

    point_cloud_loading_ready = ply_ready and pcd_ready
    true_calibration_v1_ready = bool(caps.get("calibration_v1_true_execution"))
    tests_ready = bool(_nested_value(deps, "pytest", "available"))
    external_spatiallm_ready = bool(caps.get("external_spatiallm_adapter_execution"))

    readiness = {
        "point_cloud_loading_ready": point_cloud_loading_ready,
        "true_calibration_v1_ready": true_calibration_v1_ready,
        "tests_ready": tests_ready,
        "external_spatiallm_ready": external_spatiallm_ready,
        "npy_npz_loading_ready": npy_ready,
        "v0_v1_method_comparison_ready": point_cloud_loading_ready and true_calibration_v1_ready,
    }

    reasons: list[str] = []
    if not point_cloud_loading_ready:
        reasons.append("point cloud loading backends unavailable")
    if not true_calibration_v1_ready:
        reasons.append("numpy unavailable: calibration_v1 will run fallback path")
    if not tests_ready:
        reasons.append("pytest unavailable: test suite not runnable")
    if not external_spatiallm_ready:
        reasons.append("external SpatialLM command not configured")
    readiness["blocking_reasons"] = reasons
    return readiness


def build_environment_next_actions(report: dict[str, Any]) -> list[str]:
    readiness = report.get("readiness", {})
    if not isinstance(readiness, dict):
        readiness = {}

    actions: list[str] = []
    if not bool(readiness.get("true_calibration_v1_ready")):
        actions.append("Install true-v1 dependencies: `pip install -e \".[calibration_v1]\"`.")
        actions.append("Rerun environment check: `PYTHONPATH=src python scripts/check_environment.py --format text`.")

    if not bool(readiness.get("tests_ready")):
        actions.append("Install test dependencies: `pip install -e \".[test]\"`.")

    if bool(readiness.get("true_calibration_v1_ready")):
        actions.append(
            "True calibration_v1 is available. Run eval pack: "
            "`PYTHONPATH=src python scripts/run_eval_pack.py --manifest configs/eval_pack/small_eval_pack.json --output-dir outputs/eval_pack/latest`."
        )

    if not bool(readiness.get("external_spatiallm_ready")):
        actions.append("External generator unavailable; use mock mode unless configuring SpatialLM command.")

    if not actions:
        actions.append("Environment is ready. Run eval pack and inspect `v0_v1_comparison_summary.md`.")
    return actions


def render_environment_report_text(report: dict[str, Any]) -> str:
    deps = report.get("dependencies", {})
    caps = report.get("capabilities", {})
    platform_info = report.get("platform", {})

    lines = [
        "Environment Capability Report",
        f"- generated_at: {report.get('generated_at')}",
        f"- python: {platform_info.get('python_version')} ({platform_info.get('python_implementation')})",
        f"- system: {platform_info.get('system')} / {platform_info.get('machine')}",
        "",
        "Dependencies:",
    ]

    for dep_name in ("numpy", "pytest"):
        status = deps.get(dep_name, {})
        if not isinstance(status, dict):
            continue
        lines.append(
            f"- {dep_name}: available={status.get('available')} version={status.get('version')}"
        )

    lines.extend(
        [
            "",
            "Readiness:",
        ]
    )
    readiness = report.get("readiness", {})
    if isinstance(readiness, dict):
        lines.extend(
            [
                f"- point_cloud_loading_ready: {readiness.get('point_cloud_loading_ready')}",
                f"- true_calibration_v1_ready: {readiness.get('true_calibration_v1_ready')}",
                f"- tests_ready: {readiness.get('tests_ready')}",
                f"- external_spatiallm_ready: {readiness.get('external_spatiallm_ready')}",
                f"- v0_v1_method_comparison_ready: {readiness.get('v0_v1_method_comparison_ready')}",
            ]
        )
        reasons = readiness.get("blocking_reasons", [])
        if isinstance(reasons, list) and reasons:
            lines.append("- blocking_reasons:")
            for reason in reasons:
                lines.append(f"- {reason}")

    lines.extend(
        [
            "",
            "Capabilities:",
            f"- calibration_v1_true_execution: {caps.get('calibration_v1_true_execution')}",
            f"- calibration_v1_reason: {caps.get('calibration_v1_true_execution_reason')}",
            f"- external_spatiallm_adapter_execution: {caps.get('external_spatiallm_adapter_execution')}",
            f"- external_spatiallm_adapter_reason: {caps.get('external_spatiallm_adapter_reason')}",
            "- point_cloud_loading_backends:",
        ]
    )

    backends = caps.get("point_cloud_loading_backends", {})
    if isinstance(backends, dict):
        for name in ("ply_ascii", "pcd_ascii", "npy_npz"):
            backend = backends.get(name, {})
            if isinstance(backend, dict):
                lines.append(
                    f"- {name}: supported={backend.get('supported')} notes={backend.get('notes')}"
                )

    lines.extend(["", "Next Actions:"])
    next_actions = report.get("next_actions", [])
    if isinstance(next_actions, list) and next_actions:
        for action in next_actions:
            lines.append(f"- {action}")

    return "\n".join(lines)


def _module_status(module_name: str) -> dict[str, Any]:
    try:
        module = importlib.import_module(module_name)
        version = getattr(module, "__version__", None)
        return {"available": True, "version": str(version) if version is not None else None}
    except Exception as error:  # pragma: no cover - import errors differ by env
        return {"available": False, "version": None, "error": type(error).__name__}


def _external_reason(*, env_command: str, spatiallm_binary: str | None, spatiallm_command_env_var: str) -> str:
    if env_command:
        return f"configured via env var {spatiallm_command_env_var}"
    if spatiallm_binary:
        return f"`spatiallm` binary found at {spatiallm_binary}"
    return (
        f"missing external command. Set {spatiallm_command_env_var} or install a `spatiallm` executable."
    )


def _safe_bool(backends: dict[str, Any], key: str) -> bool:
    backend = backends.get(key, {})
    if not isinstance(backend, dict):
        return False
    return bool(backend.get("supported"))


def _nested_value(mapping: dict[str, Any], first_key: str, second_key: str) -> Any:
    first = mapping.get(first_key, {})
    if not isinstance(first, dict):
        return None
    return first.get(second_key)
