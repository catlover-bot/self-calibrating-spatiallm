from self_calibrating_spatiallm.environment import build_readiness_summary, collect_environment_report


def test_environment_check_reporting_structure() -> None:
    report = collect_environment_report()

    assert "dependencies" in report
    assert "capabilities" in report

    deps = report["dependencies"]
    caps = report["capabilities"]

    assert "numpy" in deps
    assert "pytest" in deps
    assert isinstance(deps["numpy"]["available"], bool)
    assert isinstance(deps["pytest"]["available"], bool)

    assert "calibration_v1_true_execution" in caps
    assert "point_cloud_loading_backends" in caps
    assert "external_spatiallm_adapter_execution" in caps

    assert caps["calibration_v1_true_execution"] == deps["numpy"]["available"]
    backends = caps["point_cloud_loading_backends"]
    assert backends["ply_ascii"]["supported"] is True
    assert backends["pcd_ascii"]["supported"] is True
    assert backends["npy_npz"]["supported"] == deps["numpy"]["available"]

    assert "readiness" in report
    readiness = report["readiness"]
    assert isinstance(readiness["point_cloud_loading_ready"], bool)
    assert isinstance(readiness["true_calibration_v1_ready"], bool)
    assert isinstance(readiness["tests_ready"], bool)
    assert isinstance(readiness["external_spatiallm_ready"], bool)
    assert isinstance(readiness["v0_v1_method_comparison_ready"], bool)
    assert isinstance(readiness["blocking_reasons"], list)


def test_readiness_summary_generation() -> None:
    report = {
        "dependencies": {
            "numpy": {"available": False},
            "pytest": {"available": True},
        },
        "capabilities": {
            "calibration_v1_true_execution": False,
            "external_spatiallm_adapter_execution": False,
            "point_cloud_loading_backends": {
                "ply_ascii": {"supported": True},
                "pcd_ascii": {"supported": True},
                "npy_npz": {"supported": False},
            },
        },
    }
    readiness = build_readiness_summary(report)
    assert readiness["point_cloud_loading_ready"] is True
    assert readiness["true_calibration_v1_ready"] is False
    assert readiness["tests_ready"] is True
    assert readiness["external_spatiallm_ready"] is False
    assert readiness["v0_v1_method_comparison_ready"] is False
