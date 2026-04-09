from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_makefile_contains_true_v1_workflow_targets() -> None:
    makefile_text = (ROOT / "Makefile").read_text(encoding="utf-8")
    assert "install-true-v1" in makefile_text
    assert "check-env" in makefile_text
    assert "eval-v1" in makefile_text
    assert "v1-workflow" in makefile_text
    assert "post-true-v1-analysis" in makefile_text


def test_setup_and_workflow_scripts_exist() -> None:
    assert (ROOT / "scripts" / "setup_true_v1_env.sh").exists()
    assert (ROOT / "scripts" / "run_true_v1_workflow.sh").exists()
    assert (ROOT / "scripts" / "run_post_true_v1_analysis.py").exists()
