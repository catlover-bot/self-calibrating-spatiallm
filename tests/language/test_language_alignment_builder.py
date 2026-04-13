from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType


def _load_builder_module() -> ModuleType:
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "build_language_dataset.py"
    spec = importlib.util.spec_from_file_location("build_language_dataset_script", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec from {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_alignment_rows_include_pairwise_differences_and_comparison_examples() -> None:
    module = _load_builder_module()
    alignment_map = {
        "scene_demo": {
            "calibration_v0": {
                "object_labels": ["wall", "door"],
                "relation_predicates": [],
                "relation_hint_predicates": ["attached-to"],
                "object_count": 2,
                "relation_count": 0,
                "relation_evidence_level": "hinted",
            },
            "calibration_v1": {
                "object_labels": ["wall", "door", "table"],
                "relation_predicates": ["supported-by"],
                "relation_hint_predicates": [],
                "object_count": 3,
                "relation_count": 1,
                "relation_evidence_level": "explicit",
            },
            "mock_generator": {
                "object_labels": ["wall", "door"],
                "relation_predicates": ["attached-to"],
                "relation_hint_predicates": [],
                "object_count": 2,
                "relation_count": 1,
                "relation_evidence_level": "explicit",
            },
            "external_generator": {
                "object_labels": ["wall", "door", "window"],
                "relation_predicates": ["attached-to", "inside"],
                "relation_hint_predicates": [],
                "object_count": 3,
                "relation_count": 2,
                "relation_evidence_level": "explicit",
            },
        }
    }

    rows = module._build_alignment_rows(alignment_map)
    assert len(rows) == 1
    row = rows[0]

    v0_v1 = row["pairwise_differences"]["calibration_v0_vs_calibration_v1"]
    assert v0_v1["object_labels_added"] == ["table"]
    assert v0_v1["relation_predicates_added"] == ["supported-by"]
    assert v0_v1["relation_count_delta"] == 1

    mock_external = row["pairwise_differences"]["mock_generator_vs_external_generator"]
    assert mock_external["object_labels_added"] == ["window"]
    assert mock_external["relation_predicates_added"] == ["inside"]

    questions = [item.get("question") for item in row["comparison_examples"]]
    assert "How does calibration_v1 differ from calibration_v0 for this scene?" in questions
    assert "Which labels appear in external_generator but not mock_generator?" in questions
