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


def test_source_selection_prefers_explicit_structured_prediction() -> None:
    module = _load_builder_module()
    payload = {
        "sample_id": "scene_a",
        "generator_name": "mock_spatiallm",
        "objects": [
            {
                "object_id": "table_0",
                "label": "table",
                "position": {"x": 1.0, "y": 2.0, "z": 0.5},
                "size": {"x": 1.0, "y": 1.0, "z": 0.7},
                "confidence": 0.9,
                "attributes": {},
            },
            {
                "object_id": "mug_0",
                "label": "mug",
                "position": {"x": 1.1, "y": 2.0, "z": 0.8},
                "size": {"x": 0.1, "y": 0.1, "z": 0.1},
                "confidence": 0.8,
                "attributes": {},
            },
        ],
        "relations": [
            {
                "subject_id": "mug_0",
                "predicate": "supported-by",
                "object_id": "table_0",
                "score": 0.9,
                "metadata": {},
            }
        ],
        "metadata": {},
    }
    prediction, source = module._resolve_prediction_with_provenance(
        scene_id="scene_a",
        setting_name="calibration_v1",
        setting_metadata={"structured_prediction_pre_repair": payload},
        prediction_summary={"object_count": 2, "relation_count": 1},
        report_dir=None,
    )
    assert source["source_class"] == "explicit_structured_prediction"
    assert source["selected_from"] == "structured_prediction_pre_repair"
    assert len(prediction.relations) == 1


def test_source_selection_prefers_hint_only_structured_prediction_when_no_tuples() -> None:
    module = _load_builder_module()
    payload = {
        "sample_id": "scene_b",
        "generator_name": "mock_spatiallm",
        "objects": [
            {
                "object_id": "chair_0",
                "label": "chair",
                "position": {"x": 0.0, "y": 0.0, "z": 0.0},
                "size": {"x": 1.0, "y": 1.0, "z": 1.0},
                "confidence": 0.6,
                "attributes": {"geometry_unavailable": True},
            }
        ],
        "relations": [],
        "metadata": {"relation_count_hint": 2, "relation_predicates": ["attached-to"]},
    }
    prediction, source = module._resolve_prediction_with_provenance(
        scene_id="scene_b",
        setting_name="calibration_v1",
        setting_metadata={"structured_prediction_pre_repair": payload},
        prediction_summary={"object_count": 1, "relation_count": 2, "relation_predicates": ["attached-to"]},
        report_dir=None,
    )
    assert source["source_class"] == "structured_prediction_with_hint_only"
    assert source["evidence_level"] == "hinted"
    assert len(prediction.relations) == 0
    assert prediction.metadata.get("relation_count_hint") == 2


def test_relation_rows_recovery_produces_explicit_relations_without_structured_payload() -> None:
    module = _load_builder_module()
    prediction, source = module._resolve_prediction_with_provenance(
        scene_id="scene_c",
        setting_name="external_generator",
        setting_metadata={
            "prediction_relations": [
                {
                    "subject_id": "mug_0",
                    "predicate": "supported-by",
                    "object_id": "table_0",
                    "score": 0.75,
                    "metadata": {"subject_label": "mug", "object_label": "table"},
                }
            ]
        },
        prediction_summary={"object_count": 2, "relation_count": 1, "relation_predicates": ["supported-by"]},
        report_dir=None,
    )
    assert source["selected_from"] == "metadata_relation_recovery"
    assert source["source_class"] == "explicit_structured_prediction"
    assert source["evidence_level"] == "explicit"
    assert len(prediction.relations) == 1
    assert prediction.metadata.get("recovered_from_metadata") is True


def test_summary_fallback_remains_honest_when_no_richer_source_exists() -> None:
    module = _load_builder_module()
    prediction, source = module._resolve_prediction_with_provenance(
        scene_id="scene_d",
        setting_name="calibration_v0",
        setting_metadata={},
        prediction_summary={"object_count": 3, "relation_count": 2, "relation_predicates": ["attached-to"]},
        report_dir=None,
    )
    assert source["source_class"] == "summary_reconstructed"
    assert source["evidence_level"] == "hinted"
    assert prediction.metadata.get("reconstructed_from_prediction_summary") is True
    assert prediction.metadata.get("relation_count_hint") == 2


def test_source_selection_can_load_structured_prediction_from_artifact_path(tmp_path: Path) -> None:
    module = _load_builder_module()
    artifact_dir = tmp_path / "scene_setting_artifacts" / "scene_e" / "calibration_v1"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = artifact_dir / "structured_prediction_pre_repair.json"
    artifact_path.write_text(
        (
            '{"sample_id":"scene_e","generator_name":"mock","objects":[{"object_id":"table_0","label":"table",'
            '"position":{"x":0.0,"y":0.0,"z":0.0},"size":{"x":1.0,"y":1.0,"z":1.0},"confidence":0.9,"attributes":{}},'
            '{"object_id":"mug_0","label":"mug","position":{"x":0.1,"y":0.0,"z":0.8},"size":{"x":0.1,"y":0.1,"z":0.1},'
            '"confidence":0.8,"attributes":{}}],"relations":[{"subject_id":"mug_0","predicate":"supported-by",'
            '"object_id":"table_0","score":0.9,"metadata":{}}],"metadata":{}}'
        ),
        encoding="utf-8",
    )

    prediction, source = module._resolve_prediction_with_provenance(
        scene_id="scene_e",
        setting_name="calibration_v1",
        setting_metadata={
            "prediction_artifact_paths": {
                "structured_prediction_pre_repair": str(artifact_path.relative_to(tmp_path))
            }
        },
        prediction_summary={"object_count": 2, "relation_count": 1},
        report_dir=tmp_path,
    )
    assert source["selected_from"] == "artifact:structured_prediction_pre_repair"
    assert source["source_class"] == "explicit_structured_prediction"
    assert len(prediction.relations) == 1


def test_source_selection_parses_scene_prediction_wrapped_payload() -> None:
    module = _load_builder_module()
    wrapped = {
        "scene_prediction": {
            "sample_id": "scene_f",
            "generator_name": "external",
            "objects": [
                {
                    "object_id": "door_0",
                    "label": "door",
                    "position": {"x": 0.0, "y": 0.0, "z": 1.0},
                    "size": {"x": 0.9, "y": 0.15, "z": 2.0},
                    "confidence": 0.7,
                    "attributes": {},
                }
            ],
            "relations": [
                {
                    "subject_id": "door_0",
                    "predicate": "attached-to",
                    "object_id": "wall_0",
                    "score": 0.8,
                    "metadata": {},
                }
            ],
            "metadata": {},
        }
    }
    prediction, source = module._resolve_prediction_with_provenance(
        scene_id="scene_f",
        setting_name="external_generator",
        setting_metadata={"structured_prediction_pre_repair": wrapped},
        prediction_summary={"object_count": 1, "relation_count": 1},
        report_dir=None,
    )
    assert source["source_class"] == "explicit_structured_prediction"
    assert source["selected_from"] == "structured_prediction_pre_repair"
    assert len(prediction.relations) == 1


def test_source_contract_precedence_prefers_explicit_over_hint_only(tmp_path: Path) -> None:
    module = _load_builder_module()
    artifact_dir = tmp_path / "scene_setting_artifacts" / "scene_g" / "calibration_v1"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    explicit_path = artifact_dir / "structured_prediction_pre_repair.json"
    explicit_path.write_text(
        (
            '{"scene_prediction":{"sample_id":"scene_g","generator_name":"external","objects":'
            '[{"object_id":"wall_0","label":"wall","position":{"x":0.0,"y":0.0,"z":1.0},'
            '"size":{"x":3.0,"y":0.2,"z":2.5},"confidence":0.8,"attributes":{}},'
            '{"object_id":"door_0","label":"door","position":{"x":0.1,"y":0.0,"z":1.0},'
            '"size":{"x":0.9,"y":0.2,"z":2.0},"confidence":0.8,"attributes":{}}],'
            '"relations":[{"subject_id":"door_0","predicate":"attached-to","object_id":"wall_0",'
            '"score":0.9,"metadata":{}}],"metadata":{}}}'
        ),
        encoding="utf-8",
    )
    hint_only_inline = {
        "sample_id": "scene_g",
        "generator_name": "mock",
        "objects": [],
        "relations": [],
        "metadata": {"relation_count_hint": 1, "relation_predicates": ["attached-to"]},
    }
    prediction, source = module._resolve_prediction_with_provenance(
        scene_id="scene_g",
        setting_name="calibration_v1",
        setting_metadata={
            "structured_prediction_pre_repair": hint_only_inline,
            "prediction_artifact_paths": {
                "structured_prediction_pre_repair": str(explicit_path.relative_to(tmp_path))
            },
            "prediction_source_contract": {
                "explicit_structured_prediction": {
                    "available": True,
                    "inline_key": None,
                    "artifact_key": "structured_prediction_pre_repair",
                },
                "structured_prediction_with_hint_only": {
                    "available": True,
                    "inline_key": "structured_prediction_pre_repair",
                    "artifact_key": None,
                },
                "summary_reconstructed": {"available": False, "inline_key": None, "artifact_key": None},
                "source_precedence": [
                    "explicit_structured_prediction",
                    "structured_prediction_with_hint_only",
                    "summary_reconstructed",
                ],
            },
        },
        prediction_summary={"object_count": 2, "relation_count": 1},
        report_dir=tmp_path,
    )
    assert source["selected_from"] == "artifact:structured_prediction_pre_repair"
    assert source["source_class"] == "explicit_structured_prediction"
    assert len(prediction.relations) == 1
