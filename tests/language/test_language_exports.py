from self_calibrating_spatiallm.artifacts import Point3D, SceneObject, ScenePrediction, SceneRelation
from self_calibrating_spatiallm.language import (
    build_grounding_examples,
    build_language_scene_record,
    build_qa_examples,
    export_scene_prediction_to_language,
)


def _sample_prediction() -> ScenePrediction:
    return ScenePrediction(
        sample_id="scene_demo",
        generator_name="mock_spatiallm",
        objects=[
            SceneObject(
                object_id="table_0",
                label="table",
                position=Point3D(1.0, 2.0, 0.8),
                size=Point3D(1.2, 0.8, 0.7),
                confidence=0.9,
            ),
            SceneObject(
                object_id="mug_0",
                label="mug",
                position=Point3D(1.1, 2.1, 0.95),
                size=Point3D(0.1, 0.1, 0.12),
                confidence=0.85,
            ),
        ],
        relations=[
            SceneRelation(
                subject_id="mug_0",
                predicate="supported-by",
                object_id="table_0",
                score=0.8,
            )
        ],
    )


def test_scene_export_contains_deterministic_text_fields() -> None:
    prediction = _sample_prediction()
    exported = export_scene_prediction_to_language(prediction)

    assert exported["object_count"] == 2
    assert exported["relation_count"] == 1
    assert exported["relation_evidence_level"] == "explicit"
    assert "Scene scene_demo contains 2 objects" in exported["scene_summary_text"]
    assert "supported-by" in exported["relation_text"]
    assert exported["relation_statements"]


def test_hinted_relations_are_preserved_when_explicit_tuples_missing() -> None:
    prediction = ScenePrediction(
        sample_id="scene_hinted",
        generator_name="summary_fallback",
        objects=[
            SceneObject(
                object_id="chair_0",
                label="chair",
                position=Point3D(0.0, 0.0, 0.0),
                size=Point3D(1.0, 1.0, 1.0),
                confidence=0.5,
                attributes={"reconstructed_from_prediction_summary": True, "geometry_unavailable": True},
            )
        ],
        relations=[],
        metadata={
            "reconstructed_from_prediction_summary": True,
            "relation_count_hint": 2,
            "relation_predicates": ["supported-by", "attached-to"],
        },
    )
    exported = export_scene_prediction_to_language(prediction)

    assert exported["relation_evidence_level"] == "hinted"
    assert "hinted relations" in exported["relation_text"]
    assert "supported-by" in exported["relation_text"]
    assert exported["object_geometry_mode"] == "summary_reconstructed"
    assert "No explicit relations were predicted." not in exported["relation_text"]


def test_qa_generation_includes_label_and_relation_evidence_questions() -> None:
    prediction = _sample_prediction()
    qa_examples = build_qa_examples(prediction)
    task_types = {str(item.get("task_type")) for item in qa_examples}

    assert "count_objects" in task_types
    assert "contains_label" in task_types
    assert "list_object_labels" in task_types
    assert "relation_exists" in task_types
    assert "relation_predicates_list" in task_types
    assert "relation_predicate_evidence" in task_types


def test_grounding_examples_are_transparent_for_reconstructed_geometry() -> None:
    prediction = ScenePrediction(
        sample_id="scene_reconstructed",
        generator_name="summary_fallback",
        objects=[
            SceneObject(
                object_id="table_0",
                label="table",
                position=Point3D(0.0, 0.0, 0.0),
                size=Point3D(1.0, 1.0, 1.0),
                confidence=0.5,
                attributes={"reconstructed_from_prediction_summary": True, "geometry_unavailable": True},
            )
        ],
        relations=[],
        metadata={"reconstructed_from_prediction_summary": True, "relation_predicates": ["supported-by"]},
    )
    grounding_examples = build_grounding_examples(prediction)

    assert any(item.get("task_type") == "referential_statement_presence" for item in grounding_examples)
    assert all("located at" not in str(item.get("text")) for item in grounding_examples)
    assert any(item.get("task_type") == "relation_evidence_statement" for item in grounding_examples)


def test_language_scene_record_contains_required_fields() -> None:
    prediction = _sample_prediction()
    record = build_language_scene_record(
        scene_id="scene_demo",
        setting="calibration_v1",
        source_type="ply",
        prediction=prediction,
        metadata={"generator_mode": "mock"},
    )

    assert record["scene_id"] == "scene_demo"
    assert record["setting"] == "calibration_v1"
    assert "structured_prediction" in record
    assert "scene_summary_text" in record
    assert isinstance(record["relation_statements"], list)
    assert "relation_evidence_level" in record
    assert isinstance(record["qa_examples"], list)
    assert isinstance(record["grounding_examples"], list)
