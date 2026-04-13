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
    assert "Scene scene_demo contains 2 objects" in exported["scene_summary_text"]
    assert "supported-by" in exported["relation_text"]
    assert exported["relation_statements"]


def test_qa_and_grounding_examples_are_generated() -> None:
    prediction = _sample_prediction()
    qa_examples = build_qa_examples(prediction)
    grounding_examples = build_grounding_examples(prediction)

    assert any(item.get("task_type") == "count_objects" for item in qa_examples)
    assert any(item.get("task_type") == "relation_exists" for item in qa_examples)
    assert any(item.get("task_type") == "referential_statement" for item in grounding_examples)
    assert any(item.get("task_type") == "instruction_grounding" for item in grounding_examples)


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
    assert isinstance(record["qa_examples"], list)
    assert isinstance(record["grounding_examples"], list)

