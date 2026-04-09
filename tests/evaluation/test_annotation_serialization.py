from pathlib import Path

from self_calibrating_spatiallm.evaluation import (
    ExpectedRelation,
    SceneAnnotation,
    TraversabilityLabel,
)


def test_annotation_json_roundtrip(tmp_path: Path) -> None:
    annotation = SceneAnnotation(
        scene_id="scene-a",
        expected_up_axis="z",
        expected_horizontal_axis="x",
        room_orientation_hint="north_wall_parallel_to_x",
        expected_scale_hint="expected_unit:meter",
        expected_door_count=1,
        expected_window_count=0,
        expected_object_categories=["floor", "wall", "door"],
        expected_relations=[ExpectedRelation("door", "attached-to", "wall")],
        traversability_labels=[TraversabilityLabel("door", True, "Door should be reachable")],
        notes="annotation smoke test",
        metadata={"split": "dev"},
    )

    path = annotation.save_json(tmp_path / "annotation.json")
    loaded = SceneAnnotation.load_json(path)
    assert loaded == annotation
