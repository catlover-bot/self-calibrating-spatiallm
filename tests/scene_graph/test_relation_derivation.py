from self_calibrating_spatiallm.artifacts import Point3D, SceneObject
from self_calibrating_spatiallm.scene_graph import derive_basic_relations


def test_relation_derivation_extended_relations() -> None:
    objects = [
        SceneObject("obj_wall", "wall", Point3D(0.0, 1.0, 1.0), Point3D(2.0, 0.1, 2.0)),
        SceneObject("obj_table", "table", Point3D(0.0, 0.0, 0.8), Point3D(1.2, 0.8, 0.75)),
        SceneObject(
            "obj_mug",
            "mug",
            Point3D(0.1, 0.1, 1.0),
            Point3D(0.08, 0.08, 0.1),
            attributes={"facing_target_id": "obj_wall"},
        ),
    ]

    relations = derive_basic_relations(objects, near_threshold=1.0)
    keys = {(rel.subject_id, rel.predicate, rel.object_id) for rel in relations}

    assert ("obj_mug", "near", "obj_table") in keys
    assert ("obj_mug", "supported-by", "obj_table") in keys
    assert ("obj_mug", "accessible", "agent") in keys
    assert ("obj_mug", "facing", "obj_wall") in keys
