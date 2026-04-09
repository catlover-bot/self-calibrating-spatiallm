from self_calibrating_spatiallm.artifacts import Point3D, SceneObject, ScenePrediction, SceneRelation
from self_calibrating_spatiallm.repair import SimpleRuleRepairer


def test_repairer_fixes_multiple_violations() -> None:
    prediction = ScenePrediction(
        sample_id="sample-1",
        generator_name="mock",
        metadata={
            "room_bounds": {
                "min": [-1.0, -1.0, -0.1],
                "max": [1.0, 1.0, 2.2],
            }
        },
        objects=[
            SceneObject("obj_floor", "floor", Point3D(0.0, 0.0, 0.0), Point3D(4.0, 4.0, 0.1)),
            SceneObject("obj_wall", "wall", Point3D(0.0, 0.9, 1.0), Point3D(2.0, 0.1, 2.0)),
            SceneObject("obj_box", "box", Point3D(2.0, -2.0, -0.2), Point3D(0.02, 4.0, 0.01)),
            SceneObject("obj_door", "door", Point3D(0.8, 0.85, 1.0), Point3D(0.9, 0.2, 2.1)),
        ],
        relations=[SceneRelation("obj_box", "supported-by", "missing", 0.5)],
    )

    repair = SimpleRuleRepairer().repair(prediction)

    repaired_box = next(obj for obj in repair.repaired_scene.objects if obj.object_id == "obj_box")
    assert repaired_box.position.z >= 0.0
    assert repaired_box.size.x >= 0.05
    assert repaired_box.size.y <= 3.5

    assert all(rel.object_id != "missing" for rel in repair.repaired_scene.relations)
    assert any(rel.predicate == "attached-to" and rel.subject_id == "obj_door" for rel in repair.repaired_scene.relations)
    assert len(repair.fixes_applied) >= 3
