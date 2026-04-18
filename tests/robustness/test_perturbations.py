from self_calibrating_spatiallm.artifacts import Point3D, PointCloudSample
from self_calibrating_spatiallm.robustness.perturbations import (
    apply_perturbation,
    derive_variant_seed,
    severity_bucket,
)


def _base_sample() -> PointCloudSample:
    return PointCloudSample(
        sample_id="scene_a",
        points=[
            Point3D(x=-1.0, y=-1.0, z=0.0),
            Point3D(x=1.0, y=-1.0, z=0.0),
            Point3D(x=-1.0, y=1.0, z=0.0),
            Point3D(x=1.0, y=1.0, z=0.0),
            Point3D(x=0.0, y=0.0, z=1.0),
        ],
        sensor_frame="sensor",
        metadata={"source": "unit-test"},
    )


def test_perturbation_is_deterministic_for_same_seed() -> None:
    sample = _base_sample()
    seed = derive_variant_seed(base_seed=11, components=["small", "scene_a", "clutter_injection", "0.5000"])

    first = apply_perturbation(
        base_sample=sample,
        perturbation_type="clutter_injection",
        severity=0.5,
        seed=seed,
        params={"max_injection_ratio": 0.4},
    )
    second = apply_perturbation(
        base_sample=sample,
        perturbation_type="clutter_injection",
        severity=0.5,
        seed=seed,
        params={"max_injection_ratio": 0.4},
    )

    first_points = [(point.x, point.y, point.z) for point in first.sample.points]
    second_points = [(point.x, point.y, point.z) for point in second.sample.points]

    assert first_points == second_points
    assert first.metadata["num_points_before"] == 5
    assert first.metadata["num_points_after"] == len(first_points)


def test_empty_input_is_handled_without_crashing() -> None:
    sample = PointCloudSample(sample_id="empty", points=[], sensor_frame="sensor", metadata={})

    result = apply_perturbation(
        base_sample=sample,
        perturbation_type="rotation_yaw",
        severity=0.4,
        seed=7,
        params={},
    )

    assert result.metadata["num_points_before"] == 0
    assert result.metadata["num_points_after"] == 0
    assert result.metadata["parameters"]["empty_input"] is True
    assert result.sample.metadata["perturbation"]["parameters"]["empty_input"] is True


def test_severity_bucket_boundaries() -> None:
    assert severity_bucket(0.0) == "low"
    assert severity_bucket(0.33) == "low"
    assert severity_bucket(0.34) == "mid"
    assert severity_bucket(0.66) == "mid"
    assert severity_bucket(0.67) == "high"
