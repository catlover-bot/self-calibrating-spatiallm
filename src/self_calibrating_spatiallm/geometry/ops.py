"""Small geometry helpers used across pipeline stages."""

from __future__ import annotations

import math
from typing import Sequence

from self_calibrating_spatiallm.artifacts import Point3D, SceneObject


def points_to_array(points: Sequence[Point3D]) -> list[list[float]]:
    return [[float(point.x), float(point.y), float(point.z)] for point in points]


def array_to_points(array: Sequence[Sequence[float]]) -> list[Point3D]:
    points: list[Point3D] = []
    for row in array:
        values = list(row)
        if len(values) != 3:
            raise ValueError(f"Expected 3 values per point, got {len(values)}")
        points.append(Point3D(x=float(values[0]), y=float(values[1]), z=float(values[2])))
    return points


def centroid(points: Sequence[Point3D]) -> Point3D:
    if not points:
        return Point3D(x=0.0, y=0.0, z=0.0)
    inv_n = 1.0 / len(points)
    return Point3D(
        x=sum(point.x for point in points) * inv_n,
        y=sum(point.y for point in points) * inv_n,
        z=sum(point.z for point in points) * inv_n,
    )


def normalize_vector(vec: Point3D) -> Point3D:
    norm = math.sqrt((vec.x**2) + (vec.y**2) + (vec.z**2))
    if norm == 0:
        return Point3D(x=0.0, y=0.0, z=1.0)
    return Point3D(x=vec.x / norm, y=vec.y / norm, z=vec.z / norm)


def euclidean_distance(a: Point3D, b: Point3D) -> float:
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)


def horizontal_distance(a: Point3D, b: Point3D) -> float:
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)


def object_bounds(obj: SceneObject) -> tuple[Point3D, Point3D]:
    half = Point3D(x=max(obj.size.x, 0.0) / 2.0, y=max(obj.size.y, 0.0) / 2.0, z=max(obj.size.z, 0.0) / 2.0)
    return (
        Point3D(x=obj.position.x - half.x, y=obj.position.y - half.y, z=obj.position.z - half.z),
        Point3D(x=obj.position.x + half.x, y=obj.position.y + half.y, z=obj.position.z + half.z),
    )


def intersects(first: SceneObject, second: SceneObject) -> bool:
    first_min, first_max = object_bounds(first)
    second_min, second_max = object_bounds(second)

    return (
        first_min.x <= second_max.x
        and first_max.x >= second_min.x
        and first_min.y <= second_max.y
        and first_max.y >= second_min.y
        and first_min.z <= second_max.z
        and first_max.z >= second_min.z
    )


def inside(inner: SceneObject, outer: SceneObject, margin: float = 0.02) -> bool:
    inner_min, inner_max = object_bounds(inner)
    outer_min, outer_max = object_bounds(outer)

    return (
        inner_min.x >= (outer_min.x - margin)
        and inner_max.x <= (outer_max.x + margin)
        and inner_min.y >= (outer_min.y - margin)
        and inner_max.y <= (outer_max.y + margin)
        and inner_min.z >= (outer_min.z - margin)
        and inner_max.z <= (outer_max.z + margin)
    )


def point_in_bounds(point: Point3D, bounds_min: Point3D, bounds_max: Point3D) -> bool:
    return (
        bounds_min.x <= point.x <= bounds_max.x
        and bounds_min.y <= point.y <= bounds_max.y
        and bounds_min.z <= point.z <= bounds_max.z
    )


def supports(
    upper: SceneObject,
    lower: SceneObject,
    min_vertical_gap: float = 0.05,
    max_vertical_gap: float = 0.35,
    max_horizontal_gap: float = 0.8,
) -> bool:
    """Check if `upper` can plausibly be supported by `lower`."""
    vertical_gap = upper.position.z - lower.position.z
    if vertical_gap < min_vertical_gap:
        return False
    if vertical_gap > max_vertical_gap:
        return False

    return horizontal_distance(upper.position, lower.position) <= max_horizontal_gap
