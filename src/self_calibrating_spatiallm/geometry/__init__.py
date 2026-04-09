"""Geometry utility namespace."""

from self_calibrating_spatiallm.geometry.ops import (
    array_to_points,
    centroid,
    euclidean_distance,
    horizontal_distance,
    inside,
    intersects,
    normalize_vector,
    object_bounds,
    point_in_bounds,
    points_to_array,
    supports,
)

__all__ = [
    "array_to_points",
    "centroid",
    "euclidean_distance",
    "horizontal_distance",
    "inside",
    "intersects",
    "normalize_vector",
    "object_bounds",
    "point_in_bounds",
    "points_to_array",
    "supports",
]
