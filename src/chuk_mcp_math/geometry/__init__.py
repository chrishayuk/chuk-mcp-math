"""
Geometry Module

Geometric calculations including distances, intersections, areas, and transformations.
All functions are async-native and MCP-decorated for AI model integration.
"""

from .distances import (
    geom_distance,
    geom_distance_3d,
    geom_great_circle_distance,
    geom_manhattan_distance,
)

from .intersections import (
    geom_line_intersection,
    geom_segment_intersection,
    geom_circle_intersection,
    geom_point_in_polygon,
)

from .shapes import (
    geom_polygon_area,
    geom_circle_area,
    geom_circle_circumference,
    geom_triangle_area,
)

__all__ = [
    # Distances
    "geom_distance",
    "geom_distance_3d",
    "geom_great_circle_distance",
    "geom_manhattan_distance",
    # Intersections
    "geom_line_intersection",
    "geom_segment_intersection",
    "geom_circle_intersection",
    "geom_point_in_polygon",
    # Shapes
    "geom_polygon_area",
    "geom_circle_area",
    "geom_circle_circumference",
    "geom_triangle_area",
]
