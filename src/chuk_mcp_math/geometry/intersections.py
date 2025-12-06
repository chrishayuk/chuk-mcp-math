"""
Geometric Intersections

Functions for finding intersections between geometric objects.
All functions are async-native with MCP decoration for AI model integration.
"""

import asyncio
import math
from typing import Tuple, List, Dict, Any
from ..mcp_decorator import mcp_function


@mcp_function(
    description="Find intersection point of two lines in 2D space",
    namespace="geometry",
    cache_strategy="memory",
    estimated_cpu_usage="low",
    examples=[
        {
            "input": {"line1": {"p1": [0, 0], "p2": [1, 1]}, "line2": {"p1": [0, 1], "p2": [1, 0]}},
            "output": {"x": 0.5, "y": 0.5, "intersects": True},
            "description": "Two lines intersecting at (0.5, 0.5)",
        }
    ],
)
async def geom_line_intersection(
    line1: Dict[str, Tuple[float, float]], line2: Dict[str, Tuple[float, float]]
) -> Dict[str, Any]:
    """
    Find the intersection point of two infinite lines in 2D space.

    Each line is defined by two points. The function extends the lines infinitely
    and finds where they intersect (if they do).

    Args:
        line1: Dict with 'p1' and 'p2' keys, each a (x, y) tuple
        line2: Dict with 'p1' and 'p2' keys, each a (x, y) tuple

    Returns:
        Dictionary with:
        - intersects: Boolean indicating if lines intersect
        - x, y: Intersection coordinates (if intersects is True)
        - parallel: Boolean indicating if lines are parallel
        - coincident: Boolean indicating if lines are the same

    Examples:
        >>> result = await geom_line_intersection(
        ...     {"p1": (0, 0), "p2": (1, 1)},
        ...     {"p1": (0, 1), "p2": (1, 0)}
        ... )
        >>> result['x'], result['y']
        (0.5, 0.5)
    """
    await asyncio.sleep(0)

    x1, y1 = line1["p1"]
    x2, y2 = line1["p2"]
    x3, y3 = line2["p1"]
    x4, y4 = line2["p2"]

    # Calculate denominators for parametric equations
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

    # Lines are parallel if denominator is zero
    if abs(denom) < 1e-10:
        # Check if lines are coincident (the same line)
        # by checking if a point from one line lies on the other
        t_num = (x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)

        if abs(t_num) < 1e-10:
            return {
                "intersects": False,
                "parallel": True,
                "coincident": True,
            }
        else:
            return {
                "intersects": False,
                "parallel": True,
                "coincident": False,
            }

    # Calculate intersection point using parametric form
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom

    # Calculate intersection coordinates
    x = x1 + t * (x2 - x1)
    y = y1 + t * (y2 - y1)

    return {
        "intersects": True,
        "x": float(x),
        "y": float(y),
        "parallel": False,
        "coincident": False,
    }


@mcp_function(
    description="Find intersection point of two line segments in 2D space",
    namespace="geometry",
    cache_strategy="memory",
    estimated_cpu_usage="low",
)
async def geom_segment_intersection(
    seg1: Dict[str, Tuple[float, float]], seg2: Dict[str, Tuple[float, float]]
) -> Dict[str, Any]:
    """
    Find the intersection point of two line segments (not infinite lines).

    Unlike line intersection, this only returns an intersection if it occurs
    within the bounds of both segments.

    Args:
        seg1: Dict with 'p1' and 'p2' keys for segment endpoints
        seg2: Dict with 'p1' and 'p2' keys for segment endpoints

    Returns:
        Dictionary with:
        - intersects: Boolean indicating if segments intersect
        - x, y: Intersection coordinates (if intersects is True)

    Examples:
        >>> result = await geom_segment_intersection(
        ...     {"p1": (0, 0), "p2": (2, 2)},
        ...     {"p1": (0, 2), "p2": (2, 0)}
        ... )
        >>> result['x'], result['y']
        (1.0, 1.0)
    """
    await asyncio.sleep(0)

    x1, y1 = seg1["p1"]
    x2, y2 = seg1["p2"]
    x3, y3 = seg2["p1"]
    x4, y4 = seg2["p2"]

    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

    # Segments are parallel
    if abs(denom) < 1e-10:
        return {"intersects": False}

    # Calculate parameters for both segments
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom

    # Check if intersection occurs within both segments
    if 0 <= t <= 1 and 0 <= u <= 1:
        x = x1 + t * (x2 - x1)
        y = y1 + t * (y2 - y1)

        return {
            "intersects": True,
            "x": float(x),
            "y": float(y),
        }

    return {"intersects": False}


@mcp_function(
    description="Find intersection points of two circles",
    namespace="geometry",
    cache_strategy="memory",
    estimated_cpu_usage="low",
)
async def geom_circle_intersection(
    circle1: Dict[str, float], circle2: Dict[str, float]
) -> Dict[str, Any]:
    """
    Find the intersection point(s) of two circles in 2D space.

    Args:
        circle1: Dict with 'x', 'y' (center) and 'r' (radius)
        circle2: Dict with 'x', 'y' (center) and 'r' (radius)

    Returns:
        Dictionary with:
        - intersects: Boolean
        - count: Number of intersection points (0, 1, 2, or infinite)
        - points: List of intersection points [(x1, y1), (x2, y2)]
        - type: "separate", "touching", "intersecting", "coincident", or "contained"

    Examples:
        >>> result = await geom_circle_intersection(
        ...     {"x": 0, "y": 0, "r": 5},
        ...     {"x": 5, "y": 0, "r": 5}
        ... )
        >>> result['count']
        2
    """
    await asyncio.sleep(0)

    x1, y1, r1 = circle1["x"], circle1["y"], circle1["r"]
    x2, y2, r2 = circle2["x"], circle2["y"], circle2["r"]

    # Distance between centers
    d = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    # Check for various cases
    if abs(d) < 1e-10 and abs(r1 - r2) < 1e-10:
        # Circles are coincident (same circle)
        return {
            "intersects": True,
            "count": float("inf"),
            "points": [],
            "type": "coincident",
        }

    if d > r1 + r2:
        # Circles are separate
        return {
            "intersects": False,
            "count": 0,
            "points": [],
            "type": "separate",
        }

    if d < abs(r1 - r2):
        # One circle is contained in the other
        return {
            "intersects": False,
            "count": 0,
            "points": [],
            "type": "contained",
        }

    if abs(d - (r1 + r2)) < 1e-10:
        # Circles are touching externally (1 intersection point)
        t = r1 / d
        x = x1 + t * (x2 - x1)
        y = y1 + t * (y2 - y1)

        return {
            "intersects": True,
            "count": 1,
            "points": [(float(x), float(y))],
            "type": "touching",
        }

    if abs(d - abs(r1 - r2)) < 1e-10:
        # Circles are touching internally (1 intersection point)
        if r1 > r2:
            t = r1 / d
        else:
            t = -r1 / d
        x = x1 + t * (x2 - x1)
        y = y1 + t * (y2 - y1)

        return {
            "intersects": True,
            "count": 1,
            "points": [(float(x), float(y))],
            "type": "touching",
        }

    # Two intersection points
    a = (r1**2 - r2**2 + d**2) / (2 * d)
    h = math.sqrt(r1**2 - a**2)

    # Point on line between centers
    px = x1 + a * (x2 - x1) / d
    py = y1 + a * (y2 - y1) / d

    # Two intersection points
    x_i1 = px + h * (y2 - y1) / d
    y_i1 = py - h * (x2 - x1) / d

    x_i2 = px - h * (y2 - y1) / d
    y_i2 = py + h * (x2 - x1) / d

    return {
        "intersects": True,
        "count": 2,
        "points": [(float(x_i1), float(y_i1)), (float(x_i2), float(y_i2))],
        "type": "intersecting",
    }


@mcp_function(
    description="Check if a point is inside a polygon using ray casting algorithm",
    namespace="geometry",
    cache_strategy="memory",
    estimated_cpu_usage="low",
)
async def geom_point_in_polygon(
    point: Tuple[float, float], polygon: List[Tuple[float, float]]
) -> bool:
    """
    Determine if a point is inside a polygon using the ray casting algorithm.

    Casts a ray from the point to infinity and counts intersections with polygon edges.
    If odd number of intersections, point is inside; if even, it's outside.

    Args:
        point: Point to test (x, y)
        polygon: List of polygon vertices [(x1, y1), (x2, y2), ...]

    Returns:
        True if point is inside polygon, False otherwise

    Examples:
        >>> square = [(0, 0), (4, 0), (4, 4), (0, 4)]
        >>> await geom_point_in_polygon((2, 2), square)
        True
        >>> await geom_point_in_polygon((5, 5), square)
        False
    """
    await asyncio.sleep(0)

    x, y = point
    n = len(polygon)
    inside = False

    p1x, p1y = polygon[0]

    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]

        # Check if point is on a horizontal edge
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside

        p1x, p1y = p2x, p2y

    return inside
