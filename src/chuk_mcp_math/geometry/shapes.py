"""
Geometric Shapes

Functions for calculating areas, perimeters, and properties of geometric shapes.
All functions are async-native with MCP decoration for AI model integration.
"""

import asyncio
import math
from typing import List, Tuple
from ..mcp_decorator import mcp_function


@mcp_function(
    description="Calculate the area of a polygon using the Shoelace formula",
    namespace="geometry",
    cache_strategy="memory",
    estimated_cpu_usage="low",
    examples=[
        {
            "input": {"vertices": [[0, 0], [4, 0], [4, 3], [0, 3]]},
            "output": 12.0,
            "description": "Area of a 4×3 rectangle",
        }
    ],
)
async def geom_polygon_area(vertices: List[Tuple[float, float]]) -> float:
    """
    Calculate the area of a polygon using the Shoelace formula.

    Works for any simple polygon (non-self-intersecting).
    Vertices should be ordered (either clockwise or counter-clockwise).

    Formula:
        A = ½ |Σ(xᵢyᵢ₊₁ - xᵢ₊₁yᵢ)|

    Args:
        vertices: List of polygon vertices [(x1, y1), (x2, y2), ...]

    Returns:
        Area of the polygon

    Raises:
        ValueError: If polygon has fewer than 3 vertices

    Examples:
        >>> square = [(0, 0), (4, 0), (4, 4), (0, 4)]
        >>> await geom_polygon_area(square)
        16.0
        >>> triangle = [(0, 0), (4, 0), (2, 3)]
        >>> await geom_polygon_area(triangle)
        6.0
    """
    if len(vertices) < 3:
        raise ValueError(f"Polygon must have at least 3 vertices, got {len(vertices)}")

    await asyncio.sleep(0)

    n = len(vertices)
    area = 0.0

    for i in range(n):
        j = (i + 1) % n
        area += vertices[i][0] * vertices[j][1]
        area -= vertices[j][0] * vertices[i][1]

    return abs(area) / 2.0


@mcp_function(
    description="Calculate the area of a circle",
    namespace="geometry",
    cache_strategy="memory",
    estimated_cpu_usage="low",
)
async def geom_circle_area(radius: float) -> float:
    """
    Calculate the area of a circle.

    Formula:
        A = πr²

    Args:
        radius: Radius of the circle

    Returns:
        Area of the circle

    Raises:
        ValueError: If radius is negative

    Examples:
        >>> await geom_circle_area(5.0)
        78.539...  # π × 25
        >>> await geom_circle_area(1.0)
        3.141...  # π
    """
    if radius < 0:
        raise ValueError(f"Radius cannot be negative: {radius}")

    await asyncio.sleep(0)

    return float(math.pi * radius**2)


@mcp_function(
    description="Calculate the circumference of a circle",
    namespace="geometry",
    cache_strategy="memory",
    estimated_cpu_usage="low",
)
async def geom_circle_circumference(radius: float) -> float:
    """
    Calculate the circumference (perimeter) of a circle.

    Formula:
        C = 2πr

    Args:
        radius: Radius of the circle

    Returns:
        Circumference of the circle

    Raises:
        ValueError: If radius is negative

    Examples:
        >>> await geom_circle_circumference(5.0)
        31.415...  # 2π × 5
        >>> await geom_circle_circumference(1.0)
        6.283...  # 2π
    """
    if radius < 0:
        raise ValueError(f"Radius cannot be negative: {radius}")

    await asyncio.sleep(0)

    return float(2 * math.pi * radius)


@mcp_function(
    description="Calculate the area of a triangle given three vertices",
    namespace="geometry",
    cache_strategy="memory",
    estimated_cpu_usage="low",
    examples=[
        {
            "input": {"p1": [0, 0], "p2": [4, 0], "p3": [2, 3]},
            "output": 6.0,
            "description": "Area of triangle with base 4 and height 3",
        }
    ],
)
async def geom_triangle_area(
    p1: Tuple[float, float], p2: Tuple[float, float], p3: Tuple[float, float]
) -> float:
    """
    Calculate the area of a triangle given three vertices.

    Uses the cross product formula:
        A = ½ |x₁(y₂ - y₃) + x₂(y₃ - y₁) + x₃(y₁ - y₂)|

    Args:
        p1: First vertex (x, y)
        p2: Second vertex (x, y)
        p3: Third vertex (x, y)

    Returns:
        Area of the triangle

    Examples:
        >>> await geom_triangle_area((0, 0), (4, 0), (2, 3))
        6.0
        >>> await geom_triangle_area((0, 0), (1, 0), (0, 1))
        0.5
    """
    await asyncio.sleep(0)

    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3

    area = abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2.0

    return float(area)
