"""
Geometric Distances

Functions for calculating distances between points in 2D and 3D space.
All functions are async-native with MCP decoration for AI model integration.
"""

import asyncio
import math
from typing import Tuple
from ..mcp_decorator import mcp_function


@mcp_function(
    description="Calculate Euclidean distance between two 2D points",
    namespace="geometry",
    cache_strategy="memory",
    estimated_cpu_usage="low",
    examples=[
        {
            "input": {"p1": [0, 0], "p2": [3, 4]},
            "output": 5.0,
            "description": "Distance from origin to (3,4)",
        }
    ],
)
async def geom_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """
    Calculate the Euclidean distance between two points in 2D space.

    Formula:
        d = √((x₂-x₁)² + (y₂-y₁)²)

    Args:
        p1: First point (x, y)
        p2: Second point (x, y)

    Returns:
        Euclidean distance between the points

    Examples:
        >>> await geom_distance((0, 0), (3, 4))
        5.0
        >>> await geom_distance((1, 2), (4, 6))
        5.0
    """
    await asyncio.sleep(0)

    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]

    return float(math.sqrt(dx**2 + dy**2))


@mcp_function(
    description="Calculate Euclidean distance between two 3D points",
    namespace="geometry",
    cache_strategy="memory",
    estimated_cpu_usage="low",
)
async def geom_distance_3d(p1: Tuple[float, float, float], p2: Tuple[float, float, float]) -> float:
    """
    Calculate the Euclidean distance between two points in 3D space.

    Formula:
        d = √((x₂-x₁)² + (y₂-y₁)² + (z₂-z₁)²)

    Args:
        p1: First point (x, y, z)
        p2: Second point (x, y, z)

    Returns:
        Euclidean distance between the points

    Examples:
        >>> await geom_distance_3d((0, 0, 0), (1, 1, 1))
        1.732...
    """
    await asyncio.sleep(0)

    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    dz = p2[2] - p1[2]

    return float(math.sqrt(dx**2 + dy**2 + dz**2))


@mcp_function(
    description="Calculate great circle distance between two points on Earth (GPS coordinates)",
    namespace="geometry",
    cache_strategy="memory",
    estimated_cpu_usage="low",
    examples=[
        {
            "input": {"lat1": 40.7128, "lon1": -74.0060, "lat2": 51.5074, "lon2": -0.1278},
            "output": {"distance_km": 5570.0, "distance_miles": 3461.0},
            "description": "Distance from NYC to London",
        }
    ],
)
async def geom_great_circle_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> dict:
    """
    Calculate the great circle distance between two points on Earth using the Haversine formula.

    This gives the shortest distance between two points on a sphere (the Earth).
    Uses mean Earth radius of 6371 km.

    Formula:
        a = sin²(Δφ/2) + cos(φ₁) × cos(φ₂) × sin²(Δλ/2)
        c = 2 × atan2(√a, √(1−a))
        d = R × c

    Args:
        lat1: Latitude of first point in degrees
        lon1: Longitude of first point in degrees
        lat2: Latitude of second point in degrees
        lon2: Longitude of second point in degrees

    Returns:
        Dictionary with:
        - distance_km: Distance in kilometers
        - distance_miles: Distance in miles
        - distance_nm: Distance in nautical miles

    Examples:
        >>> result = await geom_great_circle_distance(40.7128, -74.0060, 51.5074, -0.1278)
        >>> result['distance_km']
        5570.2...  # NYC to London
    """
    await asyncio.sleep(0)

    # Convert to radians
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lat = math.radians(lat2 - lat1)
    delta_lon = math.radians(lon2 - lon1)

    # Haversine formula
    a = (
        math.sin(delta_lat / 2) ** 2
        + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon / 2) ** 2
    )

    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    # Distance in kilometers
    earth_radius_km = 6371.0
    distance_km = earth_radius_km * c

    # Convert to other units
    distance_miles = distance_km * 0.621371
    distance_nm = distance_km * 0.539957  # Nautical miles

    return {
        "distance_km": float(distance_km),
        "distance_miles": float(distance_miles),
        "distance_nm": float(distance_nm),
    }


@mcp_function(
    description="Calculate Manhattan (taxicab) distance between two 2D points",
    namespace="geometry",
    cache_strategy="memory",
    estimated_cpu_usage="low",
)
async def geom_manhattan_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """
    Calculate the Manhattan (taxicab) distance between two points in 2D space.

    The Manhattan distance is the sum of absolute differences of coordinates.
    Useful for grid-based navigation where diagonal movement isn't allowed.

    Formula:
        d = |x₂-x₁| + |y₂-y₁|

    Args:
        p1: First point (x, y)
        p2: Second point (x, y)

    Returns:
        Manhattan distance between the points

    Examples:
        >>> await geom_manhattan_distance((0, 0), (3, 4))
        7.0  # Must travel 3 units right + 4 units up
        >>> await geom_manhattan_distance((1, 2), (4, 6))
        7.0  # 3 + 4
    """
    await asyncio.sleep(0)

    dx = abs(p2[0] - p1[0])
    dy = abs(p2[1] - p1[1])

    return float(dx + dy)
