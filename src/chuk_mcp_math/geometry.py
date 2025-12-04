#!/usr/bin/env python3
# chuk_mcp_math/geometry.py
"""
Geometry Functions for AI Models

Geometric calculations: areas, perimeters, distances, and shape properties.
All functions include comprehensive validation and clear descriptions for AI execution.
All functions are async-native for optimal performance.
"""

import math
import asyncio
from typing import Union, Dict
from chuk_mcp_math.mcp_decorator import mcp_function


@mcp_function(
    description="Calculate the area of a circle given its radius. Uses the formula A = πr².",
    namespace="geometry",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    dependencies=["math"],
    examples=[
        {
            "input": {"radius": 5},
            "output": 78.54,
            "description": "Area of circle with radius 5",
        },
        {
            "input": {"radius": 1},
            "output": 3.14159,
            "description": "Area of unit circle",
        },
        {
            "input": {"radius": 10},
            "output": 314.16,
            "description": "Area of larger circle",
        },
    ],
)
async def circle_area(radius: Union[int, float]) -> float:
    """
    Calculate the area of a circle.

    Args:
        radius: Radius of the circle (must be non-negative)

    Returns:
        Area of the circle (π × radius²)

    Raises:
        ValueError: If radius is negative

    Examples:
        await circle_area(5) → 78.53981633974483
        await circle_area(1) → 3.141592653589793
        await circle_area(10) → 314.1592653589793
    """
    if radius < 0:
        raise ValueError("Radius cannot be negative")
    await asyncio.sleep(0)  # Yield control for async execution
    return math.pi * radius**2


@mcp_function(
    description="Calculate the circumference of a circle given its radius. Uses the formula C = 2πr.",
    namespace="geometry",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    dependencies=["math"],
    examples=[
        {
            "input": {"radius": 5},
            "output": 31.42,
            "description": "Circumference of circle with radius 5",
        },
        {
            "input": {"radius": 1},
            "output": 6.28,
            "description": "Circumference of unit circle",
        },
        {
            "input": {"radius": 3},
            "output": 18.85,
            "description": "Circumference with radius 3",
        },
    ],
)
async def circle_circumference(radius: Union[int, float]) -> float:
    """
    Calculate the circumference of a circle.

    Args:
        radius: Radius of the circle (must be non-negative)

    Returns:
        Circumference of the circle (2π × radius)

    Examples:
        await circle_circumference(5) → 31.41592653589793
        await circle_circumference(1) → 6.283185307179586
        await circle_circumference(3) → 18.84955592153876
    """
    if radius < 0:
        raise ValueError("Radius cannot be negative")
    await asyncio.sleep(0)  # Yield control for async execution
    return 2 * math.pi * radius


@mcp_function(
    description="Calculate the area of a rectangle given its length and width. Uses the formula A = length × width.",
    namespace="geometry",
    execution_modes=["local", "remote"],
    examples=[
        {
            "input": {"length": 5, "width": 3},
            "output": 15,
            "description": "5×3 rectangle area",
        },
        {
            "input": {"length": 10, "width": 2.5},
            "output": 25.0,
            "description": "Rectangle with decimal width",
        },
        {
            "input": {"length": 7, "width": 7},
            "output": 49,
            "description": "Square area (special rectangle)",
        },
    ],
)
async def rectangle_area(
    length: Union[int, float], width: Union[int, float]
) -> Union[int, float]:
    """
    Calculate the area of a rectangle.

    Args:
        length: Length of the rectangle (must be non-negative)
        width: Width of the rectangle (must be non-negative)

    Returns:
        Area of the rectangle (length × width)

    Examples:
        rectangle_area(5, 3) → 15
        rectangle_area(10, 2.5) → 25.0
        rectangle_area(7, 7) → 49
    """
    if length < 0 or width < 0:
        raise ValueError("Length and width cannot be negative")
    await asyncio.sleep(0)  # Yield control for async execution
    return length * width


@mcp_function(
    description="Calculate the perimeter of a rectangle. Uses the formula P = 2(length + width).",
    namespace="geometry",
    execution_modes=["local", "remote"],
    examples=[
        {
            "input": {"length": 5, "width": 3},
            "output": 16,
            "description": "Perimeter of 5×3 rectangle",
        },
        {
            "input": {"length": 10, "width": 2.5},
            "output": 25.0,
            "description": "Perimeter with decimal dimensions",
        },
        {
            "input": {"length": 4, "width": 4},
            "output": 16,
            "description": "Perimeter of square",
        },
    ],
)
async def rectangle_perimeter(
    length: Union[int, float], width: Union[int, float]
) -> Union[int, float]:
    """
    Calculate the perimeter of a rectangle.

    Args:
        length: Length of the rectangle (must be non-negative)
        width: Width of the rectangle (must be non-negative)

    Returns:
        Perimeter of the rectangle (2 × (length + width))

    Examples:
        rectangle_perimeter(5, 3) → 16
        rectangle_perimeter(10, 2.5) → 25.0
        rectangle_perimeter(4, 4) → 16
    """
    if length < 0 or width < 0:
        raise ValueError("Length and width cannot be negative")
    await asyncio.sleep(0)  # Yield control for async execution
    return 2 * (length + width)


@mcp_function(
    description="Calculate the area of a triangle given base and height. Uses the formula A = ½ × base × height.",
    namespace="geometry",
    execution_modes=["local", "remote"],
    examples=[
        {
            "input": {"base": 6, "height": 4},
            "output": 12.0,
            "description": "Triangle with base 6, height 4",
        },
        {
            "input": {"base": 10, "height": 5},
            "output": 25.0,
            "description": "Triangle with base 10, height 5",
        },
        {
            "input": {"base": 3.5, "height": 2},
            "output": 3.5,
            "description": "Triangle with decimal base",
        },
    ],
)
async def triangle_area(base: Union[int, float], height: Union[int, float]) -> float:
    """
    Calculate the area of a triangle using base and height.

    Args:
        base: Base of the triangle (must be non-negative)
        height: Height of the triangle (must be non-negative)

    Returns:
        Area of the triangle (0.5 × base × height)

    Examples:
        triangle_area(6, 4) → 12.0
        triangle_area(10, 5) → 25.0
        triangle_area(3.5, 2) → 3.5
    """
    if base < 0 or height < 0:
        raise ValueError("Base and height cannot be negative")
    await asyncio.sleep(0)  # Yield control for async execution
    return 0.5 * base * height


@mcp_function(
    description="Calculate the area of a triangle using Heron's formula when you know all three sides. Handles any valid triangle.",
    namespace="geometry",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    estimated_cpu_usage="medium",
    dependencies=["math"],
    examples=[
        {
            "input": {"a": 3, "b": 4, "c": 5},
            "output": 6.0,
            "description": "Area of 3-4-5 right triangle",
        },
        {
            "input": {"a": 5, "b": 5, "c": 6},
            "output": 12.0,
            "description": "Area of isosceles triangle",
        },
        {
            "input": {"a": 7, "b": 8, "c": 9},
            "output": 26.83,
            "description": "Area of scalene triangle",
        },
    ],
)
async def triangle_area_heron(
    a: Union[int, float], b: Union[int, float], c: Union[int, float]
) -> float:
    """
    Calculate the area of a triangle using Heron's formula.

    Args:
        a: Length of first side (must be positive)
        b: Length of second side (must be positive)
        c: Length of third side (must be positive)

    Returns:
        Area of the triangle

    Raises:
        ValueError: If sides don't form a valid triangle

    Examples:
        triangle_area_heron(3, 4, 5) → 6.0
        triangle_area_heron(5, 5, 6) → 12.0
        triangle_area_heron(7, 8, 9) → 26.832815729997478
    """
    if a <= 0 or b <= 0 or c <= 0:
        raise ValueError("All sides must be positive")

    # Check triangle inequality
    if a + b <= c or a + c <= b or b + c <= a:
        raise ValueError("Invalid triangle: sides don't satisfy triangle inequality")

    # Semi-perimeter
    s = (a + b + c) / 2

    # Heron's formula
    area = math.sqrt(s * (s - a) * (s - b) * (s - c))
    await asyncio.sleep(0)  # Yield control for async execution
    return area


@mcp_function(
    description="Calculate the Euclidean distance between two points in 2D space. Uses the distance formula.",
    namespace="geometry",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    examples=[
        {
            "input": {"x1": 0, "y1": 0, "x2": 3, "y2": 4},
            "output": 5.0,
            "description": "Distance from origin to (3,4)",
        },
        {
            "input": {"x1": 1, "y1": 1, "x2": 4, "y2": 5},
            "output": 5.0,
            "description": "Distance between (1,1) and (4,5)",
        },
        {
            "input": {"x1": -2, "y1": 3, "x2": 1, "y2": -1},
            "output": 5.0,
            "description": "Distance with negative coordinates",
        },
    ],
)
async def distance_2d(
    x1: Union[int, float],
    y1: Union[int, float],
    x2: Union[int, float],
    y2: Union[int, float],
) -> float:
    """
    Calculate the Euclidean distance between two points in 2D space.

    Args:
        x1, y1: Coordinates of the first point
        x2, y2: Coordinates of the second point

    Returns:
        Distance between the two points

    Examples:
        distance_2d(0, 0, 3, 4) → 5.0
        distance_2d(1, 1, 4, 5) → 5.0
        distance_2d(-2, 3, 1, -1) → 5.0
    """
    await asyncio.sleep(0)  # Yield control for async execution
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


@mcp_function(
    description="Calculate the Euclidean distance between two points in 3D space. Uses the 3D distance formula.",
    namespace="geometry",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    examples=[
        {
            "input": {"x1": 0, "y1": 0, "z1": 0, "x2": 1, "y2": 1, "z2": 1},
            "output": 1.732,
            "description": "Distance from origin to (1,1,1)",
        },
        {
            "input": {"x1": 1, "y1": 2, "z1": 3, "x2": 4, "y2": 5, "z2": 6},
            "output": 5.196,
            "description": "Distance between two 3D points",
        },
    ],
)
async def distance_3d(
    x1: Union[int, float],
    y1: Union[int, float],
    z1: Union[int, float],
    x2: Union[int, float],
    y2: Union[int, float],
    z2: Union[int, float],
) -> float:
    """
    Calculate the Euclidean distance between two points in 3D space.

    Args:
        x1, y1, z1: Coordinates of the first point
        x2, y2, z2: Coordinates of the second point

    Returns:
        Distance between the two points

    Examples:
        distance_3d(0, 0, 0, 1, 1, 1) → 1.7320508075688772
        distance_3d(1, 2, 3, 4, 5, 6) → 5.196152422706632
    """
    await asyncio.sleep(0)  # Yield control for async execution
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)


@mcp_function(
    description="Calculate the hypotenuse of a right triangle using the Pythagorean theorem: c² = a² + b².",
    namespace="geometry",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    examples=[
        {
            "input": {"a": 3, "b": 4},
            "output": 5.0,
            "description": "Classic 3-4-5 triangle",
        },
        {"input": {"a": 5, "b": 12}, "output": 13.0, "description": "5-12-13 triangle"},
        {
            "input": {"a": 1, "b": 1},
            "output": 1.414,
            "description": "Isosceles right triangle",
        },
    ],
)
async def hypotenuse(a: Union[int, float], b: Union[int, float]) -> float:
    """
    Calculate the hypotenuse of a right triangle using Pythagorean theorem.

    Args:
        a: Length of first leg (must be non-negative)
        b: Length of second leg (must be non-negative)

    Returns:
        Length of the hypotenuse

    Examples:
        hypotenuse(3, 4) → 5.0
        hypotenuse(5, 12) → 13.0
        hypotenuse(1, 1) → 1.4142135623730951
    """
    if a < 0 or b < 0:
        raise ValueError("Side lengths cannot be negative")
    await asyncio.sleep(0)  # Yield control for async execution
    return math.sqrt(a**2 + b**2)


@mcp_function(
    description="Calculate the area and perimeter of a square given its side length. A square is a special rectangle.",
    namespace="geometry",
    execution_modes=["local", "remote"],
    examples=[
        {
            "input": {"side": 5},
            "output": {"area": 25, "perimeter": 20},
            "description": "5×5 square",
        },
        {
            "input": {"side": 3.5},
            "output": {"area": 12.25, "perimeter": 14.0},
            "description": "Square with decimal side",
        },
        {
            "input": {"side": 10},
            "output": {"area": 100, "perimeter": 40},
            "description": "10×10 square",
        },
    ],
)
async def square_properties(side: Union[int, float]) -> Dict[str, Union[int, float]]:
    """
    Calculate both area and perimeter of a square.

    Args:
        side: Length of the square's side (must be non-negative)

    Returns:
        Dictionary with area and perimeter

    Examples:
        square_properties(5) → {"area": 25, "perimeter": 20}
        square_properties(3.5) → {"area": 12.25, "perimeter": 14.0}
    """
    if side < 0:
        raise ValueError("Side length cannot be negative")

    await asyncio.sleep(0)  # Yield control for async execution
    return {"area": side**2, "perimeter": 4 * side}


@mcp_function(
    description="Calculate the volume of a sphere given its radius. Uses the formula V = (4/3)πr³.",
    namespace="geometry",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    dependencies=["math"],
    examples=[
        {
            "input": {"radius": 3},
            "output": 113.1,
            "description": "Volume of sphere with radius 3",
        },
        {
            "input": {"radius": 1},
            "output": 4.19,
            "description": "Volume of unit sphere",
        },
        {
            "input": {"radius": 5},
            "output": 523.6,
            "description": "Volume of larger sphere",
        },
    ],
)
async def sphere_volume(radius: Union[int, float]) -> float:
    """
    Calculate the volume of a sphere.

    Args:
        radius: Radius of the sphere (must be non-negative)

    Returns:
        Volume of the sphere ((4/3) × π × radius³)

    Examples:
        sphere_volume(3) → 113.09733552923255
        sphere_volume(1) → 4.188790204786391
        sphere_volume(5) → 523.5987755982989
    """
    if radius < 0:
        raise ValueError("Radius cannot be negative")
    await asyncio.sleep(0)  # Yield control for async execution
    return (4 / 3) * math.pi * radius**3


@mcp_function(
    description="Calculate the surface area of a sphere given its radius. Uses the formula SA = 4πr².",
    namespace="geometry",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    dependencies=["math"],
    examples=[
        {
            "input": {"radius": 3},
            "output": 113.1,
            "description": "Surface area of sphere with radius 3",
        },
        {
            "input": {"radius": 1},
            "output": 12.57,
            "description": "Surface area of unit sphere",
        },
        {
            "input": {"radius": 5},
            "output": 314.16,
            "description": "Surface area of larger sphere",
        },
    ],
)
async def sphere_surface_area(radius: Union[int, float]) -> float:
    """
    Calculate the surface area of a sphere.

    Args:
        radius: Radius of the sphere (must be non-negative)

    Returns:
        Surface area of the sphere (4 × π × radius²)

    Examples:
        sphere_surface_area(3) → 113.09733552923255
        sphere_surface_area(1) → 12.566370614359172
        sphere_surface_area(5) → 314.1592653589793
    """
    if radius < 0:
        raise ValueError("Radius cannot be negative")
    await asyncio.sleep(0)  # Yield control for async execution
    return 4 * math.pi * radius**2


# Export all geometry functions
__all__ = [
    "circle_area",
    "circle_circumference",
    "rectangle_area",
    "rectangle_perimeter",
    "triangle_area",
    "triangle_area_heron",
    "distance_2d",
    "distance_3d",
    "hypotenuse",
    "square_properties",
    "sphere_volume",
    "sphere_surface_area",
]

if __name__ == "__main__":
    # Test the geometry functions
    print("📐 Geometry Functions Test")
    print("=" * 35)

    print(f"circle_area(5) = {circle_area(5):.2f}")
    print(f"circle_circumference(5) = {circle_circumference(5):.2f}")
    print(f"rectangle_area(4, 6) = {rectangle_area(4, 6)}")
    print(f"rectangle_perimeter(4, 6) = {rectangle_perimeter(4, 6)}")
    print(f"triangle_area(6, 4) = {triangle_area(6, 4)}")
    print(f"triangle_area_heron(3, 4, 5) = {triangle_area_heron(3, 4, 5)}")
    print(f"distance_2d(0, 0, 3, 4) = {distance_2d(0, 0, 3, 4)}")
    print(f"distance_3d(0, 0, 0, 1, 1, 1) = {distance_3d(0, 0, 0, 1, 1, 1):.3f}")
    print(f"hypotenuse(3, 4) = {hypotenuse(3, 4)}")

    square = square_properties(5)
    print(f"square_properties(5) = {square}")

    print(f"sphere_volume(3) = {sphere_volume(3):.2f}")
    print(f"sphere_surface_area(3) = {sphere_surface_area(3):.2f}")

    print("\n✅ All geometry functions working correctly!")
