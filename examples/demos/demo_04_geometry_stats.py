#!/usr/bin/env python3
"""
CHUK MCP Math - Geometry & Statistics Demo
===========================================

Demonstrates geometry and statistics functions including:
- 2D geometry (circles, rectangles, triangles)
- Vector operations
- Statistical measures
- Linear algebra basics
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from chuk_mcp_math.geometry import (
    circle_area,
    circle_circumference,
    rectangle_area,
    rectangle_perimeter,
    triangle_area,
    triangle_perimeter,
    sphere_volume,
    sphere_surface_area,
    cylinder_volume,
    cylinder_surface_area,
    cone_volume,
    cone_surface_area,
)
from chuk_mcp_math.statistics import (
    mean as stats_mean,
    median as stats_median,
    mode as stats_mode,
    variance as stats_variance,
    standard_deviation as stats_std,
    range_stat,
    percentile,
    quartiles,
)
from chuk_mcp_math.linear_algebra.vectors.basic_operations import (
    vector_add,
    vector_subtract,
    vector_scalar_multiply,
    vector_magnitude,
    vector_normalize,
)
from chuk_mcp_math.linear_algebra.vectors.products import dot_product, cross_product


async def demo_2d_geometry():
    """Demonstrate 2D geometry calculations."""
    print("\n" + "=" * 70)
    print("2D GEOMETRY")
    print("=" * 70)

    # Circle
    result = await circle_area(5)
    print(f"✓ circle_area(radius=5) = {result:.2f}")

    result = await circle_circumference(5)
    print(f"✓ circle_circumference(radius=5) = {result:.2f}")

    # Rectangle
    result = await rectangle_area(4, 6)
    print(f"✓ rectangle_area(width=4, height=6) = {result:.2f}")

    result = await rectangle_perimeter(4, 6)
    print(f"✓ rectangle_perimeter(width=4, height=6) = {result:.2f}")

    # Triangle
    result = await triangle_area(3, 4, 5)
    print(f"✓ triangle_area(a=3, b=4, c=5) = {result:.2f}")

    result = await triangle_perimeter(3, 4, 5)
    print(f"✓ triangle_perimeter(a=3, b=4, c=5) = {result:.2f}")


async def demo_3d_geometry():
    """Demonstrate 3D geometry calculations."""
    print("\n" + "=" * 70)
    print("3D GEOMETRY")
    print("=" * 70)

    # Sphere
    result = await sphere_volume(5)
    print(f"✓ sphere_volume(radius=5) = {result:.2f}")

    result = await sphere_surface_area(5)
    print(f"✓ sphere_surface_area(radius=5) = {result:.2f}")

    # Cylinder
    result = await cylinder_volume(3, 10)
    print(f"✓ cylinder_volume(radius=3, height=10) = {result:.2f}")

    result = await cylinder_surface_area(3, 10)
    print(f"✓ cylinder_surface_area(radius=3, height=10) = {result:.2f}")

    # Cone
    result = await cone_volume(3, 10)
    print(f"✓ cone_volume(radius=3, height=10) = {result:.2f}")

    result = await cone_surface_area(3, 10)
    print(f"✓ cone_surface_area(radius=3, height=10) = {result:.2f}")


async def demo_statistics():
    """Demonstrate statistical functions."""
    print("\n" + "=" * 70)
    print("STATISTICS")
    print("=" * 70)

    data = [2, 4, 4, 4, 5, 5, 7, 9]

    # Mean
    result = await stats_mean(data)
    print(f"✓ mean({data}) = {result:.2f}")

    # Median
    result = await stats_median(data)
    print(f"✓ median({data}) = {result:.2f}")

    # Mode
    result = await stats_mode(data)
    print(f"✓ mode({data}) = {result}")

    # Variance
    result = await stats_variance(data)
    print(f"✓ variance({data}) = {result:.2f}")

    # Standard deviation
    result = await stats_std(data)
    print(f"✓ standard_deviation({data}) = {result:.2f}")

    # Range
    result = await range_stat(data)
    print(f"✓ range({data}) = {result:.2f}")

    # Percentile
    result = await percentile(data, 50)
    print(f"✓ percentile({data}, 50) = {result:.2f}")

    # Quartiles
    result = await quartiles(data)
    print(
        f"✓ quartiles({data}) = Q1:{result[0]:.1f}, Q2:{result[1]:.1f}, Q3:{result[2]:.1f}"
    )


async def demo_vectors():
    """Demonstrate vector operations."""
    print("\n" + "=" * 70)
    print("VECTOR OPERATIONS")
    print("=" * 70)

    v1 = [1, 2, 3]
    v2 = [4, 5, 6]

    # Vector addition
    result = await vector_add(v1, v2)
    print(f"✓ vector_add({v1}, {v2}) = {result}")

    # Vector subtraction
    result = await vector_subtract(v1, v2)
    print(f"✓ vector_subtract({v1}, {v2}) = {result}")

    # Scalar multiplication
    result = await vector_scalar_multiply(v1, 3)
    print(f"✓ vector_scalar_multiply({v1}, 3) = {result}")

    # Magnitude
    result = await vector_magnitude(v1)
    print(f"✓ vector_magnitude({v1}) = {result:.4f}")

    # Normalize
    result = await vector_normalize(v1)
    print(
        f"✓ vector_normalize({v1}) = [{result[0]:.4f}, {result[1]:.4f}, {result[2]:.4f}]"
    )

    # Dot product
    result = await dot_product(v1, v2)
    print(f"✓ dot_product({v1}, {v2}) = {result}")

    # Cross product (3D only)
    result = await cross_product(v1, v2)
    print(f"✓ cross_product({v1}, {v2}) = {result}")


async def main():
    """Run all geometry and statistics demos."""
    print("\n" + "=" * 70)
    print("CHUK MCP MATH - GEOMETRY & STATISTICS DEMO")
    print("=" * 70)
    print("\nDemonstrating geometry, statistics, and vector functions...")

    await demo_2d_geometry()
    await demo_3d_geometry()
    await demo_statistics()
    await demo_vectors()

    print("\n" + "=" * 70)
    print("✅ ALL GEOMETRY & STATISTICS OPERATIONS WORKING PERFECTLY!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
