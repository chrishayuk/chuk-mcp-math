#!/usr/bin/env python3
# tests/math/test_geometry.py
"""
Comprehensive pytest unit tests for geometry functions.

Tests cover:
- Circle calculations (area, circumference)
- Rectangle calculations (area, perimeter)
- Triangle calculations (area with base/height and Heron's formula)
- Distance calculations (2D and 3D)
- Pythagorean theorem (hypotenuse)
- Square properties
- Sphere calculations (volume, surface area)
- Normal operation cases
- Edge cases (zero, negative)
- Error conditions
- Async behavior
- Mathematical relationships
"""

import pytest
import math
import asyncio

# Import the functions to test
from chuk_mcp_math.geometry import (
    circle_area,
    circle_circumference,
    rectangle_area,
    rectangle_perimeter,
    triangle_area,
    triangle_area_heron,
    distance_2d,
    distance_3d,
    hypotenuse,
    square_properties,
    sphere_volume,
    sphere_surface_area,
)


# Circle Tests
class TestCircleArea:
    """Test cases for circle_area function."""

    @pytest.mark.asyncio
    async def test_circle_area_unit_circle(self):
        """Test area of unit circle."""
        result = await circle_area(1)
        assert pytest.approx(result, rel=1e-10) == math.pi

    @pytest.mark.asyncio
    async def test_circle_area_radius_5(self):
        """Test area of circle with radius 5."""
        result = await circle_area(5)
        assert pytest.approx(result, rel=1e-10) == math.pi * 25

    @pytest.mark.asyncio
    async def test_circle_area_zero_radius(self):
        """Test area of circle with zero radius."""
        result = await circle_area(0)
        assert result == 0

    @pytest.mark.asyncio
    async def test_circle_area_decimal_radius(self):
        """Test area with decimal radius."""
        result = await circle_area(2.5)
        assert pytest.approx(result, rel=1e-10) == math.pi * 6.25

    @pytest.mark.asyncio
    async def test_circle_area_negative_radius_raises_error(self):
        """Test that negative radius raises ValueError."""
        with pytest.raises(ValueError, match="Radius cannot be negative"):
            await circle_area(-5)


class TestCircleCircumference:
    """Test cases for circle_circumference function."""

    @pytest.mark.asyncio
    async def test_circle_circumference_unit_circle(self):
        """Test circumference of unit circle."""
        result = await circle_circumference(1)
        assert pytest.approx(result, rel=1e-10) == 2 * math.pi

    @pytest.mark.asyncio
    async def test_circle_circumference_radius_5(self):
        """Test circumference of circle with radius 5."""
        result = await circle_circumference(5)
        assert pytest.approx(result, rel=1e-10) == 2 * math.pi * 5

    @pytest.mark.asyncio
    async def test_circle_circumference_zero_radius(self):
        """Test circumference with zero radius."""
        result = await circle_circumference(0)
        assert result == 0

    @pytest.mark.asyncio
    async def test_circle_circumference_negative_radius_raises_error(self):
        """Test that negative radius raises ValueError."""
        with pytest.raises(ValueError, match="Radius cannot be negative"):
            await circle_circumference(-3)


# Rectangle Tests
class TestRectangleArea:
    """Test cases for rectangle_area function."""

    @pytest.mark.asyncio
    async def test_rectangle_area_integers(self):
        """Test rectangle area with integers."""
        result = await rectangle_area(5, 3)
        assert result == 15

    @pytest.mark.asyncio
    async def test_rectangle_area_decimals(self):
        """Test rectangle area with decimals."""
        result = await rectangle_area(10, 2.5)
        assert result == 25.0

    @pytest.mark.asyncio
    async def test_rectangle_area_square(self):
        """Test rectangle area for square (equal sides)."""
        result = await rectangle_area(7, 7)
        assert result == 49

    @pytest.mark.asyncio
    async def test_rectangle_area_zero_dimension(self):
        """Test rectangle with zero dimension."""
        assert await rectangle_area(0, 5) == 0
        assert await rectangle_area(5, 0) == 0

    @pytest.mark.asyncio
    async def test_rectangle_area_negative_raises_error(self):
        """Test that negative dimensions raise ValueError."""
        with pytest.raises(ValueError, match="Length and width cannot be negative"):
            await rectangle_area(-5, 3)
        with pytest.raises(ValueError, match="Length and width cannot be negative"):
            await rectangle_area(5, -3)


class TestRectanglePerimeter:
    """Test cases for rectangle_perimeter function."""

    @pytest.mark.asyncio
    async def test_rectangle_perimeter_integers(self):
        """Test rectangle perimeter with integers."""
        result = await rectangle_perimeter(5, 3)
        assert result == 16

    @pytest.mark.asyncio
    async def test_rectangle_perimeter_decimals(self):
        """Test rectangle perimeter with decimals."""
        result = await rectangle_perimeter(10, 2.5)
        assert result == 25.0

    @pytest.mark.asyncio
    async def test_rectangle_perimeter_square(self):
        """Test rectangle perimeter for square."""
        result = await rectangle_perimeter(4, 4)
        assert result == 16

    @pytest.mark.asyncio
    async def test_rectangle_perimeter_negative_raises_error(self):
        """Test that negative dimensions raise ValueError."""
        with pytest.raises(ValueError, match="Length and width cannot be negative"):
            await rectangle_perimeter(-5, 3)


# Triangle Tests
class TestTriangleArea:
    """Test cases for triangle_area function."""

    @pytest.mark.asyncio
    async def test_triangle_area_standard(self):
        """Test triangle area with standard dimensions."""
        result = await triangle_area(6, 4)
        assert result == 12.0

    @pytest.mark.asyncio
    async def test_triangle_area_decimals(self):
        """Test triangle area with decimal dimensions."""
        result = await triangle_area(3.5, 2)
        assert result == 3.5

    @pytest.mark.asyncio
    async def test_triangle_area_zero_dimension(self):
        """Test triangle with zero dimension."""
        assert await triangle_area(0, 5) == 0
        assert await triangle_area(5, 0) == 0

    @pytest.mark.asyncio
    async def test_triangle_area_negative_raises_error(self):
        """Test that negative dimensions raise ValueError."""
        with pytest.raises(ValueError, match="Base and height cannot be negative"):
            await triangle_area(-6, 4)
        with pytest.raises(ValueError, match="Base and height cannot be negative"):
            await triangle_area(6, -4)


class TestTriangleAreaHeron:
    """Test cases for triangle_area_heron function."""

    @pytest.mark.asyncio
    async def test_triangle_area_heron_3_4_5(self):
        """Test Heron's formula with 3-4-5 right triangle."""
        result = await triangle_area_heron(3, 4, 5)
        assert pytest.approx(result, rel=1e-10) == 6.0

    @pytest.mark.asyncio
    async def test_triangle_area_heron_isosceles(self):
        """Test Heron's formula with isosceles triangle."""
        result = await triangle_area_heron(5, 5, 6)
        assert pytest.approx(result, rel=1e-10) == 12.0

    @pytest.mark.asyncio
    async def test_triangle_area_heron_scalene(self):
        """Test Heron's formula with scalene triangle."""
        result = await triangle_area_heron(7, 8, 9)
        assert pytest.approx(result, rel=1e-6) == 26.832815729997478

    @pytest.mark.asyncio
    async def test_triangle_area_heron_equilateral(self):
        """Test Heron's formula with equilateral triangle."""
        result = await triangle_area_heron(2, 2, 2)
        expected = math.sqrt(3)  # For side length 2
        assert pytest.approx(result, rel=1e-10) == expected

    @pytest.mark.asyncio
    async def test_triangle_area_heron_invalid_triangle(self):
        """Test that invalid triangle raises ValueError."""
        # Violates triangle inequality: a + b <= c
        with pytest.raises(ValueError, match="Invalid triangle"):
            await triangle_area_heron(1, 2, 10)

    @pytest.mark.asyncio
    async def test_triangle_area_heron_negative_side(self):
        """Test that negative side raises ValueError."""
        with pytest.raises(ValueError, match="All sides must be positive"):
            await triangle_area_heron(-3, 4, 5)

    @pytest.mark.asyncio
    async def test_triangle_area_heron_zero_side(self):
        """Test that zero side raises ValueError."""
        with pytest.raises(ValueError, match="All sides must be positive"):
            await triangle_area_heron(0, 4, 5)


# Distance Tests
class TestDistance2D:
    """Test cases for distance_2d function."""

    @pytest.mark.asyncio
    async def test_distance_2d_origin_to_point(self):
        """Test distance from origin to (3,4)."""
        result = await distance_2d(0, 0, 3, 4)
        assert result == 5.0

    @pytest.mark.asyncio
    async def test_distance_2d_same_point(self):
        """Test distance from point to itself."""
        result = await distance_2d(1, 1, 1, 1)
        assert result == 0.0

    @pytest.mark.asyncio
    async def test_distance_2d_negative_coordinates(self):
        """Test distance with negative coordinates."""
        result = await distance_2d(-2, 3, 1, -1)
        assert result == 5.0

    @pytest.mark.asyncio
    async def test_distance_2d_horizontal_line(self):
        """Test distance along horizontal line."""
        result = await distance_2d(0, 0, 5, 0)
        assert result == 5.0

    @pytest.mark.asyncio
    async def test_distance_2d_vertical_line(self):
        """Test distance along vertical line."""
        result = await distance_2d(0, 0, 0, 5)
        assert result == 5.0

    @pytest.mark.asyncio
    async def test_distance_2d_decimal_coordinates(self):
        """Test distance with decimal coordinates."""
        result = await distance_2d(0.5, 0.5, 1.5, 1.5)
        assert pytest.approx(result, rel=1e-10) == math.sqrt(2)


class TestDistance3D:
    """Test cases for distance_3d function."""

    @pytest.mark.asyncio
    async def test_distance_3d_origin_to_point(self):
        """Test distance from origin to (1,1,1)."""
        result = await distance_3d(0, 0, 0, 1, 1, 1)
        assert pytest.approx(result, rel=1e-10) == math.sqrt(3)

    @pytest.mark.asyncio
    async def test_distance_3d_same_point(self):
        """Test distance from point to itself."""
        result = await distance_3d(1, 2, 3, 1, 2, 3)
        assert result == 0.0

    @pytest.mark.asyncio
    async def test_distance_3d_negative_coordinates(self):
        """Test distance with negative coordinates."""
        result = await distance_3d(1, 2, 3, 4, 5, 6)
        assert pytest.approx(result, rel=1e-10) == math.sqrt(27)

    @pytest.mark.asyncio
    async def test_distance_3d_axis_aligned(self):
        """Test distance along x-axis."""
        result = await distance_3d(0, 0, 0, 5, 0, 0)
        assert result == 5.0


# Hypotenuse Tests
class TestHypotenuse:
    """Test cases for hypotenuse function."""

    @pytest.mark.asyncio
    async def test_hypotenuse_3_4_5(self):
        """Test classic 3-4-5 triangle."""
        result = await hypotenuse(3, 4)
        assert result == 5.0

    @pytest.mark.asyncio
    async def test_hypotenuse_5_12_13(self):
        """Test 5-12-13 triangle."""
        result = await hypotenuse(5, 12)
        assert result == 13.0

    @pytest.mark.asyncio
    async def test_hypotenuse_isosceles(self):
        """Test isosceles right triangle."""
        result = await hypotenuse(1, 1)
        assert pytest.approx(result, rel=1e-10) == math.sqrt(2)

    @pytest.mark.asyncio
    async def test_hypotenuse_zero_leg(self):
        """Test hypotenuse with zero leg."""
        result = await hypotenuse(0, 5)
        assert result == 5.0

    @pytest.mark.asyncio
    async def test_hypotenuse_negative_raises_error(self):
        """Test that negative leg raises ValueError."""
        with pytest.raises(ValueError, match="Side lengths cannot be negative"):
            await hypotenuse(-3, 4)
        with pytest.raises(ValueError, match="Side lengths cannot be negative"):
            await hypotenuse(3, -4)


# Square Tests
class TestSquareProperties:
    """Test cases for square_properties function."""

    @pytest.mark.asyncio
    async def test_square_properties_integer_side(self):
        """Test square properties with integer side."""
        result = await square_properties(5)
        assert result["area"] == 25
        assert result["perimeter"] == 20

    @pytest.mark.asyncio
    async def test_square_properties_decimal_side(self):
        """Test square properties with decimal side."""
        result = await square_properties(3.5)
        assert result["area"] == 12.25
        assert result["perimeter"] == 14.0

    @pytest.mark.asyncio
    async def test_square_properties_unit_square(self):
        """Test properties of unit square."""
        result = await square_properties(1)
        assert result["area"] == 1
        assert result["perimeter"] == 4

    @pytest.mark.asyncio
    async def test_square_properties_zero_side(self):
        """Test square with zero side."""
        result = await square_properties(0)
        assert result["area"] == 0
        assert result["perimeter"] == 0

    @pytest.mark.asyncio
    async def test_square_properties_negative_raises_error(self):
        """Test that negative side raises ValueError."""
        with pytest.raises(ValueError, match="Side length cannot be negative"):
            await square_properties(-5)


# Sphere Tests
class TestSphereVolume:
    """Test cases for sphere_volume function."""

    @pytest.mark.asyncio
    async def test_sphere_volume_radius_3(self):
        """Test sphere volume with radius 3."""
        result = await sphere_volume(3)
        expected = (4 / 3) * math.pi * 27
        assert pytest.approx(result, rel=1e-10) == expected

    @pytest.mark.asyncio
    async def test_sphere_volume_unit_sphere(self):
        """Test volume of unit sphere."""
        result = await sphere_volume(1)
        expected = (4 / 3) * math.pi
        assert pytest.approx(result, rel=1e-10) == expected

    @pytest.mark.asyncio
    async def test_sphere_volume_zero_radius(self):
        """Test sphere with zero radius."""
        result = await sphere_volume(0)
        assert result == 0

    @pytest.mark.asyncio
    async def test_sphere_volume_negative_raises_error(self):
        """Test that negative radius raises ValueError."""
        with pytest.raises(ValueError, match="Radius cannot be negative"):
            await sphere_volume(-5)


class TestSphereSurfaceArea:
    """Test cases for sphere_surface_area function."""

    @pytest.mark.asyncio
    async def test_sphere_surface_area_radius_3(self):
        """Test sphere surface area with radius 3."""
        result = await sphere_surface_area(3)
        expected = 4 * math.pi * 9
        assert pytest.approx(result, rel=1e-10) == expected

    @pytest.mark.asyncio
    async def test_sphere_surface_area_unit_sphere(self):
        """Test surface area of unit sphere."""
        result = await sphere_surface_area(1)
        expected = 4 * math.pi
        assert pytest.approx(result, rel=1e-10) == expected

    @pytest.mark.asyncio
    async def test_sphere_surface_area_zero_radius(self):
        """Test sphere with zero radius."""
        result = await sphere_surface_area(0)
        assert result == 0

    @pytest.mark.asyncio
    async def test_sphere_surface_area_negative_raises_error(self):
        """Test that negative radius raises ValueError."""
        with pytest.raises(ValueError, match="Radius cannot be negative"):
            await sphere_surface_area(-5)


# Mathematical Relationships Tests
class TestMathematicalRelationships:
    """Test mathematical relationships between geometry functions."""

    @pytest.mark.asyncio
    async def test_pythagorean_theorem(self):
        """Test Pythagorean theorem: a² + b² = c²."""
        a, b = 3, 4
        c = await hypotenuse(a, b)
        assert pytest.approx(a**2 + b**2, rel=1e-10) == c**2

    @pytest.mark.asyncio
    async def test_distance_2d_equals_hypotenuse(self):
        """Test that 2D distance from origin equals hypotenuse."""
        a, b = 3, 4
        dist = await distance_2d(0, 0, a, b)
        hyp = await hypotenuse(a, b)
        assert pytest.approx(dist, rel=1e-10) == hyp

    @pytest.mark.asyncio
    async def test_square_equals_rectangle(self):
        """Test that square properties match rectangle with equal sides."""
        side = 5
        square = await square_properties(side)
        rect_area = await rectangle_area(side, side)
        rect_perim = await rectangle_perimeter(side, side)

        assert square["area"] == rect_area
        assert square["perimeter"] == rect_perim

    @pytest.mark.asyncio
    async def test_triangle_area_consistency(self):
        """Test that both triangle area methods give same result for right triangle."""
        # For 3-4-5 right triangle
        # Base = 4, height = 3, area = 6
        area_bh = await triangle_area(4, 3)
        area_heron = await triangle_area_heron(3, 4, 5)
        assert pytest.approx(area_bh, rel=1e-10) == area_heron

    @pytest.mark.asyncio
    async def test_circle_circumference_from_area(self):
        """Test relationship: C = 2√(πA)."""
        radius = 5
        area = await circle_area(radius)
        circumference = await circle_circumference(radius)

        # C = 2πr, A = πr², so C = 2√(πA)
        expected_circumference = 2 * math.sqrt(math.pi * area)
        assert pytest.approx(circumference, rel=1e-10) == expected_circumference


# Async Behavior Tests
class TestAsyncBehavior:
    """Test async behavior of geometry functions."""

    @pytest.mark.asyncio
    async def test_all_functions_are_async(self):
        """Test that all geometry functions are properly async."""
        operations = [
            circle_area(5),
            circle_circumference(5),
            rectangle_area(4, 6),
            rectangle_perimeter(4, 6),
            triangle_area(6, 4),
            triangle_area_heron(3, 4, 5),
            distance_2d(0, 0, 3, 4),
            distance_3d(0, 0, 0, 1, 1, 1),
            hypotenuse(3, 4),
            square_properties(5),
            sphere_volume(3),
            sphere_surface_area(3),
        ]

        # Ensure all are coroutines
        for op in operations:
            assert asyncio.iscoroutine(op)

        # Run all operations concurrently
        results = await asyncio.gather(*operations)
        assert len(results) == len(operations)

    @pytest.mark.asyncio
    async def test_concurrent_geometry_calculations(self):
        """Test concurrent execution of geometry functions."""
        import time

        start = time.time()

        # Run multiple calculations concurrently
        tasks = [circle_area(i) for i in range(100)]
        await asyncio.gather(*tasks)

        duration = time.time() - start

        # Should complete quickly
        assert duration < 1.0


# Parametrized Tests
class TestParametrized:
    """Parametrized tests for comprehensive coverage."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "radius,expected_area",
        [
            (1, math.pi),
            (2, 4 * math.pi),
            (5, 25 * math.pi),
            (10, 100 * math.pi),
        ],
    )
    async def test_circle_area_parametrized(self, radius, expected_area):
        """Parametrized test for circle area."""
        result = await circle_area(radius)
        assert pytest.approx(result, rel=1e-10) == expected_area

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "a,b,c",
        [
            (3, 4, 5),
            (5, 12, 13),
            (8, 15, 17),
            (7, 24, 25),
        ],
    )
    async def test_pythagorean_triples(self, a, b, c):
        """Parametrized test for Pythagorean triples."""
        result = await hypotenuse(a, b)
        assert pytest.approx(result, rel=1e-10) == c

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "length,width,expected_area",
        [
            (1, 1, 1),
            (2, 3, 6),
            (5, 10, 50),
            (7, 9, 63),
        ],
    )
    async def test_rectangle_area_parametrized(self, length, width, expected_area):
        """Parametrized test for rectangle area."""
        result = await rectangle_area(length, width)
        assert result == expected_area


# Edge Cases Tests
class TestEdgeCases:
    """Test edge cases for geometry functions."""

    @pytest.mark.asyncio
    async def test_very_large_circles(self):
        """Test circle calculations with very large radius."""
        large_radius = 1e10
        area = await circle_area(large_radius)
        circumference = await circle_circumference(large_radius)

        assert area > 0
        assert circumference > 0
        assert area == math.pi * large_radius**2

    @pytest.mark.asyncio
    async def test_very_small_circles(self):
        """Test circle calculations with very small radius."""
        small_radius = 1e-10
        area = await circle_area(small_radius)
        circumference = await circle_circumference(small_radius)

        assert area > 0
        assert circumference > 0
        assert pytest.approx(area, rel=1e-15) == math.pi * small_radius**2

    @pytest.mark.asyncio
    async def test_degenerate_triangle(self):
        """Test degenerate triangle (sides sum to exactly the third side)."""
        # This should raise an error as it's not a valid triangle
        with pytest.raises(ValueError, match="Invalid triangle"):
            await triangle_area_heron(1, 2, 3)  # 1 + 2 = 3


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v", "--tb=short"])
