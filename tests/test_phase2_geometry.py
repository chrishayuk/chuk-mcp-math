"""
Test Phase 2 Geometry Module

Tests for all geometry functions:
- Distances (Euclidean, great circle, Manhattan)
- Intersections (lines, segments, circles)
- Shapes (polygon area, circle properties)
"""

import pytest
import math


@pytest.mark.asyncio
class TestGeometryDistances:
    """Test distance calculations."""

    async def test_geom_distance(self):
        from chuk_mcp_math.geometry.distances import geom_distance

        result = await geom_distance((0, 0), (3, 4))
        assert abs(result - 5.0) < 1e-10  # 3-4-5 triangle

    async def test_geom_distance_3d(self):
        from chuk_mcp_math.geometry.distances import geom_distance_3d

        result = await geom_distance_3d((0, 0, 0), (1, 1, 1))
        assert abs(result - math.sqrt(3)) < 1e-10

    async def test_geom_great_circle_distance(self):
        from chuk_mcp_math.geometry.distances import geom_great_circle_distance

        # NYC to London
        result = await geom_great_circle_distance(40.7128, -74.0060, 51.5074, -0.1278)
        # Should be approximately 5570 km
        assert 5500 < result["distance_km"] < 5600
        assert result["distance_miles"] > 0
        assert result["distance_nm"] > 0

    async def test_geom_manhattan_distance(self):
        from chuk_mcp_math.geometry.distances import geom_manhattan_distance

        result = await geom_manhattan_distance((0, 0), (3, 4))
        assert result == 7.0  # 3 + 4


@pytest.mark.asyncio
class TestGeometryIntersections:
    """Test intersection calculations."""

    async def test_line_intersection(self):
        from chuk_mcp_math.geometry.intersections import geom_line_intersection

        # Two lines: y=x and y=-x+1
        result = await geom_line_intersection(
            {"p1": (0, 0), "p2": (1, 1)}, {"p1": (0, 1), "p2": (1, 0)}
        )
        assert result["intersects"] is True
        assert abs(result["x"] - 0.5) < 1e-10
        assert abs(result["y"] - 0.5) < 1e-10

    async def test_line_intersection_parallel(self):
        from chuk_mcp_math.geometry.intersections import geom_line_intersection

        # Two parallel lines: y=x and y=x+1
        result = await geom_line_intersection(
            {"p1": (0, 0), "p2": (1, 1)}, {"p1": (0, 1), "p2": (1, 2)}
        )
        assert result["intersects"] is False
        assert result["parallel"] is True

    async def test_line_intersection_coincident(self):
        from chuk_mcp_math.geometry.intersections import geom_line_intersection

        # Two coincident lines (same line)
        result = await geom_line_intersection(
            {"p1": (0, 0), "p2": (1, 1)}, {"p1": (2, 2), "p2": (3, 3)}
        )
        assert result["intersects"] is False
        assert result["parallel"] is True
        assert result["coincident"] is True

    async def test_segment_intersection(self):
        from chuk_mcp_math.geometry.intersections import geom_segment_intersection

        # Two segments that intersect
        result = await geom_segment_intersection(
            {"p1": (0, 0), "p2": (2, 2)}, {"p1": (0, 2), "p2": (2, 0)}
        )
        assert result["intersects"] is True
        assert abs(result["x"] - 1.0) < 1e-10
        assert abs(result["y"] - 1.0) < 1e-10

    async def test_segment_no_intersection(self):
        from chuk_mcp_math.geometry.intersections import geom_segment_intersection

        # Two segments that don't intersect
        result = await geom_segment_intersection(
            {"p1": (0, 0), "p2": (1, 1)}, {"p1": (2, 2), "p2": (3, 3)}
        )
        assert result["intersects"] is False

    async def test_circle_intersection(self):
        from chuk_mcp_math.geometry.intersections import geom_circle_intersection

        # Two circles intersecting
        result = await geom_circle_intersection({"x": 0, "y": 0, "r": 5}, {"x": 5, "y": 0, "r": 5})
        assert result["intersects"] is True
        assert result["count"] == 2

    async def test_circle_intersection_coincident(self):
        from chuk_mcp_math.geometry.intersections import geom_circle_intersection

        # Two coincident circles (same circle)
        result = await geom_circle_intersection({"x": 0, "y": 0, "r": 5}, {"x": 0, "y": 0, "r": 5})
        assert result["intersects"] is True
        assert result["count"] == float("inf")
        assert result["type"] == "coincident"

    async def test_circle_intersection_contained(self):
        from chuk_mcp_math.geometry.intersections import geom_circle_intersection

        # One circle contained in another
        result = await geom_circle_intersection({"x": 0, "y": 0, "r": 10}, {"x": 0, "y": 0, "r": 5})
        assert result["intersects"] is False
        assert result["count"] == 0
        assert result["type"] == "contained"

    async def test_circle_intersection_touching_internal(self):
        from chuk_mcp_math.geometry.intersections import geom_circle_intersection

        # Two circles touching internally (one point)
        result = await geom_circle_intersection({"x": 0, "y": 0, "r": 10}, {"x": 5, "y": 0, "r": 5})
        assert result["intersects"] is True
        assert result["count"] == 1
        assert result["type"] == "touching"

    async def test_circle_touching(self):
        from chuk_mcp_math.geometry.intersections import geom_circle_intersection

        # Two circles touching
        result = await geom_circle_intersection({"x": 0, "y": 0, "r": 5}, {"x": 10, "y": 0, "r": 5})
        assert result["intersects"] is True
        assert result["count"] == 1
        assert result["type"] == "touching"

    async def test_circle_separate(self):
        from chuk_mcp_math.geometry.intersections import geom_circle_intersection

        # Two circles separate
        result = await geom_circle_intersection({"x": 0, "y": 0, "r": 1}, {"x": 10, "y": 0, "r": 1})
        assert result["intersects"] is False
        assert result["type"] == "separate"

    async def test_point_in_polygon_inside(self):
        from chuk_mcp_math.geometry.intersections import geom_point_in_polygon

        square = [(0, 0), (4, 0), (4, 4), (0, 4)]
        result = await geom_point_in_polygon((2, 2), square)
        assert result is True

    async def test_point_in_polygon_outside(self):
        from chuk_mcp_math.geometry.intersections import geom_point_in_polygon

        square = [(0, 0), (4, 0), (4, 4), (0, 4)]
        result = await geom_point_in_polygon((5, 5), square)
        assert result is False


@pytest.mark.asyncio
class TestGeometryShapes:
    """Test shape calculations."""

    async def test_polygon_area_square(self):
        from chuk_mcp_math.geometry.shapes import geom_polygon_area

        square = [(0, 0), (4, 0), (4, 4), (0, 4)]
        result = await geom_polygon_area(square)
        assert abs(result - 16.0) < 1e-10

    async def test_polygon_area_triangle(self):
        from chuk_mcp_math.geometry.shapes import geom_polygon_area

        triangle = [(0, 0), (4, 0), (2, 3)]
        result = await geom_polygon_area(triangle)
        assert abs(result - 6.0) < 1e-10

    async def test_circle_area(self):
        from chuk_mcp_math.geometry.shapes import geom_circle_area

        result = await geom_circle_area(5.0)
        assert abs(result - math.pi * 25) < 1e-10

    async def test_circle_circumference(self):
        from chuk_mcp_math.geometry.shapes import geom_circle_circumference

        result = await geom_circle_circumference(5.0)
        assert abs(result - 2 * math.pi * 5) < 1e-10

    async def test_triangle_area(self):
        from chuk_mcp_math.geometry.shapes import geom_triangle_area

        result = await geom_triangle_area((0, 0), (4, 0), (2, 3))
        assert abs(result - 6.0) < 1e-10


@pytest.mark.asyncio
class TestStatisticsAdvanced:
    """Test advanced statistics functions."""

    async def test_moving_average(self):
        from chuk_mcp_math.statistics import moving_average

        result = await moving_average([1, 2, 3, 4, 5], 3)
        assert result == [2.0, 3.0, 4.0]

    async def test_z_scores(self):
        from chuk_mcp_math.statistics import z_scores

        result = await z_scores([10, 20, 30, 40, 50])
        # Mean should be 30, std should distribute scores
        assert abs(sum(result)) < 1e-10  # Sum of z-scores should be ~0

    async def test_detect_outliers_zscore(self):
        from chuk_mcp_math.statistics import detect_outliers

        result = await detect_outliers([1, 2, 3, 4, 5, 100], "zscore", 2.0)
        assert result["num_outliers"] == 1
        assert 100 in result["outliers"]

    async def test_detect_outliers_iqr(self):
        from chuk_mcp_math.statistics import detect_outliers

        result = await detect_outliers([1, 2, 3, 4, 5, 100], "iqr", 1.5)
        assert result["num_outliers"] >= 1
        assert 100 in result["outliers"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
