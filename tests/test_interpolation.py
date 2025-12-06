#!/usr/bin/env python3
"""Tests for interpolation module."""

import pytest
from chuk_mcp_math.numerical import interpolation


class TestLinearInterpolate:
    """Tests for linear_interpolate function."""

    @pytest.mark.asyncio
    async def test_linear_interpolate_basic(self):
        """Test basic linear interpolation."""
        result = await interpolation.linear_interpolate(2.5, 2.0, 4.0, 3.0, 9.0)
        assert abs(result - 6.5) < 1e-10

    @pytest.mark.asyncio
    async def test_linear_interpolate_at_endpoints(self):
        """Test interpolation at endpoints."""
        result1 = await interpolation.linear_interpolate(2.0, 2.0, 4.0, 3.0, 9.0)
        assert abs(result1 - 4.0) < 1e-10

        result2 = await interpolation.linear_interpolate(3.0, 2.0, 4.0, 3.0, 9.0)
        assert abs(result2 - 9.0) < 1e-10

    @pytest.mark.asyncio
    async def test_linear_interpolate_negative_slope(self):
        """Test interpolation with negative slope."""
        result = await interpolation.linear_interpolate(1.5, 1.0, 10.0, 2.0, 5.0)
        assert abs(result - 7.5) < 1e-10

    @pytest.mark.asyncio
    async def test_linear_interpolate_errors(self):
        """Test error handling."""
        # Vertical line (x0 == x1)
        with pytest.raises(ValueError, match="cannot interpolate vertical line"):
            await interpolation.linear_interpolate(2.5, 2.0, 4.0, 2.0, 9.0)

        # x outside range
        with pytest.raises(ValueError, match="outside interpolation range"):
            await interpolation.linear_interpolate(5.0, 2.0, 4.0, 3.0, 9.0)


class TestLinearInterpolateSequence:
    """Tests for linear_interpolate_sequence function."""

    @pytest.mark.asyncio
    async def test_linear_sequence_basic(self):
        """Test basic sequence interpolation."""
        x_points = [1.0, 2.0, 3.0, 4.0]
        y_points = [1.0, 4.0, 9.0, 16.0]
        result = await interpolation.linear_interpolate_sequence(x_points, y_points, 2.5)
        assert abs(result - 6.5) < 1e-10

    @pytest.mark.asyncio
    async def test_linear_sequence_multiple_intervals(self):
        """Test interpolation across different intervals."""
        x_points = [0.0, 1.0, 2.0, 3.0]
        y_points = [0.0, 1.0, 4.0, 9.0]

        # Test in first interval
        result1 = await interpolation.linear_interpolate_sequence(x_points, y_points, 0.5)
        assert abs(result1 - 0.5) < 1e-10

        # Test in last interval
        result2 = await interpolation.linear_interpolate_sequence(x_points, y_points, 2.5)
        assert abs(result2 - 6.5) < 1e-10

    @pytest.mark.asyncio
    async def test_linear_sequence_errors(self):
        """Test error handling."""
        x_points = [1.0, 2.0, 3.0]
        y_points = [1.0, 4.0, 9.0]

        # Different lengths
        with pytest.raises(ValueError, match="same length"):
            await interpolation.linear_interpolate_sequence(x_points, [1.0, 2.0], 2.5)

        # Too few points
        with pytest.raises(ValueError, match="at least 2 points"):
            await interpolation.linear_interpolate_sequence([1.0], [1.0], 1.5)

        # Not sorted
        with pytest.raises(ValueError, match="sorted in ascending order"):
            await interpolation.linear_interpolate_sequence([1.0, 3.0, 2.0], [1.0, 9.0, 4.0], 2.0)

        # Outside range
        with pytest.raises(ValueError, match="outside interpolation range"):
            await interpolation.linear_interpolate_sequence(x_points, y_points, 5.0)


class TestLagrangeInterpolate:
    """Tests for lagrange_interpolate function."""

    @pytest.mark.asyncio
    async def test_lagrange_polynomial_quadratic(self):
        """Test Lagrange interpolation with quadratic data."""
        # Points from y = x^2
        x_points = [0.0, 1.0, 2.0]
        y_points = [0.0, 1.0, 4.0]

        # Should reconstruct quadratic exactly
        result = await interpolation.lagrange_interpolate(x_points, y_points, 1.5)
        assert abs(result - 2.25) < 1e-10

    @pytest.mark.asyncio
    async def test_lagrange_single_point(self):
        """Test with single point."""
        result = await interpolation.lagrange_interpolate([2.0], [4.0], 5.0)
        assert abs(result - 4.0) < 1e-10  # Constant polynomial

    @pytest.mark.asyncio
    async def test_lagrange_at_data_points(self):
        """Test that interpolation passes through data points."""
        x_points = [1.0, 2.0, 3.0, 4.0]
        y_points = [1.0, 4.0, 9.0, 16.0]

        for i in range(len(x_points)):
            result = await interpolation.lagrange_interpolate(x_points, y_points, x_points[i])
            assert abs(result - y_points[i]) < 1e-10

    @pytest.mark.asyncio
    async def test_lagrange_errors(self):
        """Test error handling."""
        # Different lengths
        with pytest.raises(ValueError, match="same length"):
            await interpolation.lagrange_interpolate([1.0, 2.0], [1.0], 1.5)

        # Duplicate x values
        with pytest.raises(ValueError, match="must be distinct"):
            await interpolation.lagrange_interpolate([1.0, 2.0, 2.0], [1.0, 4.0, 9.0], 1.5)

        # Empty list
        with pytest.raises(ValueError, match="at least 1 point"):
            await interpolation.lagrange_interpolate([], [], 1.0)


class TestNewtonInterpolate:
    """Tests for newton_interpolate function."""

    @pytest.mark.asyncio
    async def test_newton_polynomial_quadratic(self):
        """Test Newton interpolation with quadratic data."""
        # Points from y = x^2
        x_points = [0.0, 1.0, 2.0]
        y_points = [0.0, 1.0, 4.0]

        # Should reconstruct quadratic exactly
        result = await interpolation.newton_interpolate(x_points, y_points, 1.5)
        assert abs(result - 2.25) < 1e-10

    @pytest.mark.asyncio
    async def test_newton_same_as_lagrange(self):
        """Test that Newton and Lagrange give same results."""
        x_points = [1.0, 2.0, 3.0, 4.0]
        y_points = [1.0, 8.0, 27.0, 64.0]  # y = x^3

        x_test = 2.5
        lagrange_result = await interpolation.lagrange_interpolate(x_points, y_points, x_test)
        newton_result = await interpolation.newton_interpolate(x_points, y_points, x_test)

        assert abs(lagrange_result - newton_result) < 1e-10

    @pytest.mark.asyncio
    async def test_newton_at_data_points(self):
        """Test that interpolation passes through data points."""
        x_points = [0.0, 1.0, 2.0, 3.0]
        y_points = [0.0, 1.0, 8.0, 27.0]

        for i in range(len(x_points)):
            result = await interpolation.newton_interpolate(x_points, y_points, x_points[i])
            assert abs(result - y_points[i]) < 1e-10

    @pytest.mark.asyncio
    async def test_newton_errors(self):
        """Test error handling."""
        # Different lengths
        with pytest.raises(ValueError, match="same length"):
            await interpolation.newton_interpolate([1.0, 2.0], [1.0], 1.5)

        # Duplicate x values
        with pytest.raises(ValueError, match="must be distinct"):
            await interpolation.newton_interpolate([1.0, 2.0, 2.0], [1.0, 4.0, 9.0], 1.5)

        # Empty list
        with pytest.raises(ValueError, match="at least 1 point"):
            await interpolation.newton_interpolate([], [], 1.0)


class TestCubicSpline:
    """Tests for cubic spline interpolation."""

    @pytest.mark.asyncio
    async def test_cubic_spline_coefficients(self):
        """Test cubic spline coefficient computation."""
        x_points = [0.0, 1.0, 2.0]
        y_points = [0.0, 1.0, 4.0]

        coeffs = await interpolation.cubic_spline_coefficients(x_points, y_points)
        assert len(coeffs) == 2  # n-1 segments for n points
        assert all(len(c) == 4 for c in coeffs)  # Each segment has 4 coefficients

    @pytest.mark.asyncio
    async def test_cubic_spline_interpolate_basic(self):
        """Test basic cubic spline interpolation."""
        x_points = [0.0, 1.0, 2.0, 3.0]
        y_points = [0.0, 1.0, 4.0, 9.0]

        result = await interpolation.cubic_spline_interpolate(x_points, y_points, 1.5)
        # Should be smooth between points
        assert 1.0 <= result <= 4.0

    @pytest.mark.asyncio
    async def test_cubic_spline_at_data_points(self):
        """Test that spline passes through data points."""
        x_points = [0.0, 1.0, 2.0, 3.0]
        y_points = [0.0, 1.0, 4.0, 9.0]

        for i in range(len(x_points)):
            result = await interpolation.cubic_spline_interpolate(x_points, y_points, x_points[i])
            assert abs(result - y_points[i]) < 1e-6

    @pytest.mark.asyncio
    async def test_cubic_spline_errors(self):
        """Test error handling."""
        x_points = [0.0, 1.0, 2.0]
        y_points = [0.0, 1.0, 4.0]

        # Different lengths
        with pytest.raises(ValueError, match="same length"):
            await interpolation.cubic_spline_interpolate([1.0, 2.0], [1.0], 1.5)

        # Too few points
        with pytest.raises(ValueError, match="at least 2 points"):
            await interpolation.cubic_spline_interpolate([1.0], [1.0], 1.5)

        # Outside range
        with pytest.raises(ValueError, match="outside interpolation range"):
            await interpolation.cubic_spline_interpolate(x_points, y_points, 5.0)

        # Not sorted
        with pytest.raises(ValueError, match="sorted in ascending order"):
            await interpolation.cubic_spline_interpolate([1.0, 3.0, 2.0], [1.0, 9.0, 4.0], 2.0)


class TestBilinearInterpolate:
    """Tests for bilinear_interpolate function."""

    @pytest.mark.asyncio
    async def test_bilinear_center(self):
        """Test interpolation at center of square."""
        result = await interpolation.bilinear_interpolate(
            1.5,
            1.5,  # Center point
            1.0,
            2.0,  # x range
            1.0,
            2.0,  # y range
            0.0,
            1.0,  # f(1,1), f(1,2)
            1.0,
            2.0,  # f(2,1), f(2,2)
        )
        # Average of corners: (0 + 1 + 1 + 2) / 4 = 1.0
        assert abs(result - 1.0) < 1e-10

    @pytest.mark.asyncio
    async def test_bilinear_at_corners(self):
        """Test that interpolation returns exact values at corners."""
        # Bottom-left
        result = await interpolation.bilinear_interpolate(
            1.0, 1.0, 1.0, 2.0, 1.0, 2.0, 5.0, 10.0, 15.0, 20.0
        )
        assert abs(result - 5.0) < 1e-10

        # Bottom-right
        result = await interpolation.bilinear_interpolate(
            2.0, 1.0, 1.0, 2.0, 1.0, 2.0, 5.0, 10.0, 15.0, 20.0
        )
        assert abs(result - 15.0) < 1e-10

        # Top-left
        result = await interpolation.bilinear_interpolate(
            1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 5.0, 10.0, 15.0, 20.0
        )
        assert abs(result - 10.0) < 1e-10

        # Top-right
        result = await interpolation.bilinear_interpolate(
            2.0, 2.0, 1.0, 2.0, 1.0, 2.0, 5.0, 10.0, 15.0, 20.0
        )
        assert abs(result - 20.0) < 1e-10

    @pytest.mark.asyncio
    async def test_bilinear_linear_function(self):
        """Test with linear function f(x,y) = x + y."""
        # Corner values for f(x,y) = x + y
        result = await interpolation.bilinear_interpolate(
            1.5,
            1.5,  # Test point
            1.0,
            2.0,  # x range
            1.0,
            2.0,  # y range
            2.0,
            3.0,  # f(1,1)=2, f(1,2)=3
            3.0,
            4.0,  # f(2,1)=3, f(2,2)=4
        )
        # f(1.5, 1.5) = 1.5 + 1.5 = 3.0
        assert abs(result - 3.0) < 1e-10

    @pytest.mark.asyncio
    async def test_bilinear_errors(self):
        """Test error handling."""
        # x1 == x2
        with pytest.raises(ValueError, match="x1 and x2 must be different"):
            await interpolation.bilinear_interpolate(
                1.5, 1.5, 1.0, 1.0, 1.0, 2.0, 0.0, 1.0, 1.0, 2.0
            )

        # y1 == y2
        with pytest.raises(ValueError, match="y1 and y2 must be different"):
            await interpolation.bilinear_interpolate(
                1.5, 1.5, 1.0, 2.0, 1.0, 1.0, 0.0, 1.0, 1.0, 2.0
            )

        # x outside range
        with pytest.raises(ValueError, match="must be in range"):
            await interpolation.bilinear_interpolate(
                3.0, 1.5, 1.0, 2.0, 1.0, 2.0, 0.0, 1.0, 1.0, 2.0
            )

        # y outside range
        with pytest.raises(ValueError, match="must be in range"):
            await interpolation.bilinear_interpolate(
                1.5, 3.0, 1.0, 2.0, 1.0, 2.0, 0.0, 1.0, 1.0, 2.0
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
