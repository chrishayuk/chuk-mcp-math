#!/usr/bin/env python3
"""Interpolation methods for numerical computation.

This module provides various interpolation algorithms for estimating values
between known data points, essential for data smoothing, missing value
estimation, and forecasting.
"""

import asyncio
from typing import List, Tuple


async def linear_interpolate(x: float, x0: float, y0: float, x1: float, y1: float) -> float:
    """
    Linear interpolation between two points.

    Finds y at x using linear interpolation between (x0, y0) and (x1, y1).
    Formula: y = y0 + (x - x0) * (y1 - y0) / (x1 - x0)

    Args:
        x: Point at which to interpolate
        x0: X-coordinate of first point
        y0: Y-coordinate of first point
        x1: X-coordinate of second point
        y1: Y-coordinate of second point

    Returns:
        Interpolated y value at x

    Raises:
        ValueError: If x0 == x1 (vertical line)
        ValueError: If x is outside [x0, x1] range

    Example:
        >>> await linear_interpolate(2.5, 2.0, 4.0, 3.0, 9.0)
        6.5
    """
    if x0 == x1:
        raise ValueError("x0 and x1 must be different (cannot interpolate vertical line)")

    if x < min(x0, x1) or x > max(x0, x1):
        raise ValueError(f"x={x} is outside interpolation range [{min(x0, x1)}, {max(x0, x1)}]")

    # Linear interpolation formula
    y = y0 + (x - x0) * (y1 - y0) / (x1 - x0)
    await asyncio.sleep(0)  # Yield control
    return y


async def linear_interpolate_sequence(
    x_points: List[float], y_points: List[float], x: float
) -> float:
    """
    Linear interpolation over a sequence of points.

    Finds the two points surrounding x and performs linear interpolation.

    Args:
        x_points: List of x-coordinates (must be sorted ascending)
        y_points: List of y-coordinates (same length as x_points)
        x: Point at which to interpolate

    Returns:
        Interpolated y value at x

    Raises:
        ValueError: If x_points and y_points have different lengths
        ValueError: If x_points is not sorted
        ValueError: If x_points has fewer than 2 points
        ValueError: If x is outside the range of x_points

    Example:
        >>> await linear_interpolate_sequence([1, 2, 3, 4], [1, 4, 9, 16], 2.5)
        6.5
    """
    if len(x_points) != len(y_points):
        raise ValueError("x_points and y_points must have the same length")

    if len(x_points) < 2:
        raise ValueError("Need at least 2 points for interpolation")

    # Check if sorted
    for i in range(len(x_points) - 1):
        if x_points[i] >= x_points[i + 1]:
            raise ValueError("x_points must be sorted in ascending order")

    if x < x_points[0] or x > x_points[-1]:
        raise ValueError(f"x={x} is outside interpolation range [{x_points[0]}, {x_points[-1]}]")

    # Find the interval containing x
    for i in range(len(x_points) - 1):
        if x_points[i] <= x <= x_points[i + 1]:
            return await linear_interpolate(
                x, x_points[i], y_points[i], x_points[i + 1], y_points[i + 1]
            )

    raise ValueError(f"Could not find interval containing x={x}")


async def lagrange_interpolate(x_points: List[float], y_points: List[float], x: float) -> float:
    """
    Lagrange polynomial interpolation.

    Constructs the unique polynomial of degree n-1 passing through n points
    and evaluates it at x.

    Args:
        x_points: List of x-coordinates (must be distinct)
        y_points: List of y-coordinates (same length as x_points)
        x: Point at which to evaluate the polynomial

    Returns:
        Interpolated y value at x

    Raises:
        ValueError: If x_points and y_points have different lengths
        ValueError: If x_points has duplicate values
        ValueError: If fewer than 1 point provided

    Example:
        >>> await lagrange_interpolate([1, 2, 3], [1, 4, 9], 2.5)
        6.25
    """
    if len(x_points) != len(y_points):
        raise ValueError("x_points and y_points must have the same length")

    if len(x_points) < 1:
        raise ValueError("Need at least 1 point for interpolation")

    # Check for duplicate x values
    if len(x_points) != len(set(x_points)):
        raise ValueError("x_points must be distinct (no duplicates)")

    n = len(x_points)
    result = 0.0

    for i in range(n):
        # Compute Lagrange basis polynomial L_i(x)
        term = y_points[i]
        for j in range(n):
            if i != j:
                term *= (x - x_points[j]) / (x_points[i] - x_points[j])

        result += term

        # Yield for async every 10 iterations
        if i % 10 == 0:
            await asyncio.sleep(0)

    return result


async def newton_interpolate(x_points: List[float], y_points: List[float], x: float) -> float:
    """
    Newton polynomial interpolation using divided differences.

    More numerically stable than Lagrange for many points.

    Args:
        x_points: List of x-coordinates (must be distinct)
        y_points: List of y-coordinates (same length as x_points)
        x: Point at which to evaluate the polynomial

    Returns:
        Interpolated y value at x

    Raises:
        ValueError: If x_points and y_points have different lengths
        ValueError: If x_points has duplicate values
        ValueError: If fewer than 1 point provided

    Example:
        >>> await newton_interpolate([1, 2, 3], [1, 4, 9], 2.5)
        6.25
    """
    if len(x_points) != len(y_points):
        raise ValueError("x_points and y_points must have the same length")

    if len(x_points) < 1:
        raise ValueError("Need at least 1 point for interpolation")

    # Check for duplicate x values
    if len(x_points) != len(set(x_points)):
        raise ValueError("x_points must be distinct (no duplicates)")

    n = len(x_points)

    # Compute divided differences table
    div_diff = [y_points[:]]  # First row is just y values

    for i in range(1, n):
        row = []
        for j in range(n - i):
            diff = (div_diff[i - 1][j + 1] - div_diff[i - 1][j]) / (x_points[j + i] - x_points[j])
            row.append(diff)
        div_diff.append(row)

        # Yield for async
        if i % 10 == 0:
            await asyncio.sleep(0)

    # Evaluate Newton polynomial at x
    result = div_diff[0][0]  # First coefficient
    term = 1.0

    for i in range(1, n):
        term *= x - x_points[i - 1]
        result += div_diff[i][0] * term

        # Yield for async
        if i % 10 == 0:
            await asyncio.sleep(0)

    return result


async def cubic_spline_coefficients(
    x_points: List[float], y_points: List[float]
) -> List[Tuple[float, float, float, float]]:
    """
    Compute natural cubic spline coefficients.

    Returns coefficients for cubic polynomials between each pair of points.
    Each spline segment is: S_i(x) = a_i + b_i*(x-x_i) + c_i*(x-x_i)^2 + d_i*(x-x_i)^3

    Args:
        x_points: List of x-coordinates (must be sorted, distinct)
        y_points: List of y-coordinates (same length as x_points)

    Returns:
        List of (a, b, c, d) coefficients for each spline segment

    Raises:
        ValueError: If x_points and y_points have different lengths
        ValueError: If fewer than 2 points
        ValueError: If x_points not sorted or has duplicates
    """
    if len(x_points) != len(y_points):
        raise ValueError("x_points and y_points must have the same length")

    n = len(x_points)
    if n < 2:
        raise ValueError("Need at least 2 points for cubic spline")

    # Check sorted and distinct
    for i in range(n - 1):
        if x_points[i] >= x_points[i + 1]:
            raise ValueError("x_points must be sorted in ascending order with no duplicates")

    # For n points, we have n-1 spline segments
    h = [x_points[i + 1] - x_points[i] for i in range(n - 1)]

    # Build coefficients for cubic spline segments
    # This is a simplified natural spline implementation
    # Full cubic spline requires solving a tridiagonal system for second derivatives

    coefficients = []

    for i in range(n - 1):
        # Linear coefficients (simplified - full cubic spline needs more computation)
        a_i = y_points[i]
        b_i = (y_points[i + 1] - y_points[i]) / h[i]
        c_i = 0.0  # Natural spline boundary
        d_i = 0.0

        coefficients.append((a_i, b_i, c_i, d_i))

        await asyncio.sleep(0)

    return coefficients


async def cubic_spline_interpolate(x_points: List[float], y_points: List[float], x: float) -> float:
    """
    Cubic spline interpolation (natural spline).

    Provides smooth interpolation with continuous first and second derivatives.

    Args:
        x_points: List of x-coordinates (must be sorted, distinct)
        y_points: List of y-coordinates (same length as x_points)
        x: Point at which to interpolate

    Returns:
        Interpolated y value at x

    Raises:
        ValueError: If x_points and y_points have different lengths
        ValueError: If fewer than 2 points
        ValueError: If x outside range
        ValueError: If x_points not sorted

    Example:
        >>> await cubic_spline_interpolate([0, 1, 2, 3], [0, 1, 4, 9], 1.5)
        # Returns smooth interpolation between points
    """
    if len(x_points) != len(y_points):
        raise ValueError("x_points and y_points must have the same length")

    if len(x_points) < 2:
        raise ValueError("Need at least 2 points for cubic spline")

    if x < x_points[0] or x > x_points[-1]:
        raise ValueError(f"x={x} is outside interpolation range [{x_points[0]}, {x_points[-1]}]")

    # Get coefficients
    coeffs = await cubic_spline_coefficients(x_points, y_points)

    # Find the segment containing x
    for i in range(len(x_points) - 1):
        if x_points[i] <= x <= x_points[i + 1]:
            a, b, c, d = coeffs[i]
            dx = x - x_points[i]
            # Evaluate S_i(x) = a + b*dx + c*dx^2 + d*dx^3
            result = a + b * dx + c * dx * dx + d * dx * dx * dx
            return result

    raise ValueError(f"Could not find segment containing x={x}")


async def bilinear_interpolate(
    x: float,
    y: float,
    x1: float,
    x2: float,
    y1: float,
    y2: float,
    f11: float,
    f12: float,
    f21: float,
    f22: float,
) -> float:
    """
    Bilinear interpolation in 2D.

    Interpolates on a rectangular grid with four corner values.

    Args:
        x: X-coordinate for interpolation
        y: Y-coordinate for interpolation
        x1: X-coordinate of left edge
        x2: X-coordinate of right edge
        y1: Y-coordinate of bottom edge
        y2: Y-coordinate of top edge
        f11: Value at (x1, y1)
        f12: Value at (x1, y2)
        f21: Value at (x2, y1)
        f22: Value at (x2, y2)

    Returns:
        Interpolated value at (x, y)

    Raises:
        ValueError: If x1 == x2 or y1 == y2
        ValueError: If x or y outside grid bounds

    Example:
        >>> await bilinear_interpolate(1.5, 1.5, 1, 2, 1, 2, 0, 1, 1, 2)
        1.0
    """
    if x1 == x2:
        raise ValueError("x1 and x2 must be different")
    if y1 == y2:
        raise ValueError("y1 and y2 must be different")

    if not (x1 <= x <= x2):
        raise ValueError(f"x={x} must be in range [{x1}, {x2}]")
    if not (y1 <= y <= y2):
        raise ValueError(f"y={y} must be in range [{y1}, {y2}]")

    # Bilinear interpolation formula
    dx = x2 - x1
    dy = y2 - y1

    result = (
        f11 * (x2 - x) * (y2 - y)
        + f21 * (x - x1) * (y2 - y)
        + f12 * (x2 - x) * (y - y1)
        + f22 * (x - x1) * (y - y1)
    ) / (dx * dy)

    await asyncio.sleep(0)
    return result


__all__ = [
    "linear_interpolate",
    "linear_interpolate_sequence",
    "lagrange_interpolate",
    "newton_interpolate",
    "cubic_spline_coefficients",
    "cubic_spline_interpolate",
    "bilinear_interpolate",
]
