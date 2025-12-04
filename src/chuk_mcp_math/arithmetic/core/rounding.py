#!/usr/bin/env python3
# chuk_mcp_math/arithmetic/core/rounding.py
"""
Core Rounding Operations - Async Native

Mathematical rounding functions for various rounding strategies and precision control.

Functions:
- round_number, floor, ceil, truncate
- mround, ceiling_multiple, floor_multiple
"""

import math
import asyncio
from typing import Union
from chuk_mcp_math.mcp_decorator import mcp_function

Number = Union[int, float]


@mcp_function(
    description="Round a number to a specified number of decimal places.",
    namespace="arithmetic",
    category="core",
    execution_modes=["local", "remote"],
    examples=[
        {
            "input": {"x": 3.14159, "decimals": 2},
            "output": 3.14,
            "description": "Round to 2 decimal places",
        },
        {
            "input": {"x": 2.5},
            "output": 2,
            "description": "Round to nearest integer (banker's rounding)",
        },
        {
            "input": {"x": -3.7, "decimals": 0},
            "output": -4,
            "description": "Round negative number to integer",
        },
    ],
)
async def round_number(x: Number, decimals: int = 0) -> Number:
    """
    Round a number to a specified number of decimal places.

    Args:
        x: Number to round
        decimals: Number of decimal places (default: 0)

    Returns:
        Number rounded to specified decimal places

    Examples:
        await round_number(3.14159, 2) → 3.14
        await round_number(2.5) → 2
        await round_number(-3.7, 0) → -4
    """
    return round(x, decimals)


@mcp_function(
    description="Return the floor of a number (largest integer less than or equal to x).",
    namespace="arithmetic",
    category="core",
    execution_modes=["local", "remote"],
    examples=[
        {"input": {"x": 3.7}, "output": 3, "description": "Floor of positive number"},
        {"input": {"x": -2.3}, "output": -3, "description": "Floor of negative number"},
        {"input": {"x": 5}, "output": 5, "description": "Floor of integer"},
    ],
)
async def floor(x: Number) -> int:
    """
    Return the floor of a number.

    Args:
        x: Number to floor

    Returns:
        Largest integer less than or equal to x

    Examples:
        await floor(3.7) → 3
        await floor(-2.3) → -3
        await floor(5) → 5
    """
    return math.floor(x)


@mcp_function(
    description="Return the ceiling of a number (smallest integer greater than or equal to x).",
    namespace="arithmetic",
    category="core",
    execution_modes=["local", "remote"],
    examples=[
        {"input": {"x": 3.2}, "output": 4, "description": "Ceiling of positive number"},
        {
            "input": {"x": -2.7},
            "output": -2,
            "description": "Ceiling of negative number",
        },
        {"input": {"x": 5}, "output": 5, "description": "Ceiling of integer"},
    ],
)
async def ceil(x: Number) -> int:
    """
    Return the ceiling of a number.

    Args:
        x: Number to ceiling

    Returns:
        Smallest integer greater than or equal to x

    Examples:
        await ceil(3.2) → 4
        await ceil(-2.7) → -2
        await ceil(5) → 5
    """
    return math.ceil(x)


@mcp_function(
    description="Truncate a number towards zero (remove decimal part).",
    namespace="arithmetic",
    category="core",
    execution_modes=["local", "remote"],
    examples=[
        {"input": {"x": 3.9}, "output": 3, "description": "Truncate positive number"},
        {"input": {"x": -2.9}, "output": -2, "description": "Truncate negative number"},
        {"input": {"x": 5}, "output": 5, "description": "Truncate integer"},
    ],
)
async def truncate(x: Number) -> int:
    """
    Truncate a number towards zero.

    Args:
        x: Number to truncate

    Returns:
        Integer part of x (towards zero)

    Examples:
        await truncate(3.9) → 3
        await truncate(-2.9) → -2
        await truncate(5) → 5
    """
    return math.trunc(x)


@mcp_function(
    description="Round a number up to the nearest multiple of significance. Always rounds away from zero.",
    namespace="arithmetic",
    category="core",
    execution_modes=["local", "remote"],
    examples=[
        {
            "input": {"number": 2.5, "significance": 1},
            "output": 3,
            "description": "Round 2.5 up to nearest 1",
        },
        {
            "input": {"number": 6.7, "significance": 2},
            "output": 8,
            "description": "Round 6.7 up to nearest 2",
        },
        {
            "input": {"number": -2.1, "significance": 1},
            "output": -3,
            "description": "Round -2.1 away from zero",
        },
        {
            "input": {"number": 15, "significance": 10},
            "output": 20,
            "description": "Round 15 up to nearest 10",
        },
    ],
)
async def ceiling_multiple(number: Number, significance: Number) -> Number:
    """
    Round a number up to the nearest multiple of significance.

    Args:
        number: Number to round
        significance: Multiple to round to (must be positive and finite)

    Returns:
        Number rounded up to nearest multiple of significance

    Raises:
        ValueError: If significance is zero, negative, or infinite

    Examples:
        await ceiling_multiple(2.5, 1) → 3
        await ceiling_multiple(6.7, 2) → 8
        await ceiling_multiple(-2.1, 1) → -3
    """
    if significance <= 0:
        raise ValueError("Significance must be positive")

    if not math.isfinite(significance):
        raise ValueError("Significance must be finite")

    if number >= 0:
        return math.ceil(number / significance) * significance
    else:
        return math.floor(number / significance) * significance


@mcp_function(
    description="Round a number down to the nearest multiple of significance. Always rounds toward zero.",
    namespace="arithmetic",
    category="core",
    execution_modes=["local", "remote"],
    examples=[
        {
            "input": {"number": 2.9, "significance": 1},
            "output": 2,
            "description": "Round 2.9 down to nearest 1",
        },
        {
            "input": {"number": 7.8, "significance": 2},
            "output": 6,
            "description": "Round 7.8 down to nearest 2",
        },
        {
            "input": {"number": -2.9, "significance": 1},
            "output": -2,
            "description": "Round -2.9 toward zero",
        },
        {
            "input": {"number": 23, "significance": 10},
            "output": 20,
            "description": "Round 23 down to nearest 10",
        },
    ],
)
async def floor_multiple(number: Number, significance: Number) -> Number:
    """
    Round a number down to the nearest multiple of significance.

    Args:
        number: Number to round
        significance: Multiple to round to (must be positive and finite)

    Returns:
        Number rounded down to nearest multiple of significance

    Raises:
        ValueError: If significance is zero, negative, or infinite

    Examples:
        await floor_multiple(2.9, 1) → 2
        await floor_multiple(7.8, 2) → 6
        await floor_multiple(-2.9, 1) → -2
    """
    if significance <= 0:
        raise ValueError("Significance must be positive")

    if not math.isfinite(significance):
        raise ValueError("Significance must be finite")

    if number >= 0:
        return math.floor(number / significance) * significance
    else:
        return math.ceil(number / significance) * significance


@mcp_function(
    description="Round a number to the nearest multiple of significance. Uses standard rounding rules.",
    namespace="arithmetic",
    category="core",
    execution_modes=["local", "remote"],
    examples=[
        {
            "input": {"number": 2.4, "significance": 1},
            "output": 2,
            "description": "Round 2.4 to nearest 1",
        },
        {
            "input": {"number": 2.6, "significance": 1},
            "output": 3,
            "description": "Round 2.6 to nearest 1",
        },
        {
            "input": {"number": 7.3, "significance": 2},
            "output": 8,
            "description": "Round 7.3 to nearest 2",
        },
        {
            "input": {"number": 15, "significance": 10},
            "output": 20,
            "description": "Round 15 to nearest 10",
        },
    ],
)
async def mround(number: Number, significance: Number) -> Number:
    """
    Round a number to the nearest multiple of significance.

    Args:
        number: Number to round
        significance: Multiple to round to (must be positive and finite)

    Returns:
        Number rounded to nearest multiple of significance

    Raises:
        ValueError: If significance is zero, negative, or infinite

    Examples:
        await mround(2.4, 1) → 2
        await mround(2.6, 1) → 3
        await mround(7.3, 2) → 8
    """
    if significance <= 0:
        raise ValueError("Significance must be positive")

    if not math.isfinite(significance):
        raise ValueError("Significance must be finite")

    return round(number / significance) * significance


# Export all rounding functions
__all__ = [
    "round_number",
    "floor",
    "ceil",
    "truncate",
    "ceiling_multiple",
    "floor_multiple",
    "mround",
]

if __name__ == "__main__":
    import asyncio

    async def test_rounding_operations():
        """Test all rounding operations."""
        print("🔢 Rounding Operations Test")
        print("=" * 30)

        # Test basic rounding
        print(f"round_number(3.14159, 2) = {await round_number(3.14159, 2)}")
        print(f"floor(3.7) = {await floor(3.7)}")
        print(f"ceil(3.2) = {await ceil(3.2)}")
        print(f"truncate(3.9) = {await truncate(3.9)}")

        # Test multiple rounding
        print(f"ceiling_multiple(6.7, 2) = {await ceiling_multiple(6.7, 2)}")
        print(f"floor_multiple(7.8, 2) = {await floor_multiple(7.8, 2)}")
        print(f"mround(7.3, 2) = {await mround(7.3, 2)}")

        print("\n✅ All rounding operations working!")

    asyncio.run(test_rounding_operations())
