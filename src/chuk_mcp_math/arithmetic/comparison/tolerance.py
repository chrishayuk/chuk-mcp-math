#!/usr/bin/env python3
# chuk_mcp_math/arithmetic/comparison/tolerance.py
"""
Tolerance-based Comparison Operations - Async Native

Comparison functions that handle floating-point precision issues and tolerance checking.

Functions:
- approximately_equal, close_to_zero
- is_finite, is_nan, is_infinite, is_normal
- is_close
"""

import math
import sys  # Added missing import
import asyncio
from typing import Union
from chuk_mcp_math.mcp_decorator import mcp_function

Number = Union[int, float]


@mcp_function(
    description="Check if two floating-point numbers are approximately equal within a tolerance. Handles floating-point precision issues.",
    namespace="arithmetic",
    category="comparison",
    execution_modes=["local", "remote"],
    examples=[
        {
            "input": {"a": 0.1, "b": 0.1, "tolerance": 1e-9},
            "output": True,
            "description": "Exactly equal floats",
        },
        {
            "input": {"a": 0.1, "b": 0.10000001, "tolerance": 1e-7},
            "output": True,
            "description": "Nearly equal within tolerance",
        },
        {
            "input": {"a": 0.1, "b": 0.2, "tolerance": 1e-9},
            "output": False,
            "description": "Different numbers",
        },
        {
            "input": {"a": 1.0, "b": 1.0000001, "tolerance": 1e-6},
            "output": True,
            "description": "Close enough within tolerance",
        },
    ],
)
async def approximately_equal(a: Number, b: Number, tolerance: float = 1e-9) -> bool:
    """
    Check if two numbers are approximately equal within a tolerance.

    Args:
        a: First number
        b: Second number
        tolerance: Maximum allowed difference (default: 1e-9)

    Returns:
        True if |a - b| <= tolerance, False otherwise

    Examples:
        await approximately_equal(0.1, 0.1) → True
        await approximately_equal(0.1, 0.10000001, 1e-7) → True
        await approximately_equal(0.1, 0.2) → False
    """
    return abs(a - b) <= tolerance


@mcp_function(
    description="Check if a number is close to zero within a tolerance. Useful for floating-point comparisons.",
    namespace="arithmetic",
    category="comparison",
    execution_modes=["local", "remote"],
    examples=[
        {"input": {"x": 0}, "output": True, "description": "Exactly zero"},
        {
            "input": {"x": 1e-10, "tolerance": 1e-9},
            "output": True,
            "description": "Very small number within tolerance",
        },
        {
            "input": {"x": 0.001, "tolerance": 1e-9},
            "output": False,
            "description": "Number larger than tolerance",
        },
        {
            "input": {"x": -1e-10, "tolerance": 1e-9},
            "output": True,
            "description": "Small negative number within tolerance",
        },
    ],
)
async def close_to_zero(x: Number, tolerance: float = 1e-9) -> bool:
    """
    Check if a number is close to zero within a tolerance.

    Args:
        x: The number to check
        tolerance: Maximum allowed distance from zero (default: 1e-9)

    Returns:
        True if |x| <= tolerance, False otherwise

    Examples:
        await close_to_zero(0) → True
        await close_to_zero(1e-10, 1e-9) → True
        await close_to_zero(0.001) → False
    """
    return abs(x) <= tolerance


@mcp_function(
    description="Check if a number is finite (not infinite and not NaN).",
    namespace="arithmetic",
    category="comparison",
    execution_modes=["local", "remote"],
    examples=[
        {"input": {"x": 42.5}, "output": True, "description": "Regular finite number"},
        {"input": {"x": 0}, "output": True, "description": "Zero is finite"},
        {
            "input": {"x": float("inf")},
            "output": False,
            "description": "Positive infinity is not finite",
        },
        {
            "input": {"x": float("-inf")},
            "output": False,
            "description": "Negative infinity is not finite",
        },
    ],
)
async def is_finite(x: Number) -> bool:
    """
    Check if a number is finite (not infinite and not NaN).

    Args:
        x: The number to check

    Returns:
        True if x is finite, False otherwise

    Examples:
        await is_finite(42.5) → True
        await is_finite(0) → True
        await is_finite(float('inf')) → False
        await is_finite(float('nan')) → False
    """
    return math.isfinite(float(x))


@mcp_function(
    description="Check if a number is NaN (Not a Number).",
    namespace="arithmetic",
    category="comparison",
    execution_modes=["local", "remote"],
    examples=[
        {"input": {"x": float("nan")}, "output": True, "description": "NaN value"},
        {
            "input": {"x": 42.5},
            "output": False,
            "description": "Regular number is not NaN",
        },
        {
            "input": {"x": float("inf")},
            "output": False,
            "description": "Infinity is not NaN",
        },
        {"input": {"x": 0}, "output": False, "description": "Zero is not NaN"},
    ],
)
async def is_nan(x: Number) -> bool:
    """
    Check if a number is NaN (Not a Number).

    Args:
        x: The number to check

    Returns:
        True if x is NaN, False otherwise

    Examples:
        await is_nan(float('nan')) → True
        await is_nan(42.5) → False
        await is_nan(float('inf')) → False
        await is_nan(0) → False
    """
    return math.isnan(float(x))


@mcp_function(
    description="Check if a number is infinite (positive or negative infinity).",
    namespace="arithmetic",
    category="comparison",
    execution_modes=["local", "remote"],
    examples=[
        {
            "input": {"x": float("inf")},
            "output": True,
            "description": "Positive infinity",
        },
        {
            "input": {"x": float("-inf")},
            "output": True,
            "description": "Negative infinity",
        },
        {
            "input": {"x": 42.5},
            "output": False,
            "description": "Regular number is not infinite",
        },
        {
            "input": {"x": float("nan")},
            "output": False,
            "description": "NaN is not infinite",
        },
    ],
)
async def is_infinite(x: Number) -> bool:
    """
    Check if a number is infinite (positive or negative infinity).

    Args:
        x: The number to check

    Returns:
        True if x is infinite, False otherwise

    Examples:
        await is_infinite(float('inf')) → True
        await is_infinite(float('-inf')) → True
        await is_infinite(42.5) → False
        await is_infinite(float('nan')) → False
    """
    return math.isinf(float(x))


@mcp_function(
    description="Check if a number is normal (finite, non-zero, and not subnormal).",
    namespace="arithmetic",
    category="comparison",
    execution_modes=["local", "remote"],
    examples=[
        {"input": {"x": 42.5}, "output": True, "description": "Regular normal number"},
        {
            "input": {"x": 1e-308},
            "output": True,
            "description": "Very small but normal number",
        },
        {"input": {"x": 0}, "output": False, "description": "Zero is not normal"},
        {
            "input": {"x": float("inf")},
            "output": False,
            "description": "Infinity is not normal",
        },
    ],
)
async def is_normal(x: Number) -> bool:
    """
    Check if a number is normal (finite, non-zero, and not subnormal).

    Args:
        x: The number to check

    Returns:
        True if x is a normal number, False otherwise

    Examples:
        await is_normal(42.5) → True
        await is_normal(1e-308) → True
        await is_normal(0) → False
        await is_normal(float('inf')) → False
    """
    # Check if it's a float first
    try:
        f = float(x)
        # Use math.isfinite and check for zero and subnormal numbers
        return f != 0 and math.isfinite(f) and not (abs(f) < sys.float_info.min)
    except (ValueError, OverflowError):
        return False


@mcp_function(
    description="Relative tolerance comparison using both absolute and relative tolerances.",
    namespace="arithmetic",
    category="comparison",
    execution_modes=["local", "remote"],
    examples=[
        {
            "input": {"a": 1000000.1, "b": 1000000.2, "rel_tol": 1e-6, "abs_tol": 1e-9},
            "output": True,
            "description": "Close large numbers",
        },
        {
            "input": {"a": 0.0000001, "b": 0.0000002, "rel_tol": 1e-6, "abs_tol": 1e-9},
            "output": False,
            "description": "Small numbers need absolute tolerance",
        },
        {
            "input": {"a": 0, "b": 1e-10, "rel_tol": 1e-6, "abs_tol": 1e-9},
            "output": True,
            "description": "Near zero uses absolute tolerance",
        },
        {
            "input": {"a": 1.0, "b": 1.000001, "rel_tol": 1e-5},
            "output": True,
            "description": "Relative tolerance check",
        },
    ],
)
async def is_close(
    a: Number, b: Number, rel_tol: float = 1e-9, abs_tol: float = 0.0
) -> bool:
    """
    Check if two numbers are close using both relative and absolute tolerances.

    This is similar to Python's math.isclose() function.

    Args:
        a: First number
        b: Second number
        rel_tol: Relative tolerance (default: 1e-9)
        abs_tol: Absolute tolerance (default: 0.0)

    Returns:
        True if numbers are close within tolerances, False otherwise

    Examples:
        await is_close(1000000.1, 1000000.2, rel_tol=1e-6) → True
        await is_close(0, 1e-10, abs_tol=1e-9) → True
        await is_close(1.0, 1.000001, rel_tol=1e-5) → True
    """
    # Handle special cases
    if a == b:
        return True

    # Handle infinite values
    if math.isinf(float(a)) or math.isinf(float(b)):
        return False

    # Handle NaN values
    if math.isnan(float(a)) or math.isnan(float(b)):
        return False

    # Calculate the difference
    diff = abs(a - b)

    # Check absolute tolerance
    if diff <= abs_tol:
        return True

    # Check relative tolerance
    if diff <= rel_tol * max(abs(a), abs(b)):
        return True

    return False


# Export all tolerance-based comparison functions
__all__ = [
    "approximately_equal",
    "close_to_zero",
    "is_finite",
    "is_nan",
    "is_infinite",
    "is_normal",
    "is_close",
]

if __name__ == "__main__":
    import asyncio

    async def test_tolerance_operations():
        """Test all tolerance-based comparison operations."""
        print("🔍 Tolerance-based Comparison Operations Test")
        print("=" * 45)

        # Test approximate equality
        print(f"approximately_equal(0.1, 0.1) = {await approximately_equal(0.1, 0.1)}")
        print(
            f"approximately_equal(0.1, 0.10000001, 1e-7) = {await approximately_equal(0.1, 0.10000001, 1e-7)}"
        )
        print(f"approximately_equal(0.1, 0.2) = {await approximately_equal(0.1, 0.2)}")

        # Test close to zero
        print(f"close_to_zero(0) = {await close_to_zero(0)}")
        print(f"close_to_zero(1e-10, 1e-9) = {await close_to_zero(1e-10, 1e-9)}")
        print(f"close_to_zero(0.001) = {await close_to_zero(0.001)}")

        # Test special values
        print(f"is_finite(42.5) = {await is_finite(42.5)}")
        print(f"is_finite(float('inf')) = {await is_finite(float('inf'))}")
        print(f"is_nan(float('nan')) = {await is_nan(float('nan'))}")
        print(f"is_nan(42.5) = {await is_nan(42.5)}")
        print(f"is_infinite(float('inf')) = {await is_infinite(float('inf'))}")
        print(f"is_infinite(42.5) = {await is_infinite(42.5)}")

        # Test normal numbers
        print(f"is_normal(42.5) = {await is_normal(42.5)}")
        print(f"is_normal(0) = {await is_normal(0)}")

        # Test is_close
        print(
            f"is_close(1000000.1, 1000000.2, rel_tol=1e-6) = {await is_close(1000000.1, 1000000.2, rel_tol=1e-6)}"
        )
        print(
            f"is_close(0, 1e-10, abs_tol=1e-9) = {await is_close(0, 1e-10, abs_tol=1e-9)}"
        )

        print("\n✅ All tolerance-based comparison operations working!")

    asyncio.run(test_tolerance_operations())
