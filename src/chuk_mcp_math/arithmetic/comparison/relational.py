#!/usr/bin/env python3
# chuk_mcp_math/arithmetic/comparison/relational.py
"""
Relational Comparison Operations - Async Native

Basic comparison operations for determining relationships between numbers.

Functions:
- equal, not_equal, less_than, less_than_or_equal
- greater_than, greater_than_or_equal
- in_range, between
"""

import asyncio
from typing import Union
from chuk_mcp_math.mcp_decorator import mcp_function

Number = Union[int, float]


@mcp_function(
    description="Check if two numbers are exactly equal. For floating-point numbers, consider using approximately_equal instead.",
    namespace="arithmetic",
    category="comparison",
    execution_modes=["local", "remote"],
    examples=[
        {"input": {"a": 5, "b": 5}, "output": True, "description": "Equal integers"},
        {
            "input": {"a": 3.14, "b": 3.14},
            "output": True,
            "description": "Equal floats",
        },
        {
            "input": {"a": 5, "b": 5.0},
            "output": True,
            "description": "Integer and float with same value",
        },
        {"input": {"a": 1, "b": 2}, "output": False, "description": "Unequal numbers"},
    ],
)
async def equal(a: Number, b: Number) -> bool:
    """
    Check if two numbers are exactly equal.

    Args:
        a: First number
        b: Second number

    Returns:
        True if a equals b, False otherwise

    Examples:
        await equal(5, 5) → True
        await equal(3.14, 3.14) → True
        await equal(1, 2) → False
    """
    return a == b


@mcp_function(
    description="Check if two numbers are not equal.",
    namespace="arithmetic",
    category="comparison",
    execution_modes=["local", "remote"],
    examples=[
        {"input": {"a": 5, "b": 3}, "output": True, "description": "Different numbers"},
        {"input": {"a": 7, "b": 7}, "output": False, "description": "Same numbers"},
        {
            "input": {"a": 3.14, "b": 3.15},
            "output": True,
            "description": "Slightly different floats",
        },
    ],
)
async def not_equal(a: Number, b: Number) -> bool:
    """
    Check if two numbers are not equal.

    Args:
        a: First number
        b: Second number

    Returns:
        True if a does not equal b, False otherwise

    Examples:
        await not_equal(5, 3) → True
        await not_equal(7, 7) → False
    """
    return a != b


@mcp_function(
    description="Check if the first number is less than the second number.",
    namespace="arithmetic",
    category="comparison",
    execution_modes=["local", "remote"],
    examples=[
        {"input": {"a": 3, "b": 5}, "output": True, "description": "3 is less than 5"},
        {
            "input": {"a": 5, "b": 3},
            "output": False,
            "description": "5 is not less than 3",
        },
        {
            "input": {"a": 5, "b": 5},
            "output": False,
            "description": "5 is not less than 5",
        },
        {
            "input": {"a": -2, "b": 1},
            "output": True,
            "description": "Negative number less than positive",
        },
    ],
)
async def less_than(a: Number, b: Number) -> bool:
    """
    Check if the first number is less than the second number.

    Args:
        a: First number
        b: Second number

    Returns:
        True if a < b, False otherwise

    Examples:
        await less_than(3, 5) → True
        await less_than(5, 3) → False
        await less_than(5, 5) → False
    """
    return a < b


@mcp_function(
    description="Check if the first number is less than or equal to the second number.",
    namespace="arithmetic",
    category="comparison",
    execution_modes=["local", "remote"],
    examples=[
        {"input": {"a": 3, "b": 5}, "output": True, "description": "3 is less than 5"},
        {"input": {"a": 5, "b": 5}, "output": True, "description": "5 is equal to 5"},
        {
            "input": {"a": 7, "b": 5},
            "output": False,
            "description": "7 is greater than 5",
        },
    ],
)
async def less_than_or_equal(a: Number, b: Number) -> bool:
    """
    Check if the first number is less than or equal to the second number.

    Args:
        a: First number
        b: Second number

    Returns:
        True if a <= b, False otherwise

    Examples:
        await less_than_or_equal(3, 5) → True
        await less_than_or_equal(5, 5) → True
        await less_than_or_equal(7, 5) → False
    """
    return a <= b


@mcp_function(
    description="Check if the first number is greater than the second number.",
    namespace="arithmetic",
    category="comparison",
    execution_modes=["local", "remote"],
    examples=[
        {
            "input": {"a": 5, "b": 3},
            "output": True,
            "description": "5 is greater than 3",
        },
        {
            "input": {"a": 3, "b": 5},
            "output": False,
            "description": "3 is not greater than 5",
        },
        {
            "input": {"a": 5, "b": 5},
            "output": False,
            "description": "5 is not greater than 5",
        },
    ],
)
async def greater_than(a: Number, b: Number) -> bool:
    """
    Check if the first number is greater than the second number.

    Args:
        a: First number
        b: Second number

    Returns:
        True if a > b, False otherwise

    Examples:
        await greater_than(5, 3) → True
        await greater_than(3, 5) → False
        await greater_than(5, 5) → False
    """
    return a > b


@mcp_function(
    description="Check if the first number is greater than or equal to the second number.",
    namespace="arithmetic",
    category="comparison",
    execution_modes=["local", "remote"],
    examples=[
        {
            "input": {"a": 5, "b": 3},
            "output": True,
            "description": "5 is greater than 3",
        },
        {"input": {"a": 5, "b": 5}, "output": True, "description": "5 is equal to 5"},
        {"input": {"a": 3, "b": 5}, "output": False, "description": "3 is less than 5"},
    ],
)
async def greater_than_or_equal(a: Number, b: Number) -> bool:
    """
    Check if the first number is greater than or equal to the second number.

    Args:
        a: First number
        b: Second number

    Returns:
        True if a >= b, False otherwise

    Examples:
        await greater_than_or_equal(5, 3) → True
        await greater_than_or_equal(5, 5) → True
        await greater_than_or_equal(3, 5) → False
    """
    return a >= b


@mcp_function(
    description="Check if a number is within a specified range (inclusive by default).",
    namespace="arithmetic",
    category="comparison",
    execution_modes=["local", "remote"],
    examples=[
        {
            "input": {"value": 5, "min_val": 1, "max_val": 10, "inclusive": True},
            "output": True,
            "description": "Value within inclusive range",
        },
        {
            "input": {"value": 1, "min_val": 1, "max_val": 10, "inclusive": True},
            "output": True,
            "description": "Value at lower bound (inclusive)",
        },
        {
            "input": {"value": 1, "min_val": 1, "max_val": 10, "inclusive": False},
            "output": False,
            "description": "Value at lower bound (exclusive)",
        },
        {
            "input": {"value": 15, "min_val": 1, "max_val": 10, "inclusive": True},
            "output": False,
            "description": "Value outside range",
        },
    ],
)
async def in_range(
    value: Number, min_val: Number, max_val: Number, inclusive: bool = True
) -> bool:
    """
    Check if a value is within a specified range.

    Args:
        value: The value to check
        min_val: The minimum value of the range
        max_val: The maximum value of the range
        inclusive: If True, include endpoints; if False, exclude them

    Returns:
        True if value is in range, False otherwise

    Raises:
        ValueError: If min_val > max_val

    Examples:
        await in_range(5, 1, 10) → True
        await in_range(1, 1, 10, inclusive=True) → True
        await in_range(1, 1, 10, inclusive=False) → False
    """
    if min_val > max_val:
        raise ValueError("Minimum value cannot be greater than maximum value")

    if inclusive:
        return min_val <= value <= max_val
    else:
        return min_val < value < max_val


@mcp_function(
    description="Check if a value is between two bounds (exclusive by default, like mathematical interval notation).",
    namespace="arithmetic",
    category="comparison",
    execution_modes=["local", "remote"],
    examples=[
        {
            "input": {"value": 5, "lower": 1, "upper": 10},
            "output": True,
            "description": "Value between bounds",
        },
        {
            "input": {"value": 1, "lower": 1, "upper": 10},
            "output": False,
            "description": "Value at lower bound (exclusive)",
        },
        {
            "input": {"value": 10, "lower": 1, "upper": 10},
            "output": False,
            "description": "Value at upper bound (exclusive)",
        },
        {
            "input": {"value": 0, "lower": 1, "upper": 10},
            "output": False,
            "description": "Value below lower bound",
        },
    ],
)
async def between(value: Number, lower: Number, upper: Number) -> bool:
    """
    Check if a value is between two bounds (exclusive).

    Args:
        value: The value to check
        lower: The lower bound (exclusive)
        upper: The upper bound (exclusive)

    Returns:
        True if lower < value < upper, False otherwise

    Examples:
        await between(5, 1, 10) → True
        await between(1, 1, 10) → False
        await between(10, 1, 10) → False
    """
    return lower < value < upper


# Export all relational comparison functions
__all__ = [
    "equal",
    "not_equal",
    "less_than",
    "less_than_or_equal",
    "greater_than",
    "greater_than_or_equal",
    "in_range",
    "between",
]

if __name__ == "__main__":
    import asyncio

    async def test_relational_operations():
        """Test all relational comparison operations."""
        print("🔍 Relational Comparison Operations Test")
        print("=" * 40)

        # Test basic comparisons
        print(f"equal(5, 5) = {await equal(5, 5)}")
        print(f"not_equal(5, 3) = {await not_equal(5, 3)}")
        print(f"less_than(3, 5) = {await less_than(3, 5)}")
        print(f"less_than_or_equal(5, 5) = {await less_than_or_equal(5, 5)}")
        print(f"greater_than(5, 3) = {await greater_than(5, 3)}")
        print(f"greater_than_or_equal(5, 5) = {await greater_than_or_equal(5, 5)}")

        # Test range operations
        print(f"in_range(5, 1, 10) = {await in_range(5, 1, 10)}")
        print(
            f"in_range(1, 1, 10, inclusive=False) = {await in_range(1, 1, 10, inclusive=False)}"
        )
        print(f"between(5, 1, 10) = {await between(5, 1, 10)}")
        print(f"between(1, 1, 10) = {await between(1, 1, 10)}")

        print("\n✅ All relational comparison operations working!")

    asyncio.run(test_relational_operations())
