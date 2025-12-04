#!/usr/bin/env python3
# chuk_mcp_math/arithmetic/comparison/extrema.py
"""
Extrema and Ordering Operations - Async Native

Functions for finding minimum and maximum values, clamping, and ordering operations.

Functions:
- minimum, maximum, clamp
- sort_numbers, rank_numbers
"""

import asyncio
from typing import Union, List
from chuk_mcp_math.mcp_decorator import mcp_function

Number = Union[int, float]


@mcp_function(
    description="Find the smaller of two numbers.",
    namespace="arithmetic",
    category="comparison",
    execution_modes=["local", "remote"],
    examples=[
        {"input": {"a": 5, "b": 3}, "output": 3, "description": "Minimum of 5 and 3"},
        {
            "input": {"a": -2, "b": 1},
            "output": -2,
            "description": "Minimum with negative number",
        },
        {
            "input": {"a": 7.5, "b": 7.5},
            "output": 7.5,
            "description": "Minimum of equal numbers",
        },
        {
            "input": {"a": 2.3, "b": 2.7},
            "output": 2.3,
            "description": "Minimum of decimals",
        },
    ],
)
async def minimum(a: Number, b: Number) -> Number:
    """
    Find the smaller of two numbers.

    Args:
        a: First number
        b: Second number

    Returns:
        The smaller of a and b

    Examples:
        await minimum(5, 3) → 3
        await minimum(-2, 1) → -2
        await minimum(7.5, 7.5) → 7.5
    """
    return min(a, b)


@mcp_function(
    description="Find the larger of two numbers.",
    namespace="arithmetic",
    category="comparison",
    execution_modes=["local", "remote"],
    examples=[
        {"input": {"a": 5, "b": 3}, "output": 5, "description": "Maximum of 5 and 3"},
        {
            "input": {"a": -2, "b": 1},
            "output": 1,
            "description": "Maximum with negative number",
        },
        {
            "input": {"a": 7.5, "b": 7.5},
            "output": 7.5,
            "description": "Maximum of equal numbers",
        },
        {
            "input": {"a": 2.3, "b": 2.7},
            "output": 2.7,
            "description": "Maximum of decimals",
        },
    ],
)
async def maximum(a: Number, b: Number) -> Number:
    """
    Find the larger of two numbers.

    Args:
        a: First number
        b: Second number

    Returns:
        The larger of a and b

    Examples:
        await maximum(5, 3) → 5
        await maximum(-2, 1) → 1
        await maximum(7.5, 7.5) → 7.5
    """
    return max(a, b)


@mcp_function(
    description="Clamp a value between a minimum and maximum bound. Ensures the value stays within specified limits.",
    namespace="arithmetic",
    category="comparison",
    execution_modes=["local", "remote"],
    examples=[
        {
            "input": {"value": 5, "min_val": 1, "max_val": 10},
            "output": 5,
            "description": "Value within bounds",
        },
        {
            "input": {"value": -2, "min_val": 1, "max_val": 10},
            "output": 1,
            "description": "Value below minimum",
        },
        {
            "input": {"value": 15, "min_val": 1, "max_val": 10},
            "output": 10,
            "description": "Value above maximum",
        },
        {
            "input": {"value": 1, "min_val": 1, "max_val": 10},
            "output": 1,
            "description": "Value at minimum",
        },
    ],
)
async def clamp(value: Number, min_val: Number, max_val: Number) -> Number:
    """
    Clamp a value between minimum and maximum bounds.

    Args:
        value: The value to clamp
        min_val: The minimum allowed value
        max_val: The maximum allowed value

    Returns:
        The clamped value (between min_val and max_val inclusive)

    Raises:
        ValueError: If min_val > max_val

    Examples:
        await clamp(5, 1, 10) → 5
        await clamp(-2, 1, 10) → 1
        await clamp(15, 1, 10) → 10
    """
    if min_val > max_val:
        raise ValueError("Minimum value cannot be greater than maximum value")

    return max(min_val, min(value, max_val))


@mcp_function(
    description="Sort a list of numbers in ascending or descending order.",
    namespace="arithmetic",
    category="comparison",
    execution_modes=["local", "remote"],
    examples=[
        {
            "input": {"numbers": [3, 1, 4, 1, 5], "descending": False},
            "output": [1, 1, 3, 4, 5],
            "description": "Sort ascending",
        },
        {
            "input": {"numbers": [3, 1, 4, 1, 5], "descending": True},
            "output": [5, 4, 3, 1, 1],
            "description": "Sort descending",
        },
        {
            "input": {"numbers": [2.5, 1.1, 3.7], "descending": False},
            "output": [1.1, 2.5, 3.7],
            "description": "Sort floats",
        },
        {
            "input": {"numbers": [-2, 0, 1], "descending": False},
            "output": [-2, 0, 1],
            "description": "Sort with negatives",
        },
    ],
)
async def sort_numbers(numbers: List[Number], descending: bool = False) -> List[Number]:
    """
    Sort a list of numbers in ascending or descending order.

    Args:
        numbers: List of numbers to sort
        descending: If True, sort in descending order (default: False for ascending)

    Returns:
        New sorted list of numbers

    Examples:
        await sort_numbers([3, 1, 4, 1, 5]) → [1, 1, 3, 4, 5]
        await sort_numbers([3, 1, 4, 1, 5], descending=True) → [5, 4, 3, 1, 1]
    """
    # For large lists, yield control during sorting
    if len(numbers) > 1000:
        await asyncio.sleep(0)

    return sorted(numbers, reverse=descending)


@mcp_function(
    description="Get the rank (1-based position) of each number in a list when sorted. Handles ties appropriately.",
    namespace="arithmetic",
    category="comparison",
    execution_modes=["local", "remote"],
    examples=[
        {
            "input": {"numbers": [3, 1, 4, 1, 5]},
            "output": [3, 1, 4, 1, 5],
            "description": "Ranks with ties",
        },
        {
            "input": {"numbers": [10, 20, 30]},
            "output": [1, 2, 3],
            "description": "Simple ranking",
        },
        {
            "input": {"numbers": [1.5, 2.5, 1.5]},
            "output": [1, 3, 1],
            "description": "Ranking with float ties",
        },
    ],
)
async def rank_numbers(numbers: List[Number]) -> List[int]:
    """
    Get the rank (1-based position) of each number when sorted in ascending order.

    Args:
        numbers: List of numbers to rank

    Returns:
        List of ranks corresponding to each input number

    Examples:
        await rank_numbers([3, 1, 4, 1, 5]) → [3, 1, 4, 1, 5]
        await rank_numbers([10, 20, 30]) → [1, 2, 3]
    """
    # For large lists, yield control during processing
    if len(numbers) > 1000:
        await asyncio.sleep(0)

    # Create list of (value, original_index) pairs
    indexed_numbers = [(num, i) for i, num in enumerate(numbers)]

    # Sort by value
    sorted_indexed = sorted(indexed_numbers, key=lambda x: x[0])

    # Assign ranks
    ranks = [0] * len(numbers)
    current_rank = 1

    for i, (value, original_idx) in enumerate(sorted_indexed):
        if i > 0 and value != sorted_indexed[i - 1][0]:
            current_rank = i + 1
        ranks[original_idx] = current_rank

    return ranks


@mcp_function(
    description="Find the minimum value in a list of numbers.",
    namespace="arithmetic",
    category="comparison",
    execution_modes=["local", "remote"],
    examples=[
        {
            "input": {"numbers": [3, 1, 4, 1, 5]},
            "output": 1,
            "description": "Minimum of list",
        },
        {
            "input": {"numbers": [-2, 0, 1]},
            "output": -2,
            "description": "Minimum with negatives",
        },
        {
            "input": {"numbers": [2.5, 1.1, 3.7]},
            "output": 1.1,
            "description": "Minimum of floats",
        },
        {"input": {"numbers": [42]}, "output": 42, "description": "Single element"},
    ],
)
async def min_list(numbers: List[Number]) -> Number:
    """
    Find the minimum value in a list of numbers.

    Args:
        numbers: List of numbers (cannot be empty)

    Returns:
        The minimum value in the list

    Raises:
        ValueError: If list is empty

    Examples:
        await min_list([3, 1, 4, 1, 5]) → 1
        await min_list([-2, 0, 1]) → -2
        await min_list([2.5, 1.1, 3.7]) → 1.1
    """
    if not numbers:
        raise ValueError("Cannot find minimum of empty list")

    # For large lists, yield control
    if len(numbers) > 1000:
        await asyncio.sleep(0)

    return min(numbers)


@mcp_function(
    description="Find the maximum value in a list of numbers.",
    namespace="arithmetic",
    category="comparison",
    execution_modes=["local", "remote"],
    examples=[
        {
            "input": {"numbers": [3, 1, 4, 1, 5]},
            "output": 5,
            "description": "Maximum of list",
        },
        {
            "input": {"numbers": [-2, 0, 1]},
            "output": 1,
            "description": "Maximum with negatives",
        },
        {
            "input": {"numbers": [2.5, 1.1, 3.7]},
            "output": 3.7,
            "description": "Maximum of floats",
        },
        {"input": {"numbers": [42]}, "output": 42, "description": "Single element"},
    ],
)
async def max_list(numbers: List[Number]) -> Number:
    """
    Find the maximum value in a list of numbers.

    Args:
        numbers: List of numbers (cannot be empty)

    Returns:
        The maximum value in the list

    Raises:
        ValueError: If list is empty

    Examples:
        await max_list([3, 1, 4, 1, 5]) → 5
        await max_list([-2, 0, 1]) → 1
        await max_list([2.5, 1.1, 3.7]) → 3.7
    """
    if not numbers:
        raise ValueError("Cannot find maximum of empty list")

    # For large lists, yield control
    if len(numbers) > 1000:
        await asyncio.sleep(0)

    return max(numbers)


# Export all extrema and ordering functions
__all__ = [
    "minimum",
    "maximum",
    "clamp",
    "sort_numbers",
    "rank_numbers",
    "min_list",
    "max_list",
]

if __name__ == "__main__":
    import asyncio

    async def test_extrema_operations():
        """Test all extrema and ordering operations."""
        print("🔍 Extrema and Ordering Operations Test")
        print("=" * 40)

        # Test min/max operations
        print(f"minimum(5, 3) = {await minimum(5, 3)}")
        print(f"maximum(5, 3) = {await maximum(5, 3)}")
        print(f"clamp(15, 1, 10) = {await clamp(15, 1, 10)}")

        # Test list operations
        test_list = [3, 1, 4, 1, 5]
        print(f"sort_numbers({test_list}) = {await sort_numbers(test_list)}")
        print(
            f"sort_numbers({test_list}, descending=True) = {await sort_numbers(test_list, descending=True)}"
        )
        print(f"rank_numbers({test_list}) = {await rank_numbers(test_list)}")
        print(f"min_list({test_list}) = {await min_list(test_list)}")
        print(f"max_list({test_list}) = {await max_list(test_list)}")

        print("\n✅ All extrema and ordering operations working!")

    asyncio.run(test_extrema_operations())
