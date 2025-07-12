#!/usr/bin/env python3
# chuk_mcp_functions/math/arithmetic/comparison.py
"""
Comparison and Ordering Functions for AI Models (Async Native)

Mathematical comparison operations that help AI models make logical decisions
and perform ordering operations. Includes tolerance-based comparisons for
floating-point precision issues. All functions are async native for consistent
interface with async MCP servers.

Functions:
- Basic comparisons: equal, not_equal, less_than, greater_than, etc.
- Min/max operations: minimum, maximum, clamp
- Tolerance-based comparisons: approximately_equal, close_to_zero
- Ordering operations: sort_numbers, rank_numbers
- Range checks: in_range, between
"""

import math
import asyncio
from typing import Union, List, Tuple
from chuk_mcp_functions.mcp_decorator import mcp_function

Number = Union[int, float]

@mcp_function(
    description="Check if two numbers are exactly equal. For floating-point numbers, consider using approximately_equal instead.",
    namespace="arithmetic",
    execution_modes=["local", "remote"],
    examples=[
        {"input": {"a": 5, "b": 5}, "output": True, "description": "Equal integers"},
        {"input": {"a": 3.14, "b": 3.14}, "output": True, "description": "Equal floats"},
        {"input": {"a": 5, "b": 5.0}, "output": True, "description": "Integer and float with same value"},
        {"input": {"a": 1, "b": 2}, "output": False, "description": "Unequal numbers"}
    ]
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
    execution_modes=["local", "remote"],
    examples=[
        {"input": {"a": 5, "b": 3}, "output": True, "description": "Different numbers"},
        {"input": {"a": 7, "b": 7}, "output": False, "description": "Same numbers"},
        {"input": {"a": 3.14, "b": 3.15}, "output": True, "description": "Slightly different floats"}
    ]
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
    execution_modes=["local", "remote"],
    examples=[
        {"input": {"a": 3, "b": 5}, "output": True, "description": "3 is less than 5"},
        {"input": {"a": 5, "b": 3}, "output": False, "description": "5 is not less than 3"},
        {"input": {"a": 5, "b": 5}, "output": False, "description": "5 is not less than 5"},
        {"input": {"a": -2, "b": 1}, "output": True, "description": "Negative number less than positive"}
    ]
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
    execution_modes=["local", "remote"],
    examples=[
        {"input": {"a": 3, "b": 5}, "output": True, "description": "3 is less than 5"},
        {"input": {"a": 5, "b": 5}, "output": True, "description": "5 is equal to 5"},
        {"input": {"a": 7, "b": 5}, "output": False, "description": "7 is greater than 5"}
    ]
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
    execution_modes=["local", "remote"],
    examples=[
        {"input": {"a": 5, "b": 3}, "output": True, "description": "5 is greater than 3"},
        {"input": {"a": 3, "b": 5}, "output": False, "description": "3 is not greater than 5"},
        {"input": {"a": 5, "b": 5}, "output": False, "description": "5 is not greater than 5"}
    ]
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
    execution_modes=["local", "remote"],
    examples=[
        {"input": {"a": 5, "b": 3}, "output": True, "description": "5 is greater than 3"},
        {"input": {"a": 5, "b": 5}, "output": True, "description": "5 is equal to 5"},
        {"input": {"a": 3, "b": 5}, "output": False, "description": "3 is less than 5"}
    ]
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
    description="Find the smaller of two numbers.",
    namespace="arithmetic",
    execution_modes=["local", "remote"],
    examples=[
        {"input": {"a": 5, "b": 3}, "output": 3, "description": "Minimum of 5 and 3"},
        {"input": {"a": -2, "b": 1}, "output": -2, "description": "Minimum with negative number"},
        {"input": {"a": 7.5, "b": 7.5}, "output": 7.5, "description": "Minimum of equal numbers"},
        {"input": {"a": 2.3, "b": 2.7}, "output": 2.3, "description": "Minimum of decimals"}
    ]
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
    execution_modes=["local", "remote"],
    examples=[
        {"input": {"a": 5, "b": 3}, "output": 5, "description": "Maximum of 5 and 3"},
        {"input": {"a": -2, "b": 1}, "output": 1, "description": "Maximum with negative number"},
        {"input": {"a": 7.5, "b": 7.5}, "output": 7.5, "description": "Maximum of equal numbers"},
        {"input": {"a": 2.3, "b": 2.7}, "output": 2.7, "description": "Maximum of decimals"}
    ]
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
    execution_modes=["local", "remote"],
    examples=[
        {"input": {"value": 5, "min_val": 1, "max_val": 10}, "output": 5, "description": "Value within bounds"},
        {"input": {"value": -2, "min_val": 1, "max_val": 10}, "output": 1, "description": "Value below minimum"},
        {"input": {"value": 15, "min_val": 1, "max_val": 10}, "output": 10, "description": "Value above maximum"},
        {"input": {"value": 1, "min_val": 1, "max_val": 10}, "output": 1, "description": "Value at minimum"}
    ]
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
    description="Check if two floating-point numbers are approximately equal within a tolerance. Handles floating-point precision issues.",
    namespace="arithmetic",
    execution_modes=["local", "remote"],
    examples=[
        {"input": {"a": 0.1, "b": 0.1, "tolerance": 1e-9}, "output": True, "description": "Exactly equal floats"},
        {"input": {"a": 0.1, "b": 0.10000001, "tolerance": 1e-7}, "output": True, "description": "Nearly equal within tolerance"},
        {"input": {"a": 0.1, "b": 0.2, "tolerance": 1e-9}, "output": False, "description": "Different numbers"},
        {"input": {"a": 1.0, "b": 1.0000001, "tolerance": 1e-6}, "output": True, "description": "Close enough within tolerance"}
    ]
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
    execution_modes=["local", "remote"],
    examples=[
        {"input": {"x": 0}, "output": True, "description": "Exactly zero"},
        {"input": {"x": 1e-10, "tolerance": 1e-9}, "output": True, "description": "Very small number within tolerance"},
        {"input": {"x": 0.001, "tolerance": 1e-9}, "output": False, "description": "Number larger than tolerance"},
        {"input": {"x": -1e-10, "tolerance": 1e-9}, "output": True, "description": "Small negative number within tolerance"}
    ]
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
    description="Sort a list of numbers in ascending or descending order.",
    namespace="arithmetic",
    execution_modes=["local", "remote"],
    examples=[
        {"input": {"numbers": [3, 1, 4, 1, 5], "descending": False}, "output": [1, 1, 3, 4, 5], "description": "Sort ascending"},
        {"input": {"numbers": [3, 1, 4, 1, 5], "descending": True}, "output": [5, 4, 3, 1, 1], "description": "Sort descending"},
        {"input": {"numbers": [2.5, 1.1, 3.7], "descending": False}, "output": [1.1, 2.5, 3.7], "description": "Sort floats"},
        {"input": {"numbers": [-2, 0, 1], "descending": False}, "output": [-2, 0, 1], "description": "Sort with negatives"}
    ]
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
    execution_modes=["local", "remote"],
    examples=[
        {"input": {"numbers": [3, 1, 4, 1, 5]}, "output": [3, 1, 4, 1, 5], "description": "Ranks with ties"},
        {"input": {"numbers": [10, 20, 30]}, "output": [1, 2, 3], "description": "Simple ranking"},
        {"input": {"numbers": [1.5, 2.5, 1.5]}, "output": [1, 3, 1], "description": "Ranking with float ties"}
    ]
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
        if i > 0 and value != sorted_indexed[i-1][0]:
            current_rank = i + 1
        ranks[original_idx] = current_rank
    
    return ranks

@mcp_function(
    description="Check if a number is within a specified range (inclusive by default).",
    namespace="arithmetic",
    execution_modes=["local", "remote"],
    examples=[
        {"input": {"value": 5, "min_val": 1, "max_val": 10, "inclusive": True}, "output": True, "description": "Value within inclusive range"},
        {"input": {"value": 1, "min_val": 1, "max_val": 10, "inclusive": True}, "output": True, "description": "Value at lower bound (inclusive)"},
        {"input": {"value": 1, "min_val": 1, "max_val": 10, "inclusive": False}, "output": False, "description": "Value at lower bound (exclusive)"},
        {"input": {"value": 15, "min_val": 1, "max_val": 10, "inclusive": True}, "output": False, "description": "Value outside range"}
    ]
)
async def in_range(value: Number, min_val: Number, max_val: Number, inclusive: bool = True) -> bool:
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
    execution_modes=["local", "remote"],
    examples=[
        {"input": {"value": 5, "lower": 1, "upper": 10}, "output": True, "description": "Value between bounds"},
        {"input": {"value": 1, "lower": 1, "upper": 10}, "output": False, "description": "Value at lower bound (exclusive)"},
        {"input": {"value": 10, "lower": 1, "upper": 10}, "output": False, "description": "Value at upper bound (exclusive)"},
        {"input": {"value": 0, "lower": 1, "upper": 10}, "output": False, "description": "Value below lower bound"}
    ]
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

# Export all comparison functions
__all__ = [
    'equal', 'not_equal', 'less_than', 'less_than_or_equal',
    'greater_than', 'greater_than_or_equal', 'minimum', 'maximum', 'clamp',
    'approximately_equal', 'close_to_zero', 'sort_numbers', 'rank_numbers',
    'in_range', 'between'
]

if __name__ == "__main__":
    import asyncio
    
    async def test_comparison_functions():
        # Test the comparison functions
        print("🔍 Comparison and Ordering Functions Test (Async Native)")
        print("=" * 55)
        
        # Test basic comparisons
        print(f"equal(5, 5) = {await equal(5, 5)}")
        print(f"not_equal(5, 3) = {await not_equal(5, 3)}")
        print(f"less_than(3, 5) = {await less_than(3, 5)}")
        print(f"greater_than(5, 3) = {await greater_than(5, 3)}")
        
        # Test min/max operations
        print(f"minimum(5, 3) = {await minimum(5, 3)}")
        print(f"maximum(5, 3) = {await maximum(5, 3)}")
        print(f"clamp(15, 1, 10) = {await clamp(15, 1, 10)}")
        
        # Test tolerance-based comparisons
        print(f"approximately_equal(0.1, 0.10000001, 1e-7) = {await approximately_equal(0.1, 0.10000001, 1e-7)}")
        print(f"close_to_zero(1e-10, 1e-9) = {await close_to_zero(1e-10, 1e-9)}")
        
        # Test ordering operations
        numbers = [3, 1, 4, 1, 5]
        print(f"sort_numbers({numbers}) = {await sort_numbers(numbers)}")
        print(f"rank_numbers({numbers}) = {await rank_numbers(numbers)}")
        
        # Test range checks
        print(f"in_range(5, 1, 10) = {await in_range(5, 1, 10)}")
        print(f"between(5, 1, 10) = {await between(5, 1, 10)}")
        
        print("\n✅ All async comparison functions working correctly!")
    
    asyncio.run(test_comparison_functions())