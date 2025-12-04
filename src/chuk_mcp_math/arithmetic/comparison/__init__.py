#!/usr/bin/env python3
# chuk_mcp_math/arithmetic/comparison/__init__.py
"""
Comparison and Ordering Operations Module

Functions for comparing numbers, finding extrema, and handling floating-point precision issues.
Essential for decision making, sorting, and tolerance-based comparisons.

Submodules:
- relational: equal, not_equal, less_than, greater_than, in_range, between
- extrema: minimum, maximum, clamp, sort_numbers, rank_numbers, min_list, max_list
- tolerance: approximately_equal, close_to_zero, is_finite, is_nan, is_infinite, is_close, is_normal

All functions are async native for optimal performance in async environments.
"""

# Import all comparison submodules
from . import relational
from . import extrema
from . import tolerance

# Import all functions for direct access
from .relational import (
    equal,
    not_equal,
    less_than,
    less_than_or_equal,
    greater_than,
    greater_than_or_equal,
    in_range,
    between,
)

from .extrema import (
    minimum,
    maximum,
    clamp,
    sort_numbers,
    rank_numbers,
    min_list,
    max_list,
)

from .tolerance import (
    approximately_equal,
    close_to_zero,
    is_finite,
    is_nan,
    is_infinite,
    is_normal,
    is_close,
)

# Export all comparison functions
__all__ = [
    # Submodules
    "relational",
    "extrema",
    "tolerance",
    # Relational operations
    "equal",
    "not_equal",
    "less_than",
    "less_than_or_equal",
    "greater_than",
    "greater_than_or_equal",
    "in_range",
    "between",
    # Extrema operations
    "minimum",
    "maximum",
    "clamp",
    "sort_numbers",
    "rank_numbers",
    "min_list",
    "max_list",
    # Tolerance operations
    "approximately_equal",
    "close_to_zero",
    "is_finite",
    "is_nan",
    "is_infinite",
    "is_normal",
    "is_close",
]


async def test_comparison_functions():
    """Test all comparison functions."""
    print("🔍 Comparison and Ordering Functions Test")
    print("=" * 40)

    # Test relational operations
    print("Relational Operations:")
    print(f"  equal(5, 5) = {await equal(5, 5)}")
    print(f"  not_equal(5, 3) = {await not_equal(5, 3)}")
    print(f"  less_than(3, 5) = {await less_than(3, 5)}")
    print(f"  greater_than(5, 3) = {await greater_than(5, 3)}")
    print(f"  in_range(5, 1, 10) = {await in_range(5, 1, 10)}")
    print(f"  between(5, 1, 10) = {await between(5, 1, 10)}")

    # Test extrema operations
    print("\nExtrema Operations:")
    print(f"  minimum(5, 3) = {await minimum(5, 3)}")
    print(f"  maximum(5, 3) = {await maximum(5, 3)}")
    print(f"  clamp(15, 1, 10) = {await clamp(15, 1, 10)}")

    test_list = [3, 1, 4, 1, 5]
    print(f"  sort_numbers({test_list}) = {await sort_numbers(test_list)}")
    print(f"  rank_numbers({test_list}) = {await rank_numbers(test_list)}")
    print(f"  min_list({test_list}) = {await min_list(test_list)}")
    print(f"  max_list({test_list}) = {await max_list(test_list)}")

    # Test tolerance operations
    print("\nTolerance Operations:")
    print(
        f"  approximately_equal(0.1, 0.10000001, 1e-7) = {await approximately_equal(0.1, 0.10000001, 1e-7)}"
    )
    print(f"  close_to_zero(1e-10, 1e-9) = {await close_to_zero(1e-10, 1e-9)}")
    print(f"  is_finite(42.5) = {await is_finite(42.5)}")
    print(f"  is_nan(float('nan')) = {await is_nan(float('nan'))}")
    print(f"  is_infinite(float('inf')) = {await is_infinite(float('inf'))}")
    print(
        f"  is_close(1000000.1, 1000000.2, rel_tol=1e-6) = {await is_close(1000000.1, 1000000.2, rel_tol=1e-6)}"
    )

    print("\n✅ All comparison functions working!")


if __name__ == "__main__":
    import asyncio

    asyncio.run(test_comparison_functions())
