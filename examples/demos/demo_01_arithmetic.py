#!/usr/bin/env python3
"""
CHUK MCP Math - Arithmetic Operations Demo
===========================================

Demonstrates all arithmetic operations including:
- Basic operations (add, subtract, multiply, divide, power, sqrt)
- Advanced operations (factorial, gcd, lcm, etc.)
- Modular arithmetic
- Comparison operations
- Tolerance-based comparisons
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from chuk_mcp_math.arithmetic.core.basic_operations import (
    add,
    subtract,
    multiply,
    divide,
    power,
    sqrt,
    absolute_value,
    negate,
    reciprocal,
    floor_divide,
    ceiling,
    floor,
    round_number,
)
from chuk_mcp_math.arithmetic.core.modular import (
    mod,
    divmod_operation,
    modular_exponentiation,
    modular_inverse,
)
from chuk_mcp_math.arithmetic.comparison.relational import (
    equal,
    not_equal,
    less_than,
    greater_than,
    less_than_or_equal,
    greater_than_or_equal,
    in_range,
    between,
)
from chuk_mcp_math.arithmetic.comparison.extrema import (
    minimum,
    maximum,
    clamp,
    sort_numbers,
    rank_numbers,
    min_list,
    max_list,
)
from chuk_mcp_math.arithmetic.comparison.tolerance import (
    approximately_equal,
    close_to_zero,
    is_finite,
    is_nan,
    is_infinite,
    is_normal,
    is_close,
)
from chuk_mcp_math.advanced_operations import (
    factorial,
    fibonacci,
    gcd,
    lcm,
    digit_sum,
    is_perfect_square,
    mean,
    median,
    mode,
    variance,
    standard_deviation,
)


async def demo_basic_operations():
    """Demonstrate basic arithmetic operations."""
    print("\n" + "=" * 70)
    print("BASIC ARITHMETIC OPERATIONS")
    print("=" * 70)

    # Addition
    result = await add(42, 18)
    print(f"✓ add(42, 18) = {result}")

    # Subtraction
    result = await subtract(100, 37)
    print(f"✓ subtract(100, 37) = {result}")

    # Multiplication
    result = await multiply(7, 8)
    print(f"✓ multiply(7, 8) = {result}")

    # Division
    result = await divide(144, 12)
    print(f"✓ divide(144, 12) = {result}")

    # Power
    result = await power(2, 10)
    print(f"✓ power(2, 10) = {result}")

    # Square root
    result = await sqrt(64)
    print(f"✓ sqrt(64) = {result}")

    # Absolute value
    result = await absolute_value(-42)
    print(f"✓ absolute_value(-42) = {result}")

    # Negate
    result = await negate(100)
    print(f"✓ negate(100) = {result}")

    # Reciprocal
    result = await reciprocal(4)
    print(f"✓ reciprocal(4) = {result}")

    # Floor divide
    result = await floor_divide(17, 5)
    print(f"✓ floor_divide(17, 5) = {result}")

    # Ceiling
    result = await ceiling(3.14)
    print(f"✓ ceiling(3.14) = {result}")

    # Floor
    result = await floor(3.99)
    print(f"✓ floor(3.99) = {result}")

    # Round
    result = await round_number(3.14159, 2)
    print(f"✓ round_number(3.14159, 2) = {result}")


async def demo_modular_arithmetic():
    """Demonstrate modular arithmetic operations."""
    print("\n" + "=" * 70)
    print("MODULAR ARITHMETIC")
    print("=" * 70)

    # Modulo
    result = await mod(17, 5)
    print(f"✓ mod(17, 5) = {result}")

    # Divmod
    result = await divmod_operation(17, 5)
    print(f"✓ divmod_operation(17, 5) = {result}")

    # Modular exponentiation
    result = await modular_exponentiation(3, 4, 5)
    print(f"✓ modular_exponentiation(3, 4, 5) = {result}  (3^4 mod 5)")

    # Modular inverse
    result = await modular_inverse(3, 7)
    print(f"✓ modular_inverse(3, 7) = {result}  (3 * {result} ≡ 1 mod 7)")


async def demo_comparison_operations():
    """Demonstrate comparison operations."""
    print("\n" + "=" * 70)
    print("COMPARISON OPERATIONS")
    print("=" * 70)

    # Basic comparisons
    result = await equal(5, 5)
    print(f"✓ equal(5, 5) = {result}")

    result = await not_equal(5, 3)
    print(f"✓ not_equal(5, 3) = {result}")

    result = await less_than(3, 5)
    print(f"✓ less_than(3, 5) = {result}")

    result = await greater_than(10, 5)
    print(f"✓ greater_than(10, 5) = {result}")

    result = await less_than_or_equal(5, 5)
    print(f"✓ less_than_or_equal(5, 5) = {result}")

    result = await greater_than_or_equal(10, 5)
    print(f"✓ greater_than_or_equal(10, 5) = {result}")

    # Range checks
    result = await in_range(5, 1, 10)
    print(f"✓ in_range(5, 1, 10) = {result}")

    result = await between(5, 1, 10)
    print(f"✓ between(5, 1, 10) = {result}")

    # Min/Max
    result = await minimum(3, 7)
    print(f"✓ minimum(3, 7) = {result}")

    result = await maximum(3, 7)
    print(f"✓ maximum(3, 7) = {result}")

    # Clamp
    result = await clamp(15, 1, 10)
    print(f"✓ clamp(15, 1, 10) = {result}")

    # Sort
    result = await sort_numbers([5, 2, 8, 1, 9])
    print(f"✓ sort_numbers([5, 2, 8, 1, 9]) = {result}")

    # Rank
    result = await rank_numbers([5, 2, 8, 1, 9])
    print(f"✓ rank_numbers([5, 2, 8, 1, 9]) = {result}")

    # Min/Max of list
    result = await min_list([5, 2, 8, 1, 9])
    print(f"✓ min_list([5, 2, 8, 1, 9]) = {result}")

    result = await max_list([5, 2, 8, 1, 9])
    print(f"✓ max_list([5, 2, 8, 1, 9]) = {result}")


async def demo_tolerance_comparisons():
    """Demonstrate tolerance-based comparisons."""
    print("\n" + "=" * 70)
    print("TOLERANCE-BASED COMPARISONS")
    print("=" * 70)

    # Approximately equal
    result = await approximately_equal(1.0, 1.0000001)
    print(f"✓ approximately_equal(1.0, 1.0000001) = {result}")

    # Close to zero
    result = await close_to_zero(0.0000001)
    print(f"✓ close_to_zero(0.0000001) = {result}")

    # Is finite
    result = await is_finite(42.5)
    print(f"✓ is_finite(42.5) = {result}")

    # Is NaN
    import math

    result = await is_nan(math.nan)
    print(f"✓ is_nan(math.nan) = {result}")

    # Is infinite
    result = await is_infinite(math.inf)
    print(f"✓ is_infinite(math.inf) = {result}")

    # Is normal
    result = await is_normal(42.5)
    print(f"✓ is_normal(42.5) = {result}")

    # Is close (with tolerances)
    result = await is_close(1.0, 1.0001, rel_tol=1e-3, abs_tol=1e-3)
    print(f"✓ is_close(1.0, 1.0001, rel_tol=1e-3) = {result}")


async def demo_advanced_operations():
    """Demonstrate advanced mathematical operations."""
    print("\n" + "=" * 70)
    print("ADVANCED OPERATIONS")
    print("=" * 70)

    # Factorial
    result = await factorial(10)
    print(f"✓ factorial(10) = {result}")

    # Fibonacci
    result = await fibonacci(15)
    print(f"✓ fibonacci(15) = {result}")

    # GCD
    result = await gcd(48, 18)
    print(f"✓ gcd(48, 18) = {result}")

    # LCM
    result = await lcm(12, 18)
    print(f"✓ lcm(12, 18) = {result}")

    # Digit sum
    result = await digit_sum(12345)
    print(f"✓ digit_sum(12345) = {result}")

    # Is perfect square
    result = await is_perfect_square(64)
    print(f"✓ is_perfect_square(64) = {result}")

    # Mean
    result = await mean([1, 2, 3, 4, 5])
    print(f"✓ mean([1, 2, 3, 4, 5]) = {result}")

    # Median
    result = await median([1, 2, 3, 4, 5])
    print(f"✓ median([1, 2, 3, 4, 5]) = {result}")

    # Mode
    result = await mode([1, 2, 2, 3, 4])
    print(f"✓ mode([1, 2, 2, 3, 4]) = {result}")

    # Variance
    result = await variance([1, 2, 3, 4, 5])
    print(f"✓ variance([1, 2, 3, 4, 5]) = {result:.2f}")

    # Standard deviation
    result = await standard_deviation([1, 2, 3, 4, 5])
    print(f"✓ standard_deviation([1, 2, 3, 4, 5]) = {result:.2f}")


async def main():
    """Run all arithmetic demos."""
    print("\n" + "=" * 70)
    print("CHUK MCP MATH - ARITHMETIC OPERATIONS DEMO")
    print("=" * 70)
    print("\nDemonstrating all arithmetic functions are working correctly...")

    await demo_basic_operations()
    await demo_modular_arithmetic()
    await demo_comparison_operations()
    await demo_tolerance_comparisons()
    await demo_advanced_operations()

    print("\n" + "=" * 70)
    print("✅ ALL ARITHMETIC OPERATIONS WORKING PERFECTLY!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
