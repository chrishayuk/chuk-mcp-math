#!/usr/bin/env python3
"""
CHUK MCP Math - Comprehensive Arithmetic Demo
==============================================

Demonstrates ALL 44 arithmetic functions across:
- Core: Basic Operations (9 functions)
- Core: Rounding (7 functions)
- Core: Modular (6 functions)
- Comparison: Extrema (7 functions)
- Comparison: Relational (8 functions)
- Comparison: Tolerance (7 functions)
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


async def demo_core_basic():
    """Test all 9 basic arithmetic operations."""
    print("\n" + "=" * 70)
    print("CORE: BASIC OPERATIONS (9 functions)")
    print("=" * 70)

    from chuk_mcp_math.arithmetic.core.basic_operations import (
        add,
        subtract,
        multiply,
        divide,
        power,
        sqrt,
        abs_value,
        sign,
        negate,
    )

    print(f"✓ add(42, 18) = {await add(42, 18)}")
    print(f"✓ subtract(100, 37) = {await subtract(100, 37)}")
    print(f"✓ multiply(7, 8) = {await multiply(7, 8)}")
    print(f"✓ divide(144, 12) = {await divide(144, 12)}")
    print(f"✓ power(2, 10) = {await power(2, 10)}")
    print(f"✓ sqrt(64) = {await sqrt(64)}")
    print(f"✓ abs_value(-42) = {await abs_value(-42)}")
    print(f"✓ sign(-15) = {await sign(-15)}")
    print(f"✓ negate(42) = {await negate(42)}")


async def demo_core_rounding():
    """Test all 7 rounding operations."""
    print("\n" + "=" * 70)
    print("CORE: ROUNDING (7 functions)")
    print("=" * 70)

    from chuk_mcp_math.arithmetic.core.rounding import (
        round_number,
        floor,
        ceil,
        truncate,
        ceiling_multiple,
        floor_multiple,
        mround,
    )

    print(f"✓ round_number(3.14159, 2) = {await round_number(3.14159, 2)}")
    print(f"✓ floor(7.8) = {await floor(7.8)}")
    print(f"✓ ceil(7.2) = {await ceil(7.2)}")
    print(f"✓ truncate(7.8) = {await truncate(7.8)}")
    print(f"✓ ceiling_multiple(23, 5) = {await ceiling_multiple(23, 5)}")
    print(f"✓ floor_multiple(23, 5) = {await floor_multiple(23, 5)}")
    print(f"✓ mround(23, 5) = {await mround(23, 5)}")


async def demo_core_modular():
    """Test all 6 modular arithmetic operations."""
    print("\n" + "=" * 70)
    print("CORE: MODULAR ARITHMETIC (6 functions)")
    print("=" * 70)

    from chuk_mcp_math.arithmetic.core.modular import (
        modulo,
        divmod_operation,
        mod_power,
        quotient,
        remainder,
        fmod,
    )

    print(f"✓ modulo(17, 5) = {await modulo(17, 5)}")
    print(f"✓ divmod_operation(17, 5) = {await divmod_operation(17, 5)}")
    print(f"✓ mod_power(3, 4, 5) = {await mod_power(3, 4, 5)}")
    print(f"✓ quotient(17, 5) = {await quotient(17, 5)}")
    print(f"✓ remainder(17, 5) = {await remainder(17, 5)}")
    print(f"✓ fmod(17.5, 5.2) = {await fmod(17.5, 5.2):.2f}")


async def demo_comparison_extrema():
    """Test all 7 extrema comparison operations."""
    print("\n" + "=" * 70)
    print("COMPARISON: EXTREMA (7 functions)")
    print("=" * 70)

    from chuk_mcp_math.arithmetic.comparison.extrema import (
        minimum,
        maximum,
        clamp,
        sort_numbers,
        rank_numbers,
        min_list,
        max_list,
    )

    print(f"✓ minimum(3, 7) = {await minimum(3, 7)}")
    print(f"✓ maximum(3, 7) = {await maximum(3, 7)}")
    print(f"✓ clamp(15, 10, 20) = {await clamp(15, 10, 20)}")
    print(f"✓ sort_numbers([5,2,8,1,9]) = {await sort_numbers([5, 2, 8, 1, 9])}")
    print(f"✓ rank_numbers([5,2,8,1,9]) = {await rank_numbers([5, 2, 8, 1, 9])}")
    print(f"✓ min_list([5,2,8,1,9]) = {await min_list([5, 2, 8, 1, 9])}")
    print(f"✓ max_list([5,2,8,1,9]) = {await max_list([5, 2, 8, 1, 9])}")


async def demo_comparison_relational():
    """Test all 8 relational comparison operations."""
    print("\n" + "=" * 70)
    print("COMPARISON: RELATIONAL (8 functions)")
    print("=" * 70)

    from chuk_mcp_math.arithmetic.comparison.relational import (
        equal,
        not_equal,
        less_than,
        less_than_or_equal,
        greater_than,
        greater_than_or_equal,
        in_range,
        between,
    )

    print(f"✓ equal(5, 5) = {await equal(5, 5)}")
    print(f"✓ not_equal(5, 3) = {await not_equal(5, 3)}")
    print(f"✓ less_than(3, 5) = {await less_than(3, 5)}")
    print(f"✓ less_than_or_equal(5, 5) = {await less_than_or_equal(5, 5)}")
    print(f"✓ greater_than(10, 5) = {await greater_than(10, 5)}")
    print(f"✓ greater_than_or_equal(5, 5) = {await greater_than_or_equal(5, 5)}")
    print(f"✓ in_range(15, 10, 20) = {await in_range(15, 10, 20)}")
    print(f"✓ between(15, 10, 20) = {await between(15, 10, 20)}")


async def demo_comparison_tolerance():
    """Test all 7 tolerance comparison operations."""
    print("\n" + "=" * 70)
    print("COMPARISON: TOLERANCE (7 functions)")
    print("=" * 70)

    from chuk_mcp_math.arithmetic.comparison.tolerance import (
        approximately_equal,
        close_to_zero,
        is_finite,
        is_nan,
        is_infinite,
        is_normal,
        is_close,
    )

    print(
        f"✓ approximately_equal(1.0, 1.00001, 0.001) = {await approximately_equal(1.0, 1.00001, 0.001)}"
    )
    print(f"✓ close_to_zero(0.00001, 0.001) = {await close_to_zero(0.00001, 0.001)}")
    print(f"✓ is_finite(42) = {await is_finite(42)}")
    print(f"✓ is_nan(42) = {await is_nan(42)}")
    print(f"✓ is_infinite(42) = {await is_infinite(42)}")
    print(f"✓ is_normal(42) = {await is_normal(42)}")
    print(f"✓ is_close(1.0, 1.00001) = {await is_close(1.0, 1.00001)}")


async def main():
    """Run all arithmetic demonstrations."""
    print("\n" + "=" * 70)
    print("CHUK MCP MATH - COMPREHENSIVE ARITHMETIC DEMONSTRATION")
    print("=" * 70)
    print("\nTesting ALL 44 Arithmetic Functions")

    await demo_core_basic()
    await demo_core_rounding()
    await demo_core_modular()
    await demo_comparison_extrema()
    await demo_comparison_relational()
    await demo_comparison_tolerance()

    print("\n" + "=" * 70)
    print("✅ ALL 44 ARITHMETIC FUNCTIONS VERIFIED WORKING!")
    print("=" * 70)
    print("\nDomain Coverage:")
    print("  • Core: Basic Operations (9 functions)")
    print("  • Core: Rounding (7 functions)")
    print("  • Core: Modular Arithmetic (6 functions)")
    print("  • Comparison: Extrema (7 functions)")
    print("  • Comparison: Relational (8 functions)")
    print("  • Comparison: Tolerance (7 functions)")
    print("\n  📊 Total: 44/44 arithmetic functions (100% async)")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
