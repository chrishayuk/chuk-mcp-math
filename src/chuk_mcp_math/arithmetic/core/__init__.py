#!/usr/bin/env python3
# chuk_mcp_math/arithmetic/core/__init__.py
"""
Core Arithmetic Operations Module

Fundamental arithmetic operations that form the foundation of all mathematical computation.
This module contains the most essential arithmetic functions organized into logical groups.

Submodules:
- basic_operations: add, subtract, multiply, divide, power, sqrt, abs_value, sign, negate
- rounding: round_number, floor, ceil, truncate, mround, ceiling_multiple, floor_multiple
- modular: modulo, divmod_operation, mod_power, quotient, remainder, fmod

All functions are async native for optimal performance in async environments.
"""

# Import all core submodules
from . import basic_operations
from . import rounding
from . import modular

# Import all functions for direct access
from .basic_operations import (
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

from .rounding import (
    round_number,
    floor,
    ceil,
    truncate,
    mround,
    ceiling_multiple,
    floor_multiple,
)

from .modular import modulo, divmod_operation, mod_power, quotient, remainder, fmod

# Export all core functions
__all__ = [
    # Submodules
    "basic_operations",
    "rounding",
    "modular",
    # Basic operations
    "add",
    "subtract",
    "multiply",
    "divide",
    "power",
    "sqrt",
    "abs_value",
    "sign",
    "negate",
    # Rounding operations
    "round_number",
    "floor",
    "ceil",
    "truncate",
    "mround",
    "ceiling_multiple",
    "floor_multiple",
    # Modular operations
    "modulo",
    "divmod_operation",
    "mod_power",
    "quotient",
    "remainder",
    "fmod",
]


async def test_core_functions():
    """Test all core arithmetic functions."""
    print("🔢 Core Arithmetic Functions Test")
    print("=" * 35)

    # Test basic operations
    print("Basic Operations:")
    print(f"  add(5, 3) = {await add(5, 3)}")
    print(f"  subtract(10, 4) = {await subtract(10, 4)}")
    print(f"  multiply(6, 7) = {await multiply(6, 7)}")
    print(f"  divide(20, 4) = {await divide(20, 4)}")
    print(f"  power(2, 3) = {await power(2, 3)}")
    print(f"  sqrt(16) = {await sqrt(16)}")
    print(f"  abs_value(-5) = {await abs_value(-5)}")

    # Test rounding operations
    print("\nRounding Operations:")
    print(f"  round_number(3.14159, 2) = {await round_number(3.14159, 2)}")
    print(f"  floor(3.7) = {await floor(3.7)}")
    print(f"  ceil(3.2) = {await ceil(3.2)}")
    print(f"  mround(7.3, 2) = {await mround(7.3, 2)}")

    # Test modular operations
    print("\nModular Operations:")
    print(f"  modulo(17, 5) = {await modulo(17, 5)}")
    print(f"  quotient(17, 5) = {await quotient(17, 5)}")
    print(f"  mod_power(2, 10, 1000) = {await mod_power(2, 10, 1000)}")

    print("\n✅ All core arithmetic functions working!")


if __name__ == "__main__":
    import asyncio

    asyncio.run(test_core_functions())
