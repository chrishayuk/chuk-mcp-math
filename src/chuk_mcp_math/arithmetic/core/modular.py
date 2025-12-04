#!/usr/bin/env python3
# chuk_mcp_math/arithmetic/core/modular.py
"""
Core Modular Arithmetic Operations - Async Native

Mathematical functions for modular arithmetic and remainder operations.

Functions:
- modulo, divmod_operation, mod_power
- remainder, fmod, quotient
"""

import math
import asyncio
from typing import Union, Tuple
from chuk_mcp_math.mcp_decorator import mcp_function

Number = Union[int, float]


@mcp_function(
    description="Calculate the modulo (remainder) of division. Returns a % b.",
    namespace="arithmetic",
    category="core",
    execution_modes=["local", "remote"],
    examples=[
        {"input": {"a": 17, "b": 5}, "output": 2, "description": "17 mod 5 = 2"},
        {"input": {"a": 10, "b": 3}, "output": 1, "description": "10 mod 3 = 1"},
        {"input": {"a": -7, "b": 3}, "output": 2, "description": "Negative dividend"},
        {"input": {"a": 15, "b": 5}, "output": 0, "description": "No remainder"},
    ],
)
async def modulo(a: Number, b: Number) -> Number:
    """
    Calculate the modulo (remainder) of division.

    Args:
        a: Dividend
        b: Divisor (cannot be zero)

    Returns:
        The remainder when a is divided by b

    Raises:
        ValueError: If b is zero

    Examples:
        await modulo(17, 5) → 2
        await modulo(10, 3) → 1
        await modulo(-7, 3) → 2
    """
    if b == 0:
        raise ValueError("Cannot calculate modulo with zero divisor")
    return a % b


@mcp_function(
    description="Perform division and return both quotient and remainder. Returns (quotient, remainder).",
    namespace="arithmetic",
    category="core",
    execution_modes=["local", "remote"],
    examples=[
        {
            "input": {"a": 17, "b": 5},
            "output": [3, 2],
            "description": "17 ÷ 5 = 3 remainder 2",
        },
        {
            "input": {"a": 20, "b": 4},
            "output": [5, 0],
            "description": "20 ÷ 4 = 5 remainder 0",
        },
        {
            "input": {"a": -17, "b": 5},
            "output": [-4, 3],
            "description": "Negative dividend",
        },
        {
            "input": {"a": 7, "b": 3},
            "output": [2, 1],
            "description": "7 ÷ 3 = 2 remainder 1",
        },
    ],
)
async def divmod_operation(a: Number, b: Number) -> Tuple[Number, Number]:
    """
    Perform division and return both quotient and remainder.

    Args:
        a: Dividend
        b: Divisor (cannot be zero)

    Returns:
        Tuple of (quotient, remainder)

    Raises:
        ValueError: If b is zero

    Examples:
        await divmod_operation(17, 5) → (3, 2)
        await divmod_operation(20, 4) → (5, 0)
        await divmod_operation(-17, 5) → (-4, 3)
    """
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return divmod(a, b)


@mcp_function(
    description="Calculate modular exponentiation: (base^exponent) mod modulus. Efficient for large numbers.",
    namespace="arithmetic",
    category="core",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    estimated_cpu_usage="medium",
    examples=[
        {
            "input": {"base": 2, "exponent": 10, "modulus": 1000},
            "output": 24,
            "description": "2^10 mod 1000 = 24",
        },
        {
            "input": {"base": 3, "exponent": 5, "modulus": 7},
            "output": 5,
            "description": "3^5 mod 7 = 5",
        },
        {
            "input": {"base": 5, "exponent": 0, "modulus": 13},
            "output": 1,
            "description": "Any number^0 mod m = 1",
        },
        {
            "input": {"base": 7, "exponent": 3, "modulus": 10},
            "output": 3,
            "description": "7^3 mod 10 = 3",
        },
    ],
)
async def mod_power(base: int, exponent: int, modulus: int) -> int:
    """
    Calculate modular exponentiation efficiently.

    Args:
        base: Base number
        exponent: Non-negative exponent
        modulus: Positive modulus

    Returns:
        (base^exponent) mod modulus

    Raises:
        ValueError: If exponent is negative or modulus is not positive

    Examples:
        await mod_power(2, 10, 1000) → 24
        await mod_power(3, 5, 7) → 5
    """
    if exponent < 0:
        raise ValueError("Exponent must be non-negative")
    if modulus <= 0:
        raise ValueError("Modulus must be positive")

    # Yield control for large exponents
    if exponent > 1000:
        await asyncio.sleep(0)

    return pow(base, exponent, modulus)


@mcp_function(
    description="Return the integer quotient of division without remainder. Equivalent to a // b.",
    namespace="arithmetic",
    category="core",
    execution_modes=["local", "remote"],
    examples=[
        {
            "input": {"dividend": 17, "divisor": 5},
            "output": 3,
            "description": "17 ÷ 5 = 3 remainder 2",
        },
        {
            "input": {"dividend": 20, "divisor": 4},
            "output": 5,
            "description": "20 ÷ 4 = 5 remainder 0",
        },
        {
            "input": {"dividend": -17, "divisor": 5},
            "output": -4,
            "description": "Negative dividend",
        },
        {
            "input": {"dividend": 7, "divisor": 3},
            "output": 2,
            "description": "7 ÷ 3 = 2 remainder 1",
        },
    ],
)
async def quotient(dividend: int, divisor: int) -> int:
    """
    Return the integer quotient of division.

    Args:
        dividend: Number to be divided
        divisor: Number to divide by (cannot be zero)

    Returns:
        The integer quotient (dividend // divisor)

    Raises:
        ValueError: If divisor is zero

    Examples:
        await quotient(17, 5) → 3
        await quotient(20, 4) → 5
        await quotient(-17, 5) → -4
    """
    if divisor == 0:
        raise ValueError("Cannot divide by zero")
    return dividend // divisor


@mcp_function(
    description="Calculate floating-point remainder of division (IEEE remainder operation).",
    namespace="arithmetic",
    category="core",
    execution_modes=["local", "remote"],
    examples=[
        {
            "input": {"x": 7.5, "y": 2.5},
            "output": 0.0,
            "description": "IEEE remainder of 7.5 / 2.5",
        },
        {
            "input": {"x": 10.3, "y": 3.0},
            "output": 1.3,
            "description": "IEEE remainder with decimals",
        },
        {
            "input": {"x": -7.5, "y": 2.5},
            "output": 0.0,
            "description": "Negative dividend",
        },
        {
            "input": {"x": 5.0, "y": 2.0},
            "output": 1.0,
            "description": "Simple remainder",
        },
    ],
)
async def remainder(x: float, y: float) -> float:
    """
    Calculate the IEEE remainder of x with respect to y.

    Args:
        x: Dividend
        y: Divisor (cannot be zero)

    Returns:
        IEEE remainder of x / y

    Raises:
        ValueError: If y is zero

    Examples:
        await remainder(7.5, 2.5) → 0.0
        await remainder(10.3, 3.0) → 1.3
        await remainder(-7.5, 2.5) → 0.0
    """
    if y == 0:
        raise ValueError("Cannot calculate remainder with zero divisor")
    return math.remainder(x, y)


@mcp_function(
    description="Calculate floating-point modulo operation. Similar to % but for floats.",
    namespace="arithmetic",
    category="core",
    execution_modes=["local", "remote"],
    examples=[
        {"input": {"x": 7.5, "y": 2.5}, "output": 0.0, "description": "7.5 fmod 2.5"},
        {
            "input": {"x": 10.3, "y": 3.0},
            "output": 1.2999999999999998,
            "description": "Floating point precision",
        },
        {
            "input": {"x": -7.5, "y": 2.5},
            "output": -0.0,
            "description": "Negative dividend",
        },
        {
            "input": {"x": 5.7, "y": 2.0},
            "output": 1.7000000000000002,
            "description": "Float modulo",
        },
    ],
)
async def fmod(x: float, y: float) -> float:
    """
    Calculate the floating-point remainder of x / y.

    Args:
        x: Dividend
        y: Divisor (cannot be zero)

    Returns:
        Floating-point remainder of x / y

    Raises:
        ValueError: If y is zero

    Examples:
        await fmod(7.5, 2.5) → 0.0
        await fmod(10.3, 3.0) → 1.2999999999999998
        await fmod(-7.5, 2.5) → -0.0
    """
    if y == 0:
        raise ValueError("Cannot calculate fmod with zero divisor")
    return math.fmod(x, y)


# Export all modular arithmetic functions
__all__ = ["modulo", "divmod_operation", "mod_power", "quotient", "remainder", "fmod"]

if __name__ == "__main__":
    import asyncio

    async def test_modular_operations():
        """Test all modular arithmetic operations."""
        print("🔢 Modular Arithmetic Operations Test")
        print("=" * 40)

        # Test basic modular operations
        print(f"modulo(17, 5) = {await modulo(17, 5)}")
        print(f"divmod_operation(17, 5) = {await divmod_operation(17, 5)}")
        print(f"quotient(17, 5) = {await quotient(17, 5)}")

        # Test modular exponentiation
        print(f"mod_power(2, 10, 1000) = {await mod_power(2, 10, 1000)}")

        # Test floating point operations
        print(f"remainder(7.5, 2.5) = {await remainder(7.5, 2.5)}")
        print(f"fmod(7.5, 2.5) = {await fmod(7.5, 2.5)}")

        print("\n✅ All modular operations working!")

    asyncio.run(test_modular_operations())
