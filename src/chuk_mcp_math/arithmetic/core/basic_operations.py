#!/usr/bin/env python3
# chuk_mcp_math/arithmetic/core/basic_operations.py
"""
Core Basic Arithmetic Operations - Async Native

Fundamental arithmetic operations that form the foundation of all mathematical computation.
Clean, focused module containing only the essential arithmetic operations.

Functions:
- add, subtract, multiply, divide
- power, sqrt, abs_value, sign, negate
"""

import math
import asyncio
from typing import Union
from chuk_mcp_math.mcp_decorator import mcp_function

Number = Union[int, float]


@mcp_function(
    description="Add two numbers together. Pure arithmetic addition operation.",
    namespace="arithmetic",
    category="core",
    execution_modes=["local", "remote"],
    examples=[
        {"input": {"a": 5, "b": 3}, "output": 8, "description": "Basic addition"},
        {
            "input": {"a": -2.5, "b": 4.7},
            "output": 2.2,
            "description": "Decimal addition",
        },
        {
            "input": {"a": 1e10, "b": 1e10},
            "output": 2e10,
            "description": "Large numbers",
        },
    ],
)
async def add(a: Number, b: Number) -> Number:
    """Add two numbers together.

    Args:
        a: First number (int or float)
        b: Second number (int or float)

    Returns:
        Sum of a + b (same type as inputs)

    Example:
        add(5, 3) = 8
        add(2.5, 1.3) = 3.8
    """
    return a + b


@mcp_function(
    description="Subtract the second number from the first number.",
    namespace="arithmetic",
    category="core",
    execution_modes=["local", "remote"],
    examples=[
        {"input": {"a": 10, "b": 3}, "output": 7, "description": "Basic subtraction"},
        {"input": {"a": 3, "b": 10}, "output": -7, "description": "Negative result"},
        {
            "input": {"a": 5.5, "b": 2.3},
            "output": 3.2,
            "description": "Decimal subtraction",
        },
    ],
)
async def subtract(a: Number, b: Number) -> Number:
    """Subtract the second number from the first.

    Args:
        a: Number to subtract from (int or float)
        b: Number to subtract (int or float)

    Returns:
        Difference a - b

    Example:
        subtract(10, 3) = 7
        subtract(5.5, 2.3) = 3.2
    """
    return a - b


@mcp_function(
    description="Multiply two numbers together.",
    namespace="arithmetic",
    category="core",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    examples=[
        {
            "input": {"a": 6, "b": 7},
            "output": 42,
            "description": "Integer multiplication",
        },
        {
            "input": {"a": 2.5, "b": 4},
            "output": 10.0,
            "description": "Mixed decimal/integer",
        },
        {
            "input": {"a": -3, "b": 4},
            "output": -12,
            "description": "Negative multiplication",
        },
    ],
)
async def multiply(a: Number, b: Number) -> Number:
    """Multiply two numbers together.

    Args:
        a: First factor (int or float)
        b: Second factor (int or float)

    Returns:
        Product a * b

    Example:
        multiply(6, 7) = 42
        multiply(2.5, 4) = 10.0
    """
    return a * b


@mcp_function(
    description="Divide the first number by the second number.",
    namespace="arithmetic",
    category="core",
    execution_modes=["local", "remote"],
    examples=[
        {"input": {"a": 15, "b": 3}, "output": 5.0, "description": "Clean division"},
        {"input": {"a": 7, "b": 2}, "output": 3.5, "description": "Decimal result"},
        {
            "input": {"a": -10, "b": 2},
            "output": -5.0,
            "description": "Negative division",
        },
    ],
)
async def divide(a: Number, b: Number) -> float:
    """Divide the first number by the second.

    Args:
        a: Dividend (int or float)
        b: Divisor (int or float, must be non-zero)

    Returns:
        Quotient a / b as float

    Raises:
        ValueError: If b is zero

    Example:
        divide(15, 3) = 5.0
        divide(7, 2) = 3.5
    """
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b


@mcp_function(
    description="Raise a number to a power. Handles integer and fractional exponents.",
    namespace="arithmetic",
    category="core",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    estimated_cpu_usage="medium",
    examples=[
        {
            "input": {"base": 2, "exponent": 3},
            "output": 8,
            "description": "Integer power",
        },
        {
            "input": {"base": 4, "exponent": 0.5},
            "output": 2.0,
            "description": "Fractional power (square root)",
        },
        {
            "input": {"base": 2, "exponent": -3},
            "output": 0.125,
            "description": "Negative exponent",
        },
    ],
)
async def power(base: Number, exponent: Number) -> Number:
    """Raise a number to a power.

    Handles integer and fractional exponents including negative values.

    Args:
        base: Base number (int or float)
        exponent: Power to raise to (int or float)

    Returns:
        Result of base^exponent

    Example:
        power(2, 3) = 8
        power(4, 0.5) = 2.0  # square root
        power(2, -3) = 0.125
    """
    # Yield control for very large exponents
    if isinstance(exponent, int) and abs(exponent) > 1000:
        await asyncio.sleep(0)

    return base**exponent


@mcp_function(
    description="Calculate the square root of a number.",
    namespace="arithmetic",
    category="core",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    examples=[
        {"input": {"x": 9}, "output": 3.0, "description": "Perfect square"},
        {
            "input": {"x": 2},
            "output": 1.4142135623730951,
            "description": "Irrational result",
        },
        {"input": {"x": 0}, "output": 0.0, "description": "Square root of zero"},
    ],
)
async def sqrt(x: Number) -> float:
    """Calculate the square root of a number.

    Args:
        x: Non-negative number (int or float)

    Returns:
        Square root √x as float

    Raises:
        ValueError: If x is negative

    Example:
        sqrt(9) = 3.0
        sqrt(2) ≈ 1.414
    """
    if x < 0:
        raise ValueError("Cannot calculate square root of negative number")
    return math.sqrt(x)


@mcp_function(
    description="Calculate the absolute value of a number.",
    namespace="arithmetic",
    category="core",
    execution_modes=["local", "remote"],
    examples=[
        {"input": {"x": -5}, "output": 5, "description": "Negative to positive"},
        {"input": {"x": 3.7}, "output": 3.7, "description": "Positive unchanged"},
        {"input": {"x": 0}, "output": 0, "description": "Zero unchanged"},
    ],
)
async def abs_value(x: Number) -> Number:
    """Calculate the absolute value of a number.

    Args:
        x: Any number (int or float)

    Returns:
        Absolute value |x| (always non-negative)

    Example:
        abs_value(-5) = 5
        abs_value(3.7) = 3.7
    """
    return abs(x)


@mcp_function(
    description="Determine the sign of a number. Returns 1, -1, or 0.",
    namespace="arithmetic",
    category="core",
    execution_modes=["local", "remote"],
    examples=[
        {"input": {"x": 5}, "output": 1, "description": "Positive number"},
        {"input": {"x": -3.2}, "output": -1, "description": "Negative number"},
        {"input": {"x": 0}, "output": 0, "description": "Zero"},
    ],
)
async def sign(x: Number) -> int:
    """Determine the sign of a number.

    Args:
        x: Any number (int or float)

    Returns:
        1 if x > 0, -1 if x < 0, 0 if x = 0

    Example:
        sign(5) = 1
        sign(-3.2) = -1
        sign(0) = 0
    """
    if x > 0:
        return 1
    elif x < 0:
        return -1
    else:
        return 0


@mcp_function(
    description="Negate a number (return its additive inverse).",
    namespace="arithmetic",
    category="core",
    execution_modes=["local", "remote"],
    examples=[
        {"input": {"x": 5}, "output": -5, "description": "Positive to negative"},
        {"input": {"x": -3.2}, "output": 3.2, "description": "Negative to positive"},
        {"input": {"x": 0}, "output": 0, "description": "Zero unchanged"},
    ],
)
async def negate(x: Number) -> Number:
    """Negate a number (return its additive inverse).

    Args:
        x: Any number (int or float)

    Returns:
        Negated value -x

    Example:
        negate(5) = -5
        negate(-3.2) = 3.2
        negate(0) = 0
    """
    return -x


# Export all core basic operations
__all__ = [
    "add",
    "subtract",
    "multiply",
    "divide",
    "power",
    "sqrt",
    "abs_value",
    "sign",
    "negate",
]

if __name__ == "__main__":
    import asyncio

    async def test_core_operations():
        """Test all core basic operations."""
        print("🔢 Core Basic Operations Test")
        print("=" * 30)

        # Test basic arithmetic
        print(f"add(5, 3) = {await add(5, 3)}")
        print(f"subtract(10, 4) = {await subtract(10, 4)}")
        print(f"multiply(6, 7) = {await multiply(6, 7)}")
        print(f"divide(20, 4) = {await divide(20, 4)}")

        # Test power and roots
        print(f"power(2, 3) = {await power(2, 3)}")
        print(f"sqrt(16) = {await sqrt(16)}")

        # Test sign operations
        print(f"abs_value(-5) = {await abs_value(-5)}")
        print(f"sign(-3.2) = {await sign(-3.2)}")
        print(f"negate(10) = {await negate(10)}")

        print("\n✅ All core operations working!")

    asyncio.run(test_core_operations())
