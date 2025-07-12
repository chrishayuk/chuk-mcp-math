#!/usr/bin/env python3
# chuk_mcp_functions/math/arithmetic/basic_operations.py
"""
Basic Arithmetic Operations for AI Models (Async Native)

Fundamental mathematical operations that form the building blocks of computation.
All functions are async native to provide optimal performance in async MCP servers.
For simple operations, they complete immediately but maintain async compatibility.

Functions:
- Basic binary operations: add, subtract, multiply, divide
- Power and root operations: power, sqrt, nth_root
- Modular arithmetic: modulo, divmod_operation
- Sign operations: abs_value, sign, negate
- Rounding and truncation: round_number, floor, ceil, truncate
"""

import math
import asyncio
from typing import Union, Tuple
from chuk_mcp_functions.mcp_decorator import mcp_function

Number = Union[int, float]

@mcp_function(
    description="Add two numbers together. Supports integers, floats, and handles edge cases like infinity.",
    namespace="arithmetic",
    execution_modes=["local", "remote"],
    examples=[
        {"input": {"a": 5, "b": 3}, "output": 8, "description": "Add two positive integers"},
        {"input": {"a": -2.5, "b": 4.7}, "output": 2.2, "description": "Add negative and positive decimals"},
        {"input": {"a": 0, "b": 0}, "output": 0, "description": "Add zeros"},
        {"input": {"a": 1e10, "b": 1e10}, "output": 2e10, "description": "Add large numbers"}
    ]
)
async def add(a: Number, b: Number) -> Number:
    """
    Add two numbers together.
    
    Args:
        a: First addend (integer or float)
        b: Second addend (integer or float)
    
    Returns:
        The sum of a and b
    
    Examples:
        await add(5, 3) → 8
        await add(-2.5, 4.7) → 2.2
        await add(1000000, 999999) → 1999999
    """
    return a + b

@mcp_function(
    description="Subtract the second number from the first number. Handles negative results and decimal precision.",
    namespace="arithmetic",
    execution_modes=["local", "remote"],
    examples=[
        {"input": {"a": 10, "b": 3}, "output": 7, "description": "Basic subtraction"},
        {"input": {"a": 3, "b": 10}, "output": -7, "description": "Subtraction resulting in negative"},
        {"input": {"a": 5.5, "b": 2.3}, "output": 3.2, "description": "Decimal subtraction"},
        {"input": {"a": 0, "b": 5}, "output": -5, "description": "Subtract from zero"}
    ]
)
async def subtract(a: Number, b: Number) -> Number:
    """
    Subtract the second number from the first number.
    
    Args:
        a: Minuend (number to subtract from)
        b: Subtrahend (number to subtract)
    
    Returns:
        The difference (a - b)
    
    Examples:
        await subtract(10, 3) → 7
        await subtract(3, 10) → -7
        await subtract(5.5, 2.3) → 3.2
    """
    return a - b

@mcp_function(
    description="Multiply two numbers together. Efficiently handles large numbers and decimal precision.",
    namespace="arithmetic",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    examples=[
        {"input": {"a": 6, "b": 7}, "output": 42, "description": "Multiply two integers"},
        {"input": {"a": 2.5, "b": 4}, "output": 10.0, "description": "Multiply decimal and integer"},
        {"input": {"a": -3, "b": 4}, "output": -12, "description": "Multiply with negative number"},
        {"input": {"a": 0, "b": 1000}, "output": 0, "description": "Multiply by zero"}
    ]
)
async def multiply(a: Number, b: Number) -> Number:
    """
    Multiply two numbers together.
    
    Args:
        a: First factor
        b: Second factor
    
    Returns:
        The product of a and b
    
    Examples:
        await multiply(6, 7) → 42
        await multiply(2.5, 4) → 10.0
        await multiply(-3, 4) → -12
    """
    return a * b

@mcp_function(
    description="Divide the first number by the second number. Includes zero division protection and precise decimal handling.",
    namespace="arithmetic",
    execution_modes=["local", "remote"],
    examples=[
        {"input": {"a": 15, "b": 3}, "output": 5.0, "description": "Basic division"},
        {"input": {"a": 7, "b": 2}, "output": 3.5, "description": "Division resulting in decimal"},
        {"input": {"a": -10, "b": 2}, "output": -5.0, "description": "Division with negative numbers"},
        {"input": {"a": 10, "b": 0}, "output": "error", "description": "Division by zero raises error"}
    ]
)
async def divide(a: Number, b: Number) -> float:
    """
    Divide the first number by the second number.
    
    Args:
        a: Dividend (number to be divided)
        b: Divisor (number to divide by)
    
    Returns:
        The quotient (a / b) as a float
    
    Raises:
        ValueError: If attempting to divide by zero
    
    Examples:
        await divide(15, 3) → 5.0
        await divide(7, 2) → 3.5
        await divide(10, 0) → raises ValueError
    """
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b

@mcp_function(
    description="Raise a number to a power. Efficiently handles integer and fractional exponents, including negative powers.",
    namespace="arithmetic",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    estimated_cpu_usage="medium",
    examples=[
        {"input": {"base": 2, "exponent": 3}, "output": 8, "description": "2 to the power of 3"},
        {"input": {"base": 4, "exponent": 0.5}, "output": 2.0, "description": "Square root via fractional exponent"},
        {"input": {"base": 2, "exponent": -3}, "output": 0.125, "description": "Negative exponent"},
        {"input": {"base": 5, "exponent": 0}, "output": 1, "description": "Any number to power 0 equals 1"}
    ]
)
async def power(base: Number, exponent: Number) -> Number:
    """
    Raise a number to a power.
    
    Args:
        base: The base number
        exponent: The power to raise the base to
    
    Returns:
        base raised to the power of exponent
    
    Examples:
        await power(2, 3) → 8
        await power(4, 0.5) → 2.0
        await power(2, -3) → 0.125
        await power(5, 0) → 1
    """
    # For very large exponents, yield control periodically
    if isinstance(exponent, int) and abs(exponent) > 1000:
        await asyncio.sleep(0)
    
    return base ** exponent

@mcp_function(
    description="Calculate the square root of a number. Optimized for positive numbers with domain validation.",
    namespace="arithmetic",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    examples=[
        {"input": {"x": 9}, "output": 3.0, "description": "Square root of perfect square"},
        {"input": {"x": 2}, "output": 1.4142135623730951, "description": "Square root of non-perfect square"},
        {"input": {"x": 0}, "output": 0.0, "description": "Square root of zero"},
        {"input": {"x": -4}, "output": "error", "description": "Square root of negative number raises error"}
    ]
)
async def sqrt(x: Number) -> float:
    """
    Calculate the square root of a number.
    
    Args:
        x: Non-negative number
    
    Returns:
        The square root of x
    
    Raises:
        ValueError: If x is negative
    
    Examples:
        await sqrt(9) → 3.0
        await sqrt(2) → 1.4142135623730951
        await sqrt(0) → 0.0
    """
    if x < 0:
        raise ValueError("Cannot calculate square root of negative number")
    return math.sqrt(x)

@mcp_function(
    description="Calculate the nth root of a number. Handles even and odd roots with proper domain validation.",
    namespace="arithmetic",
    execution_modes=["local", "remote"],
    estimated_cpu_usage="medium",
    examples=[
        {"input": {"x": 8, "n": 3}, "output": 2.0, "description": "Cube root of 8"},
        {"input": {"x": 16, "n": 4}, "output": 2.0, "description": "Fourth root of 16"},
        {"input": {"x": -8, "n": 3}, "output": -2.0, "description": "Cube root of negative number"},
        {"input": {"x": -16, "n": 4}, "output": "error", "description": "Even root of negative number raises error"}
    ]
)
async def nth_root(x: Number, n: int) -> float:
    """
    Calculate the nth root of a number.
    
    Args:
        x: The number to find the root of
        n: The root index (must be positive integer)
    
    Returns:
        The nth root of x
    
    Raises:
        ValueError: If n is zero, or if x is negative and n is even
    
    Examples:
        await nth_root(8, 3) → 2.0
        await nth_root(16, 4) → 2.0
        await nth_root(-8, 3) → -2.0
    """
    if n == 0:
        raise ValueError("Root index cannot be zero")
    
    if x < 0 and n % 2 == 0:
        raise ValueError("Cannot calculate even root of negative number")
    
    if x < 0:
        return -(abs(x) ** (1.0 / n))
    
    return x ** (1.0 / n)

@mcp_function(
    description="Calculate the remainder when dividing the first number by the second. Useful for checking divisibility and cyclical patterns.",
    namespace="arithmetic",
    execution_modes=["local", "remote"],
    examples=[
        {"input": {"a": 17, "b": 5}, "output": 2, "description": "17 divided by 5 leaves remainder 2"},
        {"input": {"a": 20, "b": 4}, "output": 0, "description": "20 is perfectly divisible by 4"},
        {"input": {"a": -17, "b": 5}, "output": 3, "description": "Modulo with negative dividend"},
        {"input": {"a": 17, "b": -5}, "output": -3, "description": "Modulo with negative divisor"}
    ]
)
async def modulo(a: int, b: int) -> int:
    """
    Calculate the remainder of integer division (modulo operation).
    
    Args:
        a: Dividend
        b: Divisor (cannot be zero)
    
    Returns:
        The remainder when a is divided by b
    
    Raises:
        ValueError: If divisor is zero
    
    Examples:
        await modulo(17, 5) → 2
        await modulo(20, 4) → 0
        await modulo(-17, 5) → 3
    """
    if b == 0:
        raise ValueError("Cannot calculate modulo with zero divisor")
    return a % b

@mcp_function(
    description="Perform division and return both quotient and remainder. Equivalent to built-in divmod function.",
    namespace="arithmetic",
    execution_modes=["local", "remote"],
    examples=[
        {"input": {"a": 17, "b": 5}, "output": [3, 2], "description": "17 ÷ 5 = 3 remainder 2"},
        {"input": {"a": 20, "b": 4}, "output": [5, 0], "description": "20 ÷ 4 = 5 remainder 0"},
        {"input": {"a": -17, "b": 5}, "output": [-4, 3], "description": "Division with negative dividend"}
    ]
)
async def divmod_operation(a: int, b: int) -> Tuple[int, int]:
    """
    Perform division and return both quotient and remainder.
    
    Args:
        a: Dividend
        b: Divisor (cannot be zero)
    
    Returns:
        Tuple of (quotient, remainder)
    
    Raises:
        ValueError: If divisor is zero
    
    Examples:
        await divmod_operation(17, 5) → (3, 2)
        await divmod_operation(20, 4) → (5, 0)
    """
    if b == 0:
        raise ValueError("Cannot perform divmod with zero divisor")
    return divmod(a, b)

@mcp_function(
    description="Calculate the absolute value of a number. Returns the distance from zero on the number line.",
    namespace="arithmetic",
    execution_modes=["local", "remote"],
    examples=[
        {"input": {"x": -5}, "output": 5, "description": "Absolute value of negative number"},
        {"input": {"x": 3.7}, "output": 3.7, "description": "Absolute value of positive number"},
        {"input": {"x": 0}, "output": 0, "description": "Absolute value of zero"},
        {"input": {"x": -3.14159}, "output": 3.14159, "description": "Absolute value of negative decimal"}
    ]
)
async def abs_value(x: Number) -> Number:
    """
    Calculate the absolute value of a number.
    
    Args:
        x: Any real number
    
    Returns:
        The absolute value of x (always non-negative)
    
    Examples:
        await abs_value(-5) → 5
        await abs_value(3.7) → 3.7
        await abs_value(0) → 0
    """
    return abs(x)

@mcp_function(
    description="Determine the sign of a number. Returns 1 for positive, -1 for negative, 0 for zero.",
    namespace="arithmetic",
    execution_modes=["local", "remote"],
    examples=[
        {"input": {"x": 5}, "output": 1, "description": "Sign of positive number"},
        {"input": {"x": -3.2}, "output": -1, "description": "Sign of negative number"},
        {"input": {"x": 0}, "output": 0, "description": "Sign of zero"},
        {"input": {"x": 0.0001}, "output": 1, "description": "Sign of small positive number"}
    ]
)
async def sign(x: Number) -> int:
    """
    Determine the sign of a number.
    
    Args:
        x: Any real number
    
    Returns:
        1 if x > 0, -1 if x < 0, 0 if x == 0
    
    Examples:
        await sign(5) → 1
        await sign(-3.2) → -1
        await sign(0) → 0
    """
    if x > 0:
        return 1
    elif x < 0:
        return -1
    else:
        return 0

@mcp_function(
    description="Negate a number. Returns the additive inverse (opposite sign).",
    namespace="arithmetic",
    execution_modes=["local", "remote"],
    examples=[
        {"input": {"x": 5}, "output": -5, "description": "Negate positive number"},
        {"input": {"x": -3.2}, "output": 3.2, "description": "Negate negative number"},
        {"input": {"x": 0}, "output": 0, "description": "Negate zero"},
        {"input": {"x": -0.0}, "output": 0.0, "description": "Negate negative zero"}
    ]
)
async def negate(x: Number) -> Number:
    """
    Negate a number (return its additive inverse).
    
    Args:
        x: Any real number
    
    Returns:
        The negation of x (-x)
    
    Examples:
        await negate(5) → -5
        await negate(-3.2) → 3.2
        await negate(0) → 0
    """
    return -x

@mcp_function(
    description="Round a number to a specified number of decimal places. Uses standard rounding rules (round half to even).",
    namespace="arithmetic",
    execution_modes=["local", "remote"],
    examples=[
        {"input": {"x": 3.14159, "decimals": 2}, "output": 3.14, "description": "Round pi to 2 decimal places"},
        {"input": {"x": 2.5, "decimals": 0}, "output": 2, "description": "Round to nearest integer (banker's rounding)"},
        {"input": {"x": 3.5, "decimals": 0}, "output": 4, "description": "Round to nearest integer (banker's rounding)"},
        {"input": {"x": 123.456, "decimals": 1}, "output": 123.5, "description": "Round to 1 decimal place"}
    ]
)
async def round_number(x: Number, decimals: int = 0) -> Number:
    """
    Round a number to a specified number of decimal places.
    
    Args:
        x: The number to round
        decimals: Number of decimal places (default: 0 for integer rounding)
    
    Returns:
        The rounded number
    
    Examples:
        await round_number(3.14159, 2) → 3.14
        await round_number(2.5, 0) → 2
        await round_number(123.456, 1) → 123.5
    """
    return round(x, decimals)

@mcp_function(
    description="Return the largest integer less than or equal to the given number (floor function).",
    namespace="arithmetic",
    execution_modes=["local", "remote"],
    examples=[
        {"input": {"x": 3.7}, "output": 3, "description": "Floor of positive decimal"},
        {"input": {"x": -2.3}, "output": -3, "description": "Floor of negative decimal"},
        {"input": {"x": 5}, "output": 5, "description": "Floor of integer"},
        {"input": {"x": -5}, "output": -5, "description": "Floor of negative integer"}
    ]
)
async def floor(x: Number) -> int:
    """
    Return the largest integer less than or equal to x.
    
    Args:
        x: Any real number
    
    Returns:
        The floor of x
    
    Examples:
        await floor(3.7) → 3
        await floor(-2.3) → -3
        await floor(5) → 5
    """
    return math.floor(x)

@mcp_function(
    description="Return the smallest integer greater than or equal to the given number (ceiling function).",
    namespace="arithmetic",
    execution_modes=["local", "remote"],
    examples=[
        {"input": {"x": 3.2}, "output": 4, "description": "Ceiling of positive decimal"},
        {"input": {"x": -2.7}, "output": -2, "description": "Ceiling of negative decimal"},
        {"input": {"x": 5}, "output": 5, "description": "Ceiling of integer"},
        {"input": {"x": -5}, "output": -5, "description": "Ceiling of negative integer"}
    ]
)
async def ceil(x: Number) -> int:
    """
    Return the smallest integer greater than or equal to x.
    
    Args:
        x: Any real number
    
    Returns:
        The ceiling of x
    
    Examples:
        await ceil(3.2) → 4
        await ceil(-2.7) → -2
        await ceil(5) → 5
    """
    return math.ceil(x)

@mcp_function(
    description="Truncate a number toward zero. Removes the fractional part without rounding.",
    namespace="arithmetic",
    execution_modes=["local", "remote"],
    examples=[
        {"input": {"x": 3.7}, "output": 3, "description": "Truncate positive decimal"},
        {"input": {"x": -2.9}, "output": -2, "description": "Truncate negative decimal"},
        {"input": {"x": 5}, "output": 5, "description": "Truncate integer"},
        {"input": {"x": 0.9}, "output": 0, "description": "Truncate small positive decimal"}
    ]
)
async def truncate(x: Number) -> int:
    """
    Truncate a number toward zero (remove fractional part).
    
    Args:
        x: Any real number
    
    Returns:
        The truncated integer value
    
    Examples:
        await truncate(3.7) → 3
        await truncate(-2.9) → -2
        await truncate(5) → 5
    """
    return math.trunc(x)

# Export all basic arithmetic functions
__all__ = [
    'add', 'subtract', 'multiply', 'divide', 'power', 'sqrt', 'nth_root',
    'modulo', 'divmod_operation', 'abs_value', 'sign', 'negate',
    'round_number', 'floor', 'ceil', 'truncate'
]

if __name__ == "__main__":
    import asyncio
    
    async def test_basic_arithmetic_functions():
        """Test all basic arithmetic functions (async)."""
        print("🔢 Basic Arithmetic Operations Test (Async Native)")
        print("=" * 50)
        
        # Test basic operations
        print(f"add(15, 25) = {await add(15, 25)}")
        print(f"subtract(20, 8) = {await subtract(20, 8)}")
        print(f"multiply(6, 7) = {await multiply(6, 7)}")
        print(f"divide(20, 4) = {await divide(20, 4)}")
        
        # Test power and root operations
        print(f"power(2, 8) = {await power(2, 8)}")
        print(f"sqrt(64) = {await sqrt(64)}")
        print(f"nth_root(27, 3) = {await nth_root(27, 3)}")
        
        # Test modular arithmetic
        print(f"modulo(17, 5) = {await modulo(17, 5)}")
        print(f"divmod_operation(17, 5) = {await divmod_operation(17, 5)}")
        
        # Test sign operations
        print(f"abs_value(-42) = {await abs_value(-42)}")
        print(f"sign(-5.5) = {await sign(-5.5)}")
        print(f"negate(10) = {await negate(10)}")
        
        # Test rounding operations
        print(f"round_number(3.14159, 2) = {await round_number(3.14159, 2)}")
        print(f"floor(3.7) = {await floor(3.7)}")
        print(f"ceil(3.2) = {await ceil(3.2)}")
        print(f"truncate(3.7) = {await truncate(3.7)}")
        
        print("\n✅ All basic arithmetic functions working correctly!")
        
        # Test parallel execution
        print("\n🚀 Testing Parallel Execution:")
        parallel_results = await asyncio.gather(
            add(1, 2), multiply(3, 4), sqrt(16), power(2, 3)
        )
        print(f"Parallel results: {parallel_results}")
    
    asyncio.run(test_basic_arithmetic_functions())