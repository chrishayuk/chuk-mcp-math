#!/usr/bin/env python3
# chuk_mcp_math/arithmetic/advanced_operations.py
"""
Advanced Arithmetic Operations for AI Models (Async Native)

Extended arithmetic functions including logarithms, exponentials, advanced rounding,
number base conversions, and specialized mathematical operations commonly found
in spreadsheet applications and mathematical libraries. All functions are async
native for optimal performance in async environments.

Functions:
- Logarithmic: log, log10, log2, ln, exp
- Advanced rounding: ceiling_multiple, floor_multiple, mround
- Base conversions: decimal_to_base, base_to_decimal
- Special operations: quotient, double_factorial, multinomial
- Array operations: sum_product, sum_squares, matrix_determinant
- Random numbers: random_float, random_int, random_array
"""

import math
import random
import asyncio
from typing import Union, List
from chuk_mcp_math.mcp_decorator import mcp_function

Number = Union[int, float]


@mcp_function(
    description="Calculate the natural logarithm (base e) of a number. Returns ln(x) where e^ln(x) = x.",
    namespace="arithmetic",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    examples=[
        {"input": {"x": 2.718281828459045}, "output": 1.0, "description": "ln(e) = 1"},
        {"input": {"x": 1}, "output": 0.0, "description": "ln(1) = 0"},
        {"input": {"x": 10}, "output": 2.302585092994046, "description": "ln(10)"},
        {
            "input": {"x": 0.5},
            "output": -0.6931471805599453,
            "description": "ln(0.5) = -ln(2)",
        },
    ],
)
async def ln(x: Number) -> float:
    """
    Calculate the natural logarithm of a number.

    Args:
        x: Positive number

    Returns:
        The natural logarithm of x

    Raises:
        ValueError: If x is not positive

    Examples:
        await ln(await e()) → 1.0
        await ln(1) → 0.0
        await ln(10) → 2.302585092994046
    """
    if x <= 0:
        raise ValueError("Logarithm undefined for non-positive numbers")
    return math.log(x)


@mcp_function(
    description="Calculate the logarithm of a number to a specified base. Returns log_base(x).",
    namespace="arithmetic",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    examples=[
        {"input": {"x": 8, "base": 2}, "output": 3.0, "description": "log₂(8) = 3"},
        {
            "input": {"x": 1000, "base": 10},
            "output": 3.0,
            "description": "log₁₀(1000) = 3",
        },
        {"input": {"x": 27, "base": 3}, "output": 3.0, "description": "log₃(27) = 3"},
        {"input": {"x": 16, "base": 4}, "output": 2.0, "description": "log₄(16) = 2"},
    ],
)
async def log(x: Number, base: Number) -> float:
    """
    Calculate the logarithm of a number to a specified base.

    Args:
        x: Positive number
        base: Positive base (not equal to 1)

    Returns:
        The logarithm of x to the given base

    Raises:
        ValueError: If x or base is not positive, or if base equals 1

    Examples:
        await log(8, 2) → 3.0
        await log(1000, 10) → 3.0
        await log(27, 3) → 3.0
    """
    if x <= 0:
        raise ValueError("Logarithm undefined for non-positive numbers")
    if base <= 0 or base == 1:
        raise ValueError("Base must be positive and not equal to 1")

    return math.log(x, base)


@mcp_function(
    description="Calculate the base-10 (common) logarithm of a number. Returns log₁₀(x).",
    namespace="arithmetic",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    examples=[
        {"input": {"x": 100}, "output": 2.0, "description": "log₁₀(100) = 2"},
        {"input": {"x": 1000}, "output": 3.0, "description": "log₁₀(1000) = 3"},
        {"input": {"x": 1}, "output": 0.0, "description": "log₁₀(1) = 0"},
        {"input": {"x": 0.1}, "output": -1.0, "description": "log₁₀(0.1) = -1"},
    ],
)
async def log10(x: Number) -> float:
    """
    Calculate the base-10 logarithm of a number.

    Args:
        x: Positive number

    Returns:
        The base-10 logarithm of x

    Raises:
        ValueError: If x is not positive

    Examples:
        await log10(100) → 2.0
        await log10(1000) → 3.0
        await log10(1) → 0.0
    """
    if x <= 0:
        raise ValueError("Logarithm undefined for non-positive numbers")
    return math.log10(x)


@mcp_function(
    description="Calculate the base-2 (binary) logarithm of a number. Returns log₂(x).",
    namespace="arithmetic",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    examples=[
        {"input": {"x": 8}, "output": 3.0, "description": "log₂(8) = 3"},
        {"input": {"x": 1024}, "output": 10.0, "description": "log₂(1024) = 10"},
        {"input": {"x": 1}, "output": 0.0, "description": "log₂(1) = 0"},
        {"input": {"x": 0.5}, "output": -1.0, "description": "log₂(0.5) = -1"},
    ],
)
async def log2(x: Number) -> float:
    """
    Calculate the base-2 logarithm of a number.

    Args:
        x: Positive number

    Returns:
        The base-2 logarithm of x

    Raises:
        ValueError: If x is not positive

    Examples:
        await log2(8) → 3.0
        await log2(1024) → 10.0
        await log2(1) → 0.0
    """
    if x <= 0:
        raise ValueError("Logarithm undefined for non-positive numbers")
    return math.log2(x)


@mcp_function(
    description="Calculate e raised to the power of x. Returns e^x, the exponential function.",
    namespace="arithmetic",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    examples=[
        {"input": {"x": 0}, "output": 1.0, "description": "e⁰ = 1"},
        {"input": {"x": 1}, "output": 2.718281828459045, "description": "e¹ = e"},
        {"input": {"x": 2}, "output": 7.38905609893065, "description": "e² ≈ 7.389"},
        {"input": {"x": -1}, "output": 0.36787944117144233, "description": "e⁻¹ = 1/e"},
    ],
)
async def exp(x: Number) -> float:
    """
    Calculate e raised to the power of x.

    Args:
        x: Any real number

    Returns:
        e^x (e raised to the power x)

    Examples:
        await exp(0) → 1.0
        await exp(1) → 2.718281828459045
        await exp(2) → 7.38905609893065
    """
    return math.exp(x)


@mcp_function(
    description="Round a number up to the nearest multiple of significance. Always rounds away from zero.",
    namespace="arithmetic",
    execution_modes=["local", "remote"],
    examples=[
        {
            "input": {"number": 2.5, "significance": 1},
            "output": 3,
            "description": "Round 2.5 up to nearest 1",
        },
        {
            "input": {"number": 6.7, "significance": 2},
            "output": 8,
            "description": "Round 6.7 up to nearest 2",
        },
        {
            "input": {"number": -2.1, "significance": 1},
            "output": -3,
            "description": "Round -2.1 away from zero",
        },
        {
            "input": {"number": 15, "significance": 10},
            "output": 20,
            "description": "Round 15 up to nearest 10",
        },
    ],
)
async def ceiling_multiple(number: Number, significance: Number) -> Number:
    """
    Round a number up to the nearest multiple of significance.

    Args:
        number: Number to round
        significance: Multiple to round to (must be positive)

    Returns:
        Number rounded up to nearest multiple of significance

    Raises:
        ValueError: If significance is zero or negative

    Examples:
        await ceiling_multiple(2.5, 1) → 3
        await ceiling_multiple(6.7, 2) → 8
        await ceiling_multiple(-2.1, 1) → -3
    """
    if significance <= 0:
        raise ValueError("Significance must be positive")

    if number >= 0:
        return math.ceil(number / significance) * significance
    else:
        return math.floor(number / significance) * significance


@mcp_function(
    description="Round a number down to the nearest multiple of significance. Always rounds toward zero.",
    namespace="arithmetic",
    execution_modes=["local", "remote"],
    examples=[
        {
            "input": {"number": 2.9, "significance": 1},
            "output": 2,
            "description": "Round 2.9 down to nearest 1",
        },
        {
            "input": {"number": 7.8, "significance": 2},
            "output": 6,
            "description": "Round 7.8 down to nearest 2",
        },
        {
            "input": {"number": -2.9, "significance": 1},
            "output": -2,
            "description": "Round -2.9 toward zero",
        },
        {
            "input": {"number": 23, "significance": 10},
            "output": 20,
            "description": "Round 23 down to nearest 10",
        },
    ],
)
async def floor_multiple(number: Number, significance: Number) -> Number:
    """
    Round a number down to the nearest multiple of significance.

    Args:
        number: Number to round
        significance: Multiple to round to (must be positive)

    Returns:
        Number rounded down to nearest multiple of significance

    Raises:
        ValueError: If significance is zero or negative

    Examples:
        await floor_multiple(2.9, 1) → 2
        await floor_multiple(7.8, 2) → 6
        await floor_multiple(-2.9, 1) → -2
    """
    if significance <= 0:
        raise ValueError("Significance must be positive")

    if number >= 0:
        return math.floor(number / significance) * significance
    else:
        return math.ceil(number / significance) * significance


@mcp_function(
    description="Round a number to the nearest multiple of significance. Uses standard rounding rules.",
    namespace="arithmetic",
    execution_modes=["local", "remote"],
    examples=[
        {
            "input": {"number": 2.4, "significance": 1},
            "output": 2,
            "description": "Round 2.4 to nearest 1",
        },
        {
            "input": {"number": 2.6, "significance": 1},
            "output": 3,
            "description": "Round 2.6 to nearest 1",
        },
        {
            "input": {"number": 7.3, "significance": 2},
            "output": 8,
            "description": "Round 7.3 to nearest 2",
        },
        {
            "input": {"number": 15, "significance": 10},
            "output": 20,
            "description": "Round 15 to nearest 10",
        },
    ],
)
async def mround(number: Number, significance: Number) -> Number:
    """
    Round a number to the nearest multiple of significance.

    Args:
        number: Number to round
        significance: Multiple to round to (must be positive)

    Returns:
        Number rounded to nearest multiple of significance

    Raises:
        ValueError: If significance is zero or negative

    Examples:
        await mround(2.4, 1) → 2
        await mround(2.6, 1) → 3
        await mround(7.3, 2) → 8
    """
    if significance <= 0:
        raise ValueError("Significance must be positive")

    return round(number / significance) * significance


@mcp_function(
    description="Convert a decimal number to its representation in another base (2-36).",
    namespace="arithmetic",
    execution_modes=["local", "remote"],
    examples=[
        {
            "input": {"number": 10, "base": 2},
            "output": "1010",
            "description": "10 in binary",
        },
        {
            "input": {"number": 255, "base": 16},
            "output": "FF",
            "description": "255 in hexadecimal",
        },
        {
            "input": {"number": 8, "base": 8},
            "output": "10",
            "description": "8 in octal",
        },
        {
            "input": {"number": 100, "base": 3},
            "output": "10201",
            "description": "100 in base 3",
        },
    ],
)
async def decimal_to_base(number: int, base: int) -> str:
    """
    Convert a decimal number to its representation in another base.

    Args:
        number: Non-negative integer to convert
        base: Target base (2-36)

    Returns:
        String representation of number in the specified base

    Raises:
        ValueError: If number is negative or base is not in range 2-36

    Examples:
        await decimal_to_base(10, 2) → "1010"
        await decimal_to_base(255, 16) → "FF"
        await decimal_to_base(8, 8) → "10"
    """
    if number < 0:
        raise ValueError("Number must be non-negative")
    if base < 2 or base > 36:
        raise ValueError("Base must be between 2 and 36")

    if number == 0:
        return "0"

    digits = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    result = ""

    # Yield control for large numbers
    if number > 100000:
        await asyncio.sleep(0)

    while number > 0:
        result = digits[number % base] + result
        number //= base

    return result


@mcp_function(
    description="Convert a number from a specified base (2-36) to decimal.",
    namespace="arithmetic",
    execution_modes=["local", "remote"],
    examples=[
        {
            "input": {"number": "1010", "base": 2},
            "output": 10,
            "description": "Binary 1010 to decimal",
        },
        {
            "input": {"number": "FF", "base": 16},
            "output": 255,
            "description": "Hex FF to decimal",
        },
        {
            "input": {"number": "10", "base": 8},
            "output": 8,
            "description": "Octal 10 to decimal",
        },
        {
            "input": {"number": "10201", "base": 3},
            "output": 100,
            "description": "Base 3 to decimal",
        },
    ],
)
async def base_to_decimal(number: str, base: int) -> int:
    """
    Convert a number from a specified base to decimal.

    Args:
        number: String representation of number in given base
        base: Source base (2-36)

    Returns:
        Decimal representation of the number

    Raises:
        ValueError: If base is not in range 2-36 or number contains invalid digits

    Examples:
        await base_to_decimal("1010", 2) → 10
        await base_to_decimal("FF", 16) → 255
        await base_to_decimal("10", 8) → 8
    """
    if base < 2 or base > 36:
        raise ValueError("Base must be between 2 and 36")

    try:
        return int(number.upper(), base)
    except ValueError:
        raise ValueError(f"Invalid number '{number}' for base {base}")


@mcp_function(
    description="Return the integer quotient of division without remainder. Equivalent to a // b.",
    namespace="arithmetic",
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
    description="Calculate the double factorial of a number. n!! = n × (n-2) × (n-4) × ... × 2 or 1.",
    namespace="arithmetic",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    estimated_cpu_usage="medium",
    examples=[
        {"input": {"n": 5}, "output": 15, "description": "5!! = 5 × 3 × 1 = 15"},
        {"input": {"n": 6}, "output": 48, "description": "6!! = 6 × 4 × 2 = 48"},
        {"input": {"n": 0}, "output": 1, "description": "0!! = 1 by definition"},
        {"input": {"n": 1}, "output": 1, "description": "1!! = 1"},
    ],
)
async def double_factorial(n: int) -> int:
    """
    Calculate the double factorial of a number.

    Args:
        n: Non-negative integer

    Returns:
        The double factorial n!!

    Raises:
        ValueError: If n is negative

    Examples:
        await double_factorial(5) → 15  # 5 × 3 × 1
        await double_factorial(6) → 48  # 6 × 4 × 2
        await double_factorial(0) → 1
    """
    if n < 0:
        raise ValueError("Double factorial is not defined for negative numbers")

    if n <= 1:
        return 1

    # Yield control for large calculations
    if n > 100:
        await asyncio.sleep(0)

    result = 1
    while n > 0:
        result *= n
        n -= 2

    return result


@mcp_function(
    description="Calculate the multinomial coefficient. Generalization of binomial coefficient for multiple groups.",
    namespace="arithmetic",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    estimated_cpu_usage="high",
    examples=[
        {
            "input": {"numbers": [3, 2, 1]},
            "output": 30,
            "description": "6! / (3! × 2! × 1!) = 30",
        },
        {
            "input": {"numbers": [2, 2]},
            "output": 6,
            "description": "4! / (2! × 2!) = 6",
        },
        {"input": {"numbers": [4]}, "output": 1, "description": "4! / 4! = 1"},
        {
            "input": {"numbers": [1, 1, 1]},
            "output": 6,
            "description": "3! / (1! × 1! × 1!) = 6",
        },
    ],
)
async def multinomial(numbers: List[int]) -> int:
    """
    Calculate the multinomial coefficient.

    Args:
        numbers: List of non-negative integers

    Returns:
        The multinomial coefficient (sum!)/(n1! × n2! × ... × nk!)

    Raises:
        ValueError: If any number is negative or list is empty

    Examples:
        await multinomial([3, 2, 1]) → 30  # 6!/(3!×2!×1!)
        await multinomial([2, 2]) → 6     # 4!/(2!×2!)
    """
    if not numbers:
        raise ValueError("Numbers list cannot be empty")

    if any(n < 0 for n in numbers):
        raise ValueError("All numbers must be non-negative")

    total = sum(numbers)

    # Yield control for large calculations
    if total > 100:
        await asyncio.sleep(0)

    result = math.factorial(total)

    for n in numbers:
        result //= math.factorial(n)

    return result


@mcp_function(
    description="Calculate the sum of products of corresponding elements in multiple arrays.",
    namespace="arithmetic",
    execution_modes=["local", "remote"],
    examples=[
        {
            "input": {"arrays": [[1, 2, 3], [4, 5, 6]]},
            "output": 32,
            "description": "1×4 + 2×5 + 3×6 = 32",
        },
        {
            "input": {"arrays": [[2, 3], [4, 5], [1, 2]]},
            "output": 38,
            "description": "2×4×1 + 3×5×2 = 38",
        },
        {
            "input": {"arrays": [[1, 0, 1], [1, 1, 0]]},
            "output": 1,
            "description": "1×1 + 0×1 + 1×0 = 1",
        },
        {"input": {"arrays": [[5], [3]]}, "output": 15, "description": "5×3 = 15"},
    ],
)
async def sum_product(arrays: List[List[Number]]) -> Number:
    """
    Calculate the sum of products of corresponding elements.

    Args:
        arrays: List of arrays with equal length

    Returns:
        Sum of products of corresponding elements

    Raises:
        ValueError: If arrays have different lengths or input is empty

    Examples:
        await sum_product([[1, 2, 3], [4, 5, 6]]) → 32
        await sum_product([[2, 3], [4, 5], [1, 2]]) → 38
    """
    if not arrays:
        raise ValueError("Arrays list cannot be empty")

    if not all(arrays):
        raise ValueError("All arrays must be non-empty")

    length = len(arrays[0])
    if not all(len(arr) == length for arr in arrays):
        raise ValueError("All arrays must have the same length")

    # Yield control for large arrays
    if length > 1000:
        await asyncio.sleep(0)

    total = 0
    for i in range(length):
        product = 1
        for arr in arrays:
            product *= arr[i]  # type: ignore[assignment]
        total += product

        # Yield control every 1000 iterations for very large arrays
        if i % 1000 == 999 and length > 1000:
            await asyncio.sleep(0)

    return total


@mcp_function(
    description="Calculate the sum of squares of numbers in a list.",
    namespace="arithmetic",
    execution_modes=["local", "remote"],
    examples=[
        {
            "input": {"numbers": [1, 2, 3, 4]},
            "output": 30,
            "description": "1² + 2² + 3² + 4² = 30",
        },
        {"input": {"numbers": [3, 4]}, "output": 25, "description": "3² + 4² = 25"},
        {"input": {"numbers": [-2, 2]}, "output": 8, "description": "(-2)² + 2² = 8"},
        {"input": {"numbers": [0, 5]}, "output": 25, "description": "0² + 5² = 25"},
    ],
)
async def sum_squares(numbers: List[Number]) -> Number:
    """
    Calculate the sum of squares of numbers.

    Args:
        numbers: List of numbers

    Returns:
        Sum of squares of all numbers

    Examples:
        await sum_squares([1, 2, 3, 4]) → 30
        await sum_squares([3, 4]) → 25
        await sum_squares([-2, 2]) → 8
    """
    # Yield control for large lists
    if len(numbers) > 1000:
        await asyncio.sleep(0)

    total = 0
    for i, x in enumerate(numbers):
        total += x * x  # type: ignore[assignment]
        # Yield control every 1000 iterations for very large lists
        if i % 1000 == 999 and len(numbers) > 1000:
            await asyncio.sleep(0)

    return total


@mcp_function(
    description="Generate a random float between 0 and 1 (exclusive of 1).",
    namespace="arithmetic",
    execution_modes=["local", "remote"],
    examples=[
        {
            "input": {},
            "output": 0.37454011884736246,
            "description": "Random float between 0 and 1",
        },
        {
            "input": {},
            "output": 0.891773001115671,
            "description": "Another random float",
        },
        {
            "input": {},
            "output": 0.19661193205945462,
            "description": "Random values vary each call",
        },
    ],
)
async def random_float() -> float:
    """
    Generate a random float between 0 and 1.

    Returns:
        Random float in range [0, 1)

    Examples:
        await random_float() → 0.37454011884736246
        await random_float() → 0.891773001115671
        # Each call returns different value
    """
    return random.random()


@mcp_function(
    description="Generate a random integer between min_val and max_val (inclusive).",
    namespace="arithmetic",
    execution_modes=["local", "remote"],
    examples=[
        {
            "input": {"min_val": 1, "max_val": 10},
            "output": 7,
            "description": "Random int between 1 and 10",
        },
        {
            "input": {"min_val": -5, "max_val": 5},
            "output": -2,
            "description": "Random int between -5 and 5",
        },
        {
            "input": {"min_val": 100, "max_val": 200},
            "output": 156,
            "description": "Random int between 100 and 200",
        },
        {
            "input": {"min_val": 5, "max_val": 5},
            "output": 5,
            "description": "Single value range",
        },
    ],
)
async def random_int(min_val: int, max_val: int) -> int:
    """
    Generate a random integer in a specified range.

    Args:
        min_val: Minimum value (inclusive)
        max_val: Maximum value (inclusive)

    Returns:
        Random integer between min_val and max_val (inclusive)

    Raises:
        ValueError: If min_val > max_val

    Examples:
        await random_int(1, 10) → 7
        await random_int(-5, 5) → -2
        await random_int(100, 200) → 156
    """
    if min_val > max_val:
        raise ValueError("min_val cannot be greater than max_val")

    return random.randint(min_val, max_val)


@mcp_function(
    description="Generate an array of random numbers with specified dimensions.",
    namespace="arithmetic",
    execution_modes=["local", "remote"],
    examples=[
        {
            "input": {"rows": 2, "cols": 3},
            "output": [[0.1, 0.5, 0.8], [0.3, 0.7, 0.2]],
            "description": "2x3 array of random floats",
        },
        {
            "input": {"rows": 1, "cols": 5},
            "output": [[0.4, 0.9, 0.1, 0.6, 0.3]],
            "description": "1x5 array",
        },
        {
            "input": {"rows": 3, "cols": 1},
            "output": [[0.7], [0.2], [0.8]],
            "description": "3x1 array",
        },
    ],
)
async def random_array(rows: int, cols: int) -> List[List[float]]:
    """
    Generate a 2D array of random numbers.

    Args:
        rows: Number of rows (must be positive)
        cols: Number of columns (must be positive)

    Returns:
        2D list of random floats between 0 and 1

    Raises:
        ValueError: If rows or cols is not positive

    Examples:
        await random_array(2, 3) → [[0.1, 0.5, 0.8], [0.3, 0.7, 0.2]]
        await random_array(1, 5) → [[0.4, 0.9, 0.1, 0.6, 0.3]]
        # Values are random and will vary
    """
    if rows <= 0 or cols <= 0:
        raise ValueError("Rows and columns must be positive")

    # Yield control for large arrays
    if rows * cols > 10000:
        await asyncio.sleep(0)

    result = []
    for i in range(rows):
        row = []
        for j in range(cols):
            row.append(random.random())
        result.append(row)

        # Yield control every 100 rows for very large arrays
        if i % 100 == 99 and rows > 100:
            await asyncio.sleep(0)

    return result


@mcp_function(
    description="Calculate the product of all numbers in a list. Returns the multiplication of all elements.",
    namespace="arithmetic",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    examples=[
        {
            "input": {"numbers": [2, 3, 4]},
            "output": 24,
            "description": "2 × 3 × 4 = 24",
        },
        {
            "input": {"numbers": [1, 2, 3, 4, 5]},
            "output": 120,
            "description": "1 × 2 × 3 × 4 × 5 = 120",
        },
        {"input": {"numbers": [-2, 3]}, "output": -6, "description": "(-2) × 3 = -6"},
        {"input": {"numbers": [5]}, "output": 5, "description": "Single element"},
    ],
)
async def product(numbers: List[Number]) -> Number:
    """
    Calculate the product of all numbers in a list.

    Args:
        numbers: List of numbers

    Returns:
        Product of all numbers in the list

    Raises:
        ValueError: If list is empty

    Examples:
        await product([2, 3, 4]) → 24
        await product([1, 2, 3, 4, 5]) → 120
        await product([-2, 3]) → -6
    """
    if not numbers:
        raise ValueError("Cannot calculate product of empty list")

    # Yield control for large lists
    if len(numbers) > 1000:
        await asyncio.sleep(0)

    result = 1
    for i, num in enumerate(numbers):
        result *= num  # type: ignore[assignment]
        # Yield control every 1000 iterations for very large lists
        if i % 1000 == 999 and len(numbers) > 1000:
            await asyncio.sleep(0)

    return result


@mcp_function(
    description="Convert an Arabic number to Roman numeral representation.",
    namespace="arithmetic",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    examples=[
        {
            "input": {"number": 27},
            "output": "XXVII",
            "description": "27 in Roman numerals",
        },
        {
            "input": {"number": 1994},
            "output": "MCMXCIV",
            "description": "1994 in Roman numerals",
        },
        {"input": {"number": 4}, "output": "IV", "description": "4 in Roman numerals"},
        {"input": {"number": 9}, "output": "IX", "description": "9 in Roman numerals"},
    ],
)
async def arabic_to_roman(number: int) -> str:
    """
    Convert an Arabic number to Roman numeral.

    Args:
        number: Positive integer (1-3999)

    Returns:
        Roman numeral representation

    Raises:
        ValueError: If number is not in range 1-3999

    Examples:
        await arabic_to_roman(27) → "XXVII"
        await arabic_to_roman(1994) → "MCMXCIV"
        await arabic_to_roman(4) → "IV"
    """
    if number < 1 or number > 3999:
        raise ValueError("Number must be between 1 and 3999")

    values = [1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1]
    numerals = ["M", "CM", "D", "CD", "C", "XC", "L", "XL", "X", "IX", "V", "IV", "I"]

    result = ""
    for i, value in enumerate(values):
        count = number // value
        result += numerals[i] * count
        number -= value * count

    return result


@mcp_function(
    description="Convert a Roman numeral to Arabic number representation.",
    namespace="arithmetic",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    examples=[
        {"input": {"numeral": "XXVII"}, "output": 27, "description": "XXVII to Arabic"},
        {
            "input": {"numeral": "MCMXCIV"},
            "output": 1994,
            "description": "MCMXCIV to Arabic",
        },
        {"input": {"numeral": "IV"}, "output": 4, "description": "IV to Arabic"},
        {"input": {"numeral": "IX"}, "output": 9, "description": "IX to Arabic"},
    ],
)
async def roman_to_arabic(numeral: str) -> int:
    """
    Convert a Roman numeral to Arabic number.

    Args:
        numeral: Valid Roman numeral string

    Returns:
        Arabic number representation

    Raises:
        ValueError: If numeral contains invalid characters

    Examples:
        await roman_to_arabic("XXVII") → 27
        await roman_to_arabic("MCMXCIV") → 1994
        await roman_to_arabic("IV") → 4
    """
    roman_values = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}

    numeral = numeral.upper()

    # Validate characters
    for char in numeral:
        if char not in roman_values:
            raise ValueError(f"Invalid Roman numeral character: {char}")

    total = 0
    prev_value = 0

    for char in reversed(numeral):
        value = roman_values[char]
        if value < prev_value:
            total -= value
        else:
            total += value
        prev_value = value

    return total


@mcp_function(
    description="Calculate the sum of an arithmetic series based on first term, last term, and number of terms.",
    namespace="arithmetic",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    examples=[
        {
            "input": {"first": 1, "last": 10, "count": 10},
            "output": 55,
            "description": "Sum 1+2+...+10",
        },
        {
            "input": {"first": 2, "last": 20, "count": 10},
            "output": 110,
            "description": "Sum of arithmetic series",
        },
        {
            "input": {"first": 5, "last": 5, "count": 1},
            "output": 5,
            "description": "Single term",
        },
        {
            "input": {"first": 10, "last": 2, "count": 5},
            "output": 30,
            "description": "Decreasing series",
        },
    ],
)
async def series_sum(first: Number, last: Number, count: int) -> Number:
    """
    Calculate the sum of an arithmetic series.

    Args:
        first: First term
        last: Last term
        count: Number of terms (must be positive)

    Returns:
        Sum of the arithmetic series

    Raises:
        ValueError: If count is not positive

    Examples:
        await series_sum(1, 10, 10) → 55  # 1+2+...+10
        await series_sum(2, 20, 10) → 110
        await series_sum(5, 5, 1) → 5
    """
    if count <= 0:
        raise ValueError("Count must be positive")

    return count * (first + last) / 2


# Export all advanced arithmetic functions
__all__ = [
    "ln",
    "log",
    "log10",
    "log2",
    "exp",
    "ceiling_multiple",
    "floor_multiple",
    "mround",
    "decimal_to_base",
    "base_to_decimal",
    "quotient",
    "double_factorial",
    "multinomial",
    "sum_product",
    "sum_squares",
    "product",
    "random_float",
    "random_int",
    "random_array",
    "arabic_to_roman",
    "roman_to_arabic",
    "series_sum",
]

if __name__ == "__main__":
    import asyncio

    async def test_advanced_arithmetic_functions():
        # Test the advanced arithmetic functions
        print("🔢 Advanced Arithmetic Operations Test (Async Native)")
        print("=" * 55)

        # Test logarithmic functions
        print(f"ln(2.718281828459045) = {await ln(2.718281828459045)}")
        print(f"log(8, 2) = {await log(8, 2)}")
        print(f"log10(1000) = {await log10(1000)}")
        print(f"log2(1024) = {await log2(1024)}")
        print(f"exp(1) = {await exp(1)}")

        # Test rounding functions
        print(f"ceiling_multiple(6.7, 2) = {await ceiling_multiple(6.7, 2)}")
        print(f"floor_multiple(7.8, 2) = {await floor_multiple(7.8, 2)}")
        print(f"mround(7.3, 2) = {await mround(7.3, 2)}")

        # Test base conversions
        print(f"decimal_to_base(255, 16) = {await decimal_to_base(255, 16)}")
        print(f"base_to_decimal('FF', 16) = {await base_to_decimal('FF', 16)}")

        # Test special operations
        print(f"quotient(17, 5) = {await quotient(17, 5)}")
        print(f"double_factorial(5) = {await double_factorial(5)}")
        print(f"multinomial([3, 2, 1]) = {await multinomial([3, 2, 1])}")

        # Test array operations
        print(
            f"sum_product([[1, 2, 3], [4, 5, 6]]) = {await sum_product([[1, 2, 3], [4, 5, 6]])}"
        )
        print(f"sum_squares([1, 2, 3, 4]) = {await sum_squares([1, 2, 3, 4])}")
        print(f"product([2, 3, 4]) = {await product([2, 3, 4])}")

        # Test random functions (show structure, values will vary)
        print(f"random_float() type = {type(await random_float())}")
        print(f"random_int(1, 10) type = {type(await random_int(1, 10))}")
        array_result = await random_array(2, 2)
        print(f"random_array(2, 2) shape = 2x2, type = {type(array_result)}")

        # Test Roman numerals
        print(f"arabic_to_roman(27) = {await arabic_to_roman(27)}")
        print(f"roman_to_arabic('XXVII') = {await roman_to_arabic('XXVII')}")

        # Test series sum
        print(f"series_sum(1, 10, 10) = {await series_sum(1, 10, 10)}")

        print("\n✅ All async advanced arithmetic functions working correctly!")

    asyncio.run(test_advanced_arithmetic_functions())
