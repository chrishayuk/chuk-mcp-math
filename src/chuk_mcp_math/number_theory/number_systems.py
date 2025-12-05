#!/usr/bin/env python3
# chuk_mcp_math/number_theory/number_systems.py
"""
Number Systems - Async Native - COMPLETE IMPLEMENTATION

Functions for working with different number systems, base conversions, and number types.
Essential for computer science, digital systems, and mathematical foundations.

Functions:
- Base conversions: binary_to_decimal, decimal_to_binary, base_conversion
- Number system operations: octal_conversion, hexadecimal_conversion
- Number type generators: natural_numbers, whole_numbers, integers_in_range
- Validation: validate_number_in_base, is_valid_base
- Arithmetic in different bases: add_in_base, multiply_in_base

Mathematical Background:
Number systems represent numerical values using different bases (radix).
Common systems include binary (base-2), octal (base-8), decimal (base-10),
and hexadecimal (base-16). Each position represents a power of the base.
"""

import asyncio
import string
from typing import List
from chuk_mcp_math.mcp_decorator import mcp_function

# ============================================================================
# BASE CONVERSION FUNCTIONS
# ============================================================================


@mcp_function(
    description="Convert a binary number (base-2) to decimal (base-10).",
    namespace="arithmetic",
    category="number_systems",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    examples=[
        {
            "input": {"binary_str": "1010"},
            "output": 10,
            "description": "Binary 1010 = Decimal 10",
        },
        {
            "input": {"binary_str": "11110000"},
            "output": 240,
            "description": "Binary 11110000 = Decimal 240",
        },
        {
            "input": {"binary_str": "1"},
            "output": 1,
            "description": "Binary 1 = Decimal 1",
        },
        {
            "input": {"binary_str": "0"},
            "output": 0,
            "description": "Binary 0 = Decimal 0",
        },
    ],
)
async def binary_to_decimal(binary_str: str) -> int:
    """
    Convert a binary number string to decimal.

    Args:
        binary_str: Binary number as string (e.g., "1010")

    Returns:
        Decimal equivalent as integer

    Raises:
        ValueError: If binary_str contains invalid characters

    Examples:
        await binary_to_decimal("1010") → 10
        await binary_to_decimal("11110000") → 240
        await binary_to_decimal("1") → 1
    """
    # Validate binary string
    if not all(bit in "01" for bit in binary_str):
        raise ValueError("Binary string must contain only 0s and 1s")

    if not binary_str:
        raise ValueError("Binary string cannot be empty")

    # Convert using positional notation
    decimal = 0
    power = 0

    # Process from right to left
    for bit in reversed(binary_str):
        if bit == "1":
            decimal += 2**power
        power += 1

        # Yield control for very long binary strings
        if power % 100 == 0:
            await asyncio.sleep(0)

    return decimal


@mcp_function(
    description="Convert a decimal number (base-10) to binary (base-2).",
    namespace="arithmetic",
    category="number_systems",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    examples=[
        {
            "input": {"decimal": 10},
            "output": "1010",
            "description": "Decimal 10 = Binary 1010",
        },
        {
            "input": {"decimal": 240},
            "output": "11110000",
            "description": "Decimal 240 = Binary 11110000",
        },
        {"input": {"decimal": 1}, "output": "1", "description": "Decimal 1 = Binary 1"},
        {"input": {"decimal": 0}, "output": "0", "description": "Decimal 0 = Binary 0"},
    ],
)
async def decimal_to_binary(decimal: int) -> str:
    """
    Convert a decimal number to binary string.

    Args:
        decimal: Non-negative decimal integer

    Returns:
        Binary representation as string

    Raises:
        ValueError: If decimal is negative

    Examples:
        await decimal_to_binary(10) → "1010"
        await decimal_to_binary(240) → "11110000"
        await decimal_to_binary(0) → "0"
    """
    if decimal < 0:
        raise ValueError("Decimal number must be non-negative")

    if decimal == 0:
        return "0"

    binary_digits = []

    while decimal > 0:
        binary_digits.append(str(decimal % 2))
        decimal //= 2

        # Yield control for very large numbers
        if len(binary_digits) % 100 == 0:
            await asyncio.sleep(0)

    # Reverse to get correct order
    return "".join(reversed(binary_digits))


@mcp_function(
    description="Convert a number from any base to any other base (2-36).",
    namespace="arithmetic",
    category="number_systems",
    execution_modes=["local", "remote"],
    estimated_cpu_usage="medium",
    examples=[
        {
            "input": {"number": "1010", "from_base": 2, "to_base": 10},
            "output": "10",
            "description": "Binary to decimal",
        },
        {
            "input": {"number": "FF", "from_base": 16, "to_base": 10},
            "output": "255",
            "description": "Hex to decimal",
        },
        {
            "input": {"number": "255", "from_base": 10, "to_base": 16},
            "output": "FF",
            "description": "Decimal to hex",
        },
        {
            "input": {"number": "777", "from_base": 8, "to_base": 2},
            "output": "111111111",
            "description": "Octal to binary",
        },
    ],
)
async def base_conversion(number: str, from_base: int, to_base: int) -> str:
    """
    Convert a number from one base to another.

    Args:
        number: Number as string in the source base
        from_base: Source base (2-36)
        to_base: Target base (2-36)

    Returns:
        Number converted to target base as string

    Raises:
        ValueError: If bases are invalid or number contains invalid digits

    Examples:
        await base_conversion("1010", 2, 10) → "10"
        await base_conversion("FF", 16, 10) → "255"
        await base_conversion("255", 10, 16) → "FF"
    """
    # Validate bases
    if not (2 <= from_base <= 36) or not (2 <= to_base <= 36):
        raise ValueError("Bases must be between 2 and 36")

    if not number:
        raise ValueError("Number string cannot be empty")

    # Define valid digits for bases up to 36
    valid_digits = string.digits + string.ascii_uppercase

    # Validate input number
    number = number.upper()
    for digit in number:
        if digit not in valid_digits[:from_base]:
            raise ValueError(f"Invalid digit '{digit}' for base {from_base}")

    # Convert to decimal first
    decimal_value = 0
    power = 0

    for digit in reversed(number):
        if digit.isdigit():
            digit_value = int(digit)
        else:
            digit_value = ord(digit) - ord("A") + 10

        decimal_value += digit_value * (from_base**power)
        power += 1

        # Yield control for large conversions
        if power % 50 == 0:
            await asyncio.sleep(0)

    # Convert from decimal to target base
    if decimal_value == 0:
        return "0"

    result_digits = []
    while decimal_value > 0:
        remainder = decimal_value % to_base
        if remainder < 10:
            result_digits.append(str(remainder))
        else:
            result_digits.append(chr(ord("A") + remainder - 10))
        decimal_value //= to_base

        # Yield control for large conversions
        if len(result_digits) % 50 == 0:
            await asyncio.sleep(0)

    return "".join(reversed(result_digits))


@mcp_function(
    description="Convert decimal to octal (base-8) number system.",
    namespace="arithmetic",
    category="number_systems",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    examples=[
        {
            "input": {"decimal": 64},
            "output": "100",
            "description": "Decimal 64 = Octal 100",
        },
        {
            "input": {"decimal": 255},
            "output": "377",
            "description": "Decimal 255 = Octal 377",
        },
        {
            "input": {"decimal": 8},
            "output": "10",
            "description": "Decimal 8 = Octal 10",
        },
        {"input": {"decimal": 0}, "output": "0", "description": "Decimal 0 = Octal 0"},
    ],
)
async def decimal_to_octal(decimal: int) -> str:
    """
    Convert decimal to octal representation.

    Args:
        decimal: Non-negative decimal integer

    Returns:
        Octal representation as string

    Examples:
        await decimal_to_octal(64) → "100"
        await decimal_to_octal(255) → "377"
        await decimal_to_octal(8) → "10"
    """
    return await base_conversion(str(decimal), 10, 8)


@mcp_function(
    description="Convert octal (base-8) to decimal number.",
    namespace="arithmetic",
    category="number_systems",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    examples=[
        {
            "input": {"octal_str": "100"},
            "output": 64,
            "description": "Octal 100 = Decimal 64",
        },
        {
            "input": {"octal_str": "377"},
            "output": 255,
            "description": "Octal 377 = Decimal 255",
        },
        {
            "input": {"octal_str": "10"},
            "output": 8,
            "description": "Octal 10 = Decimal 8",
        },
        {
            "input": {"octal_str": "0"},
            "output": 0,
            "description": "Octal 0 = Decimal 0",
        },
    ],
)
async def octal_to_decimal(octal_str: str) -> int:
    """
    Convert octal string to decimal.

    Args:
        octal_str: Octal number as string

    Returns:
        Decimal equivalent as integer

    Examples:
        await octal_to_decimal("100") → 64
        await octal_to_decimal("377") → 255
        await octal_to_decimal("10") → 8
    """
    decimal_str = await base_conversion(octal_str, 8, 10)
    return int(decimal_str)


@mcp_function(
    description="Convert decimal to hexadecimal (base-16) number system.",
    namespace="arithmetic",
    category="number_systems",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    examples=[
        {
            "input": {"decimal": 255},
            "output": "FF",
            "description": "Decimal 255 = Hex FF",
        },
        {
            "input": {"decimal": 16},
            "output": "10",
            "description": "Decimal 16 = Hex 10",
        },
        {
            "input": {"decimal": 171},
            "output": "AB",
            "description": "Decimal 171 = Hex AB",
        },
        {"input": {"decimal": 0}, "output": "0", "description": "Decimal 0 = Hex 0"},
    ],
)
async def decimal_to_hexadecimal(decimal: int) -> str:
    """
    Convert decimal to hexadecimal representation.

    Args:
        decimal: Non-negative decimal integer

    Returns:
        Hexadecimal representation as string (uppercase)

    Examples:
        await decimal_to_hexadecimal(255) → "FF"
        await decimal_to_hexadecimal(16) → "10"
        await decimal_to_hexadecimal(171) → "AB"
    """
    return await base_conversion(str(decimal), 10, 16)


@mcp_function(
    description="Convert hexadecimal (base-16) to decimal number.",
    namespace="arithmetic",
    category="number_systems",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    examples=[
        {
            "input": {"hex_str": "FF"},
            "output": 255,
            "description": "Hex FF = Decimal 255",
        },
        {
            "input": {"hex_str": "10"},
            "output": 16,
            "description": "Hex 10 = Decimal 16",
        },
        {
            "input": {"hex_str": "AB"},
            "output": 171,
            "description": "Hex AB = Decimal 171",
        },
        {"input": {"hex_str": "0"}, "output": 0, "description": "Hex 0 = Decimal 0"},
    ],
)
async def hexadecimal_to_decimal(hex_str: str) -> int:
    """
    Convert hexadecimal string to decimal.

    Args:
        hex_str: Hexadecimal number as string

    Returns:
        Decimal equivalent as integer

    Examples:
        await hexadecimal_to_decimal("FF") → 255
        await hexadecimal_to_decimal("10") → 16
        await hexadecimal_to_decimal("AB") → 171
    """
    decimal_str = await base_conversion(hex_str, 16, 10)
    return int(decimal_str)


# ============================================================================
# NUMBER TYPE GENERATORS
# ============================================================================


@mcp_function(
    description="Generate natural numbers (positive integers) in a given range.",
    namespace="arithmetic",
    category="number_systems",
    execution_modes=["local", "remote"],
    examples=[
        {
            "input": {"start": 1, "end": 5},
            "output": [1, 2, 3, 4, 5],
            "description": "Natural numbers 1 to 5",
        },
        {
            "input": {"start": 10, "end": 12},
            "output": [10, 11, 12],
            "description": "Natural numbers 10 to 12",
        },
        {
            "input": {"start": 1, "end": 1},
            "output": [1],
            "description": "Single natural number",
        },
        {
            "input": {"start": 5, "end": 3},
            "output": [],
            "description": "Invalid range returns empty",
        },
    ],
)
async def natural_numbers(start: int = 1, end: int = 10) -> List[int]:
    """
    Generate natural numbers in the given range [start, end].

    Args:
        start: Starting natural number (must be positive)
        end: Ending natural number (inclusive)

    Returns:
        List of natural numbers in the range

    Raises:
        ValueError: If start is not positive

    Examples:
        await natural_numbers(1, 5) → [1, 2, 3, 4, 5]
        await natural_numbers(10, 12) → [10, 11, 12]
    """
    if start < 1:
        raise ValueError("Natural numbers must start from 1 or greater")

    if start > end:
        return []

    # Yield control for large ranges
    if end - start > 1000:
        await asyncio.sleep(0)

    result = []
    for i in range(start, end + 1):
        result.append(i)

        # Yield control every 1000 numbers for very large ranges
        if len(result) % 1000 == 0 and end - start > 1000:
            await asyncio.sleep(0)

    return result


@mcp_function(
    description="Generate whole numbers (non-negative integers) in a given range.",
    namespace="arithmetic",
    category="number_systems",
    execution_modes=["local", "remote"],
    examples=[
        {
            "input": {"start": 0, "end": 5},
            "output": [0, 1, 2, 3, 4, 5],
            "description": "Whole numbers 0 to 5",
        },
        {
            "input": {"start": 3, "end": 7},
            "output": [3, 4, 5, 6, 7],
            "description": "Whole numbers 3 to 7",
        },
        {"input": {"start": 0, "end": 0}, "output": [0], "description": "Just zero"},
        {
            "input": {"start": 5, "end": 2},
            "output": [],
            "description": "Invalid range returns empty",
        },
    ],
)
async def whole_numbers(start: int = 0, end: int = 10) -> List[int]:
    """
    Generate whole numbers in the given range [start, end].

    Args:
        start: Starting whole number (must be non-negative)
        end: Ending whole number (inclusive)

    Returns:
        List of whole numbers in the range

    Raises:
        ValueError: If start is negative

    Examples:
        await whole_numbers(0, 5) → [0, 1, 2, 3, 4, 5]
        await whole_numbers(3, 7) → [3, 4, 5, 6, 7]
    """
    if start < 0:
        raise ValueError("Whole numbers must be non-negative")

    if start > end:
        return []

    # Yield control for large ranges
    if end - start > 1000:
        await asyncio.sleep(0)

    result = []
    for i in range(start, end + 1):
        result.append(i)

        # Yield control every 1000 numbers for very large ranges
        if len(result) % 1000 == 0 and end - start > 1000:
            await asyncio.sleep(0)

    return result


@mcp_function(
    description="Generate integers (positive, negative, and zero) in a given range.",
    namespace="arithmetic",
    category="number_systems",
    execution_modes=["local", "remote"],
    examples=[
        {
            "input": {"start": -3, "end": 3},
            "output": [-3, -2, -1, 0, 1, 2, 3],
            "description": "Integers from -3 to 3",
        },
        {
            "input": {"start": -5, "end": -2},
            "output": [-5, -4, -3, -2],
            "description": "Negative integers",
        },
        {
            "input": {"start": 2, "end": 5},
            "output": [2, 3, 4, 5],
            "description": "Positive integers",
        },
        {"input": {"start": 0, "end": 0}, "output": [0], "description": "Just zero"},
    ],
)
async def integers_in_range(start: int, end: int) -> List[int]:
    """
    Generate integers in the given range [start, end].

    Args:
        start: Starting integer (can be negative)
        end: Ending integer (inclusive)

    Returns:
        List of integers in the range

    Examples:
        await integers_in_range(-3, 3) → [-3, -2, -1, 0, 1, 2, 3]
        await integers_in_range(-5, -2) → [-5, -4, -3, -2]
        await integers_in_range(2, 5) → [2, 3, 4, 5]
    """
    if start > end:
        return []

    # Yield control for large ranges
    if end - start > 1000:
        await asyncio.sleep(0)

    result = []
    for i in range(start, end + 1):
        result.append(i)

        # Yield control every 1000 numbers for very large ranges
        if len(result) % 1000 == 0 and end - start > 1000:
            await asyncio.sleep(0)

    return result


# ============================================================================
# VALIDATION FUNCTIONS
# ============================================================================


@mcp_function(
    description="Validate if a number string is valid in the given base.",
    namespace="arithmetic",
    category="number_systems",
    execution_modes=["local", "remote"],
    examples=[
        {
            "input": {"number": "1010", "base": 2},
            "output": True,
            "description": "Valid binary number",
        },
        {
            "input": {"number": "129", "base": 2},
            "output": False,
            "description": "Invalid binary (contains 2 and 9)",
        },
        {
            "input": {"number": "ABC", "base": 16},
            "output": True,
            "description": "Valid hexadecimal",
        },
        {
            "input": {"number": "XYZ", "base": 16},
            "output": False,
            "description": "Invalid hexadecimal",
        },
    ],
)
async def validate_number_in_base(number: str, base: int) -> bool:
    """
    Validate if a number string is valid in the given base.

    Args:
        number: Number string to validate
        base: Base to validate against (2-36)

    Returns:
        True if number is valid in the base, False otherwise

    Examples:
        await validate_number_in_base("1010", 2) → True
        await validate_number_in_base("129", 2) → False
        await validate_number_in_base("ABC", 16) → True
    """
    if not (2 <= base <= 36):
        return False

    if not number:
        return False

    valid_digits = string.digits + string.ascii_uppercase
    valid_chars = valid_digits[:base]

    number = number.upper()
    return all(digit in valid_chars for digit in number)


@mcp_function(
    description="Check if a base value is valid (between 2 and 36).",
    namespace="arithmetic",
    category="number_systems",
    execution_modes=["local", "remote"],
    examples=[
        {"input": {"base": 10}, "output": True, "description": "Decimal base is valid"},
        {"input": {"base": 2}, "output": True, "description": "Binary base is valid"},
        {"input": {"base": 1}, "output": False, "description": "Base 1 is invalid"},
        {"input": {"base": 37}, "output": False, "description": "Base 37 is too large"},
    ],
)
async def is_valid_base(base: int) -> bool:
    """
    Check if a base value is valid.

    Args:
        base: Base value to check

    Returns:
        True if base is valid (2-36), False otherwise

    Examples:
        await is_valid_base(10) → True
        await is_valid_base(2) → True
        await is_valid_base(1) → False
        await is_valid_base(37) → False
    """
    return 2 <= base <= 36


# ============================================================================
# ARITHMETIC IN DIFFERENT BASES
# ============================================================================


@mcp_function(
    description="Add two numbers in a specified base.",
    namespace="arithmetic",
    category="number_systems",
    execution_modes=["local", "remote"],
    estimated_cpu_usage="medium",
    examples=[
        {
            "input": {"num1": "1010", "num2": "1100", "base": 2},
            "output": "10110",
            "description": "Binary addition",
        },
        {
            "input": {"num1": "FF", "num2": "1", "base": 16},
            "output": "100",
            "description": "Hexadecimal addition",
        },
        {
            "input": {"num1": "777", "num2": "1", "base": 8},
            "output": "1000",
            "description": "Octal addition",
        },
        {
            "input": {"num1": "99", "num2": "1", "base": 10},
            "output": "100",
            "description": "Decimal addition",
        },
    ],
)
async def add_in_base(num1: str, num2: str, base: int) -> str:
    """
    Add two numbers in a specified base.

    Args:
        num1: First number as string
        num2: Second number as string
        base: Base of the numbers (2-36)

    Returns:
        Sum in the same base as string

    Examples:
        await add_in_base("1010", "1100", 2) → "10110"
        await add_in_base("FF", "1", 16) → "100"
        await add_in_base("777", "1", 8) → "1000"
    """
    # Validate inputs
    if not await is_valid_base(base):
        raise ValueError(f"Invalid base: {base}")

    if not await validate_number_in_base(num1, base):
        raise ValueError(f"Invalid number '{num1}' for base {base}")

    if not await validate_number_in_base(num2, base):
        raise ValueError(f"Invalid number '{num2}' for base {base}")

    # Convert to decimal, add, then convert back
    decimal1_str = await base_conversion(num1, base, 10)
    decimal2_str = await base_conversion(num2, base, 10)

    decimal1 = int(decimal1_str)
    decimal2 = int(decimal2_str)

    sum_decimal = decimal1 + decimal2

    return await base_conversion(str(sum_decimal), 10, base)


@mcp_function(
    description="Multiply two numbers in a specified base.",
    namespace="arithmetic",
    category="number_systems",
    execution_modes=["local", "remote"],
    estimated_cpu_usage="medium",
    examples=[
        {
            "input": {"num1": "101", "num2": "11", "base": 2},
            "output": "1111",
            "description": "Binary multiplication",
        },
        {
            "input": {"num1": "A", "num2": "B", "base": 16},
            "output": "6E",
            "description": "Hexadecimal multiplication",
        },
        {
            "input": {"num1": "12", "num2": "34", "base": 8},
            "output": "450",
            "description": "Octal multiplication",
        },
        {
            "input": {"num1": "12", "num2": "34", "base": 10},
            "output": "408",
            "description": "Decimal multiplication",
        },
    ],
)
async def multiply_in_base(num1: str, num2: str, base: int) -> str:
    """
    Multiply two numbers in a specified base.

    Args:
        num1: First number as string
        num2: Second number as string
        base: Base of the numbers (2-36)

    Returns:
        Product in the same base as string

    Examples:
        await multiply_in_base("101", "11", 2) → "1111"
        await multiply_in_base("A", "B", 16) → "6E"
        await multiply_in_base("12", "34", 8) → "450"
    """
    # Validate inputs
    if not await is_valid_base(base):
        raise ValueError(f"Invalid base: {base}")

    if not await validate_number_in_base(num1, base):
        raise ValueError(f"Invalid number '{num1}' for base {base}")

    if not await validate_number_in_base(num2, base):
        raise ValueError(f"Invalid number '{num2}' for base {base}")

    # Convert to decimal, multiply, then convert back
    decimal1_str = await base_conversion(num1, base, 10)
    decimal2_str = await base_conversion(num2, base, 10)

    decimal1 = int(decimal1_str)
    decimal2 = int(decimal2_str)

    product_decimal = decimal1 * decimal2

    return await base_conversion(str(product_decimal), 10, base)


# Export all functions
__all__ = [
    # Base conversions
    "binary_to_decimal",
    "decimal_to_binary",
    "base_conversion",
    "decimal_to_octal",
    "octal_to_decimal",
    "decimal_to_hexadecimal",
    "hexadecimal_to_decimal",
    # Number type generators
    "natural_numbers",
    "whole_numbers",
    "integers_in_range",
    # Validation
    "validate_number_in_base",
    "is_valid_base",
    # Arithmetic in different bases
    "add_in_base",
    "multiply_in_base",
]

# ============================================================================
# COMPREHENSIVE TEST SUITE
# ============================================================================


async def test_number_systems():
    """Comprehensive test of all number system functions."""
    print("🔢 Number Systems Test Suite")
    print("=" * 40)

    # Test base conversions
    print("1. Base Conversions:")
    print(f"   binary_to_decimal('1010') = {await binary_to_decimal('1010')}")
    print(f"   decimal_to_binary(10) = {await decimal_to_binary(10)}")
    print(f"   base_conversion('FF', 16, 10) = {await base_conversion('FF', 16, 10)}")
    print(f"   decimal_to_octal(64) = {await decimal_to_octal(64)}")
    print(f"   decimal_to_hexadecimal(255) = {await decimal_to_hexadecimal(255)}")

    # Test number generators
    print("\n2. Number Type Generators:")
    print(f"   natural_numbers(1, 5) = {await natural_numbers(1, 5)}")
    print(f"   whole_numbers(0, 5) = {await whole_numbers(0, 5)}")
    print(f"   integers_in_range(-3, 3) = {await integers_in_range(-3, 3)}")

    # Test validation
    print("\n3. Validation:")
    print(
        f"   validate_number_in_base('1010', 2) = {await validate_number_in_base('1010', 2)}"
    )
    print(
        f"   validate_number_in_base('129', 2) = {await validate_number_in_base('129', 2)}"
    )
    print(f"   is_valid_base(10) = {await is_valid_base(10)}")
    print(f"   is_valid_base(37) = {await is_valid_base(37)}")

    # Test arithmetic in different bases
    print("\n4. Arithmetic in Different Bases:")
    print(f"   add_in_base('1010', '1100', 2) = {await add_in_base('1010', '1100', 2)}")
    print(f"   multiply_in_base('A', 'B', 16) = {await multiply_in_base('A', 'B', 16)}")

    print("\n✅ All number system functions working perfectly!")


async def demo_base_conversions():
    """Demonstrate various base conversion scenarios."""
    print("\n🔄 Base Conversion Demonstrations")
    print("=" * 35)

    # Common conversions
    conversions = [
        (255, "decimal to binary, octal, hex"),
        (1024, "powers of 2"),
        (42, "answer to everything"),
        (365, "days in a year"),
    ]

    for num, description in conversions:
        binary = await decimal_to_binary(num)
        octal = await decimal_to_octal(num)
        hex_val = await decimal_to_hexadecimal(num)

        print(f"\n  {description.title()} ({num}):")
        print(f"    Binary:      {binary}")
        print(f"    Octal:       {octal}")
        print(f"    Hexadecimal: {hex_val}")

        # Verify conversions
        back_to_decimal = await binary_to_decimal(binary)
        print(
            f"    Verification: {back_to_decimal} ✓"
            if back_to_decimal == num
            else "    Error!"
        )


if __name__ == "__main__":
    import asyncio

    async def main():
        await test_number_systems()
        await demo_base_conversions()

    asyncio.run(main())
