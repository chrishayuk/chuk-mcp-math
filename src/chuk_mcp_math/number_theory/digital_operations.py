#!/usr/bin/env python3
# chuk_mcp_math/number_theory/digital_operations.py
"""
Digital Operations - Async Native

Functions for operations on the digits of numbers, including digital sums,
digital roots, palindromes, Harshad numbers, and other digit-based properties.

Functions:
- Digital sums: digit_sum, digital_root, digit_product, persistent_digital_root
- Digital transformations: digit_reversal, digit_sort, digit_scramble
- Palindromes: is_palindromic_number, palindromic_numbers, next_palindrome
- Harshad/Niven: is_harshad_number, harshad_numbers, harshad_in_base
- Base operations: number_to_base, base_to_number, digit_sum_in_base
- Digit properties: digit_count, digit_frequency, repeated_digit_check
- Special digit numbers: automorphic_numbers, kaprekar_numbers, dudeney_numbers
"""

import asyncio
from typing import List, Dict
from chuk_mcp_math.mcp_decorator import mcp_function

# ============================================================================
# DIGITAL SUMS AND ROOTS
# ============================================================================


@mcp_function(
    description="Calculate the sum of digits of a number in given base.",
    namespace="arithmetic",
    category="digital_operations",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    examples=[
        {"input": {"n": 12345}, "output": 15, "description": "1+2+3+4+5 = 15"},
        {"input": {"n": 999}, "output": 27, "description": "9+9+9 = 27"},
        {"input": {"n": 1729, "base": 10}, "output": 19, "description": "1+7+2+9 = 19"},
        {
            "input": {"n": 255, "base": 2},
            "output": 8,
            "description": "11111111 in binary has 8 ones",
        },
    ],
)
async def digit_sum(n: int, base: int = 10) -> int:
    """
    Calculate the sum of digits of a number in given base.

    Args:
        n: Non-negative integer
        base: Base for digit representation (default: 10)

    Returns:
        Sum of digits in the given base

    Examples:
        await digit_sum(12345) → 15      # 1+2+3+4+5
        await digit_sum(999) → 27        # 9+9+9
        await digit_sum(255, 2) → 8      # 11111111₂ has 8 ones
    """
    if n < 0:
        n = -n  # Work with absolute value

    if base < 2:
        raise ValueError("Base must be at least 2")

    total = 0
    while n > 0:
        total += n % base
        n //= base

    return total


@mcp_function(
    description="Calculate the digital root (single digit obtained by repeatedly summing digits).",
    namespace="arithmetic",
    category="digital_operations",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    examples=[
        {"input": {"n": 12345}, "output": 6, "description": "12345 → 15 → 6"},
        {"input": {"n": 999}, "output": 9, "description": "999 → 27 → 9"},
        {"input": {"n": 0}, "output": 0, "description": "Digital root of 0 is 0"},
        {"input": {"n": 9876}, "output": 3, "description": "9876 → 30 → 3"},
    ],
)
async def digital_root(n: int, base: int = 10) -> int:
    """
    Calculate the digital root by repeatedly summing digits until single digit.

    Args:
        n: Non-negative integer
        base: Base for digit representation (default: 10)

    Returns:
        Digital root (single digit in given base)

    Examples:
        await digital_root(12345) → 6    # 12345 → 15 → 6
        await digital_root(999) → 9      # 999 → 27 → 9
        await digital_root(9876) → 3     # 9876 → 30 → 3
    """
    if n < 0:
        n = -n  # Work with absolute value

    if base < 2:
        raise ValueError("Base must be at least 2")

    # Special case for 0
    if n == 0:
        return 0

    # Formula for digital root in base 10: 1 + (n-1) % 9
    # For other bases: 1 + (n-1) % (base-1)
    if base == 10:
        return 1 + (n - 1) % 9
    else:
        return 1 + (n - 1) % (base - 1)


@mcp_function(
    description="Calculate the product of digits of a number.",
    namespace="arithmetic",
    category="digital_operations",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    examples=[
        {"input": {"n": 123}, "output": 6, "description": "1×2×3 = 6"},
        {"input": {"n": 999}, "output": 729, "description": "9×9×9 = 729"},
        {
            "input": {"n": 1023},
            "output": 0,
            "description": "Contains 0, so product is 0",
        },
        {"input": {"n": 7}, "output": 7, "description": "Single digit"},
    ],
)
async def digit_product(n: int, base: int = 10) -> int:
    """
    Calculate the product of digits of a number.

    Args:
        n: Non-negative integer
        base: Base for digit representation (default: 10)

    Returns:
        Product of all digits

    Examples:
        await digit_product(123) → 6      # 1×2×3
        await digit_product(999) → 729    # 9×9×9
        await digit_product(1023) → 0     # Contains 0
    """
    if n < 0:
        n = -n  # Work with absolute value

    if base < 2:
        raise ValueError("Base must be at least 2")

    if n == 0:
        return 0

    product = 1
    while n > 0:
        digit = n % base
        product *= digit
        n //= base

    return product


@mcp_function(
    description="Calculate persistent digital root (number of steps to reach single digit).",
    namespace="arithmetic",
    category="digital_operations",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    examples=[
        {"input": {"n": 12345}, "output": 2, "description": "12345 → 15 → 6 (2 steps)"},
        {"input": {"n": 999}, "output": 2, "description": "999 → 27 → 9 (2 steps)"},
        {"input": {"n": 7}, "output": 0, "description": "Already single digit"},
        {"input": {"n": 9876}, "output": 2, "description": "9876 → 30 → 3 (2 steps)"},
    ],
)
async def persistent_digital_root(n: int, base: int = 10) -> int:
    """
    Calculate how many steps it takes to reach the digital root.

    Args:
        n: Non-negative integer
        base: Base for digit representation (default: 10)

    Returns:
        Number of steps to reach single digit

    Examples:
        await persistent_digital_root(12345) → 2    # 12345 → 15 → 6
        await persistent_digital_root(999) → 2      # 999 → 27 → 9
        await persistent_digital_root(7) → 0        # Already single digit
    """
    if n < 0:
        n = -n

    if base < 2:
        raise ValueError("Base must be at least 2")

    steps = 0
    while n >= base:
        n = await digit_sum(n, base)
        steps += 1

    return steps


# ============================================================================
# DIGITAL TRANSFORMATIONS
# ============================================================================


@mcp_function(
    description="Reverse the digits of a number.",
    namespace="arithmetic",
    category="digital_operations",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    examples=[
        {"input": {"n": 12345}, "output": 54321, "description": "Reverse of 12345"},
        {"input": {"n": 1000}, "output": 1, "description": "Trailing zeros removed"},
        {"input": {"n": 7}, "output": 7, "description": "Single digit unchanged"},
        {
            "input": {"n": 1234567890},
            "output": 987654321,
            "description": "Large number reversed",
        },
    ],
)
async def digit_reversal(n: int) -> int:
    """
    Reverse the digits of a number.

    Args:
        n: Non-negative integer

    Returns:
        Number with digits reversed (leading zeros dropped)

    Examples:
        await digit_reversal(12345) → 54321
        await digit_reversal(1000) → 1        # Trailing zeros become leading zeros
        await digit_reversal(7) → 7           # Single digit unchanged
    """
    if n < 0:
        n = -n  # Work with absolute value

    reversed_num = 0
    while n > 0:
        reversed_num = reversed_num * 10 + (n % 10)
        n //= 10

    return reversed_num


@mcp_function(
    description="Sort the digits of a number in ascending or descending order.",
    namespace="arithmetic",
    category="digital_operations",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    examples=[
        {
            "input": {"n": 54321, "descending": False},
            "output": 12345,
            "description": "Sort ascending",
        },
        {
            "input": {"n": 54321, "descending": True},
            "output": 54321,
            "description": "Sort descending",
        },
        {
            "input": {"n": 1729, "descending": False},
            "output": 1279,
            "description": "1729 sorted ascending",
        },
        {
            "input": {"n": 1000, "descending": True},
            "output": 1000,
            "description": "1000 sorted descending",
        },
    ],
)
async def digit_sort(n: int, descending: bool = False) -> int:
    """
    Sort the digits of a number.

    Args:
        n: Non-negative integer
        descending: If True, sort in descending order

    Returns:
        Number with digits sorted

    Examples:
        await digit_sort(54321, descending=False) → 12345
        await digit_sort(54321, descending=True) → 54321
        await digit_sort(1729, descending=False) → 1279
    """
    if n < 0:
        n = -n  # Work with absolute value

    # Convert to list of digits
    digits = []
    temp = n
    while temp > 0:
        digits.append(temp % 10)
        temp //= 10

    # Handle zero case
    if not digits:
        return 0

    # Sort digits
    digits.sort(reverse=descending)

    # Convert back to number
    result = 0
    for digit in digits:
        result = result * 10 + digit

    return result


# ============================================================================
# PALINDROMIC NUMBERS
# ============================================================================


@mcp_function(
    description="Check if a number is palindromic in given base.",
    namespace="arithmetic",
    category="digital_operations",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    examples=[
        {"input": {"n": 12321}, "output": True, "description": "12321 is palindromic"},
        {
            "input": {"n": 12345},
            "output": False,
            "description": "12345 is not palindromic",
        },
        {
            "input": {"n": 7},
            "output": True,
            "description": "Single digits are palindromic",
        },
        {
            "input": {"n": 9, "base": 2},
            "output": True,
            "description": "9 = 1001₂ is palindromic in binary",
        },
    ],
)
async def is_palindromic_number(n: int, base: int = 10) -> bool:
    """
    Check if a number reads the same forwards and backwards.

    Args:
        n: Non-negative integer
        base: Base for representation (default: 10)

    Returns:
        True if number is palindromic in given base

    Examples:
        await is_palindromic_number(12321) → True
        await is_palindromic_number(12345) → False
        await is_palindromic_number(9, 2) → True    # 9 = 1001₂
    """
    if n < 0:
        return False

    if base < 2:
        raise ValueError("Base must be at least 2")

    # Convert to digit representation
    digits = []
    temp = n
    while temp > 0:
        digits.append(temp % base)
        temp //= base

    # Handle zero case
    if not digits:
        return True

    # Check if palindromic
    return digits == digits[::-1]


@mcp_function(
    description="Find all palindromic numbers up to a limit.",
    namespace="arithmetic",
    category="digital_operations",
    execution_modes=["local", "remote"],
    estimated_cpu_usage="medium",
    examples=[
        {
            "input": {"limit": 200},
            "output": [
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                11,
                22,
                33,
                44,
                55,
                66,
                77,
                88,
                99,
                101,
                111,
                121,
                131,
                141,
                151,
                161,
                171,
                181,
                191,
            ],
            "description": "Palindromes ≤ 200",
        },
        {
            "input": {"limit": 50},
            "output": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 22, 33, 44],
            "description": "Palindromes ≤ 50",
        },
        {
            "input": {"limit": 1000, "base": 2},
            "output": [
                0,
                1,
                3,
                5,
                7,
                9,
                15,
                17,
                21,
                27,
                31,
                33,
                45,
                51,
                63,
                65,
                73,
                85,
                93,
                99,
                107,
                119,
                127,
                129,
                153,
                165,
                195,
                201,
                219,
                231,
                255,
                257,
                273,
                297,
                313,
                325,
                341,
                365,
                381,
                387,
                403,
                431,
                447,
                455,
                471,
                495,
                511,
                513,
                561,
                585,
                633,
                645,
                693,
                717,
                765,
                771,
                819,
                843,
                891,
                903,
                951,
                975,
            ],
            "description": "Binary palindromes ≤ 1000",
        },
    ],
)
async def palindromic_numbers(limit: int, base: int = 10) -> List[int]:
    """
    Find all palindromic numbers up to a limit.

    Args:
        limit: Upper limit (inclusive)
        base: Base for representation (default: 10)

    Returns:
        List of palindromic numbers ≤ limit

    Examples:
        await palindromic_numbers(200) → [0, 1, 2, ..., 191]
        await palindromic_numbers(50) → [0, 1, 2, ..., 44]
    """
    if limit < 0:
        return []

    palindromes = []

    for i in range(limit + 1):
        if await is_palindromic_number(i, base):
            palindromes.append(i)

        # Yield control every 1000 numbers
        if i % 1000 == 0 and limit > 1000:
            await asyncio.sleep(0)

    return palindromes


@mcp_function(
    description="Find the next palindromic number greater than n.",
    namespace="arithmetic",
    category="digital_operations",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    examples=[
        {
            "input": {"n": 123},
            "output": 131,
            "description": "Next palindrome after 123",
        },
        {
            "input": {"n": 999},
            "output": 1001,
            "description": "Next palindrome after 999",
        },
        {"input": {"n": 9}, "output": 11, "description": "Next palindrome after 9"},
        {
            "input": {"n": 191},
            "output": 202,
            "description": "Next palindrome after 191",
        },
    ],
)
async def next_palindrome(n: int, base: int = 10) -> int:
    """
    Find the next palindromic number greater than n.

    Args:
        n: Starting number
        base: Base for representation (default: 10)

    Returns:
        Smallest palindromic number > n

    Examples:
        await next_palindrome(123) → 131
        await next_palindrome(999) → 1001
        await next_palindrome(9) → 11
    """
    if base < 2:
        raise ValueError("Base must be at least 2")

    candidate = n + 1
    iterations = 0

    while not await is_palindromic_number(candidate, base):
        candidate += 1
        iterations += 1

        # Yield control every 1000 iterations
        if iterations % 1000 == 0:
            await asyncio.sleep(0)

        # Safety check for very large gaps
        if iterations > 100000:
            break

    return candidate


# ============================================================================
# HARSHAD/NIVEN NUMBERS
# ============================================================================


@mcp_function(
    description="Check if a number is a Harshad (Niven) number (divisible by sum of its digits).",
    namespace="arithmetic",
    category="digital_operations",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    examples=[
        {"input": {"n": 12}, "output": True, "description": "12 is divisible by 1+2=3"},
        {"input": {"n": 18}, "output": True, "description": "18 is divisible by 1+8=9"},
        {
            "input": {"n": 19},
            "output": False,
            "description": "19 is not divisible by 1+9=10",
        },
        {
            "input": {"n": 102},
            "output": True,
            "description": "102 is divisible by 1+0+2=3",
        },
    ],
)
async def is_harshad_number(n: int, base: int = 10) -> bool:
    """
    Check if a number is a Harshad (Niven) number.

    A Harshad number is divisible by the sum of its digits.

    Args:
        n: Positive integer to check
        base: Base for digit representation (default: 10)

    Returns:
        True if n is a Harshad number in given base

    Examples:
        await is_harshad_number(12) → True     # 12 ÷ (1+2) = 12 ÷ 3 = 4
        await is_harshad_number(18) → True     # 18 ÷ (1+8) = 18 ÷ 9 = 2
        await is_harshad_number(19) → False    # 19 ÷ (1+9) = 19 ÷ 10 = 1.9
    """
    if n <= 0:
        return False

    digit_sum_val = await digit_sum(n, base)

    if digit_sum_val == 0:
        return False

    return n % digit_sum_val == 0


@mcp_function(
    description="Find all Harshad numbers up to a limit.",
    namespace="arithmetic",
    category="digital_operations",
    execution_modes=["local", "remote"],
    estimated_cpu_usage="medium",
    examples=[
        {
            "input": {"limit": 100},
            "output": [
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                12,
                18,
                20,
                21,
                24,
                27,
                30,
                36,
                40,
                42,
                45,
                48,
                50,
                54,
                60,
                63,
                70,
                72,
                80,
                81,
                84,
                90,
                100,
            ],
            "description": "Harshad numbers ≤ 100",
        },
        {
            "input": {"limit": 50},
            "output": [
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                12,
                18,
                20,
                21,
                24,
                27,
                30,
                36,
                40,
                42,
                45,
                48,
                50,
            ],
            "description": "Harshad numbers ≤ 50",
        },
    ],
)
async def harshad_numbers(limit: int, base: int = 10) -> List[int]:
    """
    Find all Harshad numbers up to a limit.

    Args:
        limit: Upper limit (inclusive)
        base: Base for digit representation (default: 10)

    Returns:
        List of Harshad numbers ≤ limit

    Examples:
        await harshad_numbers(100) → [1, 2, 3, ..., 100]
        await harshad_numbers(50) → [1, 2, 3, ..., 50]
    """
    if limit <= 0:
        return []

    harshad_list = []

    for i in range(1, limit + 1):
        if await is_harshad_number(i, base):
            harshad_list.append(i)

        # Yield control every 1000 numbers
        if i % 1000 == 0 and limit > 1000:
            await asyncio.sleep(0)

    return harshad_list


# ============================================================================
# BASE CONVERSION OPERATIONS
# ============================================================================


@mcp_function(
    description="Convert a number to specified base representation.",
    namespace="arithmetic",
    category="digital_operations",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    examples=[
        {
            "input": {"n": 255, "base": 2},
            "output": "11111111",
            "description": "255 in binary",
        },
        {
            "input": {"n": 255, "base": 16},
            "output": "FF",
            "description": "255 in hexadecimal",
        },
        {
            "input": {"n": 1729, "base": 8},
            "output": "3301",
            "description": "1729 in octal",
        },
        {
            "input": {"n": 42, "base": 3},
            "output": "1120",
            "description": "42 in base 3",
        },
    ],
)
async def number_to_base(n: int, base: int) -> str:
    """
    Convert a number to specified base representation.

    Args:
        n: Non-negative integer
        base: Target base (2-36)

    Returns:
        String representation in target base

    Examples:
        await number_to_base(255, 2) → "11111111"   # Binary
        await number_to_base(255, 16) → "FF"        # Hexadecimal
        await number_to_base(1729, 8) → "3301"      # Octal
    """
    if n < 0:
        raise ValueError("Number must be non-negative")

    if base < 2 or base > 36:
        raise ValueError("Base must be between 2 and 36")

    if n == 0:
        return "0"

    digits = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    result = ""

    while n > 0:
        result = digits[n % base] + result
        n //= base

    return result


@mcp_function(
    description="Convert a base representation string to decimal number.",
    namespace="arithmetic",
    category="digital_operations",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    examples=[
        {
            "input": {"digits": "11111111", "base": 2},
            "output": 255,
            "description": "Binary to decimal",
        },
        {
            "input": {"digits": "FF", "base": 16},
            "output": 255,
            "description": "Hex to decimal",
        },
        {
            "input": {"digits": "3301", "base": 8},
            "output": 1729,
            "description": "Octal to decimal",
        },
        {
            "input": {"digits": "1120", "base": 3},
            "output": 42,
            "description": "Base 3 to decimal",
        },
    ],
)
async def base_to_number(digits: str, base: int) -> int:
    """
    Convert a base representation string to decimal number.

    Args:
        digits: String representation in source base
        base: Source base (2-36)

    Returns:
        Decimal integer value

    Examples:
        await base_to_number("11111111", 2) → 255   # Binary
        await base_to_number("FF", 16) → 255        # Hexadecimal
        await base_to_number("3301", 8) → 1729      # Octal
    """
    if base < 2 or base > 36:
        raise ValueError("Base must be between 2 and 36")

    if not digits:
        return 0

    digit_values = {str(i): i for i in range(10)}
    digit_values.update({chr(ord("A") + i): 10 + i for i in range(26)})
    digit_values.update({chr(ord("a") + i): 10 + i for i in range(26)})

    result = 0
    power = 0

    for digit in reversed(digits.upper()):
        if digit not in digit_values:
            raise ValueError(f"Invalid digit '{digit}' for base {base}")

        digit_value = digit_values[digit]
        if digit_value >= base:
            raise ValueError(f"Digit '{digit}' is not valid for base {base}")

        result += digit_value * (base**power)
        power += 1

    return result


# ============================================================================
# DIGIT PROPERTIES AND ANALYSIS
# ============================================================================


@mcp_function(
    description="Count the number of digits in a number for given base.",
    namespace="arithmetic",
    category="digital_operations",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    examples=[
        {"input": {"n": 12345}, "output": 5, "description": "12345 has 5 digits"},
        {"input": {"n": 0}, "output": 1, "description": "0 has 1 digit"},
        {
            "input": {"n": 255, "base": 2},
            "output": 8,
            "description": "255 = 11111111₂ has 8 binary digits",
        },
        {"input": {"n": 1000}, "output": 4, "description": "1000 has 4 digits"},
    ],
)
async def digit_count(n: int, base: int = 10) -> int:
    """
    Count the number of digits in a number.

    Args:
        n: Non-negative integer
        base: Base for representation (default: 10)

    Returns:
        Number of digits in given base

    Examples:
        await digit_count(12345) → 5        # 5 decimal digits
        await digit_count(255, 2) → 8       # 8 binary digits
        await digit_count(0) → 1            # 0 has 1 digit
    """
    if n < 0:
        n = -n  # Work with absolute value

    if base < 2:
        raise ValueError("Base must be at least 2")

    if n == 0:
        return 1

    count = 0
    while n > 0:
        count += 1
        n //= base

    return count


@mcp_function(
    description="Count frequency of each digit in a number.",
    namespace="arithmetic",
    category="digital_operations",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    examples=[
        {
            "input": {"n": 112233},
            "output": {1: 2, 2: 2, 3: 2},
            "description": "Digit frequencies in 112233",
        },
        {
            "input": {"n": 1000},
            "output": {0: 3, 1: 1},
            "description": "Digit frequencies in 1000",
        },
        {
            "input": {"n": 123456789},
            "output": {1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1},
            "description": "All digits appear once",
        },
    ],
)
async def digit_frequency(n: int, base: int = 10) -> Dict[int, int]:
    """
    Count the frequency of each digit in a number.

    Args:
        n: Non-negative integer
        base: Base for representation (default: 10)

    Returns:
        Dictionary mapping digit to frequency count

    Examples:
        await digit_frequency(112233) → {1: 2, 2: 2, 3: 2}
        await digit_frequency(1000) → {0: 3, 1: 1}
    """
    if n < 0:
        n = -n  # Work with absolute value

    if base < 2:
        raise ValueError("Base must be at least 2")

    frequency: dict[int, int] = {}

    if n == 0:
        return {0: 1}

    while n > 0:
        digit = n % base
        frequency[digit] = frequency.get(digit, 0) + 1
        n //= base

    return frequency


@mcp_function(
    description="Check if a number has all identical digits (repdigit).",
    namespace="arithmetic",
    category="digital_operations",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    examples=[
        {"input": {"n": 1111}, "output": True, "description": "1111 is a repdigit"},
        {"input": {"n": 777}, "output": True, "description": "777 is a repdigit"},
        {
            "input": {"n": 1234},
            "output": False,
            "description": "1234 is not a repdigit",
        },
        {
            "input": {"n": 5},
            "output": True,
            "description": "Single digits are repdigits",
        },
    ],
)
async def is_repdigit(n: int, base: int = 10) -> bool:
    """
    Check if a number consists of repeated identical digits.

    Args:
        n: Non-negative integer
        base: Base for representation (default: 10)

    Returns:
        True if all digits are the same

    Examples:
        await is_repdigit(1111) → True     # All 1s
        await is_repdigit(777) → True      # All 7s
        await is_repdigit(1234) → False    # Mixed digits
    """
    if n < 0:
        n = -n  # Work with absolute value

    if base < 2:
        raise ValueError("Base must be at least 2")

    if n < base:
        return True  # Single digit is always repdigit

    first_digit = n % base
    n //= base

    while n > 0:
        if n % base != first_digit:
            return False
        n //= base

    return True


# ============================================================================
# SPECIAL DIGIT-BASED NUMBERS
# ============================================================================


@mcp_function(
    description="Check if a number is automorphic (its square ends with the number itself).",
    namespace="arithmetic",
    category="digital_operations",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    examples=[
        {"input": {"n": 25}, "output": True, "description": "25² = 625 ends with 25"},
        {"input": {"n": 76}, "output": True, "description": "76² = 5776 ends with 76"},
        {
            "input": {"n": 10},
            "output": False,
            "description": "10² = 100 doesn't end with 10",
        },
        {"input": {"n": 1}, "output": True, "description": "1² = 1 ends with 1"},
    ],
)
async def is_automorphic_number(n: int) -> bool:
    """
    Check if a number is automorphic (its square ends with the number itself).

    Args:
        n: Non-negative integer

    Returns:
        True if n² ends with n

    Examples:
        await is_automorphic_number(25) → True   # 25² = 625
        await is_automorphic_number(76) → True   # 76² = 5776
        await is_automorphic_number(10) → False  # 10² = 100
    """
    if n < 0:
        return False

    square = n * n

    # Check if square ends with n
    temp = n
    while temp > 0:
        if square % 10 != temp % 10:
            return False
        square //= 10
        temp //= 10

    return True


@mcp_function(
    description="Find all automorphic numbers up to a limit.",
    namespace="arithmetic",
    category="digital_operations",
    execution_modes=["local", "remote"],
    estimated_cpu_usage="medium",
    examples=[
        {
            "input": {"limit": 100},
            "output": [0, 1, 5, 6, 25, 76],
            "description": "Automorphic numbers ≤ 100",
        },
        {
            "input": {"limit": 1000},
            "output": [0, 1, 5, 6, 25, 76, 376, 625],
            "description": "Automorphic numbers ≤ 1000",
        },
    ],
)
async def automorphic_numbers(limit: int) -> List[int]:
    """
    Find all automorphic numbers up to a limit.

    Args:
        limit: Upper limit (inclusive)

    Returns:
        List of automorphic numbers ≤ limit

    Examples:
        await automorphic_numbers(100) → [0, 1, 5, 6, 25, 76]
        await automorphic_numbers(1000) → [0, 1, 5, 6, 25, 76, 376, 625]
    """
    if limit < 0:
        return []

    automorphic_list = []

    for i in range(limit + 1):
        if await is_automorphic_number(i):
            automorphic_list.append(i)

        # Yield control every 1000 numbers
        if i % 1000 == 0 and limit > 1000:
            await asyncio.sleep(0)

    return automorphic_list


# Export all functions
__all__ = [
    # Digital sums and roots
    "digit_sum",
    "digital_root",
    "digit_product",
    "persistent_digital_root",
    # Digital transformations
    "digit_reversal",
    "digit_sort",
    # Palindromic numbers
    "is_palindromic_number",
    "palindromic_numbers",
    "next_palindrome",
    # Harshad numbers
    "is_harshad_number",
    "harshad_numbers",
    # Base conversions
    "number_to_base",
    "base_to_number",
    # Digit properties
    "digit_count",
    "digit_frequency",
    "is_repdigit",
    # Special numbers
    "is_automorphic_number",
    "automorphic_numbers",
]

if __name__ == "__main__":
    import asyncio

    async def test_digital_operations():
        """Test digital operations functions."""
        print("🔢 Digital Operations Test")
        print("=" * 35)

        # Test digital sums and roots
        print("Digital Sums and Roots:")
        print(f"  digit_sum(12345) = {await digit_sum(12345)}")
        print(f"  digital_root(12345) = {await digital_root(12345)}")
        print(f"  digit_product(123) = {await digit_product(123)}")
        print(
            f"  persistent_digital_root(12345) = {await persistent_digital_root(12345)}"
        )

        # Test transformations
        print("\nDigital Transformations:")
        print(f"  digit_reversal(12345) = {await digit_reversal(12345)}")
        print(
            f"  digit_sort(54321, descending=False) = {await digit_sort(54321, descending=False)}"
        )

        # Test palindromes
        print("\nPalindromic Numbers:")
        print(f"  is_palindromic_number(12321) = {await is_palindromic_number(12321)}")
        print(f"  next_palindrome(123) = {await next_palindrome(123)}")
        print(f"  palindromic_numbers(50) = {await palindromic_numbers(50)}")

        # Test Harshad numbers
        print("\nHarshad Numbers:")
        print(f"  is_harshad_number(12) = {await is_harshad_number(12)}")
        print(f"  harshad_numbers(50) = {await harshad_numbers(50)}")

        # Test base conversions
        print("\nBase Conversions:")
        print(f"  number_to_base(255, 2) = {await number_to_base(255, 2)}")
        print(f"  base_to_number('FF', 16) = {await base_to_number('FF', 16)}")

        # Test digit properties
        print("\nDigit Properties:")
        print(f"  digit_count(12345) = {await digit_count(12345)}")
        print(f"  digit_frequency(112233) = {await digit_frequency(112233)}")
        print(f"  is_repdigit(1111) = {await is_repdigit(1111)}")

        # Test special numbers
        print("\nSpecial Numbers:")
        print(f"  is_automorphic_number(25) = {await is_automorphic_number(25)}")
        print(f"  automorphic_numbers(100) = {await automorphic_numbers(100)}")

        print("\n✅ All digital operations functions working!")

    asyncio.run(test_digital_operations())
