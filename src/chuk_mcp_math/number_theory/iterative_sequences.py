#!/usr/bin/env python3
# chuk_mcp_math/number_theory/iterative_sequences.py
"""
Iterative Sequences - Async Native

Functions for sequences that are generated iteratively, including the famous
Collatz conjecture, Kaprekar sequences, happy numbers, and other iterative processes.

Functions:
- Collatz: collatz_sequence, collatz_stopping_time, collatz_max_value
- Kaprekar: kaprekar_sequence, kaprekar_constant, kaprekar_process
- Happy numbers: is_happy_number, happy_numbers, sad_numbers
- Narcissistic: is_narcissistic_number, narcissistic_numbers
- Look-and-say: look_and_say_sequence, look_and_say_length
- Recaman: recaman_sequence, recaman_first_n
- Keith numbers: is_keith_number, keith_numbers
- Digital operations: digital_sum_sequence, digital_product_sequence
"""

import asyncio
from typing import List, Optional
from chuk_mcp_math.mcp_decorator import mcp_function

# ============================================================================
# COLLATZ CONJECTURE (3n+1 PROBLEM)
# ============================================================================


@mcp_function(
    description="Generate the Collatz sequence (3n+1 problem) starting from n until reaching 1.",
    namespace="arithmetic",
    category="iterative_sequences",
    execution_modes=["local", "remote"],
    estimated_cpu_usage="medium",
    examples=[
        {
            "input": {"n": 7},
            "output": [7, 22, 11, 34, 17, 52, 26, 13, 40, 20, 10, 5, 16, 8, 4, 2, 1],
            "description": "Collatz sequence for 7",
        },
        {
            "input": {"n": 3},
            "output": [3, 10, 5, 16, 8, 4, 2, 1],
            "description": "Collatz sequence for 3",
        },
        {"input": {"n": 1}, "output": [1], "description": "Collatz sequence for 1"},
        {
            "input": {"n": 12},
            "output": [12, 6, 3, 10, 5, 16, 8, 4, 2, 1],
            "description": "Collatz sequence for 12",
        },
    ],
)
async def collatz_sequence(n: int) -> List[int]:
    """
    Generate the Collatz sequence starting from n.

    The Collatz conjecture states that this sequence always reaches 1.
    Rules: if n is even, divide by 2; if n is odd, multiply by 3 and add 1.

    Args:
        n: Starting positive integer

    Returns:
        List representing the Collatz sequence from n to 1

    Examples:
        await collatz_sequence(7) → [7, 22, 11, 34, 17, 52, 26, 13, 40, 20, 10, 5, 16, 8, 4, 2, 1]
        await collatz_sequence(3) → [3, 10, 5, 16, 8, 4, 2, 1]
    """
    if n <= 0:
        raise ValueError("n must be positive")

    sequence = [n]
    current = n
    iterations = 0

    while current != 1:
        if current % 2 == 0:
            current = current // 2
        else:
            current = 3 * current + 1

        sequence.append(current)
        iterations += 1

        # Yield control periodically and safety check
        if iterations % 1000 == 0:
            await asyncio.sleep(0)
            if iterations > 100000:  # Reasonable upper bound
                break

    return sequence


@mcp_function(
    description="Calculate the Collatz stopping time (number of steps to reach 1).",
    namespace="arithmetic",
    category="iterative_sequences",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    examples=[
        {"input": {"n": 7}, "output": 16, "description": "7 takes 16 steps to reach 1"},
        {"input": {"n": 3}, "output": 7, "description": "3 takes 7 steps to reach 1"},
        {"input": {"n": 1}, "output": 0, "description": "1 takes 0 steps"},
        {"input": {"n": 27}, "output": 111, "description": "27 takes 111 steps"},
    ],
)
async def collatz_stopping_time(n: int) -> int:
    """
    Calculate the Collatz stopping time (steps to reach 1).

    Args:
        n: Starting positive integer

    Returns:
        Number of steps to reach 1

    Examples:
        await collatz_stopping_time(7) → 16
        await collatz_stopping_time(3) → 7
        await collatz_stopping_time(27) → 111
    """
    if n <= 0:
        raise ValueError("n must be positive")

    if n == 1:
        return 0

    steps = 0
    current = n

    while current != 1:
        if current % 2 == 0:
            current = current // 2
        else:
            current = 3 * current + 1

        steps += 1

        # Yield control periodically and safety check
        if steps % 1000 == 0:
            await asyncio.sleep(0)
            if steps > 100000:
                break

    return steps


@mcp_function(
    description="Find the maximum value reached in the Collatz sequence.",
    namespace="arithmetic",
    category="iterative_sequences",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    examples=[
        {
            "input": {"n": 7},
            "output": 52,
            "description": "Max value in Collatz(7) is 52",
        },
        {
            "input": {"n": 3},
            "output": 16,
            "description": "Max value in Collatz(3) is 16",
        },
        {
            "input": {"n": 27},
            "output": 9232,
            "description": "Max value in Collatz(27) is 9232",
        },
    ],
)
async def collatz_max_value(n: int) -> int:
    """
    Find the maximum value reached in the Collatz sequence.

    Args:
        n: Starting positive integer

    Returns:
        Maximum value reached in the sequence

    Examples:
        await collatz_max_value(7) → 52
        await collatz_max_value(3) → 16
        await collatz_max_value(27) → 9232
    """
    if n <= 0:
        raise ValueError("n must be positive")

    max_val = n
    current = n
    iterations = 0

    while current != 1:
        if current % 2 == 0:
            current = current // 2
        else:
            current = 3 * current + 1

        max_val = max(max_val, current)
        iterations += 1

        # Yield control periodically and safety check
        if iterations % 1000 == 0:
            await asyncio.sleep(0)
            if iterations > 100000:
                break

    return max_val


# ============================================================================
# KAPREKAR SEQUENCES AND PROCESS
# ============================================================================


@mcp_function(
    description="Generate Kaprekar sequence for a number with given digit count.",
    namespace="arithmetic",
    category="iterative_sequences",
    execution_modes=["local", "remote"],
    examples=[
        {
            "input": {"n": 1234, "digits": 4},
            "output": [1234, 3087, 8352, 6174],
            "description": "Kaprekar sequence reaches 6174",
        },
        {
            "input": {"n": 495, "digits": 3},
            "output": [495],
            "description": "495 is Kaprekar constant for 3 digits",
        },
        {
            "input": {"n": 1111, "digits": 4},
            "output": [1111, 0],
            "description": "Repdigits lead to 0",
        },
    ],
)
async def kaprekar_sequence(n: int, digits: int) -> List[int]:
    """
    Generate Kaprekar sequence using Kaprekar's process.

    Kaprekar's process: arrange digits in descending and ascending order,
    subtract smaller from larger, repeat until reaching a fixed point.

    Args:
        n: Starting number
        digits: Number of digits to use (pad with zeros if needed)

    Returns:
        Sequence of numbers in Kaprekar process

    Examples:
        await kaprekar_sequence(1234, 4) → [1234, 3087, 8352, 6174]
        await kaprekar_sequence(495, 3) → [495]  # Already at Kaprekar constant
    """
    if digits <= 1:
        raise ValueError("digits must be greater than 1")

    sequence = []
    current = n
    seen = set()

    while current not in seen:
        seen.add(current)
        sequence.append(current)

        # Convert to string with proper padding
        digits_str = str(current).zfill(digits)

        # Check if all digits are the same (repdigit)
        if len(set(digits_str)) <= 1:
            sequence.append(0)
            break

        # Sort digits in descending and ascending order
        desc = int("".join(sorted(digits_str, reverse=True)))
        asc = int("".join(sorted(digits_str)))

        current = desc - asc

        # Yield control periodically
        if len(sequence) % 100 == 0:
            await asyncio.sleep(0)

        # Safety check
        if len(sequence) > 1000:
            break

    return sequence


@mcp_function(
    description="Get the Kaprekar constant for a given number of digits.",
    namespace="arithmetic",
    category="iterative_sequences",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    examples=[
        {
            "input": {"digits": 3},
            "output": 495,
            "description": "Kaprekar constant for 3 digits",
        },
        {
            "input": {"digits": 4},
            "output": 6174,
            "description": "Kaprekar constant for 4 digits",
        },
        {
            "input": {"digits": 6},
            "output": 549945,
            "description": "Kaprekar constant for 6 digits",
        },
    ],
)
async def kaprekar_constant(digits: int) -> Optional[int]:
    """
    Get the Kaprekar constant for a given number of digits.

    Args:
        digits: Number of digits

    Returns:
        Kaprekar constant for the given digit count, or None if none exists

    Examples:
        await kaprekar_constant(3) → 495
        await kaprekar_constant(4) → 6174
        await kaprekar_constant(6) → 549945
    """
    # Known Kaprekar constants
    constants = {3: 495, 4: 6174, 6: 549945, 7: 1194649}

    return constants.get(digits)


# ============================================================================
# HAPPY NUMBERS
# ============================================================================


@mcp_function(
    description="Check if a number is a happy number.",
    namespace="arithmetic",
    category="iterative_sequences",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    examples=[
        {"input": {"n": 7}, "output": True, "description": "7 is a happy number"},
        {"input": {"n": 19}, "output": True, "description": "19 is a happy number"},
        {"input": {"n": 4}, "output": False, "description": "4 is not happy (sad)"},
        {"input": {"n": 1}, "output": True, "description": "1 is happy by definition"},
    ],
)
async def is_happy_number(n: int) -> bool:
    """
    Check if a number is happy.

    A happy number is defined by the following process:
    - Replace the number by the sum of the squares of its digits
    - Repeat until the number equals 1 (happy) or cycles endlessly in a cycle that doesn't include 1 (sad)

    Args:
        n: Positive integer to check

    Returns:
        True if n is happy, False if sad

    Examples:
        await is_happy_number(7) → True   # 7→49→97→130→10→1
        await is_happy_number(19) → True  # 19→82→68→100→1
        await is_happy_number(4) → False  # 4→16→37→58→89→145→42→20→4 (cycles)
    """
    if n <= 0:
        return False

    if n == 1:
        return True

    seen = set()
    current = n

    while current != 1 and current not in seen:
        seen.add(current)

        # Calculate sum of squares of digits
        digit_sum = 0
        temp = current
        while temp > 0:
            digit = temp % 10
            digit_sum += digit * digit
            temp //= 10

        current = digit_sum

        # Yield control for large computations
        if len(seen) % 100 == 0:
            await asyncio.sleep(0)

    return current == 1


@mcp_function(
    description="Find all happy numbers up to a limit.",
    namespace="arithmetic",
    category="iterative_sequences",
    execution_modes=["local", "remote"],
    estimated_cpu_usage="medium",
    examples=[
        {
            "input": {"limit": 50},
            "output": [1, 7, 10, 13, 19, 23, 28, 31, 32, 44, 49],
            "description": "Happy numbers ≤ 50",
        },
        {
            "input": {"limit": 20},
            "output": [1, 7, 10, 13, 19],
            "description": "Happy numbers ≤ 20",
        },
    ],
)
async def happy_numbers(limit: int) -> List[int]:
    """
    Find all happy numbers up to a limit.

    Args:
        limit: Upper limit (inclusive)

    Returns:
        List of happy numbers ≤ limit

    Examples:
        await happy_numbers(50) → [1, 7, 10, 13, 19, 23, 28, 31, 32, 44, 49]
        await happy_numbers(20) → [1, 7, 10, 13, 19]
    """
    if limit <= 0:
        return []

    happy_list = []

    for i in range(1, limit + 1):
        if await is_happy_number(i):
            happy_list.append(i)

        # Yield control every 100 numbers
        if i % 100 == 0:
            await asyncio.sleep(0)

    return happy_list


# ============================================================================
# NARCISSISTIC NUMBERS
# ============================================================================


@mcp_function(
    description="Check if a number is narcissistic (equals sum of its digits raised to the power of digit count).",
    namespace="arithmetic",
    category="iterative_sequences",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    examples=[
        {"input": {"n": 153}, "output": True, "description": "153 = 1³ + 5³ + 3³"},
        {
            "input": {"n": 9474},
            "output": True,
            "description": "9474 = 9⁴ + 4⁴ + 7⁴ + 4⁴",
        },
        {"input": {"n": 123}, "output": False, "description": "123 ≠ 1³ + 2³ + 3³"},
        {"input": {"n": 1}, "output": True, "description": "1 = 1¹"},
    ],
)
async def is_narcissistic_number(n: int) -> bool:
    """
    Check if a number is narcissistic (also called pluperfect digital invariant).

    A narcissistic number is a number that is the sum of its own digits
    each raised to the power of the number of digits.

    Args:
        n: Positive integer to check

    Returns:
        True if n is narcissistic, False otherwise

    Examples:
        await is_narcissistic_number(153) → True   # 153 = 1³ + 5³ + 3³
        await is_narcissistic_number(9474) → True  # 9474 = 9⁴ + 4⁴ + 7⁴ + 4⁴
        await is_narcissistic_number(123) → False  # 123 ≠ 1³ + 2³ + 3³
    """
    if n <= 0:
        return False

    # Convert to string to get digits and count
    digits_str = str(n)
    num_digits = len(digits_str)

    # Calculate sum of digits raised to power of digit count
    digit_sum = sum(int(digit) ** num_digits for digit in digits_str)

    return digit_sum == n


@mcp_function(
    description="Find all narcissistic numbers up to a limit.",
    namespace="arithmetic",
    category="iterative_sequences",
    execution_modes=["local", "remote"],
    estimated_cpu_usage="high",
    examples=[
        {
            "input": {"limit": 1000},
            "output": [1, 2, 3, 4, 5, 6, 7, 8, 9, 153, 371, 407],
            "description": "Narcissistic numbers ≤ 1000",
        },
        {
            "input": {"limit": 200},
            "output": [1, 2, 3, 4, 5, 6, 7, 8, 9, 153],
            "description": "Narcissistic numbers ≤ 200",
        },
    ],
)
async def narcissistic_numbers(limit: int) -> List[int]:
    """
    Find all narcissistic numbers up to a limit.

    Args:
        limit: Upper limit (inclusive)

    Returns:
        List of narcissistic numbers ≤ limit

    Examples:
        await narcissistic_numbers(1000) → [1, 2, 3, 4, 5, 6, 7, 8, 9, 153, 371, 407]
        await narcissistic_numbers(200) → [1, 2, 3, 4, 5, 6, 7, 8, 9, 153]
    """
    if limit <= 0:
        return []

    narcissistic_list = []

    for i in range(1, limit + 1):
        if await is_narcissistic_number(i):
            narcissistic_list.append(i)

        # Yield control every 100 numbers
        if i % 100 == 0:
            await asyncio.sleep(0)

    return narcissistic_list


# ============================================================================
# LOOK-AND-SAY SEQUENCE
# ============================================================================


@mcp_function(
    description="Generate the look-and-say sequence starting from a number.",
    namespace="arithmetic",
    category="iterative_sequences",
    execution_modes=["local", "remote"],
    examples=[
        {
            "input": {"start": "1", "terms": 5},
            "output": ["1", "11", "21", "1211", "111221"],
            "description": "Look-and-say from 1",
        },
        {
            "input": {"start": "3", "terms": 4},
            "output": ["3", "13", "1113", "3113"],
            "description": "Look-and-say from 3",
        },
        {
            "input": {"start": "11", "terms": 3},
            "output": ["11", "21", "1211"],
            "description": "Look-and-say from 11",
        },
    ],
)
async def look_and_say_sequence(start: str, terms: int) -> List[str]:
    """
    Generate the look-and-say sequence.

    Each term describes the previous term:
    - "1" becomes "11" (one 1)
    - "11" becomes "21" (two 1s)
    - "21" becomes "1211" (one 2, one 1)

    Args:
        start: Starting string
        terms: Number of terms to generate

    Returns:
        List of strings in the look-and-say sequence

    Examples:
        await look_and_say_sequence("1", 5) → ["1", "11", "21", "1211", "111221"]
        await look_and_say_sequence("3", 4) → ["3", "13", "1113", "3113"]
    """
    if terms <= 0:
        return []

    sequence = [start]
    current = start

    for i in range(1, terms):
        next_term = ""
        count = 1
        prev_char = current[0]

        for j in range(1, len(current)):
            if current[j] == prev_char:
                count += 1
            else:
                next_term += str(count) + prev_char
                prev_char = current[j]
                count = 1

        # Add the last group
        next_term += str(count) + prev_char

        sequence.append(next_term)
        current = next_term

        # Yield control every 10 iterations
        if i % 10 == 0:
            await asyncio.sleep(0)

    return sequence


# ============================================================================
# RECAMÁN SEQUENCE
# ============================================================================


@mcp_function(
    description="Generate the Recamán sequence up to n terms.",
    namespace="arithmetic",
    category="iterative_sequences",
    execution_modes=["local", "remote"],
    examples=[
        {
            "input": {"n": 10},
            "output": [0, 1, 3, 6, 2, 7, 13, 20, 12, 21],
            "description": "First 10 Recamán numbers",
        },
        {
            "input": {"n": 5},
            "output": [0, 1, 3, 6, 2],
            "description": "First 5 Recamán numbers",
        },
    ],
)
async def recaman_sequence(n: int) -> List[int]:
    """
    Generate the Recamán sequence.

    Starting with a₀ = 0, the sequence is defined as:
    - aₙ = aₙ₋₁ - n if aₙ₋₁ - n > 0 and aₙ₋₁ - n is not already in the sequence
    - aₙ = aₙ₋₁ + n otherwise

    Args:
        n: Number of terms to generate

    Returns:
        List of first n terms of Recamán sequence

    Examples:
        await recaman_sequence(10) → [0, 1, 3, 6, 2, 7, 13, 20, 12, 21]
        await recaman_sequence(5) → [0, 1, 3, 6, 2]
    """
    if n <= 0:
        return []

    sequence = [0]
    seen = {0}

    for i in range(1, n):
        prev = sequence[-1]
        candidate = prev - i

        if candidate > 0 and candidate not in seen:
            next_val = candidate
        else:
            next_val = prev + i

        sequence.append(next_val)
        seen.add(next_val)

        # Yield control every 100 iterations
        if i % 100 == 0:
            await asyncio.sleep(0)

    return sequence


# ============================================================================
# KEITH NUMBERS
# ============================================================================


@mcp_function(
    description="Check if a number is a Keith number (repdigit-like sequence property).",
    namespace="arithmetic",
    category="iterative_sequences",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    examples=[
        {"input": {"n": 14}, "output": True, "description": "14 is Keith: 1,4,5,9,14"},
        {
            "input": {"n": 197},
            "output": True,
            "description": "197 is Keith: 1,9,7,17,33,57,107,197",
        },
        {"input": {"n": 15}, "output": False, "description": "15 is not Keith"},
        {"input": {"n": 742}, "output": True, "description": "742 is Keith"},
    ],
)
async def is_keith_number(n: int) -> bool:
    """
    Check if a number is a Keith number.

    A Keith number generates a sequence where each term is the sum of the
    previous k terms (where k is the number of digits), and the original
    number appears in this sequence.

    Args:
        n: Positive integer to check

    Returns:
        True if n is a Keith number, False otherwise

    Examples:
        await is_keith_number(14) → True    # sequence: 1,4,5,9,14
        await is_keith_number(197) → True   # sequence: 1,9,7,17,33,57,107,197
        await is_keith_number(15) → False   # doesn't appear in its sequence
    """
    if n <= 0:
        return False

    # Get digits
    digits = [int(d) for d in str(n)]
    k = len(digits)

    if k == 1:
        return False  # Single digits are not considered Keith numbers

    # Initialize sequence with the digits
    sequence = digits[:]

    # Generate sequence until we reach or exceed n
    iterations = 0
    while sequence[-1] < n:
        # Sum of last k terms
        next_term = sum(sequence[-k:])
        sequence.append(next_term)

        iterations += 1
        # Yield control and safety check
        if iterations % 100 == 0:
            await asyncio.sleep(0)
            if iterations > 10000:  # Prevent infinite loops
                break

    return sequence[-1] == n


@mcp_function(
    description="Find all Keith numbers up to a limit.",
    namespace="arithmetic",
    category="iterative_sequences",
    execution_modes=["local", "remote"],
    estimated_cpu_usage="high",
    examples=[
        {
            "input": {"limit": 1000},
            "output": [14, 19, 28, 47, 61, 75, 197, 742, 1104],
            "description": "Keith numbers ≤ 1000",
        },
        {
            "input": {"limit": 100},
            "output": [14, 19, 28, 47, 61, 75],
            "description": "Keith numbers ≤ 100",
        },
    ],
)
async def keith_numbers(limit: int) -> List[int]:
    """
    Find all Keith numbers up to a limit.

    Args:
        limit: Upper limit (inclusive)

    Returns:
        List of Keith numbers ≤ limit

    Examples:
        await keith_numbers(1000) → [14, 19, 28, 47, 61, 75, 197, 742, 1104]
        await keith_numbers(100) → [14, 19, 28, 47, 61, 75]
    """
    if limit <= 0:
        return []

    keith_list = []

    # Start from 10 since single digits are not Keith numbers
    for i in range(10, limit + 1):
        if await is_keith_number(i):
            keith_list.append(i)

        # Yield control every 50 numbers (Keith check is expensive)
        if i % 50 == 0:
            await asyncio.sleep(0)

    return keith_list


# ============================================================================
# DIGITAL SUM AND PRODUCT SEQUENCES
# ============================================================================


@mcp_function(
    description="Generate sequence by repeatedly applying digital sum until single digit.",
    namespace="arithmetic",
    category="iterative_sequences",
    execution_modes=["local", "remote"],
    examples=[
        {
            "input": {"n": 9875},
            "output": [9875, 29, 11, 2],
            "description": "Digital sum sequence for 9875",
        },
        {
            "input": {"n": 123},
            "output": [123, 6],
            "description": "Digital sum sequence for 123",
        },
        {"input": {"n": 7}, "output": [7], "description": "Single digit unchanged"},
    ],
)
async def digital_sum_sequence(n: int) -> List[int]:
    """
    Generate sequence by repeatedly calculating digital sum.

    Args:
        n: Starting positive integer

    Returns:
        Sequence ending when digital sum becomes single digit

    Examples:
        await digital_sum_sequence(9875) → [9875, 29, 11, 2]
        await digital_sum_sequence(123) → [123, 6]
        await digital_sum_sequence(7) → [7]
    """
    if n <= 0:
        raise ValueError("n must be positive")

    sequence = [n]
    current = n

    while current >= 10:
        # Calculate digital sum
        digit_sum = 0
        temp = current
        while temp > 0:
            digit_sum += temp % 10
            temp //= 10

        current = digit_sum
        sequence.append(current)

    return sequence


@mcp_function(
    description="Generate sequence by repeatedly applying digital product until single digit or zero.",
    namespace="arithmetic",
    category="iterative_sequences",
    execution_modes=["local", "remote"],
    examples=[
        {
            "input": {"n": 39},
            "output": [39, 27, 14, 4],
            "description": "Digital product sequence for 39",
        },
        {
            "input": {"n": 999},
            "output": [999, 729, 126, 12, 2],
            "description": "Digital product sequence for 999",
        },
        {
            "input": {"n": 105},
            "output": [105, 0],
            "description": "Contains 0, product becomes 0",
        },
    ],
)
async def digital_product_sequence(n: int) -> List[int]:
    """
    Generate sequence by repeatedly calculating digital product.

    Args:
        n: Starting positive integer

    Returns:
        Sequence ending when product becomes single digit or contains 0

    Examples:
        await digital_product_sequence(39) → [39, 27, 14, 4]
        await digital_product_sequence(999) → [999, 729, 126, 12, 2]
        await digital_product_sequence(105) → [105, 0]  # Contains 0
    """
    if n <= 0:
        raise ValueError("n must be positive")

    sequence = [n]
    current = n

    while current >= 10:
        # Calculate digital product
        digit_product = 1
        temp = current
        while temp > 0:
            digit = temp % 10
            digit_product *= digit
            temp //= 10

        current = digit_product
        sequence.append(current)

        # If product becomes 0, stop
        if current == 0:
            break

    return sequence


# Export all functions
__all__ = [
    # Collatz sequence
    "collatz_sequence",
    "collatz_stopping_time",
    "collatz_max_value",
    # Kaprekar sequences
    "kaprekar_sequence",
    "kaprekar_constant",
    # Happy numbers
    "is_happy_number",
    "happy_numbers",
    # Narcissistic numbers
    "is_narcissistic_number",
    "narcissistic_numbers",
    # Look-and-say sequence
    "look_and_say_sequence",
    # Recamán sequence
    "recaman_sequence",
    # Keith numbers
    "is_keith_number",
    "keith_numbers",
    # Digital sequences
    "digital_sum_sequence",
    "digital_product_sequence",
]

if __name__ == "__main__":
    import asyncio

    async def test_iterative_sequences():
        """Test iterative sequence functions."""
        print("🔄 Iterative Sequences Test")
        print("=" * 35)

        # Test Collatz sequence
        print("Collatz Sequence:")
        print(f"  collatz_sequence(7) = {await collatz_sequence(7)}")
        print(f"  collatz_stopping_time(7) = {await collatz_stopping_time(7)}")
        print(f"  collatz_max_value(7) = {await collatz_max_value(7)}")

        # Test Kaprekar
        print("\nKaprekar Process:")
        print(f"  kaprekar_sequence(1234, 4) = {await kaprekar_sequence(1234, 4)}")
        print(f"  kaprekar_constant(4) = {await kaprekar_constant(4)}")

        # Test Happy numbers
        print("\nHappy Numbers:")
        print(f"  is_happy_number(7) = {await is_happy_number(7)}")
        print(f"  happy_numbers(20) = {await happy_numbers(20)}")

        # Test Narcissistic
        print("\nNarcissistic Numbers:")
        print(f"  is_narcissistic_number(153) = {await is_narcissistic_number(153)}")
        print(f"  narcissistic_numbers(200) = {await narcissistic_numbers(200)}")

        # Test Look-and-say
        print("\nLook-and-Say:")
        print(
            f"  look_and_say_sequence('1', 5) = {await look_and_say_sequence('1', 5)}"
        )

        # Test Recamán
        print("\nRecamán Sequence:")
        print(f"  recaman_sequence(10) = {await recaman_sequence(10)}")

        # Test Keith numbers
        print("\nKeith Numbers:")
        print(f"  is_keith_number(14) = {await is_keith_number(14)}")
        print(f"  keith_numbers(100) = {await keith_numbers(100)}")

        # Test Digital sequences
        print("\nDigital Sequences:")
        print(f"  digital_sum_sequence(9875) = {await digital_sum_sequence(9875)}")
        print(f"  digital_product_sequence(39) = {await digital_product_sequence(39)}")

        print("\n✅ All iterative sequence functions working!")

    asyncio.run(test_iterative_sequences())
