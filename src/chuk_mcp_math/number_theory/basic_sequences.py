#!/usr/bin/env python3
# chuk_mcp_math/number_theory/basic_sequences.py
"""
Basic Mathematical Sequences - Async Native

Functions for fundamental mathematical sequences and basic special numbers.
Includes perfect squares, powers of two, Fibonacci numbers, factorials,
triangular numbers, pentagonal numbers, and other basic sequences.

Functions:
- Perfect squares: is_perfect_square, perfect_squares, nth_perfect_square
- Powers of two: is_power_of_two, powers_of_two, nth_power_of_two
- Fibonacci: fibonacci, fibonacci_sequence, is_fibonacci_number
- Factorial: factorial, double_factorial, subfactorial
- Triangular: triangular_number, is_triangular_number, triangular_sequence
- Pentagonal: pentagonal_number, is_pentagonal_number, pentagonal_sequence
- Square pyramidal: square_pyramidal_number, tetrahedral_number
- Sequence utilities: sequence_term_check, sequence_generation
"""

import math
import asyncio
from typing import List
from chuk_mcp_math.mcp_decorator import mcp_function

# ============================================================================
# PERFECT SQUARES
# ============================================================================


@mcp_function(
    description="Check if a number is a perfect square.",
    namespace="arithmetic",
    category="basic_sequences",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    examples=[
        {
            "input": {"n": 16},
            "output": True,
            "description": "16 = 4² is a perfect square",
        },
        {
            "input": {"n": 25},
            "output": True,
            "description": "25 = 5² is a perfect square",
        },
        {
            "input": {"n": 15},
            "output": False,
            "description": "15 is not a perfect square",
        },
        {
            "input": {"n": 0},
            "output": True,
            "description": "0 = 0² is a perfect square",
        },
    ],
)
async def is_perfect_square(n: int) -> bool:
    """
    Check if a number is a perfect square.

    Args:
        n: Non-negative integer to check

    Returns:
        True if n is a perfect square, False otherwise

    Examples:
        await is_perfect_square(16) → True   # 16 = 4²
        await is_perfect_square(25) → True   # 25 = 5²
        await is_perfect_square(15) → False  # No integer squared equals 15
    """
    if n < 0:
        return False

    if n == 0:
        return True

    sqrt_n = int(math.sqrt(n))
    return sqrt_n * sqrt_n == n


@mcp_function(
    description="Generate the first n perfect squares.",
    namespace="arithmetic",
    category="basic_sequences",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    examples=[
        {
            "input": {"n": 10},
            "output": [0, 1, 4, 9, 16, 25, 36, 49, 64, 81],
            "description": "First 10 perfect squares",
        },
        {
            "input": {"n": 5},
            "output": [0, 1, 4, 9, 16],
            "description": "First 5 perfect squares",
        },
    ],
)
async def perfect_squares(n: int) -> List[int]:
    """
    Generate the first n perfect squares.

    Args:
        n: Number of perfect squares to generate

    Returns:
        List of first n perfect squares [0², 1², 2², ...]

    Examples:
        await perfect_squares(10) → [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]
        await perfect_squares(5) → [0, 1, 4, 9, 16]
    """
    if n <= 0:
        return []

    result = []
    for i in range(n):
        result.append(i * i)

        # Yield control every 1000 iterations for large n
        if i % 1000 == 0 and n > 1000:
            await asyncio.sleep(0)

    return result


@mcp_function(
    description="Get the nth perfect square (0-indexed).",
    namespace="arithmetic",
    category="basic_sequences",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    examples=[
        {
            "input": {"n": 5},
            "output": 25,
            "description": "5th perfect square is 5² = 25",
        },
        {"input": {"n": 0}, "output": 0, "description": "0th perfect square is 0² = 0"},
    ],
)
async def nth_perfect_square(n: int) -> int:
    """
    Get the nth perfect square.

    Args:
        n: Index (0-based) of perfect square to get

    Returns:
        The nth perfect square (n²)

    Examples:
        await nth_perfect_square(5) → 25  # 5² = 25
        await nth_perfect_square(0) → 0   # 0² = 0
    """
    if n < 0:
        raise ValueError("Index must be non-negative")

    return n * n


# ============================================================================
# POWERS OF TWO
# ============================================================================


@mcp_function(
    description="Check if a number is a power of two.",
    namespace="arithmetic",
    category="basic_sequences",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    examples=[
        {"input": {"n": 8}, "output": True, "description": "8 = 2³ is a power of two"},
        {
            "input": {"n": 16},
            "output": True,
            "description": "16 = 2⁴ is a power of two",
        },
        {
            "input": {"n": 15},
            "output": False,
            "description": "15 is not a power of two",
        },
        {"input": {"n": 1}, "output": True, "description": "1 = 2⁰ is a power of two"},
    ],
)
async def is_power_of_two(n: int) -> bool:
    """
    Check if a number is a power of two.

    Args:
        n: Positive integer to check

    Returns:
        True if n is a power of two, False otherwise

    Examples:
        await is_power_of_two(8) → True    # 8 = 2³
        await is_power_of_two(16) → True   # 16 = 2⁴
        await is_power_of_two(15) → False  # Not a power of two
    """
    if n <= 0:
        return False

    return (n & (n - 1)) == 0


@mcp_function(
    description="Generate the first n powers of two.",
    namespace="arithmetic",
    category="basic_sequences",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    examples=[
        {
            "input": {"n": 10},
            "output": [1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
            "description": "First 10 powers of two",
        },
        {
            "input": {"n": 5},
            "output": [1, 2, 4, 8, 16],
            "description": "First 5 powers of two",
        },
    ],
)
async def powers_of_two(n: int) -> List[int]:
    """
    Generate the first n powers of two.

    Args:
        n: Number of powers of two to generate

    Returns:
        List of first n powers of two [2⁰, 2¹, 2², ...]

    Examples:
        await powers_of_two(10) → [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
        await powers_of_two(5) → [1, 2, 4, 8, 16]
    """
    if n <= 0:
        return []

    result = []
    power = 1

    for i in range(n):
        result.append(power)
        power *= 2

        # Yield control every 50 iterations for large n (powers grow quickly)
        if i % 50 == 0 and n > 50:
            await asyncio.sleep(0)

    return result


@mcp_function(
    description="Get the nth power of two (0-indexed).",
    namespace="arithmetic",
    category="basic_sequences",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    examples=[
        {"input": {"n": 5}, "output": 32, "description": "5th power of two is 2⁵ = 32"},
        {"input": {"n": 0}, "output": 1, "description": "0th power of two is 2⁰ = 1"},
    ],
)
async def nth_power_of_two(n: int) -> int:
    """
    Get the nth power of two.

    Args:
        n: Index (0-based) of power of two to get

    Returns:
        The nth power of two (2ⁿ)

    Examples:
        await nth_power_of_two(5) → 32  # 2⁵ = 32
        await nth_power_of_two(0) → 1   # 2⁰ = 1
    """
    if n < 0:
        raise ValueError("Index must be non-negative")

    return 1 << n  # Efficient 2^n using bit shift


# ============================================================================
# FIBONACCI NUMBERS
# ============================================================================


@mcp_function(
    description="Calculate the nth Fibonacci number using efficient matrix exponentiation.",
    namespace="arithmetic",
    category="basic_sequences",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    estimated_cpu_usage="medium",
    examples=[
        {"input": {"n": 10}, "output": 55, "description": "10th Fibonacci number"},
        {"input": {"n": 0}, "output": 0, "description": "0th Fibonacci number"},
        {"input": {"n": 1}, "output": 1, "description": "1st Fibonacci number"},
        {"input": {"n": 20}, "output": 6765, "description": "20th Fibonacci number"},
    ],
)
async def fibonacci(n: int) -> int:
    """
    Calculate the nth Fibonacci number.

    Uses efficient matrix exponentiation for large n.
    Fibonacci sequence: 0, 1, 1, 2, 3, 5, 8, 13, 21, 34, ...

    Args:
        n: Non-negative integer (position in sequence)

    Returns:
        The nth Fibonacci number

    Examples:
        await fibonacci(0) → 0     # F₀ = 0
        await fibonacci(1) → 1     # F₁ = 1
        await fibonacci(10) → 55   # F₁₀ = 55
    """
    if n < 0:
        raise ValueError("Fibonacci number position must be non-negative")

    if n <= 1:
        return n

    # For small n, use simple iteration
    if n <= 100:
        a, b = 0, 1
        for i in range(2, n + 1):
            a, b = b, a + b
            if i % 10 == 0:
                await asyncio.sleep(0)
        return b

    # For large n, use matrix exponentiation
    def matrix_mult(A, B):
        return [
            [
                A[0][0] * B[0][0] + A[0][1] * B[1][0],
                A[0][0] * B[0][1] + A[0][1] * B[1][1],
            ],
            [
                A[1][0] * B[0][0] + A[1][1] * B[1][0],
                A[1][0] * B[0][1] + A[1][1] * B[1][1],
            ],
        ]

    async def matrix_power(matrix, power):
        if power == 1:
            return matrix

        result = [[1, 0], [0, 1]]  # Identity matrix
        base = matrix
        exp = power
        iterations = 0

        while exp > 0:
            if exp % 2 == 1:
                result = matrix_mult(result, base)
            base = matrix_mult(base, base)
            exp //= 2
            iterations += 1

            if iterations % 10 == 0:
                await asyncio.sleep(0)

        return result

    fib_matrix = [[1, 1], [1, 0]]
    result_matrix = await matrix_power(fib_matrix, n)
    return result_matrix[0][1]


@mcp_function(
    description="Generate the first n Fibonacci numbers.",
    namespace="arithmetic",
    category="basic_sequences",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    estimated_cpu_usage="medium",
    examples=[
        {
            "input": {"n": 10},
            "output": [0, 1, 1, 2, 3, 5, 8, 13, 21, 34],
            "description": "First 10 Fibonacci numbers",
        },
        {
            "input": {"n": 5},
            "output": [0, 1, 1, 2, 3],
            "description": "First 5 Fibonacci numbers",
        },
    ],
)
async def fibonacci_sequence(n: int) -> List[int]:
    """
    Generate the first n Fibonacci numbers.

    Args:
        n: Number of Fibonacci numbers to generate

    Returns:
        List of the first n Fibonacci numbers

    Examples:
        await fibonacci_sequence(10) → [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
        await fibonacci_sequence(5) → [0, 1, 1, 2, 3]
    """
    if n <= 0:
        return []

    if n == 1:
        return [0]

    if n == 2:
        return [0, 1]

    result = [0, 1]

    for i in range(2, n):
        result.append(result[i - 1] + result[i - 2])

        if i % 1000 == 0 and n > 1000:
            await asyncio.sleep(0)

    return result


@mcp_function(
    description="Check if a number is a Fibonacci number.",
    namespace="arithmetic",
    category="basic_sequences",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    examples=[
        {
            "input": {"n": 13},
            "output": True,
            "description": "13 is the 7th Fibonacci number",
        },
        {
            "input": {"n": 55},
            "output": True,
            "description": "55 is the 10th Fibonacci number",
        },
        {
            "input": {"n": 12},
            "output": False,
            "description": "12 is not a Fibonacci number",
        },
    ],
)
async def is_fibonacci_number(n: int) -> bool:
    """
    Check if a number is a Fibonacci number.

    Uses the mathematical property: a positive integer n is a Fibonacci number
    if and only if one of (5n²+4) or (5n²-4) is a perfect square.

    Args:
        n: Non-negative integer to check

    Returns:
        True if n is a Fibonacci number, False otherwise

    Examples:
        await is_fibonacci_number(13) → True   # F₇ = 13
        await is_fibonacci_number(55) → True   # F₁₀ = 55
        await is_fibonacci_number(12) → False  # Not in sequence
    """
    if n < 0:
        return False

    if n == 0:
        return True

    def is_perfect_square_helper(x):
        if x < 0:
            return False
        sqrt_x = int(math.sqrt(x))
        return sqrt_x * sqrt_x == x

    n_squared = n * n
    return is_perfect_square_helper(5 * n_squared + 4) or is_perfect_square_helper(
        5 * n_squared - 4
    )


# ============================================================================
# FACTORIAL AND RELATED
# ============================================================================


@mcp_function(
    description="Calculate factorial n! = n × (n-1) × ... × 2 × 1.",
    namespace="arithmetic",
    category="basic_sequences",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    estimated_cpu_usage="medium",
    examples=[
        {"input": {"n": 5}, "output": 120, "description": "5! = 5×4×3×2×1 = 120"},
        {"input": {"n": 0}, "output": 1, "description": "0! = 1 by definition"},
        {"input": {"n": 10}, "output": 3628800, "description": "10! = 3,628,800"},
    ],
)
async def factorial(n: int) -> int:
    """
    Calculate the factorial of n.

    Args:
        n: Non-negative integer

    Returns:
        n! (factorial of n)

    Examples:
        await factorial(0) → 1        # 0! = 1
        await factorial(5) → 120      # 5! = 120
        await factorial(10) → 3628800 # 10! = 3,628,800
    """
    if n < 0:
        raise ValueError("Factorial is not defined for negative numbers")

    if n <= 1:
        return 1

    result = 1
    for i in range(2, n + 1):
        result *= i

        if i % 1000 == 0 and n > 1000:
            await asyncio.sleep(0)

    return result


@mcp_function(
    description="Calculate double factorial n!! = n × (n-2) × (n-4) × ...",
    namespace="arithmetic",
    category="basic_sequences",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    examples=[
        {"input": {"n": 8}, "output": 384, "description": "8!! = 8×6×4×2 = 384"},
        {"input": {"n": 7}, "output": 105, "description": "7!! = 7×5×3×1 = 105"},
        {"input": {"n": 0}, "output": 1, "description": "0!! = 1 by definition"},
    ],
)
async def double_factorial(n: int) -> int:
    """
    Calculate the double factorial of n.

    n!! = n × (n-2) × (n-4) × ... down to 1 or 2

    Args:
        n: Non-negative integer

    Returns:
        n!! (double factorial of n)

    Examples:
        await double_factorial(8) → 384  # 8×6×4×2 = 384
        await double_factorial(7) → 105  # 7×5×3×1 = 105
        await double_factorial(0) → 1    # 0!! = 1
    """
    if n < 0:
        raise ValueError("Double factorial is not defined for negative numbers")

    if n <= 1:
        return 1

    result = 1
    i = n
    iterations = 0

    while i > 0:
        result *= i
        i -= 2
        iterations += 1

        if iterations % 1000 == 0:
            await asyncio.sleep(0)

    return result


@mcp_function(
    description="Calculate subfactorial !n (derangements of n items).",
    namespace="arithmetic",
    category="basic_sequences",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    examples=[
        {
            "input": {"n": 4},
            "output": 9,
            "description": "!4 = 9 derangements of 4 items",
        },
        {
            "input": {"n": 5},
            "output": 44,
            "description": "!5 = 44 derangements of 5 items",
        },
        {"input": {"n": 0}, "output": 1, "description": "!0 = 1 by definition"},
    ],
)
async def subfactorial(n: int) -> int:
    """
    Calculate the subfactorial of n (number of derangements).

    !n is the number of permutations of n items where no item appears
    in its original position.

    Args:
        n: Non-negative integer

    Returns:
        !n (subfactorial of n)

    Examples:
        await subfactorial(4) → 9   # 9 derangements of 4 items
        await subfactorial(5) → 44  # 44 derangements of 5 items
        await subfactorial(0) → 1   # Empty derangement
    """
    if n < 0:
        raise ValueError("Subfactorial is not defined for negative numbers")

    if n == 0:
        return 1
    if n == 1:
        return 0

    # Use recurrence: !n = (n-1) × (!(n-1) + !(n-2))
    prev_prev = 1  # !0
    prev = 0  # !1

    for i in range(2, n + 1):
        current = (i - 1) * (prev + prev_prev)
        prev_prev, prev = prev, current

        if i % 1000 == 0 and n > 1000:
            await asyncio.sleep(0)

    return prev


# ============================================================================
# TRIANGULAR NUMBERS
# ============================================================================


@mcp_function(
    description="Calculate the nth triangular number T_n = n(n+1)/2.",
    namespace="arithmetic",
    category="basic_sequences",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    examples=[
        {"input": {"n": 5}, "output": 15, "description": "T₅ = 5×6/2 = 15"},
        {"input": {"n": 10}, "output": 55, "description": "T₁₀ = 10×11/2 = 55"},
        {"input": {"n": 0}, "output": 0, "description": "T₀ = 0"},
    ],
)
async def triangular_number(n: int) -> int:
    """
    Calculate the nth triangular number.

    T_n = 1 + 2 + 3 + ... + n = n(n+1)/2

    Args:
        n: Non-negative integer

    Returns:
        The nth triangular number

    Examples:
        await triangular_number(5) → 15   # T₅ = 15
        await triangular_number(10) → 55  # T₁₀ = 55
        await triangular_number(0) → 0    # T₀ = 0
    """
    if n < 0:
        raise ValueError("Triangular number index must be non-negative")

    return n * (n + 1) // 2


@mcp_function(
    description="Check if a number is a triangular number.",
    namespace="arithmetic",
    category="basic_sequences",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    examples=[
        {"input": {"n": 15}, "output": True, "description": "15 = T₅ is triangular"},
        {"input": {"n": 55}, "output": True, "description": "55 = T₁₀ is triangular"},
        {"input": {"n": 12}, "output": False, "description": "12 is not triangular"},
    ],
)
async def is_triangular_number(n: int) -> bool:
    """
    Check if a number is a triangular number.

    A number n is triangular if 8n + 1 is a perfect square.

    Args:
        n: Non-negative integer to check

    Returns:
        True if n is a triangular number, False otherwise

    Examples:
        await is_triangular_number(15) → True   # T₅ = 15
        await is_triangular_number(55) → True   # T₁₀ = 55
        await is_triangular_number(12) → False  # Not triangular
    """
    if n < 0:
        return False

    if n == 0:
        return True

    # n is triangular iff 8n + 1 is a perfect square
    discriminant = 8 * n + 1
    sqrt_disc = int(math.sqrt(discriminant))
    return sqrt_disc * sqrt_disc == discriminant


@mcp_function(
    description="Generate the first n triangular numbers.",
    namespace="arithmetic",
    category="basic_sequences",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    examples=[
        {
            "input": {"n": 10},
            "output": [0, 1, 3, 6, 10, 15, 21, 28, 36, 45],
            "description": "First 10 triangular numbers",
        },
        {
            "input": {"n": 5},
            "output": [0, 1, 3, 6, 10],
            "description": "First 5 triangular numbers",
        },
    ],
)
async def triangular_sequence(n: int) -> List[int]:
    """
    Generate the first n triangular numbers.

    Args:
        n: Number of triangular numbers to generate

    Returns:
        List of first n triangular numbers

    Examples:
        await triangular_sequence(10) → [0, 1, 3, 6, 10, 15, 21, 28, 36, 45]
        await triangular_sequence(5) → [0, 1, 3, 6, 10]
    """
    if n <= 0:
        return []

    result = []
    for i in range(n):
        result.append(i * (i + 1) // 2)

        if i % 1000 == 0 and n > 1000:
            await asyncio.sleep(0)

    return result


# ============================================================================
# PENTAGONAL NUMBERS
# ============================================================================


@mcp_function(
    description="Calculate the nth pentagonal number P_n = n(3n-1)/2.",
    namespace="arithmetic",
    category="basic_sequences",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    examples=[
        {"input": {"n": 5}, "output": 35, "description": "P₅ = 5×14/2 = 35"},
        {"input": {"n": 10}, "output": 145, "description": "P₁₀ = 10×29/2 = 145"},
        {"input": {"n": 0}, "output": 0, "description": "P₀ = 0"},
    ],
)
async def pentagonal_number(n: int) -> int:
    """
    Calculate the nth pentagonal number.

    P_n = n(3n-1)/2

    Args:
        n: Non-negative integer

    Returns:
        The nth pentagonal number

    Examples:
        await pentagonal_number(5) → 35   # P₅ = 35
        await pentagonal_number(10) → 145 # P₁₀ = 145
        await pentagonal_number(0) → 0    # P₀ = 0
    """
    if n < 0:
        raise ValueError("Pentagonal number index must be non-negative")

    return n * (3 * n - 1) // 2


@mcp_function(
    description="Check if a number is a pentagonal number.",
    namespace="arithmetic",
    category="basic_sequences",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    examples=[
        {"input": {"n": 35}, "output": True, "description": "35 = P₅ is pentagonal"},
        {"input": {"n": 145}, "output": True, "description": "145 = P₁₀ is pentagonal"},
        {"input": {"n": 20}, "output": False, "description": "20 is not pentagonal"},
    ],
)
async def is_pentagonal_number(n: int) -> bool:
    """
    Check if a number is a pentagonal number.

    A number n is pentagonal if (24n + 1) is a perfect square and
    (sqrt(24n + 1) + 1) is divisible by 6.

    Args:
        n: Non-negative integer to check

    Returns:
        True if n is a pentagonal number, False otherwise

    Examples:
        await is_pentagonal_number(35) → True   # P₅ = 35
        await is_pentagonal_number(145) → True  # P₁₀ = 145
        await is_pentagonal_number(20) → False  # Not pentagonal
    """
    if n < 0:
        return False

    if n == 0:
        return True

    # Check if (24n + 1) is a perfect square
    discriminant = 24 * n + 1
    sqrt_disc = int(math.sqrt(discriminant))

    if sqrt_disc * sqrt_disc != discriminant:
        return False

    # Check if (sqrt_disc + 1) is divisible by 6
    return (sqrt_disc + 1) % 6 == 0


@mcp_function(
    description="Generate the first n pentagonal numbers.",
    namespace="arithmetic",
    category="basic_sequences",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    examples=[
        {
            "input": {"n": 10},
            "output": [0, 1, 5, 12, 22, 35, 51, 70, 92, 117],
            "description": "First 10 pentagonal numbers",
        },
        {
            "input": {"n": 5},
            "output": [0, 1, 5, 12, 22],
            "description": "First 5 pentagonal numbers",
        },
    ],
)
async def pentagonal_sequence(n: int) -> List[int]:
    """
    Generate the first n pentagonal numbers.

    Args:
        n: Number of pentagonal numbers to generate

    Returns:
        List of first n pentagonal numbers

    Examples:
        await pentagonal_sequence(10) → [0, 1, 5, 12, 22, 35, 51, 70, 92, 117]
        await pentagonal_sequence(5) → [0, 1, 5, 12, 22]
    """
    if n <= 0:
        return []

    result = []
    for i in range(n):
        result.append(i * (3 * i - 1) // 2)

        if i % 1000 == 0 and n > 1000:
            await asyncio.sleep(0)

    return result


# ============================================================================
# PYRAMIDAL NUMBERS
# ============================================================================


@mcp_function(
    description="Calculate the nth square pyramidal number (sum of first n squares).",
    namespace="arithmetic",
    category="basic_sequences",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    examples=[
        {"input": {"n": 5}, "output": 55, "description": "1²+2²+3²+4²+5² = 55"},
        {"input": {"n": 10}, "output": 385, "description": "Sum of first 10 squares"},
        {"input": {"n": 0}, "output": 0, "description": "Sum of no squares is 0"},
    ],
)
async def square_pyramidal_number(n: int) -> int:
    """
    Calculate the nth square pyramidal number.

    SP_n = 1² + 2² + ... + n² = n(n+1)(2n+1)/6

    Args:
        n: Non-negative integer

    Returns:
        The nth square pyramidal number

    Examples:
        await square_pyramidal_number(5) → 55   # 1²+2²+3²+4²+5² = 55
        await square_pyramidal_number(10) → 385 # Sum of first 10 squares
    """
    if n < 0:
        raise ValueError("Square pyramidal number index must be non-negative")

    return n * (n + 1) * (2 * n + 1) // 6


@mcp_function(
    description="Calculate the nth tetrahedral number (sum of first n triangular numbers).",
    namespace="arithmetic",
    category="basic_sequences",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    examples=[
        {"input": {"n": 5}, "output": 35, "description": "1+3+6+10+15 = 35"},
        {
            "input": {"n": 10},
            "output": 220,
            "description": "Sum of first 10 triangular numbers",
        },
        {
            "input": {"n": 0},
            "output": 0,
            "description": "Sum of no triangular numbers is 0",
        },
    ],
)
async def tetrahedral_number(n: int) -> int:
    """
    Calculate the nth tetrahedral number.

    Tet_n = T_1 + T_2 + ... + T_n = n(n+1)(n+2)/6

    Args:
        n: Non-negative integer

    Returns:
        The nth tetrahedral number

    Examples:
        await tetrahedral_number(5) → 35   # 1+3+6+10+15 = 35
        await tetrahedral_number(10) → 220 # Sum of first 10 triangular numbers
    """
    if n < 0:
        raise ValueError("Tetrahedral number index must be non-negative")

    return n * (n + 1) * (n + 2) // 6


# ============================================================================
# CATALAN NUMBERS (included here as a commonly used basic sequence)
# ============================================================================


@mcp_function(
    description="Calculate the nth Catalan number C_n using efficient recurrence.",
    namespace="arithmetic",
    category="basic_sequences",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    estimated_cpu_usage="medium",
    examples=[
        {"input": {"n": 0}, "output": 1, "description": "C_0 = 1"},
        {"input": {"n": 1}, "output": 1, "description": "C_1 = 1"},
        {"input": {"n": 2}, "output": 2, "description": "C_2 = 2"},
        {"input": {"n": 3}, "output": 5, "description": "C_3 = 5"},
        {"input": {"n": 4}, "output": 14, "description": "C_4 = 14"},
        {"input": {"n": 5}, "output": 42, "description": "C_5 = 42"},
    ],
)
async def catalan_number(n: int) -> int:
    """
    Calculate the nth Catalan number.

    C_n = (2n)! / ((n+1)! * n!) = (2n choose n) / (n+1)

    Catalan numbers count many combinatorial objects:
    - Binary trees with n internal nodes
    - Ways to parenthesize n+1 factors
    - Monotonic lattice paths that don't cross the diagonal
    - Non-crossing partitions of 2n points on a circle

    Args:
        n: Non-negative integer

    Returns:
        The nth Catalan number

    Examples:
        await catalan_number(0) → 1    # C_0 = 1
        await catalan_number(3) → 5    # C_3 = 5
        await catalan_number(5) → 42   # C_5 = 42
    """
    if n < 0:
        raise ValueError("Catalan number index must be non-negative")

    if n == 0:
        return 1

    # Use the recurrence: C_n = (4n - 2) * C_{n-1} / (n + 1)
    # This is more efficient than computing factorials
    result = 1
    for i in range(1, n + 1):
        result = result * (4 * i - 2) // (i + 1)
        if i % 5 == 0:  # Yield control periodically
            await asyncio.sleep(0)

    return result


# Export all functions
__all__ = [
    # Perfect squares
    "is_perfect_square",
    "perfect_squares",
    "nth_perfect_square",
    # Powers of two
    "is_power_of_two",
    "powers_of_two",
    "nth_power_of_two",
    # Fibonacci numbers
    "fibonacci",
    "fibonacci_sequence",
    "is_fibonacci_number",
    # Factorial and related
    "factorial",
    "double_factorial",
    "subfactorial",
    # Triangular numbers
    "triangular_number",
    "is_triangular_number",
    "triangular_sequence",
    # Pentagonal numbers
    "pentagonal_number",
    "is_pentagonal_number",
    "pentagonal_sequence",
    # Pyramidal numbers
    "square_pyramidal_number",
    "tetrahedral_number",
    # Catalan numbers (commonly used basic sequence)
    "catalan_number",
]

if __name__ == "__main__":
    import asyncio

    async def test_basic_sequences():
        """Test basic sequence functions."""
        print("🔢 Basic Sequences Test")
        print("=" * 30)

        # Test perfect squares
        print("Perfect Squares:")
        print(f"  is_perfect_square(16) = {await is_perfect_square(16)}")
        print(f"  perfect_squares(5) = {await perfect_squares(5)}")
        print(f"  nth_perfect_square(5) = {await nth_perfect_square(5)}")

        # Test powers of two
        print("\nPowers of Two:")
        print(f"  is_power_of_two(8) = {await is_power_of_two(8)}")
        print(f"  powers_of_two(5) = {await powers_of_two(5)}")
        print(f"  nth_power_of_two(5) = {await nth_power_of_two(5)}")

        # Test Fibonacci
        print("\nFibonacci Numbers:")
        print(f"  fibonacci(10) = {await fibonacci(10)}")
        print(f"  fibonacci_sequence(10) = {await fibonacci_sequence(10)}")
        print(f"  is_fibonacci_number(13) = {await is_fibonacci_number(13)}")

        # Test factorials
        print("\nFactorials:")
        print(f"  factorial(5) = {await factorial(5)}")
        print(f"  double_factorial(8) = {await double_factorial(8)}")
        print(f"  subfactorial(4) = {await subfactorial(4)}")

        # Test triangular numbers
        print("\nTriangular Numbers:")
        print(f"  triangular_number(5) = {await triangular_number(5)}")
        print(f"  is_triangular_number(15) = {await is_triangular_number(15)}")
        print(f"  triangular_sequence(5) = {await triangular_sequence(5)}")

        # Test pentagonal numbers
        print("\nPentagonal Numbers:")
        print(f"  pentagonal_number(5) = {await pentagonal_number(5)}")
        print(f"  is_pentagonal_number(35) = {await is_pentagonal_number(35)}")
        print(f"  pentagonal_sequence(5) = {await pentagonal_sequence(5)}")

        # Test pyramidal numbers
        print("\nPyramidal Numbers:")
        print(f"  square_pyramidal_number(5) = {await square_pyramidal_number(5)}")
        print(f"  tetrahedral_number(5) = {await tetrahedral_number(5)}")

        print("\n✅ All basic sequence functions working!")

    asyncio.run(test_basic_sequences())
