#!/usr/bin/env python3
# chuk_mcp_math/number_theory/partitions.py
"""
Integer Partitions and Additive Number Theory - Async Native (FIXED VERSION)

Functions for integer partitions, additive decompositions, and classical
problems in additive number theory including Goldbach conjecture and Waring's problem.

Functions:
- Integer partitions: partition_count, generate_partitions, restricted_partitions
- Partition variations: distinct_partitions, partitions_into_k_parts, conjugate_partition
- Goldbach: goldbach_conjecture_check, goldbach_pairs, weak_goldbach_check
- Sum representations: sum_of_two_squares, sum_of_four_squares, sum_of_cubes
- Waring's problem: waring_representation, sum_of_k_powers
- Additive bases: additive_basis_check, sidon_sets, sum_free_sets
"""

import math
import asyncio
from typing import List, Tuple, Optional
from chuk_mcp_math.mcp_decorator import mcp_function

# Import dependencies
from .primes import is_prime

# ============================================================================
# INTEGER PARTITIONS
# ============================================================================


@mcp_function(
    description="Count the number of integer partitions of n using dynamic programming.",
    namespace="arithmetic",
    category="partitions",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    estimated_cpu_usage="medium",
    examples=[
        {
            "input": {"n": 4},
            "output": 5,
            "description": "4 has 5 partitions: [4], [3,1], [2,2], [2,1,1], [1,1,1,1]",
        },
        {"input": {"n": 5}, "output": 7, "description": "5 has 7 partitions"},
        {"input": {"n": 10}, "output": 42, "description": "10 has 42 partitions"},
        {"input": {"n": 0}, "output": 1, "description": "0 has 1 partition (empty)"},
    ],
)
async def partition_count(n: int) -> int:
    """
    Count the number of integer partitions of n.

    A partition of n is a way of writing n as a sum of positive integers,
    where order doesn't matter.

    Args:
        n: Non-negative integer

    Returns:
        Number of partitions of n

    Examples:
        await partition_count(4) → 5     # [4], [3,1], [2,2], [2,1,1], [1,1,1,1]
        await partition_count(5) → 7
        await partition_count(10) → 42
    """
    if n < 0:
        return 0
    if n == 0:
        return 1

    # Dynamic programming approach
    dp = [0] * (n + 1)
    dp[0] = 1

    # For each number from 1 to n
    for i in range(1, n + 1):
        # Update all values that can include i
        for j in range(i, n + 1):
            dp[j] += dp[j - i]

        # Yield control every 50 iterations for large n
        if i % 50 == 0 and n > 100:
            await asyncio.sleep(0)

    return dp[n]


@mcp_function(
    description="Generate all integer partitions of n.",
    namespace="arithmetic",
    category="partitions",
    execution_modes=["local", "remote"],
    estimated_cpu_usage="high",
    examples=[
        {
            "input": {"n": 4},
            "output": [[4], [3, 1], [2, 2], [2, 1, 1], [1, 1, 1, 1]],
            "description": "All partitions of 4",
        },
        {
            "input": {"n": 5},
            "output": [
                [5],
                [4, 1],
                [3, 2],
                [3, 1, 1],
                [2, 2, 1],
                [2, 1, 1, 1],
                [1, 1, 1, 1, 1],
            ],
            "description": "All partitions of 5",
        },
        {
            "input": {"n": 3},
            "output": [[3], [2, 1], [1, 1, 1]],
            "description": "All partitions of 3",
        },
    ],
)
async def generate_partitions(n: int) -> List[List[int]]:
    """
    Generate all integer partitions of n.

    Args:
        n: Non-negative integer

    Returns:
        List of all partitions (each partition is a list of positive integers)

    Examples:
        await generate_partitions(4) → [[4], [3,1], [2,2], [2,1,1], [1,1,1,1]]
        await generate_partitions(5) → [[5], [4,1], [3,2], [3,1,1], [2,2,1], [2,1,1,1], [1,1,1,1,1]]
    """
    if n <= 0:
        return [[]] if n == 0 else []

    async def partition_helper(target: int, max_val: int) -> List[List[int]]:
        """Helper function for generating partitions."""
        if target == 0:
            return [[]]
        if target < 0 or max_val <= 0:
            return []

        result = []

        # Include max_val in partition
        for partition in await partition_helper(target - max_val, max_val):
            result.append([max_val] + partition)

        # Don't include max_val
        result.extend(await partition_helper(target, max_val - 1))

        return result

    partitions = await partition_helper(n, n)

    # Sort partitions for consistent ordering
    return sorted(partitions, reverse=True)


@mcp_function(
    description="Count partitions of n into at most k parts.",
    namespace="arithmetic",
    category="partitions",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    examples=[
        {
            "input": {"n": 6, "k": 3},
            "output": 7,
            "description": "6 into at most 3 parts: [6], [5,1], [4,2], [4,1,1], [3,3], [3,2,1], [2,2,2]",
        },
        {
            "input": {"n": 5, "k": 2},
            "output": 3,
            "description": "5 into at most 2 parts: [5], [4,1], [3,2]",
        },
        {
            "input": {"n": 4, "k": 4},
            "output": 5,
            "description": "Same as unrestricted partitions when k≥n",
        },
    ],
)
async def partitions_into_k_parts(n: int, k: int) -> int:
    """
    Count partitions of n into at most k parts.

    FIXED: Uses conjugate partition principle - partitions into at most k parts
    equals partitions with largest part at most k.

    Args:
        n: Non-negative integer to partition
        k: Maximum number of parts allowed

    Returns:
        Number of partitions of n using at most k parts

    Examples:
        await partitions_into_k_parts(6, 3) → 7    # At most 3 parts
        await partitions_into_k_parts(5, 2) → 3    # At most 2 parts
    """
    if n < 0 or k < 0:
        return 0
    if n == 0:
        return 1
    if k == 0:
        return 0

    # Use conjugate partition principle:
    # Partitions into at most k parts = partitions with largest part ≤ k
    return await restricted_partitions(n, k)


@mcp_function(
    description="Count partitions of n into distinct parts (no repeated values).",
    namespace="arithmetic",
    category="partitions",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    examples=[
        {
            "input": {"n": 6},
            "output": 4,
            "description": "6 into distinct parts: [6], [5,1], [4,2], [3,2,1]",
        },
        {"input": {"n": 9}, "output": 8, "description": "9 has 8 distinct partitions"},
        {
            "input": {"n": 4},
            "output": 2,
            "description": "4 into distinct parts: [4], [3,1]",
        },
    ],
)
async def distinct_partitions(n: int) -> int:
    """
    Count partitions of n into distinct parts.

    Args:
        n: Non-negative integer

    Returns:
        Number of partitions using distinct positive integers

    Examples:
        await distinct_partitions(6) → 4     # [6], [5,1], [4,2], [3,2,1]
        await distinct_partitions(9) → 8
        await distinct_partitions(4) → 2     # [4], [3,1]
    """
    if n < 0:
        return 0
    if n == 0:
        return 1

    # Use generating function approach
    dp = [0] * (n + 1)
    dp[0] = 1

    # For each possible part size
    for part in range(1, n + 1):
        # Update dp array (each part can be used at most once)
        for total in range(n, part - 1, -1):
            dp[total] += dp[total - part]

        # Yield control every 20 iterations
        if part % 20 == 0 and n > 50:
            await asyncio.sleep(0)

    return dp[n]


@mcp_function(
    description="Count restricted partitions where largest part is at most max_part.",
    namespace="arithmetic",
    category="partitions",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    examples=[
        {
            "input": {"n": 6, "max_part": 4},
            "output": 9,
            "description": "6 with parts ≤ 4",
        },
        {
            "input": {"n": 5, "max_part": 3},
            "output": 5,
            "description": "5 with parts ≤ 3",
        },
        {
            "input": {"n": 10, "max_part": 2},
            "output": 6,
            "description": "10 with parts ≤ 2: multiple ways using 1s and 2s",
        },
    ],
)
async def restricted_partitions(n: int, max_part: int) -> int:
    """
    Count partitions where largest part is at most max_part.

    Args:
        n: Non-negative integer to partition
        max_part: Maximum allowed part size

    Returns:
        Number of restricted partitions

    Examples:
        await restricted_partitions(6, 4) → 9    # Parts ≤ 4
        await restricted_partitions(5, 3) → 5    # Parts ≤ 3
        await restricted_partitions(10, 2) → 6   # Using only 1s and 2s
    """
    if n < 0 or max_part <= 0:
        return 0
    if n == 0:
        return 1

    # Dynamic programming
    dp = [0] * (n + 1)
    dp[0] = 1

    # For each allowed part size
    for part in range(1, min(max_part + 1, n + 1)):
        for total in range(part, n + 1):
            dp[total] += dp[total - part]

        # Yield control every 20 iterations
        if part % 20 == 0 and max_part > 50:
            await asyncio.sleep(0)

    return dp[n]


# ============================================================================
# GOLDBACH CONJECTURE
# ============================================================================


@mcp_function(
    description="Check Goldbach conjecture: every even integer > 2 is sum of two primes.",
    namespace="arithmetic",
    category="additive_number_theory",
    execution_modes=["local", "remote"],
    estimated_cpu_usage="medium",
    examples=[
        {"input": {"n": 4}, "output": [2, 2], "description": "4 = 2 + 2"},
        {"input": {"n": 6}, "output": [3, 3], "description": "6 = 3 + 3"},
        {"input": {"n": 10}, "output": [3, 7], "description": "10 = 3 + 7"},
        {"input": {"n": 20}, "output": [3, 17], "description": "20 = 3 + 17"},
    ],
)
async def goldbach_conjecture_check(n: int) -> Optional[Tuple[int, int]]:
    """
    Find two primes that sum to n (Goldbach conjecture verification).

    Args:
        n: Even integer > 2

    Returns:
        Tuple of two primes that sum to n, or None if not found

    Examples:
        await goldbach_conjecture_check(4) → (2, 2)
        await goldbach_conjecture_check(6) → (3, 3)
        await goldbach_conjecture_check(10) → (3, 7)
    """
    if n <= 2 or n % 2 != 0:
        return None

    # Check all primes up to n/2
    for p in range(2, n // 2 + 1):
        if await is_prime(p):
            q = n - p
            if await is_prime(q):
                return (p, q)

        # Yield control every 100 iterations for large n
        if p % 100 == 0 and n > 1000:
            await asyncio.sleep(0)

    return None


@mcp_function(
    description="Find all Goldbach pairs for an even number.",
    namespace="arithmetic",
    category="additive_number_theory",
    execution_modes=["local", "remote"],
    estimated_cpu_usage="medium",
    examples=[
        {
            "input": {"n": 10},
            "output": [[3, 7], [5, 5]],
            "description": "10 = 3+7 = 5+5",
        },
        {
            "input": {"n": 20},
            "output": [[3, 17], [7, 13]],
            "description": "20 = 3+17 = 7+13",
        },
        {"input": {"n": 12}, "output": [[5, 7]], "description": "12 = 5+7"},
    ],
)
async def goldbach_pairs(n: int) -> List[Tuple[int, int]]:
    """
    Find all ways to express n as sum of two primes.

    Args:
        n: Even integer > 2

    Returns:
        List of all prime pairs that sum to n

    Examples:
        await goldbach_pairs(10) → [(3, 7), (5, 5)]
        await goldbach_pairs(20) → [(3, 17), (7, 13)]
        await goldbach_pairs(12) → [(5, 7)]
    """
    if n <= 2 or n % 2 != 0:
        return []

    pairs = []

    for p in range(2, n // 2 + 1):
        if await is_prime(p):
            q = n - p
            if await is_prime(q) and p <= q:  # Avoid duplicates
                pairs.append((p, q))

        # Yield control every 100 iterations
        if p % 100 == 0 and n > 1000:
            await asyncio.sleep(0)

    return pairs


@mcp_function(
    description="Check weak Goldbach conjecture: every odd integer > 5 is sum of three primes.",
    namespace="arithmetic",
    category="additive_number_theory",
    execution_modes=["local", "remote"],
    estimated_cpu_usage="high",
    examples=[
        {"input": {"n": 7}, "output": [2, 2, 3], "description": "7 = 2 + 2 + 3"},
        {"input": {"n": 9}, "output": [3, 3, 3], "description": "9 = 3 + 3 + 3"},
        {"input": {"n": 11}, "output": [3, 3, 5], "description": "11 = 3 + 3 + 5"},
        {"input": {"n": 15}, "output": [3, 5, 7], "description": "15 = 3 + 5 + 7"},
    ],
)
async def weak_goldbach_check(n: int) -> Optional[Tuple[int, int, int]]:
    """
    Find three primes that sum to n (weak Goldbach conjecture).

    Args:
        n: Odd integer > 5

    Returns:
        Tuple of three primes that sum to n, or None if not found

    Examples:
        await weak_goldbach_check(7) → (2, 2, 3)
        await weak_goldbach_check(9) → (3, 3, 3)
        await weak_goldbach_check(11) → (3, 3, 5)
    """
    if n <= 5 or n % 2 == 0:
        return None

    # Try all combinations of three primes
    max_prime = n - 4  # Since we need at least 2+2+p

    for p1 in range(2, max_prime + 1):
        if await is_prime(p1):
            for p2 in range(p1, n - p1 - 1):
                if await is_prime(p2):
                    p3 = n - p1 - p2
                    if p3 >= p2 and await is_prime(p3):
                        return (p1, p2, p3)

        # Yield control every 50 iterations
        if p1 % 50 == 0 and n > 500:
            await asyncio.sleep(0)

    return None


# ============================================================================
# SUM OF SQUARES
# ============================================================================


@mcp_function(
    description="Express a number as sum of two squares if possible.",
    namespace="arithmetic",
    category="additive_number_theory",
    execution_modes=["local", "remote"],
    estimated_cpu_usage="medium",
    examples=[
        {"input": {"n": 5}, "output": [1, 2], "description": "5 = 1² + 2²"},
        {"input": {"n": 13}, "output": [2, 3], "description": "13 = 2² + 3²"},
        {"input": {"n": 25}, "output": [3, 4], "description": "25 = 3² + 4²"},
        {
            "input": {"n": 3},
            "output": None,
            "description": "3 cannot be written as sum of two squares",
        },
    ],
)
async def sum_of_two_squares(n: int) -> Optional[Tuple[int, int]]:
    """
    Express n as sum of two squares if possible.

    Args:
        n: Non-negative integer

    Returns:
        Tuple (a, b) such that n = a² + b², or None if impossible

    Examples:
        await sum_of_two_squares(5) → (1, 2)     # 5 = 1² + 2²
        await sum_of_two_squares(13) → (2, 3)    # 13 = 2² + 3²
        await sum_of_two_squares(3) → None       # Impossible
    """
    if n < 0:
        return None
    if n == 0:
        return (0, 0)

    max_a = int(math.sqrt(n))

    for a in range(max_a + 1):
        b_squared = n - a * a
        if b_squared < 0:
            break

        b = int(math.sqrt(b_squared))
        if b * b == b_squared:
            return (a, b)

        # Yield control every 100 iterations for large n
        if a % 100 == 0 and max_a > 1000:
            await asyncio.sleep(0)

    return None


@mcp_function(
    description="Express a number as sum of four squares (Lagrange's theorem).",
    namespace="arithmetic",
    category="additive_number_theory",
    execution_modes=["local", "remote"],
    estimated_cpu_usage="high",
    examples=[
        {
            "input": {"n": 7},
            "output": [1, 1, 1, 2],
            "description": "7 = 1² + 1² + 1² + 2²",
        },
        {
            "input": {"n": 15},
            "output": [1, 1, 2, 3],
            "description": "15 = 1² + 1² + 2² + 3²",
        },
        {
            "input": {"n": 12},
            "output": [2, 2, 2, 0],
            "description": "12 = 2² + 2² + 2² + 0²",
        },
    ],
)
async def sum_of_four_squares(n: int) -> Optional[Tuple[int, int, int, int]]:
    """
    Express n as sum of four squares (always possible by Lagrange's theorem).

    Args:
        n: Non-negative integer

    Returns:
        Tuple (a, b, c, d) such that n = a² + b² + c² + d²

    Examples:
        await sum_of_four_squares(7) → (1, 1, 1, 2)   # 7 = 1² + 1² + 1² + 2²
        await sum_of_four_squares(15) → (1, 1, 2, 3)  # 15 = 1² + 1² + 2² + 3²
    """
    if n < 0:
        return None
    if n == 0:
        return (0, 0, 0, 0)

    max_val = int(math.sqrt(n))

    for a in range(max_val + 1):
        remainder_a = n - a * a
        if remainder_a < 0:
            break

        for b in range(int(math.sqrt(remainder_a)) + 1):
            remainder_b = remainder_a - b * b
            if remainder_b < 0:
                break

            for c in range(int(math.sqrt(remainder_b)) + 1):
                remainder_c = remainder_b - c * c
                if remainder_c < 0:
                    break

                d = int(math.sqrt(remainder_c))
                if d * d == remainder_c:
                    return (a, b, c, d)

        # Yield control every 50 iterations
        if a % 50 == 0 and max_val > 500:
            await asyncio.sleep(0)

    return None


# ============================================================================
# WARING'S PROBLEM
# ============================================================================


@mcp_function(
    description="Express a number as sum of k-th powers (Waring's problem).",
    namespace="arithmetic",
    category="additive_number_theory",
    execution_modes=["local", "remote"],
    estimated_cpu_usage="high",
    examples=[
        {
            "input": {"n": 23, "k": 3},
            "output": [2, 2, 1, 1, 1],
            "description": "23 = 2³ + 2³ + 1³ + 1³ + 1³",
        },
        {"input": {"n": 16, "k": 4}, "output": [2, 0, 0, 0], "description": "16 = 2⁴"},
        {
            "input": {"n": 30, "k": 3},
            "output": [3, 1, 1, 1],
            "description": "30 = 3³ + 1³ + 1³ + 1³",
        },
    ],
)
async def waring_representation(n: int, k: int) -> Optional[List[int]]:
    """
    Express n as sum of k-th powers.

    Args:
        n: Non-negative integer
        k: Power (≥ 2)

    Returns:
        List of integers whose k-th powers sum to n, or None if not found

    Examples:
        await waring_representation(23, 3) → [2, 2, 1, 1, 1]  # 23 = 2³+2³+1³+1³+1³
        await waring_representation(16, 4) → [2, 0, 0, 0]     # 16 = 2⁴
    """
    if n < 0 or k < 2:
        return None
    if n == 0:
        return [0]

    # Use dynamic programming with backtracking
    max_base = int(n ** (1.0 / k)) + 1

    # Generate all k-th powers up to n
    powers = []
    for i in range(1, max_base):
        power = i**k
        if power > n:
            break
        powers.append((i, power))

    if not powers:
        return None

    # Try to find representation using greedy approach
    result = []
    remaining = n

    while remaining > 0:
        # Find largest power ≤ remaining
        found = False
        for base, power in reversed(powers):
            if power <= remaining:
                result.append(base)
                remaining -= power
                found = True
                break

        if not found:
            # Fallback: use 1^k repeatedly
            if k == 2 or k == 3:  # For squares and cubes, this always works
                result.extend([1] * remaining)
                break
            else:
                return None  # Might not have a solution with our greedy approach

        # Yield control every 100 iterations
        if len(result) % 100 == 0 and len(result) > 100:
            await asyncio.sleep(0)

    return sorted(result, reverse=True)


@mcp_function(
    description="Find minimum number of k-th powers needed to represent n.",
    namespace="arithmetic",
    category="additive_number_theory",
    execution_modes=["local", "remote"],
    estimated_cpu_usage="high",
    examples=[
        {
            "input": {"n": 23, "k": 3},
            "output": 5,
            "description": "23 needs 5 cubes minimum",
        },
        {
            "input": {"n": 13, "k": 2},
            "output": 2,
            "description": "13 = 2² + 3² needs 2 squares",
        },
        {
            "input": {"n": 16, "k": 4},
            "output": 1,
            "description": "16 = 2⁴ needs 1 fourth power",
        },
    ],
)
async def min_waring_number(n: int, k: int) -> Optional[int]:
    """
    Find minimum number of k-th powers needed to sum to n.

    Args:
        n: Non-negative integer
        k: Power (≥ 2)

    Returns:
        Minimum number of k-th powers needed, or None if impossible

    Examples:
        await min_waring_number(23, 3) → 5    # Need 5 cubes
        await min_waring_number(13, 2) → 2    # Need 2 squares
        await min_waring_number(16, 4) → 1    # Need 1 fourth power
    """
    if n < 0 or k < 2:
        return None
    if n == 0:
        return 0

    # Dynamic programming approach
    # dp[i] = minimum number of k-th powers to make i
    dp = [float("inf")] * (n + 1)
    dp[0] = 0

    # Generate k-th powers
    powers = []
    base = 1
    while True:
        power = base**k
        if power > n:
            break
        powers.append(power)
        base += 1

    # Fill dp table
    for i in range(1, n + 1):
        for power in powers:
            if power > i:
                break
            dp[i] = min(dp[i], dp[i - power] + 1)

        # Yield control every 1000 iterations
        if i % 1000 == 0 and n > 5000:
            await asyncio.sleep(0)

    return dp[n] if dp[n] != float("inf") else None  # type: ignore[return-value]


# ============================================================================
# ADDITIVE BASES AND SPECIAL SETS
# ============================================================================


@mcp_function(
    description="Check if a set forms an additive basis for integers up to limit.",
    namespace="arithmetic",
    category="additive_number_theory",
    execution_modes=["local", "remote"],
    estimated_cpu_usage="medium",
    examples=[
        {
            "input": {"basis": [1, 2], "limit": 10},
            "output": True,
            "description": "{1,2} can represent all integers 1-10",
        },
        {
            "input": {"basis": [2, 3], "limit": 10},
            "output": False,
            "description": "{2,3} cannot represent 1",
        },
        {
            "input": {"basis": [1, 3, 5], "limit": 20},
            "output": True,
            "description": "{1,3,5} represents all integers 1-20",
        },
    ],
)
async def is_additive_basis(basis: List[int], limit: int, max_terms: int = 10) -> bool:
    """
    Check if a set can represent all integers up to limit as sums.

    Args:
        basis: List of positive integers
        limit: Upper limit to check
        max_terms: Maximum number of terms allowed in sum

    Returns:
        True if basis can represent all integers 1 to limit

    Examples:
        await is_additive_basis([1, 2], 10) → True
        await is_additive_basis([2, 3], 10) → False  # Can't make 1
    """
    if not basis or limit <= 0:
        return False

    basis = sorted(set(b for b in basis if b > 0))

    # Use dynamic programming
    representable = [False] * (limit + 1)
    representable[0] = True

    # For each number of terms
    for terms in range(1, max_terms + 1):
        new_representable = representable[:]

        for i in range(1, limit + 1):
            if not new_representable[i]:
                for b in basis:
                    if i >= b and representable[i - b]:
                        new_representable[i] = True
                        break

        representable = new_representable

        # Check if all numbers 1 to limit are representable
        if all(representable[i] for i in range(1, limit + 1)):
            return True

        # Yield control every few terms
        if terms % 3 == 0:
            await asyncio.sleep(0)

    return all(representable[i] for i in range(1, limit + 1))


@mcp_function(
    description="Generate a Sidon set (B_h sequence) - no two distinct sums are equal.",
    namespace="arithmetic",
    category="additive_number_theory",
    execution_modes=["local", "remote"],
    estimated_cpu_usage="high",
    examples=[
        {
            "input": {"limit": 20, "size": 5},
            "output": [1, 2, 5, 11, 16],
            "description": "Sidon set of size 5 with elements ≤ 20",
        },
        {
            "input": {"limit": 15, "size": 4},
            "output": [1, 2, 5, 11],
            "description": "Sidon set of size 4 with elements ≤ 15",
        },
    ],
)
async def generate_sidon_set(limit: int, size: int) -> Optional[List[int]]:
    """
    Generate a Sidon set of given size with elements up to limit.

    A Sidon set has the property that all pairwise sums are distinct.
    FIXED: Better algorithm for checking Sidon property.

    Args:
        limit: Maximum element value
        size: Desired size of Sidon set

    Returns:
        Sidon set of requested size, or None if not possible

    Examples:
        await generate_sidon_set(20, 5) → [1, 2, 5, 11, 16]
        await generate_sidon_set(15, 4) → [1, 2, 5, 11]
    """
    if size <= 0 or limit <= 0:
        return None

    def is_valid_sidon_addition(current_set: List[int], candidate: int) -> bool:
        """Check if adding candidate maintains Sidon property."""
        if not current_set:
            return True

        # Get all existing pairwise sums (including self-sums)
        existing_sums = set()
        for i in range(len(current_set)):
            for j in range(i, len(current_set)):
                existing_sums.add(current_set[i] + current_set[j])

        # Check new sums involving candidate
        for existing in current_set:
            new_sum = candidate + existing
            if new_sum in existing_sums:
                return False

        # Check candidate with itself
        self_sum = candidate + candidate
        if self_sum in existing_sums:
            return False

        return True

    # Greedy construction
    sidon_set: list[int] = []

    for candidate in range(1, limit + 1):
        if is_valid_sidon_addition(sidon_set, candidate):
            sidon_set.append(candidate)

            if len(sidon_set) == size:
                return sidon_set

        # Yield control every 100 candidates
        if candidate % 100 == 0:
            await asyncio.sleep(0)

    return sidon_set if len(sidon_set) == size else None


# Export all functions
__all__ = [
    # Integer partitions
    "partition_count",
    "generate_partitions",
    "partitions_into_k_parts",
    "distinct_partitions",
    "restricted_partitions",
    # Goldbach conjecture
    "goldbach_conjecture_check",
    "goldbach_pairs",
    "weak_goldbach_check",
    # Sum of squares
    "sum_of_two_squares",
    "sum_of_four_squares",
    # Waring's problem
    "waring_representation",
    "min_waring_number",
    # Additive bases
    "is_additive_basis",
    "generate_sidon_set",
]
