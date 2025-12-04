#!/usr/bin/env python3
# chuk_mcp_math/number_theory/sieve_algorithms.py
"""
Sieve Algorithms - Async Native - COMPLETE IMPLEMENTATION

Classical and modern sieve algorithms for efficient prime generation and counting.
Essential for number theory research, cryptography, and computational mathematics.

Functions:
- Classical sieves: sieve_of_eratosthenes, sieve_of_sundaram, sieve_of_atkin
- Segmented sieves: segmented_sieve, wheel_sieve
- Counting functions: prime_counting_sieve, mertens_function_sieve
- Optimized sieves: linear_sieve, incremental_sieve
- Analysis: sieve_performance_analysis, prime_gap_sieve

Mathematical Background:
Sieve algorithms systematically eliminate composite numbers to find primes.
The Sieve of Eratosthenes is the classical algorithm, while modern variants
like segmented sieves and wheel sieves provide memory and speed optimizations.
"""

import asyncio
import math
from typing import List, Dict
from chuk_mcp_math.mcp_decorator import mcp_function

# ============================================================================
# CLASSICAL SIEVE ALGORITHMS
# ============================================================================


@mcp_function(
    description="Sieve of Eratosthenes - the classical algorithm for finding all primes up to n.",
    namespace="arithmetic",
    category="sieve_algorithms",
    execution_modes=["local", "remote"],
    estimated_cpu_usage="medium",
    examples=[
        {
            "input": {"limit": 30},
            "output": [2, 3, 5, 7, 11, 13, 17, 19, 23, 29],
            "description": "All primes up to 30",
        },
        {
            "input": {"limit": 10},
            "output": [2, 3, 5, 7],
            "description": "Primes up to 10",
        },
        {"input": {"limit": 2}, "output": [2], "description": "Only prime 2"},
        {"input": {"limit": 1}, "output": [], "description": "No primes up to 1"},
    ],
)
async def sieve_of_eratosthenes(limit: int) -> List[int]:
    """
    Find all prime numbers up to a given limit using the Sieve of Eratosthenes.

    This is the classical algorithm that systematically eliminates multiples
    of each prime, starting with 2.

    Args:
        limit: Upper bound (inclusive) for prime search

    Returns:
        List of all prime numbers ≤ limit

    Time complexity: O(n log log n)
    Space complexity: O(n)

    Examples:
        await sieve_of_eratosthenes(30) → [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
        await sieve_of_eratosthenes(10) → [2, 3, 5, 7]
    """
    if limit < 2:
        return []

    # Initialize boolean array - True means potentially prime
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False

    # Yield control for large sieves
    if limit > 100000:
        await asyncio.sleep(0)

    # Sieve process
    for i in range(2, int(math.sqrt(limit)) + 1):
        if is_prime[i]:
            # Mark multiples of i as composite
            for j in range(i * i, limit + 1, i):
                is_prime[j] = False

            # Yield control every 1000 iterations for very large sieves
            if i % 1000 == 0 and limit > 1000000:
                await asyncio.sleep(0)

    # Collect primes
    primes = []
    for i in range(2, limit + 1):
        if is_prime[i]:
            primes.append(i)

    return primes


@mcp_function(
    description="Sieve of Sundaram - finds odd primes by eliminating specific composite patterns.",
    namespace="arithmetic",
    category="sieve_algorithms",
    execution_modes=["local", "remote"],
    estimated_cpu_usage="medium",
    examples=[
        {
            "input": {"limit": 30},
            "output": [2, 3, 5, 7, 11, 13, 17, 19, 23, 29],
            "description": "All primes up to 30",
        },
        {
            "input": {"limit": 20},
            "output": [2, 3, 5, 7, 11, 13, 17, 19],
            "description": "Primes up to 20",
        },
        {
            "input": {"limit": 10},
            "output": [2, 3, 5, 7],
            "description": "Primes up to 10",
        },
        {"input": {"limit": 3}, "output": [2, 3], "description": "First two primes"},
    ],
)
async def sieve_of_sundaram(limit: int) -> List[int]:
    """
    Find all prime numbers up to limit using the Sieve of Sundaram.

    This sieve works by eliminating numbers of the form i + j + 2ij
    where 1 ≤ i ≤ j, then converting remaining numbers to primes.

    Args:
        limit: Upper bound (inclusive) for prime search

    Returns:
        List of all prime numbers ≤ limit

    Time complexity: O(n log n)
    Space complexity: O(n)

    Examples:
        await sieve_of_sundaram(30) → [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
        await sieve_of_sundaram(20) → [2, 3, 5, 7, 11, 13, 17, 19]
    """
    if limit < 2:
        return []
    if limit == 2:
        return [2]

    # Calculate upper bound for Sundaram sieve
    n = (limit - 1) // 2

    # Initialize array - True means the number will generate a prime
    marked = [True] * (n + 1)

    # Yield control for large sieves
    if limit > 100000:
        await asyncio.sleep(0)

    # Mark numbers to be eliminated
    for i in range(1, n + 1):
        j = i
        while i + j + 2 * i * j <= n:
            marked[i + j + 2 * i * j] = False
            j += 1

        # Yield control every 1000 iterations for large sieves
        if i % 1000 == 0 and limit > 1000000:
            await asyncio.sleep(0)

    # Generate primes
    primes = [2]  # 2 is the only even prime
    for i in range(1, n + 1):
        if marked[i]:
            prime = 2 * i + 1
            if prime <= limit:
                primes.append(prime)

    return primes


@mcp_function(
    description="Sieve of Atkin - modern sieve with better asymptotic complexity.",
    namespace="arithmetic",
    category="sieve_algorithms",
    execution_modes=["local", "remote"],
    estimated_cpu_usage="high",
    examples=[
        {
            "input": {"limit": 30},
            "output": [2, 3, 5, 7, 11, 13, 17, 19, 23, 29],
            "description": "All primes up to 30",
        },
        {
            "input": {"limit": 50},
            "output": [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47],
            "description": "Primes up to 50",
        },
        {
            "input": {"limit": 10},
            "output": [2, 3, 5, 7],
            "description": "Primes up to 10",
        },
        {
            "input": {"limit": 100},
            "output": "25 primes found",
            "description": "Count for limit 100",
        },
    ],
)
async def sieve_of_atkin(limit: int) -> List[int]:
    """
    Find all prime numbers up to limit using the Sieve of Atkin.

    Modern sieve algorithm with better theoretical complexity O(n/log log n).
    Uses quadratic forms to identify potential primes.

    Args:
        limit: Upper bound (inclusive) for prime search

    Returns:
        List of all prime numbers ≤ limit

    Time complexity: O(n/log log n)
    Space complexity: O(n)

    Examples:
        await sieve_of_atkin(30) → [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
        await sieve_of_atkin(50) → [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    """
    if limit < 2:
        return []

    # Initialize sieve array
    sieve = [False] * (limit + 1)

    # Yield control for large sieves
    if limit > 100000:
        await asyncio.sleep(0)

    # Main sieving using quadratic forms
    sqrt_limit = int(math.sqrt(limit))

    # Form 4x² + y² = n (n ≡ 1 mod 4)
    for x in range(1, sqrt_limit + 1):
        for y in range(1, sqrt_limit + 1):
            n = 4 * x * x + y * y
            if n <= limit and (n % 12 == 1 or n % 12 == 5):
                sieve[n] = not sieve[n]

        # Yield control every 100 iterations for large sieves
        if x % 100 == 0 and limit > 1000000:
            await asyncio.sleep(0)

    # Form 3x² + y² = n (n ≡ 7 mod 12)
    for x in range(1, sqrt_limit + 1):
        for y in range(1, sqrt_limit + 1):
            n = 3 * x * x + y * y
            if n <= limit and n % 12 == 7:
                sieve[n] = not sieve[n]

    # Form 3x² - y² = n (n ≡ 11 mod 12, x > y)
    for x in range(1, sqrt_limit + 1):
        for y in range(1, x):
            n = 3 * x * x - y * y
            if n <= limit and n % 12 == 11:
                sieve[n] = not sieve[n]

    # Mark squares of primes as composite
    for i in range(5, sqrt_limit + 1):
        if sieve[i]:
            square = i * i
            for j in range(square, limit + 1, square):
                sieve[j] = False

    # Collect primes
    primes = []
    if limit >= 2:
        primes.append(2)
    if limit >= 3:
        primes.append(3)

    for i in range(5, limit + 1):
        if sieve[i]:
            primes.append(i)

    return primes


# ============================================================================
# SEGMENTED SIEVES
# ============================================================================


@mcp_function(
    description="Segmented sieve for finding primes in a range [low, high] with memory efficiency.",
    namespace="arithmetic",
    category="sieve_algorithms",
    execution_modes=["local", "remote"],
    estimated_cpu_usage="high",
    examples=[
        {
            "input": {"low": 10, "high": 30},
            "output": [11, 13, 17, 19, 23, 29],
            "description": "Primes between 10 and 30",
        },
        {
            "input": {"low": 100, "high": 120},
            "output": [101, 103, 107, 109, 113],
            "description": "Primes between 100 and 120",
        },
        {
            "input": {"low": 1000, "high": 1020},
            "output": [1009, 1013, 1019],
            "description": "Primes between 1000 and 1020",
        },
        {
            "input": {"low": 2, "high": 10},
            "output": [2, 3, 5, 7],
            "description": "Small range",
        },
    ],
)
async def segmented_sieve(low: int, high: int) -> List[int]:
    """
    Find all primes in range [low, high] using segmented sieve.

    Memory-efficient algorithm that processes numbers in segments,
    using only O(√high) memory regardless of range size.

    Args:
        low: Lower bound (inclusive)
        high: Upper bound (inclusive)

    Returns:
        List of prime numbers in [low, high]

    Time complexity: O((high-low+1) log log high)
    Space complexity: O(√high)

    Examples:
        await segmented_sieve(10, 30) → [11, 13, 17, 19, 23, 29]
        await segmented_sieve(100, 120) → [101, 103, 107, 109, 113]
    """
    if low > high or high < 2:
        return []

    # Adjust low to be at least 2
    low = max(low, 2)

    # Find all primes up to √high using basic sieve
    sqrt_high = int(math.sqrt(high))
    base_primes = await sieve_of_eratosthenes(sqrt_high)

    # Yield control for large ranges
    if high - low > 100000:
        await asyncio.sleep(0)

    # Segment size for memory efficiency
    segment_size = max(sqrt_high, 32 * 1024)  # At least 32KB
    primes_in_range = []

    # Process segments
    segment_low = low
    while segment_low <= high:
        segment_high = min(segment_low + segment_size - 1, high)

        # Create segment array
        segment = [True] * (segment_high - segment_low + 1)

        # Sieve with base primes
        for prime in base_primes:
            # Find first multiple of prime in segment
            start = max(prime * prime, ((segment_low + prime - 1) // prime) * prime)

            # Mark multiples as composite
            for j in range(start, segment_high + 1, prime):
                segment[j - segment_low] = False

        # Collect primes from this segment
        for i in range(len(segment)):
            if segment[i]:
                candidate = segment_low + i
                if candidate >= low:  # Ensure within original range
                    primes_in_range.append(candidate)

        segment_low = segment_high + 1

        # Yield control every segment for large ranges
        if high - low > 1000000:
            await asyncio.sleep(0)

    return primes_in_range


@mcp_function(
    description="Wheel sieve using wheel factorization for improved efficiency.",
    namespace="arithmetic",
    category="sieve_algorithms",
    execution_modes=["local", "remote"],
    estimated_cpu_usage="medium",
    examples=[
        {
            "input": {"limit": 30, "wheel_primes": [2, 3]},
            "output": [2, 3, 5, 7, 11, 13, 17, 19, 23, 29],
            "description": "2,3-wheel sieve up to 30",
        },
        {
            "input": {"limit": 50, "wheel_primes": [2, 3, 5]},
            "output": [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47],
            "description": "2,3,5-wheel sieve up to 50",
        },
        {
            "input": {"limit": 100, "wheel_primes": [2]},
            "output": "25 primes found",
            "description": "2-wheel (odd numbers only)",
        },
        {
            "input": {"limit": 20, "wheel_primes": [2, 3]},
            "output": [2, 3, 5, 7, 11, 13, 17, 19],
            "description": "Small range with 2,3-wheel",
        },
    ],
)
async def wheel_sieve(limit: int, wheel_primes: List[int] = None) -> List[int]:  # type: ignore[assignment]
    """
    Sieve using wheel factorization to skip multiples of small primes.

    Reduces the number of candidates by pre-eliminating multiples
    of small primes (the "wheel").

    Args:
        limit: Upper bound (inclusive) for prime search
        wheel_primes: Small primes to use for wheel (default: [2, 3])

    Returns:
        List of all prime numbers ≤ limit

    Examples:
        await wheel_sieve(30, [2, 3]) → [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
        await wheel_sieve(50, [2, 3, 5]) → [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    """
    if limit < 2:
        return []

    if wheel_primes is None:
        wheel_primes = [2, 3]

    # Include wheel primes in result if they're ≤ limit
    primes = [p for p in wheel_primes if p <= limit]

    if limit <= max(wheel_primes):
        return sorted(primes)

    # Calculate wheel size and generate wheel pattern
    wheel_size = 1
    for p in wheel_primes:
        wheel_size *= p

    # Generate candidates that are coprime to wheel primes
    wheel_candidates = []
    for i in range(1, wheel_size + 1):
        if all(i % p != 0 for p in wheel_primes):
            wheel_candidates.append(i)

    # Yield control for large wheels
    if wheel_size > 1000:
        await asyncio.sleep(0)

    # Generate all candidates up to limit
    candidates = set()
    base = 0
    while base <= limit:
        for offset in wheel_candidates:
            candidate = base + offset
            if candidate > limit:
                break
            if candidate > max(wheel_primes):
                candidates.add(candidate)
        base += wheel_size

        # Yield control for large limits
        if base % 1000000 == 0:
            await asyncio.sleep(0)

    # Sieve the candidates
    candidates = sorted(candidates)  # type: ignore[assignment]
    is_prime = {c: True for c in candidates}

    sqrt_limit = int(math.sqrt(limit))
    for candidate in candidates:
        if candidate > sqrt_limit:
            break
        if is_prime[candidate]:
            # Mark multiples as composite
            for multiple in range(candidate * candidate, limit + 1, candidate):
                if multiple in is_prime:
                    is_prime[multiple] = False

    # Collect remaining primes
    remaining_primes = [c for c in candidates if is_prime[c]]

    return sorted(primes + remaining_primes)


# ============================================================================
# COUNTING AND ANALYSIS FUNCTIONS
# ============================================================================


@mcp_function(
    description="Count primes up to limit using optimized sieve (returns count, not list).",
    namespace="arithmetic",
    category="sieve_algorithms",
    execution_modes=["local", "remote"],
    estimated_cpu_usage="medium",
    examples=[
        {"input": {"limit": 100}, "output": 25, "description": "π(100) = 25"},
        {"input": {"limit": 1000}, "output": 168, "description": "π(1000) = 168"},
        {"input": {"limit": 10}, "output": 4, "description": "π(10) = 4"},
        {"input": {"limit": 2}, "output": 1, "description": "π(2) = 1"},
    ],
)
async def prime_counting_sieve(limit: int) -> int:
    """
    Count the number of primes ≤ limit using memory-optimized sieve.

    More memory-efficient than generating the full list when only
    the count is needed (prime counting function π(n)).

    Args:
        limit: Upper bound (inclusive)

    Returns:
        Number of primes ≤ limit

    Examples:
        await prime_counting_sieve(100) → 25
        await prime_counting_sieve(1000) → 168
        await prime_counting_sieve(10) → 4
    """
    if limit < 2:
        return 0

    # Use bit array for memory efficiency
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False

    # Yield control for large sieves
    if limit > 100000:
        await asyncio.sleep(0)

    count = 0
    sqrt_limit = int(math.sqrt(limit))

    for i in range(2, sqrt_limit + 1):
        if is_prime[i]:
            # Mark multiples as composite
            for j in range(i * i, limit + 1, i):
                is_prime[j] = False

        # Yield control for large sieves
        if i % 1000 == 0 and limit > 1000000:
            await asyncio.sleep(0)

    # Count primes
    for i in range(2, limit + 1):
        if is_prime[i]:
            count += 1

    return count


@mcp_function(
    description="Calculate Mertens function M(n) = Σμ(k) for k=1 to n using sieve.",
    namespace="arithmetic",
    category="sieve_algorithms",
    execution_modes=["local", "remote"],
    estimated_cpu_usage="high",
    examples=[
        {"input": {"n": 10}, "output": -1, "description": "M(10) = -1"},
        {"input": {"n": 20}, "output": -2, "description": "M(20) = -2"},
        {"input": {"n": 5}, "output": -1, "description": "M(5) = -1"},
        {"input": {"n": 100}, "output": 1, "description": "M(100) = 1"},
    ],
)
async def mertens_function_sieve(n: int) -> int:
    """
    Calculate the Mertens function M(n) = Σ μ(k) for k=1 to n.

    Uses sieve to efficiently compute Möbius function values
    and their cumulative sum.

    Args:
        n: Upper bound for summation

    Returns:
        Value of Mertens function M(n)

    Examples:
        await mertens_function_sieve(10) → -1
        await mertens_function_sieve(20) → -2
        await mertens_function_sieve(100) → 1
    """
    if n < 1:
        return 0

    # Initialize Möbius function values
    mu = [1] * (n + 1)

    # Yield control for large n
    if n > 100000:
        await asyncio.sleep(0)

    # Sieve to compute Möbius function
    for i in range(2, n + 1):
        if mu[i] == 1:  # i is prime
            # Update multiples of i
            for j in range(i, n + 1, i):
                mu[j] *= -1

            # Update multiples of i²
            square = i * i
            for j in range(square, n + 1, square):
                mu[j] = 0

        # Yield control every 1000 iterations
        if i % 1000 == 0 and n > 100000:
            await asyncio.sleep(0)

    # Calculate Mertens function (cumulative sum)
    mertens_value = sum(mu[1 : n + 1])
    return mertens_value


# ============================================================================
# OPTIMIZED SIEVES
# ============================================================================


@mcp_function(
    description="Linear time sieve (Sieve of Euler) - each composite marked exactly once.",
    namespace="arithmetic",
    category="sieve_algorithms",
    execution_modes=["local", "remote"],
    estimated_cpu_usage="medium",
    examples=[
        {
            "input": {"limit": 30},
            "output": [2, 3, 5, 7, 11, 13, 17, 19, 23, 29],
            "description": "Linear sieve up to 30",
        },
        {
            "input": {"limit": 20},
            "output": [2, 3, 5, 7, 11, 13, 17, 19],
            "description": "Linear sieve up to 20",
        },
        {
            "input": {"limit": 10},
            "output": [2, 3, 5, 7],
            "description": "Linear sieve up to 10",
        },
        {
            "input": {"limit": 50},
            "output": [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47],
            "description": "Linear sieve up to 50",
        },
    ],
)
async def linear_sieve(limit: int) -> List[int]:
    """
    Linear time sieve (Sieve of Euler) with O(n) time complexity.

    Each composite number is marked exactly once by its smallest prime factor,
    achieving true linear time complexity.

    Args:
        limit: Upper bound (inclusive) for prime search

    Returns:
        List of all prime numbers ≤ limit

    Time complexity: O(n)
    Space complexity: O(n)

    Examples:
        await linear_sieve(30) → [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
        await linear_sieve(20) → [2, 3, 5, 7, 11, 13, 17, 19]
    """
    if limit < 2:
        return []

    # Initialize arrays
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    primes = []

    # Yield control for large sieves
    if limit > 100000:
        await asyncio.sleep(0)

    for i in range(2, limit + 1):
        if is_prime[i]:
            primes.append(i)

        # Mark composites using existing primes
        for prime in primes:
            if i * prime > limit:
                break

            is_prime[i * prime] = False

            # Key optimization: stop when i is divisible by prime
            # This ensures each composite is marked exactly once
            if i % prime == 0:
                break

        # Yield control every 1000 iterations for large sieves
        if i % 1000 == 0 and limit > 1000000:
            await asyncio.sleep(0)

    return primes


@mcp_function(
    description="Incremental sieve for extending existing sieve results efficiently.",
    namespace="arithmetic",
    category="sieve_algorithms",
    execution_modes=["local", "remote"],
    estimated_cpu_usage="medium",
    examples=[
        {
            "input": {
                "current_primes": [2, 3, 5, 7],
                "current_limit": 10,
                "new_limit": 20,
            },
            "output": [2, 3, 5, 7, 11, 13, 17, 19],
            "description": "Extend from 10 to 20",
        },
        {
            "input": {"current_primes": [2, 3, 5], "current_limit": 6, "new_limit": 15},
            "output": [2, 3, 5, 7, 11, 13],
            "description": "Extend from 6 to 15",
        },
        {
            "input": {"current_primes": [], "current_limit": 0, "new_limit": 10},
            "output": [2, 3, 5, 7],
            "description": "Fresh sieve to 10",
        },
        {
            "input": {"current_primes": [2], "current_limit": 2, "new_limit": 5},
            "output": [2, 3, 5],
            "description": "Small extension",
        },
    ],
)
async def incremental_sieve(
    current_primes: List[int], current_limit: int, new_limit: int
) -> List[int]:
    """
    Extend existing sieve results to a higher limit incrementally.

    Efficiently extends prime list without recalculating from scratch,
    useful for algorithms that need progressively more primes.

    Args:
        current_primes: Already found primes up to current_limit
        current_limit: Current upper bound of sieve
        new_limit: New upper bound to extend to

    Returns:
        Extended list of primes up to new_limit

    Examples:
        await incremental_sieve([2, 3, 5, 7], 10, 20) → [2, 3, 5, 7, 11, 13, 17, 19]
        await incremental_sieve([], 0, 10) → [2, 3, 5, 7]
    """
    if new_limit <= current_limit:
        return [p for p in current_primes if p <= new_limit]

    # If starting fresh, use regular sieve
    if current_limit < 2:
        return await sieve_of_eratosthenes(new_limit)

    # Start from next candidate after current_limit
    start = current_limit + 1

    # Use segmented approach for the extension
    extension_primes = await segmented_sieve(start, new_limit)

    # Combine existing primes with new ones
    all_primes = current_primes + extension_primes

    return sorted(all_primes)


# ============================================================================
# PERFORMANCE ANALYSIS
# ============================================================================


@mcp_function(
    description="Compare performance of different sieve algorithms.",
    namespace="arithmetic",
    category="sieve_algorithms",
    execution_modes=["local", "remote"],
    estimated_cpu_usage="extreme",
    examples=[
        {
            "input": {"limit": 1000},
            "output": {
                "eratosthenes": "0.002s",
                "sundaram": "0.003s",
                "atkin": "0.004s",
                "linear": "0.001s",
            },
            "description": "Performance comparison for limit 1000",
        },
        {
            "input": {"limit": 10000},
            "output": {"fastest": "linear_sieve", "slowest": "sieve_of_atkin"},
            "description": "Best and worst performers",
        },
        {
            "input": {"limit": 100},
            "output": {"all_correct": True, "prime_count": 25},
            "description": "Correctness verification",
        },
    ],
)
async def sieve_performance_analysis(limit: int) -> Dict:
    """
    Analyze and compare performance of different sieve algorithms.

    Runs multiple sieve algorithms and compares their execution times,
    memory usage, and correctness.

    Args:
        limit: Test limit for performance comparison

    Returns:
        Dictionary with performance metrics and analysis

    Examples:
        await sieve_performance_analysis(1000) → {"eratosthenes": "0.002s", ...}
        await sieve_performance_analysis(10000) → {"fastest": "linear_sieve", ...}
    """
    import time

    algorithms = {
        "eratosthenes": sieve_of_eratosthenes,
        "sundaram": sieve_of_sundaram,
        "atkin": sieve_of_atkin,
        "linear": linear_sieve,
    }

    results = {}
    prime_counts = {}

    # Yield control before starting intensive tests
    await asyncio.sleep(0)

    for name, algorithm in algorithms.items():
        try:
            start_time = time.time()
            primes = await algorithm(limit)
            end_time = time.time()

            execution_time = end_time - start_time
            prime_count = len(primes)

            results[name] = {
                "time": round(execution_time, 6),
                "prime_count": prime_count,
                "primes_per_second": round(prime_count / execution_time)
                if execution_time > 0
                else float("inf"),
            }
            prime_counts[name] = prime_count

        except Exception as e:
            results[name] = {"error": str(e)}  # type: ignore[dict-item]

    # Find fastest and slowest
    valid_results = {k: v for k, v in results.items() if "error" not in v}
    if valid_results:
        fastest = min(valid_results.keys(), key=lambda k: valid_results[k]["time"])
        slowest = max(valid_results.keys(), key=lambda k: valid_results[k]["time"])
    else:
        fastest = slowest = None  # type: ignore[assignment]

    # Check correctness (all should produce same count)
    all_counts = list(prime_counts.values())
    correctness_check = len(set(all_counts)) <= 1 if all_counts else False

    return {
        "limit": limit,
        "algorithm_results": results,
        "fastest_algorithm": fastest,
        "slowest_algorithm": slowest,
        "correctness_verified": correctness_check,
        "expected_prime_count": all_counts[0] if all_counts else 0,
    }


@mcp_function(
    description="Analyze prime gaps using sieve-generated primes.",
    namespace="arithmetic",
    category="sieve_algorithms",
    execution_modes=["local", "remote"],
    estimated_cpu_usage="medium",
    examples=[
        {
            "input": {"limit": 100},
            "output": {
                "max_gap": 8,
                "avg_gap": 3.6,
                "gap_distribution": {2: 8, 4: 7, 6: 4, 8: 1},
            },
            "description": "Gap analysis up to 100",
        },
        {
            "input": {"limit": 50},
            "output": {"max_gap": 6, "avg_gap": 2.9, "most_common_gap": 2},
            "description": "Gap analysis up to 50",
        },
        {
            "input": {"limit": 30},
            "output": {"gaps": [1, 2, 2, 4, 2, 4, 2, 4, 6]},
            "description": "Individual gaps up to 30",
        },
        {
            "input": {"limit": 20},
            "output": {"twin_prime_pairs": 2, "cousin_prime_pairs": 1},
            "description": "Special gap patterns",
        },
    ],
)
async def prime_gap_sieve(limit: int) -> Dict:
    """
    Analyze gaps between consecutive primes using sieve-generated data.

    Uses efficient sieve to generate primes then analyzes the distribution
    of gaps between consecutive primes.

    Args:
        limit: Upper bound for prime generation and gap analysis

    Returns:
        Dictionary with gap statistics and analysis

    Examples:
        await prime_gap_sieve(100) → {"max_gap": 8, "avg_gap": 3.6, ...}
        await prime_gap_sieve(50) → {"max_gap": 6, "avg_gap": 2.9, ...}
    """
    if limit < 5:  # Need at least 2 primes for gaps
        return {"error": "Limit too small for gap analysis"}

    # Generate primes using most efficient sieve
    primes = await linear_sieve(limit)

    if len(primes) < 2:
        return {"error": "Not enough primes for gap analysis"}

    # Calculate gaps
    gaps = []
    for i in range(1, len(primes)):
        gap = primes[i] - primes[i - 1]
        gaps.append(gap)

    # Statistical analysis
    max_gap = max(gaps)
    min_gap = min(gaps)
    avg_gap = sum(gaps) / len(gaps)

    # Gap distribution
    gap_distribution: dict[int, int] = {}
    for gap in gaps:
        gap_distribution[gap] = gap_distribution.get(gap, 0) + 1

    # Most common gap
    most_common_gap = max(gap_distribution.keys(), key=lambda k: gap_distribution[k])

    # Special gap patterns
    twin_prime_pairs = gap_distribution.get(2, 0)  # Gaps of 2
    cousin_prime_pairs = gap_distribution.get(4, 0)  # Gaps of 4
    sexy_prime_pairs = gap_distribution.get(6, 0)  # Gaps of 6

    return {
        "limit": limit,
        "prime_count": len(primes),
        "gap_count": len(gaps),
        "max_gap": max_gap,
        "min_gap": min_gap,
        "avg_gap": round(avg_gap, 2),
        "gap_distribution": dict(sorted(gap_distribution.items())),
        "most_common_gap": most_common_gap,
        "twin_prime_pairs": twin_prime_pairs,
        "cousin_prime_pairs": cousin_prime_pairs,
        "sexy_prime_pairs": sexy_prime_pairs,
        "gaps": gaps[:20] if len(gaps) <= 20 else gaps[:10] + ["..."] + gaps[-10:],
    }


# Export all functions
__all__ = [
    # Classical sieves
    "sieve_of_eratosthenes",
    "sieve_of_sundaram",
    "sieve_of_atkin",
    # Segmented sieves
    "segmented_sieve",
    "wheel_sieve",
    # Counting functions
    "prime_counting_sieve",
    "mertens_function_sieve",
    # Optimized sieves
    "linear_sieve",
    "incremental_sieve",
    # Analysis
    "sieve_performance_analysis",
    "prime_gap_sieve",
]

# ============================================================================
# COMPREHENSIVE TEST SUITE
# ============================================================================


async def test_sieve_algorithms():
    """Test all sieve algorithms."""
    print("🧮 Sieve Algorithms Test Suite")
    print("=" * 35)

    limit = 30

    # Test classical sieves
    print("1. Classical Sieves:")
    eratosthenes = await sieve_of_eratosthenes(limit)
    print(f"   Eratosthenes: {eratosthenes}")

    sundaram = await sieve_of_sundaram(limit)
    print(f"   Sundaram:     {sundaram}")

    atkin = await sieve_of_atkin(limit)
    print(f"   Atkin:        {atkin}")

    # Test segmented sieve
    print("\n2. Segmented Sieve:")
    segmented = await segmented_sieve(10, 30)
    print(f"   Range [10,30]: {segmented}")

    # Test wheel sieve
    print("\n3. Wheel Sieve:")
    wheel_23 = await wheel_sieve(limit, [2, 3])
    print(f"   2,3-wheel:     {wheel_23}")

    # Test optimized sieves
    print("\n4. Optimized Sieves:")
    linear = await linear_sieve(limit)
    print(f"   Linear:        {linear}")

    # Test counting
    print("\n5. Counting Functions:")
    count = await prime_counting_sieve(100)
    print(f"   π(100) = {count}")

    mertens = await mertens_function_sieve(10)
    print(f"   M(10) = {mertens}")

    # Verify all sieves produce same result
    all_results = [eratosthenes, sundaram, atkin, wheel_23, linear]
    all_equal = all(result == eratosthenes for result in all_results)
    print(
        f"\n6. Correctness Check: {'✅ All algorithms agree' if all_equal else '❌ Results differ'}"
    )

    print("\n✅ Sieve algorithms testing complete!")


if __name__ == "__main__":
    import asyncio

    asyncio.run(test_sieve_algorithms())
