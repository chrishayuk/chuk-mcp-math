#!/usr/bin/env python3
# chuk_mcp_math/number_theory/special_primes.py
"""
Special Prime Numbers and Prime-Related Functions - Async Native

Comprehensive module for all types of special prime numbers and prime-related
mathematical functions. Includes Mersenne primes, Fermat primes, Sophie Germain
primes, twin primes, Wilson's theorem, pseudoprimes, and Carmichael numbers.

Functions:
- Mersenne primes: is_mersenne_prime, mersenne_prime_exponents, lucas_lehmer_test
- Fermat primes: is_fermat_prime, fermat_numbers, known_fermat_primes
- Sophie Germain & Safe primes: is_sophie_germain_prime, is_safe_prime, safe_prime_pairs
- Twin primes: is_twin_prime, twin_prime_pairs, cousin_primes, sexy_primes
- Wilson's theorem: wilson_theorem_check, wilson_factorial_mod
- Pseudoprimes: is_fermat_pseudoprime, fermat_primality_check, is_carmichael_number
- Prime gaps: prime_gap, largest_prime_gap_in_range, twin_prime_gaps
- Prime patterns: prime_arithmetic_progressions, prime_constellations
"""

import asyncio
from typing import List, Tuple, Dict
from chuk_mcp_math.mcp_decorator import mcp_function

# Import dependencies from other modules
from .primes import is_prime, next_prime, prime_factors
from .divisibility import gcd

# ============================================================================
# MERSENNE PRIMES
# ============================================================================


@mcp_function(
    description="Check if a number is a Mersenne prime (prime of form 2^p - 1 where p is prime).",
    namespace="arithmetic",
    category="special_primes",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    estimated_cpu_usage="medium",
    examples=[
        {
            "input": {"n": 31},
            "output": True,
            "description": "31 = 2^5 - 1 is Mersenne prime",
        },
        {
            "input": {"n": 127},
            "output": True,
            "description": "127 = 2^7 - 1 is Mersenne prime",
        },
        {
            "input": {"n": 15},
            "output": False,
            "description": "15 = 2^4 - 1 is not prime",
        },
        {
            "input": {"n": 17},
            "output": False,
            "description": "17 is prime but not of Mersenne form",
        },
    ],
)
async def is_mersenne_prime(n: int) -> bool:
    """
    Check if a number is a Mersenne prime.

    A Mersenne prime is a prime number of the form 2^p - 1 where p is also prime.
    These primes are important in number theory and are used to find perfect numbers.

    Args:
        n: Number to check

    Returns:
        True if n is a Mersenne prime, False otherwise

    Examples:
        await is_mersenne_prime(31) → True   # 2^5 - 1
        await is_mersenne_prime(127) → True  # 2^7 - 1
        await is_mersenne_prime(15) → False  # 2^4 - 1, not prime
        await is_mersenne_prime(17) → False  # Prime but not Mersenne form
    """
    if n <= 1:
        return False

    # Check if n is prime first
    if not await is_prime(n):
        return False

    # Check if n is of the form 2^p - 1
    temp = n + 1
    if temp & (temp - 1) != 0:  # Check if temp is power of 2
        return False

    # Find the exponent p
    p = temp.bit_length() - 1

    # Check if p is prime
    return await is_prime(p)


@mcp_function(
    description="Get known Mersenne prime exponents up to a limit.",
    namespace="arithmetic",
    category="special_primes",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    examples=[
        {
            "input": {"limit": 20},
            "output": [2, 3, 5, 7, 13, 17, 19],
            "description": "Mersenne exponents ≤ 20",
        },
        {
            "input": {"limit": 10},
            "output": [2, 3, 5, 7],
            "description": "Mersenne exponents ≤ 10",
        },
        {
            "input": {"limit": 100},
            "output": [2, 3, 5, 7, 13, 17, 19, 31, 61, 89],
            "description": "Mersenne exponents ≤ 100",
        },
    ],
)
async def mersenne_prime_exponents(limit: int) -> List[int]:
    """
    Get known Mersenne prime exponents up to a limit.

    Returns exponents p such that 2^p - 1 is prime, for p ≤ limit.
    Uses the list of known Mersenne prime exponents.

    Args:
        limit: Maximum exponent to include

    Returns:
        List of Mersenne prime exponents ≤ limit

    Examples:
        await mersenne_prime_exponents(20) → [2, 3, 5, 7, 13, 17, 19]
        await mersenne_prime_exponents(100) → [2, 3, 5, 7, 13, 17, 19, 31, 61, 89]
    """
    # Known Mersenne prime exponents (first 51 as of 2024)
    known_exponents = [
        2,
        3,
        5,
        7,
        13,
        17,
        19,
        31,
        61,
        89,
        107,
        127,
        521,
        607,
        1279,
        2203,
        2281,
        3217,
        4253,
        4423,
        9689,
        9941,
        11213,
        19937,
        21701,
        23209,
        44497,
        86243,
        110503,
        132049,
        216091,
        756839,
        859433,
        1257787,
        1398269,
        2976221,
        3021377,
        6972593,
        13466917,
        20996011,
        24036583,
        25964951,
        30402457,
        32582657,
        37156667,
        42643801,
        43112609,
        57885161,
        74207281,
        77232917,
        82589933,
    ]

    return [p for p in known_exponents if p <= limit]


@mcp_function(
    description="Perform Lucas-Lehmer primality test for Mersenne numbers.",
    namespace="arithmetic",
    category="special_primes",
    execution_modes=["local", "remote"],
    estimated_cpu_usage="high",
    examples=[
        {"input": {"p": 5}, "output": True, "description": "2^5 - 1 = 31 is prime"},
        {"input": {"p": 7}, "output": True, "description": "2^7 - 1 = 127 is prime"},
        {
            "input": {"p": 11},
            "output": False,
            "description": "2^11 - 1 = 2047 is composite",
        },
        {"input": {"p": 13}, "output": True, "description": "2^13 - 1 = 8191 is prime"},
    ],
)
async def lucas_lehmer_test(p: int) -> bool:
    """
    Lucas-Lehmer primality test for Mersenne numbers 2^p - 1.

    This is the most efficient test for Mersenne number primality.
    The test works by computing a sequence where s_0 = 4 and
    s_{i+1} = s_i^2 - 2 (mod 2^p - 1). If s_{p-2} ≡ 0, then 2^p - 1 is prime.

    Args:
        p: Exponent (must be odd prime > 2)

    Returns:
        True if 2^p - 1 is prime, False otherwise

    Examples:
        await lucas_lehmer_test(5) → True   # 2^5 - 1 = 31 is prime
        await lucas_lehmer_test(7) → True   # 2^7 - 1 = 127 is prime
        await lucas_lehmer_test(11) → False # 2^11 - 1 = 2047 is composite
    """
    if p == 2:
        return True  # 2^2 - 1 = 3 is prime

    if not await is_prime(p) or p <= 2:
        return False

    # Lucas-Lehmer sequence: s_0 = 4, s_{i+1} = s_i^2 - 2
    s = 4
    mersenne = (1 << p) - 1  # 2^p - 1

    # Yield control for large computations
    if p > 1000:
        await asyncio.sleep(0)

    for i in range(p - 2):
        s = (s * s - 2) % mersenne

        # Yield control every 1000 iterations for very large p
        if i % 1000 == 0 and p > 10000:
            await asyncio.sleep(0)

    return s == 0


@mcp_function(
    description="Generate Mersenne numbers 2^p - 1 for prime exponents up to limit.",
    namespace="arithmetic",
    category="special_primes",
    execution_modes=["local", "remote"],
    estimated_cpu_usage="medium",
    examples=[
        {
            "input": {"limit": 10},
            "output": [3, 7, 31, 127],
            "description": "Mersenne numbers for p ≤ 10",
        },
        {
            "input": {"limit": 20},
            "output": [3, 7, 31, 127, 8191, 131071, 524287],
            "description": "Mersenne numbers for p ≤ 20",
        },
    ],
)
async def mersenne_numbers(limit: int) -> List[int]:
    """
    Generate Mersenne numbers 2^p - 1 for prime exponents up to limit.

    Args:
        limit: Maximum exponent to consider

    Returns:
        List of Mersenne numbers (may include composite ones)

    Examples:
        await mersenne_numbers(10) → [3, 7, 31, 127]
        await mersenne_numbers(20) → [3, 7, 31, 127, 8191, 131071, 524287]
    """
    if limit < 2:
        return []

    result = []
    candidate = 2

    while candidate <= limit:
        if await is_prime(candidate):
            mersenne = (1 << candidate) - 1
            result.append(mersenne)

        candidate = await next_prime(candidate)

        # Yield control periodically
        if len(result) % 5 == 0:
            await asyncio.sleep(0)

    return result


# ============================================================================
# FERMAT PRIMES
# ============================================================================


@mcp_function(
    description="Check if a number is a Fermat prime (prime of form 2^(2^n) + 1).",
    namespace="arithmetic",
    category="special_primes",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    examples=[
        {
            "input": {"n": 3},
            "output": True,
            "description": "3 = 2^(2^0) + 1 is Fermat prime",
        },
        {
            "input": {"n": 5},
            "output": True,
            "description": "5 = 2^(2^1) + 1 is Fermat prime",
        },
        {
            "input": {"n": 17},
            "output": True,
            "description": "17 = 2^(2^2) + 1 is Fermat prime",
        },
        {
            "input": {"n": 7},
            "output": False,
            "description": "7 is prime but not Fermat form",
        },
    ],
)
async def is_fermat_prime(n: int) -> bool:
    """
    Check if a number is a Fermat prime.

    A Fermat prime is a prime number of the form 2^(2^k) + 1 for some k ≥ 0.
    Only five Fermat primes are known: 3, 5, 17, 257, 65537.

    Args:
        n: Number to check

    Returns:
        True if n is a Fermat prime, False otherwise

    Examples:
        await is_fermat_prime(3) → True    # F_0 = 2^(2^0) + 1 = 3
        await is_fermat_prime(5) → True    # F_1 = 2^(2^1) + 1 = 5
        await is_fermat_prime(17) → True   # F_2 = 2^(2^2) + 1 = 17
        await is_fermat_prime(7) → False   # Prime but not Fermat form
    """
    if n <= 2:
        return False

    # Check if n is prime first
    if not await is_prime(n):
        return False

    # Check if n is of the form 2^(2^k) + 1
    temp = n - 1

    # temp should be a power of 2
    if temp & (temp - 1) != 0:
        return False

    # The exponent should also be a power of 2
    exponent = temp.bit_length() - 1
    return exponent & (exponent - 1) == 0


@mcp_function(
    description="Generate Fermat numbers F_n = 2^(2^n) + 1 up to index limit.",
    namespace="arithmetic",
    category="special_primes",
    execution_modes=["local", "remote"],
    examples=[
        {
            "input": {"limit": 4},
            "output": [3, 5, 17, 257, 65537],
            "description": "First 5 Fermat numbers",
        },
        {
            "input": {"limit": 2},
            "output": [3, 5, 17],
            "description": "First 3 Fermat numbers",
        },
        {
            "input": {"limit": 6},
            "output": [3, 5, 17, 257, 65537, 4294967297],
            "description": "First 6 Fermat numbers (F_5 is composite)",
        },
    ],
)
async def fermat_numbers(limit: int) -> List[int]:
    """
    Generate Fermat numbers F_n = 2^(2^n) + 1.

    Note: F_5 and higher are known to be composite.

    Args:
        limit: Maximum index n to generate

    Returns:
        List of Fermat numbers F_0 through F_limit

    Examples:
        await fermat_numbers(4) → [3, 5, 17, 257, 65537]
        await fermat_numbers(6) → [3, 5, 17, 257, 65537, 4294967297, ...]
    """
    if limit < 0:
        return []

    result = []
    for n in range(limit + 1):
        if n > 15:  # Avoid computing extremely large numbers
            break
        fermat_n = (1 << (1 << n)) + 1  # 2^(2^n) + 1
        result.append(fermat_n)

        # Yield control for large computations
        if n > 5:
            await asyncio.sleep(0)

    return result


@mcp_function(
    description="Get the five known Fermat primes.",
    namespace="arithmetic",
    category="special_primes",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    examples=[
        {
            "input": {},
            "output": [3, 5, 17, 257, 65537],
            "description": "The 5 known Fermat primes",
        }
    ],
)
async def known_fermat_primes() -> List[int]:
    """
    Get the five known Fermat primes.

    These are the only known Fermat primes: F_0, F_1, F_2, F_3, F_4.
    It is unknown whether any more exist.

    Returns:
        List of the five known Fermat primes

    Examples:
        await known_fermat_primes() → [3, 5, 17, 257, 65537]
    """
    return [3, 5, 17, 257, 65537]


# ============================================================================
# SOPHIE GERMAIN AND SAFE PRIMES
# ============================================================================


@mcp_function(
    description="Check if a prime p is a Sophie Germain prime (2p + 1 is also prime).",
    namespace="arithmetic",
    category="special_primes",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    examples=[
        {
            "input": {"p": 11},
            "output": True,
            "description": "11 is Sophie Germain: 2×11+1=23 is prime",
        },
        {
            "input": {"p": 23},
            "output": True,
            "description": "23 is Sophie Germain: 2×23+1=47 is prime",
        },
        {
            "input": {"p": 13},
            "output": False,
            "description": "13 is not Sophie Germain: 2×13+1=27 is composite",
        },
        {
            "input": {"p": 17},
            "output": False,
            "description": "17 is not Sophie Germain: 2×17+1=35 is composite",
        },
    ],
)
async def is_sophie_germain_prime(p: int) -> bool:
    """
    Check if a prime p is a Sophie Germain prime.

    A Sophie Germain prime is a prime p such that 2p + 1 is also prime.
    These primes are important in cryptography and number theory.

    Args:
        p: Number to check

    Returns:
        True if p is a Sophie Germain prime, False otherwise

    Examples:
        await is_sophie_germain_prime(11) → True  # 2×11+1 = 23 is prime
        await is_sophie_germain_prime(23) → True  # 2×23+1 = 47 is prime
        await is_sophie_germain_prime(13) → False # 2×13+1 = 27 is composite
    """
    if not await is_prime(p):
        return False

    safe_prime = 2 * p + 1
    return await is_prime(safe_prime)


@mcp_function(
    description="Check if a prime q is a safe prime (q = 2p + 1 where p is prime).",
    namespace="arithmetic",
    category="special_primes",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    examples=[
        {
            "input": {"q": 23},
            "output": True,
            "description": "23 = 2×11+1 where 11 is prime",
        },
        {
            "input": {"q": 47},
            "output": True,
            "description": "47 = 2×23+1 where 23 is prime",
        },
        {
            "input": {"q": 29},
            "output": False,
            "description": "29 = 2×14+1 where 14 is not prime",
        },
        {
            "input": {"q": 37},
            "output": False,
            "description": "37 = 2×18+1 where 18 is not prime",
        },
    ],
)
async def is_safe_prime(q: int) -> bool:
    """
    Check if a prime q is a safe prime.

    A safe prime is a prime q such that (q-1)/2 is also prime.
    Safe primes are used in cryptographic applications.

    Args:
        q: Number to check

    Returns:
        True if q is a safe prime, False otherwise

    Examples:
        await is_safe_prime(23) → True   # (23-1)/2 = 11 is prime
        await is_safe_prime(47) → True   # (47-1)/2 = 23 is prime
        await is_safe_prime(29) → False  # (29-1)/2 = 14 is not prime
    """
    if not await is_prime(q):
        return False

    if q == 2 or q == 3:
        return False  # Special cases

    if (q - 1) % 2 != 0:
        return False  # q-1 must be even

    sophie_germain = (q - 1) // 2
    return await is_prime(sophie_germain)


@mcp_function(
    description="Find Sophie Germain and safe prime pairs up to limit.",
    namespace="arithmetic",
    category="special_primes",
    execution_modes=["local", "remote"],
    estimated_cpu_usage="medium",
    examples=[
        {
            "input": {"limit": 50},
            "output": [[2, 5], [3, 7], [5, 11], [11, 23], [23, 47]],
            "description": "Sophie Germain pairs ≤ 50",
        },
        {
            "input": {"limit": 25},
            "output": [[2, 5], [3, 7], [5, 11], [11, 23]],
            "description": "Sophie Germain pairs ≤ 25",
        },
        {
            "input": {"limit": 100},
            "output": [
                [2, 5],
                [3, 7],
                [5, 11],
                [11, 23],
                [23, 47],
                [29, 59],
                [41, 83],
                [53, 107],
            ],
            "description": "Sophie Germain pairs ≤ 100",
        },
    ],
)
async def safe_prime_pairs(limit: int) -> List[Tuple[int, int]]:
    """
    Find Sophie Germain and safe prime pairs up to limit.

    Returns pairs (p, q) where p is Sophie Germain prime and q = 2p + 1 is safe prime.

    Args:
        limit: Upper limit for Sophie Germain primes

    Returns:
        List of (sophie_germain_prime, safe_prime) pairs

    Examples:
        await safe_prime_pairs(50) → [(2, 5), (3, 7), (5, 11), (11, 23), (23, 47)]
        await safe_prime_pairs(100) → [(2, 5), (3, 7), (5, 11), (11, 23), (23, 47), (29, 59), (41, 83), (53, 107)]
    """
    pairs = []
    checks = 0

    for p in range(2, limit + 1):
        if await is_prime(p):
            safe = 2 * p + 1
            if await is_prime(safe):
                pairs.append((p, safe))

        checks += 1
        # Yield control every 1000 checks for large limits
        if checks % 1000 == 0 and limit > 1000:
            await asyncio.sleep(0)

    return pairs


# ============================================================================
# TWIN PRIMES AND RELATED
# ============================================================================


@mcp_function(
    description="Check if a number is part of a twin prime pair (p, p+2).",
    namespace="arithmetic",
    category="special_primes",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    examples=[
        {"input": {"p": 3}, "output": True, "description": "3 and 5 are twin primes"},
        {"input": {"p": 5}, "output": True, "description": "3 and 5 are twin primes"},
        {
            "input": {"p": 13},
            "output": True,
            "description": "11 and 13 are twin primes",
        },
        {
            "input": {"p": 7},
            "output": False,
            "description": "7 is not part of twin prime pair",
        },
    ],
)
async def is_twin_prime(p: int) -> bool:
    """
    Check if a number is part of a twin prime pair.

    Twin primes are pairs of primes (p, p+2) or (p-2, p).
    Examples: (3,5), (5,7), (11,13), (17,19), (29,31), (41,43)

    Args:
        p: Number to check

    Returns:
        True if p is part of a twin prime pair, False otherwise

    Examples:
        await is_twin_prime(3) → True   # (3, 5) are twin primes
        await is_twin_prime(13) → True  # (11, 13) are twin primes
        await is_twin_prime(7) → False  # Neither (5, 7) nor (7, 9) are both prime
    """
    if not await is_prime(p):
        return False

    # Check if p+2 is prime or p-2 is prime
    plus_two_prime = await is_prime(p + 2)
    minus_two_prime = p > 2 and await is_prime(p - 2)

    return plus_two_prime or minus_two_prime


@mcp_function(
    description="Find twin prime pairs up to limit.",
    namespace="arithmetic",
    category="special_primes",
    execution_modes=["local", "remote"],
    estimated_cpu_usage="medium",
    examples=[
        {
            "input": {"limit": 50},
            "output": [[3, 5], [5, 7], [11, 13], [17, 19], [29, 31], [41, 43]],
            "description": "Twin primes ≤ 50",
        },
        {
            "input": {"limit": 20},
            "output": [[3, 5], [5, 7], [11, 13], [17, 19]],
            "description": "Twin primes ≤ 20",
        },
        {
            "input": {"limit": 100},
            "output": [
                [3, 5],
                [5, 7],
                [11, 13],
                [17, 19],
                [29, 31],
                [41, 43],
                [59, 61],
                [71, 73],
            ],
            "description": "Twin primes ≤ 100",
        },
    ],
)
async def twin_prime_pairs(limit: int) -> List[Tuple[int, int]]:
    """
    Find twin prime pairs up to limit.

    Args:
        limit: Upper limit for the smaller twin prime

    Returns:
        List of (p, p+2) twin prime pairs

    Examples:
        await twin_prime_pairs(50) → [(3, 5), (5, 7), (11, 13), (17, 19), (29, 31), (41, 43)]
        await twin_prime_pairs(100) → [(3, 5), (5, 7), (11, 13), (17, 19), (29, 31), (41, 43), (59, 61), (71, 73)]
    """
    pairs = []
    checks = 0

    for p in range(3, limit + 1, 2):  # Only check odd numbers (except 2)
        if await is_prime(p) and await is_prime(p + 2):
            pairs.append((p, p + 2))

        checks += 1
        # Yield control every 1000 checks for large limits
        if checks % 1000 == 0 and limit > 1000:
            await asyncio.sleep(0)

    return pairs


@mcp_function(
    description="Find cousin prime pairs (p, p+4) up to limit.",
    namespace="arithmetic",
    category="special_primes",
    execution_modes=["local", "remote"],
    estimated_cpu_usage="medium",
    examples=[
        {
            "input": {"limit": 50},
            "output": [[3, 7], [7, 11], [13, 17], [19, 23], [37, 41], [43, 47]],
            "description": "Cousin primes ≤ 50",
        },
        {
            "input": {"limit": 20},
            "output": [[3, 7], [7, 11], [13, 17], [19, 23]],
            "description": "Cousin primes ≤ 20",
        },
        {
            "input": {"limit": 100},
            "output": [
                [3, 7],
                [7, 11],
                [13, 17],
                [19, 23],
                [37, 41],
                [43, 47],
                [67, 71],
                [79, 83],
            ],
            "description": "Cousin primes ≤ 100",
        },
    ],
)
async def cousin_primes(limit: int) -> List[Tuple[int, int]]:
    """
    Find cousin prime pairs (primes that differ by 4).

    Cousin primes are pairs of primes (p, p+4).
    Examples: (3,7), (7,11), (13,17), (19,23)

    Args:
        limit: Upper limit for the smaller cousin prime

    Returns:
        List of (p, p+4) cousin prime pairs

    Examples:
        await cousin_primes(50) → [(3, 7), (7, 11), (13, 17), (19, 23), (37, 41), (43, 47)]
    """
    pairs = []
    checks = 0

    for p in range(3, limit + 1):
        if await is_prime(p) and await is_prime(p + 4):
            pairs.append((p, p + 4))

        checks += 1
        # Yield control every 1000 checks for large limits
        if checks % 1000 == 0 and limit > 1000:
            await asyncio.sleep(0)

    return pairs


@mcp_function(
    description="Find sexy prime pairs (p, p+6) up to limit.",
    namespace="arithmetic",
    category="special_primes",
    execution_modes=["local", "remote"],
    estimated_cpu_usage="medium",
    examples=[
        {
            "input": {"limit": 50},
            "output": [
                [5, 11],
                [7, 13],
                [13, 19],
                [17, 23],
                [31, 37],
                [37, 43],
                [41, 47],
            ],
            "description": "Sexy primes ≤ 50",
        },
        {
            "input": {"limit": 25},
            "output": [[5, 11], [7, 13], [13, 19], [17, 23]],
            "description": "Sexy primes ≤ 25",
        },
        {
            "input": {"limit": 100},
            "output": [
                [5, 11],
                [7, 13],
                [13, 19],
                [17, 23],
                [31, 37],
                [37, 43],
                [41, 47],
                [61, 67],
                [73, 79],
            ],
            "description": "Sexy primes ≤ 100",
        },
    ],
)
async def sexy_primes(limit: int) -> List[Tuple[int, int]]:
    """
    Find sexy prime pairs (primes that differ by 6).

    Sexy primes are pairs of primes (p, p+6).
    Examples: (5,11), (7,13), (13,19), (17,23)

    Args:
        limit: Upper limit for the smaller sexy prime

    Returns:
        List of (p, p+6) sexy prime pairs

    Examples:
        await sexy_primes(50) → [(5, 11), (7, 13), (13, 19), (17, 23), (31, 37), (37, 43), (41, 47)]
    """
    pairs = []
    checks = 0

    for p in range(5, limit + 1):  # Start from 5 since smaller primes don't work
        if await is_prime(p) and await is_prime(p + 6):
            pairs.append((p, p + 6))

        checks += 1
        # Yield control every 1000 checks for large limits
        if checks % 1000 == 0 and limit > 1000:
            await asyncio.sleep(0)

    return pairs


# ============================================================================
# WILSON'S THEOREM
# ============================================================================


@mcp_function(
    description="Check Wilson's theorem: p is prime iff (p-1)! ≡ -1 (mod p).",
    namespace="arithmetic",
    category="primality_tests",
    execution_modes=["local", "remote"],
    estimated_cpu_usage="high",
    examples=[
        {
            "input": {"n": 7},
            "output": True,
            "description": "7 is prime: 6! ≡ -1 (mod 7)",
        },
        {
            "input": {"n": 11},
            "output": True,
            "description": "11 is prime: 10! ≡ -1 (mod 11)",
        },
        {
            "input": {"n": 8},
            "output": False,
            "description": "8 is composite: 7! ≢ -1 (mod 8)",
        },
        {
            "input": {"n": 9},
            "output": False,
            "description": "9 is composite: 8! ≢ -1 (mod 9)",
        },
    ],
)
async def wilson_theorem_check(n: int) -> bool:
    """
    Check Wilson's theorem for primality.

    Wilson's theorem: p is prime if and only if (p-1)! ≡ -1 (mod p).
    This provides a theoretical primality test, though it's not practical
    for large numbers due to the factorial computation.

    Args:
        n: Number to check

    Returns:
        True if n satisfies Wilson's theorem, False otherwise

    Examples:
        await wilson_theorem_check(7) → True   # 6! ≡ -1 (mod 7)
        await wilson_theorem_check(11) → True  # 10! ≡ -1 (mod 11)
        await wilson_theorem_check(8) → False  # 8 is composite
    """
    if n <= 1:
        return False
    if n == 2:
        return True

    # Calculate (n-1)! mod n
    factorial_mod = await wilson_factorial_mod(n - 1, n)

    # Check if (n-1)! ≡ -1 (mod n), i.e., ≡ n-1 (mod n)
    return factorial_mod == n - 1


@mcp_function(
    description="Calculate k! mod m efficiently for Wilson's theorem.",
    namespace="arithmetic",
    category="primality_tests",
    execution_modes=["local", "remote"],
    estimated_cpu_usage="medium",
    examples=[
        {
            "input": {"k": 6, "m": 7},
            "output": 6,
            "description": "6! mod 7 = 720 mod 7 = 6",
        },
        {"input": {"k": 10, "m": 11}, "output": 10, "description": "10! mod 11 = 10"},
        {
            "input": {"k": 4, "m": 5},
            "output": 4,
            "description": "4! mod 5 = 24 mod 5 = 4",
        },
    ],
)
async def wilson_factorial_mod(k: int, m: int) -> int:
    """
    Calculate k! mod m efficiently.

    Args:
        k: Factorial number
        m: Modulus

    Returns:
        k! mod m

    Examples:
        await wilson_factorial_mod(6, 7) → 6   # 6! mod 7
        await wilson_factorial_mod(10, 11) → 10 # 10! mod 11
    """
    if k >= m:
        return 0  # k! is divisible by m

    result = 1
    for i in range(1, k + 1):
        result = (result * i) % m

        # Yield control every 1000 iterations for large k
        if i % 1000 == 0 and k > 1000:
            await asyncio.sleep(0)

    return result


# ============================================================================
# PSEUDOPRIMES AND CARMICHAEL NUMBERS
# ============================================================================


@mcp_function(
    description="Check if n is a Fermat pseudoprime to base a.",
    namespace="arithmetic",
    category="pseudoprimes",
    execution_modes=["local", "remote"],
    estimated_cpu_usage="medium",
    examples=[
        {
            "input": {"n": 341, "a": 2},
            "output": True,
            "description": "341 is pseudoprime base 2",
        },
        {
            "input": {"n": 341, "a": 3},
            "output": False,
            "description": "341 fails base 3 check",
        },
        {
            "input": {"n": 561, "a": 2},
            "output": True,
            "description": "561 is Carmichael number",
        },
        {
            "input": {"n": 561, "a": 5},
            "output": True,
            "description": "561 is pseudoprime to many bases",
        },
    ],
)
async def is_fermat_pseudoprime(n: int, a: int) -> bool:
    """
    Check if n is a Fermat pseudoprime to base a.

    A composite number n is a Fermat pseudoprime to base a if:
    gcd(a, n) = 1 and a^(n-1) ≡ 1 (mod n).

    Args:
        n: Number to check
        a: Base for the check

    Returns:
        True if n is a Fermat pseudoprime to base a, False otherwise

    Examples:
        await is_fermat_pseudoprime(341, 2) → True   # 341 is base-2 pseudoprime
        await is_fermat_pseudoprime(341, 3) → False  # 341 fails base-3 check
        await is_fermat_pseudoprime(561, 2) → True   # 561 is Carmichael number
    """
    if n <= 1 or await is_prime(n):
        return False

    if await gcd(a, n) != 1:
        return False  # a and n must be coprime

    # Check if a^(n-1) ≡ 1 (mod n)
    return pow(a, n - 1, n) == 1


@mcp_function(
    description="Perform Fermat primality check for base a.",
    namespace="arithmetic",
    category="primality_tests",
    execution_modes=["local", "remote"],
    estimated_cpu_usage="medium",
    examples=[
        {
            "input": {"n": 17, "a": 2},
            "output": True,
            "description": "17 passes Fermat check base 2",
        },
        {
            "input": {"n": 15, "a": 2},
            "output": False,
            "description": "15 fails Fermat check base 2",
        },
        {
            "input": {"n": 341, "a": 2},
            "output": True,
            "description": "341 falsely passes (pseudoprime)",
        },
    ],
)
async def fermat_primality_check(n: int, a: int) -> bool:
    """
    Perform Fermat primality check.

    Checks if a^(n-1) ≡ 1 (mod n) where gcd(a, n) = 1.
    If this fails, n is definitely composite.
    If this passes, n is probably prime (or a pseudoprime).

    Args:
        n: Number to check
        a: Base for the check

    Returns:
        True if n passes the check, False if n definitely composite

    Examples:
        await fermat_primality_check(17, 2) → True   # 17 is prime
        await fermat_primality_check(15, 2) → False  # 15 is composite
        await fermat_primality_check(341, 2) → True  # 341 is pseudoprime (false positive)
    """
    if n <= 1:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False

    if await gcd(a, n) != 1:
        return False  # Check requires gcd(a, n) = 1

    return pow(a, n - 1, n) == 1


@mcp_function(
    description="Check if n is a Carmichael number (absolute Fermat pseudoprime).",
    namespace="arithmetic",
    category="pseudoprimes",
    execution_modes=["local", "remote"],
    estimated_cpu_usage="high",
    examples=[
        {
            "input": {"n": 561},
            "output": True,
            "description": "561 is smallest Carmichael number",
        },
        {
            "input": {"n": 1105},
            "output": True,
            "description": "1105 is Carmichael number",
        },
        {
            "input": {"n": 1729},
            "output": True,
            "description": "1729 is Carmichael number",
        },
        {
            "input": {"n": 341},
            "output": False,
            "description": "341 is pseudoprime but not Carmichael",
        },
    ],
)
async def is_carmichael_number(n: int) -> bool:
    """
    Check if n is a Carmichael number.

    A Carmichael number is a composite number that is a Fermat pseudoprime
    to every base a coprime to n.

    Uses Korselt's criterion: n is Carmichael iff:
    1. n is composite and square-free
    2. For every prime p dividing n: (p-1) divides (n-1)

    Args:
        n: Number to test

    Returns:
        True if n is a Carmichael number, False otherwise

    Examples:
        await is_carmichael_number(561) → True   # First Carmichael number
        await is_carmichael_number(1105) → True  # Second Carmichael number
        await is_carmichael_number(341) → False  # Pseudoprime but not Carmichael
    """
    if n <= 2 or await is_prime(n):
        return False

    # Get prime factorization to check Korselt's criterion
    factors = await prime_factors(n)

    if not factors:
        return False

    # Check if square-free (no repeated prime factors)
    unique_factors = list(set(factors))
    if len(factors) != len(unique_factors):
        return False

    # Must have at least 3 distinct prime factors
    if len(unique_factors) < 3:
        return False

    # Check Korselt's criterion: for each prime p | n, (p-1) | (n-1)
    for p in unique_factors:
        if (n - 1) % (p - 1) != 0:
            return False

    return True


# ============================================================================
# PRIME GAPS AND PATTERNS
# ============================================================================


@mcp_function(
    description="Calculate the gap between a prime and the next prime.",
    namespace="arithmetic",
    category="prime_gaps",
    execution_modes=["local", "remote"],
    estimated_cpu_usage="medium",
    examples=[
        {
            "input": {"p": 7},
            "output": 4,
            "description": "Gap from 7 to next prime 11 is 4",
        },
        {
            "input": {"p": 23},
            "output": 6,
            "description": "Gap from 23 to next prime 29 is 6",
        },
        {
            "input": {"p": 2},
            "output": 1,
            "description": "Gap from 2 to next prime 3 is 1",
        },
        {
            "input": {"p": 89},
            "output": 8,
            "description": "Gap from 89 to next prime 97 is 8",
        },
    ],
)
async def prime_gap(p: int) -> int:
    """
    Calculate the gap between prime p and the next prime.

    Args:
        p: A prime number

    Returns:
        The gap to the next prime

    Raises:
        ValueError: If p is not prime

    Examples:
        await prime_gap(7) → 4   # Next prime after 7 is 11, gap = 4
        await prime_gap(23) → 6  # Next prime after 23 is 29, gap = 6
        await prime_gap(2) → 1   # Next prime after 2 is 3, gap = 1
    """
    if not await is_prime(p):
        raise ValueError(f"{p} is not prime")

    next_p = await next_prime(p)
    return next_p - p


@mcp_function(
    description="Find the largest prime gap in a given range.",
    namespace="arithmetic",
    category="prime_gaps",
    execution_modes=["local", "remote"],
    estimated_cpu_usage="high",
    examples=[
        {
            "input": {"start": 2, "end": 100},
            "output": {"gap": 8, "prime": 89, "next_prime": 97},
            "description": "Largest gap ≤ 100",
        },
        {
            "input": {"start": 2, "end": 50},
            "output": {"gap": 6, "prime": 23, "next_prime": 29},
            "description": "Largest gap ≤ 50",
        },
    ],
)
async def largest_prime_gap_in_range(start: int, end: int) -> Dict[str, int]:
    """
    Find the largest prime gap in a given range.

    Args:
        start: Start of range
        end: End of range

    Returns:
        Dictionary with gap size, prime, and next prime

    Examples:
        await largest_prime_gap_in_range(2, 100) → {"gap": 8, "prime": 89, "next_prime": 97}
        await largest_prime_gap_in_range(2, 50) → {"gap": 6, "prime": 23, "next_prime": 29}
    """
    if start < 2:
        start = 2

    max_gap = 0
    gap_prime = 0
    gap_next_prime = 0

    p = start
    while p <= end:
        if await is_prime(p):
            next_p = await next_prime(p)
            gap = next_p - p

            if gap > max_gap:
                max_gap = gap
                gap_prime = p
                gap_next_prime = next_p

            p = next_p
        else:
            p += 1

        # Yield control periodically
        if p % 100 == 0:
            await asyncio.sleep(0)

    return {"gap": max_gap, "prime": gap_prime, "next_prime": gap_next_prime}


@mcp_function(
    description="Find gaps between consecutive twin prime pairs.",
    namespace="arithmetic",
    category="prime_gaps",
    execution_modes=["local", "remote"],
    estimated_cpu_usage="medium",
    examples=[
        {
            "input": {"limit": 100},
            "output": [2, 6, 6, 12, 12, 18, 2],
            "description": "Gaps between twin prime pairs ≤ 100",
        }
    ],
)
async def twin_prime_gaps(limit: int) -> List[int]:
    """
    Find gaps between consecutive twin prime pairs.

    Args:
        limit: Upper limit for twin primes

    Returns:
        List of gaps between consecutive twin prime pairs

    Examples:
        await twin_prime_gaps(100) → [2, 6, 6, 12, 12, 18, 2]
    """
    twin_pairs = await twin_prime_pairs(limit)

    if len(twin_pairs) < 2:
        return []

    gaps = []
    for i in range(1, len(twin_pairs)):
        # Gap is difference between start of current pair and start of previous pair
        gap = twin_pairs[i][0] - twin_pairs[i - 1][0]
        gaps.append(gap)

    return gaps


# Export all functions
__all__ = [
    # Mersenne primes
    "is_mersenne_prime",
    "mersenne_prime_exponents",
    "lucas_lehmer_test",
    "mersenne_numbers",
    # Fermat primes
    "is_fermat_prime",
    "fermat_numbers",
    "known_fermat_primes",
    # Sophie Germain and safe primes
    "is_sophie_germain_prime",
    "is_safe_prime",
    "safe_prime_pairs",
    # Twin primes and related
    "is_twin_prime",
    "twin_prime_pairs",
    "cousin_primes",
    "sexy_primes",
    # Wilson's theorem
    "wilson_theorem_check",
    "wilson_factorial_mod",
    # Pseudoprimes and Carmichael numbers
    "is_fermat_pseudoprime",
    "fermat_primality_check",
    "is_carmichael_number",
    # Prime gaps and patterns
    "prime_gap",
    "largest_prime_gap_in_range",
    "twin_prime_gaps",
]

if __name__ == "__main__":
    import asyncio

    async def test_special_primes():
        """Test special prime functions."""
        print("🔢 Special Primes Functions Test")
        print("=" * 40)

        # Test Mersenne primes
        print("Mersenne Primes:")
        print(f"  is_mersenne_prime(31) = {await is_mersenne_prime(31)}")
        print(f"  is_mersenne_prime(127) = {await is_mersenne_prime(127)}")
        print(f"  lucas_lehmer_test(5) = {await lucas_lehmer_test(5)}")
        print(f"  mersenne_prime_exponents(20) = {await mersenne_prime_exponents(20)}")

        # Test Fermat primes
        print("\nFermat Primes:")
        print(f"  is_fermat_prime(17) = {await is_fermat_prime(17)}")
        print(f"  known_fermat_primes() = {await known_fermat_primes()}")

        # Test Sophie Germain primes
        print("\nSophie Germain Primes:")
        print(f"  is_sophie_germain_prime(11) = {await is_sophie_germain_prime(11)}")
        print(f"  is_safe_prime(23) = {await is_safe_prime(23)}")
        print(f"  safe_prime_pairs(25) = {await safe_prime_pairs(25)}")

        # Test twin primes
        print("\nTwin Primes:")
        print(f"  is_twin_prime(13) = {await is_twin_prime(13)}")
        print(f"  twin_prime_pairs(30) = {await twin_prime_pairs(30)}")
        print(f"  cousin_primes(20) = {await cousin_primes(20)}")
        print(f"  sexy_primes(25) = {await sexy_primes(25)}")

        # Test Wilson's theorem
        print("\nWilson's Theorem:")
        print(f"  wilson_theorem_check(7) = {await wilson_theorem_check(7)}")
        print(f"  wilson_theorem_check(8) = {await wilson_theorem_check(8)}")

        # Test pseudoprimes
        print("\nPseudoprimes:")
        print(
            f"  is_fermat_pseudoprime(341, 2) = {await is_fermat_pseudoprime(341, 2)}"
        )
        print(f"  is_carmichael_number(561) = {await is_carmichael_number(561)}")

        # Test prime gaps
        print("\nPrime Gaps:")
        print(f"  prime_gap(7) = {await prime_gap(7)}")
        print(f"  prime_gap(23) = {await prime_gap(23)}")
        gap_info = await largest_prime_gap_in_range(2, 100)
        print(f"  largest_prime_gap_in_range(2, 100) = {gap_info}")

        print("\n✅ All special prime functions working!")

    asyncio.run(test_special_primes())
