#!/usr/bin/env python3
# chuk_mcp_math/number_theory/advanced_prime_patterns.py
"""
Advanced Prime Patterns & Distribution - Async Native

Functions for analyzing advanced prime number patterns, distributions, and theoretical results.
Covers prime gaps, constellations, distribution theory, and various prime conjectures.

Functions:
- Prime patterns: cousin_primes, sexy_primes, prime_triplets, prime_quadruplets
- Distribution: prime_counting_function, prime_number_theorem_error, prime_gaps_analysis
- Conjectures: bertrand_postulate_verify, twin_prime_conjecture_data, prime_gap_records
- Advanced: prime_constellations, admissible_tuples, prime_k_tuples
- Analysis: prime_density_analysis, prime_spiral_analysis, ulam_spiral
"""

import math
import asyncio
from typing import List, Dict
from collections import defaultdict
from chuk_mcp_math.mcp_decorator import mcp_function


# Helper function for primality testing (reuse from existing module)
async def _is_prime(n: int) -> bool:
    """Internal helper for primality testing."""
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False

    sqrt_n = int(math.sqrt(n))
    for i in range(3, sqrt_n + 1, 2):
        if n % i == 0:
            return False
    return True


async def _sieve_of_eratosthenes(limit: int) -> List[int]:
    """Generate primes up to limit using Sieve of Eratosthenes."""
    if limit < 2:
        return []

    # Create boolean array and initialize all entries as true
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False

    p = 2
    while p * p <= limit:
        if is_prime[p]:
            # Mark all multiples of p
            for i in range(p * p, limit + 1, p):
                is_prime[i] = False
        p += 1

        # Yield control every 1000 iterations for large limits
        if p % 1000 == 0:
            await asyncio.sleep(0)

    # Collect primes
    primes = [i for i in range(2, limit + 1) if is_prime[i]]
    return primes


# ============================================================================
# PRIME CONSTELLATIONS AND PATTERNS
# ============================================================================


@mcp_function(
    description="Find all cousin prime pairs (primes differing by 4) up to limit.",
    namespace="arithmetic",
    category="prime_patterns",
    execution_modes=["local", "remote"],
    estimated_cpu_usage="medium",
    examples=[
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
            "description": "Cousin prime pairs up to 100",
        },
        {
            "input": {"limit": 50},
            "output": [[3, 7], [7, 11], [13, 17], [19, 23], [37, 41], [43, 47]],
            "description": "Cousin prime pairs up to 50",
        },
        {
            "input": {"limit": 200},
            "output": [
                [3, 7],
                [7, 11],
                [13, 17],
                [19, 23],
                [37, 41],
                [43, 47],
                [67, 71],
                [79, 83],
                [97, 101],
                [103, 107],
                [109, 113],
                [127, 131],
                [163, 167],
                [193, 197],
            ],
            "description": "Cousin prime pairs up to 200",
        },
    ],
)
async def cousin_primes(limit: int) -> List[List[int]]:
    """
    Find all cousin prime pairs (primes differing by 4).

    Cousin primes are pairs of primes (p, p+4) where both p and p+4 are prime.

    Args:
        limit: Upper bound for the search

    Returns:
        List of [p, p+4] cousin prime pairs with p+4 ≤ limit

    Examples:
        await cousin_primes(100) → [[3, 7], [7, 11], [13, 17], ...]
        await cousin_primes(50) → [[3, 7], [7, 11], [13, 17], ...]
    """
    if limit < 7:
        return []

    cousin_pairs = []
    primes = await _sieve_of_eratosthenes(limit)
    prime_set = set(primes)

    for p in primes:
        if p + 4 <= limit and p + 4 in prime_set:
            cousin_pairs.append([p, p + 4])

    return cousin_pairs


@mcp_function(
    description="Find all sexy prime pairs (primes differing by 6) up to limit.",
    namespace="arithmetic",
    category="prime_patterns",
    execution_modes=["local", "remote"],
    estimated_cpu_usage="medium",
    examples=[
        {
            "input": {"limit": 100},
            "output": [
                [5, 11],
                [7, 13],
                [13, 19],
                [17, 23],
                [23, 29],
                [31, 37],
                [37, 43],
                [41, 47],
                [61, 67],
                [67, 73],
                [73, 79],
                [83, 89],
            ],
            "description": "Sexy prime pairs up to 100",
        },
        {
            "input": {"limit": 50},
            "output": [
                [5, 11],
                [7, 13],
                [13, 19],
                [17, 23],
                [23, 29],
                [31, 37],
                [37, 43],
                [41, 47],
            ],
            "description": "Sexy prime pairs up to 50",
        },
    ],
)
async def sexy_primes(limit: int) -> List[List[int]]:
    """
    Find all sexy prime pairs (primes differing by 6).

    Sexy primes are pairs of primes (p, p+6) where both p and p+6 are prime.

    Args:
        limit: Upper bound for the search

    Returns:
        List of [p, p+6] sexy prime pairs with p+6 ≤ limit

    Examples:
        await sexy_primes(100) → [[5, 11], [7, 13], [13, 19], ...]
        await sexy_primes(50) → [[5, 11], [7, 13], [13, 19], ...]
    """
    if limit < 11:
        return []

    sexy_pairs = []
    primes = await _sieve_of_eratosthenes(limit)
    prime_set = set(primes)

    for p in primes:
        if p + 6 <= limit and p + 6 in prime_set:
            sexy_pairs.append([p, p + 6])

    return sexy_pairs


@mcp_function(
    description="Find all prime triplets up to limit.",
    namespace="arithmetic",
    category="prime_patterns",
    execution_modes=["local", "remote"],
    estimated_cpu_usage="medium",
    examples=[
        {
            "input": {"limit": 100},
            "output": [
                {
                    "type": "(p, p+2, p+6)",
                    "triplets": [
                        [5, 7, 11],
                        [11, 13, 17],
                        [17, 19, 23],
                        [41, 43, 47],
                        [61, 67, 71],
                    ],
                },
                {
                    "type": "(p, p+4, p+6)",
                    "triplets": [[7, 11, 13], [13, 17, 19], [37, 41, 43], [67, 71, 73]],
                },
            ],
            "description": "Prime triplets up to 100",
        },
        {
            "input": {"limit": 50},
            "output": [
                {
                    "type": "(p, p+2, p+6)",
                    "triplets": [[5, 7, 11], [11, 13, 17], [17, 19, 23], [41, 43, 47]],
                },
                {
                    "type": "(p, p+4, p+6)",
                    "triplets": [[7, 11, 13], [13, 17, 19], [37, 41, 43]],
                },
            ],
            "description": "Prime triplets up to 50",
        },
    ],
)
async def prime_triplets(limit: int) -> List[Dict]:
    """
    Find all prime triplets of the form (p, p+2, p+6) and (p, p+4, p+6).

    Args:
        limit: Upper bound for the search

    Returns:
        List of dictionaries with triplet types and their instances

    Examples:
        await prime_triplets(100) → [{"type": "(p, p+2, p+6)", "triplets": [[5, 7, 11], ...]}, ...]
        await prime_triplets(50) → [{"type": "(p, p+2, p+6)", "triplets": [[5, 7, 11], ...]}, ...]
    """
    if limit < 11:
        return [
            {"type": "(p, p+2, p+6)", "triplets": []},
            {"type": "(p, p+4, p+6)", "triplets": []},
        ]

    primes = await _sieve_of_eratosthenes(limit)
    prime_set = set(primes)

    triplets_2_6 = []  # (p, p+2, p+6)
    triplets_4_6 = []  # (p, p+4, p+6)

    for p in primes:
        # Check (p, p+2, p+6) triplets
        if p + 6 <= limit and p + 2 in prime_set and p + 6 in prime_set:
            triplets_2_6.append([p, p + 2, p + 6])

        # Check (p, p+4, p+6) triplets
        if p + 6 <= limit and p + 4 in prime_set and p + 6 in prime_set:
            triplets_4_6.append([p, p + 4, p + 6])

    return [
        {"type": "(p, p+2, p+6)", "triplets": triplets_2_6},
        {"type": "(p, p+4, p+6)", "triplets": triplets_4_6},
    ]


@mcp_function(
    description="Find all prime quadruplets up to limit.",
    namespace="arithmetic",
    category="prime_patterns",
    execution_modes=["local", "remote"],
    estimated_cpu_usage="medium",
    examples=[
        {
            "input": {"limit": 100},
            "output": [
                {
                    "type": "(p, p+2, p+6, p+8)",
                    "quadruplets": [[5, 7, 11, 13], [11, 13, 17, 19]],
                },
                {"type": "(p, p+4, p+6, p+10)", "quadruplets": [[7, 11, 13, 17]]},
            ],
            "description": "Prime quadruplets up to 100",
        },
        {
            "input": {"limit": 200},
            "output": [
                {
                    "type": "(p, p+2, p+6, p+8)",
                    "quadruplets": [
                        [5, 7, 11, 13],
                        [11, 13, 17, 19],
                        [101, 103, 107, 109],
                        [191, 193, 197, 199],
                    ],
                },
                {"type": "(p, p+4, p+6, p+10)", "quadruplets": [[7, 11, 13, 17]]},
            ],
            "description": "Prime quadruplets up to 200",
        },
    ],
)
async def prime_quadruplets(limit: int) -> List[Dict]:
    """
    Find all prime quadruplets of common patterns.

    Searches for patterns like (p, p+2, p+6, p+8) and (p, p+4, p+6, p+10).

    Args:
        limit: Upper bound for the search

    Returns:
        List of dictionaries with quadruplet types and their instances

    Examples:
        await prime_quadruplets(100) → [{"type": "(p, p+2, p+6, p+8)", "quadruplets": [[5, 7, 11, 13], ...]}, ...]
        await prime_quadruplets(200) → [{"type": "(p, p+2, p+6, p+8)", "quadruplets": [[5, 7, 11, 13], ...]}, ...]
    """
    if limit < 13:
        return [
            {"type": "(p, p+2, p+6, p+8)", "quadruplets": []},
            {"type": "(p, p+4, p+6, p+10)", "quadruplets": []},
        ]

    primes = await _sieve_of_eratosthenes(limit)
    prime_set = set(primes)

    quadruplets_2_6_8 = []  # (p, p+2, p+6, p+8)
    quadruplets_4_6_10 = []  # (p, p+4, p+6, p+10)

    for p in primes:
        # Check (p, p+2, p+6, p+8) quadruplets
        if (
            p + 8 <= limit
            and p + 2 in prime_set
            and p + 6 in prime_set
            and p + 8 in prime_set
        ):
            quadruplets_2_6_8.append([p, p + 2, p + 6, p + 8])

        # Check (p, p+4, p+6, p+10) quadruplets
        if (
            p + 10 <= limit
            and p + 4 in prime_set
            and p + 6 in prime_set
            and p + 10 in prime_set
        ):
            quadruplets_4_6_10.append([p, p + 4, p + 6, p + 10])

    return [
        {"type": "(p, p+2, p+6, p+8)", "quadruplets": quadruplets_2_6_8},
        {"type": "(p, p+4, p+6, p+10)", "quadruplets": quadruplets_4_6_10},
    ]


@mcp_function(
    description="Find prime constellations of specified pattern.",
    namespace="arithmetic",
    category="prime_patterns",
    execution_modes=["local", "remote"],
    estimated_cpu_usage="high",
    examples=[
        {
            "input": {"pattern": [0, 2, 6, 8], "limit": 100},
            "output": [[5, 7, 11, 13], [11, 13, 17, 19]],
            "description": "Pattern (0, 2, 6, 8) up to 100",
        },
        {
            "input": {"pattern": [0, 4, 6], "limit": 50},
            "output": [[7, 11, 13], [13, 17, 19], [37, 41, 43]],
            "description": "Pattern (0, 4, 6) up to 50",
        },
        {
            "input": {"pattern": [0, 6], "limit": 100},
            "output": [
                [5, 11],
                [7, 13],
                [13, 19],
                [17, 23],
                [23, 29],
                [31, 37],
                [37, 43],
                [41, 47],
                [61, 67],
                [67, 73],
                [73, 79],
                [83, 89],
            ],
            "description": "Sexy primes pattern up to 100",
        },
    ],
)
async def prime_constellations(pattern: List[int], limit: int) -> List[List[int]]:
    """
    Find prime constellations matching a specified pattern.

    A prime constellation is a sequence of primes with a specific difference pattern.

    Args:
        pattern: List of offsets from the first prime (first element should be 0)
        limit: Upper bound for the search

    Returns:
        List of prime constellations matching the pattern

    Examples:
        await prime_constellations([0, 2, 6, 8], 100) → [[5, 7, 11, 13], [11, 13, 17, 19]]
        await prime_constellations([0, 4, 6], 50) → [[7, 11, 13], [13, 17, 19], [37, 41, 43]]
        await prime_constellations([0, 6], 100) → [[5, 11], [7, 13], ...]  # Sexy primes
    """
    if not pattern or pattern[0] != 0:
        raise ValueError("Pattern must start with 0")

    if limit < 2:
        return []

    max_offset = max(pattern)
    if max_offset == 0:
        # Single prime pattern
        primes = await _sieve_of_eratosthenes(limit)
        return [[p] for p in primes]

    primes = await _sieve_of_eratosthenes(limit)
    prime_set = set(primes)
    constellations = []

    for p in primes:
        if p + max_offset > limit:
            break

        # Check if all offsets give primes
        constellation = []
        is_valid = True

        for offset in pattern:
            candidate = p + offset
            if candidate not in prime_set:
                is_valid = False
                break
            constellation.append(candidate)

        if is_valid:
            constellations.append(constellation)

    return constellations


@mcp_function(
    description="Check if a given pattern is admissible for prime constellations.",
    namespace="arithmetic",
    category="prime_patterns",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    examples=[
        {
            "input": {"pattern": [0, 2, 6, 8]},
            "output": {
                "admissible": True,
                "reason": "Pattern avoids all small prime divisors",
            },
            "description": "Check if (0, 2, 6, 8) is admissible",
        },
        {
            "input": {"pattern": [0, 2, 4]},
            "output": {
                "admissible": False,
                "reason": "Pattern hits all residues mod 3",
            },
            "description": "Check if (0, 2, 4) is admissible",
        },
        {
            "input": {"pattern": [0, 6, 12]},
            "output": {
                "admissible": False,
                "reason": "Pattern hits all residues mod 3",
            },
            "description": "Check if (0, 6, 12) is admissible",
        },
    ],
)
async def is_admissible_pattern(pattern: List[int]) -> Dict:
    """
    Check if a pattern is admissible for prime constellations.

    A pattern is admissible if it doesn't cover all residue classes modulo
    any prime p, which would make infinite constellations impossible.

    Args:
        pattern: List of offsets (should start with 0)

    Returns:
        Dictionary with admissibility result and explanation

    Examples:
        await is_admissible_pattern([0, 2, 6, 8]) → {"admissible": True, ...}
        await is_admissible_pattern([0, 2, 4]) → {"admissible": False, ...}
    """
    if not pattern:
        return {"admissible": True, "reason": "Empty pattern"}

    if pattern[0] != 0:
        return {"admissible": False, "reason": "Pattern must start with 0"}

    # Check small primes (up to reasonable limit)
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]

    for p in small_primes:
        residues = set(offset % p for offset in pattern)

        # If pattern covers all residue classes mod p, it's not admissible
        if len(residues) == p:
            return {
                "admissible": False,
                "reason": f"Pattern hits all residues mod {p}",
                "blocking_prime": p,
            }

    return {
        "admissible": True,
        "reason": "Pattern avoids all small prime divisors",
        "checked_primes": small_primes,
    }


# ============================================================================
# PRIME DISTRIBUTION AND COUNTING
# ============================================================================


@mcp_function(
    description="Approximate the prime counting function π(x).",
    namespace="arithmetic",
    category="prime_distribution",
    execution_modes=["local", "remote"],
    estimated_cpu_usage="medium",
    examples=[
        {
            "input": {"x": 100},
            "output": {
                "exact": 25,
                "li_approximation": 29.0809,
                "pnt_approximation": 21.7147,
                "best_approximation": "li",
            },
            "description": "π(100) with various approximations",
        },
        {
            "input": {"x": 1000},
            "output": {
                "exact": 168,
                "li_approximation": 177.6096,
                "pnt_approximation": 144.7648,
                "best_approximation": "li",
            },
            "description": "π(1000) with various approximations",
        },
        {
            "input": {"x": 10000},
            "output": {
                "exact": 1229,
                "li_approximation": 1245.0952,
                "pnt_approximation": 1085.7360,
                "best_approximation": "li",
            },
            "description": "π(10000) with various approximations",
        },
    ],
)
async def prime_counting_function(x: int) -> Dict:
    """
    Calculate π(x) (number of primes ≤ x) and compare with theoretical approximations.

    Compares exact count with:
    - Prime Number Theorem: π(x) ≈ x / ln(x)
    - Logarithmic Integral: li(x) ≈ ∫₂ˣ dt/ln(t)

    Args:
        x: Upper bound for counting primes

    Returns:
        Dictionary with exact count and various approximations

    Examples:
        await prime_counting_function(100) → {"exact": 25, "li_approximation": 29.08, ...}
        await prime_counting_function(1000) → {"exact": 168, "li_approximation": 177.61, ...}
    """
    if x < 2:
        return {"exact": 0, "li_approximation": 0, "pnt_approximation": 0}

    # Count exact primes
    primes = await _sieve_of_eratosthenes(x)
    exact_count = len(primes)

    # Prime Number Theorem approximation: π(x) ≈ x / ln(x)
    if x >= 2:
        pnt_approximation = x / math.log(x)
    else:
        pnt_approximation = 0

    # Logarithmic integral approximation (more accurate)
    # li(x) ≈ x/ln(x) + x/ln²(x) + 2x/ln³(x) + ... (asymptotic series)
    if x >= 2:
        ln_x = math.log(x)
        li_approximation = x / ln_x + x / (ln_x * ln_x) + 2 * x / (ln_x * ln_x * ln_x)
    else:
        li_approximation = 0

    # Determine which approximation is better
    pnt_error = abs(exact_count - pnt_approximation)
    li_error = abs(exact_count - li_approximation)

    best_approximation = "li" if li_error < pnt_error else "pnt"

    return {
        "exact": exact_count,
        "li_approximation": round(li_approximation, 4),
        "pnt_approximation": round(pnt_approximation, 4),
        "li_error": round(li_error, 4),
        "pnt_error": round(pnt_error, 4),
        "best_approximation": best_approximation,
        "li_error_percentage": round(100 * li_error / exact_count, 2)
        if exact_count > 0
        else 0,
        "pnt_error_percentage": round(100 * pnt_error / exact_count, 2)
        if exact_count > 0
        else 0,
    }


@mcp_function(
    description="Analyze the error in Prime Number Theorem approximation.",
    namespace="arithmetic",
    category="prime_distribution",
    execution_modes=["local", "remote"],
    estimated_cpu_usage="medium",
    examples=[
        {
            "input": {"x": 1000},
            "output": {
                "error": 23.2352,
                "relative_error": 13.83,
                "theoretical_bound": "O(x * exp(-c * sqrt(log x)))",
            },
            "description": "PNT error analysis for x=1000",
        },
        {
            "input": {"x": 10000},
            "output": {
                "error": 143.2640,
                "relative_error": 11.66,
                "theoretical_bound": "O(x * exp(-c * sqrt(log x)))",
            },
            "description": "PNT error analysis for x=10000",
        },
    ],
)
async def prime_number_theorem_error(x: int) -> Dict:
    """
    Analyze the error in Prime Number Theorem approximation π(x) ≈ x/ln(x).

    Args:
        x: Value to analyze error at

    Returns:
        Dictionary with error analysis information

    Examples:
        await prime_number_theorem_error(1000) → {"error": 23.24, "relative_error": 13.83, ...}
        await prime_number_theorem_error(10000) → {"error": 143.26, "relative_error": 11.66, ...}
    """
    if x < 2:
        return {"error": 0, "relative_error": 0}

    # Get exact count and approximation
    counting_result = await prime_counting_function(x)
    exact = counting_result["exact"]
    pnt_approx = counting_result["pnt_approximation"]

    error = abs(exact - pnt_approx)
    relative_error = (error / exact * 100) if exact > 0 else 0

    # Theoretical error bound (simplified)
    ln_x = math.log(x)
    sqrt_ln_x = math.sqrt(ln_x)

    return {
        "x": x,
        "exact_pi_x": exact,
        "pnt_approximation": round(pnt_approx, 4),
        "error": round(error, 4),
        "relative_error": round(relative_error, 2),
        "theoretical_bound": "O(x * exp(-c * sqrt(log x)))",
        "ln_x": round(ln_x, 4),
        "sqrt_ln_x": round(sqrt_ln_x, 4),
        "error_density": round(error / x, 6),
    }


@mcp_function(
    description="Analyze prime gaps in a given range.",
    namespace="arithmetic",
    category="prime_distribution",
    execution_modes=["local", "remote"],
    estimated_cpu_usage="medium",
    examples=[
        {
            "input": {"start": 2, "end": 100},
            "output": {
                "gaps": [
                    1,
                    2,
                    2,
                    4,
                    2,
                    4,
                    2,
                    4,
                    6,
                    2,
                    6,
                    4,
                    2,
                    4,
                    6,
                    6,
                    2,
                    6,
                    4,
                    2,
                    6,
                    4,
                    6,
                    8,
                    4,
                ],
                "max_gap": 8,
                "avg_gap": 3.76,
                "gap_distribution": {1: 1, 2: 8, 4: 7, 6: 7, 8: 1},
            },
            "description": "Prime gaps from 2 to 100",
        },
        {
            "input": {"start": 100, "end": 200},
            "output": {
                "gaps": [2, 6, 4, 2, 4, 6, 6, 2, 6, 6, 4, 2, 4, 6, 2, 6, 6, 4, 6, 8],
                "max_gap": 8,
                "avg_gap": 4.6,
                "gap_distribution": {2: 4, 4: 5, 6: 9, 8: 1},
            },
            "description": "Prime gaps from 100 to 200",
        },
    ],
)
async def prime_gaps_analysis(start: int, end: int) -> Dict:
    """
    Analyze the distribution of gaps between consecutive primes.

    Args:
        start: Starting value for analysis
        end: Ending value for analysis

    Returns:
        Dictionary with gap statistics and distribution

    Examples:
        await prime_gaps_analysis(2, 100) → {"gaps": [1, 2, 2, 4, ...], "max_gap": 8, ...}
        await prime_gaps_analysis(100, 200) → {"gaps": [2, 6, 4, 2, ...], "max_gap": 8, ...}
    """
    if start < 2:
        start = 2
    if end < start:
        return {"gaps": [], "max_gap": 0, "avg_gap": 0, "gap_distribution": {}}

    # Get primes in range
    all_primes = await _sieve_of_eratosthenes(end)
    primes_in_range = [p for p in all_primes if start <= p <= end]

    if len(primes_in_range) < 2:
        return {"gaps": [], "max_gap": 0, "avg_gap": 0, "gap_distribution": {}}

    # Calculate gaps
    gaps = []
    for i in range(1, len(primes_in_range)):
        gap = primes_in_range[i] - primes_in_range[i - 1]
        gaps.append(gap)

    # Statistics
    max_gap = max(gaps) if gaps else 0
    avg_gap = sum(gaps) / len(gaps) if gaps else 0

    # Gap distribution
    gap_distribution: defaultdict[int, int] = defaultdict(int)
    for gap in gaps:
        gap_distribution[gap] += 1

    # Convert defaultdict to regular dict for JSON serialization
    gap_distribution = dict(gap_distribution)  # type: ignore[assignment]

    return {
        "range": [start, end],
        "num_primes": len(primes_in_range),
        "gaps": gaps,
        "max_gap": max_gap,
        "min_gap": min(gaps) if gaps else 0,
        "avg_gap": round(avg_gap, 2),
        "gap_distribution": gap_distribution,
        "unique_gaps": len(gap_distribution),
        "most_common_gap": max(gap_distribution.items(), key=lambda x: x[1])[0]
        if gap_distribution
        else 0,
    }


# ============================================================================
# PRIME CONJECTURES AND VERIFICATION
# ============================================================================


@mcp_function(
    description="Verify Bertrand's postulate for given n.",
    namespace="arithmetic",
    category="prime_conjectures",
    execution_modes=["local", "remote"],
    examples=[
        {
            "input": {"n": 25},
            "output": {
                "holds": True,
                "primes_between": [29, 31, 37, 41, 43, 47],
                "smallest_prime": 29,
            },
            "description": "Bertrand's postulate for n=25",
        },
        {
            "input": {"n": 100},
            "output": {
                "holds": True,
                "primes_between": [
                    101,
                    103,
                    107,
                    109,
                    113,
                    127,
                    131,
                    137,
                    139,
                    149,
                    151,
                    157,
                    163,
                    167,
                    173,
                    179,
                    181,
                    191,
                    193,
                    197,
                    199,
                ],
                "smallest_prime": 101,
            },
            "description": "Bertrand's postulate for n=100",
        },
        {
            "input": {"n": 10},
            "output": {
                "holds": True,
                "primes_between": [11, 13, 17, 19],
                "smallest_prime": 11,
            },
            "description": "Bertrand's postulate for n=10",
        },
    ],
)
async def bertrand_postulate_verify(n: int) -> Dict:
    """
    Verify Bertrand's postulate: for every n > 1, there exists a prime p such that n < p < 2n.

    Args:
        n: Value to test Bertrand's postulate for

    Returns:
        Dictionary with verification result and found primes

    Examples:
        await bertrand_postulate_verify(25) → {"holds": True, "primes_between": [29, 31, ...], ...}
        await bertrand_postulate_verify(100) → {"holds": True, "primes_between": [101, 103, ...], ...}
    """
    if n <= 1:
        return {"holds": False, "reason": "Bertrand's postulate only applies for n > 1"}

    # Find primes in range (n, 2n)
    upper_bound = 2 * n
    all_primes = await _sieve_of_eratosthenes(upper_bound)
    primes_between = [p for p in all_primes if n < p < upper_bound]

    holds = len(primes_between) > 0

    result = {
        "n": n,
        "range": [n + 1, upper_bound - 1],
        "holds": holds,
        "primes_between": primes_between,
        "count": len(primes_between),
    }

    if holds:
        result["smallest_prime"] = min(primes_between)
        result["largest_prime"] = max(primes_between)

    return result


@mcp_function(
    description="Collect data about twin prime conjecture up to limit.",
    namespace="arithmetic",
    category="prime_conjectures",
    execution_modes=["local", "remote"],
    estimated_cpu_usage="medium",
    examples=[
        {
            "input": {"limit": 100},
            "output": {
                "twin_prime_pairs": [
                    [3, 5],
                    [5, 7],
                    [11, 13],
                    [17, 19],
                    [29, 31],
                    [41, 43],
                    [59, 61],
                    [71, 73],
                ],
                "count": 8,
                "density": 0.08,
                "largest_pair": [71, 73],
            },
            "description": "Twin prime data up to 100",
        },
        {
            "input": {"limit": 1000},
            "output": {
                "twin_prime_pairs": 35,
                "count": 35,
                "density": 0.035,
                "largest_pair": [881, 883],
            },
            "description": "Twin prime count up to 1000",
        },
    ],
)
async def twin_prime_conjecture_data(limit: int, return_all_pairs: bool = True) -> Dict:
    """
    Collect data supporting the twin prime conjecture.

    Args:
        limit: Upper bound for search
        return_all_pairs: If False, only return count (for large limits)

    Returns:
        Dictionary with twin prime statistics

    Examples:
        await twin_prime_conjecture_data(100) → {"twin_prime_pairs": [[3, 5], ...], "count": 8, ...}
        await twin_prime_conjecture_data(1000, False) → {"count": 35, "density": 0.035, ...}
    """
    if limit < 5:
        return {"twin_prime_pairs": [], "count": 0, "density": 0}

    primes = await _sieve_of_eratosthenes(limit)
    prime_set = set(primes)

    twin_pairs = []
    for p in primes:
        if p + 2 <= limit and p + 2 in prime_set:
            twin_pairs.append([p, p + 2])

    count = len(twin_pairs)
    density = count / limit if limit > 0 else 0

    result = {"limit": limit, "count": count, "density": round(density, 6)}

    if return_all_pairs and count <= 1000:  # Avoid huge return values
        result["twin_prime_pairs"] = twin_pairs  # type: ignore[assignment]

    if twin_pairs:
        result["largest_pair"] = twin_pairs[-1]  # type: ignore[assignment]
        result["first_pair"] = twin_pairs[0]  # type: ignore[assignment]

    return result


@mcp_function(
    description="Find record prime gaps (first occurrence of each gap size).",
    namespace="arithmetic",
    category="prime_conjectures",
    execution_modes=["local", "remote"],
    estimated_cpu_usage="high",
    examples=[
        {
            "input": {"limit": 1000},
            "output": {
                "records": {
                    1: [2, 3],
                    2: [3, 5],
                    4: [7, 11],
                    6: [23, 29],
                    8: [89, 97],
                    14: [113, 127],
                    18: [523, 541],
                    20: [887, 907],
                },
                "max_gap": 20,
                "total_records": 8,
            },
            "description": "Record prime gaps up to 1000",
        },
        {
            "input": {"limit": 500},
            "output": {
                "records": {
                    1: [2, 3],
                    2: [3, 5],
                    4: [7, 11],
                    6: [23, 29],
                    8: [89, 97],
                    14: [113, 127],
                },
                "max_gap": 14,
                "total_records": 6,
            },
            "description": "Record prime gaps up to 500",
        },
    ],
)
async def prime_gap_records(limit: int) -> Dict:
    """
    Find record prime gaps (first occurrence of each gap size).

    Args:
        limit: Upper bound for search

    Returns:
        Dictionary with record gaps and their first occurrences

    Examples:
        await prime_gap_records(1000) → {"records": {1: [2, 3], 2: [3, 5], ...}, ...}
        await prime_gap_records(500) → {"records": {1: [2, 3], 2: [3, 5], ...}, ...}
    """
    if limit < 3:
        return {"records": {}, "max_gap": 0, "total_records": 0}

    primes = await _sieve_of_eratosthenes(limit)

    if len(primes) < 2:
        return {"records": {}, "max_gap": 0, "total_records": 0}

    records = {}
    seen_gaps = set()

    for i in range(1, len(primes)):
        gap = primes[i] - primes[i - 1]

        # Record first occurrence of this gap size
        if gap not in seen_gaps:
            records[gap] = [primes[i - 1], primes[i]]
            seen_gaps.add(gap)

    max_gap = max(records.keys()) if records else 0

    return {
        "limit": limit,
        "records": records,
        "max_gap": max_gap,
        "total_records": len(records),
        "gap_sizes": sorted(records.keys()),
    }


# ============================================================================
# ADVANCED ANALYSIS AND VISUALIZATION HELPERS
# ============================================================================


@mcp_function(
    description="Analyze prime density in intervals.",
    namespace="arithmetic",
    category="prime_analysis",
    execution_modes=["local", "remote"],
    estimated_cpu_usage="medium",
    examples=[
        {
            "input": {"limit": 1000, "interval_size": 100},
            "output": {
                "intervals": [
                    [1, 100, 25],
                    [101, 200, 21],
                    [201, 300, 16],
                    [301, 400, 16],
                    [401, 500, 17],
                    [501, 600, 14],
                    [601, 700, 16],
                    [701, 800, 14],
                    [801, 900, 15],
                    [901, 1000, 14],
                ],
                "avg_density": 0.168,
                "max_density_interval": [1, 100],
                "min_density_interval": [501, 600],
            },
            "description": "Prime density in intervals of 100 up to 1000",
        },
        {
            "input": {"limit": 500, "interval_size": 50},
            "output": {
                "intervals": [
                    [1, 50, 15],
                    [51, 100, 10],
                    [101, 150, 11],
                    [151, 200, 10],
                    [201, 250, 9],
                    [251, 300, 7],
                    [301, 350, 8],
                    [351, 400, 8],
                    [401, 450, 9],
                    [451, 500, 8],
                ],
                "avg_density": 0.19,
                "max_density_interval": [1, 50],
                "min_density_interval": [251, 300],
            },
            "description": "Prime density in intervals of 50 up to 500",
        },
    ],
)
async def prime_density_analysis(limit: int, interval_size: int) -> Dict:
    """
    Analyze how prime density varies across intervals.

    Args:
        limit: Upper bound for analysis
        interval_size: Size of each interval

    Returns:
        Dictionary with density analysis across intervals

    Examples:
        await prime_density_analysis(1000, 100) → {"intervals": [[1, 100, 25], ...], ...}
        await prime_density_analysis(500, 50) → {"intervals": [[1, 50, 15], ...], ...}
    """
    if limit <= 0 or interval_size <= 0:
        return {"intervals": [], "avg_density": 0}

    primes = await _sieve_of_eratosthenes(limit)
    set(primes)

    intervals = []
    densities = []

    for start in range(1, limit + 1, interval_size):
        end = min(start + interval_size - 1, limit)

        # Count primes in this interval
        count = sum(1 for p in primes if start <= p <= end)

        density = count / (end - start + 1)
        intervals.append([start, end, count])
        densities.append(density)

    if not densities:
        return {"intervals": [], "avg_density": 0}

    avg_density = sum(densities) / len(densities)
    max_density_idx = densities.index(max(densities))
    min_density_idx = densities.index(min(densities))

    return {
        "limit": limit,
        "interval_size": interval_size,
        "intervals": intervals,
        "avg_density": round(avg_density, 3),
        "max_density": round(max(densities), 3),
        "min_density": round(min(densities), 3),
        "max_density_interval": intervals[max_density_idx][:2],
        "min_density_interval": intervals[min_density_idx][:2],
        "density_variance": round(
            sum((d - avg_density) ** 2 for d in densities) / len(densities), 6
        ),
    }


@mcp_function(
    description="Generate data for Ulam spiral prime visualization.",
    namespace="arithmetic",
    category="prime_analysis",
    execution_modes=["local", "remote"],
    estimated_cpu_usage="high",
    examples=[
        {
            "input": {"size": 7},
            "output": {
                "spiral": [
                    [37, 36, 35, 34, 33, 32, 31],
                    [38, 17, 16, 15, 14, 13, 30],
                    [39, 18, 3, 2, 1, 12, 29],
                    [40, 19, 4, 0, 0, 11, 28],
                    [41, 20, 5, 6, 7, 10, 27],
                    [42, 21, 22, 23, 24, 25, 26],
                    [43, 44, 45, 46, 47, 48, 49],
                ],
                "primes_marked": True,
                "center": [3, 3],
            },
            "description": "7x7 Ulam spiral with primes marked",
        },
        {
            "input": {"size": 5},
            "output": {
                "spiral": [
                    [21, 22, 23, 24, 25],
                    [20, 7, 8, 9, 10],
                    [19, 6, 1, 2, 11],
                    [18, 5, 4, 3, 12],
                    [17, 16, 15, 14, 13],
                ],
                "primes_marked": True,
                "center": [2, 2],
            },
            "description": "5x5 Ulam spiral with primes marked",
        },
    ],
)
async def ulam_spiral_analysis(size: int) -> Dict:
    """
    Generate Ulam spiral data for prime visualization.

    The Ulam spiral arranges natural numbers in a spiral pattern,
    revealing interesting patterns when primes are marked.

    Args:
        size: Size of the square spiral (must be odd)

    Returns:
        Dictionary with spiral data and prime information

    Examples:
        await ulam_spiral_analysis(7) → {"spiral": [[37, 36, ...]], "primes_marked": True, ...}
        await ulam_spiral_analysis(5) → {"spiral": [[21, 22, ...]], "primes_marked": True, ...}
    """
    if size <= 0 or size % 2 == 0:
        raise ValueError("Size must be a positive odd number")

    # Create spiral matrix
    spiral = [[0 for _ in range(size)] for _ in range(size)]

    # Starting position (center)
    x, y = size // 2, size // 2
    spiral[x][y] = 1

    # Direction vectors: right, down, left, up
    dx = [0, 1, 0, -1]
    dy = [1, 0, -1, 0]
    direction = 0

    num = 2
    steps = 1

    while num <= size * size:
        for _ in range(2):  # Each step count is used twice
            for _ in range(steps):
                if num > size * size:
                    break

                x += dx[direction]
                y += dy[direction]

                if 0 <= x < size and 0 <= y < size:
                    spiral[x][y] = num
                    num += 1

            direction = (direction + 1) % 4
            if num > size * size:
                break

        steps += 1

    # Mark primes (replace prime numbers with negative values for identification)
    max_num = size * size
    primes = await _sieve_of_eratosthenes(max_num)
    prime_set = set(primes)

    prime_positions = []
    for i in range(size):
        for j in range(size):
            if spiral[i][j] in prime_set:
                prime_positions.append([i, j, spiral[i][j]])
                # Keep original number but mark as prime location

    return {
        "size": size,
        "spiral": spiral,
        "primes_marked": True,
        "center": [size // 2, size // 2],
        "prime_positions": prime_positions,
        "total_numbers": size * size,
        "prime_count": len(prime_positions),
        "prime_density": round(len(prime_positions) / (size * size), 3),
    }


# Export all functions
__all__ = [
    # Prime constellations and patterns
    "cousin_primes",
    "sexy_primes",
    "prime_triplets",
    "prime_quadruplets",
    "prime_constellations",
    "is_admissible_pattern",
    # Prime distribution and counting
    "prime_counting_function",
    "prime_number_theorem_error",
    "prime_gaps_analysis",
    # Prime conjectures and verification
    "bertrand_postulate_verify",
    "twin_prime_conjecture_data",
    "prime_gap_records",
    # Advanced analysis
    "prime_density_analysis",
    "ulam_spiral_analysis",
]

if __name__ == "__main__":
    import asyncio

    async def test_advanced_prime_patterns():
        """Test advanced prime patterns and distribution functions."""
        print("🔍 Advanced Prime Patterns Test")
        print("=" * 35)

        # Test prime constellations
        print("Prime Constellations:")
        cousin = await cousin_primes(50)
        print(f"  cousin_primes(50) = {cousin}")

        sexy = await sexy_primes(50)
        print(f"  sexy_primes(50) = {sexy}")

        triplets = await prime_triplets(50)
        print(f"  prime_triplets(50) = {triplets}")

        # Test prime distribution
        print("\nPrime Distribution:")
        counting = await prime_counting_function(100)
        print(f"  prime_counting_function(100) = {counting}")

        gaps = await prime_gaps_analysis(10, 50)
        print(f"  prime_gaps_analysis(10, 50) = {gaps}")

        # Test conjectures
        print("\nPrime Conjectures:")
        bertrand = await bertrand_postulate_verify(25)
        print(f"  bertrand_postulate_verify(25) = {bertrand}")

        twin_data = await twin_prime_conjecture_data(100, False)
        print(f"  twin_prime_conjecture_data(100) = {twin_data}")

        # Test analysis
        print("\nAdvanced Analysis:")
        density = await prime_density_analysis(200, 50)
        print(f"  prime_density_analysis(200, 50) = {density}")

        print("\n✅ All advanced prime pattern functions working!")

    asyncio.run(test_advanced_prime_patterns())
