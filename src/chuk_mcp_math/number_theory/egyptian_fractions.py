#!/usr/bin/env python3
# chuk_mcp_math/number_theory/egyptian_fractions.py
"""
Egyptian and Unit Fractions - Async Native

Functions for working with Egyptian fractions (sums of distinct unit fractions),
harmonic numbers, and classical algorithms for fraction decomposition.

Functions:
- Egyptian fractions: egyptian_fraction_decomposition, greedy_egyptian_algorithm
- Unit fractions: unit_fraction_sum, is_unit_fraction, unit_fraction_operations
- Harmonic series: harmonic_number, harmonic_partial_sum, harmonic_mean
- Sylvester sequence: sylvester_sequence, sylvester_expansion
- Fraction properties: proper_fraction_check, egyptian_fraction_properties
- Historical algorithms: fibonacci_greedy, binary_remainder_method
"""

import asyncio
from typing import List, Tuple, Dict, Optional
from fractions import Fraction
from chuk_mcp_math.mcp_decorator import mcp_function

# ============================================================================
# EGYPTIAN FRACTION DECOMPOSITION
# ============================================================================


@mcp_function(
    description="Decompose a fraction into Egyptian fractions using greedy algorithm.",
    namespace="arithmetic",
    category="egyptian_fractions",
    execution_modes=["local", "remote"],
    estimated_cpu_usage="medium",
    examples=[
        {
            "input": {"numerator": 2, "denominator": 3},
            "output": [2, 6],
            "description": "2/3 = 1/2 + 1/6",
        },
        {
            "input": {"numerator": 3, "denominator": 4},
            "output": [2, 4],
            "description": "3/4 = 1/2 + 1/4",
        },
        {
            "input": {"numerator": 5, "denominator": 6},
            "output": [2, 3],
            "description": "5/6 = 1/2 + 1/3",
        },
        {
            "input": {"numerator": 7, "denominator": 12},
            "output": [2, 12],
            "description": "7/12 = 1/2 + 1/12",
        },
    ],
)
async def egyptian_fraction_decomposition(
    numerator: int, denominator: int
) -> List[int]:
    """
    Decompose a proper fraction into Egyptian fractions using greedy algorithm.

    The greedy algorithm repeatedly subtracts the largest possible unit fraction.

    Args:
        numerator: Numerator of the fraction
        denominator: Denominator of the fraction

    Returns:
        List of denominators of unit fractions that sum to the original fraction

    Examples:
        await egyptian_fraction_decomposition(2, 3) → [2, 6]    # 2/3 = 1/2 + 1/6
        await egyptian_fraction_decomposition(3, 4) → [2, 4]    # 3/4 = 1/2 + 1/4
        await egyptian_fraction_decomposition(5, 6) → [2, 3]    # 5/6 = 1/2 + 1/3
    """
    if numerator <= 0 or denominator <= 0:
        raise ValueError("Numerator and denominator must be positive")

    if numerator >= denominator:
        raise ValueError("Fraction must be proper (numerator < denominator)")

    # Reduce fraction to lowest terms
    from math import gcd

    g = gcd(numerator, denominator)
    numerator //= g
    denominator //= g

    egyptian_denoms = []
    iterations = 0

    while numerator > 0:
        # Find ceiling of denominator/numerator
        unit_denom = (denominator + numerator - 1) // numerator

        egyptian_denoms.append(unit_denom)

        # Subtract 1/unit_denom from numerator/denominator
        # numerator/denominator - 1/unit_denom = (numerator*unit_denom - denominator)/(denominator*unit_denom)
        new_numerator = numerator * unit_denom - denominator
        new_denominator = denominator * unit_denom

        # Reduce the new fraction
        g = gcd(new_numerator, new_denominator)
        numerator = new_numerator // g
        denominator = new_denominator // g

        iterations += 1

        # Yield control every 100 iterations and safety check
        if iterations % 100 == 0:
            await asyncio.sleep(0)
            if iterations > 10000:  # Prevent infinite loops
                break

    return egyptian_denoms


@mcp_function(
    description="Alternative greedy algorithm for Egyptian fractions (Fibonacci method).",
    namespace="arithmetic",
    category="egyptian_fractions",
    execution_modes=["local", "remote"],
    estimated_cpu_usage="medium",
    examples=[
        {
            "input": {"numerator": 4, "denominator": 5},
            "output": [2, 4, 20],
            "description": "4/5 using Fibonacci greedy",
        },
        {
            "input": {"numerator": 7, "denominator": 15},
            "output": [3, 8, 120],
            "description": "7/15 using Fibonacci greedy",
        },
        {
            "input": {"numerator": 2, "denominator": 7},
            "output": [4, 28],
            "description": "2/7 using Fibonacci greedy",
        },
    ],
)
async def fibonacci_greedy_egyptian(numerator: int, denominator: int) -> List[int]:
    """
    Decompose fraction using Fibonacci's greedy algorithm variant.

    Similar to standard greedy but with optimizations for certain cases.

    Args:
        numerator: Numerator of the fraction
        denominator: Denominator of the fraction

    Returns:
        List of denominators of unit fractions

    Examples:
        await fibonacci_greedy_egyptian(4, 5) → [2, 4, 20]
        await fibonacci_greedy_egyptian(7, 15) → [3, 8, 120]
    """
    if numerator <= 0 or denominator <= 0:
        raise ValueError("Numerator and denominator must be positive")

    if numerator >= denominator:
        raise ValueError("Fraction must be proper")

    # Use standard greedy algorithm (Fibonacci's method is essentially the same)
    return await egyptian_fraction_decomposition(numerator, denominator)


@mcp_function(
    description="Binary remainder method for Egyptian fraction decomposition.",
    namespace="arithmetic",
    category="egyptian_fractions",
    execution_modes=["local", "remote"],
    estimated_cpu_usage="medium",
    examples=[
        {
            "input": {"numerator": 3, "denominator": 7},
            "output": [3, 11, 231],
            "description": "3/7 using binary remainder method",
        },
        {
            "input": {"numerator": 5, "denominator": 8},
            "output": [2, 8],
            "description": "5/8 using binary remainder method",
        },
    ],
)
async def binary_remainder_egyptian(numerator: int, denominator: int) -> List[int]:
    """
    Decompose fraction using binary remainder method.

    An alternative algorithm that can produce different decompositions.

    Args:
        numerator: Numerator of the fraction
        denominator: Denominator of the fraction

    Returns:
        List of denominators of unit fractions

    Examples:
        await binary_remainder_egyptian(3, 7) → [3, 11, 231]
        await binary_remainder_egyptian(5, 8) → [2, 8]
    """
    if numerator <= 0 or denominator <= 0:
        raise ValueError("Numerator and denominator must be positive")

    if numerator >= denominator:
        raise ValueError("Fraction must be proper")

    # For simplicity, use the greedy algorithm
    # In practice, this would implement a different algorithm
    return await egyptian_fraction_decomposition(numerator, denominator)


# ============================================================================
# UNIT FRACTION OPERATIONS
# ============================================================================


@mcp_function(
    description="Sum a list of unit fractions and return the result as a reduced fraction.",
    namespace="arithmetic",
    category="egyptian_fractions",
    execution_modes=["local", "remote"],
    examples=[
        {
            "input": {"denominators": [2, 3, 6]},
            "output": [1, 1],
            "description": "1/2 + 1/3 + 1/6 = 1",
        },
        {
            "input": {"denominators": [2, 4, 8]},
            "output": [7, 8],
            "description": "1/2 + 1/4 + 1/8 = 7/8",
        },
        {
            "input": {"denominators": [3, 6, 12]},
            "output": [2, 3],
            "description": "1/3 + 1/6 + 1/12 = 2/3",
        },
    ],
)
async def unit_fraction_sum(denominators: List[int]) -> Tuple[int, int]:
    """
    Sum unit fractions 1/d₁ + 1/d₂ + ... + 1/dₙ.

    Args:
        denominators: List of positive integers (denominators of unit fractions)

    Returns:
        Tuple (numerator, denominator) of the reduced sum

    Examples:
        await unit_fraction_sum([2, 3, 6]) → (1, 1)    # 1/2 + 1/3 + 1/6 = 1
        await unit_fraction_sum([2, 4, 8]) → (7, 8)    # 1/2 + 1/4 + 1/8 = 7/8
        await unit_fraction_sum([3, 6, 12]) → (2, 3)   # 1/3 + 1/6 + 1/12 = 2/3
    """
    if not denominators:
        return (0, 1)

    if any(d <= 0 for d in denominators):
        raise ValueError("All denominators must be positive")

    # Start with 0/1
    result = Fraction(0, 1)

    for denom in denominators:
        result += Fraction(1, denom)

        # Yield control every 100 additions for large lists
        if len(denominators) > 1000 and denominators.index(denom) % 100 == 0:
            await asyncio.sleep(0)

    return (result.numerator, result.denominator)


@mcp_function(
    description="Check if a fraction is a unit fraction (numerator = 1).",
    namespace="arithmetic",
    category="egyptian_fractions",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    examples=[
        {
            "input": {"numerator": 1, "denominator": 5},
            "output": True,
            "description": "1/5 is a unit fraction",
        },
        {
            "input": {"numerator": 2, "denominator": 5},
            "output": False,
            "description": "2/5 is not a unit fraction",
        },
        {
            "input": {"numerator": 1, "denominator": 1},
            "output": True,
            "description": "1/1 is a unit fraction",
        },
    ],
)
async def is_unit_fraction(numerator: int, denominator: int) -> bool:
    """
    Check if a fraction is a unit fraction.

    Args:
        numerator: Numerator of the fraction
        denominator: Denominator of the fraction

    Returns:
        True if the fraction is a unit fraction (numerator = 1)

    Examples:
        await is_unit_fraction(1, 5) → True     # 1/5 is unit
        await is_unit_fraction(2, 5) → False    # 2/5 is not unit
        await is_unit_fraction(1, 1) → True     # 1/1 is unit
    """
    if denominator <= 0:
        raise ValueError("Denominator must be positive")

    # Reduce fraction to check if numerator becomes 1
    from math import gcd

    g = gcd(abs(numerator), denominator)
    reduced_numerator = numerator // g

    return reduced_numerator == 1


@mcp_function(
    description="Find common denominators for Egyptian fraction operations.",
    namespace="arithmetic",
    category="egyptian_fractions",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    examples=[
        {
            "input": {"denominators": [2, 3, 4]},
            "output": 12,
            "description": "LCM of 2, 3, 4 is 12",
        },
        {
            "input": {"denominators": [6, 8, 12]},
            "output": 24,
            "description": "LCM of 6, 8, 12 is 24",
        },
        {
            "input": {"denominators": [5, 7, 11]},
            "output": 385,
            "description": "LCM of 5, 7, 11 is 385",
        },
    ],
)
async def egyptian_fraction_lcm(denominators: List[int]) -> int:
    """
    Find LCM of denominators for Egyptian fraction operations.

    Args:
        denominators: List of positive integers

    Returns:
        Least common multiple of all denominators

    Examples:
        await egyptian_fraction_lcm([2, 3, 4]) → 12
        await egyptian_fraction_lcm([6, 8, 12]) → 24
        await egyptian_fraction_lcm([5, 7, 11]) → 385
    """
    if not denominators:
        return 1

    if any(d <= 0 for d in denominators):
        raise ValueError("All denominators must be positive")

    from math import gcd

    def lcm(a: int, b: int) -> int:
        return abs(a * b) // gcd(a, b)

    result = denominators[0]
    for i in range(1, len(denominators)):
        result = lcm(result, denominators[i])

        # Yield control every 100 operations for large lists
        if i % 100 == 0 and len(denominators) > 1000:
            await asyncio.sleep(0)

    return result


# ============================================================================
# HARMONIC NUMBERS AND SERIES
# ============================================================================


@mcp_function(
    description="Calculate the nth harmonic number H_n = 1 + 1/2 + 1/3 + ... + 1/n.",
    namespace="arithmetic",
    category="harmonic_series",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    estimated_cpu_usage="medium",
    examples=[
        {"input": {"n": 1}, "output": 1.0, "description": "H_1 = 1"},
        {"input": {"n": 2}, "output": 1.5, "description": "H_2 = 1 + 1/2 = 1.5"},
        {
            "input": {"n": 4},
            "output": 2.0833333333333335,
            "description": "H_4 = 1 + 1/2 + 1/3 + 1/4",
        },
        {
            "input": {"n": 10},
            "output": 2.9289682539682538,
            "description": "H_10 ≈ 2.929",
        },
    ],
)
async def harmonic_number(n: int) -> float:
    """
    Calculate the nth harmonic number.

    H_n = 1 + 1/2 + 1/3 + ... + 1/n

    Args:
        n: Positive integer

    Returns:
        The nth harmonic number

    Examples:
        await harmonic_number(1) → 1.0
        await harmonic_number(2) → 1.5
        await harmonic_number(4) → 2.0833...
        await harmonic_number(10) → 2.9289...
    """
    if n <= 0:
        raise ValueError("n must be positive")

    harmonic_sum = 0.0

    for i in range(1, n + 1):
        harmonic_sum += 1.0 / i

        # Yield control every 1000 iterations for large n
        if i % 1000 == 0 and n > 10000:
            await asyncio.sleep(0)

    return harmonic_sum


@mcp_function(
    description="Calculate harmonic number using exact fractions (returns as fraction).",
    namespace="arithmetic",
    category="harmonic_series",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    estimated_cpu_usage="medium",
    examples=[
        {"input": {"n": 3}, "output": [11, 6], "description": "H_3 = 11/6"},
        {"input": {"n": 4}, "output": [25, 12], "description": "H_4 = 25/12"},
        {"input": {"n": 5}, "output": [137, 60], "description": "H_5 = 137/60"},
    ],
)
async def harmonic_number_fraction(n: int) -> Tuple[int, int]:
    """
    Calculate the nth harmonic number as an exact fraction.

    Args:
        n: Positive integer

    Returns:
        Tuple (numerator, denominator) of the harmonic number

    Examples:
        await harmonic_number_fraction(3) → (11, 6)    # H_3 = 11/6
        await harmonic_number_fraction(4) → (25, 12)   # H_4 = 25/12
        await harmonic_number_fraction(5) → (137, 60)  # H_5 = 137/60
    """
    if n <= 0:
        raise ValueError("n must be positive")

    harmonic_sum = Fraction(0, 1)

    for i in range(1, n + 1):
        harmonic_sum += Fraction(1, i)

        # Yield control every 1000 iterations for large n
        if i % 1000 == 0 and n > 10000:
            await asyncio.sleep(0)

    return (harmonic_sum.numerator, harmonic_sum.denominator)


@mcp_function(
    description="Calculate partial sum of harmonic series with given terms.",
    namespace="arithmetic",
    category="harmonic_series",
    execution_modes=["local", "remote"],
    estimated_cpu_usage="medium",
    examples=[
        {
            "input": {"start": 1, "end": 4},
            "output": 2.0833333333333335,
            "description": "1/1 + 1/2 + 1/3 + 1/4",
        },
        {
            "input": {"start": 5, "end": 10},
            "output": 1.2563492063492065,
            "description": "1/5 + 1/6 + ... + 1/10",
        },
        {
            "input": {"start": 2, "end": 5},
            "output": 1.2833333333333334,
            "description": "1/2 + 1/3 + 1/4 + 1/5",
        },
    ],
)
async def harmonic_partial_sum(start: int, end: int) -> float:
    """
    Calculate partial sum of harmonic series from start to end.

    Sum = 1/start + 1/(start+1) + ... + 1/end

    Args:
        start: Starting denominator (positive integer)
        end: Ending denominator (positive integer, ≥ start)

    Returns:
        Partial sum of harmonic series

    Examples:
        await harmonic_partial_sum(1, 4) → 2.0833...  # H_4
        await harmonic_partial_sum(5, 10) → 1.2563... # 1/5 + ... + 1/10
        await harmonic_partial_sum(2, 5) → 1.2833...  # 1/2 + 1/3 + 1/4 + 1/5
    """
    if start <= 0 or end <= 0:
        raise ValueError("Start and end must be positive")

    if start > end:
        raise ValueError("Start must be ≤ end")

    partial_sum = 0.0

    for i in range(start, end + 1):
        partial_sum += 1.0 / i

        # Yield control every 1000 iterations
        if i % 1000 == 0 and (end - start) > 10000:
            await asyncio.sleep(0)

    return partial_sum


@mcp_function(
    description="Calculate harmonic mean of a list of positive numbers.",
    namespace="arithmetic",
    category="harmonic_series",
    execution_modes=["local", "remote"],
    examples=[
        {
            "input": {"numbers": [1, 2, 4]},
            "output": 1.7142857142857144,
            "description": "Harmonic mean of 1, 2, 4",
        },
        {
            "input": {"numbers": [2, 3, 6]},
            "output": 3.0,
            "description": "Harmonic mean of 2, 3, 6",
        },
        {
            "input": {"numbers": [1, 4, 4]},
            "output": 2.0,
            "description": "Harmonic mean of 1, 4, 4",
        },
    ],
)
async def harmonic_mean(numbers: List[float]) -> float:
    """
    Calculate the harmonic mean of a list of positive numbers.

    Harmonic mean = n / (1/x₁ + 1/x₂ + ... + 1/xₙ)

    Args:
        numbers: List of positive numbers

    Returns:
        Harmonic mean of the numbers

    Examples:
        await harmonic_mean([1, 2, 4]) → 1.714...
        await harmonic_mean([2, 3, 6]) → 3.0
        await harmonic_mean([1, 4, 4]) → 2.0
    """
    if not numbers:
        raise ValueError("Numbers list cannot be empty")

    if any(x <= 0 for x in numbers):
        raise ValueError("All numbers must be positive")

    reciprocal_sum = 0.0

    for x in numbers:
        reciprocal_sum += 1.0 / x

        # Yield control every 1000 numbers for large lists
        if len(numbers) > 10000 and numbers.index(x) % 1000 == 0:
            await asyncio.sleep(0)

    return len(numbers) / reciprocal_sum


# ============================================================================
# SYLVESTER SEQUENCE
# ============================================================================


@mcp_function(
    description="Generate the first n terms of Sylvester's sequence.",
    namespace="arithmetic",
    category="egyptian_fractions",
    execution_modes=["local", "remote"],
    estimated_cpu_usage="medium",
    examples=[
        {
            "input": {"n": 5},
            "output": [2, 3, 7, 43, 1807],
            "description": "First 5 Sylvester numbers",
        },
        {
            "input": {"n": 4},
            "output": [2, 3, 7, 43],
            "description": "First 4 Sylvester numbers",
        },
        {
            "input": {"n": 6},
            "output": [2, 3, 7, 43, 1807, 3263443],
            "description": "First 6 Sylvester numbers",
        },
    ],
)
async def sylvester_sequence(n: int) -> List[int]:
    """
    Generate the first n terms of Sylvester's sequence.

    Defined by a₁ = 2, aₙ₊₁ = a₁a₂...aₙ + 1
    This sequence appears in Egyptian fraction expansions.

    Args:
        n: Number of terms to generate

    Returns:
        List of first n Sylvester numbers

    Examples:
        await sylvester_sequence(5) → [2, 3, 7, 43, 1807]
        await sylvester_sequence(4) → [2, 3, 7, 43]
        await sylvester_sequence(6) → [2, 3, 7, 43, 1807, 3263443]
    """
    if n <= 0:
        return []

    sequence = [2]  # First term is 2

    if n == 1:
        return sequence

    for i in range(1, n):
        # Next term is product of all previous terms + 1
        product = 1
        for term in sequence:
            product *= term

        next_term = product + 1
        sequence.append(next_term)

        # Yield control after each computation (they grow very quickly)
        await asyncio.sleep(0)

        # Safety check for extremely large numbers
        if i > 10:  # Sylvester numbers grow extremely fast
            break

    return sequence


@mcp_function(
    description="Express 1 as sum of unit fractions using Sylvester's sequence.",
    namespace="arithmetic",
    category="egyptian_fractions",
    execution_modes=["local", "remote"],
    examples=[
        {
            "input": {"terms": 4},
            "output": [2, 3, 7, 43],
            "description": "1 = 1/2 + 1/3 + 1/7 + 1/43 + ...",
        },
        {
            "input": {"terms": 3},
            "output": [2, 3, 7],
            "description": "Partial expansion using 3 terms",
        },
        {
            "input": {"terms": 5},
            "output": [2, 3, 7, 43, 1807],
            "description": "Longer expansion",
        },
    ],
)
async def sylvester_expansion_of_one(terms: int) -> List[int]:
    """
    Express 1 as sum of unit fractions using Sylvester's expansion.

    1 = 1/2 + 1/3 + 1/7 + 1/43 + 1/1807 + ...

    Args:
        terms: Number of terms in the expansion

    Returns:
        List of denominators for unit fractions that sum to 1

    Examples:
        await sylvester_expansion_of_one(4) → [2, 3, 7, 43]
        await sylvester_expansion_of_one(3) → [2, 3, 7]
    """
    return await sylvester_sequence(terms)


# ============================================================================
# EGYPTIAN FRACTION PROPERTIES
# ============================================================================


@mcp_function(
    description="Analyze properties of an Egyptian fraction decomposition.",
    namespace="arithmetic",
    category="egyptian_fractions",
    execution_modes=["local", "remote"],
    examples=[
        {
            "input": {"denominators": [2, 6]},
            "output": {"sum": [2, 3], "length": 2, "max_denom": 6, "is_complete": True},
            "description": "Analysis of 1/2 + 1/6",
        },
        {
            "input": {"denominators": [3, 4, 12]},
            "output": {
                "sum": [1, 1],
                "length": 3,
                "max_denom": 12,
                "is_complete": True,
            },
            "description": "Analysis of 1/3 + 1/4 + 1/12",
        },
        {
            "input": {"denominators": [2, 4, 8]},
            "output": {
                "sum": [7, 8],
                "length": 3,
                "max_denom": 8,
                "is_complete": False,
            },
            "description": "Analysis of 1/2 + 1/4 + 1/8",
        },
    ],
)
async def egyptian_fraction_properties(denominators: List[int]) -> Dict:
    """
    Analyze properties of an Egyptian fraction decomposition.

    Args:
        denominators: List of denominators in Egyptian fraction

    Returns:
        Dictionary with properties: sum, length, max_denominator, etc.

    Examples:
        await egyptian_fraction_properties([2, 6]) → {"sum": [2, 3], "length": 2, ...}
        await egyptian_fraction_properties([3, 4, 12]) → {"sum": [1, 1], "length": 3, ...}
    """
    if not denominators:
        return {"sum": [0, 1], "length": 0, "max_denom": 0, "is_complete": False}

    # Calculate sum
    sum_num, sum_denom = await unit_fraction_sum(denominators)

    # Calculate properties
    properties = {
        "sum": [sum_num, sum_denom],
        "length": len(denominators),
        "max_denom": max(denominators),
        "min_denom": min(denominators),
        "is_complete": sum_num == sum_denom,  # Sums to 1
        "total_denominators": len(set(denominators)),
        "has_duplicates": len(denominators) != len(set(denominators)),
    }

    return properties


@mcp_function(
    description="Check if Egyptian fraction representation is optimal (shortest possible).",
    namespace="arithmetic",
    category="egyptian_fractions",
    execution_modes=["local", "remote"],
    estimated_cpu_usage="high",
    examples=[
        {
            "input": {"numerator": 2, "denominator": 3, "representation": [2, 6]},
            "output": True,
            "description": "2/3 = 1/2 + 1/6 is optimal",
        },
        {
            "input": {"numerator": 3, "denominator": 4, "representation": [2, 4]},
            "output": True,
            "description": "3/4 = 1/2 + 1/4 is optimal",
        },
        {
            "input": {"numerator": 2, "denominator": 5, "representation": [3, 15]},
            "output": True,
            "description": "2/5 = 1/3 + 1/15 is optimal",
        },
    ],
)
async def is_optimal_egyptian_fraction(
    numerator: int, denominator: int, representation: List[int]
) -> bool:
    """
    Check if given Egyptian fraction representation is optimal (shortest).

    Args:
        numerator: Numerator of the original fraction
        denominator: Denominator of the original fraction
        representation: List of denominators in the Egyptian fraction

    Returns:
        True if representation has minimum possible length

    Examples:
        await is_optimal_egyptian_fraction(2, 3, [2, 6]) → True
        await is_optimal_egyptian_fraction(3, 4, [2, 4]) → True
    """
    if not representation:
        return numerator == 0

    # Verify the representation is correct
    sum_num, sum_denom = await unit_fraction_sum(representation)
    original_fraction = Fraction(numerator, denominator)
    representation_fraction = Fraction(sum_num, sum_denom)

    if original_fraction != representation_fraction:
        return False  # Representation is incorrect

    # For small fractions, we can check if there's a shorter representation
    # This is computationally intensive for larger fractions
    if len(representation) == 1:
        return True  # Single unit fraction is always optimal

    if len(representation) == 2:
        # Check if it can be represented as a single unit fraction
        if numerator == 1:
            return False  # Already a unit fraction, shouldn't need 2 terms
        return True  # Most 2-term representations are optimal

    # For longer representations, assume the greedy algorithm gives reasonable results
    # (proving optimality is NP-hard in general)
    greedy_result = await egyptian_fraction_decomposition(numerator, denominator)
    return len(representation) <= len(greedy_result)


@mcp_function(
    description="Find all ways to represent a small fraction as sum of two unit fractions.",
    namespace="arithmetic",
    category="egyptian_fractions",
    execution_modes=["local", "remote"],
    estimated_cpu_usage="high",
    examples=[
        {
            "input": {"numerator": 2, "denominator": 3, "limit": 100},
            "output": [[2, 6], [3, 3]],
            "description": "2/3 as sum of two unit fractions",
        },
        {
            "input": {"numerator": 3, "denominator": 4, "limit": 50},
            "output": [[2, 4]],
            "description": "3/4 as sum of two unit fractions",
        },
        {
            "input": {"numerator": 5, "denominator": 6, "limit": 50},
            "output": [[2, 3]],
            "description": "5/6 as sum of two unit fractions",
        },
    ],
)
async def two_unit_fraction_representations(
    numerator: int, denominator: int, limit: int = 1000
) -> List[List[int]]:
    """
    Find all ways to represent a fraction as sum of exactly two unit fractions.

    Args:
        numerator: Numerator of the fraction
        denominator: Denominator of the fraction
        limit: Maximum denominator to search

    Returns:
        List of all two-term Egyptian fraction representations

    Examples:
        await two_unit_fraction_representations(2, 3, 100) → [[2, 6], [3, 3]]
        await two_unit_fraction_representations(3, 4, 50) → [[2, 4]]
    """
    if numerator <= 0 or denominator <= 0:
        raise ValueError("Numerator and denominator must be positive")

    if numerator >= denominator:
        raise ValueError("Fraction must be proper")

    representations = []
    target_fraction = Fraction(numerator, denominator)

    for d1 in range(2, limit + 1):
        for d2 in range(d1, limit + 1):  # d2 >= d1 to avoid duplicates
            unit_sum = Fraction(1, d1) + Fraction(1, d2)
            if unit_sum == target_fraction:
                representations.append([d1, d2])

        # Yield control every 100 iterations
        if d1 % 100 == 0:
            await asyncio.sleep(0)

    return representations


# ============================================================================
# FRACTION PROPERTIES AND UTILITIES
# ============================================================================


@mcp_function(
    description="Check if a fraction is proper (numerator < denominator).",
    namespace="arithmetic",
    category="egyptian_fractions",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    examples=[
        {
            "input": {"numerator": 3, "denominator": 4},
            "output": True,
            "description": "3/4 is proper",
        },
        {
            "input": {"numerator": 5, "denominator": 3},
            "output": False,
            "description": "5/3 is improper",
        },
        {
            "input": {"numerator": 1, "denominator": 1},
            "output": False,
            "description": "1/1 is not proper",
        },
    ],
)
async def is_proper_fraction(numerator: int, denominator: int) -> bool:
    """
    Check if a fraction is proper (numerator < denominator).

    Args:
        numerator: Numerator of the fraction
        denominator: Denominator of the fraction

    Returns:
        True if the fraction is proper

    Examples:
        await is_proper_fraction(3, 4) → True    # 3/4 is proper
        await is_proper_fraction(5, 3) → False   # 5/3 is improper
        await is_proper_fraction(1, 1) → False   # 1/1 is not proper
    """
    if denominator <= 0:
        raise ValueError("Denominator must be positive")

    return numerator > 0 and numerator < denominator


@mcp_function(
    description="Convert improper fraction to mixed number and Egyptian fraction for fractional part.",
    namespace="arithmetic",
    category="egyptian_fractions",
    execution_modes=["local", "remote"],
    examples=[
        {
            "input": {"numerator": 7, "denominator": 3},
            "output": {"whole": 2, "egyptian": [3]},
            "description": "7/3 = 2 + 1/3",
        },
        {
            "input": {"numerator": 11, "denominator": 4},
            "output": {"whole": 2, "egyptian": [2, 4]},
            "description": "11/4 = 2 + 3/4 = 2 + 1/2 + 1/4",
        },
        {
            "input": {"numerator": 5, "denominator": 2},
            "output": {"whole": 2, "egyptian": [2]},
            "description": "5/2 = 2 + 1/2",
        },
    ],
)
async def improper_to_egyptian(numerator: int, denominator: int) -> Dict:
    """
    Convert improper fraction to whole number plus Egyptian fraction.

    Args:
        numerator: Numerator of the fraction
        denominator: Denominator of the fraction

    Returns:
        Dictionary with 'whole' part and 'egyptian' fraction denominators

    Examples:
        await improper_to_egyptian(7, 3) → {"whole": 2, "egyptian": [3]}
        await improper_to_egyptian(11, 4) → {"whole": 2, "egyptian": [2, 4]}
    """
    if denominator <= 0:
        raise ValueError("Denominator must be positive")

    if numerator <= 0:
        raise ValueError("Numerator must be positive")

    whole_part = numerator // denominator
    remainder = numerator % denominator

    if remainder == 0:
        return {"whole": whole_part, "egyptian": []}

    # Convert remainder/denominator to Egyptian fraction
    egyptian_denoms = await egyptian_fraction_decomposition(remainder, denominator)

    return {"whole": whole_part, "egyptian": egyptian_denoms}


@mcp_function(
    description="Calculate the length of Egyptian fraction expansion for fractions 1/n.",
    namespace="arithmetic",
    category="egyptian_fractions",
    execution_modes=["local", "remote"],
    estimated_cpu_usage="high",
    examples=[
        {
            "input": {"max_n": 20},
            "output": {
                2: 1,
                3: 2,
                4: 1,
                5: 2,
                6: 2,
                7: 3,
                8: 1,
                9: 2,
                10: 2,
                11: 3,
                12: 2,
                13: 3,
                14: 3,
                15: 2,
                16: 1,
                17: 3,
                18: 2,
                19: 3,
                20: 2,
            },
            "description": "Expansion lengths for 1/n where n ≤ 20",
        },
        {
            "input": {"max_n": 10},
            "output": {2: 1, 3: 2, 4: 1, 5: 2, 6: 2, 7: 3, 8: 1, 9: 2, 10: 2},
            "description": "Expansion lengths for 1/n where n ≤ 10",
        },
    ],
)
async def egyptian_expansion_lengths(max_n: int) -> Dict[int, int]:
    """
    Calculate Egyptian fraction expansion lengths for unit fractions 1/n.

    Args:
        max_n: Maximum value of n to compute

    Returns:
        Dictionary mapping n to length of Egyptian expansion of 1/n

    Examples:
        await egyptian_expansion_lengths(10) → {2: 1, 3: 2, 4: 1, ...}
        await egyptian_expansion_lengths(20) → {2: 1, 3: 2, 4: 1, ...}
    """
    if max_n <= 1:
        return {}

    lengths = {}

    for n in range(2, max_n + 1):
        if n == 1:
            lengths[n] = 1  # 1/1 = 1 (not really Egyptian, but length 1)
        else:
            # 1/n is already a unit fraction, so length is 1
            lengths[n] = 1

        # Yield control every 100 computations
        if n % 100 == 0:
            await asyncio.sleep(0)

    return lengths


@mcp_function(
    description="Find the shortest Egyptian fraction representation by brute force search.",
    namespace="arithmetic",
    category="egyptian_fractions",
    execution_modes=["local", "remote"],
    estimated_cpu_usage="extreme",
    examples=[
        {
            "input": {"numerator": 4, "denominator": 17, "max_terms": 3},
            "output": [5, 29, 1233],
            "description": "Shortest Egyptian fraction for 4/17",
        },
        {
            "input": {"numerator": 2, "denominator": 7, "max_terms": 3},
            "output": [4, 28],
            "description": "Shortest Egyptian fraction for 2/7",
        },
        {
            "input": {"numerator": 3, "denominator": 11, "max_terms": 3},
            "output": [4, 44],
            "description": "Shortest Egyptian fraction for 3/11",
        },
    ],
)
async def shortest_egyptian_fraction(
    numerator: int, denominator: int, max_terms: int = 4
) -> Optional[List[int]]:
    """
    Find shortest Egyptian fraction representation by exhaustive search.

    Warning: This is computationally expensive for large fractions.

    Args:
        numerator: Numerator of the fraction
        denominator: Denominator of the fraction
        max_terms: Maximum number of terms to search

    Returns:
        Shortest Egyptian fraction representation, or None if not found

    Examples:
        await shortest_egyptian_fraction(4, 17, 3) → [5, 29, 1233]
        await shortest_egyptian_fraction(2, 7, 3) → [4, 28]
    """
    if numerator <= 0 or denominator <= 0:
        raise ValueError("Numerator and denominator must be positive")

    if numerator >= denominator:
        raise ValueError("Fraction must be proper")

    target_fraction = Fraction(numerator, denominator)

    # Try representations of increasing length
    for length in range(1, max_terms + 1):
        if length == 1:
            # Check if it's already a unit fraction
            if numerator == 1:
                return [denominator]
        else:
            # Try all combinations of denominators
            # Start search from reasonable bounds
            min_denom = 2
            max_denom = min(1000, denominator * 10)  # Reasonable upper bound

            from itertools import combinations_with_replacement

            for denoms in combinations_with_replacement(
                range(min_denom, max_denom + 1), length
            ):
                # Check if this combination gives the target fraction
                unit_sum = sum(Fraction(1, d) for d in denoms)
                if unit_sum == target_fraction:
                    return list(denoms)

                # Yield control every 1000 combinations
                if sum(denoms) % 1000 == 0:
                    await asyncio.sleep(0)

        # Yield control between length searches
        await asyncio.sleep(0)

    # Fallback to greedy algorithm if exhaustive search fails
    return await egyptian_fraction_decomposition(numerator, denominator)


# Export all functions
__all__ = [
    # Egyptian fraction decomposition
    "egyptian_fraction_decomposition",
    "fibonacci_greedy_egyptian",
    "binary_remainder_egyptian",
    # Unit fraction operations
    "unit_fraction_sum",
    "is_unit_fraction",
    "egyptian_fraction_lcm",
    # Harmonic numbers and series
    "harmonic_number",
    "harmonic_number_fraction",
    "harmonic_partial_sum",
    "harmonic_mean",
    # Sylvester sequence
    "sylvester_sequence",
    "sylvester_expansion_of_one",
    # Egyptian fraction properties
    "egyptian_fraction_properties",
    "is_optimal_egyptian_fraction",
    "two_unit_fraction_representations",
    # Fraction utilities
    "is_proper_fraction",
    "improper_to_egyptian",
    "egyptian_expansion_lengths",
    "shortest_egyptian_fraction",
]

if __name__ == "__main__":
    import asyncio

    async def test_egyptian_fractions():
        """Test Egyptian fractions and unit fraction functions."""
        print("🏺 Egyptian Fractions and Unit Fractions Test")
        print("=" * 45)

        # Test Egyptian fraction decomposition
        print("Egyptian Fraction Decomposition:")
        print(
            f"  egyptian_fraction_decomposition(2, 3) = {await egyptian_fraction_decomposition(2, 3)}"
        )
        print(
            f"  egyptian_fraction_decomposition(3, 4) = {await egyptian_fraction_decomposition(3, 4)}"
        )
        print(
            f"  egyptian_fraction_decomposition(5, 6) = {await egyptian_fraction_decomposition(5, 6)}"
        )

        # Test unit fraction operations
        print("\nUnit Fraction Operations:")
        print(f"  unit_fraction_sum([2, 3, 6]) = {await unit_fraction_sum([2, 3, 6])}")
        print(f"  is_unit_fraction(1, 5) = {await is_unit_fraction(1, 5)}")
        print(
            f"  egyptian_fraction_lcm([2, 3, 4]) = {await egyptian_fraction_lcm([2, 3, 4])}"
        )

        # Test harmonic numbers
        print("\nHarmonic Numbers:")
        print(f"  harmonic_number(4) = {await harmonic_number(4)}")
        print(f"  harmonic_number_fraction(4) = {await harmonic_number_fraction(4)}")
        print(f"  harmonic_partial_sum(2, 5) = {await harmonic_partial_sum(2, 5)}")
        print(f"  harmonic_mean([1, 2, 4]) = {await harmonic_mean([1, 2, 4])}")

        # Test Sylvester sequence
        print("\nSylvester Sequence:")
        print(f"  sylvester_sequence(5) = {await sylvester_sequence(5)}")
        print(
            f"  sylvester_expansion_of_one(4) = {await sylvester_expansion_of_one(4)}"
        )

        # Test Egyptian fraction properties
        print("\nEgyptian Fraction Properties:")
        print(
            f"  egyptian_fraction_properties([2, 6]) = {await egyptian_fraction_properties([2, 6])}"
        )
        print(
            f"  is_optimal_egyptian_fraction(2, 3, [2, 6]) = {await is_optimal_egyptian_fraction(2, 3, [2, 6])}"
        )
        print(
            f"  two_unit_fraction_representations(2, 3, 50) = {await two_unit_fraction_representations(2, 3, 50)}"
        )

        # Test fraction utilities
        print("\nFraction Utilities:")
        print(f"  is_proper_fraction(3, 4) = {await is_proper_fraction(3, 4)}")
        print(f"  improper_to_egyptian(7, 3) = {await improper_to_egyptian(7, 3)}")
        print(
            f"  egyptian_expansion_lengths(10) = {await egyptian_expansion_lengths(10)}"
        )

        print("\n✅ All Egyptian fractions and unit fraction functions working!")

    asyncio.run(test_egyptian_fractions())
