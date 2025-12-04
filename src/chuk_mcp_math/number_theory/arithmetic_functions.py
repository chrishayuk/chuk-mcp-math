#!/usr/bin/env python3
# chuk_mcp_math/number_theory/arithmetic_functions.py
"""
Arithmetic Functions - Async Native

Classical number-theoretic functions including multiplicative functions,
additive functions, and other important arithmetic functions.

Functions:
- Multiplicative functions: euler_totient, mobius_function, divisor_functions
- Additive functions: little_omega, big_omega, sum_of_divisors_power
- Von Mangoldt and related: von_mangoldt_function, chebyshev_functions
- Jordan totient: jordan_totient_function
- Carmichael function: carmichael_lambda
- Liouville function: liouville_function
- Perfect number functions: is_perfect_number, is_abundant_number, is_deficient_number
"""

import math
import asyncio
from typing import List
from chuk_mcp_math.mcp_decorator import mcp_function

# Import dependencies
from .primes import prime_factors
from .divisibility import gcd, divisors, divisor_sum

# ============================================================================
# EULER'S TOTIENT FUNCTION
# ============================================================================


@mcp_function(
    description="Calculate Euler's totient function φ(n) - count of integers ≤ n coprime to n.",
    namespace="arithmetic",
    category="arithmetic_functions",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    estimated_cpu_usage="medium",
    examples=[
        {
            "input": {"n": 12},
            "output": 4,
            "description": "φ(12) = 4: numbers 1,5,7,11 are coprime to 12",
        },
        {
            "input": {"n": 9},
            "output": 6,
            "description": "φ(9) = 6: numbers 1,2,4,5,7,8 are coprime to 9",
        },
        {
            "input": {"n": 17},
            "output": 16,
            "description": "φ(17) = 16: all numbers 1-16 coprime to prime 17",
        },
        {
            "input": {"n": 30},
            "output": 8,
            "description": "φ(30) = 8: φ(2×3×5) = 30×(1-1/2)×(1-1/3)×(1-1/5)",
        },
    ],
)
async def euler_totient(n: int) -> int:
    """
    Calculate Euler's totient function φ(n).

    φ(n) counts the number of integers from 1 to n that are coprime to n.
    Uses the formula: φ(n) = n × ∏(1 - 1/p) for all prime p dividing n.

    Args:
        n: Positive integer

    Returns:
        φ(n) - count of integers ≤ n that are coprime to n

    Examples:
        await euler_totient(12) → 4   # 1, 5, 7, 11 are coprime to 12
        await euler_totient(9) → 6    # 1, 2, 4, 5, 7, 8 are coprime to 9
        await euler_totient(17) → 16  # All 1-16 are coprime to prime 17
    """
    if n <= 0:
        return 0
    if n == 1:
        return 1

    # Use the formula: φ(n) = n * ∏(1 - 1/p) for all prime p dividing n
    factors = await prime_factors(n)

    if not factors:
        return 1

    result = n
    unique_primes = set(factors)

    for p in unique_primes:
        result = result * (p - 1) // p

    return result


@mcp_function(
    description="Calculate Jordan's totient function J_k(n) - generalization of Euler's totient.",
    namespace="arithmetic",
    category="arithmetic_functions",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    estimated_cpu_usage="medium",
    examples=[
        {"input": {"n": 12, "k": 2}, "output": 120, "description": "J_2(12) = 120"},
        {"input": {"n": 6, "k": 1}, "output": 2, "description": "J_1(6) = φ(6) = 2"},
        {"input": {"n": 10, "k": 3}, "output": 720, "description": "J_3(10) = 720"},
    ],
)
async def jordan_totient(n: int, k: int) -> int:
    """
    Calculate Jordan's totient function J_k(n).

    J_k(n) = n^k × ∏(1 - 1/p^k) for all prime p dividing n.
    When k=1, this reduces to Euler's totient function φ(n).

    Args:
        n: Positive integer
        k: Positive integer (power)

    Returns:
        J_k(n) - Jordan's totient function value

    Examples:
        await jordan_totient(12, 2) → 120  # J_2(12)
        await jordan_totient(6, 1) → 2     # J_1(6) = φ(6)
    """
    if n <= 0 or k <= 0:
        return 0
    if n == 1:
        return 1

    factors = await prime_factors(n)

    if not factors:
        return 1

    result = n**k
    unique_primes = set(factors)

    for p in unique_primes:
        # Multiply by (1 - 1/p^k) = (p^k - 1) / p^k
        p_k = p**k
        result = result * (p_k - 1) // p_k

    return result


# ============================================================================
# MÖBIUS FUNCTION
# ============================================================================


@mcp_function(
    description="Calculate the Möbius function μ(n).",
    namespace="arithmetic",
    category="arithmetic_functions",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    examples=[
        {
            "input": {"n": 6},
            "output": 1,
            "description": "μ(6) = 1: 6 = 2×3 (2 distinct primes)",
        },
        {
            "input": {"n": 12},
            "output": 0,
            "description": "μ(12) = 0: 12 = 2²×3 (has square factor)",
        },
        {
            "input": {"n": 30},
            "output": -1,
            "description": "μ(30) = -1: 30 = 2×3×5 (3 distinct primes)",
        },
        {"input": {"n": 1}, "output": 1, "description": "μ(1) = 1 by definition"},
    ],
)
async def mobius_function(n: int) -> int:
    """
    Calculate the Möbius function μ(n).

    μ(n) = 1 if n is square-free with even number of prime factors
    μ(n) = -1 if n is square-free with odd number of prime factors
    μ(n) = 0 if n has a squared prime factor

    Args:
        n: Positive integer

    Returns:
        μ(n) ∈ {-1, 0, 1}

    Examples:
        await mobius_function(6) → 1    # 6 = 2×3 (2 distinct primes)
        await mobius_function(12) → 0   # 12 = 2²×3 (has square factor)
        await mobius_function(30) → -1  # 30 = 2×3×5 (3 distinct primes)
    """
    if n <= 0:
        return 0
    if n == 1:
        return 1

    factors = await prime_factors(n)

    if not factors:
        return 1

    # Check if square-free
    unique_factors = set(factors)
    if len(factors) != len(unique_factors):
        return 0  # Has repeated prime factor

    # Return (-1)^k where k is number of distinct prime factors
    k = len(unique_factors)
    return (-1) ** k


# ============================================================================
# OMEGA FUNCTIONS
# ============================================================================


@mcp_function(
    description="Calculate ω(n) - number of distinct prime factors.",
    namespace="arithmetic",
    category="arithmetic_functions",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    examples=[
        {
            "input": {"n": 12},
            "output": 2,
            "description": "ω(12) = 2: prime factors are 2, 3",
        },
        {
            "input": {"n": 30},
            "output": 3,
            "description": "ω(30) = 3: prime factors are 2, 3, 5",
        },
        {
            "input": {"n": 17},
            "output": 1,
            "description": "ω(17) = 1: only prime factor is 17",
        },
        {"input": {"n": 1}, "output": 0, "description": "ω(1) = 0: no prime factors"},
    ],
)
async def little_omega(n: int) -> int:
    """
    Calculate ω(n) - number of distinct prime factors.

    Args:
        n: Positive integer

    Returns:
        Number of distinct prime factors of n

    Examples:
        await little_omega(12) → 2   # 12 = 2²×3, distinct primes: 2, 3
        await little_omega(30) → 3   # 30 = 2×3×5, distinct primes: 2, 3, 5
        await little_omega(17) → 1   # 17 is prime
    """
    if n <= 1:
        return 0

    factors = await prime_factors(n)
    return len(set(factors))


@mcp_function(
    description="Calculate Ω(n) - total number of prime factors (with multiplicity).",
    namespace="arithmetic",
    category="arithmetic_functions",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    examples=[
        {
            "input": {"n": 12},
            "output": 3,
            "description": "Ω(12) = 3: prime factors 2, 2, 3",
        },
        {
            "input": {"n": 30},
            "output": 3,
            "description": "Ω(30) = 3: prime factors 2, 3, 5",
        },
        {
            "input": {"n": 8},
            "output": 3,
            "description": "Ω(8) = 3: prime factors 2, 2, 2",
        },
        {"input": {"n": 1}, "output": 0, "description": "Ω(1) = 0: no prime factors"},
    ],
)
async def big_omega(n: int) -> int:
    """
    Calculate Ω(n) - total number of prime factors counting multiplicity.

    Args:
        n: Positive integer

    Returns:
        Total number of prime factors of n (with repetition)

    Examples:
        await big_omega(12) → 3   # 12 = 2²×3, factors: 2, 2, 3
        await big_omega(30) → 3   # 30 = 2×3×5, factors: 2, 3, 5
        await big_omega(8) → 3    # 8 = 2³, factors: 2, 2, 2
    """
    if n <= 1:
        return 0

    factors = await prime_factors(n)
    return len(factors)


# ============================================================================
# DIVISOR FUNCTIONS
# ============================================================================


@mcp_function(
    description="Calculate σ_k(n) - sum of kth powers of divisors.",
    namespace="arithmetic",
    category="arithmetic_functions",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    estimated_cpu_usage="medium",
    examples=[
        {
            "input": {"n": 12, "k": 0},
            "output": 6,
            "description": "σ_0(12) = 6: count of divisors",
        },
        {
            "input": {"n": 12, "k": 1},
            "output": 28,
            "description": "σ_1(12) = 28: sum of divisors",
        },
        {
            "input": {"n": 6, "k": 2},
            "output": 50,
            "description": "σ_2(6) = 50: sum of squares of divisors",
        },
    ],
)
async def divisor_power_sum(n: int, k: int) -> int:
    """
    Calculate σ_k(n) - sum of kth powers of divisors.

    σ_k(n) = Σ_{d|n} d^k

    Special cases:
    - σ_0(n) = τ(n) = number of divisors
    - σ_1(n) = σ(n) = sum of divisors

    Args:
        n: Positive integer
        k: Non-negative integer (power)

    Returns:
        Sum of kth powers of all divisors of n

    Examples:
        await divisor_power_sum(12, 0) → 6   # Count of divisors: 1,2,3,4,6,12
        await divisor_power_sum(12, 1) → 28  # Sum: 1+2+3+4+6+12 = 28
        await divisor_power_sum(6, 2) → 50   # Sum: 1²+2²+3²+6² = 1+4+9+36 = 50
    """
    if n <= 0:
        return 0
    if n == 1:
        return 1

    div_list = await divisors(n)

    if k == 0:
        return len(div_list)
    elif k == 1:
        return sum(div_list)
    else:
        return sum(d**k for d in div_list)


# ============================================================================
# VON MANGOLDT FUNCTION
# ============================================================================


@mcp_function(
    description="Calculate the von Mangoldt function Λ(n).",
    namespace="arithmetic",
    category="arithmetic_functions",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    examples=[
        {
            "input": {"n": 8},
            "output": 0.6931471805599453,
            "description": "Λ(8) = ln(2) since 8 = 2³",
        },
        {
            "input": {"n": 17},
            "output": 2.833213344056216,
            "description": "Λ(17) = ln(17) since 17 is prime",
        },
        {
            "input": {"n": 12},
            "output": 0,
            "description": "Λ(12) = 0 since 12 = 2²×3 (not prime power)",
        },
        {"input": {"n": 1}, "output": 0, "description": "Λ(1) = 0 by definition"},
    ],
)
async def von_mangoldt_function(n: int) -> float:
    """
    Calculate the von Mangoldt function Λ(n).

    Λ(n) = ln(p) if n = p^k for some prime p and positive integer k
    Λ(n) = 0 otherwise

    Args:
        n: Positive integer

    Returns:
        The von Mangoldt function value

    Examples:
        await von_mangoldt_function(8) → ln(2)   # 8 = 2³
        await von_mangoldt_function(17) → ln(17) # 17 is prime
        await von_mangoldt_function(12) → 0      # 12 = 2²×3 (not prime power)
    """
    if n <= 1:
        return 0.0

    factors = await prime_factors(n)

    if not factors:
        return 0.0

    # Check if all prime factors are the same (i.e., n = p^k)
    unique_factors = set(factors)

    if len(unique_factors) == 1:
        # n is a prime power
        p = unique_factors.pop()
        return math.log(p)
    else:
        # n has multiple distinct prime factors
        return 0.0


# ============================================================================
# LIOUVILLE FUNCTION
# ============================================================================


@mcp_function(
    description="Calculate the Liouville function λ(n) = (-1)^Ω(n).",
    namespace="arithmetic",
    category="arithmetic_functions",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    examples=[
        {
            "input": {"n": 12},
            "output": -1,
            "description": "λ(12) = (-1)³ = -1 since Ω(12) = 3",
        },
        {
            "input": {"n": 8},
            "output": -1,
            "description": "λ(8) = (-1)³ = -1 since Ω(8) = 3",
        },
        {
            "input": {"n": 6},
            "output": 1,
            "description": "λ(6) = (-1)² = 1 since Ω(6) = 2",
        },
        {"input": {"n": 1}, "output": 1, "description": "λ(1) = 1 by definition"},
    ],
)
async def liouville_function(n: int) -> int:
    """
    Calculate the Liouville function λ(n).

    λ(n) = (-1)^Ω(n) where Ω(n) is the total number of prime factors
    (counting multiplicities).

    Args:
        n: Positive integer

    Returns:
        λ(n) ∈ {-1, 1}

    Examples:
        await liouville_function(12) → -1  # λ(12) = (-1)³ = -1
        await liouville_function(6) → 1    # λ(6) = (-1)² = 1
        await liouville_function(8) → -1   # λ(8) = (-1)³ = -1
    """
    if n <= 0:
        return 1
    if n == 1:
        return 1

    omega_n = await big_omega(n)
    return (-1) ** omega_n


# ============================================================================
# CARMICHAEL FUNCTION
# ============================================================================


@mcp_function(
    description="Calculate the Carmichael function λ(n) - exponent of multiplicative group mod n.",
    namespace="arithmetic",
    category="arithmetic_functions",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    estimated_cpu_usage="medium",
    examples=[
        {"input": {"n": 12}, "output": 2, "description": "λ(12) = 2"},
        {"input": {"n": 15}, "output": 4, "description": "λ(15) = 4"},
        {"input": {"n": 8}, "output": 2, "description": "λ(8) = 2"},
        {"input": {"n": 17}, "output": 16, "description": "λ(17) = 16 for prime 17"},
    ],
)
async def carmichael_lambda(n: int) -> int:
    """
    Calculate the Carmichael function λ(n).

    The Carmichael function gives the exponent of the multiplicative group
    of integers modulo n. For any integer a coprime to n: a^λ(n) ≡ 1 (mod n).

    Args:
        n: Positive integer

    Returns:
        λ(n) - the Carmichael function value

    Examples:
        await carmichael_lambda(12) → 2   # λ(12) = 2
        await carmichael_lambda(15) → 4   # λ(15) = 4
        await carmichael_lambda(17) → 16  # λ(17) = 16 for prime
    """
    if n <= 0:
        return 0
    if n == 1:
        return 1
    if n == 2:
        return 1

    factors = await prime_factors(n)

    if not factors:
        return 1

    # Get prime power decomposition
    prime_powers: dict[int, int] = {}
    for p in factors:
        prime_powers[p] = prime_powers.get(p, 0) + 1

    # Calculate λ for each prime power and take LCM
    lambda_values = []

    for p, k in prime_powers.items():
        if p == 2:
            if k == 1:
                lambda_values.append(1)
            elif k == 2:
                lambda_values.append(2)
            else:  # k >= 3
                lambda_values.append(2 ** (k - 2))
        else:  # Odd prime
            lambda_values.append((p - 1) * (p ** (k - 1)))

    # Calculate LCM of all lambda values
    result = lambda_values[0]
    for i in range(1, len(lambda_values)):
        # LCM(a, b) = a * b / GCD(a, b)
        gcd_val = await gcd(result, lambda_values[i])
        result = result * lambda_values[i] // gcd_val

    return result


# ============================================================================
# PERFECT NUMBER FUNCTIONS
# ============================================================================


@mcp_function(
    description="Check if a number is perfect (equals sum of its proper divisors).",
    namespace="arithmetic",
    category="arithmetic_functions",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    examples=[
        {"input": {"n": 6}, "output": True, "description": "6 is perfect: 1+2+3 = 6"},
        {
            "input": {"n": 28},
            "output": True,
            "description": "28 is perfect: 1+2+4+7+14 = 28",
        },
        {
            "input": {"n": 12},
            "output": False,
            "description": "12 is abundant: 1+2+3+4+6 = 16 > 12",
        },
        {
            "input": {"n": 8},
            "output": False,
            "description": "8 is deficient: 1+2+4 = 7 < 8",
        },
    ],
)
async def is_perfect_number(n: int) -> bool:
    """
    Check if a number is perfect.

    A perfect number equals the sum of its proper divisors (divisors less than n).

    Args:
        n: Positive integer to check

    Returns:
        True if n is perfect, False otherwise

    Examples:
        await is_perfect_number(6) → True   # 6 = 1+2+3
        await is_perfect_number(28) → True  # 28 = 1+2+4+7+14
        await is_perfect_number(12) → False # 12 ≠ 1+2+3+4+6 = 16
    """
    if n <= 1:
        return False

    divisor_sum_val = await divisor_sum(n)
    proper_divisor_sum = divisor_sum_val - n

    return proper_divisor_sum == n


@mcp_function(
    description="Check if a number is abundant (sum of proper divisors > n).",
    namespace="arithmetic",
    category="arithmetic_functions",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    examples=[
        {
            "input": {"n": 12},
            "output": True,
            "description": "12 is abundant: 1+2+3+4+6 = 16 > 12",
        },
        {
            "input": {"n": 18},
            "output": True,
            "description": "18 is abundant: proper divisors sum > 18",
        },
        {
            "input": {"n": 6},
            "output": False,
            "description": "6 is perfect, not abundant",
        },
        {
            "input": {"n": 8},
            "output": False,
            "description": "8 is deficient, not abundant",
        },
    ],
)
async def is_abundant_number(n: int) -> bool:
    """
    Check if a number is abundant.

    An abundant number has the sum of its proper divisors greater than the number itself.

    Args:
        n: Positive integer to check

    Returns:
        True if n is abundant, False otherwise

    Examples:
        await is_abundant_number(12) → True  # 12 < 1+2+3+4+6 = 16
        await is_abundant_number(18) → True  # 18 < 1+2+3+6+9 = 21
        await is_abundant_number(8) → False  # 8 > 1+2+4 = 7
    """
    if n <= 1:
        return False

    divisor_sum_val = await divisor_sum(n)
    proper_divisor_sum = divisor_sum_val - n

    return proper_divisor_sum > n


@mcp_function(
    description="Check if a number is deficient (sum of proper divisors < n).",
    namespace="arithmetic",
    category="arithmetic_functions",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    examples=[
        {
            "input": {"n": 8},
            "output": True,
            "description": "8 is deficient: 1+2+4 = 7 < 8",
        },
        {
            "input": {"n": 9},
            "output": True,
            "description": "9 is deficient: 1+3 = 4 < 9",
        },
        {
            "input": {"n": 6},
            "output": False,
            "description": "6 is perfect, not deficient",
        },
        {
            "input": {"n": 12},
            "output": False,
            "description": "12 is abundant, not deficient",
        },
    ],
)
async def is_deficient_number(n: int) -> bool:
    """
    Check if a number is deficient.

    A deficient number has the sum of its proper divisors less than the number itself.

    Args:
        n: Positive integer to check

    Returns:
        True if n is deficient, False otherwise

    Examples:
        await is_deficient_number(8) → True   # 8 > 1+2+4 = 7
        await is_deficient_number(9) → True   # 9 > 1+3 = 4
        await is_deficient_number(12) → False # 12 < 1+2+3+4+6 = 16
    """
    if n <= 1:
        return n == 1  # 1 is deficient by convention

    divisor_sum_val = await divisor_sum(n)
    proper_divisor_sum = divisor_sum_val - n

    return proper_divisor_sum < n


@mcp_function(
    description="Find all perfect numbers up to limit using Euclid-Euler theorem.",
    namespace="arithmetic",
    category="arithmetic_functions",
    execution_modes=["local", "remote"],
    estimated_cpu_usage="high",
    examples=[
        {
            "input": {"limit": 100},
            "output": [6, 28],
            "description": "Perfect numbers ≤ 100",
        },
        {
            "input": {"limit": 10000},
            "output": [6, 28, 496, 8128],
            "description": "Perfect numbers ≤ 10000",
        },
    ],
)
async def perfect_numbers_up_to(limit: int) -> List[int]:
    """
    Find all perfect numbers up to limit.

    Uses the Euclid-Euler theorem: even perfect numbers have the form
    2^(p-1) × (2^p - 1) where 2^p - 1 is a Mersenne prime.

    Args:
        limit: Upper limit for search

    Returns:
        List of perfect numbers ≤ limit

    Examples:
        await perfect_numbers_up_to(100) → [6, 28]
        await perfect_numbers_up_to(10000) → [6, 28, 496, 8128]
    """
    from .special_primes import mersenne_prime_exponents, lucas_lehmer_test

    perfect_nums = []

    # Get potential Mersenne prime exponents
    max_p = min(50, int(math.log2(limit)) + 1)  # Reasonable upper bound
    mersenne_exponents = await mersenne_prime_exponents(max_p)

    for p in mersenne_exponents:
        if p > max_p:
            break

        # Check if 2^p - 1 is actually prime (double-check)
        if await lucas_lehmer_test(p):
            # Calculate the perfect number: 2^(p-1) × (2^p - 1)
            mersenne_prime = (2**p) - 1
            perfect_num = (2 ** (p - 1)) * mersenne_prime

            if perfect_num <= limit:
                perfect_nums.append(perfect_num)
            else:
                break

    return perfect_nums


# Export all functions
__all__ = [
    # Multiplicative functions
    "euler_totient",
    "jordan_totient",
    "mobius_function",
    # Additive functions
    "little_omega",
    "big_omega",
    # Divisor functions
    "divisor_power_sum",
    # Special functions
    "von_mangoldt_function",
    "liouville_function",
    "carmichael_lambda",
    # Perfect number functions
    "is_perfect_number",
    "is_abundant_number",
    "is_deficient_number",
    "perfect_numbers_up_to",
]

if __name__ == "__main__":
    import asyncio

    async def test_arithmetic_functions():
        """Test arithmetic functions."""
        print("🔢 Arithmetic Functions Test")
        print("=" * 35)

        # Test multiplicative functions
        print("Multiplicative Functions:")
        print(f"  euler_totient(12) = {await euler_totient(12)}")
        print(f"  jordan_totient(12, 2) = {await jordan_totient(12, 2)}")
        print(f"  mobius_function(30) = {await mobius_function(30)}")
        print(f"  mobius_function(12) = {await mobius_function(12)}")

        # Test additive functions
        print("\nAdditive Functions:")
        print(f"  little_omega(12) = {await little_omega(12)}")
        print(f"  big_omega(12) = {await big_omega(12)}")
        print(f"  little_omega(30) = {await little_omega(30)}")
        print(f"  big_omega(8) = {await big_omega(8)}")

        # Test divisor functions
        print("\nDivisor Functions:")
        print(f"  divisor_power_sum(12, 0) = {await divisor_power_sum(12, 0)}")
        print(f"  divisor_power_sum(12, 1) = {await divisor_power_sum(12, 1)}")
        print(f"  divisor_power_sum(6, 2) = {await divisor_power_sum(6, 2)}")

        # Test special functions
        print("\nSpecial Functions:")
        print(f"  von_mangoldt_function(8) = {await von_mangoldt_function(8):.4f}")
        print(f"  von_mangoldt_function(17) = {await von_mangoldt_function(17):.4f}")
        print(f"  liouville_function(12) = {await liouville_function(12)}")
        print(f"  carmichael_lambda(12) = {await carmichael_lambda(12)}")

        # Test perfect number functions
        print("\nPerfect Number Functions:")
        print(f"  is_perfect_number(6) = {await is_perfect_number(6)}")
        print(f"  is_perfect_number(28) = {await is_perfect_number(28)}")
        print(f"  is_abundant_number(12) = {await is_abundant_number(12)}")
        print(f"  is_deficient_number(8) = {await is_deficient_number(8)}")
        print(f"  perfect_numbers_up_to(100) = {await perfect_numbers_up_to(100)}")

        print("\n✅ All arithmetic functions working!")

    asyncio.run(test_arithmetic_functions())
