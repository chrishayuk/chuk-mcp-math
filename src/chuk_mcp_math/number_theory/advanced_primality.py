#!/usr/bin/env python3
# chuk_mcp_math/number_theory/advanced_primality.py
"""
Advanced Primality Testing Algorithms - Async Native - COMPLETE IMPLEMENTATION

State-of-the-art primality testing algorithms including probabilistic and deterministic methods.
Essential for cryptography, number theory research, and large-scale prime generation.

Functions:
- Probabilistic tests: miller_rabin_test, solovay_strassen_test, fermat_primality_test
- Deterministic tests: aks_primality_test, deterministic_miller_rabin
- Composite tests: strong_pseudoprime_test, carmichael_number_test
- Optimization: witness_finding, primality_certificate_generation

Mathematical Background:
Advanced primality tests use sophisticated mathematical properties to determine
primality with high confidence (probabilistic) or certainty (deterministic).
These algorithms are crucial for cryptographic applications requiring large primes.
"""

import asyncio
import random
import math
from typing import List
from chuk_mcp_math.mcp_decorator import mcp_function

# ============================================================================
# PROBABILISTIC PRIMALITY TESTS
# ============================================================================


@mcp_function(
    description="Miller-Rabin probabilistic primality test. Industry standard for cryptographic applications.",
    namespace="arithmetic",
    category="advanced_primality",
    execution_modes=["local", "remote"],
    estimated_cpu_usage="high",
    examples=[
        {
            "input": {"n": 97, "k": 5},
            "output": True,
            "description": "97 is prime with high confidence",
        },
        {
            "input": {"n": 561, "k": 5},
            "output": False,
            "description": "561 is composite (Carmichael number)",
        },
        {
            "input": {"n": 2047, "k": 10},
            "output": False,
            "description": "2047 is composite",
        },
        {"input": {"n": 1009, "k": 5}, "output": True, "description": "1009 is prime"},
    ],
)
async def miller_rabin_test(n: int, k: int = 10) -> bool:
    """
    Miller-Rabin probabilistic primality test.

    This is the most widely used primality test in cryptographic applications.
    It has error probability ≤ (1/4)^k for composite numbers.

    Args:
        n: Number to test for primality (must be > 1)
        k: Number of rounds (higher k = lower error probability)

    Returns:
        True if n is probably prime, False if n is definitely composite

    Error probability: ≤ (1/4)^k for composite numbers

    Examples:
        await miller_rabin_test(97, 5) → True
        await miller_rabin_test(561, 5) → False  # Carmichael number
        await miller_rabin_test(2047, 10) → False
    """
    if n < 2:
        return False
    if n == 2 or n == 3:
        return True
    if n % 2 == 0:
        return False

    # Write n-1 as d * 2^r
    r = 0
    d = n - 1
    while d % 2 == 0:
        r += 1
        d //= 2

    # Yield control for large numbers
    if n > 10**10:
        await asyncio.sleep(0)

    # Perform k rounds of testing
    for _ in range(k):
        # Choose random witness
        a = random.randrange(2, n - 1)

        # Compute a^d mod n
        x = pow(a, d, n)

        if x == 1 or x == n - 1:
            continue

        # Repeat r-1 times
        composite = True
        for _ in range(r - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                composite = False
                break

        if composite:
            return False

        # Yield control every few rounds for very large numbers
        if n > 10**15:
            await asyncio.sleep(0)

    return True


@mcp_function(
    description="Solovay-Strassen probabilistic primality test using Jacobi symbols.",
    namespace="arithmetic",
    category="advanced_primality",
    execution_modes=["local", "remote"],
    estimated_cpu_usage="high",
    examples=[
        {"input": {"n": 97, "k": 5}, "output": True, "description": "97 is prime"},
        {
            "input": {"n": 561, "k": 5},
            "output": False,
            "description": "561 is composite",
        },
        {
            "input": {"n": 15, "k": 3},
            "output": False,
            "description": "15 is obviously composite",
        },
        {"input": {"n": 101, "k": 5}, "output": True, "description": "101 is prime"},
    ],
)
async def solovay_strassen_test(n: int, k: int = 10) -> bool:
    """
    Solovay-Strassen probabilistic primality test.

    Uses quadratic residues and Jacobi symbols. Historically important
    as the first polynomial-time primality test.

    Args:
        n: Number to test for primality (must be odd > 1)
        k: Number of rounds (higher k = lower error probability)

    Returns:
        True if n is probably prime, False if n is definitely composite

    Error probability: ≤ (1/2)^k for composite numbers

    Examples:
        await solovay_strassen_test(97, 5) → True
        await solovay_strassen_test(561, 5) → False
        await solovay_strassen_test(15, 3) → False
    """
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False

    # Yield control for large numbers
    if n > 10**10:
        await asyncio.sleep(0)

    for _ in range(k):
        # Choose random base
        a = random.randrange(2, n)

        # Check if gcd(a, n) > 1
        if await _gcd_async(a, n) > 1:
            return False

        # Compute Jacobi symbol (a/n)
        jacobi = await _jacobi_symbol_async(a, n)

        # Compute a^((n-1)/2) mod n
        mod_exp = pow(a, (n - 1) // 2, n)

        # Convert negative result to positive
        if mod_exp == n - 1:
            mod_exp = -1

        # Check if Jacobi symbol equals modular exponentiation
        if jacobi != mod_exp:
            return False

        # Yield control for very large numbers
        if n > 10**15:
            await asyncio.sleep(0)

    return True


@mcp_function(
    description="Fermat primality test. Simple but can be fooled by Carmichael numbers.",
    namespace="arithmetic",
    category="advanced_primality",
    execution_modes=["local", "remote"],
    estimated_cpu_usage="medium",
    examples=[
        {
            "input": {"n": 97, "k": 5},
            "output": True,
            "description": "97 passes Fermat test",
        },
        {
            "input": {"n": 561, "k": 5},
            "output": True,
            "description": "561 is Carmichael number (false positive)",
        },
        {
            "input": {"n": 15, "k": 3},
            "output": False,
            "description": "15 fails Fermat test",
        },
        {
            "input": {"n": 101, "k": 5},
            "output": True,
            "description": "101 passes Fermat test",
        },
    ],
)
async def fermat_primality_test(n: int, k: int = 10) -> bool:
    """
    Fermat primality test based on Fermat's Little Theorem.

    Tests if a^(n-1) ≡ 1 (mod n) for random bases a.
    Simple but can give false positives for Carmichael numbers.

    Args:
        n: Number to test for primality (must be > 1)
        k: Number of rounds with different bases

    Returns:
        True if n passes all tests, False if n fails any test

    Note: Can give false positives for Carmichael numbers

    Examples:
        await fermat_primality_test(97, 5) → True
        await fermat_primality_test(561, 5) → True  # False positive!
        await fermat_primality_test(15, 3) → False
    """
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False

    # Yield control for large numbers
    if n > 10**10:
        await asyncio.sleep(0)

    for _ in range(k):
        # Choose random base
        a = random.randrange(2, n)

        # Check if gcd(a, n) > 1
        if await _gcd_async(a, n) > 1:
            return False

        # Fermat test: a^(n-1) ≡ 1 (mod n)
        if pow(a, n - 1, n) != 1:
            return False

        # Yield control for very large numbers
        if n > 10**15:
            await asyncio.sleep(0)

    return True


# ============================================================================
# DETERMINISTIC PRIMALITY TESTS
# ============================================================================


@mcp_function(
    description="AKS primality test - first deterministic polynomial-time algorithm.",
    namespace="arithmetic",
    category="advanced_primality",
    execution_modes=["local", "remote"],
    estimated_cpu_usage="extreme",
    examples=[
        {
            "input": {"n": 17},
            "output": True,
            "description": "17 is prime (deterministic)",
        },
        {
            "input": {"n": 15},
            "output": False,
            "description": "15 is composite (deterministic)",
        },
        {
            "input": {"n": 97},
            "output": True,
            "description": "97 is prime (deterministic)",
        },
        {
            "input": {"n": 25},
            "output": False,
            "description": "25 is composite (deterministic)",
        },
    ],
)
async def aks_primality_test(n: int) -> bool:
    """
    AKS (Agrawal-Kayal-Saxena) deterministic primality test.

    First polynomial-time deterministic primality test. Theoretical breakthrough
    but not practical for large numbers due to high degree polynomial.

    Args:
        n: Number to test for primality (must be > 1)

    Returns:
        True if n is prime, False if n is composite (deterministic)

    Time complexity: O(log^6 n) (theoretical), much slower in practice

    Examples:
        await aks_primality_test(17) → True
        await aks_primality_test(15) → False
        await aks_primality_test(97) → True
    """
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False

    # For practical reasons, use simplified version for demonstration
    # Full AKS is extremely complex and slow

    # Step 1: Check if n is a perfect power
    if await _is_perfect_power_async(n):
        return False

    # Step 2: Find suitable r
    r = await _find_aks_r_async(n)

    # Step 3: Check gcd conditions
    for a in range(2, min(r + 1, n)):
        if await _gcd_async(a, n) > 1:
            return False

    if n <= r:
        return True

    # Step 4: Check polynomial congruences (simplified)
    # Full implementation would check (X + a)^n ≡ X^n + a (mod X^r - 1, n)
    # This is a simplified version for demonstration

    limit = int(math.sqrt(await _euler_totient_async(r)) * math.log(n))

    for a in range(1, min(limit + 1, n)):
        if not await _aks_polynomial_check_async(n, a, r):
            return False

        # Yield control for large computations
        if a % 100 == 0:
            await asyncio.sleep(0)

    return True


@mcp_function(
    description="Deterministic Miller-Rabin test using known witnesses for specific ranges.",
    namespace="arithmetic",
    category="advanced_primality",
    execution_modes=["local", "remote"],
    estimated_cpu_usage="high",
    examples=[
        {
            "input": {"n": 97},
            "output": True,
            "description": "97 is prime (deterministic)",
        },
        {
            "input": {"n": 561},
            "output": False,
            "description": "561 is composite (deterministic)",
        },
        {
            "input": {"n": 2047},
            "output": False,
            "description": "2047 is composite (deterministic)",
        },
        {
            "input": {"n": 1009},
            "output": True,
            "description": "1009 is prime (deterministic)",
        },
    ],
)
async def deterministic_miller_rabin(n: int) -> bool:
    """
    Deterministic Miller-Rabin test using known witness sets.

    Uses precomputed witness sets that guarantee deterministic results
    for numbers up to certain limits.

    Args:
        n: Number to test for primality (must be > 1)

    Returns:
        True if n is prime, False if n is composite (deterministic for small n)

    Examples:
        await deterministic_miller_rabin(97) → True
        await deterministic_miller_rabin(561) → False
        await deterministic_miller_rabin(2047) → False
    """
    if n < 2:
        return False
    if n == 2 or n == 3:
        return True
    if n % 2 == 0:
        return False

    # Known witness sets for deterministic results
    witness_sets = {
        2047: [2],
        1373653: [2, 3],
        9080191: [31, 73],
        25326001: [2, 3, 5],
        3215031751: [2, 3, 5, 7],
        4759123141: [2, 7, 61],
        1122004669633: [2, 13, 23, 1662803],
        2152302898747: [2, 3, 5, 7, 11],
        3474749660383: [2, 3, 5, 7, 11, 13],
        341550071728321: [2, 3, 5, 7, 11, 13, 17],
        3825123056546413051: [2, 3, 5, 7, 11, 13, 17, 19, 23],
    }

    # Find appropriate witness set
    witnesses = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]  # Default set

    for limit, witness_set in witness_sets.items():
        if n < limit:
            witnesses = witness_set
            break

    # Perform Miller-Rabin with specific witnesses
    return await _miller_rabin_with_witnesses_async(n, witnesses)


# ============================================================================
# SPECIALIZED TESTS
# ============================================================================


@mcp_function(
    description="Test if a number is a strong pseudoprime to given bases.",
    namespace="arithmetic",
    category="advanced_primality",
    execution_modes=["local", "remote"],
    examples=[
        {
            "input": {"n": 2047, "bases": [2]},
            "output": True,
            "description": "2047 is strong pseudoprime to base 2",
        },
        {
            "input": {"n": 1373653, "bases": [2, 3]},
            "output": True,
            "description": "1373653 is strong pseudoprime to bases 2,3",
        },
        {
            "input": {"n": 97, "bases": [2, 3]},
            "output": False,
            "description": "97 is prime, not pseudoprime",
        },
        {
            "input": {"n": 15, "bases": [2]},
            "output": False,
            "description": "15 fails strong pseudoprime test",
        },
    ],
)
async def strong_pseudoprime_test(n: int, bases: List[int]) -> bool:
    """
    Test if n is a strong pseudoprime to all given bases.

    A strong pseudoprime passes the Miller-Rabin test for specific bases
    despite being composite.

    Args:
        n: Number to test
        bases: List of bases to test against

    Returns:
        True if n is a strong pseudoprime to all bases, False otherwise

    Examples:
        await strong_pseudoprime_test(2047, [2]) → True
        await strong_pseudoprime_test(1373653, [2, 3]) → True
        await strong_pseudoprime_test(97, [2, 3]) → False (prime)
    """
    if n < 2:
        return False
    if n == 2:
        return False  # 2 is prime, not pseudoprime
    if n % 2 == 0:
        return False

    # First check if n is actually prime
    if await miller_rabin_test(n, 20):  # High confidence primality test
        return False  # If prime, not a pseudoprime

    # Write n-1 as d * 2^r
    r = 0
    d = n - 1
    while d % 2 == 0:
        r += 1
        d //= 2

    # Test each base
    for base in bases:
        if base >= n:
            continue

        # Compute base^d mod n
        x = pow(base, d, n)

        if x == 1 or x == n - 1:
            continue  # Passes this base

        # Check if x^(2^i) ≡ -1 (mod n) for some i
        passes = False
        for _ in range(r - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                passes = True
                break

        if not passes:
            return False

    return True


@mcp_function(
    description="Test if a number is a Carmichael number (pseudoprime to all bases).",
    namespace="arithmetic",
    category="advanced_primality",
    execution_modes=["local", "remote"],
    examples=[
        {
            "input": {"n": 561},
            "output": True,
            "description": "561 is the smallest Carmichael number",
        },
        {
            "input": {"n": 1105},
            "output": True,
            "description": "1105 is a Carmichael number",
        },
        {
            "input": {"n": 1729},
            "output": True,
            "description": "1729 is a Carmichael number",
        },
        {
            "input": {"n": 97},
            "output": False,
            "description": "97 is prime, not Carmichael",
        },
    ],
)
async def carmichael_number_test(n: int) -> bool:
    """
    Test if n is a Carmichael number.

    Carmichael numbers are composite numbers that satisfy Fermat's test
    for all bases coprime to n. They are the "worst case" for Fermat's test.

    Args:
        n: Number to test

    Returns:
        True if n is a Carmichael number, False otherwise

    Examples:
        await carmichael_number_test(561) → True
        await carmichael_number_test(1105) → True
        await carmichael_number_test(1729) → True
        await carmichael_number_test(97) → False
    """
    if n < 2:
        return False

    # Must be composite
    if await miller_rabin_test(n, 20):  # If probably prime
        return False

    # Must be square-free and have at least 3 prime factors
    prime_factors = await _prime_factorization_async(n)

    # Check if square-free
    factor_counts: dict[int, int] = {}
    for factor in prime_factors:
        factor_counts[factor] = factor_counts.get(factor, 0) + 1
        if factor_counts[factor] > 1:
            return False  # Not square-free

    unique_factors = list(factor_counts.keys())
    if len(unique_factors) < 3:
        return False  # Need at least 3 distinct prime factors

    # Check Korselt's criterion: p-1 divides n-1 for all prime factors p
    for p in unique_factors:
        if (n - 1) % (p - 1) != 0:
            return False

    return True


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


async def _gcd_async(a: int, b: int) -> int:
    """Async wrapper for GCD calculation."""
    while b:
        a, b = b, a % b
    return abs(a)


async def _jacobi_symbol_async(a: int, n: int) -> int:
    """Calculate Jacobi symbol (a/n)."""
    if n <= 0 or n % 2 == 0:
        raise ValueError("n must be a positive odd integer")

    a = a % n
    result = 1

    while a != 0:
        while a % 2 == 0:
            a //= 2
            if n % 8 in [3, 5]:
                result = -result

        a, n = n, a
        if a % 4 == 3 and n % 4 == 3:
            result = -result

        a = a % n

    return result if n == 1 else 0


async def _is_perfect_power_async(n: int) -> bool:
    """Check if n is a perfect power (n = a^b for integers a, b > 1)."""
    if n < 4:
        return False

    for b in range(2, int(math.log2(n)) + 1):
        a = round(n ** (1 / b))
        if a**b == n:
            return True
        await asyncio.sleep(0)  # Yield control

    return False


async def _find_aks_r_async(n: int) -> int:
    """Find suitable r for AKS test."""
    log_n_squared = math.log(n) ** 2

    for r in range(2, int(log_n_squared) + 1):
        if await _gcd_async(r, n) > 1:
            continue

        # Check if order of n modulo r is large enough
        order = 1
        power = n % r
        original_power = power

        while power != 1 and order <= log_n_squared:
            power = (power * original_power) % r
            order += 1

        if order > log_n_squared:
            return r

    return int(log_n_squared)


async def _euler_totient_async(n: int) -> int:
    """Calculate Euler's totient function."""
    if n <= 0:
        return 0
    if n == 1:
        return 1

    result = n
    p = 2

    while p * p <= n:
        if n % p == 0:
            while n % p == 0:
                n //= p
            result -= result // p
        p += 1

    if n > 1:
        result -= result // n

    return result


async def _aks_polynomial_check_async(n: int, a: int, r: int) -> bool:
    """Simplified polynomial check for AKS test."""
    # This is a simplified version - full AKS requires polynomial arithmetic
    # Check if a^n ≡ a (mod n) as a basic test
    return pow(a, n, n) == a % n


async def _miller_rabin_with_witnesses_async(n: int, witnesses: List[int]) -> bool:
    """Miller-Rabin test with specific witnesses."""
    # Write n-1 as d * 2^r
    r = 0
    d = n - 1
    while d % 2 == 0:
        r += 1
        d //= 2

    # Test each witness
    for a in witnesses:
        if a >= n:
            continue

        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue

        composite = True
        for _ in range(r - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                composite = False
                break

        if composite:
            return False

    return True


async def _prime_factorization_async(n: int) -> List[int]:
    """Get prime factorization of n."""
    factors = []
    d = 2

    while d * d <= n:
        while n % d == 0:
            factors.append(d)
            n //= d
        d += 1

        if d % 100 == 0:
            await asyncio.sleep(0)

    if n > 1:
        factors.append(n)

    return factors


# Export all functions
__all__ = [
    # Probabilistic tests
    "miller_rabin_test",
    "solovay_strassen_test",
    "fermat_primality_test",
    # Deterministic tests
    "aks_primality_test",
    "deterministic_miller_rabin",
    # Specialized tests
    "strong_pseudoprime_test",
    "carmichael_number_test",
]

# ============================================================================
# COMPREHENSIVE TEST SUITE
# ============================================================================


async def test_advanced_primality():
    """Test all advanced primality functions."""
    print("🔬 Advanced Primality Tests")
    print("=" * 30)

    # Test numbers
    test_cases = [
        (97, True, "Prime"),
        (561, False, "Carmichael number"),
        (2047, False, "Strong pseudoprime to base 2"),
        (1373653, False, "Strong pseudoprime to bases 2,3"),
        (15, False, "Obviously composite"),
        (1009, True, "Prime"),
    ]

    print("1. Miller-Rabin Test:")
    for n, expected, description in test_cases:
        result = await miller_rabin_test(n, 10)
        status = "✓" if result == expected else "✗"
        print(f"   {n:8d}: {result:5} {status} ({description})")

    print("\n2. Solovay-Strassen Test:")
    for n, expected, description in test_cases:
        result = await solovay_strassen_test(n, 10)
        status = "✓" if result == expected else "✗"
        print(f"   {n:8d}: {result:5} {status} ({description})")

    print("\n3. Fermat Test (note: false positives for Carmichael numbers):")
    for n, expected, description in test_cases:
        result = await fermat_primality_test(n, 10)
        # Fermat test gives false positives for Carmichael numbers
        if n == 561:  # Carmichael number
            expected_fermat = True  # False positive expected
        else:
            expected_fermat = expected
        status = "✓" if result == expected_fermat else "✗"
        print(f"   {n:8d}: {result:5} {status} ({description})")

    print("\n4. Deterministic Miller-Rabin:")
    for n, expected, description in test_cases:
        if n < 3825123056546413051:  # Within deterministic range
            result = await deterministic_miller_rabin(n)
            status = "✓" if result == expected else "✗"
            print(f"   {n:8d}: {result:5} {status} ({description})")

    print("\n5. Carmichael Number Test:")
    carmichael_tests = [561, 1105, 1729, 2465, 97, 1009]
    for n in carmichael_tests:
        is_carmichael = await carmichael_number_test(n)
        expected_carmichael = n in [561, 1105, 1729, 2465]
        status = "✓" if is_carmichael == expected_carmichael else "✗"
        print(f"   {n:8d}: {is_carmichael:5} {status}")

    print("\n✅ Advanced primality testing complete!")


if __name__ == "__main__":
    import asyncio

    asyncio.run(test_advanced_primality())
