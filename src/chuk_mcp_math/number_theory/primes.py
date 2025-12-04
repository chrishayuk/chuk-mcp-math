#!/usr/bin/env python3
# chuk_mcp_math/number_theory/primes.py
"""
Prime Number Operations - Async Native

Functions for working with prime numbers, factorization, and primality testing.

Functions:
- is_prime, next_prime, nth_prime
- prime_factors, prime_count, is_coprime
"""

import math
import asyncio
from typing import List
from chuk_mcp_math.mcp_decorator import mcp_function


@mcp_function(
    description="Check if a number is prime. A prime number is only divisible by 1 and itself.",
    namespace="arithmetic",
    category="number_theory",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    estimated_cpu_usage="medium",
    examples=[
        {"input": {"n": 17}, "output": True, "description": "17 is prime"},
        {
            "input": {"n": 4},
            "output": False,
            "description": "4 is not prime (divisible by 2)",
        },
        {"input": {"n": 2}, "output": True, "description": "2 is the smallest prime"},
        {
            "input": {"n": 1},
            "output": False,
            "description": "1 is not considered prime",
        },
    ],
)
async def is_prime(n: int) -> bool:
    """
    Check if a number is prime.

    Args:
        n: Positive integer to check

    Returns:
        True if n is prime, False otherwise

    Examples:
        await is_prime(17) → True
        await is_prime(4) → False
        await is_prime(2) → True
        await is_prime(1) → False
    """
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False

    # For large numbers, yield control periodically
    sqrt_n = int(math.sqrt(n))
    if sqrt_n > 10000:
        await asyncio.sleep(0)

    # Check odd divisors up to sqrt(n)
    for i in range(3, sqrt_n + 1, 2):
        if n % i == 0:
            return False
        # Yield control every 1000 iterations for very large numbers
        if i % 1000 == 999 and sqrt_n > 10000:
            await asyncio.sleep(0)

    return True


@mcp_function(
    description="Find the next prime number greater than the given number.",
    namespace="arithmetic",
    category="number_theory",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    estimated_cpu_usage="medium",
    examples=[
        {"input": {"n": 10}, "output": 11, "description": "Next prime after 10"},
        {"input": {"n": 17}, "output": 19, "description": "Next prime after prime 17"},
        {"input": {"n": 1}, "output": 2, "description": "Next prime after 1"},
        {"input": {"n": 100}, "output": 101, "description": "Next prime after 100"},
    ],
)
async def next_prime(n: int) -> int:
    """
    Find the next prime number greater than n.

    Args:
        n: Starting number

    Returns:
        The smallest prime number greater than n

    Examples:
        await next_prime(10) → 11
        await next_prime(17) → 19
        await next_prime(1) → 2
    """
    candidate = n + 1
    checks = 0

    while not await is_prime(candidate):
        candidate += 1
        checks += 1
        # Yield control every 100 checks for large searches
        if checks % 100 == 0:
            await asyncio.sleep(0)

    return candidate


@mcp_function(
    description="Find the nth prime number (1-indexed). Uses efficient prime generation.",
    namespace="arithmetic",
    category="number_theory",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    estimated_cpu_usage="high",
    examples=[
        {"input": {"n": 1}, "output": 2, "description": "1st prime is 2"},
        {"input": {"n": 10}, "output": 29, "description": "10th prime is 29"},
        {"input": {"n": 5}, "output": 11, "description": "5th prime is 11"},
        {"input": {"n": 25}, "output": 97, "description": "25th prime is 97"},
    ],
)
async def nth_prime(n: int) -> int:
    """
    Find the nth prime number (1-indexed).

    Args:
        n: Position of prime to find (must be positive)

    Returns:
        The nth prime number

    Raises:
        ValueError: If n is not positive

    Examples:
        await nth_prime(1) → 2
        await nth_prime(10) → 29
        await nth_prime(5) → 11
    """
    if n < 1:
        raise ValueError("n must be positive")

    if n == 1:
        return 2

    primes_found = 1
    candidate = 3
    checks = 0

    while primes_found < n:
        if await is_prime(candidate):
            primes_found += 1
            if primes_found == n:
                return candidate
        candidate += 2
        checks += 1

        # Yield control every 100 checks for large n
        if checks % 100 == 0 and n > 100:
            await asyncio.sleep(0)

    return candidate


@mcp_function(
    description="Find all prime factors of a number. Returns the prime factorization as a list.",
    namespace="arithmetic",
    category="number_theory",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    estimated_cpu_usage="high",
    examples=[
        {"input": {"n": 12}, "output": [2, 2, 3], "description": "12 = 2² × 3"},
        {"input": {"n": 17}, "output": [17], "description": "17 is prime"},
        {"input": {"n": 60}, "output": [2, 2, 3, 5], "description": "60 = 2² × 3 × 5"},
        {"input": {"n": 1}, "output": [], "description": "1 has no prime factors"},
    ],
)
async def prime_factors(n: int) -> List[int]:
    """
    Find all prime factors of a number.

    Args:
        n: Positive integer to factorize

    Returns:
        List of prime factors (with repetition)

    Examples:
        await prime_factors(12) → [2, 2, 3]
        await prime_factors(17) → [17]
        await prime_factors(60) → [2, 2, 3, 5]
    """
    if n <= 1:
        return []

    factors = []
    d = 2
    original_n = n

    # Yield control for large numbers
    if n > 100000:
        await asyncio.sleep(0)

    while d * d <= n:
        while n % d == 0:
            factors.append(d)
            n //= d
        d += 1

        # Yield control periodically for large factorizations
        if d % 1000 == 0 and original_n > 100000:
            await asyncio.sleep(0)

    if n > 1:
        factors.append(n)

    return factors


@mcp_function(
    description="Count the number of prime numbers less than or equal to n (prime counting function π(n)).",
    namespace="arithmetic",
    category="number_theory",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    estimated_cpu_usage="high",
    examples=[
        {"input": {"n": 10}, "output": 4, "description": "Primes ≤ 10: 2, 3, 5, 7"},
        {"input": {"n": 20}, "output": 8, "description": "8 primes ≤ 20"},
        {"input": {"n": 2}, "output": 1, "description": "Only prime ≤ 2 is 2"},
        {"input": {"n": 1}, "output": 0, "description": "No primes ≤ 1"},
    ],
)
async def prime_count(n: int) -> int:
    """
    Count the number of prime numbers less than or equal to n.

    Args:
        n: Upper limit (inclusive)

    Returns:
        Number of primes ≤ n

    Examples:
        await prime_count(10) → 4  # 2, 3, 5, 7
        await prime_count(20) → 8
        await prime_count(2) → 1
    """
    if n < 2:
        return 0

    count = 0
    checks = 0

    for i in range(2, n + 1):
        if await is_prime(i):
            count += 1
        checks += 1

        # Yield control every 1000 checks for large n
        if checks % 1000 == 0 and n > 1000:
            await asyncio.sleep(0)

    return count


@mcp_function(
    description="Check if two numbers are coprime (their greatest common divisor is 1).",
    namespace="arithmetic",
    category="number_theory",
    execution_modes=["local", "remote"],
    examples=[
        {
            "input": {"a": 8, "b": 15},
            "output": True,
            "description": "8 and 15 are coprime (gcd=1)",
        },
        {
            "input": {"a": 12, "b": 18},
            "output": False,
            "description": "12 and 18 are not coprime (gcd=6)",
        },
        {
            "input": {"a": 7, "b": 13},
            "output": True,
            "description": "Two primes are always coprime",
        },
        {
            "input": {"a": 1, "b": 100},
            "output": True,
            "description": "1 is coprime with any number",
        },
    ],
)
async def is_coprime(a: int, b: int) -> bool:
    """
    Check if two numbers are coprime (gcd = 1).

    Args:
        a: First integer
        b: Second integer

    Returns:
        True if gcd(a, b) = 1, False otherwise

    Examples:
        await is_coprime(8, 15) → True
        await is_coprime(12, 18) → False
        await is_coprime(7, 13) → True
    """
    # Import gcd function from divisibility module
    from .divisibility import gcd

    return await gcd(abs(a), abs(b)) == 1


@mcp_function(
    description="Generate the first n prime numbers using the Sieve of Eratosthenes algorithm.",
    namespace="arithmetic",
    category="number_theory",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    estimated_cpu_usage="high",
    examples=[
        {
            "input": {"n": 5},
            "output": [2, 3, 5, 7, 11],
            "description": "First 5 primes",
        },
        {
            "input": {"n": 10},
            "output": [2, 3, 5, 7, 11, 13, 17, 19, 23, 29],
            "description": "First 10 primes",
        },
        {"input": {"n": 1}, "output": [2], "description": "First prime"},
        {"input": {"n": 0}, "output": [], "description": "No primes requested"},
    ],
)
async def first_n_primes(n: int) -> List[int]:
    """
    Generate the first n prime numbers.

    Args:
        n: Number of primes to generate (non-negative)

    Returns:
        List of the first n prime numbers

    Examples:
        await first_n_primes(5) → [2, 3, 5, 7, 11]
        await first_n_primes(10) → [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
        await first_n_primes(0) → []
    """
    if n <= 0:
        return []

    if n == 1:
        return [2]

    # Estimate upper bound for nth prime (rough approximation)
    if n < 6:
        limit = 12
    else:
        limit = int(n * (math.log(n) + math.log(math.log(n))))

    # Sieve of Eratosthenes
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False

    # Yield control for large sieves
    if limit > 10000:
        await asyncio.sleep(0)

    for i in range(2, int(math.sqrt(limit)) + 1):
        if sieve[i]:
            for j in range(i * i, limit + 1, i):
                sieve[j] = False

        # Yield control every 1000 iterations for large sieves
        if i % 1000 == 0 and limit > 10000:
            await asyncio.sleep(0)

    # Collect primes
    primes = []
    for i in range(2, limit + 1):
        if sieve[i]:
            primes.append(i)
            if len(primes) == n:
                break

    # If we didn't find enough primes, fall back to trial division
    if len(primes) < n:
        candidate = limit + 1
        while len(primes) < n:
            if await is_prime(candidate):
                primes.append(candidate)
            candidate += 1

    return primes[:n]


# Export all prime number functions
__all__ = [
    "is_prime",
    "next_prime",
    "nth_prime",
    "prime_factors",
    "prime_count",
    "is_coprime",
    "first_n_primes",
]

if __name__ == "__main__":
    import asyncio

    async def test_prime_operations():
        """Test all prime number operations."""
        print("🔢 Prime Number Operations Test")
        print("=" * 35)

        # Test basic prime functions
        print(f"is_prime(17) = {await is_prime(17)}")
        print(f"is_prime(4) = {await is_prime(4)}")
        print(f"next_prime(10) = {await next_prime(10)}")
        print(f"nth_prime(10) = {await nth_prime(10)}")

        # Test factorization
        print(f"prime_factors(60) = {await prime_factors(60)}")
        print(f"prime_factors(17) = {await prime_factors(17)}")

        # Test counting and coprimality
        print(f"prime_count(20) = {await prime_count(20)}")
        print(f"is_coprime(8, 15) = {await is_coprime(8, 15)}")
        print(f"is_coprime(12, 18) = {await is_coprime(12, 18)}")

        # Test prime generation
        print(f"first_n_primes(10) = {await first_n_primes(10)}")

        print("\n✅ All prime number operations working!")

    asyncio.run(test_prime_operations())
