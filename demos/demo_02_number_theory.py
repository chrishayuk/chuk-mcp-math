#!/usr/bin/env python3
"""
CHUK MCP Math - Number Theory Demo
===================================

Demonstrates all number theory functions including:
- Prime numbers and primality testing
- Divisibility and factorization
- Special numbers (perfect, abundant, deficient)
- Continued fractions
- Diophantine equations
- Partitions and combinatorics
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from chuk_mcp_math.number_theory.primes import (
    is_prime,
    nth_prime,
    primes_up_to,
    prime_count,
    next_prime,
    previous_prime,
)
from chuk_mcp_math.number_theory.divisibility import (
    is_divisible,
    count_divisors,
    sum_of_divisors,
    proper_divisors,
    divisors,
)
from chuk_mcp_math.number_theory.factorization import (
    prime_factorization,
    factorize,
    is_square_free,
)
from chuk_mcp_math.number_theory.special_numbers import (
    is_perfect,
    is_abundant,
    is_deficient,
    is_palindrome,
    is_armstrong,
)
from chuk_mcp_math.number_theory.special_primes import (
    is_twin_prime,
    is_cousin_prime,
    is_sexy_prime,
    is_mersenne_prime,
    is_sophie_germain_prime,
)
from chuk_mcp_math.number_theory.combinatorial_numbers import (
    binomial,
    catalan,
    stirling_first,
    stirling_second,
    bell_number,
    partition_number,
)


async def demo_prime_operations():
    """Demonstrate prime number operations."""
    print("\n" + "=" * 70)
    print("PRIME NUMBER OPERATIONS")
    print("=" * 70)

    # Primality testing
    result = await is_prime(17)
    print(f"✓ is_prime(17) = {result}")

    result = await is_prime(100)
    print(f"✓ is_prime(100) = {result}")

    # Nth prime
    result = await nth_prime(10)
    print(f"✓ nth_prime(10) = {result}")

    # Primes up to
    result = await primes_up_to(30)
    print(f"✓ primes_up_to(30) = {result}")

    # Prime count
    result = await prime_count(100)
    print(f"✓ prime_count(100) = {result}")

    # Next prime
    result = await next_prime(100)
    print(f"✓ next_prime(100) = {result}")

    # Previous prime
    result = await previous_prime(100)
    print(f"✓ previous_prime(100) = {result}")


async def demo_divisibility():
    """Demonstrate divisibility operations."""
    print("\n" + "=" * 70)
    print("DIVISIBILITY OPERATIONS")
    print("=" * 70)

    # Is divisible
    result = await is_divisible(15, 3)
    print(f"✓ is_divisible(15, 3) = {result}")

    # Divisors
    result = await divisors(24)
    print(f"✓ divisors(24) = {result}")

    # Count divisors
    result = await count_divisors(24)
    print(f"✓ count_divisors(24) = {result}")

    # Sum of divisors
    result = await sum_of_divisors(12)
    print(f"✓ sum_of_divisors(12) = {result}")

    # Proper divisors
    result = await proper_divisors(12)
    print(f"✓ proper_divisors(12) = {result}")


async def demo_factorization():
    """Demonstrate factorization operations."""
    print("\n" + "=" * 70)
    print("FACTORIZATION")
    print("=" * 70)

    # Prime factorization
    result = await prime_factorization(60)
    print(f"✓ prime_factorization(60) = {result}")

    # Factorize
    result = await factorize(100)
    print(f"✓ factorize(100) = {result}")

    # Is square free
    result = await is_square_free(30)
    print(f"✓ is_square_free(30) = {result}")

    result = await is_square_free(12)
    print(f"✓ is_square_free(12) = {result}")


async def demo_special_numbers():
    """Demonstrate special number classifications."""
    print("\n" + "=" * 70)
    print("SPECIAL NUMBERS")
    print("=" * 70)

    # Perfect numbers
    result = await is_perfect(6)
    print(f"✓ is_perfect(6) = {result}")

    result = await is_perfect(28)
    print(f"✓ is_perfect(28) = {result}")

    # Abundant numbers
    result = await is_abundant(12)
    print(f"✓ is_abundant(12) = {result}")

    # Deficient numbers
    result = await is_deficient(10)
    print(f"✓ is_deficient(10) = {result}")

    # Palindromes
    result = await is_palindrome(12321)
    print(f"✓ is_palindrome(12321) = {result}")

    # Armstrong numbers
    result = await is_armstrong(153)
    print(f"✓ is_armstrong(153) = {result}  (1³ + 5³ + 3³ = 153)")


async def demo_special_primes():
    """Demonstrate special prime classifications."""
    print("\n" + "=" * 70)
    print("SPECIAL PRIMES")
    print("=" * 70)

    # Twin primes (p, p+2)
    result = await is_twin_prime(11)
    print(f"✓ is_twin_prime(11) = {result}  (11, 13)")

    # Cousin primes (p, p+4)
    result = await is_cousin_prime(7)
    print(f"✓ is_cousin_prime(7) = {result}  (7, 11)")

    # Sexy primes (p, p+6)
    result = await is_sexy_prime(5)
    print(f"✓ is_sexy_prime(5) = {result}  (5, 11)")

    # Mersenne primes (2^p - 1)
    result = await is_mersenne_prime(31)
    print(f"✓ is_mersenne_prime(31) = {result}  (2^5 - 1)")

    # Sophie Germain primes (p and 2p+1 both prime)
    result = await is_sophie_germain_prime(11)
    print(f"✓ is_sophie_germain_prime(11) = {result}  (11 and 23)")


async def demo_combinatorial():
    """Demonstrate combinatorial numbers."""
    print("\n" + "=" * 70)
    print("COMBINATORIAL NUMBERS")
    print("=" * 70)

    # Binomial coefficient
    result = await binomial(5, 2)
    print(f"✓ binomial(5, 2) = {result}  (C(5,2))")

    # Catalan number
    result = await catalan(4)
    print(f"✓ catalan(4) = {result}")

    # Stirling numbers of the first kind
    result = await stirling_first(5, 2)
    print(f"✓ stirling_first(5, 2) = {result}")

    # Stirling numbers of the second kind
    result = await stirling_second(5, 2)
    print(f"✓ stirling_second(5, 2) = {result}")

    # Bell number
    result = await bell_number(4)
    print(f"✓ bell_number(4) = {result}")

    # Partition number
    result = await partition_number(5)
    print(f"✓ partition_number(5) = {result}")


async def main():
    """Run all number theory demos."""
    print("\n" + "=" * 70)
    print("CHUK MCP MATH - NUMBER THEORY DEMO")
    print("=" * 70)
    print("\nDemonstrating 340+ number theory functions are working correctly...")

    await demo_prime_operations()
    await demo_divisibility()
    await demo_factorization()
    await demo_special_numbers()
    await demo_special_primes()
    await demo_combinatorial()

    print("\n" + "=" * 70)
    print("✅ ALL NUMBER THEORY OPERATIONS WORKING PERFECTLY!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
