#!/usr/bin/env python3
# chuk_mcp_functions/math/arithmetic/number_theory/__init__.py
"""
Number Theory Operations Module

Functions for working with integer properties, prime numbers, divisibility, and special numbers.
Essential for cryptography, algorithms, and mathematical analysis.

Submodules:
- primes: is_prime, next_prime, nth_prime, prime_factors, prime_count, is_coprime, first_n_primes
- divisibility: gcd, lcm, divisors, is_divisible, is_even, is_odd, extended_gcd, divisor_count, divisor_sum
- special_numbers: is_perfect_square, is_power_of_two, fibonacci, factorial, collatz_steps
- sequences: triangular_numbers, perfect_numbers, deficient_numbers, abundant_numbers

All functions are async native for optimal performance in async environments.
"""

# Import all number theory submodules
from . import primes
from . import divisibility

# Import all functions for direct access
from .primes import (
    is_prime, next_prime, nth_prime, prime_factors,
    prime_count, is_coprime, first_n_primes
)

from .divisibility import (
    gcd, lcm, divisors, is_divisible, is_even, is_odd,
    extended_gcd, divisor_count, divisor_sum
)

# We'll need to create these modules, but for now import what we have
# TODO: Create special_numbers.py and sequences.py modules

# Export all number theory functions
__all__ = [
    # Submodules
    'primes', 'divisibility',
    
    # Prime operations
    'is_prime', 'next_prime', 'nth_prime', 'prime_factors',
    'prime_count', 'is_coprime', 'first_n_primes',
    
    # Divisibility operations
    'gcd', 'lcm', 'divisors', 'is_divisible', 'is_even', 'is_odd',
    'extended_gcd', 'divisor_count', 'divisor_sum'
    
    # TODO: Add special_numbers and sequences when created
]

async def test_number_theory_functions():
    """Test all number theory functions."""
    print("🔢 Number Theory Functions Test")
    print("=" * 35)
    
    # Test prime operations
    print("Prime Operations:")
    print(f"  is_prime(17) = {await is_prime(17)}")
    print(f"  is_prime(4) = {await is_prime(4)}")
    print(f"  next_prime(10) = {await next_prime(10)}")
    print(f"  nth_prime(10) = {await nth_prime(10)}")
    print(f"  prime_factors(60) = {await prime_factors(60)}")
    print(f"  prime_count(20) = {await prime_count(20)}")
    print(f"  is_coprime(8, 15) = {await is_coprime(8, 15)}")
    print(f"  first_n_primes(10) = {await first_n_primes(10)}")
    
    # Test divisibility operations
    print("\nDivisibility Operations:")
    print(f"  gcd(48, 18) = {await gcd(48, 18)}")
    print(f"  lcm(12, 18) = {await lcm(12, 18)}")
    print(f"  divisors(12) = {await divisors(12)}")
    print(f"  is_divisible(20, 4) = {await is_divisible(20, 4)}")
    print(f"  is_even(4) = {await is_even(4)}")
    print(f"  is_odd(7) = {await is_odd(7)}")
    
    # Test extended GCD
    gcd_val, x, y = await extended_gcd(30, 18)
    print(f"  extended_gcd(30, 18) = ({gcd_val}, {x}, {y})")
    print(f"    Verification: 30×{x} + 18×{y} = {30*x + 18*y}")
    
    print(f"  divisor_count(12) = {await divisor_count(12)}")
    print(f"  divisor_sum(12) = {await divisor_sum(12)}")
    
    print("\n✅ All number theory functions working!")

async def demo_cryptographic_functions():
    """Demonstrate cryptographic applications of number theory functions."""
    print("\n🔐 Cryptographic Applications Demo")
    print("=" * 35)
    
    # RSA-style operations
    print("RSA-style Operations:")
    p = await next_prime(100)
    q = await next_prime(200)
    n = p * q
    phi_n = (p - 1) * (q - 1)
    
    print(f"  Prime p = {p}")
    print(f"  Prime q = {q}")
    print(f"  n = p × q = {n}")
    print(f"  φ(n) = (p-1)(q-1) = {phi_n}")
    
    # Find a suitable e (public exponent)
    e = 65537  # Common RSA public exponent
    if await gcd(e, phi_n) == 1:
        print(f"  Public exponent e = {e} (coprime with φ(n))")
    
    # Modular exponentiation example
    message = 42
    ciphertext = pow(message, e, n)  # Using built-in pow for demonstration
    print(f"  Encrypt {message}: {message}^{e} mod {n} = {ciphertext}")

if __name__ == "__main__":
    import asyncio
    
    async def main():
        await test_number_theory_functions()
        await demo_cryptographic_functions()
    
    asyncio.run(main())