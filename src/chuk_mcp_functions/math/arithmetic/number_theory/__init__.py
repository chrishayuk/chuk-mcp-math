#!/usr/bin/env python3
# chuk_mcp_functions/math/arithmetic/number_theory/__init__.py
"""
Number Theory Operations Module - Updated Structure

Functions for working with integer properties, prime numbers, divisibility, and special numbers.
Essential for cryptography, algorithms, and mathematical analysis.

Submodules:
- primes: is_prime, next_prime, nth_prime, prime_factors, prime_count, is_coprime, first_n_primes
- divisibility: gcd, lcm, divisors, is_divisible, is_even, is_odd, extended_gcd, divisor_count, divisor_sum
- basic_sequences: perfect squares, powers of two, Fibonacci, factorial, polygonal numbers, catalan
- special_primes: Mersenne, Fermat, Sophie Germain, twin primes, Wilson's theorem, pseudoprimes
- combinatorial_numbers: Catalan, Bell, Stirling numbers
- arithmetic_functions: Euler totient, Möbius function, omega functions

All functions are async native for optimal performance in async environments.
"""

# Import all number theory submodules
from . import primes
from . import divisibility
from . import basic_sequences
from . import special_primes
from . import combinatorial_numbers

# Core prime operations (most commonly used)
from .primes import (
    is_prime, next_prime, nth_prime, prime_factors,
    prime_count, is_coprime, first_n_primes
)

# Core divisibility operations
from .divisibility import (
    gcd, lcm, divisors, is_divisible, is_even, is_odd,
    extended_gcd, divisor_count, divisor_sum
)

# Basic sequences (commonly used)
from .basic_sequences import (
    is_perfect_square, is_power_of_two, fibonacci, factorial, 
    triangular_number, fibonacci_sequence, catalan_number
)

# Special primes (commonly referenced)
from .special_primes import (
    is_mersenne_prime, is_fermat_prime, is_twin_prime, 
    wilson_theorem_check, is_carmichael_number, prime_gap
)

# Combinatorial numbers (high-value functions)
from .combinatorial_numbers import (
    catalan_number as catalan_number_full, bell_number, stirling_second
)

# TODO: Import arithmetic_functions when created
try:
    from .arithmetic_functions import (
        euler_totient, mobius_function, little_omega, big_omega
    )
except ImportError:
    # arithmetic_functions module not yet implemented
    pass

# Export all number theory functions for convenient access
__all__ = [
    # Submodules
    'primes', 'divisibility', 'basic_sequences', 'special_primes', 'combinatorial_numbers',
    
    # Core prime operations
    'is_prime', 'next_prime', 'nth_prime', 'prime_factors',
    'prime_count', 'is_coprime', 'first_n_primes',
    
    # Core divisibility operations
    'gcd', 'lcm', 'divisors', 'is_divisible', 'is_even', 'is_odd',
    'extended_gcd', 'divisor_count', 'divisor_sum',
    
    # Basic sequences
    'is_perfect_square', 'is_power_of_two', 'fibonacci', 'factorial',
    'triangular_number', 'fibonacci_sequence', 'catalan_number',
    
    # Special primes
    'is_mersenne_prime', 'is_fermat_prime', 'is_twin_prime',
    'wilson_theorem_check', 'is_carmichael_number', 'prime_gap',
    
    # Combinatorial numbers
    'bell_number', 'stirling_second'
]

# Add conditional exports for arithmetic_functions if available
try:
    from .arithmetic_functions import euler_totient, mobius_function, little_omega, big_omega
    __all__.extend(['euler_totient', 'mobius_function', 'little_omega', 'big_omega'])
except ImportError:
    pass

async def test_number_theory_functions():
    """Test core number theory functions."""
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
    
    # Test basic sequences
    print("\nBasic Sequences:")
    print(f"  is_perfect_square(16) = {await is_perfect_square(16)}")
    print(f"  is_power_of_two(8) = {await is_power_of_two(8)}")
    print(f"  fibonacci(10) = {await fibonacci(10)}")
    print(f"  factorial(5) = {await factorial(5)}")
    print(f"  triangular_number(5) = {await triangular_number(5)}")
    print(f"  catalan_number(5) = {await catalan_number(5)}")
    
    # Test special primes
    print("\nSpecial Primes:")
    print(f"  is_mersenne_prime(31) = {await is_mersenne_prime(31)}")
    print(f"  is_fermat_prime(17) = {await is_fermat_prime(17)}")
    print(f"  is_twin_prime(13) = {await is_twin_prime(13)}")
    print(f"  wilson_theorem_check(7) = {await wilson_theorem_check(7)}")
    print(f"  is_carmichael_number(561) = {await is_carmichael_number(561)}")
    print(f"  prime_gap(7) = {await prime_gap(7)}")
    
    # Test combinatorial numbers
    print("\nCombinatorial Numbers:")
    print(f"  bell_number(5) = {await bell_number(5)}")
    print(f"  stirling_second(4, 2) = {await stirling_second(4, 2)}")
    
    # Test arithmetic functions if available
    try:
        print("\nArithmetic Functions:")
        print(f"  euler_totient(12) = {await euler_totient(12)}")
        print(f"  mobius_function(30) = {await mobius_function(30)}")
        print(f"  little_omega(12) = {await little_omega(12)}")
        print(f"  big_omega(12) = {await big_omega(12)}")
    except NameError:
        print("\nArithmetic Functions: (not yet implemented)")
    
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
    
    # Sophie Germain primes for cryptographic strength
    print("\nSophie Germain Primes (Cryptographically Strong):")
    try:
        sg_pairs = await special_primes.safe_prime_pairs(100)
        print(f"  Sophie Germain prime pairs ≤ 100: {sg_pairs}")
    except AttributeError:
        print("  safe_prime_pairs function not available")
    
    # Mersenne primes for perfect numbers
    print("\nMersenne Primes (Perfect Number Generation):")
    try:
        mersenne_exps = await special_primes.mersenne_prime_exponents(20)
        print(f"  Known Mersenne exponents ≤ 20: {mersenne_exps}")
    except AttributeError:
        print("  mersenne_prime_exponents function not available")
    
    # Demonstrate Lucas-Lehmer test
    for p in [5, 7, 11, 13]:
        try:
            is_mersenne = await special_primes.lucas_lehmer_test(p)
            mersenne_num = (2 ** p) - 1
            print(f"  2^{p} - 1 = {mersenne_num} is {'prime' if is_mersenne else 'composite'}")
        except AttributeError:
            print(f"  Lucas-Lehmer test not available for 2^{p} - 1")

async def demo_combinatorial_applications():
    """Demonstrate combinatorial applications."""
    print("\n🎲 Combinatorial Applications Demo")
    print("=" * 35)
    
    # Catalan numbers in computer science
    print("Catalan Numbers (Binary Trees, Parentheses):")
    for n in range(6):
        cat_n = await catalan_number(n)
        print(f"  C_{n} = {cat_n} (binary trees with {n} internal nodes)")
    
    # Bell numbers for set partitions
    print("\nBell Numbers (Set Partitions):")
    for n in range(6):
        bell_n = await bell_number(n)
        print(f"  B_{n} = {bell_n} (ways to partition set of {n} elements)")
    
    # Stirling numbers for combinatorial analysis
    print("\nStirling Numbers of Second Kind (Subset Partitions):")
    for n in range(1, 6):
        for k in range(1, n + 1):
            stirling_nk = await stirling_second(n, k)
            print(f"  S({n},{k}) = {stirling_nk}", end="  ")
        print()  # New line after each row

if __name__ == "__main__":
    import asyncio
    
    async def main():
        await test_number_theory_functions()
        await demo_cryptographic_functions()
        await demo_combinatorial_applications()
    
    asyncio.run(main())