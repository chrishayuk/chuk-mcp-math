#!/usr/bin/env python3
# chuk_mcp_functions/math/arithmetic/number_theory/__init__.py
"""
Number Theory Operations Module - Enhanced with New Modules

Functions for working with integer properties, prime numbers, divisibility, and special numbers.
Essential for cryptography, algorithms, and mathematical analysis.

Submodules:
- primes: is_prime, next_prime, nth_prime, prime_factors, prime_count, is_coprime, first_n_primes
- divisibility: gcd, lcm, divisors, is_divisible, is_even, is_odd, extended_gcd, divisor_count, divisor_sum
- basic_sequences: perfect squares, powers of two, Fibonacci, factorial, polygonal numbers, catalan
- special_primes: Mersenne, Fermat, Sophie Germain, twin primes, Wilson's theorem, pseudoprimes
- combinatorial_numbers: Catalan, Bell, Stirling numbers, Narayana numbers
- arithmetic_functions: Euler totient, Möbius function, omega functions, perfect numbers
- iterative_sequences: Collatz, Kaprekar, happy numbers, narcissistic, look-and-say, Recamán, Keith
- mathematical_constants: Pi, e, golden ratio, Euler gamma, continued fractions, high precision
- digital_operations: digit sums, palindromes, Harshad numbers, base conversions, automorphic numbers
- partitions: integer partitions, Goldbach conjecture, sum of squares, Waring's problem, additive bases
- egyptian_fractions: Egyptian fractions, unit fractions, harmonic series, Sylvester sequence

All functions are async native for optimal performance in async environments.
"""

# Import all number theory submodules
from . import primes
from . import divisibility
from . import basic_sequences
from . import special_primes
from . import combinatorial_numbers
from . import arithmetic_functions

# Import new modules
from . import iterative_sequences
from . import mathematical_constants
from . import digital_operations
from . import partitions
from . import egyptian_fractions

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
    triangular_number, fibonacci_sequence, catalan_number,
    pentagonal_number, tetrahedral_number
)

# Special primes (commonly referenced)
from .special_primes import (
    is_mersenne_prime, is_fermat_prime, is_twin_prime, 
    wilson_theorem_check, is_carmichael_number, prime_gap,
    lucas_lehmer_test, mersenne_prime_exponents, safe_prime_pairs,
    cousin_primes, sexy_primes
)

# Combinatorial numbers (high-value functions)
from .combinatorial_numbers import (
    catalan_number as catalan_number_full, bell_number, stirling_second,
    stirling_first, narayana_number, bell_triangle, catalan_sequence,
    stirling_second_row, narayana_triangle_row
)

# Arithmetic functions
from .arithmetic_functions import (
    euler_totient, mobius_function, little_omega, big_omega,
    jordan_totient, divisor_power_sum, von_mangoldt_function,
    liouville_function, carmichael_lambda, is_perfect_number,
    is_abundant_number, is_deficient_number
)

# Iterative sequences (NEW)
from .iterative_sequences import (
    collatz_sequence, collatz_stopping_time, collatz_max_value,
    kaprekar_sequence, kaprekar_constant, is_happy_number, happy_numbers,
    is_narcissistic_number, narcissistic_numbers, look_and_say_sequence,
    recaman_sequence, is_keith_number, keith_numbers,
    digital_sum_sequence, digital_product_sequence
)

# Mathematical constants (NEW)
from .mathematical_constants import (
    compute_pi_leibniz, compute_pi_nilakantha, compute_pi_machin, compute_pi_chudnovsky,
    compute_e_series, compute_e_limit, compute_golden_ratio_fibonacci,
    compute_golden_ratio_continued_fraction, compute_euler_gamma_harmonic,
    continued_fraction_pi, continued_fraction_e, continued_fraction_golden_ratio,
    pi_digits, e_digits, approximation_error, convergence_comparison
)

# Digital operations (NEW)
from .digital_operations import (
    digit_sum, digital_root, digit_product, persistent_digital_root,
    digit_reversal, digit_sort, is_palindromic_number, palindromic_numbers,
    next_palindrome, is_harshad_number, harshad_numbers,
    number_to_base, base_to_number, digit_count, digit_frequency,
    is_repdigit, is_automorphic_number, automorphic_numbers
)

# Partitions and additive number theory (NEW)
from .partitions import (
    partition_count, generate_partitions, partitions_into_k_parts,
    distinct_partitions, restricted_partitions, goldbach_conjecture_check,
    goldbach_pairs, weak_goldbach_check, sum_of_two_squares,
    sum_of_four_squares, waring_representation, min_waring_number,
    is_additive_basis, generate_sidon_set
)

# Egyptian fractions (NEW)
from .egyptian_fractions import (
    egyptian_fraction_decomposition, fibonacci_greedy_egyptian,
    unit_fraction_sum, is_unit_fraction, harmonic_number,
    harmonic_number_fraction, harmonic_partial_sum, harmonic_mean,
    sylvester_sequence, sylvester_expansion_of_one,
    egyptian_fraction_properties, two_unit_fraction_representations,
    is_proper_fraction, improper_to_egyptian, shortest_egyptian_fraction
)

# Export all number theory functions for convenient access
__all__ = [
    # Submodules
    'primes', 'divisibility', 'basic_sequences', 'special_primes', 
    'combinatorial_numbers', 'arithmetic_functions', 'iterative_sequences',
    'mathematical_constants', 'digital_operations', 'partitions', 'egyptian_fractions',
    
    # Core prime operations
    'is_prime', 'next_prime', 'nth_prime', 'prime_factors',
    'prime_count', 'is_coprime', 'first_n_primes',
    
    # Core divisibility operations
    'gcd', 'lcm', 'divisors', 'is_divisible', 'is_even', 'is_odd',
    'extended_gcd', 'divisor_count', 'divisor_sum',
    
    # Basic sequences
    'is_perfect_square', 'is_power_of_two', 'fibonacci', 'factorial',
    'triangular_number', 'fibonacci_sequence', 'catalan_number',
    'pentagonal_number', 'tetrahedral_number',
    
    # Special primes
    'is_mersenne_prime', 'is_fermat_prime', 'is_twin_prime',
    'wilson_theorem_check', 'is_carmichael_number', 'prime_gap',
    'lucas_lehmer_test', 'mersenne_prime_exponents', 'safe_prime_pairs',
    'cousin_primes', 'sexy_primes',
    
    # Combinatorial numbers
    'bell_number', 'stirling_second', 'stirling_first', 'narayana_number',
    'bell_triangle', 'catalan_sequence', 'stirling_second_row', 
    'narayana_triangle_row',
    
    # Arithmetic functions
    'euler_totient', 'mobius_function', 'little_omega', 'big_omega',
    'jordan_totient', 'divisor_power_sum', 'von_mangoldt_function',
    'liouville_function', 'carmichael_lambda', 'is_perfect_number',
    'is_abundant_number', 'is_deficient_number',
    
    # Iterative sequences (NEW)
    'collatz_sequence', 'collatz_stopping_time', 'collatz_max_value',
    'kaprekar_sequence', 'kaprekar_constant', 'is_happy_number', 'happy_numbers',
    'is_narcissistic_number', 'narcissistic_numbers', 'look_and_say_sequence',
    'recaman_sequence', 'is_keith_number', 'keith_numbers',
    'digital_sum_sequence', 'digital_product_sequence',
    
    # Mathematical constants (NEW)
    'compute_pi_leibniz', 'compute_pi_nilakantha', 'compute_pi_machin', 'compute_pi_chudnovsky',
    'compute_e_series', 'compute_e_limit', 'compute_golden_ratio_fibonacci',
    'compute_golden_ratio_continued_fraction', 'compute_euler_gamma_harmonic',
    'continued_fraction_pi', 'continued_fraction_e', 'continued_fraction_golden_ratio',
    'pi_digits', 'e_digits', 'approximation_error', 'convergence_comparison',
    
    # Digital operations (NEW)
    'digit_sum', 'digital_root', 'digit_product', 'persistent_digital_root',
    'digit_reversal', 'digit_sort', 'is_palindromic_number', 'palindromic_numbers',
    'next_palindrome', 'is_harshad_number', 'harshad_numbers',
    'number_to_base', 'base_to_number', 'digit_count', 'digit_frequency',
    'is_repdigit', 'is_automorphic_number', 'automorphic_numbers',
    
    # Partitions and additive number theory (NEW)
    'partition_count', 'generate_partitions', 'partitions_into_k_parts',
    'distinct_partitions', 'restricted_partitions', 'goldbach_conjecture_check',
    'goldbach_pairs', 'weak_goldbach_check', 'sum_of_two_squares',
    'sum_of_four_squares', 'waring_representation', 'min_waring_number',
    'is_additive_basis', 'generate_sidon_set',
    
    # Egyptian fractions (NEW)
    'egyptian_fraction_decomposition', 'fibonacci_greedy_egyptian',
    'unit_fraction_sum', 'is_unit_fraction', 'harmonic_number',
    'harmonic_number_fraction', 'harmonic_partial_sum', 'harmonic_mean',
    'sylvester_sequence', 'sylvester_expansion_of_one',
    'egyptian_fraction_properties', 'two_unit_fraction_representations',
    'is_proper_fraction', 'improper_to_egyptian', 'shortest_egyptian_fraction'
]

async def test_number_theory_functions():
    """Test core number theory functions including new modules."""
    print("🔢 Enhanced Number Theory Functions Test")
    print("=" * 45)
    
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
    print(f"  pentagonal_number(5) = {await pentagonal_number(5)}")
    print(f"  tetrahedral_number(4) = {await tetrahedral_number(4)}")
    
    # Test special primes
    print("\nSpecial Primes:")
    print(f"  is_mersenne_prime(31) = {await is_mersenne_prime(31)}")
    print(f"  is_fermat_prime(17) = {await is_fermat_prime(17)}")
    print(f"  is_twin_prime(13) = {await is_twin_prime(13)}")
    print(f"  wilson_theorem_check(7) = {await wilson_theorem_check(7)}")
    print(f"  is_carmichael_number(561) = {await is_carmichael_number(561)}")
    print(f"  prime_gap(7) = {await prime_gap(7)}")
    print(f"  lucas_lehmer_test(5) = {await lucas_lehmer_test(5)}")
    
    # Test combinatorial numbers
    print("\nCombinatorial Numbers:")
    print(f"  bell_number(5) = {await bell_number(5)}")
    print(f"  stirling_second(4, 2) = {await stirling_second(4, 2)}")
    print(f"  stirling_first(4, 2) = {await stirling_first(4, 2)}")
    print(f"  narayana_number(4, 2) = {await narayana_number(4, 2)}")
    print(f"  catalan_sequence(5) = {await catalan_sequence(5)}")
    
    # Test arithmetic functions
    print("\nArithmetic Functions:")
    print(f"  euler_totient(12) = {await euler_totient(12)}")
    print(f"  mobius_function(30) = {await mobius_function(30)}")
    print(f"  little_omega(12) = {await little_omega(12)}")
    print(f"  big_omega(12) = {await big_omega(12)}")
    print(f"  jordan_totient(6, 2) = {await jordan_totient(6, 2)}")
    print(f"  divisor_power_sum(12, 1) = {await divisor_power_sum(12, 1)}")
    print(f"  is_perfect_number(6) = {await is_perfect_number(6)}")
    print(f"  is_abundant_number(12) = {await is_abundant_number(12)}")
    print(f"  carmichael_lambda(12) = {await carmichael_lambda(12)}")
    
    # Test NEW iterative sequences
    print("\nIterative Sequences (NEW):")
    print(f"  collatz_sequence(7) = {await collatz_sequence(7)}")
    print(f"  collatz_stopping_time(7) = {await collatz_stopping_time(7)}")
    print(f"  is_happy_number(7) = {await is_happy_number(7)}")
    print(f"  is_narcissistic_number(153) = {await is_narcissistic_number(153)}")
    print(f"  is_keith_number(14) = {await is_keith_number(14)}")
    print(f"  recaman_sequence(10) = {await recaman_sequence(10)}")
    
    # Test NEW mathematical constants
    print("\nMathematical Constants (NEW):")
    print(f"  compute_pi_leibniz(1000) ≈ {await compute_pi_leibniz(1000):.6f}")
    print(f"  compute_pi_machin(20) ≈ {await compute_pi_machin(20):.10f}")
    print(f"  compute_e_series(15) ≈ {await compute_e_series(15):.8f}")
    print(f"  compute_golden_ratio_fibonacci(20) ≈ {await compute_golden_ratio_fibonacci(20):.10f}")
    print(f"  continued_fraction_pi(10) = {await continued_fraction_pi(10)}")
    
    # Test NEW digital operations
    print("\nDigital Operations (NEW):")
    print(f"  digit_sum(12345) = {await digit_sum(12345)}")
    print(f"  digital_root(12345) = {await digital_root(12345)}")
    print(f"  is_palindromic_number(12321) = {await is_palindromic_number(12321)}")
    print(f"  is_harshad_number(12) = {await is_harshad_number(12)}")
    print(f"  digit_reversal(12345) = {await digit_reversal(12345)}")
    print(f"  is_automorphic_number(25) = {await is_automorphic_number(25)}")
    
    # Test NEW partitions
    print("\nPartitions and Additive Number Theory (NEW):")
    print(f"  partition_count(4) = {await partition_count(4)}")
    print(f"  goldbach_conjecture_check(10) = {await goldbach_conjecture_check(10)}")
    print(f"  sum_of_two_squares(13) = {await sum_of_two_squares(13)}")
    print(f"  distinct_partitions(6) = {await distinct_partitions(6)}")
    
    # Test NEW Egyptian fractions
    print("\nEgyptian Fractions (NEW):")
    print(f"  egyptian_fraction_decomposition(2, 3) = {await egyptian_fraction_decomposition(2, 3)}")
    print(f"  harmonic_number(4) = {await harmonic_number(4):.6f}")
    print(f"  sylvester_sequence(5) = {await sylvester_sequence(5)}")
    print(f"  unit_fraction_sum([2, 3, 6]) = {await unit_fraction_sum([2, 3, 6])}")
    
    print("\n✅ All enhanced number theory functions working!")

async def demo_new_functionality():
    """Demonstrate the new functionality added to the module."""
    print("\n🎯 New Functionality Showcase")
    print("=" * 35)
    
    # Collatz conjecture exploration
    print("Collatz Conjecture Exploration:")
    for n in [3, 7, 12, 27]:
        stopping_time = await collatz_stopping_time(n)
        max_value = await collatz_max_value(n)
        print(f"  Collatz({n}): {stopping_time} steps, max value {max_value}")
    
    # Pi approximation comparison
    print("\nPi Approximation Comparison:")
    methods = [
        ("Leibniz", lambda: compute_pi_leibniz(1000)),
        ("Nilakantha", lambda: compute_pi_nilakantha(100)),
        ("Machin", lambda: compute_pi_machin(20))
    ]
    for name, method in methods:
        pi_approx = await method()
        error = abs(3.141592653589793 - pi_approx)
        print(f"  {name}: {pi_approx:.10f} (error: {error:.2e})")
    
    # Digital number properties
    print("\nDigital Number Properties:")
    test_numbers = [153, 371, 407, 1634]  # Narcissistic numbers
    for num in test_numbers:
        is_narcissistic = await is_narcissistic_number(num)
        digit_sum_val = await digit_sum(num)
        digital_root_val = await digital_root(num)
        print(f"  {num}: narcissistic={is_narcissistic}, digit_sum={digit_sum_val}, digital_root={digital_root_val}")
    
    # Partition exploration
    print("\nInteger Partition Properties:")
    for n in [4, 5, 6, 7]:
        total_partitions = await partition_count(n)
        distinct_parts = await distinct_partitions(n)
        print(f"  n={n}: {total_partitions} total partitions, {distinct_parts} with distinct parts")
    
    # Egyptian fraction examples
    print("\nEgyptian Fraction Examples:")
    test_fractions = [(2, 3), (3, 4), (5, 6), (7, 12)]
    for num, den in test_fractions:
        egyptian = await egyptian_fraction_decomposition(num, den)
        print(f"  {num}/{den} = " + " + ".join(f"1/{d}" for d in egyptian))

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
    sg_pairs = await safe_prime_pairs(50)
    print(f"  Sophie Germain prime pairs ≤ 50: {sg_pairs[:3]}...")  # Show first 3
    
    # Mersenne primes for perfect numbers
    print("\nMersenne Primes (Perfect Number Generation):")
    mersenne_exps = await mersenne_prime_exponents(20)
    print(f"  Known Mersenne exponents ≤ 20: {mersenne_exps}")
    
    # Demonstrate Lucas-Lehmer test
    for p in [5, 7, 11, 13]:
        is_mersenne = await lucas_lehmer_test(p)
        mersenne_num = (2 ** p) - 1
        print(f"  2^{p} - 1 = {mersenne_num} is {'prime' if is_mersenne else 'composite'}")

async def demo_combinatorial_applications():
    """Demonstrate combinatorial applications."""
    print("\n🎲 Combinatorial Applications Demo")
    print("=" * 35)
    
    # Catalan numbers in computer science
    print("Catalan Numbers (Binary Trees, Parentheses):")
    catalan_seq = await catalan_sequence(6)
    for i, cat_n in enumerate(catalan_seq):
        print(f"  C_{i} = {cat_n} (binary trees with {i} internal nodes)")
    
    # Bell numbers for set partitions
    print("\nBell Numbers (Set Partitions):")
    for n in range(6):
        bell_n = await bell_number(n)
        print(f"  B_{n} = {bell_n} (ways to partition set of {n} elements)")
    
    # Stirling numbers for combinatorial analysis
    print("\nStirling Numbers of Second Kind (Subset Partitions):")
    for n in range(1, 6):
        stirling_row = await stirling_second_row(n)
        print(f"  Row {n}: {stirling_row}")
    
    # Narayana numbers for Dyck paths
    print("\nNarayana Numbers (Dyck Paths with Peaks):")
    for n in range(1, 5):
        narayana_row = await narayana_triangle_row(n)
        print(f"  N({n},k): {narayana_row}")

async def demo_arithmetic_function_applications():
    """Demonstrate arithmetic function applications."""
    print("\n🧮 Arithmetic Function Applications Demo")
    print("=" * 45)
    
    # Perfect numbers and their properties
    print("Perfect Numbers and Related:")
    for n in [6, 28, 12, 8]:
        is_perfect = await is_perfect_number(n)
        is_abundant = await is_abundant_number(n)
        is_deficient = await is_deficient_number(n)
        totient = await euler_totient(n)
        mobius = await mobius_function(n)
        
        classification = "perfect" if is_perfect else ("abundant" if is_abundant else "deficient")
        print(f"  {n}: {classification}, φ({n})={totient}, μ({n})={mobius}")
    
    # Multiplicative vs additive functions
    print("\nMultiplicative vs Additive Functions:")
    for n in [12, 18, 30]:
        totient = await euler_totient(n)  # Multiplicative
        omega_little = await little_omega(n)  # Additive
        omega_big = await big_omega(n)  # Additive
        divisor_sum = await divisor_power_sum(n, 1)  # Multiplicative
        
        print(f"  n={n}: φ(n)={totient}, ω(n)={omega_little}, Ω(n)={omega_big}, σ(n)={divisor_sum}")

if __name__ == "__main__":
    import asyncio
    
    async def main():
        await test_number_theory_functions()
        await demo_new_functionality()
        await demo_cryptographic_functions()
        await demo_combinatorial_applications()
        await demo_arithmetic_function_applications()
    
    asyncio.run(main())