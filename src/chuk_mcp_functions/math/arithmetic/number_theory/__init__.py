#!/usr/bin/env python3
# chuk_mcp_functions/math/arithmetic/number_theory/__init__.py
"""
Number Theory Operations Module - Enhanced with Advanced Modules

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
- figurate_numbers: polygonal, centered polygonal, pronic, star, 3D figurate, pyramidal numbers
- modular_arithmetic: Chinese Remainder Theorem, quadratic residues, Legendre symbols, primitive roots, discrete logs
- recursive_sequences: Lucas sequences, Pell numbers, Tribonacci, general linear recurrence solvers

All functions are async native for optimal performance in async environments.
"""

# Import all number theory submodules
from . import primes
from . import divisibility
from . import basic_sequences
from . import special_primes
from . import combinatorial_numbers
from . import arithmetic_functions

# Import existing modules
from . import iterative_sequences
from . import mathematical_constants
from . import digital_operations
from . import partitions
from . import egyptian_fractions

# Import new advanced modules
from . import figurate_numbers
from . import modular_arithmetic
from . import recursive_sequences

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

# Iterative sequences
from .iterative_sequences import (
    collatz_sequence, collatz_stopping_time, collatz_max_value,
    kaprekar_sequence, kaprekar_constant, is_happy_number, happy_numbers,
    is_narcissistic_number, narcissistic_numbers, look_and_say_sequence,
    recaman_sequence, is_keith_number, keith_numbers,
    digital_sum_sequence, digital_product_sequence
)

# Mathematical constants
from .mathematical_constants import (
    compute_pi_leibniz, compute_pi_nilakantha, compute_pi_machin, compute_pi_chudnovsky,
    compute_e_series, compute_e_limit, compute_golden_ratio_fibonacci,
    compute_golden_ratio_continued_fraction, compute_euler_gamma_harmonic,
    continued_fraction_pi, continued_fraction_e, continued_fraction_golden_ratio,
    pi_digits, e_digits, approximation_error, convergence_comparison
)

# Digital operations
from .digital_operations import (
    digit_sum, digital_root, digit_product, persistent_digital_root,
    digit_reversal, digit_sort, is_palindromic_number, palindromic_numbers,
    next_palindrome, is_harshad_number, harshad_numbers,
    number_to_base, base_to_number, digit_count, digit_frequency,
    is_repdigit, is_automorphic_number, automorphic_numbers
)

# Partitions and additive number theory
from .partitions import (
    partition_count, generate_partitions, partitions_into_k_parts,
    distinct_partitions, restricted_partitions, goldbach_conjecture_check,
    goldbach_pairs, weak_goldbach_check, sum_of_two_squares,
    sum_of_four_squares, waring_representation, min_waring_number,
    is_additive_basis, generate_sidon_set
)

# Egyptian fractions
from .egyptian_fractions import (
    egyptian_fraction_decomposition, fibonacci_greedy_egyptian,
    unit_fraction_sum, is_unit_fraction, harmonic_number,
    harmonic_number_fraction, harmonic_partial_sum, harmonic_mean,
    sylvester_sequence, sylvester_expansion_of_one,
    egyptian_fraction_properties, two_unit_fraction_representations,
    is_proper_fraction, improper_to_egyptian, shortest_egyptian_fraction
)

# Figurate numbers (NEW)
from .figurate_numbers import (
    polygonal_number, is_polygonal_number, polygonal_sequence,
    centered_polygonal_number, centered_triangular_number, 
    centered_square_number, centered_hexagonal_number,
    pronic_number, is_pronic_number, pronic_sequence,
    star_number, hexagram_number,
    octahedral_number, dodecahedral_number, icosahedral_number,
    triangular_pyramidal_number, square_pyramidal_number, pentagonal_pyramidal_number,
    gnomon_number
)

# Modular arithmetic (NEW)
from .modular_arithmetic import (
    crt_solve, generalized_crt,
    is_quadratic_residue, quadratic_residues, tonelli_shanks,
    legendre_symbol, jacobi_symbol,
    primitive_root, all_primitive_roots, order_modulo,
    discrete_log_naive, baby_step_giant_step
)

# Recursive sequences (NEW)
from .recursive_sequences import (
    lucas_number, lucas_sequence, lucas_u_v,
    pell_number, pell_lucas_number, pell_sequence,
    tribonacci_number, tetranacci_number, padovan_number, narayana_cow_number,
    solve_linear_recurrence, characteristic_polynomial, binet_formula
)

# Export all number theory functions for convenient access
__all__ = [
    # Submodules
    'primes', 'divisibility', 'basic_sequences', 'special_primes', 
    'combinatorial_numbers', 'arithmetic_functions', 'iterative_sequences',
    'mathematical_constants', 'digital_operations', 'partitions', 'egyptian_fractions',
    'figurate_numbers', 'modular_arithmetic', 'recursive_sequences',
    
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
    
    # Iterative sequences
    'collatz_sequence', 'collatz_stopping_time', 'collatz_max_value',
    'kaprekar_sequence', 'kaprekar_constant', 'is_happy_number', 'happy_numbers',
    'is_narcissistic_number', 'narcissistic_numbers', 'look_and_say_sequence',
    'recaman_sequence', 'is_keith_number', 'keith_numbers',
    'digital_sum_sequence', 'digital_product_sequence',
    
    # Mathematical constants
    'compute_pi_leibniz', 'compute_pi_nilakantha', 'compute_pi_machin', 'compute_pi_chudnovsky',
    'compute_e_series', 'compute_e_limit', 'compute_golden_ratio_fibonacci',
    'compute_golden_ratio_continued_fraction', 'compute_euler_gamma_harmonic',
    'continued_fraction_pi', 'continued_fraction_e', 'continued_fraction_golden_ratio',
    'pi_digits', 'e_digits', 'approximation_error', 'convergence_comparison',
    
    # Digital operations
    'digit_sum', 'digital_root', 'digit_product', 'persistent_digital_root',
    'digit_reversal', 'digit_sort', 'is_palindromic_number', 'palindromic_numbers',
    'next_palindrome', 'is_harshad_number', 'harshad_numbers',
    'number_to_base', 'base_to_number', 'digit_count', 'digit_frequency',
    'is_repdigit', 'is_automorphic_number', 'automorphic_numbers',
    
    # Partitions and additive number theory
    'partition_count', 'generate_partitions', 'partitions_into_k_parts',
    'distinct_partitions', 'restricted_partitions', 'goldbach_conjecture_check',
    'goldbach_pairs', 'weak_goldbach_check', 'sum_of_two_squares',
    'sum_of_four_squares', 'waring_representation', 'min_waring_number',
    'is_additive_basis', 'generate_sidon_set',
    
    # Egyptian fractions
    'egyptian_fraction_decomposition', 'fibonacci_greedy_egyptian',
    'unit_fraction_sum', 'is_unit_fraction', 'harmonic_number',
    'harmonic_number_fraction', 'harmonic_partial_sum', 'harmonic_mean',
    'sylvester_sequence', 'sylvester_expansion_of_one',
    'egyptian_fraction_properties', 'two_unit_fraction_representations',
    'is_proper_fraction', 'improper_to_egyptian', 'shortest_egyptian_fraction',
    
    # Figurate numbers (NEW)
    'polygonal_number', 'is_polygonal_number', 'polygonal_sequence',
    'centered_polygonal_number', 'centered_triangular_number', 
    'centered_square_number', 'centered_hexagonal_number',
    'pronic_number', 'is_pronic_number', 'pronic_sequence',
    'star_number', 'hexagram_number',
    'octahedral_number', 'dodecahedral_number', 'icosahedral_number',
    'triangular_pyramidal_number', 'square_pyramidal_number', 'pentagonal_pyramidal_number',
    'gnomon_number',
    
    # Modular arithmetic (NEW)
    'crt_solve', 'generalized_crt',
    'is_quadratic_residue', 'quadratic_residues', 'tonelli_shanks',
    'legendre_symbol', 'jacobi_symbol',
    'primitive_root', 'all_primitive_roots', 'order_modulo',
    'discrete_log_naive', 'baby_step_giant_step',
    
    # Recursive sequences (NEW)
    'lucas_number', 'lucas_sequence', 'lucas_u_v',
    'pell_number', 'pell_lucas_number', 'pell_sequence',
    'tribonacci_number', 'tetranacci_number', 'padovan_number', 'narayana_cow_number',
    'solve_linear_recurrence', 'characteristic_polynomial', 'binet_formula'
]

async def test_number_theory_functions():
    """Test core number theory functions including new advanced modules."""
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
    
    # Test iterative sequences
    print("\nIterative Sequences:")
    print(f"  collatz_sequence(7) = {await collatz_sequence(7)}")
    print(f"  collatz_stopping_time(7) = {await collatz_stopping_time(7)}")
    print(f"  is_happy_number(7) = {await is_happy_number(7)}")
    print(f"  is_narcissistic_number(153) = {await is_narcissistic_number(153)}")
    print(f"  is_keith_number(14) = {await is_keith_number(14)}")
    print(f"  recaman_sequence(10) = {await recaman_sequence(10)}")
    
    # Test mathematical constants
    print("\nMathematical Constants:")
    print(f"  compute_pi_leibniz(1000) ≈ {await compute_pi_leibniz(1000):.6f}")
    print(f"  compute_pi_machin(20) ≈ {await compute_pi_machin(20):.10f}")
    print(f"  compute_e_series(15) ≈ {await compute_e_series(15):.8f}")
    print(f"  compute_golden_ratio_fibonacci(20) ≈ {await compute_golden_ratio_fibonacci(20):.10f}")
    print(f"  continued_fraction_pi(10) = {await continued_fraction_pi(10)}")
    
    # Test digital operations
    print("\nDigital Operations:")
    print(f"  digit_sum(12345) = {await digit_sum(12345)}")
    print(f"  digital_root(12345) = {await digital_root(12345)}")
    print(f"  is_palindromic_number(12321) = {await is_palindromic_number(12321)}")
    print(f"  is_harshad_number(12) = {await is_harshad_number(12)}")
    print(f"  digit_reversal(12345) = {await digit_reversal(12345)}")
    print(f"  is_automorphic_number(25) = {await is_automorphic_number(25)}")
    
    # Test partitions
    print("\nPartitions and Additive Number Theory:")
    print(f"  partition_count(4) = {await partition_count(4)}")
    print(f"  goldbach_conjecture_check(10) = {await goldbach_conjecture_check(10)}")
    print(f"  sum_of_two_squares(13) = {await sum_of_two_squares(13)}")
    print(f"  distinct_partitions(6) = {await distinct_partitions(6)}")
    
    # Test Egyptian fractions
    print("\nEgyptian Fractions:")
    print(f"  egyptian_fraction_decomposition(2, 3) = {await egyptian_fraction_decomposition(2, 3)}")
    print(f"  harmonic_number(4) = {await harmonic_number(4):.6f}")
    print(f"  sylvester_sequence(5) = {await sylvester_sequence(5)}")
    print(f"  unit_fraction_sum([2, 3, 6]) = {await unit_fraction_sum([2, 3, 6])}")
    
    # Test NEW figurate numbers
    print("\nFigurate Numbers (NEW):")
    print(f"  polygonal_number(5, 3) = {await polygonal_number(5, 3)}  # 5th triangular")
    print(f"  centered_triangular_number(3) = {await centered_triangular_number(3)}")
    print(f"  pronic_number(4) = {await pronic_number(4)}")
    print(f"  star_number(3) = {await star_number(3)}")
    print(f"  octahedral_number(3) = {await octahedral_number(3)}")
    print(f"  triangular_pyramidal_number(4) = {await triangular_pyramidal_number(4)}")
    
    # Test NEW modular arithmetic
    print("\nModular Arithmetic (NEW):")
    crt_result = await crt_solve([2, 3], [3, 5])
    print(f"  crt_solve([2, 3], [3, 5]) = {crt_result}")
    print(f"  is_quadratic_residue(2, 7) = {await is_quadratic_residue(2, 7)}")
    print(f"  legendre_symbol(2, 7) = {await legendre_symbol(2, 7)}")
    print(f"  primitive_root(7) = {await primitive_root(7)}")
    print(f"  discrete_log_naive(3, 2, 7) = {await discrete_log_naive(3, 2, 7)}")
    
    # Test NEW recursive sequences
    print("\nRecursive Sequences (NEW):")
    print(f"  lucas_number(5) = {await lucas_number(5)}")
    print(f"  pell_number(5) = {await pell_number(5)}")
    print(f"  tribonacci_number(8) = {await tribonacci_number(8)}")
    print(f"  padovan_number(7) = {await padovan_number(7)}")
    print(f"  characteristic_polynomial([1, 1]) = {await characteristic_polynomial([1, 1])}")
    
    print("\n✅ All enhanced number theory functions working!")

async def demo_advanced_functionality():
    """Demonstrate the new advanced functionality added to the module."""
    print("\n🎯 Advanced Functionality Showcase")
    print("=" * 40)
    
    # Figurate number exploration
    print("Figurate Number Patterns:")
    for n in range(1, 6):
        triangular = await polygonal_number(n, 3)
        square = await polygonal_number(n, 4)
        pentagonal = await polygonal_number(n, 5)
        hexagonal = await polygonal_number(n, 6)
        print(f"  n={n}: T={triangular}, S={square}, P={pentagonal}, H={hexagonal}")
    
    # Modular arithmetic applications
    print("\nModular Arithmetic Applications:")
    # Chinese Remainder Theorem examples
    systems = [
        ([1, 2], [3, 5]),
        ([2, 3, 2], [3, 5, 7]),
        ([0, 0, 1], [2, 3, 5])
    ]
    for remainders, moduli in systems:
        result = await crt_solve(remainders, moduli)
        print(f"  CRT {remainders} mod {moduli} = {result}")
    
    # Quadratic residue exploration
    print("\nQuadratic Residues mod 11:")
    qr_11 = await quadratic_residues(11)
    print(f"  QR(11) = {qr_11}")
    
    # Primitive root demonstration
    for p in [7, 11, 13]:
        root = await primitive_root(p)
        print(f"  Primitive root of {p}: {root}")
    
    # Recursive sequence relationships
    print("\nRecursive Sequence Relationships:")
    for n in range(1, 8):
        lucas = await lucas_number(n)
        pell = await pell_number(n)
        tribonacci = await tribonacci_number(n)
        print(f"  n={n}: Lucas={lucas}, Pell={pell}, Tribonacci={tribonacci}")
    
    # Linear recurrence solver
    print("\nLinear Recurrence Examples:")
    # Fibonacci with general solver
    fib_coeffs = [1, 1]
    fib_initial = [0, 1]
    print(f"  Fibonacci coefficients: {fib_coeffs}, initial: {fib_initial}")
    print(f"  Characteristic polynomial: {await characteristic_polynomial(fib_coeffs)}")
    
    # 3D figurate number progression
    print("\n3D Figurate Number Progression:")
    for n in range(1, 6):
        octahedral = await octahedral_number(n)
        dodecahedral = await dodecahedral_number(n)
        icosahedral = await icosahedral_number(n)
        print(f"  n={n}: Oct={octahedral}, Dodeca={dodecahedral}, Icosa={icosahedral}")

async def demo_cryptographic_applications():
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
    
    # Discrete logarithm difficulty
    print("\nDiscrete Logarithm Examples:")
    for p in [7, 11, 13]:
        prim_root = await primitive_root(p)
        if prim_root:
            for h in [2, 3]:
                if h < p:
                    log_val = await discrete_log_naive(prim_root, h, p)
                    if log_val is not None:
                        print(f"  log_{prim_root}({h}) ≡ {log_val} (mod {p})")
    
    # Quadratic residue applications
    print("\nQuadratic Residue Applications:")
    for p in [7, 11, 13]:
        qr = await quadratic_residues(p)
        print(f"  Quadratic residues mod {p}: {qr}")
        
        # Legendre symbol examples
        for a in [2, 3]:
            if a < p:
                legendre = await legendre_symbol(a, p)
                print(f"    ({a}/{p}) = {legendre}")

async def demo_mathematical_relationships():
    """Demonstrate mathematical relationships between different number theory areas."""
    print("\n🧮 Mathematical Relationships Demo")
    print("=" * 35)
    
    # Figurate number relationships
    print("Figurate Number Relationships:")
    for n in range(1, 6):
        triangular = await polygonal_number(n, 3)
        square = await polygonal_number(n, 4)
        pronic = await pronic_number(n)
        
        # Relationship: Pronic numbers are twice triangular numbers
        print(f"  T_{n} = {triangular}, S_{n} = {square}, P_{n} = {pronic}")
        print(f"    Verify: 2×T_{n} = {2*triangular} = P_{n} ? {2*triangular == pronic}")
    
    # Lucas sequence relationships
    print("\nLucas Sequence Relationships:")
    for n in range(5):
        lucas = await lucas_number(n)
        u_n, v_n = await lucas_u_v(n, 1, -1)
        print(f"  L_{n} = {lucas}, V_{n}(1,-1) = {v_n}, match: {lucas == v_n}")
    
    # Perfect number and Mersenne prime connection
    print("\nPerfect Numbers and Mersenne Primes:")
    mersenne_primes = [3, 7, 31]  # 2^2-1, 2^3-1, 2^5-1
    for mp in mersenne_primes:
        if await is_prime(mp):
            # Find the exponent
            exp = 2
            while (2**exp - 1) != mp:
                exp += 1
            perfect = (2**(exp-1)) * mp
            is_perfect = await is_perfect_number(perfect)
            print(f"  Mersenne prime 2^{exp}-1 = {mp} → Perfect number {perfect}: {is_perfect}")

if __name__ == "__main__":
    import asyncio
    
    async def main():
        await test_number_theory_functions()
        await demo_advanced_functionality()
        await demo_cryptographic_applications()
        await demo_mathematical_relationships()
    
    asyncio.run(main())