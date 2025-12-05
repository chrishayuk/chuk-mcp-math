#!/usr/bin/env python3
# chuk_mcp_math/number_theory/__init__.py
# ruff: noqa: F401
"""
Number Theory Operations Module - Comprehensive Mathematical Library

Functions for working with integer properties, prime numbers, divisibility, and special numbers.
Essential for cryptography, algorithms, mathematical analysis, and educational applications.

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
- diophantine_equations: linear, Pell's equation, Pythagorean triples, Frobenius numbers, postage stamp problem
- advanced_prime_patterns: cousin primes, sexy primes, prime constellations, distribution analysis, gap records
- special_number_categories: amicable numbers, vampire numbers, Keith numbers, taxi numbers, digital properties
- continued_fractions: CF expansions, convergents, periodic CFs, Pell equation solutions, approximation theory
- farey_sequences: Farey sequences, Ford circles, mediants, Stern-Brocot tree, rational approximation

All functions are async native for optimal performance in async environments.
Total: 340+ mathematical functions across 18 specialized modules.
"""

# Import all number theory submodules (existing)
import math
from . import primes
from . import divisibility
from . import basic_sequences
from . import special_primes
from . import combinatorial_numbers
from . import arithmetic_functions
from . import iterative_sequences
from . import mathematical_constants
from . import digital_operations
from . import partitions
from . import egyptian_fractions
from . import figurate_numbers
from . import modular_arithmetic
from . import recursive_sequences

# Import new advanced modules
from . import diophantine_equations
from . import advanced_prime_patterns
from . import special_number_categories
from . import continued_fractions
from . import farey_sequences

# Core prime operations (most commonly used)
from .primes import (
    is_prime,
    next_prime,
    nth_prime,
    prime_factors,
    prime_count,
    is_coprime,
    first_n_primes,
)

# Core divisibility operations
from .divisibility import (
    gcd,
    lcm,
    divisors,
    is_divisible,
    is_even,
    is_odd,
    extended_gcd,
    divisor_count,
    divisor_sum,
)

# Basic sequences (commonly used)
from .basic_sequences import (
    is_perfect_square,
    is_power_of_two,
    fibonacci,
    factorial,
    triangular_number,
    fibonacci_sequence,
    catalan_number,
    pentagonal_number,
    tetrahedral_number,
)

# Special primes (commonly referenced)
from .special_primes import (
    is_mersenne_prime,
    is_fermat_prime,
    is_twin_prime,
    wilson_theorem_check,
    is_carmichael_number,
    prime_gap,
    lucas_lehmer_test,
    mersenne_prime_exponents,
    safe_prime_pairs,
    cousin_primes,
    sexy_primes,
)

# Combinatorial numbers (high-value functions)
from .combinatorial_numbers import (
    catalan_number as catalan_number_full,
    bell_number,
    stirling_second,
    stirling_first,
    narayana_number,
    bell_triangle,
    catalan_sequence,
    stirling_second_row,
    narayana_triangle_row,
)

# Arithmetic functions
from .arithmetic_functions import (
    euler_totient,
    mobius_function,
    little_omega,
    big_omega,
    jordan_totient,
    divisor_power_sum,
    von_mangoldt_function,
    liouville_function,
    carmichael_lambda,
    is_perfect_number,
    is_abundant_number,
    is_deficient_number,
)

# Iterative sequences
from .iterative_sequences import (
    collatz_sequence,
    collatz_stopping_time,
    collatz_max_value,
    kaprekar_sequence,
    kaprekar_constant,
    is_happy_number,
    happy_numbers,
    is_narcissistic_number,
    narcissistic_numbers,
    look_and_say_sequence,
    recaman_sequence,
    is_keith_number,
    keith_numbers,
    digital_sum_sequence,
    digital_product_sequence,
)

# Mathematical constants
from .mathematical_constants import (
    compute_pi_leibniz,
    compute_pi_nilakantha,
    compute_pi_machin,
    compute_pi_chudnovsky,
    compute_e_series,
    compute_e_limit,
    compute_golden_ratio_fibonacci,
    compute_golden_ratio_continued_fraction,
    compute_euler_gamma_harmonic,
    continued_fraction_pi,
    continued_fraction_e,
    continued_fraction_golden_ratio,
    pi_digits,
    e_digits,
    approximation_error,
    convergence_comparison,
)

# Digital operations
from .digital_operations import (
    digit_sum,
    digital_root,
    digit_product,
    persistent_digital_root,
    digit_reversal,
    digit_sort,
    is_palindromic_number,
    palindromic_numbers,
    next_palindrome,
    is_harshad_number,
    harshad_numbers,
    number_to_base,
    base_to_number,
    digit_count,
    digit_frequency,
    is_repdigit,
    is_automorphic_number,
    automorphic_numbers,
)

# Partitions and additive number theory
from .partitions import (
    partition_count,
    generate_partitions,
    partitions_into_k_parts,
    distinct_partitions,
    restricted_partitions,
    goldbach_conjecture_check,
    goldbach_pairs,
    weak_goldbach_check,
    sum_of_two_squares,
    sum_of_four_squares,
    waring_representation,
    min_waring_number,
    is_additive_basis,
    generate_sidon_set,
)

# Egyptian fractions
from .egyptian_fractions import (
    egyptian_fraction_decomposition,
    fibonacci_greedy_egyptian,
    unit_fraction_sum,
    is_unit_fraction,
    harmonic_number,
    harmonic_number_fraction,
    harmonic_partial_sum,
    harmonic_mean,
    sylvester_sequence,
    sylvester_expansion_of_one,
    egyptian_fraction_properties,
    two_unit_fraction_representations,
    is_proper_fraction,
    improper_to_egyptian,
    shortest_egyptian_fraction,
)

# Figurate numbers
from .figurate_numbers import (
    polygonal_number,
    is_polygonal_number,
    polygonal_sequence,
    centered_polygonal_number,
    centered_triangular_number,
    centered_square_number,
    centered_hexagonal_number,
    pronic_number,
    is_pronic_number,
    pronic_sequence,
    star_number,
    hexagram_number,
    octahedral_number,
    dodecahedral_number,
    icosahedral_number,
    triangular_pyramidal_number,
    square_pyramidal_number,
    pentagonal_pyramidal_number,
    gnomon_number,
)

# Modular arithmetic
from .modular_arithmetic import (
    crt_solve,
    generalized_crt,
    is_quadratic_residue,
    quadratic_residues,
    tonelli_shanks,
    legendre_symbol,
    jacobi_symbol,
    primitive_root,
    all_primitive_roots,
    order_modulo,
    discrete_log_naive,
    baby_step_giant_step,
)

# Recursive sequences
from .recursive_sequences import (
    lucas_number,
    lucas_sequence,
    lucas_u_v,
    pell_number,
    pell_lucas_number,
    pell_sequence,
    tribonacci_number,
    tetranacci_number,
    padovan_number,
    narayana_cow_number,
    solve_linear_recurrence,
    characteristic_polynomial,
    binet_formula,
)

# Diophantine equations (NEW)
from .diophantine_equations import (
    solve_linear_diophantine,
    count_solutions_diophantine,
    parametric_solutions_diophantine,
    solve_pell_equation,
    pell_solutions_generator,
    solve_negative_pell_equation,
    pythagorean_triples,
    sum_of_two_squares_all,
    solve_quadratic_diophantine,
    frobenius_number,
    postage_stamp_problem,
    diophantine_analysis,
)

# Advanced prime patterns (NEW)
from .advanced_prime_patterns import (
    cousin_primes as cousin_primes_advanced,
    sexy_primes as sexy_primes_advanced,
    prime_triplets,
    prime_quadruplets,
    prime_constellations,
    is_admissible_pattern,
    prime_counting_function,
    prime_number_theorem_error,
    prime_gaps_analysis,
    bertrand_postulate_verify,
    twin_prime_conjecture_data,
    prime_gap_records,
    prime_density_analysis,
    ulam_spiral_analysis,
)

# Special number categories (NEW)
from .special_number_categories import (
    find_amicable_pairs,
    is_amicable_number,
    find_social_numbers,
    aliquot_sequence_analysis,
    kaprekar_numbers,
    is_kaprekar_number,
    vampire_numbers,
    is_vampire_number,
    armstrong_numbers,
    dudeney_numbers,
    pluperfect_numbers,
    taxi_numbers,
    keith_numbers as keith_numbers_advanced,
    is_keith_number as is_keith_number_advanced,
    magic_constants,
    sum_digit_powers,
    digital_persistence,
)

# Continued fractions (NEW)
from .continued_fractions import (
    continued_fraction_expansion,
    cf_to_rational,
    rational_to_cf,
    convergents_sequence,
    best_rational_approximation,
    convergent_properties,
    sqrt_cf_expansion,
    periodic_continued_fractions,
    cf_solve_pell,
    e_continued_fraction,
    golden_ratio_cf,
    pi_cf_algorithms,
    calendar_approximations,
    cf_convergence_analysis,
)

# Farey sequences (NEW)
from .farey_sequences import (
    farey_sequence,
    farey_sequence_length,
    farey_neighbors,
    mediant,
    stern_brocot_tree,
    farey_mediant_path,
    ford_circles,
    ford_circle_properties,
    circle_tangency,
    farey_sequence_properties,
    density_analysis,
    gap_analysis,
    best_approximation_farey,
    farey_fraction_between,
    farey_sum,
    calkin_wilf_tree,
    riemann_hypothesis_connection,
)

# Export all number theory functions for convenient access
__all__ = [
    # Submodules
    "primes",
    "divisibility",
    "basic_sequences",
    "special_primes",
    "combinatorial_numbers",
    "arithmetic_functions",
    "iterative_sequences",
    "mathematical_constants",
    "digital_operations",
    "partitions",
    "egyptian_fractions",
    "figurate_numbers",
    "modular_arithmetic",
    "recursive_sequences",
    "diophantine_equations",
    "advanced_prime_patterns",
    "special_number_categories",
    "continued_fractions",
    "farey_sequences",
    # Core prime operations
    "is_prime",
    "next_prime",
    "nth_prime",
    "prime_factors",
    "prime_count",
    "is_coprime",
    "first_n_primes",
    # Core divisibility operations
    "gcd",
    "lcm",
    "divisors",
    "is_divisible",
    "is_even",
    "is_odd",
    "extended_gcd",
    "divisor_count",
    "divisor_sum",
    # Basic sequences
    "is_perfect_square",
    "is_power_of_two",
    "fibonacci",
    "factorial",
    "triangular_number",
    "fibonacci_sequence",
    "catalan_number",
    "pentagonal_number",
    "tetrahedral_number",
    # Special primes
    "is_mersenne_prime",
    "is_fermat_prime",
    "is_twin_prime",
    "wilson_theorem_check",
    "is_carmichael_number",
    "prime_gap",
    "lucas_lehmer_test",
    "mersenne_prime_exponents",
    "safe_prime_pairs",
    "cousin_primes",
    "sexy_primes",
    # Combinatorial numbers
    "bell_number",
    "stirling_second",
    "stirling_first",
    "narayana_number",
    "bell_triangle",
    "catalan_sequence",
    "stirling_second_row",
    "narayana_triangle_row",
    # Arithmetic functions
    "euler_totient",
    "mobius_function",
    "little_omega",
    "big_omega",
    "jordan_totient",
    "divisor_power_sum",
    "von_mangoldt_function",
    "liouville_function",
    "carmichael_lambda",
    "is_perfect_number",
    "is_abundant_number",
    "is_deficient_number",
    # Iterative sequences
    "collatz_sequence",
    "collatz_stopping_time",
    "collatz_max_value",
    "kaprekar_sequence",
    "kaprekar_constant",
    "is_happy_number",
    "happy_numbers",
    "is_narcissistic_number",
    "narcissistic_numbers",
    "look_and_say_sequence",
    "recaman_sequence",
    "is_keith_number",
    "keith_numbers",
    "digital_sum_sequence",
    "digital_product_sequence",
    # Mathematical constants
    "compute_pi_leibniz",
    "compute_pi_nilakantha",
    "compute_pi_machin",
    "compute_pi_chudnovsky",
    "compute_e_series",
    "compute_e_limit",
    "compute_golden_ratio_fibonacci",
    "compute_golden_ratio_continued_fraction",
    "compute_euler_gamma_harmonic",
    "continued_fraction_pi",
    "continued_fraction_e",
    "continued_fraction_golden_ratio",
    "pi_digits",
    "e_digits",
    "approximation_error",
    "convergence_comparison",
    # Digital operations
    "digit_sum",
    "digital_root",
    "digit_product",
    "persistent_digital_root",
    "digit_reversal",
    "digit_sort",
    "is_palindromic_number",
    "palindromic_numbers",
    "next_palindrome",
    "is_harshad_number",
    "harshad_numbers",
    "number_to_base",
    "base_to_number",
    "digit_count",
    "digit_frequency",
    "is_repdigit",
    "is_automorphic_number",
    "automorphic_numbers",
    # Partitions and additive number theory
    "partition_count",
    "generate_partitions",
    "partitions_into_k_parts",
    "distinct_partitions",
    "restricted_partitions",
    "goldbach_conjecture_check",
    "goldbach_pairs",
    "weak_goldbach_check",
    "sum_of_two_squares",
    "sum_of_four_squares",
    "waring_representation",
    "min_waring_number",
    "is_additive_basis",
    "generate_sidon_set",
    # Egyptian fractions
    "egyptian_fraction_decomposition",
    "fibonacci_greedy_egyptian",
    "unit_fraction_sum",
    "is_unit_fraction",
    "harmonic_number",
    "harmonic_number_fraction",
    "harmonic_partial_sum",
    "harmonic_mean",
    "sylvester_sequence",
    "sylvester_expansion_of_one",
    "egyptian_fraction_properties",
    "two_unit_fraction_representations",
    "is_proper_fraction",
    "improper_to_egyptian",
    "shortest_egyptian_fraction",
    # Figurate numbers
    "polygonal_number",
    "is_polygonal_number",
    "polygonal_sequence",
    "centered_polygonal_number",
    "centered_triangular_number",
    "centered_square_number",
    "centered_hexagonal_number",
    "pronic_number",
    "is_pronic_number",
    "pronic_sequence",
    "star_number",
    "hexagram_number",
    "octahedral_number",
    "dodecahedral_number",
    "icosahedral_number",
    "triangular_pyramidal_number",
    "square_pyramidal_number",
    "pentagonal_pyramidal_number",
    "gnomon_number",
    # Modular arithmetic
    "crt_solve",
    "generalized_crt",
    "is_quadratic_residue",
    "quadratic_residues",
    "tonelli_shanks",
    "legendre_symbol",
    "jacobi_symbol",
    "primitive_root",
    "all_primitive_roots",
    "order_modulo",
    "discrete_log_naive",
    "baby_step_giant_step",
    # Recursive sequences
    "lucas_number",
    "lucas_sequence",
    "lucas_u_v",
    "pell_number",
    "pell_lucas_number",
    "pell_sequence",
    "tribonacci_number",
    "tetranacci_number",
    "padovan_number",
    "narayana_cow_number",
    "solve_linear_recurrence",
    "characteristic_polynomial",
    "binet_formula",
    # Diophantine equations (NEW)
    "solve_linear_diophantine",
    "count_solutions_diophantine",
    "parametric_solutions_diophantine",
    "solve_pell_equation",
    "pell_solutions_generator",
    "solve_negative_pell_equation",
    "pythagorean_triples",
    "sum_of_two_squares_all",
    "solve_quadratic_diophantine",
    "frobenius_number",
    "postage_stamp_problem",
    "diophantine_analysis",
    # Advanced prime patterns (NEW)
    "cousin_primes_advanced",
    "sexy_primes_advanced",
    "prime_triplets",
    "prime_quadruplets",
    "prime_constellations",
    "is_admissible_pattern",
    "prime_counting_function",
    "prime_number_theorem_error",
    "prime_gaps_analysis",
    "bertrand_postulate_verify",
    "twin_prime_conjecture_data",
    "prime_gap_records",
    "prime_density_analysis",
    "ulam_spiral_analysis",
    # Special number categories (NEW)
    "find_amicable_pairs",
    "is_amicable_number",
    "find_social_numbers",
    "aliquot_sequence_analysis",
    "kaprekar_numbers",
    "is_kaprekar_number",
    "vampire_numbers",
    "is_vampire_number",
    "armstrong_numbers",
    "dudeney_numbers",
    "pluperfect_numbers",
    "taxi_numbers",
    "keith_numbers_advanced",
    "is_keith_number_advanced",
    "magic_constants",
    "sum_digit_powers",
    "digital_persistence",
    # Continued fractions (NEW)
    "continued_fraction_expansion",
    "cf_to_rational",
    "rational_to_cf",
    "convergents_sequence",
    "best_rational_approximation",
    "convergent_properties",
    "sqrt_cf_expansion",
    "periodic_continued_fractions",
    "cf_solve_pell",
    "e_continued_fraction",
    "golden_ratio_cf",
    "pi_cf_algorithms",
    "calendar_approximations",
    "cf_convergence_analysis",
    # Farey sequences (NEW)
    "farey_sequence",
    "farey_sequence_length",
    "farey_neighbors",
    "mediant",
    "stern_brocot_tree",
    "farey_mediant_path",
    "ford_circles",
    "ford_circle_properties",
    "circle_tangency",
    "farey_sequence_properties",
    "density_analysis",
    "gap_analysis",
    "best_approximation_farey",
    "farey_fraction_between",
    "farey_sum",
    "calkin_wilf_tree",
    "riemann_hypothesis_connection",
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
    print(f"    Verification: 30×{x} + 18×{y} = {30 * x + 18 * y}")

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

    # Test Farey sequences (NEW)
    print("\nFarey Sequences (NEW):")
    farey_seq = await farey_sequence(5)
    print(f"  farey_sequence(5) = {farey_seq}")

    farey_len = await farey_sequence_length(5)
    print(f"  farey_sequence_length(5) = {farey_len['length']}")

    mediant_result = await mediant(1, 3, 1, 2)
    print(f"  mediant(1/3, 1/2) = {mediant_result}")

    ford_result = await ford_circles(4)
    print(f"  ford_circles(4) = {ford_result['count']} circles")

    best_approx_farey = await best_approximation_farey(0.618, 10)
    print(
        f"  best_approximation_farey(0.618, 10) = {best_approx_farey['best_approximation']}"
    )

    # Test Diophantine equations
    print("\nDiophantine Equations (NEW):")
    dioph_result = await solve_linear_diophantine(3, 5, 1)
    print(f"  solve_linear_diophantine(3, 5, 1) = {dioph_result}")

    pell_result = await solve_pell_equation(2)
    print(f"  solve_pell_equation(2) = {pell_result}")

    pyth_triples = await pythagorean_triples(25, primitive_only=True)
    print(f"  pythagorean_triples(25, primitive) = {pyth_triples}")

    # Test advanced prime patterns
    print("\nAdvanced Prime Patterns (NEW):")
    cousin_result = await cousin_primes_advanced(50)
    print(f"  cousin_primes(50) = {cousin_result}")

    counting_result = await prime_counting_function(100)
    print(f"  prime_counting_function(100) = {counting_result}")

    gaps_result = await prime_gaps_analysis(10, 50)
    print(f"  prime_gaps_analysis(10, 50) = {gaps_result}")

    # Test special number categories
    print("\nSpecial Number Categories (NEW):")
    amicable_result = await find_amicable_pairs(1500)
    print(f"  find_amicable_pairs(1500) = {amicable_result}")

    kaprekar_result = await kaprekar_numbers(100)
    print(f"  kaprekar_numbers(100) = {kaprekar_result}")

    armstrong_result = await armstrong_numbers(1000)
    print(f"  armstrong_numbers(1000) = {armstrong_result}")

    # Test continued fractions
    print("\nContinued Fractions (NEW):")
    cf_result = await continued_fraction_expansion(3.14159, 8)
    print(f"  continued_fraction_expansion(π, 8) = {cf_result}")

    best_approx = await best_rational_approximation(3.14159, 1000)
    print(f"  best_rational_approximation(π, 1000) = {best_approx}")

    sqrt_cf = await sqrt_cf_expansion(2)
    print(f"  sqrt_cf_expansion(2) = {sqrt_cf}")

    print("\n✅ All enhanced number theory functions working!")


async def demo_comprehensive_functionality():
    """Demonstrate the comprehensive functionality of the enhanced module."""
    print("\n🎯 Comprehensive Functionality Showcase")
    print("=" * 45)

    # Cross-module mathematical relationships
    print("Cross-Module Mathematical Relationships:")

    # Demonstrate how different modules work together
    print("  Perfect Numbers ↔ Mersenne Primes:")
    for exp in [2, 3, 5, 7]:
        mersenne = 2**exp - 1
        if await is_prime(mersenne):
            perfect = (2 ** (exp - 1)) * mersenne
            is_perfect = await is_perfect_number(perfect)
            print(
                f"    2^{exp}-1 = {mersenne} (prime) → Perfect: {perfect} ({is_perfect})"
            )

    print("\n  Farey Sequences ↔ Continued Fractions:")
    for target in [0.618, 0.414, 0.707]:  # φ-1, √2-1, √2/2
        farey_approx = await best_approximation_farey(target, 20)
        cf_approx = await best_rational_approximation(target, 20)
        print(
            f"    {target}: Farey = {farey_approx['best_approximation']}, CF = {cf_approx['best_approximation']}"
        )

    print("\n  Continued Fractions ↔ Pell Equations:")
    for n in [2, 3, 5]:
        pell_cf = await cf_solve_pell(n)
        pell_direct = await solve_pell_equation(n)
        print(f"    √{n}: CF method = {pell_cf.get('fundamental_solution', 'N/A')}")
        print(f"         Direct method = {pell_direct.get('fundamental', 'N/A')}")

    print("\n  Diophantine ↔ Number Properties:")
    for limit in [25, 50]:
        triples = await pythagorean_triples(limit, primitive_only=True)
        print(f"    Primitive Pythagorean triples ≤ {limit}: {len(triples)} found")
        for a, b, c in triples[:3]:  # Show first 3
            gcd_abc = await gcd(await gcd(a, b), c)
            print(f"      ({a}, {b}, {c}): gcd = {gcd_abc}, sum = {a + b + c}")

    # Performance and scale demonstration
    print("\n  Scale and Performance:")
    import time

    start_time = time.time()
    large_prime = await next_prime(10000)
    prime_time = time.time() - start_time
    print(f"    next_prime(10000) = {large_prime} (computed in {prime_time:.4f}s)")

    start_time = time.time()
    large_fibonacci = await fibonacci(100)
    fib_time = time.time() - start_time
    print(
        f"    fibonacci(100) = {str(large_fibonacci)[:50]}... (computed in {fib_time:.4f}s)"
    )

    start_time = time.time()
    partitions = await partition_count(50)
    partition_time = time.time() - start_time
    print(f"    partition_count(50) = {partitions} (computed in {partition_time:.4f}s)")


async def demo_educational_applications():
    """Demonstrate educational applications across the library."""
    print("\n🎓 Educational Applications Demo")
    print("=" * 35)

    # Number theory exploration for students
    print("Student Number Theory Exploration:")

    # Explore a number comprehensively
    n = 60
    print(f"  Comprehensive analysis of {n}:")

    # Basic properties
    factors = await prime_factors(n)
    divisors_list = await divisors(n)
    totient = await euler_totient(n)

    print(f"    Prime factorization: {' × '.join(map(str, factors))}")
    print(f"    All divisors: {divisors_list}")
    print(f"    Euler's totient φ({n}) = {totient}")

    # Classifications
    is_abundant = await is_abundant_number(n)
    is_harshad = await is_harshad_number(n)
    digit_sum_val = await digit_sum(n)

    print(f"    Abundant number: {is_abundant}")
    print(f"    Harshad number: {is_harshad} (digit sum: {digit_sum_val})")

    # Related sequences
    triangular_pos = None
    for i in range(1, 20):
        if await triangular_number(i) == n:
            triangular_pos = i
            break

    if triangular_pos:
        print(f"    {n} is the {triangular_pos}th triangular number")

    # Applications in different bases
    base_2 = await number_to_base(n, 2)
    base_16 = await number_to_base(n, 16)
    print(f"    In binary: {base_2}, in hex: {base_16}")

    # Farey sequence analysis
    print("\n  Farey Sequence Analysis for fractions with denominator ≤ 8:")
    farey_8 = await farey_sequence(8)
    print(f"    F_8 contains {len(farey_8)} fractions")

    # Find n as a fraction in Farey sequence
    if [60, 1] in farey_8:
        print(f"    {n}/1 appears in F_8")
    else:
        print(f"    {n}/1 too large for F_8")


async def demo_research_applications():
    """Demonstrate research-level applications."""
    print("\n🔬 Research Applications Demo")
    print("=" * 30)

    print("Prime Distribution Research:")

    # Analyze prime gaps in different ranges
    ranges = [(100, 200), (1000, 1100), (10000, 10100)]
    for start, end in ranges:
        gaps = await prime_gaps_analysis(start, end)
        print(
            f"  Range [{start}, {end}]: avg gap = {gaps['avg_gap']}, max gap = {gaps['max_gap']}"
        )

    print("\nFarey Sequence Research:")

    # Study Farey sequence density growth
    density_data = await density_analysis(15)
    print(f"  Farey densities F_1 to F_15: {density_data['densities']}")
    print(f"  Asymptotic constant estimate: {density_data['estimated_constant']}")
    print(f"  Theoretical constant (3/π²): {density_data['theoretical_constant']}")

    # Ford circles analysis
    ford_props = await ford_circle_properties(10)
    print(
        f"  Ford circles F_10: {ford_props['total_circles']} circles, all tangent: {ford_props['all_tangent']}"
    )

    print("\nContinued Fraction Convergence:")

    # Study convergence rates for different irrationals
    constants = [
        (math.pi, "π"),
        (math.e, "e"),
        ((1 + math.sqrt(5)) / 2, "φ"),
        (math.sqrt(2), "√2"),
    ]

    for value, name in constants:
        cf_analysis = await cf_convergence_analysis(value, 8)
        print(
            f"  {name}: Hurwitz estimate = {cf_analysis.get('hurwitz_estimate', 'N/A')}"
        )
        print(f"      Type: {cf_analysis.get('diophantine_type', 'unknown')}")

    print("\nAdditive Number Theory:")

    # Goldbach conjecture verification
    even_numbers = [10, 50, 100, 500]
    for n in even_numbers:
        goldbach = await goldbach_conjecture_check(n)
        pairs = await goldbach_pairs(n)
        print(f"  {n} = sum of 2 primes: {goldbach} ({len(pairs)} representations)")


if __name__ == "__main__":
    import asyncio

    async def main():
        await test_number_theory_functions()
        await demo_comprehensive_functionality()
        await demo_educational_applications()
        await demo_research_applications()

    asyncio.run(main())
