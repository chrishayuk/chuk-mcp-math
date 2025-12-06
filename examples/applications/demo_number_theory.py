#!/usr/bin/env python3
"""
Comprehensive Number Theory Library Demonstration Script

A complete showcase of the chuk_mcp_math number theory capabilities.
This script demonstrates real-world applications, mathematical relationships,
advanced cryptographic functions, and cutting-edge number theory research
in an educational format.

Features:
- 340+ functions across 18 specialized modules
- Diophantine equations and Pell solutions
- Advanced prime patterns and distribution analysis
- Special number categories (amicable, vampire, etc.)
- Continued fractions and approximation theory
- Farey sequences and Ford circles (NEW)
- Cross-module mathematical relationships
- Research-level demonstrations

Run with: python number_theory_demo.py
"""

import asyncio
import time
import math

# Import the enhanced number theory library
from chuk_mcp_math import number_theory


async def print_header(title: str, char: str = "="):
    """Print a formatted section header."""
    print(f"\n{char * 70}")
    print(f"🧮 {title}")
    print(f"{char * 70}")


async def print_subheader(title: str):
    """Print a formatted subsection header."""
    print(f"\n📊 {title}")
    print("-" * 50)


async def demo_prime_numbers():
    """Demonstrate prime number operations and their applications."""
    await print_header("Prime Numbers & Applications")

    await print_subheader("Basic Prime Operations")

    # Test various numbers for primality
    test_numbers = [2, 17, 97, 100, 561, 1009, 1013]
    for n in test_numbers:
        is_prime = await number_theory.is_prime(n)
        print(f"  is_prime({n:4d}) = {str(is_prime):5s}")

    # Find next primes after given numbers
    print("\n  Finding next primes:")
    for n in [10, 50, 100, 1000]:
        next_p = await number_theory.next_prime(n)
        print(f"  next_prime({n:4d}) = {next_p}")

    # Get the first few primes
    print(f"\n  First 15 primes: {await number_theory.first_n_primes(15)}")

    await print_subheader("Prime Factorization")

    # Factor some interesting numbers
    factorization_examples = [60, 100, 256, 1001, 2310]
    for n in factorization_examples:
        factors = await number_theory.prime_factors(n)
        print(f"  {n:4d} = {' × '.join(map(str, factors))}")

    await print_subheader("Special Prime Properties")

    # Mersenne primes (2^p - 1 where p is prime)
    print("  Mersenne Prime Candidates:")
    mersenne_candidates = [3, 7, 31, 127, 8191]
    for candidate in mersenne_candidates:
        is_mersenne = await number_theory.is_mersenne_prime(candidate)
        # Find the exponent
        exp = 2
        while (2**exp - 1) < candidate:
            exp += 1
        if (2**exp - 1) == candidate:
            print(f"    2^{exp} - 1 = {candidate:5d} is Mersenne prime: {is_mersenne}")

    # Twin primes using new advanced module
    print("\n  Twin Prime Pairs:")
    twin_data = await number_theory.twin_prime_conjecture_data(100, return_all_pairs=True)
    if "twin_prime_pairs" in twin_data:
        twin_pairs = twin_data["twin_prime_pairs"][:8]  # Show first 8 pairs
        for p, q in twin_pairs:
            print(f"    ({p}, {q}) are twin primes")

    await print_subheader("Advanced Prime Patterns (NEW)")

    # Cousin primes (differ by 4)
    cousin_pairs = await number_theory.cousin_primes_advanced(50)
    print(f"  Cousin primes (differ by 4) up to 50: {cousin_pairs}")

    # Sexy primes (differ by 6)
    sexy_pairs = await number_theory.sexy_primes_advanced(50)
    print(f"  Sexy primes (differ by 6) up to 50: {sexy_pairs}")

    # Prime triplets
    triplets = await number_theory.prime_triplets(50)
    for triplet_type in triplets:
        if triplet_type["triplets"]:
            print(f"  {triplet_type['type']} triplets: {triplet_type['triplets']}")


async def demo_cryptographic_applications():
    """Demonstrate cryptographic applications of number theory."""
    await print_header("Cryptographic Applications")

    await print_subheader("RSA-Style Operations")

    # Choose two primes for RSA demonstration
    p = await number_theory.next_prime(100)
    q = await number_theory.next_prime(200)
    n = p * q
    phi_n = (p - 1) * (q - 1)

    print("  RSA Key Generation Example:")
    print(f"    Prime p = {p}")
    print(f"    Prime q = {q}")
    print(f"    n = p × q = {n}")
    print(f"    φ(n) = (p-1)(q-1) = {phi_n}")

    # Public exponent (commonly 65537 in practice)
    e = 65537
    gcd_result = await number_theory.gcd(e, phi_n)
    print(f"    Public exponent e = {e}")
    print(f"    gcd(e, φ(n)) = {gcd_result} (should be 1 for valid key)")

    await print_subheader("Modular Arithmetic & Chinese Remainder Theorem")

    # Chinese Remainder Theorem examples
    crt_examples = [
        ([2, 3], [3, 5]),  # x ≡ 2 (mod 3), x ≡ 3 (mod 5)
        ([1, 4, 6], [3, 5, 7]),  # System of three congruences
        ([0, 0, 1], [2, 3, 5]),  # x ≡ 0 (mod 2), x ≡ 0 (mod 3), x ≡ 1 (mod 5)
    ]

    print("  Chinese Remainder Theorem Solutions:")
    for remainders, moduli in crt_examples:
        try:
            result = await number_theory.crt_solve(remainders, moduli)
            if result is not None:
                # Handle tuple return properly
                if isinstance(result, tuple) and len(result) == 2:
                    solution, combined_modulus = result
                    print(
                        f"    System: x ≡ {remainders} (mod {moduli}) → x = {solution} (mod {combined_modulus})"
                    )

                    # Verify the solution
                    verification = []
                    for r, m in zip(remainders, moduli):
                        verification.append(f"{solution} ≡ {solution % m} (mod {m})")
                    print(f"      Verify: {' and '.join(verification)}")
                else:
                    print(f"    System: x ≡ {remainders} (mod {moduli}) → x = {result}")
            else:
                print(f"    System: x ≡ {remainders} (mod {moduli}) → No solution")
        except Exception as e:
            print(f"    System: x ≡ {remainders} (mod {moduli}) → Error: {e}")

    await print_subheader("Quadratic Residues & Legendre Symbols")

    # Quadratic residues for small primes
    small_primes = [7, 11, 13]
    for p in small_primes:
        qr = await number_theory.quadratic_residues(p)
        print(f"  Quadratic residues mod {p}: {qr}")

        # Show Legendre symbols for some values
        legendre_examples = []
        for a in range(1, min(p, 6)):
            symbol = await number_theory.legendre_symbol(a, p)
            legendre_examples.append(f"({a}/{p})={symbol:2d}")
        print(f"    Legendre symbols: {' '.join(legendre_examples)}")

    await print_subheader("Discrete Logarithms & Primitive Roots")

    # Find primitive roots for small primes
    primes_for_roots = [7, 11, 13]
    for p in primes_for_roots:
        prim_root = await number_theory.primitive_root(p)
        if prim_root:
            print(f"  Primitive root of {p}: {prim_root}")

            # Show some discrete logarithms
            print(f"    Discrete logs base {prim_root} mod {p}:")
            for target in range(1, min(p, 6)):
                try:
                    log_val = await number_theory.discrete_log_naive(prim_root, target, p)
                    if log_val is not None:
                        # Verify: prim_root^log_val ≡ target (mod p)
                        verification = pow(prim_root, log_val, p)
                        print(f"      {prim_root}^{log_val} ≡ {verification} ≡ {target} (mod {p})")
                except Exception:
                    pass


async def demo_diophantine_equations():
    """Demonstrate Diophantine equations and their solutions (NEW)."""
    await print_header("Diophantine Equations (NEW)")

    await print_subheader("Linear Diophantine Equations")

    # Linear equation examples
    linear_examples = [
        (3, 5, 1),  # 3x + 5y = 1
        (6, 9, 7),  # 6x + 9y = 7 (no solution)
        (2, 3, 7),  # 2x + 3y = 7
    ]

    for a, b, c in linear_examples:
        result = await number_theory.solve_linear_diophantine(a, b, c)
        if result["solvable"]:
            print(f"  {a}x + {b}y = {c}")
            print(f"    Solution: {result['general']}")
            print(f"    Particular: x = {result['particular'][0]}, y = {result['particular'][1]}")
        else:
            print(f"  {a}x + {b}y = {c}: No solution ({result.get('reason', 'Unknown')})")

    await print_subheader("Pell's Equation")

    # Pell equation examples x² - ny² = 1
    pell_examples = [2, 3, 5, 7]
    for n in pell_examples:
        result = await number_theory.solve_pell_equation(n)
        if result.get("exists", False):
            x, y = result["fundamental"]
            verification = x * x - n * y * y
            print(f"  x² - {n}y² = 1: fundamental solution = ({x}, {y})")
            print(f"    Verification: {x}² - {n}×{y}² = {verification}")

    # Generate multiple Pell solutions
    print("\n  First 4 solutions to x² - 2y² = 1:")
    pell_solutions = await number_theory.pell_solutions_generator(2, 4)
    for i, (x, y) in enumerate(pell_solutions, 1):
        print(f"    Solution {i}: ({x}, {y})")

    await print_subheader("Pythagorean Triples")

    # Pythagorean triples
    triples = await number_theory.pythagorean_triples(50, primitive_only=True)
    print(f"  Primitive Pythagorean triples (c ≤ 50): {len(triples)} found")
    for a, b, c in triples[:6]:  # Show first 6
        print(f"    ({a}, {b}, {c}): {a}² + {b}² = {a * a} + {b * b} = {c * c} = {c}²")

    await print_subheader("Special Diophantine Problems")

    # Frobenius coin problem
    frobenius_examples = [
        [3, 5],  # Chicken McNugget problem
        [4, 6, 9],  # Three denominations
        [6, 9, 20],  # Larger example
    ]

    for denoms in frobenius_examples:
        frobenius_num = await number_theory.frobenius_number(denoms)
        print(f"  Frobenius number for {denoms}: {frobenius_num}")
        print("    (Largest amount not expressible as non-negative combination)")

    # Postage stamp problem
    postage_examples = [
        (17, [3, 5]),  # Can we make 17 cents?
        (43, [5, 9, 20]),  # Can we make 43 cents?
        (11, [3, 5]),  # Can we make 11 cents?
    ]

    for amount, stamps in postage_examples:
        result = await number_theory.postage_stamp_problem(amount, stamps)
        if result["possible"]:
            breakdown = result.get("denomination_breakdown", {})
            print(f"  Make {amount} with stamps {stamps}: {breakdown}")
        else:
            print(f"  Cannot make {amount} with stamps {stamps}")


async def demo_special_number_categories():
    """Demonstrate special number categories (NEW)."""
    await print_header("Special Number Categories (NEW)")

    await print_subheader("Amicable and Social Numbers")

    # Amicable pairs
    amicable_pairs = await number_theory.find_amicable_pairs(1500)
    print(f"  Amicable pairs up to 1500: {amicable_pairs}")

    # Check specific numbers for amicability
    amicable_check = await number_theory.is_amicable_number(220)
    print(f"  is_amicable_number(220): {amicable_check}")

    # Aliquot sequence analysis
    aliquot_result = await number_theory.aliquot_sequence_analysis(220, 10)
    print(f"  Aliquot sequence from 220: {aliquot_result['sequence']}")
    print(
        f"    Type: {aliquot_result['type']}, cycle length: {aliquot_result.get('cycle_length', 'N/A')}"
    )

    await print_subheader("Recreational Numbers")

    # Kaprekar numbers
    kaprekar_nums = await number_theory.kaprekar_numbers(100)
    print(f"  Kaprekar numbers up to 100: {kaprekar_nums}")

    # Kaprekar number verification
    kaprekar_check = await number_theory.is_kaprekar_number(45)
    print(f"  is_kaprekar_number(45): {kaprekar_check}")

    # Armstrong numbers
    armstrong_nums = await number_theory.armstrong_numbers(1000)
    print(f"  Armstrong numbers up to 1000: {armstrong_nums}")

    # Vampire numbers (computationally intensive, small limit)
    vampire_nums = await number_theory.vampire_numbers(10000)
    print(f"  Vampire numbers up to 10,000: {len(vampire_nums)} found")
    for vamp_data in vampire_nums[:3]:  # Show first 3
        vampire = vamp_data["vampire"]
        fangs = vamp_data["fangs"]
        print(f"    {vampire}: fangs = {fangs}")

    # Keith numbers
    keith_nums = await number_theory.keith_numbers_advanced(100)
    print(f"  Keith numbers up to 100: {keith_nums}")

    # Keith number verification
    keith_check = await number_theory.is_keith_number_advanced(14)
    print(f"  is_keith_number(14): {keith_check}")

    await print_subheader("Taxi Numbers (Hardy-Ramanujan)")

    # Taxi numbers (sums of cubes in multiple ways)
    taxi_nums = await number_theory.taxi_numbers(25000, min_ways=2)
    print(f"  Taxi numbers up to 25,000: {len(taxi_nums)} found")
    for taxi_data in taxi_nums[:3]:  # Show first 3
        number = taxi_data["number"]
        representations = taxi_data["representations"]
        print(f"    {number}: {representations}")
        # Show the actual cube sums
        for a, b in representations:
            print(f"      {a}³ + {b}³ = {a**3} + {b**3} = {number}")


async def demo_continued_fractions():
    """Demonstrate continued fractions and approximation theory (NEW)."""
    await print_header("Continued Fractions & Approximation Theory (NEW)")

    await print_subheader("Basic Continued Fraction Operations")

    # Continued fraction expansions of famous constants
    constants = [
        (math.pi, "π"),
        (math.e, "e"),
        ((1 + math.sqrt(5)) / 2, "φ (golden ratio)"),
        (math.sqrt(2), "√2"),
    ]

    for value, name in constants:
        cf_result = await number_theory.continued_fraction_expansion(value, 8)
        print(f"  {name}: CF = {cf_result['cf']}")
        print(f"    Convergent: {cf_result['convergent'][0]}/{cf_result['convergent'][1]}")
        print(f"    Error: {cf_result['error']:.2e}")

    await print_subheader("Rational Approximations")

    # Best rational approximations
    approximation_examples = [
        (math.pi, 1000, "π"),
        (math.e, 100, "e"),
        (math.sqrt(2), 50, "√2"),
    ]

    for value, max_denom, name in approximation_examples:
        best_approx = await number_theory.best_rational_approximation(value, max_denom)
        fraction = best_approx["best_fraction"]
        error = best_approx["error"]
        print(f"  Best approximation to {name} with denominator ≤ {max_denom}:")
        print(f"    {fraction[0]}/{fraction[1]} (error: {error:.2e})")

    await print_subheader("Periodic Continued Fractions")

    # Square root continued fractions
    sqrt_examples = [2, 3, 5, 7]
    for n in sqrt_examples:
        sqrt_cf = await number_theory.sqrt_cf_expansion(n)
        if not sqrt_cf.get("is_perfect_square", False):
            initial = sqrt_cf["initial"]
            period = sqrt_cf["cf_period"]
            print(f"  √{n} = {initial} + CF with period {period} (length {len(period)})")

    await print_subheader("Continued Fractions and Pell Equations")

    # Solve Pell equations using continued fractions
    pell_cf_examples = [2, 3, 5]
    for n in pell_cf_examples:
        cf_pell = await number_theory.cf_solve_pell(n)
        if cf_pell.get("solution_found", False):
            solution = cf_pell["fundamental_solution"]
            verification = cf_pell["verification"]
            print(f"  x² - {n}y² = 1 via CF: solution = ({solution[0]}, {solution[1]})")
            print(f"    Verification: {verification}")

    await print_subheader("Special Continued Fractions")

    # e continued fraction
    e_cf = await number_theory.e_continued_fraction(10)
    print(f"  e CF pattern: {e_cf['cf']}")
    print(
        f"  Convergent: {e_cf['convergent'][0]}/{e_cf['convergent'][1]} ≈ {e_cf['convergent_value']:.8f}"
    )

    # Golden ratio continued fraction
    golden_cf = await number_theory.golden_ratio_cf(8)
    print(f"  φ CF (all 1s): {golden_cf['cf']}")
    print(
        f"  Convergent: {golden_cf['convergent'][0]}/{golden_cf['convergent'][1]} ≈ {golden_cf['convergent_value']:.8f}"
    )

    # Calendar approximations
    calendar_approx = await number_theory.calendar_approximations(365.24219)
    print("  Calendar approximations for tropical year (365.24219 days):")
    for approx in calendar_approx["approximations"][:3]:  # Show first 3
        fraction = approx["fraction"]
        error = approx["error"]
        interpretation = approx["calendar_interpretation"]
        print(f"    {fraction[0]}/{fraction[1]}: {interpretation} (error: {error:.6f} days)")


async def demo_farey_sequences():
    """Demonstrate Farey sequences and their geometric connections (NEW)."""
    await print_header("Farey Sequences & Ford Circles (NEW)")

    await print_subheader("Basic Farey Sequence Operations")

    # Generate Farey sequences of different orders
    farey_examples = [3, 5, 7]
    for n in farey_examples:
        farey_seq = await number_theory.farey_sequence(n)
        print(f"  F_{n} = {farey_seq}")

        # Show length calculation
        length_data = await number_theory.farey_sequence_length(n)
        print(
            f"    Length: {length_data['length']} (formula: 1 + Σφ(k) = {length_data['formula_result']})"
        )

    await print_subheader("Mediant Operations and Stern-Brocot Tree")

    # Mediant examples
    mediant_examples = [
        (1, 3, 1, 2),  # 1/3 and 1/2
        (2, 5, 3, 7),  # 2/5 and 3/7
        (0, 1, 1, 4),  # 0/1 and 1/4
    ]

    for p1, q1, p2, q2 in mediant_examples:
        mediant_result = await number_theory.mediant(p1, q1, p2, q2)
        val1, val2 = p1 / q1, p2 / q2
        mediant_val = mediant_result[0] / mediant_result[1]
        print(
            f"  mediant({p1}/{q1}, {p2}/{q2}) = {mediant_result[0]}/{mediant_result[1]} = {mediant_val:.4f}"
        )
        print(f"    Between {val1:.4f} and {val2:.4f}: {val1 < mediant_val < val2}")

    # Stern-Brocot tree paths
    sb_examples = [(3, 7), (5, 8), (2, 5)]
    print("\n  Stern-Brocot tree paths:")
    for p, q in sb_examples:
        sb_result = await number_theory.stern_brocot_tree(p, q)
        path = "".join(sb_result["path"])
        print(f"    Path to {p}/{q}: {path} (depth {sb_result['depth']})")

    await print_subheader("Ford Circles and Geometric Properties")

    # Ford circles for small Farey sequences
    ford_examples = [4, 5]
    for n in ford_examples:
        circles_data = await number_theory.ford_circles(n)
        print(f"  Ford circles for F_{n}: {circles_data['count']} circles")

        # Show a few circles
        for i, circle in enumerate(circles_data["circles"][:4]):
            frac = circle["fraction"]
            center = circle["center"]
            radius = circle["radius"]
            print(
                f"    {frac[0]}/{frac[1]}: center ({center[0]:.3f}, {center[1]:.3f}), radius {radius:.3f}"
            )

    # Circle tangency analysis
    print("\n  Ford circle tangency relationships:")
    tangency_examples = [
        (1, 3, 1, 2),  # Adjacent in F_3
        (1, 4, 1, 3),  # Adjacent in F_4
        (2, 5, 1, 2),  # Adjacent in F_5
    ]

    for p1, q1, p2, q2 in tangency_examples:
        tangency = await number_theory.circle_tangency(p1, q1, p2, q2)
        print(f"    Circles for {p1}/{q1} and {p2}/{q2}: tangent = {tangency['are_tangent']}")
        print(
            f"      Farey adjacent: {tangency['are_farey_adjacent']} (det = {tangency['determinant']})"
        )

    await print_subheader("Farey Sequence Analysis and Density")

    # Analyze properties of larger Farey sequences
    analysis_examples = [8, 10]
    for n in analysis_examples:
        props = await number_theory.farey_sequence_properties(n)
        print(f"  F_{n} properties:")
        print(f"    Length: {props['length']}, max gap: {props['max_gap']:.4f}")
        print(
            f"    Avg gap: {props['avg_gap']:.4f}, unique denominators: {props['unique_denominators']}"
        )
        print(f"    Adjacent pairs: {props['adjacent_pairs']}")

    # Density growth analysis
    print("\n  Farey sequence density growth:")
    density_data = await number_theory.density_analysis(12)
    densities = density_data["densities"]
    print(f"    |F_n| for n=1..12: {densities}")
    print(f"    Asymptotic constant 3/π² ≈ {density_data['theoretical_constant']:.4f}")
    print(f"    Estimated constant: {density_data['estimated_constant']:.4f}")

    await print_subheader("Applications and Approximations")

    # Best rational approximations using Farey sequences
    approximation_targets = [
        (math.pi - 3, "π - 3"),
        (math.sqrt(2) - 1, "√2 - 1"),
        ((1 + math.sqrt(5)) / 2 - 1, "φ - 1"),
    ]

    print("  Best Farey approximations:")
    for value, name in approximation_targets:
        if 0 <= value <= 1:
            approx = await number_theory.best_approximation_farey(value, 20)
            print(f"    {name} ≈ {value:.6f}")
            print(
                f"      Best: {approx['best_approximation'][0]}/{approx['best_approximation'][1]} (error: {approx['error']:.6f})"
            )

    # Finding fractions between given fractions
    print("\n  Finding fractions between given fractions:")
    between_examples = [
        (1, 3, 1, 2),  # Between 1/3 and 1/2
        (2, 7, 1, 3),  # Between 2/7 and 1/3
        (3, 8, 2, 5),  # Between 3/8 and 2/5
    ]

    for p1, q1, p2, q2 in between_examples:
        between_result = await number_theory.farey_fraction_between(p1, q1, p2, q2)
        between_frac = between_result["fraction_between"]
        print(f"    Between {p1}/{q1} and {p2}/{q2}: {between_frac[0]}/{between_frac[1]} (mediant)")
        print(
            f"      Value: {between_result['value']:.6f}, adjacent: {between_result['fractions_are_adjacent']}"
        )


async def demo_mathematical_sequences():
    """Demonstrate various mathematical sequences and their properties."""
    await print_header("Mathematical Sequences & Special Numbers")

    await print_subheader("Classic Sequences")

    # Generate several famous sequences
    n_terms = 10
    sequences = {
        "Fibonacci": [await number_theory.fibonacci(i) for i in range(n_terms)],
        "Lucas": [await number_theory.lucas_number(i) for i in range(n_terms)],
        "Triangular": [await number_theory.triangular_number(i) for i in range(1, n_terms + 1)],
        "Catalan": [await number_theory.catalan_number(i) for i in range(n_terms)],
        "Bell": [
            await number_theory.bell_number(i) for i in range(8)
        ],  # Bell numbers grow very fast
    }

    for name, seq in sequences.items():
        print(f"  {name:11s}: {seq}")

    # Show the golden ratio approximation using Fibonacci
    print("\n  Golden ratio approximations (Fibonacci ratios):")
    for i in range(5, 15):
        fib_i = await number_theory.fibonacci(i)
        fib_i_minus_1 = await number_theory.fibonacci(i - 1)
        if fib_i_minus_1 > 0:
            ratio = fib_i / fib_i_minus_1
            print(f"    F({i})/F({i - 1}) = {fib_i}/{fib_i_minus_1} = {ratio:.8f}")

    # Compute high-precision golden ratio
    golden_ratio = await number_theory.compute_golden_ratio_fibonacci(30)
    print(f"    High-precision golden ratio: {golden_ratio:.12f}")

    await print_subheader("Recursive Sequences")

    # Show Pell numbers and their properties
    print("  Pell Numbers (solutions to x² - 2y² = 1):")
    pell_sequence = []
    for i in range(10):
        pell = await number_theory.pell_number(i)
        pell_sequence.append(pell)
    print(f"    Pell: {pell_sequence}")

    # Tribonacci sequence
    print("  Tribonacci Numbers:")
    tribonacci_sequence = []
    for i in range(12):
        trib = await number_theory.tribonacci_number(i)
        tribonacci_sequence.append(trib)
    print(f"    Tribonacci: {tribonacci_sequence}")


async def demo_figurate_numbers():
    """Demonstrate figurate numbers and their geometric interpretations."""
    await print_header("Figurate Numbers & Geometric Patterns")

    await print_subheader("Polygonal Numbers")

    # Show various polygonal numbers
    polygon_names = {
        3: "Triangular",
        4: "Square",
        5: "Pentagonal",
        6: "Hexagonal",
        8: "Octagonal",
    }

    print("  Polygonal Numbers (first 8 terms):")
    for sides, name in polygon_names.items():
        sequence = []
        for n in range(1, 9):
            poly_num = await number_theory.polygonal_number(n, sides)
            sequence.append(poly_num)
        print(f"    {name:11s} ({sides}-gonal): {sequence}")

    await print_subheader("Centered Polygonal Numbers")

    # Centered polygonal numbers
    print("  Centered Polygonal Numbers:")
    for n in range(1, 8):
        centered_tri = await number_theory.centered_triangular_number(n)
        centered_sq = await number_theory.centered_square_number(n)
        centered_hex = await number_theory.centered_hexagonal_number(n)
        print(
            f"    n={n}: Centered triangular={centered_tri:2d}, square={centered_sq:2d}, hexagonal={centered_hex:3d}"
        )

    await print_subheader("3D Figurate Numbers")

    # Three-dimensional figurate numbers
    print("  3D Figurate Numbers:")
    for n in range(1, 7):
        octahedral = await number_theory.octahedral_number(n)
        dodecahedral = await number_theory.dodecahedral_number(n)
        icosahedral = await number_theory.icosahedral_number(n)
        print(
            f"    n={n}: Octahedral={octahedral:3d}, Dodecahedral={dodecahedral:4d}, Icosahedral={icosahedral:4d}"
        )

    await print_subheader("Special Figurate Numbers")

    # Pronic numbers (rectangular numbers)
    print("  Pronic Numbers (n(n+1)):")
    pronic_sequence = []
    for n in range(10):
        pronic = await number_theory.pronic_number(n)
        pronic_sequence.append(pronic)
    print(f"    {pronic_sequence}")

    # Star numbers
    print("  Star Numbers:")
    star_sequence = []
    for n in range(1, 8):
        star = await number_theory.star_number(n)
        star_sequence.append(star)
    print(f"    {star_sequence}")


async def demo_advanced_prime_analysis():
    """Demonstrate advanced prime distribution and analysis (NEW)."""
    await print_header("Advanced Prime Distribution & Analysis (NEW)")

    await print_subheader("Prime Counting and Distribution")

    # Prime counting function with approximations
    counting_examples = [100, 1000, 10000]
    for x in counting_examples:
        counting_result = await number_theory.prime_counting_function(x)
        exact = counting_result["exact"]
        pnt_approx = counting_result["pnt_approximation"]
        li_approx = counting_result["li_approximation"]
        print(f"  π({x}) = {exact}")
        print(
            f"    Prime Number Theorem: {pnt_approx:.2f} (error: {counting_result['pnt_error']:.2f})"
        )
        print(
            f"    Logarithmic Integral: {li_approx:.2f} (error: {counting_result['li_error']:.2f})"
        )
        print(f"    Best approximation: {counting_result['best_approximation']}")

    await print_subheader("Prime Gap Analysis")

    # Analyze prime gaps in different ranges
    gap_ranges = [(10, 100), (100, 200), (1000, 1100)]
    for start, end in gap_ranges:
        gaps_result = await number_theory.prime_gaps_analysis(start, end)
        print(f"  Prime gaps in range [{start}, {end}]:")
        print(f"    Average gap: {gaps_result['avg_gap']}")
        print(f"    Maximum gap: {gaps_result['max_gap']}")
        print(f"    Gap distribution: {dict(list(gaps_result['gap_distribution'].items())[:5])}")

    await print_subheader("Prime Conjectures")

    # Bertrand's postulate verification
    bertrand_examples = [10, 25, 100]
    for n in bertrand_examples:
        bertrand_result = await number_theory.bertrand_postulate_verify(n)
        if bertrand_result["holds"]:
            primes_between = bertrand_result["primes_between"]
            print(f"  Bertrand's postulate for n={n}: ✓")
            print(f"    Primes between {n} and {2 * n}: {len(primes_between)} found")
            print(f"    Smallest: {bertrand_result['smallest_prime']}")
        else:
            print(f"  Bertrand's postulate for n={n}: ✗")

    # Prime gap records
    gap_records = await number_theory.prime_gap_records(1000)
    print("  Prime gap records up to 1000:")
    for gap, (p1, p2) in list(gap_records["records"].items())[:6]:  # Show first 6
        print(f"    Gap {gap}: first occurs between {p1} and {p2}")


async def demo_number_properties():
    """Demonstrate various number properties and classifications."""
    await print_header("Number Properties & Classifications")

    await print_subheader("Perfect, Abundant, and Deficient Numbers")

    # Check numbers for perfect/abundant/deficient properties
    test_range = range(1, 31)
    perfect_numbers = []
    abundant_numbers = []
    deficient_numbers = []

    for n in test_range:
        is_perfect = await number_theory.is_perfect_number(n)
        is_abundant = await number_theory.is_abundant_number(n)
        is_deficient = await number_theory.is_deficient_number(n)

        if is_perfect:
            perfect_numbers.append(n)
        elif is_abundant:
            abundant_numbers.append(n)
        elif is_deficient:
            deficient_numbers.append(n)

    print(f"  Perfect numbers (1-30): {perfect_numbers}")
    print(f"  Abundant numbers (1-30): {abundant_numbers}")
    print(f"  First few deficient: {deficient_numbers[:10]}...")

    # Show divisor information for perfect numbers
    for n in perfect_numbers:
        divisors = await number_theory.divisors(n)
        divisor_sum = await number_theory.divisor_sum(n)
        print(
            f"    {n}: divisors = {divisors}, sum = {divisor_sum} (excluding {n} itself: {divisor_sum - n})"
        )

    await print_subheader("Digital Properties")

    # Palindromic numbers
    print("  Palindromic Numbers:")
    palindromes = []
    for n in range(1, 200):
        if await number_theory.is_palindromic_number(n):
            palindromes.append(n)
    print(f"    1-200: {palindromes}")

    # Harshad numbers (divisible by sum of digits)
    print("\n  Harshad Numbers (divisible by digit sum):")
    harshad_numbers = []
    for n in range(1, 51):
        if await number_theory.is_harshad_number(n):
            harshad_numbers.append(n)
    print(f"    1-50: {harshad_numbers}")

    # Show digit sum calculation for some Harshad numbers
    for n in harshad_numbers[:5]:
        digit_sum = await number_theory.digit_sum(n)
        print(f"      {n}: digit sum = {digit_sum}, {n} ÷ {digit_sum} = {n // digit_sum}")


async def demo_iterative_sequences():
    """Demonstrate iterative sequences and special number properties."""
    await print_header("Iterative Sequences & Special Properties")

    await print_subheader("Collatz Conjecture")

    # Collatz sequence examples
    collatz_examples = [7, 12, 19, 27]
    for n in collatz_examples:
        sequence = await number_theory.collatz_sequence(n)
        stopping_time = await number_theory.collatz_stopping_time(n)
        max_value = await number_theory.collatz_max_value(n)
        print(f"  Collatz({n}): length={stopping_time}, max={max_value}")
        if len(sequence) <= 20:  # Only show full sequence for short ones
            print(f"    Sequence: {sequence}")
        else:
            print(f"    Sequence: {sequence[:5]} ... {sequence[-5:]} (truncated)")

    await print_subheader("Happy Numbers")

    # Find happy numbers
    print("  Happy Numbers (1-50):")
    happy_nums = []
    for n in range(1, 51):
        if await number_theory.is_happy_number(n):
            happy_nums.append(n)
    print(f"    {happy_nums}")

    # Show the happy number calculation for one example
    n = 7
    print(f"\n  Happy number calculation for {n}:")
    current = n
    steps = []
    for _ in range(10):  # Limit iterations for display
        digit_squares = sum(int(digit) ** 2 for digit in str(current))
        steps.append(f"{current} → {digit_squares}")
        current = digit_squares
        if current == 1:
            steps.append("1 (Happy!)")
            break
    print(f"    {' → '.join(steps)}")

    await print_subheader("Narcissistic Numbers")

    # Narcissistic numbers (equal to sum of digits raised to power of number of digits)
    print("  Narcissistic Numbers:")
    narcissistic_nums = []
    for n in range(1, 1000):
        if await number_theory.is_narcissistic_number(n):
            narcissistic_nums.append(n)
    print(f"    1-1000: {narcissistic_nums}")

    # Show calculation for 153 (classic example)
    n = 153
    digits = [int(d) for d in str(n)]
    num_digits = len(digits)
    calculation = " + ".join(f"{d}^{num_digits}" for d in digits)
    result = sum(d**num_digits for d in digits)
    print(f"    Example: {n} = {calculation} = {result}")


async def demo_advanced_arithmetic_functions():
    """Demonstrate advanced arithmetic functions."""
    await print_header("Advanced Arithmetic Functions")

    await print_subheader("Euler's Totient Function")

    # Show totient values and their meanings
    totient_examples = [12, 15, 16, 20, 30]
    for n in totient_examples:
        totient = await number_theory.euler_totient(n)
        print(f"  φ({n}) = {totient} (numbers ≤ {n} coprime to {n})")

        # Show which numbers are coprime
        coprimes = []
        for k in range(1, n + 1):
            if await number_theory.gcd(k, n) == 1:
                coprimes.append(k)
        print(f"    Coprimes: {coprimes}")

    await print_subheader("Möbius Function")

    # Möbius function values
    print("  Möbius Function μ(n):")
    mobius_values = []
    for n in range(1, 21):
        mu = await number_theory.mobius_function(n)
        mobius_values.append(f"μ({n})={mu:2d}")

    # Print in rows of 5
    for i in range(0, len(mobius_values), 5):
        print(f"    {' '.join(mobius_values[i : i + 5])}")

    await print_subheader("Divisor Functions")

    # Show various divisor-related functions
    examples = [12, 18, 24, 30]
    for n in examples:
        divisors = await number_theory.divisors(n)
        divisor_count = await number_theory.divisor_count(n)
        divisor_sum = await number_theory.divisor_sum(n)
        little_omega = await number_theory.little_omega(n)  # Number of distinct prime factors
        big_omega = await number_theory.big_omega(n)  # Number of prime factors with multiplicity

        print(f"  n = {n}:")
        print(f"    Divisors: {divisors} (count: {divisor_count}, sum: {divisor_sum})")
        print(f"    ω({n}) = {little_omega}, Ω({n}) = {big_omega}")


async def demo_mathematical_constants():
    """Demonstrate high-precision computation of mathematical constants."""
    await print_header("High-Precision Mathematical Constants")

    await print_subheader("Pi Computations")

    # Various algorithms for computing π
    print("  π approximations using different methods:")

    # Leibniz series (slow convergence)
    pi_leibniz = await number_theory.compute_pi_leibniz(10000)
    print(f"    Leibniz series (10k terms):  {pi_leibniz:.10f}")

    # Machin's formula (faster convergence)
    pi_machin = await number_theory.compute_pi_machin(50)
    print(f"    Machin's formula (50 terms): {pi_machin:.15f}")

    # Show convergence of Leibniz series
    print("\n  Leibniz series convergence:")
    for terms in [100, 1000, 10000]:
        pi_approx = await number_theory.compute_pi_leibniz(terms)
        error = abs(pi_approx - 3.141592653589793)
        print(f"    {terms:5d} terms: {pi_approx:.8f} (error: {error:.2e})")

    await print_subheader("e (Euler's number)")

    # Compute e using series expansion
    print("  e approximations:")
    for terms in [10, 20, 30]:
        e_approx = await number_theory.compute_e_series(terms)
        error = abs(e_approx - 2.718281828459045)
        print(f"    {terms:2d} terms: {e_approx:.12f} (error: {error:.2e})")

    await print_subheader("Golden Ratio")

    # Golden ratio from Fibonacci ratios
    print("  Golden ratio from Fibonacci sequence:")
    for n in [20, 30, 40]:
        golden = await number_theory.compute_golden_ratio_fibonacci(n)
        theoretical = (1 + (5**0.5)) / 2
        error = abs(golden - theoretical)
        print(f"    F({n})/F({n - 1}): {golden:.12f} (error: {error:.2e})")


async def demo_partitions_and_additive_theory():
    """Demonstrate integer partitions and additive number theory."""
    await print_header("Partitions & Additive Number Theory")

    await print_subheader("Integer Partitions")

    # Show partition counts
    print("  Partition counts p(n):")
    partition_counts = []
    for n in range(1, 16):
        count = await number_theory.partition_count(n)
        partition_counts.append(f"p({n})={count}")

    # Print in rows
    for i in range(0, len(partition_counts), 5):
        print(f"    {' '.join(partition_counts[i : i + 5])}")

    # Show actual partitions for small numbers
    for n in [4, 5]:
        partitions = await number_theory.generate_partitions(n)
        print(f"\n  Partitions of {n}: {partitions}")

    await print_subheader("Goldbach Conjecture")

    # Test Goldbach conjecture for even numbers
    print("  Goldbach conjecture (every even > 2 is sum of two primes):")
    for n in range(4, 21, 2):
        goldbach_result = await number_theory.goldbach_conjecture_check(n)
        if goldbach_result:
            pairs = await number_theory.goldbach_pairs(n)
            print(f"    {n} = {pairs[0][0]} + {pairs[0][1]} (and {len(pairs) - 1} other ways)")

    await print_subheader("Sum of Squares")

    # Show representation as sum of two squares
    print("  Representation as sum of two squares:")
    for n in [5, 13, 17, 25, 29]:
        squares = await number_theory.sum_of_two_squares(n)
        if squares:
            a, b = squares
            print(f"    {n} = {a}² + {b}² = {a * a} + {b * b}")


async def demo_cross_module_relationships():
    """Demonstrate mathematical relationships across modules (NEW)."""
    await print_header("Cross-Module Mathematical Relationships (NEW)")

    await print_subheader("Perfect Numbers ↔ Mersenne Primes")

    # Connection between perfect numbers and Mersenne primes
    print("  Euclid-Euler theorem: Perfect numbers ↔ Mersenne primes")
    for exp in [2, 3, 5, 7, 13]:
        mersenne = 2**exp - 1
        if await number_theory.is_prime(mersenne):
            perfect = (2 ** (exp - 1)) * mersenne
            is_perfect = await number_theory.is_perfect_number(perfect)
            print(
                f"    2^{exp}-1 = {mersenne} (Mersenne prime) → Perfect: {perfect} ({is_perfect})"
            )

    await print_subheader("Continued Fractions ↔ Pell Equations")

    # Show how continued fractions solve Pell equations
    print("  Pell equation solutions via continued fractions:")
    for n in [2, 3, 5]:
        # Method 1: Direct Pell solver
        pell_direct = await number_theory.solve_pell_equation(n)
        # Method 2: Continued fraction method
        pell_cf = await number_theory.cf_solve_pell(n)

        if pell_direct.get("exists") and pell_cf.get("solution_found"):
            direct_sol = pell_direct["fundamental"]
            cf_sol = pell_cf["fundamental_solution"]
            print(f"    x² - {n}y² = 1:")
            print(f"      Direct method: ({direct_sol[0]}, {direct_sol[1]})")
            print(f"      CF method:     ({cf_sol[0]}, {cf_sol[1]})")
            match = direct_sol == cf_sol
            print(f"      Methods agree: {match}")

    await print_subheader("Farey Sequences ↔ Continued Fractions")

    # Compare approximation methods
    print("  Rational approximation comparison:")
    approximation_targets = [
        (math.pi - 3, "π - 3"),
        (math.sqrt(2) - 1, "√2 - 1"),
        ((1 + math.sqrt(5)) / 2 - 1, "φ - 1"),
    ]

    for value, name in approximation_targets:
        if 0 <= value <= 1:
            try:
                # Farey sequence method
                farey_approx = await number_theory.best_approximation_farey(value, 20)
                # Continued fraction method
                cf_approx = await number_theory.best_rational_approximation(value, 20)

                print(f"    {name} ≈ {value:.6f}:")
                print(
                    f"      Farey: {farey_approx['best_approximation'][0]}/{farey_approx['best_approximation'][1]} (error: {farey_approx['error']:.6f})"
                )

                # Handle different possible key names for continued fractions
                if "best_approximation" in cf_approx:
                    cf_frac = cf_approx["best_approximation"]
                elif "best_fraction" in cf_approx:
                    cf_frac = cf_approx["best_fraction"]
                else:
                    # Find the fraction key
                    cf_frac = None
                    for key in cf_approx.keys():
                        if isinstance(cf_approx[key], list) and len(cf_approx[key]) == 2:
                            cf_frac = cf_approx[key]
                            break

                if cf_frac:
                    print(
                        f"      CF:    {cf_frac[0]}/{cf_frac[1]} (error: {cf_approx['error']:.6f})"
                    )
                else:
                    print(f"      CF:    {cf_approx}")

            except Exception as e:
                print(f"    {name} ≈ {value:.6f}: Error in approximation comparison - {e}")
                # Fallback: just show Farey approximation
                try:
                    farey_approx = await number_theory.best_approximation_farey(value, 20)
                    print(
                        f"      Farey: {farey_approx['best_approximation'][0]}/{farey_approx['best_approximation'][1]} (error: {farey_approx['error']:.6f})"
                    )
                except Exception as farey_error:
                    print(f"      Farey approximation also failed: {farey_error}")

    await print_subheader("Figurate Numbers ↔ Diophantine Equations")

    # Pythagorean triples and figurate number relationships
    print("  Pythagorean triples and square numbers:")
    triples = await number_theory.pythagorean_triples(30, primitive_only=True)
    for a, b, c in triples[:4]:  # Show first 4
        # Check if any legs are square numbers
        is_a_square = await number_theory.is_perfect_square(a)
        is_b_square = await number_theory.is_perfect_square(b)
        is_c_square = await number_theory.is_perfect_square(c)

        print(f"    ({a}, {b}, {c}): a² = {is_a_square}, b² = {is_b_square}, c² = {is_c_square}")

        # Check if legs are triangular numbers
        triangular_found = False
        for i in range(1, 20):
            tri = await number_theory.triangular_number(i)
            if tri == a or tri == b:
                triangular_found = True
                break
        if triangular_found:
            print("      Contains triangular number!")

    await print_subheader("Prime Patterns ↔ Modular Arithmetic")

    # Quadratic residues and prime patterns
    print("  Quadratic residues and prime congruences:")
    primes = [5, 13, 17, 29]  # Primes ≡ 1 (mod 4)
    for p in primes:
        if p % 4 == 1:
            qr = await number_theory.quadratic_residues(p)
            # Check if -1 is a quadratic residue (should be for p ≡ 1 mod 4)
            is_neg_one_qr = (p - 1) in qr
            print(f"    p = {p} ≡ 1 (mod 4): -1 is QR? {is_neg_one_qr}")

            # Check for sum of two squares representation
            two_squares = await number_theory.sum_of_two_squares_all(p)
            if two_squares:
                print(f"      Can be written as sum of squares: {two_squares}")


async def demo_performance_and_scale():
    """Demonstrate performance and scalability of the library (NEW)."""
    await print_header("Performance & Scale Demonstration (NEW)")

    await print_subheader("Large Number Computations")

    # Test performance on large numbers
    large_number_tests = [
        ("fibonacci(1000)", lambda: number_theory.fibonacci(1000)),
        ("next_prime(100000)", lambda: number_theory.next_prime(100000)),
        ("euler_totient(123456)", lambda: number_theory.euler_totient(123456)),
        ("partition_count(100)", lambda: number_theory.partition_count(100)),
    ]

    for test_name, test_func in large_number_tests:
        start_time = time.time()
        try:
            result = await test_func()
            end_time = time.time()

            # Format result for display
            if isinstance(result, int) and result > 10**20:
                result_str = f"{str(result)[:20]}... ({len(str(result))} digits)"
            else:
                result_str = str(result)

            print(f"  {test_name}:")
            print(f"    Result: {result_str}")
            print(f"    Time: {end_time - start_time:.4f} seconds")
        except Exception as e:
            print(f"  {test_name}: Error - {e}")

    await print_subheader("Batch Operations")

    # Demonstrate batch processing
    print("  Batch processing examples:")

    # Find all primes up to 1000
    start_time = time.time()
    await number_theory.first_n_primes(168)  # π(1000) = 168
    end_time = time.time()
    print(f"    First 168 primes (all primes ≤ 1000): {end_time - start_time:.4f} seconds")

    # Find all perfect squares up to 10000
    start_time = time.time()
    perfect_squares = []
    for i in range(1, 101):  # 100² = 10000
        square = i * i
        if square <= 10000:
            perfect_squares.append(square)
    end_time = time.time()
    print(
        f"    Perfect squares ≤ 10000: {len(perfect_squares)} found in {end_time - start_time:.4f} seconds"
    )

    # Batch amicable number search
    start_time = time.time()
    amicable_pairs = await number_theory.find_amicable_pairs(10000)
    end_time = time.time()
    print(
        f"    Amicable pairs ≤ 10000: {len(amicable_pairs)} found in {end_time - start_time:.4f} seconds"
    )


async def main():
    """Main demonstration function."""
    print("🧮 COMPREHENSIVE NUMBER THEORY LIBRARY DEMONSTRATION")
    print("=" * 70)
    print("Welcome to the chuk_mcp_math enhanced number theory showcase!")
    print("This script demonstrates the extensive capabilities of our")
    print("async-native number theory library with 340+ functions across")
    print("18 specialized modules, including cutting-edge research areas.")

    # Record start time
    start_time = time.time()

    # Run all demonstrations
    demos = [
        demo_prime_numbers,
        demo_cryptographic_applications,
        demo_diophantine_equations,  # NEW
        demo_special_number_categories,  # NEW
        demo_continued_fractions,  # NEW
        demo_farey_sequences,  # NEW
        demo_mathematical_sequences,
        demo_figurate_numbers,
        demo_advanced_prime_analysis,  # NEW
        demo_number_properties,
        demo_iterative_sequences,
        demo_advanced_arithmetic_functions,
        demo_mathematical_constants,
        demo_partitions_and_additive_theory,
        demo_cross_module_relationships,  # NEW
        demo_performance_and_scale,  # NEW
    ]

    for demo in demos:
        try:
            await demo()
        except Exception as e:
            print(f"\n❌ Error in {demo.__name__}: {e}")
            print("Continuing with next demonstration...")

    # Show performance summary
    end_time = time.time()
    await print_header("Performance Summary", "=")
    print("✅ Demonstration completed successfully!")
    print(f"⏱️  Total execution time: {end_time - start_time:.2f} seconds")
    print("🚀 All functions executed asynchronously for optimal performance")
    print(f"📚 Demonstrated {len(demos)} major areas of number theory")
    print("🔢 Showcased 340+ functions across 18 specialized modules")

    print("\n🎯 Key Features Showcased:")
    features = [
        "Prime number operations and advanced prime patterns",
        "Cryptographic functions (RSA, CRT, discrete logs, quadratic residues)",
        "Diophantine equations (linear, Pell's, Pythagorean triples)",
        "Special number categories (amicable, vampire, Keith, taxi numbers)",
        "Continued fractions and rational approximation theory",
        "Farey sequences, Ford circles, and geometric number theory",
        "Mathematical sequences (Fibonacci, Lucas, Catalan, recursive)",
        "Figurate numbers (polygonal, centered, 3D geometric patterns)",
        "Prime distribution analysis and conjecture verification",
        "Number classifications (perfect, abundant, digital properties)",
        "Iterative sequences (Collatz, happy numbers, narcissistic)",
        "Advanced arithmetic functions (totient, Möbius, divisor functions)",
        "High-precision mathematical constants (π, e, φ algorithms)",
        "Integer partitions and additive number theory",
        "Cross-module mathematical relationships and connections",
        "Performance optimization and large-scale computations",
    ]

    for i, feature in enumerate(features, 1):
        print(f"   {i:2d}. {feature}")

    print("\n💡 This enhanced library is perfect for:")
    print("   • Educational mathematics and computer science curricula")
    print("   • Advanced cryptographic algorithm development and analysis")
    print("   • Mathematical research and exploration at graduate level")
    print("   • AI/ML applications requiring sophisticated number theory")
    print("   • Competitive programming and mathematical olympiads")
    print("   • Professional mathematical software development")
    print("   • Research in analytic and algebraic number theory")
    print("   • Geometric number theory and visualization applications")

    print("\n🌟 New Capabilities Added:")
    print("   • Diophantine equation solving (linear, Pell's, quadratic)")
    print("   • Advanced prime constellation analysis and verification")
    print("   • Comprehensive special number taxonomy and properties")
    print("   • Continued fraction theory and approximation algorithms")
    print("   • Farey sequences, Ford circles, and geometric connections")
    print("   • Cross-module mathematical relationship exploration")
    print("   • Research-grade performance and scalability testing")


if __name__ == "__main__":
    # Run the comprehensive enhanced demonstration
    asyncio.run(main())
