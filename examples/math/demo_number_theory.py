#!/usr/bin/env python3
"""
Number Theory Library Demonstration Script

A comprehensive showcase of the chuk_mcp_functions number theory capabilities.
This script demonstrates real-world applications, mathematical relationships,
and advanced cryptographic functions in an educational format.

Run with: python number_theory_demo.py
"""

import asyncio
import time
from typing import List, Tuple, Dict, Any

# Import the number theory library
from chuk_mcp_functions.math import number_theory

async def print_header(title: str, char: str = "="):
    """Print a formatted section header."""
    print(f"\n{char * 60}")
    print(f"🧮 {title}")
    print(f"{char * 60}")

async def print_subheader(title: str):
    """Print a formatted subsection header."""
    print(f"\n📊 {title}")
    print("-" * 40)

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
    
    # Twin primes (primes that differ by 2)
    print("\n  Twin Prime Pairs:")
    twin_candidates = [3, 5, 11, 13, 17, 19, 29, 31]
    for p in twin_candidates:
        is_twin = await number_theory.is_twin_prime(p)
        if is_twin:
            if await number_theory.is_prime(p + 2):
                print(f"    ({p}, {p+2}) are twin primes")
            elif await number_theory.is_prime(p - 2):
                print(f"    ({p-2}, {p}) are twin primes")

async def demo_cryptographic_applications():
    """Demonstrate cryptographic applications of number theory."""
    await print_header("Cryptographic Applications")
    
    await print_subheader("RSA-Style Operations")
    
    # Choose two primes for RSA demonstration
    p = await number_theory.next_prime(100)
    q = await number_theory.next_prime(200)
    n = p * q
    phi_n = (p - 1) * (q - 1)
    
    print(f"  RSA Key Generation Example:")
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
        ([2, 3], [3, 5]),      # x ≡ 2 (mod 3), x ≡ 3 (mod 5)
        ([1, 4, 6], [3, 5, 7]), # System of three congruences
        ([0, 0, 1], [2, 3, 5])  # x ≡ 0 (mod 2), x ≡ 0 (mod 3), x ≡ 1 (mod 5)
    ]
    
    print("  Chinese Remainder Theorem Solutions:")
    for remainders, moduli in crt_examples:
        try:
            result = await number_theory.crt_solve(remainders, moduli)
            if result is not None:
                # Handle tuple return properly
                if isinstance(result, tuple) and len(result) == 2:
                    solution, combined_modulus = result
                    print(f"    System: x ≡ {remainders} (mod {moduli}) → x = {solution} (mod {combined_modulus})")
                    
                    # Verify the solution
                    verification = []
                    for r, m in zip(remainders, moduli):
                        verification.append(f"{solution} ≡ {solution % m} (mod {m})")
                    print(f"      Verify: {' and '.join(verification)}")
                else:
                    # Handle unexpected return format gracefully
                    print(f"    System: x ≡ {remainders} (mod {moduli}) → x = {result}")
                    print(f"      Note: Unexpected return format from crt_solve")
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
                except:
                    pass

async def demo_mathematical_sequences():
    """Demonstrate various mathematical sequences and their properties."""
    await print_header("Mathematical Sequences & Special Numbers")
    
    await print_subheader("Classic Sequences")
    
    # Generate several famous sequences
    n_terms = 10
    sequences = {
        "Fibonacci": [await number_theory.fibonacci(i) for i in range(n_terms)],
        "Lucas": [await number_theory.lucas_number(i) for i in range(n_terms)],
        "Triangular": [await number_theory.triangular_number(i) for i in range(1, n_terms+1)],
        "Catalan": [await number_theory.catalan_number(i) for i in range(n_terms)],
        "Bell": [await number_theory.bell_number(i) for i in range(8)]  # Bell numbers grow very fast
    }
    
    for name, seq in sequences.items():
        print(f"  {name:11s}: {seq}")
    
    # Show the golden ratio approximation using Fibonacci
    print(f"\n  Golden ratio approximations (Fibonacci ratios):")
    for i in range(5, 15):
        fib_i = await number_theory.fibonacci(i)
        fib_i_minus_1 = await number_theory.fibonacci(i-1)
        if fib_i_minus_1 > 0:
            ratio = fib_i / fib_i_minus_1
            print(f"    F({i})/F({i-1}) = {fib_i}/{fib_i_minus_1} = {ratio:.8f}")
    
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
    polygon_names = {3: "Triangular", 4: "Square", 5: "Pentagonal", 6: "Hexagonal", 8: "Octagonal"}
    
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
        print(f"    n={n}: Centered triangular={centered_tri:2d}, square={centered_sq:2d}, hexagonal={centered_hex:3d}")
    
    await print_subheader("3D Figurate Numbers")
    
    # Three-dimensional figurate numbers
    print("  3D Figurate Numbers:")
    for n in range(1, 7):
        octahedral = await number_theory.octahedral_number(n)
        dodecahedral = await number_theory.dodecahedral_number(n)
        icosahedral = await number_theory.icosahedral_number(n)
        print(f"    n={n}: Octahedral={octahedral:3d}, Dodecahedral={dodecahedral:4d}, Icosahedral={icosahedral:4d}")
    
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
        print(f"    {n}: divisors = {divisors}, sum = {divisor_sum} (excluding {n} itself: {divisor_sum - n})")
    
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
        print(f"    Sequence: {sequence}")
    
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
        digit_squares = sum(int(digit)**2 for digit in str(current))
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
        print(f"    {' '.join(mobius_values[i:i+5])}")
    
    await print_subheader("Divisor Functions")
    
    # Show various divisor-related functions
    examples = [12, 18, 24, 30]
    for n in examples:
        divisors = await number_theory.divisors(n)
        divisor_count = await number_theory.divisor_count(n)
        divisor_sum = await number_theory.divisor_sum(n)
        little_omega = await number_theory.little_omega(n)  # Number of distinct prime factors
        big_omega = await number_theory.big_omega(n)       # Number of prime factors with multiplicity
        
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
    print(f"\n  Leibniz series convergence:")
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
        print(f"    F({n})/F({n-1}): {golden:.12f} (error: {error:.2e})")

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
        print(f"    {' '.join(partition_counts[i:i+5])}")
    
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
            print(f"    {n} = {pairs[0][0]} + {pairs[0][1]} (and {len(pairs)-1} other ways)")
    
    await print_subheader("Sum of Squares")
    
    # Show representation as sum of two squares
    print("  Representation as sum of two squares:")
    for n in [5, 13, 17, 25, 29]:
        squares = await number_theory.sum_of_two_squares(n)
        if squares:
            a, b = squares
            print(f"    {n} = {a}² + {b}² = {a*a} + {b*b}")

async def main():
    """Main demonstration function."""
    print("🧮 COMPREHENSIVE NUMBER THEORY LIBRARY DEMONSTRATION")
    print("=" * 60)
    print("Welcome to the chuk_mcp_functions number theory showcase!")
    print("This script demonstrates the extensive capabilities of our")
    print("async-native number theory library with real-world examples.")
    
    # Record start time
    start_time = time.time()
    
    # Run all demonstrations
    demos = [
        demo_prime_numbers,
        demo_cryptographic_applications,
        demo_mathematical_sequences,
        demo_figurate_numbers,
        demo_number_properties,
        demo_iterative_sequences,
        demo_advanced_arithmetic_functions,
        demo_mathematical_constants,
        demo_partitions_and_additive_theory
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
    print(f"✅ Demonstration completed successfully!")
    print(f"⏱️  Total execution time: {end_time - start_time:.2f} seconds")
    print(f"🚀 All functions executed asynchronously for optimal performance")
    print(f"📚 Demonstrated {len(demos)} major areas of number theory")
    
    print(f"\n🎯 Key Features Showcased:")
    features = [
        "Prime number operations and special prime types",
        "Cryptographic functions (RSA, CRT, discrete logs)",
        "Mathematical sequences (Fibonacci, Lucas, Catalan)",
        "Figurate numbers (polygonal, centered, 3D)",
        "Number classifications (perfect, abundant, etc.)",
        "Iterative sequences (Collatz, happy numbers)",
        "Advanced arithmetic functions (totient, Möbius)",
        "High-precision mathematical constants",
        "Integer partitions and additive number theory"
    ]
    
    for i, feature in enumerate(features, 1):
        print(f"   {i}. {feature}")
    
    print(f"\n💡 This library is perfect for:")
    print(f"   • Educational mathematics and computer science")
    print(f"   • Cryptographic algorithm development")
    print(f"   • Mathematical research and exploration")
    print(f"   • AI/ML applications requiring number theory")
    print(f"   • Competitive programming and puzzles")

if __name__ == "__main__":
    # Run the comprehensive demonstration
    asyncio.run(main())