#!/usr/bin/env python3
# tests/math/arithmetic/number_theory/test_special_numbers.py
"""
Comprehensive pytest unit tests for special prime numbers and arithmetic functions.

Tests cover:
- Mersenne primes: identification, Lucas-Lehmer test, known exponents
- Fermat primes: identification, generation, known primes
- Sophie Germain and safe primes: identification and pair finding
- Twin, cousin, and sexy primes: identification and pair finding
- Wilson's theorem: primality checking and factorial calculations
- Pseudoprimes and Carmichael numbers: Fermat tests and identification
- Arithmetic functions: Euler totient, Möbius, omega functions
- Prime gaps: calculation and verification
- Edge cases, error conditions, and performance testing
"""

import pytest
import asyncio
import time
from typing import List, Tuple

# Import the functions to test
from chuk_mcp_functions.math.arithmetic.number_theory.special_numbers import (
    # Basic special numbers
    is_perfect_square, is_power_of_two, fibonacci, factorial,
    fibonacci_sequence, is_fibonacci_number,
    
    # Mersenne primes
    is_mersenne_prime, mersenne_prime_exponents, lucas_lehmer_test,
    
    # Fermat primes
    is_fermat_prime, fermat_numbers, known_fermat_primes,
    
    # Sophie Germain and safe primes
    is_sophie_germain_prime, is_safe_prime, safe_prime_pairs,
    
    # Twin primes and related
    is_twin_prime, twin_prime_pairs, cousin_primes, sexy_primes,
    
    # Wilson's theorem
    wilson_theorem_check, wilson_factorial_mod,
    
    # Pseudoprimes and Carmichael numbers
    is_fermat_pseudoprime, fermat_primality_check, is_carmichael_number,
    
    # Arithmetic functions
    euler_totient, mobius_function, little_omega, big_omega,
    
    # Prime gaps
    prime_gap
)

# ============================================================================
# BASIC SPECIAL NUMBERS TESTS
# ============================================================================

class TestBasicSpecialNumbers:
    """Test cases for basic special number functions."""
    
    @pytest.mark.asyncio
    async def test_is_perfect_square_known_squares(self):
        """Test with known perfect squares."""
        perfect_squares = [0, 1, 4, 9, 16, 25, 36, 49, 64, 81, 100, 121, 144, 169, 196, 225]
        
        for square in perfect_squares:
            assert await is_perfect_square(square) == True, f"{square} should be a perfect square"
    
    @pytest.mark.asyncio
    async def test_is_perfect_square_non_squares(self):
        """Test with numbers that are not perfect squares."""
        non_squares = [2, 3, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 17, 18, 19, 20]
        
        for n in non_squares:
            assert await is_perfect_square(n) == False, f"{n} should not be a perfect square"
    
    @pytest.mark.asyncio
    async def test_is_perfect_square_large_numbers(self):
        """Test with larger perfect squares."""
        large_squares = [10000, 40000, 90000, 160000, 250000]  # 100², 200², 300², 400², 500²
        
        for square in large_squares:
            assert await is_perfect_square(square) == True, f"{square} should be a perfect square"
    
    @pytest.mark.asyncio
    async def test_is_perfect_square_edge_cases(self):
        """Test edge cases for perfect square checking."""
        assert await is_perfect_square(0) == True   # 0² = 0
        assert await is_perfect_square(1) == True   # 1² = 1
        assert await is_perfect_square(-1) == False # Negative numbers
        assert await is_perfect_square(-4) == False # Negative numbers
    
    @pytest.mark.asyncio
    async def test_is_power_of_two_known_powers(self):
        """Test with known powers of two."""
        powers_of_two = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
        
        for power in powers_of_two:
            assert await is_power_of_two(power) == True, f"{power} should be a power of two"
    
    @pytest.mark.asyncio
    async def test_is_power_of_two_non_powers(self):
        """Test with numbers that are not powers of two."""
        non_powers = [3, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19, 20]
        
        for n in non_powers:
            assert await is_power_of_two(n) == False, f"{n} should not be a power of two"
    
    @pytest.mark.asyncio
    async def test_is_power_of_two_edge_cases(self):
        """Test edge cases for power of two checking."""
        assert await is_power_of_two(1) == True   # 2⁰ = 1
        assert await is_power_of_two(0) == False  # 0 is not a power of two
        assert await is_power_of_two(-1) == False # Negative numbers
        assert await is_power_of_two(-8) == False # Negative powers
    
    @pytest.mark.asyncio
    async def test_fibonacci_basic_values(self):
        """Test Fibonacci function with basic values."""
        known_fibonacci = [
            (0, 0), (1, 1), (2, 1), (3, 2), (4, 3), (5, 5),
            (6, 8), (7, 13), (8, 21), (9, 34), (10, 55),
            (11, 89), (12, 144), (13, 233), (14, 377), (15, 610)
        ]
        
        for n, expected in known_fibonacci:
            result = await fibonacci(n)
            assert result == expected, f"F({n}) should be {expected}, got {result}"
    
    @pytest.mark.asyncio
    async def test_fibonacci_larger_values(self):
        """Test Fibonacci function with larger values."""
        # Test some larger known Fibonacci numbers
        larger_fibonacci = [
            (20, 6765), (25, 75025), (30, 832040)
        ]
        
        for n, expected in larger_fibonacci:
            result = await fibonacci(n)
            assert result == expected, f"F({n}) should be {expected}, got {result}"
    
    @pytest.mark.asyncio
    async def test_fibonacci_negative_input(self):
        """Test Fibonacci function with negative input."""
        with pytest.raises(ValueError, match="Fibonacci number position must be non-negative"):
            await fibonacci(-1)
        
        with pytest.raises(ValueError, match="Fibonacci number position must be non-negative"):
            await fibonacci(-10)
    
    @pytest.mark.asyncio
    async def test_fibonacci_sequence_basic(self):
        """Test Fibonacci sequence generation."""
        seq_10 = await fibonacci_sequence(10)
        expected_10 = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
        assert seq_10 == expected_10
        
        seq_5 = await fibonacci_sequence(5)
        expected_5 = [0, 1, 1, 2, 3]
        assert seq_5 == expected_5
    
    @pytest.mark.asyncio
    async def test_fibonacci_sequence_edge_cases(self):
        """Test Fibonacci sequence edge cases."""
        assert await fibonacci_sequence(0) == []
        assert await fibonacci_sequence(1) == [0]
        assert await fibonacci_sequence(2) == [0, 1]
        assert await fibonacci_sequence(-1) == []
    
    @pytest.mark.asyncio
    async def test_fibonacci_sequence_consistency(self):
        """Test consistency between fibonacci and fibonacci_sequence."""
        n = 15
        sequence = await fibonacci_sequence(n)
        
        for i in range(n):
            individual = await fibonacci(i)
            assert sequence[i] == individual, f"F({i}) should match sequence[{i}]"
    
    @pytest.mark.asyncio
    async def test_is_fibonacci_number_known_fibonacci(self):
        """Test with known Fibonacci numbers."""
        known_fibonacci_numbers = [0, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610]
        
        for fib in known_fibonacci_numbers:
            assert await is_fibonacci_number(fib) == True, f"{fib} should be a Fibonacci number"
    
    @pytest.mark.asyncio
    async def test_is_fibonacci_number_non_fibonacci(self):
        """Test with numbers that are not Fibonacci numbers."""
        non_fibonacci = [4, 6, 7, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 22]
        
        for n in non_fibonacci:
            assert await is_fibonacci_number(n) == False, f"{n} should not be a Fibonacci number"
    
    @pytest.mark.asyncio
    async def test_is_fibonacci_number_edge_cases(self):
        """Test edge cases for Fibonacci number checking."""
        assert await is_fibonacci_number(0) == True   # F₀ = 0
        assert await is_fibonacci_number(1) == True   # F₁ = F₂ = 1
        assert await is_fibonacci_number(-1) == False # Negative numbers
        assert await is_fibonacci_number(-5) == False # Negative numbers
    
    @pytest.mark.asyncio
    async def test_factorial_basic_values(self):
        """Test factorial function with basic values."""
        known_factorials = [
            (0, 1), (1, 1), (2, 2), (3, 6), (4, 24), (5, 120),
            (6, 720), (7, 5040), (8, 40320), (9, 362880), (10, 3628800)
        ]
        
        for n, expected in known_factorials:
            result = await factorial(n)
            assert result == expected, f"{n}! should be {expected}, got {result}"
    
    @pytest.mark.asyncio
    async def test_factorial_larger_values(self):
        """Test factorial function with larger values."""
        # Test some larger factorials
        larger_factorials = [
            (12, 479001600),
            (13, 6227020800),
            (15, 1307674368000)
        ]
        
        for n, expected in larger_factorials:
            result = await factorial(n)
            assert result == expected, f"{n}! should be {expected}, got {result}"
    
    @pytest.mark.asyncio
    async def test_factorial_negative_input(self):
        """Test factorial function with negative input."""
        with pytest.raises(ValueError, match="Factorial is not defined for negative numbers"):
            await factorial(-1)
        
        with pytest.raises(ValueError, match="Factorial is not defined for negative numbers"):
            await factorial(-5)
    
    @pytest.mark.asyncio
    async def test_factorial_growth_property(self):
        """Test that factorial grows correctly: (n+1)! = (n+1) × n!"""
        for n in range(1, 10):
            n_factorial = await factorial(n)
            n_plus_1_factorial = await factorial(n + 1)
            
            assert n_plus_1_factorial == (n + 1) * n_factorial, f"({n+1})! should equal ({n+1}) × {n}!"

# ============================================================================
# MERSENNE PRIMES TESTS
# ============================================================================

class TestMersennePrimes:
    """Test cases for Mersenne prime functions."""
    
    @pytest.mark.asyncio
    async def test_is_mersenne_prime_known_primes(self):
        """Test with known Mersenne primes."""
        known_mersenne_primes = [3, 7, 31, 127, 8191, 131071, 524287]
        
        for mp in known_mersenne_primes:
            assert await is_mersenne_prime(mp) == True, f"{mp} should be a Mersenne prime"
    
    @pytest.mark.asyncio
    async def test_is_mersenne_prime_non_mersenne(self):
        """Test with numbers that are not Mersenne primes."""
        non_mersenne = [5, 11, 13, 17, 19, 23, 29, 37, 41, 43, 47]
        
        for n in non_mersenne:
            assert await is_mersenne_prime(n) == False, f"{n} should not be a Mersenne prime"
    
    @pytest.mark.asyncio
    async def test_is_mersenne_prime_mersenne_composites(self):
        """Test with Mersenne numbers that are composite."""
        # 2^4 - 1 = 15, 2^6 - 1 = 63, 2^8 - 1 = 255, 2^9 - 1 = 511
        mersenne_composites = [15, 63, 255, 511]
        
        for mc in mersenne_composites:
            assert await is_mersenne_prime(mc) == False, f"{mc} should not be prime"
    
    @pytest.mark.asyncio
    async def test_is_mersenne_prime_edge_cases(self):
        """Test edge cases for Mersenne prime checking."""
        assert await is_mersenne_prime(0) == False
        assert await is_mersenne_prime(1) == False
        assert await is_mersenne_prime(-1) == False
        assert await is_mersenne_prime(2) == False  # Not of Mersenne form
    
    @pytest.mark.asyncio
    async def test_mersenne_prime_exponents_basic(self):
        """Test getting Mersenne prime exponents."""
        exponents_10 = await mersenne_prime_exponents(10)
        expected_10 = [2, 3, 5, 7]
        assert exponents_10 == expected_10
        
        exponents_20 = await mersenne_prime_exponents(20)
        expected_20 = [2, 3, 5, 7, 13, 17, 19]
        assert exponents_20 == expected_20
    
    @pytest.mark.asyncio
    async def test_mersenne_prime_exponents_edge_cases(self):
        """Test edge cases for Mersenne exponents."""
        assert await mersenne_prime_exponents(0) == []
        assert await mersenne_prime_exponents(1) == []
        assert await mersenne_prime_exponents(2) == [2]
    
    @pytest.mark.asyncio
    async def test_lucas_lehmer_test_known_cases(self):
        """Test Lucas-Lehmer test with known cases."""
        # Known Mersenne prime exponents
        assert await lucas_lehmer_test(2) == True   # 2^2 - 1 = 3
        assert await lucas_lehmer_test(3) == True   # 2^3 - 1 = 7
        assert await lucas_lehmer_test(5) == True   # 2^5 - 1 = 31
        assert await lucas_lehmer_test(7) == True   # 2^7 - 1 = 127
        assert await lucas_lehmer_test(13) == True  # 2^13 - 1 = 8191
        
        # Known composite Mersenne numbers
        assert await lucas_lehmer_test(11) == False # 2^11 - 1 = 2047 = 23 × 89
    
    @pytest.mark.asyncio
    async def test_lucas_lehmer_test_invalid_input(self):
        """Test Lucas-Lehmer test with invalid input."""
        assert await lucas_lehmer_test(1) == False
        assert await lucas_lehmer_test(4) == False  # Not prime
        assert await lucas_lehmer_test(6) == False  # Not prime

# ============================================================================
# FERMAT PRIMES TESTS
# ============================================================================

class TestFermatPrimes:
    """Test cases for Fermat prime functions."""
    
    @pytest.mark.asyncio
    async def test_is_fermat_prime_known_primes(self):
        """Test with the five known Fermat primes."""
        known_fermat_primes = [3, 5, 17, 257, 65537]
        
        for fp in known_fermat_primes:
            assert await is_fermat_prime(fp) == True, f"{fp} should be a Fermat prime"
    
    @pytest.mark.asyncio
    async def test_is_fermat_prime_non_fermat(self):
        """Test with primes that are not Fermat primes."""
        non_fermat_primes = [2, 7, 11, 13, 19, 23, 29, 31, 37, 41, 43, 47]
        
        for p in non_fermat_primes:
            assert await is_fermat_prime(p) == False, f"{p} should not be a Fermat prime"
    
    @pytest.mark.asyncio
    async def test_is_fermat_prime_composites(self):
        """Test with composite numbers."""
        composites = [4, 6, 8, 9, 10, 12, 14, 15, 16, 18, 20]
        
        for c in composites:
            assert await is_fermat_prime(c) == False, f"{c} should not be a Fermat prime"
    
    @pytest.mark.asyncio
    async def test_is_fermat_prime_edge_cases(self):
        """Test edge cases for Fermat prime checking."""
        assert await is_fermat_prime(0) == False
        assert await is_fermat_prime(1) == False
        assert await is_fermat_prime(2) == False
        assert await is_fermat_prime(-5) == False
    
    @pytest.mark.asyncio
    async def test_fermat_numbers_generation(self):
        """Test generation of Fermat numbers."""
        fermat_4 = await fermat_numbers(4)
        expected = [3, 5, 17, 257, 65537]
        assert fermat_4 == expected
        
        fermat_2 = await fermat_numbers(2)
        expected_2 = [3, 5, 17]
        assert fermat_2 == expected_2
    
    @pytest.mark.asyncio
    async def test_fermat_numbers_edge_cases(self):
        """Test edge cases for Fermat number generation."""
        assert await fermat_numbers(-1) == []
        assert await fermat_numbers(0) == [3]
        assert await fermat_numbers(1) == [3, 5]
    
    @pytest.mark.asyncio
    async def test_known_fermat_primes_consistency(self):
        """Test consistency of known Fermat primes."""
        known = await known_fermat_primes()
        expected = [3, 5, 17, 257, 65537]
        assert known == expected
        
        # Verify they are all actually Fermat primes
        for fp in known:
            assert await is_fermat_prime(fp) == True

# ============================================================================
# SOPHIE GERMAIN AND SAFE PRIMES TESTS
# ============================================================================

class TestSophieGermainAndSafePrimes:
    """Test cases for Sophie Germain and safe prime functions."""
    
    @pytest.mark.asyncio
    async def test_is_sophie_germain_prime_known_cases(self):
        """Test with known Sophie Germain primes."""
        known_sophie_germain = [2, 3, 5, 11, 23, 29, 41, 53, 83, 89]
        
        for sg in known_sophie_germain:
            assert await is_sophie_germain_prime(sg) == True, f"{sg} should be Sophie Germain prime"
    
    @pytest.mark.asyncio
    async def test_is_sophie_germain_prime_non_sophie_germain(self):
        """Test with primes that are not Sophie Germain."""
        non_sophie_germain = [7, 13, 17, 19, 31, 37, 43, 47]
        
        for p in non_sophie_germain:
            assert await is_sophie_germain_prime(p) == False, f"{p} should not be Sophie Germain prime"
    
    @pytest.mark.asyncio
    async def test_is_sophie_germain_prime_composites(self):
        """Test with composite numbers."""
        composites = [4, 6, 8, 9, 10, 12, 14, 15, 16, 18, 20]
        
        for c in composites:
            assert await is_sophie_germain_prime(c) == False, f"{c} should not be Sophie Germain prime"
    
    @pytest.mark.asyncio
    async def test_is_safe_prime_known_cases(self):
        """Test with known safe primes."""
        known_safe = [5, 7, 11, 23, 47, 59, 83, 107, 167, 179]
        
        for sp in known_safe:
            assert await is_safe_prime(sp) == True, f"{sp} should be safe prime"
    
    @pytest.mark.asyncio
    async def test_is_safe_prime_non_safe(self):
        """Test with primes that are not safe."""
        non_safe = [3, 13, 17, 19, 29, 31, 37, 41, 43, 53]
        
        for p in non_safe:
            assert await is_safe_prime(p) == False, f"{p} should not be safe prime"
    
    @pytest.mark.asyncio
    async def test_is_safe_prime_edge_cases(self):
        """Test edge cases for safe prime checking."""
        assert await is_safe_prime(2) == False  # Special case
        assert await is_safe_prime(3) == False  # Special case
        assert await is_safe_prime(1) == False
        assert await is_safe_prime(0) == False
    
    @pytest.mark.asyncio
    async def test_safe_prime_pairs_consistency(self):
        """Test that Sophie Germain and safe prime pairs are consistent."""
        pairs = await safe_prime_pairs(50)
        
        for sg, safe in pairs:
            # Verify Sophie Germain property
            assert await is_sophie_germain_prime(sg) == True, f"{sg} should be Sophie Germain"
            assert await is_safe_prime(safe) == True, f"{safe} should be safe prime"
            assert safe == 2 * sg + 1, f"Safe prime {safe} should equal 2×{sg}+1"
    
    @pytest.mark.asyncio
    async def test_safe_prime_pairs_known_values(self):
        """Test safe prime pairs with known values."""
        pairs = await safe_prime_pairs(25)
        expected = [(2, 5), (3, 7), (5, 11), (11, 23), (23, 47)]
        assert pairs == expected

# ============================================================================
# TWIN PRIMES AND RELATED TESTS
# ============================================================================

class TestTwinPrimesAndRelated:
    """Test cases for twin, cousin, and sexy prime functions."""
    
    @pytest.mark.asyncio
    async def test_is_twin_prime_known_cases(self):
        """Test with known twin primes."""
        known_twin_primes = [3, 5, 11, 13, 17, 19, 29, 31, 41, 43]
        
        for tp in known_twin_primes:
            assert await is_twin_prime(tp) == True, f"{tp} should be part of twin prime pair"
    
    @pytest.mark.asyncio
    async def test_is_twin_prime_non_twin(self):
        """Test with primes that are not twin primes."""
        non_twin_primes = [2, 23, 37, 47, 53, 67, 79, 83, 89, 97]
        
        for p in non_twin_primes:
            assert await is_twin_prime(p) == False, f"{p} should not be part of twin prime pair"
    
    @pytest.mark.asyncio
    async def test_is_twin_prime_composites(self):
        """Test with composite numbers."""
        composites = [4, 6, 8, 9, 10, 12, 14, 15, 16, 18, 20]
        
        for c in composites:
            assert await is_twin_prime(c) == False, f"{c} should not be twin prime"
    
    @pytest.mark.asyncio
    async def test_twin_prime_pairs_known_values(self):
        """Test twin prime pairs with known values."""
        pairs = await twin_prime_pairs(20)
        expected = [(3, 5), (5, 7), (11, 13), (17, 19)]
        assert pairs == expected
    
    @pytest.mark.asyncio
    async def test_twin_prime_pairs_verification(self):
        """Test that twin prime pairs are actually twin primes."""
        pairs = await twin_prime_pairs(100)
        
        for p1, p2 in pairs:
            assert p2 == p1 + 2, f"Twin primes should differ by 2: {p1}, {p2}"
            assert await is_twin_prime(p1) == True, f"{p1} should be twin prime"
            assert await is_twin_prime(p2) == True, f"{p2} should be twin prime"
    
    @pytest.mark.asyncio
    async def test_cousin_primes_known_values(self):
        """Test cousin primes with known values."""
        pairs = await cousin_primes(20)
        expected = [(3, 7), (7, 11), (13, 17), (19, 23)]
        assert pairs == expected
    
    @pytest.mark.asyncio
    async def test_cousin_primes_verification(self):
        """Test that cousin prime pairs differ by 4."""
        pairs = await cousin_primes(50)
        
        for p1, p2 in pairs:
            assert p2 == p1 + 4, f"Cousin primes should differ by 4: {p1}, {p2}"
            # Import is_prime for verification
            from chuk_mcp_functions.math.arithmetic.number_theory.primes import is_prime
            assert await is_prime(p1) == True, f"{p1} should be prime"
            assert await is_prime(p2) == True, f"{p2} should be prime"
    
    @pytest.mark.asyncio
    async def test_sexy_primes_known_values(self):
        """Test sexy primes with known values."""
        pairs = await sexy_primes(25)
        expected = [(5, 11), (7, 13), (11, 17), (13, 19), (17, 23), (23, 29)]
        assert pairs == expected
    
    @pytest.mark.asyncio
    async def test_sexy_primes_verification(self):
        """Test that sexy prime pairs differ by 6."""
        pairs = await sexy_primes(50)
        
        for p1, p2 in pairs:
            assert p2 == p1 + 6, f"Sexy primes should differ by 6: {p1}, {p2}"
            # Import is_prime for verification
            from chuk_mcp_functions.math.arithmetic.number_theory.primes import is_prime
            assert await is_prime(p1) == True, f"{p1} should be prime"
            assert await is_prime(p2) == True, f"{p2} should be prime"

# ============================================================================
# WILSON'S THEOREM TESTS
# ============================================================================

class TestWilsonTheorem:
    """Test cases for Wilson's theorem functions."""
    
    @pytest.mark.asyncio
    async def test_wilson_theorem_check_primes(self):
        """Test Wilson's theorem with known primes."""
        small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
        
        for p in small_primes:
            assert await wilson_theorem_check(p) == True, f"Wilson's theorem should hold for prime {p}"
    
    @pytest.mark.asyncio
    async def test_wilson_theorem_check_composites(self):
        """Test Wilson's theorem with composite numbers."""
        composites = [4, 6, 8, 9, 10, 12, 14, 15, 16, 18, 20, 21, 22]
        
        for c in composites:
            assert await wilson_theorem_check(c) == False, f"Wilson's theorem should fail for composite {c}"
    
    @pytest.mark.asyncio
    async def test_wilson_theorem_check_edge_cases(self):
        """Test Wilson's theorem with edge cases."""
        assert await wilson_theorem_check(0) == False
        assert await wilson_theorem_check(1) == False
        assert await wilson_theorem_check(2) == True
    
    @pytest.mark.asyncio
    async def test_wilson_factorial_mod_known_values(self):
        """Test factorial modulo calculation with known values."""
        # 6! mod 7 = 720 mod 7 = 6
        assert await wilson_factorial_mod(6, 7) == 6
        
        # 10! mod 11 = 10 (Wilson's theorem)
        assert await wilson_factorial_mod(10, 11) == 10
        
        # 4! mod 5 = 24 mod 5 = 4
        assert await wilson_factorial_mod(4, 5) == 4
    
    @pytest.mark.asyncio
    async def test_wilson_factorial_mod_large_k(self):
        """Test factorial modulo when k >= m."""
        # k! is divisible by m when k >= m
        assert await wilson_factorial_mod(7, 7) == 0
        assert await wilson_factorial_mod(10, 5) == 0
        assert await wilson_factorial_mod(100, 50) == 0
    
    @pytest.mark.asyncio
    async def test_wilson_factorial_mod_small_cases(self):
        """Test factorial modulo with small cases."""
        assert await wilson_factorial_mod(0, 5) == 1  # 0! = 1
        assert await wilson_factorial_mod(1, 7) == 1  # 1! = 1
        assert await wilson_factorial_mod(2, 7) == 2  # 2! = 2
        assert await wilson_factorial_mod(3, 7) == 6  # 3! = 6

# ============================================================================
# PSEUDOPRIMES AND CARMICHAEL NUMBERS TESTS
# ============================================================================

class TestPseudoprimesAndCarmichael:
    """Test cases for pseudoprime and Carmichael number functions."""
    
    @pytest.mark.asyncio
    async def test_is_fermat_pseudoprime_known_cases(self):
        """Test with known Fermat pseudoprimes."""
        # 341 is the smallest base-2 Fermat pseudoprime
        assert await is_fermat_pseudoprime(341, 2) == True
        
        # 341 = 11 × 31, fails for base 3
        assert await is_fermat_pseudoprime(341, 3) == False
        
        # 561 is Carmichael, so pseudoprime to all coprime bases
        assert await is_fermat_pseudoprime(561, 2) == True
        # Note: 561 = 3 × 11 × 17, and gcd(3, 561) = 3, so base 3 is not coprime
        assert await is_fermat_pseudoprime(561, 5) == True
        assert await is_fermat_pseudoprime(561, 7) == True
    
    @pytest.mark.asyncio
    async def test_is_fermat_pseudoprime_with_primes(self):
        """Test that primes are not pseudoprimes."""
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
        
        for p in primes:
            for base in [2, 3, 5]:
                if base != p:  # Don't test p with base p
                    assert await is_fermat_pseudoprime(p, base) == False, f"Prime {p} should not be pseudoprime"
    
    @pytest.mark.asyncio
    async def test_is_fermat_pseudoprime_gcd_condition(self):
        """Test that pseudoprime test requires gcd(a, n) = 1."""
        # 341 = 11 × 31
        assert await is_fermat_pseudoprime(341, 11) == False  # gcd(11, 341) = 11
        assert await is_fermat_pseudoprime(341, 31) == False  # gcd(31, 341) = 31
    
    @pytest.mark.asyncio
    async def test_fermat_primality_check_primes(self):
        """Test Fermat primality check with actual primes."""
        primes = [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
        
        for p in primes:
            for base in [2, 3, 5]:
                if base < p:  # Ensure gcd(base, p) = 1
                    assert await fermat_primality_check(p, base) == True, f"Prime {p} should pass Fermat test"
    
    @pytest.mark.asyncio
    async def test_fermat_primality_check_composites(self):
        """Test Fermat primality check with composites that fail."""
        # These composites should fail for base 2
        failing_composites = [9, 15, 21, 25, 27, 33, 35, 39, 45]
        
        for c in failing_composites:
            assert await fermat_primality_check(c, 2) == False, f"Composite {c} should fail Fermat test"
    
    @pytest.mark.asyncio
    async def test_fermat_primality_check_edge_cases(self):
        """Test edge cases for Fermat primality check."""
        assert await fermat_primality_check(1, 2) == False
        assert await fermat_primality_check(2, 2) == True
        assert await fermat_primality_check(4, 2) == False  # Even composite
        assert await fermat_primality_check(6, 3) == False  # gcd(3, 6) = 3
    
    @pytest.mark.asyncio
    async def test_is_carmichael_number_known_cases(self):
        """Test with known Carmichael numbers."""
        known_carmichael = [561, 1105, 1729, 2465, 2821, 6601, 8911]
        
        for cn in known_carmichael:
            assert await is_carmichael_number(cn) == True, f"{cn} should be Carmichael number"
    
    @pytest.mark.asyncio
    async def test_is_carmichael_number_non_carmichael(self):
        """Test with numbers that are not Carmichael."""
        non_carmichael = [341, 1387, 2047, 3277, 4033, 4681, 5461]  # Pseudoprimes but not Carmichael
        
        for n in non_carmichael:
            assert await is_carmichael_number(n) == False, f"{n} should not be Carmichael number"
    
    @pytest.mark.asyncio
    async def test_is_carmichael_number_primes_and_small_composites(self):
        """Test that primes and small composites are not Carmichael."""
        # Primes
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
        for p in primes:
            assert await is_carmichael_number(p) == False, f"Prime {p} should not be Carmichael"
        
        # Small composites (Carmichael numbers must have at least 3 prime factors)
        small_composites = [4, 6, 8, 9, 10, 12, 14, 15, 16, 18, 20, 21, 22, 25, 26, 27, 28]
        for c in small_composites:
            assert await is_carmichael_number(c) == False, f"Small composite {c} should not be Carmichael"
    
    @pytest.mark.asyncio
    async def test_carmichael_korselt_criterion(self):
        """Test that Carmichael numbers satisfy Korselt's criterion."""
        # Test with 561 = 3 × 11 × 17
        assert await is_carmichael_number(561) == True
        
        # Verify Korselt's criterion manually:
        # (561-1) = 560 should be divisible by (3-1)=2, (11-1)=10, (17-1)=16
        assert 560 % 2 == 0
        assert 560 % 10 == 0
        assert 560 % 16 == 0

# ============================================================================
# ARITHMETIC FUNCTIONS TESTS
# ============================================================================

class TestArithmeticFunctions:
    """Test cases for arithmetic functions."""
    
    @pytest.mark.asyncio
    async def test_euler_totient_known_values(self):
        """Test Euler totient function with known values."""
        known_values = [
            (1, 1), (2, 1), (3, 2), (4, 2), (5, 4), (6, 2),
            (7, 6), (8, 4), (9, 6), (10, 4), (11, 10), (12, 4)
        ]
        
        for n, expected in known_values:
            result = await euler_totient(n)
            assert result == expected, f"φ({n}) should be {expected}, got {result}"
    
    @pytest.mark.asyncio
    async def test_euler_totient_primes(self):
        """Test Euler totient function with primes."""
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
        
        for p in primes:
            result = await euler_totient(p)
            assert result == p - 1, f"φ({p}) should be {p-1} for prime {p}"
    
    @pytest.mark.asyncio
    async def test_euler_totient_prime_powers(self):
        """Test Euler totient function with prime powers."""
        # φ(p^k) = p^k - p^(k-1) = p^(k-1)(p-1)
        test_cases = [
            (4, 2),    # 2^2: φ(4) = 2^1(2-1) = 2
            (8, 4),    # 2^3: φ(8) = 2^2(2-1) = 4
            (9, 6),    # 3^2: φ(9) = 3^1(3-1) = 6
            (25, 20),  # 5^2: φ(25) = 5^1(5-1) = 20
            (27, 18),  # 3^3: φ(27) = 3^2(3-1) = 18
        ]
        
        for n, expected in test_cases:
            result = await euler_totient(n)
            assert result == expected, f"φ({n}) should be {expected}, got {result}"
    
    @pytest.mark.asyncio
    async def test_euler_totient_edge_cases(self):
        """Test Euler totient function edge cases."""
        assert await euler_totient(0) == 0
        assert await euler_totient(1) == 1
        assert await euler_totient(-5) == 0
    
    @pytest.mark.asyncio
    async def test_mobius_function_known_values(self):
        """Test Möbius function with known values."""
        known_values = [
            (1, 1),   # μ(1) = 1
            (2, -1),  # μ(2) = -1 (one prime factor)
            (3, -1),  # μ(3) = -1 (one prime factor)
            (4, 0),   # μ(4) = 0 (has square factor 2^2)
            (5, -1),  # μ(5) = -1 (one prime factor)
            (6, 1),   # μ(6) = 1 (two distinct prime factors: 2, 3)
            (7, -1),  # μ(7) = -1 (one prime factor)
            (8, 0),   # μ(8) = 0 (has square factor 2^3)
            (9, 0),   # μ(9) = 0 (has square factor 3^2)
            (10, 1),  # μ(10) = 1 (two distinct prime factors: 2, 5)
            (12, 0),  # μ(12) = 0 (has square factor 2^2)
            (30, -1), # μ(30) = -1 (three distinct prime factors: 2, 3, 5)
        ]
        
        for n, expected in known_values:
            result = await mobius_function(n)
            assert result == expected, f"μ({n}) should be {expected}, got {result}"
    
    @pytest.mark.asyncio
    async def test_mobius_function_edge_cases(self):
        """Test Möbius function edge cases."""
        assert await mobius_function(0) == 0
        assert await mobius_function(1) == 1
        assert await mobius_function(-5) == 0
    
    @pytest.mark.asyncio
    async def test_little_omega_known_values(self):
        """Test little omega function with known values."""
        known_values = [
            (1, 0),   # ω(1) = 0 (no prime factors)
            (2, 1),   # ω(2) = 1 (prime factor: 2)
            (6, 2),   # ω(6) = 2 (prime factors: 2, 3)
            (12, 2),  # ω(12) = 2 (distinct prime factors: 2, 3)
            (30, 3),  # ω(30) = 3 (prime factors: 2, 3, 5)
            (60, 3),  # ω(60) = 3 (distinct prime factors: 2, 3, 5)
        ]
        
        for n, expected in known_values:
            result = await little_omega(n)
            assert result == expected, f"ω({n}) should be {expected}, got {result}"
    
    @pytest.mark.asyncio
    async def test_little_omega_primes(self):
        """Test little omega function with primes."""
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
        
        for p in primes:
            result = await little_omega(p)
            assert result == 1, f"ω({p}) should be 1 for prime {p}"
    
    @pytest.mark.asyncio
    async def test_big_omega_known_values(self):
        """Test big omega function with known values."""
        known_values = [
            (1, 0),   # Ω(1) = 0 (no prime factors)
            (2, 1),   # Ω(2) = 1 (prime factor: 2)
            (4, 2),   # Ω(4) = 2 (prime factors: 2, 2)
            (8, 3),   # Ω(8) = 3 (prime factors: 2, 2, 2)
            (12, 3),  # Ω(12) = 3 (prime factors: 2, 2, 3)
            (18, 3),  # Ω(18) = 3 (prime factors: 2, 3, 3)
            (60, 4),  # Ω(60) = 4 (prime factors: 2, 2, 3, 5)
        ]
        
        for n, expected in known_values:
            result = await big_omega(n)
            assert result == expected, f"Ω({n}) should be {expected}, got {result}"
    
    @pytest.mark.asyncio
    async def test_omega_functions_relationship(self):
        """Test relationship between little and big omega functions."""
        test_numbers = [12, 18, 24, 30, 36, 60, 72, 90]
        
        for n in test_numbers:
            little = await little_omega(n)
            big = await big_omega(n)
            
            # Big omega should be >= little omega
            assert big >= little, f"Ω({n}) = {big} should be ≥ ω({n}) = {little}"
            
            # They're equal iff n is square-free
            from chuk_mcp_functions.math.arithmetic.number_theory.primes import prime_factors
            factors = await prime_factors(n)
            is_square_free = len(factors) == len(set(factors))
            
            if is_square_free:
                assert big == little, f"For square-free {n}: Ω({n}) should equal ω({n})"

# ============================================================================
# PRIME GAPS TESTS  
# ============================================================================

class TestPrimeGaps:
    """Test cases for prime gap functions."""
    
    @pytest.mark.asyncio
    async def test_prime_gap_known_values(self):
        """Test prime gap calculation with known values."""
        known_gaps = [
            (2, 1),   # gap from 2 to 3
            (3, 2),   # gap from 3 to 5
            (5, 2),   # gap from 5 to 7
            (7, 4),   # gap from 7 to 11
            (11, 2),  # gap from 11 to 13
            (13, 4),  # gap from 13 to 17
            (17, 2),  # gap from 17 to 19
            (19, 4),  # gap from 19 to 23
            (23, 6),  # gap from 23 to 29
        ]
        
        for p, expected_gap in known_gaps:
            result = await prime_gap(p)
            assert result == expected_gap, f"Gap after {p} should be {expected_gap}, got {result}"
    
    @pytest.mark.asyncio
    async def test_prime_gap_invalid_input(self):
        """Test prime gap with invalid input."""
        composites = [4, 6, 8, 9, 10, 12, 14, 15, 16]
        
        for c in composites:
            with pytest.raises(ValueError, match=f"{c} is not prime"):
                await prime_gap(c)
    
    @pytest.mark.asyncio
    async def test_prime_gap_consistency(self):
        """Test that prime gaps are consistent with next prime calculation."""
        from chuk_mcp_functions.math.arithmetic.number_theory.primes import next_prime
        
        test_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
        
        for p in test_primes:
            gap = await prime_gap(p)
            next_p = await next_prime(p)
            
            assert gap == next_p - p, f"Gap after {p} should equal next_prime({p}) - {p}"

# ============================================================================
# INTEGRATION AND PERFORMANCE TESTS
# ============================================================================

class TestIntegration:
    """Integration tests for special number functions."""
    
    @pytest.mark.asyncio
    async def test_basic_special_numbers_consistency(self):
        """Test consistency between basic special number functions."""
        # Test that perfect squares of powers of two are both perfect squares and powers
        powers_of_two_squares = [1, 4, 16, 64, 256]  # 1², 2², 4², 8², 16²
        
        for n in powers_of_two_squares:
            assert await is_perfect_square(n) == True, f"{n} should be perfect square"
            # Note: only 1, 4, 16 are both perfect squares AND powers of two
            if n in [1, 4, 16]:
                assert await is_power_of_two(n) == True, f"{n} should also be power of two"
    
    @pytest.mark.asyncio
    async def test_fibonacci_and_perfect_squares(self):
        """Test Fibonacci numbers that are also perfect squares."""
        # Known Fibonacci numbers that are perfect squares: 0, 1, 144
        fib_perfect_squares = [0, 1, 144]
        
        for n in fib_perfect_squares:
            assert await is_fibonacci_number(n) == True, f"{n} should be Fibonacci"
            assert await is_perfect_square(n) == True, f"{n} should be perfect square"
    
    @pytest.mark.asyncio
    async def test_mersenne_prime_consistency(self):
        """Test consistency between Mersenne prime functions."""
        # Get known exponents and verify corresponding numbers are Mersenne primes
        exponents = await mersenne_prime_exponents(20)
        
        for p in exponents:
            mersenne_number = (1 << p) - 1  # 2^p - 1
            assert await is_mersenne_prime(mersenne_number) == True
            assert await lucas_lehmer_test(p) == True
    
    @pytest.mark.asyncio
    async def test_factorial_and_fibonacci_relationship(self):
        """Test mathematical relationships between factorial and Fibonacci."""
        # Test that factorials grow much faster than Fibonacci numbers
        for n in range(5, 15):
            fib_n = await fibonacci(n)
            fact_n = await factorial(n)
            
            if n > 4:  # For n > 4, n! > F_n
                assert fact_n > fib_n, f"{n}! should be greater than F_{n}"
    
    @pytest.mark.asyncio
    async def test_fermat_prime_consistency(self):
        """Test consistency between Fermat prime functions."""
        known = await known_fermat_primes()
        generated = await fermat_numbers(4)
        
        assert known == generated
        
        for fp in known:
            assert await is_fermat_prime(fp) == True
    
    @pytest.mark.asyncio
    async def test_twin_prime_relationships(self):
        """Test relationships between different twin prime functions."""
        twin_pairs = await twin_prime_pairs(50)
        
        for p1, p2 in twin_pairs:
            assert await is_twin_prime(p1) == True
            assert await is_twin_prime(p2) == True
            assert p2 == p1 + 2
    
    @pytest.mark.asyncio
    async def test_arithmetic_function_properties(self):
        """Test mathematical properties of arithmetic functions."""
        test_numbers = [12, 18, 24, 30, 36, 60]
        
        for n in test_numbers:
            phi = await euler_totient(n)
            mu = await mobius_function(n)
            omega_small = await little_omega(n)
            omega_big = await big_omega(n)
            
            # Euler totient should be positive for n > 1
            if n > 1:
                assert phi > 0, f"φ({n}) should be positive"
            
            # Möbius function should be in {-1, 0, 1}
            assert mu in [-1, 0, 1], f"μ({n}) should be in {{-1, 0, 1}}"
            
            # Big omega >= little omega
            assert omega_big >= omega_small, f"Ω({n}) should be ≥ ω({n})"

class TestPerformance:
    """Performance and async behavior tests."""
    
    @pytest.mark.asyncio
    async def test_all_functions_are_async(self):
        """Test that all special number functions are properly async."""
        operations = [
            is_perfect_square(16),
            is_power_of_two(8),
            fibonacci(10),
            factorial(5),
            is_fibonacci_number(13),
            is_mersenne_prime(127),
            is_fermat_prime(257),
            is_sophie_germain_prime(11),
            is_twin_prime(13),
            wilson_theorem_check(7),
            is_carmichael_number(561),
            euler_totient(12),
            mobius_function(30),
            prime_gap(23)
        ]
        
        # Ensure all are coroutines
        for op in operations:
            assert asyncio.iscoroutine(op)
        
        # Run all operations concurrently
        results = await asyncio.gather(*operations)
        expected = [True, True, 55, 120, True, True, True, True, True, True, True, 4, -1, 6]
        
        assert results == expected
    
    @pytest.mark.asyncio
    async def test_concurrent_execution(self):
        """Test that special number operations can run concurrently."""
        import time
        
        start_time = time.time()
        
        # Run multiple operations concurrently
        tasks = []
        for i in range(2, 52):  # Test numbers 2 through 51
            tasks.append(is_perfect_square(i))
            tasks.append(is_power_of_two(i))
            tasks.append(is_fibonacci_number(i))
        
        results = await asyncio.gather(*tasks)
        duration = time.time() - start_time
        
        # Verify some results
        assert len(results) == 150  # 50 numbers × 3 functions each
        
        # Should complete quickly due to async nature
        assert duration < 2.0
    
    @pytest.mark.asyncio
    async def test_large_number_handling(self):
        """Test handling of moderately large numbers."""
        # Test with larger numbers that still complete quickly
        large_tests = [
            fibonacci(50),          # Large Fibonacci number
            factorial(20),          # Large factorial
            lucas_lehmer_test(31),  # 2^31 - 1 (large Mersenne test)
            euler_totient(1000),    # Larger totient calculation
            twin_prime_pairs(200),  # Find many twin pairs
        ]
        
        results = await asyncio.gather(*large_tests)
        
        # Verify results are reasonable
        assert isinstance(results[0], int)   # Fibonacci result
        assert isinstance(results[1], int)   # Factorial result  
        assert isinstance(results[2], bool)  # Lucas-Lehmer result
        assert isinstance(results[3], int)   # Totient result
        assert isinstance(results[4], list)  # Twin pairs result
        assert len(results[4]) > 0           # Should find some pairs

# Parametrized tests for comprehensive coverage
class TestParametrized:
    """Parametrized tests for comprehensive coverage."""
    
    @pytest.mark.asyncio
    @pytest.mark.parametrize("n,expected", [
        (0, True), (1, True), (4, True), (9, True), (16, True), (25, True),  # Perfect squares
        (2, False), (3, False), (5, False), (6, False), (7, False), (8, False)  # Non-squares
    ])
    async def test_is_perfect_square_parametrized(self, n, expected):
        """Parametrized test for perfect square checking."""
        assert await is_perfect_square(n) == expected
    
    @pytest.mark.asyncio
    @pytest.mark.parametrize("n,expected", [
        (1, True), (2, True), (4, True), (8, True), (16, True), (32, True),  # Powers of two
        (3, False), (5, False), (6, False), (7, False), (9, False), (10, False)  # Non-powers
    ])
    async def test_is_power_of_two_parametrized(self, n, expected):
        """Parametrized test for power of two checking."""
        assert await is_power_of_two(n) == expected
    
    @pytest.mark.asyncio
    @pytest.mark.parametrize("n,expected", [
        (0, 0), (1, 1), (2, 1), (3, 2), (4, 3), (5, 5),
        (6, 8), (7, 13), (8, 21), (9, 34), (10, 55)
    ])
    async def test_fibonacci_parametrized(self, n, expected):
        """Parametrized test for Fibonacci function."""
        assert await fibonacci(n) == expected
    
    @pytest.mark.asyncio
    @pytest.mark.parametrize("n,expected", [
        (0, 1), (1, 1), (2, 2), (3, 6), (4, 24), (5, 120),
        (6, 720), (7, 5040), (8, 40320), (9, 362880), (10, 3628800)
    ])
    async def test_factorial_parametrized(self, n, expected):
        """Parametrized test for factorial function."""
        assert await factorial(n) == expected
    
    @pytest.mark.asyncio
    @pytest.mark.parametrize("n,expected", [
        (3, True), (7, True), (31, True), (127, True),  # Mersenne primes
        (15, False), (63, False), (255, False),         # Mersenne composites
        (5, False), (11, False), (17, False)            # Non-Mersenne primes
    ])
    async def test_is_mersenne_prime_parametrized(self, n, expected):
        """Parametrized test for Mersenne prime checking."""
        assert await is_mersenne_prime(n) == expected
    
    @pytest.mark.asyncio
    @pytest.mark.parametrize("n,expected", [
        (3, True), (5, True), (17, True), (257, True), (65537, True),  # Fermat primes
        (2, False), (7, False), (11, False), (13, False)              # Non-Fermat primes
    ])
    async def test_is_fermat_prime_parametrized(self, n, expected):
        """Parametrized test for Fermat prime checking."""
        assert await is_fermat_prime(n) == expected
    
    @pytest.mark.asyncio
    @pytest.mark.parametrize("n,expected", [
        (1, 1), (2, 1), (3, 2), (4, 2), (5, 4), (6, 2),
        (7, 6), (8, 4), (9, 6), (10, 4), (11, 10), (12, 4)
    ])
    async def test_euler_totient_parametrized(self, n, expected):
        """Parametrized test for Euler totient function."""
        assert await euler_totient(n) == expected

# Error handling tests
class TestErrorHandling:
    """Test error handling and edge cases."""
    
    @pytest.mark.asyncio
    async def test_prime_gap_error_handling(self):
        """Test that prime_gap raises appropriate errors."""
        with pytest.raises(ValueError):
            await prime_gap(4)  # Not prime
        
        with pytest.raises(ValueError):
            await prime_gap(15) # Not prime
    
    @pytest.mark.asyncio
    async def test_basic_special_number_errors(self):
        """Test error handling for basic special number functions."""
        # Test negative inputs where appropriate
        with pytest.raises(ValueError):
            await fibonacci(-1)
        
        with pytest.raises(ValueError):
            await factorial(-1)
    
    @pytest.mark.asyncio
    async def test_edge_case_handling(self):
        """Test edge case handling across all functions."""
        # All functions should handle these edge cases gracefully
        edge_cases = [0, 1, -1, -5]
        
        for n in edge_cases:
            # These should not raise exceptions (except for functions that explicitly don't support negatives)
            await is_perfect_square(n)
            await is_power_of_two(n)
            await is_fibonacci_number(n)
            await is_mersenne_prime(n)
            await is_fermat_prime(n)
            await is_sophie_germain_prime(n)
            await is_safe_prime(n)
            await is_twin_prime(n)
            await wilson_theorem_check(n)
            await is_carmichael_number(n)
            await euler_totient(n)
            await mobius_function(n)
            await little_omega(n)
            await big_omega(n)
    
    @pytest.mark.asyncio
    async def test_error_preserves_async_context(self):
        """Test that errors are properly raised in async context."""
        try:
            await prime_gap(4)  # Should raise ValueError
            assert False, "Should have raised ValueError"
        except ValueError:
            # Should be able to continue with async operations
            result = await is_mersenne_prime(31)
            assert result == True

if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v", "--tb=short"])