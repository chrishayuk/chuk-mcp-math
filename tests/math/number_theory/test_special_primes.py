#!/usr/bin/env python3
# tests/math/number_theory/test_special_primes.py
"""
Comprehensive pytest test suite for special_primes.py module.

Tests cover:
- Mersenne primes: detection, Lucas-Lehmer test, known exponents
- Fermat primes: detection, generation, known primes
- Sophie Germain and safe primes: detection and pair finding
- Twin, cousin, and sexy primes: detection and pair finding
- Wilson's theorem: primality checking and factorial calculations
- Pseudoprimes and Carmichael numbers: Fermat tests and detection
- Prime gaps: calculation and pattern analysis
- Edge cases, error conditions, and performance testing
- Async behavior verification
"""

import pytest
import asyncio
import time

# Import the functions to test
from chuk_mcp_math.number_theory.special_primes import (
    # Mersenne primes
    is_mersenne_prime,
    mersenne_prime_exponents,
    lucas_lehmer_test,
    mersenne_numbers,
    # Fermat primes
    is_fermat_prime,
    fermat_numbers,
    known_fermat_primes,
    # Sophie Germain and safe primes
    is_sophie_germain_prime,
    is_safe_prime,
    safe_prime_pairs,
    # Twin primes and related
    is_twin_prime,
    twin_prime_pairs,
    cousin_primes,
    sexy_primes,
    # Wilson's theorem
    wilson_theorem_check,
    wilson_factorial_mod,
    # Pseudoprimes and Carmichael numbers
    is_fermat_pseudoprime,
    fermat_primality_check,
    is_carmichael_number,
    # Prime gaps and patterns
    prime_gap,
    largest_prime_gap_in_range,
    twin_prime_gaps,
)

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
            assert await is_mersenne_prime(mp), (
                f"{mp} should be a Mersenne prime"
            )

    @pytest.mark.asyncio
    async def test_is_mersenne_prime_non_mersenne(self):
        """Test with numbers that are not Mersenne primes."""
        non_mersenne = [5, 11, 13, 17, 19, 23, 29, 37, 41, 43, 47]

        for n in non_mersenne:
            assert not await is_mersenne_prime(n), (
                f"{n} should not be a Mersenne prime"
            )

    @pytest.mark.asyncio
    async def test_is_mersenne_prime_mersenne_composites(self):
        """Test with Mersenne numbers that are composite."""
        # 2^4 - 1 = 15, 2^6 - 1 = 63, 2^8 - 1 = 255, 2^9 - 1 = 511
        mersenne_composites = [15, 63, 255, 511]

        for mc in mersenne_composites:
            assert not await is_mersenne_prime(mc), f"{mc} should not be prime"

    @pytest.mark.asyncio
    async def test_is_mersenne_prime_edge_cases(self):
        """Test edge cases for Mersenne prime checking."""
        assert not await is_mersenne_prime(0)
        assert not await is_mersenne_prime(1)
        assert not await is_mersenne_prime(-1)
        assert not await is_mersenne_prime(2)  # Not of Mersenne form

    @pytest.mark.asyncio
    async def test_mersenne_prime_exponents_basic(self):
        """Test getting Mersenne prime exponents."""
        exponents_10 = await mersenne_prime_exponents(10)
        expected_10 = [2, 3, 5, 7]
        assert exponents_10 == expected_10

        exponents_20 = await mersenne_prime_exponents(20)
        expected_20 = [2, 3, 5, 7, 13, 17, 19]
        assert exponents_20 == expected_20

        exponents_100 = await mersenne_prime_exponents(100)
        expected_100 = [2, 3, 5, 7, 13, 17, 19, 31, 61, 89]
        assert exponents_100 == expected_100

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
        assert await lucas_lehmer_test(2)  # 2^2 - 1 = 3
        assert await lucas_lehmer_test(3)  # 2^3 - 1 = 7
        assert await lucas_lehmer_test(5)  # 2^5 - 1 = 31
        assert await lucas_lehmer_test(7)  # 2^7 - 1 = 127
        assert await lucas_lehmer_test(13)  # 2^13 - 1 = 8191

        # Known composite Mersenne numbers
        assert not await lucas_lehmer_test(11)  # 2^11 - 1 = 2047 = 23 × 89

    @pytest.mark.asyncio
    async def test_lucas_lehmer_test_invalid_input(self):
        """Test Lucas-Lehmer test with invalid input."""
        assert not await lucas_lehmer_test(1)
        assert not await lucas_lehmer_test(4)  # Not prime
        assert not await lucas_lehmer_test(6)  # Not prime

    @pytest.mark.asyncio
    async def test_mersenne_numbers_generation(self):
        """Test generation of Mersenne numbers."""
        mersenne_10 = await mersenne_numbers(10)
        expected = [3, 7, 31, 127]  # For primes 2, 3, 5, 7
        assert mersenne_10 == expected

        mersenne_5 = await mersenne_numbers(5)
        expected_5 = [3, 7, 31]  # For primes 2, 3, 5
        assert mersenne_5 == expected_5

    @pytest.mark.asyncio
    async def test_mersenne_lucas_lehmer_consistency(self):
        """Test consistency between Mersenne prime detection and Lucas-Lehmer test."""
        for p in [2, 3, 5, 7, 11, 13, 17, 19]:
            mersenne_number = (2**p) - 1
            is_mersenne = await is_mersenne_prime(mersenne_number)
            lucas_lehmer_result = await lucas_lehmer_test(p)

            if is_mersenne:
                assert lucas_lehmer_result, (
                    f"Lucas-Lehmer should confirm 2^{p}-1 is prime"
                )


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
            assert await is_fermat_prime(fp), f"{fp} should be a Fermat prime"

    @pytest.mark.asyncio
    async def test_is_fermat_prime_non_fermat(self):
        """Test with primes that are not Fermat primes."""
        non_fermat_primes = [2, 7, 11, 13, 19, 23, 29, 31, 37, 41, 43, 47]

        for p in non_fermat_primes:
            assert not await is_fermat_prime(p), (
                f"{p} should not be a Fermat prime"
            )

    @pytest.mark.asyncio
    async def test_is_fermat_prime_composites(self):
        """Test with composite numbers."""
        composites = [4, 6, 8, 9, 10, 12, 14, 15, 16, 18, 20]

        for c in composites:
            assert not await is_fermat_prime(c), (
                f"{c} should not be a Fermat prime"
            )

    @pytest.mark.asyncio
    async def test_is_fermat_prime_edge_cases(self):
        """Test edge cases for Fermat prime checking."""
        assert not await is_fermat_prime(0)
        assert not await is_fermat_prime(1)
        assert not await is_fermat_prime(2)
        assert not await is_fermat_prime(-5)

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
            assert await is_fermat_prime(fp)

    @pytest.mark.asyncio
    async def test_fermat_number_formula(self):
        """Test that generated Fermat numbers follow the formula F_n = 2^(2^n) + 1."""
        fermat_nums = await fermat_numbers(5)

        for i, fn in enumerate(fermat_nums):
            expected = (2 ** (2**i)) + 1
            assert fn == expected, f"F_{i} should equal 2^(2^{i}) + 1"


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
            assert await is_sophie_germain_prime(sg), (
                f"{sg} should be Sophie Germain prime"
            )

    @pytest.mark.asyncio
    async def test_is_sophie_germain_prime_non_sophie_germain(self):
        """Test with primes that are not Sophie Germain."""
        non_sophie_germain = [7, 13, 17, 19, 31, 37, 43, 47]

        for p in non_sophie_germain:
            assert not await is_sophie_germain_prime(p), (
                f"{p} should not be Sophie Germain prime"
            )

    @pytest.mark.asyncio
    async def test_is_sophie_germain_prime_composites(self):
        """Test with composite numbers."""
        composites = [4, 6, 8, 9, 10, 12, 14, 15, 16, 18, 20]

        for c in composites:
            assert not await is_sophie_germain_prime(c), (
                f"{c} should not be Sophie Germain prime"
            )

    @pytest.mark.asyncio
    async def test_is_safe_prime_known_cases(self):
        """Test with known safe primes."""
        known_safe = [5, 7, 11, 23, 47, 59, 83, 107, 167, 179]

        for sp in known_safe:
            assert await is_safe_prime(sp), f"{sp} should be safe prime"

    @pytest.mark.asyncio
    async def test_is_safe_prime_non_safe(self):
        """Test with primes that are not safe."""
        non_safe = [3, 13, 17, 19, 29, 31, 37, 41, 43, 53]

        for p in non_safe:
            assert not await is_safe_prime(p), f"{p} should not be safe prime"

    @pytest.mark.asyncio
    async def test_is_safe_prime_edge_cases(self):
        """Test edge cases for safe prime checking."""
        assert not await is_safe_prime(2)  # Special case
        assert not await is_safe_prime(3)  # Special case
        assert not await is_safe_prime(1)
        assert not await is_safe_prime(0)

    @pytest.mark.asyncio
    async def test_safe_prime_pairs_consistency(self):
        """Test that Sophie Germain and safe prime pairs are consistent."""
        pairs = await safe_prime_pairs(50)

        for sg, safe in pairs:
            # Verify Sophie Germain property
            assert await is_sophie_germain_prime(sg), (
                f"{sg} should be Sophie Germain"
            )
            assert await is_safe_prime(safe), f"{safe} should be safe prime"
            assert safe == 2 * sg + 1, f"Safe prime {safe} should equal 2×{sg}+1"

    @pytest.mark.asyncio
    async def test_safe_prime_pairs_known_values(self):
        """Test safe prime pairs with known values - CORRECTED."""
        pairs = await safe_prime_pairs(25)
        expected = [
            (2, 5),
            (3, 7),
            (5, 11),
            (11, 23),
            (23, 47),
        ]  # CORRECTED: added (23, 47)
        assert pairs == expected

        pairs_50 = await safe_prime_pairs(50)
        # All Sophie Germain primes ≤ 50 and their corresponding safe primes
        expected_50 = [(2, 5), (3, 7), (5, 11), (11, 23), (23, 47), (29, 59), (41, 83)]
        assert pairs_50 == expected_50


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
            assert await is_twin_prime(tp), (
                f"{tp} should be part of twin prime pair"
            )

    @pytest.mark.asyncio
    async def test_is_twin_prime_non_twin(self):
        """Test with primes that are not twin primes."""
        non_twin_primes = [2, 23, 37, 47, 53, 67, 79, 83, 89, 97]

        for p in non_twin_primes:
            assert not await is_twin_prime(p), (
                f"{p} should not be part of twin prime pair"
            )

    @pytest.mark.asyncio
    async def test_is_twin_prime_composites(self):
        """Test with composite numbers."""
        composites = [4, 6, 8, 9, 10, 12, 14, 15, 16, 18, 20]

        for c in composites:
            assert not await is_twin_prime(c), f"{c} should not be twin prime"

    @pytest.mark.asyncio
    async def test_twin_prime_pairs_known_values(self):
        """Test twin prime pairs with known values."""
        pairs = await twin_prime_pairs(20)
        expected = [(3, 5), (5, 7), (11, 13), (17, 19)]
        assert pairs == expected

        pairs_50 = await twin_prime_pairs(50)
        expected_50 = [(3, 5), (5, 7), (11, 13), (17, 19), (29, 31), (41, 43)]
        assert pairs_50 == expected_50

    @pytest.mark.asyncio
    async def test_twin_prime_pairs_verification(self):
        """Test that twin prime pairs are actually twin primes."""
        pairs = await twin_prime_pairs(100)

        for p1, p2 in pairs:
            assert p2 == p1 + 2, f"Twin primes should differ by 2: {p1}, {p2}"
            assert await is_twin_prime(p1), f"{p1} should be twin prime"
            assert await is_twin_prime(p2), f"{p2} should be twin prime"

    @pytest.mark.asyncio
    async def test_cousin_primes_known_values(self):
        """Test cousin primes with known values."""
        pairs = await cousin_primes(20)
        expected = [(3, 7), (7, 11), (13, 17), (19, 23)]
        assert pairs == expected

        pairs_50 = await cousin_primes(50)
        expected_50 = [(3, 7), (7, 11), (13, 17), (19, 23), (37, 41), (43, 47)]
        assert pairs_50 == expected_50

    @pytest.mark.asyncio
    async def test_cousin_primes_verification(self):
        """Test that cousin prime pairs differ by 4."""
        pairs = await cousin_primes(50)

        for p1, p2 in pairs:
            assert p2 == p1 + 4, f"Cousin primes should differ by 4: {p1}, {p2}"
            # Import is_prime for verification
            from chuk_mcp_math.number_theory.primes import is_prime

            assert await is_prime(p1), f"{p1} should be prime"
            assert await is_prime(p2), f"{p2} should be prime"

    @pytest.mark.asyncio
    async def test_sexy_primes_known_values(self):
        """Test sexy primes with known values - CORRECTED."""
        pairs = await sexy_primes(25)
        # All prime pairs (p, p+6) where p ≤ 25
        expected = [
            (5, 11),
            (7, 13),
            (11, 17),
            (13, 19),
            (17, 23),
            (23, 29),
        ]  # CORRECTED
        assert pairs == expected

        pairs_50 = await sexy_primes(50)
        expected_50 = [
            (5, 11),
            (7, 13),
            (11, 17),
            (13, 19),
            (17, 23),
            (23, 29),
            (31, 37),
            (37, 43),
            (41, 47),
            (47, 53),
        ]  # CORRECTED
        assert pairs_50 == expected_50

    @pytest.mark.asyncio
    async def test_sexy_primes_verification(self):
        """Test that sexy prime pairs differ by 6."""
        pairs = await sexy_primes(50)

        for p1, p2 in pairs:
            assert p2 == p1 + 6, f"Sexy primes should differ by 6: {p1}, {p2}"
            # Import is_prime for verification
            from chuk_mcp_math.number_theory.primes import is_prime

            assert await is_prime(p1), f"{p1} should be prime"
            assert await is_prime(p2), f"{p2} should be prime"


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
            assert await wilson_theorem_check(p), (
                f"Wilson's theorem should hold for prime {p}"
            )

    @pytest.mark.asyncio
    async def test_wilson_theorem_check_composites(self):
        """Test Wilson's theorem with composite numbers."""
        composites = [4, 6, 8, 9, 10, 12, 14, 15, 16, 18, 20, 21, 22]

        for c in composites:
            assert not await wilson_theorem_check(c), (
                f"Wilson's theorem should fail for composite {c}"
            )

    @pytest.mark.asyncio
    async def test_wilson_theorem_check_edge_cases(self):
        """Test Wilson's theorem with edge cases."""
        assert not await wilson_theorem_check(0)
        assert not await wilson_theorem_check(1)
        assert await wilson_theorem_check(2)

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
        assert await is_fermat_pseudoprime(341, 2)

        # 341 = 11 × 31, fails for base 3
        assert not await is_fermat_pseudoprime(341, 3)

        # 561 is Carmichael, so pseudoprime to coprime bases
        assert await is_fermat_pseudoprime(561, 2)
        assert await is_fermat_pseudoprime(561, 5)
        assert await is_fermat_pseudoprime(561, 7)

    @pytest.mark.asyncio
    async def test_is_fermat_pseudoprime_with_primes(self):
        """Test that primes are not pseudoprimes."""
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

        for p in primes:
            for base in [2, 3, 5]:
                if base != p:  # Don't test p with base p
                    assert not await is_fermat_pseudoprime(p, base), (
                        f"Prime {p} should not be pseudoprime"
                    )

    @pytest.mark.asyncio
    async def test_is_fermat_pseudoprime_gcd_condition(self):
        """Test that pseudoprime test requires gcd(a, n) = 1."""
        # 341 = 11 × 31
        assert not await is_fermat_pseudoprime(341, 11)  # gcd(11, 341) = 11
        assert not await is_fermat_pseudoprime(341, 31)  # gcd(31, 341) = 31

    @pytest.mark.asyncio
    async def test_fermat_primality_check_primes(self):
        """Test Fermat primality check with actual primes."""
        primes = [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]

        for p in primes:
            for base in [2, 3, 5]:
                if base < p:  # Ensure gcd(base, p) = 1
                    assert await fermat_primality_check(p, base), (
                        f"Prime {p} should pass Fermat test"
                    )

    @pytest.mark.asyncio
    async def test_fermat_primality_check_composites(self):
        """Test Fermat primality check with composites that fail."""
        # These composites should fail for base 2
        failing_composites = [9, 15, 21, 25, 27, 33, 35, 39, 45]

        for c in failing_composites:
            assert not await fermat_primality_check(c, 2), (
                f"Composite {c} should fail Fermat test"
            )

    @pytest.mark.asyncio
    async def test_fermat_primality_check_edge_cases(self):
        """Test edge cases for Fermat primality check."""
        assert not await fermat_primality_check(1, 2)
        assert await fermat_primality_check(2, 2)
        assert not await fermat_primality_check(4, 2)  # Even composite
        assert not await fermat_primality_check(6, 3)  # gcd(3, 6) = 3

    @pytest.mark.asyncio
    async def test_is_carmichael_number_known_cases(self):
        """Test with known Carmichael numbers."""
        known_carmichael = [561, 1105, 1729, 2465, 2821, 6601, 8911]

        for cn in known_carmichael:
            assert await is_carmichael_number(cn), (
                f"{cn} should be Carmichael number"
            )

    @pytest.mark.asyncio
    async def test_is_carmichael_number_non_carmichael(self):
        """Test with numbers that are not Carmichael."""
        non_carmichael = [
            341,
            1387,
            2047,
            3277,
            4033,
            4681,
            5461,
        ]  # Pseudoprimes but not Carmichael

        for n in non_carmichael:
            assert not await is_carmichael_number(n), (
                f"{n} should not be Carmichael number"
            )

    @pytest.mark.asyncio
    async def test_is_carmichael_number_primes_and_small_composites(self):
        """Test that primes and small composites are not Carmichael."""
        # Primes
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
        for p in primes:
            assert not await is_carmichael_number(p), (
                f"Prime {p} should not be Carmichael"
            )

        # Small composites (Carmichael numbers must have at least 3 prime factors)
        small_composites = [
            4,
            6,
            8,
            9,
            10,
            12,
            14,
            15,
            16,
            18,
            20,
            21,
            22,
            25,
            26,
            27,
            28,
        ]
        for c in small_composites:
            assert not await is_carmichael_number(c), (
                f"Small composite {c} should not be Carmichael"
            )

    @pytest.mark.asyncio
    async def test_carmichael_korselt_criterion(self):
        """Test that Carmichael numbers satisfy Korselt's criterion."""
        # Test with 561 = 3 × 11 × 17
        assert await is_carmichael_number(561)

        # Verify Korselt's criterion manually:
        # (561-1) = 560 should be divisible by (3-1)=2, (11-1)=10, (17-1)=16
        assert 560 % 2 == 0
        assert 560 % 10 == 0
        assert 560 % 16 == 0


# ============================================================================
# PRIME GAPS TESTS
# ============================================================================


class TestPrimeGaps:
    """Test cases for prime gap functions."""

    @pytest.mark.asyncio
    async def test_prime_gap_known_values(self):
        """Test prime gap calculation with known values."""
        known_gaps = [
            (2, 1),  # gap from 2 to 3
            (3, 2),  # gap from 3 to 5
            (5, 2),  # gap from 5 to 7
            (7, 4),  # gap from 7 to 11
            (11, 2),  # gap from 11 to 13
            (13, 4),  # gap from 13 to 17
            (17, 2),  # gap from 17 to 19
            (19, 4),  # gap from 19 to 23
            (23, 6),  # gap from 23 to 29
        ]

        for p, expected_gap in known_gaps:
            result = await prime_gap(p)
            assert result == expected_gap, (
                f"Gap after {p} should be {expected_gap}, got {result}"
            )

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
        from chuk_mcp_math.number_theory.primes import next_prime

        test_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]

        for p in test_primes:
            gap = await prime_gap(p)
            next_p = await next_prime(p)

            assert gap == next_p - p, (
                f"Gap after {p} should equal next_prime({p}) - {p}"
            )

    @pytest.mark.asyncio
    async def test_largest_prime_gap_in_range(self):
        """Test finding largest prime gap in range."""
        gap_info = await largest_prime_gap_in_range(2, 100)

        # The largest gap up to 100 should be 8 (from 89 to 97)
        assert gap_info["gap"] == 8
        assert gap_info["prime"] == 89
        assert gap_info["next_prime"] == 97

        gap_info_50 = await largest_prime_gap_in_range(2, 50)
        # The largest gap up to 50 should be 6 (from 23 to 29)
        assert gap_info_50["gap"] == 6
        assert gap_info_50["prime"] == 23
        assert gap_info_50["next_prime"] == 29

    @pytest.mark.asyncio
    async def test_twin_prime_gaps(self):
        """Test gaps between twin prime pairs."""
        gaps = await twin_prime_gaps(100)

        # Should have gaps between consecutive twin prime pairs
        assert len(gaps) > 0

        # All gaps should be positive
        for gap in gaps:
            assert gap > 0, "Twin prime gaps should be positive"

        # Verify the pattern for known twin prime pairs
        twin_pairs = await twin_prime_pairs(50)
        if len(twin_pairs) >= 2:
            manual_gaps = []
            for i in range(1, len(twin_pairs)):
                gap = twin_pairs[i][0] - twin_pairs[i - 1][0]
                manual_gaps.append(gap)

            gaps_50 = await twin_prime_gaps(50)
            assert gaps_50 == manual_gaps


# ============================================================================
# INTEGRATION AND PERFORMANCE TESTS
# ============================================================================


class TestIntegration:
    """Integration tests for special prime functions."""

    @pytest.mark.asyncio
    async def test_mersenne_fermat_sophie_germain_relationships(self):
        """Test relationships between different types of special primes."""
        # Some numbers can be multiple types of special primes

        # 3 is both Mersenne (2^2 - 1) and Fermat (2^(2^0) + 1) prime
        assert await is_mersenne_prime(3)
        assert await is_fermat_prime(3)

        # 5 is both Fermat (2^(2^1) + 1) and twin prime
        assert await is_fermat_prime(5)
        assert await is_twin_prime(5)

        # 7 is both Mersenne (2^3 - 1) and twin prime
        assert await is_mersenne_prime(7)
        assert await is_twin_prime(7)

    @pytest.mark.asyncio
    async def test_wilson_theorem_vs_primality(self):
        """Test Wilson's theorem against known primality results."""
        from chuk_mcp_math.number_theory.primes import is_prime

        # Wilson's theorem should agree with primality for reasonable range
        for n in range(2, 30):
            is_prime_result = await is_prime(n)
            wilson_result = await wilson_theorem_check(n)

            assert is_prime_result == wilson_result, (
                f"Wilson's theorem disagreement for {n}"
            )

    @pytest.mark.asyncio
    async def test_carmichael_vs_fermat_pseudoprime(self):
        """Test that Carmichael numbers are pseudoprimes to multiple bases."""
        carmichael_numbers = [561, 1105, 1729]

        for cn in carmichael_numbers:
            assert await is_carmichael_number(cn)

            # Should be pseudoprime to several coprime bases
            coprime_bases = [
                2,
                5,
                7,
                13,
            ]  # These should be coprime to the Carmichael numbers
            for base in coprime_bases:
                from chuk_mcp_math.number_theory.divisibility import gcd

                if await gcd(base, cn) == 1:
                    assert await is_fermat_pseudoprime(cn, base), (
                        f"{cn} should be pseudoprime to base {base}"
                    )


class TestPerformance:
    """Performance and async behavior tests."""

    @pytest.mark.asyncio
    async def test_all_functions_are_async(self):
        """Test that all special prime functions are properly async."""
        operations = [
            is_mersenne_prime(127),
            is_fermat_prime(257),
            is_sophie_germain_prime(11),
            is_twin_prime(13),
            wilson_theorem_check(7),
            is_carmichael_number(561),
            prime_gap(23),
            lucas_lehmer_test(7),
        ]

        # Ensure all are coroutines
        for op in operations:
            assert asyncio.iscoroutine(op)

        # Run all operations concurrently
        results = await asyncio.gather(*operations)
        expected = [True, True, True, True, True, True, 6, True]

        assert results == expected

    @pytest.mark.asyncio
    async def test_concurrent_execution(self):
        """Test that special prime operations can run concurrently."""
        start_time = time.time()

        # Run multiple operations concurrently
        tasks = []
        for i in range(10, 60):  # Test range
            tasks.append(is_twin_prime(i))
            if i % 2 == 1:  # Add some variety
                tasks.append(is_sophie_germain_prime(i))

        results = await asyncio.gather(*tasks)
        duration = time.time() - start_time

        # Should complete quickly due to async nature
        assert duration < 2.0
        assert len(results) > 0

    @pytest.mark.asyncio
    async def test_large_number_handling(self):
        """Test handling of moderately large numbers."""
        # Test with larger numbers that still complete quickly
        large_tests = [
            lucas_lehmer_test(31),  # Large Mersenne test
            twin_prime_pairs(200),  # Find many twin pairs
            safe_prime_pairs(150),  # Find Sophie Germain pairs
            largest_prime_gap_in_range(100, 300),  # Analyze gaps
        ]

        results = await asyncio.gather(*large_tests)

        # Verify results are reasonable
        assert isinstance(results[0], bool)  # Lucas-Lehmer result
        assert isinstance(results[1], list)  # Twin pairs
        assert isinstance(results[2], list)  # Sophie Germain pairs
        assert isinstance(results[3], dict)  # Gap analysis
        assert len(results[1]) > 0  # Should find some twin pairs
        assert len(results[2]) > 0  # Should find some Sophie Germain pairs


# Parametrized tests for comprehensive coverage
class TestParametrized:
    """Parametrized tests for comprehensive coverage."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "n,expected",
        [
            (3, True),
            (7, True),
            (31, True),
            (127, True),  # Mersenne primes
            (15, False),
            (63, False),
            (255, False),  # Mersenne composites
            (5, False),
            (11, False),
            (17, False),  # Non-Mersenne primes
        ],
    )
    async def test_is_mersenne_prime_parametrized(self, n, expected):
        """Parametrized test for Mersenne prime checking."""
        assert await is_mersenne_prime(n) == expected

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "n,expected",
        [
            (3, True),
            (5, True),
            (17, True),
            (257, True),
            (65537, True),  # Fermat primes
            (2, False),
            (7, False),
            (11, False),
            (13, False),  # Non-Fermat primes
        ],
    )
    async def test_is_fermat_prime_parametrized(self, n, expected):
        """Parametrized test for Fermat prime checking."""
        assert await is_fermat_prime(n) == expected

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "p,expected",
        [
            (2, True),
            (3, True),
            (5, True),
            (11, True),
            (23, True),  # Sophie Germain primes
            (7, False),
            (13, False),
            (17, False),
            (19, False),  # Non-Sophie Germain
        ],
    )
    async def test_is_sophie_germain_prime_parametrized(self, p, expected):
        """Parametrized test for Sophie Germain prime checking."""
        assert await is_sophie_germain_prime(p) == expected

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "p,expected_gap",
        [(2, 1), (3, 2), (5, 2), (7, 4), (11, 2), (13, 4), (17, 2), (19, 4), (23, 6)],
    )
    async def test_prime_gap_parametrized(self, p, expected_gap):
        """Parametrized test for prime gap calculation."""
        assert await prime_gap(p) == expected_gap


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v", "--tb=short", "--asyncio-mode=auto"])
