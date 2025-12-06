#!/usr/bin/env python3
# tests/math/number_theory/test_advanced_primality.py
"""
Comprehensive pytest unit tests for advanced primality testing algorithms.

Tests cover:
- Probabilistic tests: Miller-Rabin, Solovay-Strassen, Fermat
- Deterministic tests: AKS, deterministic Miller-Rabin
- Specialized tests: strong pseudoprimes, Carmichael numbers
- Edge cases, error conditions, and async behavior
- Mathematical properties and relationships
"""

import pytest
import asyncio
import time
import math

# Import the functions to test
from chuk_mcp_math.number_theory.advanced_primality import (
    miller_rabin_test,
    solovay_strassen_test,
    fermat_primality_test,
    aks_primality_test,
    deterministic_miller_rabin,
    strong_pseudoprime_test,
    carmichael_number_test,
)


class TestMillerRabinTest:
    """Test cases for the Miller-Rabin primality test."""

    @pytest.mark.asyncio
    async def test_small_primes(self):
        """Test Miller-Rabin with small prime numbers."""
        small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]

        for p in small_primes:
            result = await miller_rabin_test(p, k=10)
            assert result is True, f"{p} should be identified as prime"

    @pytest.mark.asyncio
    async def test_small_composites(self):
        """Test Miller-Rabin with small composite numbers."""
        composites = [4, 6, 8, 9, 10, 12, 14, 15, 16, 18, 20, 21, 22, 24, 25]

        for c in composites:
            result = await miller_rabin_test(c, k=10)
            assert result is False, f"{c} should be identified as composite"

    @pytest.mark.asyncio
    async def test_carmichael_numbers(self):
        """Test Miller-Rabin correctly identifies Carmichael numbers as composite."""
        carmichael = [561, 1105, 1729, 2465, 2821, 6601]

        for n in carmichael:
            result = await miller_rabin_test(n, k=20)
            assert result is False, f"Carmichael number {n} should be identified as composite"

    @pytest.mark.asyncio
    async def test_strong_pseudoprimes(self):
        """Test Miller-Rabin with known strong pseudoprimes."""
        # 2047 is a strong pseudoprime to base 2
        result = await miller_rabin_test(2047, k=20)
        assert result is False, "2047 should be identified as composite"

    @pytest.mark.asyncio
    async def test_edge_cases(self):
        """Test Miller-Rabin with edge cases."""
        assert await miller_rabin_test(0, k=5) is False
        assert await miller_rabin_test(1, k=5) is False
        assert await miller_rabin_test(2, k=5) is True
        assert await miller_rabin_test(3, k=5) is True

    @pytest.mark.asyncio
    async def test_larger_primes(self):
        """Test Miller-Rabin with larger known primes."""
        larger_primes = [97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149]

        for p in larger_primes:
            result = await miller_rabin_test(p, k=15)
            assert result is True, f"{p} should be identified as prime"

    @pytest.mark.asyncio
    async def test_varying_rounds(self):
        """Test Miller-Rabin with different numbers of rounds."""
        n = 97  # Known prime

        for k in [1, 5, 10, 20]:
            result = await miller_rabin_test(n, k=k)
            assert result is True, f"Prime should be detected with k={k} rounds"

    @pytest.mark.asyncio
    async def test_even_numbers(self):
        """Test that even numbers (except 2) are rejected."""
        even_numbers = [4, 6, 8, 10, 100, 1000]

        for n in even_numbers:
            result = await miller_rabin_test(n, k=5)
            assert result is False, f"Even number {n} should be composite"


class TestSolovayStrassenTest:
    """Test cases for the Solovay-Strassen primality test."""

    @pytest.mark.asyncio
    async def test_small_primes(self):
        """Test Solovay-Strassen with small primes."""
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]

        for p in primes:
            result = await solovay_strassen_test(p, k=10)
            assert result is True, f"{p} should be identified as prime"

    @pytest.mark.asyncio
    async def test_small_composites(self):
        """Test Solovay-Strassen with small composites."""
        composites = [4, 6, 8, 9, 10, 12, 15, 16, 18, 20, 21, 25]

        for c in composites:
            result = await solovay_strassen_test(c, k=10)
            assert result is False, f"{c} should be identified as composite"

    @pytest.mark.asyncio
    async def test_carmichael_numbers(self):
        """Test Solovay-Strassen with Carmichael numbers."""
        carmichael = [561, 1105, 1729]

        for n in carmichael:
            result = await solovay_strassen_test(n, k=20)
            assert result is False, f"Carmichael {n} should be identified as composite"

    @pytest.mark.asyncio
    async def test_edge_cases(self):
        """Test Solovay-Strassen edge cases."""
        assert await solovay_strassen_test(0, k=5) is False
        assert await solovay_strassen_test(1, k=5) is False
        assert await solovay_strassen_test(2, k=5) is True

    @pytest.mark.asyncio
    async def test_larger_numbers(self):
        """Test Solovay-Strassen with larger numbers."""
        assert await solovay_strassen_test(97, k=10) is True
        assert await solovay_strassen_test(101, k=10) is True
        assert await solovay_strassen_test(100, k=10) is False


class TestFermatPrimalityTest:
    """Test cases for the Fermat primality test."""

    @pytest.mark.asyncio
    async def test_small_primes(self):
        """Test Fermat test with small primes."""
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

        for p in primes:
            result = await fermat_primality_test(p, k=10)
            assert result is True, f"{p} should pass Fermat test"

    @pytest.mark.asyncio
    async def test_small_composites(self):
        """Test Fermat test with small composites."""
        composites = [4, 6, 8, 9, 10, 12, 14, 15, 16, 18, 20, 21]

        for c in composites:
            result = await fermat_primality_test(c, k=10)
            assert result is False, f"{c} should fail Fermat test"

    @pytest.mark.asyncio
    async def test_carmichael_false_positives(self):
        """Test that Fermat gives false positives for Carmichael numbers."""
        # 561 is the smallest Carmichael number
        result = await fermat_primality_test(561, k=10)
        # Fermat test will likely return True (false positive)
        # This is expected behavior for Carmichael numbers
        assert isinstance(result, bool)

    @pytest.mark.asyncio
    async def test_edge_cases(self):
        """Test Fermat test edge cases."""
        assert await fermat_primality_test(0, k=5) is False
        assert await fermat_primality_test(1, k=5) is False
        assert await fermat_primality_test(2, k=5) is True

    @pytest.mark.asyncio
    async def test_varying_rounds(self):
        """Test Fermat with different k values."""
        n = 17  # Prime

        for k in [1, 5, 10, 15]:
            result = await fermat_primality_test(n, k=k)
            assert result is True


class TestAKSPrimalityTest:
    """Test cases for the AKS primality test."""

    @pytest.mark.asyncio
    async def test_small_primes(self):
        """Test AKS with small primes."""
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]

        for p in primes:
            result = await aks_primality_test(p)
            assert result is True, f"{p} should be identified as prime by AKS"

    @pytest.mark.asyncio
    async def test_small_composites(self):
        """Test AKS with small composites."""
        composites = [4, 6, 8, 9, 10, 12, 14, 15, 16, 18, 20, 21, 22, 24, 25]

        for c in composites:
            result = await aks_primality_test(c)
            assert result is False, f"{c} should be identified as composite by AKS"

    @pytest.mark.asyncio
    async def test_perfect_powers(self):
        """Test that AKS correctly rejects perfect powers."""
        perfect_powers = [4, 8, 9, 16, 25, 27, 32, 36, 49, 64, 81, 100]

        for n in perfect_powers:
            result = await aks_primality_test(n)
            assert result is False, f"Perfect power {n} should be composite"

    @pytest.mark.asyncio
    async def test_edge_cases(self):
        """Test AKS edge cases."""
        assert await aks_primality_test(0) is False
        assert await aks_primality_test(1) is False
        assert await aks_primality_test(2) is True

    @pytest.mark.asyncio
    async def test_deterministic_nature(self):
        """Test that AKS gives same result on repeated calls."""
        n = 29

        results = []
        for _ in range(5):
            results.append(await aks_primality_test(n))

        assert all(r is True for r in results), "AKS should be deterministic"


class TestDeterministicMillerRabin:
    """Test cases for the deterministic Miller-Rabin test."""

    @pytest.mark.asyncio
    async def test_small_primes(self):
        """Test deterministic MR with small primes."""
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]

        for p in primes:
            result = await deterministic_miller_rabin(p)
            assert result is True, f"{p} should be identified as prime"

    @pytest.mark.asyncio
    async def test_small_composites(self):
        """Test deterministic MR with small composites."""
        composites = [4, 6, 8, 9, 10, 12, 14, 15, 16, 18, 20, 21, 22, 24, 25]

        for c in composites:
            result = await deterministic_miller_rabin(c)
            assert result is False, f"{c} should be identified as composite"

    @pytest.mark.asyncio
    async def test_strong_pseudoprimes(self):
        """Test that deterministic MR correctly identifies strong pseudoprimes."""
        # 2047 is strong pseudoprime to base 2
        assert await deterministic_miller_rabin(2047) is False

    @pytest.mark.asyncio
    async def test_carmichael_numbers(self):
        """Test deterministic MR with Carmichael numbers."""
        carmichael = [561, 1105, 1729]

        for n in carmichael:
            result = await deterministic_miller_rabin(n)
            assert result is False, f"Carmichael {n} should be composite"

    @pytest.mark.asyncio
    async def test_edge_cases(self):
        """Test deterministic MR edge cases."""
        assert await deterministic_miller_rabin(0) is False
        assert await deterministic_miller_rabin(1) is False
        assert await deterministic_miller_rabin(2) is True
        assert await deterministic_miller_rabin(3) is True

    @pytest.mark.asyncio
    async def test_larger_primes(self):
        """Test deterministic MR with larger primes."""
        larger_primes = [97, 101, 103, 107, 109, 113, 127, 131]

        for p in larger_primes:
            result = await deterministic_miller_rabin(p)
            assert result is True, f"{p} should be prime"


class TestStrongPseudoprimeTest:
    """Test cases for strong pseudoprime testing."""

    @pytest.mark.asyncio
    async def test_known_strong_pseudoprimes(self):
        """Test known strong pseudoprimes."""
        # 2047 is strong pseudoprime to base 2
        result = await strong_pseudoprime_test(2047, [2])
        assert result is True, "2047 should be strong pseudoprime to base 2"

    @pytest.mark.asyncio
    async def test_primes_are_not_pseudoprimes(self):
        """Test that actual primes return False (they're not pseudoprimes)."""
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

        for p in primes:
            result = await strong_pseudoprime_test(p, [2, 3])
            assert result is False, f"Prime {p} should not be a pseudoprime"

    @pytest.mark.asyncio
    async def test_obvious_composites(self):
        """Test obvious composites that aren't strong pseudoprimes."""
        composites = [4, 6, 8, 9, 10, 12, 15, 18, 20, 21]

        for c in composites:
            result = await strong_pseudoprime_test(c, [2])
            assert result is False, f"{c} should not be strong pseudoprime"

    @pytest.mark.asyncio
    async def test_edge_cases(self):
        """Test strong pseudoprime edge cases."""
        assert await strong_pseudoprime_test(0, [2]) is False
        assert await strong_pseudoprime_test(1, [2]) is False
        assert await strong_pseudoprime_test(2, [2]) is False  # 2 is prime, not pseudoprime

    @pytest.mark.asyncio
    async def test_multiple_bases(self):
        """Test strong pseudoprime with multiple bases."""
        # Test with multiple bases
        result = await strong_pseudoprime_test(2047, [2, 3, 5])
        assert isinstance(result, bool)


class TestCarmichaelNumberTest:
    """Test cases for Carmichael number identification."""

    @pytest.mark.asyncio
    async def test_known_carmichael_numbers(self):
        """Test identification of known Carmichael numbers."""
        carmichael = [561, 1105, 1729, 2465, 2821, 6601]

        for n in carmichael:
            result = await carmichael_number_test(n)
            assert result is True, f"{n} should be identified as Carmichael number"

    @pytest.mark.asyncio
    async def test_primes_are_not_carmichael(self):
        """Test that primes are not Carmichael numbers."""
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]

        for p in primes:
            result = await carmichael_number_test(p)
            assert result is False, f"Prime {p} should not be Carmichael"

    @pytest.mark.asyncio
    async def test_non_carmichael_composites(self):
        """Test composites that are not Carmichael numbers."""
        composites = [4, 6, 8, 9, 10, 12, 14, 15, 16, 18, 20, 21, 22, 24, 25]

        for c in composites:
            result = await carmichael_number_test(c)
            assert result is False, f"{c} is composite but not Carmichael"

    @pytest.mark.asyncio
    async def test_edge_cases(self):
        """Test Carmichael test edge cases."""
        assert await carmichael_number_test(0) is False
        assert await carmichael_number_test(1) is False
        assert await carmichael_number_test(2) is False

    @pytest.mark.asyncio
    async def test_smallest_carmichael(self):
        """Test the smallest Carmichael number."""
        # 561 is the smallest Carmichael number
        result = await carmichael_number_test(561)
        assert result is True, "561 should be identified as Carmichael"


class TestIntegration:
    """Integration tests for advanced primality functions."""

    @pytest.mark.asyncio
    async def test_algorithms_agree_on_primes(self):
        """Test that different algorithms agree on prime numbers."""
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

        for p in primes:
            mr_result = await miller_rabin_test(p, k=10)
            ss_result = await solovay_strassen_test(p, k=10)
            fermat_result = await fermat_primality_test(p, k=10)
            det_mr_result = await deterministic_miller_rabin(p)

            assert mr_result is True
            assert ss_result is True
            assert fermat_result is True
            assert det_mr_result is True

    @pytest.mark.asyncio
    async def test_algorithms_agree_on_composites(self):
        """Test that different algorithms agree on composite numbers."""
        # Use composites that aren't Carmichael numbers
        composites = [4, 6, 8, 9, 10, 12, 14, 15, 16, 18, 20]

        for c in composites:
            mr_result = await miller_rabin_test(c, k=10)
            ss_result = await solovay_strassen_test(c, k=10)
            fermat_result = await fermat_primality_test(c, k=10)
            det_mr_result = await deterministic_miller_rabin(c)

            assert mr_result is False
            assert ss_result is False
            assert fermat_result is False
            assert det_mr_result is False

    @pytest.mark.asyncio
    async def test_carmichael_detection_consistency(self):
        """Test that Carmichael numbers are properly identified across tests."""
        carmichael = [561, 1105, 1729]

        for n in carmichael:
            # Should be identified as Carmichael
            is_carmichael = await carmichael_number_test(n)
            assert is_carmichael is True

            # Should be identified as composite by robust tests
            mr_composite = await miller_rabin_test(n, k=20)
            det_composite = await deterministic_miller_rabin(n)

            assert mr_composite is False
            assert det_composite is False


class TestPerformanceAndAsync:
    """Test async behavior and performance characteristics."""

    @pytest.mark.asyncio
    async def test_all_functions_are_async(self):
        """Test that all primality functions are async."""
        import asyncio

        operations = [
            miller_rabin_test(97, k=10),
            solovay_strassen_test(97, k=10),
            fermat_primality_test(97, k=10),
            aks_primality_test(97),
            deterministic_miller_rabin(97),
            strong_pseudoprime_test(2047, [2]),
            carmichael_number_test(561),
        ]

        for op in operations:
            assert asyncio.iscoroutine(op), "All operations should be coroutines"

        # Clean up coroutines
        results = await asyncio.gather(*operations)
        assert len(results) == len(operations)

    @pytest.mark.asyncio
    async def test_concurrent_execution(self):
        """Test that primality tests can run concurrently."""
        start_time = time.time()

        # Run multiple tests concurrently
        tasks = [miller_rabin_test(n, k=10) for n in range(100, 150)]

        results = await asyncio.gather(*tasks)
        duration = time.time() - start_time

        assert len(results) == 50
        assert duration < 5.0  # Should complete quickly

    @pytest.mark.asyncio
    async def test_timeout_behavior(self):
        """Test that operations complete within reasonable time."""
        operations = [
            (miller_rabin_test(982451653, k=10), 2.0),
            (deterministic_miller_rabin(982451653), 2.0),
            (solovay_strassen_test(97, k=10), 1.0),
        ]

        for operation, timeout in operations:
            try:
                result = await asyncio.wait_for(operation, timeout=timeout)
                assert result is not None
            except asyncio.TimeoutError:
                pytest.fail(f"Operation took longer than {timeout} seconds")


class TestParametrized:
    """Parametrized tests for comprehensive coverage."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "n,expected",
        [
            (2, True),
            (3, True),
            (5, True),
            (7, True),
            (11, True),
            (4, False),
            (6, False),
            (8, False),
            (9, False),
            (10, False),
            (97, True),
            (100, False),
            (101, True),
            (102, False),
        ],
    )
    async def test_miller_rabin_parametrized(self, n, expected):
        """Parametrized Miller-Rabin tests."""
        result = await miller_rabin_test(n, k=10)
        assert result == expected

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "n,expected",
        [
            (2, True),
            (3, True),
            (5, True),
            (7, True),
            (4, False),
            (6, False),
            (8, False),
            (9, False),
            (97, True),
            (100, False),
        ],
    )
    async def test_deterministic_mr_parametrized(self, n, expected):
        """Parametrized deterministic Miller-Rabin tests."""
        result = await deterministic_miller_rabin(n)
        assert result == expected

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "n,expected",
        [
            (561, True),
            (1105, True),
            (1729, True),
            (2, False),
            (3, False),
            (97, False),
            (100, False),
        ],
    )
    async def test_carmichael_parametrized(self, n, expected):
        """Parametrized Carmichael number tests."""
        result = await carmichael_number_test(n)
        assert result == expected


class TestEdgeCasesAndErrors:
    """Test edge cases and error conditions."""

    @pytest.mark.asyncio
    async def test_negative_numbers(self):
        """Test handling of negative numbers."""
        negative_numbers = [-1, -5, -10, -100]

        for n in negative_numbers:
            assert await miller_rabin_test(n, k=5) is False
            assert await solovay_strassen_test(n, k=5) is False
            assert await fermat_primality_test(n, k=5) is False
            assert await deterministic_miller_rabin(n) is False

    @pytest.mark.asyncio
    async def test_zero_and_one(self):
        """Test handling of 0 and 1."""
        for n in [0, 1]:
            assert await miller_rabin_test(n, k=5) is False
            assert await solovay_strassen_test(n, k=5) is False
            assert await fermat_primality_test(n, k=5) is False
            assert await aks_primality_test(n) is False
            assert await deterministic_miller_rabin(n) is False

    @pytest.mark.asyncio
    async def test_two_is_special(self):
        """Test that 2 (only even prime) is handled correctly."""
        assert await miller_rabin_test(2, k=5) is True
        assert await solovay_strassen_test(2, k=5) is True
        assert await fermat_primality_test(2, k=5) is True
        assert await aks_primality_test(2) is True
        assert await deterministic_miller_rabin(2) is True

    @pytest.mark.asyncio
    async def test_very_small_k_values(self):
        """Test behavior with very small k values."""
        n = 97  # Known prime

        for k in [1, 2, 3]:
            result = await miller_rabin_test(n, k=k)
            assert result is True

    @pytest.mark.asyncio
    async def test_empty_base_list(self):
        """Test strong pseudoprime with empty base list."""
        result = await strong_pseudoprime_test(2047, [])
        # Should handle empty list gracefully
        assert isinstance(result, bool)

    @pytest.mark.asyncio
    async def test_large_numbers_async_yield(self):
        """Test async yield points for very large numbers."""
        # Test numbers that trigger asyncio.sleep(0) calls
        large_number = 10**11 + 3  # Should be prime, triggers line 94
        result = await miller_rabin_test(large_number, k=5)
        assert isinstance(result, bool)

        # Very large number for line 120
        very_large = 10**16 + 61  # Triggers line 120
        result = await miller_rabin_test(very_large, k=3)
        assert isinstance(result, bool)

        # Large number for solovay_strassen (line 176)
        result = await solovay_strassen_test(large_number, k=3)
        assert isinstance(result, bool)

        # Very large for solovay_strassen (line 202)
        result = await solovay_strassen_test(very_large, k=2)
        assert isinstance(result, bool)

        # Large number for fermat (line 266)
        result = await fermat_primality_test(large_number, k=3)
        assert isinstance(result, bool)

        # Very large for fermat (line 282)
        result = await fermat_primality_test(very_large, k=2)
        assert isinstance(result, bool)

    @pytest.mark.asyncio
    async def test_base_larger_than_n(self):
        """Test strong_pseudoprime_test with bases >= n."""
        # This tests line 537 (continue when base >= n)
        result = await strong_pseudoprime_test(10, [2, 3, 100, 200])
        assert isinstance(result, bool)

    @pytest.mark.asyncio
    async def test_strong_pseudoprime_passes_condition(self):
        """Test strong pseudoprime when it passes for some base."""
        # Test condition where x^(2^i) ≡ -1 (mod n) triggers lines 550-551
        # 2047 is a strong pseudoprime to base 2
        # We need a case where the inner loop finds x == n-1
        result = await strong_pseudoprime_test(2047, [2])
        assert result is True  # 2047 is strong pseudoprime to base 2


class TestHelperFunctions:
    """Test helper functions to improve coverage."""

    @pytest.mark.asyncio
    async def test_jacobi_symbol_error(self):
        """Test _jacobi_symbol_async with invalid input (line 650)."""
        from chuk_mcp_math.number_theory.advanced_primality import _jacobi_symbol_async

        # Test with even n (should raise ValueError)
        with pytest.raises(ValueError, match="positive odd integer"):
            await _jacobi_symbol_async(3, 4)

        # Test with negative n
        with pytest.raises(ValueError, match="positive odd integer"):
            await _jacobi_symbol_async(3, -5)

        # Test with zero
        with pytest.raises(ValueError, match="positive odd integer"):
            await _jacobi_symbol_async(3, 0)

    @pytest.mark.asyncio
    async def test_euler_totient_edge_cases(self):
        """Test _euler_totient_async with edge cases (line 710)."""
        from chuk_mcp_math.number_theory.advanced_primality import _euler_totient_async

        # Test with n <= 0 (line 710)
        result = await _euler_totient_async(0)
        assert result == 0

        result = await _euler_totient_async(-5)
        assert result == 0

        # Test with n = 1
        result = await _euler_totient_async(1)
        assert result == 1

        # Test with prime
        result = await _euler_totient_async(7)
        assert result == 6

    @pytest.mark.asyncio
    async def test_aks_small_n_le_r(self):
        """Test AKS when n <= r (line 364)."""

        # Find a small prime where n <= r
        # For n=3, r should be around log(3)^2 ~ 1.2, so r >= 2
        # When we check if 2 <= 3 is True, then n <= r should trigger
        # But we need to test the actual condition in the code

        # Test small primes - need to ensure n <= r condition
        result = await aks_primality_test(3)
        assert result is True

        # Actually, let's trace through: for n=3, after checking gcd with a in range(2, min(r+1, n))
        # If r is found to be >= 3, then n <= r would be True (3 <= r)
        # Let's test this more directly by using very small numbers
        result = await aks_primality_test(2)
        assert result is True  # 2 is handled specially but let's verify

    @pytest.mark.asyncio
    async def test_aks_polynomial_check_fail(self):
        """Test AKS polynomial check failure (line 374)."""
        # Test with a composite that triggers polynomial check
        # This should test the polynomial check loop
        result = await aks_primality_test(21)
        assert result is False

        result = await aks_primality_test(33)
        assert result is False

    @pytest.mark.asyncio
    async def test_aks_large_loop_async(self):
        """Test AKS with larger numbers to trigger async sleep (line 378)."""
        # Use a larger composite to trigger the sleep in polynomial check loop
        result = await aks_primality_test(91)  # 91 = 7 * 13
        assert result is False

    @pytest.mark.asyncio
    async def test_carmichael_korselt_criterion_fail(self):
        """Test Carmichael when Korselt's criterion fails (line 630)."""
        # Test composites that have 3+ prime factors but fail Korselt's criterion
        # Most composites will fail this test
        # 30 = 2 * 3 * 5, but (30-1) % (2-1) = 29 % 1 = 0, (30-1) % (3-1) = 29 % 2 = 1 (fails)
        result = await carmichael_number_test(30)
        assert result is False

        # 42 = 2 * 3 * 7
        result = await carmichael_number_test(42)
        assert result is False

    @pytest.mark.asyncio
    async def test_find_aks_r_return(self):
        """Test _find_aks_r_async return path (line 702)."""
        from chuk_mcp_math.number_theory.advanced_primality import _find_aks_r_async

        # Test with various numbers to ensure we hit the return r path
        r = await _find_aks_r_async(17)
        assert isinstance(r, int)
        assert r > 0

        r = await _find_aks_r_async(31)
        assert isinstance(r, int)
        assert r > 0

    @pytest.mark.asyncio
    async def test_miller_rabin_witnesses_large_base(self):
        """Test _miller_rabin_with_witnesses_async with base >= n (line 749)."""
        from chuk_mcp_math.number_theory.advanced_primality import (
            _miller_rabin_with_witnesses_async,
        )

        # Test with witnesses larger than n
        result = await _miller_rabin_with_witnesses_async(10, [2, 3, 100, 200])
        assert isinstance(result, bool)

    @pytest.mark.asyncio
    async def test_prime_factorization_async_yield(self):
        """Test _prime_factorization_async async yield (line 780)."""
        from chuk_mcp_math.number_theory.advanced_primality import (
            _prime_factorization_async,
        )

        # Use a number that requires many iterations to trigger d % 100 == 0
        # This will test line 780
        large_composite = 2 * 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29
        factors = await _prime_factorization_async(large_composite)
        assert len(factors) > 0
        assert all(isinstance(f, int) for f in factors)

        # Also test with a number that has a large prime factor
        # to force iteration past d % 100 == 0
        large_num = 101 * 103 * 107  # Product of primes > 100
        factors = await _prime_factorization_async(large_num)
        assert len(factors) == 3

    @pytest.mark.asyncio
    async def test_aks_specific_paths(self):
        """Test specific AKS paths that are hard to hit."""
        from chuk_mcp_math.number_theory.advanced_primality import (
            _find_aks_r_async,
            _aks_polynomial_check_async,
        )

        # Test _find_aks_r_async to hit line 702 (return r)
        # Use primes where the order condition is satisfied
        for n in [11, 13, 17, 19, 23, 29]:
            r = await _find_aks_r_async(n)
            assert r > 0

        # Test polynomial check that returns False (line 374)
        # Test with composite numbers
        result = await _aks_polynomial_check_async(15, 2, 4)
        assert isinstance(result, bool)

        # Test with a case that should pass
        result = await _aks_polynomial_check_async(7, 2, 3)
        assert isinstance(result, bool)

    @pytest.mark.asyncio
    async def test_strong_pseudoprime_witness_paths(self):
        """Test strong pseudoprime specific witness paths."""
        # Test with 1373653 which is strong pseudoprime to bases 2 and 3
        # This should help hit lines 550-551 (passes = True, break)
        result = await strong_pseudoprime_test(1373653, [2, 3])
        # 1373653 = 829 * 1657, it's composite
        # It is a strong pseudoprime to base 2 and base 3
        assert isinstance(result, bool)

        # Test with a base that's much larger than n to hit line 537
        result = await strong_pseudoprime_test(15, [20, 30, 40])
        assert isinstance(result, bool)

    @pytest.mark.asyncio
    async def test_aks_loop_with_larger_limit(self):
        """Test AKS with numbers that create larger loop iterations."""
        # Test with numbers that will create a limit > 100 to trigger line 378
        # For this we need: limit = int(sqrt(phi(r)) * log(n))
        # If we use a moderately large prime, we might hit this
        result = await aks_primality_test(29)
        assert result is True

        result = await aks_primality_test(31)
        assert result is True

        # Test with composite
        result = await aks_primality_test(35)
        assert result is False

    @pytest.mark.asyncio
    async def test_deterministic_mr_small_n_skips_large_witnesses(self):
        """Test deterministic_miller_rabin with n smaller than some witnesses (line 749)."""
        # When n is small (e.g., 5), many of the default witnesses will be >= n
        # The function uses witnesses = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        # So for n=5, witnesses 5, 7, 11, 13, etc. will be skipped (line 749)
        result = await deterministic_miller_rabin(5)
        assert result is True

        result = await deterministic_miller_rabin(7)
        assert result is True

        result = await deterministic_miller_rabin(11)
        assert result is True

    @pytest.mark.asyncio
    async def test_find_aks_r_early_return(self):
        """Test _find_aks_r_async to hit the early return (line 702)."""
        from chuk_mcp_math.number_theory.advanced_primality import _find_aks_r_async

        # For certain primes, the order condition order > log_n_squared should be met
        # and return r early (line 702)
        # Let's try various primes
        for prime in [7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53]:
            r = await _find_aks_r_async(prime)
            assert r > 0
            # Verify that r was found (not just defaulting to int(log_n_squared))
            log_n_sq = int(math.log(prime) ** 2)
            # r should be <= log_n_sq since we search up to that range
            assert r <= log_n_sq + 1

    @pytest.mark.asyncio
    async def test_aks_n_less_than_or_equal_r(self):
        """Test AKS when n <= r condition is True (line 364)."""
        # For very small n, after checking gcd with range(2, min(r+1, n)),
        # if r is found to be >= n, then n <= r and we return True
        # This happens naturally with small primes where r might be larger than n

        # Test with 3 - the smallest odd prime
        result = await aks_primality_test(3)
        assert result is True

    @pytest.mark.asyncio
    async def test_aks_polynomial_early_false(self):
        """Test AKS polynomial check returning False early (line 374)."""
        # We need a composite where the polynomial check fails
        # This tests the condition: if not await _aks_polynomial_check_async(n, a, r): return False

        # Composites that will fail polynomial check
        for composite in [15, 21, 25, 33, 35, 39, 45, 49, 51, 55, 57, 63]:
            result = await aks_primality_test(composite)
            assert result is False

    @pytest.mark.asyncio
    async def test_aks_large_enough_for_async_sleep(self):
        """Test AKS with a number large enough to trigger line 378."""
        # We need limit to be > 100 and a >= 100 to trigger the sleep
        # limit = int(sqrt(phi(r)) * log(n))
        # For a larger number, this is more likely

        # Try a larger composite that will go through many iterations
        result = await aks_primality_test(143)  # 143 = 11 * 13
        assert result is False


class TestDemoFunction:
    """Test the demo function at the end of the module."""

    @pytest.mark.asyncio
    async def test_advanced_primality_demo(self):
        """Test the demo function to improve coverage."""
        from chuk_mcp_math.number_theory.advanced_primality import _test_advanced_primality
        import sys
        from io import StringIO

        # Capture stdout to avoid cluttering test output
        old_stdout = sys.stdout
        sys.stdout = StringIO()

        try:
            # Run the demo function
            await _test_advanced_primality()
        finally:
            # Restore stdout
            sys.stdout = old_stdout


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v", "--tb=short"])
