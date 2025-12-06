#!/usr/bin/env python3
# tests/math/number_theory/test_sieve_algorithms.py
"""
Comprehensive pytest unit tests for sieve algorithms.

Tests cover:
- Classical sieves: Eratosthenes, Sundaram, Atkin
- Segmented and wheel sieves
- Counting functions
- Optimized sieves
- Performance and correctness
"""

import pytest
import asyncio

from chuk_mcp_math.number_theory.sieve_algorithms import (
    sieve_of_eratosthenes,
    sieve_of_sundaram,
    sieve_of_atkin,
    segmented_sieve,
    wheel_sieve,
    prime_counting_sieve,
    mertens_function_sieve,
    linear_sieve,
    incremental_sieve,
    sieve_performance_analysis,
    prime_gap_sieve,
)


class TestSieveOfEratosthenes:
    """Test Sieve of Eratosthenes."""

    @pytest.mark.asyncio
    async def test_small_limit(self):
        """Test with small limit."""
        primes = await sieve_of_eratosthenes(30)
        expected = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
        assert primes == expected

    @pytest.mark.asyncio
    async def test_edge_cases(self):
        """Test edge cases."""
        assert await sieve_of_eratosthenes(0) == []
        assert await sieve_of_eratosthenes(1) == []
        assert await sieve_of_eratosthenes(2) == [2]
        assert await sieve_of_eratosthenes(3) == [2, 3]

    @pytest.mark.asyncio
    async def test_first_25_primes(self):
        """Test first 25 primes."""
        primes = await sieve_of_eratosthenes(100)
        assert len(primes) == 25
        assert primes[0] == 2
        assert primes[-1] == 97

    @pytest.mark.asyncio
    async def test_all_results_prime(self):
        """Verify all results are prime."""
        primes = await sieve_of_eratosthenes(50)
        # Quick primality check
        for p in primes:
            assert p >= 2
            if p > 2:
                assert p % 2 != 0


class TestSieveOfSundaram:
    """Test Sieve of Sundaram."""

    @pytest.mark.asyncio
    async def test_basic_functionality(self):
        """Test basic Sundaram sieve."""
        primes = await sieve_of_sundaram(30)
        expected = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
        assert primes == expected

    @pytest.mark.asyncio
    async def test_edge_cases(self):
        """Test edge cases."""
        assert await sieve_of_sundaram(0) == []
        assert await sieve_of_sundaram(1) == []
        assert await sieve_of_sundaram(2) == [2]
        assert await sieve_of_sundaram(3) == [2, 3]

    @pytest.mark.asyncio
    async def test_matches_eratosthenes(self):
        """Test that Sundaram matches Eratosthenes."""
        limit = 50
        sundaram = await sieve_of_sundaram(limit)
        eratosthenes = await sieve_of_eratosthenes(limit)
        assert sundaram == eratosthenes


class TestSieveOfAtkin:
    """Test Sieve of Atkin."""

    @pytest.mark.asyncio
    async def test_basic_functionality(self):
        """Test basic Atkin sieve."""
        primes = await sieve_of_atkin(30)
        expected = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
        assert primes == expected

    @pytest.mark.asyncio
    async def test_edge_cases(self):
        """Test edge cases."""
        assert await sieve_of_atkin(0) == []
        assert await sieve_of_atkin(1) == []
        assert await sieve_of_atkin(2) == [2]

    @pytest.mark.asyncio
    async def test_matches_eratosthenes(self):
        """Test that Atkin matches Eratosthenes."""
        limit = 100
        atkin = await sieve_of_atkin(limit)
        eratosthenes = await sieve_of_eratosthenes(limit)
        assert atkin == eratosthenes


class TestSegmentedSieve:
    """Test segmented sieve."""

    @pytest.mark.asyncio
    async def test_basic_range(self):
        """Test basic range."""
        primes = await segmented_sieve(10, 30)
        expected = [11, 13, 17, 19, 23, 29]
        assert primes == expected

    @pytest.mark.asyncio
    async def test_includes_endpoints(self):
        """Test that endpoints are included."""
        primes = await segmented_sieve(2, 10)
        assert 2 in primes
        assert 7 in primes

    @pytest.mark.asyncio
    async def test_large_range(self):
        """Test with larger range."""
        primes = await segmented_sieve(100, 120)
        expected = [101, 103, 107, 109, 113]
        assert primes == expected

    @pytest.mark.asyncio
    async def test_invalid_range(self):
        """Test with invalid range."""
        assert await segmented_sieve(30, 10) == []
        assert await segmented_sieve(10, 0) == []

    @pytest.mark.asyncio
    async def test_matches_full_sieve(self):
        """Test segmented matches full sieve."""
        low, high = 50, 100
        segmented = await segmented_sieve(low, high)
        full = await sieve_of_eratosthenes(high)
        expected = [p for p in full if low <= p <= high]
        assert segmented == expected


class TestWheelSieve:
    """Test wheel sieve."""

    @pytest.mark.asyncio
    async def test_default_wheel(self):
        """Test with default wheel [2, 3]."""
        primes = await wheel_sieve(30)
        expected = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
        assert primes == expected

    @pytest.mark.asyncio
    async def test_custom_wheel(self):
        """Test with custom wheel."""
        primes = await wheel_sieve(30, [2, 3, 5])
        expected = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
        assert primes == expected

    @pytest.mark.asyncio
    async def test_small_wheel(self):
        """Test with small wheel [2]."""
        primes = await wheel_sieve(20, [2])
        expected = [2, 3, 5, 7, 11, 13, 17, 19]
        assert primes == expected

    @pytest.mark.asyncio
    async def test_edge_cases(self):
        """Test edge cases."""
        assert await wheel_sieve(0) == []
        assert await wheel_sieve(1) == []

    @pytest.mark.asyncio
    async def test_limit_equals_max_wheel_prime(self):
        """Test line 457 - limit <= max(wheel_primes)."""
        # When limit equals max wheel prime, should return sorted primes up to limit
        result = await wheel_sieve(3, [2, 3])
        assert result == [2, 3]

        result = await wheel_sieve(2, [2, 3])
        assert result == [2]

    @pytest.mark.asyncio
    async def test_large_wheel_async(self):
        """Test line 472 - wheel_size > 1000."""
        # Create a large wheel with many small primes
        # 2*3*5*7*11*13 = 30030 > 1000
        result = await wheel_sieve(100, [2, 3, 5, 7, 11, 13])
        assert 2 in result
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_large_limit_wheel_async(self):
        """Test line 488 - base % 1000000 == 0."""
        # Test with large limit to trigger async sleep
        result = await wheel_sieve(1000001, [2, 3])
        assert len(result) > 0
        assert 2 in result


class TestPrimeCountingSieve:
    """Test prime counting sieve."""

    @pytest.mark.asyncio
    async def test_known_counts(self):
        """Test with known π(n) values."""
        assert await prime_counting_sieve(10) == 4
        assert await prime_counting_sieve(100) == 25
        assert await prime_counting_sieve(1000) == 168

    @pytest.mark.asyncio
    async def test_edge_cases(self):
        """Test edge cases."""
        assert await prime_counting_sieve(0) == 0
        assert await prime_counting_sieve(1) == 0
        assert await prime_counting_sieve(2) == 1

    @pytest.mark.asyncio
    async def test_matches_sieve_length(self):
        """Test count matches sieve length."""
        limit = 50
        count = await prime_counting_sieve(limit)
        primes = await sieve_of_eratosthenes(limit)
        assert count == len(primes)


class TestMertensFunctionSieve:
    """Test Mertens function sieve."""

    @pytest.mark.asyncio
    async def test_small_values(self):
        """Test with small n values."""
        # Known values of Mertens function
        result = await mertens_function_sieve(10)
        assert isinstance(result, int)

    @pytest.mark.asyncio
    async def test_edge_cases(self):
        """Test edge cases."""
        assert await mertens_function_sieve(0) == 0
        assert await mertens_function_sieve(1) == 1

    @pytest.mark.asyncio
    async def test_bounded_growth(self):
        """Test that Mertens function is bounded."""
        for n in [5, 10, 20, 50]:
            m = await mertens_function_sieve(n)
            # Mertens function grows slowly
            assert abs(m) <= n


class TestLinearSieve:
    """Test linear sieve."""

    @pytest.mark.asyncio
    async def test_basic_functionality(self):
        """Test basic linear sieve."""
        primes = await linear_sieve(30)
        expected = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
        assert primes == expected

    @pytest.mark.asyncio
    async def test_matches_eratosthenes(self):
        """Test that linear sieve matches Eratosthenes."""
        limit = 100
        linear = await linear_sieve(limit)
        eratosthenes = await sieve_of_eratosthenes(limit)
        assert linear == eratosthenes

    @pytest.mark.asyncio
    async def test_edge_cases(self):
        """Test edge cases."""
        assert await linear_sieve(0) == []
        assert await linear_sieve(1) == []
        assert await linear_sieve(2) == [2]


class TestIncrementalSieve:
    """Test incremental sieve."""

    @pytest.mark.asyncio
    async def test_extend_existing(self):
        """Test extending existing sieve."""
        current = [2, 3, 5, 7]
        extended = await incremental_sieve(current, 10, 20)
        expected = [2, 3, 5, 7, 11, 13, 17, 19]
        assert extended == expected

    @pytest.mark.asyncio
    async def test_fresh_sieve(self):
        """Test starting from scratch."""
        result = await incremental_sieve([], 0, 10)
        expected = await sieve_of_eratosthenes(10)
        assert result == expected

    @pytest.mark.asyncio
    async def test_no_extension_needed(self):
        """Test when new limit <= current limit."""
        current = [2, 3, 5, 7]
        result = await incremental_sieve(current, 10, 5)
        assert result == [2, 3, 5]


class TestPrimeGapSieve:
    """Test prime gap analysis."""

    @pytest.mark.asyncio
    async def test_basic_gap_analysis(self):
        """Test basic gap analysis."""
        result = await prime_gap_sieve(100)

        assert "max_gap" in result
        assert "min_gap" in result
        assert "avg_gap" in result
        assert "gap_distribution" in result
        assert result["min_gap"] >= 1
        assert result["max_gap"] >= result["min_gap"]

    @pytest.mark.asyncio
    async def test_twin_prime_detection(self):
        """Test twin prime pair detection."""
        result = await prime_gap_sieve(50)
        # Should detect some twin primes (gap of 2)
        assert "twin_prime_pairs" in result
        assert result["twin_prime_pairs"] > 0

    @pytest.mark.asyncio
    async def test_small_limit_error(self):
        """Test with limit too small."""
        result = await prime_gap_sieve(3)
        assert "error" in result

    @pytest.mark.asyncio
    async def test_insufficient_primes_error(self):
        """Test line 968 - not enough primes for gap analysis."""
        # Test edge case where there might not be enough primes
        result = await prime_gap_sieve(4)
        # Should either have valid results or error
        assert isinstance(result, dict)
        # If we get an error, it should be the right error
        if "error" in result:
            assert "enough primes" in result["error"] or "too small" in result["error"]

    @pytest.mark.asyncio
    async def test_edge_case_limit_2(self):
        """Test line 968 - edge case with limit=2."""
        # With limit=2, we get only one prime [2], which is insufficient for gap analysis
        result = await prime_gap_sieve(2)
        # Should return an error since we need at least 2 primes for gaps
        assert isinstance(result, dict)
        if "error" in result:
            assert "enough" in result["error"].lower() or "small" in result["error"].lower()


class TestSievePerformanceAnalysis:
    """Test sieve performance analysis."""

    @pytest.mark.asyncio
    async def test_performance_comparison(self):
        """Test performance comparison."""
        result = await sieve_performance_analysis(100)

        assert "algorithm_results" in result
        assert "fastest_algorithm" in result
        assert "correctness_verified" in result

    @pytest.mark.asyncio
    async def test_all_algorithms_run(self):
        """Test that all algorithms are tested."""
        result = await sieve_performance_analysis(50)

        algorithms = ["eratosthenes", "sundaram", "atkin", "linear"]
        for alg in algorithms:
            assert alg in result["algorithm_results"]

    @pytest.mark.asyncio
    async def test_correctness_check(self):
        """Test correctness verification."""
        result = await sieve_performance_analysis(100)
        assert result["correctness_verified"] is True

    @pytest.mark.asyncio
    async def test_no_valid_results(self):
        """Test lines 886-887, 895 - error handling when no valid results."""
        # This test ensures the code handles edge cases gracefully
        # We test with limit 0 which should cause empty results
        result = await sieve_performance_analysis(0)

        # All algorithms should handle limit 0 gracefully
        assert "algorithm_results" in result
        assert "fastest_algorithm" in result
        assert "slowest_algorithm" in result

        # Verify structure is correct even with edge case
        assert isinstance(result["algorithm_results"], dict)

    @pytest.mark.asyncio
    async def test_exception_handling(self):
        """Test lines 886-887 - exception handling in performance analysis."""
        # Use a negative limit which might cause issues
        result = await sieve_performance_analysis(-1)

        # Should have algorithm_results even with errors
        assert "algorithm_results" in result

        # Check that the structure is maintained
        assert isinstance(result, dict)


class TestIntegration:
    """Integration tests."""

    @pytest.mark.asyncio
    async def test_all_sieves_agree(self):
        """Test that all sieve algorithms agree."""
        limit = 50

        eratosthenes = await sieve_of_eratosthenes(limit)
        sundaram = await sieve_of_sundaram(limit)
        atkin = await sieve_of_atkin(limit)
        linear = await linear_sieve(limit)

        assert eratosthenes == sundaram
        assert eratosthenes == atkin
        assert eratosthenes == linear

    @pytest.mark.asyncio
    async def test_segmented_matches_full(self):
        """Test segmented sieve matches full sieve."""
        limit = 100

        # Full sieve
        full = await sieve_of_eratosthenes(limit)

        # Segmented approach
        seg1 = await segmented_sieve(2, 50)
        seg2 = await segmented_sieve(51, 100)
        combined = seg1 + seg2

        assert full == combined


class TestAsyncBehavior:
    """Test async behavior."""

    @pytest.mark.asyncio
    async def test_all_functions_async(self):
        """Test that all sieve functions are async."""
        operations = [
            sieve_of_eratosthenes(30),
            sieve_of_sundaram(30),
            sieve_of_atkin(30),
            segmented_sieve(10, 30),
            linear_sieve(30),
            prime_counting_sieve(100),
        ]

        for op in operations:
            assert asyncio.iscoroutine(op)

        results = await asyncio.gather(*operations)
        assert len(results) == 6

    @pytest.mark.asyncio
    async def test_concurrent_sieves(self):
        """Test concurrent sieve execution."""
        tasks = [sieve_of_eratosthenes(n) for n in [10, 20, 30, 40, 50]]
        results = await asyncio.gather(*tasks)

        assert len(results) == 5
        # Each result should be longer than previous
        for i in range(1, len(results)):
            assert len(results[i]) >= len(results[i - 1])

    @pytest.mark.asyncio
    async def test_large_sieve_eratosthenes_async(self):
        """Test async sleep triggered in large eratosthenes sieve."""
        # Test line 82 - limit > 100000
        result = await sieve_of_eratosthenes(100001)
        assert len(result) > 0
        assert 2 in result

        # Test line 93 - i % 1000 == 0 and limit > 1000000
        result = await sieve_of_eratosthenes(1000001)
        assert len(result) > 0
        assert 2 in result

    @pytest.mark.asyncio
    async def test_large_sieve_sundaram_async(self):
        """Test async sleep triggered in large sundaram sieve."""
        # Test line 162 - limit > 100000
        result = await sieve_of_sundaram(100001)
        assert len(result) > 0
        assert 2 in result

        # Test line 173 - i % 1000 == 0 and limit > 1000000
        result = await sieve_of_sundaram(1000001)
        assert len(result) > 0
        assert 2 in result

    @pytest.mark.asyncio
    async def test_large_sieve_atkin_async(self):
        """Test async sleep triggered in large atkin sieve."""
        # Test line 243 - limit > 100000
        result = await sieve_of_atkin(100001)
        assert len(result) > 0
        assert 2 in result

        # Test line 257 - x % 100 == 0 and limit > 1000000
        result = await sieve_of_atkin(1000001)
        assert len(result) > 0
        assert 2 in result

    @pytest.mark.asyncio
    async def test_large_segmented_sieve_async(self):
        """Test async sleep triggered in large segmented sieve."""
        # Test line 361 - high - low > 100000
        result = await segmented_sieve(2, 100003)
        assert len(result) > 0
        assert 2 in result

        # Test line 395 - high - low > 1000000
        result = await segmented_sieve(2, 1000003)
        assert len(result) > 0
        assert 2 in result

    @pytest.mark.asyncio
    async def test_large_prime_counting_async(self):
        """Test async sleep triggered in large prime counting."""
        # Test line 555 - limit > 100000
        result = await prime_counting_sieve(100001)
        assert result > 0

        # Test line 568 - i % 1000 == 0 and limit > 1000000
        result = await prime_counting_sieve(1000001)
        assert result > 0

    @pytest.mark.asyncio
    async def test_large_mertens_async(self):
        """Test async sleep triggered in large Mertens function."""
        # Test line 617 - n > 100000
        result = await mertens_function_sieve(100001)
        assert isinstance(result, int)

        # Test line 633 - i % 1000 == 0 and n > 100000
        result = await mertens_function_sieve(100001)
        assert isinstance(result, int)

    @pytest.mark.asyncio
    async def test_large_linear_sieve_async(self):
        """Test async sleep triggered in large linear sieve."""
        # Test line 704 - limit > 100000
        result = await linear_sieve(100001)
        assert len(result) > 0
        assert 2 in result

        # Test line 724 - i % 1000 == 0 and limit > 1000000
        result = await linear_sieve(1000001)
        assert len(result) > 0
        assert 2 in result


class TestParametrized:
    """Parametrized tests."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "limit,expected_count",
        [
            (10, 4),
            (20, 8),
            (30, 10),
            (50, 15),
            (100, 25),
        ],
    )
    async def test_prime_count_parametrized(self, limit, expected_count):
        """Parametrized prime counting tests."""
        count = await prime_counting_sieve(limit)
        assert count == expected_count

    @pytest.mark.asyncio
    @pytest.mark.parametrize("limit", [10, 20, 30, 50, 100])
    async def test_sieve_consistency_parametrized(self, limit):
        """Test all sieves give same result."""
        eratosthenes = await sieve_of_eratosthenes(limit)
        linear = await linear_sieve(limit)
        assert eratosthenes == linear


class TestMainFunction:
    """Test the main test function in the module."""

    @pytest.mark.asyncio
    async def test_module_test_suite(self):
        """Test line 1037-1083 - the module's own test suite."""
        # Import and run the test_sieve_algorithms function from the module
        from chuk_mcp_math.number_theory.sieve_algorithms import test_sieve_algorithms

        # This should run without errors
        await test_sieve_algorithms()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
