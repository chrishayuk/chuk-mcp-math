#!/usr/bin/env python3
# tests/math/test_sequences.py
"""
Comprehensive pytest unit tests for sequences and series functions.

Tests cover:
- Arithmetic sequences and sums
- Geometric sequences and sums (finite and infinite)
- Special sequences (triangular, square, cube numbers)
- Series calculations (harmonic, power series)
- Pattern analysis (differences, arithmetic/geometric detection)
- Normal operation cases
- Edge cases (zero, negative, empty)
- Error conditions
- Async behavior
- Mathematical relationships
"""

import pytest
import asyncio

# Import the functions to test
from chuk_mcp_math.sequences import (
    arithmetic_sequence,
    arithmetic_sum,
    geometric_sequence,
    geometric_sum,
    triangular_numbers,
    square_numbers,
    cube_numbers,
    harmonic_series,
    power_series_sum,
    find_differences,
    is_arithmetic,
    is_geometric,
)


# Arithmetic Sequence Tests
class TestArithmeticSequence:
    """Test cases for arithmetic_sequence function."""

    @pytest.mark.asyncio
    async def test_arithmetic_sequence_positive_diff(self):
        """Test arithmetic sequence with positive common difference."""
        result = await arithmetic_sequence(2, 3, 5)
        assert result == [2, 5, 8, 11, 14]

    @pytest.mark.asyncio
    async def test_arithmetic_sequence_negative_diff(self):
        """Test arithmetic sequence with negative common difference."""
        result = await arithmetic_sequence(10, -2, 4)
        assert result == [10, 8, 6, 4]

    @pytest.mark.asyncio
    async def test_arithmetic_sequence_zero_diff(self):
        """Test arithmetic sequence with zero difference (constant)."""
        result = await arithmetic_sequence(5, 0, 4)
        assert result == [5, 5, 5, 5]

    @pytest.mark.asyncio
    async def test_arithmetic_sequence_natural_numbers(self):
        """Test generating natural numbers."""
        result = await arithmetic_sequence(0, 1, 6)
        assert result == [0, 1, 2, 3, 4, 5]

    @pytest.mark.asyncio
    async def test_arithmetic_sequence_single_term(self):
        """Test arithmetic sequence with single term."""
        result = await arithmetic_sequence(7, 3, 1)
        assert result == [7]

    @pytest.mark.asyncio
    async def test_arithmetic_sequence_invalid_count(self):
        """Test that non-positive count raises error."""
        with pytest.raises(ValueError, match="Number of terms must be positive"):
            await arithmetic_sequence(1, 1, 0)
        with pytest.raises(ValueError, match="Number of terms must be positive"):
            await arithmetic_sequence(1, 1, -5)


class TestArithmeticSum:
    """Test cases for arithmetic_sum function."""

    @pytest.mark.asyncio
    async def test_arithmetic_sum_standard(self):
        """Test arithmetic sum."""
        result = await arithmetic_sum(2, 3, 5)
        assert result == 40  # 2 + 5 + 8 + 11 + 14

    @pytest.mark.asyncio
    async def test_arithmetic_sum_natural_numbers(self):
        """Test sum of first 10 natural numbers."""
        result = await arithmetic_sum(1, 1, 10)
        assert result == 55  # 1+2+...+10

    @pytest.mark.asyncio
    async def test_arithmetic_sum_constant_sequence(self):
        """Test sum of constant sequence."""
        result = await arithmetic_sum(5, 0, 4)
        assert result == 20  # 5+5+5+5

    @pytest.mark.asyncio
    async def test_arithmetic_sum_decreasing(self):
        """Test sum of decreasing sequence."""
        result = await arithmetic_sum(10, -1, 5)
        assert result == 40  # 10+9+8+7+6

    @pytest.mark.asyncio
    async def test_arithmetic_sum_single_term(self):
        """Test sum with single term."""
        result = await arithmetic_sum(7, 3, 1)
        assert result == 7

    @pytest.mark.asyncio
    async def test_arithmetic_sum_invalid_count(self):
        """Test that non-positive count raises error."""
        with pytest.raises(ValueError, match="Number of terms must be positive"):
            await arithmetic_sum(1, 1, 0)


# Geometric Sequence Tests
class TestGeometricSequence:
    """Test cases for geometric_sequence function."""

    @pytest.mark.asyncio
    async def test_geometric_sequence_ratio_3(self):
        """Test geometric sequence with ratio 3."""
        result = await geometric_sequence(2, 3, 5)
        assert result == [2, 6, 18, 54, 162]

    @pytest.mark.asyncio
    async def test_geometric_sequence_fractional_ratio(self):
        """Test geometric sequence with fractional ratio."""
        result = await geometric_sequence(1, 0.5, 4)
        assert result == [1, 0.5, 0.25, 0.125]

    @pytest.mark.asyncio
    async def test_geometric_sequence_ratio_1(self):
        """Test geometric sequence with ratio 1 (constant)."""
        result = await geometric_sequence(5, 1, 3)
        assert result == [5, 5, 5]

    @pytest.mark.asyncio
    async def test_geometric_sequence_negative_ratio(self):
        """Test geometric sequence with negative ratio (alternating)."""
        result = await geometric_sequence(1, -2, 4)
        assert result == [1, -2, 4, -8]

    @pytest.mark.asyncio
    async def test_geometric_sequence_powers_of_2(self):
        """Test powers of 2."""
        result = await geometric_sequence(1, 2, 5)
        assert result == [1, 2, 4, 8, 16]

    @pytest.mark.asyncio
    async def test_geometric_sequence_invalid_count(self):
        """Test that non-positive count raises error."""
        with pytest.raises(ValueError, match="Number of terms must be positive"):
            await geometric_sequence(1, 2, 0)


class TestGeometricSum:
    """Test cases for geometric_sum function."""

    @pytest.mark.asyncio
    async def test_geometric_sum_finite(self):
        """Test finite geometric sum."""
        result = await geometric_sum(2, 3, 5)
        assert result == 242  # 2 + 6 + 18 + 54 + 162

    @pytest.mark.asyncio
    async def test_geometric_sum_ratio_half(self):
        """Test geometric sum with ratio 0.5."""
        result = await geometric_sum(1, 0.5, 10)
        assert pytest.approx(result, rel=1e-6) == 1.998046875

    @pytest.mark.asyncio
    async def test_geometric_sum_ratio_1(self):
        """Test geometric sum with ratio 1."""
        result = await geometric_sum(3, 1, 5)
        assert result == 15  # 3+3+3+3+3

    @pytest.mark.asyncio
    async def test_geometric_sum_infinite_convergent(self):
        """Test infinite geometric series (|r| < 1)."""
        result = await geometric_sum(1, 0.5, None)
        assert result == 2.0  # 1/(1-0.5) = 2

    @pytest.mark.asyncio
    async def test_geometric_sum_infinite_divergent_error(self):
        """Test that infinite series with |r| >= 1 raises error."""
        with pytest.raises(ValueError, match="Infinite geometric series only converges"):
            await geometric_sum(1, 2, None)
        with pytest.raises(ValueError, match="Infinite geometric series only converges"):
            await geometric_sum(1, 1, None)
        with pytest.raises(ValueError, match="Infinite geometric series only converges"):
            await geometric_sum(1, -1, None)

    @pytest.mark.asyncio
    async def test_geometric_sum_invalid_count(self):
        """Test that non-positive count raises error."""
        with pytest.raises(ValueError, match="Number of terms must be positive"):
            await geometric_sum(1, 2, 0)


# Special Sequences Tests
class TestTriangularNumbers:
    """Test cases for triangular_numbers function."""

    @pytest.mark.asyncio
    async def test_triangular_numbers_standard(self):
        """Test first 5 triangular numbers."""
        result = await triangular_numbers(5)
        assert result == [1, 3, 6, 10, 15]

    @pytest.mark.asyncio
    async def test_triangular_numbers_first(self):
        """Test first triangular number."""
        result = await triangular_numbers(1)
        assert result == [1]

    @pytest.mark.asyncio
    async def test_triangular_numbers_eight(self):
        """Test first 8 triangular numbers."""
        result = await triangular_numbers(8)
        assert result == [1, 3, 6, 10, 15, 21, 28, 36]

    @pytest.mark.asyncio
    async def test_triangular_numbers_zero(self):
        """Test with n=0 returns empty list."""
        result = await triangular_numbers(0)
        assert result == []

    @pytest.mark.asyncio
    async def test_triangular_numbers_negative_error(self):
        """Test that negative n raises error."""
        with pytest.raises(ValueError, match="n must be non-negative"):
            await triangular_numbers(-1)


class TestSquareNumbers:
    """Test cases for square_numbers function."""

    @pytest.mark.asyncio
    async def test_square_numbers_standard(self):
        """Test first 5 square numbers."""
        result = await square_numbers(5)
        assert result == [1, 4, 9, 16, 25]

    @pytest.mark.asyncio
    async def test_square_numbers_first(self):
        """Test first square number."""
        result = await square_numbers(1)
        assert result == [1]

    @pytest.mark.asyncio
    async def test_square_numbers_seven(self):
        """Test first 7 square numbers."""
        result = await square_numbers(7)
        assert result == [1, 4, 9, 16, 25, 36, 49]

    @pytest.mark.asyncio
    async def test_square_numbers_zero(self):
        """Test with n=0 returns empty list."""
        result = await square_numbers(0)
        assert result == []

    @pytest.mark.asyncio
    async def test_square_numbers_negative_error(self):
        """Test that negative n raises error."""
        with pytest.raises(ValueError, match="n must be non-negative"):
            await square_numbers(-1)


class TestCubeNumbers:
    """Test cases for cube_numbers function."""

    @pytest.mark.asyncio
    async def test_cube_numbers_standard(self):
        """Test first 5 cube numbers."""
        result = await cube_numbers(5)
        assert result == [1, 8, 27, 64, 125]

    @pytest.mark.asyncio
    async def test_cube_numbers_first(self):
        """Test first cube number."""
        result = await cube_numbers(1)
        assert result == [1]

    @pytest.mark.asyncio
    async def test_cube_numbers_six(self):
        """Test first 6 cube numbers."""
        result = await cube_numbers(6)
        assert result == [1, 8, 27, 64, 125, 216]

    @pytest.mark.asyncio
    async def test_cube_numbers_zero(self):
        """Test with n=0 returns empty list."""
        result = await cube_numbers(0)
        assert result == []

    @pytest.mark.asyncio
    async def test_cube_numbers_negative_error(self):
        """Test that negative n raises error."""
        with pytest.raises(ValueError, match="n must be non-negative"):
            await cube_numbers(-1)


# Series Tests
class TestHarmonicSeries:
    """Test cases for harmonic_series function."""

    @pytest.mark.asyncio
    async def test_harmonic_series_h1(self):
        """Test H₁ = 1."""
        result = await harmonic_series(1)
        assert result == 1.0

    @pytest.mark.asyncio
    async def test_harmonic_series_h2(self):
        """Test H₂ = 1.5."""
        result = await harmonic_series(2)
        assert result == 1.5

    @pytest.mark.asyncio
    async def test_harmonic_series_h4(self):
        """Test H₄ = 1 + 1/2 + 1/3 + 1/4."""
        result = await harmonic_series(4)
        expected = 1 + 0.5 + 1 / 3 + 0.25
        assert pytest.approx(result, rel=1e-10) == expected

    @pytest.mark.asyncio
    async def test_harmonic_series_h10(self):
        """Test H₁₀."""
        result = await harmonic_series(10)
        expected = sum(1 / i for i in range(1, 11))
        assert pytest.approx(result, rel=1e-10) == expected

    @pytest.mark.asyncio
    async def test_harmonic_series_invalid_n(self):
        """Test that n <= 0 raises error."""
        with pytest.raises(ValueError, match="n must be positive"):
            await harmonic_series(0)
        with pytest.raises(ValueError, match="n must be positive"):
            await harmonic_series(-1)


class TestPowerSeriesSum:
    """Test cases for power_series_sum function."""

    @pytest.mark.asyncio
    async def test_power_series_sum_x2_n4(self):
        """Test 1 + 2 + 4 + 8 + 16 = 31."""
        result = await power_series_sum(2, 4)
        assert result == 31

    @pytest.mark.asyncio
    async def test_power_series_sum_x_half(self):
        """Test 1 + 0.5 + 0.25 + 0.125 = 1.875."""
        result = await power_series_sum(0.5, 3)
        assert pytest.approx(result, rel=1e-10) == 1.875

    @pytest.mark.asyncio
    async def test_power_series_sum_x1(self):
        """Test x=1: 1 + 1 + 1 + 1 + 1 + 1 = 6."""
        result = await power_series_sum(1, 5)
        assert result == 6

    @pytest.mark.asyncio
    async def test_power_series_sum_x_negative(self):
        """Test x=-1: 1 - 1 + 1 - 1 = 0."""
        result = await power_series_sum(-1, 3)
        assert result == 0

    @pytest.mark.asyncio
    async def test_power_series_sum_n0(self):
        """Test n=0: just x⁰ = 1."""
        result = await power_series_sum(5, 0)
        assert result == 1

    @pytest.mark.asyncio
    async def test_power_series_sum_negative_n_error(self):
        """Test that negative n raises error."""
        with pytest.raises(ValueError, match="n must be non-negative"):
            await power_series_sum(2, -1)


# Pattern Analysis Tests
class TestFindDifferences:
    """Test cases for find_differences function."""

    @pytest.mark.asyncio
    async def test_find_differences_square_numbers(self):
        """Test differences in square numbers."""
        result = await find_differences([1, 4, 9, 16, 25])
        assert result == [3, 5, 7, 9]

    @pytest.mark.asyncio
    async def test_find_differences_arithmetic(self):
        """Test differences in arithmetic sequence."""
        result = await find_differences([2, 5, 8, 11, 14])
        assert result == [3, 3, 3, 3]

    @pytest.mark.asyncio
    async def test_find_differences_powers_of_2(self):
        """Test differences in powers of 2."""
        result = await find_differences([1, 2, 4, 8, 16])
        assert result == [1, 2, 4, 8]

    @pytest.mark.asyncio
    async def test_find_differences_single_element(self):
        """Test single element returns empty list."""
        result = await find_differences([10])
        assert result == []

    @pytest.mark.asyncio
    async def test_find_differences_two_elements(self):
        """Test two elements returns single difference."""
        result = await find_differences([5, 8])
        assert result == [3]

    @pytest.mark.asyncio
    async def test_find_differences_negative_numbers(self):
        """Test differences with negative numbers."""
        result = await find_differences([-5, -2, 1, 4])
        assert result == [3, 3, 3]


class TestIsArithmetic:
    """Test cases for is_arithmetic function."""

    @pytest.mark.asyncio
    async def test_is_arithmetic_true(self):
        """Test arithmetic sequence detection."""
        assert await is_arithmetic([2, 5, 8, 11, 14]) is True

    @pytest.mark.asyncio
    async def test_is_arithmetic_constant(self):
        """Test constant sequence is arithmetic."""
        assert await is_arithmetic([10, 10, 10]) is True

    @pytest.mark.asyncio
    async def test_is_arithmetic_false(self):
        """Test non-arithmetic sequence."""
        assert await is_arithmetic([1, 4, 9, 16, 25]) is False

    @pytest.mark.asyncio
    async def test_is_arithmetic_single_element(self):
        """Test single element is trivially arithmetic."""
        assert await is_arithmetic([5]) is True

    @pytest.mark.asyncio
    async def test_is_arithmetic_two_elements(self):
        """Test two elements is always arithmetic."""
        assert await is_arithmetic([3, 7]) is True

    @pytest.mark.asyncio
    async def test_is_arithmetic_empty(self):
        """Test empty sequence is trivially arithmetic."""
        assert await is_arithmetic([]) is True

    @pytest.mark.asyncio
    async def test_is_arithmetic_with_floats(self):
        """Test arithmetic sequence with floating point numbers."""
        assert await is_arithmetic([1.5, 2.5, 3.5, 4.5]) is True


class TestIsGeometric:
    """Test cases for is_geometric function."""

    @pytest.mark.asyncio
    async def test_is_geometric_true(self):
        """Test geometric sequence detection."""
        assert await is_geometric([2, 6, 18, 54]) is True

    @pytest.mark.asyncio
    async def test_is_geometric_constant(self):
        """Test constant sequence is geometric (ratio 1)."""
        assert await is_geometric([5, 5, 5]) is True

    @pytest.mark.asyncio
    async def test_is_geometric_false(self):
        """Test non-geometric sequence."""
        assert await is_geometric([1, 2, 4, 7]) is False

    @pytest.mark.asyncio
    async def test_is_geometric_alternating(self):
        """Test alternating geometric sequence (negative ratio)."""
        assert await is_geometric([1, -2, 4, -8]) is True

    @pytest.mark.asyncio
    async def test_is_geometric_single_element(self):
        """Test single element is trivially geometric."""
        assert await is_geometric([5]) is True

    @pytest.mark.asyncio
    async def test_is_geometric_two_elements(self):
        """Test two elements is always geometric."""
        assert await is_geometric([3, 9]) is True

    @pytest.mark.asyncio
    async def test_is_geometric_empty(self):
        """Test empty sequence is trivially geometric."""
        assert await is_geometric([]) is True

    @pytest.mark.asyncio
    async def test_is_geometric_with_zero(self):
        """Test sequence with zero is not geometric (except if all zeros after first)."""
        assert await is_geometric([2, 0, 0]) is False


# Mathematical Relationships Tests
class TestMathematicalRelationships:
    """Test mathematical relationships between sequences."""

    @pytest.mark.asyncio
    async def test_arithmetic_sequence_matches_sum(self):
        """Test that summing arithmetic sequence matches arithmetic_sum."""
        seq = await arithmetic_sequence(2, 3, 5)
        manual_sum = sum(seq)
        formula_sum = await arithmetic_sum(2, 3, 5)
        assert manual_sum == formula_sum

    @pytest.mark.asyncio
    async def test_geometric_sequence_matches_sum(self):
        """Test that summing geometric sequence matches geometric_sum."""
        seq = await geometric_sequence(2, 3, 5)
        manual_sum = sum(seq)
        formula_sum = await geometric_sum(2, 3, 5)
        assert manual_sum == formula_sum

    @pytest.mark.asyncio
    async def test_triangular_number_formula(self):
        """Test triangular number formula: T_n = n(n+1)/2."""
        result = await triangular_numbers(10)
        for i, val in enumerate(result, 1):
            assert val == i * (i + 1) // 2

    @pytest.mark.asyncio
    async def test_square_number_formula(self):
        """Test square number formula: S_n = n²."""
        result = await square_numbers(10)
        for i, val in enumerate(result, 1):
            assert val == i * i

    @pytest.mark.asyncio
    async def test_cube_number_formula(self):
        """Test cube number formula: C_n = n³."""
        result = await cube_numbers(10)
        for i, val in enumerate(result, 1):
            assert val == i**3


# Async Behavior Tests
class TestAsyncBehavior:
    """Test async behavior of sequence functions."""

    @pytest.mark.asyncio
    async def test_all_functions_are_async(self):
        """Test that all sequence functions are properly async."""
        operations = [
            arithmetic_sequence(2, 3, 5),
            arithmetic_sum(2, 3, 5),
            geometric_sequence(2, 3, 4),
            geometric_sum(2, 3, 4),
            triangular_numbers(5),
            square_numbers(5),
            cube_numbers(5),
            harmonic_series(4),
            power_series_sum(2, 4),
            find_differences([1, 2, 3]),
            is_arithmetic([2, 5, 8]),
            is_geometric([2, 6, 18]),
        ]

        # Ensure all are coroutines
        for op in operations:
            assert asyncio.iscoroutine(op)

        # Run all operations concurrently
        results = await asyncio.gather(*operations)
        assert len(results) == len(operations)

    @pytest.mark.asyncio
    async def test_concurrent_sequence_generation(self):
        """Test concurrent sequence generation."""
        import time

        start = time.time()

        # Run multiple sequence generations concurrently
        tasks = [arithmetic_sequence(i, 1, 10) for i in range(100)]
        await asyncio.gather(*tasks)

        duration = time.time() - start

        # Should complete quickly
        assert duration < 1.0


# Parametrized Tests
class TestParametrized:
    """Parametrized tests for comprehensive coverage."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "first,diff,count,expected",
        [
            (0, 1, 5, [0, 1, 2, 3, 4]),
            (5, 2, 3, [5, 7, 9]),
            (10, -1, 4, [10, 9, 8, 7]),
        ],
    )
    async def test_arithmetic_sequence_parametrized(self, first, diff, count, expected):
        """Parametrized test for arithmetic sequences."""
        result = await arithmetic_sequence(first, diff, count)
        assert result == expected

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "sequence,expected",
        [
            ([2, 5, 8, 11], True),
            ([1, 4, 9, 16], False),
            ([10, 10, 10], True),
            ([1, 2, 4, 8], False),
        ],
    )
    async def test_is_arithmetic_parametrized(self, sequence, expected):
        """Parametrized test for arithmetic detection."""
        result = await is_arithmetic(sequence)
        assert result is expected

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "sequence,expected",
        [
            ([2, 6, 18, 54], True),
            ([1, 2, 4, 7], False),
            ([5, 5, 5], True),
            ([1, -2, 4, -8], True),
        ],
    )
    async def test_is_geometric_parametrized(self, sequence, expected):
        """Parametrized test for geometric detection."""
        result = await is_geometric(sequence)
        assert result is expected


# Edge Cases Tests
class TestEdgeCases:
    """Test edge cases for sequence functions."""

    @pytest.mark.asyncio
    async def test_large_sequence_generation(self):
        """Test generating large sequences."""
        result = await arithmetic_sequence(1, 1, 1000)
        assert len(result) == 1000
        assert result[0] == 1
        assert result[-1] == 1000

    @pytest.mark.asyncio
    async def test_infinite_geometric_series_edge_case(self):
        """Test infinite geometric series at boundary |r| = 0.9999."""
        result = await geometric_sum(1, 0.9999, None)
        # Should converge since |r| < 1
        assert result > 0
        assert result == pytest.approx(1 / (1 - 0.9999), rel=1e-10)

    @pytest.mark.asyncio
    async def test_very_large_arithmetic_sequence(self):
        """Test arithmetic sequence with > 1000 terms (covers asyncio.sleep lines 78, 85)."""
        result = await arithmetic_sequence(1, 1, 2500)
        assert len(result) == 2500
        assert result[0] == 1
        assert result[-1] == 2500
        # Verify it's still correct at key points
        assert result[1000] == 1001
        assert result[1999] == 2000

    @pytest.mark.asyncio
    async def test_very_large_geometric_sequence(self):
        """Test geometric sequence with > 1000 terms (covers asyncio.sleep lines 197, 207)."""
        result = await geometric_sequence(1, 1.001, 2500)
        assert len(result) == 2500
        assert result[0] == 1
        # Just verify length and first term, as values get very large

    @pytest.mark.asyncio
    async def test_very_large_triangular_numbers(self):
        """Test triangular numbers with > 1000 terms (covers asyncio.sleep lines 318, 325)."""
        result = await triangular_numbers(2500)
        assert len(result) == 2500
        assert result[0] == 1
        assert result[999] == 1000 * 1001 // 2
        assert result[-1] == 2500 * 2501 // 2

    @pytest.mark.asyncio
    async def test_very_large_square_numbers(self):
        """Test square numbers with > 1000 terms (covers asyncio.sleep lines 370, 377)."""
        result = await square_numbers(2500)
        assert len(result) == 2500
        assert result[0] == 1
        assert result[999] == 1000 * 1000
        assert result[-1] == 2500 * 2500

    @pytest.mark.asyncio
    async def test_very_large_cube_numbers(self):
        """Test cube numbers with > 1000 terms (covers asyncio.sleep lines 422, 429)."""
        result = await cube_numbers(2500)
        assert len(result) == 2500
        assert result[0] == 1
        assert result[999] == 1000**3
        assert result[-1] == 2500**3

    @pytest.mark.asyncio
    async def test_very_large_harmonic_series(self):
        """Test harmonic series with > 10000 terms (covers asyncio.sleep lines 474, 481)."""
        result = await harmonic_series(25000)
        # Harmonic series grows slowly
        assert result > 0
        # H_25000 should be around 10.98 (ln(25000) + gamma)
        assert result > 10

    @pytest.mark.asyncio
    async def test_very_large_find_differences(self):
        """Test find_differences with > 1000 elements (covers asyncio.sleep lines 585, 592)."""
        # Create a large arithmetic sequence and find differences
        large_seq = list(range(2500))
        result = await find_differences(large_seq)
        assert len(result) == 2499
        # All differences should be 1
        assert all(d == 1 for d in result)

    @pytest.mark.asyncio
    async def test_is_arithmetic_empty_differences(self):
        """Test is_arithmetic when differences list is empty (covers line 646)."""
        # This happens with single element or empty sequences
        # Already covered by other tests, but let's be explicit
        result = await is_arithmetic([])
        assert result is True
        result = await is_arithmetic([42])
        assert result is True

    @pytest.mark.asyncio
    async def test_is_geometric_zero_first_term_all_zeros(self):
        """Test is_geometric with first term zero and all zeros (covers line 705)."""
        # According to the logic, if any term before index i is zero, it returns False
        # So [0, 0, 0, 0] will return False because at i=1, sequence[0]=0
        result = await is_geometric([0, 0, 0, 0])
        assert result is False

    @pytest.mark.asyncio
    async def test_is_geometric_first_term_zero_special_case(self):
        """Test is_geometric when only first term is zero but later check (line 703-705)."""
        # To truly cover line 705, we need a case where first term is zero
        # but we get past line 699-701 check. However, that check will always
        # trigger if there's a zero before any element. So line 705 is actually
        # unreachable with the current logic. Let's test what happens.
        # If we have just [0], it returns True before reaching line 703
        result = await is_geometric([0])
        assert result is True

    @pytest.mark.asyncio
    async def test_is_geometric_zero_first_term_nonzero_after(self):
        """Test is_geometric with first term zero but nonzero after (covers line 705)."""
        result = await is_geometric([0, 1, 2, 3])
        assert result is False

    @pytest.mark.asyncio
    async def test_is_geometric_zero_in_middle(self):
        """Test is_geometric with zero in middle (covers line 711)."""
        # If any term (except possibly first) is preceded by zero, it's not geometric
        result = await is_geometric([5, 0, 3])
        assert result is False

    @pytest.mark.asyncio
    async def test_is_geometric_empty_ratios(self):
        """Test is_geometric when ratios list is empty (covers line 715)."""
        # This shouldn't happen with the current logic, but test single element case
        result = await is_geometric([5.5])
        assert result is True


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v", "--tb=short"])
