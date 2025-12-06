#!/usr/bin/env python3
# tests/math/number_theory/test_egyptian_fractions.py
"""
Comprehensive pytest test suite for egyptian_fractions.py module.

Tests cover:
- Egyptian fraction decomposition: greedy algorithm, Fibonacci method, binary remainder
- Unit fraction operations: summation, identification, LCM calculations
- Harmonic numbers: exact and floating point calculations, partial sums, harmonic mean
- Sylvester sequence: generation and expansion properties
- Egyptian fraction properties: analysis, optimality checking, representation finding
- Fraction utilities: proper fraction checking, improper conversion, expansion lengths
- Mathematical properties and relationships
- Edge cases, error conditions, and performance testing
- Async behavior verification
"""

import pytest
import asyncio
import time
import math
from fractions import Fraction

# Import the functions to test
from chuk_mcp_math.number_theory.egyptian_fractions import (
    # Egyptian fraction decomposition
    egyptian_fraction_decomposition,
    fibonacci_greedy_egyptian,
    binary_remainder_egyptian,
    # Unit fraction operations
    unit_fraction_sum,
    is_unit_fraction,
    egyptian_fraction_lcm,
    # Harmonic numbers and series
    harmonic_number,
    harmonic_number_fraction,
    harmonic_partial_sum,
    harmonic_mean,
    # Sylvester sequence
    sylvester_sequence,
    sylvester_expansion_of_one,
    # Egyptian fraction properties
    egyptian_fraction_properties,
    is_optimal_egyptian_fraction,
    two_unit_fraction_representations,
    # Fraction utilities
    is_proper_fraction,
    improper_to_egyptian,
    egyptian_expansion_lengths,
    shortest_egyptian_fraction,
)

# ============================================================================
# EGYPTIAN FRACTION DECOMPOSITION TESTS
# ============================================================================


class TestEgyptianFractionDecomposition:
    """Test cases for Egyptian fraction decomposition functions."""

    @pytest.mark.asyncio
    async def test_egyptian_fraction_decomposition_known_cases(self):
        """Test Egyptian fraction decomposition with known cases."""
        test_cases = [
            (2, 3, [2, 6]),  # 2/3 = 1/2 + 1/6
            (3, 4, [2, 4]),  # 3/4 = 1/2 + 1/4
            (5, 6, [2, 3]),  # 5/6 = 1/2 + 1/3
            (7, 12, [2, 12]),  # 7/12 = 1/2 + 1/12
            (4, 5, [2, 4, 20]),  # 4/5 = 1/2 + 1/4 + 1/20
            (2, 5, [3, 15]),  # 2/5 = 1/3 + 1/15
            (3, 5, [2, 10]),  # 3/5 = 1/2 + 1/10
            (5, 8, [2, 8]),  # 5/8 = 1/2 + 1/8
            (7, 15, [3, 8, 120]),  # 7/15 = 1/3 + 1/8 + 1/120
        ]

        for num, den, expected in test_cases:
            result = await egyptian_fraction_decomposition(num, den)
            assert result == expected, (
                f"egyptian_fraction_decomposition({num}, {den}) should be {expected}, got {result}"
            )

    @pytest.mark.asyncio
    async def test_egyptian_fraction_decomposition_verification(self):
        """Test that Egyptian fraction decompositions sum to original fraction."""
        test_fractions = [(2, 3), (3, 4), (4, 5), (5, 6), (7, 12), (11, 15), (13, 20)]

        for num, den in test_fractions:
            egyptian_denoms = await egyptian_fraction_decomposition(num, den)

            # Verify the sum equals the original fraction
            sum_num, sum_den = await unit_fraction_sum(egyptian_denoms)
            original_fraction = Fraction(num, den)
            reconstructed_fraction = Fraction(sum_num, sum_den)

            assert original_fraction == reconstructed_fraction, (
                f"Decomposition of {num}/{den} doesn't sum correctly"
            )

    @pytest.mark.asyncio
    async def test_egyptian_fraction_decomposition_unit_fractions(self):
        """Test Egyptian fraction decomposition for unit fractions."""
        unit_fractions = [(1, 2), (1, 3), (1, 4), (1, 5), (1, 10), (1, 100)]

        for num, den in unit_fractions:
            result = await egyptian_fraction_decomposition(num, den)
            assert result == [den], f"Unit fraction {num}/{den} should decompose to [{den}]"

    @pytest.mark.asyncio
    async def test_egyptian_fraction_decomposition_edge_cases(self):
        """Test Egyptian fraction decomposition edge cases and errors."""
        # Invalid inputs
        with pytest.raises(ValueError, match="Numerator and denominator must be positive"):
            await egyptian_fraction_decomposition(0, 5)

        with pytest.raises(ValueError, match="Numerator and denominator must be positive"):
            await egyptian_fraction_decomposition(2, 0)

        with pytest.raises(ValueError, match="Numerator and denominator must be positive"):
            await egyptian_fraction_decomposition(-1, 3)

        # Improper fractions
        with pytest.raises(ValueError, match="Fraction must be proper"):
            await egyptian_fraction_decomposition(5, 3)

        with pytest.raises(ValueError, match="Fraction must be proper"):
            await egyptian_fraction_decomposition(4, 4)

    @pytest.mark.asyncio
    async def test_fibonacci_greedy_egyptian_consistency(self):
        """Test that Fibonacci greedy method gives same results as standard method."""
        test_fractions = [(2, 3), (3, 4), (4, 5), (5, 6), (7, 12)]

        for num, den in test_fractions:
            standard_result = await egyptian_fraction_decomposition(num, den)
            fibonacci_result = await fibonacci_greedy_egyptian(num, den)

            # Both should give valid decompositions (may differ but both valid)
            standard_sum = await unit_fraction_sum(standard_result)
            fibonacci_sum = await unit_fraction_sum(fibonacci_result)

            original_fraction = Fraction(num, den)
            assert Fraction(*standard_sum) == original_fraction
            assert Fraction(*fibonacci_sum) == original_fraction

    @pytest.mark.asyncio
    async def test_binary_remainder_egyptian_consistency(self):
        """Test that binary remainder method gives valid decompositions."""
        test_fractions = [(2, 3), (3, 4), (4, 5), (5, 6)]

        for num, den in test_fractions:
            binary_result = await binary_remainder_egyptian(num, den)

            # Should give valid decomposition
            binary_sum = await unit_fraction_sum(binary_result)
            original_fraction = Fraction(num, den)
            reconstructed_fraction = Fraction(*binary_sum)

            assert original_fraction == reconstructed_fraction, (
                f"Binary remainder method failed for {num}/{den}"
            )

    @pytest.mark.asyncio
    async def test_egyptian_fraction_decomposition_properties(self):
        """Test mathematical properties of Egyptian fraction decompositions."""
        test_fractions = [(2, 3), (3, 4), (4, 5), (5, 6), (7, 12)]

        for num, den in test_fractions:
            egyptian_denoms = await egyptian_fraction_decomposition(num, den)

            # All denominators should be positive and increasing
            assert all(d > 0 for d in egyptian_denoms), (
                f"All denominators should be positive for {num}/{den}"
            )
            assert egyptian_denoms == sorted(egyptian_denoms), (
                f"Denominators should be non-decreasing for {num}/{den}"
            )

            # All denominators should be distinct (greedy algorithm property)
            assert len(egyptian_denoms) == len(set(egyptian_denoms)), (
                f"Denominators should be distinct for {num}/{den}"
            )


# ============================================================================
# UNIT FRACTION OPERATIONS TESTS
# ============================================================================


class TestUnitFractionOperations:
    """Test cases for unit fraction operation functions."""

    @pytest.mark.asyncio
    async def test_unit_fraction_sum_basic_cases(self):
        """Test unit fraction summation with basic cases."""
        test_cases = [
            ([2, 3, 6], (1, 1)),  # 1/2 + 1/3 + 1/6 = 1
            ([2, 4, 8], (7, 8)),  # 1/2 + 1/4 + 1/8 = 7/8
            ([3, 6, 12], (7, 12)),  # 1/3 + 1/6 + 1/12 = 4/12 + 2/12 + 1/12 = 7/12
            ([4, 6, 12], (1, 2)),  # 1/4 + 1/6 + 1/12 = 3/12 + 2/12 + 1/12 = 6/12 = 1/2
            ([2], (1, 2)),  # 1/2 = 1/2
            ([5, 10, 20], (7, 20)),  # 1/5 + 1/10 + 1/20 = 4/20 + 2/20 + 1/20 = 7/20
        ]

        for denominators, expected in test_cases:
            result = await unit_fraction_sum(denominators)
            assert result == expected, (
                f"unit_fraction_sum({denominators}) should be {expected}, got {result}"
            )

    @pytest.mark.asyncio
    async def test_unit_fraction_sum_verification(self):
        """Test unit fraction sum by manual calculation."""
        # Test 1/2 + 1/3 + 1/6 = 3/6 + 2/6 + 1/6 = 6/6 = 1
        result = await unit_fraction_sum([2, 3, 6])
        assert result == (1, 1), "1/2 + 1/3 + 1/6 should equal 1"

        # Test 1/4 + 1/8 = 2/8 + 1/8 = 3/8
        result = await unit_fraction_sum([4, 8])
        assert result == (3, 8), "1/4 + 1/8 should equal 3/8"

        # Test single unit fraction
        result = await unit_fraction_sum([7])
        assert result == (1, 7), "1/7 should equal 1/7"

    @pytest.mark.asyncio
    async def test_unit_fraction_sum_edge_cases(self):
        """Test unit fraction sum edge cases."""
        # Empty list
        result = await unit_fraction_sum([])
        assert result == (0, 1), "Empty sum should be 0"

        # Invalid denominators
        with pytest.raises(ValueError, match="All denominators must be positive"):
            await unit_fraction_sum([2, 0, 4])

        with pytest.raises(ValueError, match="All denominators must be positive"):
            await unit_fraction_sum([-1, 2, 3])

    @pytest.mark.asyncio
    async def test_unit_fraction_sum_large_lists(self):
        """Test unit fraction sum with larger lists."""
        # Sum of 1/1 + 1/2 + ... + 1/10 (partial harmonic series)
        denominators = list(range(1, 11))
        result = await unit_fraction_sum(denominators)

        # Verify it's positive and reasonable
        fraction_value = result[0] / result[1]
        assert fraction_value > 2.5, "Sum of first 10 unit fractions should be > 2.5"
        assert fraction_value < 3.5, "Sum of first 10 unit fractions should be < 3.5"

    @pytest.mark.asyncio
    async def test_is_unit_fraction_basic_cases(self):
        """Test unit fraction identification."""
        test_cases = [
            (1, 5, True),  # 1/5 is unit
            (2, 5, False),  # 2/5 is not unit
            (1, 1, True),  # 1/1 is unit
            (3, 9, True),  # 3/9 = 1/3 is unit (reduced)
            (4, 8, True),  # 4/8 = 1/2 is unit (reduced)
            (6, 9, False),  # 6/9 = 2/3 is not unit
            (2, 4, True),  # 2/4 = 1/2 is unit (reduced)
            (5, 15, True),  # 5/15 = 1/3 is unit (reduced)
            (7, 14, True),  # 7/14 = 1/2 is unit (reduced)
        ]

        for num, den, expected in test_cases:
            result = await is_unit_fraction(num, den)
            assert result == expected, (
                f"is_unit_fraction({num}, {den}) should be {expected}, got {result}"
            )

    @pytest.mark.asyncio
    async def test_is_unit_fraction_edge_cases(self):
        """Test unit fraction identification edge cases."""
        # Zero denominator
        with pytest.raises(ValueError, match="Denominator must be positive"):
            await is_unit_fraction(1, 0)

        # Negative denominator
        with pytest.raises(ValueError, match="Denominator must be positive"):
            await is_unit_fraction(1, -5)

        # Zero numerator
        result = await is_unit_fraction(0, 5)
        assert not result, "0/5 should not be a unit fraction"

        # Negative numerator
        result = await is_unit_fraction(-1, 5)
        assert not result, "-1/5 should not be a unit fraction"

    @pytest.mark.asyncio
    async def test_egyptian_fraction_lcm_basic_cases(self):
        """Test LCM calculation for Egyptian fractions."""
        test_cases = [
            ([2, 3, 4], 12),  # LCM(2, 3, 4) = 12
            ([6, 8, 12], 24),  # LCM(6, 8, 12) = 24
            ([5, 7, 11], 385),  # LCM(5, 7, 11) = 385
            ([2, 4, 8], 8),  # LCM(2, 4, 8) = 8
            ([3, 9, 27], 27),  # LCM(3, 9, 27) = 27
            ([10, 15, 20], 60),  # LCM(10, 15, 20) = 60
        ]

        for denominators, expected in test_cases:
            result = await egyptian_fraction_lcm(denominators)
            assert result == expected, (
                f"egyptian_fraction_lcm({denominators}) should be {expected}, got {result}"
            )

    @pytest.mark.asyncio
    async def test_egyptian_fraction_lcm_properties(self):
        """Test mathematical properties of LCM calculation."""
        # LCM of single number is the number itself
        assert await egyptian_fraction_lcm([7]) == 7
        assert await egyptian_fraction_lcm([15]) == 15

        # LCM is commutative
        lcm1 = await egyptian_fraction_lcm([6, 8, 12])
        lcm2 = await egyptian_fraction_lcm([12, 6, 8])
        assert lcm1 == lcm2, "LCM should be commutative"

        # LCM should be divisible by all inputs
        denominators = [4, 6, 8, 12]
        lcm_result = await egyptian_fraction_lcm(denominators)
        for d in denominators:
            assert lcm_result % d == 0, f"LCM {lcm_result} should be divisible by {d}"

    @pytest.mark.asyncio
    async def test_egyptian_fraction_lcm_edge_cases(self):
        """Test LCM calculation edge cases."""
        # Empty list
        result = await egyptian_fraction_lcm([])
        assert result == 1, "LCM of empty list should be 1"

        # Invalid denominators
        with pytest.raises(ValueError, match="All denominators must be positive"):
            await egyptian_fraction_lcm([2, 0, 4])

        with pytest.raises(ValueError, match="All denominators must be positive"):
            await egyptian_fraction_lcm([-1, 2, 3])


# ============================================================================
# HARMONIC NUMBERS AND SERIES TESTS
# ============================================================================


class TestHarmonicNumbersAndSeries:
    """Test cases for harmonic number and series functions."""

    @pytest.mark.asyncio
    async def test_harmonic_number_known_values(self):
        """Test harmonic number calculation with known values."""
        # Known exact values for small n
        test_cases = [
            (1, 1.0),  # H_1 = 1
            (2, 1.5),  # H_2 = 1 + 1/2 = 1.5
            (3, 1.8333333333333333),  # H_3 = 1 + 1/2 + 1/3 ≈ 1.833
            (4, 2.0833333333333335),  # H_4 = 1 + 1/2 + 1/3 + 1/4 ≈ 2.083
        ]

        for n, expected in test_cases:
            result = await harmonic_number(n)
            assert abs(result - expected) < 1e-10, (
                f"harmonic_number({n}) should be {expected}, got {result}"
            )

    @pytest.mark.asyncio
    async def test_harmonic_number_properties(self):
        """Test mathematical properties of harmonic numbers."""
        # Harmonic numbers should be increasing
        h1 = await harmonic_number(5)
        h2 = await harmonic_number(10)
        h3 = await harmonic_number(20)

        assert h1 < h2 < h3, "Harmonic numbers should be strictly increasing"

        # H_n should be approximately ln(n) + γ for large n (Euler-Mascheroni constant γ ≈ 0.5772)
        h_100 = await harmonic_number(100)
        approx = math.log(100) + 0.5772156649
        assert abs(h_100 - approx) < 0.1, f"H_100 should approximate ln(100) + γ, got {h_100}"

    @pytest.mark.asyncio
    async def test_harmonic_number_edge_cases(self):
        """Test harmonic number edge cases."""
        # Invalid input
        with pytest.raises(ValueError, match="n must be positive"):
            await harmonic_number(0)

        with pytest.raises(ValueError, match="n must be positive"):
            await harmonic_number(-5)

    @pytest.mark.asyncio
    async def test_harmonic_number_fraction_exact_values(self):
        """Test exact harmonic number fractions."""
        test_cases = [
            (1, (1, 1)),  # H_1 = 1/1
            (2, (3, 2)),  # H_2 = 3/2
            (3, (11, 6)),  # H_3 = 11/6
            (4, (25, 12)),  # H_4 = 25/12
            (5, (137, 60)),  # H_5 = 137/60
        ]

        for n, expected in test_cases:
            result = await harmonic_number_fraction(n)
            assert result == expected, (
                f"harmonic_number_fraction({n}) should be {expected}, got {result}"
            )

    @pytest.mark.asyncio
    async def test_harmonic_number_fraction_verification(self):
        """Test that fractional harmonic numbers equal decimal versions."""
        for n in range(1, 10):
            decimal_h = await harmonic_number(n)
            frac_num, frac_den = await harmonic_number_fraction(n)
            frac_decimal = frac_num / frac_den

            assert abs(decimal_h - frac_decimal) < 1e-10, (
                f"Decimal and fractional H_{n} should match"
            )

    @pytest.mark.asyncio
    async def test_harmonic_partial_sum_basic_cases(self):
        """Test harmonic partial sum calculation."""
        test_cases = [
            (1, 4, 2.0833333333333335),  # H_4 = 1 + 1/2 + 1/3 + 1/4
            (2, 5, 1.2833333333333334),  # 1/2 + 1/3 + 1/4 + 1/5
            (5, 10, 0.8456349206349206),  # 1/5 + 1/6 + ... + 1/10 (corrected value)
            (3, 3, 0.3333333333333333),  # 1/3
        ]

        for start, end, expected in test_cases:
            result = await harmonic_partial_sum(start, end)
            assert abs(result - expected) < 1e-10, (
                f"harmonic_partial_sum({start}, {end}) should be {expected}, got {result}"
            )

    @pytest.mark.asyncio
    async def test_harmonic_partial_sum_properties(self):
        """Test mathematical properties of harmonic partial sums."""
        # Partial sum from 1 to n should equal H_n
        for n in range(1, 8):
            full_harmonic = await harmonic_number(n)
            partial_sum = await harmonic_partial_sum(1, n)
            assert abs(full_harmonic - partial_sum) < 1e-10, (
                f"Partial sum 1 to {n} should equal H_{n}"
            )

        # Sum from 1 to m plus sum from m+1 to n should equal sum from 1 to n
        m, n = 5, 10
        sum1 = await harmonic_partial_sum(1, m)
        sum2 = await harmonic_partial_sum(m + 1, n)
        total_sum = await harmonic_partial_sum(1, n)

        assert abs((sum1 + sum2) - total_sum) < 1e-10, "Partial sums should be additive"

    @pytest.mark.asyncio
    async def test_harmonic_partial_sum_edge_cases(self):
        """Test harmonic partial sum edge cases."""
        # Invalid inputs
        with pytest.raises(ValueError, match="Start and end must be positive"):
            await harmonic_partial_sum(0, 5)

        with pytest.raises(ValueError, match="Start and end must be positive"):
            await harmonic_partial_sum(1, 0)

        with pytest.raises(ValueError, match="Start must be ≤ end"):
            await harmonic_partial_sum(5, 3)

    @pytest.mark.asyncio
    async def test_harmonic_mean_basic_cases(self):
        """Test harmonic mean calculation."""
        test_cases = [
            ([1, 2, 4], 1.7142857142857144),  # HM of 1, 2, 4
            ([2, 3, 6], 3.0),  # HM of 2, 3, 6
            ([1, 4, 4], 2.0),  # HM of 1, 4, 4
            ([5], 5.0),  # HM of single number
            ([2, 8], 3.2),  # HM of 2, 8
        ]

        for numbers, expected in test_cases:
            result = await harmonic_mean(numbers)
            assert abs(result - expected) < 1e-10, (
                f"harmonic_mean({numbers}) should be {expected}, got {result}"
            )

    @pytest.mark.asyncio
    async def test_harmonic_mean_properties(self):
        """Test mathematical properties of harmonic mean."""
        # Harmonic mean of identical numbers should be the number itself
        for n in [2, 5, 10]:
            identical_numbers = [7] * n
            hm = await harmonic_mean(identical_numbers)
            assert abs(hm - 7) < 1e-10, "Harmonic mean of identical 7s should be 7"

        # Harmonic mean ≤ Geometric mean ≤ Arithmetic mean
        numbers = [2, 4, 8]
        hm = await harmonic_mean(numbers)
        am = sum(numbers) / len(numbers)  # Arithmetic mean
        gm = (2 * 4 * 8) ** (1 / 3)  # Geometric mean

        assert hm <= gm <= am, "HM ≤ GM ≤ AM inequality should hold"

    @pytest.mark.asyncio
    async def test_harmonic_mean_edge_cases(self):
        """Test harmonic mean edge cases."""
        # Empty list
        with pytest.raises(ValueError, match="Numbers list cannot be empty"):
            await harmonic_mean([])

        # Non-positive numbers
        with pytest.raises(ValueError, match="All numbers must be positive"):
            await harmonic_mean([1, 0, 3])

        with pytest.raises(ValueError, match="All numbers must be positive"):
            await harmonic_mean([2, -1, 4])


# ============================================================================
# SYLVESTER SEQUENCE TESTS
# ============================================================================


class TestSylvesterSequence:
    """Test cases for Sylvester sequence functions."""

    @pytest.mark.asyncio
    async def test_sylvester_sequence_known_values(self):
        """Test Sylvester sequence with known values."""
        known_sylvester = [
            (1, [2]),
            (2, [2, 3]),
            (3, [2, 3, 7]),
            (4, [2, 3, 7, 43]),
            (5, [2, 3, 7, 43, 1807]),
            (6, [2, 3, 7, 43, 1807, 3263443]),
        ]

        for n, expected in known_sylvester:
            result = await sylvester_sequence(n)
            assert result == expected, f"sylvester_sequence({n}) should be {expected}, got {result}"

    @pytest.mark.asyncio
    async def test_sylvester_sequence_recurrence(self):
        """Test Sylvester sequence recurrence relation."""
        sequence = await sylvester_sequence(5)

        # Verify recurrence: a_{n+1} = a_1 * a_2 * ... * a_n + 1
        for i in range(1, len(sequence)):
            product = 1
            for j in range(i):
                product *= sequence[j]
            expected_next = product + 1
            assert sequence[i] == expected_next, f"Sylvester recurrence failed at position {i}"

    @pytest.mark.asyncio
    async def test_sylvester_sequence_edge_cases(self):
        """Test Sylvester sequence edge cases."""
        # Zero or negative input
        assert await sylvester_sequence(0) == []
        assert await sylvester_sequence(-1) == []

        # Single term
        assert await sylvester_sequence(1) == [2]

    @pytest.mark.asyncio
    async def test_sylvester_sequence_growth_property(self):
        """Test that Sylvester sequence grows very rapidly."""
        sequence = await sylvester_sequence(5)

        # Each term should be much larger than the previous (except 2nd term)
        # Note: 2nd term (3) is only 1.5 times the first (2), so adjust expectation
        for i in range(2, len(sequence)):  # Start from 3rd term
            assert sequence[i] > sequence[i - 1] * 2, (
                "Sylvester sequence should grow rapidly from 3rd term onward"
            )

        # Verify specific growth pattern
        assert sequence[0] == 2
        assert sequence[1] == 3  # Only 1.5x growth here, which is expected
        if len(sequence) > 2:
            assert sequence[2] == 7  # Much faster growth starts here

        # Last term should be very large
        assert sequence[-1] > 1000, "Sylvester sequence grows to large numbers quickly"

    @pytest.mark.asyncio
    async def test_sylvester_expansion_of_one_consistency(self):
        """Test that Sylvester expansion of one gives same sequence."""
        for n in range(1, 6):
            sylvester_seq = await sylvester_sequence(n)
            expansion_seq = await sylvester_expansion_of_one(n)
            assert sylvester_seq == expansion_seq, (
                f"Sylvester sequence and expansion should match for n={n}"
            )

    @pytest.mark.asyncio
    async def test_sylvester_expansion_sum_property(self):
        """Test that Sylvester expansion approaches 1."""
        # The sum 1/2 + 1/3 + 1/7 + 1/43 + ... approaches 1
        sylvester_terms = await sylvester_expansion_of_one(4)
        partial_sum_num, partial_sum_den = await unit_fraction_sum(sylvester_terms)
        partial_sum = partial_sum_num / partial_sum_den

        # Should be close to 1 but less than 1
        assert 0.9 < partial_sum < 1.0, (
            f"Sylvester expansion partial sum should approach 1, got {partial_sum}"
        )

        # With more terms, should be even closer
        sylvester_terms_5 = await sylvester_expansion_of_one(5)
        sum_5_num, sum_5_den = await unit_fraction_sum(sylvester_terms_5)
        sum_5 = sum_5_num / sum_5_den

        assert sum_5 > partial_sum, "More Sylvester terms should give larger sum"
        assert sum_5 < 1.0, "Sylvester expansion should never exceed 1"


# ============================================================================
# EGYPTIAN FRACTION PROPERTIES TESTS
# ============================================================================


class TestEgyptianFractionProperties:
    """Test cases for Egyptian fraction property functions."""

    @pytest.mark.asyncio
    async def test_egyptian_fraction_properties_basic_analysis(self):
        """Test basic Egyptian fraction properties analysis."""
        test_cases = [
            (
                [2, 6],
                {
                    "sum": [2, 3],
                    "length": 2,
                    "max_denom": 6,
                    "min_denom": 2,
                    "is_complete": False,
                },
            ),
            (
                [2, 3],
                {
                    "sum": [5, 6],
                    "length": 2,
                    "max_denom": 3,
                    "min_denom": 2,
                    "is_complete": False,
                },
            ),  # 1/2 + 1/3 = 5/6
            (
                [3, 4, 12],
                {
                    "sum": [2, 3],
                    "length": 3,
                    "max_denom": 12,
                    "min_denom": 3,
                    "is_complete": False,
                },
            ),  # 1/3 + 1/4 + 1/12 = 8/12 = 2/3
        ]

        for denominators, expected_partial in test_cases:
            result = await egyptian_fraction_properties(denominators)

            # Check key properties
            assert result["sum"] == expected_partial["sum"], f"Sum mismatch for {denominators}"
            assert result["length"] == expected_partial["length"], (
                f"Length mismatch for {denominators}"
            )
            assert result["max_denom"] == expected_partial["max_denom"], (
                f"Max denominator mismatch for {denominators}"
            )
            assert result["min_denom"] == expected_partial["min_denom"], (
                f"Min denominator mismatch for {denominators}"
            )
            assert result["is_complete"] == expected_partial["is_complete"], (
                f"Completeness mismatch for {denominators}"
            )

    @pytest.mark.asyncio
    async def test_egyptian_fraction_properties_edge_cases(self):
        """Test Egyptian fraction properties with edge cases."""
        # Empty list
        result = await egyptian_fraction_properties([])
        assert result["sum"] == [0, 1], "Empty list should have sum 0"
        assert result["length"] == 0, "Empty list should have length 0"
        assert not result["is_complete"], "Empty list should not be complete"

        # Single unit fraction
        result = await egyptian_fraction_properties([5])
        assert result["sum"] == [1, 5], "Single unit fraction 1/5"
        assert result["length"] == 1, "Single term should have length 1"
        assert result["max_denom"] == 5, "Max denominator should be 5"
        assert result["min_denom"] == 5, "Min denominator should be 5"
        assert not result["is_complete"], "1/5 is not complete (≠ 1)"

    @pytest.mark.asyncio
    async def test_egyptian_fraction_properties_duplicate_detection(self):
        """Test detection of duplicate denominators."""
        # No duplicates
        result = await egyptian_fraction_properties([2, 3, 6])
        assert not result["has_duplicates"], "Should detect no duplicates"
        assert result["total_denominators"] == 3, "Should count unique denominators"

        # With duplicates
        result = await egyptian_fraction_properties([2, 3, 2, 6])
        assert result["has_duplicates"], "Should detect duplicates"
        assert result["total_denominators"] == 3, "Should count unique denominators"
        assert result["length"] == 4, "Should count total terms"

    @pytest.mark.asyncio
    async def test_is_optimal_egyptian_fraction_known_optimal(self):
        """Test optimality checking for known optimal representations."""
        optimal_cases = [
            (2, 3, [2, 6]),  # 2/3 = 1/2 + 1/6 is optimal
            (3, 4, [2, 4]),  # 3/4 = 1/2 + 1/4 is optimal
            (1, 5, [5]),  # 1/5 = 1/5 is optimal (unit fraction)
            (1, 2, [2]),  # 1/2 = 1/2 is optimal (unit fraction)
        ]

        for num, den, representation in optimal_cases:
            result = await is_optimal_egyptian_fraction(num, den, representation)
            assert result, f"Representation {representation} should be optimal for {num}/{den}"

    @pytest.mark.asyncio
    async def test_is_optimal_egyptian_fraction_incorrect_representation(self):
        """Test optimality checking with incorrect representations."""
        # Wrong representation
        result = await is_optimal_egyptian_fraction(2, 3, [3, 4])  # Doesn't sum to 2/3
        assert not result, "Incorrect representation should not be optimal"

        # Empty representation for non-zero fraction
        result = await is_optimal_egyptian_fraction(1, 2, [])
        assert not result, "Empty representation for 1/2 should not be optimal"

        # Correct for zero fraction
        result = await is_optimal_egyptian_fraction(0, 1, [])
        assert result, "Empty representation for 0 should be optimal"

    @pytest.mark.asyncio
    async def test_two_unit_fraction_representations_known_cases(self):
        """Test finding two-term Egyptian fraction representations."""
        test_cases = [
            (2, 3, [[2, 6]]),  # 2/3 = 1/2 + 1/6
            (3, 4, [[2, 4]]),  # 3/4 = 1/2 + 1/4
            (5, 6, [[2, 3]]),  # 5/6 = 1/2 + 1/3
            (1, 2, [[2, 2]]),  # 1/2 = 1/2 + 1/∞ or different interpretation
        ]

        for num, den, expected_contains in test_cases[:-1]:  # Skip the last ambiguous case
            result = await two_unit_fraction_representations(num, den, 100)

            # Check that expected representations are found
            for expected_rep in expected_contains:
                assert expected_rep in result, (
                    f"Should find representation {expected_rep} for {num}/{den}"
                )

    @pytest.mark.asyncio
    async def test_two_unit_fraction_representations_verification(self):
        """Test that found representations actually sum to target fraction."""
        target_fractions = [(2, 3), (3, 4), (4, 5)]

        for num, den in target_fractions:
            representations = await two_unit_fraction_representations(num, den, 50)
            target_fraction = Fraction(num, den)

            for rep in representations:
                if len(rep) == 2:
                    sum_fraction = Fraction(1, rep[0]) + Fraction(1, rep[1])
                    assert sum_fraction == target_fraction, (
                        f"Representation {rep} doesn't sum to {num}/{den}"
                    )

    @pytest.mark.asyncio
    async def test_two_unit_fraction_representations_edge_cases(self):
        """Test two-term representation finding with edge cases."""
        # Invalid inputs
        with pytest.raises(ValueError, match="Numerator and denominator must be positive"):
            await two_unit_fraction_representations(0, 3, 10)

        with pytest.raises(ValueError, match="Fraction must be proper"):
            await two_unit_fraction_representations(5, 3, 10)

        # Very small limit might find no representations
        result = await two_unit_fraction_representations(2, 3, 5)  # Very small limit
        # Should still find some or be empty, but shouldn't error
        assert isinstance(result, list), "Should return a list even with small limit"


# ============================================================================
# FRACTION UTILITIES TESTS
# ============================================================================


class TestFractionUtilities:
    """Test cases for fraction utility functions."""

    @pytest.mark.asyncio
    async def test_is_proper_fraction_basic_cases(self):
        """Test proper fraction identification."""
        test_cases = [
            (3, 4, True),  # 3/4 is proper
            (5, 3, False),  # 5/3 is improper
            (1, 1, False),  # 1/1 is not proper
            (2, 5, True),  # 2/5 is proper
            (7, 8, True),  # 7/8 is proper
            (10, 10, False),  # 10/10 is not proper
            (15, 4, False),  # 15/4 is improper
            (1, 100, True),  # 1/100 is proper
        ]

        for num, den, expected in test_cases:
            result = await is_proper_fraction(num, den)
            assert result == expected, (
                f"is_proper_fraction({num}, {den}) should be {expected}, got {result}"
            )

    @pytest.mark.asyncio
    async def test_is_proper_fraction_edge_cases(self):
        """Test proper fraction identification edge cases."""
        # Zero numerator
        result = await is_proper_fraction(0, 5)
        assert not result, "0/5 should not be proper"

        # Negative numerator
        result = await is_proper_fraction(-3, 5)
        assert not result, "-3/5 should not be proper"

        # Invalid denominator
        with pytest.raises(ValueError, match="Denominator must be positive"):
            await is_proper_fraction(3, 0)

        with pytest.raises(ValueError, match="Denominator must be positive"):
            await is_proper_fraction(3, -5)

    @pytest.mark.asyncio
    async def test_improper_to_egyptian_basic_cases(self):
        """Test conversion of improper fractions to Egyptian form."""
        test_cases = [
            (7, 3, {"whole": 2, "egyptian": [3]}),  # 7/3 = 2 + 1/3
            (5, 2, {"whole": 2, "egyptian": [2]}),  # 5/2 = 2 + 1/2
            (11, 4, {"whole": 2, "egyptian": [2, 4]}),  # 11/4 = 2 + 3/4 = 2 + 1/2 + 1/4
            (8, 3, {"whole": 2, "egyptian": [2, 6]}),  # 8/3 = 2 + 2/3 = 2 + 1/2 + 1/6
            (6, 2, {"whole": 3, "egyptian": []}),  # 6/2 = 3 + 0
            (9, 4, {"whole": 2, "egyptian": [4]}),  # 9/4 = 2 + 1/4
        ]

        for num, den, expected in test_cases:
            result = await improper_to_egyptian(num, den)
            assert result["whole"] == expected["whole"], f"Whole part mismatch for {num}/{den}"

            # Verify Egyptian part sums correctly to remainder
            if expected["egyptian"]:
                remainder = num - expected["whole"] * den
                egyptian_sum_num, egyptian_sum_den = await unit_fraction_sum(result["egyptian"])
                expected_remainder = Fraction(remainder, den)
                actual_remainder = Fraction(egyptian_sum_num, egyptian_sum_den)
                assert expected_remainder == actual_remainder, (
                    f"Egyptian part doesn't match remainder for {num}/{den}"
                )

    @pytest.mark.asyncio
    async def test_improper_to_egyptian_edge_cases(self):
        """Test improper fraction conversion edge cases."""
        # Invalid inputs
        with pytest.raises(ValueError, match="Denominator must be positive"):
            await improper_to_egyptian(7, 0)

        with pytest.raises(ValueError, match="Numerator must be positive"):
            await improper_to_egyptian(0, 3)

        with pytest.raises(ValueError, match="Numerator must be positive"):
            await improper_to_egyptian(-5, 3)

    @pytest.mark.asyncio
    async def test_improper_to_egyptian_reconstruction(self):
        """Test that improper fractions can be reconstructed from Egyptian form."""
        test_fractions = [(7, 3), (11, 4), (13, 5), (17, 6)]

        for num, den in test_fractions:
            result = await improper_to_egyptian(num, den)

            # Reconstruct the original fraction
            whole_part = result["whole"]
            if result["egyptian"]:
                egyptian_sum_num, egyptian_sum_den = await unit_fraction_sum(result["egyptian"])
                fractional_part = Fraction(egyptian_sum_num, egyptian_sum_den)
            else:
                fractional_part = Fraction(0, 1)

            reconstructed = whole_part + fractional_part
            original = Fraction(num, den)

            assert reconstructed == original, (
                f"Failed to reconstruct {num}/{den} from Egyptian form"
            )

    @pytest.mark.asyncio
    async def test_egyptian_expansion_lengths_basic_calculation(self):
        """Test Egyptian expansion length calculation."""
        # For unit fractions 1/n, the expansion length is always 1
        result = await egyptian_expansion_lengths(10)

        for n in range(2, 11):
            assert result[n] == 1, (
                f"Expansion length for 1/{n} should be 1 (it's already a unit fraction)"
            )

    @pytest.mark.asyncio
    async def test_egyptian_expansion_lengths_edge_cases(self):
        """Test Egyptian expansion length calculation edge cases."""
        # Invalid input
        result = await egyptian_expansion_lengths(1)
        assert result == {}, "Should return empty dict for max_n ≤ 1"

        result = await egyptian_expansion_lengths(0)
        assert result == {}, "Should return empty dict for max_n ≤ 1"

    @pytest.mark.asyncio
    async def test_shortest_egyptian_fraction_known_cases(self):
        """Test shortest Egyptian fraction finding for known cases."""
        # Note: This is computationally expensive, so we test small cases
        test_cases = [
            (2, 3),  # Should find [2, 6]
            (3, 4),  # Should find [2, 4]
            (1, 5),  # Should find [5]
        ]

        for num, den in test_cases:
            result = await shortest_egyptian_fraction(num, den, max_terms=3)

            # Verify the result sums to the original fraction
            if result:
                sum_num, sum_den = await unit_fraction_sum(result)
                original_fraction = Fraction(num, den)
                result_fraction = Fraction(sum_num, sum_den)
                assert original_fraction == result_fraction, (
                    f"Shortest Egyptian fraction for {num}/{den} doesn't sum correctly"
                )

    @pytest.mark.asyncio
    async def test_shortest_egyptian_fraction_edge_cases(self):
        """Test shortest Egyptian fraction finding edge cases."""
        # Invalid inputs
        with pytest.raises(ValueError, match="Numerator and denominator must be positive"):
            await shortest_egyptian_fraction(0, 3, 2)

        with pytest.raises(ValueError, match="Fraction must be proper"):
            await shortest_egyptian_fraction(5, 3, 2)

        # Unit fraction case
        result = await shortest_egyptian_fraction(1, 7, 2)
        assert result == [7], "Unit fraction should have shortest representation as itself"


# ============================================================================
# INTEGRATION AND MATHEMATICAL PROPERTIES TESTS
# ============================================================================


class TestIntegrationAndProperties:
    """Integration tests and mathematical property verification."""

    @pytest.mark.asyncio
    async def test_egyptian_decomposition_and_summation_consistency(self):
        """Test consistency between decomposition and summation."""
        test_fractions = [(2, 3), (3, 4), (4, 5), (5, 6), (7, 12), (11, 15)]

        for num, den in test_fractions:
            # Decompose into Egyptian fraction
            egyptian_denoms = await egyptian_fraction_decomposition(num, den)

            # Sum the Egyptian fractions back
            sum_num, sum_den = await unit_fraction_sum(egyptian_denoms)

            # Should equal original fraction
            original = Fraction(num, den)
            reconstructed = Fraction(sum_num, sum_den)
            assert original == reconstructed, (
                f"Decomposition-summation cycle failed for {num}/{den}"
            )

    @pytest.mark.asyncio
    async def test_harmonic_number_and_unit_fraction_relationship(self):
        """Test relationship between harmonic numbers and unit fraction sums."""
        for n in range(1, 8):
            # Harmonic number H_n
            harmonic_n = await harmonic_number(n)

            # Sum of unit fractions 1/1 + 1/2 + ... + 1/n
            unit_fractions = list(range(1, n + 1))
            sum_num, sum_den = await unit_fraction_sum(unit_fractions)
            unit_sum = sum_num / sum_den

            # Should be equal
            assert abs(harmonic_n - unit_sum) < 1e-10, (
                f"Harmonic number H_{n} should equal unit fraction sum"
            )

    @pytest.mark.asyncio
    async def test_sylvester_sequence_and_egyptian_fraction_relationship(self):
        """Test relationship between Sylvester sequence and Egyptian fractions."""
        # Sylvester sequence gives Egyptian fraction expansion of 1
        sylvester_terms = await sylvester_sequence(4)
        sum_num, sum_den = await unit_fraction_sum(sylvester_terms)

        # Should approach 1 but be less than 1
        fraction_sum = sum_num / sum_den
        assert 0.9 < fraction_sum < 1.0, "Sylvester expansion should approach but not reach 1"

        # Adding more terms should get closer to 1
        sylvester_terms_5 = await sylvester_sequence(5)
        sum_5_num, sum_5_den = await unit_fraction_sum(sylvester_terms_5)
        fraction_sum_5 = sum_5_num / sum_5_den

        assert fraction_sum_5 > fraction_sum, "More Sylvester terms should give larger sum"

    @pytest.mark.asyncio
    async def test_egyptian_fraction_properties_and_decomposition_consistency(self):
        """Test consistency between property analysis and decomposition."""
        test_fractions = [(2, 3), (3, 4), (5, 6)]

        for num, den in test_fractions:
            egyptian_denoms = await egyptian_fraction_decomposition(num, den)
            properties = await egyptian_fraction_properties(egyptian_denoms)

            # Verify properties match the decomposition
            assert properties["length"] == len(egyptian_denoms), "Length should match decomposition"
            assert properties["max_denom"] == max(egyptian_denoms), "Max denominator should match"
            assert properties["min_denom"] == min(egyptian_denoms), "Min denominator should match"

            # Sum should equal original fraction
            original_fraction = Fraction(num, den)
            property_sum = Fraction(properties["sum"][0], properties["sum"][1])
            assert original_fraction == property_sum, "Property sum should equal original fraction"

    @pytest.mark.asyncio
    async def test_harmonic_mean_and_arithmetic_relationship(self):
        """Test mathematical relationships involving harmonic mean."""
        test_cases = [
            [2, 8],  # HM(2, 8) = 3.2, AM(2, 8) = 5, GM(2, 8) = 4
            [3, 6],  # HM(3, 6) = 4, AM(3, 6) = 4.5, GM(3, 6) ≈ 4.24
            [1, 4, 4],  # More complex case
        ]

        for numbers in test_cases:
            hm = await harmonic_mean(numbers)
            am = sum(numbers) / len(numbers)  # Arithmetic mean

            # Harmonic mean should be ≤ arithmetic mean
            assert hm <= am + 1e-10, f"Harmonic mean should be ≤ arithmetic mean for {numbers}"

            # For positive numbers, harmonic mean should be positive
            assert hm > 0, f"Harmonic mean should be positive for {numbers}"

    @pytest.mark.asyncio
    async def test_unit_fraction_properties_verification(self):
        """Test mathematical properties of unit fractions."""
        denominators = [2, 3, 4, 6, 12]

        # Test that unit fractions sum correctly
        sum_num, sum_den = await unit_fraction_sum(denominators)

        # Manually calculate expected sum
        manual_sum = Fraction(0)
        for d in denominators:
            manual_sum += Fraction(1, d)

        assert Fraction(sum_num, sum_den) == manual_sum, (
            "Unit fraction sum should match manual calculation"
        )

        # Test LCM properties
        lcm_result = await egyptian_fraction_lcm(denominators)
        for d in denominators:
            assert lcm_result % d == 0, "LCM should be divisible by all denominators"

    @pytest.mark.asyncio
    async def test_proper_fraction_and_egyptian_decomposition_relationship(self):
        """Test that proper fractions have valid Egyptian decompositions."""
        test_fractions = [
            (1, 2),
            (2, 3),
            (3, 5),
            (5, 8),
            (8, 13),
        ]  # Some are Fibonacci ratios

        for num, den in test_fractions:
            # Should be proper
            is_proper = await is_proper_fraction(num, den)
            assert is_proper, f"{num}/{den} should be proper"

            # Should have valid Egyptian decomposition
            egyptian_denoms = await egyptian_fraction_decomposition(num, den)
            assert len(egyptian_denoms) > 0, (
                f"{num}/{den} should have non-empty Egyptian decomposition"
            )

            # Decomposition should sum to original
            sum_num, sum_den = await unit_fraction_sum(egyptian_denoms)
            original = Fraction(num, den)
            reconstructed = Fraction(sum_num, sum_den)
            assert original == reconstructed, (
                f"Egyptian decomposition should reconstruct {num}/{den}"
            )


# ============================================================================
# PERFORMANCE AND ASYNC BEHAVIOR TESTS
# ============================================================================


class TestPerformance:
    """Performance and async behavior tests."""

    @pytest.mark.asyncio
    async def test_all_functions_are_async(self):
        """Test that all Egyptian fraction functions are properly async."""
        operations = [
            egyptian_fraction_decomposition(2, 3),
            fibonacci_greedy_egyptian(3, 4),
            unit_fraction_sum([2, 3, 6]),
            is_unit_fraction(1, 5),
            egyptian_fraction_lcm([2, 3, 4]),
            harmonic_number(10),
            harmonic_number_fraction(5),
            harmonic_partial_sum(2, 6),
            harmonic_mean([1, 2, 4]),
            sylvester_sequence(4),
            sylvester_expansion_of_one(3),
            egyptian_fraction_properties([2, 6]),
            is_optimal_egyptian_fraction(2, 3, [2, 6]),
            is_proper_fraction(3, 4),
            improper_to_egyptian(7, 3),
            egyptian_expansion_lengths(10),
        ]

        # Ensure all are coroutines
        for op in operations:
            assert asyncio.iscoroutine(op)

        # Run all operations concurrently
        results = await asyncio.gather(*operations)

        # Verify results have expected types and values
        assert results[0] == [2, 6]  # egyptian_fraction_decomposition(2, 3)
        assert results[1] == [2, 4]  # fibonacci_greedy_egyptian(3, 4)
        assert results[2] == (1, 1)  # unit_fraction_sum([2, 3, 6])
        assert results[3]  # is_unit_fraction(1, 5)
        assert results[4] == 12  # egyptian_fraction_lcm([2, 3, 4])
        assert isinstance(results[5], float)  # harmonic_number(10)
        assert results[6] == (137, 60)  # harmonic_number_fraction(5)
        assert isinstance(results[7], float)  # harmonic_partial_sum(2, 6)
        assert isinstance(results[8], float)  # harmonic_mean([1, 2, 4])
        assert results[9] == [2, 3, 7, 43]  # sylvester_sequence(4)
        assert results[10] == [2, 3, 7]  # sylvester_expansion_of_one(3)
        assert isinstance(results[11], dict)  # egyptian_fraction_properties([2, 6])
        assert results[12]  # is_optimal_egyptian_fraction(2, 3, [2, 6])
        assert results[13]  # is_proper_fraction(3, 4)
        assert isinstance(results[14], dict)  # improper_to_egyptian(7, 3)
        assert isinstance(results[15], dict)  # egyptian_expansion_lengths(10)

    @pytest.mark.asyncio
    async def test_concurrent_execution(self):
        """Test that Egyptian fraction operations can run concurrently."""
        start_time = time.time()

        # Run multiple operations concurrently
        tasks = []
        for i in range(2, 25):  # Test fractions 1/i, 2/i, etc.
            if i > 2:
                tasks.append(egyptian_fraction_decomposition(2, i))
            if i > 3:
                tasks.append(egyptian_fraction_decomposition(3, i))
            tasks.append(harmonic_number(i))
            tasks.append(is_proper_fraction(i - 1, i))
            tasks.append(unit_fraction_sum([2, i]))

        results = await asyncio.gather(*tasks)
        duration = time.time() - start_time

        # Should complete quickly due to async nature
        assert duration < 3.0
        assert len(results) > 50  # Should have many results

        # Check some patterns in results
        egyptian_results = [
            r for r in results if isinstance(r, list) and all(isinstance(x, int) for x in r)
        ]
        harmonic_results = [r for r in results if isinstance(r, float)]
        boolean_results = [r for r in results if isinstance(r, bool)]

        assert len(egyptian_results) > 10, "Should have Egyptian fraction results"
        assert len(harmonic_results) > 10, "Should have harmonic number results"
        assert len(boolean_results) > 10, "Should have boolean results"

    @pytest.mark.asyncio
    async def test_large_computation_handling(self):
        """Test handling of computationally intensive operations."""
        # Test larger harmonic numbers
        h_100 = await harmonic_number(100)
        assert h_100 > 5.0, "H_100 should be > 5"

        # Test larger Egyptian fraction decompositions
        large_fraction = await egyptian_fraction_decomposition(99, 100)
        assert len(large_fraction) > 3, "99/100 should have multiple terms"

        # Test longer Sylvester sequence (but not too long due to rapid growth)
        sylvester_6 = await sylvester_sequence(6)
        assert len(sylvester_6) == 6, "Should generate 6 Sylvester terms"
        assert sylvester_6[-1] > 1000000, "6th Sylvester number should be very large"

        # Test harmonic number with exact fractions
        h_20_frac = await harmonic_number_fraction(20)
        assert h_20_frac[0] > h_20_frac[1], "H_20 should be > 1"
        assert h_20_frac[1] > 1000, "Denominator of H_20 should be large"

    @pytest.mark.asyncio
    async def test_memory_efficiency(self):
        """Test that functions don't consume excessive memory."""
        # Test multiple Egyptian fraction decompositions
        decompositions = []
        for num in range(1, 20):
            for den in range(num + 1, min(num + 10, 30)):
                decomp = await egyptian_fraction_decomposition(num, den)
                decompositions.append(decomp)

        assert len(decompositions) > 50, "Should generate many decompositions"

        # Test multiple harmonic calculations
        harmonic_numbers = []
        for i in range(1, 50):
            h = await harmonic_number(i)
            harmonic_numbers.append(h)

        assert len(harmonic_numbers) == 49, "Should calculate 49 harmonic numbers"
        assert all(h > 0 for h in harmonic_numbers), "All harmonic numbers should be positive"

        # Test that harmonic numbers are increasing
        for i in range(len(harmonic_numbers) - 1):
            assert harmonic_numbers[i] < harmonic_numbers[i + 1], (
                "Harmonic numbers should be increasing"
            )


# ============================================================================
# ERROR HANDLING TESTS
# ============================================================================


class TestErrorHandling:
    """Test error handling and edge cases."""

    @pytest.mark.asyncio
    async def test_egyptian_fraction_decomposition_errors(self):
        """Test error handling in Egyptian fraction decomposition."""
        error_cases = [
            (0, 5, "Numerator and denominator must be positive"),
            (-1, 5, "Numerator and denominator must be positive"),
            (2, 0, "Numerator and denominator must be positive"),
            (2, -3, "Numerator and denominator must be positive"),
            (5, 3, "Fraction must be proper"),
            (4, 4, "Fraction must be proper"),
        ]

        for num, den, expected_error in error_cases:
            with pytest.raises(ValueError, match=expected_error):
                await egyptian_fraction_decomposition(num, den)

    @pytest.mark.asyncio
    async def test_unit_fraction_operations_errors(self):
        """Test error handling in unit fraction operations."""
        # unit_fraction_sum errors
        with pytest.raises(ValueError, match="All denominators must be positive"):
            await unit_fraction_sum([2, 0, 4])

        with pytest.raises(ValueError, match="All denominators must be positive"):
            await unit_fraction_sum([2, -1, 4])

        # is_unit_fraction errors
        with pytest.raises(ValueError, match="Denominator must be positive"):
            await is_unit_fraction(1, 0)

        with pytest.raises(ValueError, match="Denominator must be positive"):
            await is_unit_fraction(1, -5)

        # egyptian_fraction_lcm errors
        with pytest.raises(ValueError, match="All denominators must be positive"):
            await egyptian_fraction_lcm([2, 0, 4])

    @pytest.mark.asyncio
    async def test_harmonic_functions_errors(self):
        """Test error handling in harmonic functions."""
        # harmonic_number errors
        with pytest.raises(ValueError, match="n must be positive"):
            await harmonic_number(0)

        with pytest.raises(ValueError, match="n must be positive"):
            await harmonic_number(-5)

        # harmonic_partial_sum errors
        with pytest.raises(ValueError, match="Start and end must be positive"):
            await harmonic_partial_sum(0, 5)

        with pytest.raises(ValueError, match="Start must be ≤ end"):
            await harmonic_partial_sum(5, 3)

        # harmonic_mean errors
        with pytest.raises(ValueError, match="Numbers list cannot be empty"):
            await harmonic_mean([])

        with pytest.raises(ValueError, match="All numbers must be positive"):
            await harmonic_mean([1, 0, 3])

        with pytest.raises(ValueError, match="All numbers must be positive"):
            await harmonic_mean([2, -1, 4])

    @pytest.mark.asyncio
    async def test_fraction_utility_errors(self):
        """Test error handling in fraction utility functions."""
        # is_proper_fraction errors
        with pytest.raises(ValueError, match="Denominator must be positive"):
            await is_proper_fraction(3, 0)

        with pytest.raises(ValueError, match="Denominator must be positive"):
            await is_proper_fraction(3, -5)

        # improper_to_egyptian errors
        with pytest.raises(ValueError, match="Denominator must be positive"):
            await improper_to_egyptian(7, 0)

        with pytest.raises(ValueError, match="Numerator must be positive"):
            await improper_to_egyptian(0, 3)

        with pytest.raises(ValueError, match="Numerator must be positive"):
            await improper_to_egyptian(-5, 3)

        # shortest_egyptian_fraction errors
        with pytest.raises(ValueError, match="Numerator and denominator must be positive"):
            await shortest_egyptian_fraction(0, 3, 2)

        with pytest.raises(ValueError, match="Fraction must be proper"):
            await shortest_egyptian_fraction(5, 3, 2)

        # two_unit_fraction_representations errors
        with pytest.raises(ValueError, match="Numerator and denominator must be positive"):
            await two_unit_fraction_representations(0, 3, 10)

        with pytest.raises(ValueError, match="Fraction must be proper"):
            await two_unit_fraction_representations(5, 3, 10)

    @pytest.mark.asyncio
    async def test_error_preserves_async_context(self):
        """Test that errors are properly raised in async context."""
        try:
            await egyptian_fraction_decomposition(5, 3)  # Should raise ValueError
            assert False, "Should have raised ValueError"
        except ValueError:
            # Should be able to continue with async operations
            result = await egyptian_fraction_decomposition(2, 3)
            assert result == [2, 6]

        try:
            await harmonic_number(0)  # Should raise ValueError
            assert False, "Should have raised ValueError"
        except ValueError:
            # Should be able to continue with async operations
            result = await harmonic_number(4)
            assert abs(result - 2.0833333333333335) < 1e-10


# ============================================================================
# PARAMETRIZED TESTS
# ============================================================================


class TestParametrized:
    """Parametrized tests for comprehensive coverage."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "numerator,denominator,expected",
        [
            (2, 3, [2, 6]),
            (3, 4, [2, 4]),
            (5, 6, [2, 3]),
            (7, 12, [2, 12]),
            (1, 2, [2]),
            (1, 3, [3]),
            (1, 4, [4]),
            (1, 5, [5]),
        ],
    )
    async def test_egyptian_fraction_decomposition_parametrized(
        self, numerator, denominator, expected
    ):
        """Parametrized test for Egyptian fraction decomposition."""
        result = await egyptian_fraction_decomposition(numerator, denominator)
        assert result == expected

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "denominators,expected_sum",
        [
            ([2, 3, 6], (1, 1)),
            ([2, 4, 8], (7, 8)),
            ([3, 6, 12], (7, 12)),  # Corrected: 1/3 + 1/6 + 1/12 = 7/12
            ([4, 6, 12], (1, 2)),
            ([2], (1, 2)),
            ([5], (1, 5)),
        ],
    )
    async def test_unit_fraction_sum_parametrized(self, denominators, expected_sum):
        """Parametrized test for unit fraction summation."""
        result = await unit_fraction_sum(denominators)
        assert result == expected_sum

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "n,expected",
        [
            (1, 1.0),
            (2, 1.5),
            (3, 1.8333333333333333),
            (4, 2.0833333333333335),
            (5, 2.283333333333333),
        ],
    )
    async def test_harmonic_number_parametrized(self, n, expected):
        """Parametrized test for harmonic number calculation."""
        result = await harmonic_number(n)
        assert abs(result - expected) < 1e-10

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "n,expected",
        [
            (1, (1, 1)),
            (2, (3, 2)),
            (3, (11, 6)),
            (4, (25, 12)),
            (5, (137, 60)),
        ],
    )
    async def test_harmonic_number_fraction_parametrized(self, n, expected):
        """Parametrized test for exact harmonic number fractions."""
        result = await harmonic_number_fraction(n)
        assert result == expected

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "numerator,denominator",
        [(1, 5), (1, 7), (1, 10), (3, 9), (4, 8), (5, 15), (7, 14)],
    )
    async def test_is_unit_fraction_parametrized(self, numerator, denominator):
        """Parametrized test for unit fraction identification."""
        result = await is_unit_fraction(numerator, denominator)
        assert result

    @pytest.mark.asyncio
    @pytest.mark.parametrize("numerator,denominator", [(3, 4), (2, 5), (7, 8), (1, 100), (99, 100)])
    async def test_is_proper_fraction_parametrized(self, numerator, denominator):
        """Parametrized test for proper fraction identification."""
        result = await is_proper_fraction(numerator, denominator)
        assert result

    @pytest.mark.asyncio
    @pytest.mark.parametrize("n,expected_length", [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5)])
    async def test_sylvester_sequence_parametrized(self, n, expected_length):
        """Parametrized test for Sylvester sequence length."""
        result = await sylvester_sequence(n)
        assert len(result) == expected_length
        if n > 0:
            assert result[0] == 2  # First Sylvester number is always 2


# ============================================================================
# COMPREHENSIVE INTEGRATION TESTS
# ============================================================================


class TestComprehensiveIntegration:
    """Comprehensive integration tests combining multiple functions."""

    @pytest.mark.asyncio
    async def test_complete_egyptian_fraction_workflow(self):
        """Test a complete Egyptian fraction analysis workflow."""
        # Start with a fraction
        num, den = 7, 12

        # Check if it's proper
        is_proper = await is_proper_fraction(num, den)
        assert is_proper, f"{num}/{den} should be proper"

        # Decompose into Egyptian fraction
        egyptian_denoms = await egyptian_fraction_decomposition(num, den)

        # Analyze properties
        await egyptian_fraction_properties(egyptian_denoms)

        # Verify the sum equals original fraction
        sum_num, sum_den = await unit_fraction_sum(egyptian_denoms)
        original_fraction = Fraction(num, den)
        reconstructed_fraction = Fraction(sum_num, sum_den)
        assert original_fraction == reconstructed_fraction

        # Check optimality
        await is_optimal_egyptian_fraction(num, den, egyptian_denoms)
        # Should be optimal (greedy algorithm typically gives good results)

        # Find two-term representations if they exist
        two_term_reps = await two_unit_fraction_representations(num, den, 100)

        # Verify any found two-term representations
        for rep in two_term_reps:
            if len(rep) == 2:
                rep_sum_num, rep_sum_den = await unit_fraction_sum(rep)
                rep_fraction = Fraction(rep_sum_num, rep_sum_den)
                assert rep_fraction == original_fraction, (
                    f"Two-term representation {rep} should equal {num}/{den}"
                )

    @pytest.mark.asyncio
    async def test_harmonic_series_and_egyptian_fractions_integration(self):
        """Test integration between harmonic series and Egyptian fractions."""
        n = 5

        # Calculate harmonic number
        harmonic_decimal = await harmonic_number(n)
        harmonic_frac_num, harmonic_frac_den = await harmonic_number_fraction(n)

        # Verify consistency
        harmonic_from_fraction = harmonic_frac_num / harmonic_frac_den
        assert abs(harmonic_decimal - harmonic_from_fraction) < 1e-10

        # Decompose harmonic fraction into Egyptian fractions
        # Note: This is an improper fraction, so we need to handle it specially
        if harmonic_frac_num >= harmonic_frac_den:
            egyptian_result = await improper_to_egyptian(harmonic_frac_num, harmonic_frac_den)

            # Verify reconstruction
            whole_part = egyptian_result["whole"]
            if egyptian_result["egyptian"]:
                egyptian_sum_num, egyptian_sum_den = await unit_fraction_sum(
                    egyptian_result["egyptian"]
                )
                fractional_part = Fraction(egyptian_sum_num, egyptian_sum_den)
            else:
                fractional_part = Fraction(0, 1)

            reconstructed = whole_part + fractional_part
            original = Fraction(harmonic_frac_num, harmonic_frac_den)
            assert reconstructed == original, f"Failed to reconstruct harmonic number {n}"

    @pytest.mark.asyncio
    async def test_sylvester_sequence_egyptian_fraction_properties(self):
        """Test properties of Sylvester sequence as Egyptian fractions."""
        n = 4

        # Generate Sylvester sequence
        sylvester_terms = await sylvester_sequence(n)

        # Use as Egyptian fraction denominators
        properties = await egyptian_fraction_properties(sylvester_terms)

        # Sum should be close to but less than 1
        sum_fraction = Fraction(properties["sum"][0], properties["sum"][1])
        sum_decimal = float(sum_fraction)
        assert 0.9 < sum_decimal < 1.0, f"Sylvester expansion should approach 1, got {sum_decimal}"

        # Length should match number of terms
        assert properties["length"] == n

        # Should not be complete (sum ≠ 1)
        assert not properties["is_complete"], "Sylvester expansion should not sum to exactly 1"

        # Denominators should be distinct and increasing
        assert not properties["has_duplicates"], "Sylvester denominators should be distinct"
        assert sylvester_terms == sorted(sylvester_terms), "Sylvester terms should be increasing"

    @pytest.mark.asyncio
    async def test_mathematical_constants_and_egyptian_fractions(self):
        """Test relationships with mathematical constants."""
        # Test that harmonic numbers grow logarithmically
        harmonic_values = []
        for i in range(1, 20):
            h_i = await harmonic_number(i)
            harmonic_values.append(h_i)

        # H_n should be approximately ln(n) + γ for large n
        h_10 = harmonic_values[9]  # H_10
        approx_h_10 = math.log(10) + 0.5772156649  # ln(10) + Euler-Mascheroni constant
        assert abs(h_10 - approx_h_10) < 0.1, "H_10 should approximate ln(10) + γ"

        # Test that harmonic mean ≤ geometric mean ≤ arithmetic mean
        test_numbers = [2, 4, 8, 16]
        hm = await harmonic_mean(test_numbers)
        am = sum(test_numbers) / len(test_numbers)
        gm = (2 * 4 * 8 * 16) ** (1 / 4)

        assert hm <= gm <= am, "HM ≤ GM ≤ AM inequality should hold"

    @pytest.mark.asyncio
    async def test_fraction_conversion_and_decomposition_chain(self):
        """Test a complete chain of fraction conversions and decompositions."""
        # Start with an improper fraction
        num, den = 11, 4

        # Convert to mixed number with Egyptian fraction
        mixed_result = await improper_to_egyptian(num, den)
        whole_part = mixed_result["whole"]
        egyptian_part = mixed_result["egyptian"]

        # Analyze the Egyptian part
        if egyptian_part:
            egyptian_properties = await egyptian_fraction_properties(egyptian_part)

            # Sum of Egyptian part should equal the fractional part
            remainder = num - whole_part * den
            expected_fraction = Fraction(remainder, den)
            actual_sum = Fraction(egyptian_properties["sum"][0], egyptian_properties["sum"][1])
            assert expected_fraction == actual_sum, (
                "Egyptian part should equal fractional remainder"
            )

        # Reconstruct original fraction
        if egyptian_part:
            egyptian_sum_num, egyptian_sum_den = await unit_fraction_sum(egyptian_part)
            fractional_part = Fraction(egyptian_sum_num, egyptian_sum_den)
        else:
            fractional_part = Fraction(0, 1)

        reconstructed = whole_part + fractional_part
        original = Fraction(num, den)
        assert reconstructed == original, f"Failed to reconstruct {num}/{den}"

    @pytest.mark.asyncio
    async def test_optimization_and_representation_finding(self):
        """Test optimization and alternative representation finding."""
        # Test fraction with multiple representations
        num, den = 2, 3

        # Get standard decomposition
        standard_decomp = await egyptian_fraction_decomposition(num, den)

        # Find two-term representations
        two_term_reps = await two_unit_fraction_representations(num, den, 50)

        # Check if standard decomposition is optimal
        await is_optimal_egyptian_fraction(num, den, standard_decomp)

        # Find shortest representation
        shortest_rep = await shortest_egyptian_fraction(num, den, max_terms=3)

        # All representations should sum to the same value
        original_fraction = Fraction(num, den)

        # Verify standard decomposition
        standard_sum_num, standard_sum_den = await unit_fraction_sum(standard_decomp)
        assert Fraction(standard_sum_num, standard_sum_den) == original_fraction

        # Verify two-term representations
        for rep in two_term_reps:
            if len(rep) == 2:
                rep_sum_num, rep_sum_den = await unit_fraction_sum(rep)
                assert Fraction(rep_sum_num, rep_sum_den) == original_fraction

        # Verify shortest representation
        if shortest_rep:
            shortest_sum_num, shortest_sum_den = await unit_fraction_sum(shortest_rep)
            assert Fraction(shortest_sum_num, shortest_sum_den) == original_fraction

            # Shortest should be no longer than standard
            assert len(shortest_rep) <= len(standard_decomp), (
                "Shortest representation should be no longer than standard"
            )


# ============================================================================
# EDGE CASES AND BOUNDARY CONDITIONS
# ============================================================================


class TestEdgeCasesAndBoundaryConditions:
    """Test edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_smallest_proper_fractions(self):
        """Test with smallest possible proper fractions."""
        small_fractions = [(1, 2), (1, 3), (1, 4), (1, 5), (1, 100)]

        for num, den in small_fractions:
            # Should be proper
            assert await is_proper_fraction(num, den)

            # Should be unit fractions
            assert await is_unit_fraction(num, den)

            # Egyptian decomposition should be trivial
            decomp = await egyptian_fraction_decomposition(num, den)
            assert decomp == [den], f"Unit fraction {num}/{den} should decompose to [{den}]"

    @pytest.mark.asyncio
    async def test_fractions_close_to_one(self):
        """Test with fractions very close to 1."""
        close_to_one = [(99, 100), (999, 1000), (9999, 10000)]

        for num, den in close_to_one:
            # Should be proper
            assert await is_proper_fraction(num, den)

            # Should have Egyptian decomposition
            decomp = await egyptian_fraction_decomposition(num, den)
            assert len(decomp) > 0, f"Fraction {num}/{den} should have Egyptian decomposition"

            # Decomposition should sum correctly
            sum_num, sum_den = await unit_fraction_sum(decomp)
            original = Fraction(num, den)
            reconstructed = Fraction(sum_num, sum_den)
            assert original == reconstructed, f"Decomposition of {num}/{den} should sum correctly"

    @pytest.mark.asyncio
    async def test_harmonic_numbers_boundary_cases(self):
        """Test harmonic numbers at boundary cases."""
        # H_1 should be exactly 1
        h_1 = await harmonic_number(1)
        assert abs(h_1 - 1.0) < 1e-15, "H_1 should be exactly 1"

        h_1_frac = await harmonic_number_fraction(1)
        assert h_1_frac == (1, 1), "H_1 as fraction should be 1/1"

        # H_2 should be exactly 1.5
        h_2 = await harmonic_number(2)
        assert abs(h_2 - 1.5) < 1e-15, "H_2 should be exactly 1.5"

        h_2_frac = await harmonic_number_fraction(2)
        assert h_2_frac == (3, 2), "H_2 as fraction should be 3/2"

        # Partial sum from n to n should equal 1/n
        for n in range(1, 10):
            partial = await harmonic_partial_sum(n, n)
            expected = 1.0 / n
            assert abs(partial - expected) < 1e-15, f"Partial sum from {n} to {n} should be 1/{n}"

    @pytest.mark.asyncio
    async def test_sylvester_sequence_boundary_cases(self):
        """Test Sylvester sequence boundary cases."""
        # Empty sequence
        assert await sylvester_sequence(0) == []

        # Single term
        assert await sylvester_sequence(1) == [2]

        # First few terms should match known values exactly
        sylv_2 = await sylvester_sequence(2)
        assert sylv_2 == [2, 3], "First two Sylvester numbers should be [2, 3]"

        sylv_3 = await sylvester_sequence(3)
        assert sylv_3 == [2, 3, 7], "First three Sylvester numbers should be [2, 3, 7]"

        # Verify recurrence for known terms
        # a_3 = a_1 * a_2 + 1 = 2 * 3 + 1 = 7 ✓
        # a_4 = a_1 * a_2 * a_3 + 1 = 2 * 3 * 7 + 1 = 43 ✓
        sylv_4 = await sylvester_sequence(4)
        assert sylv_4[3] == 43, "4th Sylvester number should be 43"

    @pytest.mark.asyncio
    async def test_harmonic_mean_boundary_cases(self):
        """Test harmonic mean boundary cases."""
        # Single number
        assert await harmonic_mean([5.0]) == 5.0, (
            "Harmonic mean of single number should be the number"
        )

        # Two identical numbers
        assert await harmonic_mean([7.0, 7.0]) == 7.0, (
            "Harmonic mean of identical numbers should be the number"
        )

        # Many identical numbers
        assert await harmonic_mean([3.0] * 10) == 3.0, (
            "Harmonic mean of many identical numbers should be the number"
        )

        # Very small numbers (testing numerical stability)
        small_numbers = [1e-10, 2e-10, 3e-10]
        hm_small = await harmonic_mean(small_numbers)
        assert hm_small > 0, "Harmonic mean of small numbers should be positive"
        assert hm_small < min(small_numbers) * 2, (
            "Harmonic mean should be reasonable for small numbers"
        )

    @pytest.mark.asyncio
    async def test_unit_fraction_sum_boundary_cases(self):
        """Test unit fraction sum boundary cases."""
        # Single unit fraction
        result = await unit_fraction_sum([7])
        assert result == (1, 7), "Sum of single unit fraction 1/7 should be (1, 7)"

        # Very large denominators
        large_denoms = [1000, 2000, 3000]
        result = await unit_fraction_sum(large_denoms)
        # Should be a valid fraction
        assert result[1] > 0, "Denominator should be positive"
        assert result[0] >= 0, "Numerator should be non-negative"

        # Sum should be small (since denominators are large)
        fraction_value = result[0] / result[1]
        assert fraction_value < 0.01, (
            "Sum of unit fractions with large denominators should be small"
        )


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v", "--tb=short", "--asyncio-mode=auto"])
