#!/usr/bin/env python3
# tests/math/number_theory/test_combinatorial_numbers.py
"""
Comprehensive pytest test suite for combinatorial_numbers.py module.

Tests cover:
- Catalan numbers: calculation, sequence generation, identification
- Bell numbers: calculation, sequence generation, Bell's triangle
- Stirling numbers: first and second kind, row generation
- Narayana numbers: calculation and triangle generation
- Mathematical properties and relationships
- Edge cases, error conditions, and performance testing
- Async behavior verification
"""

import pytest
import asyncio
import time

# Import the functions to test
from chuk_mcp_math.number_theory.combinatorial_numbers import (
    # Catalan numbers
    catalan_number,
    catalan_sequence,
    is_catalan_number,
    # Bell numbers
    bell_number,
    bell_sequence,
    bell_triangle,
    # Stirling numbers
    stirling_first,
    stirling_second,
    stirling_second_row,
    # Narayana numbers
    narayana_number,
    narayana_triangle_row,
)

# ============================================================================
# CATALAN NUMBERS TESTS
# ============================================================================


class TestCatalanNumbers:
    """Test cases for Catalan number functions."""

    @pytest.mark.asyncio
    async def test_catalan_number_known_values(self):
        """Test Catalan function with known values."""
        known_catalan = [
            (0, 1),
            (1, 1),
            (2, 2),
            (3, 5),
            (4, 14),
            (5, 42),
            (6, 132),
            (7, 429),
            (8, 1430),
            (9, 4862),
            (10, 16796),
            (11, 58786),
            (12, 208012),
            (13, 742900),
            (14, 2674440),
            (15, 9694845),
        ]

        for n, expected in known_catalan:
            result = await catalan_number(n)
            assert result == expected, f"C({n}) should be {expected}, got {result}"

    @pytest.mark.asyncio
    async def test_catalan_number_negative_input(self):
        """Test Catalan function with negative input."""
        with pytest.raises(
            ValueError, match="Catalan number index must be non-negative"
        ):
            await catalan_number(-1)

        with pytest.raises(
            ValueError, match="Catalan number index must be non-negative"
        ):
            await catalan_number(-5)

    @pytest.mark.asyncio
    async def test_catalan_number_recurrence_relation(self):
        """Test Catalan recurrence relation: C_n = (4n-2)*C_{n-1}/(n+1)."""
        # Test the recurrence for several values
        for n in range(1, 10):
            c_n = await catalan_number(n)
            c_n_minus_1 = await catalan_number(n - 1)
            expected = c_n_minus_1 * (4 * n - 2) // (n + 1)
            assert c_n == expected, f"Recurrence failed for n={n}"

    @pytest.mark.asyncio
    async def test_catalan_sequence_basic(self):
        """Test Catalan sequence generation."""
        seq_10 = await catalan_sequence(10)
        expected_10 = [1, 1, 2, 5, 14, 42, 132, 429, 1430, 4862]
        assert seq_10 == expected_10

        seq_5 = await catalan_sequence(5)
        expected_5 = [1, 1, 2, 5, 14]
        assert seq_5 == expected_5

    @pytest.mark.asyncio
    async def test_catalan_sequence_edge_cases(self):
        """Test Catalan sequence edge cases."""
        assert await catalan_sequence(0) == []
        assert await catalan_sequence(1) == [1]
        assert await catalan_sequence(2) == [1, 1]
        assert await catalan_sequence(-1) == []

    @pytest.mark.asyncio
    async def test_catalan_sequence_consistency(self):
        """Test consistency between catalan_number and catalan_sequence."""
        n = 12
        sequence = await catalan_sequence(n)

        for i in range(n):
            individual = await catalan_number(i)
            assert sequence[i] == individual, f"C({i}) should match sequence[{i}]"

    @pytest.mark.asyncio
    async def test_is_catalan_number_known_catalan(self):
        """Test with known Catalan numbers."""
        known_catalan_numbers = [1, 2, 5, 14, 42, 132, 429, 1430, 4862, 16796]

        for cat in known_catalan_numbers:
            assert await is_catalan_number(cat), (
                f"{cat} should be a Catalan number"
            )

    @pytest.mark.asyncio
    async def test_is_catalan_number_non_catalan(self):
        """Test with numbers that are not Catalan numbers."""
        non_catalan = [3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20]

        for n in non_catalan:
            assert not await is_catalan_number(n), (
                f"{n} should not be a Catalan number"
            )

    @pytest.mark.asyncio
    async def test_is_catalan_number_edge_cases(self):
        """Test edge cases for Catalan number checking."""
        assert (
            not await is_catalan_number(0)
        )  # 0 is not typically considered Catalan
        assert await is_catalan_number(1)  # C_0 = C_1 = 1
        assert not await is_catalan_number(-1)  # Negative numbers
        assert not await is_catalan_number(-5)  # Negative numbers

    @pytest.mark.asyncio
    async def test_catalan_large_values(self):
        """Test Catalan numbers for larger values."""
        # Test that we can compute reasonably large Catalan numbers
        large_catalan = [(16, 35357670), (17, 129644790), (18, 477638700)]

        for n, expected in large_catalan:
            result = await catalan_number(n)
            assert result == expected, f"C({n}) should be {expected}, got {result}"


# ============================================================================
# BELL NUMBERS TESTS
# ============================================================================


class TestBellNumbers:
    """Test cases for Bell number functions."""

    @pytest.mark.asyncio
    async def test_bell_number_known_values(self):
        """Test Bell function with known values."""
        known_bell = [
            (0, 1),
            (1, 1),
            (2, 2),
            (3, 5),
            (4, 15),
            (5, 52),
            (6, 203),
            (7, 877),
            (8, 4140),
            (9, 21147),
            (10, 115975),
        ]

        for n, expected in known_bell:
            result = await bell_number(n)
            assert result == expected, f"B({n}) should be {expected}, got {result}"

    @pytest.mark.asyncio
    async def test_bell_number_negative_input(self):
        """Test Bell function with negative input."""
        with pytest.raises(ValueError, match="Bell number index must be non-negative"):
            await bell_number(-1)

        with pytest.raises(ValueError, match="Bell number index must be non-negative"):
            await bell_number(-3)

    @pytest.mark.asyncio
    async def test_bell_sequence_basic(self):
        """Test Bell sequence generation."""
        seq_8 = await bell_sequence(8)
        expected_8 = [1, 1, 2, 5, 15, 52, 203, 877]
        assert seq_8 == expected_8

        seq_5 = await bell_sequence(5)
        expected_5 = [1, 1, 2, 5, 15]
        assert seq_5 == expected_5

    @pytest.mark.asyncio
    async def test_bell_sequence_edge_cases(self):
        """Test Bell sequence edge cases."""
        assert await bell_sequence(0) == []
        assert await bell_sequence(1) == [1]
        assert await bell_sequence(2) == [1, 1]
        assert await bell_sequence(-1) == []

    @pytest.mark.asyncio
    async def test_bell_sequence_consistency(self):
        """Test consistency between bell_number and bell_sequence."""
        n = 10
        sequence = await bell_sequence(n)

        for i in range(n):
            individual = await bell_number(i)
            assert sequence[i] == individual, f"B({i}) should match sequence[{i}]"

    @pytest.mark.asyncio
    async def test_bell_triangle_construction(self):
        """Test Bell's triangle construction."""
        triangle = await bell_triangle(5)
        expected = [[1], [1, 2], [2, 3, 5], [5, 7, 10, 15], [15, 20, 27, 37, 52]]
        assert triangle == expected

    @pytest.mark.asyncio
    async def test_bell_triangle_properties(self):
        """Test properties of Bell's triangle."""
        triangle = await bell_triangle(6)

        # First element of each row should be a Bell number
        for i, row in enumerate(triangle):
            if i > 0:  # Skip the first row (index 0)
                bell_i = await bell_number(i)
                assert row[0] == bell_i, f"First element of row {i} should be B({i})"

        # Last element of each row should be the first element of the next row
        for i in range(len(triangle) - 1):
            assert triangle[i][-1] == triangle[i + 1][0], (
                f"Row {i} last != Row {i + 1} first"
            )

    @pytest.mark.asyncio
    async def test_bell_triangle_edge_cases(self):
        """Test Bell's triangle edge cases."""
        assert await bell_triangle(0) == []
        assert await bell_triangle(1) == [[1]]

        with pytest.raises(ValueError, match="Number of rows must be non-negative"):
            await bell_triangle(-1)

    @pytest.mark.asyncio
    async def test_bell_triangle_row_sum_property(self):
        """Test that each element equals sum of element above + element to left."""
        triangle = await bell_triangle(5)

        for i in range(1, len(triangle)):  # Start from row 1
            for j in range(1, len(triangle[i])):  # Start from column 1
                expected = triangle[i - 1][j - 1] + triangle[i][j - 1]
                actual = triangle[i][j]
                assert actual == expected, (
                    f"Element at ({i},{j}) should equal sum pattern"
                )


# ============================================================================
# STIRLING NUMBERS TESTS
# ============================================================================


class TestStirlingNumbers:
    """Test cases for Stirling number functions."""

    @pytest.mark.asyncio
    async def test_stirling_second_known_values(self):
        """Test Stirling numbers of the second kind with known values."""
        known_stirling_second = [
            (4, 2, 7),
            (5, 3, 25),
            (6, 3, 90),
            (5, 2, 15),
            (3, 2, 3),
            (4, 3, 6),
            (6, 4, 65),
            (7, 3, 301),
        ]

        for n, k, expected in known_stirling_second:
            result = await stirling_second(n, k)
            assert result == expected, f"S({n},{k}) should be {expected}, got {result}"

    @pytest.mark.asyncio
    async def test_stirling_first_known_values(self):
        """Test Stirling numbers of the first kind with known values - CORRECTED."""
        known_stirling_first = [
            (4, 2, 11),
            (5, 3, 35),
            (6, 3, 225),
            (5, 2, 50),
            (3, 2, 3),
            (4, 3, 6),
            (6, 4, 85),
            (7, 3, 1624),  # CORRECTED: was 735, but s(7,3) = 1624
            (7, 4, 735),  # ADDED: s(7,4) = 735 (the confused value)
        ]

        for n, k, expected in known_stirling_first:
            result = await stirling_first(n, k)
            assert result == expected, f"s({n},{k}) should be {expected}, got {result}"

    @pytest.mark.asyncio
    async def test_stirling_numbers_edge_cases(self):
        """Test edge cases for Stirling numbers."""
        # S(n,0) = 0 for n > 0, S(0,0) = 1
        assert await stirling_second(0, 0) == 1
        assert await stirling_second(5, 0) == 0
        assert await stirling_second(0, 3) == 0

        # S(n,n) = 1
        for n in range(1, 8):
            assert await stirling_second(n, n) == 1, f"S({n},{n}) should be 1"
            assert await stirling_first(n, n) == 1, f"s({n},{n}) should be 1"

        # S(n,k) = 0 for k > n
        assert await stirling_second(5, 7) == 0
        assert await stirling_first(3, 5) == 0

    @pytest.mark.asyncio
    async def test_stirling_numbers_negative_input(self):
        """Test Stirling numbers with negative input."""
        with pytest.raises(ValueError, match="n and k must be non-negative"):
            await stirling_second(-1, 2)

        with pytest.raises(ValueError, match="n and k must be non-negative"):
            await stirling_first(3, -2)

    @pytest.mark.asyncio
    async def test_stirling_second_recurrence(self):
        """Test Stirling second kind recurrence: S(n,k) = k*S(n-1,k) + S(n-1,k-1)."""
        # Test for several values
        test_cases = [(5, 3), (6, 4), (7, 3), (4, 2)]

        for n, k in test_cases:
            if n > 1 and k > 1:
                s_nk = await stirling_second(n, k)
                s_n1k = await stirling_second(n - 1, k)
                s_n1k1 = await stirling_second(n - 1, k - 1)
                expected = k * s_n1k + s_n1k1
                assert s_nk == expected, f"Recurrence failed for S({n},{k})"

    @pytest.mark.asyncio
    async def test_stirling_first_recurrence(self):
        """Test Stirling first kind recurrence: s(n,k) = (n-1)*s(n-1,k) + s(n-1,k-1)."""
        # Test for several values
        test_cases = [(5, 3), (6, 4), (7, 3), (4, 2)]

        for n, k in test_cases:
            if n > 1 and k > 1:
                s_nk = await stirling_first(n, k)
                s_n1k = await stirling_first(n - 1, k)
                s_n1k1 = await stirling_first(n - 1, k - 1)
                expected = (n - 1) * s_n1k + s_n1k1
                assert s_nk == expected, f"Recurrence failed for s({n},{k})"

    @pytest.mark.asyncio
    async def test_stirling_second_row_generation(self):
        """Test generation of Stirling second kind rows."""
        row_4 = await stirling_second_row(4)
        expected_4 = [0, 1, 7, 6, 1]  # S(4,0), S(4,1), S(4,2), S(4,3), S(4,4)
        assert row_4 == expected_4

        row_5 = await stirling_second_row(5)
        expected_5 = [0, 1, 15, 25, 10, 1]
        assert row_5 == expected_5

    @pytest.mark.asyncio
    async def test_stirling_second_row_consistency(self):
        """Test consistency between stirling_second and stirling_second_row."""
        n = 6
        row = await stirling_second_row(n)

        for k in range(n + 1):
            individual = await stirling_second(n, k)
            assert row[k] == individual, f"S({n},{k}) should match row[{k}]"

    @pytest.mark.asyncio
    async def test_stirling_second_row_edge_cases(self):
        """Test edge cases for Stirling second row generation."""
        assert await stirling_second_row(0) == [1]  # Only S(0,0) = 1

        with pytest.raises(ValueError, match="Row number must be non-negative"):
            await stirling_second_row(-1)

    @pytest.mark.asyncio
    async def test_stirling_bell_relationship(self):
        """Test relationship between Stirling second kind and Bell numbers."""
        # Bell number B_n = sum of S(n,k) for k = 0 to n
        for n in range(1, 8):
            bell_n = await bell_number(n)
            stirling_row = await stirling_second_row(n)
            stirling_sum = sum(stirling_row)
            assert bell_n == stirling_sum, f"B({n}) should equal sum of S({n},k)"


# ============================================================================
# NARAYANA NUMBERS TESTS
# ============================================================================


class TestNarayanaNumbers:
    """Test cases for Narayana number functions."""

    @pytest.mark.asyncio
    async def test_narayana_number_known_values(self):
        """Test Narayana numbers with known values - CORRECTED."""
        known_narayana = [
            (3, 2, 3),
            (4, 2, 6),
            (4, 3, 6),  # CORRECTED: was 4, but N(4,3) = 6
            (5, 2, 10),
            (5, 3, 20),
            (5, 4, 10),
            (6, 3, 50),
            (6, 4, 50),  # CORRECTED: was 60, but N(6,4) = 50
        ]

        for n, k, expected in known_narayana:
            result = await narayana_number(n, k)
            assert result == expected, f"N({n},{k}) should be {expected}, got {result}"

    @pytest.mark.asyncio
    async def test_narayana_triangle_row_generation(self):
        """Test generation of Narayana triangle rows - CORRECTED."""
        row_4 = await narayana_triangle_row(4)
        expected_4 = [1, 6, 6, 1]  # CORRECTED: was [1, 6, 6, 4, 1]
        assert row_4 == expected_4

        row_5 = await narayana_triangle_row(5)
        expected_5 = [1, 10, 20, 10, 1]  # CORRECTED: was [1, 10, 20, 20, 10, 1]
        assert row_5 == expected_5

    @pytest.mark.asyncio
    async def test_narayana_number_edge_cases(self):
        """Test edge cases for Narayana numbers."""
        # N(n,k) = 0 for k <= 0 or k > n
        assert await narayana_number(5, 0) == 0
        assert await narayana_number(5, 6) == 0
        assert await narayana_number(0, 1) == 0

        # N(n,1) = N(n,n) = 1 for n > 0
        for n in range(1, 8):
            assert await narayana_number(n, 1) == 1, f"N({n},1) should be 1"
            assert await narayana_number(n, n) == 1, f"N({n},{n}) should be 1"

    @pytest.mark.asyncio
    async def test_narayana_triangle_consistency(self):
        """Test consistency between narayana_number and narayana_triangle_row."""
        n = 6
        row = await narayana_triangle_row(n)

        for k in range(1, n + 1):
            individual = await narayana_number(n, k)
            assert row[k - 1] == individual, f"N({n},{k}) should match row[{k - 1}]"

    @pytest.mark.asyncio
    async def test_narayana_triangle_edge_cases(self):
        """Test edge cases for Narayana triangle generation."""
        assert await narayana_triangle_row(0) == []
        assert await narayana_triangle_row(1) == [1]
        assert await narayana_triangle_row(2) == [1, 1]

    @pytest.mark.asyncio
    async def test_narayana_catalan_relationship(self):
        """Test relationship between Narayana numbers and Catalan numbers."""
        # Catalan number C_n = sum of N(n,k) for k = 1 to n
        for n in range(1, 8):
            catalan_n = await catalan_number(n)
            narayana_row = await narayana_triangle_row(n)
            narayana_sum = sum(narayana_row)
            assert catalan_n == narayana_sum, f"C({n}) should equal sum of N({n},k)"

    @pytest.mark.asyncio
    async def test_narayana_symmetry(self):
        """Test symmetry property of Narayana numbers: N(n,k) = N(n,n+1-k)."""
        for n in range(2, 8):
            for k in range(1, n + 1):
                narayana_k = await narayana_number(n, k)
                narayana_symmetric = await narayana_number(n, n + 1 - k)
                assert narayana_k == narayana_symmetric, (
                    f"N({n},{k}) should equal N({n},{n + 1 - k})"
                )


# ============================================================================
# INTEGRATION AND MATHEMATICAL PROPERTIES TESTS
# ============================================================================


class TestIntegrationAndProperties:
    """Integration tests and mathematical property verification."""

    @pytest.mark.asyncio
    async def test_catalan_bell_stirling_relationships(self):
        """Test mathematical relationships between different combinatorial numbers."""
        # For small values, verify some known relationships

        # C_3 = 5, B_3 = 5 (coincidence for n=3)
        assert await catalan_number(3) == await bell_number(3) == 5

        # C_4 = 14, B_4 = 15 (close but different)
        c4 = await catalan_number(4)
        b4 = await bell_number(4)
        assert c4 == 14 and b4 == 15

    @pytest.mark.asyncio
    async def test_combinatorial_identities(self):
        """Test various combinatorial identities."""
        # Test that Stirling numbers sum to Bell numbers
        for n in range(1, 8):
            bell_n = await bell_number(n)
            stirling_sum = 0
            for k in range(n + 1):
                stirling_sum += await stirling_second(n, k)
            assert bell_n == stirling_sum, f"Bell({n}) identity failed"

        # Test that Narayana numbers sum to Catalan numbers
        for n in range(1, 8):
            catalan_n = await catalan_number(n)
            narayana_sum = 0
            for k in range(1, n + 1):
                narayana_sum += await narayana_number(n, k)
            assert catalan_n == narayana_sum, f"Catalan({n}) identity failed"

    @pytest.mark.asyncio
    async def test_growth_rates(self):
        """Test that the sequences have expected growth patterns."""
        # Catalan numbers grow exponentially
        catalan_seq = await catalan_sequence(15)
        for i in range(2, len(catalan_seq)):
            ratio = catalan_seq[i] / catalan_seq[i - 1]
            assert ratio > 1, "Catalan numbers should be increasing"
            if i > 5:  # For larger n, ratio approaches 4
                assert 2 < ratio < 5, f"Catalan growth ratio seems wrong at index {i}"

        # Bell numbers grow very rapidly
        bell_seq = await bell_sequence(10)
        for i in range(2, len(bell_seq)):
            ratio = bell_seq[i] / bell_seq[i - 1]
            assert ratio > 1, "Bell numbers should be increasing"
            if i > 3:
                assert ratio > 2, "Bell numbers should grow rapidly"

    @pytest.mark.asyncio
    async def test_boundary_conditions(self):
        """Test boundary conditions and special cases."""
        # Test n=0 cases
        assert await catalan_number(0) == 1
        assert await bell_number(0) == 1
        assert await stirling_second(0, 0) == 1
        assert await stirling_first(0, 0) == 1

        # Test n=1 cases
        assert await catalan_number(1) == 1
        assert await bell_number(1) == 1
        assert await stirling_second(1, 1) == 1
        assert await stirling_first(1, 1) == 1

        # Test k=1 cases for Stirling numbers
        for n in range(1, 8):
            assert await stirling_second(n, 1) == 1, f"S({n},1) should be 1"

    @pytest.mark.asyncio
    async def test_cross_validation(self):
        """Cross-validate results using alternative computation methods."""

        # For small Catalan numbers, verify using binomial coefficient formula
        # C_n = (2n choose n) / (n+1)
        def binomial(n, k):
            if k > n or k < 0:
                return 0
            if k == 0 or k == n:
                return 1

            k = min(k, n - k)  # Take advantage of symmetry
            result = 1
            for i in range(k):
                result = result * (n - i) // (i + 1)
            return result

        for n in range(10):
            catalan_n = await catalan_number(n)
            binomial_formula = binomial(2 * n, n) // (n + 1) if n > 0 else 1
            assert catalan_n == binomial_formula, f"Catalan({n}) formula mismatch"


# ============================================================================
# PERFORMANCE AND ASYNC BEHAVIOR TESTS
# ============================================================================


class TestPerformance:
    """Performance and async behavior tests."""

    @pytest.mark.asyncio
    async def test_all_functions_are_async(self):
        """Test that all combinatorial functions are properly async."""
        operations = [
            catalan_number(10),
            bell_number(8),
            stirling_second(6, 3),
            stirling_first(5, 2),
            narayana_number(5, 3),
            catalan_sequence(8),
            bell_sequence(6),
            bell_triangle(4),
            stirling_second_row(5),
            narayana_triangle_row(4),
        ]

        # Ensure all are coroutines
        for op in operations:
            assert asyncio.iscoroutine(op)

        # Run all operations concurrently
        results = await asyncio.gather(*operations)
        expected_types = [int, int, int, int, int, list, list, list, list, list]

        for i, (result, expected_type) in enumerate(zip(results, expected_types)):
            assert isinstance(result, expected_type), (
                f"Operation {i} returned wrong type"
            )

    @pytest.mark.asyncio
    async def test_concurrent_execution(self):
        """Test that combinatorial operations can run concurrently."""
        start_time = time.time()

        # Run multiple operations concurrently
        tasks = []
        for n in range(1, 15):
            tasks.append(catalan_number(n))
            tasks.append(bell_number(n))
            if n <= 8:  # Stirling numbers are more expensive
                tasks.append(stirling_second(n, min(n // 2 + 1, n)))

        results = await asyncio.gather(*tasks)
        duration = time.time() - start_time

        # Should complete quickly due to async nature
        assert duration < 3.0
        assert len(results) > 0

        # All results should be positive integers
        for result in results:
            assert isinstance(result, int)
            assert result > 0

    @pytest.mark.asyncio
    async def test_large_number_handling(self):
        """Test handling of moderately large numbers."""
        # Test with larger numbers that still complete quickly
        large_tests = [
            catalan_number(20),  # Large Catalan number
            bell_number(15),  # Large Bell number
            stirling_second(12, 6),  # Large Stirling computation
            catalan_sequence(18),  # Long sequence
            bell_triangle(8),  # Large triangle
        ]

        results = await asyncio.gather(*large_tests)

        # Verify results are reasonable
        assert isinstance(results[0], int)  # Catalan result
        assert isinstance(results[1], int)  # Bell result
        assert isinstance(results[2], int)  # Stirling result
        assert isinstance(results[3], list)  # Catalan sequence
        assert isinstance(results[4], list)  # Bell triangle

        # Check that large numbers are actually large
        assert results[0] > 1000000  # C_20 is large
        assert results[1] > 1000000  # B_15 is large
        assert len(results[3]) == 18  # Sequence length
        assert len(results[4]) == 8  # Triangle rows

    @pytest.mark.asyncio
    async def test_memory_efficiency(self):
        """Test that functions don't consume excessive memory."""
        # Generate several sequences and verify they complete
        sequences = await asyncio.gather(
            catalan_sequence(25), bell_sequence(20), bell_triangle(10)
        )

        # Verify sequences have expected lengths
        assert len(sequences[0]) == 25
        assert len(sequences[1]) == 20
        assert len(sequences[2]) == 10

        # Verify triangle structure
        triangle = sequences[2]
        for i, row in enumerate(triangle):
            assert len(row) == i + 1, f"Triangle row {i} should have {i + 1} elements"


# ============================================================================
# ERROR HANDLING TESTS
# ============================================================================


class TestErrorHandling:
    """Test error handling and edge cases."""

    @pytest.mark.asyncio
    async def test_catalan_error_conditions(self):
        """Test error conditions for Catalan functions."""
        # Negative inputs should raise ValueError
        with pytest.raises(ValueError):
            await catalan_number(-1)

        with pytest.raises(ValueError):
            await catalan_number(-10)

    @pytest.mark.asyncio
    async def test_bell_error_conditions(self):
        """Test error conditions for Bell functions."""
        # Negative inputs should raise ValueError
        with pytest.raises(ValueError):
            await bell_number(-1)

        with pytest.raises(ValueError):
            await bell_triangle(-1)

    @pytest.mark.asyncio
    async def test_stirling_error_conditions(self):
        """Test error conditions for Stirling functions."""
        # Negative inputs should raise ValueError
        with pytest.raises(ValueError):
            await stirling_second(-1, 2)

        with pytest.raises(ValueError):
            await stirling_first(3, -1)

        with pytest.raises(ValueError):
            await stirling_second_row(-1)

    @pytest.mark.asyncio
    async def test_edge_case_handling(self):
        """Test edge case handling across all functions."""
        # All functions should handle edge cases gracefully
        edge_cases = [0, 1]

        for n in edge_cases:
            # These should not raise exceptions
            await catalan_number(n)
            await bell_number(n)
            await catalan_sequence(n + 2)
            await bell_sequence(n + 2)

            if n > 0:
                await stirling_second(n, n)
                await stirling_first(n, n)

    @pytest.mark.asyncio
    async def test_error_preserves_async_context(self):
        """Test that errors are properly raised in async context."""
        try:
            await catalan_number(-1)  # Should raise ValueError
            assert False, "Should have raised ValueError"
        except ValueError:
            # Should be able to continue with async operations
            result = await catalan_number(5)
            assert result == 42


# ============================================================================
# PARAMETRIZED TESTS
# ============================================================================


class TestParametrized:
    """Parametrized tests for comprehensive coverage."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "n,expected",
        [
            (0, 1),
            (1, 1),
            (2, 2),
            (3, 5),
            (4, 14),
            (5, 42),
            (6, 132),
            (7, 429),
            (8, 1430),
            (9, 4862),
            (10, 16796),
        ],
    )
    async def test_catalan_number_parametrized(self, n, expected):
        """Parametrized test for Catalan number calculation."""
        assert await catalan_number(n) == expected

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "n,expected",
        [
            (0, 1),
            (1, 1),
            (2, 2),
            (3, 5),
            (4, 15),
            (5, 52),
            (6, 203),
            (7, 877),
            (8, 4140),
            (9, 21147),
            (10, 115975),
        ],
    )
    async def test_bell_number_parametrized(self, n, expected):
        """Parametrized test for Bell number calculation."""
        assert await bell_number(n) == expected

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "n,k,expected",
        [
            (4, 2, 7),
            (5, 3, 25),
            (6, 3, 90),
            (5, 2, 15),
            (3, 2, 3),
            (4, 3, 6),
            (6, 4, 65),
            (7, 3, 301),
        ],
    )
    async def test_stirling_second_parametrized(self, n, k, expected):
        """Parametrized test for Stirling numbers of the second kind."""
        assert await stirling_second(n, k) == expected

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "n,k,expected",
        [
            (4, 2, 11),
            (5, 3, 35),
            (6, 3, 225),
            (5, 2, 50),
            (3, 2, 3),
            (4, 3, 6),
            (6, 4, 85),
            (7, 3, 1624),  # CORRECTED: was 735
        ],
    )
    async def test_stirling_first_parametrized(self, n, k, expected):
        """Parametrized test for Stirling numbers of the first kind - CORRECTED."""
        assert await stirling_first(n, k) == expected

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "n,k,expected",
        [
            (3, 2, 3),
            (4, 2, 6),
            (4, 3, 6),  # CORRECTED: was 4
            (5, 2, 10),
            (5, 3, 20),
            (5, 4, 10),
            (6, 3, 50),
            (6, 4, 50),  # CORRECTED: was 60
        ],
    )
    async def test_narayana_number_parametrized(self, n, k, expected):
        """Parametrized test for Narayana numbers - CORRECTED."""
        assert await narayana_number(n, k) == expected

    @pytest.mark.asyncio
    @pytest.mark.parametrize("catalan_num", [1, 2, 5, 14, 42, 132, 429, 1430, 4862])
    async def test_is_catalan_number_parametrized(self, catalan_num):
        """Parametrized test for Catalan number identification."""
        assert await is_catalan_number(catalan_num)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("non_catalan", [3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16])
    async def test_is_not_catalan_number_parametrized(self, non_catalan):
        """Parametrized test for non-Catalan number identification."""
        assert not await is_catalan_number(non_catalan)


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v", "--tb=short", "--asyncio-mode=auto"])
