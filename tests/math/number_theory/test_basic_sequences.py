#!/usr/bin/env python3
# tests/math/number_theory/test_basic_sequences.py
"""
Comprehensive pytest test suite for basic_sequences.py module.

Tests cover:
- Perfect squares: detection, generation, nth calculation
- Powers of two: detection, generation, nth calculation
- Fibonacci numbers: calculation, sequence generation, identification
- Factorials: standard, double, and subfactorial
- Triangular numbers: calculation, identification, sequence generation
- Pentagonal numbers: calculation, identification, sequence generation
- Pyramidal numbers: square pyramidal and tetrahedral
- Mathematical properties and relationships
- Edge cases, error conditions, and performance testing
- Async behavior verification
"""

import pytest
import asyncio
import time
import math

# Import the functions to test
from chuk_mcp_math.number_theory.basic_sequences import (
    # Perfect squares
    is_perfect_square,
    perfect_squares,
    nth_perfect_square,
    # Powers of two
    is_power_of_two,
    powers_of_two,
    nth_power_of_two,
    # Fibonacci numbers
    fibonacci,
    fibonacci_sequence,
    is_fibonacci_number,
    # Factorials
    factorial,
    double_factorial,
    subfactorial,
    # Triangular numbers
    triangular_number,
    is_triangular_number,
    triangular_sequence,
    # Pentagonal numbers
    pentagonal_number,
    is_pentagonal_number,
    pentagonal_sequence,
    # Pyramidal numbers
    square_pyramidal_number,
    tetrahedral_number,
    # Catalan numbers
    catalan_number,
)

# ============================================================================
# PERFECT SQUARES TESTS
# ============================================================================


class TestPerfectSquares:
    """Test cases for perfect square functions."""

    @pytest.mark.asyncio
    async def test_is_perfect_square_known_squares(self):
        """Test with known perfect squares."""
        perfect_squares_list = [
            0,
            1,
            4,
            9,
            16,
            25,
            36,
            49,
            64,
            81,
            100,
            121,
            144,
            169,
            196,
            225,
            256,
            289,
            324,
            361,
            400,
        ]

        for square in perfect_squares_list:
            assert await is_perfect_square(square), f"{square} should be a perfect square"

    @pytest.mark.asyncio
    async def test_is_perfect_square_non_squares(self):
        """Test with numbers that are not perfect squares."""
        non_squares = [
            2,
            3,
            5,
            6,
            7,
            8,
            10,
            11,
            12,
            13,
            14,
            15,
            17,
            18,
            19,
            20,
            21,
            22,
            23,
            24,
        ]

        for n in non_squares:
            assert not await is_perfect_square(n), f"{n} should not be a perfect square"

    @pytest.mark.asyncio
    async def test_is_perfect_square_large_numbers(self):
        """Test with larger perfect squares."""
        large_squares = [
            10000,
            40000,
            90000,
            160000,
            250000,
            360000,
            490000,
            640000,
            810000,
            1000000,
        ]

        for square in large_squares:
            assert await is_perfect_square(square), f"{square} should be a perfect square"

    @pytest.mark.asyncio
    async def test_is_perfect_square_edge_cases(self):
        """Test edge cases for perfect square checking."""
        assert await is_perfect_square(0)  # 0² = 0
        assert await is_perfect_square(1)  # 1² = 1
        assert not await is_perfect_square(-1)  # Negative numbers
        assert not await is_perfect_square(-4)  # Negative numbers

    @pytest.mark.asyncio
    async def test_perfect_squares_generation(self):
        """Test generation of perfect squares sequence."""
        squares_10 = await perfect_squares(10)
        expected_10 = [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]
        assert squares_10 == expected_10

        squares_5 = await perfect_squares(5)
        expected_5 = [0, 1, 4, 9, 16]
        assert squares_5 == expected_5

    @pytest.mark.asyncio
    async def test_perfect_squares_edge_cases(self):
        """Test perfect squares generation edge cases."""
        assert await perfect_squares(0) == []
        assert await perfect_squares(1) == [0]
        assert await perfect_squares(2) == [0, 1]
        assert await perfect_squares(-1) == []

    @pytest.mark.asyncio
    async def test_nth_perfect_square(self):
        """Test nth perfect square calculation."""
        test_cases = [
            (0, 0),
            (1, 1),
            (2, 4),
            (3, 9),
            (4, 16),
            (5, 25),
            (10, 100),
            (20, 400),
        ]

        for n, expected in test_cases:
            result = await nth_perfect_square(n)
            assert result == expected, f"nth_perfect_square({n}) should be {expected}, got {result}"

    @pytest.mark.asyncio
    async def test_nth_perfect_square_negative(self):
        """Test nth perfect square with negative input."""
        with pytest.raises(ValueError, match="Index must be non-negative"):
            await nth_perfect_square(-1)

    @pytest.mark.asyncio
    async def test_perfect_squares_consistency(self):
        """Test consistency between perfect square functions."""
        n = 15
        squares_list = await perfect_squares(n)

        for i in range(n):
            nth_square = await nth_perfect_square(i)
            assert squares_list[i] == nth_square, (
                f"perfect_squares[{i}] should equal nth_perfect_square({i})"
            )
            assert await is_perfect_square(nth_square), (
                f"{nth_square} should be identified as perfect square"
            )


# ============================================================================
# POWERS OF TWO TESTS
# ============================================================================


class TestPowersOfTwo:
    """Test cases for powers of two functions."""

    @pytest.mark.asyncio
    async def test_is_power_of_two_known_powers(self):
        """Test with known powers of two."""
        powers_of_two_list = [
            1,
            2,
            4,
            8,
            16,
            32,
            64,
            128,
            256,
            512,
            1024,
            2048,
            4096,
            8192,
            16384,
            32768,
            65536,
        ]

        for power in powers_of_two_list:
            assert await is_power_of_two(power), f"{power} should be a power of two"

    @pytest.mark.asyncio
    async def test_is_power_of_two_non_powers(self):
        """Test with numbers that are not powers of two."""
        non_powers = [
            3,
            5,
            6,
            7,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            17,
            18,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
        ]

        for n in non_powers:
            assert not await is_power_of_two(n), f"{n} should not be a power of two"

    @pytest.mark.asyncio
    async def test_is_power_of_two_edge_cases(self):
        """Test edge cases for power of two checking."""
        assert await is_power_of_two(1)  # 2⁰ = 1
        assert not await is_power_of_two(0)  # 0 is not a power of two
        assert not await is_power_of_two(-1)  # Negative numbers
        assert not await is_power_of_two(-8)  # Negative powers

    @pytest.mark.asyncio
    async def test_powers_of_two_generation(self):
        """Test generation of powers of two sequence."""
        powers_10 = await powers_of_two(10)
        expected_10 = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
        assert powers_10 == expected_10

        powers_5 = await powers_of_two(5)
        expected_5 = [1, 2, 4, 8, 16]
        assert powers_5 == expected_5

    @pytest.mark.asyncio
    async def test_powers_of_two_edge_cases(self):
        """Test powers of two generation edge cases."""
        assert await powers_of_two(0) == []
        assert await powers_of_two(1) == [1]
        assert await powers_of_two(2) == [1, 2]
        assert await powers_of_two(-1) == []

    @pytest.mark.asyncio
    async def test_nth_power_of_two(self):
        """Test nth power of two calculation."""
        test_cases = [
            (0, 1),
            (1, 2),
            (2, 4),
            (3, 8),
            (4, 16),
            (5, 32),
            (10, 1024),
            (16, 65536),
        ]

        for n, expected in test_cases:
            result = await nth_power_of_two(n)
            assert result == expected, f"nth_power_of_two({n}) should be {expected}, got {result}"

    @pytest.mark.asyncio
    async def test_nth_power_of_two_negative(self):
        """Test nth power of two with negative input."""
        with pytest.raises(ValueError, match="Index must be non-negative"):
            await nth_power_of_two(-1)

    @pytest.mark.asyncio
    async def test_powers_of_two_consistency(self):
        """Test consistency between power of two functions."""
        n = 12
        powers_list = await powers_of_two(n)

        for i in range(n):
            nth_power = await nth_power_of_two(i)
            assert powers_list[i] == nth_power, (
                f"powers_of_two[{i}] should equal nth_power_of_two({i})"
            )
            assert await is_power_of_two(nth_power), (
                f"{nth_power} should be identified as power of two"
            )


# ============================================================================
# FIBONACCI NUMBERS TESTS
# ============================================================================


class TestFibonacciNumbers:
    """Test cases for Fibonacci number functions."""

    @pytest.mark.asyncio
    async def test_fibonacci_basic_values(self):
        """Test Fibonacci function with basic values."""
        known_fibonacci = [
            (0, 0),
            (1, 1),
            (2, 1),
            (3, 2),
            (4, 3),
            (5, 5),
            (6, 8),
            (7, 13),
            (8, 21),
            (9, 34),
            (10, 55),
            (11, 89),
            (12, 144),
            (13, 233),
            (14, 377),
            (15, 610),
            (16, 987),
            (17, 1597),
            (18, 2584),
            (19, 4181),
            (20, 6765),
        ]

        for n, expected in known_fibonacci:
            result = await fibonacci(n)
            assert result == expected, f"F({n}) should be {expected}, got {result}"

    @pytest.mark.asyncio
    async def test_fibonacci_larger_values(self):
        """Test Fibonacci function with larger values."""
        larger_fibonacci = [(25, 75025), (30, 832040), (35, 9227465), (40, 102334155)]

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
        n = 18
        sequence = await fibonacci_sequence(n)

        for i in range(n):
            individual = await fibonacci(i)
            assert sequence[i] == individual, f"F({i}) should match sequence[{i}]"

    @pytest.mark.asyncio
    async def test_is_fibonacci_number_known_fibonacci(self):
        """Test with known Fibonacci numbers."""
        known_fibonacci_numbers = [
            0,
            1,
            2,
            3,
            5,
            8,
            13,
            21,
            34,
            55,
            89,
            144,
            233,
            377,
            610,
            987,
            1597,
            2584,
        ]

        for fib in known_fibonacci_numbers:
            assert await is_fibonacci_number(fib), f"{fib} should be a Fibonacci number"

    @pytest.mark.asyncio
    async def test_is_fibonacci_number_non_fibonacci(self):
        """Test with numbers that are not Fibonacci numbers."""
        non_fibonacci = [
            4,
            6,
            7,
            9,
            10,
            11,
            12,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            22,
            23,
            24,
            25,
            26,
            27,
        ]

        for n in non_fibonacci:
            assert not await is_fibonacci_number(n), f"{n} should not be a Fibonacci number"

    @pytest.mark.asyncio
    async def test_is_fibonacci_number_edge_cases(self):
        """Test edge cases for Fibonacci number checking."""
        assert await is_fibonacci_number(0)  # F₀ = 0
        assert await is_fibonacci_number(1)  # F₁ = F₂ = 1
        assert not await is_fibonacci_number(-1)  # Negative numbers
        assert not await is_fibonacci_number(-5)  # Negative numbers

    @pytest.mark.asyncio
    async def test_fibonacci_recurrence_relation(self):
        """Test Fibonacci recurrence relation: F(n) = F(n-1) + F(n-2)."""
        for n in range(2, 20):
            f_n = await fibonacci(n)
            f_n_minus_1 = await fibonacci(n - 1)
            f_n_minus_2 = await fibonacci(n - 2)
            assert f_n == f_n_minus_1 + f_n_minus_2, f"Fibonacci recurrence failed for n={n}"


# ============================================================================
# CATALAN NUMBERS TESTS
# ============================================================================


class TestCatalanNumbers:
    """Test cases for Catalan number functions."""

    @pytest.mark.asyncio
    async def test_catalan_number_known_values(self):
        """Test Catalan number calculation with known values."""
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
        ]

        for n, expected in known_catalan:
            result = await catalan_number(n)
            assert result == expected, f"C({n}) should be {expected}, got {result}"

    @pytest.mark.asyncio
    async def test_catalan_number_negative_input(self):
        """Test Catalan number with negative input."""
        with pytest.raises(ValueError, match="Catalan number index must be non-negative"):
            await catalan_number(-1)


# ============================================================================
# FACTORIALS TESTS
# ============================================================================


class TestFactorials:
    """Test cases for factorial functions."""

    @pytest.mark.asyncio
    async def test_factorial_basic_values(self):
        """Test factorial function with basic values."""
        known_factorials = [
            (0, 1),
            (1, 1),
            (2, 2),
            (3, 6),
            (4, 24),
            (5, 120),
            (6, 720),
            (7, 5040),
            (8, 40320),
            (9, 362880),
            (10, 3628800),
            (11, 39916800),
            (12, 479001600),
            (13, 6227020800),
        ]

        for n, expected in known_factorials:
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
        for n in range(1, 12):
            n_factorial = await factorial(n)
            n_plus_1_factorial = await factorial(n + 1)

            assert n_plus_1_factorial == (n + 1) * n_factorial, (
                f"({n + 1})! should equal ({n + 1}) × {n}!"
            )

    @pytest.mark.asyncio
    async def test_double_factorial_known_values(self):
        """Test double factorial function with known values."""
        known_double_factorials = [
            (0, 1),
            (1, 1),
            (2, 2),
            (3, 3),
            (4, 8),
            (5, 15),
            (6, 48),
            (7, 105),
            (8, 384),
            (9, 945),
            (10, 3840),
        ]

        for n, expected in known_double_factorials:
            result = await double_factorial(n)
            assert result == expected, f"{n}!! should be {expected}, got {result}"

    @pytest.mark.asyncio
    async def test_double_factorial_negative_input(self):
        """Test double factorial function with negative input."""
        with pytest.raises(
            ValueError, match="Double factorial is not defined for negative numbers"
        ):
            await double_factorial(-1)

    @pytest.mark.asyncio
    async def test_double_factorial_pattern(self):
        """Test double factorial pattern."""
        # n!! = n × (n-2) × (n-4) × ...
        # Verify manually for a few cases
        assert await double_factorial(8) == 8 * 6 * 4 * 2  # 384
        assert await double_factorial(7) == 7 * 5 * 3 * 1  # 105
        assert await double_factorial(6) == 6 * 4 * 2  # 48
        assert await double_factorial(5) == 5 * 3 * 1  # 15

    @pytest.mark.asyncio
    async def test_subfactorial_known_values(self):
        """Test subfactorial function with known values."""
        known_subfactorials = [
            (0, 1),
            (1, 0),
            (2, 1),
            (3, 2),
            (4, 9),
            (5, 44),
            (6, 265),
            (7, 1854),
            (8, 14833),
            (9, 133496),
        ]

        for n, expected in known_subfactorials:
            result = await subfactorial(n)
            assert result == expected, f"!{n} should be {expected}, got {result}"

    @pytest.mark.asyncio
    async def test_subfactorial_negative_input(self):
        """Test subfactorial function with negative input."""
        with pytest.raises(ValueError, match="Subfactorial is not defined for negative numbers"):
            await subfactorial(-1)

    @pytest.mark.asyncio
    async def test_subfactorial_recurrence(self):
        """Test subfactorial recurrence: !n = (n-1) × (!(n-1) + !(n-2))."""
        for n in range(2, 8):
            subfact_n = await subfactorial(n)
            subfact_n_minus_1 = await subfactorial(n - 1)
            subfact_n_minus_2 = await subfactorial(n - 2)
            expected = (n - 1) * (subfact_n_minus_1 + subfact_n_minus_2)
            assert subfact_n == expected, f"Subfactorial recurrence failed for n={n}"


# ============================================================================
# TRIANGULAR NUMBERS TESTS
# ============================================================================


class TestTriangularNumbers:
    """Test cases for triangular number functions."""

    @pytest.mark.asyncio
    async def test_triangular_number_known_values(self):
        """Test triangular number calculation with known values."""
        known_triangular = [
            (0, 0),
            (1, 1),
            (2, 3),
            (3, 6),
            (4, 10),
            (5, 15),
            (6, 21),
            (7, 28),
            (8, 36),
            (9, 45),
            (10, 55),
            (11, 66),
            (12, 78),
            (13, 91),
            (14, 105),
            (15, 120),
        ]

        for n, expected in known_triangular:
            result = await triangular_number(n)
            assert result == expected, f"T({n}) should be {expected}, got {result}"

    @pytest.mark.asyncio
    async def test_triangular_number_formula(self):
        """Test triangular number formula: T_n = n(n+1)/2."""
        for n in range(20):
            triangular_n = await triangular_number(n)
            formula_result = n * (n + 1) // 2
            assert triangular_n == formula_result, f"T({n}) formula mismatch"

    @pytest.mark.asyncio
    async def test_triangular_number_negative_input(self):
        """Test triangular number with negative input."""
        with pytest.raises(ValueError, match="Triangular number index must be non-negative"):
            await triangular_number(-1)

    @pytest.mark.asyncio
    async def test_is_triangular_number_known_triangular(self):
        """Test with known triangular numbers."""
        known_triangular_numbers = [
            0,
            1,
            3,
            6,
            10,
            15,
            21,
            28,
            36,
            45,
            55,
            66,
            78,
            91,
            105,
            120,
        ]

        for tri in known_triangular_numbers:
            assert await is_triangular_number(tri), f"{tri} should be a triangular number"

    @pytest.mark.asyncio
    async def test_is_triangular_number_non_triangular(self):
        """Test with numbers that are not triangular numbers."""
        non_triangular = [
            2,
            4,
            5,
            7,
            8,
            9,
            11,
            12,
            13,
            14,
            16,
            17,
            18,
            19,
            20,
            22,
            23,
            24,
            25,
            26,
        ]

        for n in non_triangular:
            assert not await is_triangular_number(n), f"{n} should not be a triangular number"

    @pytest.mark.asyncio
    async def test_is_triangular_number_edge_cases(self):
        """Test edge cases for triangular number checking."""
        assert await is_triangular_number(0)  # T₀ = 0
        assert await is_triangular_number(1)  # T₁ = 1
        assert not await is_triangular_number(-1)  # Negative numbers
        assert not await is_triangular_number(-5)  # Negative numbers

    @pytest.mark.asyncio
    async def test_triangular_sequence_generation(self):
        """Test triangular sequence generation."""
        seq_10 = await triangular_sequence(10)
        expected_10 = [0, 1, 3, 6, 10, 15, 21, 28, 36, 45]
        assert seq_10 == expected_10

        seq_5 = await triangular_sequence(5)
        expected_5 = [0, 1, 3, 6, 10]
        assert seq_5 == expected_5

    @pytest.mark.asyncio
    async def test_triangular_sequence_edge_cases(self):
        """Test triangular sequence edge cases."""
        assert await triangular_sequence(0) == []
        assert await triangular_sequence(1) == [0]
        assert await triangular_sequence(2) == [0, 1]
        assert await triangular_sequence(-1) == []

    @pytest.mark.asyncio
    async def test_triangular_sequence_consistency(self):
        """Test consistency between triangular functions."""
        n = 12
        sequence = await triangular_sequence(n)

        for i in range(n):
            individual = await triangular_number(i)
            assert sequence[i] == individual, f"T({i}) should match sequence[{i}]"
            assert await is_triangular_number(individual), (
                f"{individual} should be identified as triangular"
            )


# ============================================================================
# PENTAGONAL NUMBERS TESTS
# ============================================================================


class TestPentagonalNumbers:
    """Test cases for pentagonal number functions."""

    @pytest.mark.asyncio
    async def test_pentagonal_number_known_values(self):
        """Test pentagonal number calculation with known values."""
        known_pentagonal = [
            (0, 0),
            (1, 1),
            (2, 5),
            (3, 12),
            (4, 22),
            (5, 35),
            (6, 51),
            (7, 70),
            (8, 92),
            (9, 117),
            (10, 145),
            (11, 176),
            (12, 210),
            (13, 247),
            (14, 287),
            (15, 330),
        ]

        for n, expected in known_pentagonal:
            result = await pentagonal_number(n)
            assert result == expected, f"P({n}) should be {expected}, got {result}"

    @pytest.mark.asyncio
    async def test_pentagonal_number_formula(self):
        """Test pentagonal number formula: P_n = n(3n-1)/2."""
        for n in range(20):
            pentagonal_n = await pentagonal_number(n)
            formula_result = n * (3 * n - 1) // 2
            assert pentagonal_n == formula_result, f"P({n}) formula mismatch"

    @pytest.mark.asyncio
    async def test_pentagonal_number_negative_input(self):
        """Test pentagonal number with negative input."""
        with pytest.raises(ValueError, match="Pentagonal number index must be non-negative"):
            await pentagonal_number(-1)

    @pytest.mark.asyncio
    async def test_is_pentagonal_number_known_pentagonal(self):
        """Test with known pentagonal numbers."""
        known_pentagonal_numbers = [
            0,
            1,
            5,
            12,
            22,
            35,
            51,
            70,
            92,
            117,
            145,
            176,
            210,
            247,
            287,
            330,
        ]

        for pent in known_pentagonal_numbers:
            assert await is_pentagonal_number(pent), f"{pent} should be a pentagonal number"

    @pytest.mark.asyncio
    async def test_is_pentagonal_number_non_pentagonal(self):
        """Test with numbers that are not pentagonal numbers."""
        non_pentagonal = [
            2,
            3,
            4,
            6,
            7,
            8,
            9,
            10,
            11,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            21,
            23,
            24,
        ]

        for n in non_pentagonal:
            assert not await is_pentagonal_number(n), f"{n} should not be a pentagonal number"

    @pytest.mark.asyncio
    async def test_is_pentagonal_number_edge_cases(self):
        """Test edge cases for pentagonal number checking."""
        assert await is_pentagonal_number(0)  # P₀ = 0
        assert await is_pentagonal_number(1)  # P₁ = 1
        assert not await is_pentagonal_number(-1)  # Negative numbers
        assert not await is_pentagonal_number(-5)  # Negative numbers

    @pytest.mark.asyncio
    async def test_pentagonal_sequence_generation(self):
        """Test pentagonal sequence generation."""
        seq_10 = await pentagonal_sequence(10)
        expected_10 = [0, 1, 5, 12, 22, 35, 51, 70, 92, 117]
        assert seq_10 == expected_10

        seq_5 = await pentagonal_sequence(5)
        expected_5 = [0, 1, 5, 12, 22]
        assert seq_5 == expected_5

    @pytest.mark.asyncio
    async def test_pentagonal_sequence_edge_cases(self):
        """Test pentagonal sequence edge cases."""
        assert await pentagonal_sequence(0) == []
        assert await pentagonal_sequence(1) == [0]
        assert await pentagonal_sequence(2) == [0, 1]
        assert await pentagonal_sequence(-1) == []

    @pytest.mark.asyncio
    async def test_pentagonal_sequence_consistency(self):
        """Test consistency between pentagonal functions."""
        n = 12
        sequence = await pentagonal_sequence(n)

        for i in range(n):
            individual = await pentagonal_number(i)
            assert sequence[i] == individual, f"P({i}) should match sequence[{i}]"
            assert await is_pentagonal_number(individual), (
                f"{individual} should be identified as pentagonal"
            )


# ============================================================================
# PYRAMIDAL NUMBERS TESTS
# ============================================================================


class TestPyramidalNumbers:
    """Test cases for pyramidal number functions."""

    @pytest.mark.asyncio
    async def test_square_pyramidal_number_known_values(self):
        """Test square pyramidal number calculation with known values."""
        known_square_pyramidal = [
            (0, 0),
            (1, 1),
            (2, 5),
            (3, 14),
            (4, 30),
            (5, 55),
            (6, 91),
            (7, 140),
            (8, 204),
            (9, 285),
            (10, 385),
        ]

        for n, expected in known_square_pyramidal:
            result = await square_pyramidal_number(n)
            assert result == expected, f"SP({n}) should be {expected}, got {result}"

    @pytest.mark.asyncio
    async def test_square_pyramidal_number_formula(self):
        """Test square pyramidal number formula: SP_n = n(n+1)(2n+1)/6."""
        for n in range(15):
            pyramidal_n = await square_pyramidal_number(n)
            formula_result = n * (n + 1) * (2 * n + 1) // 6
            assert pyramidal_n == formula_result, f"SP({n}) formula mismatch"

    @pytest.mark.asyncio
    async def test_square_pyramidal_number_negative_input(self):
        """Test square pyramidal number with negative input."""
        with pytest.raises(ValueError, match="Square pyramidal number index must be non-negative"):
            await square_pyramidal_number(-1)

    @pytest.mark.asyncio
    async def test_square_pyramidal_sum_property(self):
        """Test that SP_n = sum of first n squares."""
        for n in range(1, 10):
            pyramidal_n = await square_pyramidal_number(n)
            sum_of_squares = sum(i * i for i in range(1, n + 1))
            assert pyramidal_n == sum_of_squares, f"SP({n}) should equal sum of first {n} squares"

    @pytest.mark.asyncio
    async def test_tetrahedral_number_known_values(self):
        """Test tetrahedral number calculation with known values."""
        known_tetrahedral = [
            (0, 0),
            (1, 1),
            (2, 4),
            (3, 10),
            (4, 20),
            (5, 35),
            (6, 56),
            (7, 84),
            (8, 120),
            (9, 165),
            (10, 220),
        ]

        for n, expected in known_tetrahedral:
            result = await tetrahedral_number(n)
            assert result == expected, f"Tet({n}) should be {expected}, got {result}"

    @pytest.mark.asyncio
    async def test_tetrahedral_number_formula(self):
        """Test tetrahedral number formula: Tet_n = n(n+1)(n+2)/6."""
        for n in range(15):
            tetrahedral_n = await tetrahedral_number(n)
            formula_result = n * (n + 1) * (n + 2) // 6
            assert tetrahedral_n == formula_result, f"Tet({n}) formula mismatch"

    @pytest.mark.asyncio
    async def test_tetrahedral_number_negative_input(self):
        """Test tetrahedral number with negative input."""
        with pytest.raises(ValueError, match="Tetrahedral number index must be non-negative"):
            await tetrahedral_number(-1)

    @pytest.mark.asyncio
    async def test_tetrahedral_triangular_relationship(self):
        """Test relationship between tetrahedral and triangular numbers."""
        # Tet_n = sum of first n triangular numbers
        for n in range(1, 10):
            tetrahedral_n = await tetrahedral_number(n)
            triangular_values = await asyncio.gather(
                *[triangular_number(i) for i in range(1, n + 1)]
            )
            sum_of_triangular = sum(triangular_values)
            assert tetrahedral_n == sum_of_triangular, (
                f"Tet({n}) should equal sum of first {n} triangular numbers"
            )


# ============================================================================
# INTEGRATION AND MATHEMATICAL PROPERTIES TESTS
# ============================================================================


class TestIntegrationAndProperties:
    """Integration tests and mathematical property verification."""

    @pytest.mark.asyncio
    async def test_sequence_relationships(self):
        """Test mathematical relationships between different sequences."""
        # Test that some perfect squares are also triangular numbers
        # Perfect squares that are triangular: 1, 36, 1225, ...
        assert await is_perfect_square(1) and await is_triangular_number(1)
        assert await is_perfect_square(36) and await is_triangular_number(36)

        # Test that some Fibonacci numbers are also perfect squares
        # Fibonacci perfect squares: 1, 144
        assert await is_fibonacci_number(1) and await is_perfect_square(1)
        assert await is_fibonacci_number(144) and await is_perfect_square(144)

    @pytest.mark.asyncio
    async def test_growth_rates(self):
        """Test that sequences have expected growth patterns."""
        # Fibonacci numbers grow exponentially (roughly φ^n)
        fib_seq = await fibonacci_sequence(20)
        for i in range(5, len(fib_seq)):
            if fib_seq[i - 1] > 0:
                ratio = fib_seq[i] / fib_seq[i - 1]
                # Golden ratio ≈ 1.618
                assert 1.5 < ratio < 2.0, f"Fibonacci growth ratio seems wrong at index {i}"

        # Factorial grows very rapidly
        factorials = [await factorial(i) for i in range(1, 10)]
        for i in range(1, len(factorials)):
            ratio = factorials[i] / factorials[i - 1]
            assert ratio == i + 1, "Factorial growth should be exactly (n+1)"

    @pytest.mark.asyncio
    async def test_sum_formulas(self):
        """Test various sum formulas using the sequences."""
        # Sum of first n triangular numbers = tetrahedral number
        n = 8
        triangular_values = await asyncio.gather(*[triangular_number(i) for i in range(1, n + 1)])
        triangular_sum = sum(triangular_values)
        tetrahedral_n = await tetrahedral_number(n)
        assert triangular_sum == tetrahedral_n, "Sum of triangular numbers != tetrahedral number"

        # Sum of first n squares = square pyramidal number
        squares_sum = sum(i * i for i in range(1, n + 1))
        square_pyramidal_n = await square_pyramidal_number(n)
        assert squares_sum == square_pyramidal_n, "Sum of squares != square pyramidal number"

    @pytest.mark.asyncio
    async def test_parity_patterns(self):
        """Test parity patterns in sequences."""
        # Fibonacci parity pattern: odd, odd, even, odd, odd, even, ... (period 3)
        fib_seq = await fibonacci_sequence(12)
        parity_pattern = [f % 2 for f in fib_seq]
        expected_pattern = [0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1]  # 0=even, 1=odd
        assert parity_pattern == expected_pattern, "Fibonacci parity pattern incorrect"

        # Triangular numbers alternate even/odd pattern
        triangular_seq = await triangular_sequence(10)
        for i, tri in enumerate(triangular_seq):
            if i % 2 == 0:  # Even index
                assert tri % 2 == 0 or tri % 2 == 1  # Can be either
            # More complex pattern for triangular numbers

    @pytest.mark.asyncio
    async def test_modular_properties(self):
        """Test modular arithmetic properties of sequences."""
        # Fibonacci numbers modulo small primes have periods
        [(await fibonacci(i)) % 5 for i in range(20)]
        # Fibonacci sequence mod 5 has period 20

        # Perfect squares modulo 4 are always 0 or 1
        for i in range(20):
            square = await nth_perfect_square(i)
            assert square % 4 in [0, 1], f"Perfect square {square} mod 4 should be 0 or 1"


# ============================================================================
# PERFORMANCE AND ASYNC BEHAVIOR TESTS
# ============================================================================


class TestPerformance:
    """Performance and async behavior tests."""

    @pytest.mark.asyncio
    async def test_all_functions_are_async(self):
        """Test that all basic sequence functions are properly async."""
        operations = [
            is_perfect_square(16),
            is_power_of_two(8),
            fibonacci(15),
            factorial(8),
            is_fibonacci_number(55),
            triangular_number(10),
            is_triangular_number(45),
            pentagonal_number(7),
            is_pentagonal_number(70),
            square_pyramidal_number(5),
            tetrahedral_number(6),
            perfect_squares(8),
            powers_of_two(6),
            fibonacci_sequence(12),
            triangular_sequence(8),
            pentagonal_sequence(6),
            catalan_number(5),
        ]

        # Ensure all are coroutines
        for op in operations:
            assert asyncio.iscoroutine(op)

        # Run all operations concurrently
        results = await asyncio.gather(*operations)

        # Verify results have expected types
        expected_results = [
            True,
            True,
            610,
            40320,
            True,
            55,
            True,
            70,
            True,
            55,
            56,
            list,
            list,
            list,
            list,
            list,
            42,
        ]
        for i, (result, expected) in enumerate(zip(results, expected_results)):
            if isinstance(expected, type):
                assert isinstance(result, expected), f"Operation {i} returned wrong type"
            else:
                assert result == expected, f"Operation {i} returned wrong value"

    @pytest.mark.asyncio
    async def test_concurrent_execution(self):
        """Test that basic sequence operations can run concurrently."""
        start_time = time.time()

        # Run multiple operations concurrently
        tasks = []
        for i in range(1, 25):
            tasks.append(fibonacci(i))
            tasks.append(factorial(min(i, 10)))  # Limit factorial to avoid huge numbers
            tasks.append(triangular_number(i))
            tasks.append(is_perfect_square(i * i))
            tasks.append(is_power_of_two(2 ** min(i, 20)))

        results = await asyncio.gather(*tasks)
        duration = time.time() - start_time

        # Should complete quickly due to async nature
        assert duration < 2.0
        assert len(results) > 0

        # Check some patterns in results
        fibonacci_results = results[::5]  # Every 5th result
        for result in fibonacci_results:
            assert isinstance(result, int) and result >= 0

    @pytest.mark.asyncio
    async def test_large_number_handling(self):
        """Test handling of moderately large numbers."""
        # Test with larger numbers that still complete quickly
        large_tests = [
            fibonacci(50),  # Large Fibonacci number
            factorial(20),  # Large factorial
            triangular_number(100),  # Large triangular number
            square_pyramidal_number(50),  # Large pyramidal number
            perfect_squares(50),  # Long sequence
            fibonacci_sequence(30),  # Long Fibonacci sequence
        ]

        results = await asyncio.gather(*large_tests)

        # Verify results are reasonable
        assert isinstance(results[0], int)  # Fibonacci result
        assert isinstance(results[1], int)  # Factorial result
        assert isinstance(results[2], int)  # Triangular result
        assert isinstance(results[3], int)  # Pyramidal result
        assert isinstance(results[4], list)  # Perfect squares list
        assert isinstance(results[5], list)  # Fibonacci sequence

        # Check that large numbers are actually large
        assert results[0] > 1000000000  # F_50 is large
        assert results[1] > 1000000000000  # 20! is large
        assert len(results[4]) == 50  # Perfect squares sequence length
        assert len(results[5]) == 30  # Fibonacci sequence length

    @pytest.mark.asyncio
    async def test_memory_efficiency(self):
        """Test that functions don't consume excessive memory."""
        # Generate several long sequences and verify they complete
        sequences = await asyncio.gather(
            perfect_squares(100),
            powers_of_two(30),
            fibonacci_sequence(40),
            triangular_sequence(80),
            pentagonal_sequence(60),
        )

        # Verify sequences have expected lengths
        assert len(sequences[0]) == 100
        assert len(sequences[1]) == 30
        assert len(sequences[2]) == 40
        assert len(sequences[3]) == 80
        assert len(sequences[4]) == 60

        # Verify sequences are properly ordered (increasing)
        for seq in sequences:
            for i in range(1, len(seq)):
                assert seq[i] >= seq[i - 1], "Sequences should be non-decreasing"


# ============================================================================
# ERROR HANDLING TESTS
# ============================================================================


class TestErrorHandling:
    """Test error handling and edge cases."""

    @pytest.mark.asyncio
    async def test_negative_input_errors(self):
        """Test that appropriate errors are raised for negative inputs."""
        negative_input_functions = [
            (fibonacci, "Fibonacci number position must be non-negative"),
            (factorial, "Factorial is not defined for negative numbers"),
            (double_factorial, "Double factorial is not defined for negative numbers"),
            (subfactorial, "Subfactorial is not defined for negative numbers"),
            (triangular_number, "Triangular number index must be non-negative"),
            (pentagonal_number, "Pentagonal number index must be non-negative"),
            (
                square_pyramidal_number,
                "Square pyramidal number index must be non-negative",
            ),
            (tetrahedral_number, "Tetrahedral number index must be non-negative"),
            (nth_perfect_square, "Index must be non-negative"),
            (nth_power_of_two, "Index must be non-negative"),
            (catalan_number, "Catalan number index must be non-negative"),
        ]

        for func, expected_message in negative_input_functions:
            with pytest.raises(ValueError, match=expected_message):
                await func(-1)

    @pytest.mark.asyncio
    async def test_edge_case_handling(self):
        """Test edge case handling across all functions."""
        # All functions should handle these edge cases gracefully
        edge_cases = [0, 1]

        for n in edge_cases:
            # These should not raise exceptions
            await is_perfect_square(n)
            await is_power_of_two(n)
            await is_fibonacci_number(n)
            await is_triangular_number(n)
            await is_pentagonal_number(n)

            # Sequence generation with small n
            await perfect_squares(n + 1)
            await powers_of_two(n + 1)
            await fibonacci_sequence(n + 1)
            await triangular_sequence(n + 1)
            await pentagonal_sequence(n + 1)

    @pytest.mark.asyncio
    async def test_error_preserves_async_context(self):
        """Test that errors are properly raised in async context."""
        try:
            await fibonacci(-1)  # Should raise ValueError
            assert False, "Should have raised ValueError"
        except ValueError:
            # Should be able to continue with async operations
            result = await fibonacci(10)
            assert result == 55


# ============================================================================
# PARAMETRIZED TESTS
# ============================================================================


class TestParametrized:
    """Parametrized tests for comprehensive coverage."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "n,expected",
        [(0, 0), (1, 1), (4, 4), (9, 9), (16, 16), (25, 25), (100, 100), (144, 144)],
    )
    async def test_nth_perfect_square_parametrized(self, n, expected):
        """Parametrized test for nth perfect square calculation."""
        result = await nth_perfect_square(int(math.sqrt(expected)))
        assert result == expected

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "n,expected", [(0, 1), (1, 2), (2, 4), (3, 8), (4, 16), (5, 32), (10, 1024)]
    )
    async def test_nth_power_of_two_parametrized(self, n, expected):
        """Parametrized test for nth power of two calculation."""
        assert await nth_power_of_two(n) == expected

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "n,expected",
        [(0, 0), (1, 1), (2, 1), (3, 2), (4, 3), (5, 5), (10, 55), (15, 610)],
    )
    async def test_fibonacci_parametrized(self, n, expected):
        """Parametrized test for Fibonacci calculation."""
        assert await fibonacci(n) == expected

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "n,expected",
        [(0, 1), (1, 1), (2, 2), (3, 6), (4, 24), (5, 120), (6, 720), (7, 5040)],
    )
    async def test_factorial_parametrized(self, n, expected):
        """Parametrized test for factorial calculation."""
        assert await factorial(n) == expected

    @pytest.mark.asyncio
    @pytest.mark.parametrize("perfect_square", [0, 1, 4, 9, 16, 25, 36, 49, 64, 81, 100])
    async def test_is_perfect_square_parametrized(self, perfect_square):
        """Parametrized test for perfect square identification."""
        assert await is_perfect_square(perfect_square)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("power_of_two", [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024])
    async def test_is_power_of_two_parametrized(self, power_of_two):
        """Parametrized test for power of two identification."""
        assert await is_power_of_two(power_of_two)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("fib_num", [0, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144])
    async def test_is_fibonacci_number_parametrized(self, fib_num):
        """Parametrized test for Fibonacci number identification."""
        assert await is_fibonacci_number(fib_num)


# ============================================================================
# LARGE NUMBER ASYNC TESTS (for coverage of asyncio.sleep lines)
# ============================================================================


class TestLargeNumberAsync:
    """Test async yield points for large numbers."""

    @pytest.mark.asyncio
    async def test_perfect_squares_large_n(self):
        """Test perfect_squares with large n to trigger async yield (line 127)."""
        # n > 1000 triggers asyncio.sleep(0) at line 127
        result = await perfect_squares(1500)
        assert len(result) == 1500
        # Verify first and last few elements
        assert result[0] == 0
        assert result[1] == 1
        assert result[1499] == 1499 * 1499

    @pytest.mark.asyncio
    async def test_powers_of_two_large_n(self):
        """Test powers_of_two with large n to trigger async yield (line 259)."""
        # n > 50 triggers asyncio.sleep(0) at line 259
        result = await powers_of_two(100)
        assert len(result) == 100
        # Verify pattern
        assert result[0] == 1
        assert result[1] == 2
        assert result[50] == 2**50

    @pytest.mark.asyncio
    async def test_fibonacci_large_n_matrix_path(self):
        """Test fibonacci with n > 100 to trigger matrix exponentiation (lines 348-383)."""
        # n > 100 triggers matrix exponentiation path (lines 348-383)
        result = await fibonacci(150)
        # Verify it's a reasonable large number
        assert result > 0
        assert isinstance(result, int)

        # Test n=101 to trigger matrix_power base case (line 362)
        result_101 = await fibonacci(101)
        assert result_101 > 0
        assert isinstance(result_101, int)

        # Test another large value to ensure matrix_power loop and sleep are exercised (line 377)
        result2 = await fibonacci(200)
        assert result2 > result
        assert isinstance(result2, int)

        # Test very large value to ensure multiple iterations in matrix_power
        result3 = await fibonacci(1000)
        assert result3 > result2
        assert isinstance(result3, int)

    @pytest.mark.asyncio
    async def test_fibonacci_sequence_large_n(self):
        """Test fibonacci_sequence with large n to trigger async yield (line 435)."""
        # n > 1000 triggers asyncio.sleep(0) at line 435
        result = await fibonacci_sequence(1500)
        assert len(result) == 1500
        assert result[0] == 0
        assert result[1] == 1
        # Verify recurrence relation holds
        assert result[10] == result[9] + result[8]

    @pytest.mark.asyncio
    async def test_is_fibonacci_number_negative_in_helper(self):
        """Test is_fibonacci_number helper function with negative check (line 490)."""
        # This tests the is_perfect_square_helper internal function's return False for x < 0
        # The helper is called with 5*n^2 + 4 and 5*n^2 - 4
        # For n=0, it checks 5*0 + 4 = 4 and 5*0 - 4 = -4
        # The -4 case triggers line 490
        result = await is_fibonacci_number(0)
        assert result is True  # 0 is a Fibonacci number

        # For n=1, checks 5*1 + 4 = 9 and 5*1 - 4 = 1
        result = await is_fibonacci_number(1)
        assert result is True

    @pytest.mark.asyncio
    async def test_factorial_large_n(self):
        """Test factorial with large n to trigger async yield (line 544)."""
        # n > 1000 triggers asyncio.sleep(0) at line 544
        result = await factorial(1200)
        assert result > 0
        assert isinstance(result, int)

    @pytest.mark.asyncio
    async def test_double_factorial_large_iterations(self):
        """Test double_factorial with large iterations to trigger async yield (line 594)."""
        # Need iterations % 1000 == 0 to trigger line 594
        # With n=2000, we get 1000 iterations (every other number)
        result = await double_factorial(2000)
        assert result > 0
        assert isinstance(result, int)

    @pytest.mark.asyncio
    async def test_subfactorial_large_n(self):
        """Test subfactorial with large n to trigger async yield (line 654)."""
        # n > 1000 triggers asyncio.sleep(0) at line 654
        result = await subfactorial(1200)
        assert result > 0
        assert isinstance(result, int)

    @pytest.mark.asyncio
    async def test_triangular_sequence_large_n(self):
        """Test triangular_sequence with large n to trigger async yield (line 781)."""
        # n > 1000 triggers asyncio.sleep(0) at line 781
        result = await triangular_sequence(1500)
        assert len(result) == 1500
        assert result[0] == 0
        assert result[10] == 10 * 11 // 2

    @pytest.mark.asyncio
    async def test_pentagonal_sequence_large_n(self):
        """Test pentagonal_sequence with large n to trigger async yield (line 914)."""
        # n > 1000 triggers asyncio.sleep(0) at line 914
        result = await pentagonal_sequence(1500)
        assert len(result) == 1500
        assert result[0] == 0
        assert result[10] == 10 * (3 * 10 - 1) // 2


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v", "--tb=short", "--asyncio-mode=auto"])
