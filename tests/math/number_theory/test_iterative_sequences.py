#!/usr/bin/env python3
# tests/math/number_theory/test_iterative_sequences.py
"""
Comprehensive pytest test suite for iterative_sequences.py module.

Tests cover:
- Collatz conjecture: sequence generation, stopping times, max values
- Kaprekar sequences: process generation, constants for different digit counts
- Happy numbers: identification, sequence generation, sad number cycles
- Narcissistic numbers: identification, sequence generation, digit power properties
- Look-and-say sequence: generation, pattern analysis, length properties
- Recamán sequence: generation, uniqueness properties, growth patterns
- Keith numbers: identification, sequence generation, digit-based properties
- Digital sequences: sum and product iterations, convergence properties
- Mathematical properties and relationships
- Edge cases, error conditions, and performance testing
- Async behavior verification
"""

import pytest
import asyncio
import time

# Import the functions to test
from chuk_mcp_math.number_theory.iterative_sequences import (
    # Collatz sequence
    collatz_sequence,
    collatz_stopping_time,
    collatz_max_value,
    # Kaprekar sequences
    kaprekar_sequence,
    kaprekar_constant,
    # Happy numbers
    is_happy_number,
    happy_numbers,
    # Narcissistic numbers
    is_narcissistic_number,
    narcissistic_numbers,
    # Look-and-say sequence
    look_and_say_sequence,
    # Recamán sequence
    recaman_sequence,
    # Keith numbers
    is_keith_number,
    keith_numbers,
    # Digital sequences
    digital_sum_sequence,
    digital_product_sequence,
)

# ============================================================================
# COLLATZ CONJECTURE TESTS
# ============================================================================


class TestCollatzConjecture:
    """Test cases for Collatz conjecture functions."""

    @pytest.mark.asyncio
    async def test_collatz_sequence_known_cases(self):
        """Test Collatz sequence with known cases."""
        test_cases = [
            (1, [1]),
            (2, [2, 1]),
            (3, [3, 10, 5, 16, 8, 4, 2, 1]),
            (4, [4, 2, 1]),
            (5, [5, 16, 8, 4, 2, 1]),
            (6, [6, 3, 10, 5, 16, 8, 4, 2, 1]),
            (7, [7, 22, 11, 34, 17, 52, 26, 13, 40, 20, 10, 5, 16, 8, 4, 2, 1]),
            (8, [8, 4, 2, 1]),
        ]

        for n, expected in test_cases:
            result = await collatz_sequence(n)
            assert result == expected, (
                f"collatz_sequence({n}) should be {expected}, got {result}"
            )

    @pytest.mark.asyncio
    async def test_collatz_sequence_properties(self):
        """Test mathematical properties of Collatz sequences."""
        test_numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 16, 31, 32]

        for n in test_numbers:
            sequence = await collatz_sequence(n)

            # Sequence should start with n
            assert sequence[0] == n, f"Collatz sequence should start with {n}"

            # Sequence should end with 1
            assert sequence[-1] == 1, "Collatz sequence should end with 1"

            # Verify Collatz rules
            for i in range(len(sequence) - 1):
                current = sequence[i]
                next_val = sequence[i + 1]

                if current % 2 == 0:
                    assert next_val == current // 2, (
                        f"Even rule failed: {current} -> {next_val}"
                    )
                else:
                    assert next_val == 3 * current + 1, (
                        f"Odd rule failed: {current} -> {next_val}"
                    )

    @pytest.mark.asyncio
    async def test_collatz_stopping_time_known_values(self):
        """Test Collatz stopping times with known values."""
        test_cases = [
            (1, 0),  # Already at 1
            (2, 1),  # 2 -> 1
            (3, 7),  # 3 -> 10 -> 5 -> 16 -> 8 -> 4 -> 2 -> 1
            (4, 2),  # 4 -> 2 -> 1
            (5, 5),  # 5 -> 16 -> 8 -> 4 -> 2 -> 1
            (6, 8),  # 6 -> 3 -> 10 -> 5 -> 16 -> 8 -> 4 -> 2 -> 1
            (7, 16),  # Known stopping time for 7
            (8, 3),  # 8 -> 4 -> 2 -> 1
        ]

        for n, expected in test_cases:
            result = await collatz_stopping_time(n)
            assert result == expected, (
                f"collatz_stopping_time({n}) should be {expected}, got {result}"
            )

    @pytest.mark.asyncio
    async def test_collatz_stopping_time_consistency(self):
        """Test consistency between stopping time and sequence length."""
        test_numbers = [1, 3, 5, 7, 9, 11, 13, 15, 27]

        for n in test_numbers:
            sequence = await collatz_sequence(n)
            stopping_time = await collatz_stopping_time(n)

            # Stopping time should equal sequence length - 1
            assert stopping_time == len(sequence) - 1, (
                f"Stopping time inconsistency for {n}"
            )

    @pytest.mark.asyncio
    async def test_collatz_max_value_known_cases(self):
        """Test Collatz maximum values with known cases."""
        test_cases = [
            (1, 1),  # Max in [1] is 1
            (2, 2),  # Max in [2, 1] is 2
            (3, 16),  # Max in sequence is 16
            (7, 52),  # Max in sequence is 52
            (15, 160),  # Known max for 15
            (27, 9232),  # Known large max for 27
        ]

        for n, expected in test_cases:
            result = await collatz_max_value(n)
            assert result == expected, (
                f"collatz_max_value({n}) should be {expected}, got {result}"
            )

    @pytest.mark.asyncio
    async def test_collatz_max_value_consistency(self):
        """Test consistency between max value and sequence."""
        test_numbers = [1, 3, 5, 7, 9, 11, 13, 15]

        for n in test_numbers:
            sequence = await collatz_sequence(n)
            max_value = await collatz_max_value(n)

            # Max value should equal maximum in sequence
            assert max_value == max(sequence), f"Max value inconsistency for {n}"

            # Max value should be >= starting number
            assert max_value >= n, f"Max value should be >= starting number for {n}"

    @pytest.mark.asyncio
    async def test_collatz_edge_cases(self):
        """Test Collatz functions with edge cases."""
        # Invalid inputs
        with pytest.raises(ValueError, match="n must be positive"):
            await collatz_sequence(0)

        with pytest.raises(ValueError, match="n must be positive"):
            await collatz_sequence(-1)

        with pytest.raises(ValueError, match="n must be positive"):
            await collatz_stopping_time(0)

        with pytest.raises(ValueError, match="n must be positive"):
            await collatz_max_value(-5)


# ============================================================================
# KAPREKAR SEQUENCES TESTS
# ============================================================================


class TestKaprekarSequences:
    """Test cases for Kaprekar sequence functions."""

    @pytest.mark.asyncio
    async def test_kaprekar_sequence_known_cases(self):
        """Test Kaprekar sequence with known cases."""
        test_cases = [
            (1234, 4, [1234, 3087, 8352, 6174]),  # Reaches 6174
            (6174, 4, [6174]),  # Already at Kaprekar constant
            (495, 3, [495]),  # 3-digit Kaprekar constant
            (1111, 4, [1111, 0]),  # Repdigit leads to 0
            (2222, 4, [2222, 0]),  # Another repdigit
        ]

        for n, digits, expected in test_cases:
            result = await kaprekar_sequence(n, digits)
            assert result == expected, (
                f"kaprekar_sequence({n}, {digits}) should be {expected}, got {result}"
            )

    @pytest.mark.asyncio
    async def test_kaprekar_sequence_properties(self):
        """Test mathematical properties of Kaprekar sequences."""
        # Test 4-digit numbers converging to 6174
        test_numbers = [1234, 5432, 9876, 1357, 2468]

        for n in test_numbers:
            sequence = await kaprekar_sequence(n, 4)

            # Should start with the input number
            assert sequence[0] == n, f"Kaprekar sequence should start with {n}"

            # For 4-digit non-repdigits, should eventually reach 6174 or 0
            if not self._is_repdigit(n, 4):
                assert 6174 in sequence or 0 in sequence, (
                    "4-digit sequence should reach 6174 or 0"
                )

    def _is_repdigit(self, n: int, digits: int) -> bool:
        """Helper to check if number is a repdigit."""
        digits_str = str(n).zfill(digits)
        return len(set(digits_str)) <= 1

    @pytest.mark.asyncio
    async def test_kaprekar_sequence_3_digit(self):
        """Test 3-digit Kaprekar sequences."""
        test_numbers = [123, 456, 789, 321]

        for n in test_numbers:
            sequence = await kaprekar_sequence(n, 3)

            # Should reach 495 (3-digit Kaprekar constant) or 0
            if not self._is_repdigit(n, 3):
                assert 495 in sequence or 0 in sequence, (
                    "3-digit sequence should reach 495 or 0"
                )

    @pytest.mark.asyncio
    async def test_kaprekar_sequence_repdigits(self):
        """Test Kaprekar sequences with repdigits."""
        repdigit_cases = [
            (1111, 4, 0),  # All 1s
            (2222, 4, 0),  # All 2s
            (9999, 4, 0),  # All 9s
            (333, 3, 0),  # 3-digit repdigit
            (777, 3, 0),  # Another 3-digit repdigit
        ]

        for n, digits, expected_end in repdigit_cases:
            sequence = await kaprekar_sequence(n, digits)
            assert expected_end in sequence, (
                f"Repdigit {n} should lead to {expected_end}"
            )

    @pytest.mark.asyncio
    async def test_kaprekar_constant_known_values(self):
        """Test known Kaprekar constants."""
        known_constants = {
            3: 495,
            4: 6174,
            6: 549945,
            7: 1194649,
        }

        for digits, expected in known_constants.items():
            result = await kaprekar_constant(digits)
            assert result == expected, (
                f"Kaprekar constant for {digits} digits should be {expected}"
            )

    @pytest.mark.asyncio
    async def test_kaprekar_constant_unknown_values(self):
        """Test Kaprekar constants for digit counts without known constants."""
        unknown_digits = [2, 5, 8, 9, 10]

        for digits in unknown_digits:
            result = await kaprekar_constant(digits)
            assert result is None, f"No known Kaprekar constant for {digits} digits"

    @pytest.mark.asyncio
    async def test_kaprekar_sequence_edge_cases(self):
        """Test Kaprekar sequence edge cases."""
        # Invalid digit count
        with pytest.raises(ValueError, match="digits must be greater than 1"):
            await kaprekar_sequence(123, 1)

        with pytest.raises(ValueError, match="digits must be greater than 1"):
            await kaprekar_sequence(123, 0)


# ============================================================================
# HAPPY NUMBERS TESTS
# ============================================================================


class TestHappyNumbers:
    """Test cases for happy number functions."""

    @pytest.mark.asyncio
    async def test_is_happy_number_known_happy(self):
        """Test with known happy numbers."""
        known_happy = [
            1,
            7,
            10,
            13,
            19,
            23,
            28,
            31,
            32,
            44,
            49,
            68,
            70,
            79,
            82,
            86,
            91,
            94,
            97,
            100,
        ]

        for happy_num in known_happy:
            result = await is_happy_number(happy_num)
            assert result, f"{happy_num} should be happy"

    @pytest.mark.asyncio
    async def test_is_happy_number_known_sad(self):
        """Test with known sad (unhappy) numbers."""
        known_sad = [
            2,
            3,
            4,
            5,
            6,
            8,
            9,
            11,
            12,
            14,
            15,
            16,
            17,
            18,
            20,
            21,
            22,
            24,
            25,
            26,
            27,
            29,
            30,
        ]

        for sad_num in known_sad:
            result = await is_happy_number(sad_num)
            assert not result, f"{sad_num} should be sad (not happy)"

    @pytest.mark.asyncio
    async def test_is_happy_number_manual_verification(self):
        """Test happy number calculation with manual verification."""
        # 7: 7 -> 49 -> 97 -> 130 -> 10 -> 1 (happy)
        assert await is_happy_number(7)

        # 19: 19 -> 82 -> 68 -> 100 -> 1 (happy)
        assert await is_happy_number(19)

        # 4: 4 -> 16 -> 37 -> 58 -> 89 -> 145 -> 42 -> 20 -> 4 (cycles, sad)
        assert not await is_happy_number(4)

    @pytest.mark.asyncio
    async def test_is_happy_number_edge_cases(self):
        """Test happy number edge cases."""
        # 1 is happy by definition
        assert await is_happy_number(1)

        # 0 and negative numbers are not happy
        assert not await is_happy_number(0)
        assert not await is_happy_number(-1)
        assert not await is_happy_number(-7)

    @pytest.mark.asyncio
    async def test_happy_numbers_generation(self):
        """Test generation of happy numbers up to a limit."""
        # Known happy numbers up to 50
        expected_50 = [1, 7, 10, 13, 19, 23, 28, 31, 32, 44, 49]
        result_50 = await happy_numbers(50)
        assert result_50 == expected_50, (
            f"Happy numbers up to 50 should be {expected_50}"
        )

        # Known happy numbers up to 20
        expected_20 = [1, 7, 10, 13, 19]
        result_20 = await happy_numbers(20)
        assert result_20 == expected_20, (
            f"Happy numbers up to 20 should be {expected_20}"
        )

    @pytest.mark.asyncio
    async def test_happy_numbers_edge_cases(self):
        """Test happy numbers generation edge cases."""
        # Empty range
        assert await happy_numbers(0) == []
        assert await happy_numbers(-5) == []

        # Single number
        assert await happy_numbers(1) == [1]

    @pytest.mark.asyncio
    async def test_happy_numbers_properties(self):
        """Test mathematical properties of happy numbers."""
        happy_list = await happy_numbers(100)

        # All numbers in the list should be happy
        for num in happy_list:
            assert await is_happy_number(num), (
                f"{num} in happy list should be happy"
            )

        # Should be in ascending order
        assert happy_list == sorted(happy_list), (
            "Happy numbers should be in ascending order"
        )

        # Should contain 1
        assert 1 in happy_list, "Happy numbers should contain 1"


# ============================================================================
# NARCISSISTIC NUMBERS TESTS
# ============================================================================


class TestNarcissisticNumbers:
    """Test cases for narcissistic number functions."""

    @pytest.mark.asyncio
    async def test_is_narcissistic_number_known_narcissistic(self):
        """Test with known narcissistic numbers."""
        known_narcissistic = [
            # 1-digit (trivial)
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            # 3-digit
            153,
            371,
            407,
            # 4-digit
            1634,
            8208,
            9474,
            # 5-digit
            54748,
            92727,
            93084,
        ]

        for narcissistic_num in known_narcissistic:
            result = await is_narcissistic_number(narcissistic_num)
            assert result, f"{narcissistic_num} should be narcissistic"

    @pytest.mark.asyncio
    async def test_is_narcissistic_number_manual_verification(self):
        """Test narcissistic numbers with manual verification."""
        # 153 = 1³ + 5³ + 3³ = 1 + 125 + 27 = 153
        assert await is_narcissistic_number(153)

        # 371 = 3³ + 7³ + 1³ = 27 + 343 + 1 = 371
        assert await is_narcissistic_number(371)

        # 1634 = 1⁴ + 6⁴ + 3⁴ + 4⁴ = 1 + 1296 + 81 + 256 = 1634
        assert await is_narcissistic_number(1634)

        # 123 ≠ 1³ + 2³ + 3³ = 1 + 8 + 27 = 36
        assert not await is_narcissistic_number(123)

    @pytest.mark.asyncio
    async def test_is_narcissistic_number_non_narcissistic(self):
        """Test with numbers that are not narcissistic."""
        non_narcissistic = [10, 11, 12, 99, 100, 123, 456, 789, 1000, 1235, 5678]

        for non_narcissistic_num in non_narcissistic:
            result = await is_narcissistic_number(non_narcissistic_num)
            assert not result, f"{non_narcissistic_num} should not be narcissistic"

    @pytest.mark.asyncio
    async def test_is_narcissistic_number_edge_cases(self):
        """Test narcissistic number edge cases."""
        # 0 and negative numbers are not narcissistic
        assert not await is_narcissistic_number(0)
        assert not await is_narcissistic_number(-1)
        assert not await is_narcissistic_number(-153)

    @pytest.mark.asyncio
    async def test_narcissistic_numbers_generation(self):
        """Test generation of narcissistic numbers."""
        # Known narcissistic numbers up to 1000 (corrected to include 370)
        expected_1000 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 153, 370, 371, 407]
        result_1000 = await narcissistic_numbers(1000)
        assert result_1000 == expected_1000, (
            f"Narcissistic numbers up to 1000 should be {expected_1000}"
        )

        # Up to 200
        expected_200 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 153]
        result_200 = await narcissistic_numbers(200)
        assert result_200 == expected_200, (
            f"Narcissistic numbers up to 200 should be {expected_200}"
        )

    @pytest.mark.asyncio
    async def test_narcissistic_numbers_properties(self):
        """Test mathematical properties of narcissistic numbers."""
        narcissistic_list = await narcissistic_numbers(500)

        # All numbers in the list should be narcissistic
        for num in narcissistic_list:
            assert await is_narcissistic_number(num), (
                f"{num} should be narcissistic"
            )

        # Should be in ascending order
        assert narcissistic_list == sorted(narcissistic_list), (
            "Narcissistic numbers should be in ascending order"
        )

        # Should contain all single digits
        for i in range(1, 10):
            assert i in narcissistic_list, f"Single digit {i} should be narcissistic"

    @pytest.mark.asyncio
    async def test_narcissistic_numbers_edge_cases(self):
        """Test narcissistic numbers generation edge cases."""
        assert await narcissistic_numbers(0) == []
        assert await narcissistic_numbers(-5) == []
        assert await narcissistic_numbers(5) == [1, 2, 3, 4, 5]


# ============================================================================
# LOOK-AND-SAY SEQUENCE TESTS
# ============================================================================


class TestLookAndSaySequence:
    """Test cases for look-and-say sequence functions."""

    @pytest.mark.asyncio
    async def test_look_and_say_sequence_from_1(self):
        """Test look-and-say sequence starting from 1."""
        expected = ["1", "11", "21", "1211", "111221", "312211"]
        result = await look_and_say_sequence("1", 6)
        assert result == expected, (
            f"Look-and-say from '1' should be {expected}, got {result}"
        )

    @pytest.mark.asyncio
    async def test_look_and_say_sequence_from_3(self):
        """Test look-and-say sequence starting from 3."""
        expected = ["3", "13", "1113", "3113", "132113"]
        result = await look_and_say_sequence("3", 5)
        assert result == expected, (
            f"Look-and-say from '3' should be {expected}, got {result}"
        )

    @pytest.mark.asyncio
    async def test_look_and_say_sequence_from_11(self):
        """Test look-and-say sequence starting from 11."""
        expected = ["11", "21", "1211", "111221"]
        result = await look_and_say_sequence("11", 4)
        assert result == expected, (
            f"Look-and-say from '11' should be {expected}, got {result}"
        )

    @pytest.mark.asyncio
    async def test_look_and_say_sequence_properties(self):
        """Test mathematical properties of look-and-say sequence."""
        # Test that sequence starts with the given input
        for start in ["1", "2", "3", "11", "22"]:
            sequence = await look_and_say_sequence(start, 3)
            assert sequence[0] == start, f"Sequence should start with {start}"
            assert len(sequence) == 3, "Should generate 3 terms"

    @pytest.mark.asyncio
    async def test_look_and_say_manual_verification(self):
        """Test look-and-say with manual verification."""
        # "1" -> "11" (one 1)
        # "11" -> "21" (two 1s)
        # "21" -> "1211" (one 2, one 1)
        # "1211" -> "111221" (one 1, one 2, two 1s)

        result = await look_and_say_sequence("1", 4)
        assert result[1] == "11", "'1' should become '11'"
        assert result[2] == "21", "'11' should become '21'"
        assert result[3] == "1211", "'21' should become '1211'"

    @pytest.mark.asyncio
    async def test_look_and_say_edge_cases(self):
        """Test look-and-say edge cases."""
        # Zero terms
        assert await look_and_say_sequence("1", 0) == []

        # Negative terms
        assert await look_and_say_sequence("1", -1) == []

        # Single term
        result = await look_and_say_sequence("123", 1)
        assert result == ["123"], "Single term should be the input"

    @pytest.mark.asyncio
    async def test_look_and_say_length_growth(self):
        """Test that look-and-say sequences generally grow in length."""
        sequence = await look_and_say_sequence("1", 8)

        # Generally, each term should be longer than or equal to the previous
        # (with some possible exceptions for special patterns)
        for i in range(1, len(sequence)):
            # Length should generally increase or stay the same
            assert len(sequence[i]) >= len(sequence[i - 1]) - 2, (
                "Length shouldn't decrease dramatically"
            )


# ============================================================================
# RECAMÁN SEQUENCE TESTS
# ============================================================================


class TestRecamanSequence:
    """Test cases for Recamán sequence functions."""

    @pytest.mark.asyncio
    async def test_recaman_sequence_known_values(self):
        """Test Recamán sequence with known values."""
        # First 15 terms of Recamán sequence
        expected_15 = [0, 1, 3, 6, 2, 7, 13, 20, 12, 21, 11, 22, 10, 23, 9]
        result_15 = await recaman_sequence(15)
        assert result_15 == expected_15, (
            f"First 15 Recamán numbers should be {expected_15}"
        )

        # First 10 terms
        expected_10 = [0, 1, 3, 6, 2, 7, 13, 20, 12, 21]
        result_10 = await recaman_sequence(10)
        assert result_10 == expected_10, (
            f"First 10 Recamán numbers should be {expected_10}"
        )

    @pytest.mark.asyncio
    async def test_recaman_sequence_properties(self):
        """Test mathematical properties of Recamán sequence."""
        sequence = await recaman_sequence(20)

        # Should start with 0
        assert sequence[0] == 0, "Recamán sequence should start with 0"

        # All values should be non-negative
        assert all(x >= 0 for x in sequence), (
            "All Recamán numbers should be non-negative"
        )

        # No duplicates in the sequence (by definition)
        assert len(sequence) == len(set(sequence)), (
            "Recamán sequence should have no duplicates"
        )

    @pytest.mark.asyncio
    async def test_recaman_sequence_recurrence_verification(self):
        """Test Recamán sequence recurrence relation."""
        sequence = await recaman_sequence(10)

        for i in range(1, len(sequence)):
            prev = sequence[i - 1]
            current = sequence[i]

            # Check if current follows Recamán rules
            candidate_subtract = prev - i
            if candidate_subtract > 0 and candidate_subtract not in sequence[:i]:
                # Should subtract
                assert current == candidate_subtract, (
                    f"Should subtract at step {i}: {prev} - {i} = {candidate_subtract}"
                )
            else:
                # Should add
                assert current == prev + i, (
                    f"Should add at step {i}: {prev} + {i} = {prev + i}"
                )

    @pytest.mark.asyncio
    async def test_recaman_sequence_edge_cases(self):
        """Test Recamán sequence edge cases."""
        # Zero terms
        assert await recaman_sequence(0) == []

        # Negative terms
        assert await recaman_sequence(-1) == []

        # Single term
        assert await recaman_sequence(1) == [0]


# ============================================================================
# KEITH NUMBERS TESTS
# ============================================================================


class TestKeithNumbers:
    """Test cases for Keith number functions."""

    @pytest.mark.asyncio
    async def test_is_keith_number_known_keith(self):
        """Test with known Keith numbers."""
        known_keith = [
            14,
            19,
            28,
            47,
            61,
            75,
            197,
            742,
            1104,
            1537,
            2208,
            2580,
            3684,
            4788,
            7385,
            7647,
            7909,
        ]

        for keith_num in known_keith[:10]:  # Test first 10 to avoid long computation
            result = await is_keith_number(keith_num)
            assert result, f"{keith_num} should be a Keith number"

    @pytest.mark.asyncio
    async def test_is_keith_number_manual_verification(self):
        """Test Keith numbers with manual verification."""
        # 14: digits [1, 4], sequence: 1, 4, 5, 9, 14 ✓
        assert await is_keith_number(14)

        # 197: digits [1, 9, 7], sequence: 1, 9, 7, 17, 33, 57, 107, 197 ✓
        assert await is_keith_number(197)

        # 15: not Keith
        assert not await is_keith_number(15)

    @pytest.mark.asyncio
    async def test_is_keith_number_non_keith(self):
        """Test with numbers that are not Keith numbers."""
        non_keith = [
            10,
            11,
            12,
            13,
            15,
            16,
            17,
            18,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            29,
            30,
        ]

        for non_keith_num in non_keith:
            result = await is_keith_number(non_keith_num)
            assert not result, f"{non_keith_num} should not be a Keith number"

    @pytest.mark.asyncio
    async def test_is_keith_number_edge_cases(self):
        """Test Keith number edge cases."""
        # Single digits are not Keith numbers
        for i in range(1, 10):
            assert not await is_keith_number(i), (
                f"Single digit {i} should not be Keith"
            )

        # Zero and negative numbers
        assert not await is_keith_number(0)
        assert not await is_keith_number(-14)

    @pytest.mark.asyncio
    async def test_keith_numbers_generation(self):
        """Test generation of Keith numbers."""
        # Known Keith numbers up to 100
        expected_100 = [14, 19, 28, 47, 61, 75]
        result_100 = await keith_numbers(100)
        assert result_100 == expected_100, (
            f"Keith numbers up to 100 should be {expected_100}"
        )

        # Keith numbers up to 50
        expected_50 = [14, 19, 28, 47]
        result_50 = await keith_numbers(50)
        assert result_50 == expected_50, (
            f"Keith numbers up to 50 should be {expected_50}"
        )

    @pytest.mark.asyncio
    async def test_keith_numbers_properties(self):
        """Test mathematical properties of Keith numbers."""
        keith_list = await keith_numbers(200)

        # All numbers in the list should be Keith numbers
        for num in keith_list:
            assert await is_keith_number(num), f"{num} should be a Keith number"

        # Should be in ascending order
        assert keith_list == sorted(keith_list), (
            "Keith numbers should be in ascending order"
        )

        # All should be >= 10 (no single digits)
        assert all(num >= 10 for num in keith_list), "All Keith numbers should be >= 10"

    @pytest.mark.asyncio
    async def test_keith_numbers_edge_cases(self):
        """Test Keith numbers generation edge cases."""
        assert await keith_numbers(0) == []
        assert await keith_numbers(-5) == []
        assert await keith_numbers(9) == []  # No Keith numbers < 10


# ============================================================================
# DIGITAL SEQUENCES TESTS
# ============================================================================


class TestDigitalSequences:
    """Test cases for digital sequence functions."""

    @pytest.mark.asyncio
    async def test_digital_sum_sequence_known_cases(self):
        """Test digital sum sequence with known cases."""
        test_cases = [
            (7, [7]),  # Single digit unchanged
            (123, [123, 6]),  # 123 -> 1+2+3=6
            (9875, [9875, 29, 11, 2]),  # 9875 -> 29 -> 11 -> 2
            (999, [999, 27, 9]),  # 999 -> 27 -> 9
            (1999, [1999, 28, 10, 1]),  # 1999 -> 28 -> 10 -> 1
        ]

        for n, expected in test_cases:
            result = await digital_sum_sequence(n)
            assert result == expected, (
                f"digital_sum_sequence({n}) should be {expected}, got {result}"
            )

    @pytest.mark.asyncio
    async def test_digital_sum_sequence_properties(self):
        """Test properties of digital sum sequences."""
        test_numbers = [12, 99, 123, 456, 999, 1234]

        for n in test_numbers:
            sequence = await digital_sum_sequence(n)

            # Should start with the input
            assert sequence[0] == n, f"Digital sum sequence should start with {n}"

            # Should end with single digit
            assert sequence[-1] < 10, (
                "Digital sum sequence should end with single digit"
            )

            # Should be decreasing until single digit
            for i in range(len(sequence) - 1):
                assert sequence[i] >= sequence[i + 1], (
                    "Digital sum sequence should be non-increasing"
                )

    @pytest.mark.asyncio
    async def test_digital_product_sequence_known_cases(self):
        """Test digital product sequence with known cases."""
        test_cases = [
            (7, [7]),  # Single digit unchanged
            (39, [39, 27, 14, 4]),  # 39 -> 3×9=27 -> 2×7=14 -> 1×4=4
            (
                999,
                [999, 729, 126, 12, 2],
            ),  # 999 -> 9×9×9=729 -> 7×2×9=126 -> 1×2×6=12 -> 1×2=2
            (105, [105, 0]),  # Contains 0, product becomes 0
            (1000, [1000, 0]),  # Contains 0, product becomes 0
        ]

        for n, expected in test_cases:
            result = await digital_product_sequence(n)
            assert result == expected, (
                f"digital_product_sequence({n}) should be {expected}, got {result}"
            )

    @pytest.mark.asyncio
    async def test_digital_product_sequence_properties(self):
        """Test properties of digital product sequences."""
        test_numbers = [12, 23, 39, 99, 123, 234]

        for n in test_numbers:
            sequence = await digital_product_sequence(n)

            # Should start with the input
            assert sequence[0] == n, f"Digital product sequence should start with {n}"

            # Should end with single digit or 0
            assert sequence[-1] < 10, (
                "Digital product sequence should end with single digit or 0"
            )

            # If no zeros in digits, should generally decrease
            if "0" not in str(n):
                for i in range(len(sequence) - 1):
                    assert sequence[i] >= sequence[i + 1], (
                        "Digital product sequence should generally decrease"
                    )

    @pytest.mark.asyncio
    async def test_digital_product_sequence_zero_cases(self):
        """Test digital product sequences with zeros."""
        zero_cases = [10, 20, 100, 105, 1000, 2030]

        for n in zero_cases:
            sequence = await digital_product_sequence(n)

            # Should contain 0 and end with 0
            assert 0 in sequence, "Digital product sequence with zeros should contain 0"
            assert sequence[-1] == 0, (
                "Digital product sequence with zeros should end with 0"
            )

    @pytest.mark.asyncio
    async def test_digital_sequences_edge_cases(self):
        """Test digital sequence edge cases."""
        # Invalid inputs
        with pytest.raises(ValueError, match="n must be positive"):
            await digital_sum_sequence(0)

        with pytest.raises(ValueError, match="n must be positive"):
            await digital_sum_sequence(-123)

        with pytest.raises(ValueError, match="n must be positive"):
            await digital_product_sequence(0)

        with pytest.raises(ValueError, match="n must be positive"):
            await digital_product_sequence(-39)


# ============================================================================
# INTEGRATION AND MATHEMATICAL PROPERTIES TESTS
# ============================================================================


class TestIntegrationAndProperties:
    """Integration tests and mathematical property verification."""

    @pytest.mark.asyncio
    async def test_sequence_convergence_properties(self):
        """Test convergence properties of iterative sequences."""
        # Collatz sequences always reach 1 (conjecture)
        for n in range(1, 20):
            sequence = await collatz_sequence(n)
            assert sequence[-1] == 1, f"Collatz({n}) should reach 1"

        # Digital sum sequences reach single digits
        for n in [123, 456, 999, 1234]:
            sequence = await digital_sum_sequence(n)
            assert sequence[-1] < 10, "Digital sum sequence should reach single digit"

        # Happy number determination is consistent
        for n in [1, 7, 19, 23]:
            assert await is_happy_number(n), f"{n} should be happy"

    @pytest.mark.asyncio
    async def test_sequence_length_relationships(self):
        """Test relationships between sequence lengths and properties."""
        # Collatz stopping time equals sequence length - 1
        for n in [3, 5, 7, 11]:
            sequence = await collatz_sequence(n)
            stopping_time = await collatz_stopping_time(n)
            assert len(sequence) - 1 == stopping_time, (
                f"Length relationship failed for Collatz({n})"
            )

        # Look-and-say sequences grow in complexity
        sequence = await look_and_say_sequence("1", 6)
        for i in range(1, len(sequence)):
            # Generally increases in length (Conway's constant ≈ 1.303)
            assert len(sequence[i]) >= len(sequence[i - 1]) * 0.8, (
                "Look-and-say should generally grow"
            )

    @pytest.mark.asyncio
    async def test_mathematical_invariants(self):
        """Test mathematical invariants across sequences."""
        # Kaprekar sequences reach fixed points
        kaprekar_seq = await kaprekar_sequence(1234, 4)
        if 6174 in kaprekar_seq:
            # If it reaches 6174, 6174 should be stable
            next_after_6174 = await kaprekar_sequence(6174, 4)
            assert next_after_6174 == [6174], "6174 should be a fixed point"

        # Recamán sequence property: early terms have unique values, but duplicates can occur later
        # The mathematical property is that we avoid duplicates when possible using the subtraction rule
        recaman_seq = await recaman_sequence(
            20
        )  # Test with smaller sequence to avoid later duplicates
        assert len(recaman_seq) == len(set(recaman_seq)), (
            "Recamán sequence should have no duplicates in early terms"
        )

        # Alternative test: verify the Recamán generation rule is followed correctly
        recaman_seq_30 = await recaman_sequence(30)
        seen_values = set()

        for i in range(len(recaman_seq_30)):
            if i == 0:
                assert recaman_seq_30[i] == 0, "Recamán should start with 0"
                seen_values.add(0)
            else:
                prev = recaman_seq_30[i - 1]
                current = recaman_seq_30[i]
                candidate_subtract = prev - i

                # Verify the rule: subtract if positive and not seen, otherwise add
                if candidate_subtract > 0 and candidate_subtract not in seen_values:
                    assert current == candidate_subtract, f"Should subtract at step {i}"
                else:
                    assert current == prev + i, f"Should add at step {i}"

                # Note: duplicates can occur in later terms when both subtract and add options are used
                seen_values.add(current)

        # Narcissistic numbers equal sum of powered digits
        for n in [153, 370, 371, 1634]:
            assert await is_narcissistic_number(n), (
                f"{n} should be narcissistic"
            )

    @pytest.mark.asyncio
    async def test_cross_sequence_relationships(self):
        """Test relationships between different types of sequences."""
        # Some numbers appear in multiple special sequences

        # Check if any Keith numbers are also happy
        keith_list = await keith_numbers(100)
        happy_list = await happy_numbers(100)
        keith_and_happy = set(keith_list) & set(happy_list)

        # Verify any intersections
        for num in keith_and_happy:
            assert await is_keith_number(num), f"{num} should be Keith"
            assert await is_happy_number(num), f"{num} should be happy"

        # Check narcissistic vs happy overlap
        narcissistic_list = await narcissistic_numbers(500)
        narcissistic_and_happy = set(narcissistic_list) & set(happy_list)

        for num in narcissistic_and_happy:
            assert await is_narcissistic_number(num), (
                f"{num} should be narcissistic"
            )
            assert await is_happy_number(num), f"{num} should be happy"


# ============================================================================
# PERFORMANCE AND ASYNC BEHAVIOR TESTS
# ============================================================================


class TestPerformance:
    """Performance and async behavior tests."""

    @pytest.mark.asyncio
    async def test_all_functions_are_async(self):
        """Test that all iterative sequence functions are properly async."""
        operations = [
            collatz_sequence(7),
            collatz_stopping_time(7),
            collatz_max_value(7),
            kaprekar_sequence(1234, 4),
            kaprekar_constant(4),
            is_happy_number(7),
            happy_numbers(20),
            is_narcissistic_number(153),
            narcissistic_numbers(200),
            look_and_say_sequence("1", 5),
            recaman_sequence(10),
            is_keith_number(14),
            keith_numbers(50),
            digital_sum_sequence(123),
            digital_product_sequence(39),
        ]

        # Ensure all are coroutines
        for op in operations:
            assert asyncio.iscoroutine(op)

        # Run all operations concurrently
        results = await asyncio.gather(*operations)

        # Verify results have expected types and values
        assert results[0] == [
            7,
            22,
            11,
            34,
            17,
            52,
            26,
            13,
            40,
            20,
            10,
            5,
            16,
            8,
            4,
            2,
            1,
        ]  # collatz_sequence(7)
        assert results[1] == 16  # collatz_stopping_time(7)
        assert results[2] == 52  # collatz_max_value(7)
        assert results[3] == [1234, 3087, 8352, 6174]  # kaprekar_sequence(1234, 4)
        assert results[4] == 6174  # kaprekar_constant(4)
        assert results[5]  # is_happy_number(7)
        assert isinstance(results[6], list)  # happy_numbers(20)
        assert results[7]  # is_narcissistic_number(153)
        assert isinstance(results[8], list)  # narcissistic_numbers(200)
        assert isinstance(results[9], list)  # look_and_say_sequence("1", 5)
        assert isinstance(results[10], list)  # recaman_sequence(10)
        assert results[11]  # is_keith_number(14)
        assert isinstance(results[12], list)  # keith_numbers(50)
        assert isinstance(results[13], list)  # digital_sum_sequence(123)
        assert isinstance(results[14], list)  # digital_product_sequence(39)

    @pytest.mark.asyncio
    async def test_concurrent_execution(self):
        """Test that iterative sequence operations can run concurrently."""
        start_time = time.time()

        # Run multiple operations concurrently
        tasks = []
        for i in range(1, 25):
            tasks.append(collatz_stopping_time(i))
            tasks.append(is_happy_number(i))
            if i >= 10:  # Keith numbers start from 10
                tasks.append(is_keith_number(i))
            tasks.append(is_narcissistic_number(i))
            tasks.append(digital_sum_sequence(i))

        results = await asyncio.gather(*tasks)
        duration = time.time() - start_time

        # Should complete quickly due to async nature
        assert duration < 5.0
        assert len(results) > 50  # Should have many results

        # Check some patterns in results
        stopping_times = [r for r in results if isinstance(r, int) and 0 <= r <= 200]
        boolean_results = [r for r in results if isinstance(r, bool)]
        sequence_results = [r for r in results if isinstance(r, list)]

        assert len(stopping_times) > 10, "Should have stopping time results"
        assert len(boolean_results) > 20, "Should have boolean results"
        assert len(sequence_results) > 10, "Should have sequence results"

    @pytest.mark.asyncio
    async def test_large_computation_handling(self):
        """Test handling of computationally intensive operations."""
        # Test larger Collatz computations
        collatz_27 = await collatz_sequence(27)
        assert len(collatz_27) > 100, "Collatz(27) should have many steps"
        assert collatz_27[-1] == 1, "Should reach 1"

        # Test longer sequences
        longer_sequences = await asyncio.gather(
            happy_numbers(200),
            narcissistic_numbers(1000),
            keith_numbers(200),
            look_and_say_sequence("1", 8),
            recaman_sequence(50),
        )

        happy_200, narcissistic_1000, keith_200, look_say_8, recaman_50 = (
            longer_sequences
        )

        # Verify reasonable results
        assert len(happy_200) > 20, "Should find many happy numbers"
        assert len(narcissistic_1000) >= 12, "Should find known narcissistic numbers"
        assert len(keith_200) >= 6, "Should find known Keith numbers"
        assert len(look_say_8) == 8, "Should generate 8 look-and-say terms"
        assert len(recaman_50) == 50, "Should generate 50 Recamán terms"

    @pytest.mark.asyncio
    async def test_memory_efficiency(self):
        """Test that functions don't consume excessive memory."""
        # Generate multiple sequences
        sequences = await asyncio.gather(
            *[collatz_sequence(i) for i in range(1, 20)],
            *[digital_sum_sequence(i * 111) for i in range(1, 15)],
            *[digital_product_sequence(i * 123) for i in range(1, 10)],
        )

        # Verify all sequences completed
        assert len(sequences) == 19 + 14 + 9  # Total expected sequences

        # All should be lists
        assert all(isinstance(seq, list) for seq in sequences), (
            "All results should be lists"
        )

        # All Collatz sequences should end with 1
        collatz_sequences = sequences[:19]
        assert all(seq[-1] == 1 for seq in collatz_sequences), (
            "All Collatz sequences should end with 1"
        )


# ============================================================================
# ERROR HANDLING TESTS
# ============================================================================


class TestErrorHandling:
    """Test error handling and edge cases."""

    @pytest.mark.asyncio
    async def test_collatz_function_errors(self):
        """Test error handling in Collatz functions."""
        collatz_functions = [collatz_sequence, collatz_stopping_time, collatz_max_value]

        for func in collatz_functions:
            with pytest.raises(ValueError, match="n must be positive"):
                await func(0)

            with pytest.raises(ValueError, match="n must be positive"):
                await func(-1)

    @pytest.mark.asyncio
    async def test_kaprekar_function_errors(self):
        """Test error handling in Kaprekar functions."""
        with pytest.raises(ValueError, match="digits must be greater than 1"):
            await kaprekar_sequence(123, 1)

        with pytest.raises(ValueError, match="digits must be greater than 1"):
            await kaprekar_sequence(123, 0)

    @pytest.mark.asyncio
    async def test_digital_sequence_errors(self):
        """Test error handling in digital sequence functions."""
        digital_functions = [digital_sum_sequence, digital_product_sequence]

        for func in digital_functions:
            with pytest.raises(ValueError, match="n must be positive"):
                await func(0)

            with pytest.raises(ValueError, match="n must be positive"):
                await func(-123)

    @pytest.mark.asyncio
    async def test_edge_case_handling(self):
        """Test edge case handling across all functions."""
        # Functions that should handle edge cases gracefully

        # Happy number functions with edge cases
        assert await is_happy_number(1), "1 should be happy"
        assert not await is_happy_number(0), "0 should not be happy"
        assert await happy_numbers(0) == [], "Empty range should return empty list"

        # Narcissistic number functions with edge cases
        assert await is_narcissistic_number(1), "1 should be narcissistic"
        assert not await is_narcissistic_number(0), "0 should not be narcissistic"
        assert await narcissistic_numbers(0) == [], (
            "Empty range should return empty list"
        )

        # Keith number functions with edge cases
        assert not await is_keith_number(9), "Single digits should not be Keith"
        assert await keith_numbers(9) == [], "No Keith numbers < 10"

        # Look-and-say with edge cases
        assert await look_and_say_sequence("1", 0) == [], (
            "Zero terms should return empty"
        )
        assert await look_and_say_sequence("1", 1) == ["1"], (
            "One term should return input"
        )

        # Recamán with edge cases
        assert await recaman_sequence(0) == [], "Zero terms should return empty"
        assert await recaman_sequence(1) == [0], "One term should return [0]"

    @pytest.mark.asyncio
    async def test_error_preserves_async_context(self):
        """Test that errors are properly raised in async context."""
        try:
            await collatz_sequence(-1)  # Should raise ValueError
            assert False, "Should have raised ValueError"
        except ValueError:
            # Should be able to continue with async operations
            result = await collatz_sequence(3)
            assert result == [3, 10, 5, 16, 8, 4, 2, 1]

        try:
            await digital_sum_sequence(0)  # Should raise ValueError
            assert False, "Should have raised ValueError"
        except ValueError:
            # Should be able to continue with async operations
            result = await digital_sum_sequence(123)
            assert result == [123, 6]


# ============================================================================
# PARAMETRIZED TESTS
# ============================================================================


class TestParametrized:
    """Parametrized tests for comprehensive coverage."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "n,expected_length",
        [(1, 1), (2, 2), (3, 8), (4, 3), (5, 6), (6, 9), (7, 17), (8, 4)],
    )
    async def test_collatz_sequence_length_parametrized(self, n, expected_length):
        """Parametrized test for Collatz sequence length."""
        result = await collatz_sequence(n)
        assert len(result) == expected_length

    @pytest.mark.asyncio
    @pytest.mark.parametrize("happy_num", [1, 7, 10, 13, 19, 23, 28, 31, 32, 44, 49])
    async def test_is_happy_number_parametrized(self, happy_num):
        """Parametrized test for happy number identification."""
        result = await is_happy_number(happy_num)
        assert result

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "narcissistic_num", [1, 2, 3, 4, 5, 6, 7, 8, 9, 153, 370, 371, 407]
    )
    async def test_is_narcissistic_number_parametrized(self, narcissistic_num):
        """Parametrized test for narcissistic number identification."""
        result = await is_narcissistic_number(narcissistic_num)
        assert result

    @pytest.mark.asyncio
    @pytest.mark.parametrize("keith_num", [14, 19, 28, 47, 61, 75])
    async def test_is_keith_number_parametrized(self, keith_num):
        """Parametrized test for Keith number identification."""
        result = await is_keith_number(keith_num)
        assert result

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "digits,expected", [(3, 495), (4, 6174), (6, 549945), (7, 1194649)]
    )
    async def test_kaprekar_constant_parametrized(self, digits, expected):
        """Parametrized test for Kaprekar constants."""
        result = await kaprekar_constant(digits)
        assert result == expected


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v", "--tb=short", "--asyncio-mode=auto"])
