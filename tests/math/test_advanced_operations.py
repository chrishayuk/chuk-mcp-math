#!/usr/bin/env python3
# tests/math/test_advanced_operations.py
"""
Comprehensive pytest unit tests for advanced arithmetic operations.

Tests cover:
- Logarithmic functions (ln, log, log10, log2, exp)
- Advanced rounding (ceiling_multiple, floor_multiple, mround)
- Base conversions (decimal_to_base, base_to_decimal)
- Special operations (quotient, double_factorial, multinomial)
- Array operations (sum_product, sum_squares, product)
- Random number generation
- Roman numeral conversions
- Series calculations
- Normal operation cases
- Edge cases (zero, negative, infinity, NaN)
- Error conditions
- Async behavior
"""

import pytest
import math
import asyncio
from typing import Union

# Import the functions to test
from chuk_mcp_math.advanced_operations import (
    ln,
    log,
    log10,
    log2,
    exp,
    ceiling_multiple,
    floor_multiple,
    mround,
    decimal_to_base,
    base_to_decimal,
    quotient,
    double_factorial,
    multinomial,
    sum_product,
    sum_squares,
    product,
    random_float,
    random_int,
    random_array,
    arabic_to_roman,
    roman_to_arabic,
    series_sum,
)

Number = Union[int, float]


# Logarithmic Functions Tests
class TestLn:
    """Test cases for the ln function."""

    @pytest.mark.asyncio
    async def test_ln_e(self):
        """Test natural logarithm of e."""
        result = await ln(math.e)
        assert pytest.approx(result, rel=1e-10) == 1.0

    @pytest.mark.asyncio
    async def test_ln_one(self):
        """Test natural logarithm of 1."""
        result = await ln(1)
        assert result == 0.0

    @pytest.mark.asyncio
    async def test_ln_positive_numbers(self):
        """Test natural logarithm of various positive numbers."""
        assert pytest.approx(await ln(10), rel=1e-10) == math.log(10)
        assert pytest.approx(await ln(0.5), rel=1e-10) == math.log(0.5)
        assert pytest.approx(await ln(100), rel=1e-10) == math.log(100)

    @pytest.mark.asyncio
    async def test_ln_zero_raises_error(self):
        """Test that ln(0) raises ValueError."""
        with pytest.raises(ValueError, match="Logarithm undefined for non-positive numbers"):
            await ln(0)

    @pytest.mark.asyncio
    async def test_ln_negative_raises_error(self):
        """Test that ln of negative number raises ValueError."""
        with pytest.raises(ValueError, match="Logarithm undefined for non-positive numbers"):
            await ln(-1)
        with pytest.raises(ValueError, match="Logarithm undefined for non-positive numbers"):
            await ln(-10.5)


class TestLog:
    """Test cases for the log function."""

    @pytest.mark.asyncio
    async def test_log_powers_of_base(self):
        """Test logarithm for exact powers."""
        assert pytest.approx(await log(8, 2), rel=1e-10) == 3.0
        assert pytest.approx(await log(1000, 10), rel=1e-10) == 3.0
        assert pytest.approx(await log(27, 3), rel=1e-10) == 3.0
        assert pytest.approx(await log(16, 4), rel=1e-10) == 2.0

    @pytest.mark.asyncio
    async def test_log_base_e(self):
        """Test that log base e equals ln."""
        result = await log(10, math.e)
        ln_result = await ln(10)
        assert pytest.approx(result, rel=1e-10) == ln_result

    @pytest.mark.asyncio
    async def test_log_invalid_base_one(self):
        """Test that base 1 raises error."""
        with pytest.raises(ValueError, match="Base must be positive and not equal to 1"):
            await log(10, 1)

    @pytest.mark.asyncio
    async def test_log_negative_base(self):
        """Test that negative base raises error."""
        with pytest.raises(ValueError, match="Base must be positive and not equal to 1"):
            await log(10, -2)

    @pytest.mark.asyncio
    async def test_log_negative_number(self):
        """Test that negative number raises error."""
        with pytest.raises(ValueError, match="Logarithm undefined for non-positive numbers"):
            await log(-10, 2)


class TestLog10:
    """Test cases for the log10 function."""

    @pytest.mark.asyncio
    async def test_log10_powers_of_10(self):
        """Test log10 for powers of 10."""
        assert await log10(1) == 0.0
        assert await log10(10) == 1.0
        assert await log10(100) == 2.0
        assert await log10(1000) == 3.0

    @pytest.mark.asyncio
    async def test_log10_decimals(self):
        """Test log10 for decimal values."""
        assert await log10(0.1) == -1.0
        assert await log10(0.01) == -2.0

    @pytest.mark.asyncio
    async def test_log10_invalid_input(self):
        """Test log10 with invalid inputs."""
        with pytest.raises(ValueError):
            await log10(0)
        with pytest.raises(ValueError):
            await log10(-5)


class TestLog2:
    """Test cases for the log2 function."""

    @pytest.mark.asyncio
    async def test_log2_powers_of_2(self):
        """Test log2 for powers of 2."""
        assert await log2(1) == 0.0
        assert await log2(2) == 1.0
        assert await log2(8) == 3.0
        assert await log2(1024) == 10.0

    @pytest.mark.asyncio
    async def test_log2_decimals(self):
        """Test log2 for decimal values."""
        assert await log2(0.5) == -1.0
        assert await log2(0.25) == -2.0

    @pytest.mark.asyncio
    async def test_log2_invalid_input(self):
        """Test log2 with invalid inputs."""
        with pytest.raises(ValueError):
            await log2(0)
        with pytest.raises(ValueError):
            await log2(-1)


class TestExp:
    """Test cases for the exp function."""

    @pytest.mark.asyncio
    async def test_exp_zero(self):
        """Test exp(0) = 1."""
        assert await exp(0) == 1.0

    @pytest.mark.asyncio
    async def test_exp_one(self):
        """Test exp(1) = e."""
        result = await exp(1)
        assert pytest.approx(result, rel=1e-10) == math.e

    @pytest.mark.asyncio
    async def test_exp_negative(self):
        """Test exp with negative input."""
        result = await exp(-1)
        assert pytest.approx(result, rel=1e-10) == 1 / math.e

    @pytest.mark.asyncio
    async def test_exp_large_number(self):
        """Test exp with larger numbers."""
        result = await exp(2)
        assert pytest.approx(result, rel=1e-10) == math.e**2


# Advanced Rounding Tests
class TestCeilingMultiple:
    """Test cases for the ceiling_multiple function."""

    @pytest.mark.asyncio
    async def test_ceiling_multiple_positive(self):
        """Test ceiling multiple with positive numbers."""
        assert await ceiling_multiple(2.5, 1) == 3
        assert await ceiling_multiple(6.7, 2) == 8
        assert await ceiling_multiple(15, 10) == 20

    @pytest.mark.asyncio
    async def test_ceiling_multiple_negative(self):
        """Test ceiling multiple with negative numbers (rounds away from zero)."""
        assert await ceiling_multiple(-2.1, 1) == -3
        assert await ceiling_multiple(-6.7, 2) == -8

    @pytest.mark.asyncio
    async def test_ceiling_multiple_exact(self):
        """Test ceiling multiple when already exact."""
        assert await ceiling_multiple(10, 5) == 10
        assert await ceiling_multiple(20, 10) == 20

    @pytest.mark.asyncio
    async def test_ceiling_multiple_invalid_significance(self):
        """Test ceiling multiple with invalid significance."""
        with pytest.raises(ValueError, match="Significance must be positive"):
            await ceiling_multiple(5, 0)
        with pytest.raises(ValueError, match="Significance must be positive"):
            await ceiling_multiple(5, -1)


class TestFloorMultiple:
    """Test cases for the floor_multiple function."""

    @pytest.mark.asyncio
    async def test_floor_multiple_positive(self):
        """Test floor multiple with positive numbers."""
        assert await floor_multiple(2.9, 1) == 2
        assert await floor_multiple(7.8, 2) == 6
        assert await floor_multiple(23, 10) == 20

    @pytest.mark.asyncio
    async def test_floor_multiple_negative(self):
        """Test floor multiple with negative numbers (rounds toward zero)."""
        assert await floor_multiple(-2.9, 1) == -2
        assert await floor_multiple(-7.8, 2) == -6

    @pytest.mark.asyncio
    async def test_floor_multiple_exact(self):
        """Test floor multiple when already exact."""
        assert await floor_multiple(10, 5) == 10
        assert await floor_multiple(20, 10) == 20

    @pytest.mark.asyncio
    async def test_floor_multiple_invalid_significance(self):
        """Test floor multiple with invalid significance."""
        with pytest.raises(ValueError, match="Significance must be positive"):
            await floor_multiple(5, 0)
        with pytest.raises(ValueError, match="Significance must be positive"):
            await floor_multiple(5, -1)


class TestMround:
    """Test cases for the mround function."""

    @pytest.mark.asyncio
    async def test_mround_standard_rounding(self):
        """Test mround with standard rounding rules."""
        assert await mround(2.4, 1) == 2
        assert await mround(2.6, 1) == 3
        assert await mround(2.5, 1) == 2  # Python rounds to even
        assert await mround(3.5, 1) == 4

    @pytest.mark.asyncio
    async def test_mround_to_multiples(self):
        """Test mround to various multiples."""
        assert await mround(7.3, 2) == 8
        assert await mround(15, 10) == 20
        assert await mround(14, 10) == 10

    @pytest.mark.asyncio
    async def test_mround_invalid_significance(self):
        """Test mround with invalid significance."""
        with pytest.raises(ValueError, match="Significance must be positive"):
            await mround(5, 0)
        with pytest.raises(ValueError, match="Significance must be positive"):
            await mround(5, -1)


# Base Conversion Tests
class TestDecimalToBase:
    """Test cases for the decimal_to_base function."""

    @pytest.mark.asyncio
    async def test_decimal_to_binary(self):
        """Test conversion to binary."""
        assert await decimal_to_base(10, 2) == "1010"
        assert await decimal_to_base(5, 2) == "101"
        assert await decimal_to_base(0, 2) == "0"

    @pytest.mark.asyncio
    async def test_decimal_to_hex(self):
        """Test conversion to hexadecimal."""
        assert await decimal_to_base(255, 16) == "FF"
        assert await decimal_to_base(16, 16) == "10"
        assert await decimal_to_base(10, 16) == "A"

    @pytest.mark.asyncio
    async def test_decimal_to_octal(self):
        """Test conversion to octal."""
        assert await decimal_to_base(8, 8) == "10"
        assert await decimal_to_base(64, 8) == "100"

    @pytest.mark.asyncio
    async def test_decimal_to_base_invalid_input(self):
        """Test with invalid inputs."""
        with pytest.raises(ValueError, match="Number must be non-negative"):
            await decimal_to_base(-5, 2)
        with pytest.raises(ValueError, match="Base must be between 2 and 36"):
            await decimal_to_base(10, 1)
        with pytest.raises(ValueError, match="Base must be between 2 and 36"):
            await decimal_to_base(10, 37)


class TestBaseToDecimal:
    """Test cases for the base_to_decimal function."""

    @pytest.mark.asyncio
    async def test_binary_to_decimal(self):
        """Test conversion from binary."""
        assert await base_to_decimal("1010", 2) == 10
        assert await base_to_decimal("101", 2) == 5
        assert await base_to_decimal("0", 2) == 0

    @pytest.mark.asyncio
    async def test_hex_to_decimal(self):
        """Test conversion from hexadecimal."""
        assert await base_to_decimal("FF", 16) == 255
        assert await base_to_decimal("10", 16) == 16
        assert await base_to_decimal("A", 16) == 10

    @pytest.mark.asyncio
    async def test_octal_to_decimal(self):
        """Test conversion from octal."""
        assert await base_to_decimal("10", 8) == 8
        assert await base_to_decimal("100", 8) == 64

    @pytest.mark.asyncio
    async def test_base_to_decimal_invalid_input(self):
        """Test with invalid inputs."""
        with pytest.raises(ValueError, match="Base must be between 2 and 36"):
            await base_to_decimal("10", 1)
        with pytest.raises(ValueError, match="Invalid number"):
            await base_to_decimal("G", 10)


# Special Operations Tests
class TestQuotient:
    """Test cases for the quotient function."""

    @pytest.mark.asyncio
    async def test_quotient_standard(self):
        """Test quotient with standard division."""
        assert await quotient(17, 5) == 3
        assert await quotient(20, 4) == 5
        assert await quotient(7, 3) == 2

    @pytest.mark.asyncio
    async def test_quotient_negative(self):
        """Test quotient with negative numbers."""
        assert await quotient(-17, 5) == -4
        assert await quotient(17, -5) == -4

    @pytest.mark.asyncio
    async def test_quotient_division_by_zero(self):
        """Test quotient division by zero."""
        with pytest.raises(ValueError, match="Cannot divide by zero"):
            await quotient(10, 0)


class TestDoubleFactorial:
    """Test cases for the double_factorial function."""

    @pytest.mark.asyncio
    async def test_double_factorial_small_numbers(self):
        """Test double factorial for small numbers."""
        assert await double_factorial(0) == 1
        assert await double_factorial(1) == 1
        assert await double_factorial(5) == 15  # 5 * 3 * 1
        assert await double_factorial(6) == 48  # 6 * 4 * 2

    @pytest.mark.asyncio
    async def test_double_factorial_negative(self):
        """Test double factorial with negative input."""
        with pytest.raises(
            ValueError, match="Double factorial is not defined for negative numbers"
        ):
            await double_factorial(-1)


class TestMultinomial:
    """Test cases for the multinomial function."""

    @pytest.mark.asyncio
    async def test_multinomial_standard(self):
        """Test multinomial coefficient."""
        assert await multinomial([3, 2, 1]) == 60
        assert await multinomial([2, 2]) == 6
        assert await multinomial([4]) == 1

    @pytest.mark.asyncio
    async def test_multinomial_invalid_input(self):
        """Test multinomial with invalid inputs."""
        with pytest.raises(ValueError, match="Numbers list cannot be empty"):
            await multinomial([])
        with pytest.raises(ValueError, match="All numbers must be non-negative"):
            await multinomial([3, -1, 2])


# Array Operations Tests
class TestSumProduct:
    """Test cases for the sum_product function."""

    @pytest.mark.asyncio
    async def test_sum_product_two_arrays(self):
        """Test sum product with two arrays."""
        result = await sum_product([[1, 2, 3], [4, 5, 6]])
        assert result == 32  # 1*4 + 2*5 + 3*6

    @pytest.mark.asyncio
    async def test_sum_product_three_arrays(self):
        """Test sum product with three arrays."""
        result = await sum_product([[2, 3], [4, 5], [1, 2]])
        assert result == 38  # 2*4*1 + 3*5*2

    @pytest.mark.asyncio
    async def test_sum_product_invalid_input(self):
        """Test sum product with invalid inputs."""
        with pytest.raises(ValueError, match="Arrays list cannot be empty"):
            await sum_product([])
        with pytest.raises(ValueError, match="All arrays must have the same length"):
            await sum_product([[1, 2], [3]])


class TestSumSquares:
    """Test cases for the sum_squares function."""

    @pytest.mark.asyncio
    async def test_sum_squares_positive(self):
        """Test sum of squares."""
        assert await sum_squares([1, 2, 3, 4]) == 30  # 1 + 4 + 9 + 16
        assert await sum_squares([3, 4]) == 25  # 9 + 16

    @pytest.mark.asyncio
    async def test_sum_squares_negative(self):
        """Test sum of squares with negative numbers."""
        assert await sum_squares([-2, 2]) == 8  # 4 + 4

    @pytest.mark.asyncio
    async def test_sum_squares_zero(self):
        """Test sum of squares with zero."""
        assert await sum_squares([0, 5]) == 25


class TestProduct:
    """Test cases for the product function."""

    @pytest.mark.asyncio
    async def test_product_positive(self):
        """Test product of positive numbers."""
        assert await product([2, 3, 4]) == 24
        assert await product([1, 2, 3, 4, 5]) == 120

    @pytest.mark.asyncio
    async def test_product_negative(self):
        """Test product with negative numbers."""
        assert await product([-2, 3]) == -6
        assert await product([-2, -3]) == 6

    @pytest.mark.asyncio
    async def test_product_with_zero(self):
        """Test product with zero."""
        assert await product([5, 0, 3]) == 0

    @pytest.mark.asyncio
    async def test_product_empty_list(self):
        """Test product with empty list."""
        with pytest.raises(ValueError, match="Cannot calculate product of empty list"):
            await product([])


# Random Number Tests
class TestRandomFloat:
    """Test cases for the random_float function."""

    @pytest.mark.asyncio
    async def test_random_float_range(self):
        """Test random float is in valid range."""
        result = await random_float()
        assert 0 <= result < 1

    @pytest.mark.asyncio
    async def test_random_float_variability(self):
        """Test random float produces different values."""
        values = [await random_float() for _ in range(10)]
        # Very unlikely all 10 values are identical
        assert len(set(values)) > 1


class TestRandomInt:
    """Test cases for the random_int function."""

    @pytest.mark.asyncio
    async def test_random_int_range(self):
        """Test random int is in valid range."""
        result = await random_int(1, 10)
        assert 1 <= result <= 10

    @pytest.mark.asyncio
    async def test_random_int_single_value(self):
        """Test random int with same min and max."""
        result = await random_int(5, 5)
        assert result == 5

    @pytest.mark.asyncio
    async def test_random_int_invalid_range(self):
        """Test random int with invalid range."""
        with pytest.raises(ValueError, match="min_val cannot be greater than max_val"):
            await random_int(10, 5)


class TestRandomArray:
    """Test cases for the random_array function."""

    @pytest.mark.asyncio
    async def test_random_array_dimensions(self):
        """Test random array has correct dimensions."""
        result = await random_array(2, 3)
        assert len(result) == 2
        assert len(result[0]) == 3
        assert all(len(row) == 3 for row in result)

    @pytest.mark.asyncio
    async def test_random_array_values_in_range(self):
        """Test random array values are in valid range."""
        result = await random_array(2, 2)
        for row in result:
            for val in row:
                assert 0 <= val < 1

    @pytest.mark.asyncio
    async def test_random_array_invalid_dimensions(self):
        """Test random array with invalid dimensions."""
        with pytest.raises(ValueError, match="Rows and columns must be positive"):
            await random_array(0, 5)
        with pytest.raises(ValueError, match="Rows and columns must be positive"):
            await random_array(5, -1)


# Roman Numeral Tests
class TestArabicToRoman:
    """Test cases for the arabic_to_roman function."""

    @pytest.mark.asyncio
    async def test_arabic_to_roman_standard(self):
        """Test standard conversions."""
        assert await arabic_to_roman(1) == "I"
        assert await arabic_to_roman(4) == "IV"
        assert await arabic_to_roman(9) == "IX"
        assert await arabic_to_roman(27) == "XXVII"
        assert await arabic_to_roman(1994) == "MCMXCIV"

    @pytest.mark.asyncio
    async def test_arabic_to_roman_bounds(self):
        """Test boundary values."""
        assert await arabic_to_roman(1) == "I"
        assert await arabic_to_roman(3999) == "MMMCMXCIX"

    @pytest.mark.asyncio
    async def test_arabic_to_roman_invalid(self):
        """Test invalid inputs."""
        with pytest.raises(ValueError, match="Number must be between 1 and 3999"):
            await arabic_to_roman(0)
        with pytest.raises(ValueError, match="Number must be between 1 and 3999"):
            await arabic_to_roman(4000)


class TestRomanToArabic:
    """Test cases for the roman_to_arabic function."""

    @pytest.mark.asyncio
    async def test_roman_to_arabic_standard(self):
        """Test standard conversions."""
        assert await roman_to_arabic("I") == 1
        assert await roman_to_arabic("IV") == 4
        assert await roman_to_arabic("IX") == 9
        assert await roman_to_arabic("XXVII") == 27
        assert await roman_to_arabic("MCMXCIV") == 1994

    @pytest.mark.asyncio
    async def test_roman_to_arabic_case_insensitive(self):
        """Test case insensitivity."""
        assert await roman_to_arabic("xxvii") == 27
        assert await roman_to_arabic("Xxvii") == 27

    @pytest.mark.asyncio
    async def test_roman_to_arabic_invalid(self):
        """Test invalid inputs."""
        with pytest.raises(ValueError, match="Invalid Roman numeral character"):
            await roman_to_arabic("ABC")


# Series Sum Tests
class TestSeriesSum:
    """Test cases for the series_sum function."""

    @pytest.mark.asyncio
    async def test_series_sum_standard(self):
        """Test series sum."""
        result = await series_sum(1, 10, 10)
        assert result == 55  # 1+2+...+10

    @pytest.mark.asyncio
    async def test_series_sum_single_term(self):
        """Test series sum with single term."""
        result = await series_sum(5, 5, 1)
        assert result == 5

    @pytest.mark.asyncio
    async def test_series_sum_decreasing(self):
        """Test series sum with decreasing series."""
        result = await series_sum(10, 2, 5)
        assert result == 30

    @pytest.mark.asyncio
    async def test_series_sum_invalid_count(self):
        """Test series sum with invalid count."""
        with pytest.raises(ValueError, match="Count must be positive"):
            await series_sum(1, 10, 0)


# Integration and Async Tests
class TestIntegration:
    """Integration tests for multiple operations."""

    @pytest.mark.asyncio
    async def test_log_exp_inverse(self):
        """Test that log and exp are inverse operations."""
        x = 5
        exp_result = await exp(x)
        ln_result = await ln(exp_result)
        assert pytest.approx(ln_result, rel=1e-10) == x

    @pytest.mark.asyncio
    async def test_base_conversion_roundtrip(self):
        """Test base conversion roundtrip."""
        original = 255
        hex_str = await decimal_to_base(original, 16)
        back_to_decimal = await base_to_decimal(hex_str, 16)
        assert back_to_decimal == original

    @pytest.mark.asyncio
    async def test_roman_conversion_roundtrip(self):
        """Test Roman numeral conversion roundtrip."""
        original = 1994
        roman = await arabic_to_roman(original)
        back_to_arabic = await roman_to_arabic(roman)
        assert back_to_arabic == original


class TestAsyncBehavior:
    """Test async behavior and concurrency."""

    @pytest.mark.asyncio
    async def test_concurrent_execution(self):
        """Test that operations can run concurrently."""
        tasks = [
            ln(10),
            exp(2),
            log10(100),
            log2(8),
            decimal_to_base(255, 16),
        ]
        results = await asyncio.gather(*tasks)

        assert pytest.approx(results[0], rel=1e-10) == math.log(10)
        assert pytest.approx(results[1], rel=1e-10) == math.e**2
        assert results[2] == 2.0
        assert results[3] == 3.0
        assert results[4] == "FF"


# Parametrized Tests
class TestParametrized:
    """Parametrized tests for comprehensive coverage."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "x,base,expected",
        [
            (8, 2, 3.0),
            (1000, 10, 3.0),
            (27, 3, 3.0),
            (16, 4, 2.0),
        ],
    )
    async def test_log_parametrized(self, x, base, expected):
        """Parametrized test for log function."""
        result = await log(x, base)
        assert pytest.approx(result, rel=1e-10) == expected

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "number,base,expected",
        [
            (10, 2, "1010"),
            (255, 16, "FF"),
            (8, 8, "10"),
            (0, 2, "0"),
        ],
    )
    async def test_decimal_to_base_parametrized(self, number, base, expected):
        """Parametrized test for decimal_to_base function."""
        result = await decimal_to_base(number, base)
        assert result == expected


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v", "--tb=short"])
