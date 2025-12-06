#!/usr/bin/env python3
# tests/math/number_theory/test_number_systems.py
"""
Comprehensive pytest unit tests for number systems and base conversions.

Tests cover:
- Binary, octal, hexadecimal conversions
- General base conversion (2-36)
- Number type generators
- Validation functions
- Arithmetic in different bases
- Edge cases and error conditions
"""

import pytest
import asyncio

from chuk_mcp_math.number_theory.number_systems import (
    binary_to_decimal,
    decimal_to_binary,
    base_conversion,
    decimal_to_octal,
    octal_to_decimal,
    decimal_to_hexadecimal,
    hexadecimal_to_decimal,
    natural_numbers,
    whole_numbers,
    integers_in_range,
    validate_number_in_base,
    is_valid_base,
    add_in_base,
    multiply_in_base,
)


class TestBinaryConversions:
    """Test binary-decimal conversions."""

    @pytest.mark.asyncio
    async def test_binary_to_decimal_basic(self):
        """Test basic binary to decimal conversion."""
        assert await binary_to_decimal("0") == 0
        assert await binary_to_decimal("1") == 1
        assert await binary_to_decimal("10") == 2
        assert await binary_to_decimal("11") == 3
        assert await binary_to_decimal("100") == 4
        assert await binary_to_decimal("1010") == 10
        assert await binary_to_decimal("11110000") == 240

    @pytest.mark.asyncio
    async def test_decimal_to_binary_basic(self):
        """Test basic decimal to binary conversion."""
        assert await decimal_to_binary(0) == "0"
        assert await decimal_to_binary(1) == "1"
        assert await decimal_to_binary(2) == "10"
        assert await decimal_to_binary(3) == "11"
        assert await decimal_to_binary(4) == "100"
        assert await decimal_to_binary(10) == "1010"
        assert await decimal_to_binary(240) == "11110000"

    @pytest.mark.asyncio
    async def test_binary_roundtrip(self):
        """Test binary conversion roundtrip."""
        for n in [0, 1, 7, 15, 31, 63, 127, 255, 1024]:
            binary = await decimal_to_binary(n)
            result = await binary_to_decimal(binary)
            assert result == n

    @pytest.mark.asyncio
    async def test_binary_invalid_input(self):
        """Test binary conversion with invalid input."""
        with pytest.raises(ValueError):
            await binary_to_decimal("12")  # Contains non-binary digit

        with pytest.raises(ValueError):
            await binary_to_decimal("")  # Empty string

    @pytest.mark.asyncio
    async def test_decimal_to_binary_negative(self):
        """Test decimal to binary with negative number."""
        with pytest.raises(ValueError):
            await decimal_to_binary(-5)


class TestOctalConversions:
    """Test octal conversions."""

    @pytest.mark.asyncio
    async def test_decimal_to_octal(self):
        """Test decimal to octal conversion."""
        assert await decimal_to_octal(0) == "0"
        assert await decimal_to_octal(8) == "10"
        assert await decimal_to_octal(64) == "100"
        assert await decimal_to_octal(255) == "377"

    @pytest.mark.asyncio
    async def test_octal_to_decimal(self):
        """Test octal to decimal conversion."""
        assert await octal_to_decimal("0") == 0
        assert await octal_to_decimal("10") == 8
        assert await octal_to_decimal("100") == 64
        assert await octal_to_decimal("377") == 255

    @pytest.mark.asyncio
    async def test_octal_roundtrip(self):
        """Test octal conversion roundtrip."""
        for n in [0, 7, 8, 63, 64, 511, 512]:
            octal = await decimal_to_octal(n)
            result = await octal_to_decimal(octal)
            assert result == n


class TestHexadecimalConversions:
    """Test hexadecimal conversions."""

    @pytest.mark.asyncio
    async def test_decimal_to_hex(self):
        """Test decimal to hexadecimal conversion."""
        assert await decimal_to_hexadecimal(0) == "0"
        assert await decimal_to_hexadecimal(10) == "A"
        assert await decimal_to_hexadecimal(15) == "F"
        assert await decimal_to_hexadecimal(16) == "10"
        assert await decimal_to_hexadecimal(255) == "FF"
        assert await decimal_to_hexadecimal(171) == "AB"

    @pytest.mark.asyncio
    async def test_hex_to_decimal(self):
        """Test hexadecimal to decimal conversion."""
        assert await hexadecimal_to_decimal("0") == 0
        assert await hexadecimal_to_decimal("A") == 10
        assert await hexadecimal_to_decimal("F") == 15
        assert await hexadecimal_to_decimal("10") == 16
        assert await hexadecimal_to_decimal("FF") == 255
        assert await hexadecimal_to_decimal("AB") == 171

    @pytest.mark.asyncio
    async def test_hex_case_insensitive(self):
        """Test hex conversion is case insensitive."""
        assert await hexadecimal_to_decimal("ff") == 255
        assert await hexadecimal_to_decimal("FF") == 255
        assert await hexadecimal_to_decimal("aB") == 171

    @pytest.mark.asyncio
    async def test_hex_roundtrip(self):
        """Test hex conversion roundtrip."""
        for n in [0, 10, 15, 16, 255, 256, 4095]:
            hex_val = await decimal_to_hexadecimal(n)
            result = await hexadecimal_to_decimal(hex_val)
            assert result == n


class TestGeneralBaseConversion:
    """Test general base conversion function."""

    @pytest.mark.asyncio
    async def test_base_2_conversions(self):
        """Test base 2 conversions."""
        assert await base_conversion("1010", 2, 10) == "10"
        assert await base_conversion("10", 10, 2) == "1010"

    @pytest.mark.asyncio
    async def test_base_16_conversions(self):
        """Test base 16 conversions."""
        assert await base_conversion("FF", 16, 10) == "255"
        assert await base_conversion("255", 10, 16) == "FF"

    @pytest.mark.asyncio
    async def test_base_8_conversions(self):
        """Test base 8 conversions."""
        assert await base_conversion("777", 8, 2) == "111111111"
        assert await base_conversion("100", 8, 10) == "64"

    @pytest.mark.asyncio
    async def test_same_base_conversion(self):
        """Test conversion between same bases."""
        assert await base_conversion("123", 10, 10) == "123"
        assert await base_conversion("FF", 16, 16) == "FF"

    @pytest.mark.asyncio
    async def test_base_36_conversion(self):
        """Test maximum base (36) conversion."""
        # Base 36 uses 0-9 and A-Z
        result = await base_conversion("Z", 36, 10)
        assert result == "35"  # Z is the last digit in base 36

    @pytest.mark.asyncio
    async def test_invalid_base(self):
        """Test with invalid base values."""
        with pytest.raises(ValueError):
            await base_conversion("10", 1, 10)  # Base too small

        with pytest.raises(ValueError):
            await base_conversion("10", 10, 37)  # Base too large

    @pytest.mark.asyncio
    async def test_invalid_digit_for_base(self):
        """Test with digits invalid for given base."""
        with pytest.raises(ValueError):
            await base_conversion("9", 8, 10)  # 9 not valid in base 8

        with pytest.raises(ValueError):
            await base_conversion("G", 16, 10)  # G not valid in base 16


class TestNumberTypeGenerators:
    """Test number type generator functions."""

    @pytest.mark.asyncio
    async def test_natural_numbers(self):
        """Test natural number generation."""
        assert await natural_numbers(1, 5) == [1, 2, 3, 4, 5]
        assert await natural_numbers(10, 12) == [10, 11, 12]
        assert await natural_numbers(1, 1) == [1]

    @pytest.mark.asyncio
    async def test_natural_numbers_invalid_range(self):
        """Test natural numbers with invalid range."""
        assert await natural_numbers(5, 3) == []  # End < start

    @pytest.mark.asyncio
    async def test_natural_numbers_error(self):
        """Test natural numbers with invalid start."""
        with pytest.raises(ValueError):
            await natural_numbers(0, 5)  # Start must be positive

    @pytest.mark.asyncio
    async def test_whole_numbers(self):
        """Test whole number generation."""
        assert await whole_numbers(0, 5) == [0, 1, 2, 3, 4, 5]
        assert await whole_numbers(3, 7) == [3, 4, 5, 6, 7]
        assert await whole_numbers(0, 0) == [0]

    @pytest.mark.asyncio
    async def test_whole_numbers_error(self):
        """Test whole numbers with negative start."""
        with pytest.raises(ValueError):
            await whole_numbers(-1, 5)

    @pytest.mark.asyncio
    async def test_integers_in_range(self):
        """Test integer range generation."""
        assert await integers_in_range(-3, 3) == [-3, -2, -1, 0, 1, 2, 3]
        assert await integers_in_range(-5, -2) == [-5, -4, -3, -2]
        assert await integers_in_range(2, 5) == [2, 3, 4, 5]
        assert await integers_in_range(0, 0) == [0]

    @pytest.mark.asyncio
    async def test_integers_invalid_range(self):
        """Test integers with invalid range."""
        assert await integers_in_range(5, 2) == []


class TestValidationFunctions:
    """Test validation functions."""

    @pytest.mark.asyncio
    async def test_validate_binary(self):
        """Test binary number validation."""
        assert await validate_number_in_base("1010", 2) is True
        assert await validate_number_in_base("0101", 2) is True
        assert await validate_number_in_base("129", 2) is False
        assert await validate_number_in_base("", 2) is False

    @pytest.mark.asyncio
    async def test_validate_hex(self):
        """Test hexadecimal validation."""
        assert await validate_number_in_base("ABC", 16) is True
        assert await validate_number_in_base("123", 16) is True
        assert await validate_number_in_base("XYZ", 16) is False

    @pytest.mark.asyncio
    async def test_validate_octal(self):
        """Test octal validation."""
        assert await validate_number_in_base("777", 8) is True
        assert await validate_number_in_base("123", 8) is True
        assert await validate_number_in_base("89", 8) is False

    @pytest.mark.asyncio
    async def test_is_valid_base(self):
        """Test base validation."""
        assert await is_valid_base(2) is True
        assert await is_valid_base(10) is True
        assert await is_valid_base(16) is True
        assert await is_valid_base(36) is True
        assert await is_valid_base(1) is False
        assert await is_valid_base(37) is False
        assert await is_valid_base(0) is False
        assert await is_valid_base(-1) is False


class TestArithmeticInBases:
    """Test arithmetic operations in different bases."""

    @pytest.mark.asyncio
    async def test_add_in_binary(self):
        """Test binary addition."""
        assert await add_in_base("1010", "1100", 2) == "10110"
        assert await add_in_base("1", "1", 2) == "10"
        assert await add_in_base("0", "0", 2) == "0"

    @pytest.mark.asyncio
    async def test_add_in_hex(self):
        """Test hexadecimal addition."""
        assert await add_in_base("FF", "1", 16) == "100"
        assert await add_in_base("A", "B", 16) == "15"

    @pytest.mark.asyncio
    async def test_add_in_octal(self):
        """Test octal addition."""
        assert await add_in_base("777", "1", 8) == "1000"
        assert await add_in_base("7", "7", 8) == "16"

    @pytest.mark.asyncio
    async def test_add_in_decimal(self):
        """Test decimal addition."""
        assert await add_in_base("99", "1", 10) == "100"
        assert await add_in_base("25", "75", 10) == "100"

    @pytest.mark.asyncio
    async def test_multiply_in_binary(self):
        """Test binary multiplication."""
        assert await multiply_in_base("101", "11", 2) == "1111"
        assert await multiply_in_base("10", "10", 2) == "100"

    @pytest.mark.asyncio
    async def test_multiply_in_hex(self):
        """Test hexadecimal multiplication."""
        assert await multiply_in_base("A", "B", 16) == "6E"
        assert await multiply_in_base("2", "8", 16) == "10"

    @pytest.mark.asyncio
    async def test_multiply_in_decimal(self):
        """Test decimal multiplication."""
        assert await multiply_in_base("12", "34", 10) == "408"
        assert await multiply_in_base("5", "7", 10) == "35"

    @pytest.mark.asyncio
    async def test_arithmetic_invalid_base(self):
        """Test arithmetic with invalid base."""
        with pytest.raises(ValueError):
            await add_in_base("10", "20", 1)

    @pytest.mark.asyncio
    async def test_arithmetic_invalid_number(self):
        """Test arithmetic with invalid number for base."""
        with pytest.raises(ValueError):
            await add_in_base("129", "1", 2)  # 9 invalid in binary


class TestIntegration:
    """Integration tests."""

    @pytest.mark.asyncio
    async def test_multi_base_conversion_chain(self):
        """Test chained conversions between multiple bases."""
        # Decimal -> Binary -> Hex -> Decimal
        n = 42
        binary = await decimal_to_binary(n)
        hex_from_bin = await base_conversion(binary, 2, 16)
        decimal_from_hex = await hexadecimal_to_decimal(hex_from_bin)

        assert decimal_from_hex == n

    @pytest.mark.asyncio
    async def test_arithmetic_matches_decimal(self):
        """Test that arithmetic in different bases gives correct results."""
        # Test in different bases
        a, b = 10, 7

        # Decimal
        decimal_sum = await add_in_base(str(a), str(b), 10)
        assert int(decimal_sum) == 17

        # Binary
        binary_a = await decimal_to_binary(a)
        binary_b = await decimal_to_binary(b)
        binary_sum = await add_in_base(binary_a, binary_b, 2)
        decimal_result = await binary_to_decimal(binary_sum)
        assert decimal_result == 17


class TestAsyncBehavior:
    """Test async behavior."""

    @pytest.mark.asyncio
    async def test_all_functions_async(self):
        """Test that all functions are async."""
        operations = [
            binary_to_decimal("1010"),
            decimal_to_binary(10),
            base_conversion("FF", 16, 10),
            natural_numbers(1, 5),
            validate_number_in_base("123", 10),
            add_in_base("5", "3", 10),
        ]

        for op in operations:
            assert asyncio.iscoroutine(op)

        results = await asyncio.gather(*operations)
        assert len(results) == 6

    @pytest.mark.asyncio
    async def test_concurrent_conversions(self):
        """Test concurrent base conversions."""
        tasks = [decimal_to_binary(n) for n in range(0, 20)]
        results = await asyncio.gather(*tasks)

        assert len(results) == 20
        assert all(isinstance(r, str) for r in results)


class TestParametrized:
    """Parametrized tests."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "decimal,binary",
        [
            (0, "0"),
            (1, "1"),
            (2, "10"),
            (3, "11"),
            (4, "100"),
            (10, "1010"),
            (15, "1111"),
            (16, "10000"),
            (255, "11111111"),
        ],
    )
    async def test_binary_conversion_parametrized(self, decimal, binary):
        """Parametrized binary conversion tests."""
        assert await decimal_to_binary(decimal) == binary
        assert await binary_to_decimal(binary) == decimal

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "decimal,hex_val",
        [
            (0, "0"),
            (10, "A"),
            (15, "F"),
            (16, "10"),
            (255, "FF"),
            (256, "100"),
            (4095, "FFF"),
        ],
    )
    async def test_hex_conversion_parametrized(self, decimal, hex_val):
        """Parametrized hex conversion tests."""
        assert await decimal_to_hexadecimal(decimal) == hex_val
        assert await hexadecimal_to_decimal(hex_val) == decimal


class TestLargeNumberHandling:
    """Test async yielding with large numbers and ranges."""

    @pytest.mark.asyncio
    async def test_binary_to_decimal_long_string(self):
        """Test binary to decimal with very long binary string (triggers async sleep)."""
        # Create a binary string with > 100 bits to trigger line 98
        long_binary = "1" + "0" * 150
        result = await binary_to_decimal(long_binary)
        assert result == 2**150

    @pytest.mark.asyncio
    async def test_decimal_to_binary_large_number(self):
        """Test decimal to binary with large number (triggers async sleep)."""
        # Large number that creates > 100 binary digits to trigger line 156
        large_num = 2**150
        result = await decimal_to_binary(large_num)
        assert len(result) > 100
        assert result[0] == "1"

    @pytest.mark.asyncio
    async def test_base_conversion_large_number(self):
        """Test base conversion with large numbers (triggers async sleep)."""
        # Create a number with > 50 digits to trigger lines 242 and 259
        large_hex = "F" * 60
        result = await base_conversion(large_hex, 16, 10)
        assert len(result) > 50

    @pytest.mark.asyncio
    async def test_natural_numbers_large_range(self):
        """Test natural numbers with large range (triggers async sleep)."""
        # Range > 1000 to trigger lines 501 and 509
        result = await natural_numbers(1, 1500)
        assert len(result) == 1500
        assert result[0] == 1
        assert result[-1] == 1500

    @pytest.mark.asyncio
    async def test_whole_numbers_large_range(self):
        """Test whole numbers with large range (triggers async sleep)."""
        # Range > 1000 to trigger lines 564 and 572
        result = await whole_numbers(0, 1500)
        assert len(result) == 1501
        assert result[0] == 0
        assert result[-1] == 1500

    @pytest.mark.asyncio
    async def test_whole_numbers_invalid_range(self):
        """Test whole numbers with invalid range (triggers line 560)."""
        result = await whole_numbers(10, 5)
        assert result == []

    @pytest.mark.asyncio
    async def test_integers_in_range_large(self):
        """Test integers with large range (triggers async sleep)."""
        # Range > 1000 to trigger lines 622 and 630
        result = await integers_in_range(-500, 1000)
        assert len(result) == 1501
        assert result[0] == -500
        assert result[-1] == 1000


class TestEdgeCasesAndErrors:
    """Test additional edge cases and error conditions."""

    @pytest.mark.asyncio
    async def test_base_conversion_empty_string(self):
        """Test base conversion with empty string (triggers line 216)."""
        with pytest.raises(ValueError, match="Number string cannot be empty"):
            await base_conversion("", 10, 2)

    @pytest.mark.asyncio
    async def test_validate_number_invalid_base(self):
        """Test validate_number_in_base with invalid base (triggers line 685)."""
        # Base outside 2-36 range
        result = await validate_number_in_base("123", 1)
        assert result is False

        result = await validate_number_in_base("123", 37)
        assert result is False

    @pytest.mark.asyncio
    async def test_add_in_base_invalid_second_number(self):
        """Test add_in_base with invalid second number (triggers line 787)."""
        with pytest.raises(ValueError, match="Invalid number"):
            await add_in_base("10", "99", 2)  # 99 is invalid in binary

    @pytest.mark.asyncio
    async def test_multiply_in_base_invalid_base(self):
        """Test multiply_in_base with invalid base (triggers line 849)."""
        with pytest.raises(ValueError, match="Invalid base"):
            await multiply_in_base("10", "10", 1)

    @pytest.mark.asyncio
    async def test_multiply_in_base_invalid_first_number(self):
        """Test multiply_in_base with invalid first number (triggers line 852)."""
        with pytest.raises(ValueError, match="Invalid number"):
            await multiply_in_base("G", "10", 10)  # G is invalid in base 10

    @pytest.mark.asyncio
    async def test_multiply_in_base_invalid_second_number(self):
        """Test multiply_in_base with invalid second number (triggers line 855)."""
        with pytest.raises(ValueError, match="Invalid number"):
            await multiply_in_base("10", "Z", 16)  # Z is invalid in base 16


class TestDemoFunctions:
    """Test the demo functions at end of module."""

    @pytest.mark.asyncio
    async def test_demo_functions_execute(self):
        """Test that demo functions execute without errors."""
        from chuk_mcp_math.number_theory import number_systems

        # These are demo/test functions in the module - just verify they run
        if hasattr(number_systems, "test_number_systems"):
            await number_systems.test_number_systems()

        if hasattr(number_systems, "demo_base_conversions"):
            await number_systems.demo_base_conversions()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
