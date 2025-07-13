#!/usr/bin/env python3
# tests/math/arithmetic/number_theory/test_digital_operations.py
"""
Comprehensive pytest test suite for digital_operations.py module.

Tests cover:
- Digital sums and roots: digit_sum, digital_root, digit_product, persistent_digital_root
- Digital transformations: digit_reversal, digit_sort
- Palindromic numbers: is_palindromic_number, palindromic_numbers, next_palindrome
- Harshad numbers: is_harshad_number, harshad_numbers
- Base conversions: number_to_base, base_to_number
- Digit properties: digit_count, digit_frequency, is_repdigit
- Special numbers: is_automorphic_number, automorphic_numbers
- Mathematical properties and relationships
- Edge cases, error conditions, and performance testing
- Async behavior verification
"""

import pytest
import asyncio
import time
import math
from typing import List, Dict

# Import the functions to test
from chuk_mcp_functions.math.arithmetic.number_theory.digital_operations import (
    # Digital sums and roots
    digit_sum, digital_root, digit_product, persistent_digital_root,
    
    # Digital transformations
    digit_reversal, digit_sort,
    
    # Palindromic numbers
    is_palindromic_number, palindromic_numbers, next_palindrome,
    
    # Harshad numbers
    is_harshad_number, harshad_numbers,
    
    # Base conversions
    number_to_base, base_to_number,
    
    # Digit properties
    digit_count, digit_frequency, is_repdigit,
    
    # Special numbers
    is_automorphic_number, automorphic_numbers
)

# ============================================================================
# DIGITAL SUMS AND ROOTS TESTS
# ============================================================================

class TestDigitalSumsAndRoots:
    """Test cases for digital sum and root functions."""
    
    @pytest.mark.asyncio
    async def test_digit_sum_basic_cases(self):
        """Test digit sum with basic cases."""
        test_cases = [
            (0, 0),
            (9, 9),
            (12, 3),
            (123, 6),
            (999, 27),
            (12345, 15),
            (9876, 30),
            (1000, 1),
            (555, 15)
        ]
        
        for n, expected in test_cases:
            result = await digit_sum(n)
            assert result == expected, f"digit_sum({n}) should be {expected}, got {result}"
    
    @pytest.mark.asyncio
    async def test_digit_sum_different_bases(self):
        """Test digit sum in different bases."""
        # 255 in binary (11111111) should have sum 8
        assert await digit_sum(255, 2) == 8
        
        # 15 in binary (1111) should have sum 4
        assert await digit_sum(15, 2) == 4
        
        # 255 in base 16 (FF) should have sum 30 (15+15)
        assert await digit_sum(255, 16) == 30
        
        # 100 in base 8 (144) should have sum 9 (1+4+4)
        assert await digit_sum(100, 8) == 9
    
    @pytest.mark.asyncio
    async def test_digit_sum_negative_numbers(self):
        """Test digit sum with negative numbers (should use absolute value)."""
        assert await digit_sum(-123) == 6
        assert await digit_sum(-999) == 27
        assert await digit_sum(-1) == 1
    
    @pytest.mark.asyncio
    async def test_digit_sum_invalid_base(self):
        """Test digit sum with invalid base."""
        with pytest.raises(ValueError, match="Base must be at least 2"):
            await digit_sum(123, 1)
        
        with pytest.raises(ValueError, match="Base must be at least 2"):
            await digit_sum(123, 0)
    
    @pytest.mark.asyncio
    async def test_digital_root_basic_cases(self):
        """Test digital root with known values."""
        test_cases = [
            (0, 0),
            (9, 9),
            (12, 3),
            (123, 6),
            (999, 9),
            (12345, 6),  # 12345 → 15 → 6
            (9876, 3),   # 9876 → 30 → 3
            (1999, 1),   # 1999 → 28 → 10 → 1
            (65, 2)      # 65 → 11 → 2
        ]
        
        for n, expected in test_cases:
            result = await digital_root(n)
            assert result == expected, f"digital_root({n}) should be {expected}, got {result}"
    
    @pytest.mark.asyncio
    async def test_digital_root_formula_base_10(self):
        """Test digital root formula for base 10: 1 + (n-1) % 9."""
        for n in range(1, 100):
            expected = 1 + (n - 1) % 9
            result = await digital_root(n)
            assert result == expected, f"digital_root({n}) formula mismatch"
        
        # Special case for 0
        assert await digital_root(0) == 0
    
    @pytest.mark.asyncio
    async def test_digital_root_different_bases(self):
        """Test digital root in different bases."""
        # In base 8, formula is 1 + (n-1) % 7
        assert await digital_root(10, 8) == 1 + (10 - 1) % 7  # Should be 3
        assert await digital_root(15, 8) == 1 + (15 - 1) % 7  # Should be 1
        
        # Test with base 2
        assert await digital_root(7, 2) == 1  # 111₂ → 3₂ → 11₂ → 2₂ → 10₂ → 1₂
    
    @pytest.mark.asyncio
    async def test_digital_root_negative_numbers(self):
        """Test digital root with negative numbers."""
        assert await digital_root(-123) == 6
        assert await digital_root(-999) == 9
        assert await digital_root(-12345) == 6
    
    @pytest.mark.asyncio
    async def test_digit_product_basic_cases(self):
        """Test digit product with basic cases."""
        test_cases = [
            (0, 0),
            (1, 1),
            (123, 6),    # 1×2×3
            (999, 729),  # 9×9×9
            (1023, 0),   # Contains 0
            (456, 120),  # 4×5×6
            (789, 504),  # 7×8×9
            (1111, 1),   # 1×1×1×1
            (25, 10)     # 2×5
        ]
        
        for n, expected in test_cases:
            result = await digit_product(n)
            assert result == expected, f"digit_product({n}) should be {expected}, got {result}"
    
    @pytest.mark.asyncio
    async def test_digit_product_with_zeros(self):
        """Test digit product with numbers containing zeros."""
        numbers_with_zeros = [10, 100, 1000, 1023, 5060, 70809]
        
        for n in numbers_with_zeros:
            result = await digit_product(n)
            assert result == 0, f"digit_product({n}) should be 0 when number contains zero"
    
    @pytest.mark.asyncio
    async def test_digit_product_different_bases(self):
        """Test digit product in different bases."""
        # 7 in binary is 111, product should be 1×1×1 = 1
        assert await digit_product(7, 2) == 1
        
        # 15 in binary is 1111, product should be 1×1×1×1 = 1
        assert await digit_product(15, 2) == 1
        
        # 255 in base 16 is FF, product should be 15×15 = 225
        assert await digit_product(255, 16) == 225
    
    @pytest.mark.asyncio
    async def test_persistent_digital_root_known_cases(self):
        """Test persistent digital root with known cases."""
        test_cases = [
            (7, 0),      # Already single digit
            (12, 1),     # 12 → 3
            (123, 1),    # 123 → 6
            (1234, 2),   # 1234 → 10 → 1 (2 steps)
            (12345, 2),  # 12345 → 15 → 6
            (999, 2),    # 999 → 27 → 9
            (9876, 2),   # 9876 → 30 → 3
            (88, 2),     # 88 → 16 → 7 (2 steps)
            (195, 2)     # 195 → 15 → 6
        ]
        
        for n, expected in test_cases:
            result = await persistent_digital_root(n)
            assert result == expected, f"persistent_digital_root({n}) should be {expected}, got {result}"
    
    @pytest.mark.asyncio
    async def test_persistent_digital_root_single_digits(self):
        """Test persistent digital root for single digits."""
        for i in range(10):
            result = await persistent_digital_root(i)
            assert result == 0, f"Single digit {i} should have persistent digital root 0"
    
    @pytest.mark.asyncio
    async def test_persistent_digital_root_consistency(self):
        """Test consistency between persistent digital root and digital root."""
        for n in [123, 456, 789, 1234, 9876]:
            persistent_steps = await persistent_digital_root(n)
            
            # Apply digit sum repeatedly and verify steps
            temp = n
            steps = 0
            while temp >= 10:
                temp = await digit_sum(temp)
                steps += 1
            
            assert steps == persistent_steps, f"Persistent digital root steps mismatch for {n}"
            
            # Final result should match digital root
            final_root = await digital_root(n)
            assert temp == final_root, f"Final digital root mismatch for {n}"

# ============================================================================
# DIGITAL TRANSFORMATIONS TESTS
# ============================================================================

class TestDigitalTransformations:
    """Test cases for digital transformation functions."""
    
    @pytest.mark.asyncio
    async def test_digit_reversal_basic_cases(self):
        """Test digit reversal with basic cases."""
        test_cases = [
            (0, 0),
            (1, 1),
            (12, 21),
            (123, 321),
            (1234, 4321),
            (12345, 54321),
            (1000, 1),     # Trailing zeros become leading zeros (dropped)
            (1200, 21),    # Multiple trailing zeros
            (54321, 12345)
        ]
        
        for n, expected in test_cases:
            result = await digit_reversal(n)
            assert result == expected, f"digit_reversal({n}) should be {expected}, got {result}"
    
    @pytest.mark.asyncio
    async def test_digit_reversal_palindromes(self):
        """Test digit reversal with palindromic numbers."""
        palindromes = [1, 11, 121, 1221, 12321, 123321]
        
        for palindrome in palindromes:
            result = await digit_reversal(palindrome)
            assert result == palindrome, f"Palindrome {palindrome} should remain unchanged when reversed"
    
    @pytest.mark.asyncio
    async def test_digit_reversal_negative_numbers(self):
        """Test digit reversal with negative numbers."""
        assert await digit_reversal(-123) == 321
        assert await digit_reversal(-1000) == 1
        assert await digit_reversal(-54321) == 12345
    
    @pytest.mark.asyncio
    async def test_digit_reversal_involution_property(self):
        """Test that reversing twice gives original number (for numbers without trailing zeros)."""
        test_numbers = [1, 12, 123, 1234, 12345, 54321, 9876, 111, 2468]
        
        for n in test_numbers:
            reversed_once = await digit_reversal(n)
            reversed_twice = await digit_reversal(reversed_once)
            assert reversed_twice == n, f"Double reversal of {n} should give original number"
    
    @pytest.mark.asyncio
    async def test_digit_sort_ascending(self):
        """Test digit sorting in ascending order."""
        test_cases = [
            (0, 0),
            (5, 5),
            (54321, 12345),
            (987654321, 123456789),
            (1111, 1111),
            (1032, 123),   # Note: leading zero is dropped
            (5432, 2345),
            (9876, 6789)
        ]
        
        for n, expected in test_cases:
            result = await digit_sort(n, descending=False)
            assert result == expected, f"digit_sort({n}, ascending) should be {expected}, got {result}"
    
    @pytest.mark.asyncio
    async def test_digit_sort_descending(self):
        """Test digit sorting in descending order."""
        test_cases = [
            (0, 0),
            (5, 5),
            (12345, 54321),
            (123456789, 987654321),
            (1111, 1111),
            (1032, 3210),
            (2345, 5432),
            (6789, 9876)
        ]
        
        for n, expected in test_cases:
            result = await digit_sort(n, descending=True)
            assert result == expected, f"digit_sort({n}, descending) should be {expected}, got {result}"
    
    @pytest.mark.asyncio
    async def test_digit_sort_with_repeated_digits(self):
        """Test digit sorting with repeated digits."""
        test_cases = [
            (112233, 123123),  # Ascending: 112233
            (332211, 123123),  # Ascending: 112233
            (112233, 332211),  # Descending: 332211
            (555, 555),        # All same digits
            (100200, 1200),    # With zeros (ascending: 000012 → 12)
            (100200, 210000)   # With zeros (descending: 210000)
        ]
        
        # Test ascending
        assert await digit_sort(112233, descending=False) == 112233
        assert await digit_sort(332211, descending=False) == 112233
        assert await digit_sort(555, descending=False) == 555
        assert await digit_sort(100200, descending=False) == 12
        
        # Test descending
        assert await digit_sort(112233, descending=True) == 332211
        assert await digit_sort(332211, descending=True) == 332211
        assert await digit_sort(555, descending=True) == 555
        assert await digit_sort(100200, descending=True) == 210000
    
    @pytest.mark.asyncio
    async def test_digit_sort_negative_numbers(self):
        """Test digit sorting with negative numbers."""
        assert await digit_sort(-54321, descending=False) == 12345
        assert await digit_sort(-54321, descending=True) == 54321
        assert await digit_sort(-1000, descending=False) == 1
        assert await digit_sort(-1000, descending=True) == 1000

# ============================================================================
# PALINDROMIC NUMBERS TESTS
# ============================================================================

class TestPalindromicNumbers:
    """Test cases for palindromic number functions."""
    
    @pytest.mark.asyncio
    async def test_is_palindromic_number_known_palindromes(self):
        """Test with known palindromic numbers."""
        palindromes_base_10 = [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9,  # Single digits
            11, 22, 33, 44, 55, 66, 77, 88, 99,  # Two-digit palindromes
            101, 111, 121, 131, 141, 151, 161, 171, 181, 191,  # Three-digit
            1001, 1111, 1221, 1331, 1441, 1551,  # Four-digit
            12321, 12421, 12521, 12621  # Five-digit
        ]
        
        for palindrome in palindromes_base_10:
            assert await is_palindromic_number(palindrome) == True, f"{palindrome} should be palindromic"
    
    @pytest.mark.asyncio
    async def test_is_palindromic_number_non_palindromes(self):
        """Test with non-palindromic numbers."""
        non_palindromes = [
            10, 12, 13, 14, 15, 16, 17, 18, 19,  # Two-digit non-palindromes
            100, 102, 103, 110, 112, 120, 123,   # Three-digit non-palindromes
            1000, 1010, 1020, 1100, 1200, 1230,  # Four-digit non-palindromes
            12345, 12346, 12347, 54321, 98765     # Multi-digit non-palindromes
        ]
        
        for non_palindrome in non_palindromes:
            assert await is_palindromic_number(non_palindrome) == False, f"{non_palindrome} should not be palindromic"
    
    @pytest.mark.asyncio
    async def test_is_palindromic_number_different_bases(self):
        """Test palindromic numbers in different bases."""
        # 9 in binary is 1001, which is palindromic
        assert await is_palindromic_number(9, 2) == True
        
        # 15 in binary is 1111, which is palindromic
        assert await is_palindromic_number(15, 2) == True
        
        # 10 in binary is 1010, which is not palindromic
        assert await is_palindromic_number(10, 2) == False
        
        # 17 in base 8 is 21, which is not palindromic
        assert await is_palindromic_number(17, 8) == False
        
        # 85 in base 16 is 55, which is palindromic
        assert await is_palindromic_number(85, 16) == True
    
    @pytest.mark.asyncio
    async def test_is_palindromic_number_negative(self):
        """Test palindromic check with negative numbers."""
        assert await is_palindromic_number(-1) == False
        assert await is_palindromic_number(-121) == False
        assert await is_palindromic_number(-1221) == False
    
    @pytest.mark.asyncio
    async def test_is_palindromic_number_invalid_base(self):
        """Test palindromic check with invalid base."""
        with pytest.raises(ValueError, match="Base must be at least 2"):
            await is_palindromic_number(123, 1)
    
    @pytest.mark.asyncio
    async def test_palindromic_numbers_generation(self):
        """Test generation of palindromic numbers."""
        palindromes_50 = await palindromic_numbers(50)
        expected_50 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 22, 33, 44]
        assert palindromes_50 == expected_50
        
        palindromes_200 = await palindromic_numbers(200)
        # Should include all single digits, double digit palindromes, and three-digit palindromes up to 191
        assert 0 in palindromes_200
        assert 11 in palindromes_200
        assert 121 in palindromes_200
        assert 191 in palindromes_200
        assert 200 not in palindromes_200  # 200 is not palindromic
    
    @pytest.mark.asyncio
    async def test_palindromic_numbers_edge_cases(self):
        """Test palindromic numbers generation edge cases."""
        assert await palindromic_numbers(0) == [0]
        assert await palindromic_numbers(-1) == []
        assert await palindromic_numbers(5) == [0, 1, 2, 3, 4, 5]
    
    @pytest.mark.asyncio
    async def test_palindromic_numbers_different_bases(self):
        """Test palindromic numbers in different bases."""
        # Binary palindromes up to 15
        binary_palindromes_15 = await palindromic_numbers(15, 2)
        expected_binary = [0, 1, 3, 5, 7, 9, 15]  # 0, 1, 11, 101, 111, 1001, 1111 in binary
        assert binary_palindromes_15 == expected_binary
    
    @pytest.mark.asyncio
    async def test_next_palindrome_basic_cases(self):
        """Test next palindrome function."""
        test_cases = [
            (0, 1),
            (1, 2),
            (9, 11),
            (11, 22),
            (22, 33),
            (99, 101),
            (101, 111),
            (123, 131),
            (191, 202),
            (999, 1001),
            (1001, 1111)
        ]
        
        for n, expected in test_cases:
            result = await next_palindrome(n)
            assert result == expected, f"next_palindrome({n}) should be {expected}, got {result}"
    
    @pytest.mark.asyncio
    async def test_next_palindrome_invalid_base(self):
        """Test next palindrome with invalid base."""
        with pytest.raises(ValueError, match="Base must be at least 2"):
            await next_palindrome(123, 1)
    
    @pytest.mark.asyncio
    async def test_next_palindrome_property(self):
        """Test that next_palindrome always returns a palindrome."""
        test_numbers = [0, 5, 12, 25, 99, 123, 456, 789, 1000]
        
        for n in test_numbers:
            next_pal = await next_palindrome(n)
            assert next_pal > n, f"next_palindrome({n}) should be greater than {n}"
            assert await is_palindromic_number(next_pal) == True, f"next_palindrome({n}) = {next_pal} should be palindromic"

# ============================================================================
# HARSHAD NUMBERS TESTS
# ============================================================================

class TestHarshadNumbers:
    """Test cases for Harshad number functions."""
    
    @pytest.mark.asyncio
    async def test_is_harshad_number_known_harshad(self):
        """Test with known Harshad numbers."""
        known_harshad = [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  # Single digits and 10
            12, 18, 20, 21, 24, 27, 30,      # Two-digit Harshad numbers
            36, 40, 42, 45, 48, 50, 54,      # More two-digit
            60, 63, 70, 72, 80, 81, 84,      # Even more two-digit
            90, 100, 102, 108, 110, 111      # Three-digit Harshad numbers
        ]
        
        for harshad in known_harshad:
            assert await is_harshad_number(harshad) == True, f"{harshad} should be Harshad"
    
    @pytest.mark.asyncio
    async def test_is_harshad_number_non_harshad(self):
        """Test with non-Harshad numbers."""
        non_harshad = [
            11, 13, 14, 15, 16, 17, 19,      # Two-digit non-Harshad
            22, 23, 25, 26, 28, 29, 31,      # More two-digit non-Harshad
            32, 34, 35, 37, 38, 39, 41,      # Even more two-digit
            43, 44, 46, 47, 49, 101, 103     # Mixed non-Harshad
        ]
        
        for non_harshad in non_harshad:
            assert await is_harshad_number(non_harshad) == False, f"{non_harshad} should not be Harshad"
    
    @pytest.mark.asyncio
    async def test_is_harshad_number_edge_cases(self):
        """Test Harshad number edge cases."""
        assert await is_harshad_number(0) == False   # 0 is not positive
        assert await is_harshad_number(-12) == False # Negative numbers
        assert await is_harshad_number(-1) == False  # Negative numbers
    
    @pytest.mark.asyncio
    async def test_is_harshad_number_manual_verification(self):
        """Test Harshad numbers with manual verification."""
        # 12: digit sum = 1+2 = 3, 12 % 3 = 0 ✓
        assert await is_harshad_number(12) == True
        
        # 18: digit sum = 1+8 = 9, 18 % 9 = 0 ✓
        assert await is_harshad_number(18) == True
        
        # 19: digit sum = 1+9 = 10, 19 % 10 = 9 ≠ 0 ✗
        assert await is_harshad_number(19) == False
        
        # 102: digit sum = 1+0+2 = 3, 102 % 3 = 0 ✓
        assert await is_harshad_number(102) == True
        
        # 103: digit sum = 1+0+3 = 4, 103 % 4 = 3 ≠ 0 ✗
        assert await is_harshad_number(103) == False
    
    @pytest.mark.asyncio
    async def test_is_harshad_number_different_bases(self):
        """Test Harshad numbers in different bases."""
        # In base 2: 6 = 110₂, digit sum = 1+1+0 = 2, 6 % 2 = 0 ✓
        assert await is_harshad_number(6, 2) == True
        
        # In base 2: 7 = 111₂, digit sum = 1+1+1 = 3, 7 % 3 = 1 ≠ 0 ✗
        assert await is_harshad_number(7, 2) == False
        
        # In base 16: 255 = FF₁₆, digit sum = 15+15 = 30, 255 % 30 = 15 ≠ 0 ✗
        assert await is_harshad_number(255, 16) == False
        
        # In base 16: 240 = F0₁₆, digit sum = 15+0 = 15, 240 % 15 = 0 ✓
        assert await is_harshad_number(240, 16) == True
    
    @pytest.mark.asyncio
    async def test_harshad_numbers_generation(self):
        """Test generation of Harshad numbers."""
        harshad_50 = await harshad_numbers(50)
        expected_50 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 18, 20, 21, 24, 27, 30, 36, 40, 42, 45, 48, 50]
        assert harshad_50 == expected_50
        
        harshad_100 = await harshad_numbers(100)
        # Should include all from 50 plus additional ones
        for h in harshad_50:
            assert h in harshad_100
        
        # Check some additional ones
        assert 54 in harshad_100
        assert 60 in harshad_100
        assert 63 in harshad_100
        assert 70 in harshad_100
        assert 72 in harshad_100
        assert 80 in harshad_100
        assert 81 in harshad_100
        assert 84 in harshad_100
        assert 90 in harshad_100
        assert 100 in harshad_100
    
    @pytest.mark.asyncio
    async def test_harshad_numbers_edge_cases(self):
        """Test Harshad numbers generation edge cases."""
        assert await harshad_numbers(0) == []
        assert await harshad_numbers(-1) == []
        assert await harshad_numbers(1) == [1]
        assert await harshad_numbers(5) == [1, 2, 3, 4, 5]

# ============================================================================
# BASE CONVERSION TESTS
# ============================================================================

class TestBaseConversions:
    """Test cases for base conversion functions."""
    
    @pytest.mark.asyncio
    async def test_number_to_base_binary(self):
        """Test conversion to binary."""
        test_cases = [
            (0, "0"),
            (1, "1"),
            (2, "10"),
            (3, "11"),
            (4, "100"),
            (7, "111"),
            (8, "1000"),
            (15, "1111"),
            (16, "10000"),
            (255, "11111111"),
            (1024, "10000000000")
        ]
        
        for n, expected in test_cases:
            result = await number_to_base(n, 2)
            assert result == expected, f"number_to_base({n}, 2) should be {expected}, got {result}"
    
    @pytest.mark.asyncio
    async def test_number_to_base_octal(self):
        """Test conversion to octal."""
        test_cases = [
            (0, "0"),
            (1, "1"),
            (7, "7"),
            (8, "10"),
            (64, "100"),
            (255, "377"),
            (512, "1000"),
            (1729, "3301")
        ]
        
        for n, expected in test_cases:
            result = await number_to_base(n, 8)
            assert result == expected, f"number_to_base({n}, 8) should be {expected}, got {result}"
    
    @pytest.mark.asyncio
    async def test_number_to_base_hexadecimal(self):
        """Test conversion to hexadecimal."""
        test_cases = [
            (0, "0"),
            (1, "1"),
            (9, "9"),
            (10, "A"),
            (15, "F"),
            (16, "10"),
            (255, "FF"),
            (256, "100"),
            (4095, "FFF"),
            (65535, "FFFF")
        ]
        
        for n, expected in test_cases:
            result = await number_to_base(n, 16)
            assert result == expected, f"number_to_base({n}, 16) should be {expected}, got {result}"
    
    @pytest.mark.asyncio
    async def test_number_to_base_other_bases(self):
        """Test conversion to other bases."""
        # Base 3
        assert await number_to_base(0, 3) == "0"
        assert await number_to_base(1, 3) == "1"
        assert await number_to_base(2, 3) == "2"
        assert await number_to_base(3, 3) == "10"
        assert await number_to_base(9, 3) == "100"
        assert await number_to_base(27, 3) == "1000"
        
        # Base 36 (maximum supported)
        assert await number_to_base(35, 36) == "Z"
        assert await number_to_base(36, 36) == "10"
    
    @pytest.mark.asyncio
    async def test_number_to_base_edge_cases(self):
        """Test number to base edge cases and errors."""
        # Invalid base
        with pytest.raises(ValueError, match="Base must be between 2 and 36"):
            await number_to_base(10, 1)
        
        with pytest.raises(ValueError, match="Base must be between 2 and 36"):
            await number_to_base(10, 37)
        
        # Negative number
        with pytest.raises(ValueError, match="Number must be non-negative"):
            await number_to_base(-1, 10)
    
    @pytest.mark.asyncio
    async def test_base_to_number_binary(self):
        """Test conversion from binary."""
        test_cases = [
            ("0", 0),
            ("1", 1),
            ("10", 2),
            ("11", 3),
            ("100", 4),
            ("111", 7),
            ("1000", 8),
            ("1111", 15),
            ("10000", 16),
            ("11111111", 255),
            ("10000000000", 1024)
        ]
        
        for digits, expected in test_cases:
            result = await base_to_number(digits, 2)
            assert result == expected, f"base_to_number('{digits}', 2) should be {expected}, got {result}"
    
    @pytest.mark.asyncio
    async def test_base_to_number_hexadecimal(self):
        """Test conversion from hexadecimal."""
        test_cases = [
            ("0", 0),
            ("1", 1),
            ("9", 9),
            ("A", 10),
            ("F", 15),
            ("10", 16),
            ("FF", 255),
            ("100", 256),
            ("FFF", 4095),
            ("FFFF", 65535)
        ]
        
        for digits, expected in test_cases:
            result = await base_to_number(digits, 16)
            assert result == expected, f"base_to_number('{digits}', 16) should be {expected}, got {result}"
    
    @pytest.mark.asyncio
    async def test_base_to_number_case_insensitive(self):
        """Test that base conversion is case insensitive."""
        assert await base_to_number("ff", 16) == 255
        assert await base_to_number("FF", 16) == 255
        assert await base_to_number("Ff", 16) == 255
        assert await base_to_number("fF", 16) == 255
    
    @pytest.mark.asyncio
    async def test_base_to_number_edge_cases(self):
        """Test base to number edge cases and errors."""
        # Empty string
        assert await base_to_number("", 10) == 0
        
        # Invalid base
        with pytest.raises(ValueError, match="Base must be between 2 and 36"):
            await base_to_number("10", 1)
        
        with pytest.raises(ValueError, match="Base must be between 2 and 36"):
            await base_to_number("10", 37)
        
        # Invalid digit for base - fix the regex pattern to match actual error message
        with pytest.raises(ValueError, match="Digit .* is not valid for base"):
            await base_to_number("2", 2)  # '2' is invalid in binary
        
        with pytest.raises(ValueError, match="Digit .* is not valid for base"):
            await base_to_number("G", 16)  # 'G' is invalid in hex
    
    @pytest.mark.asyncio
    async def test_base_conversion_round_trip(self):
        """Test that conversion to base and back gives original number."""
        test_numbers = [0, 1, 7, 15, 16, 42, 100, 255, 1000, 1729, 65535]
        test_bases = [2, 3, 8, 10, 16, 36]
        
        for n in test_numbers:
            for base in test_bases:
                base_repr = await number_to_base(n, base)
                converted_back = await base_to_number(base_repr, base)
                assert converted_back == n, f"Round trip failed for {n} in base {base}"

# ============================================================================
# DIGIT PROPERTIES TESTS
# ============================================================================

class TestDigitProperties:
    """Test cases for digit property functions."""
    
    @pytest.mark.asyncio
    async def test_digit_count_basic_cases(self):
        """Test digit count with basic cases."""
        test_cases = [
            (0, 1),
            (1, 1),
            (9, 1),
            (10, 2),
            (99, 2),
            (100, 3),
            (999, 3),
            (1000, 4),
            (12345, 5),
            (1000000, 7)
        ]
        
        for n, expected in test_cases:
            result = await digit_count(n)
            assert result == expected, f"digit_count({n}) should be {expected}, got {result}"
    
    @pytest.mark.asyncio
    async def test_digit_count_different_bases(self):
        """Test digit count in different bases."""
        # 255 in binary has 8 digits (11111111)
        assert await digit_count(255, 2) == 8
        
        # 255 in hex has 2 digits (FF)
        assert await digit_count(255, 16) == 2
        
        # 64 in binary has 7 digits (1000000)
        assert await digit_count(64, 2) == 7
        
        # 64 in octal has 3 digits (100)
        assert await digit_count(64, 8) == 3
    
    @pytest.mark.asyncio
    async def test_digit_count_negative_numbers(self):
        """Test digit count with negative numbers."""
        assert await digit_count(-123) == 3
        assert await digit_count(-1) == 1
        assert await digit_count(-1000) == 4
    
    @pytest.mark.asyncio
    async def test_digit_count_invalid_base(self):
        """Test digit count with invalid base."""
        with pytest.raises(ValueError, match="Base must be at least 2"):
            await digit_count(123, 1)
    
    @pytest.mark.asyncio
    async def test_digit_frequency_basic_cases(self):
        """Test digit frequency with basic cases."""
        test_cases = [
            (0, {0: 1}),
            (123, {1: 1, 2: 1, 3: 1}),
            (1122, {1: 2, 2: 2}),
            (112233, {1: 2, 2: 2, 3: 2}),
            (1000, {1: 1, 0: 3}),
            (555, {5: 3}),
            (123456789, {1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1})
        ]
        
        for n, expected in test_cases:
            result = await digit_frequency(n)
            assert result == expected, f"digit_frequency({n}) should be {expected}, got {result}"
    
    @pytest.mark.asyncio
    async def test_digit_frequency_different_bases(self):
        """Test digit frequency in different bases."""
        # 15 in binary is 1111, should have {1: 4}
        result_binary = await digit_frequency(15, 2)
        assert result_binary == {1: 4}
        
        # 255 in hex is FF, should have {15: 2}
        result_hex = await digit_frequency(255, 16)
        assert result_hex == {15: 2}
    
    @pytest.mark.asyncio
    async def test_digit_frequency_properties(self):
        """Test digit frequency properties."""
        for n in [123, 1234, 12345, 112233, 555]:
            freq = await digit_frequency(n)
            
            # Sum of frequencies should equal number of digits
            total_freq = sum(freq.values())
            expected_digit_count = await digit_count(n)
            assert total_freq == expected_digit_count, f"Frequency sum mismatch for {n}"
            
            # All frequencies should be positive
            for digit, count in freq.items():
                assert count > 0, f"Frequency for digit {digit} should be positive"
                assert 0 <= digit <= 9, f"Digit {digit} should be between 0 and 9"
    
    @pytest.mark.asyncio
    async def test_is_repdigit_basic_cases(self):
        """Test repdigit detection with basic cases."""
        # Single digits are repdigits
        for i in range(10):
            assert await is_repdigit(i) == True, f"Single digit {i} should be repdigit"
        
        # Known repdigits
        repdigits = [11, 22, 33, 44, 55, 66, 77, 88, 99, 111, 222, 333, 444, 555, 666, 777, 888, 999, 1111, 2222]
        for repdigit in repdigits:
            assert await is_repdigit(repdigit) == True, f"{repdigit} should be repdigit"
        
        # Non-repdigits
        non_repdigits = [10, 12, 21, 23, 100, 101, 110, 123, 1000, 1001, 1010, 1100, 1234]
        for non_repdigit in non_repdigits:
            assert await is_repdigit(non_repdigit) == False, f"{non_repdigit} should not be repdigit"
    
    @pytest.mark.asyncio
    async def test_is_repdigit_different_bases(self):
        """Test repdigit detection in different bases."""
        # 15 in binary is 1111, which is a repdigit
        assert await is_repdigit(15, 2) == True
        
        # 7 in binary is 111, which is a repdigit
        assert await is_repdigit(7, 2) == True
        
        # 9 in binary is 1001, which is not a repdigit
        assert await is_repdigit(9, 2) == False
        
        # 85 in hex is 55, which is a repdigit
        assert await is_repdigit(85, 16) == True
    
    @pytest.mark.asyncio
    async def test_is_repdigit_negative_numbers(self):
        """Test repdigit detection with negative numbers."""
        assert await is_repdigit(-11) == True   # Absolute value is repdigit
        assert await is_repdigit(-123) == False # Absolute value is not repdigit
        assert await is_repdigit(-5) == True    # Single digit absolute value

# ============================================================================
# SPECIAL NUMBERS TESTS
# ============================================================================

class TestSpecialNumbers:
    """Test cases for special number functions."""
    
    @pytest.mark.asyncio
    async def test_is_automorphic_number_known_automorphic(self):
        """Test with known automorphic numbers."""
        known_automorphic = [0, 1, 5, 6, 25, 76, 376, 625, 9376, 90625]
        
        for automorphic in known_automorphic:
            assert await is_automorphic_number(automorphic) == True, f"{automorphic} should be automorphic"
    
    @pytest.mark.asyncio
    async def test_is_automorphic_number_manual_verification(self):
        """Test automorphic numbers with manual verification."""
        # 25² = 625, ends with 25 ✓
        assert await is_automorphic_number(25) == True
        assert 25 * 25 == 625
        
        # 76² = 5776, ends with 76 ✓
        assert await is_automorphic_number(76) == True
        assert 76 * 76 == 5776
        
        # 625² = 390625, ends with 625 ✓
        assert await is_automorphic_number(625) == True
        assert 625 * 625 == 390625
        
        # 376² = 141376, ends with 376 ✓
        assert await is_automorphic_number(376) == True
        assert 376 * 376 == 141376
    
    @pytest.mark.asyncio
    async def test_is_automorphic_number_non_automorphic(self):
        """Test with non-automorphic numbers."""
        non_automorphic = [2, 3, 4, 7, 8, 9, 10, 11, 12, 15, 20, 23, 24, 26, 27, 30, 50, 75, 100]
        
        for non_automorphic in non_automorphic:
            assert await is_automorphic_number(non_automorphic) == False, f"{non_automorphic} should not be automorphic"
    
    @pytest.mark.asyncio
    async def test_is_automorphic_number_edge_cases(self):
        """Test automorphic number edge cases."""
        assert await is_automorphic_number(0) == True   # 0² = 0, ends with 0
        assert await is_automorphic_number(1) == True   # 1² = 1, ends with 1
        assert await is_automorphic_number(-1) == False # Negative numbers
        assert await is_automorphic_number(-25) == False # Negative numbers
    
    @pytest.mark.asyncio
    async def test_automorphic_numbers_generation(self):
        """Test generation of automorphic numbers."""
        automorphic_100 = await automorphic_numbers(100)
        expected_100 = [0, 1, 5, 6, 25, 76]
        assert automorphic_100 == expected_100
        
        automorphic_1000 = await automorphic_numbers(1000)
        expected_1000 = [0, 1, 5, 6, 25, 76, 376, 625]
        assert automorphic_1000 == expected_1000
    
    @pytest.mark.asyncio
    async def test_automorphic_numbers_edge_cases(self):
        """Test automorphic numbers generation edge cases."""
        assert await automorphic_numbers(0) == [0]
        assert await automorphic_numbers(-1) == []
        assert await automorphic_numbers(1) == [0, 1]
        assert await automorphic_numbers(5) == [0, 1, 5]

# ============================================================================
# INTEGRATION AND MATHEMATICAL PROPERTIES TESTS
# ============================================================================

class TestIntegrationAndProperties:
    """Integration tests and mathematical property verification."""
    
    @pytest.mark.asyncio
    async def test_digital_root_and_digit_sum_relationship(self):
        """Test relationship between digital root and digit sum."""
        test_numbers = [123, 456, 789, 1234, 9876, 12345]
        
        for n in test_numbers:
            digital_root_val = await digital_root(n)
            
            # Apply digit sum repeatedly until single digit
            temp = n
            while temp >= 10:
                temp = await digit_sum(temp)
            
            assert temp == digital_root_val, f"Digital root calculation mismatch for {n}"
    
    @pytest.mark.asyncio
    async def test_palindrome_and_reversal_relationship(self):
        """Test relationship between palindromes and digit reversal."""
        test_numbers = [121, 1221, 12321, 123321, 1234321]
        
        for palindrome in test_numbers:
            assert await is_palindromic_number(palindrome) == True, f"{palindrome} should be palindromic"
            
            reversed_num = await digit_reversal(palindrome)
            assert reversed_num == palindrome, f"Palindrome {palindrome} should equal its reversal"
    
    @pytest.mark.asyncio
    async def test_digit_sort_and_frequency_consistency(self):
        """Test consistency between digit sorting and frequency counting."""
        test_numbers = [12345, 54321, 112233, 999, 1000]
        
        for n in test_numbers:
            freq = await digit_frequency(n)
            sorted_asc = await digit_sort(n, descending=False)
            sorted_desc = await digit_sort(n, descending=True)
            
            # Count digits in sorted numbers
            freq_asc = await digit_frequency(sorted_asc)
            freq_desc = await digit_frequency(sorted_desc)
            
            # Frequencies should be preserved (except for leading zeros in ascending)
            assert freq_desc == freq, f"Frequency mismatch after descending sort for {n}"
    
    @pytest.mark.asyncio
    async def test_base_conversion_properties(self):
        """Test mathematical properties of base conversions."""
        test_numbers = [42, 255, 1729]
        
        for n in test_numbers:
            # Converting to different bases should preserve digit sum relationships
            base_10_digit_sum = await digit_sum(n, 10)
            base_2_digit_sum = await digit_sum(n, 2)
            base_16_digit_sum = await digit_sum(n, 16)
            
            # All should be positive
            assert base_10_digit_sum > 0
            assert base_2_digit_sum > 0
            assert base_16_digit_sum > 0
            
            # Binary digit sum equals number of 1 bits
            binary_repr = await number_to_base(n, 2)
            expected_binary_sum = binary_repr.count('1')
            assert base_2_digit_sum == expected_binary_sum, f"Binary digit sum mismatch for {n}"
    
    @pytest.mark.asyncio
    async def test_automorphic_and_digit_properties(self):
        """Test relationships between automorphic numbers and digit properties."""
        automorphic_nums = [1, 5, 6, 25, 76, 376, 625]
        
        for automorphic in automorphic_nums:
            assert await is_automorphic_number(automorphic) == True
            
            # Check that square actually ends with the number
            square = automorphic * automorphic
            square_str = str(square)
            automorphic_str = str(automorphic)
            
            assert square_str.endswith(automorphic_str), f"{automorphic}² = {square} should end with {automorphic}"
    
    @pytest.mark.asyncio
    async def test_harshad_and_digit_sum_relationship(self):
        """Test fundamental relationship for Harshad numbers."""
        harshad_nums = await harshad_numbers(100)
        
        for harshad in harshad_nums:
            digit_sum_val = await digit_sum(harshad)
            assert harshad % digit_sum_val == 0, f"Harshad number {harshad} should be divisible by its digit sum {digit_sum_val}"
            
            # Check that non-multiples are not Harshad
            if harshad < 100:  # Avoid going over our test range
                non_harshad_candidate = harshad + 1
                if non_harshad_candidate <= 100:
                    candidate_digit_sum = await digit_sum(non_harshad_candidate)
                    is_harshad_candidate = await is_harshad_number(non_harshad_candidate)
                    
                    if non_harshad_candidate % candidate_digit_sum != 0:
                        assert is_harshad_candidate == False, f"{non_harshad_candidate} should not be Harshad"

# ============================================================================
# PERFORMANCE AND ASYNC BEHAVIOR TESTS
# ============================================================================

class TestPerformance:
    """Performance and async behavior tests."""
    
    @pytest.mark.asyncio
    async def test_all_functions_are_async(self):
        """Test that all digital operations functions are properly async."""
        operations = [
            digit_sum(12345),
            digital_root(12345),
            digit_product(123),
            persistent_digital_root(12345),
            digit_reversal(12345),
            digit_sort(54321),
            is_palindromic_number(12321),
            next_palindrome(123),
            is_harshad_number(12),
            number_to_base(255, 2),
            base_to_number("FF", 16),
            digit_count(12345),
            digit_frequency(112233),
            is_repdigit(1111),
            is_automorphic_number(25),
            palindromic_numbers(50),
            harshad_numbers(50),
            automorphic_numbers(100)
        ]
        
        # Ensure all are coroutines
        for op in operations:
            assert asyncio.iscoroutine(op)
        
        # Run all operations concurrently
        results = await asyncio.gather(*operations)
        
        # Verify results have expected types and values
        assert results[0] == 15                    # digit_sum(12345)
        assert results[1] == 6                     # digital_root(12345)
        assert results[2] == 6                     # digit_product(123)
        assert results[3] == 2                     # persistent_digital_root(12345)
        assert results[4] == 54321                 # digit_reversal(12345)
        assert results[5] == 12345                 # digit_sort(54321)
        assert results[6] == True                  # is_palindromic_number(12321)
        assert results[7] == 131                   # next_palindrome(123)
        assert results[8] == True                  # is_harshad_number(12)
        assert results[9] == "11111111"            # number_to_base(255, 2)
        assert results[10] == 255                  # base_to_number("FF", 16)
        assert results[11] == 5                    # digit_count(12345)
        assert results[12] == {1: 2, 2: 2, 3: 2}  # digit_frequency(112233)
        assert results[13] == True                 # is_repdigit(1111)
        assert results[14] == True                 # is_automorphic_number(25)
        assert isinstance(results[15], list)       # palindromic_numbers(50)
        assert isinstance(results[16], list)       # harshad_numbers(50)
        assert isinstance(results[17], list)       # automorphic_numbers(100)
    
    @pytest.mark.asyncio
    async def test_concurrent_execution(self):
        """Test that digital operations can run concurrently."""
        start_time = time.time()
        
        # Run multiple operations concurrently
        tasks = []
        for i in range(1, 50):  # 49 numbers (1-49)
            tasks.append(digit_sum(i * 123))
            tasks.append(digital_root(i * 456))
            tasks.append(is_palindromic_number(i * 11))
            tasks.append(is_harshad_number(i))
            tasks.append(digit_count(i * 1000))
        
        results = await asyncio.gather(*tasks)
        duration = time.time() - start_time
        
        # Should complete quickly due to async nature
        assert duration < 2.0
        assert len(results) == 49 * 5  # 49 numbers × 5 operations each
        
        # Check some patterns in results
        digit_sum_results = results[::5]  # Every 5th result
        for result in digit_sum_results:
            assert isinstance(result, int) and result > 0
    
    @pytest.mark.asyncio
    async def test_large_number_handling(self):
        """Test handling of moderately large numbers."""
        large_tests = [
            digit_sum(123456789),
            digital_root(987654321),
            digit_reversal(1234567890),
            digit_count(10**10),
            is_palindromic_number(1234554321),
            number_to_base(2**20, 2),
            palindromic_numbers(500),
            harshad_numbers(500),
        ]
        
        results = await asyncio.gather(*large_tests)
        
        # Verify results are reasonable
        assert results[0] == 45                    # digit_sum(123456789)
        assert results[1] == 9                     # digital_root(987654321)
        assert results[2] == 987654321             # digit_reversal(1234567890)
        assert results[3] == 11                    # digit_count(10^10)
        assert results[4] == True                  # is_palindromic_number(1234554321)
        assert len(results[5]) == 21               # binary representation of 2^20
        assert isinstance(results[6], list)        # palindromic_numbers(500)
        assert isinstance(results[7], list)        # harshad_numbers(500)
        
        # Check that sequences have reasonable lengths
        assert len(results[6]) > 20                # Should have many palindromes ≤ 500
        assert len(results[7]) > 50                # Should have many Harshad numbers ≤ 500
    
    @pytest.mark.asyncio
    async def test_memory_efficiency(self):
        """Test that functions don't consume excessive memory."""
        # Generate several sequences and verify they complete
        sequences = await asyncio.gather(
            palindromic_numbers(200),
            harshad_numbers(200),
            automorphic_numbers(1000),
            # Test digit operations on various numbers
            asyncio.gather(*[digit_frequency(i * 12345) for i in range(1, 21)]),
            asyncio.gather(*[digit_sum(i * 98765) for i in range(1, 31)])
        )
        
        # Verify sequences have expected properties
        palindromes, harshads, automorphics, frequencies, digit_sums = sequences
        
        assert len(palindromes) > 10
        assert len(harshads) > 20
        assert len(automorphics) >= 8  # Should have at least [0, 1, 5, 6, 25, 76, 376, 625]
        assert len(frequencies) == 20
        assert len(digit_sums) == 30
        
        # Check that all palindromes are actually palindromic
        for palindrome in palindromes[:10]:  # Check first 10
            assert await is_palindromic_number(palindrome)
        
        # Check that all Harshad numbers are actually Harshad
        for harshad in harshads[:10]:  # Check first 10
            assert await is_harshad_number(harshad)

# ============================================================================
# ERROR HANDLING TESTS
# ============================================================================

class TestErrorHandling:
    """Test error handling and edge cases."""
    
    @pytest.mark.asyncio
    async def test_base_validation_errors(self):
        """Test that appropriate errors are raised for invalid bases."""
        base_functions = [
            (digit_sum, (123,), {}),
            (digital_root, (123,), {}),
            (digit_product, (123,), {}),
            (persistent_digital_root, (123,), {}),
            (is_palindromic_number, (123,), {}),
            (is_harshad_number, (123,), {}),
            (digit_count, (123,), {}),
            (digit_frequency, (123,), {}),
            (is_repdigit, (123,), {})
        ]
        
        for func, args, kwargs in base_functions:
            # Test base < 2
            with pytest.raises(ValueError, match="Base must be at least 2"):
                await func(*args, **kwargs, base=1)
            
            with pytest.raises(ValueError, match="Base must be at least 2"):
                await func(*args, **kwargs, base=0)
    
    @pytest.mark.asyncio
    async def test_base_conversion_errors(self):
        """Test base conversion specific errors."""
        # number_to_base errors
        with pytest.raises(ValueError, match="Number must be non-negative"):
            await number_to_base(-1, 10)
        
        with pytest.raises(ValueError, match="Base must be between 2 and 36"):
            await number_to_base(10, 1)
        
        with pytest.raises(ValueError, match="Base must be between 2 and 36"):
            await number_to_base(10, 37)
        
        # base_to_number errors
        with pytest.raises(ValueError, match="Base must be between 2 and 36"):
            await base_to_number("10", 1)
        
        # Fix the regex pattern to match the actual error message format
        with pytest.raises(ValueError, match="Digit .* is not valid for base"):
            await base_to_number("2", 2)  # '2' invalid in binary
        
        with pytest.raises(ValueError, match="Digit .* is not valid for base"):
            await base_to_number("G", 16)  # 'G' invalid in hex
    
    @pytest.mark.asyncio
    async def test_edge_case_handling(self):
        """Test edge case handling across all functions."""
        # All functions should handle zero gracefully
        edge_cases = [0]
        
        for n in edge_cases:
            # These should not raise exceptions
            await digit_sum(n)
            await digital_root(n)
            await digit_product(n)
            await persistent_digital_root(n)
            await digit_reversal(n)
            await digit_sort(n)
            await is_palindromic_number(n)
            await digit_count(n)
            await digit_frequency(n)
            await is_repdigit(n)
            await is_automorphic_number(n)
            
            # Sequence generation with small limits
            await palindromic_numbers(n + 1)
            await harshad_numbers(max(1, n))  # Harshad needs positive limit
            await automorphic_numbers(n + 1)
    
    @pytest.mark.asyncio
    async def test_error_preserves_async_context(self):
        """Test that errors are properly raised in async context."""
        try:
            await number_to_base(-1, 10)  # Should raise ValueError
            assert False, "Should have raised ValueError"
        except ValueError:
            # Should be able to continue with async operations
            result = await digit_sum(12345)
            assert result == 15
        
        try:
            await base_to_number("G", 16)  # Should raise ValueError
            assert False, "Should have raised ValueError"
        except ValueError:
            # Should be able to continue with async operations
            result = await is_palindromic_number(12321)
            assert result == True

# ============================================================================
# PARAMETRIZED TESTS
# ============================================================================

class TestParametrized:
    """Parametrized tests for comprehensive coverage."""
    
    @pytest.mark.asyncio
    @pytest.mark.parametrize("n,expected", [
        (0, 0), (9, 9), (12, 3), (123, 6), (999, 27), (12345, 15), (9876, 30)
    ])
    async def test_digit_sum_parametrized(self, n, expected):
        """Parametrized test for digit sum calculation."""
        assert await digit_sum(n) == expected
    
    @pytest.mark.asyncio
    @pytest.mark.parametrize("n,expected", [
        (0, 0), (9, 9), (12, 3), (123, 6), (999, 9), (12345, 6), (9876, 3)
    ])
    async def test_digital_root_parametrized(self, n, expected):
        """Parametrized test for digital root calculation."""
        assert await digital_root(n) == expected
    
    @pytest.mark.asyncio
    @pytest.mark.parametrize("n,expected", [
        (0, 0), (1, 1), (123, 6), (999, 729), (1023, 0), (456, 120)
    ])
    async def test_digit_product_parametrized(self, n, expected):
        """Parametrized test for digit product calculation."""
        assert await digit_product(n) == expected
    
    @pytest.mark.asyncio
    @pytest.mark.parametrize("n,expected", [
        (0, 0), (1, 1), (12, 21), (123, 321), (1000, 1), (54321, 12345)
    ])
    async def test_digit_reversal_parametrized(self, n, expected):
        """Parametrized test for digit reversal."""
        assert await digit_reversal(n) == expected
    
    @pytest.mark.asyncio
    @pytest.mark.parametrize("palindrome", [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 22, 33, 44, 55, 66, 77, 88, 99,
        101, 111, 121, 131, 141, 151, 161, 171, 181, 191, 202, 212, 222
    ])
    async def test_is_palindromic_number_parametrized(self, palindrome):
        """Parametrized test for palindrome identification."""
        assert await is_palindromic_number(palindrome) == True
    
    @pytest.mark.asyncio
    @pytest.mark.parametrize("harshad", [
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 18, 20, 21, 24, 27, 30, 36, 40, 42, 45, 48, 50
    ])
    async def test_is_harshad_number_parametrized(self, harshad):
        """Parametrized test for Harshad number identification."""
        assert await is_harshad_number(harshad) == True
    
    @pytest.mark.asyncio
    @pytest.mark.parametrize("n,base,expected", [
        (0, 2, "0"), (1, 2, "1"), (2, 2, "10"), (7, 2, "111"), (15, 2, "1111"),
        (255, 2, "11111111"), (0, 16, "0"), (10, 16, "A"), (255, 16, "FF")
    ])
    async def test_number_to_base_parametrized(self, n, base, expected):
        """Parametrized test for number to base conversion."""
        assert await number_to_base(n, base) == expected
    
    @pytest.mark.asyncio
    @pytest.mark.parametrize("digits,base,expected", [
        ("0", 2, 0), ("1", 2, 1), ("10", 2, 2), ("111", 2, 7), ("1111", 2, 15),
        ("11111111", 2, 255), ("0", 16, 0), ("A", 16, 10), ("FF", 16, 255)
    ])
    async def test_base_to_number_parametrized(self, digits, base, expected):
        """Parametrized test for base to number conversion."""
        assert await base_to_number(digits, base) == expected
    
    @pytest.mark.asyncio
    @pytest.mark.parametrize("repdigit", [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 22, 33, 44, 55, 66, 77, 88, 99,
        111, 222, 333, 444, 555, 666, 777, 888, 999, 1111, 2222, 3333
    ])
    async def test_is_repdigit_parametrized(self, repdigit):
        """Parametrized test for repdigit identification."""
        assert await is_repdigit(repdigit) == True
    
    @pytest.mark.asyncio
    @pytest.mark.parametrize("automorphic", [0, 1, 5, 6, 25, 76, 376, 625])
    async def test_is_automorphic_number_parametrized(self, automorphic):
        """Parametrized test for automorphic number identification."""
        assert await is_automorphic_number(automorphic) == True

# ============================================================================
# COMPREHENSIVE INTEGRATION TESTS
# ============================================================================

class TestComprehensiveIntegration:
    """Comprehensive integration tests combining multiple functions."""
    
    @pytest.mark.asyncio
    async def test_digit_operations_workflow(self):
        """Test a complete workflow using multiple digit operations."""
        n = 12345
        
        # Basic digit operations
        ds = await digit_sum(n)
        dr = await digital_root(n)
        dp = await digit_product(n)
        pdr = await persistent_digital_root(n)
        
        # Transformations
        reversed_n = await digit_reversal(n)
        sorted_asc = await digit_sort(n, descending=False)
        sorted_desc = await digit_sort(n, descending=True)
        
        # Properties
        is_palindrome = await is_palindromic_number(n)
        is_harshad = await is_harshad_number(n)
        is_automorphic = await is_automorphic_number(n)
        
        # Verify expected results
        assert ds == 15
        assert dr == 6
        assert dp == 120
        assert pdr == 2
        assert reversed_n == 54321
        assert sorted_asc == 12345
        assert sorted_desc == 54321
        assert is_palindrome == False
        assert is_harshad == True  # 12345 % 15 = 0
        assert is_automorphic == False
        
        # Test relationships
        assert dr == 1 + (ds - 1) % 9  # Digital root formula
        assert reversed_n == sorted_desc  # For this specific number
    
    @pytest.mark.asyncio
    async def test_base_conversion_workflow(self):
        """Test complete base conversion workflows."""
        n = 255
        
        # Convert to various bases
        binary = await number_to_base(n, 2)
        octal = await number_to_base(n, 8)
        hex_repr = await number_to_base(n, 16)
        
        # Convert back to decimal
        from_binary = await base_to_number(binary, 2)
        from_octal = await base_to_number(octal, 8)
        from_hex = await base_to_number(hex_repr, 16)
        
        # Verify round trips
        assert from_binary == n
        assert from_octal == n
        assert from_hex == n
        
        # Verify expected representations
        assert binary == "11111111"
        assert octal == "377"
        assert hex_repr == "FF"
        
        # Test digit sums in different bases
        binary_digit_sum = await digit_sum(n, 2)
        hex_digit_sum = await digit_sum(n, 16)
        
        assert binary_digit_sum == 8  # Eight 1s in binary
        assert hex_digit_sum == 30    # F + F = 15 + 15
    
    @pytest.mark.asyncio
    async def test_palindrome_generation_and_verification(self):
        """Test palindrome generation and verification workflow."""
        limit = 200
        
        # Generate palindromes
        palindromes = await palindromic_numbers(limit)
        
        # Verify all generated numbers are palindromic
        for palindrome in palindromes:
            assert await is_palindromic_number(palindrome) == True
            assert palindrome <= limit
        
        # Test next palindrome function
        for i in range(0, min(20, len(palindromes) - 1)):
            current_palindrome = palindromes[i]
            next_palindrome_val = await next_palindrome(current_palindrome)
            
            # Next palindrome should be greater
            assert next_palindrome_val > current_palindrome
            
            # Should be palindromic
            assert await is_palindromic_number(next_palindrome_val) == True
            
            # Should be the next one in our list (if within limit)
            if next_palindrome_val <= limit:
                next_index = palindromes.index(next_palindrome_val)
                assert next_index == i + 1
    
    @pytest.mark.asyncio
    async def test_special_number_relationships(self):
        """Test relationships between different types of special numbers."""
        limit = 1000
        
        # Generate different types of special numbers
        harshad_nums = await harshad_numbers(limit)
        automorphic_nums = await automorphic_numbers(limit)
        palindromes = await palindromic_numbers(limit)
        
        # Find intersections
        harshad_palindromes = [n for n in harshad_nums if n in palindromes]
        automorphic_palindromes = [n for n in automorphic_nums if n in palindromes]
        harshad_automorphic = [n for n in harshad_nums if n in automorphic_nums]
        
        # Verify intersections have both properties
        for n in harshad_palindromes:
            assert await is_harshad_number(n) == True
            assert await is_palindromic_number(n) == True
        
        for n in automorphic_palindromes:
            assert await is_automorphic_number(n) == True
            assert await is_palindromic_number(n) == True
        
        for n in harshad_automorphic:
            assert await is_harshad_number(n) == True
            assert await is_automorphic_number(n) == True
        
        # Some expected intersections
        assert 1 in harshad_palindromes  # 1 is both Harshad and palindromic
        assert 1 in automorphic_palindromes  # 1 is both automorphic and palindromic
        assert 1 in harshad_automorphic  # 1 is both Harshad and automorphic
    
    @pytest.mark.asyncio
    async def test_digit_transformation_chains(self):
        """Test chains of digit transformations."""
        start_numbers = [12345, 54321, 98765, 13579]
        
        for n in start_numbers:
            # Apply transformation chain
            step1 = await digit_reversal(n)
            step2 = await digit_sort(step1, descending=True)
            step3 = await digit_reversal(step2)
            step4 = await digit_sort(step3, descending=False)
            
            # Verify transformations preserve digit frequencies
            original_freq = await digit_frequency(n)
            final_freq = await digit_frequency(step4)
            
            # Frequencies should be preserved (digit transformations don't add/remove digits)
            assert original_freq == final_freq, f"Digit frequency not preserved for {n}"
            
            # All intermediate results should have same digit count
            original_count = await digit_count(n)
            assert await digit_count(step1) <= original_count  # Reversal might remove leading zeros
            assert await digit_count(step2) <= original_count  # Sorting might remove leading zeros
            assert await digit_count(step3) <= original_count  # Reversal might remove leading zeros
            assert await digit_count(step4) <= original_count  # Sorting might remove leading zeros
    
    @pytest.mark.asyncio
    async def test_mathematical_property_verification(self):
        """Verify mathematical properties across functions."""
        test_numbers = [123, 456, 789, 1234, 5678, 9999]
        
        for n in test_numbers:
            # Digital root properties
            dr = await digital_root(n)
            assert 0 <= dr <= 9, f"Digital root {dr} should be single digit"
            
            # Digit sum vs digital root relationship
            ds = await digit_sum(n)
            if n > 0:
                expected_dr = 1 + (n - 1) % 9
                assert dr == expected_dr, f"Digital root formula failed for {n}"
            
            # Persistence properties
            persistence = await persistent_digital_root(n)
            assert persistence >= 0, f"Persistence should be non-negative"
            
            # Apply digit sum persistence times and verify we get digital root
            temp = n
            for _ in range(persistence):
                temp = await digit_sum(temp)
            assert temp == dr, f"Persistence calculation inconsistent for {n}"
            
            # Palindrome properties
            reversed_n = await digit_reversal(n)
            is_palindrome = await is_palindromic_number(n)
            
            if is_palindrome:
                assert reversed_n == n, f"Palindrome {n} should equal its reversal"
            
            # Harshad properties
            is_harshad = await is_harshad_number(n)
            if is_harshad:
                assert n % ds == 0, f"Harshad number {n} should be divisible by digit sum {ds}"
    
    @pytest.mark.asyncio
    async def test_performance_stress_test(self):
        """Stress test with larger datasets to ensure performance."""
        # Test with moderately large numbers
        large_numbers = [10**i for i in range(1, 8)]  # 10, 100, 1000, ..., 10000000
        
        start_time = time.time()
        
        # Run comprehensive tests on large numbers
        tasks = []
        for n in large_numbers:
            tasks.extend([
                digit_sum(n),
                digital_root(n),
                digit_count(n),
                is_palindromic_number(n),
                digit_reversal(n),
                digit_frequency(n)
            ])
        
        results = await asyncio.gather(*tasks)
        duration = time.time() - start_time
        
        # Should complete in reasonable time
        assert duration < 5.0, f"Large number processing took too long: {duration}s"
        
        # Verify results are reasonable
        assert len(results) == len(large_numbers) * 6
        
        # Test sequence generation performance
        start_time = time.time()
        
        sequences = await asyncio.gather(
            palindromic_numbers(1000),
            harshad_numbers(1000),
            automorphic_numbers(10000)
        )
        
        duration = time.time() - start_time
        assert duration < 10.0, f"Sequence generation took too long: {duration}s"
        
        # Verify sequences are properly ordered and contain expected elements
        palindromes, harshads, automorphics = sequences
        
        # Should be in ascending order
        assert palindromes == sorted(palindromes)
        assert harshads == sorted(harshads)
        assert automorphics == sorted(automorphics)
        
        # Should contain expected elements
        assert 121 in palindromes  # Known palindrome
        assert 12 in harshads      # Known Harshad number
        assert 25 in automorphics  # Known automorphic number

if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v", "--tb=short", "--asyncio-mode=auto"])