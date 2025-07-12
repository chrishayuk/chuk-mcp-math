#!/usr/bin/env python3
# tests/math/arithmetic/core/test_rounding.py
"""
Comprehensive pytest unit tests for core rounding operations.

Tests cover:
- Basic rounding functions (round_number, floor, ceil, truncate)
- Multiple-based rounding (ceiling_multiple, floor_multiple, mround)
- Edge cases and precision handling
- Banker's rounding vs standard rounding
- Negative number handling
- Error conditions and validation
- Async behavior and performance
"""

import pytest
import math
import asyncio
from typing import Union

# Import the functions to test
from chuk_mcp_functions.math.arithmetic.core.rounding import (
    round_number, floor, ceil, truncate,
    ceiling_multiple, floor_multiple, mround
)

Number = Union[int, float]

class TestRoundNumber:
    """Test cases for the round_number function."""
    
    @pytest.mark.asyncio
    async def test_round_number_to_integer(self):
        """Test rounding to integer (0 decimal places)."""
        assert await round_number(3.14159) == 3
        assert await round_number(3.7) == 4
        assert await round_number(-3.7) == -4
        assert await round_number(2.5) == 2  # Banker's rounding
        assert await round_number(3.5) == 4  # Banker's rounding
    
    @pytest.mark.asyncio
    async def test_round_number_to_decimal_places(self):
        """Test rounding to specific decimal places."""
        assert await round_number(3.14159, 2) == 3.14
        assert await round_number(3.14159, 3) == 3.142
        assert await round_number(3.14159, 4) == 3.1416
        assert await round_number(2.675, 2) == 2.67  # Note: floating point precision
    
    @pytest.mark.asyncio
    async def test_round_number_negative_decimals(self):
        """Test rounding to negative decimal places (round to tens, hundreds, etc.)."""
        assert await round_number(1234.567, -1) == 1230
        assert await round_number(1234.567, -2) == 1200
        assert await round_number(1234.567, -3) == 1000
        assert await round_number(1567, -2) == 1600
    
    @pytest.mark.asyncio
    async def test_round_number_zero(self):
        """Test rounding zero."""
        assert await round_number(0) == 0
        assert await round_number(0.0) == 0
        assert await round_number(0, 2) == 0
    
    @pytest.mark.asyncio
    async def test_round_number_integers(self):
        """Test rounding integers (should remain unchanged)."""
        assert await round_number(5) == 5
        assert await round_number(-3) == -3
        assert await round_number(100, 2) == 100
    
    @pytest.mark.asyncio
    async def test_round_number_banker_rounding(self):
        """Test banker's rounding (round half to even)."""
        # When the digit to be rounded is exactly 5, round to the nearest even number
        assert await round_number(0.5) == 0  # Round to even
        assert await round_number(1.5) == 2  # Round to even
        assert await round_number(2.5) == 2  # Round to even
        assert await round_number(3.5) == 4  # Round to even
        assert await round_number(-0.5) == 0  # Round to even
        assert await round_number(-1.5) == -2  # Round to even
    
    @pytest.mark.asyncio
    async def test_round_number_large_numbers(self):
        """Test rounding large numbers."""
        assert await round_number(1234567890.123456, 2) == 1234567890.12
        assert await round_number(1e15 + 0.7) == 1e15 + 1  # May lose precision
    
    @pytest.mark.asyncio
    async def test_round_number_very_small_numbers(self):
        """Test rounding very small numbers."""
        assert await round_number(1e-10, 12) == 1e-10
        assert await round_number(1.23e-15, 17) == 1.23e-15

class TestFloor:
    """Test cases for the floor function."""
    
    @pytest.mark.asyncio
    async def test_floor_positive_numbers(self):
        """Test floor of positive numbers."""
        assert await floor(3.7) == 3
        assert await floor(3.1) == 3
        assert await floor(3.9) == 3
        assert await floor(3.0) == 3
    
    @pytest.mark.asyncio
    async def test_floor_negative_numbers(self):
        """Test floor of negative numbers."""
        assert await floor(-2.3) == -3
        assert await floor(-2.7) == -3
        assert await floor(-2.0) == -2
        assert await floor(-0.1) == -1
    
    @pytest.mark.asyncio
    async def test_floor_zero(self):
        """Test floor of zero."""
        assert await floor(0) == 0
        assert await floor(0.0) == 0
        assert await floor(-0.0) == 0
    
    @pytest.mark.asyncio
    async def test_floor_integers(self):
        """Test floor of integers (should remain unchanged)."""
        assert await floor(5) == 5
        assert await floor(-3) == -3
        assert await floor(0) == 0
        assert await floor(100) == 100
    
    @pytest.mark.asyncio
    async def test_floor_edge_cases(self):
        """Test floor edge cases."""
        assert await floor(0.999999) == 0
        assert await floor(-0.000001) == -1
        assert await floor(1.0) == 1
        assert await floor(-1.0) == -1
    
    @pytest.mark.asyncio
    async def test_floor_return_type(self):
        """Test that floor always returns int."""
        result = await floor(3.7)
        assert isinstance(result, int)
        
        result = await floor(-2.3)
        assert isinstance(result, int)

class TestCeil:
    """Test cases for the ceil function."""
    
    @pytest.mark.asyncio
    async def test_ceil_positive_numbers(self):
        """Test ceiling of positive numbers."""
        assert await ceil(3.2) == 4
        assert await ceil(3.1) == 4
        assert await ceil(3.9) == 4
        assert await ceil(3.0) == 3
    
    @pytest.mark.asyncio
    async def test_ceil_negative_numbers(self):
        """Test ceiling of negative numbers."""
        assert await ceil(-2.7) == -2
        assert await ceil(-2.1) == -2
        assert await ceil(-2.0) == -2
        assert await ceil(-0.1) == 0
    
    @pytest.mark.asyncio
    async def test_ceil_zero(self):
        """Test ceiling of zero."""
        assert await ceil(0) == 0
        assert await ceil(0.0) == 0
        assert await ceil(-0.0) == 0
    
    @pytest.mark.asyncio
    async def test_ceil_integers(self):
        """Test ceiling of integers (should remain unchanged)."""
        assert await ceil(5) == 5
        assert await ceil(-3) == -3
        assert await ceil(0) == 0
        assert await ceil(100) == 100
    
    @pytest.mark.asyncio
    async def test_ceil_edge_cases(self):
        """Test ceiling edge cases."""
        assert await ceil(0.000001) == 1
        assert await ceil(-0.999999) == 0
        assert await ceil(1.0) == 1
        assert await ceil(-1.0) == -1
    
    @pytest.mark.asyncio
    async def test_ceil_return_type(self):
        """Test that ceil always returns int."""
        result = await ceil(3.2)
        assert isinstance(result, int)
        
        result = await ceil(-2.7)
        assert isinstance(result, int)

class TestTruncate:
    """Test cases for the truncate function."""
    
    @pytest.mark.asyncio
    async def test_truncate_positive_numbers(self):
        """Test truncation of positive numbers."""
        assert await truncate(3.9) == 3
        assert await truncate(3.1) == 3
        assert await truncate(3.0) == 3
        assert await truncate(0.9) == 0
    
    @pytest.mark.asyncio
    async def test_truncate_negative_numbers(self):
        """Test truncation of negative numbers."""
        assert await truncate(-2.9) == -2
        assert await truncate(-2.1) == -2
        assert await truncate(-2.0) == -2
        assert await truncate(-0.9) == 0
    
    @pytest.mark.asyncio
    async def test_truncate_zero(self):
        """Test truncation of zero."""
        assert await truncate(0) == 0
        assert await truncate(0.0) == 0
        assert await truncate(-0.0) == 0
    
    @pytest.mark.asyncio
    async def test_truncate_integers(self):
        """Test truncation of integers (should remain unchanged)."""
        assert await truncate(5) == 5
        assert await truncate(-3) == -3
        assert await truncate(0) == 0
        assert await truncate(100) == 100
    
    @pytest.mark.asyncio
    async def test_truncate_vs_floor_positive(self):
        """Test that truncate and floor are the same for positive numbers."""
        test_values = [3.1, 3.9, 0.1, 0.9, 5.0]
        for value in test_values:
            assert await truncate(value) == await floor(value)
    
    @pytest.mark.asyncio
    async def test_truncate_vs_ceil_negative(self):
        """Test that truncate and ceil are the same for negative numbers."""
        test_values = [-3.1, -3.9, -0.1, -0.9, -5.0]
        for value in test_values:
            assert await truncate(value) == await ceil(value)
    
    @pytest.mark.asyncio
    async def test_truncate_return_type(self):
        """Test that truncate always returns int."""
        result = await truncate(3.9)
        assert isinstance(result, int)
        
        result = await truncate(-2.1)
        assert isinstance(result, int)

class TestCeilingMultiple:
    """Test cases for the ceiling_multiple function."""
    
    @pytest.mark.asyncio
    async def test_ceiling_multiple_basic(self):
        """Test basic ceiling multiple operations."""
        assert await ceiling_multiple(2.5, 1) == 3
        assert await ceiling_multiple(6.7, 2) == 8
        assert await ceiling_multiple(15, 10) == 20
        assert await ceiling_multiple(7.3, 2) == 8
    
    @pytest.mark.asyncio
    async def test_ceiling_multiple_exact_multiples(self):
        """Test ceiling multiple with exact multiples."""
        assert await ceiling_multiple(6, 2) == 6
        assert await ceiling_multiple(10, 5) == 10
        assert await ceiling_multiple(15, 3) == 15
        assert await ceiling_multiple(0, 5) == 0
    
    @pytest.mark.asyncio
    async def test_ceiling_multiple_negative_numbers(self):
        """Test ceiling multiple with negative numbers."""
        # For negative numbers, "away from zero" means more negative
        assert await ceiling_multiple(-2.1, 1) == -3
        assert await ceiling_multiple(-5.5, 2) == -6
        assert await ceiling_multiple(-7, 3) == -9
    
    @pytest.mark.asyncio
    async def test_ceiling_multiple_fractional_significance(self):
        """Test ceiling multiple with fractional significance."""
        assert await ceiling_multiple(1.23, 0.5) == 1.5
        assert await ceiling_multiple(2.1, 0.25) == 2.25
        assert pytest.approx(await ceiling_multiple(3.14, 0.1)) == 3.2
    
    @pytest.mark.asyncio
    async def test_ceiling_multiple_zero_significance_raises(self):
        """Test that zero significance raises ValueError."""
        with pytest.raises(ValueError, match="Significance must be positive"):
            await ceiling_multiple(5, 0)
    
    @pytest.mark.asyncio
    async def test_ceiling_multiple_negative_significance_raises(self):
        """Test that negative significance raises ValueError."""
        with pytest.raises(ValueError, match="Significance must be positive"):
            await ceiling_multiple(5, -2)

class TestFloorMultiple:
    """Test cases for the floor_multiple function."""
    
    @pytest.mark.asyncio
    async def test_floor_multiple_basic(self):
        """Test basic floor multiple operations."""
        assert await floor_multiple(2.9, 1) == 2
        assert await floor_multiple(7.8, 2) == 6
        assert await floor_multiple(23, 10) == 20
        assert await floor_multiple(8.7, 2) == 8
    
    @pytest.mark.asyncio
    async def test_floor_multiple_exact_multiples(self):
        """Test floor multiple with exact multiples."""
        assert await floor_multiple(6, 2) == 6
        assert await floor_multiple(10, 5) == 10
        assert await floor_multiple(15, 3) == 15
        assert await floor_multiple(0, 5) == 0
    
    @pytest.mark.asyncio
    async def test_floor_multiple_negative_numbers(self):
        """Test floor multiple with negative numbers."""
        # For negative numbers, "toward zero" means less negative
        assert await floor_multiple(-2.9, 1) == -2
        assert await floor_multiple(-5.5, 2) == -4
        assert await floor_multiple(-7.1, 3) == -6
    
    @pytest.mark.asyncio
    async def test_floor_multiple_fractional_significance(self):
        """Test floor multiple with fractional significance."""
        assert await floor_multiple(1.77, 0.5) == 1.5
        assert await floor_multiple(2.9, 0.25) == 2.75
        assert pytest.approx(await floor_multiple(3.14, 0.1)) == 3.1
    
    @pytest.mark.asyncio
    async def test_floor_multiple_zero_significance_raises(self):
        """Test that zero significance raises ValueError."""
        with pytest.raises(ValueError, match="Significance must be positive"):
            await floor_multiple(5, 0)
    
    @pytest.mark.asyncio
    async def test_floor_multiple_negative_significance_raises(self):
        """Test that negative significance raises ValueError."""
        with pytest.raises(ValueError, match="Significance must be positive"):
            await floor_multiple(5, -2)

class TestMround:
    """Test cases for the mround function."""
    
    @pytest.mark.asyncio
    async def test_mround_basic(self):
        """Test basic mround operations."""
        assert await mround(2.4, 1) == 2
        assert await mround(2.6, 1) == 3
        assert await mround(7.3, 2) == 8
        assert await mround(15, 10) == 20
    
    @pytest.mark.asyncio
    async def test_mround_exact_multiples(self):
        """Test mround with exact multiples."""
        assert await mround(6, 2) == 6
        assert await mround(10, 5) == 10
        assert await mround(15, 3) == 15
        assert await mround(0, 5) == 0
    
    @pytest.mark.asyncio
    async def test_mround_midpoint_cases(self):
        """Test mround with midpoint cases (exactly between multiples)."""
        assert await mround(2.5, 1) == 2  # Banker's rounding
        assert await mround(3.5, 1) == 4  # Banker's rounding
        assert await mround(5, 2) == 4    # 5 is halfway between 4 and 6
        assert await mround(7, 2) == 8    # 7 is closer to 8 than 6
    
    @pytest.mark.asyncio
    async def test_mround_negative_numbers(self):
        """Test mround with negative numbers."""
        assert await mround(-2.4, 1) == -2
        assert await mround(-2.6, 1) == -3
        assert await mround(-7.3, 2) == -8
        assert await mround(-5, 2) == -4  # -5 rounds to even (-4)
    
    @pytest.mark.asyncio
    async def test_mround_fractional_significance(self):
        """Test mround with fractional significance."""
        assert await mround(1.65, 0.5) == 1.5  # Banker's rounding
        assert await mround(1.75, 0.5) == 2.0  # Banker's rounding
        assert pytest.approx(await mround(3.14, 0.1)) == 3.1
        assert pytest.approx(await mround(3.16, 0.1)) == 3.2
    
    @pytest.mark.asyncio
    async def test_mround_zero_significance_raises(self):
        """Test that zero significance raises ValueError."""
        with pytest.raises(ValueError, match="Significance must be positive"):
            await mround(5, 0)
    
    @pytest.mark.asyncio
    async def test_mround_negative_significance_raises(self):
        """Test that negative significance raises ValueError."""
        with pytest.raises(ValueError, match="Significance must be positive"):
            await mround(5, -2)

class TestIntegration:
    """Integration tests for rounding operations."""
    
    @pytest.mark.asyncio
    async def test_floor_ceil_relationship(self):
        """Test the relationship between floor and ceil."""
        test_values = [3.1, 3.9, -2.1, -2.9, 0, 5.0, -5.0]
        
        for value in test_values:
            floor_result = await floor(value)
            ceil_result = await ceil(value)
            
            if value == int(value):  # If value is an integer
                assert floor_result == ceil_result == int(value)
            else:
                assert ceil_result == floor_result + 1
    
    @pytest.mark.asyncio
    async def test_truncate_relationship(self):
        """Test relationship between truncate, floor, and ceil."""
        test_values = [3.7, -3.7, 0.5, -0.5]
        
        for value in test_values:
            trunc_result = await truncate(value)
            floor_result = await floor(value)
            ceil_result = await ceil(value)
            
            if value >= 0:
                assert trunc_result == floor_result
            else:
                assert trunc_result == ceil_result
    
    @pytest.mark.asyncio
    async def test_multiple_rounding_consistency(self):
        """Test consistency between different multiple rounding functions."""
        test_cases = [
            (2.7, 1), (7.3, 2), (15.5, 5), (23.8, 10)
        ]
        
        for number, significance in test_cases:
            floor_mult = await floor_multiple(number, significance)
            ceil_mult = await ceiling_multiple(number, significance)
            
            # Floor multiple should be <= original number
            assert floor_mult <= number
            # Ceiling multiple should be >= original number
            assert ceil_mult >= number
            
            # The difference between ceiling and floor multiple should be significance
            if floor_mult != ceil_mult:
                assert pytest.approx(ceil_mult - floor_mult) == significance
    
    @pytest.mark.asyncio
    async def test_round_number_vs_mround_consistency(self):
        """Test consistency between round_number and mround for significance=1."""
        test_values = [2.4, 2.6, 3.5, -2.4, -2.6, -3.5]
        
        for value in test_values:
            round_result = await round_number(value)
            mround_result = await mround(value, 1)
            assert round_result == mround_result

class TestPerformance:
    """Performance and async behavior tests."""
    
    @pytest.mark.asyncio
    async def test_all_operations_are_async(self):
        """Test that all rounding operations are properly async."""
        operations = [
            round_number(3.14159, 2),
            floor(3.7),
            ceil(3.2),
            truncate(3.9),
            ceiling_multiple(6.7, 2),
            floor_multiple(7.8, 2),
            mround(7.3, 2)
        ]
        
        # Ensure all are coroutines
        for op in operations:
            assert asyncio.iscoroutine(op)
        
        # Run all operations concurrently
        results = await asyncio.gather(*operations)
        expected = [3.14, 3, 4, 3, 8, 6, 8]
        
        for result, expected_val in zip(results, expected):
            if isinstance(expected_val, float):
                assert pytest.approx(result) == expected_val
            else:
                assert result == expected_val
    
    @pytest.mark.asyncio
    async def test_concurrent_execution(self):
        """Test that rounding operations can run concurrently."""
        import time
        
        start_time = time.time()
        
        # Run multiple rounding operations concurrently
        tasks = [
            round_number(i + 0.7, 1) for i in range(100)
        ]
        
        results = await asyncio.gather(*tasks)
        duration = time.time() - start_time
        
        # Verify results are correct
        for i, result in enumerate(results):
            expected = round(i + 0.7, 1)
            assert pytest.approx(result) == expected
        
        # Should complete quickly due to async nature
        assert duration < 1.0

# Parametrized tests for comprehensive coverage
class TestParametrized:
    """Parametrized tests for comprehensive coverage."""
    
    @pytest.mark.asyncio
    @pytest.mark.parametrize("value,decimals,expected", [
        (3.14159, 0, 3),
        (3.14159, 1, 3.1),
        (3.14159, 2, 3.14),
        (3.14159, 3, 3.142),
        (2.5, 0, 2),  # Banker's rounding
        (3.5, 0, 4),  # Banker's rounding
        (-3.7, 0, -4),
        (0, 2, 0)
    ])
    async def test_round_number_parametrized(self, value, decimals, expected):
        """Parametrized test for round_number function."""
        result = await round_number(value, decimals)
        if isinstance(expected, float):
            assert pytest.approx(result) == expected
        else:
            assert result == expected
    
    @pytest.mark.asyncio
    @pytest.mark.parametrize("value,expected", [
        (3.7, 3), (3.1, 3), (-2.3, -3), (-2.7, -3),
        (0, 0), (5, 5), (-5, -5), (0.999, 0), (-0.001, -1)
    ])
    async def test_floor_parametrized(self, value, expected):
        """Parametrized test for floor function."""
        result = await floor(value)
        assert result == expected
    
    @pytest.mark.asyncio
    @pytest.mark.parametrize("value,expected", [
        (3.2, 4), (3.9, 4), (-2.7, -2), (-2.1, -2),
        (0, 0), (5, 5), (-5, -5), (0.001, 1), (-0.999, 0)
    ])
    async def test_ceil_parametrized(self, value, expected):
        """Parametrized test for ceil function."""
        result = await ceil(value)
        assert result == expected
    
    @pytest.mark.asyncio
    @pytest.mark.parametrize("number,significance,expected", [
        (2.4, 1, 2), (2.6, 1, 3), (7.3, 2, 8), (15, 10, 20),
        (2.5, 1, 2), (3.5, 1, 4),  # Banker's rounding
        (-2.4, 1, -2), (-2.6, 1, -3)
    ])
    async def test_mround_parametrized(self, number, significance, expected):
        """Parametrized test for mround function."""
        result = await mround(number, significance)
        assert result == expected

# Error handling tests
class TestErrorHandling:
    """Test error handling and edge cases."""
    
    @pytest.mark.asyncio
    async def test_multiple_functions_significance_validation(self):
        """Test that multiple-based functions validate significance properly."""
        functions = [ceiling_multiple, floor_multiple, mround]
        
        for func in functions:
            # Test zero significance
            with pytest.raises(ValueError, match="Significance must be positive"):
                await func(5, 0)
            
            # Test negative significance
            with pytest.raises(ValueError, match="Significance must be positive"):
                await func(5, -1)
            
            # Test infinite significance
            with pytest.raises(ValueError, match="Significance must be finite"):
                await func(5, float('inf'))
    
    @pytest.mark.asyncio
    async def test_extreme_values(self):
        """Test rounding with extreme values."""
        # Very large numbers
        large_num = 1e15
        assert await floor(large_num) == int(large_num)
        assert await ceil(large_num) == int(large_num)
        
        # Very small numbers
        small_num = 1e-15
        assert await floor(small_num) == 0
        assert await ceil(small_num) == 1
    
    @pytest.mark.asyncio
    async def test_infinity_and_nan_handling(self):
        """Test handling of infinity and NaN values."""
        import math
        
        # Test with infinity - math.floor/ceil raise OverflowError for infinity
        inf = float('inf')
        neg_inf = float('-inf')
        
        # Floor and ceil of infinity should raise OverflowError
        with pytest.raises(OverflowError):
            await floor(inf)
        
        with pytest.raises(OverflowError):
            await ceil(inf)
        
        with pytest.raises(OverflowError):
            await floor(neg_inf)
        
        with pytest.raises(OverflowError):
            await ceil(neg_inf)
        
        # Truncate should also raise OverflowError for infinity
        with pytest.raises(OverflowError):
            await truncate(inf)
        
        with pytest.raises(OverflowError):
            await truncate(neg_inf)
        
        # NaN should propagate through round_number (which uses Python's round())
        nan = float('nan')
        result = await round_number(nan)
        assert math.isnan(result)
        
        # NaN with decimal places
        result = await round_number(nan, 2)
        assert math.isnan(result)
        
        # Test multiple rounding functions with infinity should raise ValueError
        # since significance must be positive and finite
        with pytest.raises(ValueError):
            await ceiling_multiple(5, inf)
        
        with pytest.raises(ValueError):
            await floor_multiple(5, inf)
        
        with pytest.raises(ValueError):
            await mround(5, inf)

if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v", "--tb=short"])