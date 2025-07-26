#!/usr/bin/env python3
# tests/math/arithmetic/comparison/test_relational.py
"""
Comprehensive pytest unit tests for relational comparison operations.

Tests cover:
- Basic equality and inequality operations (equal, not_equal)
- Ordering operations (less_than, less_than_or_equal, greater_than, greater_than_or_equal)
- Range checking operations (in_range, between)
- Edge cases with extreme values, NaN, infinity
- Type consistency and mixed type comparisons
- Error conditions and validation
- Async behavior and performance
"""

import pytest
import math
import asyncio
from typing import Union

# Import the functions to test
from chuk_mcp_math.arithmetic.comparison.relational import (
    equal, not_equal, less_than, less_than_or_equal,
    greater_than, greater_than_or_equal, in_range, between
)

Number = Union[int, float]

class TestEqual:
    """Test cases for the equal function."""
    
    @pytest.mark.asyncio
    async def test_equal_integers(self):
        """Test equality with integer inputs."""
        assert await equal(5, 5) == True
        assert await equal(0, 0) == True
        assert await equal(-3, -3) == True
        assert await equal(5, 3) == False
        assert await equal(-2, 2) == False
    
    @pytest.mark.asyncio
    async def test_equal_floats(self):
        """Test equality with float inputs."""
        assert await equal(3.14, 3.14) == True
        assert await equal(0.0, 0.0) == True
        assert await equal(-2.5, -2.5) == True
        assert await equal(3.14, 3.15) == False
        assert await equal(1.0, 1.1) == False
    
    @pytest.mark.asyncio
    async def test_equal_mixed_types(self):
        """Test equality with mixed int/float inputs."""
        assert await equal(5, 5.0) == True
        assert await equal(5.0, 5) == True
        assert await equal(0, 0.0) == True
        assert await equal(-3, -3.0) == True
        assert await equal(5, 5.1) == False
        assert await equal(5.1, 5) == False
    
    @pytest.mark.asyncio
    async def test_equal_extreme_values(self):
        """Test equality with extreme values."""
        inf = float('inf')
        neg_inf = float('-inf')
        
        assert await equal(inf, inf) == True
        assert await equal(neg_inf, neg_inf) == True
        assert await equal(inf, neg_inf) == False
        assert await equal(inf, 1e10) == False
        assert await equal(1e-10, 0) == False
    
    @pytest.mark.asyncio
    async def test_equal_with_nan(self):
        """Test equality with NaN values."""
        nan = float('nan')
        
        # NaN is never equal to anything, including itself
        assert await equal(nan, nan) == False
        assert await equal(nan, 5) == False
        assert await equal(5, nan) == False
        assert await equal(nan, float('inf')) == False
    
    @pytest.mark.asyncio
    async def test_equal_floating_point_precision(self):
        """Test equality with floating point precision issues."""
        # These should be exactly equal due to binary representation
        assert await equal(0.1 + 0.2, 0.3) == False  # Classic floating point issue
        
        # But these should be equal
        assert await equal(0.5 + 0.5, 1.0) == True
        assert await equal(0.25 + 0.25, 0.5) == True

class TestNotEqual:
    """Test cases for the not_equal function."""
    
    @pytest.mark.asyncio
    async def test_not_equal_basic(self):
        """Test basic not_equal functionality."""
        assert await not_equal(5, 3) == True
        assert await not_equal(3, 5) == True
        assert await not_equal(-2, 2) == True
        assert await not_equal(5, 5) == False
        assert await not_equal(0, 0) == False
    
    @pytest.mark.asyncio
    async def test_not_equal_floats(self):
        """Test not_equal with float inputs."""
        assert await not_equal(3.14, 3.15) == True
        assert await not_equal(1.0, 1.1) == True
        assert await not_equal(3.14, 3.14) == False
        assert await not_equal(0.0, 0.0) == False
    
    @pytest.mark.asyncio
    async def test_not_equal_mixed_types(self):
        """Test not_equal with mixed types."""
        assert await not_equal(5, 5.0) == False
        assert await not_equal(5.0, 5) == False
        assert await not_equal(5, 5.1) == True
        assert await not_equal(5.1, 5) == True
    
    @pytest.mark.asyncio
    async def test_not_equal_with_nan(self):
        """Test not_equal with NaN values."""
        nan = float('nan')
        
        # NaN is not equal to anything, so not_equal should return True
        assert await not_equal(nan, nan) == True
        assert await not_equal(nan, 5) == True
        assert await not_equal(5, nan) == True
    
    @pytest.mark.asyncio
    async def test_not_equal_consistency_with_equal(self):
        """Test that not_equal is consistent with equal."""
        test_pairs = [
            (5, 5), (5, 3), (3.14, 3.14), (3.14, 3.15),
            (5, 5.0), (0, 0.0), (float('inf'), float('inf'))
        ]
        
        for a, b in test_pairs:
            equal_result = await equal(a, b)
            not_equal_result = await not_equal(a, b)
            assert equal_result != not_equal_result  # Should be opposites

class TestLessThan:
    """Test cases for the less_than function."""
    
    @pytest.mark.asyncio
    async def test_less_than_integers(self):
        """Test less_than with integers."""
        assert await less_than(3, 5) == True
        assert await less_than(-5, -3) == True
        assert await less_than(-2, 1) == True
        assert await less_than(0, 1) == True
        assert await less_than(5, 3) == False
        assert await less_than(5, 5) == False
    
    @pytest.mark.asyncio
    async def test_less_than_floats(self):
        """Test less_than with floats."""
        assert await less_than(2.71, 3.14) == True
        assert await less_than(-2.5, -1.5) == True
        assert await less_than(0.0, 0.1) == True
        assert await less_than(3.14, 2.71) == False
        assert await less_than(2.5, 2.5) == False
    
    @pytest.mark.asyncio
    async def test_less_than_mixed_types(self):
        """Test less_than with mixed types."""
        assert await less_than(3, 3.5) == True
        assert await less_than(3.5, 4) == True
        assert await less_than(5, 5.0) == False
        assert await less_than(5.1, 5) == False
    
    @pytest.mark.asyncio
    async def test_less_than_extreme_values(self):
        """Test less_than with extreme values."""
        inf = float('inf')
        neg_inf = float('-inf')
        
        assert await less_than(neg_inf, inf) == True
        assert await less_than(neg_inf, 0) == True
        assert await less_than(0, inf) == True
        assert await less_than(1e10, inf) == True
        assert await less_than(inf, inf) == False
        assert await less_than(inf, 0) == False
    
    @pytest.mark.asyncio
    async def test_less_than_with_nan(self):
        """Test less_than with NaN values."""
        nan = float('nan')
        
        # NaN comparisons always return False
        assert await less_than(nan, 5) == False
        assert await less_than(5, nan) == False
        assert await less_than(nan, nan) == False
        assert await less_than(nan, float('inf')) == False

class TestLessThanOrEqual:
    """Test cases for the less_than_or_equal function."""
    
    @pytest.mark.asyncio
    async def test_less_than_or_equal_basic(self):
        """Test basic less_than_or_equal functionality."""
        assert await less_than_or_equal(3, 5) == True
        assert await less_than_or_equal(5, 5) == True
        assert await less_than_or_equal(-3, -3) == True
        assert await less_than_or_equal(5, 3) == False
    
    @pytest.mark.asyncio
    async def test_less_than_or_equal_consistency(self):
        """Test consistency with less_than and equal."""
        test_pairs = [
            (3, 5), (5, 3), (5, 5), (-2, 1), (3.14, 3.15)
        ]
        
        for a, b in test_pairs:
            less_than_result = await less_than(a, b)
            equal_result = await equal(a, b)
            less_than_or_equal_result = await less_than_or_equal(a, b)
            
            # less_than_or_equal should be True if either less_than OR equal is True
            assert less_than_or_equal_result == (less_than_result or equal_result)

class TestGreaterThan:
    """Test cases for the greater_than function."""
    
    @pytest.mark.asyncio
    async def test_greater_than_integers(self):
        """Test greater_than with integers."""
        assert await greater_than(5, 3) == True
        assert await greater_than(-3, -5) == True
        assert await greater_than(1, -2) == True
        assert await greater_than(1, 0) == True
        assert await greater_than(3, 5) == False
        assert await greater_than(5, 5) == False
    
    @pytest.mark.asyncio
    async def test_greater_than_floats(self):
        """Test greater_than with floats."""
        assert await greater_than(3.14, 2.71) == True
        assert await greater_than(-1.5, -2.5) == True
        assert await greater_than(0.1, 0.0) == True
        assert await greater_than(2.71, 3.14) == False
        assert await greater_than(2.5, 2.5) == False
    
    @pytest.mark.asyncio
    async def test_greater_than_consistency_with_less_than(self):
        """Test that greater_than is consistent with less_than."""
        test_pairs = [
            (5, 3), (3, 5), (5, 5), (-2, 1), (3.14, 2.71)
        ]
        
        for a, b in test_pairs:
            greater_than_result = await greater_than(a, b)
            less_than_result = await less_than(b, a)  # Swap arguments
            
            # a > b should be equivalent to b < a
            assert greater_than_result == less_than_result

class TestGreaterThanOrEqual:
    """Test cases for the greater_than_or_equal function."""
    
    @pytest.mark.asyncio
    async def test_greater_than_or_equal_basic(self):
        """Test basic greater_than_or_equal functionality."""
        assert await greater_than_or_equal(5, 3) == True
        assert await greater_than_or_equal(5, 5) == True
        assert await greater_than_or_equal(-3, -3) == True
        assert await greater_than_or_equal(3, 5) == False
    
    @pytest.mark.asyncio
    async def test_greater_than_or_equal_consistency(self):
        """Test consistency with greater_than and equal."""
        test_pairs = [
            (5, 3), (3, 5), (5, 5), (1, -2), (3.15, 3.14)
        ]
        
        for a, b in test_pairs:
            greater_than_result = await greater_than(a, b)
            equal_result = await equal(a, b)
            greater_than_or_equal_result = await greater_than_or_equal(a, b)
            
            # greater_than_or_equal should be True if either greater_than OR equal is True
            assert greater_than_or_equal_result == (greater_than_result or equal_result)

class TestInRange:
    """Test cases for the in_range function."""
    
    @pytest.mark.asyncio
    async def test_in_range_inclusive_basic(self):
        """Test in_range with inclusive bounds (default)."""
        assert await in_range(5, 1, 10) == True
        assert await in_range(1, 1, 10) == True  # At lower bound
        assert await in_range(10, 1, 10) == True  # At upper bound
        assert await in_range(0, 1, 10) == False  # Below range
        assert await in_range(11, 1, 10) == False  # Above range
    
    @pytest.mark.asyncio
    async def test_in_range_inclusive_explicit(self):
        """Test in_range with explicitly inclusive bounds."""
        assert await in_range(5, 1, 10, inclusive=True) == True
        assert await in_range(1, 1, 10, inclusive=True) == True
        assert await in_range(10, 1, 10, inclusive=True) == True
        assert await in_range(0, 1, 10, inclusive=True) == False
        assert await in_range(11, 1, 10, inclusive=True) == False
    
    @pytest.mark.asyncio
    async def test_in_range_exclusive(self):
        """Test in_range with exclusive bounds."""
        assert await in_range(5, 1, 10, inclusive=False) == True
        assert await in_range(1, 1, 10, inclusive=False) == False  # At lower bound
        assert await in_range(10, 1, 10, inclusive=False) == False  # At upper bound
        assert await in_range(0, 1, 10, inclusive=False) == False  # Below range
        assert await in_range(11, 1, 10, inclusive=False) == False  # Above range
    
    @pytest.mark.asyncio
    async def test_in_range_floats(self):
        """Test in_range with float values."""
        assert await in_range(3.14, 1.0, 10.0) == True
        assert await in_range(1.0, 1.0, 10.0, inclusive=True) == True
        assert await in_range(1.0, 1.0, 10.0, inclusive=False) == False
        assert await in_range(0.5, 1.0, 10.0) == False
    
    @pytest.mark.asyncio
    async def test_in_range_negative_bounds(self):
        """Test in_range with negative bounds."""
        assert await in_range(-2, -5, 5) == True
        assert await in_range(-5, -5, 5, inclusive=True) == True
        assert await in_range(-5, -5, 5, inclusive=False) == False
        assert await in_range(-10, -5, 5) == False
    
    @pytest.mark.asyncio
    async def test_in_range_zero_width_range(self):
        """Test in_range with zero-width range (min == max)."""
        assert await in_range(5, 5, 5, inclusive=True) == True
        assert await in_range(5, 5, 5, inclusive=False) == False
        assert await in_range(3, 5, 5, inclusive=True) == False
        assert await in_range(7, 5, 5, inclusive=True) == False
    
    @pytest.mark.asyncio
    async def test_in_range_invalid_bounds(self):
        """Test in_range with invalid bounds (min > max)."""
        with pytest.raises(ValueError, match="Minimum value cannot be greater than maximum value"):
            await in_range(5, 10, 1)
        
        with pytest.raises(ValueError, match="Minimum value cannot be greater than maximum value"):
            await in_range(0, 5.5, 2.3)
    
    @pytest.mark.asyncio
    async def test_in_range_extreme_values(self):
        """Test in_range with extreme values."""
        inf = float('inf')
        neg_inf = float('-inf')
        
        assert await in_range(1000, neg_inf, inf) == True
        assert await in_range(inf, 1, 10) == False
        assert await in_range(neg_inf, 1, 10) == False
        assert await in_range(5, neg_inf, 10) == True
        assert await in_range(5, 1, inf) == True
    
    @pytest.mark.asyncio
    async def test_in_range_with_nan(self):
        """Test in_range with NaN values."""
        nan = float('nan')
        
        # NaN comparisons should return False
        assert await in_range(nan, 1, 10) == False
        assert await in_range(5, nan, 10) == False
        assert await in_range(5, 1, nan) == False

class TestBetween:
    """Test cases for the between function."""
    
    @pytest.mark.asyncio
    async def test_between_basic(self):
        """Test basic between functionality (exclusive bounds)."""
        assert await between(5, 1, 10) == True
        assert await between(2, 1, 10) == True
        assert await between(9, 1, 10) == True
        assert await between(1, 1, 10) == False  # At lower bound
        assert await between(10, 1, 10) == False  # At upper bound
        assert await between(0, 1, 10) == False  # Below range
        assert await between(11, 1, 10) == False  # Above range
    
    @pytest.mark.asyncio
    async def test_between_floats(self):
        """Test between with float values."""
        assert await between(3.14, 1.0, 10.0) == True
        assert await between(1.0, 1.0, 10.0) == False  # At bound
        assert await between(1.01, 1.0, 10.0) == True  # Just inside
        assert await between(9.99, 1.0, 10.0) == True  # Just inside
    
    @pytest.mark.asyncio
    async def test_between_negative_bounds(self):
        """Test between with negative bounds."""
        assert await between(0, -5, 5) == True
        assert await between(-2, -5, 5) == True
        assert await between(-5, -5, 5) == False  # At lower bound
        assert await between(5, -5, 5) == False  # At upper bound
    
    @pytest.mark.asyncio
    async def test_between_consistency_with_in_range(self):
        """Test that between is consistent with in_range(exclusive=True)."""
        test_cases = [
            (5, 1, 10), (1, 1, 10), (10, 1, 10), (0, 1, 10), (11, 1, 10),
            (3.14, 1.0, 10.0), (-2, -5, 5)
        ]
        
        for value, lower, upper in test_cases:
            between_result = await between(value, lower, upper)
            in_range_result = await in_range(value, lower, upper, inclusive=False)
            
            assert between_result == in_range_result
    
    @pytest.mark.asyncio
    async def test_between_extreme_values(self):
        """Test between with extreme values."""
        inf = float('inf')
        neg_inf = float('-inf')
        
        assert await between(1000, neg_inf, inf) == True
        assert await between(inf, 1, 10) == False
        assert await between(neg_inf, 1, 10) == False
        assert await between(5, neg_inf, 10) == True
        assert await between(5, 1, inf) == True
    
    @pytest.mark.asyncio
    async def test_between_with_nan(self):
        """Test between with NaN values."""
        nan = float('nan')
        
        # NaN comparisons should return False
        assert await between(nan, 1, 10) == False
        assert await between(5, nan, 10) == False
        assert await between(5, 1, nan) == False

class TestIntegration:
    """Integration tests for relational operations."""
    
    @pytest.mark.asyncio
    async def test_comparison_transitivity(self):
        """Test transitivity of comparison operations."""
        # If a < b and b < c, then a < c
        a, b, c = 1, 5, 10
        
        assert await less_than(a, b) == True
        assert await less_than(b, c) == True
        assert await less_than(a, c) == True
        
        # Similar for greater_than
        assert await greater_than(c, b) == True
        assert await greater_than(b, a) == True
        assert await greater_than(c, a) == True
    
    @pytest.mark.asyncio
    async def test_comparison_antisymmetry(self):
        """Test antisymmetry of comparison operations."""
        test_pairs = [(3, 5), (5, 3), (5, 5)]
        
        for a, b in test_pairs:
            less_than_result = await less_than(a, b)
            greater_than_result = await greater_than(a, b)
            equal_result = await equal(a, b)
            
            # Exactly one of <, >, = should be true
            true_count = sum([less_than_result, greater_than_result, equal_result])
            assert true_count == 1
    
    @pytest.mark.asyncio
    async def test_range_operations_consistency(self):
        """Test consistency between different range operations."""
        test_cases = [
            (5, 1, 10), (1, 1, 10), (10, 1, 10), (0, 1, 10), (11, 1, 10)
        ]
        
        for value, min_val, max_val in test_cases:
            # in_range(inclusive=True) should be equivalent to min_val <= value <= max_val
            in_range_inclusive = await in_range(value, min_val, max_val, inclusive=True)
            manual_check_inclusive = (await less_than_or_equal(min_val, value) and 
                                    await less_than_or_equal(value, max_val))
            assert in_range_inclusive == manual_check_inclusive
            
            # in_range(inclusive=False) should be equivalent to min_val < value < max_val
            in_range_exclusive = await in_range(value, min_val, max_val, inclusive=False)
            manual_check_exclusive = (await less_than(min_val, value) and 
                                    await less_than(value, max_val))
            assert in_range_exclusive == manual_check_exclusive
    
    @pytest.mark.asyncio
    async def test_logical_relationships(self):
        """Test logical relationships between comparison operations."""
        test_pairs = [(3, 5), (5, 3), (5, 5), (-2, 1)]
        
        for a, b in test_pairs:
            equal_result = await equal(a, b)
            not_equal_result = await not_equal(a, b)
            less_than_result = await less_than(a, b)
            greater_than_result = await greater_than(a, b)
            less_than_or_equal_result = await less_than_or_equal(a, b)
            greater_than_or_equal_result = await greater_than_or_equal(a, b)
            
            # Test De Morgan's laws and logical relationships
            assert equal_result == (not not_equal_result)
            assert less_than_or_equal_result == (less_than_result or equal_result)
            assert greater_than_or_equal_result == (greater_than_result or equal_result)
            
            # Exactly one of <, =, > should be true
            comparison_results = [less_than_result, equal_result, greater_than_result]
            assert sum(comparison_results) == 1

class TestPerformance:
    """Performance and async behavior tests."""
    
    @pytest.mark.asyncio
    async def test_all_operations_are_async(self):
        """Test that all relational operations are properly async."""
        operations = [
            equal(5, 5),
            not_equal(5, 3),
            less_than(3, 5),
            less_than_or_equal(3, 5),
            greater_than(5, 3),
            greater_than_or_equal(5, 3),
            in_range(5, 1, 10),
            between(5, 1, 10)
        ]
        
        # Ensure all are coroutines
        for op in operations:
            assert asyncio.iscoroutine(op)
        
        # Run all operations concurrently
        results = await asyncio.gather(*operations)
        expected = [True, True, True, True, True, True, True, True]
        
        assert results == expected
    
    @pytest.mark.asyncio
    async def test_concurrent_execution(self):
        """Test that relational operations can run concurrently."""
        import time
        
        start_time = time.time()
        
        # Run multiple operations concurrently
        tasks = [
            equal(i, i) for i in range(100)
        ]
        
        results = await asyncio.gather(*tasks)
        duration = time.time() - start_time
        
        # All should be True (i equals i)
        assert all(results)
        assert len(results) == 100
        
        # Should complete quickly due to async nature
        assert duration < 1.0
    
    @pytest.mark.asyncio
    async def test_comparison_performance(self):
        """Test performance of comparison operations."""
        import time
        
        large_numbers = list(range(1000))
        
        start_time = time.time()
        
        # Test multiple comparison operations
        tasks = []
        for i in range(len(large_numbers) - 1):
            tasks.append(less_than(large_numbers[i], large_numbers[i + 1]))
        
        results = await asyncio.gather(*tasks)
        duration = time.time() - start_time
        
        # All should be True (consecutive numbers in ascending order)
        assert all(results)
        assert len(results) == 999
        
        # Should handle many comparisons efficiently
        assert duration < 1.0

# Parametrized tests for comprehensive coverage
class TestParametrized:
    """Parametrized tests for comprehensive coverage."""
    
    @pytest.mark.asyncio
    @pytest.mark.parametrize("a,b,expected_equal,expected_less,expected_greater", [
        (5, 5, True, False, False),
        (3, 5, False, True, False),
        (5, 3, False, False, True),
        (-2, 1, False, True, False),
        (0, 0, True, False, False),
        (3.14, 3.14, True, False, False),
        (2.71, 3.14, False, True, False)
    ])
    async def test_basic_comparisons_parametrized(self, a, b, expected_equal, expected_less, expected_greater):
        """Parametrized test for basic comparison operations."""
        assert await equal(a, b) == expected_equal
        assert await not_equal(a, b) == (not expected_equal)
        assert await less_than(a, b) == expected_less
        assert await greater_than(a, b) == expected_greater
        assert await less_than_or_equal(a, b) == (expected_less or expected_equal)
        assert await greater_than_or_equal(a, b) == (expected_greater or expected_equal)
    
    @pytest.mark.asyncio
    @pytest.mark.parametrize("value,min_val,max_val,expected_inclusive,expected_exclusive", [
        (5, 1, 10, True, True),      # Within range
        (1, 1, 10, True, False),     # At lower bound
        (10, 1, 10, True, False),    # At upper bound
        (0, 1, 10, False, False),    # Below range
        (11, 1, 10, False, False),   # Above range
        (5, 5, 5, True, False),      # Zero-width range
        (3.14, 1.0, 10.0, True, True),  # Float within range
        (-2, -5, 5, True, True)      # Negative ranges
    ])
    async def test_range_operations_parametrized(self, value, min_val, max_val, expected_inclusive, expected_exclusive):
        """Parametrized test for range operations."""
        assert await in_range(value, min_val, max_val, inclusive=True) == expected_inclusive
        assert await in_range(value, min_val, max_val, inclusive=False) == expected_exclusive
        assert await between(value, min_val, max_val) == expected_exclusive

# Error handling tests
class TestErrorHandling:
    """Test error handling and edge cases."""
    
    @pytest.mark.asyncio
    async def test_in_range_invalid_bounds_error_message(self):
        """Test specific error message for invalid in_range bounds."""
        with pytest.raises(ValueError) as exc_info:
            await in_range(5, 10, 1)
        
        assert "Minimum value cannot be greater than maximum value" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_error_preserves_async_context(self):
        """Test that errors are properly raised in async context."""
        try:
            await in_range(5, 10, 1)
            assert False, "Should have raised ValueError"
        except ValueError:
            # Should be able to continue with async operations
            result = await equal(3, 3)
            assert result == True
    
    @pytest.mark.asyncio
    async def test_nan_behavior_consistency(self):
        """Test that NaN behavior is consistent across all comparison operations."""
        nan = float('nan')
        regular_value = 5
        
        # All comparison operations with NaN should return False except not_equal
        assert await equal(nan, regular_value) == False
        assert await equal(regular_value, nan) == False
        assert await equal(nan, nan) == False
        
        assert await not_equal(nan, regular_value) == True
        assert await not_equal(regular_value, nan) == True
        assert await not_equal(nan, nan) == True
        
        assert await less_than(nan, regular_value) == False
        assert await less_than(regular_value, nan) == False
        assert await less_than_or_equal(nan, regular_value) == False
        assert await less_than_or_equal(regular_value, nan) == False
        
        assert await greater_than(nan, regular_value) == False
        assert await greater_than(regular_value, nan) == False
        assert await greater_than_or_equal(nan, regular_value) == False
        assert await greater_than_or_equal(regular_value, nan) == False
        
        assert await in_range(nan, 1, 10) == False
        assert await between(nan, 1, 10) == False

if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v", "--tb=short"])