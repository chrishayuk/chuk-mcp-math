#!/usr/bin/env python3
# tests/math/arithmetic/core/test_basic_operations.py
"""
Comprehensive pytest unit tests for core basic arithmetic operations.

Tests cover:
- Normal operation cases
- Edge cases (zero, negative, infinity, NaN)
- Error conditions
- Async behavior
- Type validation
- Performance characteristics
- MCP compliance
"""

import pytest
import math
import asyncio
from typing import Union

# Import the functions to test
from chuk_mcp_math.arithmetic.core.basic_operations import (
    add,
    subtract,
    multiply,
    divide,
    power,
    sqrt,
    abs_value,
    sign,
    negate,
)

Number = Union[int, float]


class TestAdd:
    """Test cases for the add function."""

    @pytest.mark.asyncio
    async def test_add_positive_integers(self):
        """Test addition of positive integers."""
        result = await add(5, 3)
        assert result == 8
        assert isinstance(result, int)

    @pytest.mark.asyncio
    async def test_add_negative_integers(self):
        """Test addition of negative integers."""
        result = await add(-5, -3)
        assert result == -8
        assert isinstance(result, int)

    @pytest.mark.asyncio
    async def test_add_mixed_sign_integers(self):
        """Test addition of mixed sign integers."""
        result = await add(-5, 3)
        assert result == -2
        assert isinstance(result, int)

    @pytest.mark.asyncio
    async def test_add_floats(self):
        """Test addition of floating point numbers."""
        result = await add(2.5, 3.7)
        assert pytest.approx(result) == 6.2
        assert isinstance(result, float)

    @pytest.mark.asyncio
    async def test_add_mixed_types(self):
        """Test addition of integer and float."""
        result = await add(5, 3.2)
        assert pytest.approx(result) == 8.2
        assert isinstance(result, float)

    @pytest.mark.asyncio
    async def test_add_zero(self):
        """Test addition with zero."""
        assert await add(5, 0) == 5
        assert await add(0, 5) == 5
        assert await add(0, 0) == 0

    @pytest.mark.asyncio
    async def test_add_large_numbers(self):
        """Test addition of large numbers."""
        result = await add(1e15, 1e15)
        assert result == 2e15

    @pytest.mark.asyncio
    async def test_add_very_small_numbers(self):
        """Test addition of very small numbers."""
        result = await add(1e-15, 1e-15)
        assert pytest.approx(result) == 2e-15

    @pytest.mark.asyncio
    async def test_add_infinity(self):
        """Test addition with infinity."""
        inf = float("inf")
        assert await add(inf, 5) == inf
        assert await add(5, inf) == inf
        assert await add(inf, inf) == inf

    @pytest.mark.asyncio
    async def test_add_negative_infinity(self):
        """Test addition with negative infinity."""
        neg_inf = float("-inf")
        assert await add(neg_inf, 5) == neg_inf
        assert await add(5, neg_inf) == neg_inf
        assert await add(neg_inf, neg_inf) == neg_inf

    @pytest.mark.asyncio
    async def test_add_infinity_mixed(self):
        """Test addition of positive and negative infinity."""
        result = await add(float("inf"), float("-inf"))
        assert math.isnan(result)

    @pytest.mark.asyncio
    async def test_add_nan(self):
        """Test addition with NaN."""
        nan = float("nan")
        result1 = await add(nan, 5)
        result2 = await add(5, nan)
        result3 = await add(nan, nan)

        assert math.isnan(result1)
        assert math.isnan(result2)
        assert math.isnan(result3)

    @pytest.mark.asyncio
    async def test_add_precision(self):
        """Test addition precision with floating point arithmetic."""
        # Test case that might have precision issues
        result = await add(0.1, 0.2)
        assert pytest.approx(result, rel=1e-9) == 0.3


class TestSubtract:
    """Test cases for the subtract function."""

    @pytest.mark.asyncio
    async def test_subtract_positive_integers(self):
        """Test subtraction of positive integers."""
        result = await subtract(10, 3)
        assert result == 7
        assert isinstance(result, int)

    @pytest.mark.asyncio
    async def test_subtract_negative_result(self):
        """Test subtraction resulting in negative number."""
        result = await subtract(3, 10)
        assert result == -7
        assert isinstance(result, int)

    @pytest.mark.asyncio
    async def test_subtract_same_numbers(self):
        """Test subtraction of identical numbers."""
        assert await subtract(5, 5) == 0
        assert await subtract(-3, -3) == 0
        assert await subtract(0, 0) == 0

    @pytest.mark.asyncio
    async def test_subtract_floats(self):
        """Test subtraction of floating point numbers."""
        result = await subtract(5.5, 2.3)
        assert pytest.approx(result) == 3.2

    @pytest.mark.asyncio
    async def test_subtract_with_zero(self):
        """Test subtraction with zero."""
        assert await subtract(5, 0) == 5
        assert await subtract(0, 5) == -5

    @pytest.mark.asyncio
    async def test_subtract_infinity(self):
        """Test subtraction with infinity."""
        inf = float("inf")
        neg_inf = float("-inf")

        assert await subtract(inf, 5) == inf
        assert await subtract(5, inf) == neg_inf
        result = await subtract(inf, inf)
        assert math.isnan(result)


class TestMultiply:
    """Test cases for the multiply function."""

    @pytest.mark.asyncio
    async def test_multiply_positive_integers(self):
        """Test multiplication of positive integers."""
        result = await multiply(6, 7)
        assert result == 42
        assert isinstance(result, int)

    @pytest.mark.asyncio
    async def test_multiply_with_zero(self):
        """Test multiplication with zero."""
        assert await multiply(5, 0) == 0
        assert await multiply(0, 5) == 0
        assert await multiply(0, 0) == 0

    @pytest.mark.asyncio
    async def test_multiply_with_one(self):
        """Test multiplication with one."""
        assert await multiply(5, 1) == 5
        assert await multiply(1, 5) == 5

    @pytest.mark.asyncio
    async def test_multiply_negative_numbers(self):
        """Test multiplication with negative numbers."""
        assert await multiply(-3, 4) == -12
        assert await multiply(3, -4) == -12
        assert await multiply(-3, -4) == 12

    @pytest.mark.asyncio
    async def test_multiply_floats(self):
        """Test multiplication of floating point numbers."""
        result = await multiply(2.5, 4.0)
        assert pytest.approx(result) == 10.0

    @pytest.mark.asyncio
    async def test_multiply_mixed_types(self):
        """Test multiplication of mixed types."""
        result = await multiply(2.5, 4)
        assert pytest.approx(result) == 10.0
        assert isinstance(result, float)

    @pytest.mark.asyncio
    async def test_multiply_large_numbers(self):
        """Test multiplication of large numbers."""
        result = await multiply(1e10, 1e10)
        assert result == 1e20

    @pytest.mark.asyncio
    async def test_multiply_infinity(self):
        """Test multiplication with infinity."""
        inf = float("inf")
        assert await multiply(inf, 5) == inf
        assert await multiply(5, inf) == inf
        assert await multiply(inf, inf) == inf

        # Zero times infinity is NaN
        result = await multiply(0, inf)
        assert math.isnan(result)


class TestDivide:
    """Test cases for the divide function."""

    @pytest.mark.asyncio
    async def test_divide_integers(self):
        """Test division of integers."""
        result = await divide(15, 3)
        assert result == 5.0
        assert isinstance(result, float)

    @pytest.mark.asyncio
    async def test_divide_with_remainder(self):
        """Test division with decimal result."""
        result = await divide(7, 2)
        assert result == 3.5

    @pytest.mark.asyncio
    async def test_divide_floats(self):
        """Test division of floating point numbers."""
        result = await divide(7.5, 2.5)
        assert result == 3.0

    @pytest.mark.asyncio
    async def test_divide_negative_numbers(self):
        """Test division with negative numbers."""
        assert await divide(-10, 2) == -5.0
        assert await divide(10, -2) == -5.0
        assert await divide(-10, -2) == 5.0

    @pytest.mark.asyncio
    async def test_divide_by_zero_raises_error(self):
        """Test that division by zero raises ValueError."""
        with pytest.raises(ValueError, match="Cannot divide by zero"):
            await divide(5, 0)

        with pytest.raises(ValueError, match="Cannot divide by zero"):
            await divide(-5, 0)

        with pytest.raises(ValueError, match="Cannot divide by zero"):
            await divide(0, 0)

    @pytest.mark.asyncio
    async def test_divide_zero_by_number(self):
        """Test division of zero by non-zero number."""
        assert await divide(0, 5) == 0.0
        assert await divide(0, -5) == 0.0

    @pytest.mark.asyncio
    async def test_divide_infinity(self):
        """Test division with infinity."""
        inf = float("inf")

        # Number divided by infinity
        result = await divide(5, inf)
        assert result == 0.0

        # Infinity divided by number
        result = await divide(inf, 5)
        assert result == inf

        # Infinity divided by infinity
        result = await divide(inf, inf)
        assert math.isnan(result)

    @pytest.mark.asyncio
    async def test_divide_very_small_number(self):
        """Test division by very small number (near zero but not zero)."""
        result = await divide(1, 1e-10)
        assert result == 1e10


class TestPower:
    """Test cases for the power function."""

    @pytest.mark.asyncio
    async def test_power_positive_integer_exponent(self):
        """Test power with positive integer exponent."""
        result = await power(2, 3)
        assert result == 8
        assert isinstance(result, int)

    @pytest.mark.asyncio
    async def test_power_zero_exponent(self):
        """Test power with zero exponent."""
        assert await power(5, 0) == 1
        assert await power(0, 0) == 1  # 0^0 is defined as 1 in Python
        assert await power(-3, 0) == 1

    @pytest.mark.asyncio
    async def test_power_one_exponent(self):
        """Test power with exponent of one."""
        assert await power(5, 1) == 5
        assert await power(-3, 1) == -3
        assert await power(0, 1) == 0

    @pytest.mark.asyncio
    async def test_power_negative_exponent(self):
        """Test power with negative exponent."""
        result = await power(2, -3)
        assert pytest.approx(result) == 0.125

    @pytest.mark.asyncio
    async def test_power_fractional_exponent(self):
        """Test power with fractional exponent."""
        result = await power(4, 0.5)
        assert pytest.approx(result) == 2.0

        result = await power(8, 1 / 3)
        assert pytest.approx(result, rel=1e-10) == 2.0

    @pytest.mark.asyncio
    async def test_power_negative_base(self):
        """Test power with negative base."""
        assert await power(-2, 3) == -8
        assert await power(-2, 2) == 4
        assert await power(-1, 100) == 1
        assert await power(-1, 101) == -1

    @pytest.mark.asyncio
    async def test_power_zero_base(self):
        """Test power with zero base."""
        assert await power(0, 5) == 0
        assert await power(0, 1) == 0

        # 0^0 case
        assert await power(0, 0) == 1

    @pytest.mark.asyncio
    async def test_power_large_exponent(self):
        """Test power with large exponent (should yield control)."""
        # This should trigger the async yield
        result = await power(2, 1001)
        expected = 2**1001
        assert result == expected

    @pytest.mark.asyncio
    async def test_power_infinity(self):
        """Test power with infinity."""
        inf = float("inf")

        # Positive base to infinity
        assert await power(2, inf) == inf
        assert await power(0.5, inf) == 0.0

        # Infinity to positive power
        assert await power(inf, 2) == inf

        # Special cases - Python returns 1.0 for 1^inf
        result = await power(1, inf)
        assert result == 1.0  # Python's behavior for 1^inf

    @pytest.mark.asyncio
    async def test_power_async_behavior(self):
        """Test that power function works properly in async context."""
        import time

        # Test that we can run multiple power operations concurrently
        start_time = time.time()

        # Run several power operations concurrently including one with large exponent
        tasks = [
            power(2, 100),
            power(3, 50),
            power(5, 30),
            power(2, 1001),  # This should handle large exponent properly
        ]

        results = await asyncio.gather(*tasks)
        duration = time.time() - start_time

        # Verify results are correct
        assert results[0] == 2**100
        assert results[1] == 3**50
        assert results[2] == 5**30
        assert results[3] == 2**1001

        # Should complete in reasonable time
        assert duration < 5.0


class TestSqrt:
    """Test cases for the sqrt function."""

    @pytest.mark.asyncio
    async def test_sqrt_perfect_squares(self):
        """Test square root of perfect squares."""
        assert await sqrt(0) == 0.0
        assert await sqrt(1) == 1.0
        assert await sqrt(4) == 2.0
        assert await sqrt(9) == 3.0
        assert await sqrt(16) == 4.0
        assert await sqrt(25) == 5.0

    @pytest.mark.asyncio
    async def test_sqrt_non_perfect_squares(self):
        """Test square root of non-perfect squares."""
        result = await sqrt(2)
        assert pytest.approx(result, rel=1e-10) == math.sqrt(2)

        result = await sqrt(3)
        assert pytest.approx(result, rel=1e-10) == math.sqrt(3)

    @pytest.mark.asyncio
    async def test_sqrt_decimal_numbers(self):
        """Test square root of decimal numbers."""
        result = await sqrt(2.25)
        assert pytest.approx(result) == 1.5

        result = await sqrt(0.25)
        assert pytest.approx(result) == 0.5

    @pytest.mark.asyncio
    async def test_sqrt_large_numbers(self):
        """Test square root of large numbers."""
        result = await sqrt(1e10)
        assert pytest.approx(result) == 1e5

    @pytest.mark.asyncio
    async def test_sqrt_small_numbers(self):
        """Test square root of small numbers."""
        result = await sqrt(1e-10)
        assert pytest.approx(result) == 1e-5

    @pytest.mark.asyncio
    async def test_sqrt_negative_raises_error(self):
        """Test that square root of negative number raises ValueError."""
        with pytest.raises(
            ValueError, match="Cannot calculate square root of negative number"
        ):
            await sqrt(-1)

        with pytest.raises(
            ValueError, match="Cannot calculate square root of negative number"
        ):
            await sqrt(-0.1)

    @pytest.mark.asyncio
    async def test_sqrt_infinity(self):
        """Test square root of infinity."""
        result = await sqrt(float("inf"))
        assert result == float("inf")

    @pytest.mark.asyncio
    async def test_sqrt_return_type(self):
        """Test that sqrt always returns float."""
        result = await sqrt(4)
        assert isinstance(result, float)
        assert result == 2.0


class TestAbsValue:
    """Test cases for the abs_value function."""

    @pytest.mark.asyncio
    async def test_abs_positive_numbers(self):
        """Test absolute value of positive numbers."""
        assert await abs_value(5) == 5
        assert await abs_value(3.7) == 3.7
        assert await abs_value(0.1) == 0.1

    @pytest.mark.asyncio
    async def test_abs_negative_numbers(self):
        """Test absolute value of negative numbers."""
        assert await abs_value(-5) == 5
        assert await abs_value(-3.7) == 3.7
        assert await abs_value(-0.1) == 0.1

    @pytest.mark.asyncio
    async def test_abs_zero(self):
        """Test absolute value of zero."""
        assert await abs_value(0) == 0
        assert await abs_value(-0) == 0
        assert await abs_value(0.0) == 0.0

    @pytest.mark.asyncio
    async def test_abs_large_numbers(self):
        """Test absolute value of large numbers."""
        assert await abs_value(1e15) == 1e15
        assert await abs_value(-1e15) == 1e15

    @pytest.mark.asyncio
    async def test_abs_small_numbers(self):
        """Test absolute value of small numbers."""
        assert await abs_value(1e-15) == 1e-15
        assert await abs_value(-1e-15) == 1e-15

    @pytest.mark.asyncio
    async def test_abs_infinity(self):
        """Test absolute value of infinity."""
        assert await abs_value(float("inf")) == float("inf")
        assert await abs_value(float("-inf")) == float("inf")

    @pytest.mark.asyncio
    async def test_abs_nan(self):
        """Test absolute value of NaN."""
        result = await abs_value(float("nan"))
        assert math.isnan(result)

    @pytest.mark.asyncio
    async def test_abs_preserves_type(self):
        """Test that absolute value preserves number type."""
        result_int = await abs_value(-5)
        assert isinstance(result_int, int)

        result_float = await abs_value(-5.0)
        assert isinstance(result_float, float)


class TestSign:
    """Test cases for the sign function."""

    @pytest.mark.asyncio
    async def test_sign_positive_numbers(self):
        """Test sign of positive numbers."""
        assert await sign(5) == 1
        assert await sign(0.1) == 1
        assert await sign(1e10) == 1
        assert await sign(1e-10) == 1

    @pytest.mark.asyncio
    async def test_sign_negative_numbers(self):
        """Test sign of negative numbers."""
        assert await sign(-5) == -1
        assert await sign(-0.1) == -1
        assert await sign(-1e10) == -1
        assert await sign(-1e-10) == -1

    @pytest.mark.asyncio
    async def test_sign_zero(self):
        """Test sign of zero."""
        assert await sign(0) == 0
        assert await sign(0.0) == 0
        assert await sign(-0.0) == 0

    @pytest.mark.asyncio
    async def test_sign_infinity(self):
        """Test sign of infinity."""
        assert await sign(float("inf")) == 1
        assert await sign(float("-inf")) == -1

    @pytest.mark.asyncio
    async def test_sign_nan(self):
        """Test sign of NaN."""
        # NaN comparisons are always False, so sign should be 0
        result = await sign(float("nan"))
        assert result == 0

    @pytest.mark.asyncio
    async def test_sign_return_type(self):
        """Test that sign always returns int."""
        assert isinstance(await sign(5.5), int)
        assert isinstance(await sign(-5.5), int)
        assert isinstance(await sign(0.0), int)


class TestNegate:
    """Test cases for the negate function."""

    @pytest.mark.asyncio
    async def test_negate_positive_numbers(self):
        """Test negation of positive numbers."""
        assert await negate(5) == -5
        assert await negate(3.7) == -3.7
        assert await negate(0.1) == -0.1

    @pytest.mark.asyncio
    async def test_negate_negative_numbers(self):
        """Test negation of negative numbers."""
        assert await negate(-5) == 5
        assert await negate(-3.7) == 3.7
        assert await negate(-0.1) == 0.1

    @pytest.mark.asyncio
    async def test_negate_zero(self):
        """Test negation of zero."""
        assert await negate(0) == 0
        assert await negate(0.0) == 0.0
        # Note: -0.0 in Python is still 0.0

    @pytest.mark.asyncio
    async def test_negate_large_numbers(self):
        """Test negation of large numbers."""
        assert await negate(1e15) == -1e15
        assert await negate(-1e15) == 1e15

    @pytest.mark.asyncio
    async def test_negate_infinity(self):
        """Test negation of infinity."""
        assert await negate(float("inf")) == float("-inf")
        assert await negate(float("-inf")) == float("inf")

    @pytest.mark.asyncio
    async def test_negate_nan(self):
        """Test negation of NaN."""
        result = await negate(float("nan"))
        assert math.isnan(result)

    @pytest.mark.asyncio
    async def test_negate_preserves_type(self):
        """Test that negation preserves number type."""
        result_int = await negate(5)
        assert isinstance(result_int, int)

        result_float = await negate(5.0)
        assert isinstance(result_float, float)

    @pytest.mark.asyncio
    async def test_double_negation(self):
        """Test that double negation returns original value."""
        original = 5
        result = await negate(await negate(original))
        assert result == original

        original_float = 5.7
        result_float = await negate(await negate(original_float))
        assert result_float == original_float


# Integration and property-based tests
class TestIntegration:
    """Integration tests for multiple operations."""

    @pytest.mark.asyncio
    async def test_arithmetic_operations_integration(self):
        """Test combination of arithmetic operations."""
        # Test: (5 + 3) * 2 - 4 = 12
        step1 = await add(5, 3)  # 8
        step2 = await multiply(step1, 2)  # 16
        result = await subtract(step2, 4)  # 12
        assert result == 12

    @pytest.mark.asyncio
    async def test_power_and_sqrt_inverse(self):
        """Test that sqrt and power are inverse operations."""
        original = 9
        squared = await power(original, 2)  # 81
        sqrt_result = await sqrt(squared)  # 9
        assert pytest.approx(sqrt_result) == original

    @pytest.mark.asyncio
    async def test_abs_and_negate_properties(self):
        """Test properties of absolute value and negation."""
        x = -5

        # |x| = |-x|
        abs_x = await abs_value(x)
        neg_x = await negate(x)
        abs_neg_x = await abs_value(neg_x)
        assert abs_x == abs_neg_x

        # |x| ≥ 0
        assert abs_x >= 0


class TestPerformance:
    """Performance and async behavior tests."""

    @pytest.mark.asyncio
    async def test_all_operations_are_async(self):
        """Test that all operations are properly async."""
        # All these should be coroutines
        operations = [
            add(1, 2),
            subtract(5, 3),
            multiply(2, 3),
            divide(6, 2),
            power(2, 3),
            sqrt(9),
            abs_value(-5),
            sign(5),
            negate(5),
        ]

        # Ensure all are coroutines
        for op in operations:
            assert asyncio.iscoroutine(op)

        # Run all operations concurrently
        results = await asyncio.gather(*operations)
        expected = [3, 2, 6, 3.0, 8, 3.0, 5, 1, -5]

        for result, expected_val in zip(results, expected):
            if isinstance(expected_val, float):
                assert pytest.approx(result) == expected_val
            else:
                assert result == expected_val

    @pytest.mark.asyncio
    async def test_concurrent_execution(self):
        """Test that operations can run concurrently."""
        import time

        async def timed_operation():
            start = time.time()
            # Run multiple operations concurrently
            tasks = [add(i, i + 1) for i in range(100)]
            await asyncio.gather(*tasks)
            return time.time() - start

        # This should complete quickly due to async nature
        duration = await timed_operation()
        assert duration < 1.0  # Should be much faster than 1 second


# Fixtures for reusable test data
@pytest.fixture
def sample_numbers():
    """Provide sample numbers for testing."""
    return [0, 1, -1, 5, -5, 3.14, -3.14, 1e10, -1e10, 1e-10, -1e-10]


@pytest.fixture
def special_values():
    """Provide special floating point values."""
    return [float("inf"), float("-inf"), float("nan"), 0.0, -0.0]


# Parametrized tests for comprehensive coverage
class TestParametrized:
    """Parametrized tests for comprehensive coverage."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "a,b,expected",
        [
            (0, 0, 0),
            (1, 0, 1),
            (0, 1, 1),
            (5, 3, 8),
            (-5, 3, -2),
            (5, -3, 2),
            (-5, -3, -8),
            (1.5, 2.5, 4.0),
            (1e10, 1e10, 2e10),
        ],
    )
    async def test_add_parametrized(self, a, b, expected):
        """Parametrized test for add function."""
        result = await add(a, b)
        if isinstance(expected, float):
            assert pytest.approx(result) == expected
        else:
            assert result == expected

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "x,expected",
        [(5, 5), (-5, 5), (0, 0), (3.7, 3.7), (-3.7, 3.7), (1e10, 1e10), (-1e10, 1e10)],
    )
    async def test_abs_value_parametrized(self, x, expected):
        """Parametrized test for abs_value function."""
        result = await abs_value(x)
        assert result == expected

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "x,expected",
        [(5, 1), (-5, -1), (0, 0), (0.1, 1), (-0.1, -1), (1e10, 1), (-1e10, -1)],
    )
    async def test_sign_parametrized(self, x, expected):
        """Parametrized test for sign function."""
        result = await sign(x)
        assert result == expected


# Error handling tests
class TestErrorHandling:
    """Test error handling and edge cases."""

    @pytest.mark.asyncio
    async def test_division_by_zero_error_message(self):
        """Test that division by zero has appropriate error message."""
        with pytest.raises(ValueError) as exc_info:
            await divide(5, 0)

        assert "Cannot divide by zero" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_sqrt_negative_error_message(self):
        """Test that sqrt of negative has appropriate error message."""
        with pytest.raises(ValueError) as exc_info:
            await sqrt(-1)

        assert "Cannot calculate square root of negative number" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_error_preserves_async_context(self):
        """Test that errors don't break async context."""
        try:
            await divide(1, 0)
            assert False, "Should have raised an error"
        except ValueError:
            # Should still be able to perform async operations
            result = await add(1, 2)
            assert result == 3


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v", "--tb=short"])
