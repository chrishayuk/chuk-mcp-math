#!/usr/bin/env python3
# tests/math/arithmetic/core/test_modular.py
"""
Comprehensive pytest unit tests for core modular arithmetic operations.

Tests cover:
- Modular arithmetic operations (modulo, divmod, quotient)
- Modular exponentiation (mod_power)
- IEEE remainder and floating-point modulo
- Edge cases and error conditions
- Async behavior and performance
- Mathematical properties and invariants
"""

import pytest
import math
import asyncio
from typing import Union

# Import the functions to test
from chuk_mcp_math.arithmetic.core.modular import (
    modulo,
    divmod_operation,
    mod_power,
    quotient,
    remainder,
    fmod,
)

Number = Union[int, float]


class TestModulo:
    """Test cases for the modulo function."""

    @pytest.mark.asyncio
    async def test_modulo_positive_integers(self):
        """Test modulo with positive integers."""
        assert await modulo(17, 5) == 2
        assert await modulo(10, 3) == 1
        assert await modulo(15, 5) == 0
        assert await modulo(7, 3) == 1

    @pytest.mark.asyncio
    async def test_modulo_negative_dividend(self):
        """Test modulo with negative dividend."""
        # Python's modulo follows the sign of the divisor
        assert await modulo(-7, 3) == 2  # -7 % 3 = 2 in Python
        assert await modulo(-10, 3) == 2  # -10 % 3 = 2 in Python
        assert await modulo(-15, 5) == 0  # -15 % 5 = 0

    @pytest.mark.asyncio
    async def test_modulo_negative_divisor(self):
        """Test modulo with negative divisor."""
        assert await modulo(7, -3) == -2  # 7 % -3 = -2 in Python
        assert await modulo(10, -3) == -2  # 10 % -3 = -2 in Python
        assert await modulo(15, -5) == 0  # 15 % -5 = 0

    @pytest.mark.asyncio
    async def test_modulo_both_negative(self):
        """Test modulo with both negative numbers."""
        assert await modulo(-7, -3) == -1  # -7 % -3 = -1 in Python
        assert await modulo(-10, -3) == -1  # -10 % -3 = -1 in Python
        assert await modulo(-15, -5) == 0  # -15 % -5 = 0

    @pytest.mark.asyncio
    async def test_modulo_zero_dividend(self):
        """Test modulo with zero dividend."""
        assert await modulo(0, 5) == 0
        assert await modulo(0, -5) == 0
        assert await modulo(0, 1) == 0

    @pytest.mark.asyncio
    async def test_modulo_one_divisor(self):
        """Test modulo with divisor of one."""
        assert await modulo(17, 1) == 0
        assert await modulo(0, 1) == 0
        assert await modulo(-5, 1) == 0

    @pytest.mark.asyncio
    async def test_modulo_floats(self):
        """Test modulo with floating point numbers."""
        result = await modulo(7.5, 2.5)
        assert pytest.approx(result) == 0.0

        result = await modulo(10.3, 3.0)
        assert pytest.approx(result) == 1.3

    @pytest.mark.asyncio
    async def test_modulo_large_numbers(self):
        """Test modulo with large numbers."""
        assert await modulo(1000000, 7) == 1  # 1000000 % 7 = 1, not 6
        assert await modulo(1234567890, 123) == 39

    @pytest.mark.asyncio
    async def test_modulo_zero_divisor_raises(self):
        """Test that modulo by zero raises ValueError."""
        with pytest.raises(ValueError, match="Cannot calculate modulo with zero divisor"):
            await modulo(17, 0)

        with pytest.raises(ValueError, match="Cannot calculate modulo with zero divisor"):
            await modulo(-5, 0)


class TestDivmodOperation:
    """Test cases for the divmod_operation function."""

    @pytest.mark.asyncio
    async def test_divmod_positive_integers(self):
        """Test divmod with positive integers."""
        quotient, remainder = await divmod_operation(17, 5)
        assert quotient == 3
        assert remainder == 2

        quotient, remainder = await divmod_operation(20, 4)
        assert quotient == 5
        assert remainder == 0

    @pytest.mark.asyncio
    async def test_divmod_negative_dividend(self):
        """Test divmod with negative dividend."""
        quotient, remainder = await divmod_operation(-17, 5)
        assert quotient == -4
        assert remainder == 3

        quotient, remainder = await divmod_operation(-20, 4)
        assert quotient == -5
        assert remainder == 0

    @pytest.mark.asyncio
    async def test_divmod_negative_divisor(self):
        """Test divmod with negative divisor."""
        quotient, remainder = await divmod_operation(17, -5)
        assert quotient == -4
        assert remainder == -3

        quotient, remainder = await divmod_operation(20, -4)
        assert quotient == -5
        assert remainder == 0

    @pytest.mark.asyncio
    async def test_divmod_both_negative(self):
        """Test divmod with both negative numbers."""
        quotient, remainder = await divmod_operation(-17, -5)
        assert quotient == 3
        assert remainder == -2

    @pytest.mark.asyncio
    async def test_divmod_zero_dividend(self):
        """Test divmod with zero dividend."""
        quotient, remainder = await divmod_operation(0, 5)
        assert quotient == 0
        assert remainder == 0

    @pytest.mark.asyncio
    async def test_divmod_floats(self):
        """Test divmod with floating point numbers."""
        quotient, remainder = await divmod_operation(7.5, 2.5)
        assert quotient == 3
        assert pytest.approx(remainder) == 0.0

        quotient, remainder = await divmod_operation(10.7, 3.0)
        assert quotient == 3
        assert pytest.approx(remainder) == 1.7

    @pytest.mark.asyncio
    async def test_divmod_property(self):
        """Test that divmod satisfies the fundamental property: a = q*b + r."""
        test_cases = [
            (17, 5),
            (-17, 5),
            (17, -5),
            (-17, -5),
            (20, 4),
            (7, 3),
            (100, 7),
            (1000, 13),
        ]

        for a, b in test_cases:
            quotient, remainder = await divmod_operation(a, b)
            reconstructed = quotient * b + remainder
            assert reconstructed == a, (
                f"Failed for {a}, {b}: {quotient}*{b} + {remainder} = {reconstructed} != {a}"
            )

    @pytest.mark.asyncio
    async def test_divmod_zero_divisor_raises(self):
        """Test that divmod by zero raises ValueError."""
        with pytest.raises(ValueError, match="Cannot divide by zero"):
            await divmod_operation(17, 0)


class TestModPower:
    """Test cases for the mod_power function."""

    @pytest.mark.asyncio
    async def test_mod_power_basic_cases(self):
        """Test basic modular exponentiation cases."""
        assert await mod_power(2, 10, 1000) == 24  # 2^10 = 1024, 1024 % 1000 = 24
        assert await mod_power(3, 5, 7) == 5  # 3^5 = 243, 243 % 7 = 5
        assert await mod_power(7, 3, 10) == 3  # 7^3 = 343, 343 % 10 = 3

    @pytest.mark.asyncio
    async def test_mod_power_zero_exponent(self):
        """Test modular exponentiation with zero exponent."""
        assert await mod_power(5, 0, 13) == 1
        assert await mod_power(100, 0, 7) == 1
        assert await mod_power(2, 0, 1000) == 1

    @pytest.mark.asyncio
    async def test_mod_power_one_exponent(self):
        """Test modular exponentiation with exponent of one."""
        assert await mod_power(5, 1, 13) == 5
        assert await mod_power(7, 1, 10) == 7
        assert await mod_power(15, 1, 10) == 5  # 15 % 10 = 5

    @pytest.mark.asyncio
    async def test_mod_power_one_base(self):
        """Test modular exponentiation with base of one."""
        assert await mod_power(1, 100, 7) == 1
        assert await mod_power(1, 1000, 13) == 1
        assert await mod_power(1, 5, 10) == 1

    @pytest.mark.asyncio
    async def test_mod_power_large_exponent(self):
        """Test modular exponentiation with large exponent (should yield)."""
        # This should trigger async yielding
        result = await mod_power(2, 1001, 1000000)
        # Just verify it completes without error and is in correct range
        assert 0 <= result < 1000000

    @pytest.mark.asyncio
    async def test_mod_power_cryptographic_example(self):
        """Test a cryptographic-style example."""
        # Common in RSA: compute a^e mod n
        base, exponent, modulus = 123, 65537, 3233  # Small RSA-style example
        result = await mod_power(base, exponent, modulus)
        assert 0 <= result < modulus

        # Verify using Python's built-in pow
        expected = pow(base, exponent, modulus)
        assert result == expected

    @pytest.mark.asyncio
    async def test_mod_power_fermat_little_theorem(self):
        """Test using Fermat's Little Theorem: a^(p-1) ≡ 1 (mod p) for prime p."""
        # Test with prime p = 7 and a = 3
        p = 7
        a = 3
        result = await mod_power(a, p - 1, p)
        assert result == 1  # 3^6 mod 7 should be 1

        # Test with prime p = 13 and a = 5
        p = 13
        a = 5
        result = await mod_power(a, p - 1, p)
        assert result == 1  # 5^12 mod 13 should be 1

    @pytest.mark.asyncio
    async def test_mod_power_negative_exponent_raises(self):
        """Test that negative exponent raises ValueError."""
        with pytest.raises(ValueError, match="Exponent must be non-negative"):
            await mod_power(2, -1, 7)

    @pytest.mark.asyncio
    async def test_mod_power_non_positive_modulus_raises(self):
        """Test that non-positive modulus raises ValueError."""
        with pytest.raises(ValueError, match="Modulus must be positive"):
            await mod_power(2, 3, 0)

        with pytest.raises(ValueError, match="Modulus must be positive"):
            await mod_power(2, 3, -5)


class TestQuotient:
    """Test cases for the quotient function."""

    @pytest.mark.asyncio
    async def test_quotient_positive_integers(self):
        """Test quotient with positive integers."""
        assert await quotient(17, 5) == 3
        assert await quotient(20, 4) == 5
        assert await quotient(7, 3) == 2
        assert await quotient(15, 5) == 3

    @pytest.mark.asyncio
    async def test_quotient_negative_dividend(self):
        """Test quotient with negative dividend."""
        assert await quotient(-17, 5) == -4
        assert await quotient(-20, 4) == -5
        assert await quotient(-7, 3) == -3

    @pytest.mark.asyncio
    async def test_quotient_negative_divisor(self):
        """Test quotient with negative divisor."""
        assert await quotient(17, -5) == -4
        assert await quotient(20, -4) == -5
        assert await quotient(7, -3) == -3

    @pytest.mark.asyncio
    async def test_quotient_both_negative(self):
        """Test quotient with both negative numbers."""
        assert await quotient(-17, -5) == 3
        assert await quotient(-20, -4) == 5
        assert await quotient(-7, -3) == 2

    @pytest.mark.asyncio
    async def test_quotient_zero_dividend(self):
        """Test quotient with zero dividend."""
        assert await quotient(0, 5) == 0
        assert await quotient(0, -5) == 0
        assert await quotient(0, 1) == 0

    @pytest.mark.asyncio
    async def test_quotient_large_numbers(self):
        """Test quotient with large numbers."""
        assert await quotient(1000000, 7) == 142857
        assert await quotient(1234567890, 123) == 10037137

    @pytest.mark.asyncio
    async def test_quotient_zero_divisor_raises(self):
        """Test that quotient by zero raises ValueError."""
        with pytest.raises(ValueError, match="Cannot divide by zero"):
            await quotient(17, 0)


class TestRemainder:
    """Test cases for the IEEE remainder function."""

    @pytest.mark.asyncio
    async def test_remainder_basic_cases(self):
        """Test basic IEEE remainder cases."""
        result = await remainder(7.5, 2.5)
        assert pytest.approx(result) == 0.0

        result = await remainder(5.0, 2.0)
        assert pytest.approx(result) == 1.0

        result = await remainder(6.0, 3.0)
        assert pytest.approx(result) == 0.0

    @pytest.mark.asyncio
    async def test_remainder_negative_numbers(self):
        """Test IEEE remainder with negative numbers."""
        result = await remainder(-7.5, 2.5)
        assert pytest.approx(result) == 0.0

        result = await remainder(7.5, -2.5)
        assert pytest.approx(result) == 0.0

        result = await remainder(-7.5, -2.5)
        assert pytest.approx(result) == 0.0

    @pytest.mark.asyncio
    async def test_remainder_vs_python_remainder(self):
        """Test that our remainder matches Python's math.remainder."""
        test_cases = [
            (10.3, 3.0),
            (7.5, 2.5),
            (-7.5, 2.5),
            (5.7, 2.0),
            (9.5, 3.5),
            (-10.3, 3.0),
        ]

        for x, y in test_cases:
            our_result = await remainder(x, y)
            python_result = math.remainder(x, y)
            assert pytest.approx(our_result, rel=1e-15) == python_result

    @pytest.mark.asyncio
    async def test_remainder_zero_divisor_raises(self):
        """Test that remainder by zero raises ValueError."""
        with pytest.raises(ValueError, match="Cannot calculate remainder with zero divisor"):
            await remainder(7.5, 0.0)


class TestFmod:
    """Test cases for the floating-point modulo function."""

    @pytest.mark.asyncio
    async def test_fmod_basic_cases(self):
        """Test basic floating-point modulo cases."""
        result = await fmod(7.5, 2.5)
        assert pytest.approx(result) == 0.0

        result = await fmod(5.7, 2.0)
        assert pytest.approx(result, rel=1e-14) == 1.7

    @pytest.mark.asyncio
    async def test_fmod_negative_numbers(self):
        """Test fmod with negative numbers."""
        result = await fmod(-7.5, 2.5)
        assert pytest.approx(result) == -0.0  # Note: -0.0 is distinct from 0.0

        result = await fmod(7.5, -2.5)
        assert pytest.approx(result) == 0.0

    @pytest.mark.asyncio
    async def test_fmod_vs_python_fmod(self):
        """Test that our fmod matches Python's math.fmod."""
        test_cases = [
            (10.3, 3.0),
            (7.5, 2.5),
            (-7.5, 2.5),
            (5.7, 2.0),
            (9.5, 3.5),
            (-10.3, 3.0),
        ]

        for x, y in test_cases:
            our_result = await fmod(x, y)
            python_result = math.fmod(x, y)
            assert pytest.approx(our_result, rel=1e-14) == python_result

    @pytest.mark.asyncio
    async def test_fmod_precision_example(self):
        """Test fmod with precision example from docstring."""
        result = await fmod(10.3, 3.0)
        # This should be approximately 1.3, but floating point precision means it's not exact
        assert pytest.approx(result, rel=1e-14) == 1.2999999999999998

    @pytest.mark.asyncio
    async def test_fmod_zero_divisor_raises(self):
        """Test that fmod by zero raises ValueError."""
        with pytest.raises(ValueError, match="Cannot calculate fmod with zero divisor"):
            await fmod(7.5, 0.0)


class TestIntegration:
    """Integration tests for modular arithmetic operations."""

    @pytest.mark.asyncio
    async def test_modulo_divmod_consistency(self):
        """Test that modulo and divmod are consistent."""
        test_cases = [(17, 5), (-17, 5), (17, -5), (-17, -5), (20, 4), (7, 3), (100, 7)]

        for a, b in test_cases:
            mod_result = await modulo(a, b)
            quotient_result, remainder_result = await divmod_operation(a, b)

            assert mod_result == remainder_result, f"Inconsistent for {a} % {b}"

    @pytest.mark.asyncio
    async def test_quotient_divmod_consistency(self):
        """Test that quotient and divmod quotient are consistent."""
        test_cases = [(17, 5), (-17, 5), (17, -5), (-17, -5), (20, 4), (7, 3), (100, 7)]

        for a, b in test_cases:
            quotient_result = await quotient(a, b)
            divmod_quotient, _ = await divmod_operation(a, b)

            assert quotient_result == divmod_quotient, f"Inconsistent for {a} // {b}"

    @pytest.mark.asyncio
    async def test_mod_power_property(self):
        """Test modular exponentiation properties."""
        # Test: (a * b) mod m = ((a mod m) * (b mod m)) mod m
        a, b, m = 123, 456, 789

        left_side = await modulo(a * b, m)

        a_mod = await modulo(a, m)
        b_mod = await modulo(b, m)
        right_side = await modulo(a_mod * b_mod, m)

        assert left_side == right_side

    @pytest.mark.asyncio
    async def test_remainder_vs_fmod_differences(self):
        """Test the differences between IEEE remainder and fmod."""
        # For some inputs, remainder and fmod give the same result
        x, y = 6.0, 3.0
        remainder_result = await remainder(x, y)
        fmod_result = await fmod(x, y)
        assert pytest.approx(remainder_result) == pytest.approx(fmod_result)  # Both should be 0.0

        # But for other cases, they can differ significantly
        # IEEE remainder can return negative values when fmod returns positive
        x, y = 5.5, 2.0
        remainder_result = await remainder(x, y)  # IEEE remainder: -0.5
        fmod_result = await fmod(x, y)  # fmod: 1.5

        # Verify they are different (this is expected behavior)
        assert remainder_result != fmod_result
        assert pytest.approx(remainder_result) == -0.5
        assert pytest.approx(fmod_result) == 1.5

        # Test case where they agree
        x, y = 7.0, 3.0
        remainder_result = await remainder(x, y)  # Should be 1.0
        fmod_result = await fmod(x, y)  # Should also be 1.0
        assert pytest.approx(remainder_result) == 1.0
        assert pytest.approx(fmod_result) == 1.0


class TestPerformance:
    """Performance and async behavior tests."""

    @pytest.mark.asyncio
    async def test_all_operations_are_async(self):
        """Test that all modular operations are properly async."""
        operations = [
            modulo(17, 5),
            divmod_operation(17, 5),
            mod_power(2, 10, 1000),
            quotient(17, 5),
            remainder(7.5, 2.5),
            fmod(7.5, 2.5),
        ]

        # Ensure all are coroutines
        for op in operations:
            assert asyncio.iscoroutine(op)

        # Run all operations concurrently
        results = await asyncio.gather(*operations)
        expected = [2, (3, 2), 24, 3, 0.0, 0.0]

        for result, expected_val in zip(results, expected):
            if isinstance(expected_val, tuple):
                assert result == expected_val
            elif isinstance(expected_val, float):
                assert pytest.approx(result) == expected_val
            else:
                assert result == expected_val

    @pytest.mark.asyncio
    async def test_mod_power_async_yielding(self):
        """Test that mod_power yields for large exponents."""
        import time

        start_time = time.time()
        result = await mod_power(2, 2000, 1000000)  # Large exponent
        duration = time.time() - start_time

        # Should complete in reasonable time
        assert duration < 5.0
        assert 0 <= result < 1000000


# Parametrized tests for comprehensive coverage
class TestParametrized:
    """Parametrized tests for comprehensive coverage."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "a,b,expected",
        [
            (17, 5, 2),
            (10, 3, 1),
            (15, 5, 0),
            (-7, 3, 2),
            (7, -3, -2),
            (-7, -3, -1),
            (0, 5, 0),
            (100, 7, 2),
        ],
    )
    async def test_modulo_parametrized(self, a, b, expected):
        """Parametrized test for modulo function."""
        result = await modulo(a, b)
        assert result == expected

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "base,exp,mod,expected",
        [(2, 10, 1000, 24), (3, 5, 7, 5), (5, 0, 13, 1), (7, 3, 10, 3), (1, 100, 7, 1)],
    )
    async def test_mod_power_parametrized(self, base, exp, mod, expected):
        """Parametrized test for mod_power function."""
        result = await mod_power(base, exp, mod)
        assert result == expected


# Error handling tests
class TestErrorHandling:
    """Test error handling and edge cases."""

    @pytest.mark.asyncio
    async def test_division_by_zero_errors(self):
        """Test that all division-by-zero cases raise appropriate errors."""
        functions_and_args = [
            (modulo, (17, 0)),
            (divmod_operation, (17, 0)),
            (quotient, (17, 0)),
            (remainder, (7.5, 0.0)),
            (fmod, (7.5, 0.0)),
        ]

        for func, args in functions_and_args:
            with pytest.raises(ValueError):
                await func(*args)

    @pytest.mark.asyncio
    async def test_mod_power_validation_errors(self):
        """Test mod_power input validation."""
        # Negative exponent
        with pytest.raises(ValueError, match="Exponent must be non-negative"):
            await mod_power(2, -1, 7)

        # Zero modulus
        with pytest.raises(ValueError, match="Modulus must be positive"):
            await mod_power(2, 3, 0)

        # Negative modulus
        with pytest.raises(ValueError, match="Modulus must be positive"):
            await mod_power(2, 3, -5)


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v", "--tb=short"])
