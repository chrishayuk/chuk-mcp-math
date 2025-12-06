#!/usr/bin/env python3
"""
Comprehensive tests for hyperbolic functions module.
Tests sinh, cosh, tanh and their reciprocals.
"""

import pytest
import math
from chuk_mcp_math.trigonometry.hyperbolic import (
    sinh,
    cosh,
    tanh,
    csch,
    sech,
    coth,
    hyperbolic_functions,
    verify_hyperbolic_identity,
    catenary_curve,
)


class TestPrimaryHyperbolicFunctions:
    """Test primary hyperbolic functions."""

    @pytest.mark.asyncio
    async def test_sinh_zero(self):
        """Test sinh(0) = 0."""
        assert abs(await sinh(0)) < 1e-15

    @pytest.mark.asyncio
    async def test_sinh_known_values(self):
        """Test sinh at known values."""
        # sinh(1) ≈ 1.175201194
        assert abs(await sinh(1) - 1.1752011936438014) < 1e-10

        # sinh(ln(2)) = 3/4
        assert abs(await sinh(math.log(2)) - 0.75) < 1e-10

    @pytest.mark.asyncio
    async def test_sinh_odd_function(self):
        """Test that sinh is odd: sinh(-x) = -sinh(x)."""
        test_values = [0.5, 1.0, 2.0, 3.0]
        for x in test_values:
            sinh_pos = await sinh(x)
            sinh_neg = await sinh(-x)
            assert abs(sinh_pos + sinh_neg) < 1e-10

    @pytest.mark.asyncio
    async def test_sinh_large_values(self):
        """Test sinh with large values."""
        # Should handle large values without overflow
        result = await sinh(100)
        assert result > 1e40
        assert math.isfinite(result)

    @pytest.mark.asyncio
    async def test_cosh_zero(self):
        """Test cosh(0) = 1."""
        assert abs(await cosh(0) - 1.0) < 1e-15

    @pytest.mark.asyncio
    async def test_cosh_known_values(self):
        """Test cosh at known values."""
        # cosh(1) ≈ 1.543080635
        assert abs(await cosh(1) - 1.5430806348152437) < 1e-10

        # cosh(ln(2)) = 5/4
        assert abs(await cosh(math.log(2)) - 1.25) < 1e-10

    @pytest.mark.asyncio
    async def test_cosh_even_function(self):
        """Test that cosh is even: cosh(-x) = cosh(x)."""
        test_values = [0.5, 1.0, 2.0, 3.0]
        for x in test_values:
            cosh_pos = await cosh(x)
            cosh_neg = await cosh(-x)
            assert abs(cosh_pos - cosh_neg) < 1e-10

    @pytest.mark.asyncio
    async def test_cosh_always_positive(self):
        """Test that cosh(x) >= 1 for all x."""
        test_values = [-10, -5, -1, 0, 1, 5, 10]
        for x in test_values:
            result = await cosh(x)
            assert result >= 1.0

    @pytest.mark.asyncio
    async def test_tanh_zero(self):
        """Test tanh(0) = 0."""
        assert abs(await tanh(0)) < 1e-15

    @pytest.mark.asyncio
    async def test_tanh_known_values(self):
        """Test tanh at known values."""
        # tanh(1) ≈ 0.761594156
        assert abs(await tanh(1) - 0.7615941559557649) < 1e-10

    @pytest.mark.asyncio
    async def test_tanh_odd_function(self):
        """Test that tanh is odd: tanh(-x) = -tanh(x)."""
        test_values = [0.5, 1.0, 2.0, 3.0]
        for x in test_values:
            tanh_pos = await tanh(x)
            tanh_neg = await tanh(-x)
            assert abs(tanh_pos + tanh_neg) < 1e-10

    @pytest.mark.asyncio
    async def test_tanh_bounds(self):
        """Test that -1 <= tanh(x) <= 1 for all x."""
        test_values = [-100, -10, -1, 0, 1, 10, 100]
        for x in test_values:
            result = await tanh(x)
            assert -1 <= result <= 1

    @pytest.mark.asyncio
    async def test_tanh_saturation(self):
        """Test tanh saturation at large |x|."""
        # For large positive x, tanh → 1
        assert abs(await tanh(10) - 1.0) < 1e-8
        assert abs(await tanh(20) - 1.0) < 1e-15

        # For large negative x, tanh → -1
        assert abs(await tanh(-10) + 1.0) < 1e-8
        assert abs(await tanh(-20) + 1.0) < 1e-15


class TestReciprocalHyperbolicFunctions:
    """Test reciprocal hyperbolic functions."""

    @pytest.mark.asyncio
    async def test_csch_values(self):
        """Test cosecant hyperbolic values."""
        # csch(1) = 1/sinh(1)
        sinh_1 = await sinh(1)
        csch_1 = await csch(1)
        assert abs(csch_1 * sinh_1 - 1.0) < 1e-10

    @pytest.mark.asyncio
    async def test_csch_zero_error(self):
        """Test that csch(0) raises error."""
        with pytest.raises(ValueError, match="csch undefined at x = 0"):
            await csch(0)

    @pytest.mark.asyncio
    async def test_csch_odd_function(self):
        """Test that csch is odd: csch(-x) = -csch(x)."""
        test_values = [0.5, 1.0, 2.0]
        for x in test_values:
            csch_pos = await csch(x)
            csch_neg = await csch(-x)
            assert abs(csch_pos + csch_neg) < 1e-10

    @pytest.mark.asyncio
    async def test_sech_values(self):
        """Test secant hyperbolic values."""
        # sech(0) = 1
        assert abs(await sech(0) - 1.0) < 1e-15

        # sech(1) = 1/cosh(1)
        cosh_1 = await cosh(1)
        sech_1 = await sech(1)
        assert abs(sech_1 * cosh_1 - 1.0) < 1e-10

    @pytest.mark.asyncio
    async def test_sech_even_function(self):
        """Test that sech is even: sech(-x) = sech(x)."""
        test_values = [0.5, 1.0, 2.0]
        for x in test_values:
            sech_pos = await sech(x)
            sech_neg = await sech(-x)
            assert abs(sech_pos - sech_neg) < 1e-10

    @pytest.mark.asyncio
    async def test_sech_bounds(self):
        """Test that 0 < sech(x) <= 1."""
        test_values = [-10, -5, -1, 0, 1, 5, 10]
        for x in test_values:
            result = await sech(x)
            assert 0 < result <= 1

    @pytest.mark.asyncio
    async def test_coth_values(self):
        """Test cotangent hyperbolic values."""
        # coth(1) = 1/tanh(1)
        tanh_1 = await tanh(1)
        coth_1 = await coth(1)
        assert abs(coth_1 * tanh_1 - 1.0) < 1e-10

    @pytest.mark.asyncio
    async def test_coth_zero_error(self):
        """Test that coth(0) raises error."""
        with pytest.raises(ValueError, match="coth undefined at x = 0"):
            await coth(0)

    @pytest.mark.asyncio
    async def test_coth_odd_function(self):
        """Test that coth is odd: coth(-x) = -coth(x)."""
        test_values = [0.5, 1.0, 2.0]
        for x in test_values:
            coth_pos = await coth(x)
            coth_neg = await coth(-x)
            assert abs(coth_pos + coth_neg) < 1e-10


class TestHyperbolicIdentities:
    """Test hyperbolic identities."""

    @pytest.mark.asyncio
    async def test_fundamental_identity(self):
        """Test cosh²(x) - sinh²(x) = 1."""
        test_values = [0, 0.5, 1.0, 2.0, 5.0, -1.0, -2.0]
        for x in test_values:
            sinh_x = await sinh(x)
            cosh_x = await cosh(x)
            result = cosh_x**2 - sinh_x**2
            assert abs(result - 1.0) < 1e-10

    @pytest.mark.asyncio
    async def test_verify_identity_function(self):
        """Test the verify_hyperbolic_identity function."""
        test_values = [0, 1, 2, -1, -2]
        for x in test_values:
            result = await verify_hyperbolic_identity(x)
            assert result["identity_holds"] is True
            assert result["error"] < 1e-12

    @pytest.mark.asyncio
    async def test_quotient_identity(self):
        """Test tanh(x) = sinh(x)/cosh(x)."""
        test_values = [0.5, 1.0, 2.0, -1.0]
        for x in test_values:
            tanh_x = await tanh(x)
            sinh_x = await sinh(x)
            cosh_x = await cosh(x)
            quotient = sinh_x / cosh_x
            assert abs(tanh_x - quotient) < 1e-10

    @pytest.mark.asyncio
    async def test_double_angle_sinh(self):
        """Test sinh(2x) = 2sinh(x)cosh(x)."""
        test_values = [0.5, 1.0, 1.5]
        for x in test_values:
            sinh_2x = await sinh(2 * x)
            sinh_x = await sinh(x)
            cosh_x = await cosh(x)
            expected = 2 * sinh_x * cosh_x
            assert abs(sinh_2x - expected) < 1e-10

    @pytest.mark.asyncio
    async def test_double_angle_cosh(self):
        """Test cosh(2x) = cosh²(x) + sinh²(x)."""
        test_values = [0.5, 1.0, 1.5]
        for x in test_values:
            cosh_2x = await cosh(2 * x)
            sinh_x = await sinh(x)
            cosh_x = await cosh(x)
            expected = cosh_x**2 + sinh_x**2
            assert abs(cosh_2x - expected) < 1e-10


class TestHyperbolicFunctionsUtility:
    """Test the hyperbolic_functions utility."""

    @pytest.mark.asyncio
    async def test_all_functions_at_one(self):
        """Test all hyperbolic functions at x=1."""
        result = await hyperbolic_functions(1)

        assert "sinh" in result
        assert "cosh" in result
        assert "tanh" in result
        assert "csch" in result
        assert "sech" in result
        assert "coth" in result

        # All should have finite values
        assert math.isfinite(result["sinh"])
        assert math.isfinite(result["cosh"])
        assert math.isfinite(result["tanh"])
        assert math.isfinite(result["csch"])
        assert math.isfinite(result["sech"])
        assert math.isfinite(result["coth"])

    @pytest.mark.asyncio
    async def test_all_functions_at_zero(self):
        """Test all hyperbolic functions at x=0."""
        result = await hyperbolic_functions(0)

        assert result["sinh"] == 0.0
        assert result["cosh"] == 1.0
        assert result["tanh"] == 0.0
        assert result["csch"] is None  # Undefined at 0
        assert result["sech"] == 1.0
        assert result["coth"] is None  # Undefined at 0


class TestCatenaryCurve:
    """Test catenary curve calculations."""

    @pytest.mark.asyncio
    async def test_catenary_at_center(self):
        """Test catenary at lowest point (x=0)."""
        result = await catenary_curve(1.0, 0)

        assert abs(result["y"] - 1.0) < 1e-10  # y = a at x=0
        assert abs(result["slope"]) < 1e-10  # Slope = 0 at lowest point
        assert abs(result["arc_length"]) < 1e-10  # Arc length = 0 at center

    @pytest.mark.asyncio
    async def test_catenary_properties(self):
        """Test catenary curve properties."""
        a = 2.0
        x = 1.0
        result = await catenary_curve(a, x)

        # y = a * cosh(x/a)
        expected_y = a * await cosh(x / a)
        assert abs(result["y"] - expected_y) < 1e-10

        # slope = sinh(x/a)
        expected_slope = await sinh(x / a)
        assert abs(result["slope"] - expected_slope) < 1e-10

    @pytest.mark.asyncio
    async def test_catenary_negative_a_error(self):
        """Test that negative a raises error."""
        with pytest.raises(ValueError, match="parameter 'a' must be positive"):
            await catenary_curve(-1.0, 0)

    @pytest.mark.asyncio
    async def test_catenary_symmetry(self):
        """Test catenary is symmetric around x=0."""
        a = 1.5
        x = 1.0

        result_pos = await catenary_curve(a, x)
        result_neg = await catenary_curve(a, -x)

        # y values should be equal (even function)
        assert abs(result_pos["y"] - result_neg["y"]) < 1e-10

        # Slopes should be opposite (odd derivative)
        assert abs(result_pos["slope"] + result_neg["slope"]) < 1e-10

        # Arc lengths should be equal in magnitude
        assert abs(result_pos["arc_length"] - abs(result_neg["arc_length"])) < 1e-10


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_very_small_values(self):
        """Test hyperbolic functions with very small values."""
        small = 1e-10

        # For small x: sinh(x) ≈ x
        sinh_val = await sinh(small)
        assert abs(sinh_val - small) < 1e-9

        # For small x: cosh(x) ≈ 1
        cosh_val = await cosh(small)
        assert abs(cosh_val - 1.0) < 1e-9

        # For small x: tanh(x) ≈ x
        tanh_val = await tanh(small)
        assert abs(tanh_val - small) < 1e-9

    @pytest.mark.asyncio
    async def test_very_large_positive_values(self):
        """Test hyperbolic functions with large positive values."""
        large = 50

        sinh_val = await sinh(large)
        cosh_val = await cosh(large)

        # For large x: sinh(x) ≈ cosh(x) ≈ e^x/2
        assert abs(sinh_val - cosh_val) / sinh_val < 1e-10

        # tanh should approach 1
        tanh_val = await tanh(large)
        assert abs(tanh_val - 1.0) < 1e-15

    @pytest.mark.asyncio
    async def test_very_large_negative_values(self):
        """Test hyperbolic functions with large negative values."""
        large = -50

        tanh_val = await tanh(large)
        # tanh should approach -1
        assert abs(tanh_val + 1.0) < 1e-15


class TestNumericalStability:
    """Test numerical stability."""

    @pytest.mark.asyncio
    async def test_stability_near_zero(self):
        """Test numerical stability near zero."""
        tiny_values = [1e-15, 1e-12, 1e-10, 1e-8]
        for x in tiny_values:
            sinh_val = await sinh(x)
            cosh_val = await cosh(x)

            # sinh(x) should be close to x for small x
            assert abs(sinh_val - x) / x < 0.01 if x > 1e-14 else True

            # cosh(x) should be close to 1 for small x
            assert abs(cosh_val - 1.0) < 0.01

    @pytest.mark.asyncio
    async def test_overflow_protection(self):
        """Test that very large values don't cause overflow."""
        large_values = [100, 200, 500]
        for x in large_values:
            sinh_val = await sinh(x)
            cosh_val = await cosh(x)

            # Should return finite values
            assert math.isfinite(sinh_val)
            assert math.isfinite(cosh_val)

    @pytest.mark.asyncio
    async def test_reciprocal_precision(self):
        """Test precision of reciprocal relationships."""
        test_values = [0.5, 1.0, 2.0, 5.0]
        for x in test_values:
            # csch = 1/sinh
            sinh_x = await sinh(x)
            csch_x = await csch(x)
            assert abs(sinh_x * csch_x - 1.0) < 1e-10

            # sech = 1/cosh
            cosh_x = await cosh(x)
            sech_x = await sech(x)
            assert abs(cosh_x * sech_x - 1.0) < 1e-10

            # coth = 1/tanh
            tanh_x = await tanh(x)
            coth_x = await coth(x)
            assert abs(tanh_x * coth_x - 1.0) < 1e-10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
