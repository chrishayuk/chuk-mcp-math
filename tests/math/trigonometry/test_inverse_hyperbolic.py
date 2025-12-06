#!/usr/bin/env python3
"""Tests for inverse_hyperbolic functions module."""

import pytest
from chuk_mcp_math.trigonometry.inverse_hyperbolic import (
    asinh,
    acosh,
    atanh,
    acsch,
    asech,
    acoth,
    inverse_hyperbolic_functions,
    verify_inverse_hyperbolic_identities,
)


class TestInverseHyperbolicFunctions:
    @pytest.mark.asyncio
    async def test_asinh_values(self):
        assert abs(await asinh(0)) < 1e-15
        assert abs(await asinh(1) - 0.881373587) < 1e-6

    @pytest.mark.asyncio
    async def test_asinh_odd_function(self):
        vals = [0.5, 1, 2]
        for x in vals:
            assert abs(await asinh(x) + await asinh(-x)) < 1e-10

    @pytest.mark.asyncio
    async def test_acosh_values(self):
        assert abs(await acosh(1)) < 1e-15
        assert abs(await acosh(2) - 1.316957897) < 1e-6

    @pytest.mark.asyncio
    async def test_acosh_domain_error(self):
        with pytest.raises(ValueError):
            await acosh(0.5)

    @pytest.mark.asyncio
    async def test_atanh_values(self):
        assert abs(await atanh(0)) < 1e-15
        assert abs(await atanh(0.5) - 0.549306144) < 1e-6

    @pytest.mark.asyncio
    async def test_atanh_domain_error(self):
        with pytest.raises(ValueError):
            await atanh(1.1)
        with pytest.raises(ValueError):
            await atanh(1)

    @pytest.mark.asyncio
    async def test_acsch_values(self):
        assert abs(await acsch(1) - 0.881373587) < 1e-6

    @pytest.mark.asyncio
    async def test_acsch_zero_error(self):
        with pytest.raises(ValueError):
            await acsch(0)

    @pytest.mark.asyncio
    async def test_asech_values(self):
        assert abs(await asech(1)) < 1e-15
        assert abs(await asech(0.5) - 1.316957897) < 1e-6

    @pytest.mark.asyncio
    async def test_asech_domain_error(self):
        with pytest.raises(ValueError):
            await asech(1.1)
        with pytest.raises(ValueError):
            await asech(0)

    @pytest.mark.asyncio
    async def test_acoth_values(self):
        assert abs(await acoth(2) - 0.549306144) < 1e-6

    @pytest.mark.asyncio
    async def test_acoth_domain_error(self):
        with pytest.raises(ValueError):
            await acoth(0.5)

    @pytest.mark.asyncio
    async def test_inverse_hyperbolic_identities(self):
        vals = [0.5, 1.5, 2]
        for x in vals:
            result = await verify_inverse_hyperbolic_identities(x)
            assert result["sinh_asinh_identity"] is True
            assert result["sinh_asinh_error"] < 1e-10

    @pytest.mark.asyncio
    async def test_all_functions_utility(self):
        result = await inverse_hyperbolic_functions(2)
        assert result["asinh"] is not None
        assert result["acosh"] is not None
        assert result["atanh"] is None
        assert result["acoth"] is not None

    @pytest.mark.asyncio
    async def test_asinh_very_small_values(self):
        """Test series expansion for very small x (lines 81-82)"""
        x = 1e-11
        result = await asinh(x)
        assert abs(result - x) < 1e-10  # For very small x, asinh(x) ≈ x

    @pytest.mark.asyncio
    async def test_asinh_very_large_values(self):
        """Test asymptotic expansion for large |x| (line 87)"""
        x = 1e9
        result = await asinh(x)
        # For large x, asinh(x) ≈ ln(2x)
        import math

        expected = math.log(2 * x)
        assert abs(result - expected) < 1e-6

        # Test negative large value
        x_neg = -1e9
        result_neg = await asinh(x_neg)
        assert abs(result_neg + expected) < 1e-6

    @pytest.mark.asyncio
    async def test_acosh_very_large_values(self):
        """Test asymptotic expansion for large x (line 162)"""
        x = 1e9
        result = await acosh(x)
        # For large x, acosh(x) ≈ ln(2x)
        import math

        expected = math.log(2 * x)
        assert abs(result - expected) < 1e-6

    @pytest.mark.asyncio
    async def test_acosh_near_one(self):
        """Test special handling near x=1 (lines 168-170)"""
        x = 1.05  # Just above 1, triggers x < 1.1 condition
        result = await acosh(x)
        # Should still give accurate result
        assert result > 0
        assert result < 1

    @pytest.mark.asyncio
    async def test_atanh_very_small_values(self):
        """Test series expansion for very small x (lines 237-238)"""
        x = 1e-11
        result = await atanh(x)
        assert abs(result - x) < 1e-10  # For very small x, atanh(x) ≈ x

    @pytest.mark.asyncio
    async def test_atanh_near_boundary_positive(self):
        """Test special handling for x close to 1 (lines 244-245)"""
        x = 0.96  # Close to 1, triggers abs(x) > 0.95
        result = await atanh(x)
        assert result > 0  # Should be positive
        assert result < float("inf")  # Should be finite

    @pytest.mark.asyncio
    async def test_atanh_near_boundary_negative(self):
        """Test special handling for x close to -1 (lines 246-247)"""
        x = -0.96  # Close to -1, triggers abs(x) > 0.95
        result = await atanh(x)
        assert result < 0  # Should be negative
        assert result > float("-inf")  # Should be finite

    @pytest.mark.asyncio
    async def test_inverse_hyperbolic_functions_all_paths(self):
        """Test all code paths in inverse_hyperbolic_functions"""
        # Test value < 1: acosh=None (line 489), atanh=result (line 493), asech=result (line 505)
        result = await inverse_hyperbolic_functions(0.5)
        assert result["asinh"] is not None
        assert result["acosh"] is None  # Line 489
        assert result["atanh"] is not None  # Line 493
        assert result["acsch"] is not None
        assert result["asech"] is not None  # Line 505
        assert result["acoth"] is None  # Line 513

        # Test value = 0: acsch=None (line 501)
        result = await inverse_hyperbolic_functions(0)
        assert result["asinh"] is not None
        assert result["acosh"] is None
        assert result["atanh"] is not None
        assert result["acsch"] is None  # Line 501
        assert result["asech"] is None
        assert result["acoth"] is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
