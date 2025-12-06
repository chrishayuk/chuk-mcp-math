#!/usr/bin/env python3
"""
Comprehensive tests for inverse_functions module.
Tests asin, acos, atan, atan2 and inverse reciprocal functions.
"""

import pytest
import math
from chuk_mcp_math.trigonometry.inverse_functions import (
    asin,
    acos,
    atan,
    atan2,
    acsc,
    asec,
    acot,
    asin_degrees,
    acos_degrees,
    atan_degrees,
    atan2_degrees,
)


class TestArcsin:
    """Test arcsine function."""

    @pytest.mark.asyncio
    async def test_asin_special_values(self):
        """Test asin at special values."""
        assert abs(await asin(0)) < 1e-15
        assert abs(await asin(0.5) - math.pi / 6) < 1e-10
        assert abs(await asin(math.sqrt(2) / 2) - math.pi / 4) < 1e-10
        assert abs(await asin(math.sqrt(3) / 2) - math.pi / 3) < 1e-10
        assert abs(await asin(1) - math.pi / 2) < 1e-10
        assert abs(await asin(-1) + math.pi / 2) < 1e-10
        # Test negative exact values (lines 93, 97)
        assert abs(await asin(-math.sqrt(2) / 2) + math.pi / 4) < 1e-10
        assert abs(await asin(-math.sqrt(3) / 2) + math.pi / 3) < 1e-10
        assert abs(await asin(-0.5) + math.pi / 6) < 1e-10

    @pytest.mark.asyncio
    async def test_asin_domain_error(self):
        """Test asin domain validation."""
        with pytest.raises(ValueError, match="asin domain error"):
            await asin(1.1)
        with pytest.raises(ValueError, match="asin domain error"):
            await asin(-1.1)
        with pytest.raises(ValueError, match="asin domain error"):
            await asin(2)

    @pytest.mark.asyncio
    async def test_asin_range(self):
        """Test asin range is [-π/2, π/2]."""
        test_values = [-1, -0.5, 0, 0.5, 1]
        for val in test_values:
            result = await asin(val)
            assert -math.pi / 2 <= result <= math.pi / 2

    @pytest.mark.asyncio
    async def test_asin_odd_function(self):
        """Test asin is odd: asin(-x) = -asin(x)."""
        test_values = [0.25, 0.5, 0.75]
        for val in test_values:
            asin_pos = await asin(val)
            asin_neg = await asin(-val)
            assert abs(asin_pos + asin_neg) < 1e-10


class TestArccos:
    """Test arccosine function."""

    @pytest.mark.asyncio
    async def test_acos_special_values(self):
        """Test acos at special values."""
        assert abs(await acos(1)) < 1e-15
        assert abs(await acos(0.5) - math.pi / 3) < 1e-10
        assert abs(await acos(math.sqrt(2) / 2) - math.pi / 4) < 1e-10
        assert abs(await acos(0) - math.pi / 2) < 1e-10
        assert abs(await acos(-1) - math.pi) < 1e-10
        # Test negative exact values (lines 168, 170, 172)
        assert abs(await acos(-math.sqrt(2) / 2) - 3 * math.pi / 4) < 1e-10
        assert abs(await acos(math.sqrt(3) / 2) - math.pi / 6) < 1e-10
        assert abs(await acos(-math.sqrt(3) / 2) - 5 * math.pi / 6) < 1e-10
        assert abs(await acos(-0.5) - 2 * math.pi / 3) < 1e-10

    @pytest.mark.asyncio
    async def test_acos_domain_error(self):
        """Test acos domain validation."""
        with pytest.raises(ValueError, match="acos domain error"):
            await acos(1.1)
        with pytest.raises(ValueError, match="acos domain error"):
            await acos(-1.1)

    @pytest.mark.asyncio
    async def test_acos_range(self):
        """Test acos range is [0, π]."""
        test_values = [-1, -0.5, 0, 0.5, 1]
        for val in test_values:
            result = await acos(val)
            assert 0 <= result <= math.pi

    @pytest.mark.asyncio
    async def test_asin_acos_relationship(self):
        """Test asin(x) + acos(x) = π/2."""
        test_values = [-0.8, -0.5, 0, 0.3, 0.7, 0.9]
        for val in test_values:
            asin_val = await asin(val)
            acos_val = await acos(val)
            assert abs(asin_val + acos_val - math.pi / 2) < 1e-10


class TestArctan:
    """Test arctangent function."""

    @pytest.mark.asyncio
    async def test_atan_special_values(self):
        """Test atan at special values."""
        assert abs(await atan(0)) < 1e-15
        assert abs(await atan(1) - math.pi / 4) < 1e-10
        assert abs(await atan(math.sqrt(3)) - math.pi / 3) < 1e-10
        assert abs(await atan(-1) + math.pi / 4) < 1e-10
        # Test additional exact values (lines 232, 234, 236)
        assert abs(await atan(-math.sqrt(3)) + math.pi / 3) < 1e-10
        assert abs(await atan(1 / math.sqrt(3)) - math.pi / 6) < 1e-10
        assert abs(await atan(-1 / math.sqrt(3)) + math.pi / 6) < 1e-10

    @pytest.mark.asyncio
    async def test_atan_range(self):
        """Test atan range is (-π/2, π/2)."""
        test_values = [-1000, -10, -1, 0, 1, 10, 1000]
        for val in test_values:
            result = await atan(val)
            assert -math.pi / 2 < result < math.pi / 2

    @pytest.mark.asyncio
    async def test_atan_odd_function(self):
        """Test atan is odd: atan(-x) = -atan(x)."""
        test_values = [0.5, 1, 2, 10]
        for val in test_values:
            atan_pos = await atan(val)
            atan_neg = await atan(-val)
            assert abs(atan_pos + atan_neg) < 1e-10

    @pytest.mark.asyncio
    async def test_atan_asymptotes(self):
        """Test atan approaches ±π/2 for large values."""
        assert abs(await atan(10000) - math.pi / 2) < 0.0001
        assert abs(await atan(-10000) + math.pi / 2) < 0.0001


class TestArctan2:
    """Test two-argument arctangent function."""

    @pytest.mark.asyncio
    async def test_atan2_quadrants(self):
        """Test atan2 returns correct values in all quadrants."""
        # Q1: positive x, positive y
        assert abs(await atan2(1, 1) - math.pi / 4) < 1e-10

        # Q2: negative x, positive y
        assert abs(await atan2(1, -1) - 3 * math.pi / 4) < 1e-10

        # Q3: negative x, negative y
        assert abs(await atan2(-1, -1) + 3 * math.pi / 4) < 1e-10

        # Q4: positive x, negative y
        assert abs(await atan2(-1, 1) + math.pi / 4) < 1e-10

    @pytest.mark.asyncio
    async def test_atan2_axes(self):
        """Test atan2 on coordinate axes."""
        # Positive x-axis
        assert abs(await atan2(0, 1)) < 1e-15

        # Positive y-axis
        assert abs(await atan2(1, 0) - math.pi / 2) < 1e-15

        # Negative x-axis
        assert abs(await atan2(0, -1) - math.pi) < 1e-15

        # Negative y-axis
        assert abs(await atan2(-1, 0) + math.pi / 2) < 1e-15

    @pytest.mark.asyncio
    async def test_atan2_undefined(self):
        """Test atan2(0, 0) raises error."""
        with pytest.raises(ValueError, match="atan2 undefined"):
            await atan2(0, 0)

    @pytest.mark.asyncio
    async def test_atan2_range(self):
        """Test atan2 range is (-π, π]."""
        test_cases = [(1, 1), (1, -1), (-1, -1), (-1, 1), (2, 3), (-2, 3)]
        for y, x in test_cases:
            result = await atan2(y, x)
            assert -math.pi < result <= math.pi


class TestInverseReciprocalFunctions:
    """Test inverse reciprocal functions."""

    @pytest.mark.asyncio
    async def test_acsc_values(self):
        """Test arccosecant values."""
        assert abs(await acsc(1) - math.pi / 2) < 1e-10
        assert abs(await acsc(-1) + math.pi / 2) < 1e-10
        assert abs(await acsc(2) - math.pi / 6) < 1e-10

    @pytest.mark.asyncio
    async def test_acsc_domain_error(self):
        """Test acsc domain validation."""
        with pytest.raises(ValueError, match="acsc domain error"):
            await acsc(0.5)
        with pytest.raises(ValueError, match="acsc domain error"):
            await acsc(-0.5)

    @pytest.mark.asyncio
    async def test_acsc_reciprocal(self):
        """Test acsc(x) = asin(1/x)."""
        test_values = [1, 2, -1, -2, 1.5]
        for val in test_values:
            acsc_val = await acsc(val)
            asin_val = await asin(1 / val)
            assert abs(acsc_val - asin_val) < 1e-10

    @pytest.mark.asyncio
    async def test_asec_values(self):
        """Test arcsecant values."""
        assert abs(await asec(1)) < 1e-15
        assert abs(await asec(2) - math.pi / 3) < 1e-10
        assert abs(await asec(-1) - math.pi) < 1e-10

    @pytest.mark.asyncio
    async def test_asec_domain_error(self):
        """Test asec domain validation."""
        with pytest.raises(ValueError, match="asec domain error"):
            await asec(0.5)
        with pytest.raises(ValueError, match="asec domain error"):
            await asec(-0.5)

    @pytest.mark.asyncio
    async def test_asec_reciprocal(self):
        """Test asec(x) = acos(1/x)."""
        test_values = [1, 2, -1, -2]
        for val in test_values:
            asec_val = await asec(val)
            acos_val = await acos(1 / val)
            assert abs(asec_val - acos_val) < 1e-10

    @pytest.mark.asyncio
    async def test_acot_values(self):
        """Test arccotangent values."""
        assert abs(await acot(1) - math.pi / 4) < 1e-10
        assert abs(await acot(0) - math.pi / 2) < 1e-10
        assert abs(await acot(math.sqrt(3)) - math.pi / 6) < 1e-10
        # Test additional exact values (lines 497, 499, 501)
        assert abs(await acot(-math.sqrt(3)) - 5 * math.pi / 6) < 1e-10
        assert abs(await acot(1 / math.sqrt(3)) - math.pi / 3) < 1e-10
        assert abs(await acot(-1 / math.sqrt(3)) - 2 * math.pi / 3) < 1e-10
        assert abs(await acot(-1) - 3 * math.pi / 4) < 1e-10

    @pytest.mark.asyncio
    async def test_acot_range(self):
        """Test acot range is (0, π)."""
        test_values = [-100, -1, 0, 1, 100]
        for val in test_values:
            result = await acot(val)
            assert 0 < result < math.pi


class TestDegreeVariants:
    """Test degree-output variants."""

    @pytest.mark.asyncio
    async def test_asin_degrees(self):
        """Test asin_degrees function."""
        assert abs(await asin_degrees(0)) < 1e-15
        assert abs(await asin_degrees(0.5) - 30) < 1e-10
        assert abs(await asin_degrees(1) - 90) < 1e-10

    @pytest.mark.asyncio
    async def test_acos_degrees(self):
        """Test acos_degrees function."""
        assert abs(await acos_degrees(1)) < 1e-15
        assert abs(await acos_degrees(0.5) - 60) < 1e-10
        assert abs(await acos_degrees(0) - 90) < 1e-10

    @pytest.mark.asyncio
    async def test_atan_degrees(self):
        """Test atan_degrees function."""
        assert abs(await atan_degrees(0)) < 1e-15
        assert abs(await atan_degrees(1) - 45) < 1e-10
        assert abs(await atan_degrees(math.sqrt(3)) - 60) < 1e-10

    @pytest.mark.asyncio
    async def test_atan2_degrees(self):
        """Test atan2_degrees function."""
        assert abs(await atan2_degrees(1, 1) - 45) < 1e-10
        assert abs(await atan2_degrees(1, -1) - 135) < 1e-10
        assert abs(await atan2_degrees(1, 0) - 90) < 1e-10


class TestInverseFunctionProperties:
    """Test inverse function mathematical properties."""

    @pytest.mark.asyncio
    async def test_sin_asin_identity(self):
        """Test sin(asin(x)) = x."""
        from chuk_mcp_math.trigonometry.basic_functions import sin

        test_values = [-1, -0.5, 0, 0.5, 1]
        for val in test_values:
            asin_val = await asin(val)
            sin_asin = await sin(asin_val)
            assert abs(sin_asin - val) < 1e-10

    @pytest.mark.asyncio
    async def test_cos_acos_identity(self):
        """Test cos(acos(x)) = x."""
        from chuk_mcp_math.trigonometry.basic_functions import cos

        test_values = [-1, -0.5, 0, 0.5, 1]
        for val in test_values:
            acos_val = await acos(val)
            cos_acos = await cos(acos_val)
            assert abs(cos_acos - val) < 1e-10

    @pytest.mark.asyncio
    async def test_tan_atan_identity(self):
        """Test tan(atan(x)) = x."""
        from chuk_mcp_math.trigonometry.basic_functions import tan

        test_values = [-10, -1, 0, 1, 10]
        for val in test_values:
            atan_val = await atan(val)
            tan_atan = await tan(atan_val)
            assert abs(tan_atan - val) < 1e-10


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_boundary_values(self):
        """Test at domain boundaries."""
        # asin at boundaries
        assert abs(await asin(-1) + math.pi / 2) < 1e-10
        assert abs(await asin(1) - math.pi / 2) < 1e-10

        # acos at boundaries
        assert abs(await acos(-1) - math.pi) < 1e-10
        assert abs(await acos(1)) < 1e-10

    @pytest.mark.asyncio
    async def test_very_small_values(self):
        """Test with very small values."""
        small = 1e-10

        # For small x: asin(x) ≈ x
        asin_val = await asin(small)
        assert abs(asin_val - small) < 1e-9

        # For small x: acos(x) ≈ π/2 - x
        acos_val = await acos(small)
        assert abs(acos_val - (math.pi / 2 - small)) < 1e-9

        # For small x: atan(x) ≈ x
        atan_val = await atan(small)
        assert abs(atan_val - small) < 1e-9

    @pytest.mark.asyncio
    async def test_precision_near_boundaries(self):
        """Test precision near domain boundaries."""
        # Near 1 for asin
        near_one = 1 - 1e-10
        result = await asin(near_one)
        assert result < math.pi / 2
        assert result > 1.5  # Should be close to π/2

        # Near -1 for asin
        near_neg_one = -1 + 1e-10
        result = await asin(near_neg_one)
        assert result > -math.pi / 2
        assert result < -1.5


class TestModuleExports:
    """Test module exports and MCP decorator integration."""

    def test_all_functions_exported(self):
        """Test that all functions are properly exported (lines 698-720)."""
        import chuk_mcp_math.trigonometry.inverse_functions as inv_funcs

        # Test that all functions in __all__ are accessible
        expected_functions = [
            "asin",
            "acos",
            "atan",
            "atan2",
            "acsc",
            "asec",
            "acot",
            "asin_degrees",
            "acos_degrees",
            "atan_degrees",
            "atan2_degrees",
        ]

        for func_name in expected_functions:
            assert hasattr(inv_funcs, func_name), f"Function {func_name} not exported"
            func = getattr(inv_funcs, func_name)
            assert callable(func), f"{func_name} is not callable"

    def test_module_all_attribute(self):
        """Test __all__ attribute exists and contains expected functions."""
        import chuk_mcp_math.trigonometry.inverse_functions as inv_funcs

        assert hasattr(inv_funcs, "__all__")
        assert len(inv_funcs.__all__) == 11  # 11 functions total

    def test_module_reloadable(self):
        """Test module can be imported multiple times without errors."""
        import importlib
        import chuk_mcp_math.trigonometry.inverse_functions

        # Reload should not raise errors
        importlib.reload(chuk_mcp_math.trigonometry.inverse_functions)

        # Functions should still be available
        from chuk_mcp_math.trigonometry.inverse_functions import asin, acos

        assert callable(asin)
        assert callable(acos)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
