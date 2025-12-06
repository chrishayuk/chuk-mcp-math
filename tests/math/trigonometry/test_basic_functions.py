#!/usr/bin/env python3
"""
Comprehensive tests for basic_functions module.
Tests sin, cos, tan and their reciprocal functions.
"""

import pytest
import math
from chuk_mcp_math.trigonometry.basic_functions import (
    sin,
    cos,
    tan,
    csc,
    sec,
    cot,
    sin_degrees,
    cos_degrees,
    tan_degrees,
)


class TestPrimaryTrigFunctions:
    """Test primary trigonometric functions (sin, cos, tan)."""

    @pytest.mark.asyncio
    async def test_sin_special_angles(self):
        """Test sine at special angles."""
        assert abs(await sin(0)) < 1e-15
        assert abs(await sin(math.pi / 6) - 0.5) < 1e-10
        assert abs(await sin(math.pi / 4) - math.sqrt(2) / 2) < 1e-10
        assert abs(await sin(math.pi / 3) - math.sqrt(3) / 2) < 1e-10
        assert abs(await sin(math.pi / 2) - 1.0) < 1e-10
        assert abs(await sin(math.pi)) < 1e-10
        assert abs(await sin(3 * math.pi / 2) + 1.0) < 1e-10

    @pytest.mark.asyncio
    async def test_sin_negative_angles(self):
        """Test sine with negative angles (odd function)."""
        angle = math.pi / 6
        sin_pos = await sin(angle)
        sin_neg = await sin(-angle)
        assert abs(sin_pos + sin_neg) < 1e-10

    @pytest.mark.asyncio
    async def test_sin_periodicity(self):
        """Test sine periodicity (period = 2π)."""
        angle = math.pi / 4
        sin1 = await sin(angle)
        sin2 = await sin(angle + 2 * math.pi)
        sin3 = await sin(angle + 4 * math.pi)
        assert abs(sin1 - sin2) < 1e-10
        assert abs(sin1 - sin3) < 1e-10

    @pytest.mark.asyncio
    async def test_sin_range(self):
        """Test that sine is bounded in [-1, 1]."""
        test_angles = [0, 0.5, 1, 1.5, 2, 2.5, 3, math.pi, 4, 5, 6]
        for angle in test_angles:
            value = await sin(angle)
            assert -1 <= value <= 1

    @pytest.mark.asyncio
    async def test_cos_special_angles(self):
        """Test cosine at special angles."""
        assert abs(await cos(0) - 1.0) < 1e-15
        assert abs(await cos(math.pi / 6) - math.sqrt(3) / 2) < 1e-10
        assert abs(await cos(math.pi / 4) - math.sqrt(2) / 2) < 1e-10
        assert abs(await cos(math.pi / 3) - 0.5) < 1e-10
        assert abs(await cos(math.pi / 2)) < 1e-10
        assert abs(await cos(math.pi) + 1.0) < 1e-10

    @pytest.mark.asyncio
    async def test_cos_negative_angles(self):
        """Test cosine with negative angles (even function)."""
        angle = math.pi / 3
        cos_pos = await cos(angle)
        cos_neg = await cos(-angle)
        assert abs(cos_pos - cos_neg) < 1e-10

    @pytest.mark.asyncio
    async def test_cos_periodicity(self):
        """Test cosine periodicity (period = 2π)."""
        angle = math.pi / 6
        cos1 = await cos(angle)
        cos2 = await cos(angle + 2 * math.pi)
        assert abs(cos1 - cos2) < 1e-10

    @pytest.mark.asyncio
    async def test_cos_range(self):
        """Test that cosine is bounded in [-1, 1]."""
        test_angles = [0, 0.5, 1, 1.5, 2, 2.5, 3, math.pi, 4, 5, 6]
        for angle in test_angles:
            value = await cos(angle)
            assert -1 <= value <= 1

    @pytest.mark.asyncio
    async def test_tan_special_angles(self):
        """Test tangent at special angles."""
        assert abs(await tan(0)) < 1e-15
        assert abs(await tan(math.pi / 4) - 1.0) < 1e-10
        assert abs(await tan(math.pi / 6) - 1 / math.sqrt(3)) < 1e-10
        assert abs(await tan(math.pi / 3) - math.sqrt(3)) < 1e-10

    @pytest.mark.asyncio
    async def test_tan_singularities(self):
        """Test tangent at singularities (odd multiples of π/2)."""
        with pytest.raises(ValueError, match="Tangent undefined"):
            await tan(math.pi / 2)

        with pytest.raises(ValueError, match="Tangent undefined"):
            await tan(3 * math.pi / 2)

        with pytest.raises(ValueError, match="Tangent undefined"):
            await tan(-math.pi / 2)

    @pytest.mark.asyncio
    async def test_tan_negative_angles(self):
        """Test tangent with negative angles (odd function)."""
        angle = math.pi / 4
        tan_pos = await tan(angle)
        tan_neg = await tan(-angle)
        assert abs(tan_pos + tan_neg) < 1e-10


class TestReciprocalFunctions:
    """Test reciprocal trigonometric functions (csc, sec, cot)."""

    @pytest.mark.asyncio
    async def test_csc_special_angles(self):
        """Test cosecant at special angles."""
        assert abs(await csc(math.pi / 6) - 2.0) < 1e-10
        assert abs(await csc(math.pi / 4) - math.sqrt(2)) < 1e-10
        assert abs(await csc(math.pi / 2) - 1.0) < 1e-10

    @pytest.mark.asyncio
    async def test_csc_singularities(self):
        """Test cosecant at singularities (multiples of π)."""
        with pytest.raises(ValueError, match="Cosecant undefined"):
            await csc(0)

        with pytest.raises(ValueError, match="Cosecant undefined"):
            await csc(math.pi)

        with pytest.raises(ValueError, match="Cosecant undefined"):
            await csc(2 * math.pi)

    @pytest.mark.asyncio
    async def test_csc_reciprocal_relationship(self):
        """Test that csc = 1/sin."""
        test_angles = [math.pi / 6, math.pi / 4, math.pi / 3, math.pi / 2]
        for angle in test_angles:
            sin_val = await sin(angle)
            csc_val = await csc(angle)
            assert abs(csc_val * sin_val - 1.0) < 1e-10

    @pytest.mark.asyncio
    async def test_sec_special_angles(self):
        """Test secant at special angles."""
        assert abs(await sec(0) - 1.0) < 1e-15
        assert abs(await sec(math.pi / 3) - 2.0) < 1e-10
        assert abs(await sec(math.pi / 4) - math.sqrt(2)) < 1e-10

    @pytest.mark.asyncio
    async def test_sec_singularities(self):
        """Test secant at singularities (odd multiples of π/2)."""
        with pytest.raises(ValueError, match="Secant undefined"):
            await sec(math.pi / 2)

        with pytest.raises(ValueError, match="Secant undefined"):
            await sec(3 * math.pi / 2)

    @pytest.mark.asyncio
    async def test_sec_reciprocal_relationship(self):
        """Test that sec = 1/cos."""
        test_angles = [0, math.pi / 6, math.pi / 4, math.pi / 3]
        for angle in test_angles:
            cos_val = await cos(angle)
            sec_val = await sec(angle)
            assert abs(sec_val * cos_val - 1.0) < 1e-10

    @pytest.mark.asyncio
    async def test_cot_special_angles(self):
        """Test cotangent at special angles."""
        assert abs(await cot(math.pi / 4) - 1.0) < 1e-10
        assert abs(await cot(math.pi / 6) - math.sqrt(3)) < 1e-10
        assert abs(await cot(math.pi / 3) - 1 / math.sqrt(3)) < 1e-10

    @pytest.mark.asyncio
    async def test_cot_singularities(self):
        """Test cotangent at singularities (multiples of π)."""
        with pytest.raises(ValueError, match="Cotangent undefined"):
            await cot(0)

        with pytest.raises(ValueError, match="Cotangent undefined"):
            await cot(math.pi)

    @pytest.mark.asyncio
    async def test_cot_reciprocal_relationship(self):
        """Test that cot = 1/tan."""
        test_angles = [math.pi / 6, math.pi / 4, math.pi / 3]
        for angle in test_angles:
            tan_val = await tan(angle)
            cot_val = await cot(angle)
            assert abs(cot_val * tan_val - 1.0) < 1e-10


class TestDegreeVariants:
    """Test degree-input variants of trig functions."""

    @pytest.mark.asyncio
    async def test_sin_degrees_special_angles(self):
        """Test sin_degrees at special angles."""
        assert abs(await sin_degrees(0)) < 1e-15
        assert abs(await sin_degrees(30) - 0.5) < 1e-10
        assert abs(await sin_degrees(45) - math.sqrt(2) / 2) < 1e-10
        assert abs(await sin_degrees(60) - math.sqrt(3) / 2) < 1e-10
        assert abs(await sin_degrees(90) - 1.0) < 1e-10
        assert abs(await sin_degrees(180)) < 1e-10

    @pytest.mark.asyncio
    async def test_cos_degrees_special_angles(self):
        """Test cos_degrees at special angles."""
        assert abs(await cos_degrees(0) - 1.0) < 1e-15
        assert abs(await cos_degrees(30) - math.sqrt(3) / 2) < 1e-10
        assert abs(await cos_degrees(45) - math.sqrt(2) / 2) < 1e-10
        assert abs(await cos_degrees(60) - 0.5) < 1e-10
        assert abs(await cos_degrees(90)) < 1e-10
        assert abs(await cos_degrees(180) + 1.0) < 1e-10

    @pytest.mark.asyncio
    async def test_tan_degrees_special_angles(self):
        """Test tan_degrees at special angles."""
        assert abs(await tan_degrees(0)) < 1e-15
        assert abs(await tan_degrees(45) - 1.0) < 1e-10
        assert abs(await tan_degrees(30) - 1 / math.sqrt(3)) < 1e-10
        assert abs(await tan_degrees(60) - math.sqrt(3)) < 1e-10

    @pytest.mark.asyncio
    async def test_tan_degrees_singularities(self):
        """Test tan_degrees at singularities."""
        with pytest.raises(ValueError, match="undefined"):
            await tan_degrees(90)

        with pytest.raises(ValueError, match="undefined"):
            await tan_degrees(270)

    @pytest.mark.asyncio
    async def test_degree_variants_consistency(self):
        """Test that degree variants match radian versions."""
        test_angles = [0, 30, 45, 60, 90, 120, 180]
        for deg in test_angles:
            rad = math.radians(deg)

            # Test sin
            sin_deg = await sin_degrees(deg)
            sin_rad = await sin(rad)
            assert abs(sin_deg - sin_rad) < 1e-10

            # Test cos
            cos_deg = await cos_degrees(deg)
            cos_rad = await cos(rad)
            assert abs(cos_deg - cos_rad) < 1e-10


class TestPythagoreanIdentity:
    """Test Pythagorean identity: sin²θ + cos²θ = 1."""

    @pytest.mark.asyncio
    async def test_pythagorean_identity_special_angles(self):
        """Test Pythagorean identity at special angles."""
        test_angles = [0, math.pi / 6, math.pi / 4, math.pi / 3, math.pi / 2, math.pi]
        for angle in test_angles:
            sin_val = await sin(angle)
            cos_val = await cos(angle)
            result = sin_val**2 + cos_val**2
            assert abs(result - 1.0) < 1e-10

    @pytest.mark.asyncio
    async def test_pythagorean_identity_random_angles(self):
        """Test Pythagorean identity at various angles."""
        test_angles = [0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0]
        for angle in test_angles:
            sin_val = await sin(angle)
            cos_val = await cos(angle)
            result = sin_val**2 + cos_val**2
            assert abs(result - 1.0) < 1e-10


class TestQuotientIdentity:
    """Test quotient identity: tan = sin/cos."""

    @pytest.mark.asyncio
    async def test_quotient_identity(self):
        """Test that tan(θ) = sin(θ)/cos(θ)."""
        test_angles = [math.pi / 6, math.pi / 4, math.pi / 3, math.pi / 8]
        for angle in test_angles:
            tan_val = await tan(angle)
            sin_val = await sin(angle)
            cos_val = await cos(angle)
            quotient = sin_val / cos_val
            assert abs(tan_val - quotient) < 1e-10


class TestSymmetryProperties:
    """Test symmetry properties of trig functions."""

    @pytest.mark.asyncio
    async def test_sin_odd_function(self):
        """Test that sin is an odd function: sin(-x) = -sin(x)."""
        test_angles = [0.5, 1.0, math.pi / 6, math.pi / 4, math.pi / 3]
        for angle in test_angles:
            sin_pos = await sin(angle)
            sin_neg = await sin(-angle)
            assert abs(sin_pos + sin_neg) < 1e-10

    @pytest.mark.asyncio
    async def test_cos_even_function(self):
        """Test that cos is an even function: cos(-x) = cos(x)."""
        test_angles = [0.5, 1.0, math.pi / 6, math.pi / 4, math.pi / 3]
        for angle in test_angles:
            cos_pos = await cos(angle)
            cos_neg = await cos(-angle)
            assert abs(cos_pos - cos_neg) < 1e-10

    @pytest.mark.asyncio
    async def test_tan_odd_function(self):
        """Test that tan is an odd function: tan(-x) = -tan(x)."""
        test_angles = [0.5, 1.0, math.pi / 6, math.pi / 4, math.pi / 3]
        for angle in test_angles:
            tan_pos = await tan(angle)
            tan_neg = await tan(-angle)
            assert abs(tan_pos + tan_neg) < 1e-10


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_very_small_angles(self):
        """Test trig functions with very small angles."""
        small_angle = 1e-10
        sin_val = await sin(small_angle)
        # For small x, sin(x) ≈ x
        assert abs(sin_val - small_angle) < 1e-9

        cos_val = await cos(small_angle)
        # For small x, cos(x) ≈ 1
        assert abs(cos_val - 1.0) < 1e-9

    @pytest.mark.asyncio
    async def test_large_angles(self):
        """Test trig functions with large angles."""
        large_angle = 100 * math.pi
        sin_val = await sin(large_angle)
        # Should still be bounded
        assert -1 <= sin_val <= 1

        cos_val = await cos(large_angle)
        assert -1 <= cos_val <= 1

    @pytest.mark.asyncio
    async def test_zero_angle(self):
        """Test trig functions at zero."""
        assert abs(await sin(0)) < 1e-15
        assert abs(await cos(0) - 1.0) < 1e-15
        assert abs(await tan(0)) < 1e-15

    @pytest.mark.asyncio
    async def test_negative_zero(self):
        """Test that -0.0 is handled correctly."""
        assert abs(await sin(-0.0)) < 1e-15
        assert abs(await cos(-0.0) - 1.0) < 1e-15


class TestNumericalStability:
    """Test numerical stability and precision."""

    @pytest.mark.asyncio
    async def test_near_singularity_stability(self):
        """Test behavior near singularities."""
        # Just before π/2
        angle = math.pi / 2 - 1e-8
        tan_val = await tan(angle)
        assert tan_val > 1e7  # Should be very large but finite

    @pytest.mark.asyncio
    async def test_angle_normalization(self):
        """Test that large angles are normalized correctly."""
        # 10π should give same result as 0
        sin_0 = await sin(0)
        sin_10pi = await sin(10 * math.pi)
        assert abs(sin_0 - sin_10pi) < 1e-10

    @pytest.mark.asyncio
    async def test_precision_at_special_values(self):
        """Test precision at mathematically exact values."""
        # sin(π) should be very close to 0
        sin_pi = await sin(math.pi)
        assert abs(sin_pi) < 1e-10

        # cos(π) should be very close to -1
        cos_pi = await cos(math.pi)
        assert abs(cos_pi + 1.0) < 1e-10

    @pytest.mark.asyncio
    async def test_tan_near_zero_cleanup(self):
        """Test tan function's near-zero result cleanup (line 235)."""
        # Test angles where tan should be very close to zero
        # Using multiples of π where tan should be exactly 0
        tan_pi = await tan(math.pi)
        assert tan_pi == 0.0  # Should return exact 0.0, not small float

        tan_2pi = await tan(2 * math.pi)
        assert tan_2pi == 0.0

        tan_3pi = await tan(3 * math.pi)
        assert tan_3pi == 0.0

        # Negative multiples of π
        tan_neg_pi = await tan(-math.pi)
        assert tan_neg_pi == 0.0


class TestModuleExports:
    """Test module export mechanism and MCP decorator integration."""

    def test_all_functions_exported(self):
        """Test that all functions in __all__ are accessible."""
        import chuk_mcp_math.trigonometry.basic_functions as bf

        # Check that __all__ exists
        assert hasattr(bf, "__all__")

        # Check that all functions in __all__ are accessible as module attributes
        for func_name in bf.__all__:
            assert hasattr(bf, func_name), f"Function {func_name} not accessible"
            func = getattr(bf, func_name)
            assert callable(func), f"{func_name} is not callable"

    def test_module_has_required_functions(self):
        """Test that module has all required trigonometric functions."""
        import chuk_mcp_math.trigonometry.basic_functions as bf

        required_functions = [
            "sin",
            "cos",
            "tan",
            "csc",
            "sec",
            "cot",
            "sin_degrees",
            "cos_degrees",
            "tan_degrees",
        ]

        for func_name in required_functions:
            assert hasattr(bf, func_name), f"Module missing required function: {func_name}"

    @pytest.mark.asyncio
    async def test_exported_functions_are_callable(self):
        """Test that exported functions work correctly when called."""
        import chuk_mcp_math.trigonometry.basic_functions as bf

        # Test that we can call the functions from the module
        result = await bf.sin(0)
        assert result == 0.0

        result = await bf.cos(0)
        assert result == 1.0

        result = await bf.tan(0)
        assert result == 0.0

    def test_mcp_decorator_integration(self):
        """Test MCP decorator integration for functions (lines 584-606)."""
        import chuk_mcp_math.trigonometry.basic_functions as bf
        import sys

        # Verify the module is properly set up
        current_module = sys.modules["chuk_mcp_math.trigonometry.basic_functions"]
        assert current_module is not None

        # Verify functions are in globals
        for func_name in bf.__all__:
            # Either in module namespace or in globals
            assert hasattr(bf, func_name) or func_name in dir(bf)

        # Test that temporary variables were cleaned up (line 606)
        # These should NOT exist in the module after initialization
        assert not hasattr(bf, "_current_module")
        assert not hasattr(bf, "_func_name")
        # _func might exist or not depending on the import path

    def test_module_export_mechanism_reimport(self):
        """Test the module export mechanism by forcing a reimport scenario."""
        import sys

        # Get the module
        module_name = "chuk_mcp_math.trigonometry.basic_functions"

        # The module is already imported, but we can verify its state
        if module_name in sys.modules:
            bf = sys.modules[module_name]

            # Verify all functions from __all__ are accessible
            # This exercises the export logic that ran at import time (lines 584-606)
            for func_name in bf.__all__:
                # Check function is accessible
                assert hasattr(bf, func_name), f"Function {func_name} not found in module"

                # Get the function
                func = getattr(bf, func_name)

                # Verify it's callable
                assert callable(func), f"{func_name} is not callable"

            # Verify cleanup happened (line 604-606)
            # These temp variables should not exist after module initialization
            assert not hasattr(bf, "_current_module"), "_current_module should be deleted"
            assert not hasattr(bf, "_func_name"), "_func_name should be deleted"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
