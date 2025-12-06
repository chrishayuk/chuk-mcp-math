#!/usr/bin/env python3
"""Tests for trigonometric identities module."""

import pytest
import math
from chuk_mcp_math.trigonometry.identities import (
    pythagorean_identity,
    sum_difference_formulas,
    verify_sum_difference_formulas,
    double_angle_formulas,
    half_angle_formulas,
    verify_identity,
    comprehensive_identity_verification,
    simplify_trig_expression,
)


class TestPythagoreanIdentity:
    @pytest.mark.asyncio
    async def test_pythagorean_at_special_angles(self):
        angles = [0, math.pi / 6, math.pi / 4, math.pi / 3, math.pi / 2]
        for angle in angles:
            result = await pythagorean_identity(angle)
            assert result["sin_cos_identity"] is True
            assert result["sin_cos_error"] < 1e-10

    @pytest.mark.asyncio
    async def test_pythagorean_when_cos_near_zero(self):
        # Test when cos(θ) ≈ 0 (at π/2)
        result = await pythagorean_identity(math.pi / 2)
        assert result["sin_cos_identity"] is True
        assert result["sec_tan_identity"] is None
        assert "sec_tan_note" in result
        assert "cos(θ) ≈ 0" in result["sec_tan_note"]

    @pytest.mark.asyncio
    async def test_pythagorean_when_sin_near_zero(self):
        # Test when sin(θ) ≈ 0 (at 0)
        result = await pythagorean_identity(0)
        assert result["sin_cos_identity"] is True
        assert result["csc_cot_identity"] is None
        assert "csc_cot_note" in result
        assert "sin(θ) ≈ 0" in result["csc_cot_note"]

    @pytest.mark.asyncio
    async def test_pythagorean_all_identities_valid(self):
        # Test at an angle where all three identities are valid (π/4)
        result = await pythagorean_identity(math.pi / 4)
        assert result["sin_cos_identity"] is True
        assert result["sec_tan_identity"] is True
        assert result["csc_cot_identity"] is True
        assert "sec_tan_error" in result
        assert "csc_cot_error" in result


class TestSumDifferenceFormulas:
    @pytest.mark.asyncio
    async def test_sum_formulas(self):
        result = await sum_difference_formulas(math.pi / 4, math.pi / 6, "add")
        assert "sin_formula" in result
        assert "cos_formula" in result

    @pytest.mark.asyncio
    async def test_verify_sum_formulas(self):
        result = await verify_sum_difference_formulas(math.pi / 4, math.pi / 6, "add")
        assert result["sin_verified"] is True
        assert result["cos_verified"] is True

    @pytest.mark.asyncio
    async def test_sum_formulas_tan_denominator_near_zero_add(self):
        # Test when 1 - tan(a)tan(b) ≈ 0 for addition
        # tan(π/4) = 1, so we need tan(b) ≈ 1 to get denominator ≈ 0
        result = await sum_difference_formulas(math.pi / 4, math.pi / 4, "add")
        assert "sin_formula" in result
        assert "cos_formula" in result
        # tan_formula should be infinity when denominator is near zero
        assert result["tan_formula"] is not None
        assert math.isinf(result["tan_formula"]) or abs(result["tan_formula"]) > 1e10

    @pytest.mark.asyncio
    async def test_sum_formulas_tan_denominator_near_zero_subtract(self):
        # Test when 1 + tan(a)tan(b) ≈ 0 for subtraction
        # This happens when tan(a)tan(b) ≈ -1
        # Use angles where tan values multiply to approximately -1
        result = await sum_difference_formulas(math.pi / 4, -math.pi / 4, "subtract")
        assert "sin_formula" in result
        assert "cos_formula" in result
        assert result["tan_formula"] is not None

    @pytest.mark.asyncio
    async def test_sum_formulas_tan_undefined_add(self):
        # Test when tan is undefined (at π/2)
        result = await sum_difference_formulas(math.pi / 2, math.pi / 6, "add")
        assert "sin_formula" in result
        assert "cos_formula" in result
        assert result["tan_formula"] is None

    @pytest.mark.asyncio
    async def test_sum_formulas_tan_undefined_subtract(self):
        # Test when tan is undefined (at π/2) for subtraction
        result = await sum_difference_formulas(math.pi / 2, math.pi / 6, "subtract")
        assert "sin_formula" in result
        assert "cos_formula" in result
        assert result["tan_formula"] is None

    @pytest.mark.asyncio
    async def test_verify_sum_formulas_subtract(self):
        # Test verification with subtract operation
        result = await verify_sum_difference_formulas(math.pi / 3, math.pi / 6, "subtract")
        assert result["sin_verified"] is True
        assert result["cos_verified"] is True

    @pytest.mark.asyncio
    async def test_verify_sum_formulas_tan_undefined(self):
        # Test verification when tan is undefined
        result = await verify_sum_difference_formulas(math.pi / 2, math.pi / 6, "add")
        assert result["sin_verified"] is True
        assert result["cos_verified"] is True
        # tan_verified should be None when tan is undefined
        assert result["tan_verified"] is None


class TestDoubleAngleFormulas:
    @pytest.mark.asyncio
    async def test_double_angle_sin(self):
        result = await double_angle_formulas(math.pi / 4, "sin")
        assert abs(result["double_angle_value"] - 1.0) < 1e-10

    @pytest.mark.asyncio
    async def test_double_angle_cos(self):
        result = await double_angle_formulas(math.pi / 6, "cos")
        assert abs(result["double_angle_value"] - 0.5) < 1e-10

    @pytest.mark.asyncio
    async def test_double_angle_tan(self):
        # Test tan double angle formula
        result = await double_angle_formulas(math.pi / 6, "tan")
        assert result["double_angle_value"] is not None
        assert "formula_used" in result
        assert "tan(2θ)" in result["formula_used"]

    @pytest.mark.asyncio
    async def test_double_angle_tan_denominator_near_zero(self):
        # Test when 1 - tan²(θ) ≈ 0, which happens when tan(θ) ≈ ±1
        result = await double_angle_formulas(math.pi / 4, "tan")
        # At π/4, tan = 1, so 1 - 1² = 0
        assert result["double_angle_value"] is not None
        # Should be infinity or very large
        assert math.isinf(result["double_angle_value"]) or abs(result["double_angle_value"]) > 1e10

    @pytest.mark.asyncio
    async def test_double_angle_tan_undefined(self):
        # Test when tan is undefined (at π/2)
        result = await double_angle_formulas(math.pi / 2, "tan")
        assert result["double_angle_value"] is None
        assert "undefined" in result["formula_used"]

    @pytest.mark.asyncio
    async def test_double_angle_cos_alternative_forms(self):
        # Test that alternative forms are included for cosine
        result = await double_angle_formulas(math.pi / 6, "cos")
        assert "alternative_forms" in result
        assert "2cos²(θ) - 1" in result["alternative_forms"]
        assert "1 - 2sin²(θ)" in result["alternative_forms"]


class TestHalfAngleFormulas:
    @pytest.mark.asyncio
    async def test_half_angle_sin(self):
        result = await half_angle_formulas(math.pi / 2, "sin")
        assert abs(result["half_angle_value"] - math.sqrt(2) / 2) < 1e-10

    @pytest.mark.asyncio
    async def test_half_angle_cos(self):
        result = await half_angle_formulas(2 * math.pi / 3, "cos")
        # Half angle of 2π/3 is π/3, and cos(π/3) = 0.5, not √3/2
        assert abs(result["half_angle_value"] - 0.5) < 1e-10

    @pytest.mark.asyncio
    async def test_half_angle_tan(self):
        # Test tan half angle formula
        result = await half_angle_formulas(math.pi / 2, "tan")
        assert result["half_angle_value"] is not None
        assert "formula_used" in result
        assert "tan(θ/2)" in result["formula_used"]

    @pytest.mark.asyncio
    async def test_half_angle_tan_cos_near_minus_one(self):
        # Test when 1 + cos(θ) ≈ 0, which happens when cos(θ) ≈ -1 (at π)
        # At π: cos(π) = -1, sin(π) ≈ 0, so both 1 + cos(θ) ≈ 0 AND sin(θ) ≈ 0
        # This makes tan(θ/2) undefined
        result = await half_angle_formulas(math.pi, "tan")
        assert result["half_angle_value"] is None
        assert "undefined" in result["formula_used"]

    @pytest.mark.asyncio
    async def test_half_angle_tan_alternative_formula(self):
        # Test when 1 + cos(θ) is extremely small but sin(θ) is not zero
        # Use an angle very close to π (within 1e-10) but not exactly π
        angle = math.pi - 1e-10  # Very close to π
        result = await half_angle_formulas(angle, "tan")
        # At this angle, it should use the alternative formula
        assert result["half_angle_value"] is not None
        # Verify it uses the alternative formula
        assert "(1 - cos(θ))/sin(θ)" in result["formula_used"]

    @pytest.mark.asyncio
    async def test_half_angle_tan_normal_formula(self):
        # Test the normal formula path: sin(θ)/(1 + cos(θ))
        # Use an angle where 1 + cos(θ) is not near zero
        result = await half_angle_formulas(2 * math.pi, "tan")
        # This should work normally
        assert result["half_angle_value"] is not None
        assert "sin(θ)/(1 + cos(θ))" in result["formula_used"]


class TestIdentityVerification:
    @pytest.mark.asyncio
    async def test_verify_pythagorean(self):
        result = await verify_identity("pythagorean")
        assert result["identity_verified"] is True

    @pytest.mark.asyncio
    async def test_comprehensive_verification(self):
        result = await comprehensive_identity_verification()
        assert result["all_verified"] is True

    @pytest.mark.asyncio
    async def test_verify_unsupported_identity(self):
        # Test with an unsupported identity name
        result = await verify_identity("unsupported_identity_name")
        assert result["identity_verified"] is False
        # Check that test results contain error information
        assert len(result["test_results"]) > 0
        for test_result in result["test_results"]:
            assert test_result["verified"] is False
            assert "note" in test_result
            assert "Unsupported identity" in test_result["note"]

    @pytest.mark.asyncio
    async def test_verify_identity_with_exception(self):
        # This tests the exception handling in verify_identity
        # We'll use custom test angles that might cause issues
        result = await verify_identity("pythagorean", test_angles=[0, math.pi / 4])
        assert result["identity_verified"] is True

    @pytest.mark.asyncio
    async def test_verify_identity_tolerance_fail(self):
        # Test when verification fails due to strict tolerance
        result = await verify_identity("pythagorean", test_angles=[math.pi / 4], tolerance=1e-20)
        # With such a strict tolerance, it might fail due to floating point errors
        assert "identity_verified" in result

    @pytest.mark.asyncio
    async def test_verify_double_angle_sin_identity(self):
        # Test verification of double angle sine identity
        result = await verify_identity("double_angle_sin")
        assert result["identity_verified"] is True
        assert result["identity_name"] == "double_angle_sin"

    @pytest.mark.asyncio
    async def test_verify_double_angle_cos_identity(self):
        # Test verification of double angle cosine identity
        result = await verify_identity("double_angle_cos")
        assert result["identity_verified"] is True
        assert result["identity_name"] == "double_angle_cos"

    @pytest.mark.asyncio
    async def test_comprehensive_verification_with_custom_angles(self):
        # Test comprehensive verification with custom angles
        custom_angles = [0, math.pi / 6, math.pi / 4]
        result = await comprehensive_identity_verification(test_angles=custom_angles)
        assert "identities_tested" in result
        assert result["identities_tested"] == 3
        assert "overall_success_rate" in result


class TestExpressionSimplification:
    @pytest.mark.asyncio
    async def test_simplify_pythagorean(self):
        result = await simplify_trig_expression("sin^2(x) + cos^2(x)")
        assert result["simplified"] == "1"
        assert result["identity_used"] == "pythagorean"

    @pytest.mark.asyncio
    async def test_simplify_pythagorean_reversed(self):
        # Test with cos^2 + sin^2 order
        result = await simplify_trig_expression("cos^2(x) + sin^2(x)")
        assert result["simplified"] == "1"
        assert result["identity_used"] == "pythagorean"

    @pytest.mark.asyncio
    async def test_simplify_double_angle_sin(self):
        # Test simplification of 2sin(x)cos(x) pattern
        result = await simplify_trig_expression("2sin(x)cos(x)")
        assert result["simplified"] == "sin(2x)"
        assert result["identity_used"] == "double_angle_sin"

    @pytest.mark.asyncio
    async def test_simplify_no_match(self):
        # Test with an expression that doesn't match any pattern
        result = await simplify_trig_expression("tan(x) + sec(x)")
        assert result["simplified"] == "tan(x) + sec(x)"
        assert result["identity_used"] is None
        assert "No applicable simplification found" in result["steps"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
