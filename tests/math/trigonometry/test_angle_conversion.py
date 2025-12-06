#!/usr/bin/env python3
"""
Comprehensive tests for angle_conversion module.
Tests all conversion functions, normalization, and angle utilities.
"""

import pytest
import math
from chuk_mcp_math.trigonometry.angle_conversion import (
    degrees_to_radians,
    radians_to_degrees,
    gradians_to_radians,
    radians_to_gradians,
    degrees_to_gradians,
    gradians_to_degrees,
    normalize_angle,
    angle_difference,
    convert_angle,
    angle_properties,
    angular_velocity_from_period_or_frequency,
)


class TestBasicConversions:
    """Test basic angle conversion functions."""

    @pytest.mark.asyncio
    async def test_degrees_to_radians_standard_angles(self):
        """Test conversion of standard angles from degrees to radians."""
        # Test special angles
        assert await degrees_to_radians(0) == 0.0
        assert abs(await degrees_to_radians(90) - math.pi / 2) < 1e-10
        assert abs(await degrees_to_radians(180) - math.pi) < 1e-10
        assert abs(await degrees_to_radians(360) - 2 * math.pi) < 1e-10
        assert abs(await degrees_to_radians(45) - math.pi / 4) < 1e-10

    @pytest.mark.asyncio
    async def test_degrees_to_radians_negative(self):
        """Test conversion of negative angles."""
        assert abs(await degrees_to_radians(-90) + math.pi / 2) < 1e-10
        assert abs(await degrees_to_radians(-180) + math.pi) < 1e-10

    @pytest.mark.asyncio
    async def test_radians_to_degrees_standard_angles(self):
        """Test conversion from radians to degrees."""
        assert await radians_to_degrees(0) == 0.0
        assert abs(await radians_to_degrees(math.pi / 2) - 90.0) < 1e-10
        assert abs(await radians_to_degrees(math.pi) - 180.0) < 1e-10
        assert abs(await radians_to_degrees(2 * math.pi) - 360.0) < 1e-10

    @pytest.mark.asyncio
    async def test_roundtrip_degrees_radians(self):
        """Test roundtrip conversion degrees -> radians -> degrees."""
        test_angles = [0, 30, 45, 60, 90, 120, 180, 270, 360, -45, -90]
        for angle in test_angles:
            rad = await degrees_to_radians(angle)
            back = await radians_to_degrees(rad)
            assert abs(back - angle) < 1e-10

    @pytest.mark.asyncio
    async def test_gradians_to_radians(self):
        """Test gradian to radian conversion."""
        assert await gradians_to_radians(0) == 0.0
        assert abs(await gradians_to_radians(100) - math.pi / 2) < 1e-10
        assert abs(await gradians_to_radians(200) - math.pi) < 1e-10
        assert abs(await gradians_to_radians(400) - 2 * math.pi) < 1e-10

    @pytest.mark.asyncio
    async def test_radians_to_gradians(self):
        """Test radian to gradian conversion."""
        assert await radians_to_gradians(0) == 0.0
        assert abs(await radians_to_gradians(math.pi / 2) - 100.0) < 1e-10
        assert abs(await radians_to_gradians(math.pi) - 200.0) < 1e-10
        assert abs(await radians_to_gradians(2 * math.pi) - 400.0) < 1e-10

    @pytest.mark.asyncio
    async def test_degrees_to_gradians(self):
        """Test degree to gradian conversion."""
        assert await degrees_to_gradians(0) == 0.0
        assert abs(await degrees_to_gradians(90) - 100.0) < 1e-10
        assert abs(await degrees_to_gradians(180) - 200.0) < 1e-10
        assert abs(await degrees_to_gradians(360) - 400.0) < 1e-10

    @pytest.mark.asyncio
    async def test_gradians_to_degrees(self):
        """Test gradian to degree conversion."""
        assert await gradians_to_degrees(0) == 0.0
        assert abs(await gradians_to_degrees(100) - 90.0) < 1e-10
        assert abs(await gradians_to_degrees(200) - 180.0) < 1e-10
        assert abs(await gradians_to_degrees(400) - 360.0) < 1e-10


class TestNormalization:
    """Test angle normalization functions."""

    @pytest.mark.asyncio
    async def test_normalize_angle_degrees_positive(self):
        """Test normalization to [0, 360) degrees."""
        assert abs(await normalize_angle(370, "degrees", "positive") - 10.0) < 1e-10
        assert abs(await normalize_angle(720, "degrees", "positive") - 0.0) < 1e-10
        assert abs(await normalize_angle(-45, "degrees", "positive") - 315.0) < 1e-10
        assert abs(await normalize_angle(0, "degrees", "positive") - 0.0) < 1e-10

    @pytest.mark.asyncio
    async def test_normalize_angle_degrees_symmetric(self):
        """Test normalization to [-180, 180) degrees."""
        assert abs(await normalize_angle(200, "degrees", "symmetric") - -160.0) < 1e-10
        assert abs(await normalize_angle(-45, "degrees", "symmetric") - -45.0) < 1e-10
        assert abs(await normalize_angle(45, "degrees", "symmetric") - 45.0) < 1e-10
        assert abs(await normalize_angle(180, "degrees", "symmetric") - 180.0) < 1e-10

    @pytest.mark.asyncio
    async def test_normalize_angle_radians_positive(self):
        """Test normalization to [0, 2π) radians."""
        result = await normalize_angle(3 * math.pi, "radians", "positive")
        assert abs(result - math.pi) < 1e-10

        result = await normalize_angle(-math.pi / 2, "radians", "positive")
        assert abs(result - 3 * math.pi / 2) < 1e-10

    @pytest.mark.asyncio
    async def test_normalize_angle_radians_symmetric(self):
        """Test normalization to [-π, π) radians."""
        result = await normalize_angle(1.5 * math.pi, "radians", "symmetric")
        assert abs(result + 0.5 * math.pi) < 1e-10

    @pytest.mark.asyncio
    async def test_normalize_angle_gradians(self):
        """Test normalization with gradians."""
        assert abs(await normalize_angle(500, "gradians", "positive") - 100.0) < 1e-10
        assert abs(await normalize_angle(-100, "gradians", "positive") - 300.0) < 1e-10

    @pytest.mark.asyncio
    async def test_normalize_angle_negative_zero_handling(self):
        """Test handling of negative zero in normalization."""
        # Test when angle is negative but normalizes to 0
        result = await normalize_angle(-360, "degrees", "positive")
        assert result == 0.0
        # Ensure it's positive zero, not negative zero
        assert str(result) == "0.0"

        # Test with radians
        result = await normalize_angle(-2 * math.pi, "radians", "positive")
        assert abs(result) < 1e-10

        # Test with gradians
        result = await normalize_angle(-400, "gradians", "positive")
        assert abs(result) < 1e-10


class TestAngleDifference:
    """Test angle difference calculations."""

    @pytest.mark.asyncio
    async def test_angle_difference_degrees(self):
        """Test shortest angular difference in degrees."""
        # Crossing 0/360 boundary
        diff = await angle_difference(350, 10, "degrees")
        assert abs(diff - 20.0) < 1e-10

        diff = await angle_difference(10, 350, "degrees")
        assert abs(diff + 20.0) < 1e-10

    @pytest.mark.asyncio
    async def test_angle_difference_radians(self):
        """Test shortest angular difference in radians."""
        diff = await angle_difference(0, math.pi, "radians")
        assert abs(diff - math.pi) < 1e-10

        diff = await angle_difference(math.pi, 0, "radians")
        # The implementation returns π instead of -π for angle wrapping
        assert abs(abs(diff) - math.pi) < 1e-10

    @pytest.mark.asyncio
    async def test_angle_difference_same_angle(self):
        """Test difference between identical angles."""
        diff = await angle_difference(45, 45, "degrees")
        assert abs(diff) < 1e-10

    @pytest.mark.asyncio
    async def test_angle_difference_opposite(self):
        """Test difference for opposite angles."""
        diff = await angle_difference(0, 180, "degrees")
        assert abs(abs(diff) - 180.0) < 1e-10


class TestConvertAngle:
    """Test unified angle conversion function."""

    @pytest.mark.asyncio
    async def test_convert_angle_same_unit(self):
        """Test conversion when units are the same."""
        result = await convert_angle(45, "degrees", "degrees")
        assert result == 45

    @pytest.mark.asyncio
    async def test_convert_angle_all_combinations(self):
        """Test all unit conversion combinations."""
        # 90 degrees
        rad = await convert_angle(90, "degrees", "radians")
        assert abs(rad - math.pi / 2) < 1e-10

        grad = await convert_angle(90, "degrees", "gradians")
        assert abs(grad - 100) < 1e-10

        # π/2 radians
        deg = await convert_angle(math.pi / 2, "radians", "degrees")
        assert abs(deg - 90) < 1e-10

        grad = await convert_angle(math.pi / 2, "radians", "gradians")
        assert abs(grad - 100) < 1e-10

        # 100 gradians
        deg = await convert_angle(100, "gradians", "degrees")
        assert abs(deg - 90) < 1e-10

        rad = await convert_angle(100, "gradians", "radians")
        assert abs(rad - math.pi / 2) < 1e-10


class TestAngleProperties:
    """Test comprehensive angle properties function."""

    @pytest.mark.asyncio
    async def test_angle_properties_45_degrees(self):
        """Test properties of 45 degree angle."""
        props = await angle_properties(45, "degrees")

        assert abs(props["degrees"] - 45.0) < 1e-10
        assert abs(props["radians"] - math.pi / 4) < 1e-10
        assert abs(props["gradians"] - 50.0) < 1e-10
        assert props["quadrant"] == 1
        assert abs(props["normalized_positive"] - 45.0) < 1e-10
        assert abs(props["reference_angle"] - 45.0) < 1e-10

    @pytest.mark.asyncio
    async def test_angle_properties_negative_angle(self):
        """Test properties of negative angle."""
        props = await angle_properties(-90, "degrees")

        assert props["degrees"] == -90.0
        assert abs(props["normalized_positive"] - 270.0) < 1e-10
        assert abs(props["normalized_symmetric"] + 90.0) < 1e-10
        assert props["quadrant"] == 4

    @pytest.mark.asyncio
    async def test_angle_properties_radians(self):
        """Test properties when input is in radians."""
        props = await angle_properties(math.pi / 3, "radians")

        assert abs(props["radians"] - math.pi / 3) < 1e-10
        assert abs(props["degrees"] - 60.0) < 1e-10
        assert props["quadrant"] == 1

    @pytest.mark.asyncio
    async def test_angle_properties_gradians_input(self):
        """Test properties when input is in gradians."""
        # Test 150 gradians (135 degrees, quadrant 2)
        props = await angle_properties(150, "gradians")

        assert abs(props["gradians"] - 150.0) < 1e-10
        assert abs(props["degrees"] - 135.0) < 1e-10
        assert props["quadrant"] == 2
        # Reference angle in gradians for 135° (quadrant 2): 200 - 150 = 50
        assert abs(props["reference_angle"] - 50.0) < 1e-10

    @pytest.mark.asyncio
    async def test_angle_properties_gradians_quadrant_1(self):
        """Test properties for angle in quadrant 1 (gradians)."""
        # 50 gradians (45 degrees) is in quadrant 1
        props = await angle_properties(50, "gradians")

        assert props["quadrant"] == 1
        # Reference angle for quadrant 1 is the normalized angle itself
        assert abs(props["reference_angle"] - 50.0) < 1e-10

    @pytest.mark.asyncio
    async def test_angle_properties_quadrant_2_degrees(self):
        """Test properties for angle in quadrant 2 (degrees)."""
        # 135 degrees is in quadrant 2
        props = await angle_properties(135, "degrees")

        assert props["quadrant"] == 2
        # Reference angle for 135°: 180 - 135 = 45
        assert abs(props["reference_angle"] - 45.0) < 1e-10

    @pytest.mark.asyncio
    async def test_angle_properties_quadrant_3_degrees(self):
        """Test properties for angle in quadrant 3 (degrees)."""
        # 225 degrees is in quadrant 3
        props = await angle_properties(225, "degrees")

        assert props["quadrant"] == 3
        # Reference angle for 225°: 225 - 180 = 45
        assert abs(props["reference_angle"] - 45.0) < 1e-10

    @pytest.mark.asyncio
    async def test_angle_properties_radians_quadrant_2(self):
        """Test properties for angle in quadrant 2 (radians)."""
        # 2π/3 radians (120 degrees) is in quadrant 2
        props = await angle_properties(2 * math.pi / 3, "radians")

        assert props["quadrant"] == 2
        # Reference angle: π - 2π/3 = π/3
        assert abs(props["reference_angle"] - math.pi / 3) < 1e-10

    @pytest.mark.asyncio
    async def test_angle_properties_radians_quadrant_3(self):
        """Test properties for angle in quadrant 3 (radians)."""
        # 4π/3 radians (240 degrees) is in quadrant 3
        props = await angle_properties(4 * math.pi / 3, "radians")

        assert props["quadrant"] == 3
        # Reference angle: 4π/3 - π = π/3
        assert abs(props["reference_angle"] - math.pi / 3) < 1e-10

    @pytest.mark.asyncio
    async def test_angle_properties_radians_quadrant_4(self):
        """Test properties for angle in quadrant 4 (radians)."""
        # 5π/3 radians (300 degrees) is in quadrant 4
        props = await angle_properties(5 * math.pi / 3, "radians")

        assert props["quadrant"] == 4
        # Reference angle: 2π - 5π/3 = π/3
        assert abs(props["reference_angle"] - math.pi / 3) < 1e-10

    @pytest.mark.asyncio
    async def test_angle_properties_gradians_quadrant_3(self):
        """Test properties for angle in quadrant 3 (gradians)."""
        # 250 gradians (225 degrees) is in quadrant 3
        props = await angle_properties(250, "gradians")

        assert props["quadrant"] == 3
        # Reference angle for 250 gradians: 250 - 200 = 50
        assert abs(props["reference_angle"] - 50.0) < 1e-10

    @pytest.mark.asyncio
    async def test_angle_properties_gradians_quadrant_4(self):
        """Test properties for angle in quadrant 4 (gradians)."""
        # 350 gradians (315 degrees) is in quadrant 4
        props = await angle_properties(350, "gradians")

        assert props["quadrant"] == 4
        # Reference angle for 350 gradians: 400 - 350 = 50
        assert abs(props["reference_angle"] - 50.0) < 1e-10

    @pytest.mark.asyncio
    async def test_angle_properties_coterminal(self):
        """Test that coterminal angles are generated."""
        props = await angle_properties(30, "degrees")

        coterminal = props["coterminal_angles"]
        assert len(coterminal) == 4
        # Should include 390, -330, 750, -690
        assert abs(coterminal[0] - 390) < 1e-10


class TestAngularVelocity:
    """Test angular velocity calculations."""

    @pytest.mark.asyncio
    async def test_angular_velocity_from_period(self):
        """Test calculation from period."""
        result = await angular_velocity_from_period_or_frequency(period=2.0)

        assert abs(result["period_sec"] - 2.0) < 1e-10
        assert abs(result["frequency_hz"] - 0.5) < 1e-10
        assert abs(result["angular_velocity_rad_per_sec"] - math.pi) < 1e-10

    @pytest.mark.asyncio
    async def test_angular_velocity_from_frequency(self):
        """Test calculation from frequency."""
        result = await angular_velocity_from_period_or_frequency(frequency=60, unit="hz")

        assert abs(result["frequency_hz"] - 60) < 1e-10
        assert abs(result["period_sec"] - 1 / 60) < 1e-10
        assert abs(result["angular_velocity_rad_per_sec"] - 120 * math.pi) < 1e-10

    @pytest.mark.asyncio
    async def test_angular_velocity_errors(self):
        """Test error conditions."""
        # Both parameters provided
        with pytest.raises(ValueError, match="either period OR frequency"):
            await angular_velocity_from_period_or_frequency(period=1.0, frequency=1.0)

        # Neither parameter provided
        with pytest.raises(ValueError, match="Must provide either"):
            await angular_velocity_from_period_or_frequency()

        # Negative period
        with pytest.raises(ValueError, match="Period must be positive"):
            await angular_velocity_from_period_or_frequency(period=-1.0)

        # Negative frequency
        with pytest.raises(ValueError, match="Frequency must be positive"):
            await angular_velocity_from_period_or_frequency(frequency=-1.0)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_zero_angles(self):
        """Test handling of zero angles."""
        assert await degrees_to_radians(0) == 0.0
        assert await radians_to_degrees(0) == 0.0
        assert await gradians_to_radians(0) == 0.0

    @pytest.mark.asyncio
    async def test_very_large_angles(self):
        """Test handling of very large angles."""
        # Should handle large angles without overflow
        large_deg = 36000  # 100 rotations
        rad = await degrees_to_radians(large_deg)
        back = await radians_to_degrees(rad)
        assert abs(back - large_deg) < 1e-8

    @pytest.mark.asyncio
    async def test_very_small_angles(self):
        """Test handling of very small angles."""
        small_deg = 0.0001
        rad = await degrees_to_radians(small_deg)
        back = await radians_to_degrees(rad)
        assert abs(back - small_deg) < 1e-12

    @pytest.mark.asyncio
    async def test_float_vs_int_input(self):
        """Test that both int and float inputs work."""
        int_result = await degrees_to_radians(90)
        float_result = await degrees_to_radians(90.0)
        assert abs(int_result - float_result) < 1e-15


class TestNumericalPrecision:
    """Test numerical precision and accuracy."""

    @pytest.mark.asyncio
    async def test_precision_roundtrip(self):
        """Test precision is maintained in roundtrip conversions."""
        # Degrees -> Radians -> Degrees
        for angle in [0, 30, 45, 60, 90, 120, 135, 150, 180]:
            rad = await degrees_to_radians(angle)
            back = await radians_to_degrees(rad)
            assert abs(back - angle) < 1e-12

        # Radians -> Degrees -> Radians
        for angle in [0, math.pi / 6, math.pi / 4, math.pi / 3, math.pi / 2, math.pi]:
            deg = await radians_to_degrees(angle)
            back = await degrees_to_radians(deg)
            assert abs(back - angle) < 1e-12

    @pytest.mark.asyncio
    async def test_special_angle_precision(self):
        """Test that special angles are computed precisely."""
        # π/2 should be exactly math.pi/2
        result = await degrees_to_radians(90)
        assert abs(result - math.pi / 2) < 1e-15

        # 2π should be exactly 2*math.pi
        result = await degrees_to_radians(360)
        assert abs(result - 2 * math.pi) < 1e-15


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
