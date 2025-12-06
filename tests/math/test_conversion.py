#!/usr/bin/env python3
# tests/math/test_conversion.py
"""
Comprehensive pytest unit tests for conversion functions.

Tests cover:
- Temperature conversions (Celsius, Fahrenheit, Kelvin, Rankine)
- Length conversions (meters, feet, miles, etc.)
- Weight conversions (kilograms, pounds, etc.)
- Area conversions (square meters, acres, etc.)
- Volume conversions (liters, gallons, etc.)
- Time conversions (seconds, hours, days, etc.)
- Speed conversions (mph, kph, etc.)
- Normal operation cases
- Edge cases (zero, negative)
- Error conditions
- Async behavior
- Roundtrip conversions
"""

import pytest
import asyncio

# Import the functions to test
from chuk_mcp_math.conversion import (
    convert_temperature,
    convert_length,
    convert_weight,
    convert_area,
    convert_volume,
    convert_time,
    convert_speed,
)


# Temperature Conversion Tests
class TestConvertTemperature:
    """Test cases for temperature conversion."""

    @pytest.mark.asyncio
    async def test_celsius_to_fahrenheit(self):
        """Test Celsius to Fahrenheit conversion."""
        result = await convert_temperature(0, "celsius", "fahrenheit")
        assert result["result"] == 32.0
        assert result["formula"] == "C × 9/5 + 32"

        result = await convert_temperature(100, "celsius", "fahrenheit")
        assert result["result"] == 212.0

    @pytest.mark.asyncio
    async def test_fahrenheit_to_celsius(self):
        """Test Fahrenheit to Celsius conversion."""
        result = await convert_temperature(32, "fahrenheit", "celsius")
        assert result["result"] == 0.0
        assert result["formula"] == "(F - 32) × 5/9"

        result = await convert_temperature(212, "fahrenheit", "celsius")
        assert pytest.approx(result["result"]) == 100.0

    @pytest.mark.asyncio
    async def test_celsius_to_kelvin(self):
        """Test Celsius to Kelvin conversion."""
        result = await convert_temperature(0, "celsius", "kelvin")
        assert result["result"] == 273.15
        assert result["formula"] == "C + 273.15"

        result = await convert_temperature(-273.15, "celsius", "kelvin")
        assert result["result"] == 0.0

    @pytest.mark.asyncio
    async def test_kelvin_to_celsius(self):
        """Test Kelvin to Celsius conversion."""
        result = await convert_temperature(273.15, "kelvin", "celsius")
        assert result["result"] == 0.0

        result = await convert_temperature(0, "kelvin", "celsius")
        assert result["result"] == -273.15

    @pytest.mark.asyncio
    async def test_celsius_to_rankine(self):
        """Test Celsius to Rankine conversion."""
        result = await convert_temperature(0, "celsius", "rankine")
        assert pytest.approx(result["result"]) == 491.67

    @pytest.mark.asyncio
    async def test_rankine_to_celsius(self):
        """Test Rankine to Celsius conversion."""
        result = await convert_temperature(491.67, "rankine", "celsius")
        assert pytest.approx(result["result"]) == 0.0

    @pytest.mark.asyncio
    async def test_temperature_same_scale(self):
        """Test conversion to same scale."""
        result = await convert_temperature(25, "celsius", "celsius")
        assert result["result"] == 25

    @pytest.mark.asyncio
    async def test_temperature_metadata(self):
        """Test that result includes metadata."""
        result = await convert_temperature(100, "celsius", "fahrenheit")
        assert "result" in result
        assert "original_value" in result
        assert "original_scale" in result
        assert "target_scale" in result
        assert "formula" in result
        assert "celsius_equivalent" in result

    @pytest.mark.asyncio
    async def test_temperature_invalid_scale(self):
        """Test with invalid temperature scale."""
        with pytest.raises(ValueError, match="Invalid scale"):
            await convert_temperature(100, "celsius", "invalid")

    @pytest.mark.asyncio
    async def test_temperature_below_absolute_zero(self):
        """Test with temperature below absolute zero."""
        with pytest.raises(ValueError, match="cannot be below absolute zero"):
            await convert_temperature(-300, "celsius", "fahrenheit")

        with pytest.raises(ValueError, match="cannot be below absolute zero"):
            await convert_temperature(-500, "fahrenheit", "celsius")

    @pytest.mark.asyncio
    async def test_temperature_negative_kelvin(self):
        """Test that negative Kelvin raises error."""
        with pytest.raises(ValueError, match="Kelvin temperature cannot be negative"):
            await convert_temperature(-1, "kelvin", "celsius")


# Length Conversion Tests
class TestConvertLength:
    """Test cases for length conversion."""

    @pytest.mark.asyncio
    async def test_meter_to_feet(self):
        """Test meter to feet conversion."""
        result = await convert_length(1, "meter", "feet")
        assert pytest.approx(result, rel=1e-6) == 3.2808398950131233

    @pytest.mark.asyncio
    async def test_feet_to_meter(self):
        """Test feet to meter conversion."""
        result = await convert_length(3.2808398950131233, "feet", "meter")
        assert pytest.approx(result, rel=1e-6) == 1.0

    @pytest.mark.asyncio
    async def test_kilometer_to_mile(self):
        """Test kilometer to mile conversion."""
        result = await convert_length(1, "kilometer", "mile")
        assert pytest.approx(result, rel=1e-6) == 0.6213711922373339

    @pytest.mark.asyncio
    async def test_mile_to_kilometer(self):
        """Test mile to kilometer conversion."""
        result = await convert_length(1, "mile", "kilometer")
        assert pytest.approx(result, rel=1e-6) == 1.609344

    @pytest.mark.asyncio
    async def test_feet_to_inches(self):
        """Test feet to inches conversion."""
        result = await convert_length(5, "feet", "inches")
        assert result == 60.0

    @pytest.mark.asyncio
    async def test_inch_to_centimeter(self):
        """Test inch to centimeter conversion."""
        result = await convert_length(1, "inch", "centimeter")
        assert result == 2.54

    @pytest.mark.asyncio
    async def test_length_same_unit(self):
        """Test conversion to same unit."""
        result = await convert_length(10, "meter", "meter")
        assert result == 10

    @pytest.mark.asyncio
    async def test_length_case_insensitive(self):
        """Test case insensitivity."""
        result1 = await convert_length(1, "METER", "FEET")
        result2 = await convert_length(1, "meter", "feet")
        assert result1 == result2

    @pytest.mark.asyncio
    async def test_length_negative_raises_error(self):
        """Test that negative length raises error."""
        with pytest.raises(ValueError, match="Length cannot be negative"):
            await convert_length(-5, "meter", "feet")

    @pytest.mark.asyncio
    async def test_length_invalid_unit(self):
        """Test with invalid unit."""
        with pytest.raises(ValueError, match="Unsupported"):
            await convert_length(10, "meter", "invalid")


# Weight Conversion Tests
class TestConvertWeight:
    """Test cases for weight conversion."""

    @pytest.mark.asyncio
    async def test_kilogram_to_pound(self):
        """Test kilogram to pound conversion."""
        result = await convert_weight(1, "kilogram", "pound")
        assert pytest.approx(result, rel=1e-6) == 2.2046226218487757

    @pytest.mark.asyncio
    async def test_pound_to_kilogram(self):
        """Test pound to kilogram conversion."""
        result = await convert_weight(2.2046226218487757, "pound", "kilogram")
        assert pytest.approx(result, rel=1e-6) == 1.0

    @pytest.mark.asyncio
    async def test_ounce_to_pound(self):
        """Test ounce to pound conversion."""
        result = await convert_weight(16, "ounce", "pound")
        assert pytest.approx(result, rel=1e-6) == 1.0

    @pytest.mark.asyncio
    async def test_gram_to_kilogram(self):
        """Test gram to kilogram conversion."""
        result = await convert_weight(1000, "gram", "kilogram")
        assert result == 1.0

    @pytest.mark.asyncio
    async def test_ton_to_kilogram(self):
        """Test ton to kilogram conversion."""
        result = await convert_weight(1, "ton", "kilogram")
        assert result == 1000.0

    @pytest.mark.asyncio
    async def test_weight_same_unit(self):
        """Test conversion to same unit."""
        result = await convert_weight(10, "kilogram", "kilogram")
        assert result == 10

    @pytest.mark.asyncio
    async def test_weight_negative_raises_error(self):
        """Test that negative weight raises error."""
        with pytest.raises(ValueError, match="Weight cannot be negative"):
            await convert_weight(-5, "kilogram", "pound")

    @pytest.mark.asyncio
    async def test_weight_invalid_unit(self):
        """Test with invalid unit."""
        with pytest.raises(ValueError, match="Unsupported"):
            await convert_weight(10, "kilogram", "invalid")


# Area Conversion Tests
class TestConvertArea:
    """Test cases for area conversion."""

    @pytest.mark.asyncio
    async def test_square_meter_to_square_feet(self):
        """Test square meter to square feet conversion."""
        result = await convert_area(1, "square_meter", "square_feet")
        assert pytest.approx(result, rel=1e-6) == 10.76391041670972

    @pytest.mark.asyncio
    async def test_acre_to_hectare(self):
        """Test acre to hectare conversion."""
        result = await convert_area(1, "acre", "hectare")
        assert pytest.approx(result, rel=1e-6) == 0.40468564224

    @pytest.mark.asyncio
    async def test_hectare_to_square_meter(self):
        """Test hectare to square meter conversion."""
        result = await convert_area(1, "hectare", "square_meter")
        assert result == 10000.0

    @pytest.mark.asyncio
    async def test_square_kilometer_to_square_meter(self):
        """Test square kilometer to square meter conversion."""
        result = await convert_area(1, "square_kilometer", "square_meter")
        assert result == 1000000.0

    @pytest.mark.asyncio
    async def test_area_same_unit(self):
        """Test conversion to same unit."""
        result = await convert_area(100, "square_meter", "square_meter")
        assert result == 100

    @pytest.mark.asyncio
    async def test_area_negative_raises_error(self):
        """Test that negative area raises error."""
        with pytest.raises(ValueError, match="Area cannot be negative"):
            await convert_area(-5, "square_meter", "square_feet")

    @pytest.mark.asyncio
    async def test_area_invalid_unit(self):
        """Test with invalid unit."""
        with pytest.raises(ValueError, match="Unsupported"):
            await convert_area(10, "square_meter", "invalid")


# Volume Conversion Tests
class TestConvertVolume:
    """Test cases for volume conversion."""

    @pytest.mark.asyncio
    async def test_liter_to_gallon(self):
        """Test liter to gallon conversion."""
        result = await convert_volume(1, "liter", "gallon")
        assert pytest.approx(result, rel=1e-6) == 0.26417205235814845

    @pytest.mark.asyncio
    async def test_gallon_to_liter(self):
        """Test gallon to liter conversion."""
        result = await convert_volume(1, "gallon", "liter")
        assert pytest.approx(result, rel=1e-6) == 3.785411784

    @pytest.mark.asyncio
    async def test_cubic_meter_to_liter(self):
        """Test cubic meter to liter conversion."""
        result = await convert_volume(1, "cubic_meter", "liter")
        assert result == 1000.0

    @pytest.mark.asyncio
    async def test_milliliter_to_liter(self):
        """Test milliliter to liter conversion."""
        result = await convert_volume(1000, "milliliter", "liter")
        assert result == 1.0

    @pytest.mark.asyncio
    async def test_quart_to_gallon(self):
        """Test quart to gallon conversion."""
        result = await convert_volume(4, "quart", "gallon")
        assert pytest.approx(result, rel=1e-6) == 1.0

    @pytest.mark.asyncio
    async def test_volume_same_unit(self):
        """Test conversion to same unit."""
        result = await convert_volume(50, "liter", "liter")
        assert result == 50

    @pytest.mark.asyncio
    async def test_volume_negative_raises_error(self):
        """Test that negative volume raises error."""
        with pytest.raises(ValueError, match="Volume cannot be negative"):
            await convert_volume(-5, "liter", "gallon")

    @pytest.mark.asyncio
    async def test_volume_invalid_unit(self):
        """Test with invalid unit."""
        with pytest.raises(ValueError, match="Unsupported"):
            await convert_volume(10, "liter", "invalid")


# Time Conversion Tests
class TestConvertTime:
    """Test cases for time conversion."""

    @pytest.mark.asyncio
    async def test_hour_to_minute(self):
        """Test hour to minute conversion."""
        result = await convert_time(1, "hour", "minute")
        assert result == 60.0

    @pytest.mark.asyncio
    async def test_day_to_hour(self):
        """Test day to hour conversion."""
        result = await convert_time(1, "day", "hour")
        assert result == 24.0

    @pytest.mark.asyncio
    async def test_week_to_day(self):
        """Test week to day conversion."""
        result = await convert_time(1, "week", "day")
        assert result == 7.0

    @pytest.mark.asyncio
    async def test_minute_to_second(self):
        """Test minute to second conversion."""
        result = await convert_time(1, "minute", "second")
        assert result == 60.0

    @pytest.mark.asyncio
    async def test_year_to_day(self):
        """Test year to day conversion."""
        result = await convert_time(1, "year", "day")
        # Source uses Gregorian calendar average (365.2425), not Julian (365.25)
        assert pytest.approx(result, rel=1e-6) == 365.2425

    @pytest.mark.asyncio
    async def test_time_same_unit(self):
        """Test conversion to same unit."""
        result = await convert_time(100, "second", "second")
        assert result == 100

    @pytest.mark.asyncio
    async def test_time_negative_raises_error(self):
        """Test that negative time raises error."""
        with pytest.raises(ValueError, match="Time cannot be negative"):
            await convert_time(-5, "hour", "minute")

    @pytest.mark.asyncio
    async def test_time_invalid_unit(self):
        """Test with invalid unit."""
        with pytest.raises(ValueError, match="Unsupported"):
            await convert_time(10, "hour", "invalid")


# Speed Conversion Tests
class TestConvertSpeed:
    """Test cases for speed conversion."""

    @pytest.mark.asyncio
    async def test_mph_to_kph(self):
        """Test mph to kph conversion."""
        result = await convert_speed(60, "mph", "kph")
        assert pytest.approx(result, rel=1e-6) == 96.56064

    @pytest.mark.asyncio
    async def test_kph_to_mph(self):
        """Test kph to mph conversion."""
        result = await convert_speed(100, "kph", "mph")
        assert pytest.approx(result, rel=1e-6) == 62.13711922373339

    @pytest.mark.asyncio
    async def test_mps_to_kph(self):
        """Test m/s to kph conversion."""
        result = await convert_speed(10, "m/s", "kph")
        assert result == 36.0

    @pytest.mark.asyncio
    async def test_knot_to_mph(self):
        """Test knot to mph conversion."""
        result = await convert_speed(1, "knot", "mph")
        assert pytest.approx(result, rel=1e-6) == 1.15077944802354

    @pytest.mark.asyncio
    async def test_speed_same_unit(self):
        """Test conversion to same unit."""
        result = await convert_speed(100, "kph", "kph")
        assert result == 100

    @pytest.mark.asyncio
    async def test_speed_negative_raises_error(self):
        """Test that negative speed raises error."""
        with pytest.raises(ValueError, match="Speed cannot be negative"):
            await convert_speed(-5, "mph", "kph")

    @pytest.mark.asyncio
    async def test_speed_invalid_unit(self):
        """Test with invalid unit."""
        with pytest.raises(ValueError, match="Unsupported"):
            await convert_speed(10, "mph", "invalid")


# Roundtrip Conversion Tests
class TestRoundtripConversions:
    """Test roundtrip conversions (A → B → A should equal A)."""

    @pytest.mark.asyncio
    async def test_temperature_roundtrip(self):
        """Test temperature roundtrip conversion."""
        original = 25
        temp_f = await convert_temperature(original, "celsius", "fahrenheit")
        back_to_c = await convert_temperature(temp_f["result"], "fahrenheit", "celsius")
        assert pytest.approx(back_to_c["result"], rel=1e-10) == original

    @pytest.mark.asyncio
    async def test_length_roundtrip(self):
        """Test length roundtrip conversion."""
        original = 100
        feet = await convert_length(original, "meter", "feet")
        back_to_m = await convert_length(feet, "feet", "meter")
        assert pytest.approx(back_to_m, rel=1e-10) == original

    @pytest.mark.asyncio
    async def test_weight_roundtrip(self):
        """Test weight roundtrip conversion."""
        original = 50
        pounds = await convert_weight(original, "kilogram", "pound")
        back_to_kg = await convert_weight(pounds, "pound", "kilogram")
        assert pytest.approx(back_to_kg, rel=1e-10) == original

    @pytest.mark.asyncio
    async def test_area_roundtrip(self):
        """Test area roundtrip conversion."""
        original = 1000
        sq_ft = await convert_area(original, "square_meter", "square_feet")
        back_to_sqm = await convert_area(sq_ft, "square_feet", "square_meter")
        assert pytest.approx(back_to_sqm, rel=1e-10) == original

    @pytest.mark.asyncio
    async def test_volume_roundtrip(self):
        """Test volume roundtrip conversion."""
        original = 10
        gallons = await convert_volume(original, "liter", "gallon")
        back_to_l = await convert_volume(gallons, "gallon", "liter")
        assert pytest.approx(back_to_l, rel=1e-10) == original

    @pytest.mark.asyncio
    async def test_time_roundtrip(self):
        """Test time roundtrip conversion."""
        original = 120
        minutes = await convert_time(original, "second", "minute")
        back_to_s = await convert_time(minutes, "minute", "second")
        assert pytest.approx(back_to_s, rel=1e-10) == original

    @pytest.mark.asyncio
    async def test_speed_roundtrip(self):
        """Test speed roundtrip conversion."""
        original = 100
        mph = await convert_speed(original, "kph", "mph")
        back_to_kph = await convert_speed(mph, "mph", "kph")
        assert pytest.approx(back_to_kph, rel=1e-10) == original


# Async Behavior Tests
class TestAsyncBehavior:
    """Test async behavior of conversion functions."""

    @pytest.mark.asyncio
    async def test_all_conversions_are_async(self):
        """Test that all conversion functions are properly async."""
        operations = [
            convert_temperature(100, "celsius", "fahrenheit"),
            convert_length(1, "meter", "feet"),
            convert_weight(1, "kilogram", "pound"),
            convert_area(1, "square_meter", "square_feet"),
            convert_volume(1, "liter", "gallon"),
            convert_time(1, "hour", "minute"),
            convert_speed(60, "mph", "kph"),
        ]

        # Ensure all are coroutines
        for op in operations:
            assert asyncio.iscoroutine(op)

        # Run all operations concurrently
        results = await asyncio.gather(*operations)
        assert len(results) == len(operations)

    @pytest.mark.asyncio
    async def test_concurrent_conversions(self):
        """Test concurrent conversion execution."""
        import time

        start = time.time()

        # Run multiple conversions concurrently
        tasks = [convert_length(i, "meter", "feet") for i in range(100)]

        await asyncio.gather(*tasks)
        duration = time.time() - start

        # Should complete quickly
        assert duration < 1.0


# Parametrized Tests
class TestParametrized:
    """Parametrized tests for comprehensive coverage."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "celsius,fahrenheit",
        [
            (0, 32),
            (100, 212),
            (-40, -40),
            (37, 98.6),
        ],
    )
    async def test_temperature_conversions_parametrized(self, celsius, fahrenheit):
        """Parametrized test for temperature conversions."""
        c_to_f = await convert_temperature(celsius, "celsius", "fahrenheit")
        assert pytest.approx(c_to_f["result"], rel=1e-6) == fahrenheit

        f_to_c = await convert_temperature(fahrenheit, "fahrenheit", "celsius")
        assert pytest.approx(f_to_c["result"], rel=1e-6) == celsius

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "value,from_unit,to_unit,expected",
        [
            (1, "meter", "centimeter", 100),
            (1, "kilometer", "meter", 1000),
            (1, "foot", "inch", 12),
            (1, "mile", "feet", 5280),
        ],
    )
    async def test_length_conversions_parametrized(self, value, from_unit, to_unit, expected):
        """Parametrized test for length conversions."""
        result = await convert_length(value, from_unit, to_unit)
        assert pytest.approx(result, rel=1e-6) == expected


# Edge Cases Tests
class TestEdgeCases:
    """Test edge cases for conversions."""

    @pytest.mark.asyncio
    async def test_zero_conversions(self):
        """Test converting zero values."""
        assert await convert_length(0, "meter", "feet") == 0
        assert await convert_weight(0, "kilogram", "pound") == 0
        assert await convert_area(0, "square_meter", "square_feet") == 0
        assert await convert_volume(0, "liter", "gallon") == 0
        assert await convert_time(0, "hour", "minute") == 0
        assert await convert_speed(0, "mph", "kph") == 0

    @pytest.mark.asyncio
    async def test_large_number_conversions(self):
        """Test converting large numbers."""
        large = 1e10
        result = await convert_length(large, "meter", "kilometer")
        assert result == large / 1000

    @pytest.mark.asyncio
    async def test_small_number_conversions(self):
        """Test converting small numbers."""
        small = 1e-10
        result = await convert_length(small, "meter", "millimeter")
        assert pytest.approx(result, rel=1e-15) == small * 1000


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v", "--tb=short"])
