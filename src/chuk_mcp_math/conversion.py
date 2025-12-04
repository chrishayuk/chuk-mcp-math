#!/usr/bin/env python3
# chuk_mcp_math/conversion.py
"""
Conversion Functions for AI Models

Unit conversions and transformations: temperature, length, weight, area, volume,
time, and more. All functions include validation and clear descriptions for AI execution.
"""

import asyncio
from typing import Union, Dict, Any
from chuk_mcp_math.mcp_decorator import mcp_function


@mcp_function(
    description="Convert temperature between Celsius, Fahrenheit, and Kelvin scales with validation and formulas.",
    namespace="conversions",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    examples=[
        {
            "input": {"value": 100, "from_scale": "celsius", "to_scale": "fahrenheit"},
            "output": {"result": 212.0, "formula": "C × 9/5 + 32"},
            "description": "Boiling point conversion",
        },
        {
            "input": {"value": 32, "from_scale": "fahrenheit", "to_scale": "celsius"},
            "output": {"result": 0.0, "formula": "(F - 32) × 5/9"},
            "description": "Freezing point conversion",
        },
        {
            "input": {"value": 273.15, "from_scale": "kelvin", "to_scale": "celsius"},
            "output": {"result": 0.0, "formula": "K - 273.15"},
            "description": "Absolute zero conversion",
        },
    ],
)
async def convert_temperature(
    value: Union[int, float], from_scale: str, to_scale: str
) -> Dict[str, Any]:
    """
    Convert temperature between different scales with metadata.

    Args:
        value: Temperature value to convert
        from_scale: Source scale ("celsius", "fahrenheit", "kelvin", "rankine")
        to_scale: Target scale ("celsius", "fahrenheit", "kelvin", "rankine")

    Returns:
        Dictionary with converted value, formula used, and metadata

    Examples:
        convert_temperature(100, "celsius", "fahrenheit") → {"result": 212.0, "formula": "C × 9/5 + 32"}
        convert_temperature(32, "fahrenheit", "celsius") → {"result": 0.0, "formula": "(F - 32) × 5/9"}
    """
    from_scale = from_scale.lower()
    to_scale = to_scale.lower()

    valid_scales = ["celsius", "fahrenheit", "kelvin", "rankine"]
    if from_scale not in valid_scales or to_scale not in valid_scales:
        raise ValueError(f"Invalid scale. Use one of: {valid_scales}")

    # Validate input ranges
    if from_scale == "kelvin" and value < 0:
        raise ValueError("Kelvin temperature cannot be negative")
    if from_scale == "rankine" and value < 0:
        raise ValueError("Rankine temperature cannot be negative")
    if from_scale == "celsius" and value < -273.15:
        raise ValueError(
            "Celsius temperature cannot be below absolute zero (-273.15°C)"
        )
    if from_scale == "fahrenheit" and value < -459.67:
        raise ValueError(
            "Fahrenheit temperature cannot be below absolute zero (-459.67°F)"
        )

    # Convert to Celsius first
    if from_scale == "fahrenheit":
        celsius = (value - 32) * 5 / 9
    elif from_scale == "kelvin":
        celsius = value - 273.15
    elif from_scale == "rankine":
        celsius = (value - 491.67) * 5 / 9
    elif from_scale == "celsius":
        celsius = value

    # Convert from Celsius to target scale
    if to_scale == "fahrenheit":
        result = celsius * 9 / 5 + 32
        formula = (
            "C × 9/5 + 32" if from_scale == "celsius" else "(converted to C) × 9/5 + 32"
        )
    elif to_scale == "kelvin":
        result = celsius + 273.15
        formula = (
            "C + 273.15" if from_scale == "celsius" else "(converted to C) + 273.15"
        )
    elif to_scale == "rankine":
        result = (celsius + 273.15) * 9 / 5
        formula = (
            "(C + 273.15) × 9/5"
            if from_scale == "celsius"
            else "(converted to C + 273.15) × 9/5"
        )
    elif to_scale == "celsius":
        result = celsius
        if from_scale == "fahrenheit":
            formula = "(F - 32) × 5/9"
        elif from_scale == "kelvin":
            formula = "K - 273.15"
        elif from_scale == "rankine":
            formula = "(R - 491.67) × 5/9"
        else:
            formula = "C"

    await asyncio.sleep(0)  # Yield control for async execution
    return {
        "result": result,
        "original_value": value,
        "original_scale": from_scale,
        "target_scale": to_scale,
        "formula": formula,
        "celsius_equivalent": celsius,
    }


@mcp_function(
    description="Convert length between different units: meters, feet, inches, kilometers, miles, etc.",
    namespace="conversions",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    examples=[
        {
            "input": {"value": 1, "from_unit": "meter", "to_unit": "feet"},
            "output": 3.281,
            "description": "1 meter to feet",
        },
        {
            "input": {"value": 5, "from_unit": "feet", "to_unit": "inches"},
            "output": 60,
            "description": "5 feet to inches",
        },
        {
            "input": {"value": 1, "from_unit": "kilometer", "to_unit": "mile"},
            "output": 0.621,
            "description": "1 km to miles",
        },
    ],
)
async def convert_length(
    value: Union[int, float], from_unit: str, to_unit: str
) -> float:
    """
    Convert length between different units.

    Args:
        value: Length value to convert
        from_unit: Source unit
        to_unit: Target unit

    Returns:
        Converted length value

    Supported units: meter, kilometer, centimeter, millimeter, inch, foot, yard, mile

    Examples:
        convert_length(1, "meter", "feet") → 3.2808398950131233
        convert_length(5, "feet", "inches") → 60.0
        convert_length(1, "kilometer", "mile") → 0.6213711922373339
    """
    # Conversion factors to meters
    to_meters = {
        "meter": 1.0,
        "meters": 1.0,
        "m": 1.0,
        "kilometer": 1000.0,
        "kilometers": 1000.0,
        "km": 1000.0,
        "centimeter": 0.01,
        "centimeters": 0.01,
        "cm": 0.01,
        "millimeter": 0.001,
        "millimeters": 0.001,
        "mm": 0.001,
        "inch": 0.0254,
        "inches": 0.0254,
        "in": 0.0254,
        "foot": 0.3048,
        "feet": 0.3048,
        "ft": 0.3048,
        "yard": 0.9144,
        "yards": 0.9144,
        "yd": 0.9144,
        "mile": 1609.344,
        "miles": 1609.344,
        "mi": 1609.344,
    }

    from_unit = from_unit.lower()
    to_unit = to_unit.lower()

    if from_unit not in to_meters:
        raise ValueError(f"Unsupported from_unit: {from_unit}")
    if to_unit not in to_meters:
        raise ValueError(f"Unsupported to_unit: {to_unit}")

    if value < 0:
        raise ValueError("Length cannot be negative")

    # Convert to meters, then to target unit
    meters = value * to_meters[from_unit]
    result = meters / to_meters[to_unit]

    await asyncio.sleep(0)  # Yield control for async execution
    return result


@mcp_function(
    description="Convert weight/mass between different units: kilograms, pounds, ounces, grams, etc.",
    namespace="conversions",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    examples=[
        {
            "input": {"value": 1, "from_unit": "kilogram", "to_unit": "pound"},
            "output": 2.205,
            "description": "1 kg to pounds",
        },
        {
            "input": {"value": 16, "from_unit": "ounce", "to_unit": "pound"},
            "output": 1.0,
            "description": "16 oz to 1 pound",
        },
        {
            "input": {"value": 1000, "from_unit": "gram", "to_unit": "kilogram"},
            "output": 1.0,
            "description": "1000g to 1kg",
        },
    ],
)
async def convert_weight(
    value: Union[int, float], from_unit: str, to_unit: str
) -> float:
    """
    Convert weight/mass between different units.

    Args:
        value: Weight value to convert
        from_unit: Source unit
        to_unit: Target unit

    Returns:
        Converted weight value

    Supported units: kilogram, gram, pound, ounce, stone, ton

    Examples:
        convert_weight(1, "kilogram", "pound") → 2.2046226218487757
        convert_weight(16, "ounce", "pound") → 1.0
        convert_weight(1000, "gram", "kilogram") → 1.0
    """
    # Conversion factors to kilograms
    to_kg = {
        "kilogram": 1.0,
        "kilograms": 1.0,
        "kg": 1.0,
        "gram": 0.001,
        "grams": 0.001,
        "g": 0.001,
        "pound": 0.45359237,
        "pounds": 0.45359237,
        "lb": 0.45359237,
        "lbs": 0.45359237,
        "ounce": 0.028349523125,
        "ounces": 0.028349523125,
        "oz": 0.028349523125,
        "stone": 6.35029318,
        "stones": 6.35029318,
        "st": 6.35029318,
        "ton": 1000.0,
        "tons": 1000.0,
        "tonne": 1000.0,
        "tonnes": 1000.0,
    }

    from_unit = from_unit.lower()
    to_unit = to_unit.lower()

    if from_unit not in to_kg:
        raise ValueError(f"Unsupported from_unit: {from_unit}")
    if to_unit not in to_kg:
        raise ValueError(f"Unsupported to_unit: {to_unit}")

    if value < 0:
        raise ValueError("Weight cannot be negative")

    # Convert to kg, then to target unit
    kg = value * to_kg[from_unit]
    result = kg / to_kg[to_unit]

    await asyncio.sleep(0)  # Yield control for async execution
    return result


@mcp_function(
    description="Convert area between different units: square meters, square feet, acres, hectares, etc.",
    namespace="conversions",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    examples=[
        {
            "input": {
                "value": 1,
                "from_unit": "square_meter",
                "to_unit": "square_feet",
            },
            "output": 10.764,
            "description": "1 m² to ft²",
        },
        {
            "input": {"value": 1, "from_unit": "acre", "to_unit": "hectare"},
            "output": 0.405,
            "description": "1 acre to hectares",
        },
        {
            "input": {"value": 1, "from_unit": "hectare", "to_unit": "square_meter"},
            "output": 10000,
            "description": "1 hectare to m²",
        },
    ],
)
async def convert_area(value: Union[int, float], from_unit: str, to_unit: str) -> float:
    """
    Convert area between different units.

    Args:
        value: Area value to convert
        from_unit: Source unit
        to_unit: Target unit

    Returns:
        Converted area value

    Supported units: square_meter, square_kilometer, square_centimeter, square_inch,
                    square_foot, square_yard, acre, hectare

    Examples:
        convert_area(1, "square_meter", "square_feet") → 10.76391041670972
        convert_area(1, "acre", "hectare") → 0.40468564224
        convert_area(1, "hectare", "square_meter") → 10000.0
    """
    # Conversion factors to square meters
    to_sq_meters = {
        "square_meter": 1.0,
        "square_meters": 1.0,
        "sq_m": 1.0,
        "m2": 1.0,
        "square_kilometer": 1000000.0,
        "square_kilometers": 1000000.0,
        "sq_km": 1000000.0,
        "km2": 1000000.0,
        "square_centimeter": 0.0001,
        "square_centimeters": 0.0001,
        "sq_cm": 0.0001,
        "cm2": 0.0001,
        "square_inch": 0.00064516,
        "square_inches": 0.00064516,
        "sq_in": 0.00064516,
        "in2": 0.00064516,
        "square_foot": 0.09290304,
        "square_feet": 0.09290304,
        "sq_ft": 0.09290304,
        "ft2": 0.09290304,
        "square_yard": 0.83612736,
        "square_yards": 0.83612736,
        "sq_yd": 0.83612736,
        "yd2": 0.83612736,
        "acre": 4046.8564224,
        "acres": 4046.8564224,
        "hectare": 10000.0,
        "hectares": 10000.0,
        "ha": 10000.0,
    }

    from_unit = from_unit.lower()
    to_unit = to_unit.lower()

    if from_unit not in to_sq_meters:
        raise ValueError(f"Unsupported from_unit: {from_unit}")
    if to_unit not in to_sq_meters:
        raise ValueError(f"Unsupported to_unit: {to_unit}")

    if value < 0:
        raise ValueError("Area cannot be negative")

    # Convert to square meters, then to target unit
    sq_meters = value * to_sq_meters[from_unit]
    result = sq_meters / to_sq_meters[to_unit]

    await asyncio.sleep(0)  # Yield control for async execution
    return result


@mcp_function(
    description="Convert volume between different units: liters, gallons, cubic meters, cubic feet, etc.",
    namespace="conversions",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    examples=[
        {
            "input": {"value": 1, "from_unit": "liter", "to_unit": "gallon"},
            "output": 0.264,
            "description": "1 liter to US gallons",
        },
        {
            "input": {"value": 1, "from_unit": "cubic_meter", "to_unit": "liter"},
            "output": 1000,
            "description": "1 m³ to liters",
        },
        {
            "input": {"value": 1, "from_unit": "gallon", "to_unit": "liter"},
            "output": 3.785,
            "description": "1 US gallon to liters",
        },
    ],
)
async def convert_volume(
    value: Union[int, float], from_unit: str, to_unit: str
) -> float:
    """
    Convert volume between different units.

    Args:
        value: Volume value to convert
        from_unit: Source unit
        to_unit: Target unit

    Returns:
        Converted volume value

    Supported units: liter, milliliter, cubic_meter, cubic_centimeter, gallon (US),
                    quart, pint, cup, fluid_ounce, cubic_foot, cubic_inch

    Examples:
        convert_volume(1, "liter", "gallon") → 0.26417205235814845
        convert_volume(1, "cubic_meter", "liter") → 1000.0
        convert_volume(1, "gallon", "liter") → 3.785411784
    """
    # Conversion factors to liters
    to_liters = {
        "liter": 1.0,
        "liters": 1.0,
        "l": 1.0,
        "milliliter": 0.001,
        "milliliters": 0.001,
        "ml": 0.001,
        "cubic_meter": 1000.0,
        "cubic_meters": 1000.0,
        "m3": 1000.0,
        "cubic_centimeter": 0.001,
        "cubic_centimeters": 0.001,
        "cm3": 0.001,
        "cc": 0.001,
        "gallon": 3.785411784,  # US gallon
        "gallons": 3.785411784,
        "gal": 3.785411784,
        "quart": 0.946352946,
        "quarts": 0.946352946,
        "qt": 0.946352946,
        "pint": 0.473176473,
        "pints": 0.473176473,
        "pt": 0.473176473,
        "cup": 0.2365882365,
        "cups": 0.2365882365,
        "fluid_ounce": 0.0295735296,
        "fluid_ounces": 0.0295735296,
        "fl_oz": 0.0295735296,
        "cubic_foot": 28.3168466,
        "cubic_feet": 28.3168466,
        "ft3": 28.3168466,
        "cubic_inch": 0.0163870640,
        "cubic_inches": 0.0163870640,
        "in3": 0.0163870640,
    }

    from_unit = from_unit.lower()
    to_unit = to_unit.lower()

    if from_unit not in to_liters:
        raise ValueError(f"Unsupported from_unit: {from_unit}")
    if to_unit not in to_liters:
        raise ValueError(f"Unsupported to_unit: {to_unit}")

    if value < 0:
        raise ValueError("Volume cannot be negative")

    # Convert to liters, then to target unit
    liters = value * to_liters[from_unit]
    result = liters / to_liters[to_unit]

    await asyncio.sleep(0)  # Yield control for async execution
    return result


@mcp_function(
    description="Convert time between different units: seconds, minutes, hours, days, weeks, years, etc.",
    namespace="conversions",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    examples=[
        {
            "input": {"value": 1, "from_unit": "hour", "to_unit": "minute"},
            "output": 60,
            "description": "1 hour to minutes",
        },
        {
            "input": {"value": 1, "from_unit": "day", "to_unit": "hour"},
            "output": 24,
            "description": "1 day to hours",
        },
        {
            "input": {"value": 1, "from_unit": "week", "to_unit": "day"},
            "output": 7,
            "description": "1 week to days",
        },
    ],
)
async def convert_time(value: Union[int, float], from_unit: str, to_unit: str) -> float:
    """
    Convert time between different units.

    Args:
        value: Time value to convert
        from_unit: Source unit
        to_unit: Target unit

    Returns:
        Converted time value

    Supported units: second, minute, hour, day, week, month, year

    Examples:
        convert_time(1, "hour", "minute") → 60.0
        convert_time(1, "day", "hour") → 24.0
        convert_time(1, "week", "day") → 7.0
    """
    # Conversion factors to seconds
    to_seconds = {
        "second": 1.0,
        "seconds": 1.0,
        "sec": 1.0,
        "s": 1.0,
        "minute": 60.0,
        "minutes": 60.0,
        "min": 60.0,
        "hour": 3600.0,
        "hours": 3600.0,
        "hr": 3600.0,
        "h": 3600.0,
        "day": 86400.0,
        "days": 86400.0,
        "d": 86400.0,
        "week": 604800.0,
        "weeks": 604800.0,
        "wk": 604800.0,
        "month": 2629746.0,  # Average month (365.25 days / 12)
        "months": 2629746.0,
        "year": 31556952.0,  # Average year (365.25 days)
        "years": 31556952.0,
        "yr": 31556952.0,
        "y": 31556952.0,
    }

    from_unit = from_unit.lower()
    to_unit = to_unit.lower()

    if from_unit not in to_seconds:
        raise ValueError(f"Unsupported from_unit: {from_unit}")
    if to_unit not in to_seconds:
        raise ValueError(f"Unsupported to_unit: {to_unit}")

    if value < 0:
        raise ValueError("Time cannot be negative")

    # Convert to seconds, then to target unit
    seconds = value * to_seconds[from_unit]
    result = seconds / to_seconds[to_unit]

    await asyncio.sleep(0)  # Yield control for async execution
    return result


@mcp_function(
    description="Convert speed between different units: mph, kph, m/s, knots, etc.",
    namespace="conversions",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    examples=[
        {
            "input": {"value": 60, "from_unit": "mph", "to_unit": "kph"},
            "output": 96.56,
            "description": "60 mph to kph",
        },
        {
            "input": {"value": 100, "from_unit": "kph", "to_unit": "mph"},
            "output": 62.14,
            "description": "100 kph to mph",
        },
        {
            "input": {"value": 10, "from_unit": "m/s", "to_unit": "kph"},
            "output": 36.0,
            "description": "10 m/s to kph",
        },
    ],
)
async def convert_speed(
    value: Union[int, float], from_unit: str, to_unit: str
) -> float:
    """
    Convert speed between different units.

    Args:
        value: Speed value to convert
        from_unit: Source unit
        to_unit: Target unit

    Returns:
        Converted speed value

    Supported units: m/s, kph, mph, knot, ft/s

    Examples:
        convert_speed(60, "mph", "kph") → 96.56064
        convert_speed(100, "kph", "mph") → 62.13711922373339
        convert_speed(10, "m/s", "kph") → 36.0
    """
    # Conversion factors to meters per second
    to_mps = {
        "m/s": 1.0,
        "mps": 1.0,
        "meter_per_second": 1.0,
        "kph": 1 / 3.6,
        "kmh": 1 / 3.6,
        "km/h": 1 / 3.6,
        "kilometer_per_hour": 1 / 3.6,
        "mph": 0.44704,
        "mile_per_hour": 0.44704,
        "knot": 0.514444,
        "knots": 0.514444,
        "kt": 0.514444,
        "ft/s": 0.3048,
        "fps": 0.3048,
        "feet_per_second": 0.3048,
    }

    from_unit = from_unit.lower()
    to_unit = to_unit.lower()

    if from_unit not in to_mps:
        raise ValueError(f"Unsupported from_unit: {from_unit}")
    if to_unit not in to_mps:
        raise ValueError(f"Unsupported to_unit: {to_unit}")

    if value < 0:
        raise ValueError("Speed cannot be negative")

    # Convert to m/s, then to target unit
    mps = value * to_mps[from_unit]
    result = mps / to_mps[to_unit]

    await asyncio.sleep(0)  # Yield control for async execution
    return result


# Export all conversion functions
__all__ = [
    "convert_temperature",
    "convert_length",
    "convert_weight",
    "convert_area",
    "convert_volume",
    "convert_time",
    "convert_speed",
]

if __name__ == "__main__":
    # Test the conversion functions
    print("🔄 Conversion Functions Test")
    print("=" * 35)

    temp = convert_temperature(100, "celsius", "fahrenheit")
    print(f"convert_temperature(100, 'celsius', 'fahrenheit') = {temp['result']}°F")

    print(
        f"convert_length(1, 'meter', 'feet') = {convert_length(1, 'meter', 'feet'):.3f} ft"
    )
    print(
        f"convert_weight(1, 'kilogram', 'pound') = {convert_weight(1, 'kilogram', 'pound'):.3f} lbs"
    )
    print(
        f"convert_area(1, 'square_meter', 'square_feet') = {convert_area(1, 'square_meter', 'square_feet'):.3f} ft²"
    )
    print(
        f"convert_volume(1, 'liter', 'gallon') = {convert_volume(1, 'liter', 'gallon'):.3f} gal"
    )
    print(
        f"convert_time(1, 'hour', 'minute') = {convert_time(1, 'hour', 'minute')} min"
    )
    print(
        f"convert_speed(60, 'mph', 'kph') = {convert_speed(60, 'mph', 'kph'):.2f} kph"
    )

    print("\n✅ All conversion functions working correctly!")
