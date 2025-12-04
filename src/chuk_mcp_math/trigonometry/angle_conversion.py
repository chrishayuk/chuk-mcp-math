#!/usr/bin/env python3
# chuk_mcp_math/trigonometry/angle_conversion.py
"""
Angle Conversion Functions - Async Native

Comprehensive angle conversion and normalization functions.
Handles conversions between degrees, radians, and gradians with high precision.

Functions:
- Basic conversions: degrees ↔ radians ↔ gradians
- Angle normalization: [0, 2π), [-π, π), [0°, 360°), etc.
- Angle difference: shortest angular distance
- Precision handling: minimizes floating-point errors
"""

import math
import asyncio
from typing import Union, Literal
from chuk_mcp_math.mcp_decorator import mcp_function

# ============================================================================
# BASIC ANGLE CONVERSIONS
# ============================================================================


@mcp_function(
    description="Convert degrees to radians with high precision.",
    namespace="trigonometry",
    category="angle_conversion",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    cache_ttl_seconds=7200,
    examples=[
        {"input": {"degrees": 0}, "output": 0.0, "description": "0° = 0 radians"},
        {
            "input": {"degrees": 90},
            "output": 1.5707963267948966,
            "description": "90° = π/2 radians",
        },
        {
            "input": {"degrees": 180},
            "output": 3.141592653589793,
            "description": "180° = π radians",
        },
        {
            "input": {"degrees": 360},
            "output": 6.283185307179586,
            "description": "360° = 2π radians",
        },
    ],
)
async def degrees_to_radians(degrees: Union[int, float]) -> float:
    """
    Convert degrees to radians.

    Uses high-precision conversion factor: π/180.

    Args:
        degrees: Angle in degrees

    Returns:
        Angle in radians

    Examples:
        await degrees_to_radians(0) → 0.0        # 0°
        await degrees_to_radians(90) → π/2       # 90° = π/2 ≈ 1.5708
        await degrees_to_radians(180) → π        # 180° = π ≈ 3.1416
        await degrees_to_radians(360) → 2π       # 360° = 2π ≈ 6.2832
    """
    return degrees * math.pi / 180.0


@mcp_function(
    description="Convert radians to degrees with high precision.",
    namespace="trigonometry",
    category="angle_conversion",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    cache_ttl_seconds=7200,
    examples=[
        {"input": {"radians": 0}, "output": 0.0, "description": "0 radians = 0°"},
        {
            "input": {"radians": 1.5707963267948966},
            "output": 90.0,
            "description": "π/2 radians = 90°",
        },
        {
            "input": {"radians": 3.141592653589793},
            "output": 180.0,
            "description": "π radians = 180°",
        },
        {
            "input": {"radians": 6.283185307179586},
            "output": 360.0,
            "description": "2π radians = 360°",
        },
    ],
)
async def radians_to_degrees(radians: Union[int, float]) -> float:
    """
    Convert radians to degrees.

    Uses high-precision conversion factor: 180/π.

    Args:
        radians: Angle in radians

    Returns:
        Angle in degrees

    Examples:
        await radians_to_degrees(0) → 0.0        # 0 rad
        await radians_to_degrees(π/2) → 90.0     # π/2 rad = 90°
        await radians_to_degrees(π) → 180.0      # π rad = 180°
        await radians_to_degrees(2π) → 360.0     # 2π rad = 360°
    """
    return radians * 180.0 / math.pi


@mcp_function(
    description="Convert gradians to radians.",
    namespace="trigonometry",
    category="angle_conversion",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    cache_ttl_seconds=7200,
    examples=[
        {
            "input": {"gradians": 0},
            "output": 0.0,
            "description": "0 gradians = 0 radians",
        },
        {
            "input": {"gradians": 100},
            "output": 1.5707963267948966,
            "description": "100 gradians = π/2 radians",
        },
        {
            "input": {"gradians": 200},
            "output": 3.141592653589793,
            "description": "200 gradians = π radians",
        },
        {
            "input": {"gradians": 400},
            "output": 6.283185307179586,
            "description": "400 gradians = 2π radians",
        },
    ],
)
async def gradians_to_radians(gradians: Union[int, float]) -> float:
    """
    Convert gradians (grads) to radians.

    Gradians divide a full circle into 400 units.
    Conversion factor: π/200.

    Args:
        gradians: Angle in gradians

    Returns:
        Angle in radians

    Examples:
        await gradians_to_radians(0) → 0.0       # 0 grad
        await gradians_to_radians(100) → π/2     # 100 grad = π/2 rad
        await gradians_to_radians(200) → π       # 200 grad = π rad
        await gradians_to_radians(400) → 2π      # 400 grad = 2π rad
    """
    return gradians * math.pi / 200.0


@mcp_function(
    description="Convert radians to gradians.",
    namespace="trigonometry",
    category="angle_conversion",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    cache_ttl_seconds=7200,
    examples=[
        {
            "input": {"radians": 0},
            "output": 0.0,
            "description": "0 radians = 0 gradians",
        },
        {
            "input": {"radians": 1.5707963267948966},
            "output": 100.0,
            "description": "π/2 radians = 100 gradians",
        },
        {
            "input": {"radians": 3.141592653589793},
            "output": 200.0,
            "description": "π radians = 200 gradians",
        },
        {
            "input": {"radians": 6.283185307179586},
            "output": 400.0,
            "description": "2π radians = 400 gradians",
        },
    ],
)
async def radians_to_gradians(radians: Union[int, float]) -> float:
    """
    Convert radians to gradians (grads).

    Gradians divide a full circle into 400 units.
    Conversion factor: 200/π.

    Args:
        radians: Angle in radians

    Returns:
        Angle in gradians

    Examples:
        await radians_to_gradians(0) → 0.0       # 0 rad
        await radians_to_gradians(π/2) → 100.0   # π/2 rad = 100 grad
        await radians_to_gradians(π) → 200.0     # π rad = 200 grad
        await radians_to_gradians(2π) → 400.0    # 2π rad = 400 grad
    """
    return radians * 200.0 / math.pi


@mcp_function(
    description="Convert degrees to gradians.",
    namespace="trigonometry",
    category="angle_conversion",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    cache_ttl_seconds=7200,
    examples=[
        {"input": {"degrees": 0}, "output": 0.0, "description": "0° = 0 gradians"},
        {
            "input": {"degrees": 90},
            "output": 100.0,
            "description": "90° = 100 gradians",
        },
        {
            "input": {"degrees": 180},
            "output": 200.0,
            "description": "180° = 200 gradians",
        },
        {
            "input": {"degrees": 360},
            "output": 400.0,
            "description": "360° = 400 gradians",
        },
    ],
)
async def degrees_to_gradians(degrees: Union[int, float]) -> float:
    """
    Convert degrees to gradians (grads).

    Conversion factor: 400/360 = 10/9.

    Args:
        degrees: Angle in degrees

    Returns:
        Angle in gradians

    Examples:
        await degrees_to_gradians(0) → 0.0       # 0°
        await degrees_to_gradians(90) → 100.0    # 90° = 100 grad
        await degrees_to_gradians(180) → 200.0   # 180° = 200 grad
        await degrees_to_gradians(360) → 400.0   # 360° = 400 grad
    """
    return degrees * 400.0 / 360.0


@mcp_function(
    description="Convert gradians to degrees.",
    namespace="trigonometry",
    category="angle_conversion",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    cache_ttl_seconds=7200,
    examples=[
        {"input": {"gradians": 0}, "output": 0.0, "description": "0 gradians = 0°"},
        {
            "input": {"gradians": 100},
            "output": 90.0,
            "description": "100 gradians = 90°",
        },
        {
            "input": {"gradians": 200},
            "output": 180.0,
            "description": "200 gradians = 180°",
        },
        {
            "input": {"gradians": 400},
            "output": 360.0,
            "description": "400 gradians = 360°",
        },
    ],
)
async def gradians_to_degrees(gradians: Union[int, float]) -> float:
    """
    Convert gradians (grads) to degrees.

    Conversion factor: 360/400 = 9/10.

    Args:
        gradians: Angle in gradians

    Returns:
        Angle in degrees

    Examples:
        await gradians_to_degrees(0) → 0.0       # 0 grad
        await gradians_to_degrees(100) → 90.0    # 100 grad = 90°
        await gradians_to_degrees(200) → 180.0   # 200 grad = 180°
        await gradians_to_degrees(400) → 360.0   # 400 grad = 360°
    """
    return gradians * 360.0 / 400.0


# ============================================================================
# ANGLE NORMALIZATION FUNCTIONS
# ============================================================================


@mcp_function(
    description="Normalize an angle to a standard range.",
    namespace="trigonometry",
    category="angle_conversion",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    cache_ttl_seconds=3600,
    examples=[
        {
            "input": {"angle": 370, "unit": "degrees", "range": "positive"},
            "output": 10.0,
            "description": "370° normalized to [0, 360°)",
        },
        {
            "input": {"angle": -45, "unit": "degrees", "range": "symmetric"},
            "output": -45.0,
            "description": "-45° in [-180°, 180°)",
        },
        {
            "input": {"angle": 7.5, "unit": "radians", "range": "positive"},
            "output": 1.2168146928204138,
            "description": "7.5 rad normalized to [0, 2π)",
        },
    ],
)
async def normalize_angle(
    angle: Union[int, float],
    unit: Literal["radians", "degrees", "gradians"] = "radians",
    range_type: Literal["positive", "symmetric"] = "positive",
) -> float:
    """
    Normalize an angle to a standard range.

    Args:
        angle: Input angle
        unit: Unit of the angle ("radians", "degrees", "gradians")
        range_type: Type of normalization:
            - "positive": [0, full_circle)
            - "symmetric": [-half_circle, half_circle)

    Returns:
        Normalized angle in the same unit

    Examples:
        await normalize_angle(370, "degrees", "positive") → 10.0    # 370° → 10°
        await normalize_angle(-45, "degrees", "symmetric") → -45.0  # -45° stays -45°
        await normalize_angle(7.5, "radians", "positive") → 1.217   # 7.5 rad → ~1.217 rad
    """
    # Define full circle and half circle for each unit
    full_circles = {"radians": 2 * math.pi, "degrees": 360.0, "gradians": 400.0}

    full_circle = full_circles[unit]
    half_circle = full_circle / 2

    if range_type == "positive":
        # Normalize to [0, full_circle)
        normalized = angle % full_circle
        # Handle negative zero
        if normalized == 0.0 and angle < 0:
            normalized = 0.0
    else:  # symmetric
        # Normalize to [-half_circle, half_circle)
        # First normalize to [0, full_circle)
        temp = angle % full_circle
        # Then shift to symmetric range
        if temp > half_circle:
            normalized = temp - full_circle
        else:
            normalized = temp

    return normalized


@mcp_function(
    description="Calculate the shortest angular difference between two angles.",
    namespace="trigonometry",
    category="angle_conversion",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    cache_ttl_seconds=3600,
    examples=[
        {
            "input": {"angle1": 350, "angle2": 10, "unit": "degrees"},
            "output": 20.0,
            "description": "Shortest path: 350° to 10° = 20°",
        },
        {
            "input": {"angle1": 10, "angle2": 350, "unit": "degrees"},
            "output": -20.0,
            "description": "Shortest path: 10° to 350° = -20°",
        },
        {
            "input": {"angle1": 0, "angle2": 3.14159, "unit": "radians"},
            "output": 3.141592653589793,
            "description": "0 to π radians",
        },
    ],
)
async def angle_difference(
    angle1: Union[int, float],
    angle2: Union[int, float],
    unit: Literal["radians", "degrees", "gradians"] = "radians",
) -> float:
    """
    Calculate the shortest angular difference from angle1 to angle2.

    The result is positive if the shortest rotation is counterclockwise,
    negative if clockwise.

    Args:
        angle1: Starting angle
        angle2: Ending angle
        unit: Unit of the angles

    Returns:
        Shortest angular difference (angle2 - angle1) in the same unit
        Range: (-half_circle, half_circle]

    Examples:
        await angle_difference(350, 10, "degrees") → 20.0      # 350° → 10° (CCW)
        await angle_difference(10, 350, "degrees") → -20.0     # 10° → 350° (CW)
        await angle_difference(0, π, "radians") → π            # 0 → π (CCW)
        await angle_difference(π, 0, "radians") → -π           # π → 0 (CW)
    """
    # Calculate raw difference
    diff = angle2 - angle1

    # Normalize to symmetric range
    normalized_diff = await normalize_angle(diff, unit, "symmetric")

    return normalized_diff


# ============================================================================
# ADVANCED ANGLE UTILITIES
# ============================================================================


@mcp_function(
    description="Convert between different angle units in one operation.",
    namespace="trigonometry",
    category="angle_conversion",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    cache_ttl_seconds=3600,
    examples=[
        {
            "input": {"angle": 90, "from_unit": "degrees", "to_unit": "radians"},
            "output": 1.5707963267948966,
            "description": "90° to radians",
        },
        {
            "input": {"angle": 3.14159, "from_unit": "radians", "to_unit": "degrees"},
            "output": 179.99847695156393,
            "description": "π radians to degrees",
        },
        {
            "input": {"angle": 200, "from_unit": "gradians", "to_unit": "degrees"},
            "output": 180.0,
            "description": "200 gradians to degrees",
        },
    ],
)
async def convert_angle(
    angle: Union[int, float],
    from_unit: Literal["radians", "degrees", "gradians"],
    to_unit: Literal["radians", "degrees", "gradians"],
) -> float:
    """
    Convert an angle from one unit to another.

    Supports all combinations of radians, degrees, and gradians.

    Args:
        angle: Input angle value
        from_unit: Source unit
        to_unit: Target unit

    Returns:
        Converted angle in the target unit

    Examples:
        await convert_angle(90, "degrees", "radians") → π/2     # 90° → π/2 rad
        await convert_angle(π, "radians", "degrees") → 180.0    # π rad → 180°
        await convert_angle(200, "gradians", "degrees") → 180.0 # 200 grad → 180°
    """
    if from_unit == to_unit:
        return angle

    # Convert to radians first (common intermediate)
    if from_unit == "degrees":
        radians = await degrees_to_radians(angle)
    elif from_unit == "gradians":
        radians = await gradians_to_radians(angle)
    else:  # from_unit == "radians"
        radians = angle

    # Convert from radians to target unit
    if to_unit == "degrees":
        return await radians_to_degrees(radians)
    elif to_unit == "gradians":
        return await radians_to_gradians(radians)
    else:  # to_unit == "radians"
        return radians


@mcp_function(
    description="Get angle properties and equivalent representations.",
    namespace="trigonometry",
    category="angle_conversion",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    cache_ttl_seconds=3600,
    examples=[
        {
            "input": {"angle": 45, "unit": "degrees"},
            "output": {
                "radians": 0.7853981633974483,
                "degrees": 45.0,
                "gradians": 50.0,
                "normalized_positive": 45.0,
                "normalized_symmetric": 45.0,
                "quadrant": 1,
            },
            "description": "45° angle properties",
        },
        {
            "input": {"angle": -90, "unit": "degrees"},
            "output": {
                "radians": -1.5707963267948966,
                "degrees": -90.0,
                "gradians": -100.0,
                "normalized_positive": 270.0,
                "normalized_symmetric": -90.0,
                "quadrant": 4,
            },
            "description": "-90° angle properties",
        },
    ],
)
async def angle_properties(
    angle: Union[int, float],
    unit: Literal["radians", "degrees", "gradians"] = "degrees",
) -> dict:
    """
    Get comprehensive properties of an angle.

    Returns angle in all units, normalized forms, quadrant, and reference angle.

    Args:
        angle: Input angle
        unit: Unit of the input angle

    Returns:
        Dictionary containing:
        - radians: Angle in radians
        - degrees: Angle in degrees
        - gradians: Angle in gradians
        - normalized_positive: Normalized to [0, full_circle)
        - normalized_symmetric: Normalized to [-half_circle, half_circle)
        - quadrant: Quadrant number (1-4)
        - reference_angle: Reference angle in input unit
        - coterminal_angles: List of coterminal angles

    Examples:
        await angle_properties(45, "degrees") → {...}
        await angle_properties(π/4, "radians") → {...}
    """
    # Convert to all units
    if unit == "degrees":
        degrees = angle
        radians = await degrees_to_radians(angle)
        gradians = await degrees_to_gradians(angle)
    elif unit == "radians":
        radians = angle
        degrees = await radians_to_degrees(angle)
        gradians = await radians_to_gradians(angle)
    else:  # gradians
        gradians = angle
        degrees = await gradians_to_degrees(angle)
        radians = await gradians_to_radians(angle)

    # Normalize angle
    normalized_positive = await normalize_angle(angle, unit, "positive")
    normalized_symmetric = await normalize_angle(angle, unit, "symmetric")

    # Determine quadrant (based on degrees for simplicity)
    norm_deg = await normalize_angle(degrees, "degrees", "positive")
    if 0 <= norm_deg < 90:
        quadrant = 1
    elif 90 <= norm_deg < 180:
        quadrant = 2
    elif 180 <= norm_deg < 270:
        quadrant = 3
    else:
        quadrant = 4

    # Calculate reference angle (acute angle to x-axis)
    if unit == "degrees":
        if quadrant == 1:
            ref_angle = norm_deg
        elif quadrant == 2:
            ref_angle = 180 - norm_deg
        elif quadrant == 3:
            ref_angle = norm_deg - 180
        else:  # quadrant == 4
            ref_angle = 360 - norm_deg
    elif unit == "radians":
        norm_rad = await normalize_angle(radians, "radians", "positive")
        if quadrant == 1:
            ref_angle = norm_rad
        elif quadrant == 2:
            ref_angle = math.pi - norm_rad
        elif quadrant == 3:
            ref_angle = norm_rad - math.pi
        else:  # quadrant == 4
            ref_angle = 2 * math.pi - norm_rad
    else:  # gradians
        norm_grad = await normalize_angle(gradians, "gradians", "positive")
        if quadrant == 1:
            ref_angle = norm_grad
        elif quadrant == 2:
            ref_angle = 200 - norm_grad
        elif quadrant == 3:
            ref_angle = norm_grad - 200
        else:  # quadrant == 4
            ref_angle = 400 - norm_grad

    # Generate some coterminal angles
    full_circles = {"radians": 2 * math.pi, "degrees": 360.0, "gradians": 400.0}
    full_circle = full_circles[unit]
    coterminal = [
        angle + full_circle,
        angle - full_circle,
        angle + 2 * full_circle,
        angle - 2 * full_circle,
    ]

    return {
        "radians": radians,
        "degrees": degrees,
        "gradians": gradians,
        "normalized_positive": normalized_positive,
        "normalized_symmetric": normalized_symmetric,
        "quadrant": quadrant,
        "reference_angle": ref_angle,
        "coterminal_angles": coterminal,
    }


@mcp_function(
    description="Calculate the angular velocity from period or frequency.",
    namespace="trigonometry",
    category="angle_conversion",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    examples=[
        {
            "input": {"period": 2.0, "unit": "seconds"},
            "output": {
                "angular_velocity_rad_per_sec": 3.141592653589793,
                "frequency_hz": 0.5,
                "period_sec": 2.0,
            },
            "description": "2-second period",
        },
        {
            "input": {"frequency": 60, "unit": "hz"},
            "output": {
                "angular_velocity_rad_per_sec": 377.0,
                "frequency_hz": 60,
                "period_sec": 0.016666666666666666,
            },
            "description": "60 Hz frequency",
        },
    ],
)
async def angular_velocity_from_period_or_frequency(
    period: Union[int, float, None] = None,
    frequency: Union[int, float, None] = None,
    unit: Literal["seconds", "hz"] = "seconds",
) -> dict:
    """
    Calculate angular velocity from period or frequency.

    Provide either period OR frequency, not both.

    Args:
        period: Period in seconds (if provided)
        frequency: Frequency in Hz (if provided)
        unit: Unit specification ("seconds" for period, "hz" for frequency)

    Returns:
        Dictionary with angular velocity (rad/s), frequency (Hz), and period (s)

    Examples:
        await angular_velocity_from_period_or_frequency(period=2.0) → {...}
        await angular_velocity_from_period_or_frequency(frequency=60.0, unit="hz") → {...}
    """
    if period is not None and frequency is not None:
        raise ValueError("Provide either period OR frequency, not both")

    if period is None and frequency is None:
        raise ValueError("Must provide either period or frequency")

    if period is not None:
        if period <= 0:
            raise ValueError("Period must be positive")
        freq_hz: float = 1.0 / period
        period_sec: float = period
    else:  # frequency is not None
        assert frequency is not None  # Type narrowing
        if frequency <= 0:
            raise ValueError("Frequency must be positive")
        freq_hz = frequency
        period_sec = 1.0 / frequency

    angular_velocity = 2 * math.pi * freq_hz

    return {
        "angular_velocity_rad_per_sec": angular_velocity,
        "frequency_hz": freq_hz,
        "period_sec": period_sec,
    }


# Export all functions
__all__ = [
    # Basic conversions
    "degrees_to_radians",
    "radians_to_degrees",
    "gradians_to_radians",
    "radians_to_gradians",
    "degrees_to_gradians",
    "gradians_to_degrees",
    # Normalization and utilities
    "normalize_angle",
    "angle_difference",
    "convert_angle",
    "angle_properties",
    "angular_velocity_from_period_or_frequency",
]

if __name__ == "__main__":
    import asyncio

    async def test_angle_conversion_functions():
        """Test angle conversion functions."""
        print("🔄 Angle Conversion Functions Test")
        print("=" * 35)

        # Test basic conversions
        print("Basic Conversions:")
        test_angles = [0, 30, 45, 90, 180, 270, 360]
        for deg in test_angles:
            rad = await degrees_to_radians(deg)
            back_to_deg = await radians_to_degrees(rad)
            grad = await degrees_to_gradians(deg)
            print(f"  {deg}° = {rad:.6f} rad = {grad:.1f} grad")
            print(f"    Back to degrees: {back_to_deg:.6f}°")

        print("\nAngle Normalization:")
        test_cases = [
            (370, "degrees", "positive"),
            (-45, "degrees", "symmetric"),
            (7.5, "radians", "positive"),
            (-2.5, "radians", "symmetric"),
        ]

        for angle, unit, range_type in test_cases:
            normalized = await normalize_angle(angle, unit, range_type)
            print(
                f"  normalize_angle({angle}, {unit}, {range_type}) = {normalized:.6f}"
            )

        print("\nAngle Differences:")
        diff_cases = [
            (350, 10, "degrees"),
            (10, 350, "degrees"),
            (0, math.pi, "radians"),
            (math.pi, 0, "radians"),
        ]

        for a1, a2, unit in diff_cases:
            diff = await angle_difference(a1, a2, unit)
            print(f"  angle_difference({a1}, {a2}, {unit}) = {diff:.6f}")

        print("\nAngle Properties:")
        props = await angle_properties(225, "degrees")
        print("  225° properties:")
        print(f"    Radians: {props['radians']:.6f}")
        print(f"    Quadrant: {props['quadrant']}")
        print(f"    Reference angle: {props['reference_angle']:.6f}°")
        print(f"    Normalized positive: {props['normalized_positive']:.6f}°")

        print("\nAngular Velocity:")
        velocity_data = await angular_velocity_from_period_or_frequency(period=2.0)
        print("  Period = 2.0 s:")
        print(
            f"    Angular velocity: {velocity_data['angular_velocity_rad_per_sec']:.6f} rad/s"
        )
        print(f"    Frequency: {velocity_data['frequency_hz']:.6f} Hz")

        print("\n✅ All angle conversion functions working!")

    asyncio.run(test_angle_conversion_functions())
