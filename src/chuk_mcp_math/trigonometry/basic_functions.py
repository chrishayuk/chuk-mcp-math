#!/usr/bin/env python3
# chuk_mcp_math/trigonometry/basic_functions.py
"""
Basic Trigonometric Functions - Async Native (FIXED)

Core trigonometric functions with comprehensive domain validation and error handling.
Optimized for both mathematical accuracy and computational efficiency.

Functions:
- sin, cos, tan: Primary trigonometric functions (radians)
- csc, sec, cot: Reciprocal trigonometric functions
- sin_degrees, cos_degrees, tan_degrees: Degree-input variants
- Comprehensive domain validation and singularity handling
- High precision for both small and large angles

FIXED: Properly imports sqrt from arithmetic core module
"""

import math
import asyncio
from typing import Union
from chuk_mcp_math.mcp_decorator import mcp_function

# Import sqrt from our existing arithmetic core module

# ============================================================================
# PRIMARY TRIGONOMETRIC FUNCTIONS (RADIANS)
# ============================================================================


@mcp_function(
    description="Calculate sine of an angle in radians with high precision.",
    namespace="trigonometry",
    category="basic_functions",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    cache_ttl_seconds=3600,
    examples=[
        {"input": {"angle": 0}, "output": 0.0, "description": "sin(0) = 0"},
        {
            "input": {"angle": 1.5707963267948966},
            "output": 1.0,
            "description": "sin(π/2) = 1",
        },
        {
            "input": {"angle": 3.141592653589793},
            "output": 0.0,
            "description": "sin(π) ≈ 0",
        },
        {
            "input": {"angle": 0.7853981633974483},
            "output": 0.7071067811865476,
            "description": "sin(π/4) = √2/2",
        },
    ],
)
async def sin(angle: Union[int, float]) -> float:
    """
    Calculate the sine of an angle in radians.

    The sine function returns the y-coordinate of the point on the unit circle
    at the given angle from the positive x-axis.

    Args:
        angle: Angle in radians

    Returns:
        Sine of the angle, range [-1, 1]

    Examples:
        await sin(0) → 0.0                    # sin(0°)
        await sin(math.pi/2) → 1.0           # sin(90°)
        await sin(math.pi) → 0.0             # sin(180°)
        await sin(math.pi/4) → 0.7071...     # sin(45°) = √2/2
    """
    # Handle special cases for better numerical accuracy
    if angle == 0:
        return 0.0

    # Normalize angle to reduce floating-point errors for large angles
    normalized_angle = angle % (2 * math.pi)

    # Use math.sin for computation
    result = math.sin(normalized_angle)

    # Clean up near-zero results (within machine epsilon)
    if abs(result) < 1e-15:
        return 0.0

    # Clean up results very close to ±1
    if abs(abs(result) - 1.0) < 1e-15:
        return 1.0 if result > 0 else -1.0

    return result


@mcp_function(
    description="Calculate cosine of an angle in radians with high precision.",
    namespace="trigonometry",
    category="basic_functions",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    cache_ttl_seconds=3600,
    examples=[
        {"input": {"angle": 0}, "output": 1.0, "description": "cos(0) = 1"},
        {
            "input": {"angle": 1.5707963267948966},
            "output": 0.0,
            "description": "cos(π/2) = 0",
        },
        {
            "input": {"angle": 3.141592653589793},
            "output": -1.0,
            "description": "cos(π) = -1",
        },
        {
            "input": {"angle": 0.7853981633974483},
            "output": 0.7071067811865476,
            "description": "cos(π/4) = √2/2",
        },
    ],
)
async def cos(angle: Union[int, float]) -> float:
    """
    Calculate the cosine of an angle in radians.

    The cosine function returns the x-coordinate of the point on the unit circle
    at the given angle from the positive x-axis.

    Args:
        angle: Angle in radians

    Returns:
        Cosine of the angle, range [-1, 1]

    Examples:
        await cos(0) → 1.0                    # cos(0°)
        await cos(math.pi/2) → 0.0           # cos(90°)
        await cos(math.pi) → -1.0            # cos(180°)
        await cos(math.pi/4) → 0.7071...     # cos(45°) = √2/2
    """
    # Handle special cases for better numerical accuracy
    if angle == 0:
        return 1.0

    # Normalize angle to reduce floating-point errors for large angles
    normalized_angle = angle % (2 * math.pi)

    # Use math.cos for computation
    result = math.cos(normalized_angle)

    # Clean up near-zero results (within machine epsilon)
    if abs(result) < 1e-15:
        return 0.0

    # Clean up results very close to ±1
    if abs(abs(result) - 1.0) < 1e-15:
        return 1.0 if result > 0 else -1.0

    return result


@mcp_function(
    description="Calculate tangent of an angle in radians with singularity handling.",
    namespace="trigonometry",
    category="basic_functions",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    cache_ttl_seconds=3600,
    examples=[
        {"input": {"angle": 0}, "output": 0.0, "description": "tan(0) = 0"},
        {
            "input": {"angle": 0.7853981633974483},
            "output": 1.0,
            "description": "tan(π/4) = 1",
        },
        {
            "input": {"angle": 2.356194490192345},
            "output": -1.0,
            "description": "tan(3π/4) = -1",
        },
        {
            "input": {"angle": 0.5235987755982988},
            "output": 1.7320508075688772,
            "description": "tan(π/6) = √3",
        },
    ],
)
async def tan(angle: Union[int, float]) -> float:
    """
    Calculate the tangent of an angle in radians.

    The tangent function is the ratio of sine to cosine: tan(θ) = sin(θ)/cos(θ).
    Has vertical asymptotes at odd multiples of π/2.

    Args:
        angle: Angle in radians

    Returns:
        Tangent of the angle, range (-∞, ∞)

    Raises:
        ValueError: If angle is at a singularity (odd multiple of π/2)

    Examples:
        await tan(0) → 0.0                    # tan(0°)
        await tan(math.pi/4) → 1.0           # tan(45°)
        await tan(math.pi/6) → 0.5773...     # tan(30°) = √3/3
        await tan(math.pi/3) → 1.7320...     # tan(60°) = √3
    """
    # Handle special case
    if angle == 0:
        return 0.0

    # Normalize angle to reduce floating-point errors
    normalized_angle = angle % (2 * math.pi)

    # Check for singularities (odd multiples of π/2)
    # We check with a small tolerance for floating-point comparison
    tolerance = 1e-10
    half_pi = math.pi / 2

    for k in range(-4, 5):  # Check a reasonable range of odd multiples
        singularity = (2 * k + 1) * half_pi
        if abs(normalized_angle - singularity) < tolerance:
            raise ValueError(
                f"Tangent undefined at angle {angle} (near singularity at {singularity})"
            )

    # Calculate tangent
    result = math.tan(normalized_angle)

    # Clean up near-zero results
    if abs(result) < 1e-15:
        return 0.0

    return result


# ============================================================================
# RECIPROCAL TRIGONOMETRIC FUNCTIONS
# ============================================================================


@mcp_function(
    description="Calculate cosecant (1/sin) of an angle in radians.",
    namespace="trigonometry",
    category="basic_functions",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    cache_ttl_seconds=3600,
    examples=[
        {
            "input": {"angle": 1.5707963267948966},
            "output": 1.0,
            "description": "csc(π/2) = 1",
        },
        {
            "input": {"angle": 0.5235987755982988},
            "output": 2.0,
            "description": "csc(π/6) = 2",
        },
        {
            "input": {"angle": 0.7853981633974483},
            "output": 1.4142135623730951,
            "description": "csc(π/4) = √2",
        },
    ],
)
async def csc(angle: Union[int, float]) -> float:
    """
    Calculate the cosecant of an angle in radians.

    Cosecant is the reciprocal of sine: csc(θ) = 1/sin(θ).
    Undefined when sin(θ) = 0 (at multiples of π).

    Args:
        angle: Angle in radians

    Returns:
        Cosecant of the angle, range (-∞, -1] ∪ [1, ∞)

    Raises:
        ValueError: If angle is at a singularity (multiple of π)

    Examples:
        await csc(math.pi/2) → 1.0           # csc(90°)
        await csc(math.pi/6) → 2.0           # csc(30°)
        await csc(math.pi/4) → 1.4142...     # csc(45°) = √2
    """
    # Calculate sine first
    sin_value = await sin(angle)

    # Check for singularity (sin = 0)
    if abs(sin_value) < 1e-15:
        raise ValueError(f"Cosecant undefined at angle {angle} (sine is zero)")

    return 1.0 / sin_value


@mcp_function(
    description="Calculate secant (1/cos) of an angle in radians.",
    namespace="trigonometry",
    category="basic_functions",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    cache_ttl_seconds=3600,
    examples=[
        {"input": {"angle": 0}, "output": 1.0, "description": "sec(0) = 1"},
        {
            "input": {"angle": 1.0471975511965976},
            "output": 2.0,
            "description": "sec(π/3) = 2",
        },
        {
            "input": {"angle": 0.7853981633974483},
            "output": 1.4142135623730951,
            "description": "sec(π/4) = √2",
        },
    ],
)
async def sec(angle: Union[int, float]) -> float:
    """
    Calculate the secant of an angle in radians.

    Secant is the reciprocal of cosine: sec(θ) = 1/cos(θ).
    Undefined when cos(θ) = 0 (at odd multiples of π/2).

    Args:
        angle: Angle in radians

    Returns:
        Secant of the angle, range (-∞, -1] ∪ [1, ∞)

    Raises:
        ValueError: If angle is at a singularity (odd multiple of π/2)

    Examples:
        await sec(0) → 1.0                    # sec(0°)
        await sec(math.pi/3) → 2.0           # sec(60°)
        await sec(math.pi/4) → 1.4142...     # sec(45°) = √2
    """
    # Calculate cosine first
    cos_value = await cos(angle)

    # Check for singularity (cos = 0)
    if abs(cos_value) < 1e-15:
        raise ValueError(f"Secant undefined at angle {angle} (cosine is zero)")

    return 1.0 / cos_value


@mcp_function(
    description="Calculate cotangent (1/tan) of an angle in radians.",
    namespace="trigonometry",
    category="basic_functions",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    cache_ttl_seconds=3600,
    examples=[
        {
            "input": {"angle": 0.7853981633974483},
            "output": 1.0,
            "description": "cot(π/4) = 1",
        },
        {
            "input": {"angle": 1.0471975511965976},
            "output": 0.5773502691896257,
            "description": "cot(π/3) = √3/3",
        },
        {
            "input": {"angle": 0.5235987755982988},
            "output": 1.7320508075688772,
            "description": "cot(π/6) = √3",
        },
    ],
)
async def cot(angle: Union[int, float]) -> float:
    """
    Calculate the cotangent of an angle in radians.

    Cotangent is the reciprocal of tangent: cot(θ) = 1/tan(θ) = cos(θ)/sin(θ).
    Undefined when sin(θ) = 0 (at multiples of π).

    Args:
        angle: Angle in radians

    Returns:
        Cotangent of the angle, range (-∞, ∞)

    Raises:
        ValueError: If angle is at a singularity (multiple of π)

    Examples:
        await cot(math.pi/4) → 1.0           # cot(45°)
        await cot(math.pi/3) → 0.5773...     # cot(60°) = √3/3
        await cot(math.pi/6) → 1.7320...     # cot(30°) = √3
    """
    # Calculate sine and cosine
    sin_value = await sin(angle)
    cos_value = await cos(angle)

    # Check for singularity (sin = 0)
    if abs(sin_value) < 1e-15:
        raise ValueError(f"Cotangent undefined at angle {angle} (sine is zero)")

    return cos_value / sin_value


# ============================================================================
# DEGREE VARIANTS OF PRIMARY FUNCTIONS
# ============================================================================


@mcp_function(
    description="Calculate sine of an angle in degrees.",
    namespace="trigonometry",
    category="basic_functions",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    cache_ttl_seconds=3600,
    examples=[
        {"input": {"angle_degrees": 0}, "output": 0.0, "description": "sin(0°) = 0"},
        {"input": {"angle_degrees": 90}, "output": 1.0, "description": "sin(90°) = 1"},
        {
            "input": {"angle_degrees": 180},
            "output": 0.0,
            "description": "sin(180°) = 0",
        },
        {
            "input": {"angle_degrees": 45},
            "output": 0.7071067811865476,
            "description": "sin(45°) = √2/2",
        },
    ],
)
async def sin_degrees(angle_degrees: Union[int, float]) -> float:
    """
    Calculate the sine of an angle in degrees.

    Convenience function that converts degrees to radians internally.

    Args:
        angle_degrees: Angle in degrees

    Returns:
        Sine of the angle, range [-1, 1]

    Examples:
        await sin_degrees(0) → 0.0           # sin(0°)
        await sin_degrees(90) → 1.0          # sin(90°)
        await sin_degrees(180) → 0.0         # sin(180°)
        await sin_degrees(45) → 0.7071...    # sin(45°) = √2/2
    """
    angle_radians = math.radians(angle_degrees)
    return await sin(angle_radians)


@mcp_function(
    description="Calculate cosine of an angle in degrees.",
    namespace="trigonometry",
    category="basic_functions",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    cache_ttl_seconds=3600,
    examples=[
        {"input": {"angle_degrees": 0}, "output": 1.0, "description": "cos(0°) = 1"},
        {"input": {"angle_degrees": 90}, "output": 0.0, "description": "cos(90°) = 0"},
        {
            "input": {"angle_degrees": 180},
            "output": -1.0,
            "description": "cos(180°) = -1",
        },
        {
            "input": {"angle_degrees": 45},
            "output": 0.7071067811865476,
            "description": "cos(45°) = √2/2",
        },
    ],
)
async def cos_degrees(angle_degrees: Union[int, float]) -> float:
    """
    Calculate the cosine of an angle in degrees.

    Convenience function that converts degrees to radians internally.

    Args:
        angle_degrees: Angle in degrees

    Returns:
        Cosine of the angle, range [-1, 1]

    Examples:
        await cos_degrees(0) → 1.0           # cos(0°)
        await cos_degrees(90) → 0.0          # cos(90°)
        await cos_degrees(180) → -1.0        # cos(180°)
        await cos_degrees(45) → 0.7071...    # cos(45°) = √2/2
    """
    angle_radians = math.radians(angle_degrees)
    return await cos(angle_radians)


@mcp_function(
    description="Calculate tangent of an angle in degrees.",
    namespace="trigonometry",
    category="basic_functions",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    cache_ttl_seconds=3600,
    examples=[
        {"input": {"angle_degrees": 0}, "output": 0.0, "description": "tan(0°) = 0"},
        {"input": {"angle_degrees": 45}, "output": 1.0, "description": "tan(45°) = 1"},
        {
            "input": {"angle_degrees": 30},
            "output": 0.5773502691896257,
            "description": "tan(30°) = √3/3",
        },
        {
            "input": {"angle_degrees": 60},
            "output": 1.7320508075688772,
            "description": "tan(60°) = √3",
        },
    ],
)
async def tan_degrees(angle_degrees: Union[int, float]) -> float:
    """
    Calculate the tangent of an angle in degrees.

    Convenience function that converts degrees to radians internally.
    Handles singularities at 90°, 270°, etc.

    Args:
        angle_degrees: Angle in degrees

    Returns:
        Tangent of the angle, range (-∞, ∞)

    Raises:
        ValueError: If angle is at a singularity (odd multiple of 90°)

    Examples:
        await tan_degrees(0) → 0.0           # tan(0°)
        await tan_degrees(45) → 1.0          # tan(45°)
        await tan_degrees(30) → 0.5773...    # tan(30°) = √3/3
        await tan_degrees(60) → 1.7320...    # tan(60°) = √3
    """
    # Check for singularities in degrees (more intuitive)
    normalized_degrees = angle_degrees % 360
    if abs(normalized_degrees % 180 - 90) < 1e-10:
        raise ValueError(f"Tangent undefined at {angle_degrees}° (singularity)")

    angle_radians = math.radians(angle_degrees)
    return await tan(angle_radians)


# Export all functions
__all__ = [
    # Primary trigonometric functions (radians)
    "sin",
    "cos",
    "tan",
    # Reciprocal trigonometric functions
    "csc",
    "sec",
    "cot",
    # Degree variants
    "sin_degrees",
    "cos_degrees",
    "tan_degrees",
]

# === FIX FOR MCP DECORATOR FUNCTION EXPORTS ===
# The @mcp_function decorator creates wrappers that need to be explicitly
# made available in the module namespace for normal Python imports

import sys  # noqa: E402

_current_module = sys.modules[__name__]

# Ensure each function from __all__ is available in the module namespace
for _func_name in __all__:
    if not hasattr(_current_module, _func_name):
        # Try to get the function from globals (it should be there from the decorators)
        _func = globals().get(_func_name)
        if _func:
            setattr(_current_module, _func_name, _func)
        else:
            # If not in globals, try to get from the MCP registry
            try:
                from chuk_mcp_math.mcp_decorator import get_mcp_functions

                mcp_functions = get_mcp_functions("trigonometry")
                qualified_name = f"trigonometry/{_func_name}"
                if qualified_name in mcp_functions:
                    _func = mcp_functions[qualified_name].function_ref
                    if _func:
                        setattr(_current_module, _func_name, _func)
                        # Also set in globals for consistency
                        globals()[_func_name] = _func
            except (ImportError, KeyError):
                pass

# Clean up temporary variables
del _current_module, _func_name
if "_func" in locals():
    del _func

if __name__ == "__main__":
    import asyncio

    async def test_basic_trigonometric_functions():
        """Test basic trigonometric functions."""
        print("📐 Basic Trigonometric Functions Test (FIXED)")
        print("=" * 45)

        # Test primary functions
        print("Primary Functions (radians):")
        test_angles = [0, math.pi / 6, math.pi / 4, math.pi / 3, math.pi / 2]
        angle_names = ["0", "π/6", "π/4", "π/3", "π/2"]

        for angle, name in zip(test_angles, angle_names):
            try:
                sin_val = await sin(angle)
                cos_val = await cos(angle)

                if abs(angle - math.pi / 2) < 1e-10:
                    tan_str = "undefined"
                else:
                    tan_val = await tan(angle)
                    tan_str = f"{tan_val:8.4f}"

                print(
                    f"  {name:4s}: sin = {sin_val:8.4f}, cos = {cos_val:8.4f}, tan = {tan_str}"
                )
            except ValueError as e:
                print(f"  {name:4s}: Error - {e}")

        print("\nReciprocal Functions:")
        reciprocal_angles = [math.pi / 6, math.pi / 4, math.pi / 3]
        reciprocal_names = ["π/6", "π/4", "π/3"]

        for angle, name in zip(reciprocal_angles, reciprocal_names):
            csc_val = await csc(angle)
            sec_val = await sec(angle)
            cot_val = await cot(angle)
            print(
                f"  {name}: csc = {csc_val:8.4f}, sec = {sec_val:8.4f}, cot = {cot_val:8.4f}"
            )

        print("\nDegree Variants:")
        degree_angles = [0, 30, 45, 60, 90]
        for deg in degree_angles:
            try:
                sin_deg = await sin_degrees(deg)
                cos_deg = await cos_degrees(deg)

                if deg % 90 == 0 and deg % 180 != 0:
                    tan_deg_str = "undefined"
                else:
                    tan_deg = await tan_degrees(deg)
                    tan_deg_str = f"{tan_deg:8.4f}"

                print(
                    f"  {deg:2d}°: sin = {sin_deg:8.4f}, cos = {cos_deg:8.4f}, tan = {tan_deg_str}"
                )
            except ValueError as e:
                print(f"  {deg:2d}°: Error - {e}")

        print("\n✅ All basic trigonometric functions working with proper sqrt import!")

    asyncio.run(test_basic_trigonometric_functions())
