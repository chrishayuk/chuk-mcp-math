#!/usr/bin/env python3
# chuk_mcp_math/trigonometry/inverse_functions.py
"""
Inverse Trigonometric Functions - Async Native

Comprehensive inverse trigonometric functions with domain validation and range handling.
Includes both standard inverse functions and their degree variants.

Functions:
- asin, acos, atan: Standard inverse functions (radians output)
- atan2: Two-argument arctangent for full quadrant determination
- acsc, asec, acot: Inverse reciprocal functions
- asin_degrees, acos_degrees, atan_degrees: Degree output variants
- Comprehensive domain validation and range restrictions
"""

import math
import asyncio
from typing import Union
from chuk_mcp_math.mcp_decorator import mcp_function

# ============================================================================
# STANDARD INVERSE TRIGONOMETRIC FUNCTIONS (RADIANS OUTPUT)
# ============================================================================


@mcp_function(
    description="Calculate arcsine (inverse sine) with domain validation.",
    namespace="trigonometry",
    category="inverse_functions",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    cache_ttl_seconds=3600,
    examples=[
        {"input": {"value": 0}, "output": 0.0, "description": "asin(0) = 0"},
        {
            "input": {"value": 1},
            "output": 1.5707963267948966,
            "description": "asin(1) = π/2",
        },
        {
            "input": {"value": -1},
            "output": -1.5707963267948966,
            "description": "asin(-1) = -π/2",
        },
        {
            "input": {"value": 0.7071067811865476},
            "output": 0.7853981633974483,
            "description": "asin(√2/2) = π/4",
        },
    ],
)
async def asin(value: Union[int, float]) -> float:
    """
    Calculate the arcsine (inverse sine) of a value.

    Returns the angle whose sine is the given value.
    Domain: [-1, 1], Range: [-π/2, π/2]

    Args:
        value: Input value, must be in [-1, 1]

    Returns:
        Arcsine in radians, range [-π/2, π/2]

    Raises:
        ValueError: If value is outside [-1, 1]

    Examples:
        await asin(0) → 0.0                  # asin(0) = 0°
        await asin(1) → π/2                  # asin(1) = 90°
        await asin(-1) → -π/2                # asin(-1) = -90°
        await asin(0.5) → π/6                # asin(0.5) = 30°
    """
    # Domain validation
    if not -1 <= value <= 1:
        raise ValueError(f"asin domain error: value {value} not in [-1, 1]")

    # Handle exact values for better precision
    if value == 0:
        return 0.0
    elif value == 1:
        return math.pi / 2
    elif value == -1:
        return -math.pi / 2
    elif abs(value - 0.5) < 1e-15:
        return math.pi / 6  # 30°
    elif abs(value + 0.5) < 1e-15:
        return -math.pi / 6  # -30°
    elif abs(value - math.sqrt(2) / 2) < 1e-15:
        return math.pi / 4  # 45°
    elif abs(value + math.sqrt(2) / 2) < 1e-15:
        return -math.pi / 4  # -45°
    elif abs(value - math.sqrt(3) / 2) < 1e-15:
        return math.pi / 3  # 60°
    elif abs(value + math.sqrt(3) / 2) < 1e-15:
        return -math.pi / 3  # -60°

    return math.asin(value)


@mcp_function(
    description="Calculate arccosine (inverse cosine) with domain validation.",
    namespace="trigonometry",
    category="inverse_functions",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    cache_ttl_seconds=3600,
    examples=[
        {"input": {"value": 1}, "output": 0.0, "description": "acos(1) = 0"},
        {
            "input": {"value": 0},
            "output": 1.5707963267948966,
            "description": "acos(0) = π/2",
        },
        {
            "input": {"value": -1},
            "output": 3.141592653589793,
            "description": "acos(-1) = π",
        },
        {
            "input": {"value": 0.7071067811865476},
            "output": 0.7853981633974483,
            "description": "acos(√2/2) = π/4",
        },
    ],
)
async def acos(value: Union[int, float]) -> float:
    """
    Calculate the arccosine (inverse cosine) of a value.

    Returns the angle whose cosine is the given value.
    Domain: [-1, 1], Range: [0, π]

    Args:
        value: Input value, must be in [-1, 1]

    Returns:
        Arccosine in radians, range [0, π]

    Raises:
        ValueError: If value is outside [-1, 1]

    Examples:
        await acos(1) → 0.0                  # acos(1) = 0°
        await acos(0) → π/2                  # acos(0) = 90°
        await acos(-1) → π                   # acos(-1) = 180°
        await acos(0.5) → π/3                # acos(0.5) = 60°
    """
    # Domain validation
    if not -1 <= value <= 1:
        raise ValueError(f"acos domain error: value {value} not in [-1, 1]")

    # Handle exact values for better precision
    if value == 1:
        return 0.0
    elif value == 0:
        return math.pi / 2
    elif value == -1:
        return math.pi
    elif abs(value - 0.5) < 1e-15:
        return math.pi / 3  # 60°
    elif abs(value + 0.5) < 1e-15:
        return 2 * math.pi / 3  # 120°
    elif abs(value - math.sqrt(2) / 2) < 1e-15:
        return math.pi / 4  # 45°
    elif abs(value + math.sqrt(2) / 2) < 1e-15:
        return 3 * math.pi / 4  # 135°
    elif abs(value - math.sqrt(3) / 2) < 1e-15:
        return math.pi / 6  # 30°
    elif abs(value + math.sqrt(3) / 2) < 1e-15:
        return 5 * math.pi / 6  # 150°

    return math.acos(value)


@mcp_function(
    description="Calculate arctangent (inverse tangent).",
    namespace="trigonometry",
    category="inverse_functions",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    cache_ttl_seconds=3600,
    examples=[
        {"input": {"value": 0}, "output": 0.0, "description": "atan(0) = 0"},
        {
            "input": {"value": 1},
            "output": 0.7853981633974483,
            "description": "atan(1) = π/4",
        },
        {
            "input": {"value": -1},
            "output": -0.7853981633974483,
            "description": "atan(-1) = -π/4",
        },
        {
            "input": {"value": 1.7320508075688772},
            "output": 1.0471975511965976,
            "description": "atan(√3) = π/3",
        },
    ],
)
async def atan(value: Union[int, float]) -> float:
    """
    Calculate the arctangent (inverse tangent) of a value.

    Returns the angle whose tangent is the given value.
    Domain: (-∞, ∞), Range: (-π/2, π/2)

    Args:
        value: Input value (any real number)

    Returns:
        Arctangent in radians, range (-π/2, π/2)

    Examples:
        await atan(0) → 0.0                  # atan(0) = 0°
        await atan(1) → π/4                  # atan(1) = 45°
        await atan(-1) → -π/4                # atan(-1) = -45°
        await atan(√3) → π/3                 # atan(√3) = 60°
    """
    # Handle exact values for better precision
    if value == 0:
        return 0.0
    elif value == 1:
        return math.pi / 4  # 45°
    elif value == -1:
        return -math.pi / 4  # -45°
    elif abs(value - math.sqrt(3)) < 1e-15:
        return math.pi / 3  # 60°
    elif abs(value + math.sqrt(3)) < 1e-15:
        return -math.pi / 3  # -60°
    elif abs(value - 1 / math.sqrt(3)) < 1e-15:
        return math.pi / 6  # 30°
    elif abs(value + 1 / math.sqrt(3)) < 1e-15:
        return -math.pi / 6  # -30°

    return math.atan(value)


@mcp_function(
    description="Calculate two-argument arctangent for full quadrant determination.",
    namespace="trigonometry",
    category="inverse_functions",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    cache_ttl_seconds=3600,
    examples=[
        {
            "input": {"y": 1, "x": 1},
            "output": 0.7853981633974483,
            "description": "atan2(1, 1) = π/4 (Q1)",
        },
        {
            "input": {"y": 1, "x": -1},
            "output": 2.356194490192345,
            "description": "atan2(1, -1) = 3π/4 (Q2)",
        },
        {
            "input": {"y": -1, "x": -1},
            "output": -2.356194490192345,
            "description": "atan2(-1, -1) = -3π/4 (Q3)",
        },
        {
            "input": {"y": -1, "x": 1},
            "output": -0.7853981633974483,
            "description": "atan2(-1, 1) = -π/4 (Q4)",
        },
    ],
)
async def atan2(y: Union[int, float], x: Union[int, float]) -> float:
    """
    Calculate the two-argument arctangent of y/x.

    Returns the angle from the positive x-axis to the point (x, y).
    Handles all quadrants correctly and the cases where x = 0.
    Domain: x, y not both zero, Range: (-π, π]

    Args:
        y: Y coordinate
        x: X coordinate

    Returns:
        Angle in radians, range (-π, π]

    Raises:
        ValueError: If both x and y are zero

    Examples:
        await atan2(1, 1) → π/4              # 45° in Q1
        await atan2(1, -1) → 3π/4            # 135° in Q2
        await atan2(-1, -1) → -3π/4          # -135° in Q3
        await atan2(-1, 1) → -π/4            # -45° in Q4
        await atan2(1, 0) → π/2              # 90° (positive y-axis)
    """
    # Check for the undefined case
    if x == 0 and y == 0:
        raise ValueError("atan2 undefined for (0, 0)")

    # Handle special cases for better precision
    if x == 0:
        if y > 0:
            return math.pi / 2  # 90°
        else:  # y < 0
            return -math.pi / 2  # -90°
    elif y == 0:
        if x > 0:
            return 0.0  # 0°
        else:  # x < 0
            return math.pi  # 180°

    # Handle quadrant cases with exact values
    if x > 0 and y > 0:  # Q1
        if abs(y / x - 1) < 1e-15:
            return math.pi / 4  # 45°
    elif x < 0 and y > 0:  # Q2
        if abs(y / (-x) - 1) < 1e-15:
            return 3 * math.pi / 4  # 135°
    elif x < 0 and y < 0:  # Q3
        if abs((-y) / (-x) - 1) < 1e-15:
            return -3 * math.pi / 4  # -135°
    elif x > 0 and y < 0:  # Q4
        if abs((-y) / x - 1) < 1e-15:
            return -math.pi / 4  # -45°

    return math.atan2(y, x)


# ============================================================================
# INVERSE RECIPROCAL TRIGONOMETRIC FUNCTIONS
# ============================================================================


@mcp_function(
    description="Calculate arccosecant (inverse cosecant).",
    namespace="trigonometry",
    category="inverse_functions",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    cache_ttl_seconds=3600,
    examples=[
        {
            "input": {"value": 1},
            "output": 1.5707963267948966,
            "description": "acsc(1) = π/2",
        },
        {
            "input": {"value": -1},
            "output": -1.5707963267948966,
            "description": "acsc(-1) = -π/2",
        },
        {
            "input": {"value": 2},
            "output": 0.5235987755982988,
            "description": "acsc(2) = π/6",
        },
    ],
)
async def acsc(value: Union[int, float]) -> float:
    """
    Calculate the arccosecant (inverse cosecant) of a value.

    acsc(x) = asin(1/x)
    Domain: (-∞, -1] ∪ [1, ∞), Range: [-π/2, 0) ∪ (0, π/2]

    Args:
        value: Input value, must satisfy |value| ≥ 1

    Returns:
        Arccosecant in radians

    Raises:
        ValueError: If |value| < 1

    Examples:
        await acsc(1) → π/2                  # acsc(1) = 90°
        await acsc(-1) → -π/2                # acsc(-1) = -90°
        await acsc(2) → π/6                  # acsc(2) = 30°
        await acsc(-2) → -π/6                # acsc(-2) = -30°
    """
    # Domain validation
    if abs(value) < 1:
        raise ValueError(f"acsc domain error: |{value}| must be ≥ 1")

    # acsc(x) = asin(1/x)
    return await asin(1.0 / value)


@mcp_function(
    description="Calculate arcsecant (inverse secant).",
    namespace="trigonometry",
    category="inverse_functions",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    cache_ttl_seconds=3600,
    examples=[
        {"input": {"value": 1}, "output": 0.0, "description": "asec(1) = 0"},
        {
            "input": {"value": -1},
            "output": 3.141592653589793,
            "description": "asec(-1) = π",
        },
        {
            "input": {"value": 2},
            "output": 1.0471975511965976,
            "description": "asec(2) = π/3",
        },
    ],
)
async def asec(value: Union[int, float]) -> float:
    """
    Calculate the arcsecant (inverse secant) of a value.

    asec(x) = acos(1/x)
    Domain: (-∞, -1] ∪ [1, ∞), Range: [0, π/2) ∪ (π/2, π]

    Args:
        value: Input value, must satisfy |value| ≥ 1

    Returns:
        Arcsecant in radians

    Raises:
        ValueError: If |value| < 1

    Examples:
        await asec(1) → 0.0                  # asec(1) = 0°
        await asec(-1) → π                   # asec(-1) = 180°
        await asec(2) → π/3                  # asec(2) = 60°
        await asec(-2) → 2π/3                # asec(-2) = 120°
    """
    # Domain validation
    if abs(value) < 1:
        raise ValueError(f"asec domain error: |{value}| must be ≥ 1")

    # asec(x) = acos(1/x)
    return await acos(1.0 / value)


@mcp_function(
    description="Calculate arccotangent (inverse cotangent).",
    namespace="trigonometry",
    category="inverse_functions",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    cache_ttl_seconds=3600,
    examples=[
        {
            "input": {"value": 1},
            "output": 0.7853981633974483,
            "description": "acot(1) = π/4",
        },
        {
            "input": {"value": 0},
            "output": 1.5707963267948966,
            "description": "acot(0) = π/2",
        },
        {
            "input": {"value": 1.7320508075688772},
            "output": 0.5235987755982988,
            "description": "acot(√3) = π/6",
        },
    ],
)
async def acot(value: Union[int, float]) -> float:
    """
    Calculate the arccotangent (inverse cotangent) of a value.

    acot(x) = π/2 - atan(x) for x > 0
    acot(x) = -π/2 - atan(x) for x < 0
    acot(0) = π/2
    Domain: (-∞, ∞), Range: (0, π)

    Args:
        value: Input value (any real number)

    Returns:
        Arccotangent in radians, range (0, π)

    Examples:
        await acot(1) → π/4                  # acot(1) = 45°
        await acot(0) → π/2                  # acot(0) = 90°
        await acot(-1) → 3π/4                # acot(-1) = 135°
        await acot(√3) → π/6                 # acot(√3) = 30°
    """
    if value == 0:
        return math.pi / 2

    # Handle exact values
    if value == 1:
        return math.pi / 4  # 45°
    elif value == -1:
        return 3 * math.pi / 4  # 135°
    elif abs(value - math.sqrt(3)) < 1e-15:
        return math.pi / 6  # 30°
    elif abs(value + math.sqrt(3)) < 1e-15:
        return 5 * math.pi / 6  # 150°
    elif abs(value - 1 / math.sqrt(3)) < 1e-15:
        return math.pi / 3  # 60°
    elif abs(value + 1 / math.sqrt(3)) < 1e-15:
        return 2 * math.pi / 3  # 120°

    # General formula: acot(x) = atan(1/x) + (π if x < 0 else 0)
    if value > 0:
        return math.atan(1.0 / value)
    else:  # value < 0
        return math.atan(1.0 / value) + math.pi


# ============================================================================
# DEGREE VARIANTS OF INVERSE FUNCTIONS
# ============================================================================


@mcp_function(
    description="Calculate arcsine with result in degrees.",
    namespace="trigonometry",
    category="inverse_functions",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    cache_ttl_seconds=3600,
    examples=[
        {"input": {"value": 0}, "output": 0.0, "description": "asin_degrees(0) = 0°"},
        {"input": {"value": 1}, "output": 90.0, "description": "asin_degrees(1) = 90°"},
        {
            "input": {"value": 0.5},
            "output": 30.0,
            "description": "asin_degrees(0.5) = 30°",
        },
    ],
)
async def asin_degrees(value: Union[int, float]) -> float:
    """
    Calculate the arcsine with result in degrees.

    Args:
        value: Input value, must be in [-1, 1]

    Returns:
        Arcsine in degrees, range [-90°, 90°]

    Examples:
        await asin_degrees(0) → 0.0          # asin(0) = 0°
        await asin_degrees(1) → 90.0         # asin(1) = 90°
        await asin_degrees(0.5) → 30.0       # asin(0.5) = 30°
    """
    radians_result = await asin(value)
    return math.degrees(radians_result)


@mcp_function(
    description="Calculate arccosine with result in degrees.",
    namespace="trigonometry",
    category="inverse_functions",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    cache_ttl_seconds=3600,
    examples=[
        {"input": {"value": 1}, "output": 0.0, "description": "acos_degrees(1) = 0°"},
        {"input": {"value": 0}, "output": 90.0, "description": "acos_degrees(0) = 90°"},
        {
            "input": {"value": 0.5},
            "output": 60.0,
            "description": "acos_degrees(0.5) = 60°",
        },
    ],
)
async def acos_degrees(value: Union[int, float]) -> float:
    """
    Calculate the arccosine with result in degrees.

    Args:
        value: Input value, must be in [-1, 1]

    Returns:
        Arccosine in degrees, range [0°, 180°]

    Examples:
        await acos_degrees(1) → 0.0          # acos(1) = 0°
        await acos_degrees(0) → 90.0         # acos(0) = 90°
        await acos_degrees(0.5) → 60.0       # acos(0.5) = 60°
    """
    radians_result = await acos(value)
    return math.degrees(radians_result)


@mcp_function(
    description="Calculate arctangent with result in degrees.",
    namespace="trigonometry",
    category="inverse_functions",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    cache_ttl_seconds=3600,
    examples=[
        {"input": {"value": 0}, "output": 0.0, "description": "atan_degrees(0) = 0°"},
        {"input": {"value": 1}, "output": 45.0, "description": "atan_degrees(1) = 45°"},
        {
            "input": {"value": 1.7320508075688772},
            "output": 60.0,
            "description": "atan_degrees(√3) = 60°",
        },
    ],
)
async def atan_degrees(value: Union[int, float]) -> float:
    """
    Calculate the arctangent with result in degrees.

    Args:
        value: Input value (any real number)

    Returns:
        Arctangent in degrees, range (-90°, 90°)

    Examples:
        await atan_degrees(0) → 0.0          # atan(0) = 0°
        await atan_degrees(1) → 45.0         # atan(1) = 45°
        await atan_degrees(√3) → 60.0        # atan(√3) = 60°
    """
    radians_result = await atan(value)
    return math.degrees(radians_result)


@mcp_function(
    description="Calculate two-argument arctangent with result in degrees.",
    namespace="trigonometry",
    category="inverse_functions",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    cache_ttl_seconds=3600,
    examples=[
        {
            "input": {"y": 1, "x": 1},
            "output": 45.0,
            "description": "atan2_degrees(1, 1) = 45°",
        },
        {
            "input": {"y": 1, "x": -1},
            "output": 135.0,
            "description": "atan2_degrees(1, -1) = 135°",
        },
        {
            "input": {"y": 1, "x": 0},
            "output": 90.0,
            "description": "atan2_degrees(1, 0) = 90°",
        },
    ],
)
async def atan2_degrees(y: Union[int, float], x: Union[int, float]) -> float:
    """
    Calculate the two-argument arctangent with result in degrees.

    Args:
        y: Y coordinate
        x: X coordinate

    Returns:
        Angle in degrees, range (-180°, 180°]

    Examples:
        await atan2_degrees(1, 1) → 45.0     # 45° in Q1
        await atan2_degrees(1, -1) → 135.0   # 135° in Q2
        await atan2_degrees(1, 0) → 90.0     # 90° (positive y-axis)
    """
    radians_result = await atan2(y, x)
    return math.degrees(radians_result)


# Export all functions
__all__ = [
    # Standard inverse functions (radians)
    "asin",
    "acos",
    "atan",
    "atan2",
    # Inverse reciprocal functions
    "acsc",
    "asec",
    "acot",
    # Degree variants
    "asin_degrees",
    "acos_degrees",
    "atan_degrees",
    "atan2_degrees",
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

    async def test_inverse_trigonometric_functions():
        """Test inverse trigonometric functions."""
        print("🔄 Inverse Trigonometric Functions Test")
        print("=" * 40)

        # Test standard inverse functions
        print("Standard Inverse Functions (radians):")
        test_values = [0, 0.5, math.sqrt(2) / 2, math.sqrt(3) / 2, 1]

        for value in test_values:
            try:
                asin_val = await asin(value)
                acos_val = await acos(value)
                print(
                    f"  asin({value:.4f}) = {asin_val:.6f}, acos({value:.4f}) = {acos_val:.6f}"
                )
            except ValueError as e:
                print(f"  Error with value {value}: {e}")

        print("\nArctangent and atan2:")
        atan_values = [0, 1, -1, math.sqrt(3), 1 / math.sqrt(3)]
        for value in atan_values:
            atan_val = await atan(value)
            print(f"  atan({value:.4f}) = {atan_val:.6f}")

        # Test atan2 in all quadrants
        quadrant_tests = [(1, 1), (1, -1), (-1, -1), (-1, 1), (1, 0), (0, 1)]
        for y, x in quadrant_tests:
            atan2_val = await atan2(y, x)
            atan2_deg = await atan2_degrees(y, x)
            print(f"  atan2({y}, {x}) = {atan2_val:.6f} rad = {atan2_deg:.1f}°")

        print("\nInverse Reciprocal Functions:")
        recip_values = [1, -1, 2, -2, math.sqrt(2)]
        for value in recip_values:
            try:
                acsc_val = await acsc(value)
                asec_val = await asec(value)
                print(
                    f"  acsc({value:.4f}) = {acsc_val:.6f}, asec({value:.4f}) = {asec_val:.6f}"
                )
            except ValueError as e:
                print(f"  Error with value {value}: {e}")

        # Test acot (works for all real numbers)
        cot_values = [0, 1, -1, math.sqrt(3), 1 / math.sqrt(3)]
        for value in cot_values:
            acot_val = await acot(value)
            print(f"  acot({value:.4f}) = {acot_val:.6f}")

        print("\nDegree Variants:")
        degree_test_values = [0, 0.5, math.sqrt(2) / 2, 1]
        for value in degree_test_values:
            try:
                asin_deg = await asin_degrees(value)
                acos_deg = await acos_degrees(value)
                print(
                    f"  asin_degrees({value:.4f}) = {asin_deg:.1f}°, acos_degrees({value:.4f}) = {acos_deg:.1f}°"
                )
            except ValueError as e:
                print(f"  Error with value {value}: {e}")

        print("\n✅ All inverse trigonometric functions working!")

    asyncio.run(test_inverse_trigonometric_functions())
