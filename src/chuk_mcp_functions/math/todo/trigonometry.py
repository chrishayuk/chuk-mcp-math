#!/usr/bin/env python3
# chuk_mcp_functions/math/trigonometery.py
"""
Trigonometry Functions for AI Models

Trigonometric functions and conversions: sine, cosine, tangent, inverse functions,
and angle conversions. All functions include clear descriptions for AI model execution.
"""

import math
from typing import Union
from chuk_mcp_functions.mcp_decorator import mcp_function, CacheStrategy, ExecutionMode

@mcp_function(
    description="Calculate the sine of an angle in degrees. Returns the y-coordinate on the unit circle.",
    namespace="trigonometry",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    dependencies=["math"],
    examples=[
        {"input": {"degrees": 0}, "output": 0.0, "description": "sin(0°) = 0"},
        {"input": {"degrees": 30}, "output": 0.5, "description": "sin(30°) = 0.5"},
        {"input": {"degrees": 90}, "output": 1.0, "description": "sin(90°) = 1"}
    ]
)
def sin_degrees(degrees: Union[int, float]) -> float:
    """
    Calculate the sine of an angle in degrees.
    
    Args:
        degrees: Angle in degrees
    
    Returns:
        Sine of the angle
    
    Examples:
        sin_degrees(0) → 0.0
        sin_degrees(30) → 0.5
        sin_degrees(90) → 1.0
    """
    return math.sin(math.radians(degrees))

@mcp_function(
    description="Calculate the cosine of an angle in degrees. Returns the x-coordinate on the unit circle.",
    namespace="trigonometry",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    dependencies=["math"],
    examples=[
        {"input": {"degrees": 0}, "output": 1.0, "description": "cos(0°) = 1"},
        {"input": {"degrees": 60}, "output": 0.5, "description": "cos(60°) = 0.5"},
        {"input": {"degrees": 90}, "output": 0.0, "description": "cos(90°) = 0 (approx)"}
    ]
)
def cos_degrees(degrees: Union[int, float]) -> float:
    """
    Calculate the cosine of an angle in degrees.
    
    Args:
        degrees: Angle in degrees
    
    Returns:
        Cosine of the angle
    
    Examples:
        cos_degrees(0) → 1.0
        cos_degrees(60) → 0.5
        cos_degrees(90) → 6.123233995736766e-17 (essentially 0)
    """
    return math.cos(math.radians(degrees))

@mcp_function(
    description="Calculate the tangent of an angle in degrees. Returns the ratio of sine to cosine.",
    namespace="trigonometry",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    dependencies=["math"],
    examples=[
        {"input": {"degrees": 0}, "output": 0.0, "description": "tan(0°) = 0"},
        {"input": {"degrees": 45}, "output": 1.0, "description": "tan(45°) = 1"},
        {"input": {"degrees": 90}, "output": "very_large", "description": "tan(90°) approaches infinity"}
    ]
)
def tan_degrees(degrees: Union[int, float]) -> float:
    """
    Calculate the tangent of an angle in degrees.
    
    Args:
        degrees: Angle in degrees
    
    Returns:
        Tangent of the angle
    
    Note:
        Returns very large values near 90°, 270°, etc. where tangent approaches infinity
    
    Examples:
        tan_degrees(0) → 0.0
        tan_degrees(45) → 1.0
        tan_degrees(90) → 1.633123935319537e+16 (very large number)
    """
    return math.tan(math.radians(degrees))

@mcp_function(
    description="Calculate the sine of an angle in radians. Fundamental trigonometric function.",
    namespace="trigonometry",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    dependencies=["math"],
    examples=[
        {"input": {"radians": 0}, "output": 0.0, "description": "sin(0) = 0"},
        {"input": {"radians": 1.5708}, "output": 1.0, "description": "sin(π/2) ≈ 1"},
        {"input": {"radians": 3.14159}, "output": 0.0, "description": "sin(π) ≈ 0"}
    ]
)
def sin_radians(radians: Union[int, float]) -> float:
    """
    Calculate the sine of an angle in radians.
    
    Args:
        radians: Angle in radians
    
    Returns:
        Sine of the angle
    
    Examples:
        sin_radians(0) → 0.0
        sin_radians(math.pi/2) → 1.0
        sin_radians(math.pi) → 1.2246467991473532e-16 (essentially 0)
    """
    return math.sin(radians)

@mcp_function(
    description="Calculate the cosine of an angle in radians. Fundamental trigonometric function.",
    namespace="trigonometry",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    dependencies=["math"],
    examples=[
        {"input": {"radians": 0}, "output": 1.0, "description": "cos(0) = 1"},
        {"input": {"radians": 1.5708}, "output": 0.0, "description": "cos(π/2) ≈ 0"},
        {"input": {"radians": 3.14159}, "output": -1.0, "description": "cos(π) ≈ -1"}
    ]
)
def cos_radians(radians: Union[int, float]) -> float:
    """
    Calculate the cosine of an angle in radians.
    
    Args:
        radians: Angle in radians
    
    Returns:
        Cosine of the angle
    
    Examples:
        cos_radians(0) → 1.0
        cos_radians(math.pi/2) → 6.123233995736766e-17 (essentially 0)
        cos_radians(math.pi) → -1.0
    """
    return math.cos(radians)

@mcp_function(
    description="Calculate the tangent of an angle in radians. Fundamental trigonometric function.",
    namespace="trigonometry",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    dependencies=["math"],
    examples=[
        {"input": {"radians": 0}, "output": 0.0, "description": "tan(0) = 0"},
        {"input": {"radians": 0.7854}, "output": 1.0, "description": "tan(π/4) ≈ 1"},
        {"input": {"radians": 1.5708}, "output": "very_large", "description": "tan(π/2) approaches infinity"}
    ]
)
def tan_radians(radians: Union[int, float]) -> float:
    """
    Calculate the tangent of an angle in radians.
    
    Args:
        radians: Angle in radians
    
    Returns:
        Tangent of the angle
    
    Examples:
        tan_radians(0) → 0.0
        tan_radians(math.pi/4) → 0.9999999999999999 (essentially 1)
        tan_radians(math.pi/2) → 1.633123935319537e+16 (very large)
    """
    return math.tan(radians)

@mcp_function(
    description="Calculate the arcsine (inverse sine) of a value, returning the angle in degrees whose sine is the input.",
    namespace="trigonometry",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    dependencies=["math"],
    examples=[
        {"input": {"value": 0}, "output": 0.0, "description": "arcsin(0) = 0°"},
        {"input": {"value": 0.5}, "output": 30.0, "description": "arcsin(0.5) = 30°"},
        {"input": {"value": 1}, "output": 90.0, "description": "arcsin(1) = 90°"}
    ]
)
def arcsin_degrees(value: Union[int, float]) -> float:
    """
    Calculate the arcsine (inverse sine) in degrees.
    
    Args:
        value: Value between -1 and 1
    
    Returns:
        Angle in degrees whose sine is the input value
    
    Raises:
        ValueError: If value is outside [-1, 1]
    
    Examples:
        arcsin_degrees(0) → 0.0
        arcsin_degrees(0.5) → 30.0
        arcsin_degrees(1) → 90.0
    """
    if not -1 <= value <= 1:
        raise ValueError("Arcsine input must be between -1 and 1")
    return math.degrees(math.asin(value))

@mcp_function(
    description="Calculate the arccosine (inverse cosine) of a value, returning the angle in degrees whose cosine is the input.",
    namespace="trigonometry",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    dependencies=["math"],
    examples=[
        {"input": {"value": 1}, "output": 0.0, "description": "arccos(1) = 0°"},
        {"input": {"value": 0.5}, "output": 60.0, "description": "arccos(0.5) = 60°"},
        {"input": {"value": 0}, "output": 90.0, "description": "arccos(0) = 90°"}
    ]
)
def arccos_degrees(value: Union[int, float]) -> float:
    """
    Calculate the arccosine (inverse cosine) in degrees.
    
    Args:
        value: Value between -1 and 1
    
    Returns:
        Angle in degrees whose cosine is the input value
    
    Raises:
        ValueError: If value is outside [-1, 1]
    
    Examples:
        arccos_degrees(1) → 0.0
        arccos_degrees(0.5) → 60.0
        arccos_degrees(0) → 90.0
    """
    if not -1 <= value <= 1:
        raise ValueError("Arccosine input must be between -1 and 1")
    return math.degrees(math.acos(value))

@mcp_function(
    description="Calculate the arctangent (inverse tangent) of a value, returning the angle in degrees whose tangent is the input.",
    namespace="trigonometry",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    dependencies=["math"],
    examples=[
        {"input": {"value": 0}, "output": 0.0, "description": "arctan(0) = 0°"},
        {"input": {"value": 1}, "output": 45.0, "description": "arctan(1) = 45°"},
        {"input": {"value": 1.732}, "output": 60.0, "description": "arctan(√3) ≈ 60°"}
    ]
)
def arctan_degrees(value: Union[int, float]) -> float:
    """
    Calculate the arctangent (inverse tangent) in degrees.
    
    Args:
        value: Any real number
    
    Returns:
        Angle in degrees whose tangent is the input value
    
    Examples:
        arctan_degrees(0) → 0.0
        arctan_degrees(1) → 45.0
        arctan_degrees(math.sqrt(3)) → 59.99999999999999 (essentially 60°)
    """
    return math.degrees(math.atan(value))

@mcp_function(
    description="Convert degrees to radians. Multiply degrees by π/180. Essential for trigonometric calculations.",
    namespace="trigonometry",
    execution_modes=["local", "remote"],
    dependencies=["math"],
    examples=[
        {"input": {"degrees": 180}, "output": 3.14159, "description": "180° = π radians"},
        {"input": {"degrees": 90}, "output": 1.5708, "description": "90° = π/2 radians"},
        {"input": {"degrees": 360}, "output": 6.28318, "description": "360° = 2π radians"}
    ]
)
def degrees_to_radians(degrees: Union[int, float]) -> float:
    """
    Convert degrees to radians.
    
    Args:
        degrees: Angle in degrees
    
    Returns:
        Angle in radians
    
    Examples:
        degrees_to_radians(180) → 3.141592653589793
        degrees_to_radians(90) → 1.5707963267948966
        degrees_to_radians(360) → 6.283185307179586
    """
    return math.radians(degrees)

@mcp_function(
    description="Convert radians to degrees. Multiply radians by 180/π. Useful for displaying angles in familiar units.",
    namespace="trigonometry",
    execution_modes=["local", "remote"],
    dependencies=["math"],
    examples=[
        {"input": {"radians": 3.14159}, "output": 180.0, "description": "π radians = 180°"},
        {"input": {"radians": 1.5708}, "output": 90.0, "description": "π/2 radians = 90°"},
        {"input": {"radians": 6.28318}, "output": 360.0, "description": "2π radians = 360°"}
    ]
)
def radians_to_degrees(radians: Union[int, float]) -> float:
    """
    Convert radians to degrees.
    
    Args:
        radians: Angle in radians
    
    Returns:
        Angle in degrees
    
    Examples:
        radians_to_degrees(math.pi) → 180.0
        radians_to_degrees(math.pi/2) → 90.0
        radians_to_degrees(2*math.pi) → 360.0
    """
    return math.degrees(radians)

@mcp_function(
    description="Calculate the secant of an angle in degrees. Secant is the reciprocal of cosine: sec(θ) = 1/cos(θ).",
    namespace="trigonometry",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    dependencies=["math"],
    examples=[
        {"input": {"degrees": 0}, "output": 1.0, "description": "sec(0°) = 1/cos(0°) = 1"},
        {"input": {"degrees": 60}, "output": 2.0, "description": "sec(60°) = 1/cos(60°) = 2"},
        {"input": {"degrees": 90}, "output": "undefined", "description": "sec(90°) is undefined (cos(90°) = 0)"}
    ]
)
def sec_degrees(degrees: Union[int, float]) -> float:
    """
    Calculate the secant of an angle in degrees.
    
    Args:
        degrees: Angle in degrees
    
    Returns:
        Secant of the angle (1/cosine)
    
    Raises:
        ValueError: If cosine is zero (angle is 90°, 270°, etc.)
    
    Examples:
        sec_degrees(0) → 1.0
        sec_degrees(60) → 2.0
        sec_degrees(90) → raises ValueError
    """
    cos_val = math.cos(math.radians(degrees))
    if abs(cos_val) < 1e-10:  # Effectively zero
        raise ValueError(f"Secant is undefined at {degrees} degrees (cosine is zero)")
    return 1.0 / cos_val

@mcp_function(
    description="Calculate the cosecant of an angle in degrees. Cosecant is the reciprocal of sine: csc(θ) = 1/sin(θ).",
    namespace="trigonometry",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    dependencies=["math"],
    examples=[
        {"input": {"degrees": 90}, "output": 1.0, "description": "csc(90°) = 1/sin(90°) = 1"},
        {"input": {"degrees": 30}, "output": 2.0, "description": "csc(30°) = 1/sin(30°) = 2"},
        {"input": {"degrees": 0}, "output": "undefined", "description": "csc(0°) is undefined (sin(0°) = 0)"}
    ]
)
def csc_degrees(degrees: Union[int, float]) -> float:
    """
    Calculate the cosecant of an angle in degrees.
    
    Args:
        degrees: Angle in degrees
    
    Returns:
        Cosecant of the angle (1/sine)
    
    Raises:
        ValueError: If sine is zero (angle is 0°, 180°, etc.)
    
    Examples:
        csc_degrees(90) → 1.0
        csc_degrees(30) → 2.0
        csc_degrees(0) → raises ValueError
    """
    sin_val = math.sin(math.radians(degrees))
    if abs(sin_val) < 1e-10:  # Effectively zero
        raise ValueError(f"Cosecant is undefined at {degrees} degrees (sine is zero)")
    return 1.0 / sin_val

@mcp_function(
    description="Calculate the cotangent of an angle in degrees. Cotangent is the reciprocal of tangent: cot(θ) = 1/tan(θ) = cos(θ)/sin(θ).",
    namespace="trigonometry",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    dependencies=["math"],
    examples=[
        {"input": {"degrees": 45}, "output": 1.0, "description": "cot(45°) = 1/tan(45°) = 1"},
        {"input": {"degrees": 30}, "output": 1.732, "description": "cot(30°) = √3 ≈ 1.732"},
        {"input": {"degrees": 0}, "output": "undefined", "description": "cot(0°) is undefined (tan(0°) = 0)"}
    ]
)
def cot_degrees(degrees: Union[int, float]) -> float:
    """
    Calculate the cotangent of an angle in degrees.
    
    Args:
        degrees: Angle in degrees
    
    Returns:
        Cotangent of the angle (1/tangent)
    
    Raises:
        ValueError: If tangent is zero
    
    Examples:
        cot_degrees(45) → 1.0
        cot_degrees(30) → 1.7320508075688772
        cot_degrees(0) → raises ValueError
    """
    sin_val = math.sin(math.radians(degrees))
    cos_val = math.cos(math.radians(degrees))
    
    if abs(sin_val) < 1e-10:  # Effectively zero
        raise ValueError(f"Cotangent is undefined at {degrees} degrees (sine is zero)")
    
    return cos_val / sin_val

# Export all trigonometry functions
__all__ = [
    'sin_degrees', 'cos_degrees', 'tan_degrees',
    'sin_radians', 'cos_radians', 'tan_radians',
    'arcsin_degrees', 'arccos_degrees', 'arctan_degrees',
    'degrees_to_radians', 'radians_to_degrees',
    'sec_degrees', 'csc_degrees', 'cot_degrees'
]

if __name__ == "__main__":
    # Test the trigonometry functions
    print("📐 Trigonometry Functions Test")
    print("=" * 40)
    
    print(f"sin_degrees(30) = {sin_degrees(30)}")
    print(f"cos_degrees(60) = {cos_degrees(60)}")
    print(f"tan_degrees(45) = {tan_degrees(45)}")
    
    print(f"sin_radians(π/2) = {sin_radians(math.pi/2):.6f}")
    print(f"cos_radians(π) = {cos_radians(math.pi):.6f}")
    
    print(f"arcsin_degrees(0.5) = {arcsin_degrees(0.5)}")
    print(f"arccos_degrees(0.5) = {arccos_degrees(0.5)}")
    print(f"arctan_degrees(1) = {arctan_degrees(1)}")
    
    print(f"degrees_to_radians(180) = {degrees_to_radians(180):.6f}")
    print(f"radians_to_degrees(π) = {radians_to_degrees(math.pi):.6f}")
    
    print(f"sec_degrees(60) = {sec_degrees(60)}")
    print(f"csc_degrees(30) = {csc_degrees(30)}")
    print(f"cot_degrees(45) = {cot_degrees(45):.6f}")
    
    print("\n✅ All trigonometry functions working correctly!")