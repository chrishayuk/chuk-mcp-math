#!/usr/bin/env python3
# chuk_mcp_math/trigonometry/__init__.py
"""
Trigonometry Functions Library - Async Native

A comprehensive collection of trigonometric functions organized by domain.
Designed specifically for AI model execution with clear documentation,
examples, and robust error handling. All functions are async native for
optimal performance in async environments.

Trigonometric Domains:
- basic_functions: sin, cos, tan, csc, sec, cot - ASYNC NATIVE ✅
- inverse_functions: asin, acos, atan, atan2, acsc, asec, acot - ASYNC NATIVE ✅
- hyperbolic: sinh, cosh, tanh, csch, sech, coth - ASYNC NATIVE ✅
- inverse_hyperbolic: asinh, acosh, atanh, acsch, asech, acoth - ASYNC NATIVE ✅
- angle_conversion: deg_to_rad, rad_to_deg, grad_to_rad, etc. - ASYNC NATIVE ✅
- identities: verify_identity, simplify_expression, pythagorean_identity - ASYNC NATIVE ✅
- wave_analysis: amplitude, frequency, phase_shift, wave_properties - ASYNC NATIVE ✅
- applications: distance_calculation, bearing_navigation, oscillations - ASYNC NATIVE ✅

All functions support:
- Async native execution for optimal performance
- Local and remote execution modes
- Comprehensive error handling and domain validation
- Performance optimization with caching where appropriate
- Rich examples for AI understanding
- Type safety and validation
- Strategic yielding for long-running operations
"""

from typing import Dict, List, Any
import math
import asyncio

# Import trigonometry submodules
from . import basic_functions
from . import inverse_functions
from . import hyperbolic
from . import inverse_hyperbolic
from . import angle_conversion
from . import identities
from . import wave_analysis
from . import applications

# Import core functions for easier access
try:
    from chuk_mcp_math.mcp_decorator import get_mcp_functions

    _mcp_decorator_available = True
except ImportError:
    _mcp_decorator_available = False

# Core trigonometric functions (most commonly used)
from .basic_functions import (
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

# Inverse trigonometric functions
from .inverse_functions import (
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
)

# Hyperbolic functions
from .hyperbolic import sinh, cosh, tanh, csch, sech, coth

# Inverse hyperbolic functions
from .inverse_hyperbolic import asinh, acosh, atanh, acsch, asech, acoth

# Angle conversion functions
from .angle_conversion import (
    degrees_to_radians,
    radians_to_degrees,
    gradians_to_radians,
    radians_to_gradians,
    degrees_to_gradians,
    gradians_to_degrees,
    normalize_angle,
    angle_difference,
)

# Identity verification functions
from .identities import (
    pythagorean_identity,
    sum_difference_formulas,
    double_angle_formulas,
    half_angle_formulas,
    verify_identity,
    simplify_trig_expression,
)

# Wave analysis functions
from .wave_analysis import (
    amplitude_from_coefficients,
    frequency_from_period,
    phase_shift_analysis,
    wave_equation,
    harmonic_analysis,
    fourier_coefficients_basic,
)

# Application functions
from .applications import (
    distance_haversine,
    bearing_calculation,
    triangulation,
    oscillation_analysis,
    pendulum_period,
    spring_oscillation,
)


async def get_trigonometry_functions() -> Dict[str, Any]:
    """Get all trigonometric functions organized by domain (async)."""
    if not _mcp_decorator_available:
        return {
            "basic_functions": {},
            "inverse_functions": {},
            "hyperbolic": {},
            "inverse_hyperbolic": {},
            "angle_conversion": {},
            "identities": {},
            "wave_analysis": {},
            "applications": {},
        }

    all_funcs = get_mcp_functions()

    trig_domains: list[dict[str, Any]] = {  # type: ignore[assignment]
        "basic_functions": {},
        "inverse_functions": {},
        "hyperbolic": {},
        "inverse_hyperbolic": {},
        "angle_conversion": {},
        "identities": {},
        "wave_analysis": {},
        "applications": {},
    }

    # Organize functions by their namespace
    for name, spec in all_funcs.items():
        domain = spec.namespace
        if domain in trig_domains:
            trig_domains[domain][spec.function_name] = spec  # type: ignore[call-overload]

    return trig_domains  # type: ignore[return-value]


def get_trig_constants() -> Dict[str, float]:
    """Get all trigonometric constants."""
    return {
        "pi": math.pi,
        "tau": math.tau,  # 2π
        "e": math.e,
        "pi_2": math.pi / 2,  # π/2
        "pi_4": math.pi / 4,  # π/4
        "pi_3": math.pi / 3,  # π/3
        "pi_6": math.pi / 6,  # π/6
        "sqrt_2": math.sqrt(2),
        "sqrt_3": math.sqrt(3),
        "golden_ratio": (1 + math.sqrt(5)) / 2,
        "degrees_per_radian": 180.0 / math.pi,
        "radians_per_degree": math.pi / 180.0,
        "gradians_per_radian": 200.0 / math.pi,
        "radians_per_gradian": math.pi / 200.0,
    }


async def print_trigonometry_summary():
    """Print a summary of all trigonometric functions by domain (async)."""
    print("📐 Trigonometric Functions Library (Async Native)")
    print("=" * 50)

    print("📊 Available Domains:")
    print("📐 basic_functions - sin, cos, tan, csc, sec, cot (radians & degrees)")
    print("🔄 inverse_functions - asin, acos, atan, atan2, etc.")
    print("📈 hyperbolic - sinh, cosh, tanh, csch, sech, coth")
    print("🔄 inverse_hyperbolic - asinh, acosh, atanh, etc.")
    print("🔄 angle_conversion - deg/rad/grad conversions, normalization")
    print("⚖️  identities - Pythagorean, sum/difference, double/half angle")
    print("🌊 wave_analysis - amplitude, frequency, phase shift, harmonics")
    print("🎯 applications - navigation, oscillations, real-world problems")
    print()

    print("📐 Basic Trigonometry Capabilities:")
    print("   • Primary functions: sin, cos, tan with domain validation")
    print("   • Reciprocal functions: csc, sec, cot with singularity handling")
    print("   • Degree variants: sin_degrees, cos_degrees, tan_degrees")
    print("   • High precision: optimized for both small and large angles")

    print("\n🔄 Inverse Functions:")
    print("   • Standard inverse: asin, acos, atan with range restrictions")
    print("   • Two-argument atan2: full quadrant determination")
    print("   • Reciprocal inverse: acsc, asec, acot")
    print("   • Degree outputs: asin_degrees, acos_degrees, etc.")

    print("\n📈 Hyperbolic Functions:")
    print("   • Basic hyperbolic: sinh, cosh, tanh")
    print("   • Reciprocal hyperbolic: csch, sech, coth")
    print("   • Inverse hyperbolic: asinh, acosh, atanh, etc.")
    print("   • Applications: exponential growth, catenary curves")

    print("\n🔄 Angle Conversions:")
    print("   • Standard conversions: degrees ↔ radians ↔ gradians")
    print("   • Angle normalization: [0, 2π), [-π, π), [0°, 360°)")
    print("   • Angle difference: shortest angular distance")
    print("   • Precision handling: minimizes floating-point errors")

    print("\n⚖️  Identity Verification:")
    print("   • Pythagorean identities: sin²θ + cos²θ = 1, etc.")
    print("   • Sum/difference formulas: sin(a±b), cos(a±b), tan(a±b)")
    print("   • Double angle: sin(2θ), cos(2θ), tan(2θ)")
    print("   • Half angle: sin(θ/2), cos(θ/2), tan(θ/2)")
    print("   • Identity verification: numerical validation within tolerance")

    print("\n🌊 Wave Analysis:")
    print("   • Amplitude extraction: from A*sin(ωt + φ) + B")
    print("   • Frequency analysis: period, angular frequency, Hz")
    print("   • Phase shift: horizontal displacement, time delay")
    print("   • Harmonic analysis: fundamental + overtones")
    print("   • Fourier basics: coefficient calculation for simple waves")

    print("\n🎯 Real-World Applications:")
    print("   • Navigation: haversine distance, bearing calculation")
    print("   • Triangulation: position from multiple reference points")
    print("   • Oscillations: pendulum, spring-mass systems")
    print("   • Signal processing: wave interference, modulation")


def get_function_recommendations(operation_type: str) -> List[str]:
    """Get function recommendations based on operation type."""
    recommendations = {
        # Basic trigonometry
        "basic": ["sin", "cos", "tan", "sin_degrees", "cos_degrees", "tan_degrees"],
        "reciprocal": ["csc", "sec", "cot"],
        "inverse": ["asin", "acos", "atan", "atan2", "asin_degrees", "acos_degrees"],
        # Hyperbolic functions
        "hyperbolic": ["sinh", "cosh", "tanh", "csch", "sech", "coth"],
        "inverse_hyperbolic": ["asinh", "acosh", "atanh"],
        # Conversions and utilities
        "conversion": ["degrees_to_radians", "radians_to_degrees", "normalize_angle"],
        "angles": ["angle_difference", "gradians_to_radians", "degrees_to_gradians"],
        # Advanced applications
        "identities": [
            "pythagorean_identity",
            "double_angle_formulas",
            "verify_identity",
        ],
        "waves": [
            "amplitude_from_coefficients",
            "frequency_from_period",
            "phase_shift_analysis",
        ],
        "navigation": ["distance_haversine", "bearing_calculation", "triangulation"],
        "physics": ["oscillation_analysis", "pendulum_period", "spring_oscillation"],
    }

    return recommendations.get(operation_type.lower(), [])


def validate_trig_domain(domain: str) -> bool:
    """Validate if a trigonometric domain exists."""
    valid_domains = {
        "basic_functions",
        "inverse_functions",
        "hyperbolic",
        "inverse_hyperbolic",
        "angle_conversion",
        "identities",
        "wave_analysis",
        "applications",
    }
    return domain.lower() in valid_domains


async def get_async_performance_stats() -> Dict[str, Any]:
    """Get performance statistics for async trigonometric functions."""
    if not _mcp_decorator_available:
        return {
            "total_async_functions": 0,
            "cached_functions": 0,
            "streaming_functions": 0,
            "high_performance_functions": 0,
            "domains_implemented": 8,
        }

    trig_funcs = await get_trigonometry_functions()

    stats = {
        "total_async_functions": 0,
        "cached_functions": 0,
        "streaming_functions": 0,
        "high_performance_functions": 0,
        "domains_implemented": 0,
    }

    for domain_name, functions in trig_funcs.items():
        if functions:  # Domain has functions
            stats["domains_implemented"] += 1

        for func_name, spec in functions.items():
            stats["total_async_functions"] += 1

            if spec.cache_strategy.value != "none":
                stats["cached_functions"] += 1

            if spec.supports_streaming:
                stats["streaming_functions"] += 1

            if spec.estimated_cpu_usage.value == "high":
                stats["high_performance_functions"] += 1

    return stats


def trigonometry_quick_reference() -> str:
    """Generate a quick reference guide for trigonometric functions."""
    reference = """
📐 Trigonometric Functions Quick Reference (Async Native)

🚀 BASIC TRIGONOMETRIC FUNCTIONS (use await):
   
📐 PRIMARY FUNCTIONS:
   await sin(angle), await cos(angle), await tan(angle)
   await sin_degrees(angle), await cos_degrees(angle), await tan_degrees(angle)
   
📐 RECIPROCAL FUNCTIONS:
   await csc(angle), await sec(angle), await cot(angle)
   
🔄 INVERSE FUNCTIONS:
   await asin(value), await acos(value), await atan(value), await atan2(y, x)
   await asin_degrees(value), await acos_degrees(value), await atan_degrees(value)
   await acsc(value), await asec(value), await acot(value)

📈 HYPERBOLIC FUNCTIONS (use await):
   
📈 BASIC HYPERBOLIC:
   await sinh(x), await cosh(x), await tanh(x)
   await csch(x), await sech(x), await coth(x)
   
🔄 INVERSE HYPERBOLIC:
   await asinh(x), await acosh(x), await atanh(x)
   await acsch(x), await asech(x), await acoth(x)

🔄 ANGLE CONVERSIONS (use await):
   
🔄 BASIC CONVERSIONS:
   await degrees_to_radians(degrees), await radians_to_degrees(radians)
   await gradians_to_radians(grad), await radians_to_gradians(rad)
   await degrees_to_gradians(deg), await gradians_to_degrees(grad)
   
🔄 ANGLE UTILITIES:
   await normalize_angle(angle, unit='radians'), await angle_difference(a1, a2)

⚖️ IDENTITY VERIFICATION (use await):
   
⚖️ BASIC IDENTITIES:
   await pythagorean_identity(angle), await sum_difference_formulas(a, b, operation)
   await double_angle_formulas(angle, function), await half_angle_formulas(angle, function)
   await verify_identity(expression1, expression2, angle), await simplify_trig_expression(expr)

🌊 WAVE ANALYSIS (use await):
   
🌊 WAVE PROPERTIES:
   await amplitude_from_coefficients(a, b), await frequency_from_period(period)
   await phase_shift_analysis(coefficients), await wave_equation(t, amplitude, freq, phase)
   await harmonic_analysis(signal), await fourier_coefficients_basic(function, n_terms)

🎯 APPLICATIONS (use await):
   
🗺️ NAVIGATION:
   await distance_haversine(lat1, lon1, lat2, lon2), await bearing_calculation(lat1, lon1, lat2, lon2)
   await triangulation(point1, point2, distance1, distance2)
   
⚡ PHYSICS:
   await oscillation_analysis(displacement_function), await pendulum_period(length, g)
   await spring_oscillation(mass, spring_constant, amplitude, phase)

🎯 IMPORT PATTERNS:
   # Basic trigonometry
   from chuk_mcp_math.trigonometry.basic_functions import sin, cos, tan
   from chuk_mcp_math.trigonometry.inverse_functions import asin, atan2
   
   # Hyperbolic functions
   from chuk_mcp_math.trigonometry.hyperbolic import sinh, cosh, tanh
   from chuk_mcp_math.trigonometry.inverse_hyperbolic import asinh
   
   # Conversions and applications
   from chuk_mcp_math.trigonometry.angle_conversion import degrees_to_radians
   from chuk_mcp_math.trigonometry.applications import distance_haversine
   
   # Or use submodules
   from chuk_mcp_math import trigonometry
   result = await trigonometry.sin(math.pi/4)
   distance = await trigonometry.distance_haversine(lat1, lon1, lat2, lon2)
"""
    return reference.strip()


# Export main components
__all__ = [
    # Trigonometric domains (async native)
    "basic_functions",
    "inverse_functions",
    "hyperbolic",
    "inverse_hyperbolic",
    "angle_conversion",
    "identities",
    "wave_analysis",
    "applications",
    # Basic trigonometric functions
    "sin",
    "cos",
    "tan",
    "csc",
    "sec",
    "cot",
    "sin_degrees",
    "cos_degrees",
    "tan_degrees",
    # Inverse trigonometric functions
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
    # Hyperbolic functions
    "sinh",
    "cosh",
    "tanh",
    "csch",
    "sech",
    "coth",
    # Inverse hyperbolic functions
    "asinh",
    "acosh",
    "atanh",
    "acsch",
    "asech",
    "acoth",
    # Angle conversion functions
    "degrees_to_radians",
    "radians_to_degrees",
    "gradians_to_radians",
    "radians_to_gradians",
    "degrees_to_gradians",
    "gradians_to_degrees",
    "normalize_angle",
    "angle_difference",
    # Identity verification
    "pythagorean_identity",
    "sum_difference_formulas",
    "double_angle_formulas",
    "half_angle_formulas",
    "verify_identity",
    "simplify_trig_expression",
    # Wave analysis
    "amplitude_from_coefficients",
    "frequency_from_period",
    "phase_shift_analysis",
    "wave_equation",
    "harmonic_analysis",
    "fourier_coefficients_basic",
    # Applications
    "distance_haversine",
    "bearing_calculation",
    "triangulation",
    "oscillation_analysis",
    "pendulum_period",
    "spring_oscillation",
    # Utility functions
    "get_trigonometry_functions",
    "get_trig_constants",
    "print_trigonometry_summary",
    "get_function_recommendations",
    "validate_trig_domain",
    "get_async_performance_stats",
    "trigonometry_quick_reference",
]

if __name__ == "__main__":
    import asyncio

    async def main():
        await print_trigonometry_summary()
        print("\n" + "=" * 50)
        print(trigonometry_quick_reference())

        # Test basic functions if available
        print("\n🧪 Testing Basic Trigonometric Functions:")

        try:
            from .basic_functions import sin, cos, tan

            angle = math.pi / 4  # 45 degrees
            sin_result = await sin(angle)
            cos_result = await cos(angle)
            tan_result = await tan(angle)
            print(f"✅ sin(π/4) = {sin_result:.6f}")
            print(f"✅ cos(π/4) = {cos_result:.6f}")
            print(f"✅ tan(π/4) = {tan_result:.6f}")
        except Exception as e:
            print(f"⚠️  Basic trigonometric test failed: {e}")

        try:
            from .angle_conversion import degrees_to_radians, radians_to_degrees

            degrees = 45
            radians = await degrees_to_radians(degrees)
            back_to_degrees = await radians_to_degrees(radians)
            print(f"✅ 45° = {radians:.6f} rad = {back_to_degrees:.6f}°")
        except Exception as e:
            print(f"⚠️  Angle conversion test failed: {e}")

    asyncio.run(main())
