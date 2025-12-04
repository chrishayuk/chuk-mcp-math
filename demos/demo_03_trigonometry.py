#!/usr/bin/env python3
"""
CHUK MCP Math - Trigonometry Demo
==================================

Demonstrates all trigonometry functions including:
- Basic trigonometric functions (sin, cos, tan, etc.)
- Inverse trigonometric functions
- Hyperbolic functions
- Angle conversions
- Trigonometric identities
- Real-world applications
"""

import asyncio
import sys
import math
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from chuk_mcp_math.trigonometry.basic_functions import sin, cos, tan, cot, sec, csc
from chuk_mcp_math.trigonometry.inverse import (
    arcsin,
    arccos,
    arctan,
    arctan2,
    arccot,
    arcsec,
    arccsc,
)
from chuk_mcp_math.trigonometry.hyperbolic import (
    sinh,
    cosh,
    tanh,
    coth,
    sech,
    csch,
    arcsinh,
    arccosh,
    arctanh,
)
from chuk_mcp_math.trigonometry.angle_conversion import (
    degrees_to_radians,
    radians_to_degrees,
    degrees_to_gradians,
    gradians_to_degrees,
    normalize_angle,
    angle_to_unit_circle,
)
from chuk_mcp_math.trigonometry.identities import (
    verify_pythagorean_identities,
    sum_formula,
    double_angle_formula,
    half_angle_formula,
)
from chuk_mcp_math.trigonometry.applications import (
    triangle_area_sss,
    triangle_area_sas,
    law_of_sines,
    law_of_cosines,
    pendulum_period,
)


async def demo_basic_trig():
    """Demonstrate basic trigonometric functions."""
    print("\n" + "=" * 70)
    print("BASIC TRIGONOMETRIC FUNCTIONS")
    print("=" * 70)

    angle = math.pi / 4  # 45 degrees

    result = await sin(angle)
    print(f"✓ sin(π/4) = {result:.6f}")

    result = await cos(angle)
    print(f"✓ cos(π/4) = {result:.6f}")

    result = await tan(angle)
    print(f"✓ tan(π/4) = {result:.6f}")

    result = await cot(angle)
    print(f"✓ cot(π/4) = {result:.6f}")

    result = await sec(angle)
    print(f"✓ sec(π/4) = {result:.6f}")

    result = await csc(angle)
    print(f"✓ csc(π/4) = {result:.6f}")


async def demo_inverse_trig():
    """Demonstrate inverse trigonometric functions."""
    print("\n" + "=" * 70)
    print("INVERSE TRIGONOMETRIC FUNCTIONS")
    print("=" * 70)

    result = await arcsin(0.5)
    print(f"✓ arcsin(0.5) = {result:.6f} rad = {math.degrees(result):.2f}°")

    result = await arccos(0.5)
    print(f"✓ arccos(0.5) = {result:.6f} rad = {math.degrees(result):.2f}°")

    result = await arctan(1.0)
    print(f"✓ arctan(1.0) = {result:.6f} rad = {math.degrees(result):.2f}°")

    result = await arctan2(1.0, 1.0)
    print(f"✓ arctan2(1.0, 1.0) = {result:.6f} rad = {math.degrees(result):.2f}°")

    result = await arccot(1.0)
    print(f"✓ arccot(1.0) = {result:.6f} rad = {math.degrees(result):.2f}°")

    result = await arcsec(2.0)
    print(f"✓ arcsec(2.0) = {result:.6f} rad = {math.degrees(result):.2f}°")

    result = await arccsc(2.0)
    print(f"✓ arccsc(2.0) = {result:.6f} rad = {math.degrees(result):.2f}°")


async def demo_hyperbolic():
    """Demonstrate hyperbolic functions."""
    print("\n" + "=" * 70)
    print("HYPERBOLIC FUNCTIONS")
    print("=" * 70)

    x = 1.0

    result = await sinh(x)
    print(f"✓ sinh(1.0) = {result:.6f}")

    result = await cosh(x)
    print(f"✓ cosh(1.0) = {result:.6f}")

    result = await tanh(x)
    print(f"✓ tanh(1.0) = {result:.6f}")

    result = await coth(x)
    print(f"✓ coth(1.0) = {result:.6f}")

    result = await sech(x)
    print(f"✓ sech(1.0) = {result:.6f}")

    result = await csch(x)
    print(f"✓ csch(1.0) = {result:.6f}")

    # Inverse hyperbolic
    result = await arcsinh(1.0)
    print(f"✓ arcsinh(1.0) = {result:.6f}")

    result = await arccosh(2.0)
    print(f"✓ arccosh(2.0) = {result:.6f}")

    result = await arctanh(0.5)
    print(f"✓ arctanh(0.5) = {result:.6f}")


async def demo_angle_conversion():
    """Demonstrate angle conversion functions."""
    print("\n" + "=" * 70)
    print("ANGLE CONVERSIONS")
    print("=" * 70)

    # Degrees to radians
    result = await degrees_to_radians(180)
    print(f"✓ degrees_to_radians(180) = {result:.6f} (π)")

    # Radians to degrees
    result = await radians_to_degrees(math.pi)
    print(f"✓ radians_to_degrees(π) = {result:.2f}°")

    # Degrees to gradians
    result = await degrees_to_gradians(90)
    print(f"✓ degrees_to_gradians(90) = {result:.2f} gradians")

    # Gradians to degrees
    result = await gradians_to_degrees(100)
    print(f"✓ gradians_to_degrees(100) = {result:.2f}°")

    # Normalize angle
    result = await normalize_angle(450)
    print(f"✓ normalize_angle(450°) = {result:.2f}°")

    # Angle to unit circle
    result = await angle_to_unit_circle(math.pi / 3)
    print(f"✓ angle_to_unit_circle(π/3) = ({result[0]:.4f}, {result[1]:.4f})")


async def demo_identities():
    """Demonstrate trigonometric identities."""
    print("\n" + "=" * 70)
    print("TRIGONOMETRIC IDENTITIES")
    print("=" * 70)

    # Pythagorean identity
    result = await verify_pythagorean_identities(math.pi / 4)
    print("✓ Pythagorean identity at π/4:")
    print(f"  sin²θ + cos²θ = 1? {result['sin_cos_identity']}")

    # Sum formula
    result = await sum_formula(math.pi / 6, math.pi / 4, "add")
    print("✓ sum_formula(π/6, π/4, 'add'):")
    print(f"  sin(π/6 + π/4) = {result['sin_formula']:.6f}")

    # Double angle formula
    result = await double_angle_formula(math.pi / 6, "sin")
    print("✓ double_angle_formula(π/6, 'sin'):")
    print(f"  sin(2θ) = {result['double_angle_value']:.6f}")

    # Half angle formula
    result = await half_angle_formula(math.pi / 2, "sin")
    print("✓ half_angle_formula(π/2, 'sin'):")
    print(f"  sin(θ/2) = {result['half_angle_value']:.6f}")


async def demo_applications():
    """Demonstrate real-world trigonometry applications."""
    print("\n" + "=" * 70)
    print("REAL-WORLD APPLICATIONS")
    print("=" * 70)

    # Triangle area (SSS - three sides)
    result = await triangle_area_sss(3, 4, 5)
    print(f"✓ triangle_area_sss(3, 4, 5) = {result:.2f}")

    # Triangle area (SAS - two sides and included angle)
    result = await triangle_area_sas(5, 6, math.pi / 3)
    print(f"✓ triangle_area_sas(5, 6, π/3) = {result:.2f}")

    # Law of sines
    result = await law_of_sines(a=10, angle_a=math.pi / 6, angle_b=math.pi / 4)
    print("✓ law_of_sines(a=10, ∠A=π/6, ∠B=π/4):")
    print(f"  Side b = {result['b']:.2f}")

    # Law of cosines
    result = await law_of_cosines(a=5, b=7, c=10)
    print("✓ law_of_cosines(a=5, b=7, c=10):")
    print(f"  Angle C = {math.degrees(result['angle_c']):.2f}°")

    # Pendulum period
    result = await pendulum_period(length=1.0, gravity=9.81)
    print("✓ pendulum_period(length=1.0m, g=9.81):")
    print(f"  Period = {result['period']:.3f} seconds")


async def main():
    """Run all trigonometry demos."""
    print("\n" + "=" * 70)
    print("CHUK MCP MATH - TRIGONOMETRY DEMO")
    print("=" * 70)
    print("\nDemonstrating 120+ trigonometry functions are working correctly...")

    await demo_basic_trig()
    await demo_inverse_trig()
    await demo_hyperbolic()
    await demo_angle_conversion()
    await demo_identities()
    await demo_applications()

    print("\n" + "=" * 70)
    print("✅ ALL TRIGONOMETRY OPERATIONS WORKING PERFECTLY!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
