#!/usr/bin/env python3
# chuk_mcp_math/trigonometry/hyperbolic.py
"""
Hyperbolic Functions - Async Native

Comprehensive hyperbolic trigonometric functions with high precision and optimizations.
Hyperbolic functions are analogs of trigonometric functions but for hyperbolic geometry.

Functions:
- sinh, cosh, tanh: Primary hyperbolic functions
- csch, sech, coth: Reciprocal hyperbolic functions
- Optimized for numerical stability and performance
- Applications in exponential growth, catenary curves, relativity
"""

import math
import asyncio
from typing import Union
from chuk_mcp_math.mcp_decorator import mcp_function

# ============================================================================
# PRIMARY HYPERBOLIC FUNCTIONS
# ============================================================================


@mcp_function(
    description="Calculate hyperbolic sine with high precision and overflow protection.",
    namespace="trigonometry",
    category="hyperbolic",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    cache_ttl_seconds=3600,
    examples=[
        {"input": {"x": 0}, "output": 0.0, "description": "sinh(0) = 0"},
        {
            "input": {"x": 1},
            "output": 1.1752011936438014,
            "description": "sinh(1) ≈ 1.175",
        },
        {
            "input": {"x": -1},
            "output": -1.1752011936438014,
            "description": "sinh(-1) ≈ -1.175 (odd function)",
        },
        {
            "input": {"x": 0.5},
            "output": 0.5210953054937474,
            "description": "sinh(0.5) ≈ 0.521",
        },
    ],
)
async def sinh(x: Union[int, float]) -> float:
    """
    Calculate the hyperbolic sine of x.

    sinh(x) = (e^x - e^(-x)) / 2

    Hyperbolic sine is an odd function: sinh(-x) = -sinh(x)
    Used in catenary curves, special relativity, and exponential growth models.

    Args:
        x: Input value (any real number)

    Returns:
        Hyperbolic sine of x

    Examples:
        await sinh(0) → 0.0                  # sinh(0) = 0
        await sinh(1) → 1.175...             # sinh(1) ≈ 1.175
        await sinh(-1) → -1.175...           # sinh(-1) = -sinh(1)
        await sinh(ln(2)) → 0.75             # sinh(ln(2)) = 3/4
    """
    # Handle special cases
    if x == 0:
        return 0.0

    # For very small x, use series expansion to avoid precision loss
    if abs(x) < 1e-10:
        # sinh(x) ≈ x + x³/6 + x⁵/120 + ... for small x
        x_squared = x * x
        return x * (1 + x_squared / 6 * (1 + x_squared / 20 * (1 + x_squared / 42)))

    # For large |x|, use optimized computation to avoid overflow
    if abs(x) > 700:  # Threshold to prevent overflow
        # For large positive x: sinh(x) ≈ e^x / 2
        # For large negative x: sinh(x) ≈ -e^|x| / 2
        if x > 0:
            # Use log-exp trick to handle very large values
            return math.exp(x - math.log(2))
        else:
            return -math.exp(-x - math.log(2))

    # Use the standard definition for moderate values
    return math.sinh(x)


@mcp_function(
    description="Calculate hyperbolic cosine with numerical stability.",
    namespace="trigonometry",
    category="hyperbolic",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    cache_ttl_seconds=3600,
    examples=[
        {"input": {"x": 0}, "output": 1.0, "description": "cosh(0) = 1"},
        {
            "input": {"x": 1},
            "output": 1.5430806348152437,
            "description": "cosh(1) ≈ 1.543",
        },
        {
            "input": {"x": -1},
            "output": 1.5430806348152437,
            "description": "cosh(-1) = cosh(1) (even function)",
        },
        {
            "input": {"x": 0.5},
            "output": 1.1276259652063807,
            "description": "cosh(0.5) ≈ 1.128",
        },
    ],
)
async def cosh(x: Union[int, float]) -> float:
    """
    Calculate the hyperbolic cosine of x.

    cosh(x) = (e^x + e^(-x)) / 2

    Hyperbolic cosine is an even function: cosh(-x) = cosh(x)
    Always ≥ 1, describes the shape of a hanging chain (catenary).

    Args:
        x: Input value (any real number)

    Returns:
        Hyperbolic cosine of x, always ≥ 1

    Examples:
        await cosh(0) → 1.0                  # cosh(0) = 1
        await cosh(1) → 1.543...             # cosh(1) ≈ 1.543
        await cosh(-1) → 1.543...            # cosh(-1) = cosh(1)
        await cosh(ln(2)) → 1.25             # cosh(ln(2)) = 5/4
    """
    # Handle special cases
    if x == 0:
        return 1.0

    # For very small x, use series expansion
    if abs(x) < 1e-10:
        # cosh(x) ≈ 1 + x²/2 + x⁴/24 + ... for small x
        x_squared = x * x
        return 1 + x_squared / 2 * (1 + x_squared / 12 * (1 + x_squared / 30))

    # For large |x|, use optimized computation
    if abs(x) > 700:
        # For large |x|: cosh(x) ≈ e^|x| / 2
        return math.exp(abs(x) - math.log(2))

    # Use the standard definition for moderate values
    return math.cosh(x)


@mcp_function(
    description="Calculate hyperbolic tangent with saturation handling.",
    namespace="trigonometry",
    category="hyperbolic",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    cache_ttl_seconds=3600,
    examples=[
        {"input": {"x": 0}, "output": 0.0, "description": "tanh(0) = 0"},
        {
            "input": {"x": 1},
            "output": 0.7615941559557649,
            "description": "tanh(1) ≈ 0.762",
        },
        {
            "input": {"x": -1},
            "output": -0.7615941559557649,
            "description": "tanh(-1) = -tanh(1) (odd function)",
        },
        {
            "input": {"x": 10},
            "output": 0.9999999958776927,
            "description": "tanh(10) ≈ 1 (saturates)",
        },
    ],
)
async def tanh(x: Union[int, float]) -> float:
    """
    Calculate the hyperbolic tangent of x.

    tanh(x) = sinh(x) / cosh(x) = (e^x - e^(-x)) / (e^x + e^(-x))

    Hyperbolic tangent is an odd function: tanh(-x) = -tanh(x)
    Range: (-1, 1), commonly used as activation function in neural networks.

    Args:
        x: Input value (any real number)

    Returns:
        Hyperbolic tangent of x, range (-1, 1)

    Examples:
        await tanh(0) → 0.0                  # tanh(0) = 0
        await tanh(1) → 0.762...             # tanh(1) ≈ 0.762
        await tanh(-1) → -0.762...           # tanh(-1) = -tanh(1)
        await tanh(∞) → 1.0                  # tanh approaches ±1
    """
    # Handle special cases
    if x == 0:
        return 0.0

    # For very small x, use series expansion
    if abs(x) < 1e-10:
        # tanh(x) ≈ x - x³/3 + 2x⁵/15 - ... for small x
        x_squared = x * x
        return x * (
            1 - x_squared / 3 * (1 - 2 * x_squared / 15 * (1 - 17 * x_squared / 315))
        )

    # For large |x|, tanh saturates to ±1
    if abs(x) > 20:
        return 1.0 if x > 0 else -1.0

    # Use optimized formula to avoid overflow: tanh(x) = (e^(2x) - 1) / (e^(2x) + 1)
    if abs(x) > 1:
        exp_2x = math.exp(2 * x)
        return (exp_2x - 1) / (exp_2x + 1)

    # Use standard definition for small to moderate values
    return math.tanh(x)


# ============================================================================
# RECIPROCAL HYPERBOLIC FUNCTIONS
# ============================================================================


@mcp_function(
    description="Calculate hyperbolic cosecant (1/sinh) with singularity handling.",
    namespace="trigonometry",
    category="hyperbolic",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    cache_ttl_seconds=3600,
    examples=[
        {
            "input": {"x": 1},
            "output": 0.8509181282393216,
            "description": "csch(1) ≈ 0.851",
        },
        {
            "input": {"x": -1},
            "output": -0.8509181282393216,
            "description": "csch(-1) = -csch(1) (odd function)",
        },
        {
            "input": {"x": 0.5},
            "output": 1.9190347513349437,
            "description": "csch(0.5) ≈ 1.919",
        },
    ],
)
async def csch(x: Union[int, float]) -> float:
    """
    Calculate the hyperbolic cosecant of x.

    csch(x) = 1 / sinh(x)

    Undefined at x = 0. Odd function: csch(-x) = -csch(x)

    Args:
        x: Input value, must not be zero

    Returns:
        Hyperbolic cosecant of x

    Raises:
        ValueError: If x is zero

    Examples:
        await csch(1) → 0.851...             # csch(1) ≈ 0.851
        await csch(-1) → -0.851...           # csch(-1) = -csch(1)
        await csch(0.1) → 9.983...           # Large for small x
    """
    if x == 0:
        raise ValueError("csch undefined at x = 0 (sinh(0) = 0)")

    sinh_x = await sinh(x)

    # Additional check for very small sinh values
    if abs(sinh_x) < 1e-15:
        raise ValueError(f"csch undefined: sinh({x}) is effectively zero")

    return 1.0 / sinh_x


@mcp_function(
    description="Calculate hyperbolic secant (1/cosh).",
    namespace="trigonometry",
    category="hyperbolic",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    cache_ttl_seconds=3600,
    examples=[
        {"input": {"x": 0}, "output": 1.0, "description": "sech(0) = 1"},
        {
            "input": {"x": 1},
            "output": 0.6480542736638855,
            "description": "sech(1) ≈ 0.648",
        },
        {
            "input": {"x": -1},
            "output": 0.6480542736638855,
            "description": "sech(-1) = sech(1) (even function)",
        },
        {
            "input": {"x": 2},
            "output": 0.26580222883407969,
            "description": "sech(2) ≈ 0.266",
        },
    ],
)
async def sech(x: Union[int, float]) -> float:
    """
    Calculate the hyperbolic secant of x.

    sech(x) = 1 / cosh(x)

    Even function: sech(-x) = sech(x)
    Range: (0, 1], maximum value of 1 at x = 0.
    Used in soliton theory and as activation function.

    Args:
        x: Input value (any real number)

    Returns:
        Hyperbolic secant of x, range (0, 1]

    Examples:
        await sech(0) → 1.0                  # sech(0) = 1 (maximum)
        await sech(1) → 0.648...             # sech(1) ≈ 0.648
        await sech(-1) → 0.648...            # sech(-1) = sech(1)
        await sech(∞) → 0.0                  # sech approaches 0
    """
    cosh_x = await cosh(x)
    return 1.0 / cosh_x


@mcp_function(
    description="Calculate hyperbolic cotangent (1/tanh) with singularity handling.",
    namespace="trigonometry",
    category="hyperbolic",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    cache_ttl_seconds=3600,
    examples=[
        {
            "input": {"x": 1},
            "output": 1.3130352854993312,
            "description": "coth(1) ≈ 1.313",
        },
        {
            "input": {"x": -1},
            "output": -1.3130352854993312,
            "description": "coth(-1) = -coth(1) (odd function)",
        },
        {
            "input": {"x": 0.1},
            "output": 10.033331119703451,
            "description": "coth(0.1) ≈ 10.033 (large for small x)",
        },
    ],
)
async def coth(x: Union[int, float]) -> float:
    """
    Calculate the hyperbolic cotangent of x.

    coth(x) = 1 / tanh(x) = cosh(x) / sinh(x)

    Undefined at x = 0. Odd function: coth(-x) = -coth(x)
    Range: (-∞, -1) ∪ (1, ∞)

    Args:
        x: Input value, must not be zero

    Returns:
        Hyperbolic cotangent of x

    Raises:
        ValueError: If x is zero

    Examples:
        await coth(1) → 1.313...             # coth(1) ≈ 1.313
        await coth(-1) → -1.313...           # coth(-1) = -coth(1)
        await coth(0.1) → 10.033...          # Large for small x
    """
    if x == 0:
        raise ValueError("coth undefined at x = 0 (tanh(0) = 0)")

    tanh_x = await tanh(x)

    # Additional check for very small tanh values
    if abs(tanh_x) < 1e-15:
        raise ValueError(f"coth undefined: tanh({x}) is effectively zero")

    return 1.0 / tanh_x


# ============================================================================
# HYPERBOLIC UTILITY FUNCTIONS
# ============================================================================


@mcp_function(
    description="Calculate all hyperbolic functions for a given x simultaneously.",
    namespace="trigonometry",
    category="hyperbolic",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    cache_ttl_seconds=3600,
    examples=[
        {
            "input": {"x": 1},
            "output": {
                "sinh": 1.1752011936438014,
                "cosh": 1.5430806348152437,
                "tanh": 0.7615941559557649,
                "csch": 0.8509181282393216,
                "sech": 0.6480542736638855,
                "coth": 1.3130352854993312,
            },
            "description": "All hyperbolic functions at x=1",
        },
        {
            "input": {"x": 0},
            "output": {
                "sinh": 0.0,
                "cosh": 1.0,
                "tanh": 0.0,
                "csch": None,
                "sech": 1.0,
                "coth": None,
            },
            "description": "Functions at x=0 (some undefined)",
        },
    ],
)
async def hyperbolic_functions(x: Union[int, float]) -> dict:
    """
    Calculate all hyperbolic functions for a given value.

    Efficiently computes all six hyperbolic functions simultaneously.
    Handles singularities appropriately.

    Args:
        x: Input value

    Returns:
        Dictionary containing all hyperbolic function values:
        - sinh, cosh, tanh: Primary functions
        - csch, sech, coth: Reciprocal functions (None if undefined)

    Examples:
        await hyperbolic_functions(1) → {"sinh": 1.175, "cosh": 1.543, ...}
        await hyperbolic_functions(0) → {"sinh": 0, "cosh": 1, "csch": None, ...}
    """
    # Calculate primary functions
    sinh_x = await sinh(x)
    cosh_x = await cosh(x)
    tanh_x = await tanh(x)

    # Calculate reciprocal functions, handling singularities
    csch_x = None
    coth_x = None
    if x != 0 and abs(sinh_x) > 1e-15:
        csch_x = 1.0 / sinh_x
        coth_x = 1.0 / tanh_x

    sech_x = 1.0 / cosh_x  # Always defined since cosh(x) ≥ 1

    return {
        "sinh": sinh_x,
        "cosh": cosh_x,
        "tanh": tanh_x,
        "csch": csch_x,
        "sech": sech_x,
        "coth": coth_x,
    }


@mcp_function(
    description="Verify hyperbolic identity: cosh²(x) - sinh²(x) = 1.",
    namespace="trigonometry",
    category="hyperbolic",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    examples=[
        {
            "input": {"x": 1},
            "output": {"identity_holds": True, "calculated_value": 1.0, "error": 0.0},
            "description": "Identity verification at x=1",
        },
        {
            "input": {"x": 5},
            "output": {
                "identity_holds": True,
                "calculated_value": 1.0000000000000009,
                "error": 8.881784197001252e-16,
            },
            "description": "Identity with small numerical error",
        },
    ],
)
async def verify_hyperbolic_identity(
    x: Union[int, float], tolerance: float = 1e-12
) -> dict:
    """
    Verify the fundamental hyperbolic identity: cosh²(x) - sinh²(x) = 1.

    This is the hyperbolic analog of the Pythagorean identity.

    Args:
        x: Input value to test
        tolerance: Tolerance for numerical verification

    Returns:
        Dictionary with verification results:
        - identity_holds: Boolean indicating if identity holds within tolerance
        - calculated_value: Actual value of cosh²(x) - sinh²(x)
        - error: Absolute error from expected value of 1

    Examples:
        await verify_hyperbolic_identity(1) → {"identity_holds": True, ...}
        await verify_hyperbolic_identity(10) → {"identity_holds": True, ...}
    """
    sinh_x = await sinh(x)
    cosh_x = await cosh(x)

    calculated_value = cosh_x * cosh_x - sinh_x * sinh_x
    error = abs(calculated_value - 1.0)
    identity_holds = error <= tolerance

    return {
        "identity_holds": identity_holds,
        "calculated_value": calculated_value,
        "error": error,
        "tolerance": tolerance,
    }


@mcp_function(
    description="Calculate catenary curve parameters and properties.",
    namespace="trigonometry",
    category="hyperbolic",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    examples=[
        {
            "input": {"a": 1, "x": 0},
            "output": {
                "y": 1.0,
                "slope": 0.0,
                "arc_length": 0.0,
                "curve_type": "catenary",
            },
            "description": "Catenary at lowest point",
        },
        {
            "input": {"a": 2, "x": 1},
            "output": {
                "y": 2.5430806348152437,
                "slope": 1.1752011936438014,
                "arc_length": 2.3504023872876028,
                "curve_type": "catenary",
            },
            "description": "Catenary curve properties",
        },
    ],
)
async def catenary_curve(a: Union[int, float], x: Union[int, float]) -> dict:
    """
    Calculate properties of a catenary curve.

    A catenary is the curve formed by a hanging chain or cable.
    Equation: y = a * cosh(x/a)

    Args:
        a: Catenary parameter (related to tension and weight)
        x: Horizontal position

    Returns:
        Dictionary with catenary properties:
        - y: Height at position x
        - slope: Slope (dy/dx) at position x
        - arc_length: Arc length from center to position x
        - curve_type: "catenary"

    Examples:
        await catenary_curve(1, 0) → {"y": 1, "slope": 0, ...}  # Lowest point
        await catenary_curve(2, 1) → {"y": 2.543, "slope": 1.175, ...}
    """
    if a <= 0:
        raise ValueError("Catenary parameter 'a' must be positive")

    # Height: y = a * cosh(x/a)
    x_over_a = x / a
    cosh_val = await cosh(x_over_a)
    sinh_val = await sinh(x_over_a)

    y = a * cosh_val

    # Slope: dy/dx = sinh(x/a)
    slope = sinh_val

    # Arc length from center (x=0) to position x: s = a * sinh(x/a)
    arc_length = a * sinh_val

    return {
        "y": y,
        "slope": slope,
        "arc_length": arc_length,
        "curve_type": "catenary",
        "parameter_a": a,
        "position_x": x,
    }


# Export all functions
__all__ = [
    # Primary hyperbolic functions
    "sinh",
    "cosh",
    "tanh",
    # Reciprocal hyperbolic functions
    "csch",
    "sech",
    "coth",
    # Utility functions
    "hyperbolic_functions",
    "verify_hyperbolic_identity",
    "catenary_curve",
]

if __name__ == "__main__":
    import asyncio

    async def test_hyperbolic_functions():
        """Test hyperbolic functions."""
        print("📈 Hyperbolic Functions Test")
        print("=" * 30)

        # Test primary functions
        print("Primary Hyperbolic Functions:")
        test_values = [0, 0.5, 1, 2, -1, -0.5]

        for x in test_values:
            sinh_x = await sinh(x)
            cosh_x = await cosh(x)
            tanh_x = await tanh(x)
            print(
                f"  x = {x:4.1f}: sinh = {sinh_x:8.4f}, cosh = {cosh_x:8.4f}, tanh = {tanh_x:8.4f}"
            )

        print("\nReciprocal Hyperbolic Functions:")
        test_values_nonzero = [0.5, 1, 2, -1, -0.5]

        for x in test_values_nonzero:
            try:
                csch_x = await csch(x)
                sech_x = await sech(x)
                coth_x = await coth(x)
                print(
                    f"  x = {x:4.1f}: csch = {csch_x:8.4f}, sech = {sech_x:8.4f}, coth = {coth_x:8.4f}"
                )
            except ValueError as e:
                print(f"  x = {x:4.1f}: Error - {e}")

        print("\nHyperbolic Identity Verification:")
        for x in [0, 1, 2, 5, -1]:
            identity_result = await verify_hyperbolic_identity(x)
            holds = identity_result["identity_holds"]
            error = identity_result["error"]
            print(f"  cosh²({x}) - sinh²({x}) = 1: {holds} (error: {error:.2e})")

        print("\nCatenary Curve Examples:")
        catenary_cases = [(1, 0), (1, 1), (2, 1), (0.5, 0.5)]
        for a, x in catenary_cases:
            catenary = await catenary_curve(a, x)
            y = catenary["y"]
            slope = catenary["slope"]
            arc_len = catenary["arc_length"]
            print(
                f"  Catenary a={a}, x={x}: y={y:.3f}, slope={slope:.3f}, arc_length={arc_len:.3f}"
            )

        print("\nAll Hyperbolic Functions at x=1:")
        all_funcs = await hyperbolic_functions(1)
        for func_name, value in all_funcs.items():
            if value is not None:
                print(f"  {func_name}(1) = {value:.6f}")
            else:
                print(f"  {func_name}(1) = undefined")

        print("\n✅ All hyperbolic functions working!")

    asyncio.run(test_hyperbolic_functions())
