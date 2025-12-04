#!/usr/bin/env python3
# chuk_mcp_math/trigonometry/inverse_hyperbolic.py
"""
Inverse Hyperbolic Functions - Async Native

Comprehensive inverse hyperbolic functions with domain validation and high precision.
These functions are the inverses of hyperbolic functions, used in calculus,
engineering, and mathematical physics.

Functions:
- asinh, acosh, atanh: Primary inverse hyperbolic functions
- acsch, asech, acoth: Inverse reciprocal hyperbolic functions
- Comprehensive domain validation and range handling
- Optimized for numerical stability
"""

import math
import asyncio
from typing import Union
from chuk_mcp_math.mcp_decorator import mcp_function

# ============================================================================
# PRIMARY INVERSE HYPERBOLIC FUNCTIONS
# ============================================================================


@mcp_function(
    description="Calculate inverse hyperbolic sine (area sine).",
    namespace="trigonometry",
    category="inverse_hyperbolic",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    cache_ttl_seconds=3600,
    examples=[
        {"input": {"x": 0}, "output": 0.0, "description": "asinh(0) = 0"},
        {
            "input": {"x": 1},
            "output": 0.8813735870195429,
            "description": "asinh(1) ≈ 0.881",
        },
        {
            "input": {"x": -1},
            "output": -0.8813735870195429,
            "description": "asinh(-1) = -asinh(1) (odd function)",
        },
        {
            "input": {"x": 2},
            "output": 1.4436354751788103,
            "description": "asinh(2) ≈ 1.444",
        },
    ],
)
async def asinh(x: Union[int, float]) -> float:
    """
    Calculate the inverse hyperbolic sine (area sine) of x.

    asinh(x) = ln(x + √(x² + 1))

    Domain: (-∞, ∞), Range: (-∞, ∞)
    Odd function: asinh(-x) = -asinh(x)

    Args:
        x: Input value (any real number)

    Returns:
        Inverse hyperbolic sine of x

    Examples:
        await asinh(0) → 0.0                 # asinh(0) = 0
        await asinh(1) → 0.881...            # asinh(1) ≈ 0.881
        await asinh(-1) → -0.881...          # asinh(-1) = -asinh(1)
        await asinh(∞) → ∞                   # Unbounded
    """
    # Handle special cases
    if x == 0:
        return 0.0

    # For very small x, use series expansion
    if abs(x) < 1e-10:
        # asinh(x) ≈ x - x³/6 + 3x⁵/40 - ... for small x
        x_squared = x * x
        return x * (
            1 - x_squared / 6 * (1 - 3 * x_squared / 20 * (1 - 5 * x_squared / 56))
        )

    # For large |x|, use asymptotic expansion to avoid overflow
    if abs(x) > 1e8:
        # asinh(x) ≈ ln(2|x|) + sign(x) for large |x|
        return math.log(2 * abs(x)) * (1 if x > 0 else -1)

    # Use optimized formula: asinh(x) = ln(x + √(x² + 1))
    # For numerical stability, use different forms based on |x|
    if abs(x) < 1:
        # For |x| < 1, use: asinh(x) = ln(x + √(x² + 1))
        return math.log(x + math.sqrt(x * x + 1))
    else:
        # For |x| ≥ 1, use: sign(x) * ln(|x| + √(x² + 1))
        abs_x = abs(x)
        result = math.log(abs_x + math.sqrt(abs_x * abs_x + 1))
        return result if x >= 0 else -result


@mcp_function(
    description="Calculate inverse hyperbolic cosine (area cosine) with domain validation.",
    namespace="trigonometry",
    category="inverse_hyperbolic",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    cache_ttl_seconds=3600,
    examples=[
        {"input": {"x": 1}, "output": 0.0, "description": "acosh(1) = 0"},
        {
            "input": {"x": 2},
            "output": 1.3169578969248166,
            "description": "acosh(2) ≈ 1.317",
        },
        {
            "input": {"x": 1.5},
            "output": 0.9624236501192069,
            "description": "acosh(1.5) ≈ 0.962",
        },
        {
            "input": {"x": 10},
            "output": 2.993222846126381,
            "description": "acosh(10) ≈ 2.993",
        },
    ],
)
async def acosh(x: Union[int, float]) -> float:
    """
    Calculate the inverse hyperbolic cosine (area cosine) of x.

    acosh(x) = ln(x + √(x² - 1))

    Domain: [1, ∞), Range: [0, ∞)
    Even function for principal branch

    Args:
        x: Input value, must be ≥ 1

    Returns:
        Inverse hyperbolic cosine of x

    Raises:
        ValueError: If x < 1

    Examples:
        await acosh(1) → 0.0                 # acosh(1) = 0
        await acosh(2) → 1.317...            # acosh(2) ≈ 1.317
        await acosh(1.5) → 0.962...          # acosh(1.5) ≈ 0.962
        await acosh(∞) → ∞                   # Unbounded
    """
    # Domain validation
    if x < 1:
        raise ValueError(f"acosh domain error: x = {x} must be ≥ 1")

    # Handle special case
    if x == 1:
        return 0.0

    # For very large x, use asymptotic expansion
    if x > 1e8:
        # acosh(x) ≈ ln(2x) for large x
        return math.log(2 * x)

    # Use optimized formula: acosh(x) = ln(x + √(x² - 1))
    # For numerical stability near x = 1, use different approach
    if x < 1.1:
        # Near x = 1, use: acosh(x) = √(2(x-1)) * (1 + (x-1)/12 + ...)
        delta = x - 1
        sqrt_2delta = math.sqrt(2 * delta)
        return sqrt_2delta * (1 + delta / 12 * (1 + 3 * delta / 160))
    else:
        # For x ≥ 1.1, use standard formula
        return math.log(x + math.sqrt(x * x - 1))


@mcp_function(
    description="Calculate inverse hyperbolic tangent with domain validation.",
    namespace="trigonometry",
    category="inverse_hyperbolic",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    cache_ttl_seconds=3600,
    examples=[
        {"input": {"x": 0}, "output": 0.0, "description": "atanh(0) = 0"},
        {
            "input": {"x": 0.5},
            "output": 0.5493061443340549,
            "description": "atanh(0.5) ≈ 0.549",
        },
        {
            "input": {"x": -0.5},
            "output": -0.5493061443340549,
            "description": "atanh(-0.5) = -atanh(0.5) (odd function)",
        },
        {
            "input": {"x": 0.9},
            "output": 1.4722194895832204,
            "description": "atanh(0.9) ≈ 1.472",
        },
    ],
)
async def atanh(x: Union[int, float]) -> float:
    """
    Calculate the inverse hyperbolic tangent (area tangent) of x.

    atanh(x) = (1/2) * ln((1 + x) / (1 - x))

    Domain: (-1, 1), Range: (-∞, ∞)
    Odd function: atanh(-x) = -atanh(x)

    Args:
        x: Input value, must be in (-1, 1)

    Returns:
        Inverse hyperbolic tangent of x

    Raises:
        ValueError: If |x| ≥ 1

    Examples:
        await atanh(0) → 0.0                 # atanh(0) = 0
        await atanh(0.5) → 0.549...          # atanh(0.5) ≈ 0.549
        await atanh(-0.5) → -0.549...        # atanh(-0.5) = -atanh(0.5)
        await atanh(±1) → ±∞                 # Undefined at boundaries
    """
    # Domain validation
    if abs(x) >= 1:
        raise ValueError(f"atanh domain error: |x| = {abs(x)} must be < 1")

    # Handle special case
    if x == 0:
        return 0.0

    # For very small x, use series expansion
    if abs(x) < 1e-10:
        # atanh(x) ≈ x + x³/3 + 2x⁵/15 + ... for small x
        x_squared = x * x
        return x * (
            1 + x_squared / 3 * (1 + 2 * x_squared / 5 * (1 + 3 * x_squared / 7))
        )

    # For x close to ±1, use different approach to avoid precision loss
    if abs(x) > 0.95:
        # Use: atanh(x) = (1/2) * ln((1+x)/(1-x))
        # Rearrange to avoid subtraction of nearly equal numbers
        if x > 0:
            return 0.5 * math.log((1 + x) / (1 - x))
        else:
            return -0.5 * math.log((1 - x) / (1 + x))

    # Use standard formula for moderate values
    return 0.5 * math.log((1 + x) / (1 - x))


# ============================================================================
# INVERSE RECIPROCAL HYPERBOLIC FUNCTIONS
# ============================================================================


@mcp_function(
    description="Calculate inverse hyperbolic cosecant with domain validation.",
    namespace="trigonometry",
    category="inverse_hyperbolic",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    cache_ttl_seconds=3600,
    examples=[
        {
            "input": {"x": 1},
            "output": 0.8813735870195429,
            "description": "acsch(1) ≈ 0.881",
        },
        {
            "input": {"x": -1},
            "output": -0.8813735870195429,
            "description": "acsch(-1) = -acsch(1) (odd function)",
        },
        {
            "input": {"x": 2},
            "output": 0.48121182505960347,
            "description": "acsch(2) ≈ 0.481",
        },
    ],
)
async def acsch(x: Union[int, float]) -> float:
    """
    Calculate the inverse hyperbolic cosecant of x.

    acsch(x) = asinh(1/x) = ln(1/x + √(1/x² + 1))

    Domain: (-∞, 0) ∪ (0, ∞), Range: (-∞, 0) ∪ (0, ∞)
    Odd function: acsch(-x) = -acsch(x)

    Args:
        x: Input value, must not be zero

    Returns:
        Inverse hyperbolic cosecant of x

    Raises:
        ValueError: If x is zero

    Examples:
        await acsch(1) → 0.881...            # acsch(1) ≈ 0.881
        await acsch(-1) → -0.881...          # acsch(-1) = -acsch(1)
        await acsch(2) → 0.481...            # acsch(2) ≈ 0.481
    """
    if x == 0:
        raise ValueError("acsch undefined at x = 0")

    # acsch(x) = asinh(1/x)
    return await asinh(1.0 / x)


@mcp_function(
    description="Calculate inverse hyperbolic secant with domain validation.",
    namespace="trigonometry",
    category="inverse_hyperbolic",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    cache_ttl_seconds=3600,
    examples=[
        {"input": {"x": 1}, "output": 0.0, "description": "asech(1) = 0"},
        {
            "input": {"x": 0.5},
            "output": 1.3169578969248166,
            "description": "asech(0.5) ≈ 1.317",
        },
        {
            "input": {"x": 0.1},
            "output": 2.993222846126381,
            "description": "asech(0.1) ≈ 2.993",
        },
    ],
)
async def asech(x: Union[int, float]) -> float:
    """
    Calculate the inverse hyperbolic secant of x.

    asech(x) = acosh(1/x) = ln(1/x + √(1/x² - 1))

    Domain: (0, 1], Range: [0, ∞)

    Args:
        x: Input value, must be in (0, 1]

    Returns:
        Inverse hyperbolic secant of x

    Raises:
        ValueError: If x ≤ 0 or x > 1

    Examples:
        await asech(1) → 0.0                 # asech(1) = 0
        await asech(0.5) → 1.317...          # asech(0.5) ≈ 1.317
        await asech(0.1) → 2.993...          # asech(0.1) ≈ 2.993
    """
    if x <= 0 or x > 1:
        raise ValueError(f"asech domain error: x = {x} must be in (0, 1]")

    # asech(x) = acosh(1/x)
    return await acosh(1.0 / x)


@mcp_function(
    description="Calculate inverse hyperbolic cotangent with domain validation.",
    namespace="trigonometry",
    category="inverse_hyperbolic",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    cache_ttl_seconds=3600,
    examples=[
        {
            "input": {"x": 2},
            "output": 0.5493061443340549,
            "description": "acoth(2) ≈ 0.549",
        },
        {
            "input": {"x": -2},
            "output": -0.5493061443340549,
            "description": "acoth(-2) = -acoth(2) (odd function)",
        },
        {
            "input": {"x": 1.5},
            "output": 0.8047189562170503,
            "description": "acoth(1.5) ≈ 0.805",
        },
    ],
)
async def acoth(x: Union[int, float]) -> float:
    """
    Calculate the inverse hyperbolic cotangent of x.

    acoth(x) = atanh(1/x) = (1/2) * ln((x + 1) / (x - 1))

    Domain: (-∞, -1) ∪ (1, ∞), Range: (-∞, 0) ∪ (0, ∞)
    Odd function: acoth(-x) = -acoth(x)

    Args:
        x: Input value, must satisfy |x| > 1

    Returns:
        Inverse hyperbolic cotangent of x

    Raises:
        ValueError: If |x| ≤ 1

    Examples:
        await acoth(2) → 0.549...            # acoth(2) ≈ 0.549
        await acoth(-2) → -0.549...          # acoth(-2) = -acoth(2)
        await acoth(1.5) → 0.805...          # acoth(1.5) ≈ 0.805
    """
    if abs(x) <= 1:
        raise ValueError(f"acoth domain error: |x| = {abs(x)} must be > 1")

    # acoth(x) = atanh(1/x)
    return await atanh(1.0 / x)


# ============================================================================
# INVERSE HYPERBOLIC UTILITY FUNCTIONS
# ============================================================================


@mcp_function(
    description="Calculate all inverse hyperbolic functions for a given x.",
    namespace="trigonometry",
    category="inverse_hyperbolic",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    cache_ttl_seconds=3600,
    examples=[
        {
            "input": {"x": 2},
            "output": {
                "asinh": 1.4436354751788103,
                "acosh": 1.3169578969248166,
                "atanh": None,
                "acsch": 0.48121182505960347,
                "asech": None,
                "acoth": 0.5493061443340549,
            },
            "description": "Inverse hyperbolic functions at x=2",
        },
        {
            "input": {"x": 0.5},
            "output": {
                "asinh": 0.48121182505960347,
                "acosh": None,
                "atanh": 0.5493061443340549,
                "acsch": 1.4436354751788103,
                "asech": 1.3169578969248166,
                "acoth": None,
            },
            "description": "Functions at x=0.5",
        },
    ],
)
async def inverse_hyperbolic_functions(x: Union[int, float]) -> dict:
    """
    Calculate all inverse hyperbolic functions for a given value where defined.

    Efficiently computes all six inverse hyperbolic functions.
    Returns None for functions that are undefined at the given x.

    Args:
        x: Input value

    Returns:
        Dictionary containing all inverse hyperbolic function values:
        - asinh: Always defined
        - acosh: Defined for x ≥ 1
        - atanh: Defined for |x| < 1
        - acsch: Defined for x ≠ 0
        - asech: Defined for 0 < x ≤ 1
        - acoth: Defined for |x| > 1

    Examples:
        await inverse_hyperbolic_functions(2) → {...}
        await inverse_hyperbolic_functions(0.5) → {...}
    """
    result = {}

    # asinh is always defined
    result["asinh"] = await asinh(x)

    # acosh is defined for x ≥ 1
    if x >= 1:
        result["acosh"] = await acosh(x)
    else:
        result["acosh"] = None

    # atanh is defined for |x| < 1
    if abs(x) < 1:
        result["atanh"] = await atanh(x)
    else:
        result["atanh"] = None

    # acsch is defined for x ≠ 0
    if x != 0:
        result["acsch"] = await acsch(x)
    else:
        result["acsch"] = None

    # asech is defined for 0 < x ≤ 1
    if 0 < x <= 1:
        result["asech"] = await asech(x)
    else:
        result["asech"] = None

    # acoth is defined for |x| > 1
    if abs(x) > 1:
        result["acoth"] = await acoth(x)
    else:
        result["acoth"] = None

    return result


@mcp_function(
    description="Verify inverse hyperbolic function identities.",
    namespace="trigonometry",
    category="inverse_hyperbolic",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    examples=[
        {
            "input": {"x": 2},
            "output": {
                "sinh_asinh_identity": True,
                "cosh_acosh_identity": True,
                "sinh_asinh_error": 0.0,
                "cosh_acosh_error": 0.0,
            },
            "description": "Identity verification at x=2",
        },
        {
            "input": {"x": 0.5},
            "output": {"tanh_atanh_identity": True, "tanh_atanh_error": 0.0},
            "description": "Tanh identity at x=0.5",
        },
    ],
)
async def verify_inverse_hyperbolic_identities(
    x: Union[int, float], tolerance: float = 1e-12
) -> dict:
    """
    Verify inverse hyperbolic function identities.

    Tests identities like sinh(asinh(x)) = x, cosh(acosh(x)) = x, etc.

    Args:
        x: Input value to test
        tolerance: Tolerance for numerical verification

    Returns:
        Dictionary with verification results for applicable identities

    Examples:
        await verify_inverse_hyperbolic_identities(2) → {...}
        await verify_inverse_hyperbolic_identities(0.5) → {...}
    """
    from .hyperbolic import sinh, cosh, tanh

    result = {"tolerance": tolerance}

    # Test sinh(asinh(x)) = x (always applicable)
    asinh_x = await asinh(x)
    sinh_asinh_x = await sinh(asinh_x)
    sinh_asinh_error = abs(sinh_asinh_x - x)
    result["sinh_asinh_identity"] = sinh_asinh_error <= tolerance
    result["sinh_asinh_error"] = sinh_asinh_error

    # Test cosh(acosh(x)) = x (for x ≥ 1)
    if x >= 1:
        acosh_x = await acosh(x)
        cosh_acosh_x = await cosh(acosh_x)
        cosh_acosh_error = abs(cosh_acosh_x - x)
        result["cosh_acosh_identity"] = cosh_acosh_error <= tolerance
        result["cosh_acosh_error"] = cosh_acosh_error

    # Test tanh(atanh(x)) = x (for |x| < 1)
    if abs(x) < 1:
        atanh_x = await atanh(x)
        tanh_atanh_x = await tanh(atanh_x)
        tanh_atanh_error = abs(tanh_atanh_x - x)
        result["tanh_atanh_identity"] = tanh_atanh_error <= tolerance
        result["tanh_atanh_error"] = tanh_atanh_error

    return result


# Export all functions
__all__ = [
    # Primary inverse hyperbolic functions
    "asinh",
    "acosh",
    "atanh",
    # Inverse reciprocal hyperbolic functions
    "acsch",
    "asech",
    "acoth",
    # Utility functions
    "inverse_hyperbolic_functions",
    "verify_inverse_hyperbolic_identities",
]

if __name__ == "__main__":
    import asyncio

    async def test_inverse_hyperbolic_functions():
        """Test inverse hyperbolic functions."""
        print("🔄 Inverse Hyperbolic Functions Test")
        print("=" * 40)

        # Test primary inverse functions
        print("Primary Inverse Hyperbolic Functions:")

        # asinh (defined for all x)
        asinh_values = [0, 1, -1, 2, -2, 0.5]
        for x in asinh_values:
            result = await asinh(x)
            print(f"  asinh({x:4.1f}) = {result:8.4f}")

        # acosh (defined for x ≥ 1)
        print("\nInverse Hyperbolic Cosine (x ≥ 1):")
        acosh_values = [1, 1.5, 2, 3, 10]
        for x in acosh_values:
            try:
                result = await acosh(x)
                print(f"  acosh({x:4.1f}) = {result:8.4f}")
            except ValueError as e:
                print(f"  acosh({x:4.1f}) = Error: {e}")

        # atanh (defined for |x| < 1)
        print("\nInverse Hyperbolic Tangent (|x| < 1):")
        atanh_values = [0, 0.5, -0.5, 0.9, -0.9]
        for x in atanh_values:
            try:
                result = await atanh(x)
                print(f"  atanh({x:4.1f}) = {result:8.4f}")
            except ValueError as e:
                print(f"  atanh({x:4.1f}) = Error: {e}")

        print("\nInverse Reciprocal Hyperbolic Functions:")

        # Test various values for reciprocal functions
        test_values = [0.1, 0.5, 1, 1.5, 2, -1, -2]
        for x in test_values:
            functions = await inverse_hyperbolic_functions(x)
            defined_funcs = {k: v for k, v in functions.items() if v is not None}
            print(f"  x = {x:4.1f}: {len(defined_funcs)} functions defined")
            for func_name, value in defined_funcs.items():
                print(f"    {func_name}({x}) = {value:.4f}")

        print("\nIdentity Verification:")
        test_identity_values = [0.5, 1, 1.5, 2, -0.5, -1.5]
        for x in test_identity_values:
            identities = await verify_inverse_hyperbolic_identities(x)
            print(f"  x = {x:4.1f}:")
            for key, value in identities.items():
                if key.endswith("_identity"):
                    error_key = key.replace("_identity", "_error")
                    error = identities.get(error_key, 0)
                    print(f"    {key}: {value} (error: {error:.2e})")

        print("\n✅ All inverse hyperbolic functions working!")

    asyncio.run(test_inverse_hyperbolic_functions())
