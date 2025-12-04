#!/usr/bin/env python3
# chuk_mcp_math/trigonometry/identities.py
"""
Trigonometric Identities - Async Native

Comprehensive trigonometric identity verification and computation functions.
Includes Pythagorean identities, sum/difference formulas, double/half angle formulas,
and identity verification tools.

Functions:
- Pythagorean identities: sin²θ + cos²θ = 1, etc.
- Sum and difference formulas: sin(a±b), cos(a±b), tan(a±b)
- Double angle formulas: sin(2θ), cos(2θ), tan(2θ)
- Half angle formulas: sin(θ/2), cos(θ/2), tan(θ/2)
- Identity verification: numerical validation within tolerance
- Expression simplification: basic trigonometric expression handling
"""

import math
from typing import Union, Literal, Dict, Any, List, Optional
from chuk_mcp_math.mcp_decorator import mcp_function

# ============================================================================
# PYTHAGOREAN IDENTITIES
# ============================================================================


@mcp_function(
    description="Verify and compute Pythagorean trigonometric identities.",
    namespace="trigonometry",
    category="identities",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    cache_ttl_seconds=3600,
    examples=[
        {
            "input": {"angle": 0.7853981633974483},
            "output": {
                "sin_cos_identity": True,
                "sin_cos_value": 1.0,
                "sin_cos_error": 0.0,
                "sec_tan_identity": True,
                "csc_cot_identity": True,
            },
            "description": "Pythagorean identities at π/4",
        },
        {
            "input": {"angle": 1.0471975511965976},
            "output": {
                "sin_cos_identity": True,
                "sin_cos_value": 1.0000000000000002,
                "sin_cos_error": 2.220446049250313e-16,
            },
            "description": "Identities at π/3 with small numerical error",
        },
    ],
)
async def pythagorean_identity(
    angle: Union[int, float], tolerance: float = 1e-12
) -> Dict[str, Any]:
    """
    Verify all Pythagorean trigonometric identities at a given angle.

    Verifies:
    1. sin²θ + cos²θ = 1
    2. 1 + tan²θ = sec²θ (when cos θ ≠ 0)
    3. 1 + cot²θ = csc²θ (when sin θ ≠ 0)

    Args:
        angle: Angle in radians
        tolerance: Tolerance for numerical verification

    Returns:
        Dictionary with verification results for all applicable identities

    Examples:
        await pythagorean_identity(π/4) → {"sin_cos_identity": True, ...}
        await pythagorean_identity(π/6) → {"sin_cos_identity": True, ...}
    """
    from .basic_functions import sin, cos, tan, sec, csc, cot

    result: Dict[str, Any] = {"angle": angle, "tolerance": tolerance}

    # Get trigonometric values
    sin_val = await sin(angle)
    cos_val = await cos(angle)

    # Identity 1: sin²θ + cos²θ = 1
    sin_cos_value = sin_val**2 + cos_val**2
    sin_cos_error = abs(sin_cos_value - 1.0)
    result["sin_cos_identity"] = sin_cos_error <= tolerance
    result["sin_cos_value"] = sin_cos_value
    result["sin_cos_error"] = sin_cos_error

    # Identity 2: 1 + tan²θ = sec²θ (when cos θ ≠ 0)
    if abs(cos_val) > tolerance:
        try:
            tan_val = await tan(angle)
            sec_val = await sec(angle)

            sec_tan_left = 1 + tan_val**2
            sec_tan_right = sec_val**2
            sec_tan_error = abs(sec_tan_left - sec_tan_right)

            result["sec_tan_identity"] = sec_tan_error <= tolerance
            result["sec_tan_left"] = sec_tan_left
            result["sec_tan_right"] = sec_tan_right
            result["sec_tan_error"] = sec_tan_error
        except ValueError:
            result["sec_tan_identity"] = None
            result["sec_tan_note"] = "tan or sec undefined at this angle"
    else:
        result["sec_tan_identity"] = None
        result["sec_tan_note"] = "cos(θ) ≈ 0, identity not applicable"

    # Identity 3: 1 + cot²θ = csc²θ (when sin θ ≠ 0)
    if abs(sin_val) > tolerance:
        try:
            cot_val = await cot(angle)
            csc_val = await csc(angle)

            csc_cot_left = 1 + cot_val**2
            csc_cot_right = csc_val**2
            csc_cot_error = abs(csc_cot_left - csc_cot_right)

            result["csc_cot_identity"] = csc_cot_error <= tolerance
            result["csc_cot_left"] = csc_cot_left
            result["csc_cot_right"] = csc_cot_right
            result["csc_cot_error"] = csc_cot_error
        except ValueError:
            result["csc_cot_identity"] = None
            result["csc_cot_note"] = "cot or csc undefined at this angle"
    else:
        result["csc_cot_identity"] = None
        result["csc_cot_note"] = "sin(θ) ≈ 0, identity not applicable"

    return result


# ============================================================================
# SUM AND DIFFERENCE FORMULAS
# ============================================================================


@mcp_function(
    description="Calculate trigonometric functions using sum and difference formulas.",
    namespace="trigonometry",
    category="identities",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    cache_ttl_seconds=3600,
    examples=[
        {
            "input": {
                "a": 0.7853981633974483,
                "b": 0.5235987755982988,
                "operation": "add",
            },
            "output": {
                "sin_formula": 0.9659258262890683,
                "cos_formula": 0.25881904510252074,
                "tan_formula": 3.7320508075688776,
            },
            "description": "sin(π/4 + π/6), cos(π/4 + π/6), tan(π/4 + π/6)",
        },
        {
            "input": {
                "a": 1.0471975511965976,
                "b": 0.5235987755982988,
                "operation": "subtract",
            },
            "output": {
                "sin_formula": 0.5000000000000001,
                "cos_formula": 0.8660254037844386,
                "tan_formula": 0.5773502691896257,
            },
            "description": "sin(π/3 - π/6), cos(π/3 - π/6), tan(π/3 - π/6)",
        },
    ],
)
async def sum_difference_formulas(
    a: Union[int, float],
    b: Union[int, float],
    operation: Literal["add", "subtract"] = "add",
) -> Dict[str, float]:
    """
    Calculate trigonometric functions using sum and difference formulas.

    Formulas:
    - sin(a ± b) = sin(a)cos(b) ± cos(a)sin(b)
    - cos(a ± b) = cos(a)cos(b) ∓ sin(a)sin(b)
    - tan(a ± b) = (tan(a) ± tan(b)) / (1 ∓ tan(a)tan(b))

    Args:
        a: First angle in radians
        b: Second angle in radians
        operation: "add" for a+b, "subtract" for a-b

    Returns:
        Dictionary containing sin, cos, and tan of (a ± b)

    Examples:
        await sum_difference_formulas(π/4, π/6, "add") → {...}
        await sum_difference_formulas(π/3, π/6, "subtract") → {...}
    """
    from .basic_functions import sin, cos, tan

    # Get trigonometric values for both angles
    sin_a = await sin(a)
    cos_a = await cos(a)
    sin_b = await sin(b)
    cos_b = await cos(b)

    # Calculate using sum/difference formulas
    if operation == "add":
        # sin(a + b) = sin(a)cos(b) + cos(a)sin(b)
        sin_formula = sin_a * cos_b + cos_a * sin_b

        # cos(a + b) = cos(a)cos(b) - sin(a)sin(b)
        cos_formula = cos_a * cos_b - sin_a * sin_b

        # tan(a + b) = (tan(a) + tan(b)) / (1 - tan(a)tan(b))
        try:
            tan_a = await tan(a)
            tan_b = await tan(b)
            denominator = 1 - tan_a * tan_b
            if abs(denominator) < 1e-15:
                tan_formula = float("inf") if denominator >= 0 else float("-inf")
            else:
                tan_formula = (tan_a + tan_b) / denominator
        except ValueError:
            # Handle cases where tan is undefined
            tan_formula = None

    else:  # subtract
        # sin(a - b) = sin(a)cos(b) - cos(a)sin(b)
        sin_formula = sin_a * cos_b - cos_a * sin_b

        # cos(a - b) = cos(a)cos(b) + sin(a)sin(b)
        cos_formula = cos_a * cos_b + sin_a * sin_b

        # tan(a - b) = (tan(a) - tan(b)) / (1 + tan(a)tan(b))
        try:
            tan_a = await tan(a)
            tan_b = await tan(b)
            denominator = 1 + tan_a * tan_b
            if abs(denominator) < 1e-15:
                tan_formula = float("inf") if denominator >= 0 else float("-inf")
            else:
                tan_formula = (tan_a - tan_b) / denominator
        except ValueError:
            tan_formula = None

    return {
        "sin_formula": sin_formula,
        "cos_formula": cos_formula,
        "tan_formula": tan_formula,  # type: ignore[dict-item]
        "operation": operation,  # type: ignore[dict-item]
        "angle_a": a,
        "angle_b": b,
    }


@mcp_function(
    description="Verify sum and difference formulas by comparison with direct calculation.",
    namespace="trigonometry",
    category="identities",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    examples=[
        {
            "input": {
                "a": 0.7853981633974483,
                "b": 0.5235987755982988,
                "operation": "add",
            },
            "output": {
                "sin_verified": True,
                "cos_verified": True,
                "tan_verified": True,
                "max_error": 2.220446049250313e-16,
            },
            "description": "Verification of formulas at π/4 + π/6",
        },
        {
            "input": {
                "a": 1.0471975511965976,
                "b": 0.5235987755982988,
                "operation": "subtract",
            },
            "output": {
                "sin_verified": True,
                "cos_verified": True,
                "tan_verified": True,
                "max_error": 1.1102230246251565e-16,
            },
            "description": "Verification at π/3 - π/6",
        },
    ],
)
async def verify_sum_difference_formulas(
    a: Union[int, float],
    b: Union[int, float],
    operation: Literal["add", "subtract"] = "add",
    tolerance: float = 1e-12,
) -> Dict[str, Any]:
    """
    Verify sum and difference formulas by comparing with direct calculation.

    Args:
        a: First angle in radians
        b: Second angle in radians
        operation: "add" or "subtract"
        tolerance: Tolerance for numerical verification

    Returns:
        Dictionary with verification results and error analysis
    """
    from .basic_functions import sin, cos, tan

    # Calculate using formulas
    formula_results = await sum_difference_formulas(a, b, operation)

    # Calculate directly
    result_angle = a + b if operation == "add" else a - b
    sin_direct = await sin(result_angle)
    cos_direct = await cos(result_angle)

    try:
        tan_direct = await tan(result_angle)
    except ValueError:
        tan_direct = None

    # Compare results
    sin_error = abs(formula_results["sin_formula"] - sin_direct)
    cos_error = abs(formula_results["cos_formula"] - cos_direct)

    sin_verified = sin_error <= tolerance
    cos_verified = cos_error <= tolerance

    tan_verified = None
    tan_error = None
    if formula_results["tan_formula"] is not None and tan_direct is not None:
        if math.isfinite(formula_results["tan_formula"]) and math.isfinite(tan_direct):
            tan_error = abs(formula_results["tan_formula"] - tan_direct)
            tan_verified = tan_error <= tolerance

    max_error = max(sin_error, cos_error)
    if tan_error is not None:
        max_error = max(max_error, tan_error)

    return {
        "sin_verified": sin_verified,
        "cos_verified": cos_verified,
        "tan_verified": tan_verified,
        "sin_error": sin_error,
        "cos_error": cos_error,
        "tan_error": tan_error,
        "max_error": max_error,
        "tolerance": tolerance,
        "result_angle": result_angle,
    }


# ============================================================================
# DOUBLE ANGLE FORMULAS
# ============================================================================


@mcp_function(
    description="Calculate trigonometric functions using double angle formulas.",
    namespace="trigonometry",
    category="identities",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    cache_ttl_seconds=3600,
    examples=[
        {
            "input": {"angle": 0.7853981633974483, "function": "sin"},
            "output": {
                "double_angle_value": 1.0,
                "formula_used": "sin(2θ) = 2sin(θ)cos(θ)",
                "original_angle": 0.7853981633974483,
                "double_angle": 1.5707963267948966,
            },
            "description": "sin(2 × π/4) = sin(π/2) = 1",
        },
        {
            "input": {"angle": 0.5235987755982988, "function": "cos"},
            "output": {
                "double_angle_value": 0.5000000000000001,
                "formula_used": "cos(2θ) = cos²(θ) - sin²(θ)",
                "original_angle": 0.5235987755982988,
                "double_angle": 1.0471975511965976,
            },
            "description": "cos(2 × π/6) = cos(π/3) = 0.5",
        },
    ],
)
async def double_angle_formulas(
    angle: Union[int, float], function: Literal["sin", "cos", "tan"] = "sin"
) -> Dict[str, Any]:
    """
    Calculate trigonometric functions using double angle formulas.

    Formulas:
    - sin(2θ) = 2sin(θ)cos(θ)
    - cos(2θ) = cos²(θ) - sin²(θ) = 2cos²(θ) - 1 = 1 - 2sin²(θ)
    - tan(2θ) = 2tan(θ) / (1 - tan²(θ))

    Args:
        angle: Angle in radians
        function: Which function to calculate ("sin", "cos", or "tan")

    Returns:
        Dictionary containing the calculated value and formula information

    Examples:
        await double_angle_formulas(π/4, "sin") → {"double_angle_value": 1.0, ...}
        await double_angle_formulas(π/6, "cos") → {"double_angle_value": 0.5, ...}
    """
    from .basic_functions import sin, cos, tan

    # Get trigonometric values for the original angle
    sin_val = await sin(angle)
    cos_val = await cos(angle)

    double_angle = 2 * angle

    if function == "sin":
        # sin(2θ) = 2sin(θ)cos(θ)
        double_angle_value = 2 * sin_val * cos_val
        formula_used = "sin(2θ) = 2sin(θ)cos(θ)"

    elif function == "cos":
        # cos(2θ) = cos²(θ) - sin²(θ) (primary form)
        double_angle_value = cos_val**2 - sin_val**2
        formula_used = "cos(2θ) = cos²(θ) - sin²(θ)"

        # Also calculate alternative forms for verification
        alt_form1 = 2 * cos_val**2 - 1
        alt_form2 = 1 - 2 * sin_val**2

    elif function == "tan":
        try:
            tan_val = await tan(angle)
            denominator = 1 - tan_val**2

            if abs(denominator) < 1e-15:
                double_angle_value = float("inf") if denominator >= 0 else float("-inf")
            else:
                double_angle_value = (2 * tan_val) / denominator

            formula_used = "tan(2θ) = 2tan(θ) / (1 - tan²(θ))"
        except ValueError:
            double_angle_value = None
            formula_used = "tan(2θ) undefined (tan(θ) undefined)"

    result: Dict[str, Any] = {
        "double_angle_value": double_angle_value,
        "formula_used": formula_used,
        "original_angle": angle,
        "double_angle": double_angle,
        "function": function,
    }

    # Add alternative forms for cosine
    if function == "cos":
        result["alternative_forms"] = {
            "2cos²(θ) - 1": alt_form1,
            "1 - 2sin²(θ)": alt_form2,
        }

    return result


# ============================================================================
# HALF ANGLE FORMULAS
# ============================================================================


@mcp_function(
    description="Calculate trigonometric functions using half angle formulas.",
    namespace="trigonometry",
    category="identities",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    cache_ttl_seconds=3600,
    examples=[
        {
            "input": {"angle": 1.5707963267948966, "function": "sin"},
            "output": {
                "half_angle_value": 0.7071067811865476,
                "formula_used": "sin(θ/2) = ±√((1 - cos(θ))/2)",
                "original_angle": 1.5707963267948966,
                "half_angle": 0.7853981633974483,
            },
            "description": "sin(π/4) using half angle formula",
        },
        {
            "input": {"angle": 2.0943951023931953, "function": "cos"},
            "output": {
                "half_angle_value": 0.8660254037844387,
                "formula_used": "cos(θ/2) = ±√((1 + cos(θ))/2)",
                "original_angle": 2.0943951023931953,
                "half_angle": 1.0471975511965976,
            },
            "description": "cos(π/3) using half angle formula",
        },
    ],
)
async def half_angle_formulas(
    angle: Union[int, float], function: Literal["sin", "cos", "tan"] = "sin"
) -> Dict[str, Any]:
    """
    Calculate trigonometric functions using half angle formulas.

    Formulas:
    - sin(θ/2) = ±√((1 - cos(θ))/2)
    - cos(θ/2) = ±√((1 + cos(θ))/2)
    - tan(θ/2) = ±√((1 - cos(θ))/(1 + cos(θ))) = sin(θ)/(1 + cos(θ)) = (1 - cos(θ))/sin(θ)

    Sign is determined by the quadrant of θ/2.

    Args:
        angle: Angle in radians
        function: Which function to calculate ("sin", "cos", or "tan")

    Returns:
        Dictionary containing the calculated value and formula information

    Examples:
        await half_angle_formulas(π/2, "sin") → {"half_angle_value": √2/2, ...}
        await half_angle_formulas(2π/3, "cos") → {"half_angle_value": √3/2, ...}
    """
    from .basic_functions import sin, cos
    from .angle_conversion import normalize_angle

    # Get cosine of the original angle
    cos_val = await cos(angle)
    sin_val = await sin(angle)

    half_angle = angle / 2

    # Determine the quadrant of θ/2 to get the correct sign
    normalized_half = await normalize_angle(half_angle, "radians", "positive")

    if function == "sin":
        # sin(θ/2) = ±√((1 - cos(θ))/2)
        magnitude = math.sqrt((1 - cos_val) / 2)

        # sin is positive in Q1 and Q2 (0 to π)
        sign = 1 if normalized_half <= math.pi else -1
        half_angle_value = sign * magnitude
        formula_used = "sin(θ/2) = ±√((1 - cos(θ))/2)"

    elif function == "cos":
        # cos(θ/2) = ±√((1 + cos(θ))/2)
        magnitude = math.sqrt((1 + cos_val) / 2)

        # cos is positive in Q1 and Q4 (0 to π/2 and 3π/2 to 2π)
        sign = (
            1
            if (normalized_half <= math.pi / 2 or normalized_half >= 3 * math.pi / 2)
            else -1
        )
        half_angle_value = sign * magnitude
        formula_used = "cos(θ/2) = ±√((1 + cos(θ))/2)"

    elif function == "tan":
        # tan(θ/2) = sin(θ)/(1 + cos(θ)) (this form avoids ± ambiguity)
        if abs(1 + cos_val) < 1e-15:
            # Use alternative form: tan(θ/2) = (1 - cos(θ))/sin(θ)
            if abs(sin_val) < 1e-15:
                half_angle_value = None  # Undefined
                formula_used = "tan(θ/2) undefined"
            else:
                half_angle_value = (1 - cos_val) / sin_val
                formula_used = "tan(θ/2) = (1 - cos(θ))/sin(θ)"
        else:
            half_angle_value = sin_val / (1 + cos_val)
            formula_used = "tan(θ/2) = sin(θ)/(1 + cos(θ))"

    return {
        "half_angle_value": half_angle_value,
        "formula_used": formula_used,
        "original_angle": angle,
        "half_angle": half_angle,
        "function": function,
        "quadrant_half_angle": int(normalized_half // (math.pi / 2)) + 1,
    }


# ============================================================================
# IDENTITY VERIFICATION FUNCTIONS
# ============================================================================


@mcp_function(
    description="Verify a trigonometric identity numerically at multiple test points.",
    namespace="trigonometry",
    category="identities",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    examples=[
        {
            "input": {
                "identity_name": "pythagorean",
                "test_angles": [0, 0.7853981633974483, 1.5707963267948966],
            },
            "output": {
                "identity_verified": True,
                "test_results": [
                    {"angle": 0, "verified": True, "error": 0.0},
                    {"angle": 0.7853981633974483, "verified": True, "error": 0.0},
                ],
                "max_error": 0.0,
            },
            "description": "Verify Pythagorean identity at multiple angles",
        },
        {
            "input": {
                "identity_name": "double_angle_sin",
                "test_angles": [0.5235987755982988, 0.7853981633974483],
            },
            "output": {
                "identity_verified": True,
                "test_results": [
                    {
                        "angle": 0.5235987755982988,
                        "verified": True,
                        "error": 2.220446049250313e-16,
                    }
                ],
                "max_error": 2.220446049250313e-16,
            },
            "description": "Verify double angle sine identity",
        },
    ],
)
async def verify_identity(
    identity_name: str,
    test_angles: Optional[List[float]] = None,
    tolerance: float = 1e-12,
) -> Dict[str, Any]:
    """
    Verify a trigonometric identity numerically at multiple test points.

    Supported identities:
    - "pythagorean": sin²θ + cos²θ = 1
    - "double_angle_sin": sin(2θ) = 2sin(θ)cos(θ)
    - "double_angle_cos": cos(2θ) = cos²(θ) - sin²(θ)
    - "sum_sin": sin(a+b) = sin(a)cos(b) + cos(a)sin(b)
    - "sum_cos": cos(a+b) = cos(a)cos(b) - sin(a)sin(b)

    Args:
        identity_name: Name of the identity to verify
        test_angles: List of angles to test (uses default set if None)
        tolerance: Tolerance for numerical verification

    Returns:
        Dictionary with comprehensive verification results

    Examples:
        await verify_identity("pythagorean") → {...}
        await verify_identity("double_angle_sin", [π/6, π/4, π/3]) → {...}
    """
    from .basic_functions import sin, cos

    # Default test angles if not provided
    if test_angles is None:
        test_angles = [
            0,
            math.pi / 6,
            math.pi / 4,
            math.pi / 3,
            math.pi / 2,
            2 * math.pi / 3,
            3 * math.pi / 4,
            5 * math.pi / 6,
            math.pi,
            -math.pi / 6,
            -math.pi / 4,
            -math.pi / 3,
        ]

    test_results = []
    max_error = 0.0
    all_verified = True

    for angle in test_angles:
        try:
            if identity_name == "pythagorean":
                sin_val = await sin(angle)
                cos_val = await cos(angle)
                left_side = sin_val**2 + cos_val**2
                right_side = 1.0
                error = abs(left_side - right_side)

            elif identity_name == "double_angle_sin":
                sin_val = await sin(angle)
                cos_val = await cos(angle)
                sin_2theta = await sin(2 * angle)
                left_side = sin_2theta
                right_side = 2 * sin_val * cos_val
                error = abs(left_side - right_side)

            elif identity_name == "double_angle_cos":
                sin_val = await sin(angle)
                cos_val = await cos(angle)
                cos_2theta = await cos(2 * angle)
                left_side = cos_2theta
                right_side = cos_val**2 - sin_val**2
                error = abs(left_side - right_side)

            else:
                # Unsupported identity
                test_results.append(
                    {
                        "angle": angle,
                        "verified": False,
                        "error": None,
                        "note": f"Unsupported identity: {identity_name}",
                    }
                )
                all_verified = False
                continue

            verified = error <= tolerance
            max_error = max(max_error, error)

            if not verified:
                all_verified = False

            test_results.append(
                {
                    "angle": angle,
                    "verified": verified,
                    "error": error,
                    "left_side": left_side,
                    "right_side": right_side,
                }
            )

        except Exception as e:
            test_results.append(
                {"angle": angle, "verified": False, "error": None, "exception": str(e)}
            )
            all_verified = False

    return {
        "identity_name": identity_name,
        "identity_verified": all_verified,
        "test_results": test_results,
        "max_error": max_error,
        "tolerance": tolerance,
        "num_tests": len(test_angles),
        "num_verified": sum(1 for r in test_results if r.get("verified", False)),
    }


@mcp_function(
    description="Generate a comprehensive identity verification report.",
    namespace="trigonometry",
    category="identities",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    examples=[
        {
            "input": {"test_angles": [0, 0.7853981633974483, 1.5707963267948966]},
            "output": {
                "identities_tested": 3,
                "all_verified": True,
                "summary": {
                    "pythagorean": True,
                    "double_angle_sin": True,
                    "double_angle_cos": True,
                },
            },
            "description": "Comprehensive verification report",
        }
    ],
)
async def comprehensive_identity_verification(
    test_angles: Optional[List[float]] = None, tolerance: float = 1e-12
) -> Dict[str, Any]:
    """
    Generate a comprehensive verification report for multiple identities.

    Tests all supported identities at the given angles and provides
    a summary report.

    Args:
        test_angles: List of angles to test
        tolerance: Tolerance for numerical verification

    Returns:
        Dictionary with comprehensive verification report

    Examples:
        await comprehensive_identity_verification() → {...}
        await comprehensive_identity_verification([π/6, π/4, π/3]) → {...}
    """
    identities_to_test = ["pythagorean", "double_angle_sin", "double_angle_cos"]

    results = {}
    summary = {}
    all_verified = True
    total_tests = 0
    total_verified = 0

    for identity in identities_to_test:
        verification = await verify_identity(identity, test_angles, tolerance)
        results[identity] = verification
        summary[identity] = verification["identity_verified"]

        if not verification["identity_verified"]:
            all_verified = False

        total_tests += verification["num_tests"]
        total_verified += verification["num_verified"]

    return {
        "identities_tested": len(identities_to_test),
        "all_verified": all_verified,
        "summary": summary,
        "detailed_results": results,
        "total_tests": total_tests,
        "total_verified": total_verified,
        "overall_success_rate": total_verified / total_tests if total_tests > 0 else 0,
        "tolerance": tolerance,
    }


# Add the missing simplify_trig_expression function for compatibility
@mcp_function(
    description="Basic trigonometric expression simplification (placeholder implementation).",
    namespace="trigonometry",
    category="identities",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    examples=[
        {
            "input": {"expression": "sin^2(x) + cos^2(x)"},
            "output": {
                "simplified": "1",
                "original": "sin^2(x) + cos^2(x)",
                "steps": ["Applied Pythagorean identity"],
            },
            "description": "Simplify basic Pythagorean identity",
        }
    ],
)
async def simplify_trig_expression(expression: str) -> Dict[str, Any]:
    """
    Basic trigonometric expression simplification.

    This is a placeholder implementation that handles a few common cases.
    For more advanced simplification, consider using a symbolic math library.

    Args:
        expression: String representation of trigonometric expression

    Returns:
        Dictionary with simplified expression and steps

    Examples:
        await simplify_trig_expression("sin^2(x) + cos^2(x)") → {"simplified": "1", ...}
    """
    # Basic pattern matching for common identities
    expr = expression.lower().replace(" ", "")

    # Pythagorean identities
    if "sin^2" in expr and "cos^2" in expr and "+" in expr:
        if "sin^2(x)+cos^2(x)" in expr or "cos^2(x)+sin^2(x)" in expr:
            return {
                "simplified": "1",
                "original": expression,
                "steps": ["Applied Pythagorean identity: sin²(x) + cos²(x) = 1"],
                "identity_used": "pythagorean",
            }

    # Double angle patterns
    if "2sin" in expr and "cos" in expr:
        return {
            "simplified": "sin(2x)",
            "original": expression,
            "steps": ["Applied double angle identity: 2sin(x)cos(x) = sin(2x)"],
            "identity_used": "double_angle_sin",
        }

    # If no simplification found, return original
    return {
        "simplified": expression,
        "original": expression,
        "steps": ["No applicable simplification found"],
        "identity_used": None,
    }


# Export all functions
__all__ = [
    # Pythagorean identities
    "pythagorean_identity",
    # Sum and difference formulas
    "sum_difference_formulas",
    "verify_sum_difference_formulas",
    # Double angle formulas
    "double_angle_formulas",
    # Half angle formulas
    "half_angle_formulas",
    # Identity verification
    "verify_identity",
    "comprehensive_identity_verification",
    # Expression simplification
    "simplify_trig_expression",
]
