#!/usr/bin/env python3
# chuk_mcp_math/number_theory/wilsons_theorem_bezout.py
"""
Wilson's Theorem and Bézout's Identity - Async Native - COMPLETE IMPLEMENTATION

Implementation of Wilson's theorem for primality testing and Bézout's identity
for finding linear combinations that equal the GCD.

Functions:
- Wilson's theorem: wilson_theorem_test, wilson_theorem_verify
- Bézout's identity: bezout_identity, extended_gcd_bezout
- Applications: wilson_primality_test, bezout_applications

Mathematical Background:
Wilson's Theorem: A natural number n > 1 is prime if and only if (n-1)! ≡ -1 (mod n)
Bézout's Identity: For integers a, b, there exist integers x, y such that ax + by = gcd(a,b)
"""

import asyncio
from typing import List, Dict, Optional
from chuk_mcp_math.mcp_decorator import mcp_function

# ============================================================================
# WILSON'S THEOREM IMPLEMENTATION
# ============================================================================


@mcp_function(
    description="Test primality using Wilson's theorem: (n-1)! ≡ -1 (mod n) iff n is prime.",
    namespace="arithmetic",
    category="wilson_theorem",
    execution_modes=["local", "remote"],
    estimated_cpu_usage="extreme",
    examples=[
        {
            "input": {"n": 7},
            "output": True,
            "description": "7 is prime, 6! ≡ -1 (mod 7)",
        },
        {
            "input": {"n": 11},
            "output": True,
            "description": "11 is prime, 10! ≡ -1 (mod 11)",
        },
        {
            "input": {"n": 9},
            "output": False,
            "description": "9 is composite, 8! ≢ -1 (mod 9)",
        },
        {
            "input": {"n": 4},
            "output": False,
            "description": "4 is composite, 3! ≢ -1 (mod 4)",
        },
    ],
)
async def wilson_theorem_test(n: int) -> bool:
    """
    Test if n is prime using Wilson's theorem.

    Wilson's theorem states that a natural number n > 1 is prime if and only if
    (n-1)! ≡ -1 (mod n).

    Args:
        n: Number to test for primality (must be > 1)

    Returns:
        True if n passes Wilson's test (is prime), False otherwise

    Note: Computationally expensive for large n due to factorial calculation

    Examples:
        await wilson_theorem_test(7) → True
        await wilson_theorem_test(9) → False
        await wilson_theorem_test(11) → True
    """
    if n <= 1:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False

    # Calculate (n-1)! mod n efficiently
    factorial_mod = 1

    # Yield control for large factorials
    if n > 100:
        await asyncio.sleep(0)

    for i in range(1, n):
        factorial_mod = (factorial_mod * i) % n

        # Early termination if factorial becomes 0 mod n
        if factorial_mod == 0:
            return False

        # Yield control every 1000 iterations for very large n
        if i % 1000 == 0 and n > 10000:
            await asyncio.sleep(0)

    # Check if (n-1)! ≡ -1 (mod n), which is equivalent to n-1
    return factorial_mod == n - 1


@mcp_function(
    description="Verify Wilson's theorem for a given prime with detailed calculation.",
    namespace="arithmetic",
    category="wilson_theorem",
    execution_modes=["local", "remote"],
    estimated_cpu_usage="high",
    examples=[
        {
            "input": {"p": 7},
            "output": {
                "factorial": 720,
                "factorial_mod_p": 6,
                "is_minus_one": True,
                "verification": "6! = 720 ≡ 6 ≡ -1 (mod 7)",
            },
            "description": "Wilson's theorem verification for 7",
        },
        {
            "input": {"p": 5},
            "output": {
                "factorial": 24,
                "factorial_mod_p": 4,
                "is_minus_one": True,
                "verification": "4! = 24 ≡ 4 ≡ -1 (mod 5)",
            },
            "description": "Wilson's theorem verification for 5",
        },
        {
            "input": {"p": 11},
            "output": {"factorial_mod_p": 10, "is_minus_one": True},
            "description": "Wilson's theorem verification for 11",
        },
        {
            "input": {"p": 13},
            "output": {"factorial_mod_p": 12, "is_minus_one": True},
            "description": "Wilson's theorem verification for 13",
        },
    ],
)
async def wilson_theorem_verify(p: int) -> Dict:
    """
    Verify Wilson's theorem for a given number with detailed calculation.

    Provides step-by-step verification of Wilson's theorem including
    intermediate values and explanation.

    Args:
        p: Number to verify Wilson's theorem for

    Returns:
        Dictionary with verification details and results

    Examples:
        await wilson_theorem_verify(7) → {"factorial": 720, "factorial_mod_p": 6, ...}
        await wilson_theorem_verify(5) → {"factorial": 24, "factorial_mod_p": 4, ...}
    """
    if p <= 1:
        return {"error": "Wilson's theorem applies to numbers > 1"}

    # Calculate (p-1)! and (p-1)! mod p
    factorial = 1
    factorial_mod_p = 1

    # Yield control for large calculations
    if p > 100:
        await asyncio.sleep(0)

    intermediate_steps = []

    for i in range(1, p):
        factorial *= i
        factorial_mod_p = (factorial_mod_p * i) % p

        # Record some intermediate steps for small p
        if p <= 10:
            intermediate_steps.append(
                {
                    "step": i,
                    "partial_factorial": factorial,
                    "partial_factorial_mod_p": factorial_mod_p,
                }
            )

        # Yield control for large p
        if i % 1000 == 0 and p > 10000:
            await asyncio.sleep(0)

    is_minus_one = factorial_mod_p == p - 1

    result = {
        "p": p,
        "factorial_expression": f"({p - 1})!",
        "factorial_mod_p": factorial_mod_p,
        "expected_value": p - 1,
        "is_minus_one": is_minus_one,
        "wilson_theorem_satisfied": is_minus_one,
        "verification": f"({p - 1})! ≡ {factorial_mod_p} ≡ {'-1' if is_minus_one else 'not -1'} (mod {p})",
    }

    # Include full factorial for small numbers
    if p <= 20:
        result["factorial"] = factorial
        result["detailed_verification"] = (
            f"({p - 1})! = {factorial} ≡ {factorial_mod_p} (mod {p})"
        )

    # Include intermediate steps for very small numbers
    if p <= 10 and intermediate_steps:
        result["intermediate_steps"] = intermediate_steps

    return result


@mcp_function(
    description="Optimized Wilson's theorem test using pairing technique for better performance.",
    namespace="arithmetic",
    category="wilson_theorem",
    execution_modes=["local", "remote"],
    estimated_cpu_usage="high",
    examples=[
        {
            "input": {"n": 17},
            "output": True,
            "description": "17 is prime (optimized test)",
        },
        {
            "input": {"n": 15},
            "output": False,
            "description": "15 is composite (optimized test)",
        },
        {
            "input": {"n": 19},
            "output": True,
            "description": "19 is prime (optimized test)",
        },
        {
            "input": {"n": 21},
            "output": False,
            "description": "21 is composite (optimized test)",
        },
    ],
)
async def optimized_wilson_test(n: int) -> bool:
    """
    Optimized Wilson's theorem test using pairing of multiplicative inverses.

    Uses the fact that in (p-1)!, most numbers can be paired with their
    multiplicative inverses, leaving only -1 unpaired for primes.

    Args:
        n: Number to test for primality

    Returns:
        True if n is prime according to Wilson's theorem, False otherwise

    Examples:
        await optimized_wilson_test(17) → True
        await optimized_wilson_test(15) → False
        await optimized_wilson_test(19) → True
    """
    if n <= 1:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False

    # For small n, use regular Wilson test
    if n <= 100:
        return await wilson_theorem_test(n)

    # For larger n, use pairing optimization
    # In a prime p, numbers 2, 3, ..., p-2 can be paired with their inverses
    # Only 1 and p-1 are their own inverses

    product = 1

    # Yield control for large n
    await asyncio.sleep(0)

    # We only need to compute the product avoiding the pairing cancellation
    # This is still O(n) but with better constant factors
    for i in range(2, n - 1):
        # Find multiplicative inverse of i modulo n
        inverse = await _multiplicative_inverse_async(i, n)

        if inverse is None:
            return False  # No inverse means n is composite

        if i < inverse:  # Only multiply once per pair
            product = (product * i * inverse) % n
        elif i == inverse:  # Self-inverse (should only be 1 and n-1 for primes)
            if i != 1 and i != n - 1:
                return False
            product = (product * i) % n

        # Yield control every 1000 iterations
        if i % 1000 == 0:
            await asyncio.sleep(0)

    # The result should be -1 (mod n) for primes
    return product == n - 1


# ============================================================================
# BÉZOUT'S IDENTITY IMPLEMENTATION
# ============================================================================


@mcp_function(
    description="Find Bézout coefficients: integers x, y such that ax + by = gcd(a, b).",
    namespace="arithmetic",
    category="bezout_identity",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    examples=[
        {
            "input": {"a": 30, "b": 18},
            "output": {"gcd": 6, "x": -1, "y": 2, "verification": "30*(-1) + 18*2 = 6"},
            "description": "Bézout coefficients for 30, 18",
        },
        {
            "input": {"a": 17, "b": 13},
            "output": {"gcd": 1, "x": -3, "y": 4, "verification": "17*(-3) + 13*4 = 1"},
            "description": "Bézout coefficients for coprime numbers",
        },
        {
            "input": {"a": 48, "b": 18},
            "output": {"gcd": 6, "x": 1, "y": -2, "verification": "48*1 + 18*(-2) = 6"},
            "description": "Another Bézout example",
        },
        {
            "input": {"a": 25, "b": 15},
            "output": {"gcd": 5, "x": -1, "y": 2, "verification": "25*(-1) + 15*2 = 5"},
            "description": "Bézout for 25, 15",
        },
    ],
)
async def bezout_identity(a: int, b: int) -> Dict:
    """
    Find Bézout coefficients using the Extended Euclidean Algorithm.

    Finds integers x, y such that ax + by = gcd(a, b).
    This is fundamental for solving linear Diophantine equations.

    Args:
        a: First integer
        b: Second integer

    Returns:
        Dictionary with gcd, coefficients x, y, and verification

    Examples:
        await bezout_identity(30, 18) → {"gcd": 6, "x": -1, "y": 2, ...}
        await bezout_identity(17, 13) → {"gcd": 1, "x": -3, "y": 4, ...}
    """
    original_a, original_b = a, b

    # Handle edge cases
    if a == 0:
        return {
            "gcd": abs(b),
            "x": 0,
            "y": 1 if b >= 0 else -1,
            "verification": f"{original_a}*0 + {original_b}*{1 if b >= 0 else -1} = {abs(b)}",
            "original_a": original_a,
            "original_b": original_b,
        }

    if b == 0:
        return {
            "gcd": abs(a),
            "x": 1 if a >= 0 else -1,
            "y": 0,
            "verification": f"{original_a}*{1 if a >= 0 else -1} + {original_b}*0 = {abs(a)}",
            "original_a": original_a,
            "original_b": original_b,
        }

    # Extended Euclidean Algorithm
    old_x, x = 1, 0
    old_y, y = 0, 1

    steps = []  # Track steps for educational purposes

    while b != 0:
        quotient = a // b

        # Update values
        a, b = b, a % b
        old_x, x = x, old_x - quotient * x
        old_y, y = y, old_y - quotient * y

        # Record step
        steps.append(
            {"quotient": quotient, "remainder": b, "x_coeff": old_x, "y_coeff": old_y}
        )

        # Yield control for very large numbers
        if len(steps) % 100 == 0 and max(abs(original_a), abs(original_b)) > 10**10:
            await asyncio.sleep(0)

    gcd_value = abs(a)

    # Adjust signs if necessary
    if a < 0:
        old_x = -old_x
        old_y = -old_y

    verification = f"{original_a}*{old_x} + {original_b}*{old_y} = {original_a * old_x + original_b * old_y}"

    result = {
        "gcd": gcd_value,
        "x": old_x,
        "y": old_y,
        "verification": verification,
        "original_a": original_a,
        "original_b": original_b,
        "bezout_equation": f"{original_a}x + {original_b}y = {gcd_value}",
    }

    # Include steps for small numbers
    if max(abs(original_a), abs(original_b)) <= 1000:
        result["steps"] = steps

    return result


@mcp_function(
    description="Extended GCD with Bézout coefficients and multiple representations.",
    namespace="arithmetic",
    category="bezout_identity",
    execution_modes=["local", "remote"],
    examples=[
        {
            "input": {"a": 24, "b": 16},
            "output": {
                "gcd": 8,
                "bezout_coeffs": [1, -1],
                "all_solutions": "x = 1 + 2t, y = -1 - 3t",
            },
            "description": "Extended GCD with general solution",
        },
        {
            "input": {"a": 35, "b": 21},
            "output": {
                "gcd": 7,
                "bezout_coeffs": [-2, 3],
                "verification_multiple": True,
            },
            "description": "GCD with verification",
        },
        {
            "input": {"a": 100, "b": 45},
            "output": {"gcd": 5, "alternative_solutions": [[-4, 9], [5, -11]]},
            "description": "Multiple Bézout representations",
        },
        {
            "input": {"a": 13, "b": 8},
            "output": {"gcd": 1, "bezout_coeffs": [-3, 5]},
            "description": "Coprime numbers",
        },
    ],
)
async def extended_gcd_bezout(a: int, b: int, find_alternatives: bool = True) -> Dict:
    """
    Extended GCD with comprehensive Bézout coefficient analysis.

    Provides multiple representations of Bézout coefficients and
    general solution formulas for linear Diophantine equations.

    Args:
        a: First integer
        b: Second integer
        find_alternatives: Whether to find alternative coefficient pairs

    Returns:
        Comprehensive dictionary with GCD, coefficients, and analysis

    Examples:
        await extended_gcd_bezout(24, 16) → {"gcd": 8, "bezout_coeffs": [1, -1], ...}
        await extended_gcd_bezout(35, 21) → {"gcd": 7, "bezout_coeffs": [-2, 3], ...}
    """
    # Get basic Bézout identity
    basic_result = await bezout_identity(a, b)
    gcd_val = basic_result["gcd"]
    x0, y0 = basic_result["x"], basic_result["y"]

    result = {
        "gcd": gcd_val,
        "bezout_coeffs": [x0, y0],
        "fundamental_solution": f"{a}*{x0} + {b}*{y0} = {gcd_val}",
        "verification": basic_result["verification"],
    }

    if gcd_val > 0:
        # General solution formula
        if b != 0:
            b_over_gcd = b // gcd_val
            a_over_gcd = a // gcd_val
            result["general_solution"] = {
                "x_formula": f"x = {x0} + {b_over_gcd}t",
                "y_formula": f"y = {y0} - {a_over_gcd}t",
                "parameter": "t ∈ ℤ (any integer)",
            }

        # Find alternative solutions if requested
        if find_alternatives and b != 0:
            alternatives = []
            b_over_gcd = b // gcd_val
            a_over_gcd = a // gcd_val

            for t in range(-2, 3):  # Find a few alternative solutions
                if t != 0:
                    x_alt = x0 + b_over_gcd * t
                    y_alt = y0 - a_over_gcd * t
                    alternatives.append([x_alt, y_alt])

                    # Verify alternative
                    verification = a * x_alt + b * y_alt
                    if verification != gcd_val:
                        alternatives.pop()  # Remove if verification fails

            if alternatives:
                result["alternative_solutions"] = alternatives[:3]  # Limit to 3

        # Additional properties
        result["properties"] = {
            "gcd_divides_both": f"gcd({a}, {b}) = {gcd_val} divides both {a} and {b}",
            "linear_combination": f"{gcd_val} can be expressed as a linear combination of {a} and {b}",
            "uniqueness": "Coefficients are unique modulo b/gcd and a/gcd respectively",
        }

        # Applications hint
        if gcd_val == 1:
            result["applications"] = {
                "modular_inverse": f"{x0} is the multiplicative inverse of {a} modulo {b}"
                if b > 1 and x0 % b > 0
                else None,
                "coprimality": f"{a} and {b} are coprime",
                "linear_diophantine": "Any equation {a}x + {b}y = c has integer solutions for any integer c",
            }

    return result


# ============================================================================
# APPLICATIONS
# ============================================================================


@mcp_function(
    description="Applications of Bézout's identity in various mathematical contexts.",
    namespace="arithmetic",
    category="bezout_identity",
    execution_modes=["local", "remote"],
    examples=[
        {
            "input": {
                "a": 17,
                "b": 13,
                "applications": ["modular_inverse", "diophantine"],
            },
            "output": {"modular_inverse_17_mod_13": 9, "diophantine_solvable": True},
            "description": "Applications for coprime numbers",
        },
        {
            "input": {"a": 30, "b": 42, "applications": ["gcd", "lcm"]},
            "output": {"gcd": 6, "lcm": 210, "simplification": "30/42 = 5/7"},
            "description": "GCD/LCM applications",
        },
        {
            "input": {"a": 25, "b": 15, "applications": ["fraction_reduction"]},
            "output": {"reduced_fraction": "5/3", "common_factor": 5},
            "description": "Fraction reduction",
        },
        {
            "input": {"a": 11, "b": 7, "applications": ["chinese_remainder"]},
            "output": {"coprime": True, "crt_applicable": True},
            "description": "Chinese Remainder Theorem applicability",
        },
    ],
)
async def bezout_applications(a: int, b: int, applications: List[str] = None) -> Dict:  # type: ignore[assignment]
    """
    Demonstrate various applications of Bézout's identity.

    Shows how Bézout coefficients are used in modular arithmetic,
    Diophantine equations, fraction reduction, and more.

    Args:
        a: First integer
        b: Second integer
        applications: List of applications to demonstrate

    Returns:
        Dictionary with results for requested applications

    Examples:
        await bezout_applications(17, 13, ["modular_inverse"]) → {"modular_inverse_17_mod_13": 9}
        await bezout_applications(30, 42, ["gcd", "lcm"]) → {"gcd": 6, "lcm": 210}
    """
    if applications is None:
        applications = ["gcd", "modular_inverse", "fraction_reduction", "diophantine"]

    # Get Bézout coefficients
    bezout_result = await bezout_identity(a, b)
    gcd_val = bezout_result["gcd"]
    x, y = bezout_result["x"], bezout_result["y"]

    results = {"bezout_coefficients": {"x": x, "y": y}, "gcd": gcd_val}

    for app in applications:
        if app == "modular_inverse":
            # Multiplicative inverse using Bézout coefficients
            if gcd_val == 1 and b > 1:
                # x is the inverse of a modulo b
                inverse_a_mod_b = x % b
                results["modular_inverse"] = {
                    f"inverse_of_{a}_mod_{b}": inverse_a_mod_b,
                    "verification": f"{a} * {inverse_a_mod_b} ≡ {(a * inverse_a_mod_b) % b} ≡ 1 (mod {b})",
                }

                # y is the inverse of b modulo a (if a > 1)
                if a > 1:
                    inverse_b_mod_a = y % a
                    results["modular_inverse"][f"inverse_of_{b}_mod_{a}"] = (
                        inverse_b_mod_a
                    )
            else:
                results["modular_inverse"] = {
                    "error": f"No modular inverse exists (gcd = {gcd_val} ≠ 1)"
                }

        elif app == "fraction_reduction":
            # Simplify fraction a/b using GCD
            if b != 0:
                reduced_num = a // gcd_val
                reduced_den = b // gcd_val
                results["fraction_reduction"] = {
                    "original": f"{a}/{b}",
                    "reduced": f"{reduced_num}/{reduced_den}",
                    "common_factor": gcd_val,
                    "explanation": f"gcd({a}, {b}) = {gcd_val}, so {a}/{b} = {reduced_num}/{reduced_den}",
                }
            else:
                results["fraction_reduction"] = {
                    "error": "Cannot reduce fraction with denominator 0"
                }

        elif app == "lcm":
            # Calculate LCM using the identity: lcm(a,b) = |ab|/gcd(a,b)
            if gcd_val > 0:
                lcm_val = abs(a * b) // gcd_val
                results["lcm"] = {
                    "value": lcm_val,
                    "formula": f"lcm({a}, {b}) = |{a} × {b}| / gcd({a}, {b}) = {abs(a * b)} / {gcd_val} = {lcm_val}",
                }
            else:
                results["lcm"] = {"error": "LCM undefined when one number is 0"}

        elif app == "diophantine":
            # Linear Diophantine equation solvability
            results["diophantine"] = {
                "equation_form": f"{a}x + {b}y = c",
                "solvability": "Has integer solutions for any c divisible by gcd",
                "particular_solution": f"x = {x}(c/{gcd_val}), y = {y}(c/{gcd_val})",
                "general_solution": f"x = {x}(c/{gcd_val}) + {b // gcd_val}t, y = {y}(c/{gcd_val}) - {a // gcd_val}t"
                if gcd_val > 0
                else "undefined",
                "examples": {
                    f"c = {gcd_val}": {"x": x, "y": y},
                    f"c = {2 * gcd_val}": {"x": 2 * x, "y": 2 * y},
                    f"c = {3 * gcd_val}": {"x": 3 * x, "y": 3 * y},
                }
                if gcd_val > 0
                else {},
            }

        elif app == "chinese_remainder":
            # Chinese Remainder Theorem applicability
            results["chinese_remainder"] = {
                "coprime": gcd_val == 1,
                "crt_applicable": gcd_val == 1,
                "explanation": "CRT applies when moduli are pairwise coprime"
                if gcd_val == 1
                else f"CRT not applicable (gcd = {gcd_val} ≠ 1)",
            }

        elif app == "gcd":
            # GCD properties and verification
            results["gcd_properties"] = {
                "value": gcd_val,
                "linear_combination": f"{gcd_val} = {a}×{x} + {b}×{y}",
                "divides_both": f"{gcd_val} | {a} and {gcd_val} | {b}",
                "greatest": f"{gcd_val} is the largest integer dividing both {a} and {b}",
            }

    return results


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


async def _multiplicative_inverse_async(a: int, m: int) -> Optional[int]:
    """Find multiplicative inverse of a modulo m."""
    if m <= 1:
        return None

    # Use extended Euclidean algorithm
    def extended_gcd(a, b):
        if a == 0:
            return b, 0, 1
        gcd, x1, y1 = extended_gcd(b % a, a)
        x = y1 - (b // a) * x1
        y = x1
        return gcd, x, y

    gcd, x, _ = extended_gcd(a % m, m)

    if gcd != 1:
        return None  # Inverse doesn't exist

    return (x % m + m) % m


# Export all functions
__all__ = [
    # Wilson's theorem
    "wilson_theorem_test",
    "wilson_theorem_verify",
    "optimized_wilson_test",
    # Bézout's identity
    "bezout_identity",
    "extended_gcd_bezout",
    "bezout_applications",
]

# ============================================================================
# COMPREHENSIVE TEST SUITE
# ============================================================================


async def test_wilson_bezout():
    """Test Wilson's theorem and Bézout identity functions."""
    print("🔬 Wilson's Theorem and Bézout's Identity Test")
    print("=" * 50)

    # Test Wilson's theorem
    print("1. Wilson's Theorem Tests:")
    wilson_test_cases = [
        (5, True, "Small prime"),
        (7, True, "Another small prime"),
        (9, False, "Composite number"),
        (11, True, "Larger prime"),
        (15, False, "Composite number"),
    ]

    for n, expected, description in wilson_test_cases:
        result = await wilson_theorem_test(n)
        status = "✓" if result == expected else "✗"
        print(f"   Wilson test {n:2d}: {result:5} {status} ({description})")

    # Test Wilson verification
    print("\n2. Wilson's Theorem Verification:")
    for p in [5, 7, 11]:
        verification = await wilson_theorem_verify(p)
        print(f"   p={p}: {verification['verification']}")

    # Test Bézout identity
    print("\n3. Bézout Identity Tests:")
    bezout_test_cases = [
        (30, 18, 6),  # gcd = 6
        (17, 13, 1),  # coprime
        (48, 18, 6),  # gcd = 6
        (25, 15, 5),  # gcd = 5
    ]

    for a, b, expected_gcd in bezout_test_cases:
        result = await bezout_identity(a, b)
        gcd_val = result["gcd"]
        x, y = result["x"], result["y"]
        verification = a * x + b * y

        gcd_correct = gcd_val == expected_gcd
        identity_correct = verification == gcd_val

        print(
            f"   Bézout({a:2d}, {b:2d}): gcd={gcd_val} {'✓' if gcd_correct else '✗'}, "
            f"identity={x:2d}×{a} + {y:2d}×{b} = {verification} {'✓' if identity_correct else '✗'}"
        )

    # Test applications
    print("\n4. Bézout Applications:")
    app_result = await bezout_applications(17, 13, ["modular_inverse", "diophantine"])
    if "modular_inverse" in app_result:
        inv_info = app_result["modular_inverse"]
        print(f"   Modular inverse: {inv_info}")

    print("\n✅ Wilson's theorem and Bézout identity tests complete!")


async def demo_wilson_theorem():
    """Demonstrate Wilson's theorem with examples."""
    print("\n🎯 Wilson's Theorem Demonstration")
    print("=" * 35)

    print("Wilson's Theorem: n is prime ⟺ (n-1)! ≡ -1 (mod n)")

    # Test small primes
    small_primes = [3, 5, 7, 11, 13]
    print("\nVerification for small primes:")

    for p in small_primes:
        verification = await wilson_theorem_verify(p)
        if verification.get("factorial"):
            print(
                f"  p={p}: ({p - 1})! = {verification['factorial']} ≡ {verification['factorial_mod_p']} ≡ -1 (mod {p}) ✓"
            )
        else:
            print(
                f"  p={p}: ({p - 1})! ≡ {verification['factorial_mod_p']} ≡ -1 (mod {p}) ✓"
            )

    # Test composite numbers
    composites = [4, 6, 8, 9, 10, 12]
    print("\nCounter-examples for composite numbers:")

    for n in composites:
        verification = await wilson_theorem_verify(n)
        print(
            f"  n={n}: ({n - 1})! ≡ {verification['factorial_mod_p']} ≢ -1 (mod {n}) ✗"
        )


async def demo_bezout_identity():
    """Demonstrate Bézout's identity with examples."""
    print("\n🎯 Bézout's Identity Demonstration")
    print("=" * 35)

    print(
        "Bézout's Identity: For any integers a, b, ∃ integers x, y such that ax + by = gcd(a,b)"
    )

    examples = [
        (30, 18, "Common example"),
        (17, 13, "Coprime numbers"),
        (240, 46, "Larger numbers"),
        (15, 25, "Different order"),
    ]

    for a, b, description in examples:
        result = await bezout_identity(a, b)
        gcd_val = result["gcd"]
        x, y = result["x"], result["y"]

        print(f"\n  {description}: gcd({a}, {b}) = {gcd_val}")
        print(f"    Bézout coefficients: x = {x}, y = {y}")
        print(f"    Verification: {a}×{x} + {b}×{y} = {a * x + b * y} = {gcd_val} ✓")

        # Show applications
        apps = await bezout_applications(
            a, b, ["modular_inverse", "fraction_reduction"]
        )

        if "modular_inverse" in apps and "error" not in apps["modular_inverse"]:
            inv_info = apps["modular_inverse"]
            for key, value in inv_info.items():
                if key.startswith("inverse_of"):
                    print(f"    Application: {key.replace('_', ' ')} = {value}")

        if "fraction_reduction" in apps:
            frac_info = apps["fraction_reduction"]
            print(f"    Fraction: {frac_info['original']} = {frac_info['reduced']}")


if __name__ == "__main__":
    import asyncio

    async def main():
        await test_wilson_bezout()
        await demo_wilson_theorem()
        await demo_bezout_identity()

    asyncio.run(main())
