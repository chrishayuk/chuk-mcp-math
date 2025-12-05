#!/usr/bin/env python3
# chuk_mcp_math/number_theory/mobius_inversion.py
"""
Möbius Inversion and Advanced Applications - Async Native - COMPLETE IMPLEMENTATION

Implementation of Möbius inversion formula and its applications in number theory.
Essential for advanced arithmetic functions and analytic number theory.

Functions:
- Möbius inversion: mobius_inversion_formula, apply_mobius_inversion
- Applications: euler_totient_inversion, divisor_function_inversion
- Advanced: multiplicative_function_analysis, number_theoretic_transforms

Mathematical Background:
The Möbius inversion formula allows recovery of arithmetic functions from their summatory functions.
If g(n) = Σ_{d|n} f(d), then f(n) = Σ_{d|n} μ(n/d) g(d), where μ is the Möbius function.
"""

import asyncio
import math
from typing import List, Dict
from chuk_mcp_math.mcp_decorator import mcp_function

# ============================================================================
# MÖBIUS FUNCTION AND BASIC OPERATIONS
# ============================================================================


@mcp_function(
    description="Calculate Möbius function values for a range of numbers efficiently.",
    namespace="arithmetic",
    category="mobius_functions",
    execution_modes=["local", "remote"],
    estimated_cpu_usage="medium",
    examples=[
        {
            "input": {"n": 10},
            "output": {1: 1, 2: -1, 3: -1, 4: 0, 5: -1, 6: 1, 7: -1, 8: 0, 9: 0, 10: 1},
            "description": "Möbius values up to 10",
        },
        {
            "input": {"n": 20},
            "output": "μ(1)=1, μ(2)=-1, μ(6)=1, μ(10)=1, etc.",
            "description": "Extended range",
        },
        {
            "input": {"n": 5},
            "output": {1: 1, 2: -1, 3: -1, 4: 0, 5: -1},
            "description": "Small range",
        },
        {
            "input": {"n": 15},
            "output": {1: 1, 6: 1, 10: 1, 14: 1, 15: 1},
            "description": "Positive values up to 15",
        },
    ],
)
async def mobius_function_range(n: int) -> Dict[int, int]:
    """
    Calculate Möbius function μ(k) for all k from 1 to n.

    Uses sieve-like algorithm for efficient computation of multiple values.
    μ(n) = 1 if n is square-free with even number of prime factors
    μ(n) = -1 if n is square-free with odd number of prime factors
    μ(n) = 0 if n is not square-free

    Args:
        n: Upper bound for Möbius function calculation

    Returns:
        Dictionary mapping integers to their Möbius function values

    Examples:
        await mobius_function_range(10) → {1: 1, 2: -1, 3: -1, 4: 0, 5: -1, 6: 1, 7: -1, 8: 0, 9: 0, 10: 1}
        await mobius_function_range(5) → {1: 1, 2: -1, 3: -1, 4: 0, 5: -1}
    """
    if n < 1:
        return {}

    # Initialize all values to 1
    mu = [1] * (n + 1)

    # Yield control for large ranges
    if n > 100000:
        await asyncio.sleep(0)

    # Sieve to compute Möbius function
    for i in range(2, n + 1):
        if mu[i] == 1:  # i is prime (not yet modified)
            # Mark multiples of i
            for j in range(i, n + 1, i):
                mu[j] *= -1

            # Mark multiples of i² as 0 (not square-free)
            square = i * i
            for j in range(square, n + 1, square):
                mu[j] = 0

        # Yield control every 1000 iterations for large n
        if i % 1000 == 0 and n > 100000:
            await asyncio.sleep(0)

    # Convert to dictionary (excluding 0)
    return {k: mu[k] for k in range(1, n + 1)}


@mcp_function(
    description="Apply Möbius inversion formula to recover function from its summatory function.",
    namespace="arithmetic",
    category="mobius_functions",
    execution_modes=["local", "remote"],
    estimated_cpu_usage="high",
    examples=[
        {
            "input": {"g_values": {1: 1, 2: 3, 3: 4, 4: 7, 6: 12}, "n": 6},
            "output": {1: 1, 2: 2, 3: 1, 4: 4, 6: 2},
            "description": "Invert summatory function",
        },
        {
            "input": {"g_values": {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6}, "n": 6},
            "output": {1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1},
            "description": "Identity function inversion",
        },
        {
            "input": {"g_values": {1: 1, 2: 4, 3: 7, 4: 13}, "n": 4},
            "output": {1: 1, 2: 3, 3: 3, 4: 6},
            "description": "Small example",
        },
        {
            "input": {"g_values": {1: 2, 2: 6, 4: 14, 8: 30}, "n": 8},
            "output": {1: 2, 2: 4, 4: 8, 8: 16},
            "description": "Powers of 2",
        },
    ],
)
async def mobius_inversion_formula(g_values: Dict[int, int], n: int) -> Dict[int, int]:
    """
    Apply Möbius inversion to recover f from g where g(n) = Σ_{d|n} f(d).

    Uses the classical inversion formula: f(n) = Σ_{d|n} μ(n/d) g(d)

    Args:
        g_values: Dictionary of g(k) values for various k
        n: Maximum value to compute f(k) for

    Returns:
        Dictionary of recovered f(k) values

    Mathematical formula: f(n) = Σ_{d|n} μ(n/d) g(d)

    Examples:
        await mobius_inversion_formula({1: 1, 2: 3, 4: 7}, 4) → {1: 1, 2: 2, 4: 4}
        await mobius_inversion_formula({1: 1, 2: 2, 3: 3}, 3) → {1: 1, 2: 1, 3: 1}
    """
    if n < 1:
        return {}

    # Get Möbius function values
    mu_values = await mobius_function_range(n)

    # Yield control for large computations
    if n > 1000:
        await asyncio.sleep(0)

    f_values = {}

    for k in range(1, n + 1):
        if k in g_values:
            f_k = 0

            # Find all divisors of k
            divisors = await _get_divisors_async(k)

            for d in divisors:
                if d in g_values:
                    quotient = k // d
                    f_k += mu_values[quotient] * g_values[d]

            f_values[k] = f_k

        # Yield control every 100 iterations for large n
        if k % 100 == 0 and n > 10000:
            await asyncio.sleep(0)

    return f_values


@mcp_function(
    description="Apply Möbius inversion to a function defined by a formula.",
    namespace="arithmetic",
    category="mobius_functions",
    execution_modes=["local", "remote"],
    estimated_cpu_usage="high",
    examples=[
        {
            "input": {"formula": "lambda d: d", "n": 6},
            "output": {1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1},
            "description": "Invert divisor sum function",
        },
        {
            "input": {"formula": "lambda d: 1", "n": 5},
            "output": {1: 1, 2: -1, 3: -1, 4: 0, 5: -1},
            "description": "Invert constant function (gives Möbius)",
        },
        {
            "input": {"formula": "lambda d: d*d", "n": 4},
            "output": "f values from d² summatory",
            "description": "Quadratic function",
        },
        {
            "input": {"formula": "lambda d: 2**d", "n": 3},
            "output": "f values from 2^d summatory",
            "description": "Exponential function",
        },
    ],
)
async def apply_mobius_inversion(formula: str, n: int, description: str = "") -> Dict:
    """
    Apply Möbius inversion to a function defined by a formula.

    Given a formula for f(d), computes g(n) = Σ_{d|n} f(d), then applies
    Möbius inversion to recover the original function.

    Args:
        formula: String representation of function (e.g., "lambda d: d")
        n: Maximum value to compute for
        description: Human-readable description of the function

    Returns:
        Dictionary with original function, summatory function, and inverted result

    Examples:
        await apply_mobius_inversion("lambda d: d", 6) → {"original": {...}, "summatory": {...}, "inverted": {...}}
        await apply_mobius_inversion("lambda d: 1", 5) → Möbius function values
    """
    try:
        # Parse the formula
        f_func = eval(formula)
    except Exception:
        return {"error": f"Invalid formula: {formula}"}

    if n < 1:
        return {"error": "n must be positive"}

    # Compute original function values
    original_f = {}
    for k in range(1, n + 1):
        try:
            original_f[k] = f_func(k)
        except Exception:
            original_f[k] = 0

    # Compute summatory function g(n) = Σ_{d|n} f(d)
    summatory_g = {}

    for k in range(1, n + 1):
        divisors = await _get_divisors_async(k)
        g_k = sum(original_f.get(d, 0) for d in divisors)
        summatory_g[k] = g_k

        # Yield control every 100 iterations
        if k % 100 == 0 and n > 1000:
            await asyncio.sleep(0)

    # Apply Möbius inversion
    inverted_f = await mobius_inversion_formula(summatory_g, n)

    # Verify inversion worked
    inversion_error = {}
    for k in range(1, n + 1):
        if k in original_f and k in inverted_f:
            error = abs(original_f[k] - inverted_f[k])
            if error > 0:
                inversion_error[k] = error

    return {
        "formula": formula,
        "description": description or "User-defined function",
        "n": n,
        "original_function": original_f,
        "summatory_function": summatory_g,
        "inverted_function": inverted_f,
        "inversion_successful": len(inversion_error) == 0,
        "inversion_errors": inversion_error if inversion_error else "None",
    }


# ============================================================================
# APPLICATIONS OF MÖBIUS INVERSION
# ============================================================================


@mcp_function(
    description="Derive Euler's totient function using Möbius inversion.",
    namespace="arithmetic",
    category="mobius_functions",
    execution_modes=["local", "remote"],
    examples=[
        {
            "input": {"n": 12},
            "output": {
                "phi": 4,
                "derivation": "φ(12) = 12 × Σ μ(d)/d for d|12",
                "verification": True,
            },
            "description": "Totient via Möbius inversion",
        },
        {
            "input": {"n": 15},
            "output": {"phi": 8, "mobius_formula": "φ(n) = n × Σ_{d|n} μ(d)/d"},
            "description": "Another totient example",
        },
        {
            "input": {"n": 10},
            "output": {"phi": 4, "step_by_step": "detailed calculation"},
            "description": "Step-by-step derivation",
        },
        {
            "input": {"n": 6},
            "output": {"phi": 2, "comparison": "direct vs Möbius"},
            "description": "Method comparison",
        },
    ],
)
async def euler_totient_inversion(n: int) -> Dict:
    """
    Compute Euler's totient function using Möbius inversion.

    Uses the formula: φ(n) = n × Σ_{d|n} μ(d)/d
    This demonstrates how totient function can be derived via inversion.

    Args:
        n: Number to compute φ(n) for

    Returns:
        Dictionary with totient value and derivation details

    Examples:
        await euler_totient_inversion(12) → {"phi": 4, "derivation": "...", ...}
        await euler_totient_inversion(15) → {"phi": 8, "mobius_formula": "...", ...}
    """
    if n < 1:
        return {"error": "n must be positive"}

    # Get divisors and Möbius values
    divisors = await _get_divisors_async(n)
    mu_values = await mobius_function_range(n)

    # Calculate φ(n) = n × Σ_{d|n} μ(d)/d
    mobius_sum = 0
    calculation_steps = []

    for d in divisors:
        mu_d = mu_values[d]
        term = mu_d / d
        mobius_sum += term

        calculation_steps.append(
            {
                "divisor": d,
                "mobius_value": mu_d,
                "term": f"μ({d})/{d} = {mu_d}/{d} = {term:.6f}",
            }
        )

    phi_n = int(n * mobius_sum)

    # Verify with direct calculation
    direct_phi = await _euler_totient_direct_async(n)
    verification_passed = phi_n == direct_phi

    return {
        "n": n,
        "phi": phi_n,
        "mobius_formula": f"φ({n}) = {n} × Σ_{{d|{n}}} μ(d)/d",
        "divisors": divisors,
        "mobius_sum": round(mobius_sum, 6),
        "calculation_steps": calculation_steps,
        "verification": {
            "mobius_result": phi_n,
            "direct_result": direct_phi,
            "passed": verification_passed,
        },
        "formula_explanation": "The totient function counts numbers ≤ n coprime to n",
    }


@mcp_function(
    description="Analyze divisor functions using Möbius inversion techniques.",
    namespace="arithmetic",
    category="mobius_functions",
    execution_modes=["local", "remote"],
    examples=[
        {
            "input": {"n": 12, "function_type": "count"},
            "output": {"divisor_count": 6, "inversion_derivation": "τ(n) via Möbius"},
            "description": "Divisor count function",
        },
        {
            "input": {"n": 10, "function_type": "sum"},
            "output": {"divisor_sum": 18, "mobius_approach": "σ(n) analysis"},
            "description": "Divisor sum function",
        },
        {
            "input": {"n": 8, "function_type": "power_sum", "power": 2},
            "output": {"power_sum": 85, "formula": "σ₂(n)"},
            "description": "Sum of squares of divisors",
        },
        {
            "input": {"n": 6, "function_type": "all"},
            "output": {"count": 4, "sum": 12, "analysis": "complete"},
            "description": "All divisor functions",
        },
    ],
)
async def divisor_function_inversion(
    n: int, function_type: str = "count", power: int = 1
) -> Dict:
    """
    Analyze divisor functions using Möbius inversion relationships.

    Explores how various divisor functions can be expressed and inverted
    using Möbius function techniques.

    Args:
        n: Number to analyze
        function_type: Type of divisor function ("count", "sum", "power_sum", "all")
        power: Power for power sum (σₖ(n))

    Returns:
        Dictionary with analysis results and Möbius relationships

    Examples:
        await divisor_function_inversion(12, "count") → {"divisor_count": 6, ...}
        await divisor_function_inversion(10, "sum") → {"divisor_sum": 18, ...}
    """
    if n < 1:
        return {"error": "n must be positive"}

    divisors = await _get_divisors_async(n)
    results = {"n": n, "divisors": divisors}

    if function_type in ["count", "all"]:
        # Divisor count function τ(n)
        tau_n = len(divisors)
        results["divisor_count"] = {
            "value": tau_n,
            "formula": f"τ({n}) = {tau_n}",
            "mobius_relation": "Related to sum of μ²(d) over divisors",
            "divisors_list": divisors,
        }

    if function_type in ["sum", "all"]:
        # Divisor sum function σ(n)
        sigma_n = sum(divisors)
        results["divisor_sum"] = {
            "value": sigma_n,
            "formula": f"σ({n}) = {' + '.join(map(str, divisors))} = {sigma_n}",
            "mobius_relation": "Can be inverted to find multiplicative inverse",
            "average_divisor": sigma_n / len(divisors),
        }

    if function_type in ["power_sum", "all"]:
        # Power sum σₖ(n)
        power_sum = sum(d**power for d in divisors)
        results["power_sum"] = {
            "power": power,
            "value": power_sum,
            "formula": f"σ_{power}({n}) = {' + '.join(f'{d}^{power}' for d in divisors)} = {power_sum}",
            "mobius_inversion": f"Related to inversion of n^{power} summatory function",
        }

    # Möbius function analysis
    mu_values = await mobius_function_range(n)
    divisor_mobius = {d: mu_values[d] for d in divisors}

    results["mobius_analysis"] = {
        "divisor_mobius_values": divisor_mobius,
        "mobius_sum": sum(divisor_mobius.values()),
        "interpretation": "Möbius sum = 1 if n=1, else 0 (fundamental property)",
    }

    return results


# ============================================================================
# ADVANCED APPLICATIONS
# ============================================================================


@mcp_function(
    description="Analyze multiplicative functions using Möbius techniques.",
    namespace="arithmetic",
    category="mobius_functions",
    execution_modes=["local", "remote"],
    estimated_cpu_usage="high",
    examples=[
        {
            "input": {"function_name": "totient", "range_end": 20},
            "output": {"multiplicative": True, "mobius_representation": "φ(n) formula"},
            "description": "Totient function analysis",
        },
        {
            "input": {"function_name": "mobius", "range_end": 15},
            "output": {
                "completely_multiplicative": False,
                "zero_values": [4, 8, 9, 12],
            },
            "description": "Möbius function properties",
        },
        {
            "input": {"function_name": "divisor_count", "range_end": 12},
            "output": {"multiplicative": True, "prime_power_formula": "τ(p^k) = k+1"},
            "description": "Divisor count analysis",
        },
        {
            "input": {"function_name": "unit", "range_end": 10},
            "output": {"completely_multiplicative": True, "constant": 1},
            "description": "Constant function",
        },
    ],
)
async def multiplicative_function_analysis(
    function_name: str, range_end: int = 20
) -> Dict:
    """
    Analyze multiplicative properties of arithmetic functions using Möbius methods.

    Studies whether functions are multiplicative, completely multiplicative,
    and how they relate to Möbius inversion.

    Args:
        function_name: Name of function to analyze ("totient", "mobius", "divisor_count", etc.)
        range_end: Range to analyze function over

    Returns:
        Dictionary with multiplicativity analysis and Möbius relationships

    Examples:
        await multiplicative_function_analysis("totient", 20) → {"multiplicative": True, ...}
        await multiplicative_function_analysis("mobius", 15) → {"completely_multiplicative": False, ...}
    """
    if range_end < 2:
        return {"error": "Range must be at least 2"}

    # Generate function values
    function_values = {}

    for n in range(1, range_end + 1):
        if function_name == "totient":
            function_values[n] = await _euler_totient_direct_async(n)
        elif function_name == "mobius":
            mu_vals = await mobius_function_range(n)
            function_values[n] = mu_vals[n]
        elif function_name == "divisor_count":
            divisors = await _get_divisors_async(n)
            function_values[n] = len(divisors)
        elif function_name == "divisor_sum":
            divisors = await _get_divisors_async(n)
            function_values[n] = sum(divisors)
        elif function_name == "unit":
            function_values[n] = 1
        else:
            return {"error": f"Unknown function: {function_name}"}

        # Yield control every 50 iterations
        if n % 50 == 0:
            await asyncio.sleep(0)

    # Test multiplicativity
    multiplicative_violations = []
    completely_multiplicative_violations = []

    for a in range(1, range_end // 2 + 1):
        for b in range(a, range_end // a + 1):
            if a * b <= range_end:
                gcd_ab = await _gcd_async(a, b)

                # Test multiplicative property: f(ab) = f(a)f(b) when gcd(a,b) = 1
                if gcd_ab == 1:
                    f_ab = function_values[a * b]
                    f_a_f_b = function_values[a] * function_values[b]

                    if f_ab != f_a_f_b:
                        multiplicative_violations.append((a, b, f_ab, f_a_f_b))

                # Test completely multiplicative: f(ab) = f(a)f(b) always
                f_ab = function_values[a * b]
                f_a_f_b = function_values[a] * function_values[b]

                if f_ab != f_a_f_b:
                    completely_multiplicative_violations.append((a, b, f_ab, f_a_f_b))

    is_multiplicative = len(multiplicative_violations) == 0
    is_completely_multiplicative = len(completely_multiplicative_violations) == 0

    # Analyze zeros and special values
    zero_values = [n for n, val in function_values.items() if val == 0]
    unit_values = [n for n, val in function_values.items() if val == 1]
    negative_values = [n for n, val in function_values.items() if val < 0]

    results = {
        "function_name": function_name,
        "range_analyzed": f"1 to {range_end}",
        "function_values": function_values,
        "multiplicative": is_multiplicative,
        "completely_multiplicative": is_completely_multiplicative,
        "special_values": {
            "zeros": zero_values,
            "units": unit_values,
            "negatives": negative_values,
        },
    }

    if multiplicative_violations:
        results["multiplicative_violations"] = multiplicative_violations[
            :5
        ]  # Show first 5

    if completely_multiplicative_violations:
        results["completely_multiplicative_violations"] = (
            completely_multiplicative_violations[:5]
        )

    # Add function-specific analysis
    if function_name == "totient":
        results["mobius_formula"] = "φ(n) = n × Σ_{d|n} μ(d)/d"
        results["euler_product"] = "φ(n) = n × ∏_{p|n} (1 - 1/p)"

    elif function_name == "mobius":
        results["definition"] = "μ(n) = 0 if n not square-free, (-1)^k if n = p₁p₂...pₖ"
        results["inversion_property"] = "Σ_{d|n} μ(d) = 1 if n=1, else 0"

    elif function_name == "divisor_count":
        results["prime_power_formula"] = "τ(p^k) = k + 1"
        results["mobius_relation"] = "Related to Σ_{d|n} μ²(d)"

    return results


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


async def _get_divisors_async(n: int) -> List[int]:
    """Get all positive divisors of n."""
    if n <= 0:
        return []

    divisors = []
    sqrt_n = int(math.sqrt(n))

    for i in range(1, sqrt_n + 1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)

    return sorted(divisors)


async def _euler_totient_direct_async(n: int) -> int:
    """Direct calculation of Euler's totient function."""
    if n <= 0:
        return 0
    if n == 1:
        return 1

    result = n
    p = 2

    while p * p <= n:
        if n % p == 0:
            while n % p == 0:
                n //= p
            result -= result // p
        p += 1

    if n > 1:
        result -= result // n

    return result


async def _gcd_async(a: int, b: int) -> int:
    """Async wrapper for GCD calculation."""
    while b:
        a, b = b, a % b
    return abs(a)


# Export all functions
__all__ = [
    # Basic Möbius operations
    "mobius_function_range",
    "mobius_inversion_formula",
    "apply_mobius_inversion",
    # Applications
    "euler_totient_inversion",
    "divisor_function_inversion",
    # Advanced analysis
    "multiplicative_function_analysis",
]

# ============================================================================
# COMPREHENSIVE TEST SUITE
# ============================================================================


async def test_mobius_inversion():
    """Test all Möbius inversion functions."""
    print("🔬 Möbius Inversion Test Suite")
    print("=" * 35)

    # Test Möbius function
    print("1. Möbius Function Values:")
    mu_vals = await mobius_function_range(12)
    for n in range(1, 13):
        print(f"   μ({n:2d}) = {mu_vals[n]:2d}")

    # Test basic inversion
    print("\n2. Basic Möbius Inversion:")
    # Test with g(n) = n (should give f(n) = φ(n))
    g_values = {n: n for n in range(1, 7)}
    f_recovered = await mobius_inversion_formula(g_values, 6)

    for n in range(1, 7):
        if n in f_recovered:
            phi_n = await _euler_totient_direct_async(n)
            print(
                f"   n={n}: recovered f({n})={f_recovered[n]}, φ({n})={phi_n} {'✓' if f_recovered[n] == phi_n else '✗'}"
            )

    # Test totient inversion
    print("\n3. Euler Totient via Möbius Inversion:")
    for n in [6, 10, 12]:
        result = await euler_totient_inversion(n)
        print(
            f"   φ({n}) = {result['phi']} (verification: {result['verification']['passed']})"
        )

    # Test multiplicative function analysis
    print("\n4. Multiplicative Function Analysis:")
    analysis = await multiplicative_function_analysis("totient", 10)
    print(f"   Totient function is multiplicative: {analysis['multiplicative']}")
    print(f"   Completely multiplicative: {analysis['completely_multiplicative']}")

    print("\n✅ Möbius inversion tests complete!")


if __name__ == "__main__":
    import asyncio

    asyncio.run(test_mobius_inversion())
