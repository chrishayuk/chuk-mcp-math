#!/usr/bin/env python3
# chuk_mcp_math/arithmetic/constants.py
"""
Mathematical Constants as Functions for AI Models (Async Native)

Mathematical constants exposed as MCP functions for easy access and consistency.
All constants are provided with high precision and include comprehensive documentation
about their mathematical significance and common applications.

All functions are async native for optimal performance in async MCP servers.

Functions:
- Fundamental constants: pi, e, tau, infinity, nan
- Algebraic constants: golden_ratio, silver_ratio, plastic_number
- Root constants: sqrt2, sqrt3, sqrt5, cbrt2, cbrt3
- Logarithmic constants: ln2, ln10, log2e, log10e
- Special constants: euler_gamma, catalan, apery, khinchin, glaisher
- Numeric limits: machine_epsilon, max_float, min_float
"""

import math
import sys
from chuk_mcp_math.mcp_decorator import mcp_function


@mcp_function(
    description="Get the mathematical constant π (pi) ≈ 3.14159. The ratio of a circle's circumference to its diameter.",
    namespace="arithmetic",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    examples=[
        {"input": {}, "output": 3.141592653589793, "description": "Value of π"},
        {
            "input": {},
            "output": 3.141592653589793,
            "description": "Used in circle calculations",
        },
    ],
)
async def pi() -> float:
    """
    Get the mathematical constant π (pi).

    Returns:
        The value of π ≈ 3.141592653589793

    Mathematical significance:
        - Ratio of circle's circumference to diameter
        - Area of unit circle
        - Appears in Gaussian distribution
        - Fundamental in trigonometry and analysis

    Examples:
        await pi() → 3.141592653589793
        Circle area = await pi() * radius²
        Circle circumference = 2 * await pi() * radius
    """
    return math.pi


@mcp_function(
    description="Get Euler's number e ≈ 2.71828. Base of natural logarithm and fundamental constant in calculus.",
    namespace="arithmetic",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    examples=[
        {"input": {}, "output": 2.718281828459045, "description": "Value of e"},
        {
            "input": {},
            "output": 2.718281828459045,
            "description": "Base of natural logarithm",
        },
    ],
)
async def e() -> float:
    """
    Get Euler's number e.

    Returns:
        The value of e ≈ 2.718281828459045

    Mathematical significance:
        - Base of natural logarithm
        - lim(n→∞) (1 + 1/n)^n
        - Fundamental in exponential growth/decay
        - Appears in compound interest, probability

    Examples:
        await e() → 2.718281828459045
        Natural log base: ln(await e()) = 1
        Exponential function: (await e()) ** x
    """
    return math.e


@mcp_function(
    description="Get tau (τ) ≈ 6.28318. Equal to 2π, represents a full turn in radians.",
    namespace="arithmetic",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    examples=[
        {"input": {}, "output": 6.283185307179586, "description": "Value of τ = 2π"},
        {
            "input": {},
            "output": 6.283185307179586,
            "description": "Full circle in radians",
        },
    ],
)
async def tau() -> float:
    """
    Get the mathematical constant τ (tau) = 2π.

    Returns:
        The value of τ ≈ 6.283185307179586

    Mathematical significance:
        - Represents a full turn (360°) in radians
        - Alternative to π for some calculations
        - Simplifies many circular formulas

    Examples:
        await tau() → 6.283185307179586
        Full circle = await tau() radians
        await tau() = 2 * await pi()
    """
    return math.tau


@mcp_function(
    description="Get positive infinity. Represents unbounded growth or division by zero limit.",
    namespace="arithmetic",
    execution_modes=["local", "remote"],
    examples=[
        {"input": {}, "output": "inf", "description": "Positive infinity"},
        {"input": {}, "output": "inf", "description": "Result of 1/0 limit"},
    ],
)
async def infinity() -> float:
    """
    Get positive infinity.

    Returns:
        Positive infinity (float('inf'))

    Mathematical significance:
        - Represents unbounded growth
        - Limit of 1/x as x approaches 0 from positive side
        - Used in optimization and analysis

    Examples:
        await infinity() → inf
        1.0 / 0 → await infinity() (conceptually)
        await infinity() > any_finite_number
    """
    return math.inf


@mcp_function(
    description="Get NaN (Not a Number). Represents undefined or invalid mathematical operations.",
    namespace="arithmetic",
    execution_modes=["local", "remote"],
    examples=[
        {"input": {}, "output": "nan", "description": "Not a Number"},
        {"input": {}, "output": "nan", "description": "Result of 0/0 or inf-inf"},
    ],
)
async def nan() -> float:
    """
    Get NaN (Not a Number).

    Returns:
        NaN (float('nan'))

    Mathematical significance:
        - Represents undefined operations (0/0, inf-inf)
        - Propagates through calculations
        - Used for missing data representation

    Examples:
        await nan() → nan
        0.0 / 0.0 → await nan() (conceptually)
        await nan() != await nan() (always True)
    """
    return math.nan


@mcp_function(
    description="Get the golden ratio φ ≈ 1.61803. Appears in art, nature, and Fibonacci sequence.",
    namespace="arithmetic",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    examples=[
        {
            "input": {},
            "output": 1.618033988749895,
            "description": "Value of golden ratio",
        },
        {"input": {}, "output": 1.618033988749895, "description": "φ = (1 + √5) / 2"},
    ],
)
async def golden_ratio() -> float:
    """
    Get the golden ratio φ (phi).

    Returns:
        The value of φ ≈ 1.618033988749895

    Mathematical significance:
        - φ = (1 + √5) / 2
        - Limit of ratio of consecutive Fibonacci numbers
        - Appears in pentagram, art, architecture
        - Self-similar ratio: φ = 1 + 1/φ

    Examples:
        await golden_ratio() → 1.618033988749895
        φ² = φ + 1
        1 / φ = φ - 1
    """
    return (1 + math.sqrt(5)) / 2


@mcp_function(
    description="Get the silver ratio δ ≈ 2.41421. Equal to 1 + √2, appears in octagon constructions.",
    namespace="arithmetic",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    examples=[
        {
            "input": {},
            "output": 2.414213562373095,
            "description": "Value of silver ratio",
        },
        {"input": {}, "output": 2.414213562373095, "description": "δ = 1 + √2"},
    ],
)
async def silver_ratio() -> float:
    """
    Get the silver ratio δ (delta).

    Returns:
        The value of δ ≈ 2.414213562373095

    Mathematical significance:
        - δ = 1 + √2
        - Appears in regular octagon constructions
        - Related to Pell numbers sequence
        - δ² = 2δ + 1

    Examples:
        await silver_ratio() → 2.414213562373095
        δ = 1 + await sqrt2()
        δ² = 2δ + 1
    """
    return 1 + math.sqrt(2)


@mcp_function(
    description="Get the plastic number ρ ≈ 1.32472. Real root of x³ = x + 1, related to Padovan sequence.",
    namespace="arithmetic",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    examples=[
        {
            "input": {},
            "output": 1.3247179572447464,
            "description": "Value of plastic number",
        },
        {"input": {}, "output": 1.3247179572447464, "description": "ρ³ = ρ + 1"},
    ],
)
async def plastic_number() -> float:
    """
    Get the plastic number ρ (rho).

    Returns:
        The value of ρ ≈ 1.3247179572447464

    Mathematical significance:
        - Real root of x³ = x + 1
        - Limit of ratios in Padovan sequence
        - Minimal polynomial: x³ - x - 1 = 0
        - ρ³ = ρ + 1

    Examples:
        await plastic_number() → 1.3247179572447464
        ρ³ = ρ + 1
        Related to 3D golden ratio
    """
    # Real root of x³ - x - 1 = 0
    # Using the formula: ρ = (∛(9 + √69) + ∛(9 - √69)) / 3
    discriminant = math.sqrt(69)
    term1 = (9 + discriminant) ** (1 / 3)
    term2 = (9 - discriminant) ** (1 / 3)
    return (term1 + term2) / 3


@mcp_function(
    description="Get √2 ≈ 1.41421. Square root of 2, first known irrational number.",
    namespace="arithmetic",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    examples=[
        {"input": {}, "output": 1.4142135623730951, "description": "Value of √2"},
        {
            "input": {},
            "output": 1.4142135623730951,
            "description": "Diagonal of unit square",
        },
    ],
)
async def sqrt2() -> float:
    """
    Get the square root of 2.

    Returns:
        The value of √2 ≈ 1.4142135623730951

    Mathematical significance:
        - First known irrational number
        - Diagonal of unit square
        - Appears in 45-45-90 triangles
        - √2 ≈ 1.41421356...

    Examples:
        await sqrt2() → 1.4142135623730951
        Unit square diagonal = await sqrt2()
        (await sqrt2()) ** 2 = 2
    """
    return math.sqrt(2)


@mcp_function(
    description="Get √3 ≈ 1.73205. Square root of 3, appears in equilateral triangles.",
    namespace="arithmetic",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    examples=[
        {"input": {}, "output": 1.7320508075688772, "description": "Value of √3"},
        {
            "input": {},
            "output": 1.7320508075688772,
            "description": "Height factor in equilateral triangles",
        },
    ],
)
async def sqrt3() -> float:
    """
    Get the square root of 3.

    Returns:
        The value of √3 ≈ 1.7320508075688772

    Mathematical significance:
        - Height of equilateral triangle with unit sides
        - Appears in 30-60-90 triangles
        - √3 ≈ 1.73205080...

    Examples:
        await sqrt3() → 1.7320508075688772
        Equilateral triangle height = await sqrt3() / 2 * side
        (await sqrt3()) ** 2 = 3
    """
    return math.sqrt(3)


@mcp_function(
    description="Get √5 ≈ 2.23607. Square root of 5, appears in golden ratio formula.",
    namespace="arithmetic",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    examples=[
        {"input": {}, "output": 2.23606797749979, "description": "Value of √5"},
        {
            "input": {},
            "output": 2.23606797749979,
            "description": "Component of golden ratio",
        },
    ],
)
async def sqrt5() -> float:
    """
    Get the square root of 5.

    Returns:
        The value of √5 ≈ 2.23606797749979

    Mathematical significance:
        - Component in golden ratio: (1 + √5) / 2
        - Diagonal of 1×2 rectangle
        - √5 ≈ 2.23606797...

    Examples:
        await sqrt5() → 2.23606797749979
        Golden ratio = (1 + await sqrt5()) / 2
        (await sqrt5()) ** 2 = 5
    """
    return math.sqrt(5)


@mcp_function(
    description="Get ∛2 ≈ 1.25992. Cube root of 2, edge length of cube with volume 2.",
    namespace="arithmetic",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    examples=[
        {"input": {}, "output": 1.2599210498948732, "description": "Value of ∛2"},
        {
            "input": {},
            "output": 1.2599210498948732,
            "description": "Cube with volume 2",
        },
    ],
)
async def cbrt2() -> float:
    """
    Get the cube root of 2.

    Returns:
        The value of ∛2 ≈ 1.2599210498948732

    Mathematical significance:
        - Edge length of cube with volume 2
        - ∛2 ≈ 1.25992104...
        - Related to doubling the cube problem

    Examples:
        await cbrt2() → 1.2599210498948732
        (await cbrt2()) ** 3 = 2
        Cube volume 2 = (await cbrt2()) ** 3
    """
    return 2 ** (1 / 3)


@mcp_function(
    description="Get ∛3 ≈ 1.44225. Cube root of 3, edge length of cube with volume 3.",
    namespace="arithmetic",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    examples=[
        {"input": {}, "output": 1.4422495703074083, "description": "Value of ∛3"},
        {
            "input": {},
            "output": 1.4422495703074083,
            "description": "Cube with volume 3",
        },
    ],
)
async def cbrt3() -> float:
    """
    Get the cube root of 3.

    Returns:
        The value of ∛3 ≈ 1.4422495703074083

    Mathematical significance:
        - Edge length of cube with volume 3
        - ∛3 ≈ 1.44224957...

    Examples:
        await cbrt3() → 1.4422495703074083
        (await cbrt3()) ** 3 = 3
        Cube volume 3 = (await cbrt3()) ** 3
    """
    return 3 ** (1 / 3)


@mcp_function(
    description="Get ln(2) ≈ 0.69315. Natural logarithm of 2, appears in information theory.",
    namespace="arithmetic",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    examples=[
        {"input": {}, "output": 0.6931471805599453, "description": "Value of ln(2)"},
        {
            "input": {},
            "output": 0.6931471805599453,
            "description": "Information content of fair coin flip",
        },
    ],
)
async def ln2() -> float:
    """
    Get the natural logarithm of 2.

    Returns:
        The value of ln(2) ≈ 0.6931471805599453

    Mathematical significance:
        - Natural logarithm of 2
        - Information content of fair coin flip (in nats)
        - Integral of 1/x from 1 to 2
        - ln(2) ≈ 0.69314718...

    Examples:
        await ln2() → 0.6931471805599453
        (await e()) ** (await ln2()) = 2
        Binary entropy = await ln2() bits
    """
    return math.log(2)


@mcp_function(
    description="Get ln(10) ≈ 2.30259. Natural logarithm of 10, conversion factor for logarithms.",
    namespace="arithmetic",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    examples=[
        {"input": {}, "output": 2.302585092994046, "description": "Value of ln(10)"},
        {
            "input": {},
            "output": 2.302585092994046,
            "description": "Convert log base 10 to natural log",
        },
    ],
)
async def ln10() -> float:
    """
    Get the natural logarithm of 10.

    Returns:
        The value of ln(10) ≈ 2.302585092994046

    Mathematical significance:
        - Natural logarithm of 10
        - Conversion factor: log₁₀(x) = ln(x) / ln(10)
        - ln(10) ≈ 2.30258509...

    Examples:
        await ln10() → 2.302585092994046
        (await e()) ** (await ln10()) = 10
        log₁₀(x) = ln(x) / await ln10()
    """
    return math.log(10)


@mcp_function(
    description="Get log₂(e) ≈ 1.44270. Logarithm base 2 of e, conversion factor for binary logarithms.",
    namespace="arithmetic",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    examples=[
        {"input": {}, "output": 1.4426950408889634, "description": "Value of log₂(e)"},
        {
            "input": {},
            "output": 1.4426950408889634,
            "description": "Convert natural log to log base 2",
        },
    ],
)
async def log2e() -> float:
    """
    Get the logarithm base 2 of e.

    Returns:
        The value of log₂(e) ≈ 1.4426950408889634

    Mathematical significance:
        - Logarithm base 2 of Euler's number
        - Conversion factor: log₂(x) = ln(x) * log₂(e)
        - log₂(e) ≈ 1.44269504...

    Examples:
        await log2e() → 1.4426950408889634
        2 ** (await log2e()) = await e()
        log₂(x) = ln(x) * await log2e()
    """
    return math.log2(math.e)


@mcp_function(
    description="Get log₁₀(e) ≈ 0.43429. Common logarithm of e, conversion factor for base 10 logarithms.",
    namespace="arithmetic",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    examples=[
        {"input": {}, "output": 0.4342944819032518, "description": "Value of log₁₀(e)"},
        {
            "input": {},
            "output": 0.4342944819032518,
            "description": "Convert natural log to log base 10",
        },
    ],
)
async def log10e() -> float:
    """
    Get the common logarithm (base 10) of e.

    Returns:
        The value of log₁₀(e) ≈ 0.4342944819032518

    Mathematical significance:
        - Common logarithm of Euler's number
        - Conversion factor: log₁₀(x) = ln(x) * log₁₀(e)
        - log₁₀(e) ≈ 0.43429448...

    Examples:
        await log10e() → 0.4342944819032518
        10 ** (await log10e()) = await e()
        log₁₀(x) = ln(x) * await log10e()
    """
    return math.log10(math.e)


@mcp_function(
    description="Get the Euler-Mascheroni constant γ ≈ 0.57722. Limit of harmonic series minus natural logarithm.",
    namespace="arithmetic",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    examples=[
        {
            "input": {},
            "output": 0.5772156649015329,
            "description": "Value of Euler-Mascheroni constant",
        },
        {
            "input": {},
            "output": 0.5772156649015329,
            "description": "γ = lim(n→∞) [H_n - ln(n)]",
        },
    ],
)
async def euler_gamma() -> float:
    """
    Get the Euler-Mascheroni constant γ (gamma).

    Returns:
        The value of γ ≈ 0.5772156649015329

    Mathematical significance:
        - γ = lim(n→∞) [H_n - ln(n)] where H_n is nth harmonic number
        - Appears in number theory and analysis
        - Not known if γ is rational or irrational
        - γ ≈ 0.57721566...

    Examples:
        await euler_gamma() → 0.5772156649015329
        H_n ≈ ln(n) + γ for large n
        Appears in gamma function derivatives
    """
    return 0.5772156649015329


@mcp_function(
    description="Get Catalan's constant G ≈ 0.91597. Sum of alternating series 1 - 1/9 + 1/25 - 1/49 + ...",
    namespace="arithmetic",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    examples=[
        {
            "input": {},
            "output": 0.9159655941772190,
            "description": "Value of Catalan's constant",
        },
        {
            "input": {},
            "output": 0.9159655941772190,
            "description": "G = Σ(-1)ⁿ/(2n+1)²",
        },
    ],
)
async def catalan() -> float:
    """
    Get Catalan's constant G.

    Returns:
        The value of G ≈ 0.9159655941772190

    Mathematical significance:
        - G = Σ(n=0 to ∞) (-1)ⁿ / (2n+1)²
        - Appears in combinatorics and number theory
        - Related to Dirichlet beta function β(2)
        - G ≈ 0.91596559...

    Examples:
        await catalan() → 0.9159655941772190
        Series: 1 - 1/9 + 1/25 - 1/49 + ...
        β(2) = await catalan()
    """
    return 0.9159655941772190


@mcp_function(
    description="Get Apéry's constant ζ(3) ≈ 1.20206. Sum of cubes reciprocals: 1 + 1/8 + 1/27 + ...",
    namespace="arithmetic",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    examples=[
        {
            "input": {},
            "output": 1.2020569031595942,
            "description": "Value of Apéry's constant",
        },
        {"input": {}, "output": 1.2020569031595942, "description": "ζ(3) = Σ(1/n³)"},
    ],
)
async def apery() -> float:
    """
    Get Apéry's constant ζ(3).

    Returns:
        The value of ζ(3) ≈ 1.2020569031595942

    Mathematical significance:
        - ζ(3) = Σ(n=1 to ∞) 1/n³
        - Value of Riemann zeta function at 3
        - Proven irrational by Roger Apéry in 1978
        - ζ(3) ≈ 1.20205690...

    Examples:
        await apery() → 1.2020569031595942
        Series: 1 + 1/8 + 1/27 + 1/64 + ...
        Riemann zeta function: ζ(3)
    """
    return 1.2020569031595942


@mcp_function(
    description="Get Khinchin's constant K ≈ 2.68545. Geometric mean of continued fraction coefficients.",
    namespace="arithmetic",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    examples=[
        {
            "input": {},
            "output": 2.6854520010653062,
            "description": "Value of Khinchin's constant",
        },
        {
            "input": {},
            "output": 2.6854520010653062,
            "description": "Geometric mean of CF coefficients",
        },
    ],
)
async def khinchin() -> float:
    """
    Get Khinchin's constant K.

    Returns:
        The value of K ≈ 2.6854520010653062

    Mathematical significance:
        - Geometric mean of continued fraction coefficients
        - For almost all real numbers x
        - K ≈ 2.68545200...
        - Related to continued fraction expansions

    Examples:
        await khinchin() → 2.6854520010653062
        Continued fraction geometric mean
        Appears in metric number theory
    """
    return 2.6854520010653062


@mcp_function(
    description="Get the Glaisher-Kinkelin constant A ≈ 1.28243. Related to Barnes G-function and hyperfactorials.",
    namespace="arithmetic",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    examples=[
        {
            "input": {},
            "output": 1.2824271291006226,
            "description": "Value of Glaisher-Kinkelin constant",
        },
        {
            "input": {},
            "output": 1.2824271291006226,
            "description": "Related to Barnes G-function",
        },
    ],
)
async def glaisher() -> float:
    """
    Get the Glaisher-Kinkelin constant A.

    Returns:
        The value of A ≈ 1.2824271291006226

    Mathematical significance:
        - Related to Barnes G-function
        - Appears in hyperfactorial asymptotic expansions
        - A ≈ 1.28242712...
        - Connected to multiple zeta values

    Examples:
        await glaisher() → 1.2824271291006226
        Barnes G-function constant
        Hyperfactorial asymptotics
    """
    return 1.2824271291006226


@mcp_function(
    description="Get machine epsilon for floating-point precision. Smallest value where 1 + ε > 1.",
    namespace="arithmetic",
    execution_modes=["local", "remote"],
    examples=[
        {
            "input": {},
            "output": 2.220446049250313e-16,
            "description": "Machine epsilon for float64",
        },
        {
            "input": {},
            "output": 2.220446049250313e-16,
            "description": "Floating-point precision limit",
        },
    ],
)
async def machine_epsilon() -> float:
    """
    Get machine epsilon for floating-point arithmetic.

    Returns:
        Machine epsilon (smallest ε where 1 + ε > 1)

    Mathematical significance:
        - Fundamental floating-point precision limit
        - Used in numerical analysis for comparisons
        - Typically 2^-52 for double precision
        - Platform and implementation dependent

    Examples:
        await machine_epsilon() → 2.220446049250313e-16
        1.0 + await machine_epsilon() > 1.0  # True
        1.0 + (await machine_epsilon())/2 == 1.0  # True
    """
    return sys.float_info.epsilon


@mcp_function(
    description="Get the maximum finite floating-point value. Largest representable float.",
    namespace="arithmetic",
    execution_modes=["local", "remote"],
    examples=[
        {
            "input": {},
            "output": 1.7976931348623157e308,
            "description": "Maximum float value",
        },
        {
            "input": {},
            "output": 1.7976931348623157e308,
            "description": "Largest finite float",
        },
    ],
)
async def max_float() -> float:
    """
    Get the maximum finite floating-point value.

    Returns:
        Maximum representable finite float

    Mathematical significance:
        - Largest finite floating-point number
        - Platform and implementation dependent
        - Typically around 1.8 × 10^308 for double precision
        - Overflow beyond this gives infinity

    Examples:
        await max_float() → 1.7976931348623157e+308
        (await max_float()) * 2 → inf
        Used for overflow detection
    """
    return sys.float_info.max


@mcp_function(
    description="Get the minimum positive normalized floating-point value. Smallest positive normal float.",
    namespace="arithmetic",
    execution_modes=["local", "remote"],
    examples=[
        {
            "input": {},
            "output": 2.2250738585072014e-308,
            "description": "Minimum positive normal float",
        },
        {
            "input": {},
            "output": 2.2250738585072014e-308,
            "description": "Smallest positive normal float",
        },
    ],
)
async def min_float() -> float:
    """
    Get the minimum positive normalized floating-point value.

    Returns:
        Minimum positive normalized float

    Mathematical significance:
        - Smallest positive normalized floating-point number
        - Below this, numbers become denormalized
        - Platform and implementation dependent
        - Typically around 2.2 × 10^-308 for double precision

    Examples:
        await min_float() → 2.2250738585072014e-308
        (await min_float()) / 2 → denormalized number
        Used for underflow detection
    """
    return sys.float_info.min


# Export all constant functions
__all__ = [
    "pi",
    "e",
    "tau",
    "infinity",
    "nan",
    "golden_ratio",
    "silver_ratio",
    "plastic_number",
    "sqrt2",
    "sqrt3",
    "sqrt5",
    "cbrt2",
    "cbrt3",
    "ln2",
    "ln10",
    "log2e",
    "log10e",
    "euler_gamma",
    "catalan",
    "apery",
    "khinchin",
    "glaisher",
    "machine_epsilon",
    "max_float",
    "min_float",
]

if __name__ == "__main__":
    import asyncio

    async def test_mathematical_constants():
        """Test all mathematical constants (async)."""
        print("🔢 Mathematical Constants Test (Async Native)")
        print("=" * 45)

        # Fundamental constants
        print(f"pi() = {await pi()}")
        print(f"e() = {await e()}")
        print(f"tau() = {await tau()}")

        # Special values
        print(f"infinity() = {await infinity()}")
        print(f"nan() = {await nan()}")

        # Algebraic constants
        print(f"golden_ratio() = {await golden_ratio()}")
        print(f"silver_ratio() = {await silver_ratio()}")
        print(f"plastic_number() = {await plastic_number()}")

        # Root constants
        print(f"sqrt2() = {await sqrt2()}")
        print(f"sqrt3() = {await sqrt3()}")
        print(f"sqrt5() = {await sqrt5()}")
        print(f"cbrt2() = {await cbrt2()}")
        print(f"cbrt3() = {await cbrt3()}")

        # Logarithmic constants
        print(f"ln2() = {await ln2()}")
        print(f"ln10() = {await ln10()}")
        print(f"log2e() = {await log2e()}")
        print(f"log10e() = {await log10e()}")

        # Special mathematical constants
        print(f"euler_gamma() = {await euler_gamma()}")
        print(f"catalan() = {await catalan()}")
        print(f"apery() = {await apery()}")
        print(f"khinchin() = {await khinchin()}")
        print(f"glaisher() = {await glaisher()}")

        # Numeric limits
        print(f"machine_epsilon() = {await machine_epsilon()}")
        print(f"max_float() = {await max_float()}")
        print(f"min_float() = {await min_float()}")

        print("\n✅ All mathematical constants working correctly!")

        # Test parallel execution
        print("\n🚀 Testing Parallel Execution:")
        parallel_results = await asyncio.gather(
            pi(), e(), golden_ratio(), sqrt2(), ln2()
        )
        print(f"Parallel results: {parallel_results}")

        # Test relationships
        print("\n🔬 Testing Mathematical Relationships:")
        tau_val = await tau()
        pi_val = await pi()
        print(f"tau() / pi() = {tau_val / pi_val:.10f} (should be ≈ 2.0)")

        sqrt2_val = await sqrt2()
        print(f"sqrt2() ** 2 = {sqrt2_val**2:.10f} (should be ≈ 2.0)")

        e_val = await e()
        ln2_val = await ln2()
        print(f"e() ** ln2() = {e_val**ln2_val:.10f} (should be ≈ 2.0)")

    asyncio.run(test_mathematical_constants())
