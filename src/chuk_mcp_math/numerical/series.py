#!/usr/bin/env python3
"""Series expansions for function approximation.

This module provides Taylor series, power series, and Fourier series expansions
for approximating functions, essential for numerical analysis and signal processing.
"""

import asyncio
import math
from typing import List, Callable


async def taylor_series(
    f: Callable[[float], float],
    derivatives: List[Callable[[float], float]],
    a: float,
    x: float,
    n: int,
) -> float:
    """
    Taylor series approximation of f around point a.

    Approximates f(x) using Taylor series:
    f(x) ≈ Σ(k=0 to n) [f^(k)(a) / k!] * (x-a)^k

    Args:
        f: Function to approximate
        derivatives: List of derivative functions [f, f', f'', ..., f^(n)]
        a: Center point for expansion
        x: Point at which to evaluate
        n: Order of expansion (number of terms - 1)

    Returns:
        Taylor series approximation at x

    Raises:
        ValueError: If n < 0 or derivatives list is too short

    Example:
        >>> # Approximate e^x around 0
        >>> f = lambda x: math.exp(x)
        >>> derivatives = [f] * 5  # All derivatives of e^x are e^x
        >>> result = await taylor_series(f, derivatives, 0.0, 0.5, 4)
        >>> # Approximates e^0.5 ≈ 1.648
    """
    if n < 0:
        raise ValueError("n must be >= 0")
    if len(derivatives) < n + 1:
        raise ValueError(f"Need at least {n + 1} derivatives, got {len(derivatives)}")

    result = 0.0
    factorial = 1

    for k in range(n + 1):
        # Evaluate k-th derivative at a
        if asyncio.iscoroutinefunction(derivatives[k]):
            deriv_k = await derivatives[k](a)  # type: ignore[misc]
        else:
            deriv_k = derivatives[k](a)

        # Add term: f^(k)(a) / k! * (x-a)^k
        term = deriv_k / factorial * (x - a) ** k
        result += term

        # Update factorial for next term
        factorial *= k + 1

        if k % 10 == 0:
            await asyncio.sleep(0)

    return result


async def power_series(coefficients: List[float], x: float, x0: float = 0.0) -> float:
    """
    Evaluate power series at x.

    Evaluates: Σ(k=0 to n) a_k * (x - x0)^k

    Args:
        coefficients: List of coefficients [a_0, a_1, ..., a_n]
        x: Point at which to evaluate
        x0: Center point (default 0)

    Returns:
        Value of power series at x

    Raises:
        ValueError: If coefficients is empty

    Example:
        >>> # 1 + 2x + 3x^2
        >>> result = await power_series([1, 2, 3], 2.0)
        >>> # = 1 + 2(2) + 3(2^2) = 1 + 4 + 12 = 17
    """
    if not coefficients:
        raise ValueError("coefficients cannot be empty")

    result = 0.0
    power = 1.0  # (x - x0)^k

    for k, coeff in enumerate(coefficients):
        result += coeff * power
        power *= x - x0

        if k % 10 == 0:
            await asyncio.sleep(0)

    return result


async def horner_method(coefficients: List[float], x: float) -> float:
    """
    Evaluate polynomial using Horner's method (numerically stable).

    More efficient and stable than power_series for polynomials.
    Evaluates: a_n*x^n + a_(n-1)*x^(n-1) + ... + a_1*x + a_0

    Args:
        coefficients: Coefficients in ascending order [a_0, a_1, ..., a_n]
        x: Point at which to evaluate

    Returns:
        Polynomial value at x

    Raises:
        ValueError: If coefficients is empty

    Example:
        >>> # 2 + 3x + 4x^2 at x=1
        >>> result = await horner_method([2, 3, 4], 1.0)
        >>> # = 2 + 3 + 4 = 9
    """
    if not coefficients:
        raise ValueError("coefficients cannot be empty")

    # Horner's method: work backwards from highest degree
    result = coefficients[-1]

    for i in range(len(coefficients) - 2, -1, -1):
        result = result * x + coefficients[i]

        if i % 10 == 0:
            await asyncio.sleep(0)

    return result


async def fourier_series_approximation(
    f: Callable[[float], float],
    period: float,
    n_terms: int,
    x: float,
    n_samples: int = 100,
) -> float:
    """
    Fourier series approximation of periodic function.

    Approximates f using Fourier series with n_terms harmonics.

    Args:
        f: Periodic function to approximate
        period: Period of the function
        n_terms: Number of Fourier terms to use
        x: Point at which to evaluate
        n_samples: Number of samples for coefficient integration

    Returns:
        Fourier series approximation at x

    Raises:
        ValueError: If period <= 0 or n_terms < 1 or n_samples < 1

    Example:
        >>> # Approximate square wave
        >>> f = lambda x: 1.0 if x % 2 < 1 else -1.0
        >>> result = await fourier_series_approximation(f, 2.0, 10, 0.5)
    """
    if period <= 0:
        raise ValueError("period must be positive")
    if n_terms < 1:
        raise ValueError("n_terms must be >= 1")
    if n_samples < 1:
        raise ValueError("n_samples must be >= 1")

    # Compute a0 (DC component)
    a0 = 0.0
    dx = period / n_samples
    for i in range(n_samples):
        t = i * dx
        f_t = await f(t) if asyncio.iscoroutinefunction(f) else f(t)
        a0 += f_t * dx
    a0 /= period

    # Initialize series with a0/2
    result = a0 / 2

    # Compute Fourier coefficients and sum
    omega = 2 * math.pi / period

    for n in range(1, n_terms + 1):
        # Compute a_n (cosine coefficient)
        a_n = 0.0
        for i in range(n_samples):
            t = i * dx
            f_t = await f(t) if asyncio.iscoroutinefunction(f) else f(t)
            a_n += f_t * math.cos(n * omega * t) * dx
        a_n *= 2 / period

        # Compute b_n (sine coefficient)
        b_n = 0.0
        for i in range(n_samples):
            t = i * dx
            f_t = await f(t) if asyncio.iscoroutinefunction(f) else f(t)
            b_n += f_t * math.sin(n * omega * t) * dx
        b_n *= 2 / period

        # Add n-th harmonic to result
        result += a_n * math.cos(n * omega * x) + b_n * math.sin(n * omega * x)

        if n % 5 == 0:
            await asyncio.sleep(0)

    return result


async def maclaurin_series(derivatives: List[Callable[[float], float]], x: float, n: int) -> float:
    """
    Maclaurin series (Taylor series centered at 0).

    Special case of Taylor series with a=0.

    Args:
        derivatives: List of derivative functions evaluated at 0
        x: Point at which to evaluate
        n: Order of expansion

    Returns:
        Maclaurin series approximation

    Example:
        >>> # e^x Maclaurin series
        >>> exp = lambda x: math.exp(x)
        >>> derivatives = [exp] * 5
        >>> result = await maclaurin_series(derivatives, 1.0, 4)
    """
    return await taylor_series(lambda t: 0.0, derivatives, 0.0, x, n)


async def binomial_series(x: float, alpha: float, n: int) -> float:
    """
    Binomial series expansion.

    Expands (1 + x)^alpha using binomial series.

    Args:
        x: Argument (|x| < 1 for convergence)
        alpha: Exponent (can be non-integer)
        n: Number of terms

    Returns:
        Binomial series approximation

    Raises:
        ValueError: If n < 0

    Example:
        >>> # sqrt(1 + x) = (1 + x)^0.5
        >>> result = await binomial_series(0.2, 0.5, 10)
    """
    if n < 0:
        raise ValueError("n must be >= 0")

    result = 1.0  # First term
    term = 1.0  # Current term

    for k in range(1, n + 1):
        term *= (alpha - k + 1) * x / k
        result += term

        if k % 10 == 0:
            await asyncio.sleep(0)

    return result


async def geometric_series(a: float, r: float, n: int) -> float:
    """
    Geometric series sum.

    Computes: a + ar + ar^2 + ... + ar^(n-1)

    Args:
        a: First term
        r: Common ratio
        n: Number of terms

    Returns:
        Sum of first n terms

    Raises:
        ValueError: If n < 1

    Example:
        >>> # 1 + 2 + 4 + 8 + 16 = 31
        >>> result = await geometric_series(1, 2, 5)
    """
    if n < 1:
        raise ValueError("n must be >= 1")

    if abs(r - 1.0) < 1e-10:
        # r ≈ 1: sum is n*a
        return n * a

    # Use closed form: a * (1 - r^n) / (1 - r)
    result = a * (1 - r**n) / (1 - r)
    await asyncio.sleep(0)
    return result


async def arithmetic_series(a: float, d: float, n: int) -> float:
    """
    Arithmetic series sum.

    Computes: a + (a+d) + (a+2d) + ... + (a+(n-1)d)

    Args:
        a: First term
        d: Common difference
        n: Number of terms

    Returns:
        Sum of first n terms

    Raises:
        ValueError: If n < 1

    Example:
        >>> # 1 + 3 + 5 + 7 + 9 = 25
        >>> result = await arithmetic_series(1, 2, 5)
    """
    if n < 1:
        raise ValueError("n must be >= 1")

    # Use closed form: n/2 * (2a + (n-1)d)
    result = n / 2 * (2 * a + (n - 1) * d)
    await asyncio.sleep(0)
    return result


async def exp_series(x: float, n: int) -> float:
    """
    Exponential function via Taylor series.

    Computes e^x = Σ(k=0 to n) x^k / k!

    Args:
        x: Argument
        n: Number of terms

    Returns:
        Approximation of e^x

    Raises:
        ValueError: If n < 0

    Example:
        >>> result = await exp_series(1.0, 10)
        >>> # Approximates e ≈ 2.71828
    """
    if n < 0:
        raise ValueError("n must be >= 0")

    result = 0.0
    term = 1.0  # x^k / k!

    for k in range(n + 1):
        result += term
        term *= x / (k + 1)  # Update for next term

        if k % 10 == 0:
            await asyncio.sleep(0)

    return result


async def sin_series(x: float, n: int) -> float:
    """
    Sine function via Taylor series.

    Computes sin(x) = Σ(k=0 to n) (-1)^k * x^(2k+1) / (2k+1)!

    Args:
        x: Argument in radians
        n: Number of terms

    Returns:
        Approximation of sin(x)

    Raises:
        ValueError: If n < 0

    Example:
        >>> result = await sin_series(math.pi/2, 10)
        >>> # Approximates sin(π/2) = 1
    """
    if n < 0:
        raise ValueError("n must be >= 0")

    result = 0.0
    term = x  # x^(2k+1) / (2k+1)!
    sign = 1

    for k in range(n + 1):
        result += sign * term
        # Update for next term: multiply by x^2 / ((2k+2)(2k+3))
        term *= x * x / ((2 * k + 2) * (2 * k + 3))
        sign *= -1

        if k % 10 == 0:
            await asyncio.sleep(0)

    return result


async def cos_series(x: float, n: int) -> float:
    """
    Cosine function via Taylor series.

    Computes cos(x) = Σ(k=0 to n) (-1)^k * x^(2k) / (2k)!

    Args:
        x: Argument in radians
        n: Number of terms

    Returns:
        Approximation of cos(x)

    Raises:
        ValueError: If n < 0

    Example:
        >>> result = await cos_series(0.0, 10)
        >>> # Approximates cos(0) = 1
    """
    if n < 0:
        raise ValueError("n must be >= 0")

    result = 0.0
    term = 1.0  # x^(2k) / (2k)!
    sign = 1

    for k in range(n + 1):
        result += sign * term
        # Update for next term: multiply by x^2 / ((2k+1)(2k+2))
        term *= x * x / ((2 * k + 1) * (2 * k + 2))
        sign *= -1

        if k % 10 == 0:
            await asyncio.sleep(0)

    return result


async def ln_series(x: float, n: int) -> float:
    """
    Natural logarithm via Taylor series (for 0 < x <= 2).

    Computes ln(x) using series around x=1:
    ln(x) = Σ(k=1 to n) (-1)^(k+1) * (x-1)^k / k

    Args:
        x: Argument (must be in (0, 2] for convergence)
        n: Number of terms

    Returns:
        Approximation of ln(x)

    Raises:
        ValueError: If x <= 0 or x > 2 or n < 1

    Example:
        >>> result = await ln_series(1.5, 20)
        >>> # Approximates ln(1.5) ≈ 0.405
    """
    if x <= 0:
        raise ValueError("x must be positive")
    if x > 2:
        raise ValueError("x must be <= 2 for convergence")
    if n < 1:
        raise ValueError("n must be >= 1")

    result = 0.0
    term = x - 1  # (x-1)^k / k
    sign = 1

    for k in range(1, n + 1):
        result += sign * term
        # Update for next term
        term *= (x - 1) * k / (k + 1)
        sign *= -1

        if k % 10 == 0:
            await asyncio.sleep(0)

    return result


__all__ = [
    "taylor_series",
    "power_series",
    "horner_method",
    "fourier_series_approximation",
    "maclaurin_series",
    "binomial_series",
    "geometric_series",
    "arithmetic_series",
    "exp_series",
    "sin_series",
    "cos_series",
    "ln_series",
]
