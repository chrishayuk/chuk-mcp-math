"""
Numeric Integration

Numeric integration using various quadrature methods.
All functions are async-native with MCP decoration for AI model integration.
"""

import asyncio
from typing import Callable
from ..mcp_decorator import mcp_function


@mcp_function(
    description="Calculate definite integral using the trapezoidal rule",
    namespace="calculus",
    cache_strategy="memory",
    estimated_cpu_usage="medium",
    examples=[
        {
            "input": {"func": "lambda x: x**2", "a": 0.0, "b": 1.0, "n_steps": 1000},
            "output": 0.333,
            "description": "Integral of x² from 0 to 1 is 1/3",
        }
    ],
)
async def integrate_trapezoid(
    func: Callable[[float], float], a: float, b: float, n_steps: int = 1000
) -> float:
    """
    Calculate definite integral using the trapezoidal rule.

    Formula:
        ∫[a,b] f(x)dx ≈ (h/2) × [f(a) + 2f(a+h) + 2f(a+2h) + ... + 2f(b-h) + f(b)]
        where h = (b-a)/n

    Args:
        func: Function to integrate
        a: Lower limit of integration
        b: Upper limit of integration
        n_steps: Number of trapezoids (default: 1000)

    Returns:
        Approximate integral value

    Raises:
        ValueError: If n_steps < 1 or a > b

    Examples:
        >>> f = lambda x: x**2
        >>> await integrate_trapezoid(f, 0.0, 1.0, 1000)
        0.333...
        >>> g = lambda x: 1.0
        >>> await integrate_trapezoid(g, 0.0, 5.0, 100)
        5.0
    """
    if n_steps < 1:
        raise ValueError(f"Number of steps must be >= 1, got {n_steps}")
    if a > b:
        raise ValueError(f"Lower limit a={a} must be <= upper limit b={b}")

    # Yield for large computations
    if n_steps > 1000:
        await asyncio.sleep(0)

    h = (b - a) / n_steps

    # Calculate sum
    # First and last points get weight 1, interior points get weight 2
    total = func(a) + func(b)

    for i in range(1, n_steps):
        x = a + i * h
        total += 2 * func(x)

        # Yield periodically for very large integrations
        if i % 10000 == 0:
            await asyncio.sleep(0)

    integral = (h / 2) * total
    return float(integral)


@mcp_function(
    description="Calculate definite integral using Simpson's rule (more accurate than trapezoid)",
    namespace="calculus",
    cache_strategy="memory",
    estimated_cpu_usage="medium",
    examples=[
        {
            "input": {"func": "lambda x: x**2", "a": 0.0, "b": 1.0, "n_steps": 1000},
            "output": 0.33333,
            "description": "Integral of x² from 0 to 1 using Simpson's rule",
        }
    ],
)
async def integrate_simpson(
    func: Callable[[float], float], a: float, b: float, n_steps: int = 1000
) -> float:
    """
    Calculate definite integral using Simpson's rule.

    Formula:
        ∫[a,b] f(x)dx ≈ (h/3) × [f(a) + 4f(a+h) + 2f(a+2h) + 4f(a+3h) + ... + f(b)]

    More accurate than trapezoidal rule for smooth functions.

    Args:
        func: Function to integrate
        a: Lower limit of integration
        b: Upper limit of integration
        n_steps: Number of intervals (must be even, default: 1000)

    Returns:
        Approximate integral value

    Raises:
        ValueError: If n_steps is odd, < 2, or a > b

    Examples:
        >>> f = lambda x: x**2
        >>> await integrate_simpson(f, 0.0, 1.0, 1000)
        0.333333...
        >>> g = lambda x: x**3
        >>> await integrate_simpson(g, 0.0, 2.0, 1000)
        4.0
    """
    if n_steps < 2:
        raise ValueError(f"Number of steps must be >= 2, got {n_steps}")
    if n_steps % 2 != 0:
        raise ValueError(f"Number of steps must be even for Simpson's rule, got {n_steps}")
    if a > b:
        raise ValueError(f"Lower limit a={a} must be <= upper limit b={b}")

    # Yield for large computations
    if n_steps > 1000:
        await asyncio.sleep(0)

    h = (b - a) / n_steps

    # Simpson's rule: sum = f(a) + 4*f(odd points) + 2*f(even points) + f(b)
    total = func(a) + func(b)

    for i in range(1, n_steps):
        x = a + i * h
        if i % 2 == 1:
            # Odd indices get weight 4
            total += 4 * func(x)
        else:
            # Even indices get weight 2
            total += 2 * func(x)

        # Yield periodically for very large integrations
        if i % 10000 == 0:
            await asyncio.sleep(0)

    integral = (h / 3) * total
    return float(integral)


@mcp_function(
    description="Calculate definite integral using the midpoint rule",
    namespace="calculus",
    cache_strategy="memory",
    estimated_cpu_usage="medium",
)
async def integrate_midpoint(
    func: Callable[[float], float], a: float, b: float, n_steps: int = 1000
) -> float:
    """
    Calculate definite integral using the midpoint rule.

    Formula:
        ∫[a,b] f(x)dx ≈ h × [f(a+h/2) + f(a+3h/2) + ... + f(b-h/2)]

    Args:
        func: Function to integrate
        a: Lower limit of integration
        b: Upper limit of integration
        n_steps: Number of intervals (default: 1000)

    Returns:
        Approximate integral value

    Raises:
        ValueError: If n_steps < 1 or a > b

    Examples:
        >>> f = lambda x: x**2
        >>> await integrate_midpoint(f, 0.0, 1.0, 1000)
        0.333...
    """
    if n_steps < 1:
        raise ValueError(f"Number of steps must be >= 1, got {n_steps}")
    if a > b:
        raise ValueError(f"Lower limit a={a} must be <= upper limit b={b}")

    # Yield for large computations
    if n_steps > 1000:
        await asyncio.sleep(0)

    h = (b - a) / n_steps

    total = 0.0
    for i in range(n_steps):
        # Evaluate at midpoint of each interval
        x_mid = a + (i + 0.5) * h
        total += func(x_mid)

        # Yield periodically for very large integrations
        if i % 10000 == 0:
            await asyncio.sleep(0)

    integral = h * total
    return float(integral)
