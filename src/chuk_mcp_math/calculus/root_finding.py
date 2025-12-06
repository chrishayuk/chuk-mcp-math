"""
Root Finding Algorithms

Numeric methods for finding roots (zeros) of functions.
All functions are async-native with MCP decoration for AI model integration.
"""

import asyncio
from typing import Callable
from ..mcp_decorator import mcp_function


@mcp_function(
    description="Find root of a function using the bisection method (robust and guaranteed)",
    namespace="calculus",
    cache_strategy="memory",
    estimated_cpu_usage="medium",
    examples=[
        {
            "input": {
                "func": "lambda x: x**2 - 4",
                "a": 0.0,
                "b": 3.0,
                "tol": 1e-6,
                "max_iter": 100,
            },
            "output": 2.0,
            "description": "Find root of x² - 4 = 0 (answer: x = 2)",
        }
    ],
)
async def root_find_bisection(
    func: Callable[[float], float], a: float, b: float, tol: float = 1e-6, max_iter: int = 100
) -> float:
    """
    Find a root of the function using the bisection method.

    The bisection method is robust and guaranteed to converge if:
    - f(a) and f(b) have opposite signs
    - f is continuous on [a, b]

    Algorithm:
    1. Start with interval [a, b] where f(a) and f(b) have opposite signs
    2. Calculate midpoint c = (a + b) / 2
    3. If f(c) is close enough to zero, return c
    4. Otherwise, replace either a or b with c to maintain opposite signs
    5. Repeat until convergence or max iterations

    Args:
        func: Function to find root of
        a: Left endpoint of interval
        b: Right endpoint of interval
        tol: Tolerance for convergence (default: 1e-6)
        max_iter: Maximum number of iterations (default: 100)

    Returns:
        Approximate root of the function

    Raises:
        ValueError: If f(a) and f(b) have the same sign or max iterations exceeded

    Examples:
        >>> f = lambda x: x**2 - 4  # Root at x = 2
        >>> await root_find_bisection(f, 0.0, 3.0)
        2.0
        >>> g = lambda x: x**3 - x - 2  # Root around x = 1.52
        >>> await root_find_bisection(g, 1.0, 2.0)
        1.521...
    """
    if a > b:
        a, b = b, a

    fa = func(a)
    fb = func(b)

    # Check that f(a) and f(b) have opposite signs
    if fa * fb > 0:
        raise ValueError(
            f"Function must have opposite signs at endpoints: f({a})={fa}, f({b})={fb}"
        )

    # Yield for async
    await asyncio.sleep(0)

    for iteration in range(max_iter):
        # Calculate midpoint
        c = (a + b) / 2
        fc = func(c)

        # Check for convergence
        if abs(fc) < tol or (b - a) / 2 < tol:
            return float(c)

        # Yield every 10 iterations
        if iteration % 10 == 0:
            await asyncio.sleep(0)

        # Determine which half to keep
        if fa * fc < 0:
            # Root is in left half
            b = c
            fb = fc
        else:
            # Root is in right half
            a = c
            fa = fc

    # Max iterations exceeded
    raise ValueError(
        f"Bisection failed to converge within {max_iter} iterations. "
        f"Last interval: [{a}, {b}], f(midpoint) = {func((a + b) / 2)}"
    )


@mcp_function(
    description="Find root of a function using Newton's method (fast but requires derivative)",
    namespace="calculus",
    cache_strategy="memory",
    estimated_cpu_usage="medium",
)
async def root_find_newton(
    func: Callable[[float], float],
    fprime: Callable[[float], float],
    x0: float,
    tol: float = 1e-6,
    max_iter: int = 50,
) -> float:
    """
    Find a root using Newton's method (Newton-Raphson).

    Formula:
        x_{n+1} = x_n - f(x_n) / f'(x_n)

    Converges quadratically when close to root, but:
    - Requires derivative
    - May not converge if started far from root
    - Fails if derivative is zero

    Args:
        func: Function to find root of
        fprime: Derivative of the function
        x0: Initial guess
        tol: Tolerance for convergence (default: 1e-6)
        max_iter: Maximum number of iterations (default: 50)

    Returns:
        Approximate root of the function

    Raises:
        ValueError: If derivative is zero or max iterations exceeded

    Examples:
        >>> f = lambda x: x**2 - 4
        >>> fp = lambda x: 2*x
        >>> await root_find_newton(f, fp, 1.0)
        2.0
    """
    x = x0

    await asyncio.sleep(0)

    for iteration in range(max_iter):
        fx = func(x)

        # Check for convergence
        if abs(fx) < tol:
            return float(x)

        # Calculate derivative
        fpx = fprime(x)

        if abs(fpx) < 1e-15:
            raise ValueError(
                f"Derivative too close to zero at x={x}: f'(x)={fpx}. "
                "Newton's method cannot continue."
            )

        # Newton's update
        x_new = x - fx / fpx

        # Yield every 10 iterations
        if iteration % 10 == 0:
            await asyncio.sleep(0)

        x = x_new

    raise ValueError(
        f"Newton's method failed to converge within {max_iter} iterations. "
        f"Last value: x={x}, f(x)={func(x)}"
    )


@mcp_function(
    description="Find root of a function using the secant method (like Newton but without derivative)",
    namespace="calculus",
    cache_strategy="memory",
    estimated_cpu_usage="medium",
)
async def root_find_secant(
    func: Callable[[float], float], x0: float, x1: float, tol: float = 1e-6, max_iter: int = 50
) -> float:
    """
    Find a root using the secant method.

    Like Newton's method but approximates the derivative numerically:
        f'(x) ≈ [f(x1) - f(x0)] / (x1 - x0)

    Formula:
        x_{n+1} = x_n - f(x_n) × (x_n - x_{n-1}) / (f(x_n) - f(x_{n-1}))

    Args:
        func: Function to find root of
        x0: First initial guess
        x1: Second initial guess
        tol: Tolerance for convergence (default: 1e-6)
        max_iter: Maximum number of iterations (default: 50)

    Returns:
        Approximate root of the function

    Raises:
        ValueError: If points converge or max iterations exceeded

    Examples:
        >>> f = lambda x: x**2 - 4
        >>> await root_find_secant(f, 1.0, 3.0)
        2.0
    """
    await asyncio.sleep(0)

    for iteration in range(max_iter):
        fx0 = func(x0)
        fx1 = func(x1)

        # Check for convergence
        if abs(fx1) < tol:
            return float(x1)

        # Check that function values are different
        if abs(fx1 - fx0) < 1e-15:
            raise ValueError(
                f"Function values too close: f({x0})={fx0}, f({x1})={fx1}. "
                "Secant method cannot continue."
            )

        # Secant update
        x_new = x1 - fx1 * (x1 - x0) / (fx1 - fx0)

        # Yield every 10 iterations
        if iteration % 10 == 0:
            await asyncio.sleep(0)

        # Shift points
        x0, x1 = x1, x_new

    raise ValueError(
        f"Secant method failed to converge within {max_iter} iterations. "
        f"Last value: x={x1}, f(x)={func(x1)}"
    )
