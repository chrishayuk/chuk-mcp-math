"""
Numeric Derivatives

Numeric differentiation using finite difference methods.
All functions are async-native with MCP decoration for AI model integration.
"""

import asyncio
from typing import Callable
from ..mcp_decorator import mcp_function


@mcp_function(
    description="Calculate numeric derivative using central difference method (most accurate)",
    namespace="calculus",
    cache_strategy="memory",
    estimated_cpu_usage="low",
    examples=[
        {
            "input": {"func": "lambda x: x**2", "x": 3.0, "h": 0.0001},
            "output": 6.0,
            "description": "Derivative of x² at x=3 is 2x = 6",
        }
    ],
)
async def derivative_central(func: Callable[[float], float], x: float, h: float = 1e-5) -> float:
    """
    Calculate the numeric derivative using the central difference method.

    Formula:
        f'(x) ≈ [f(x+h) - f(x-h)] / (2h)

    This is more accurate than forward/backward differences (O(h²) error).

    Args:
        func: Function to differentiate (must accept single float, return float)
        x: Point at which to evaluate derivative
        h: Step size (default: 1e-5)

    Returns:
        Approximate derivative value

    Raises:
        ValueError: If h is too small or zero

    Examples:
        >>> f = lambda x: x**2
        >>> await derivative_central(f, 3.0)
        6.0
        >>> g = lambda x: math.sin(x)
        >>> await derivative_central(g, 0.0)  # cos(0) = 1
        1.0
    """
    if abs(h) < 1e-15:
        raise ValueError(f"Step size h={h} is too small, may cause numerical instability")

    await asyncio.sleep(0)

    # Central difference: [f(x+h) - f(x-h)] / 2h
    f_forward = func(x + h)
    f_backward = func(x - h)

    derivative = (f_forward - f_backward) / (2 * h)
    return float(derivative)


@mcp_function(
    description="Calculate numeric derivative using forward difference method",
    namespace="calculus",
    cache_strategy="memory",
    estimated_cpu_usage="low",
)
async def derivative_forward(func: Callable[[float], float], x: float, h: float = 1e-5) -> float:
    """
    Calculate the numeric derivative using the forward difference method.

    Formula:
        f'(x) ≈ [f(x+h) - f(x)] / h

    Less accurate than central difference (O(h) error).

    Args:
        func: Function to differentiate
        x: Point at which to evaluate derivative
        h: Step size (default: 1e-5)

    Returns:
        Approximate derivative value

    Raises:
        ValueError: If h is too small or zero

    Examples:
        >>> f = lambda x: x**2
        >>> await derivative_forward(f, 3.0)
        6.0001  # Slightly less accurate than central
    """
    if abs(h) < 1e-15:
        raise ValueError(f"Step size h={h} is too small, may cause numerical instability")

    await asyncio.sleep(0)

    # Forward difference: [f(x+h) - f(x)] / h
    f_current = func(x)
    f_forward = func(x + h)

    derivative = (f_forward - f_current) / h
    return float(derivative)


@mcp_function(
    description="Calculate numeric derivative using backward difference method",
    namespace="calculus",
    cache_strategy="memory",
    estimated_cpu_usage="low",
)
async def derivative_backward(func: Callable[[float], float], x: float, h: float = 1e-5) -> float:
    """
    Calculate the numeric derivative using the backward difference method.

    Formula:
        f'(x) ≈ [f(x) - f(x-h)] / h

    Less accurate than central difference (O(h) error).

    Args:
        func: Function to differentiate
        x: Point at which to evaluate derivative
        h: Step size (default: 1e-5)

    Returns:
        Approximate derivative value

    Raises:
        ValueError: If h is too small or zero

    Examples:
        >>> f = lambda x: x**2
        >>> await derivative_backward(f, 3.0)
        5.9999  # Slightly less accurate than central
    """
    if abs(h) < 1e-15:
        raise ValueError(f"Step size h={h} is too small, may cause numerical instability")

    await asyncio.sleep(0)

    # Backward difference: [f(x) - f(x-h)] / h
    f_current = func(x)
    f_backward = func(x - h)

    derivative = (f_current - f_backward) / h
    return float(derivative)
