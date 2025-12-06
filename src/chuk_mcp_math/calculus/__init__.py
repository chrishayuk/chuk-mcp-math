"""
Calculus Module

Numeric calculus operations including derivatives, integrals, and root finding.
All functions are async-native and MCP-decorated for AI model integration.
"""

from .derivatives import (
    derivative_central,
    derivative_forward,
    derivative_backward,
)

from .integration import (
    integrate_trapezoid,
    integrate_simpson,
    integrate_midpoint,
)

from .root_finding import (
    root_find_bisection,
    root_find_newton,
    root_find_secant,
)

__all__ = [
    # Derivatives
    "derivative_central",
    "derivative_forward",
    "derivative_backward",
    # Integration
    "integrate_trapezoid",
    "integrate_simpson",
    "integrate_midpoint",
    # Root Finding
    "root_find_bisection",
    "root_find_newton",
    "root_find_secant",
]
