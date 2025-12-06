#!/usr/bin/env python3
"""Optimization algorithms for finding minima and maxima of functions.

This module provides various optimization methods for both unconstrained and
constrained optimization problems, essential for machine learning, engineering,
and business applications.
"""

import asyncio
import math
from typing import List, Callable, Dict, Any


async def gradient_descent(
    f: Callable[[List[float]], float],
    grad_f: Callable[[List[float]], List[float]],
    x0: List[float],
    learning_rate: float = 0.01,
    max_iterations: int = 1000,
    tolerance: float = 1e-6,
) -> Dict[str, Any]:
    """
    Vanilla gradient descent optimization.

    Minimizes a function using gradient descent with fixed learning rate.

    Args:
        f: Function to minimize (takes list of floats, returns float)
        grad_f: Gradient function (takes list of floats, returns list of floats)
        x0: Initial point
        learning_rate: Step size for each iteration
        max_iterations: Maximum number of iterations
        tolerance: Convergence tolerance for gradient norm

    Returns:
        Dictionary with 'x' (optimal point), 'f_x' (function value),
        'iterations', 'converged'

    Raises:
        ValueError: If learning_rate <= 0 or max_iterations < 1
        ValueError: If x0 is empty or tolerance <= 0

    Example:
        >>> f = lambda x: x[0]**2 + x[1]**2
        >>> grad = lambda x: [2*x[0], 2*x[1]]
        >>> result = await gradient_descent(f, grad, [1.0, 1.0])
        >>> # Converges to [0, 0]
    """
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if max_iterations < 1:
        raise ValueError("max_iterations must be >= 1")
    if not x0:
        raise ValueError("x0 cannot be empty")
    if tolerance <= 0:
        raise ValueError("tolerance must be positive")

    x = list(x0)  # Copy to avoid modifying input
    n = len(x)

    for iteration in range(max_iterations):
        # Compute gradient
        grad = await grad_f(x) if asyncio.iscoroutinefunction(grad_f) else grad_f(x)

        # Check convergence
        grad_norm = math.sqrt(sum(g**2 for g in grad))
        if grad_norm < tolerance:
            f_x = await f(x) if asyncio.iscoroutinefunction(f) else f(x)
            return {
                "x": x,
                "f_x": f_x,
                "iterations": iteration + 1,
                "converged": True,
                "grad_norm": grad_norm,
            }

        # Update step: x = x - learning_rate * grad
        for i in range(n):
            x[i] = x[i] - learning_rate * grad[i]

        # Yield for async
        if iteration % 10 == 0:
            await asyncio.sleep(0)

    # Max iterations reached
    f_x = await f(x) if asyncio.iscoroutinefunction(f) else f(x)
    grad = await grad_f(x) if asyncio.iscoroutinefunction(grad_f) else grad_f(x)
    grad_norm = math.sqrt(sum(g**2 for g in grad))

    return {
        "x": x,
        "f_x": f_x,
        "iterations": max_iterations,
        "converged": False,
        "grad_norm": grad_norm,
    }


async def gradient_descent_momentum(
    f: Callable[[List[float]], float],
    grad_f: Callable[[List[float]], List[float]],
    x0: List[float],
    learning_rate: float = 0.01,
    momentum: float = 0.9,
    max_iterations: int = 1000,
    tolerance: float = 1e-6,
) -> Dict[str, Any]:
    """
    Gradient descent with momentum.

    Accelerates convergence by accumulating velocity in consistent directions.

    Args:
        f: Function to minimize
        grad_f: Gradient function
        x0: Initial point
        learning_rate: Step size
        momentum: Momentum coefficient (typically 0.9)
        max_iterations: Maximum iterations
        tolerance: Convergence tolerance

    Returns:
        Dictionary with optimization results

    Raises:
        ValueError: If parameters are invalid

    Example:
        >>> result = await gradient_descent_momentum(f, grad, [1.0, 1.0], momentum=0.9)
    """
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if not 0 <= momentum < 1:
        raise ValueError("momentum must be in [0, 1)")
    if max_iterations < 1:
        raise ValueError("max_iterations must be >= 1")
    if not x0:
        raise ValueError("x0 cannot be empty")
    if tolerance <= 0:
        raise ValueError("tolerance must be positive")

    x = list(x0)
    n = len(x)
    velocity = [0.0] * n

    for iteration in range(max_iterations):
        grad = await grad_f(x) if asyncio.iscoroutinefunction(grad_f) else grad_f(x)

        # Check convergence
        grad_norm = math.sqrt(sum(g**2 for g in grad))
        if grad_norm < tolerance:
            f_x = await f(x) if asyncio.iscoroutinefunction(f) else f(x)
            return {
                "x": x,
                "f_x": f_x,
                "iterations": iteration + 1,
                "converged": True,
                "grad_norm": grad_norm,
            }

        # Update velocity and position
        for i in range(n):
            velocity[i] = momentum * velocity[i] - learning_rate * grad[i]
            x[i] = x[i] + velocity[i]

        if iteration % 10 == 0:
            await asyncio.sleep(0)

    f_x = await f(x) if asyncio.iscoroutinefunction(f) else f(x)
    grad = await grad_f(x) if asyncio.iscoroutinefunction(grad_f) else grad_f(x)
    grad_norm = math.sqrt(sum(g**2 for g in grad))

    return {
        "x": x,
        "f_x": f_x,
        "iterations": max_iterations,
        "converged": False,
        "grad_norm": grad_norm,
    }


async def adam_optimizer(
    f: Callable[[List[float]], float],
    grad_f: Callable[[List[float]], List[float]],
    x0: List[float],
    learning_rate: float = 0.001,
    beta1: float = 0.9,
    beta2: float = 0.999,
    epsilon: float = 1e-8,
    max_iterations: int = 1000,
    tolerance: float = 1e-6,
) -> Dict[str, Any]:
    """
    Adam (Adaptive Moment Estimation) optimizer.

    Combines momentum and adaptive learning rates for robust optimization.

    Args:
        f: Function to minimize
        grad_f: Gradient function
        x0: Initial point
        learning_rate: Initial learning rate
        beta1: Exponential decay rate for first moment (typically 0.9)
        beta2: Exponential decay rate for second moment (typically 0.999)
        epsilon: Small constant for numerical stability
        max_iterations: Maximum iterations
        tolerance: Convergence tolerance

    Returns:
        Dictionary with optimization results

    Example:
        >>> result = await adam_optimizer(f, grad, [1.0, 1.0])
    """
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if not 0 <= beta1 < 1:
        raise ValueError("beta1 must be in [0, 1)")
    if not 0 <= beta2 < 1:
        raise ValueError("beta2 must be in [0, 1)")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if max_iterations < 1:
        raise ValueError("max_iterations must be >= 1")
    if not x0:
        raise ValueError("x0 cannot be empty")

    x = list(x0)
    n = len(x)
    m = [0.0] * n  # First moment estimate
    v = [0.0] * n  # Second moment estimate

    for t in range(1, max_iterations + 1):
        grad = await grad_f(x) if asyncio.iscoroutinefunction(grad_f) else grad_f(x)

        # Check convergence
        grad_norm = math.sqrt(sum(g**2 for g in grad))
        if grad_norm < tolerance:
            f_x = await f(x) if asyncio.iscoroutinefunction(f) else f(x)
            return {
                "x": x,
                "f_x": f_x,
                "iterations": t,
                "converged": True,
                "grad_norm": grad_norm,
            }

        # Update biased first and second moments
        for i in range(n):
            m[i] = beta1 * m[i] + (1 - beta1) * grad[i]
            v[i] = beta2 * v[i] + (1 - beta2) * grad[i] ** 2

        # Bias correction
        m_hat = [m[i] / (1 - beta1**t) for i in range(n)]
        v_hat = [v[i] / (1 - beta2**t) for i in range(n)]

        # Update parameters
        for i in range(n):
            x[i] = x[i] - learning_rate * m_hat[i] / (math.sqrt(v_hat[i]) + epsilon)

        if t % 10 == 0:
            await asyncio.sleep(0)

    f_x = await f(x) if asyncio.iscoroutinefunction(f) else f(x)
    grad = await grad_f(x) if asyncio.iscoroutinefunction(grad_f) else grad_f(x)
    grad_norm = math.sqrt(sum(g**2 for g in grad))

    return {
        "x": x,
        "f_x": f_x,
        "iterations": max_iterations,
        "converged": False,
        "grad_norm": grad_norm,
    }


async def golden_section_search(
    f: Callable[[float], float],
    a: float,
    b: float,
    tolerance: float = 1e-6,
    max_iterations: int = 100,
) -> Dict[str, Any]:
    """
    Golden section search for 1D minimization.

    Finds the minimum of a unimodal function on interval [a, b].

    Args:
        f: Function to minimize (single variable)
        a: Left endpoint of interval
        b: Right endpoint of interval
        tolerance: Convergence tolerance
        max_iterations: Maximum iterations

    Returns:
        Dictionary with 'x' (minimum point), 'f_x', 'iterations'

    Raises:
        ValueError: If a >= b or tolerance <= 0

    Example:
        >>> f = lambda x: (x - 2)**2
        >>> result = await golden_section_search(f, 0, 5)
        >>> # Finds minimum at x=2
    """
    if a >= b:
        raise ValueError("a must be < b")
    if tolerance <= 0:
        raise ValueError("tolerance must be positive")
    if max_iterations < 1:
        raise ValueError("max_iterations must be >= 1")

    phi = (1 + math.sqrt(5)) / 2  # Golden ratio
    resphi = 2 - phi

    # Initial points
    x1 = a + resphi * (b - a)
    x2 = b - resphi * (b - a)
    f1 = await f(x1) if asyncio.iscoroutinefunction(f) else f(x1)
    f2 = await f(x2) if asyncio.iscoroutinefunction(f) else f(x2)

    for iteration in range(max_iterations):
        if abs(b - a) < tolerance:
            x_min = (a + b) / 2
            f_x = await f(x_min) if asyncio.iscoroutinefunction(f) else f(x_min)
            return {"x": x_min, "f_x": f_x, "iterations": iteration + 1, "converged": True}

        if f1 < f2:
            b = x2
            x2 = x1
            f2 = f1
            x1 = a + resphi * (b - a)
            f1 = await f(x1) if asyncio.iscoroutinefunction(f) else f(x1)
        else:
            a = x1
            x1 = x2
            f1 = f2
            x2 = b - resphi * (b - a)
            f2 = await f(x2) if asyncio.iscoroutinefunction(f) else f(x2)

        if iteration % 10 == 0:
            await asyncio.sleep(0)

    x_min = (a + b) / 2
    f_x = await f(x_min) if asyncio.iscoroutinefunction(f) else f(x_min)
    return {"x": x_min, "f_x": f_x, "iterations": max_iterations, "converged": False}


async def nelder_mead(
    f: Callable[[List[float]], float],
    x0: List[float],
    alpha: float = 1.0,
    gamma: float = 2.0,
    rho: float = 0.5,
    sigma: float = 0.5,
    max_iterations: int = 1000,
    tolerance: float = 1e-6,
) -> Dict[str, Any]:
    """
    Nelder-Mead simplex optimization (derivative-free).

    Finds minimum without requiring gradient information.

    Args:
        f: Function to minimize
        x0: Initial point
        alpha: Reflection coefficient (typically 1.0)
        gamma: Expansion coefficient (typically 2.0)
        rho: Contraction coefficient (typically 0.5)
        sigma: Shrink coefficient (typically 0.5)
        max_iterations: Maximum iterations
        tolerance: Convergence tolerance

    Returns:
        Dictionary with optimization results

    Raises:
        ValueError: If parameters are invalid

    Example:
        >>> f = lambda x: x[0]**2 + x[1]**2
        >>> result = await nelder_mead(f, [1.0, 1.0])
    """
    if not x0:
        raise ValueError("x0 cannot be empty")
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    if gamma <= 0:
        raise ValueError("gamma must be positive")
    if not 0 < rho < 1:
        raise ValueError("rho must be in (0, 1)")
    if not 0 < sigma < 1:
        raise ValueError("sigma must be in (0, 1)")
    if max_iterations < 1:
        raise ValueError("max_iterations must be >= 1")
    if tolerance <= 0:
        raise ValueError("tolerance must be positive")

    n = len(x0)

    # Initialize simplex
    simplex = [list(x0)]
    for i in range(n):
        vertex = list(x0)
        vertex[i] += 1.0  # Offset by 1 in each dimension
        simplex.append(vertex)

    # Evaluate function at each vertex
    f_values = []
    for vertex in simplex:
        f_val = await f(vertex) if asyncio.iscoroutinefunction(f) else f(vertex)
        f_values.append(f_val)

    for iteration in range(max_iterations):
        # Sort by function value
        sorted_indices = sorted(range(len(f_values)), key=lambda i: f_values[i])
        simplex = [simplex[i] for i in sorted_indices]
        f_values = [f_values[i] for i in sorted_indices]

        # Check convergence
        f_std = math.sqrt(
            sum((f_values[i] - f_values[0]) ** 2 for i in range(len(f_values))) / len(f_values)
        )
        if f_std < tolerance:
            return {
                "x": simplex[0],
                "f_x": f_values[0],
                "iterations": iteration + 1,
                "converged": True,
            }

        # Compute centroid of all but worst point
        centroid = [0.0] * n
        for i in range(n):
            centroid[i] = sum(simplex[j][i] for j in range(n)) / n

        # Reflection
        reflected = [centroid[i] + alpha * (centroid[i] - simplex[-1][i]) for i in range(n)]
        f_reflected = await f(reflected) if asyncio.iscoroutinefunction(f) else f(reflected)

        if f_values[0] <= f_reflected < f_values[-2]:
            # Accept reflected point
            simplex[-1] = reflected
            f_values[-1] = f_reflected
        elif f_reflected < f_values[0]:
            # Try expansion
            expanded = [centroid[i] + gamma * (reflected[i] - centroid[i]) for i in range(n)]
            f_expanded = await f(expanded) if asyncio.iscoroutinefunction(f) else f(expanded)
            if f_expanded < f_reflected:
                simplex[-1] = expanded
                f_values[-1] = f_expanded
            else:
                simplex[-1] = reflected
                f_values[-1] = f_reflected
        else:
            # Contraction
            contracted = [centroid[i] + rho * (simplex[-1][i] - centroid[i]) for i in range(n)]
            f_contracted = await f(contracted) if asyncio.iscoroutinefunction(f) else f(contracted)
            if f_contracted < f_values[-1]:
                simplex[-1] = contracted
                f_values[-1] = f_contracted
            else:
                # Shrink
                for i in range(1, len(simplex)):
                    simplex[i] = [
                        simplex[0][j] + sigma * (simplex[i][j] - simplex[0][j]) for j in range(n)
                    ]
                    f_values[i] = (
                        await f(simplex[i]) if asyncio.iscoroutinefunction(f) else f(simplex[i])
                    )

        if iteration % 10 == 0:
            await asyncio.sleep(0)

    # Return best point
    best_idx = min(range(len(f_values)), key=lambda i: f_values[i])
    return {
        "x": simplex[best_idx],
        "f_x": f_values[best_idx],
        "iterations": max_iterations,
        "converged": False,
    }


async def coordinate_descent(
    f: Callable[[List[float]], float],
    x0: List[float],
    step_size: float = 0.01,
    max_iterations: int = 1000,
    tolerance: float = 1e-6,
) -> Dict[str, Any]:
    """
    Coordinate descent optimization.

    Optimizes one coordinate at a time, cycling through all dimensions.

    Args:
        f: Function to minimize
        x0: Initial point
        step_size: Step size for line search
        max_iterations: Maximum iterations
        tolerance: Convergence tolerance

    Returns:
        Dictionary with optimization results

    Example:
        >>> result = await coordinate_descent(f, [1.0, 1.0])
    """
    if not x0:
        raise ValueError("x0 cannot be empty")
    if step_size <= 0:
        raise ValueError("step_size must be positive")
    if max_iterations < 1:
        raise ValueError("max_iterations must be >= 1")
    if tolerance <= 0:
        raise ValueError("tolerance must be positive")

    x = list(x0)
    n = len(x)
    f_prev = await f(x) if asyncio.iscoroutinefunction(f) else f(x)

    for iteration in range(max_iterations):
        # Optimize each coordinate
        for i in range(n):
            # Try step in positive direction
            x[i] += step_size
            f_plus = await f(x) if asyncio.iscoroutinefunction(f) else f(x)

            # Try step in negative direction
            x[i] -= 2 * step_size
            f_minus = await f(x) if asyncio.iscoroutinefunction(f) else f(x)

            # Choose best direction
            if f_plus < f_prev and f_plus < f_minus:
                x[i] += step_size  # Move to positive step
            elif f_minus < f_prev:
                pass  # Already at negative step
            else:
                x[i] += step_size  # Revert to original

        f_curr = await f(x) if asyncio.iscoroutinefunction(f) else f(x)

        # Check convergence
        if abs(f_curr - f_prev) < tolerance:
            return {
                "x": x,
                "f_x": f_curr,
                "iterations": iteration + 1,
                "converged": True,
            }

        f_prev = f_curr

        if iteration % 10 == 0:
            await asyncio.sleep(0)

    return {
        "x": x,
        "f_x": f_prev,
        "iterations": max_iterations,
        "converged": False,
    }


__all__ = [
    "gradient_descent",
    "gradient_descent_momentum",
    "adam_optimizer",
    "golden_section_search",
    "nelder_mead",
    "coordinate_descent",
]
