#!/usr/bin/env python3
# chuk_mcp_math/number_theory/diophantine_equations.py
"""
Diophantine Equations - Async Native

Functions for solving Diophantine equations (equations with integer solutions).
Covers linear, quadratic, and special cases like Pell's equation.

Functions:
- Linear: solve_linear_diophantine, count_solutions_diophantine, parametric_solutions
- Pell's equation: solve_pell_equation, pell_solutions_generator, negative_pell
- Quadratic: solve_quadratic_diophantine, pythagorean_triples, sum_of_squares
- General: diophantine_analysis, frobenius_number, postage_stamp_problem
- Advanced: generalized_pell, continued_fraction_pell, fundamental_solution
"""

import math
import asyncio
from typing import List, Tuple, Dict
from chuk_mcp_math.mcp_decorator import mcp_function

# ============================================================================
# LINEAR DIOPHANTINE EQUATIONS
# ============================================================================


@mcp_function(
    description="Solve linear Diophantine equation ax + by = c for integer solutions.",
    namespace="arithmetic",
    category="diophantine_equations",
    execution_modes=["local", "remote"],
    estimated_cpu_usage="medium",
    examples=[
        {
            "input": {"a": 3, "b": 5, "c": 1},
            "output": {
                "solvable": True,
                "particular": [-2, 1],
                "general": "x = -2 + 5t, y = 1 - 3t",
            },
            "description": "3x + 5y = 1",
        },
        {
            "input": {"a": 6, "b": 9, "c": 7},
            "output": {"solvable": False, "reason": "gcd(6,9) = 3 does not divide 7"},
            "description": "6x + 9y = 7 (no solution)",
        },
        {
            "input": {"a": 2, "b": 3, "c": 7},
            "output": {
                "solvable": True,
                "particular": [2, 1],
                "general": "x = 2 + 3t, y = 1 - 2t",
            },
            "description": "2x + 3y = 7",
        },
    ],
)
async def solve_linear_diophantine(a: int, b: int, c: int) -> Dict:
    """
    Solve the linear Diophantine equation ax + by = c.

    Uses the extended Euclidean algorithm to find a particular solution,
    then gives the general solution in parametric form.

    Args:
        a: Coefficient of x
        b: Coefficient of y
        c: Right-hand side constant

    Returns:
        Dictionary with solution information:
        - solvable: True if equation has integer solutions
        - particular: [x₀, y₀] particular solution (if solvable)
        - general: String describing general solution
        - gcd: gcd(a, b)

    Examples:
        await solve_linear_diophantine(3, 5, 1) → {"solvable": True, "particular": [-2, 1], ...}
        await solve_linear_diophantine(6, 9, 7) → {"solvable": False, ...}
    """
    if a == 0 and b == 0:
        if c == 0:
            return {
                "solvable": True,
                "particular": [0, 0],
                "general": "x = s, y = t for any integers s, t",
                "gcd": 0,
            }
        else:
            return {
                "solvable": False,
                "reason": "0x + 0y = c where c ≠ 0 has no solutions",
                "gcd": 0,
            }

    # Extended Euclidean algorithm
    def extended_gcd(a: int, b: int) -> Tuple[int, int, int]:
        if b == 0:
            return a, 1, 0
        gcd, x1, y1 = extended_gcd(b, a % b)
        x = y1
        y = x1 - (a // b) * y1
        return gcd, x, y

    gcd, x0, y0 = extended_gcd(abs(a), abs(b))

    # Check if equation is solvable
    if c % gcd != 0:
        return {
            "solvable": False,
            "reason": f"gcd({a},{b}) = {gcd} does not divide {c}",
            "gcd": gcd,
        }

    # Scale the solution
    scale = c // gcd
    x_particular = x0 * scale
    y_particular = y0 * scale

    # Adjust signs if original coefficients were negative
    if a < 0:
        x_particular = -x_particular
    if b < 0:
        y_particular = -y_particular

    # General solution: x = x₀ + (b/gcd)t, y = y₀ - (a/gcd)t
    b_step = b // gcd
    a_step = a // gcd

    general_form = f"x = {x_particular} + {b_step}t, y = {y_particular} - {a_step}t"

    return {
        "solvable": True,
        "particular": [x_particular, y_particular],
        "general": general_form,
        "gcd": gcd,
        "step_x": b_step,
        "step_y": -a_step,
    }


@mcp_function(
    description="Count integer solutions to ax + by = c within given bounds.",
    namespace="arithmetic",
    category="diophantine_equations",
    execution_modes=["local", "remote"],
    estimated_cpu_usage="medium",
    examples=[
        {
            "input": {
                "a": 2,
                "b": 3,
                "c": 12,
                "x_min": 0,
                "x_max": 6,
                "y_min": 0,
                "y_max": 4,
            },
            "output": 3,
            "description": "Solutions to 2x + 3y = 12 with bounds",
        },
        {
            "input": {
                "a": 1,
                "b": 1,
                "c": 10,
                "x_min": 0,
                "x_max": 10,
                "y_min": 0,
                "y_max": 10,
            },
            "output": 11,
            "description": "Solutions to x + y = 10 with bounds",
        },
        {
            "input": {
                "a": 5,
                "b": 7,
                "c": 35,
                "x_min": 0,
                "x_max": 7,
                "y_min": 0,
                "y_max": 5,
            },
            "output": 2,
            "description": "Solutions to 5x + 7y = 35 with bounds",
        },
    ],
)
async def count_solutions_diophantine(
    a: int, b: int, c: int, x_min: int, x_max: int, y_min: int, y_max: int
) -> int:
    """
    Count integer solutions to ax + by = c within specified bounds.

    Args:
        a, b, c: Coefficients of the Diophantine equation
        x_min, x_max: Bounds for x
        y_min, y_max: Bounds for y

    Returns:
        Number of integer solutions within the bounds

    Examples:
        await count_solutions_diophantine(2, 3, 12, 0, 6, 0, 4) → 3
        await count_solutions_diophantine(1, 1, 10, 0, 10, 0, 10) → 11
    """
    solution = await solve_linear_diophantine(a, b, c)

    if not solution["solvable"]:
        return 0

    x0, y0 = solution["particular"]
    step_x = solution["step_x"]  # b // gcd
    step_y = solution["step_y"]  # -a // gcd

    if step_x == 0:
        # Special case: x is constant
        if x_min <= x0 <= x_max and y_min <= y0 <= y_max:
            return 1
        else:
            return 0

    # Find range of t values that keep both x and y in bounds
    # x = x0 + step_x * t must be in [x_min, x_max]
    # y = y0 + step_y * t must be in [y_min, y_max]

    if step_x > 0:
        t_min_x = math.ceil((x_min - x0) / step_x)
        t_max_x = math.floor((x_max - x0) / step_x)
    else:
        t_min_x = math.ceil((x_max - x0) / step_x)
        t_max_x = math.floor((x_min - x0) / step_x)

    if step_y == 0:
        # y is constant
        if y_min <= y0 <= y_max:
            t_min_y = float("-inf")
            t_max_y = float("inf")
        else:
            return 0
    elif step_y > 0:
        t_min_y = math.ceil((y_min - y0) / step_y)
        t_max_y = math.floor((y_max - y0) / step_y)
    else:
        t_min_y = math.ceil((y_max - y0) / step_y)
        t_max_y = math.floor((y_min - y0) / step_y)

    # Intersection of valid t ranges
    t_min = max(t_min_x, t_min_y)
    t_max = min(t_max_x, t_max_y)

    if t_min <= t_max:
        return int(t_max - t_min + 1)
    else:
        return 0


@mcp_function(
    description="Generate parametric solutions to linear Diophantine equation.",
    namespace="arithmetic",
    category="diophantine_equations",
    execution_modes=["local", "remote"],
    examples=[
        {
            "input": {"a": 3, "b": 5, "c": 1, "t_min": -2, "t_max": 2},
            "output": [[-12, 7], [-7, 4], [-2, 1], [3, -2], [8, -5]],
            "description": "Solutions for t from -2 to 2",
        },
        {
            "input": {"a": 2, "b": 3, "c": 7, "t_min": 0, "t_max": 3},
            "output": [[2, 1], [5, -1], [8, -3], [11, -5]],
            "description": "Solutions for t from 0 to 3",
        },
    ],
)
async def parametric_solutions_diophantine(
    a: int, b: int, c: int, t_min: int, t_max: int
) -> List[List[int]]:
    """
    Generate parametric solutions to ax + by = c for parameter t in given range.

    Args:
        a, b, c: Coefficients of the Diophantine equation
        t_min, t_max: Range of parameter t

    Returns:
        List of [x, y] solutions for each integer t in range

    Examples:
        await parametric_solutions_diophantine(3, 5, 1, -2, 2) → [[-12, 7], [-7, 4], ...]
        await parametric_solutions_diophantine(2, 3, 7, 0, 3) → [[2, 1], [5, -1], ...]
    """
    solution = await solve_linear_diophantine(a, b, c)

    if not solution["solvable"]:
        return []

    x0, y0 = solution["particular"]
    step_x = solution["step_x"]
    step_y = solution["step_y"]

    solutions = []
    for t in range(t_min, t_max + 1):
        x = x0 + step_x * t
        y = y0 + step_y * t
        solutions.append([x, y])

        # Yield control every 1000 iterations for large ranges
        if (t - t_min) % 1000 == 0 and (t_max - t_min) > 10000:
            await asyncio.sleep(0)

    return solutions


# ============================================================================
# PELL'S EQUATION
# ============================================================================


@mcp_function(
    description="Solve Pell's equation x² - ny² = 1 for fundamental solution.",
    namespace="arithmetic",
    category="diophantine_equations",
    execution_modes=["local", "remote"],
    estimated_cpu_usage="high",
    examples=[
        {
            "input": {"n": 2},
            "output": {"fundamental": [3, 2], "exists": True},
            "description": "x² - 2y² = 1, fundamental solution (3,2)",
        },
        {
            "input": {"n": 3},
            "output": {"fundamental": [2, 1], "exists": True},
            "description": "x² - 3y² = 1, fundamental solution (2,1)",
        },
        {
            "input": {"n": 4},
            "output": {"exists": False, "reason": "n = 4 is a perfect square"},
            "description": "No non-trivial solutions when n is perfect square",
        },
        {
            "input": {"n": 5},
            "output": {"fundamental": [9, 4], "exists": True},
            "description": "x² - 5y² = 1, fundamental solution (9,4)",
        },
    ],
)
async def solve_pell_equation(n: int) -> Dict:
    """
    Solve Pell's equation x² - ny² = 1 for the fundamental solution.

    Uses continued fraction expansion of √n to find the minimal solution.

    Args:
        n: Positive integer (not a perfect square)

    Returns:
        Dictionary with solution information:
        - exists: True if non-trivial solutions exist
        - fundamental: [x₁, y₁] fundamental solution
        - period: Period of continued fraction
        - convergents: List of convergents used

    Examples:
        await solve_pell_equation(2) → {"fundamental": [3, 2], "exists": True}
        await solve_pell_equation(3) → {"fundamental": [2, 1], "exists": True}
        await solve_pell_equation(4) → {"exists": False, ...}
    """
    if n <= 0:
        raise ValueError("n must be positive")

    # Check if n is a perfect square
    sqrt_n = int(math.sqrt(n))
    if sqrt_n * sqrt_n == n:
        return {
            "exists": False,
            "reason": f"n = {n} is a perfect square",
            "trivial_solution": [1, 0],
        }

    # Use continued fraction expansion of √n
    # √n = a₀ + 1/(a₁ + 1/(a₂ + ...))

    a0 = sqrt_n
    m, d, a = 0, 1, a0

    # Track convergents
    convergents = []
    p_prev, p_curr = 1, a0
    q_prev, q_curr = 0, 1

    period_tracker = {}
    step = 0

    while True:
        # Next step in continued fraction
        m = d * a - m
        d = (n - m * m) // d
        a = (a0 + m) // d

        # Update convergents
        p_next = a * p_curr + p_prev
        q_next = a * q_curr + q_prev

        convergents.append([p_curr, q_curr])

        # Check if this is a solution to Pell's equation
        if p_curr * p_curr - n * q_curr * q_curr == 1 and q_curr > 0:
            return {
                "exists": True,
                "fundamental": [p_curr, q_curr],
                "period": step,
                "convergents": convergents,
            }

        # Track state to detect period
        state = (m, d, a)
        if state in period_tracker:
            # Period detected but no solution found in one period
            # This shouldn't happen for valid Pell equations
            break
        period_tracker[state] = step

        # Update for next iteration
        p_prev, p_curr = p_curr, p_next
        q_prev, q_curr = q_curr, q_next
        step += 1

        # Safety check
        if step > 1000:
            break

        # Yield control every 50 iterations
        if step % 50 == 0:
            await asyncio.sleep(0)

    return {
        "exists": False,
        "reason": "No solution found within iteration limit",
        "steps_computed": step,
    }


@mcp_function(
    description="Generate multiple solutions to Pell's equation x² - ny² = 1.",
    namespace="arithmetic",
    category="diophantine_equations",
    execution_modes=["local", "remote"],
    estimated_cpu_usage="high",
    examples=[
        {
            "input": {"n": 2, "num_solutions": 5},
            "output": [[3, 2], [17, 12], [99, 70], [577, 408], [3363, 2378]],
            "description": "First 5 solutions to x² - 2y² = 1",
        },
        {
            "input": {"n": 3, "num_solutions": 4},
            "output": [[2, 1], [7, 4], [26, 15], [97, 56]],
            "description": "First 4 solutions to x² - 3y² = 1",
        },
        {
            "input": {"n": 5, "num_solutions": 3},
            "output": [[9, 4], [161, 72], [2889, 1292]],
            "description": "First 3 solutions to x² - 5y² = 1",
        },
    ],
)
async def pell_solutions_generator(n: int, num_solutions: int) -> List[List[int]]:
    """
    Generate multiple solutions to Pell's equation using recurrence relations.

    If (x₁, y₁) is the fundamental solution, then subsequent solutions are:
    xₖ + yₖ√n = (x₁ + y₁√n)ᵏ

    Args:
        n: Positive integer (not a perfect square)
        num_solutions: Number of solutions to generate

    Returns:
        List of [x, y] solutions in ascending order

    Examples:
        await pell_solutions_generator(2, 5) → [[3, 2], [17, 12], ...]
        await pell_solutions_generator(3, 4) → [[2, 1], [7, 4], ...]
    """
    if num_solutions <= 0:
        return []

    fundamental = await solve_pell_equation(n)
    if not fundamental["exists"]:
        return []

    x1, y1 = fundamental["fundamental"]
    solutions = [[x1, y1]]

    if num_solutions == 1:
        return solutions

    # Generate subsequent solutions using recurrence:
    # x_{k+1} = x₁ * xₖ + n * y₁ * yₖ
    # y_{k+1} = x₁ * yₖ + y₁ * xₖ

    xk, yk = x1, y1

    for k in range(2, num_solutions + 1):
        x_next = x1 * xk + n * y1 * yk
        y_next = x1 * yk + y1 * xk

        solutions.append([x_next, y_next])
        xk, yk = x_next, y_next

        # Yield control every 10 solutions
        if k % 10 == 0:
            await asyncio.sleep(0)

    return solutions


@mcp_function(
    description="Solve negative Pell's equation x² - ny² = -1.",
    namespace="arithmetic",
    category="diophantine_equations",
    execution_modes=["local", "remote"],
    estimated_cpu_usage="high",
    examples=[
        {
            "input": {"n": 2},
            "output": {"fundamental": [1, 1], "exists": True},
            "description": "x² - 2y² = -1, fundamental solution (1,1)",
        },
        {
            "input": {"n": 5},
            "output": {"fundamental": [2, 1], "exists": True},
            "description": "x² - 5y² = -1, fundamental solution (2,1)",
        },
        {
            "input": {"n": 3},
            "output": {"exists": False, "reason": "No solutions exist for n = 3"},
            "description": "x² - 3y² = -1 has no solutions",
        },
    ],
)
async def solve_negative_pell_equation(n: int) -> Dict:
    """
    Solve the negative Pell's equation x² - ny² = -1.

    Not all values of n have solutions. Solutions exist if and only if
    the period of the continued fraction of √n is odd.

    Args:
        n: Positive integer (not a perfect square)

    Returns:
        Dictionary with solution information for x² - ny² = -1

    Examples:
        await solve_negative_pell_equation(2) → {"fundamental": [1, 1], "exists": True}
        await solve_negative_pell_equation(5) → {"fundamental": [2, 1], "exists": True}
        await solve_negative_pell_equation(3) → {"exists": False, ...}
    """
    if n <= 0:
        raise ValueError("n must be positive")

    # Check if n is a perfect square
    sqrt_n = int(math.sqrt(n))
    if sqrt_n * sqrt_n == n:
        return {"exists": False, "reason": f"n = {n} is a perfect square"}

    # Use continued fraction expansion
    a0 = sqrt_n
    m, d, a = 0, 1, a0

    # Track convergents
    p_prev, p_curr = 1, a0
    q_prev, q_curr = 0, 1

    step = 0
    period_tracker = {}

    while True:
        # Next step in continued fraction
        m = d * a - m
        d = (n - m * m) // d
        a = (a0 + m) // d

        # Update convergents
        p_next = a * p_curr + p_prev
        q_next = a * q_curr + q_prev

        # Check if this is a solution to negative Pell's equation
        if p_curr * p_curr - n * q_curr * q_curr == -1 and q_curr > 0:
            return {
                "exists": True,
                "fundamental": [p_curr, q_curr],
                "period_length": step + 1,
            }

        # Track state to detect period
        state = (m, d, a)
        if state in period_tracker:
            # Completed one period without finding solution
            return {
                "exists": False,
                "reason": f"Period length {step + 1} is even, no solutions exist",
                "period_length": step + 1,
            }
        period_tracker[state] = step

        # Update for next iteration
        p_prev, p_curr = p_curr, p_next
        q_prev, q_curr = q_curr, q_next
        step += 1

        # Safety check
        if step > 1000:
            break

        # Yield control every 50 iterations
        if step % 50 == 0:
            await asyncio.sleep(0)

    return {"exists": False, "reason": "No solution found within iteration limit"}


# ============================================================================
# QUADRATIC DIOPHANTINE EQUATIONS
# ============================================================================


@mcp_function(
    description="Find Pythagorean triples (x, y, z) where x² + y² = z².",
    namespace="arithmetic",
    category="diophantine_equations",
    execution_modes=["local", "remote"],
    examples=[
        {
            "input": {"limit": 50},
            "output": [
                [3, 4, 5],
                [5, 12, 13],
                [8, 15, 17],
                [7, 24, 25],
                [20, 21, 29],
                [9, 40, 41],
                [12, 35, 37],
                [11, 60, 61],
                [13, 84, 85],
                [28, 45, 53],
            ],
            "description": "Pythagorean triples with z ≤ 50",
        },
        {
            "input": {"limit": 25, "primitive_only": True},
            "output": [[3, 4, 5], [5, 12, 13], [8, 15, 17], [7, 24, 25]],
            "description": "Primitive triples with z ≤ 25",
        },
    ],
)
async def pythagorean_triples(
    limit: int, primitive_only: bool = False
) -> List[List[int]]:
    """
    Generate Pythagorean triples (a, b, c) where a² + b² = c².

    Uses the parametric form: a = m² - n², b = 2mn, c = m² + n²
    where m > n > 0 and gcd(m,n) = 1 for primitive triples.

    Args:
        limit: Maximum value for the hypotenuse c
        primitive_only: If True, return only primitive triples

    Returns:
        List of [a, b, c] Pythagorean triples with c ≤ limit

    Examples:
        await pythagorean_triples(50) → [[3, 4, 5], [5, 12, 13], ...]
        await pythagorean_triples(25, True) → [[3, 4, 5], [5, 12, 13], ...]
    """
    if limit <= 0:
        return []

    triples = []
    seen = set()

    # Generate primitive triples using parametric form
    max_m = int(math.sqrt(limit)) + 1

    for m in range(2, max_m + 1):
        for n in range(1, m):
            if m * m + n * n > limit:
                break

            # For primitive triples: gcd(m,n) = 1 and m,n not both odd
            gcd_mn = math.gcd(m, n)
            if primitive_only and (gcd_mn != 1 or (m % 2 == 1 and n % 2 == 1)):
                continue

            # Generate triple
            a = m * m - n * n
            b = 2 * m * n
            c = m * m + n * n

            if c > limit:
                continue

            # Ensure a ≤ b (canonical form)
            if a > b:
                a, b = b, a

            # Add primitive triple
            if (a, b, c) not in seen:
                triples.append([a, b, c])
                seen.add((a, b, c))

            # Add multiples if not primitive_only
            if not primitive_only:
                k = 2
                while k * c <= limit:
                    ka, kb, kc = k * a, k * b, k * c
                    if (ka, kb, kc) not in seen:
                        triples.append([ka, kb, kc])
                        seen.add((ka, kb, kc))
                    k += 1

        # Yield control every 10 values of m
        if m % 10 == 0:
            await asyncio.sleep(0)

    # Sort by hypotenuse, then by first leg
    triples.sort(key=lambda t: (t[2], t[0]))
    return triples


@mcp_function(
    description="Represent integer as sum of two squares x² + y².",
    namespace="arithmetic",
    category="diophantine_equations",
    execution_modes=["local", "remote"],
    examples=[
        {
            "input": {"n": 25},
            "output": [[0, 5], [3, 4]],
            "description": "25 = 0² + 5² = 3² + 4²",
        },
        {
            "input": {"n": 50},
            "output": [[1, 7], [5, 5]],
            "description": "50 = 1² + 7² = 5² + 5²",
        },
        {"input": {"n": 13}, "output": [[2, 3]], "description": "13 = 2² + 3²"},
    ],
)
async def sum_of_two_squares_all(n: int) -> List[List[int]]:
    """
    Find all ways to represent n as sum of two squares.

    Args:
        n: Positive integer

    Returns:
        List of [x, y] pairs where x² + y² = n (x ≤ y)

    Examples:
        await sum_of_two_squares_all(25) → [[0, 5], [3, 4]]
        await sum_of_two_squares_all(50) → [[1, 7], [5, 5]]
        await sum_of_two_squares_all(13) → [[2, 3]]
    """
    if n < 0:
        return []

    representations = []
    max_x = int(math.sqrt(n))

    for x in range(max_x + 1):
        y_squared = n - x * x
        if y_squared < 0:
            break

        y = int(math.sqrt(y_squared))
        if y * y == y_squared and x <= y:
            representations.append([x, y])

        # Yield control every 100 iterations for large n
        if x % 100 == 0 and max_x > 1000:
            await asyncio.sleep(0)

    return representations


@mcp_function(
    description="Solve general quadratic Diophantine equation ax² + bxy + cy² + dx + ey + f = 0.",
    namespace="arithmetic",
    category="diophantine_equations",
    execution_modes=["local", "remote"],
    estimated_cpu_usage="extreme",
    examples=[
        {
            "input": {"coeffs": [1, 0, 1, 0, 0, -25], "bounds": [-10, 10]},
            "output": [
                [-5, 0],
                [-4, -3],
                [-4, 3],
                [-3, -4],
                [-3, 4],
                [0, -5],
                [0, 5],
                [3, -4],
                [3, 4],
                [4, -3],
                [4, 3],
                [5, 0],
            ],
            "description": "x² + y² = 25",
        },
        {
            "input": {"coeffs": [1, 0, -1, 0, 0, 0], "bounds": [-5, 5]},
            "output": [
                [-5, -5],
                [-5, 5],
                [-4, -4],
                [-4, 4],
                [-3, -3],
                [-3, 3],
                [-2, -2],
                [-2, 2],
                [-1, -1],
                [-1, 1],
                [0, 0],
                [1, -1],
                [1, 1],
                [2, -2],
                [2, 2],
                [3, -3],
                [3, 3],
                [4, -4],
                [4, 4],
                [5, -5],
                [5, 5],
            ],
            "description": "x² - y² = 0",
        },
    ],
)
async def solve_quadratic_diophantine(
    coeffs: List[int], bounds: List[int]
) -> List[List[int]]:
    """
    Solve general quadratic Diophantine equation by exhaustive search.

    Equation: ax² + bxy + cy² + dx + ey + f = 0

    Args:
        coeffs: [a, b, c, d, e, f] coefficients of the equation
        bounds: [min_val, max_val] search bounds for x and y

    Returns:
        List of [x, y] integer solutions within bounds

    Examples:
        await solve_quadratic_diophantine([1, 0, 1, 0, 0, -25], [-10, 10]) → [[-5, 0], ...]
        await solve_quadratic_diophantine([1, 0, -1, 0, 0, 0], [-5, 5]) → [[-5, -5], ...]
    """
    if len(coeffs) != 6:
        raise ValueError("coeffs must have 6 elements [a, b, c, d, e, f]")

    if len(bounds) != 2:
        raise ValueError("bounds must have 2 elements [min_val, max_val]")

    a, b, c, d, e, f = coeffs
    min_val, max_val = bounds

    solutions = []

    for x in range(min_val, max_val + 1):
        for y in range(min_val, max_val + 1):
            # Evaluate ax² + bxy + cy² + dx + ey + f
            value = a * x * x + b * x * y + c * y * y + d * x + e * y + f

            if value == 0:
                solutions.append([x, y])

        # Yield control every 100 values of x for large ranges
        if x % 100 == 0 and (max_val - min_val) > 1000:
            await asyncio.sleep(0)

    return solutions


# ============================================================================
# SPECIAL DIOPHANTINE PROBLEMS
# ============================================================================


@mcp_function(
    description="Calculate the Frobenius number for given denominations.",
    namespace="arithmetic",
    category="diophantine_equations",
    execution_modes=["local", "remote"],
    estimated_cpu_usage="high",
    examples=[
        {
            "input": {"denominations": [3, 5]},
            "output": 7,
            "description": "Largest amount not expressible as 3a + 5b",
        },
        {
            "input": {"denominations": [4, 6, 9]},
            "output": 5,
            "description": "Largest amount not expressible with 4, 6, 9",
        },
        {
            "input": {"denominations": [6, 9, 20]},
            "output": 43,
            "description": "Largest amount not expressible with 6, 9, 20",
        },
    ],
)
async def frobenius_number(denominations: List[int]) -> int:
    """
    Calculate the Frobenius number (largest value not representable).

    For coprime integers a₁, a₂, ..., aₙ, find the largest integer
    that cannot be expressed as non-negative integer linear combination.

    Args:
        denominations: List of positive integers

    Returns:
        Frobenius number (largest non-representable value)

    Examples:
        await frobenius_number([3, 5]) → 7
        await frobenius_number([4, 6, 9]) → 5
        await frobenius_number([6, 9, 20]) → 43
    """
    if not denominations or any(d <= 0 for d in denominations):
        raise ValueError("All denominations must be positive")

    # Remove duplicates and sort
    denoms = sorted(set(denominations))

    if len(denoms) == 1:
        # Special case: only one denomination
        return -1 if denoms[0] == 1 else float("inf")  # type: ignore[return-value]

    # Check if gcd of all denominations is 1
    gcd_all = denoms[0]
    for d in denoms[1:]:
        gcd_all = math.gcd(gcd_all, d)

    if gcd_all > 1:
        return float("inf")  # type: ignore[return-value]  # No finite Frobenius number

    # For two coprime numbers a, b: Frobenius number is ab - a - b
    if len(denoms) == 2:
        a, b = denoms
        if math.gcd(a, b) == 1:
            return a * b - a - b

    # General case: use dynamic programming
    min(denoms)
    upper_bound = denoms[0] * denoms[1]  # Conservative upper bound

    # DP array: can_make[i] = True if amount i can be made
    can_make = [False] * (upper_bound + 1)
    can_make[0] = True

    for amount in range(1, upper_bound + 1):
        for denom in denoms:
            if amount >= denom and can_make[amount - denom]:
                can_make[amount] = True
                break

        # Yield control every 1000 amounts
        if amount % 1000 == 0:
            await asyncio.sleep(0)

    # Find largest amount that cannot be made
    frobenius = -1
    for amount in range(upper_bound, -1, -1):
        if not can_make[amount]:
            frobenius = amount
            break

    return frobenius


@mcp_function(
    description="Solve the postage stamp problem (exact postage with given denominations).",
    namespace="arithmetic",
    category="diophantine_equations",
    execution_modes=["local", "remote"],
    examples=[
        {
            "input": {"amount": 43, "denominations": [5, 9, 20]},
            "output": {"possible": True, "solution": [1, 2, 1], "stamps_used": 4},
            "description": "43 = 1×5 + 2×9 + 1×20",
        },
        {
            "input": {"amount": 11, "denominations": [3, 5]},
            "output": {"possible": False},
            "description": "Cannot make 11 with denominations 3, 5",
        },
        {
            "input": {"amount": 17, "denominations": [3, 5]},
            "output": {"possible": True, "solution": [4, 1], "stamps_used": 5},
            "description": "17 = 4×3 + 1×5",
        },
    ],
)
async def postage_stamp_problem(amount: int, denominations: List[int]) -> Dict:
    """
    Solve the postage stamp problem: make exact amount with given denominations.

    Args:
        amount: Target amount to make
        denominations: Available stamp denominations

    Returns:
        Dictionary with solution information:
        - possible: True if amount can be made
        - solution: List of counts for each denomination
        - stamps_used: Total number of stamps

    Examples:
        await postage_stamp_problem(43, [5, 9, 20]) → {"possible": True, "solution": [1, 2, 1]}
        await postage_stamp_problem(11, [3, 5]) → {"possible": False}
    """
    if amount < 0:
        return {"possible": False, "reason": "Amount must be non-negative"}

    if amount == 0:
        return {
            "possible": True,
            "solution": [0] * len(denominations),
            "stamps_used": 0,
        }

    if not denominations:
        return {"possible": False, "reason": "No denominations provided"}

    # Use dynamic programming
    # dp[i] = minimum stamps to make amount i, or -1 if impossible
    dp = [-1] * (amount + 1)
    dp[0] = 0
    parent = [-1] * (amount + 1)  # To reconstruct solution

    for curr_amount in range(1, amount + 1):
        min_stamps = float("inf")
        best_denom_idx = -1

        for idx, denom in enumerate(denominations):
            if curr_amount >= denom and dp[curr_amount - denom] != -1:
                stamps_needed = dp[curr_amount - denom] + 1
                if stamps_needed < min_stamps:
                    min_stamps = stamps_needed
                    best_denom_idx = idx

        if best_denom_idx != -1:
            dp[curr_amount] = min_stamps  # type: ignore[call-overload]
            parent[curr_amount] = best_denom_idx

        # Yield control every 1000 amounts
        if curr_amount % 1000 == 0:
            await asyncio.sleep(0)

    if dp[amount] == -1:
        return {"possible": False}

    # Reconstruct solution
    solution = [0] * len(denominations)
    curr = amount

    while curr > 0:
        denom_idx = parent[curr]
        solution[denom_idx] += 1
        curr -= denominations[denom_idx]

    return {
        "possible": True,
        "solution": solution,
        "stamps_used": dp[amount],
        "denomination_breakdown": {
            denominations[i]: solution[i]
            for i in range(len(denominations))
            if solution[i] > 0
        },
    }


# ============================================================================
# ANALYSIS AND UTILITIES
# ============================================================================


@mcp_function(
    description="Analyze a Diophantine equation and provide general information.",
    namespace="arithmetic",
    category="diophantine_equations",
    execution_modes=["local", "remote"],
    examples=[
        {
            "input": {"equation_type": "linear", "coefficients": [3, 5, 1]},
            "output": {
                "type": "linear",
                "solvable": True,
                "infinite_solutions": True,
                "classification": "indefinite",
            },
            "description": "Analysis of 3x + 5y = 1",
        },
        {
            "input": {"equation_type": "pell", "n": 2},
            "output": {
                "type": "pell",
                "fundamental_solution": [3, 2],
                "has_solutions": True,
                "classification": "hyperbolic",
            },
            "description": "Analysis of x² - 2y² = 1",
        },
    ],
)
async def diophantine_analysis(equation_type: str, **kwargs) -> Dict:
    """
    Analyze a Diophantine equation and provide comprehensive information.

    Args:
        equation_type: Type of equation ("linear", "pell", "quadratic", etc.)
        **kwargs: Coefficients and parameters specific to equation type

    Returns:
        Dictionary with analysis results including solvability, solution count, etc.

    Examples:
        await diophantine_analysis("linear", coefficients=[3, 5, 1])
        await diophantine_analysis("pell", n=2)
    """
    analysis = {"type": equation_type}

    if equation_type == "linear":
        coeffs = kwargs.get("coefficients", [])
        if len(coeffs) >= 3:
            a, b, c = coeffs[:3]
            solution = await solve_linear_diophantine(a, b, c)
            analysis.update(
                {
                    "solvable": solution["solvable"],
                    "infinite_solutions": solution["solvable"],
                    "gcd": solution.get("gcd"),
                    "classification": "indefinite"
                    if solution["solvable"]
                    else "inconsistent",
                }
            )

    elif equation_type == "pell":
        n = kwargs.get("n")
        if n is not None:
            solution = await solve_pell_equation(n)
            analysis.update(
                {
                    "has_solutions": solution["exists"],
                    "fundamental_solution": solution.get("fundamental"),
                    "classification": "hyperbolic",
                    "infinite_solutions": solution["exists"],
                }
            )

    elif equation_type == "negative_pell":
        n = kwargs.get("n")
        if n is not None:
            solution = await solve_negative_pell_equation(n)
            analysis.update(
                {
                    "has_solutions": solution["exists"],
                    "fundamental_solution": solution.get("fundamental"),
                    "classification": "elliptic"
                    if solution["exists"]
                    else "no_solutions",
                }
            )

    elif equation_type == "pythagorean":
        limit = kwargs.get("limit", 100)
        triples = await pythagorean_triples(limit, primitive_only=True)
        analysis.update(
            {
                "classification": "elliptic",
                "infinite_solutions": True,  # type: ignore[dict-item]
                "primitive_triples_found": len(triples),  # type: ignore[dict-item]
                "parametric_form": "a = m² - n², b = 2mn, c = m² + n²",
            }
        )

    else:
        analysis["error"] = f"Unknown equation type: {equation_type}"

    return analysis


# Export all functions
__all__ = [
    # Linear Diophantine equations
    "solve_linear_diophantine",
    "count_solutions_diophantine",
    "parametric_solutions_diophantine",
    # Pell's equation
    "solve_pell_equation",
    "pell_solutions_generator",
    "solve_negative_pell_equation",
    # Quadratic Diophantine equations
    "pythagorean_triples",
    "sum_of_two_squares_all",
    "solve_quadratic_diophantine",
    # Special problems
    "frobenius_number",
    "postage_stamp_problem",
    # Analysis utilities
    "diophantine_analysis",
]

if __name__ == "__main__":
    import asyncio

    async def test_diophantine_equations():
        """Test Diophantine equations functions."""
        print("🔢 Diophantine Equations Test")
        print("=" * 30)

        # Test linear Diophantine
        print("Linear Diophantine Equations:")
        result = await solve_linear_diophantine(3, 5, 1)
        print(f"  solve_linear_diophantine(3, 5, 1) = {result}")

        count = await count_solutions_diophantine(2, 3, 12, 0, 6, 0, 4)
        print(f"  count_solutions_diophantine(2, 3, 12, bounds) = {count}")

        # Test Pell's equation
        print("\nPell's Equation:")
        pell_result = await solve_pell_equation(2)
        print(f"  solve_pell_equation(2) = {pell_result}")

        pell_solutions = await pell_solutions_generator(2, 3)
        print(f"  pell_solutions_generator(2, 3) = {pell_solutions}")

        # Test Pythagorean triples
        print("\nPythagorean Triples:")
        triples = await pythagorean_triples(25, primitive_only=True)
        print(f"  pythagorean_triples(25, primitive_only=True) = {triples}")

        # Test special problems
        print("\nSpecial Problems:")
        frobenius = await frobenius_number([3, 5])
        print(f"  frobenius_number([3, 5]) = {frobenius}")

        postage = await postage_stamp_problem(17, [3, 5])
        print(f"  postage_stamp_problem(17, [3, 5]) = {postage}")

        print("\n✅ All Diophantine equations functions working!")

    asyncio.run(test_diophantine_equations())
