#!/usr/bin/env python3
# chuk_mcp_math/number_theory/recursive_sequences.py
"""
Lucas & Recursive Sequences - Async Native

Functions for Lucas numbers, Pell numbers, Tribonacci, Padovan sequences,
and general linear recurrence sequence solvers.

Functions:
- Lucas sequences: lucas_number, lucas_sequence, lucas_u_v
- Pell sequences: pell_number, pell_lucas_number, pell_sequence
- Higher order: tribonacci_number, tetranacci_number, padovan_number
- Narayana's cow: narayana_cow_sequence, fibonacci_like_sequences
- General solvers: solve_linear_recurrence, characteristic_polynomial
- Properties: binet_formula, generating_functions, sequence_identities
"""

import math
import asyncio
from typing import List, Tuple
from chuk_mcp_math.mcp_decorator import mcp_function

# ============================================================================
# LUCAS SEQUENCES
# ============================================================================


@mcp_function(
    description="Calculate the nth Lucas number L_n.",
    namespace="arithmetic",
    category="recursive_sequences",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    estimated_cpu_usage="medium",
    examples=[
        {"input": {"n": 0}, "output": 2, "description": "L_0 = 2"},
        {"input": {"n": 1}, "output": 1, "description": "L_1 = 1"},
        {"input": {"n": 5}, "output": 11, "description": "L_5 = 11"},
        {"input": {"n": 10}, "output": 123, "description": "L_10 = 123"},
    ],
)
async def lucas_number(n: int) -> int:
    """
    Calculate the nth Lucas number.

    Lucas sequence: L_0 = 2, L_1 = 1, L_n = L_{n-1} + L_{n-2}
    Sequence: 2, 1, 3, 4, 7, 11, 18, 29, 47, 76, 123, ...

    Args:
        n: Non-negative integer index

    Returns:
        The nth Lucas number

    Examples:
        await lucas_number(0) → 2    # L_0 = 2
        await lucas_number(5) → 11   # L_5 = 11
        await lucas_number(10) → 123 # L_10 = 123
    """
    if n < 0:
        raise ValueError("Lucas number index must be non-negative")

    if n == 0:
        return 2
    if n == 1:
        return 1

    # Use iterative approach for efficiency
    a, b = 2, 1
    for i in range(2, n + 1):
        a, b = b, a + b

        # Yield control every 1000 iterations
        if i % 1000 == 0:
            await asyncio.sleep(0)

    return b


@mcp_function(
    description="Generate the first n Lucas numbers.",
    namespace="arithmetic",
    category="recursive_sequences",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    examples=[
        {
            "input": {"n": 10},
            "output": [2, 1, 3, 4, 7, 11, 18, 29, 47, 76],
            "description": "First 10 Lucas numbers",
        },
        {
            "input": {"n": 5},
            "output": [2, 1, 3, 4, 7],
            "description": "First 5 Lucas numbers",
        },
    ],
)
async def lucas_sequence(n: int) -> List[int]:
    """
    Generate the first n Lucas numbers.

    Args:
        n: Number of Lucas numbers to generate

    Returns:
        List of the first n Lucas numbers

    Examples:
        await lucas_sequence(10) → [2, 1, 3, 4, 7, 11, 18, 29, 47, 76]
        await lucas_sequence(5) → [2, 1, 3, 4, 7]
    """
    if n <= 0:
        return []

    if n == 1:
        return [2]

    sequence = [2, 1]

    for i in range(2, n):
        next_val = sequence[i - 1] + sequence[i - 2]
        sequence.append(next_val)

        # Yield control every 1000 iterations
        if i % 1000 == 0:
            await asyncio.sleep(0)

    return sequence


@mcp_function(
    description="Calculate Lucas U_n and V_n sequences with parameters P, Q.",
    namespace="arithmetic",
    category="recursive_sequences",
    execution_modes=["local", "remote"],
    estimated_cpu_usage="medium",
    examples=[
        {
            "input": {"n": 5, "P": 1, "Q": -1},
            "output": [5, 11],
            "description": "U_5 and V_5 for Fibonacci/Lucas",
        },
        {
            "input": {"n": 4, "P": 2, "Q": -1},
            "output": [6, 14],
            "description": "U_4 and V_4 for Pell sequences",
        },
        {
            "input": {"n": 3, "P": 3, "Q": 2},
            "output": [13, 17],
            "description": "Custom Lucas sequences",
        },
    ],
)
async def lucas_u_v(n: int, P: int, Q: int) -> Tuple[int, int]:
    """
    Calculate Lucas U_n and V_n sequences with parameters P and Q.

    U_n: U_0 = 0, U_1 = 1, U_n = P*U_{n-1} - Q*U_{n-2}
    V_n: V_0 = 2, V_1 = P, V_n = P*V_{n-1} - Q*V_{n-2}

    Args:
        n: Non-negative integer index
        P: Parameter P
        Q: Parameter Q

    Returns:
        Tuple (U_n, V_n)

    Examples:
        await lucas_u_v(5, 1, -1) → (5, 11)   # Fibonacci and Lucas numbers
        await lucas_u_v(4, 2, -1) → (6, 14)   # Pell and Pell-Lucas numbers
    """
    if n < 0:
        raise ValueError("Index must be non-negative")

    if n == 0:
        return (0, 2)
    if n == 1:
        return (1, P)

    # Use efficient doubling method for large n
    if n > 1000:
        return await _lucas_uv_fast(n, P, Q)

    # Iterative calculation
    u_prev, u_curr = 0, 1
    v_prev, v_curr = 2, P

    for i in range(2, n + 1):
        u_next = P * u_curr - Q * u_prev
        v_next = P * v_curr - Q * v_prev

        u_prev, u_curr = u_curr, u_next
        v_prev, v_curr = v_curr, v_next

        # Yield control every 100 iterations
        if i % 100 == 0:
            await asyncio.sleep(0)

    return (u_curr, v_curr)


# ============================================================================
# PELL SEQUENCES
# ============================================================================


@mcp_function(
    description="Calculate the nth Pell number P_n.",
    namespace="arithmetic",
    category="recursive_sequences",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    examples=[
        {"input": {"n": 0}, "output": 0, "description": "P_0 = 0"},
        {"input": {"n": 1}, "output": 1, "description": "P_1 = 1"},
        {"input": {"n": 5}, "output": 29, "description": "P_5 = 29"},
        {"input": {"n": 10}, "output": 2378, "description": "P_10 = 2378"},
    ],
)
async def pell_number(n: int) -> int:
    """
    Calculate the nth Pell number.

    Pell sequence: P_0 = 0, P_1 = 1, P_n = 2*P_{n-1} + P_{n-2}
    Sequence: 0, 1, 2, 5, 12, 29, 70, 169, 408, 985, 2378, ...

    Args:
        n: Non-negative integer index

    Returns:
        The nth Pell number

    Examples:
        await pell_number(0) → 0     # P_0 = 0
        await pell_number(5) → 29    # P_5 = 29
        await pell_number(10) → 2378 # P_10 = 2378
    """
    if n < 0:
        raise ValueError("Pell number index must be non-negative")

    if n == 0:
        return 0
    if n == 1:
        return 1

    # Use Lucas sequence with P=2, Q=-1
    u_n, _ = await lucas_u_v(n, 2, -1)
    return u_n


@mcp_function(
    description="Calculate the nth Pell-Lucas number (companion to Pell numbers).",
    namespace="arithmetic",
    category="recursive_sequences",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    examples=[
        {"input": {"n": 0}, "output": 2, "description": "Q_0 = 2"},
        {"input": {"n": 1}, "output": 2, "description": "Q_1 = 2"},
        {"input": {"n": 5}, "output": 82, "description": "Q_5 = 82"},
        {"input": {"n": 10}, "output": 6730, "description": "Q_10 = 6730"},
    ],
)
async def pell_lucas_number(n: int) -> int:
    """
    Calculate the nth Pell-Lucas number (also called Q_n).

    Pell-Lucas sequence: Q_0 = 2, Q_1 = 2, Q_n = 2*Q_{n-1} + Q_{n-2}
    Sequence: 2, 2, 6, 14, 34, 82, 198, 478, 1154, 2786, 6730, ...

    Args:
        n: Non-negative integer index

    Returns:
        The nth Pell-Lucas number

    Examples:
        await pell_lucas_number(0) → 2     # Q_0 = 2
        await pell_lucas_number(5) → 82    # Q_5 = 82
        await pell_lucas_number(10) → 6730 # Q_10 = 6730
    """
    if n < 0:
        raise ValueError("Pell-Lucas number index must be non-negative")

    # Use Lucas sequence with P=2, Q=-1
    _, v_n = await lucas_u_v(n, 2, -1)
    return v_n


@mcp_function(
    description="Generate the first n Pell numbers.",
    namespace="arithmetic",
    category="recursive_sequences",
    execution_modes=["local", "remote"],
    examples=[
        {
            "input": {"n": 10},
            "output": [0, 1, 2, 5, 12, 29, 70, 169, 408, 985],
            "description": "First 10 Pell numbers",
        },
        {
            "input": {"n": 5},
            "output": [0, 1, 2, 5, 12],
            "description": "First 5 Pell numbers",
        },
    ],
)
async def pell_sequence(n: int) -> List[int]:
    """
    Generate the first n Pell numbers.

    Args:
        n: Number of Pell numbers to generate

    Returns:
        List of the first n Pell numbers

    Examples:
        await pell_sequence(10) → [0, 1, 2, 5, 12, 29, 70, 169, 408, 985]
        await pell_sequence(5) → [0, 1, 2, 5, 12]
    """
    if n <= 0:
        return []

    if n == 1:
        return [0]

    sequence = [0, 1]

    for i in range(2, n):
        next_val = 2 * sequence[i - 1] + sequence[i - 2]
        sequence.append(next_val)

        # Yield control every 1000 iterations
        if i % 1000 == 0:
            await asyncio.sleep(0)

    return sequence


# ============================================================================
# HIGHER ORDER SEQUENCES
# ============================================================================


@mcp_function(
    description="Calculate the nth Tribonacci number T_n.",
    namespace="arithmetic",
    category="recursive_sequences",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    examples=[
        {"input": {"n": 0}, "output": 0, "description": "T_0 = 0"},
        {"input": {"n": 3}, "output": 1, "description": "T_3 = 1"},
        {"input": {"n": 10}, "output": 149, "description": "T_10 = 149"},
        {"input": {"n": 15}, "output": 3136, "description": "T_15 = 3136"},
    ],
)
async def tribonacci_number(n: int) -> int:
    """
    Calculate the nth Tribonacci number.

    Tribonacci sequence: T_0 = 0, T_1 = 0, T_2 = 1, T_n = T_{n-1} + T_{n-2} + T_{n-3}
    Sequence: 0, 0, 1, 1, 2, 4, 7, 13, 24, 44, 81, 149, 274, 504, 927, 1705, 3136, ...

    Args:
        n: Non-negative integer index

    Returns:
        The nth Tribonacci number

    Examples:
        await tribonacci_number(0) → 0    # T_0 = 0
        await tribonacci_number(10) → 149 # T_10 = 149
        await tribonacci_number(15) → 3136 # T_15 = 3136
    """
    if n < 0:
        raise ValueError("Tribonacci number index must be non-negative")

    if n == 0 or n == 1:
        return 0
    if n == 2:
        return 1

    # Use iterative approach
    a, b, c = 0, 0, 1

    for i in range(3, n + 1):
        a, b, c = b, c, a + b + c

        # Yield control every 1000 iterations
        if i % 1000 == 0:
            await asyncio.sleep(0)

    return c


@mcp_function(
    description="Calculate the nth Tetranacci number (4th order Fibonacci).",
    namespace="arithmetic",
    category="recursive_sequences",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    examples=[
        {"input": {"n": 0}, "output": 0, "description": "Tet_0 = 0"},
        {"input": {"n": 4}, "output": 1, "description": "Tet_4 = 1"},
        {"input": {"n": 10}, "output": 56, "description": "Tet_10 = 56"},
        {"input": {"n": 15}, "output": 1705, "description": "Tet_15 = 1705"},
    ],
)
async def tetranacci_number(n: int) -> int:
    """
    Calculate the nth Tetranacci number.

    Tetranacci sequence: first 4 terms are 0,0,0,1, then each term is sum of previous 4
    Sequence: 0, 0, 0, 1, 1, 2, 4, 8, 15, 29, 56, 108, 208, 401, 773, 1490, ...

    Args:
        n: Non-negative integer index

    Returns:
        The nth Tetranacci number

    Examples:
        await tetranacci_number(0) → 0    # Tet_0 = 0
        await tetranacci_number(10) → 56  # Tet_10 = 56
        await tetranacci_number(15) → 1490 # Tet_15 = 1490
    """
    if n < 0:
        raise ValueError("Tetranacci number index must be non-negative")

    if n < 3:
        return 0
    if n == 3:
        return 1

    # Use iterative approach with 4 terms
    a, b, c, d = 0, 0, 0, 1

    for i in range(4, n + 1):
        a, b, c, d = b, c, d, a + b + c + d

        # Yield control every 1000 iterations
        if i % 1000 == 0:
            await asyncio.sleep(0)

    return d


@mcp_function(
    description="Calculate the nth Padovan number P_n.",
    namespace="arithmetic",
    category="recursive_sequences",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    examples=[
        {"input": {"n": 0}, "output": 1, "description": "P_0 = 1"},
        {"input": {"n": 5}, "output": 2, "description": "P_5 = 2"},
        {"input": {"n": 10}, "output": 7, "description": "P_10 = 7"},
        {"input": {"n": 15}, "output": 21, "description": "P_15 = 21"},
    ],
)
async def padovan_number(n: int) -> int:
    """
    Calculate the nth Padovan number.

    Padovan sequence: P_0 = 1, P_1 = 1, P_2 = 1, P_n = P_{n-2} + P_{n-3}
    Sequence: 1, 1, 1, 2, 2, 3, 4, 5, 7, 9, 12, 16, 21, 28, 37, 49, ...

    Args:
        n: Non-negative integer index

    Returns:
        The nth Padovan number

    Examples:
        await padovan_number(0) → 1   # P_0 = 1
        await padovan_number(10) → 7  # P_10 = 7
        await padovan_number(15) → 21 # P_15 = 21
    """
    if n < 0:
        raise ValueError("Padovan number index must be non-negative")

    if n <= 2:
        return 1

    # Use iterative approach
    a, b, c = 1, 1, 1

    for i in range(3, n + 1):
        a, b, c = b, c, a + b

        # Yield control every 1000 iterations
        if i % 1000 == 0:
            await asyncio.sleep(0)

    return c


@mcp_function(
    description="Calculate Narayana's cow sequence (cows that reproduce).",
    namespace="arithmetic",
    category="recursive_sequences",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    examples=[
        {"input": {"n": 0}, "output": 1, "description": "N_0 = 1"},
        {"input": {"n": 5}, "output": 4, "description": "N_5 = 4"},
        {"input": {"n": 10}, "output": 19, "description": "N_10 = 19"},
        {"input": {"n": 15}, "output": 84, "description": "N_15 = 84"},
    ],
)
async def narayana_cow_number(n: int) -> int:
    """
    Calculate the nth number in Narayana's cow sequence.

    Narayana's cow sequence: N_0 = 1, N_1 = 1, N_2 = 1, N_n = N_{n-1} + N_{n-3}
    Each cow produces offspring after 3 years, and both parent and offspring continue producing.
    Sequence: 1, 1, 1, 2, 3, 4, 6, 9, 13, 19, 28, 41, 60, 88, 129, ...

    Args:
        n: Non-negative integer index

    Returns:
        The nth Narayana cow number

    Examples:
        await narayana_cow_number(0) → 1   # N_0 = 1
        await narayana_cow_number(10) → 28 # N_10 = 28
        await narayana_cow_number(15) → 129 # N_15 = 129
    """
    if n < 0:
        raise ValueError("Narayana cow number index must be non-negative")

    if n <= 2:
        return 1

    # Use iterative approach
    sequence = [1, 1, 1]

    for i in range(3, n + 1):
        next_val = sequence[i - 1] + sequence[i - 3]
        sequence.append(next_val)

        # Yield control every 1000 iterations
        if i % 1000 == 0:
            await asyncio.sleep(0)

    return sequence[n]


# ============================================================================
# GENERAL LINEAR RECURRENCE SOLVER
# ============================================================================


@mcp_function(
    description="Solve general linear recurrence relation with given coefficients and initial values.",
    namespace="arithmetic",
    category="recursive_sequences",
    execution_modes=["local", "remote"],
    estimated_cpu_usage="high",
    examples=[
        {
            "input": {"coeffs": [1, 1], "initial": [0, 1], "n": 10},
            "output": 55,
            "description": "Fibonacci: a_n = a_{n-1} + a_{n-2}",
        },
        {
            "input": {"coeffs": [2, 1], "initial": [0, 1], "n": 5},
            "output": 29,
            "description": "Pell: a_n = 2*a_{n-1} + a_{n-2}",
        },
        {
            "input": {"coeffs": [1, 1, 1], "initial": [0, 0, 1], "n": 10},
            "output": 149,
            "description": "Tribonacci: a_n = a_{n-1} + a_{n-2} + a_{n-3}",
        },
    ],
)
async def solve_linear_recurrence(coeffs: List[int], initial: List[int], n: int) -> int:
    """
    Solve general linear recurrence relation.

    For recurrence a_n = c_1*a_{n-1} + c_2*a_{n-2} + ... + c_k*a_{n-k}

    Args:
        coeffs: Coefficients [c_1, c_2, ..., c_k]
        initial: Initial values [a_0, a_1, ..., a_{k-1}]
        n: Index of term to calculate

    Returns:
        The nth term of the sequence

    Examples:
        await solve_linear_recurrence([1, 1], [0, 1], 10) → 55  # Fibonacci F_10
        await solve_linear_recurrence([2, 1], [0, 1], 5) → 29   # Pell P_5
    """
    if len(coeffs) != len(initial):
        raise ValueError("Number of coefficients must equal number of initial values")

    if n < 0:
        raise ValueError("Index must be non-negative")

    k = len(coeffs)
    if n < k:
        return initial[n]

    # Use iterative calculation
    sequence = list(initial)

    for i in range(k, n + 1):
        # Calculate next value using the last k values
        next_val = sum(coeffs[j] * sequence[-(j + 1)] for j in range(k))
        sequence.append(next_val)

        # Keep only the last k values to save memory
        if len(sequence) > k:
            sequence = sequence[-k:]

        # Yield control every 1000 iterations
        if i % 1000 == 0:
            await asyncio.sleep(0)

    return sequence[-1]


@mcp_function(
    description="Find characteristic polynomial of linear recurrence relation.",
    namespace="arithmetic",
    category="recursive_sequences",
    execution_modes=["local", "remote"],
    examples=[
        {
            "input": {"coeffs": [1, 1]},
            "output": [1, -1, -1],
            "description": "Fibonacci: x² - x - 1",
        },
        {
            "input": {"coeffs": [2, 1]},
            "output": [1, -2, -1],
            "description": "Pell: x² - 2x - 1",
        },
        {
            "input": {"coeffs": [1, 1, 1]},
            "output": [1, -1, -1, -1],
            "description": "Tribonacci: x³ - x² - x - 1",
        },
    ],
)
async def characteristic_polynomial(coeffs: List[int]) -> List[int]:
    """
    Find characteristic polynomial of linear recurrence relation.

    For recurrence a_n = c_1*a_{n-1} + c_2*a_{n-2} + ... + c_k*a_{n-k}
    Returns polynomial x^k - c_1*x^{k-1} - c_2*x^{k-2} - ... - c_k

    Args:
        coeffs: Recurrence coefficients [c_1, c_2, ..., c_k]

    Returns:
        Polynomial coefficients [1, -c_1, -c_2, ..., -c_k]

    Examples:
        await characteristic_polynomial([1, 1]) → [1, -1, -1]    # x² - x - 1
        await characteristic_polynomial([2, 1]) → [1, -2, -1]    # x² - 2x - 1
    """
    if not coeffs:
        return [1]

    # Characteristic polynomial: x^k - c_1*x^{k-1} - ... - c_k = 0
    poly = [1] + [-c for c in coeffs]
    return poly


@mcp_function(
    description="Calculate Binet-style formula for sequence using characteristic roots.",
    namespace="arithmetic",
    category="recursive_sequences",
    execution_modes=["local", "remote"],
    estimated_cpu_usage="high",
    examples=[
        {
            "input": {"coeffs": [1, 1], "initial": [0, 1], "n": 10},
            "output": 55.0,
            "description": "Fibonacci using Binet formula",
        },
        {
            "input": {"coeffs": [1], "initial": [2], "n": 5},
            "output": 11.0,
            "description": "Lucas numbers",
        },
    ],
)
async def binet_formula(coeffs: List[int], initial: List[int], n: int) -> float:
    """
    Calculate nth term using Binet-style formula with characteristic roots.

    Args:
        coeffs: Recurrence coefficients
        initial: Initial values
        n: Index of term to calculate

    Returns:
        Approximate nth term using closed form

    Examples:
        await binet_formula([1, 1], [0, 1], 10) → 55.0  # Fibonacci F_10
    """
    if len(coeffs) != len(initial):
        raise ValueError("Coefficients and initial values must have same length")

    if n < 0:
        return 0.0

    if n < len(initial):
        return float(initial[n])

    # Get characteristic polynomial
    char_poly = await characteristic_polynomial(coeffs)

    # Find roots (simplified for degree 2)
    if len(char_poly) == 3:  # Quadratic case
        a, b, c = char_poly
        discriminant = b * b - 4 * a * c

        if discriminant >= 0:
            sqrt_disc = math.sqrt(discriminant)
            r1 = (-b + sqrt_disc) / (2 * a)
            r2 = (-b - sqrt_disc) / (2 * a)

            # Solve for constants using initial conditions
            if len(initial) >= 2:
                # For F_0 = initial[0], F_1 = initial[1]
                # A*r1^0 + B*r2^0 = initial[0] → A + B = initial[0]
                # A*r1^1 + B*r2^1 = initial[1] → A*r1 + B*r2 = initial[1]
                det = r1 - r2
                if abs(det) > 1e-10:
                    A = (initial[1] - initial[0] * r2) / det
                    B = (initial[0] * r1 - initial[1]) / det

                    result = A * (r1**n) + B * (r2**n)
                    return result

    # Fallback to iterative method
    return float(await solve_linear_recurrence(coeffs, initial, n))


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


async def _lucas_uv_fast(n: int, P: int, Q: int) -> Tuple[int, int]:
    """Fast calculation of Lucas U_n, V_n using binary method."""
    if n == 0:
        return (0, 2)
    if n == 1:
        return (1, P)

    # Use doubling formulas for efficiency
    # U_{2k} = U_k * V_k
    # V_{2k} = V_k^2 - 2*Q^k
    # U_{2k+1} = (P*U_{2k} + V_{2k}) / 2
    # V_{2k+1} = (P*V_{2k} + D*U_{2k}) / 2, where D = P^2 - 4Q

    # For now, use iterative method with some optimizations
    u_prev, u_curr = 0, 1
    v_prev, v_curr = 2, P

    for i in range(2, n + 1):
        u_next = P * u_curr - Q * u_prev
        v_next = P * v_curr - Q * v_prev

        u_prev, u_curr = u_curr, u_next
        v_prev, v_curr = v_curr, v_next

        # Yield control every 100 iterations
        if i % 100 == 0:
            await asyncio.sleep(0)

    return (u_curr, v_curr)


# Export all functions
__all__ = [
    # Lucas sequences
    "lucas_number",
    "lucas_sequence",
    "lucas_u_v",
    # Pell sequences
    "pell_number",
    "pell_lucas_number",
    "pell_sequence",
    # Higher order sequences
    "tribonacci_number",
    "tetranacci_number",
    "padovan_number",
    "narayana_cow_number",
    # General solvers
    "solve_linear_recurrence",
    "characteristic_polynomial",
    "binet_formula",
]
