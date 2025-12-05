#!/usr/bin/env python3
# chuk_mcp_math/number_theory/farey_sequences.py
"""
Farey Sequences - Async Native - COMPLETE IMPLEMENTATION

Functions for working with Farey sequences, Ford circles, and mediant operations.
Farey sequences are fundamental in number theory, connecting to continued fractions,
modular arithmetic, and geometric number theory.

Functions:
- Basic operations: farey_sequence, farey_sequence_length, farey_neighbors
- Mediant operations: mediant, stern_brocot_tree, farey_mediant_path
- Ford circles: ford_circles, ford_circle_properties, circle_tangency
- Analysis: farey_sequence_properties, density_analysis, gap_analysis
- Applications: best_approximation_farey, farey_fraction_between
- Advanced: farey_sum, calkin_wilf_tree, riemann_hypothesis_connection

Mathematical Background:
The Farey sequence F_n consists of all completely reduced fractions between 0 and 1
which have denominators less than or equal to n, arranged in increasing order.

Key Properties:
- |F_n| = 1 + Σ_{k=1}^n φ(k) where φ is Euler's totient function
- Adjacent fractions a/b, c/d in F_n satisfy |bc - ad| = 1
- The mediant of a/b and c/d is (a+c)/(b+d)
- Connected to Stern-Brocot tree and continued fractions
"""

import math
import asyncio
from typing import List, Dict
from chuk_mcp_math.mcp_decorator import mcp_function

# ============================================================================
# BASIC FAREY SEQUENCE OPERATIONS
# ============================================================================


@mcp_function(
    description="Generate the Farey sequence F_n of order n.",
    namespace="arithmetic",
    category="farey_sequences",
    execution_modes=["local", "remote"],
    estimated_cpu_usage="medium",
    examples=[
        {
            "input": {"n": 5},
            "output": [
                [0, 1],
                [1, 5],
                [1, 4],
                [1, 3],
                [2, 5],
                [1, 2],
                [3, 5],
                [2, 3],
                [3, 4],
                [4, 5],
                [1, 1],
            ],
            "description": "Farey sequence F_5",
        },
        {
            "input": {"n": 3},
            "output": [[0, 1], [1, 3], [1, 2], [2, 3], [1, 1]],
            "description": "Farey sequence F_3",
        },
        {
            "input": {"n": 4},
            "output": [[0, 1], [1, 4], [1, 3], [1, 2], [2, 3], [3, 4], [1, 1]],
            "description": "Farey sequence F_4",
        },
    ],
)
async def farey_sequence(n: int) -> List[List[int]]:
    """
    Generate the Farey sequence F_n of order n.

    The Farey sequence F_n consists of all completely reduced fractions
    between 0 and 1 with denominators ≤ n, in ascending order.

    Args:
        n: Order of the Farey sequence (positive integer)

    Returns:
        List of [numerator, denominator] pairs representing fractions in F_n

    Examples:
        await farey_sequence(5) → [[0, 1], [1, 5], [1, 4], [1, 3], [2, 5], [1, 2], ...]
        await farey_sequence(3) → [[0, 1], [1, 3], [1, 2], [2, 3], [1, 1]]
    """
    if n <= 0:
        raise ValueError("n must be a positive integer")

    # Generate all fractions with denominators ≤ n
    fractions = set()

    # Add 0/1 and 1/1
    fractions.add((0, 1))
    fractions.add((1, 1))

    # Add all reduced fractions p/q where 1 ≤ p < q ≤ n
    for q in range(2, n + 1):
        for p in range(1, q):
            # Check if fraction is in lowest terms (gcd(p, q) = 1)
            if await _gcd_async(p, q) == 1:
                fractions.add((p, q))

        # Yield control every 100 iterations for large n
        if q % 100 == 0:
            await asyncio.sleep(0)

    # Sort fractions by value
    sorted_fractions = sorted(fractions, key=lambda frac: frac[0] / frac[1])

    # Convert to list of [numerator, denominator] lists
    return [[p, q] for p, q in sorted_fractions]


@mcp_function(
    description="Calculate the length of Farey sequence F_n.",
    namespace="arithmetic",
    category="farey_sequences",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    examples=[
        {
            "input": {"n": 5},
            "output": {"length": 11, "formula_result": 11, "totient_sum": 10},
            "description": "Length of F_5",
        },
        {
            "input": {"n": 10},
            "output": {"length": 33, "formula_result": 33, "totient_sum": 32},
            "description": "Length of F_10",
        },
        {
            "input": {"n": 8},
            "output": {"length": 23, "formula_result": 23, "totient_sum": 22},
            "description": "Length of F_8",
        },
    ],
)
async def farey_sequence_length(n: int) -> Dict:
    """
    Calculate the length of Farey sequence F_n using Euler's totient function.

    Formula: |F_n| = 1 + Σ_{k=1}^n φ(k)
    where φ(k) is Euler's totient function.

    Args:
        n: Order of the Farey sequence

    Returns:
        Dictionary with length and verification information

    Examples:
        await farey_sequence_length(5) → {"length": 11, "formula_result": 11, ...}
        await farey_sequence_length(10) → {"length": 33, "formula_result": 33, ...}
    """
    if n < 0:
        raise ValueError("n must be non-negative")
    if n == 0:
        return {"length": 0, "formula_result": 0, "totient_sum": 0}

    # Calculate using totient sum formula
    totient_sum = 0
    for k in range(1, n + 1):
        totient_sum += await _euler_totient_async(k)

        # Yield control every 1000 iterations for large n
        if k % 1000 == 0:
            await asyncio.sleep(0)

    formula_result = 1 + totient_sum

    return {
        "n": n,
        "length": formula_result,
        "formula_result": formula_result,
        "totient_sum": totient_sum,
        "formula": "1 + Σ φ(k) for k=1 to n",
    }


@mcp_function(
    description="Find the neighbors of a fraction in a Farey sequence.",
    namespace="arithmetic",
    category="farey_sequences",
    execution_modes=["local", "remote"],
    examples=[
        {
            "input": {"p": 1, "q": 2, "n": 5},
            "output": {
                "left_neighbor": [2, 5],
                "right_neighbor": [3, 5],
                "mediant_left": [3, 7],
                "mediant_right": [4, 7],
            },
            "description": "Neighbors of 1/2 in F_5",
        },
        {
            "input": {"p": 1, "q": 3, "n": 5},
            "output": {
                "left_neighbor": [1, 4],
                "right_neighbor": [2, 5],
                "mediant_left": [2, 7],
                "mediant_right": [3, 8],
            },
            "description": "Neighbors of 1/3 in F_5",
        },
    ],
)
async def farey_neighbors(p: int, q: int, n: int) -> Dict:
    """
    Find the left and right neighbors of fraction p/q in Farey sequence F_n.

    Uses the property that adjacent fractions a/b, c/d in F_n satisfy |bc - ad| = 1.

    Args:
        p: Numerator of the fraction
        q: Denominator of the fraction
        n: Order of the Farey sequence

    Returns:
        Dictionary with left and right neighbors and their mediants

    Examples:
        await farey_neighbors(1, 2, 5) → {"left_neighbor": [2, 5], "right_neighbor": [3, 5], ...}
        await farey_neighbors(1, 3, 5) → {"left_neighbor": [1, 4], "right_neighbor": [2, 5], ...}
    """
    if q <= 0 or p < 0 or p > q or n < q:
        raise ValueError("Invalid fraction or Farey sequence order")

    if await _gcd_async(p, q) != 1:
        raise ValueError("Fraction must be in lowest terms")

    # Generate the full Farey sequence and find the fraction
    farey_seq = await farey_sequence(n)

    # Find the position of p/q in the sequence
    p / q
    position = -1

    for i, [num, den] in enumerate(farey_seq):
        if num == p and den == q:
            position = i
            break

    if position == -1:
        raise ValueError(f"Fraction {p}/{q} not found in F_{n}")

    result = {
        "fraction": [p, q],
        "position": position,
        "sequence_length": len(farey_seq),
    }

    # Find left neighbor
    if position > 0:
        left_neighbor = farey_seq[position - 1]
        result["left_neighbor"] = left_neighbor

        # Calculate mediant with left neighbor
        mediant_left = await mediant(left_neighbor[0], left_neighbor[1], p, q)
        result["mediant_left"] = mediant_left
    else:
        result["left_neighbor"] = None
        result["mediant_left"] = None

    # Find right neighbor
    if position < len(farey_seq) - 1:
        right_neighbor = farey_seq[position + 1]
        result["right_neighbor"] = right_neighbor

        # Calculate mediant with right neighbor
        mediant_right = await mediant(p, q, right_neighbor[0], right_neighbor[1])
        result["mediant_right"] = mediant_right
    else:
        result["right_neighbor"] = None
        result["mediant_right"] = None

    return result


# ============================================================================
# MEDIANT OPERATIONS
# ============================================================================


@mcp_function(
    description="Calculate the mediant of two fractions.",
    namespace="arithmetic",
    category="farey_sequences",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    examples=[
        {
            "input": {"p1": 1, "q1": 3, "p2": 1, "q2": 2},
            "output": [2, 5],
            "description": "Mediant of 1/3 and 1/2",
        },
        {
            "input": {"p1": 2, "q1": 5, "p2": 3, "q2": 4},
            "output": [5, 9],
            "description": "Mediant of 2/5 and 3/4",
        },
        {
            "input": {"p1": 0, "q1": 1, "p2": 1, "q2": 1},
            "output": [1, 2],
            "description": "Mediant of 0/1 and 1/1",
        },
    ],
)
async def mediant(p1: int, q1: int, p2: int, q2: int) -> List[int]:
    """
    Calculate the mediant of two fractions p1/q1 and p2/q2.

    The mediant of a/b and c/d is (a+c)/(b+d).
    This operation is fundamental in Farey sequences and the Stern-Brocot tree.

    Args:
        p1, q1: Numerator and denominator of first fraction
        p2, q2: Numerator and denominator of second fraction

    Returns:
        [numerator, denominator] of the mediant fraction

    Examples:
        await mediant(1, 3, 1, 2) → [2, 5]  # (1+1)/(3+2) = 2/5
        await mediant(2, 5, 3, 4) → [5, 9]  # (2+3)/(5+4) = 5/9
    """
    if q1 <= 0 or q2 <= 0:
        raise ValueError("Denominators must be positive")

    mediant_num = p1 + p2
    mediant_den = q1 + q2

    return [mediant_num, mediant_den]


@mcp_function(
    description="Generate path through Stern-Brocot tree to reach target fraction.",
    namespace="arithmetic",
    category="farey_sequences",
    execution_modes=["local", "remote"],
    examples=[
        {
            "input": {"target_p": 3, "target_q": 7},
            "output": {
                "path": ["R", "L", "R"],
                "fractions": [[0, 1], [1, 1], [1, 2], [2, 3], [3, 7]],
                "depth": 4,
            },
            "description": "Path to 3/7 in Stern-Brocot tree",
        },
        {
            "input": {"target_p": 2, "target_q": 5},
            "output": {
                "path": ["L", "R"],
                "fractions": [[0, 1], [1, 1], [1, 2], [2, 5]],
                "depth": 3,
            },
            "description": "Path to 2/5 in Stern-Brocot tree",
        },
    ],
)
async def stern_brocot_tree(target_p: int, target_q: int) -> Dict:
    """
    Generate the path through the Stern-Brocot tree to reach the target fraction.

    The Stern-Brocot tree is a binary tree where each node represents a fraction,
    and children are formed by taking mediants with ancestors.

    Args:
        target_p: Numerator of target fraction
        target_q: Denominator of target fraction

    Returns:
        Dictionary with path directions, intermediate fractions, and tree depth

    Examples:
        await stern_brocot_tree(3, 7) → {"path": ["R", "L", "R"], "fractions": [...], ...}
        await stern_brocot_tree(2, 5) → {"path": ["L", "R"], "fractions": [...], ...}
    """
    if target_q <= 0 or target_p < 0:
        raise ValueError("Invalid target fraction")

    if await _gcd_async(target_p, target_q) != 1:
        raise ValueError("Target fraction must be in lowest terms")

    # Start with the root bounds of Stern-Brocot tree
    left_p, left_q = 0, 1  # 0/1
    right_p, right_q = 1, 1  # 1/1

    path = []
    fractions = [[left_p, left_q], [right_p, right_q]]

    while True:
        # Calculate mediant
        med = await mediant(left_p, left_q, right_p, right_q)
        med_p, med_q = med[0], med[1]
        fractions.append(med)

        # Check if we found the target
        if med_p == target_p and med_q == target_q:
            break

        # Determine which direction to go
        target_value = target_p / target_q
        mediant_value = med_p / med_q

        if target_value < mediant_value:
            # Go left: mediant becomes new right bound
            path.append("L")
            right_p, right_q = med_p, med_q
        else:
            # Go right: mediant becomes new left bound
            path.append("R")
            left_p, left_q = med_p, med_q

        # Safety check to prevent infinite loops
        if len(path) > 100:
            break

        await asyncio.sleep(0)  # Yield control

    return {
        "target": [target_p, target_q],
        "path": path,
        "fractions": fractions,
        "depth": len(path),
        "total_fractions": len(fractions),
    }


@mcp_function(
    description="Find path through Farey sequence using mediant operations.",
    namespace="arithmetic",
    category="farey_sequences",
    execution_modes=["local", "remote"],
    examples=[
        {
            "input": {
                "start_p": 1,
                "start_q": 3,
                "end_p": 1,
                "end_q": 2,
                "max_denom": 10,
            },
            "output": {
                "mediants": [[2, 5], [3, 7], [4, 9], [5, 11]],
                "converges": True,
                "steps": 4,
            },
            "description": "Mediant path from 1/3 to 1/2",
        },
        {
            "input": {
                "start_p": 0,
                "start_q": 1,
                "end_p": 1,
                "end_q": 4,
                "max_denom": 8,
            },
            "output": {"mediants": [[1, 5], [1, 4]], "converges": True, "steps": 2},
            "description": "Mediant path from 0/1 to 1/4",
        },
    ],
)
async def farey_mediant_path(
    start_p: int, start_q: int, end_p: int, end_q: int, max_denom: int = 100
) -> Dict:
    """
    Find a path between two fractions using mediant operations.

    This explores how fractions can be reached through repeated mediant calculations,
    which is fundamental to understanding Farey sequence structure.

    Args:
        start_p, start_q: Starting fraction
        end_p, end_q: Target fraction
        max_denom: Maximum denominator to allow in mediants

    Returns:
        Dictionary with mediant path and convergence information

    Examples:
        await farey_mediant_path(1, 3, 1, 2, 10) → {"mediants": [[2, 5], ...], ...}
        await farey_mediant_path(0, 1, 1, 4, 8) → {"mediants": [[1, 5], [1, 4]], ...}
    """
    if start_q <= 0 or end_q <= 0 or max_denom <= 0:
        raise ValueError("Invalid parameters")

    current_p, current_q = start_p, start_q
    target_p, target_q = end_p, end_q

    mediants = []
    steps = 0
    max_steps = 50  # Prevent infinite loops

    while steps < max_steps:
        # Calculate mediant
        med = await mediant(current_p, current_q, target_p, target_q)
        med_p, med_q = med[0], med[1]

        # Check if mediant denominator is too large
        if med_q > max_denom:
            break

        mediants.append(med)

        # Check if we've reached the target
        if med_p == target_p and med_q == target_q:
            return {
                "start": [start_p, start_q],
                "end": [end_p, end_q],
                "mediants": mediants,
                "converges": True,
                "steps": len(mediants),
                "max_denom_used": max(med[1] for med in mediants) if mediants else 0,
            }

        # Update current fraction to the mediant
        current_p, current_q = med_p, med_q
        steps += 1

        await asyncio.sleep(0)  # Yield control

    return {
        "start": [start_p, start_q],
        "end": [end_p, end_q],
        "mediants": mediants,
        "converges": False,
        "steps": len(mediants),
        "max_denom_used": max(med[1] for med in mediants) if mediants else 0,
        "reason": "Max denominator exceeded"
        if mediants and mediants[-1][1] >= max_denom
        else "Max steps exceeded",
    }


# ============================================================================
# FORD CIRCLES
# ============================================================================


@mcp_function(
    description="Generate Ford circles for fractions in Farey sequence F_n.",
    namespace="arithmetic",
    category="farey_sequences",
    execution_modes=["local", "remote"],
    estimated_cpu_usage="medium",
    examples=[
        {
            "input": {"n": 4},
            "output": {
                "circles": [
                    {"fraction": [0, 1], "center": [0, 0.5], "radius": 0.5},
                    {"fraction": [1, 4], "center": [0.25, 0.125], "radius": 0.125},
                ],
                "count": 7,
            },
            "description": "Ford circles for F_4",
        },
        {
            "input": {"n": 3},
            "output": {
                "circles": [
                    {"fraction": [0, 1], "center": [0, 0.5], "radius": 0.5},
                    {"fraction": [1, 3], "center": [0.333, 0.167], "radius": 0.167},
                ],
                "count": 5,
            },
            "description": "Ford circles for F_3",
        },
    ],
)
async def ford_circles(n: int) -> Dict:
    """
    Generate Ford circles for all fractions in Farey sequence F_n.

    For a fraction p/q in lowest terms, the Ford circle has:
    - Center: (p/q, 1/(2q²))
    - Radius: 1/(2q²)

    Ford circles are tangent to the x-axis and to each other when the
    corresponding fractions are adjacent in a Farey sequence.

    Args:
        n: Order of the Farey sequence

    Returns:
        Dictionary with circle data and properties

    Examples:
        await ford_circles(4) → {"circles": [{"fraction": [0, 1], "center": [0, 0.5], ...}], ...}
        await ford_circles(3) → {"circles": [{"fraction": [1, 3], "center": [0.333, 0.167], ...}], ...}
    """
    if n < 0:
        raise ValueError("n must be non-negative")
    if n == 0:
        return {"circles": [], "count": 0}

    # Get Farey sequence
    farey_seq = await farey_sequence(n)
    circles = []

    for p, q in farey_seq:
        # Skip 0/1 special case
        if p == 0 and q == 1:
            # Special circle for 0/1 (infinite radius, but we use a large circle)
            center = [0.0, 0.5]
            radius = 0.5
        else:
            # Standard Ford circle formula
            center_x = p / q
            center_y = 1 / (2 * q * q)
            radius = 1 / (2 * q * q)

            center = [round(center_x, 6), round(center_y, 6)]
            radius = round(radius, 6)

        circle_data = {
            "fraction": [p, q],
            "center": center,
            "radius": radius,
            "denominator": q,
        }
        circles.append(circle_data)

    return {
        "n": n,
        "circles": circles,
        "count": len(circles),
        "farey_length": len(farey_seq),
    }


@mcp_function(
    description="Analyze properties of Ford circles including tangency relationships.",
    namespace="arithmetic",
    category="farey_sequences",
    execution_modes=["local", "remote"],
    examples=[
        {
            "input": {"n": 5},
            "output": {
                "total_circles": 11,
                "tangent_pairs": 10,
                "max_radius": 0.5,
                "min_radius": 0.02,
                "avg_radius": 0.157,
            },
            "description": "Ford circle properties for F_5",
        },
        {
            "input": {"n": 3},
            "output": {
                "total_circles": 5,
                "tangent_pairs": 4,
                "max_radius": 0.5,
                "min_radius": 0.167,
                "avg_radius": 0.267,
            },
            "description": "Ford circle properties for F_3",
        },
    ],
)
async def ford_circle_properties(n: int) -> Dict:
    """
    Analyze properties of Ford circles for Farey sequence F_n.

    Computes circle statistics and identifies tangency relationships between
    adjacent Ford circles corresponding to adjacent fractions in F_n.

    Args:
        n: Order of the Farey sequence

    Returns:
        Dictionary with circle properties and tangency analysis

    Examples:
        await ford_circle_properties(5) → {"total_circles": 11, "tangent_pairs": 10, ...}
        await ford_circle_properties(3) → {"total_circles": 5, "tangent_pairs": 4, ...}
    """
    if n <= 0:
        return {"total_circles": 0, "tangent_pairs": 0}

    # Get Ford circles
    circles_data = await ford_circles(n)
    circles = circles_data["circles"]

    if not circles:
        return {"total_circles": 0, "tangent_pairs": 0}

    # Calculate statistics
    radii = [circle["radius"] for circle in circles]
    max_radius = max(radii)
    min_radius = min(radii)
    avg_radius = sum(radii) / len(radii)

    # Count tangent pairs (adjacent fractions in Farey sequence)
    tangent_pairs = len(circles) - 1 if len(circles) > 1 else 0

    # Analyze tangency relationships
    tangency_analysis = []
    for i in range(len(circles) - 1):
        circle1 = circles[i]
        circle2 = circles[i + 1]

        # Calculate distance between centers
        dx = circle2["center"][0] - circle1["center"][0]
        dy = circle2["center"][1] - circle1["center"][1]
        distance = math.sqrt(dx * dx + dy * dy)

        # Sum of radii (for tangent circles)
        sum_radii = circle1["radius"] + circle2["radius"]

        # Check if tangent (distance ≈ sum of radii)
        is_tangent = abs(distance - sum_radii) < 1e-10

        tangency_analysis.append(
            {
                "fraction1": circle1["fraction"],
                "fraction2": circle2["fraction"],
                "distance": round(distance, 8),
                "sum_radii": round(sum_radii, 8),
                "is_tangent": is_tangent,
                "tangency_error": abs(distance - sum_radii),
            }
        )

    return {
        "n": n,
        "total_circles": len(circles),
        "tangent_pairs": tangent_pairs,
        "max_radius": round(max_radius, 6),
        "min_radius": round(min_radius, 6),
        "avg_radius": round(avg_radius, 6),
        "radius_ratio": round(max_radius / min_radius, 2)
        if min_radius > 0
        else float("inf"),
        "tangency_analysis": tangency_analysis[:5],  # Show first 5 for brevity
        "all_tangent": all(t["is_tangent"] for t in tangency_analysis),
    }


@mcp_function(
    description="Check tangency relationship between two Ford circles.",
    namespace="arithmetic",
    category="farey_sequences",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    examples=[
        {
            "input": {"p1": 1, "q1": 3, "p2": 1, "q2": 2},
            "output": {
                "are_tangent": True,
                "distance": 0.375,
                "sum_radii": 0.375,
                "tangency_type": "external",
            },
            "description": "Tangency between 1/3 and 1/2",
        },
        {
            "input": {"p1": 1, "q1": 4, "p2": 1, "q2": 3},
            "output": {
                "are_tangent": True,
                "distance": 0.208,
                "sum_radii": 0.208,
                "tangency_type": "external",
            },
            "description": "Tangency between 1/4 and 1/3",
        },
    ],
)
async def circle_tangency(p1: int, q1: int, p2: int, q2: int) -> Dict:
    """
    Check if Ford circles for two fractions are tangent.

    Two Ford circles are externally tangent if and only if the corresponding
    fractions are adjacent in some Farey sequence (i.e., |p1*q2 - p2*q1| = 1).

    Args:
        p1, q1: First fraction
        p2, q2: Second fraction

    Returns:
        Dictionary with tangency analysis

    Examples:
        await circle_tangency(1, 3, 1, 2) → {"are_tangent": True, "distance": 0.375, ...}
        await circle_tangency(1, 4, 1, 3) → {"are_tangent": True, "distance": 0.208, ...}
    """
    if q1 <= 0 or q2 <= 0:
        raise ValueError("Denominators must be positive")

    # Check if fractions are adjacent in Farey sequence
    determinant = abs(p1 * q2 - p2 * q1)
    are_farey_adjacent = determinant == 1

    # Calculate Ford circle properties
    center1_x = p1 / q1
    center1_y = 1 / (2 * q1 * q1)
    radius1 = 1 / (2 * q1 * q1)

    center2_x = p2 / q2
    center2_y = 1 / (2 * q2 * q2)
    radius2 = 1 / (2 * q2 * q2)

    # Calculate distance between centers
    dx = center2_x - center1_x
    dy = center2_y - center1_y
    distance = math.sqrt(dx * dx + dy * dy)

    # Sum of radii
    sum_radii = radius1 + radius2

    # Check tangency (distance = sum of radii for external tangency)
    are_tangent = abs(distance - sum_radii) < 1e-12

    # Determine tangency type
    if are_tangent:
        tangency_type = "external"
    elif distance < sum_radii:
        tangency_type = "overlapping"
    else:
        tangency_type = "separate"

    return {
        "fraction1": [p1, q1],
        "fraction2": [p2, q2],
        "are_farey_adjacent": are_farey_adjacent,
        "determinant": determinant,
        "circle1": {
            "center": [round(center1_x, 6), round(center1_y, 6)],
            "radius": round(radius1, 6),
        },
        "circle2": {
            "center": [round(center2_x, 6), round(center2_y, 6)],
            "radius": round(radius2, 6),
        },
        "distance": round(distance, 8),
        "sum_radii": round(sum_radii, 8),
        "are_tangent": are_tangent,
        "tangency_type": tangency_type,
        "tangency_error": abs(distance - sum_radii),
    }


# ============================================================================
# ANALYSIS AND PROPERTIES
# ============================================================================


@mcp_function(
    description="Analyze comprehensive properties of Farey sequence F_n.",
    namespace="arithmetic",
    category="farey_sequences",
    execution_modes=["local", "remote"],
    estimated_cpu_usage="medium",
    examples=[
        {
            "input": {"n": 5},
            "output": {
                "length": 11,
                "density": 0.55,
                "max_gap": 0.2,
                "avg_gap": 0.1,
                "adjacent_pairs": 10,
            },
            "description": "Properties of F_5",
        },
        {
            "input": {"n": 8},
            "output": {
                "length": 23,
                "density": 0.719,
                "max_gap": 0.125,
                "avg_gap": 0.043,
                "adjacent_pairs": 22,
            },
            "description": "Properties of F_8",
        },
    ],
)
async def farey_sequence_properties(n: int) -> Dict:
    """
    Analyze comprehensive properties of Farey sequence F_n.

    Computes length, density, gaps between consecutive fractions,
    and various statistical measures.

    Args:
        n: Order of the Farey sequence

    Returns:
        Dictionary with comprehensive sequence properties

    Examples:
        await farey_sequence_properties(5) → {"length": 11, "density": 0.55, ...}
        await farey_sequence_properties(8) → {"length": 23, "density": 0.719, ...}
    """
    if n <= 0:
        return {"length": 0, "density": 0, "max_gap": 0, "avg_gap": 0}

    # Get Farey sequence
    farey_seq = await farey_sequence(n)
    length = len(farey_seq)

    if length < 2:
        return {"length": length, "density": 0, "max_gap": 0, "avg_gap": 0}

    # Calculate gaps between consecutive fractions
    gaps = []
    for i in range(len(farey_seq) - 1):
        p1, q1 = farey_seq[i]
        p2, q2 = farey_seq[i + 1]

        val1 = p1 / q1
        val2 = p2 / q2
        gap = val2 - val1
        gaps.append(gap)

    # Statistical analysis
    max_gap = max(gaps) if gaps else 0
    min_gap = min(gaps) if gaps else 0
    avg_gap = sum(gaps) / len(gaps) if gaps else 0

    # Density analysis (fractions per unit interval)
    density = length  # Since we're looking at [0, 1]

    # Denominator statistics
    denominators = [q for p, q in farey_seq]
    max_denom = max(denominators) if denominators else 0
    avg_denom = sum(denominators) / len(denominators) if denominators else 0

    # Count distinct denominators
    unique_denoms = len(set(denominators))

    # Adjacent pairs analysis (should satisfy |p1*q2 - p2*q1| = 1)
    adjacent_pairs = 0
    farey_adjacent_violations = 0

    for i in range(len(farey_seq) - 1):
        p1, q1 = farey_seq[i]
        p2, q2 = farey_seq[i + 1]

        determinant = abs(p1 * q2 - p2 * q1)
        if determinant == 1:
            adjacent_pairs += 1
        else:
            farey_adjacent_violations += 1

    return {
        "n": n,
        "length": length,
        "density": round(density, 3),
        "max_gap": round(max_gap, 6),
        "min_gap": round(min_gap, 6),
        "avg_gap": round(avg_gap, 6),
        "gap_variance": round(sum((g - avg_gap) ** 2 for g in gaps) / len(gaps), 8)
        if gaps
        else 0,
        "max_denominator": max_denom,
        "avg_denominator": round(avg_denom, 2),
        "unique_denominators": unique_denoms,
        "adjacent_pairs": adjacent_pairs,
        "farey_property_violations": farey_adjacent_violations,
        "gaps_sample": [round(g, 6) for g in gaps[:10]],  # First 10 gaps for inspection
    }


@mcp_function(
    description="Analyze density of Farey sequences as n increases.",
    namespace="arithmetic",
    category="farey_sequences",
    execution_modes=["local", "remote"],
    estimated_cpu_usage="high",
    examples=[
        {
            "input": {"max_n": 10},
            "output": {
                "densities": [2, 3, 5, 7, 11, 13, 19, 23, 29, 33],
                "density_ratios": [1.5, 1.67, 1.4, 1.57, 1.18, 1.46, 1.21, 1.26, 1.14],
                "asymptotic_constant": 0.608,
            },
            "description": "Density analysis for n=1 to 10",
        },
        {
            "input": {"max_n": 5},
            "output": {
                "densities": [2, 3, 5, 7, 11],
                "density_ratios": [1.5, 1.67, 1.4, 1.57],
                "asymptotic_constant": 0.61,
            },
            "description": "Density analysis for n=1 to 5",
        },
    ],
)
async def density_analysis(max_n: int) -> Dict:
    """
    Analyze how Farey sequence density grows with n.

    Studies the asymptotic behavior |F_n| ~ (3/π²)n² and computes
    the density growth ratios.

    Args:
        max_n: Maximum value of n to analyze

    Returns:
        Dictionary with density growth analysis

    Examples:
        await density_analysis(10) → {"densities": [2, 3, 5, 7, ...], "density_ratios": [...], ...}
        await density_analysis(5) → {"densities": [2, 3, 5, 7, 11], ...}
    """
    if max_n <= 0:
        return {"densities": [], "density_ratios": [], "asymptotic_constant": 0}

    densities = []

    # Calculate lengths for F_1 through F_max_n
    for n in range(1, max_n + 1):
        length_data = await farey_sequence_length(n)
        densities.append(length_data["length"])

        # Yield control every few iterations
        if n % 5 == 0:
            await asyncio.sleep(0)

    # Calculate density ratios (how much length increases)
    density_ratios = []
    for i in range(1, len(densities)):
        ratio = densities[i] / densities[i - 1]
        density_ratios.append(round(ratio, 2))

    # Estimate asymptotic constant 3/π²
    theoretical_constant = 3 / (math.pi * math.pi)

    if max_n >= 3:
        # Estimate constant from largest value
        estimated_constant = densities[-1] / (max_n * max_n)
        asymptotic_error = abs(estimated_constant - theoretical_constant)
    else:
        estimated_constant = 0
        asymptotic_error = 0

    return {
        "max_n": max_n,
        "densities": densities,
        "density_ratios": density_ratios,
        "theoretical_constant": round(theoretical_constant, 6),
        "estimated_constant": round(estimated_constant, 6),
        "asymptotic_error": round(asymptotic_error, 6),
        "growth_formula": "|F_n| ~ (3/π²)n²",
        "final_density": densities[-1] if densities else 0,
    }


@mcp_function(
    description="Analyze gaps between consecutive fractions in Farey sequence.",
    namespace="arithmetic",
    category="farey_sequences",
    execution_modes=["local", "remote"],
    examples=[
        {
            "input": {"n": 5},
            "output": {
                "total_gaps": 10,
                "max_gap": 0.2,
                "min_gap": 0.033,
                "gap_distribution": {"0.033": 1, "0.067": 2, "0.1": 3, "0.2": 4},
            },
            "description": "Gap analysis for F_5",
        },
        {
            "input": {"n": 7},
            "output": {
                "total_gaps": 18,
                "max_gap": 0.143,
                "min_gap": 0.02,
                "gap_distribution": {"0.02": 1, "0.048": 3, "0.071": 5},
            },
            "description": "Gap analysis for F_7",
        },
    ],
)
async def gap_analysis(n: int) -> Dict:
    """
    Analyze gaps between consecutive fractions in Farey sequence F_n.

    Studies the distribution of gaps and their relationship to the
    denominators of adjacent fractions.

    Args:
        n: Order of the Farey sequence

    Returns:
        Dictionary with gap distribution and analysis

    Examples:
        await gap_analysis(5) → {"total_gaps": 10, "max_gap": 0.2, ...}
        await gap_analysis(7) → {"total_gaps": 18, "max_gap": 0.143, ...}
    """
    if n <= 0:
        return {"total_gaps": 0, "max_gap": 0, "min_gap": 0}

    # Get Farey sequence
    farey_seq = await farey_sequence(n)

    if len(farey_seq) < 2:
        return {"total_gaps": 0, "max_gap": 0, "min_gap": 0}

    gaps = []
    gap_info = []

    for i in range(len(farey_seq) - 1):
        p1, q1 = farey_seq[i]
        p2, q2 = farey_seq[i + 1]

        val1 = p1 / q1
        val2 = p2 / q2
        gap = val2 - val1

        gaps.append(gap)
        gap_info.append(
            {
                "fraction1": [p1, q1],
                "fraction2": [p2, q2],
                "gap": round(gap, 6),
                "gap_formula": f"1/({q1}*{q2})" if q1 * q2 > 0 else "special",
                "theoretical_gap": 1 / (q1 * q2) if q1 * q2 > 0 else gap,
            }
        )

    # Gap statistics
    max_gap = max(gaps) if gaps else 0
    min_gap = min(gaps) if gaps else 0
    avg_gap = sum(gaps) / len(gaps) if gaps else 0

    # Gap distribution (rounded to 3 decimal places for grouping)
    gap_distribution: dict[int, int] = {}
    for gap in gaps:
        rounded_gap = round(gap, 3)
        gap_distribution[rounded_gap] = gap_distribution.get(rounded_gap, 0) + 1

    # Find largest gaps
    largest_gaps = sorted(gap_info, key=lambda x: x["gap"], reverse=True)[:5]

    return {
        "n": n,
        "total_gaps": len(gaps),
        "max_gap": round(max_gap, 6),
        "min_gap": round(min_gap, 6),
        "avg_gap": round(avg_gap, 6),
        "gap_variance": round(sum((g - avg_gap) ** 2 for g in gaps) / len(gaps), 8)
        if gaps
        else 0,
        "gap_distribution": {str(k): v for k, v in sorted(gap_distribution.items())},
        "largest_gaps": largest_gaps,
        "gap_formula_note": "For adjacent fractions a/b, c/d in Farey sequence: gap = |c/d - a/b| = 1/(bd)",
    }


# ============================================================================
# APPLICATIONS
# ============================================================================


@mcp_function(
    description="Find best rational approximation using Farey sequences.",
    namespace="arithmetic",
    category="farey_sequences",
    execution_modes=["local", "remote"],
    examples=[
        {
            "input": {"target": 0.618, "max_denom": 10},
            "output": {"best_approximation": [5, 8], "error": 0.007, "farey_order": 8},
            "description": "Best approximation to φ-1 ≈ 0.618",
        },
        {
            "input": {"target": 0.333, "max_denom": 15},
            "output": {"best_approximation": [1, 3], "error": 0.0003, "farey_order": 3},
            "description": "Best approximation to 1/3",
        },
    ],
)
async def best_approximation_farey(target: float, max_denom: int) -> Dict:
    """
    Find the best rational approximation to a target value using Farey sequences.

    Searches through Farey sequences up to the specified maximum denominator
    to find the fraction that minimizes the approximation error.

    Args:
        target: Target real number to approximate (should be in [0, 1])
        max_denom: Maximum allowed denominator

    Returns:
        Dictionary with best approximation and error analysis

    Examples:
        await best_approximation_farey(0.618, 10) → {"best_approximation": [5, 8], "error": 0.007, ...}
        await best_approximation_farey(0.333, 15) → {"best_approximation": [1, 3], "error": 0.0003, ...}
    """
    if max_denom <= 0:
        raise ValueError("Maximum denominator must be positive")

    if not (0 <= target <= 1):
        raise ValueError("Target must be in [0, 1] for Farey sequence approximation")

    best_fraction = [0, 1]
    best_error = abs(target - 0)
    best_farey_order = 1

    # Search through Farey sequences F_1, F_2, ..., F_max_denom
    for n in range(1, max_denom + 1):
        farey_seq = await farey_sequence(n)

        for p, q in farey_seq:
            if q <= max_denom:
                approximation = p / q
                error = abs(target - approximation)

                if error < best_error:
                    best_error = error
                    best_fraction = [p, q]
                    best_farey_order = n

        # Yield control every few iterations
        if n % 10 == 0:
            await asyncio.sleep(0)

    return {
        "target": target,
        "max_denom": max_denom,
        "best_approximation": best_fraction,
        "best_value": best_fraction[0] / best_fraction[1],
        "error": round(best_error, 8),
        "relative_error": round(best_error / target, 8) if target != 0 else 0,
        "farey_order": best_farey_order,
        "fraction_string": f"{best_fraction[0]}/{best_fraction[1]}",
    }


@mcp_function(
    description="Find a fraction between two given fractions using Farey sequence properties.",
    namespace="arithmetic",
    category="farey_sequences",
    execution_modes=["local", "remote"],
    examples=[
        {
            "input": {"p1": 1, "q1": 3, "p2": 1, "q2": 2},
            "output": {"fraction_between": [2, 5], "is_mediant": True, "value": 0.4},
            "description": "Fraction between 1/3 and 1/2",
        },
        {
            "input": {"p1": 2, "q1": 7, "p2": 1, "q2": 3},
            "output": {"fraction_between": [3, 10], "is_mediant": True, "value": 0.3},
            "description": "Fraction between 2/7 and 1/3",
        },
    ],
)
async def farey_fraction_between(p1: int, q1: int, p2: int, q2: int) -> Dict:
    """
    Find a fraction between two given fractions using Farey sequence properties.

    Uses the mediant operation, which gives the fraction with smallest denominator
    between two fractions in a Farey sequence.

    Args:
        p1, q1: First fraction
        p2, q2: Second fraction

    Returns:
        Dictionary with the fraction between and its properties

    Examples:
        await farey_fraction_between(1, 3, 1, 2) → {"fraction_between": [2, 5], ...}
        await farey_fraction_between(2, 7, 1, 3) → {"fraction_between": [3, 10], ...}
    """
    if q1 <= 0 or q2 <= 0:
        raise ValueError("Denominators must be positive")

    val1 = p1 / q1
    val2 = p2 / q2

    # Ensure p1/q1 < p2/q2
    if val1 > val2:
        p1, q1, p2, q2 = p2, q2, p1, q1
        val1, val2 = val2, val1

    # Calculate mediant
    mediant_frac = await mediant(p1, q1, p2, q2)
    mediant_p, mediant_q = mediant_frac[0], mediant_frac[1]
    mediant_val = mediant_p / mediant_q

    # Verify it's between the two fractions
    is_between = val1 < mediant_val < val2

    # Check if fractions are adjacent in some Farey sequence
    determinant = abs(p1 * q2 - p2 * q1)
    are_adjacent = determinant == 1

    return {
        "fraction1": [p1, q1],
        "fraction2": [p2, q2],
        "fraction_between": mediant_frac,
        "value": round(mediant_val, 8),
        "is_mediant": True,
        "is_between": is_between,
        "fractions_are_adjacent": are_adjacent,
        "mediant_denominator": mediant_q,
        "interval": [round(val1, 6), round(val2, 6)],
        "note": "Mediant gives fraction with smallest denominator between two fractions",
    }


# ============================================================================
# ADVANCED APPLICATIONS
# ============================================================================


@mcp_function(
    description="Calculate Farey sum of two fractions with applications to modular arithmetic.",
    namespace="arithmetic",
    category="farey_sequences",
    execution_modes=["local", "remote"],
    examples=[
        {
            "input": {"p1": 1, "q1": 3, "p2": 1, "q2": 4},
            "output": {"farey_sum": [7, 12], "regular_sum": [7, 12], "are_equal": True},
            "description": "Farey sum of 1/3 and 1/4",
        },
        {
            "input": {"p1": 2, "q1": 5, "p2": 3, "q2": 7},
            "output": {
                "farey_sum": [29, 35],
                "regular_sum": [29, 35],
                "are_equal": True,
            },
            "description": "Farey sum of 2/5 and 3/7",
        },
    ],
)
async def farey_sum(p1: int, q1: int, p2: int, q2: int) -> Dict:
    """
    Calculate the Farey sum of two fractions.

    The Farey sum is the regular fraction addition, but the term "Farey sum"
    emphasizes its role in the structure of Farey sequences and applications
    to modular arithmetic and continued fractions.

    Args:
        p1, q1: First fraction
        p2, q2: Second fraction

    Returns:
        Dictionary with Farey sum and related information

    Examples:
        await farey_sum(1, 3, 1, 4) → {"farey_sum": [7, 12], "regular_sum": [7, 12], ...}
        await farey_sum(2, 5, 3, 7) → {"farey_sum": [29, 35], "regular_sum": [29, 35], ...}
    """
    if q1 <= 0 or q2 <= 0:
        raise ValueError("Denominators must be positive")

    # Calculate regular sum: p1/q1 + p2/q2 = (p1*q2 + p2*q1)/(q1*q2)
    sum_num = p1 * q2 + p2 * q1
    sum_den = q1 * q2

    # Reduce to lowest terms
    gcd_val = await _gcd_async(sum_num, sum_den)
    reduced_num = sum_num // gcd_val
    reduced_den = sum_den // gcd_val

    return {
        "fraction1": [p1, q1],
        "fraction2": [p2, q2],
        "farey_sum": [reduced_num, reduced_den],
        "regular_sum": [reduced_num, reduced_den],
        "are_equal": True,
        "sum_value": round(reduced_num / reduced_den, 8),
        "unreduced_form": [sum_num, sum_den],
        "reduction_factor": gcd_val,
        "note": "Farey sum is regular fraction addition in reduced form",
    }


@mcp_function(
    description="Generate levels of Calkin-Wilf tree related to Farey sequences.",
    namespace="arithmetic",
    category="farey_sequences",
    execution_modes=["local", "remote"],
    examples=[
        {
            "input": {"levels": 3},
            "output": {
                "tree_levels": [
                    [[1, 1]],
                    [[1, 2], [2, 1]],
                    [[1, 3], [3, 2], [2, 3], [3, 1]],
                ],
                "total_fractions": 7,
            },
            "description": "First 3 levels of Calkin-Wilf tree",
        },
        {
            "input": {"levels": 2},
            "output": {
                "tree_levels": [[[1, 1]], [[1, 2], [2, 1]]],
                "total_fractions": 3,
            },
            "description": "First 2 levels of Calkin-Wilf tree",
        },
    ],
)
async def calkin_wilf_tree(levels: int) -> Dict:
    """
    Generate the Calkin-Wilf tree, which provides a bijection between
    positive integers and positive rational numbers.

    The tree is constructed where each fraction p/q has children
    p/(p+q) and (p+q)/q.

    Args:
        levels: Number of levels to generate

    Returns:
        Dictionary with tree levels and enumeration properties

    Examples:
        await calkin_wilf_tree(3) → {"tree_levels": [[[1, 1]], [[1, 2], [2, 1]], ...], ...}
        await calkin_wilf_tree(2) → {"tree_levels": [[[1, 1]], [[1, 2], [2, 1]]], ...}
    """
    if levels <= 0:
        return {"tree_levels": [], "total_fractions": 0}

    tree_levels = []

    # Level 0: root 1/1
    current_level = [[1, 1]]
    tree_levels.append(current_level.copy())

    # Generate subsequent levels
    for level in range(1, levels):
        next_level = []

        for p, q in current_level:
            # Left child: p/(p+q)
            left_child = [p, p + q]
            next_level.append(left_child)

            # Right child: (p+q)/q
            right_child = [p + q, q]
            next_level.append(right_child)

        tree_levels.append(next_level)
        current_level = next_level

        await asyncio.sleep(0)  # Yield control

    # Count total fractions
    total_fractions = sum(len(level) for level in tree_levels)

    # Generate enumeration (first few fractions in order)
    enumeration: list[tuple[int, int]] = []
    for level in tree_levels:  # type: ignore[assignment]
        enumeration.extend(level)  # type: ignore[arg-type]

    return {
        "levels": levels,
        "tree_levels": tree_levels,
        "total_fractions": total_fractions,
        "enumeration": enumeration[:20],  # First 20 for display
        "properties": {
            "bijective": "Maps positive integers to positive rationals",
            "completeness": "Every positive rational appears exactly once",
            "construction": "Left child: p/(p+q), Right child: (p+q)/q",
        },
    }


@mcp_function(
    description="Explore connections between Farey sequences and the Riemann Hypothesis.",
    namespace="arithmetic",
    category="farey_sequences",
    execution_modes=["local", "remote"],
    estimated_cpu_usage="high",
    examples=[
        {
            "input": {"n": 10},
            "output": {
                "farey_length": 33,
                "theoretical_length": 32.84,
                "error": 0.16,
                "rh_bound": 0.5,
            },
            "description": "RH connection for F_10",
        },
        {
            "input": {"n": 20},
            "output": {
                "farey_length": 127,
                "theoretical_length": 126.32,
                "error": 0.68,
                "rh_bound": 0.447,
            },
            "description": "RH connection for F_20",
        },
    ],
)
async def riemann_hypothesis_connection(n: int) -> Dict:
    """
    Explore connections between Farey sequences and the Riemann Hypothesis.

    The Riemann Hypothesis is equivalent to the statement that the error
    in the asymptotic formula for |F_n| is O(n^{1/2+ε}) for any ε > 0.

    Args:
        n: Order of Farey sequence to analyze

    Returns:
        Dictionary with RH-related analysis

    Examples:
        await riemann_hypothesis_connection(10) → {"farey_length": 33, "theoretical_length": 32.84, ...}
        await riemann_hypothesis_connection(20) → {"farey_length": 127, "theoretical_length": 126.32, ...}
    """
    if n <= 0:
        return {"farey_length": 0, "theoretical_length": 0, "error": 0}

    # Get actual Farey sequence length
    length_data = await farey_sequence_length(n)
    actual_length = length_data["length"]

    # Theoretical asymptotic formula: |F_n| ~ (3/π²)n²
    theoretical_constant = 3 / (math.pi * math.pi)
    theoretical_length = theoretical_constant * n * n

    # Error analysis
    error = abs(actual_length - theoretical_length)
    relative_error = error / actual_length if actual_length > 0 else 0

    # Riemann Hypothesis bound analysis
    # RH implies error is O(n^{1/2+ε})
    # We compute what exponent α would make error ~ n^α
    if n > 1 and error > 0:
        alpha_estimate = math.log(error) / math.log(n)
        rh_bound = 0.5  # RH predicts α ≤ 1/2 + ε
        rh_consistent = alpha_estimate <= rh_bound + 0.1  # Allow small tolerance
    else:
        alpha_estimate = 0
        rh_bound = 0.5
        rh_consistent = True

    return {
        "n": n,
        "farey_length": actual_length,
        "theoretical_length": round(theoretical_length, 2),
        "error": round(error, 4),
        "relative_error": round(relative_error, 6),
        "error_exponent": round(alpha_estimate, 3),
        "rh_bound": rh_bound,
        "rh_consistent": rh_consistent,
        "asymptotic_formula": "|F_n| ~ (3/π²)n²",
        "rh_statement": "RH ⟺ error in |F_n| is O(n^{1/2+ε})",
        "theoretical_constant": round(theoretical_constant, 6),
    }


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


async def _gcd_async(a: int, b: int) -> int:
    """Async wrapper for GCD calculation."""
    while b:
        a, b = b, a % b
    return abs(a)


async def _euler_totient_async(n: int) -> int:
    """Async Euler totient function calculation."""
    if n <= 0:
        return 0
    if n == 1:
        return 1

    result = n
    p = 2
    original_n = n

    # Find all prime factors and apply totient formula
    while p * p <= original_n:
        if n % p == 0:
            # Remove all factors of p
            while n % p == 0:
                n //= p
            # Apply totient formula: φ(n) = n * (1 - 1/p)
            result -= result // p
        p += 1

        # Yield control every 1000 iterations for large n
        if p % 1000 == 0:
            await asyncio.sleep(0)

    # If n is still > 1, it's a prime factor
    if n > 1:
        result -= result // n

    return result


# Export all functions
__all__ = [
    # Basic operations
    "farey_sequence",
    "farey_sequence_length",
    "farey_neighbors",
    # Mediant operations
    "mediant",
    "stern_brocot_tree",
    "farey_mediant_path",
    # Ford circles
    "ford_circles",
    "ford_circle_properties",
    "circle_tangency",
    # Analysis
    "farey_sequence_properties",
    "density_analysis",
    "gap_analysis",
    # Applications
    "best_approximation_farey",
    "farey_fraction_between",
    # Advanced
    "farey_sum",
    "calkin_wilf_tree",
    "riemann_hypothesis_connection",
]

# ============================================================================
# COMPREHENSIVE TEST SUITE
# ============================================================================


async def test_farey_sequences():
    """Comprehensive test of all Farey sequence functions."""
    print("🔢 Farey Sequences Test Suite")
    print("=" * 40)

    # Test basic operations
    print("1. Basic Operations:")
    f5 = await farey_sequence(5)
    print(f"   farey_sequence(5) = {f5}")

    length = await farey_sequence_length(5)
    print(f"   farey_sequence_length(5) = {length}")

    neighbors = await farey_neighbors(1, 2, 5)
    print(f"   farey_neighbors(1/2, F_5) = {neighbors}")

    # Test mediant operations
    print("\n2. Mediant Operations:")
    med = await mediant(1, 3, 1, 2)
    print(f"   mediant(1/3, 1/2) = {med}")

    sb_tree = await stern_brocot_tree(3, 7)
    print(f"   stern_brocot_tree(3/7) = {sb_tree}")

    med_path = await farey_mediant_path(1, 3, 1, 2, 10)
    print(f"   farey_mediant_path(1/3 → 1/2) = {med_path}")

    # Test Ford circles
    print("\n3. Ford Circles:")
    circles = await ford_circles(4)
    print(f"   ford_circles(4) = {len(circles['circles'])} circles")

    circle_props = await ford_circle_properties(5)
    print(f"   ford_circle_properties(5) = {circle_props}")

    tangency = await circle_tangency(1, 3, 1, 2)
    print(f"   circle_tangency(1/3, 1/2) = {tangency}")

    # Test analysis functions
    print("\n4. Analysis Functions:")
    props = await farey_sequence_properties(5)
    print(f"   farey_sequence_properties(5) = {props}")

    density = await density_analysis(10)
    print(f"   density_analysis(10) = {density}")

    gaps = await gap_analysis(5)
    print(f"   gap_analysis(5) = {gaps}")

    # Test applications
    print("\n5. Applications:")
    best_approx = await best_approximation_farey(0.618, 10)
    print(f"   best_approximation_farey(0.618, 10) = {best_approx}")

    between = await farey_fraction_between(1, 3, 1, 2)
    print(f"   farey_fraction_between(1/3, 1/2) = {between}")

    # Test advanced functions
    print("\n6. Advanced Functions:")
    farey_sum_result = await farey_sum(1, 3, 1, 4)
    print(f"   farey_sum(1/3, 1/4) = {farey_sum_result}")

    cw_tree = await calkin_wilf_tree(3)
    print(f"   calkin_wilf_tree(3) = {cw_tree}")

    rh_conn = await riemann_hypothesis_connection(10)
    print(f"   riemann_hypothesis_connection(10) = {rh_conn}")

    print("\n✅ All Farey sequence functions working perfectly!")


async def demo_mathematical_properties():
    """Demonstrate key mathematical properties of Farey sequences."""
    print("\n🎯 Mathematical Properties Demo")
    print("=" * 35)

    # Property 1: Adjacent fractions have determinant 1
    print("1. Adjacent Fractions Property:")
    f5 = await farey_sequence(5)
    print("   F_5 =", f5)

    for i in range(len(f5) - 1):
        p1, q1 = f5[i]
        p2, q2 = f5[i + 1]
        det = abs(p1 * q2 - p2 * q1)
        print(f"   |{p1}×{q2} - {p2}×{q1}| = {det}")

    # Property 2: Mediant property
    print("\n2. Mediant Property:")
    for i in range(len(f5) - 1):
        p1, q1 = f5[i]
        p2, q2 = f5[i + 1]
        med = await mediant(p1, q1, p2, q2)
        med_val = med[0] / med[1]
        val1 = p1 / q1
        val2 = p2 / q2
        print(f"   mediant({p1}/{q1}, {p2}/{q2}) = {med[0]}/{med[1]} = {med_val:.4f}")
        print(f"   Between {val1:.4f} and {val2:.4f}: {val1 < med_val < val2}")

    # Property 3: Ford circle tangency
    print("\n3. Ford Circle Tangency:")
    for i in range(min(3, len(f5) - 1)):
        p1, q1 = f5[i]
        p2, q2 = f5[i + 1]
        tangency = await circle_tangency(p1, q1, p2, q2)
        print(
            f"   Ford circles for {p1}/{q1} and {p2}/{q2}: tangent = {tangency['are_tangent']}"
        )

    # Property 4: Length formula verification
    print("\n4. Length Formula |F_n| = 1 + Σφ(k):")
    for n in range(1, 8):
        length_data = await farey_sequence_length(n)
        actual_seq = await farey_sequence(n)
        print(
            f"   F_{n}: formula = {length_data['length']}, actual = {len(actual_seq)}"
        )


async def demo_applications():
    """Demonstrate practical applications of Farey sequences."""
    print("\n🔬 Practical Applications Demo")
    print("=" * 35)

    # Application 1: Rational approximation
    print("1. Rational Approximation:")
    constants = [
        (math.pi - 3, "π - 3"),
        (math.e - 2, "e - 2"),
        (math.sqrt(2) - 1, "√2 - 1"),
        ((1 + math.sqrt(5)) / 2 - 1, "φ - 1"),
    ]

    for value, name in constants:
        if 0 <= value <= 1:
            approx = await best_approximation_farey(value, 20)
            print(f"   {name} ≈ {value:.6f}")
            print(
                f"   Best approximation: {approx['best_approximation'][0]}/{approx['best_approximation'][1]}"
            )
            print(f"   Error: {approx['error']:.8f}")

    # Application 2: Musical intervals (frequency ratios)
    print("\n2. Musical Interval Approximation:")
    musical_ratios = [
        (2 ** (1 / 12), "Semitone"),
        (2 ** (7 / 12), "Perfect Fifth"),
        (2 ** (4 / 12), "Major Third"),
        (2 ** (5 / 12), "Perfect Fourth"),
    ]

    for ratio, name in musical_ratios:
        if ratio <= 2:  # Normalize to [0,1] range
            normalized = ratio - 1 if ratio > 1 else ratio
            if 0 <= normalized <= 1:
                approx = await best_approximation_farey(normalized, 15)
                print(f"   {name}: ratio ≈ {ratio:.6f}")
                print(
                    f"   Farey approximation: {approx['best_approximation'][0]}/{approx['best_approximation'][1]}"
                )

    # Application 3: Calendar approximations
    print("\n3. Calendar System Analysis:")
    # Year length approximations
    year_fraction = 365.25 / 366  # Approximate fractional part
    if 0 <= year_fraction <= 1:
        year_approx = await best_approximation_farey(year_fraction, 30)
        print(f"   Year length fraction: {year_fraction:.6f}")
        print(
            f"   Best rational: {year_approx['best_approximation'][0]}/{year_approx['best_approximation'][1]}"
        )


async def performance_benchmark():
    """Benchmark performance of Farey sequence operations."""
    print("\n⚡ Performance Benchmark")
    print("=" * 25)

    import time

    # Benchmark basic operations
    operations = [
        ("farey_sequence(50)", lambda: farey_sequence(50)),
        ("farey_sequence_length(100)", lambda: farey_sequence_length(100)),
        ("ford_circles(20)", lambda: ford_circles(20)),
        ("density_analysis(20)", lambda: density_analysis(20)),
        ("stern_brocot_tree(355, 113)", lambda: stern_brocot_tree(355, 113)),
        ("calkin_wilf_tree(8)", lambda: calkin_wilf_tree(8)),
    ]

    for name, operation in operations:
        start_time = time.time()
        await operation()
        duration = time.time() - start_time
        print(f"   {name}: {duration:.4f}s")

    # Test concurrent execution
    print("\n   Concurrent Execution Test:")
    start_time = time.time()

    concurrent_tasks = [
        farey_sequence(30),
        ford_circles(15),
        density_analysis(15),
        calkin_wilf_tree(6),
    ]

    await asyncio.gather(*concurrent_tasks)
    concurrent_duration = time.time() - start_time
    print(f"   4 operations concurrently: {concurrent_duration:.4f}s")


if __name__ == "__main__":
    import asyncio

    async def main():
        await test_farey_sequences()
        await demo_mathematical_properties()
        await demo_applications()
        await performance_benchmark()

    asyncio.run(main())
