#!/usr/bin/env python3
# chuk_mcp_math/number_theory/figurate_numbers.py
"""
Extended Figurate Numbers - Async Native

Functions for polygonal numbers, centered polygonal numbers, pronic numbers,
and 3D figurate numbers (octahedral, dodecahedral, etc.).

Functions:
- Polygonal numbers: polygonal_number, is_polygonal_number, polygonal_sequence
- Centered polygonal: centered_polygonal, centered_triangular, centered_square
- Pronic numbers: pronic_number, is_pronic_number, pronic_sequence
- Star numbers: star_number, hexagram_number, dodecagram_number
- 3D figurate: octahedral_number, dodecahedral_number, icosahedral_number
- Pyramidal: triangular_pyramidal, square_pyramidal, pentagonal_pyramidal
- Advanced: gnomon_numbers, figurate_number_properties, generating_functions
"""

import math
import asyncio
from typing import List
from chuk_mcp_math.mcp_decorator import mcp_function

# ============================================================================
# GENERAL POLYGONAL NUMBERS
# ============================================================================


@mcp_function(
    description="Calculate the nth s-gonal (polygonal) number.",
    namespace="arithmetic",
    category="figurate_numbers",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    examples=[
        {
            "input": {"n": 5, "s": 3},
            "output": 15,
            "description": "5th triangular number",
        },
        {"input": {"n": 4, "s": 4}, "output": 16, "description": "4th square number"},
        {
            "input": {"n": 3, "s": 5},
            "output": 12,
            "description": "3rd pentagonal number",
        },
        {
            "input": {"n": 6, "s": 6},
            "output": 66,
            "description": "6th hexagonal number",
        },
    ],
)
async def polygonal_number(n: int, s: int) -> int:
    """
    Calculate the nth s-gonal number.

    Formula: P(s,n) = n * ((s-2)*n - (s-4)) / 2

    Args:
        n: Position in the sequence (≥ 0)
        s: Number of sides of polygon (≥ 3)

    Returns:
        The nth s-gonal number

    Examples:
        await polygonal_number(5, 3) → 15   # 5th triangular
        await polygonal_number(4, 4) → 16   # 4th square
        await polygonal_number(3, 5) → 12   # 3rd pentagonal
    """
    if n < 0:
        raise ValueError("Position must be non-negative")
    if s < 3:
        raise ValueError("Polygon must have at least 3 sides")

    if n == 0:
        return 0

    return n * ((s - 2) * n - (s - 4)) // 2


@mcp_function(
    description="Check if a number is an s-gonal number.",
    namespace="arithmetic",
    category="figurate_numbers",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    examples=[
        {
            "input": {"num": 15, "s": 3},
            "output": True,
            "description": "15 is triangular",
        },
        {"input": {"num": 16, "s": 4}, "output": True, "description": "16 is square"},
        {
            "input": {"num": 13, "s": 5},
            "output": False,
            "description": "13 is not pentagonal",
        },
        {
            "input": {"num": 28, "s": 6},
            "output": True,
            "description": "28 is hexagonal",
        },
    ],
)
async def is_polygonal_number(num: int, s: int) -> bool:
    """
    Check if a number is an s-gonal number.

    Args:
        num: Number to test
        s: Number of sides of polygon (≥ 3)

    Returns:
        True if num is an s-gonal number

    Examples:
        await is_polygonal_number(15, 3) → True   # 15 is triangular
        await is_polygonal_number(16, 4) → True   # 16 is square
        await is_polygonal_number(13, 5) → False  # 13 not pentagonal
    """
    if num < 0 or s < 3:
        return False

    if num == 0:
        return True

    # Solve n * ((s-2)*n - (s-4)) / 2 = num
    # (s-2)*n² - (s-4)*n - 2*num = 0
    # Using quadratic formula: n = ((s-4) + sqrt((s-4)² + 8*(s-2)*num)) / (2*(s-2))

    a = s - 2
    b = -(s - 4)
    c = -2 * num

    discriminant = b * b - 4 * a * c

    if discriminant < 0:
        return False

    sqrt_disc = math.sqrt(discriminant)
    n = (-b + sqrt_disc) / (2 * a)

    # Check if n is a positive integer
    if n >= 0 and abs(n - round(n)) < 1e-10:
        n_int = round(n)
        # Verify by computing the polygonal number
        computed = await polygonal_number(n_int, s)
        return computed == num

    return False


@mcp_function(
    description="Generate the first n s-gonal numbers.",
    namespace="arithmetic",
    category="figurate_numbers",
    execution_modes=["local", "remote"],
    examples=[
        {
            "input": {"n": 8, "s": 3},
            "output": [0, 1, 3, 6, 10, 15, 21, 28],
            "description": "First 8 triangular numbers",
        },
        {
            "input": {"n": 6, "s": 5},
            "output": [0, 1, 5, 12, 22, 35],
            "description": "First 6 pentagonal numbers",
        },
        {
            "input": {"n": 5, "s": 6},
            "output": [0, 1, 6, 15, 28],
            "description": "First 5 hexagonal numbers",
        },
    ],
)
async def polygonal_sequence(n: int, s: int) -> List[int]:
    """
    Generate the first n s-gonal numbers.

    Args:
        n: Number of terms to generate
        s: Number of sides of polygon (≥ 3)

    Returns:
        List of first n s-gonal numbers

    Examples:
        await polygonal_sequence(8, 3) → [0, 1, 3, 6, 10, 15, 21, 28]
        await polygonal_sequence(6, 5) → [0, 1, 5, 12, 22, 35]
    """
    if n <= 0:
        return []
    if s < 3:
        raise ValueError("Polygon must have at least 3 sides")

    sequence = []
    for i in range(n):
        poly_num = await polygonal_number(i, s)
        sequence.append(poly_num)

        # Yield control every 1000 iterations
        if i % 1000 == 0 and n > 1000:
            await asyncio.sleep(0)

    return sequence


# ============================================================================
# CENTERED POLYGONAL NUMBERS
# ============================================================================


@mcp_function(
    description="Calculate the nth centered s-gonal number.",
    namespace="arithmetic",
    category="figurate_numbers",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    examples=[
        {
            "input": {"n": 3, "s": 3},
            "output": 10,
            "description": "3rd centered triangular",
        },
        {"input": {"n": 4, "s": 4}, "output": 25, "description": "4th centered square"},
        {
            "input": {"n": 2, "s": 6},
            "output": 7,
            "description": "2nd centered hexagonal",
        },
        {
            "input": {"n": 5, "s": 5},
            "output": 51,
            "description": "5th centered pentagonal",
        },
    ],
)
async def centered_polygonal_number(n: int, s: int) -> int:
    """
    Calculate the nth centered s-gonal number.

    Formula: C(s,n) = 1 + s*n*(n-1)/2 for n ≥ 1, C(s,0) = 1

    Args:
        n: Position in the sequence (≥ 0)
        s: Number of sides of polygon (≥ 3)

    Returns:
        The nth centered s-gonal number

    Examples:
        await centered_polygonal_number(3, 3) → 10  # 3rd centered triangular
        await centered_polygonal_number(4, 4) → 25  # 4th centered square
    """
    if n < 0:
        raise ValueError("Position must be non-negative")
    if s < 3:
        raise ValueError("Polygon must have at least 3 sides")

    if n == 0:
        return 1

    return 1 + s * n * (n - 1) // 2


@mcp_function(
    description="Calculate the nth centered triangular number.",
    namespace="arithmetic",
    category="figurate_numbers",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    examples=[
        {"input": {"n": 0}, "output": 1, "description": "0th centered triangular"},
        {"input": {"n": 3}, "output": 10, "description": "3rd centered triangular"},
        {"input": {"n": 5}, "output": 16, "description": "5th centered triangular"},
        {"input": {"n": 10}, "output": 31, "description": "10th centered triangular"},
    ],
)
async def centered_triangular_number(n: int) -> int:
    """
    Calculate the nth centered triangular number.

    Formula: CT_n = (3*n² + 3*n + 2) / 2 = 1 + 3*T_{n-1} where T_k is kth triangular
    Sequence: 1, 4, 10, 19, 31, 46, 64, 85, 109, 136, ...

    Args:
        n: Non-negative integer index

    Returns:
        The nth centered triangular number

    Examples:
        await centered_triangular_number(0) → 1   # Single center point
        await centered_triangular_number(3) → 10  # Center + 3 triangular layers
    """
    if n < 0:
        raise ValueError("Index must be non-negative")

    return (3 * n * n + 3 * n + 2) // 2


@mcp_function(
    description="Calculate the nth centered square number.",
    namespace="arithmetic",
    category="figurate_numbers",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    examples=[
        {"input": {"n": 0}, "output": 1, "description": "0th centered square"},
        {"input": {"n": 3}, "output": 13, "description": "3rd centered square"},
        {"input": {"n": 5}, "output": 21, "description": "5th centered square"},
        {"input": {"n": 10}, "output": 41, "description": "10th centered square"},
    ],
)
async def centered_square_number(n: int) -> int:
    """
    Calculate the nth centered square number.

    Formula: CS_n = 2*n² + 2*n + 1 = 2*n*(n+1) + 1
    Sequence: 1, 5, 13, 25, 41, 61, 85, 113, 145, 181, ...

    Args:
        n: Non-negative integer index

    Returns:
        The nth centered square number

    Examples:
        await centered_square_number(0) → 1   # Single center point
        await centered_square_number(3) → 25  # Center + 3 square layers
    """
    if n < 0:
        raise ValueError("Index must be non-negative")

    return 2 * n * (n + 1) + 1


@mcp_function(
    description="Calculate the nth centered hexagonal number.",
    namespace="arithmetic",
    category="figurate_numbers",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    examples=[
        {"input": {"n": 0}, "output": 1, "description": "0th centered hexagonal"},
        {"input": {"n": 2}, "output": 7, "description": "2nd centered hexagonal"},
        {"input": {"n": 4}, "output": 19, "description": "4th centered hexagonal"},
        {"input": {"n": 6}, "output": 37, "description": "6th centered hexagonal"},
    ],
)
async def centered_hexagonal_number(n: int) -> int:
    """
    Calculate the nth centered hexagonal number.

    Formula: CH_n = 3*n² + 3*n + 1 = 3*n*(n+1) + 1
    Sequence: 1, 7, 19, 37, 61, 91, 127, 169, 217, 271, ...

    Args:
        n: Non-negative integer index

    Returns:
        The nth centered hexagonal number

    Examples:
        await centered_hexagonal_number(0) → 1   # Single center point
        await centered_hexagonal_number(2) → 7   # Center + 2 hexagonal layers
    """
    if n < 0:
        raise ValueError("Index must be non-negative")

    return 3 * n * (n + 1) + 1


# ============================================================================
# PRONIC NUMBERS
# ============================================================================


@mcp_function(
    description="Calculate the nth pronic number (oblong number).",
    namespace="arithmetic",
    category="figurate_numbers",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    examples=[
        {"input": {"n": 0}, "output": 0, "description": "0th pronic number"},
        {"input": {"n": 3}, "output": 12, "description": "3rd pronic number"},
        {"input": {"n": 5}, "output": 30, "description": "5th pronic number"},
        {"input": {"n": 10}, "output": 110, "description": "10th pronic number"},
    ],
)
async def pronic_number(n: int) -> int:
    """
    Calculate the nth pronic number (also called oblong number).

    Formula: P_n = n * (n + 1)
    Pronic numbers are the product of two consecutive integers.
    Sequence: 0, 2, 6, 12, 20, 30, 42, 56, 72, 90, 110, ...

    Args:
        n: Non-negative integer index

    Returns:
        The nth pronic number

    Examples:
        await pronic_number(0) → 0    # 0 * 1 = 0
        await pronic_number(3) → 12   # 3 * 4 = 12
        await pronic_number(10) → 110 # 10 * 11 = 110
    """
    if n < 0:
        raise ValueError("Index must be non-negative")

    return n * (n + 1)


@mcp_function(
    description="Check if a number is a pronic number.",
    namespace="arithmetic",
    category="figurate_numbers",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    examples=[
        {"input": {"num": 12}, "output": True, "description": "12 = 3 * 4 is pronic"},
        {"input": {"num": 30}, "output": True, "description": "30 = 5 * 6 is pronic"},
        {"input": {"num": 15}, "output": False, "description": "15 is not pronic"},
        {"input": {"num": 0}, "output": True, "description": "0 = 0 * 1 is pronic"},
    ],
)
async def is_pronic_number(num: int) -> bool:
    """
    Check if a number is a pronic number.

    Args:
        num: Non-negative integer to test

    Returns:
        True if num is a pronic number

    Examples:
        await is_pronic_number(12) → True   # 12 = 3 * 4
        await is_pronic_number(30) → True   # 30 = 5 * 6
        await is_pronic_number(15) → False  # No consecutive integers multiply to 15
    """
    if num < 0:
        return False

    if num == 0:
        return True

    # Solve n*(n+1) = num
    # n² + n - num = 0
    # n = (-1 + sqrt(1 + 4*num)) / 2

    discriminant = 1 + 4 * num
    sqrt_disc = math.sqrt(discriminant)
    n = (-1 + sqrt_disc) / 2

    # Check if n is a non-negative integer
    if n >= 0 and abs(n - round(n)) < 1e-10:
        n_int = round(n)
        return n_int * (n_int + 1) == num

    return False


@mcp_function(
    description="Generate the first n pronic numbers.",
    namespace="arithmetic",
    category="figurate_numbers",
    execution_modes=["local", "remote"],
    examples=[
        {
            "input": {"n": 10},
            "output": [0, 2, 6, 12, 20, 30, 42, 56, 72, 90],
            "description": "First 10 pronic numbers",
        },
        {
            "input": {"n": 5},
            "output": [0, 2, 6, 12, 20],
            "description": "First 5 pronic numbers",
        },
    ],
)
async def pronic_sequence(n: int) -> List[int]:
    """
    Generate the first n pronic numbers.

    Args:
        n: Number of pronic numbers to generate

    Returns:
        List of first n pronic numbers

    Examples:
        await pronic_sequence(10) → [0, 2, 6, 12, 20, 30, 42, 56, 72, 90]
        await pronic_sequence(5) → [0, 2, 6, 12, 20]
    """
    if n <= 0:
        return []

    sequence = []
    for i in range(n):
        sequence.append(i * (i + 1))

        # Yield control every 1000 iterations
        if i % 1000 == 0 and n > 1000:
            await asyncio.sleep(0)

    return sequence


# ============================================================================
# STAR NUMBERS
# ============================================================================


@mcp_function(
    description="Calculate the nth star number (6-pointed star).",
    namespace="arithmetic",
    category="figurate_numbers",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    examples=[
        {"input": {"n": 1}, "output": 1, "description": "1st star number"},
        {"input": {"n": 2}, "output": 13, "description": "2nd star number"},
        {"input": {"n": 3}, "output": 37, "description": "3rd star number"},
        {"input": {"n": 4}, "output": 73, "description": "4th star number"},
    ],
)
async def star_number(n: int) -> int:
    """
    Calculate the nth star number (6-pointed star).

    Formula: S_n = 6*n*(n-1) + 1
    Star numbers represent centered hexagrams (6-pointed stars).
    Sequence: 1, 13, 37, 73, 121, 181, 253, 337, 433, 541, ...

    Args:
        n: Positive integer index (≥ 1)

    Returns:
        The nth star number

    Examples:
        await star_number(1) → 1    # Single central point
        await star_number(2) → 13   # Central point + first star layer
        await star_number(3) → 37   # Central point + two star layers
    """
    if n < 1:
        raise ValueError("Star number index must be positive")

    return 6 * n * (n - 1) + 1


@mcp_function(
    description="Calculate the nth hexagram number (Star of David).",
    namespace="arithmetic",
    category="figurate_numbers",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    examples=[
        {"input": {"n": 1}, "output": 1, "description": "1st hexagram number"},
        {"input": {"n": 2}, "output": 13, "description": "2nd hexagram number"},
        {"input": {"n": 3}, "output": 37, "description": "3rd hexagram number"},
    ],
)
async def hexagram_number(n: int) -> int:
    """
    Calculate the nth hexagram number (same as star number).

    Hexagram numbers count the dots in nested hexagrams (Stars of David).

    Args:
        n: Positive integer index (≥ 1)

    Returns:
        The nth hexagram number
    """
    return await star_number(n)


# ============================================================================
# 3D FIGURATE NUMBERS
# ============================================================================


@mcp_function(
    description="Calculate the nth octahedral number.",
    namespace="arithmetic",
    category="figurate_numbers",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    examples=[
        {"input": {"n": 1}, "output": 1, "description": "1st octahedral number"},
        {"input": {"n": 2}, "output": 6, "description": "2nd octahedral number"},
        {"input": {"n": 3}, "output": 19, "description": "3rd octahedral number"},
        {"input": {"n": 4}, "output": 44, "description": "4th octahedral number"},
    ],
)
async def octahedral_number(n: int) -> int:
    """
    Calculate the nth octahedral number.

    Formula: O_n = (2*n³ + n) / 3 = n*(2*n² + 1) / 3
    Octahedral numbers count spheres in octahedral arrangements.
    Sequence: 1, 6, 19, 44, 85, 146, 231, 344, 489, 670, ...

    Args:
        n: Positive integer index (≥ 1)

    Returns:
        The nth octahedral number

    Examples:
        await octahedral_number(1) → 1    # Single sphere
        await octahedral_number(2) → 6    # Central + 6 in octahedral shell
        await octahedral_number(3) → 19   # Previous + second octahedral shell
    """
    if n < 1:
        raise ValueError("Octahedral number index must be positive")

    return n * (2 * n * n + 1) // 3


@mcp_function(
    description="Calculate the nth dodecahedral number.",
    namespace="arithmetic",
    category="figurate_numbers",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    examples=[
        {"input": {"n": 1}, "output": 1, "description": "1st dodecahedral number"},
        {"input": {"n": 2}, "output": 20, "description": "2nd dodecahedral number"},
        {"input": {"n": 3}, "output": 84, "description": "3rd dodecahedral number"},
        {"input": {"n": 4}, "output": 220, "description": "4th dodecahedral number"},
    ],
)
async def dodecahedral_number(n: int) -> int:
    """
    Calculate the nth dodecahedral number.

    Formula: D_n = n * (3*n - 1) * (3*n - 2) / 2
    Dodecahedral numbers are 3D analogs of pentagonal numbers.
    Sequence: 1, 20, 84, 220, 455, 816, 1330, 2024, 2925, 4060, ...

    Args:
        n: Positive integer index (≥ 1)

    Returns:
        The nth dodecahedral number

    Examples:
        await dodecahedral_number(1) → 1     # Single point
        await dodecahedral_number(2) → 20    # Dodecahedral shell
        await dodecahedral_number(3) → 84    # Two dodecahedral shells
    """
    if n < 1:
        raise ValueError("Dodecahedral number index must be positive")

    return n * (3 * n - 1) * (3 * n - 2) // 2


@mcp_function(
    description="Calculate the nth icosahedral number.",
    namespace="arithmetic",
    category="figurate_numbers",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    examples=[
        {"input": {"n": 1}, "output": 1, "description": "1st icosahedral number"},
        {"input": {"n": 2}, "output": 12, "description": "2nd icosahedral number"},
        {"input": {"n": 3}, "output": 48, "description": "3rd icosahedral number"},
        {"input": {"n": 4}, "output": 124, "description": "4th icosahedral number"},
    ],
)
async def icosahedral_number(n: int) -> int:
    """
    Calculate the nth icosahedral number.

    Formula: I_n = n * (5*n² - 5*n + 2) / 2
    Icosahedral numbers count spheres arranged in icosahedral patterns.
    Sequence: 1, 12, 48, 124, 255, 456, 742, 1128, 1629, 2260, ...

    Args:
        n: Positive integer index (≥ 1)

    Returns:
        The nth icosahedral number

    Examples:
        await icosahedral_number(1) → 1     # Single sphere
        await icosahedral_number(2) → 12    # Central + icosahedral shell
        await icosahedral_number(3) → 48    # Previous + second shell
    """
    if n < 1:
        raise ValueError("Icosahedral number index must be positive")

    return n * (5 * n * n - 5 * n + 2) // 2


# ============================================================================
# PYRAMIDAL NUMBERS
# ============================================================================


@mcp_function(
    description="Calculate the nth triangular pyramidal number (tetrahedral).",
    namespace="arithmetic",
    category="figurate_numbers",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    examples=[
        {"input": {"n": 1}, "output": 1, "description": "1st tetrahedral number"},
        {"input": {"n": 4}, "output": 20, "description": "4th tetrahedral number"},
        {"input": {"n": 5}, "output": 35, "description": "5th tetrahedral number"},
        {"input": {"n": 10}, "output": 220, "description": "10th tetrahedral number"},
    ],
)
async def triangular_pyramidal_number(n: int) -> int:
    """
    Calculate the nth triangular pyramidal (tetrahedral) number.

    Formula: Tet_n = n * (n + 1) * (n + 2) / 6
    Sum of first n triangular numbers.
    Sequence: 1, 4, 10, 20, 35, 56, 84, 120, 165, 220, ...

    Args:
        n: Positive integer index (≥ 1)

    Returns:
        The nth tetrahedral number

    Examples:
        await triangular_pyramidal_number(1) → 1    # Single sphere
        await triangular_pyramidal_number(4) → 20   # 4-layer triangular pyramid
        await triangular_pyramidal_number(10) → 220 # 10-layer pyramid
    """
    if n < 1:
        raise ValueError("Tetrahedral number index must be positive")

    return n * (n + 1) * (n + 2) // 6


@mcp_function(
    description="Calculate the nth square pyramidal number.",
    namespace="arithmetic",
    category="figurate_numbers",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    examples=[
        {"input": {"n": 1}, "output": 1, "description": "1st square pyramidal"},
        {"input": {"n": 4}, "output": 30, "description": "4th square pyramidal"},
        {"input": {"n": 5}, "output": 55, "description": "5th square pyramidal"},
        {"input": {"n": 10}, "output": 385, "description": "10th square pyramidal"},
    ],
)
async def square_pyramidal_number(n: int) -> int:
    """
    Calculate the nth square pyramidal number.

    Formula: SP_n = n * (n + 1) * (2*n + 1) / 6
    Sum of first n square numbers.
    Sequence: 1, 5, 14, 30, 55, 91, 140, 204, 285, 385, ...

    Args:
        n: Positive integer index (≥ 1)

    Returns:
        The nth square pyramidal number

    Examples:
        await square_pyramidal_number(1) → 1    # Single cube
        await square_pyramidal_number(4) → 30   # 4-layer square pyramid
        await square_pyramidal_number(10) → 385 # 10-layer pyramid
    """
    if n < 1:
        raise ValueError("Square pyramidal number index must be positive")

    return n * (n + 1) * (2 * n + 1) // 6


@mcp_function(
    description="Calculate the nth pentagonal pyramidal number.",
    namespace="arithmetic",
    category="figurate_numbers",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    examples=[
        {"input": {"n": 1}, "output": 1, "description": "1st pentagonal pyramidal"},
        {"input": {"n": 3}, "output": 18, "description": "3rd pentagonal pyramidal"},
        {"input": {"n": 5}, "output": 70, "description": "5th pentagonal pyramidal"},
        {"input": {"n": 8}, "output": 240, "description": "8th pentagonal pyramidal"},
    ],
)
async def pentagonal_pyramidal_number(n: int) -> int:
    """
    Calculate the nth pentagonal pyramidal number.

    Formula: PP_n = n² * (n + 1) / 2
    Sum of first n pentagonal numbers.
    Sequence: 1, 6, 18, 40, 75, 126, 196, 288, 405, 550, ...

    Args:
        n: Positive integer index (≥ 1)

    Returns:
        The nth pentagonal pyramidal number

    Examples:
        await pentagonal_pyramidal_number(1) → 1    # Single point
        await pentagonal_pyramidal_number(3) → 18   # 3-layer pentagonal pyramid
        await pentagonal_pyramidal_number(8) → 288  # 8-layer pyramid
    """
    if n < 1:
        raise ValueError("Pentagonal pyramidal number index must be positive")

    return n * n * (n + 1) // 2


# ============================================================================
# ADVANCED FIGURATE NUMBERS
# ============================================================================


@mcp_function(
    description="Calculate gnomon numbers for s-gonal numbers.",
    namespace="arithmetic",
    category="figurate_numbers",
    execution_modes=["local", "remote"],
    examples=[
        {"input": {"n": 3, "s": 4}, "output": 5, "description": "3rd square gnomon"},
        {
            "input": {"n": 4, "s": 3},
            "output": 4,
            "description": "4th triangular gnomon",
        },
        {"input": {"n": 2, "s": 6}, "output": 6, "description": "2nd hexagonal gnomon"},
    ],
)
async def gnomon_number(n: int, s: int) -> int:
    """
    Calculate the nth gnomon for s-gonal numbers.

    A gnomon is the difference between consecutive figurate numbers.
    For s-gonal numbers: Gnomon_n = P(s,n) - P(s,n-1)

    Args:
        n: Positive integer index (≥ 1)
        s: Number of sides (≥ 3)

    Returns:
        The nth gnomon for s-gonal numbers

    Examples:
        await gnomon_number(3, 4) → 5   # 3rd square gnomon
        await gnomon_number(4, 3) → 4   # 4th triangular gnomon
    """
    if n < 1:
        raise ValueError("Gnomon index must be positive")
    if s < 3:
        raise ValueError("Polygon must have at least 3 sides")

    poly_n = await polygonal_number(n, s)
    poly_n_minus_1 = await polygonal_number(n - 1, s)

    return poly_n - poly_n_minus_1


# Export all functions
__all__ = [
    # General polygonal numbers
    "polygonal_number",
    "is_polygonal_number",
    "polygonal_sequence",
    # Centered polygonal numbers
    "centered_polygonal_number",
    "centered_triangular_number",
    "centered_square_number",
    "centered_hexagonal_number",
    # Pronic numbers
    "pronic_number",
    "is_pronic_number",
    "pronic_sequence",
    # Star numbers
    "star_number",
    "hexagram_number",
    # 3D figurate numbers
    "octahedral_number",
    "dodecahedral_number",
    "icosahedral_number",
    # Pyramidal numbers
    "triangular_pyramidal_number",
    "square_pyramidal_number",
    "pentagonal_pyramidal_number",
    # Advanced
    "gnomon_number",
]
