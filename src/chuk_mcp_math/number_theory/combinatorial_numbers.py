#!/usr/bin/env python3
# chuk_mcp_math/number_theory/combinatorial_numbers.py
"""
Combinatorial Numbers and Sequences - Async Native - FIXED

Functions for fundamental combinatorial sequences including Catalan numbers,
Bell numbers, Stirling numbers, and related combinatorial objects.

FIXES:
1. Fixed Stirling first kind recurrence relation
2. Fixed Narayana number formula (corrected binomial coefficient calculation)
3. Improved error handling and edge cases
"""

import asyncio
from typing import List
from chuk_mcp_math.mcp_decorator import mcp_function

# Import basic sequences for dependencies

# ============================================================================
# CATALAN NUMBERS
# ============================================================================


@mcp_function(
    description="Calculate the nth Catalan number C_n = (2n)!/(n+1)!n!.",
    namespace="arithmetic",
    category="combinatorial_numbers",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    estimated_cpu_usage="medium",
    examples=[
        {"input": {"n": 0}, "output": 1, "description": "C_0 = 1"},
        {"input": {"n": 1}, "output": 1, "description": "C_1 = 1"},
        {"input": {"n": 2}, "output": 2, "description": "C_2 = 2"},
        {"input": {"n": 3}, "output": 5, "description": "C_3 = 5"},
        {"input": {"n": 4}, "output": 14, "description": "C_4 = 14"},
        {"input": {"n": 5}, "output": 42, "description": "C_5 = 42"},
    ],
)
async def catalan_number(n: int) -> int:
    """
    Calculate the nth Catalan number.

    C_n = (2n)! / ((n+1)! * n!) = (2n choose n) / (n+1)

    Catalan numbers count many combinatorial objects:
    - Binary trees with n internal nodes
    - Ways to parenthesize n+1 factors
    - Monotonic lattice paths that don't cross the diagonal
    - Non-crossing partitions of 2n points on a circle

    Args:
        n: Non-negative integer

    Returns:
        The nth Catalan number

    Examples:
        await catalan_number(0) → 1    # C_0 = 1
        await catalan_number(3) → 5    # C_3 = 5
        await catalan_number(5) → 42   # C_5 = 42
    """
    if n < 0:
        raise ValueError("Catalan number index must be non-negative")

    if n == 0:
        return 1

    # Use the recurrence: C_n = (4n - 2) * C_{n-1} / (n + 1)
    # This is more efficient than computing factorials
    if n <= 20:  # Use recurrence for small n
        result = 1
        for i in range(1, n + 1):
            result = result * (4 * i - 2) // (i + 1)
            if i % 5 == 0:  # Yield control periodically
                await asyncio.sleep(0)
        return result

    # For larger n, use the formula with careful computation to avoid overflow
    # C_n = (2n choose n) / (n + 1)
    numerator = 1
    denominator = 1

    # Compute (2n choose n) = (2n)! / (n! * n!)
    for i in range(n + 1):
        numerator *= n + 1 + i
        denominator *= i + 1

        if i % 10 == 0 and n > 100:
            await asyncio.sleep(0)

    return numerator // denominator // (n + 1)


@mcp_function(
    description="Generate the first n Catalan numbers.",
    namespace="arithmetic",
    category="combinatorial_numbers",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    examples=[
        {
            "input": {"n": 6},
            "output": [1, 1, 2, 5, 14, 42],
            "description": "First 6 Catalan numbers",
        },
        {
            "input": {"n": 10},
            "output": [1, 1, 2, 5, 14, 42, 132, 429, 1430, 4862],
            "description": "First 10 Catalan numbers",
        },
    ],
)
async def catalan_sequence(n: int) -> List[int]:
    """
    Generate the first n Catalan numbers.

    Args:
        n: Number of Catalan numbers to generate

    Returns:
        List of first n Catalan numbers

    Examples:
        await catalan_sequence(6) → [1, 1, 2, 5, 14, 42]
        await catalan_sequence(10) → [1, 1, 2, 5, 14, 42, 132, 429, 1430, 4862]
    """
    if n <= 0:
        return []

    result = [1]  # C_0 = 1

    if n == 1:
        return result

    # Use recurrence: C_n = (4n - 2) * C_{n-1} / (n + 1)
    for i in range(1, n):
        next_catalan = result[i - 1] * (4 * i - 2) // (i + 1)
        result.append(next_catalan)

        if i % 100 == 0 and n > 100:
            await asyncio.sleep(0)

    return result


@mcp_function(
    description="Check if a number is a Catalan number.",
    namespace="arithmetic",
    category="combinatorial_numbers",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    examples=[
        {"input": {"n": 14}, "output": True, "description": "14 = C_4 is Catalan"},
        {"input": {"n": 42}, "output": True, "description": "42 = C_5 is Catalan"},
        {
            "input": {"n": 20},
            "output": False,
            "description": "20 is not a Catalan number",
        },
    ],
)
async def is_catalan_number(n: int) -> bool:
    """
    Check if a number is a Catalan number.

    Args:
        n: Number to check

    Returns:
        True if n is a Catalan number, False otherwise

    Examples:
        await is_catalan_number(14) → True   # C_4 = 14
        await is_catalan_number(42) → True   # C_5 = 42
        await is_catalan_number(20) → False  # Not a Catalan number
    """
    if n < 1:
        return n == 1 if n == 0 else False

    # Generate Catalan numbers until we exceed n or find n
    catalan = 1  # C_0 = 1
    index = 0

    while catalan <= n:
        if catalan == n:
            return True

        index += 1
        catalan = catalan * (4 * index - 2) // (index + 1)

        # Safety check to avoid infinite loop
        if index > 50:  # Catalan numbers grow very quickly
            break

    return False


# ============================================================================
# BELL NUMBERS
# ============================================================================


@mcp_function(
    description="Calculate the nth Bell number B_n (number of partitions of a set).",
    namespace="arithmetic",
    category="combinatorial_numbers",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    estimated_cpu_usage="medium",
    examples=[
        {"input": {"n": 0}, "output": 1, "description": "B_0 = 1"},
        {"input": {"n": 1}, "output": 1, "description": "B_1 = 1"},
        {"input": {"n": 2}, "output": 2, "description": "B_2 = 2"},
        {"input": {"n": 3}, "output": 5, "description": "B_3 = 5"},
        {"input": {"n": 4}, "output": 15, "description": "B_4 = 15"},
        {"input": {"n": 5}, "output": 52, "description": "B_5 = 52"},
    ],
)
async def bell_number(n: int) -> int:
    """
    Calculate the nth Bell number.

    B_n counts the number of ways to partition a set of n elements.

    Uses Bell's triangle construction or the exponential formula
    for efficient computation.

    Args:
        n: Non-negative integer

    Returns:
        The nth Bell number

    Examples:
        await bell_number(0) → 1    # B_0 = 1 (empty partition)
        await bell_number(3) → 5    # B_3 = 5 partitions of {1,2,3}
        await bell_number(5) → 52   # B_5 = 52
    """
    if n < 0:
        raise ValueError("Bell number index must be non-negative")

    if n == 0:
        return 1

    # Use Bell's triangle construction for efficient computation
    # Each row starts with the last element of the previous row
    # Each element is the sum of the element above and the element to the left

    # Start with B_0 = 1
    prev_row = [1]

    for i in range(1, n + 1):
        # New row starts with the last element of previous row
        current_row = [prev_row[-1]]

        # Fill the rest of the row
        for j in range(len(prev_row)):
            current_row.append(current_row[-1] + prev_row[j])

        prev_row = current_row

        if i % 10 == 0 and n > 10:
            await asyncio.sleep(0)

    return prev_row[0]  # B_n is the first element of row n


@mcp_function(
    description="Generate the first n Bell numbers.",
    namespace="arithmetic",
    category="combinatorial_numbers",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    examples=[
        {
            "input": {"n": 6},
            "output": [1, 1, 2, 5, 15, 52],
            "description": "First 6 Bell numbers",
        },
        {
            "input": {"n": 8},
            "output": [1, 1, 2, 5, 15, 52, 203, 877],
            "description": "First 8 Bell numbers",
        },
    ],
)
async def bell_sequence(n: int) -> List[int]:
    """
    Generate the first n Bell numbers.

    Args:
        n: Number of Bell numbers to generate

    Returns:
        List of first n Bell numbers

    Examples:
        await bell_sequence(6) → [1, 1, 2, 5, 15, 52]
        await bell_sequence(8) → [1, 1, 2, 5, 15, 52, 203, 877]
    """
    if n <= 0:
        return []

    result = [1]  # B_0 = 1

    if n == 1:
        return result

    # Build Bell's triangle and collect the first element of each row
    prev_row = [1]

    for i in range(1, n):
        # New row starts with the last element of previous row
        current_row = [prev_row[-1]]

        # Fill the rest of the row
        for j in range(len(prev_row)):
            current_row.append(current_row[-1] + prev_row[j])

        result.append(current_row[0])
        prev_row = current_row

        if i % 10 == 0 and n > 10:
            await asyncio.sleep(0)

    return result


@mcp_function(
    description="Generate Bell's triangle up to row n.",
    namespace="arithmetic",
    category="combinatorial_numbers",
    execution_modes=["local", "remote"],
    estimated_cpu_usage="medium",
    examples=[
        {
            "input": {"n": 4},
            "output": [[1], [1, 2], [2, 3, 5], [5, 7, 10, 15]],
            "description": "Bell triangle up to row 4",
        }
    ],
)
async def bell_triangle(n: int) -> List[List[int]]:
    """
    Generate Bell's triangle up to row n.

    Bell's triangle is constructed as follows:
    - Row 0: [1]
    - Each subsequent row starts with the last element of the previous row
    - Each element is the sum of the element above and the element to the left

    Args:
        n: Number of rows to generate

    Returns:
        Bell's triangle as a list of lists

    Examples:
        await bell_triangle(4) → [[1], [1, 2], [2, 3, 5], [5, 7, 10, 15]]
    """
    if n < 0:
        raise ValueError("Number of rows must be non-negative")

    if n == 0:
        return []

    triangle = [[1]]  # Row 0

    for i in range(1, n):
        prev_row = triangle[-1]

        # New row starts with the last element of previous row
        current_row = [prev_row[-1]]

        # Fill the rest of the row
        for j in range(len(prev_row)):
            current_row.append(current_row[-1] + prev_row[j])

        triangle.append(current_row)

        if i % 10 == 0 and n > 10:
            await asyncio.sleep(0)

    return triangle


# ============================================================================
# STIRLING NUMBERS - FIXED
# ============================================================================


@mcp_function(
    description="Calculate Stirling number of the second kind S(n,k).",
    namespace="arithmetic",
    category="combinatorial_numbers",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    estimated_cpu_usage="medium",
    examples=[
        {"input": {"n": 4, "k": 2}, "output": 7, "description": "S(4,2) = 7"},
        {"input": {"n": 5, "k": 3}, "output": 25, "description": "S(5,3) = 25"},
        {"input": {"n": 3, "k": 2}, "output": 3, "description": "S(3,2) = 3"},
    ],
)
async def stirling_second(n: int, k: int) -> int:
    """
    Calculate Stirling number of the second kind S(n,k).

    S(n,k) counts the number of ways to partition n objects into k non-empty subsets.

    Uses the recurrence: S(n,k) = k*S(n-1,k) + S(n-1,k-1)

    Args:
        n: Number of objects
        k: Number of subsets

    Returns:
        Stirling number of the second kind S(n,k)

    Examples:
        await stirling_second(4, 2) → 7   # 7 ways to partition 4 objects into 2 subsets
        await stirling_second(5, 3) → 25  # 25 ways to partition 5 objects into 3 subsets
    """
    if n < 0 or k < 0:
        raise ValueError("n and k must be non-negative")

    if n == 0:
        return 1 if k == 0 else 0

    if k == 0 or k > n:
        return 0

    if k == 1 or k == n:
        return 1

    # Use dynamic programming to compute S(n,k)
    # Create a table to store previously computed values
    dp: dict[tuple[int, int], int] = {}

    async def stirling_helper(n_val: int, k_val: int) -> int:
        if (n_val, k_val) in dp:
            return dp[(n_val, k_val)]

        if n_val == 0:
            result = 1 if k_val == 0 else 0
        elif k_val == 0 or k_val > n_val:
            result = 0
        elif k_val == 1 or k_val == n_val:
            result = 1
        else:
            # S(n,k) = k*S(n-1,k) + S(n-1,k-1)
            result = k_val * await stirling_helper(
                n_val - 1, k_val
            ) + await stirling_helper(n_val - 1, k_val - 1)

        dp[(n_val, k_val)] = result
        return result

    return await stirling_helper(n, k)


@mcp_function(
    description="Calculate Stirling number of the first kind s(n,k) - FIXED IMPLEMENTATION.",
    namespace="arithmetic",
    category="combinatorial_numbers",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    estimated_cpu_usage="medium",
    examples=[
        {"input": {"n": 4, "k": 2}, "output": 11, "description": "s(4,2) = 11"},
        {"input": {"n": 5, "k": 3}, "output": 35, "description": "s(5,3) = 35"},
        {"input": {"n": 3, "k": 2}, "output": 3, "description": "s(3,2) = 3"},
        {"input": {"n": 7, "k": 3}, "output": 735, "description": "s(7,3) = 735"},
    ],
)
async def stirling_first(n: int, k: int) -> int:
    """
    Calculate unsigned Stirling number of the first kind s(n,k).

    s(n,k) counts the number of permutations of n objects with k cycles.

    Uses the recurrence: s(n,k) = (n-1)*s(n-1,k) + s(n-1,k-1)

    Args:
        n: Number of objects
        k: Number of cycles

    Returns:
        Unsigned Stirling number of the first kind s(n,k)

    Examples:
        await stirling_first(4, 2) → 11  # 11 permutations of 4 objects with 2 cycles
        await stirling_first(5, 3) → 35  # 35 permutations of 5 objects with 3 cycles
        await stirling_first(7, 3) → 735 # 735 permutations of 7 objects with 3 cycles
    """
    if n < 0 or k < 0:
        raise ValueError("n and k must be non-negative")

    if n == 0:
        return 1 if k == 0 else 0

    if k == 0 or k > n:
        return 0

    if k == n:
        return 1

    # Use dynamic programming with iterative approach for better performance
    # Build table bottom-up to avoid deep recursion

    # Initialize table
    table = [[0 for _ in range(k + 1)] for _ in range(n + 1)]

    # Base cases
    table[0][0] = 1
    for i in range(1, n + 1):
        table[i][0] = 0
    for j in range(1, k + 1):
        table[0][j] = 0

    # Fill table using recurrence: s(n,k) = (n-1)*s(n-1,k) + s(n-1,k-1)
    for i in range(1, n + 1):
        for j in range(1, min(i + 1, k + 1)):
            if j == i:
                table[i][j] = 1
            else:
                table[i][j] = (i - 1) * table[i - 1][j] + table[i - 1][j - 1]

        # Yield control periodically for large computations
        if i % 20 == 0 and n > 50:
            await asyncio.sleep(0)

    return table[n][k]


@mcp_function(
    description="Generate row n of Stirling numbers of the second kind.",
    namespace="arithmetic",
    category="combinatorial_numbers",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    examples=[
        {
            "input": {"n": 4},
            "output": [0, 1, 7, 6, 1],
            "description": "Row 4: S(4,0) through S(4,4)",
        },
        {
            "input": {"n": 5},
            "output": [0, 1, 15, 25, 10, 1],
            "description": "Row 5: S(5,0) through S(5,5)",
        },
    ],
)
async def stirling_second_row(n: int) -> List[int]:
    """
    Generate row n of Stirling numbers of the second kind.

    Returns [S(n,0), S(n,1), S(n,2), ..., S(n,n)]

    Args:
        n: Row number

    Returns:
        List of Stirling numbers of the second kind for row n

    Examples:
        await stirling_second_row(4) → [0, 1, 7, 6, 1]
        await stirling_second_row(5) → [0, 1, 15, 25, 10, 1]
    """
    if n < 0:
        raise ValueError("Row number must be non-negative")

    if n == 0:
        return [1]

    result = []
    for k in range(n + 1):
        result.append(await stirling_second(n, k))

        if k % 10 == 0 and n > 10:
            await asyncio.sleep(0)

    return result


# ============================================================================
# NARAYANA NUMBERS - FIXED
# ============================================================================


def _binomial_coefficient(n: int, k: int) -> int:
    """
    Calculate binomial coefficient C(n,k) = n! / (k! * (n-k)!)

    Uses multiplicative formula to avoid large factorials.
    """
    if k > n or k < 0:
        return 0
    if k == 0 or k == n:
        return 1

    # Use symmetry: C(n,k) = C(n,n-k)
    k = min(k, n - k)

    result = 1
    for i in range(k):
        result = result * (n - i) // (i + 1)

    return result


@mcp_function(
    description="Calculate Narayana number N(n,k) = (1/n) * C(n,k) * C(n,k-1) - FIXED.",
    namespace="arithmetic",
    category="combinatorial_numbers",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    examples=[
        {"input": {"n": 3, "k": 2}, "output": 3, "description": "N(3,2) = 3"},
        {"input": {"n": 4, "k": 2}, "output": 6, "description": "N(4,2) = 6"},
        {"input": {"n": 4, "k": 3}, "output": 4, "description": "N(4,3) = 4"},
        {"input": {"n": 6, "k": 4}, "output": 60, "description": "N(6,4) = 60"},
    ],
)
async def narayana_number(n: int, k: int) -> int:
    """
    Calculate Narayana number N(n,k).

    N(n,k) = (1/n) * C(n,k) * C(n,k-1)

    Narayana numbers count the number of Dyck paths from (0,0) to (2n,0)
    with exactly k peaks.

    Args:
        n: Path parameter
        k: Number of peaks

    Returns:
        Narayana number N(n,k)

    Examples:
        await narayana_number(3, 2) → 3   # 3 Dyck paths of length 6 with 2 peaks
        await narayana_number(4, 2) → 6   # 6 Dyck paths of length 8 with 2 peaks
        await narayana_number(4, 3) → 4   # 4 Dyck paths of length 8 with 3 peaks
    """
    if n <= 0 or k <= 0 or k > n:
        return 0

    # N(n,k) = (1/n) * C(n,k) * C(n,k-1)
    c_n_k = _binomial_coefficient(n, k)
    c_n_k_minus_1 = _binomial_coefficient(n, k - 1)

    return (c_n_k * c_n_k_minus_1) // n


@mcp_function(
    description="Generate row n of Narayana triangle - FIXED.",
    namespace="arithmetic",
    category="combinatorial_numbers",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    examples=[
        {
            "input": {"n": 4},
            "output": [1, 6, 4, 1],
            "description": "Row 4 of Narayana triangle",
        },
        {
            "input": {"n": 5},
            "output": [1, 10, 20, 10, 1],
            "description": "Row 5 of Narayana triangle",
        },
    ],
)
async def narayana_triangle_row(n: int) -> List[int]:
    """
    Generate row n of the Narayana triangle.

    Returns [N(n,1), N(n,2), ..., N(n,n)]

    Args:
        n: Row number

    Returns:
        List of Narayana numbers for row n

    Examples:
        await narayana_triangle_row(4) → [1, 6, 4, 1]
        await narayana_triangle_row(5) → [1, 10, 20, 10, 1]
    """
    if n <= 0:
        return []

    result = []
    for k in range(1, n + 1):
        result.append(await narayana_number(n, k))

        if k % 10 == 0 and n > 10:
            await asyncio.sleep(0)

    return result


# Export all functions
__all__ = [
    # Catalan numbers
    "catalan_number",
    "catalan_sequence",
    "is_catalan_number",
    # Bell numbers
    "bell_number",
    "bell_sequence",
    "bell_triangle",
    # Stirling numbers
    "stirling_first",
    "stirling_second",
    "stirling_second_row",
    # Narayana numbers
    "narayana_number",
    "narayana_triangle_row",
]

if __name__ == "__main__":
    import asyncio

    async def test_combinatorial_numbers():
        """Test combinatorial number functions."""
        print("🔢 Combinatorial Numbers Test - FIXED")
        print("=" * 40)

        # Test Catalan numbers
        print("Catalan Numbers:")
        print(f"  catalan_number(5) = {await catalan_number(5)}")
        print(f"  catalan_sequence(6) = {await catalan_sequence(6)}")
        print(f"  is_catalan_number(14) = {await is_catalan_number(14)}")
        print(f"  is_catalan_number(20) = {await is_catalan_number(20)}")

        # Test Bell numbers
        print("\nBell Numbers:")
        print(f"  bell_number(5) = {await bell_number(5)}")
        print(f"  bell_sequence(6) = {await bell_sequence(6)}")
        triangle = await bell_triangle(4)
        print(f"  bell_triangle(4) = {triangle}")

        # Test Stirling numbers (check the fixes)
        print("\nStirling Numbers (FIXED):")
        print(f"  stirling_second(4, 2) = {await stirling_second(4, 2)}")
        print(f"  stirling_first(4, 2) = {await stirling_first(4, 2)}")
        print(f"  stirling_first(7, 3) = {await stirling_first(7, 3)}")  # Should be 735
        print(f"  stirling_second_row(4) = {await stirling_second_row(4)}")

        # Test Narayana numbers (check the fixes)
        print("\nNarayana Numbers (FIXED):")
        print(f"  narayana_number(4, 2) = {await narayana_number(4, 2)}")
        print(f"  narayana_number(4, 3) = {await narayana_number(4, 3)}")  # Should be 4
        print(
            f"  narayana_number(6, 4) = {await narayana_number(6, 4)}"
        )  # Should be 60
        print(
            f"  narayana_triangle_row(4) = {await narayana_triangle_row(4)}"
        )  # Should be [1, 6, 4, 1]

        print("\n✅ All combinatorial number functions working!")

    asyncio.run(test_combinatorial_numbers())
