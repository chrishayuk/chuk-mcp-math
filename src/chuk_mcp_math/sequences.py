#!/usr/bin/env python3
# chuk_mcp_math/arithmetic/sequences.py
"""
Sequences and Series Functions for AI Models (Async Native)

Mathematical functions for generating and analyzing sequences, series, and patterns.
Essential for pattern recognition, mathematical modeling, and algorithmic analysis.
All functions are async native for optimal performance in async environments.

Functions:
- Arithmetic sequences: arithmetic_sequence, arithmetic_sum
- Geometric sequences: geometric_sequence, geometric_sum
- Special sequences: triangular_numbers, square_numbers, cube_numbers
- Series calculations: harmonic_series, power_series_sum
- Pattern analysis: find_differences, is_arithmetic, is_geometric
"""

import asyncio
from typing import List, Union, Optional
from chuk_mcp_math.mcp_decorator import mcp_function

Number = Union[int, float]


@mcp_function(
    description="Generate an arithmetic sequence with given first term, common difference, and number of terms.",
    namespace="arithmetic",
    execution_modes=["local", "remote"],
    examples=[
        {
            "input": {"first_term": 2, "common_diff": 3, "num_terms": 5},
            "output": [2, 5, 8, 11, 14],
            "description": "Arithmetic sequence starting at 2, diff 3",
        },
        {
            "input": {"first_term": 10, "common_diff": -2, "num_terms": 4},
            "output": [10, 8, 6, 4],
            "description": "Decreasing arithmetic sequence",
        },
        {
            "input": {"first_term": 0, "common_diff": 1, "num_terms": 6},
            "output": [0, 1, 2, 3, 4, 5],
            "description": "Natural numbers starting from 0",
        },
        {
            "input": {"first_term": 1, "common_diff": 0, "num_terms": 3},
            "output": [1, 1, 1],
            "description": "Constant sequence",
        },
    ],
)
async def arithmetic_sequence(
    first_term: Number, common_diff: Number, num_terms: int
) -> List[Number]:
    """
    Generate an arithmetic sequence.

    Args:
        first_term: The first term of the sequence
        common_diff: The common difference between consecutive terms
        num_terms: Number of terms to generate (must be positive)

    Returns:
        List containing the arithmetic sequence

    Raises:
        ValueError: If num_terms is not positive

    Examples:
        await arithmetic_sequence(2, 3, 5) → [2, 5, 8, 11, 14]
        await arithmetic_sequence(10, -2, 4) → [10, 8, 6, 4]
    """
    if num_terms <= 0:
        raise ValueError("Number of terms must be positive")

    # Yield control for large sequences
    if num_terms > 1000:
        await asyncio.sleep(0)

    result = []
    for i in range(num_terms):
        result.append(first_term + i * common_diff)
        # Yield control every 1000 terms for very large sequences
        if i % 1000 == 999 and num_terms > 1000:
            await asyncio.sleep(0)

    return result


@mcp_function(
    description="Calculate the sum of an arithmetic sequence using the formula: n/2 * (2a + (n-1)d).",
    namespace="arithmetic",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    examples=[
        {
            "input": {"first_term": 2, "common_diff": 3, "num_terms": 5},
            "output": 40,
            "description": "Sum of [2, 5, 8, 11, 14]",
        },
        {
            "input": {"first_term": 1, "common_diff": 1, "num_terms": 10},
            "output": 55,
            "description": "Sum of first 10 natural numbers",
        },
        {
            "input": {"first_term": 5, "common_diff": 0, "num_terms": 4},
            "output": 20,
            "description": "Sum of constant sequence",
        },
        {
            "input": {"first_term": 10, "common_diff": -1, "num_terms": 5},
            "output": 40,
            "description": "Sum of decreasing sequence",
        },
    ],
)
async def arithmetic_sum(
    first_term: Number, common_diff: Number, num_terms: int
) -> Number:
    """
    Calculate the sum of an arithmetic sequence.

    Args:
        first_term: The first term of the sequence
        common_diff: The common difference between consecutive terms
        num_terms: Number of terms to sum (must be positive)

    Returns:
        The sum of the arithmetic sequence

    Raises:
        ValueError: If num_terms is not positive

    Examples:
        await arithmetic_sum(2, 3, 5) → 40
        await arithmetic_sum(1, 1, 10) → 55
    """
    if num_terms <= 0:
        raise ValueError("Number of terms must be positive")

    # Formula: S_n = n/2 * (2a + (n-1)d)
    return num_terms * (2 * first_term + (num_terms - 1) * common_diff) / 2


@mcp_function(
    description="Generate a geometric sequence with given first term, common ratio, and number of terms.",
    namespace="arithmetic",
    execution_modes=["local", "remote"],
    examples=[
        {
            "input": {"first_term": 2, "common_ratio": 3, "num_terms": 5},
            "output": [2, 6, 18, 54, 162],
            "description": "Geometric sequence with ratio 3",
        },
        {
            "input": {"first_term": 1, "common_ratio": 0.5, "num_terms": 4},
            "output": [1, 0.5, 0.25, 0.125],
            "description": "Decreasing geometric sequence",
        },
        {
            "input": {"first_term": 5, "common_ratio": 1, "num_terms": 3},
            "output": [5, 5, 5],
            "description": "Constant geometric sequence",
        },
        {
            "input": {"first_term": 1, "common_ratio": -2, "num_terms": 4},
            "output": [1, -2, 4, -8],
            "description": "Alternating geometric sequence",
        },
    ],
)
async def geometric_sequence(
    first_term: Number, common_ratio: Number, num_terms: int
) -> List[Number]:
    """
    Generate a geometric sequence.

    Args:
        first_term: The first term of the sequence
        common_ratio: The common ratio between consecutive terms
        num_terms: Number of terms to generate (must be positive)

    Returns:
        List containing the geometric sequence

    Raises:
        ValueError: If num_terms is not positive

    Examples:
        await geometric_sequence(2, 3, 5) → [2, 6, 18, 54, 162]
        await geometric_sequence(1, 0.5, 4) → [1, 0.5, 0.25, 0.125]
    """
    if num_terms <= 0:
        raise ValueError("Number of terms must be positive")

    # Yield control for large sequences
    if num_terms > 1000:
        await asyncio.sleep(0)

    result = []
    current_term = first_term

    for i in range(num_terms):
        result.append(current_term)
        current_term *= common_ratio
        # Yield control every 1000 terms for very large sequences
        if i % 1000 == 999 and num_terms > 1000:
            await asyncio.sleep(0)

    return result


@mcp_function(
    description="Calculate the sum of a geometric sequence. For |r| < 1, can calculate infinite series sum.",
    namespace="arithmetic",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    examples=[
        {
            "input": {"first_term": 2, "common_ratio": 3, "num_terms": 5},
            "output": 242,
            "description": "Sum of [2, 6, 18, 54, 162]",
        },
        {
            "input": {"first_term": 1, "common_ratio": 0.5, "num_terms": 10},
            "output": 1.998046875,
            "description": "Sum of decreasing geometric sequence",
        },
        {
            "input": {"first_term": 1, "common_ratio": 0.5, "num_terms": None},
            "output": 2.0,
            "description": "Infinite geometric series sum",
        },
        {
            "input": {"first_term": 3, "common_ratio": 1, "num_terms": 5},
            "output": 15,
            "description": "Sum when ratio = 1",
        },
    ],
)
async def geometric_sum(
    first_term: Number, common_ratio: Number, num_terms: Optional[int] = None
) -> Number:
    """
    Calculate the sum of a geometric sequence.

    Args:
        first_term: The first term of the sequence
        common_ratio: The common ratio between consecutive terms
        num_terms: Number of terms to sum (None for infinite series)

    Returns:
        The sum of the geometric sequence or series

    Raises:
        ValueError: If requesting infinite sum with |ratio| >= 1, or num_terms is not positive

    Examples:
        await geometric_sum(2, 3, 5) → 242
        await geometric_sum(1, 0.5, 10) → 1.998046875
        await geometric_sum(1, 0.5, None) → 2.0  # Infinite series
    """
    if num_terms is None:
        # Infinite series
        if abs(common_ratio) >= 1:
            raise ValueError(
                "Infinite geometric series only converges when |ratio| < 1"
            )
        return first_term / (1 - common_ratio)

    if num_terms <= 0:
        raise ValueError("Number of terms must be positive")

    if common_ratio == 1:
        return first_term * num_terms

    # Formula: S_n = a * (1 - r^n) / (1 - r)
    return first_term * (1 - common_ratio**num_terms) / (1 - common_ratio)


@mcp_function(
    description="Generate the first n triangular numbers (1, 3, 6, 10, 15, ...). nth triangular number = n(n+1)/2.",
    namespace="arithmetic",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    examples=[
        {
            "input": {"n": 5},
            "output": [1, 3, 6, 10, 15],
            "description": "First 5 triangular numbers",
        },
        {"input": {"n": 1}, "output": [1], "description": "First triangular number"},
        {
            "input": {"n": 8},
            "output": [1, 3, 6, 10, 15, 21, 28, 36],
            "description": "First 8 triangular numbers",
        },
        {"input": {"n": 0}, "output": [], "description": "Empty sequence"},
    ],
)
async def triangular_numbers(n: int) -> List[int]:
    """
    Generate the first n triangular numbers.

    Args:
        n: Number of triangular numbers to generate (non-negative)

    Returns:
        List of the first n triangular numbers

    Examples:
        await triangular_numbers(5) → [1, 3, 6, 10, 15]
        await triangular_numbers(1) → [1]
        await triangular_numbers(0) → []
    """
    if n < 0:
        raise ValueError("n must be non-negative")

    # Yield control for large sequences
    if n > 1000:
        await asyncio.sleep(0)

    result = []
    for i in range(1, n + 1):
        result.append(i * (i + 1) // 2)
        # Yield control every 1000 terms for very large sequences
        if i % 1000 == 0 and n > 1000:
            await asyncio.sleep(0)

    return result


@mcp_function(
    description="Generate the first n square numbers (1, 4, 9, 16, 25, ...). nth square number = n².",
    namespace="arithmetic",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    examples=[
        {
            "input": {"n": 5},
            "output": [1, 4, 9, 16, 25],
            "description": "First 5 square numbers",
        },
        {"input": {"n": 1}, "output": [1], "description": "First square number"},
        {
            "input": {"n": 7},
            "output": [1, 4, 9, 16, 25, 36, 49],
            "description": "First 7 square numbers",
        },
        {"input": {"n": 0}, "output": [], "description": "Empty sequence"},
    ],
)
async def square_numbers(n: int) -> List[int]:
    """
    Generate the first n square numbers.

    Args:
        n: Number of square numbers to generate (non-negative)

    Returns:
        List of the first n square numbers

    Examples:
        await square_numbers(5) → [1, 4, 9, 16, 25]
        await square_numbers(1) → [1]
        await square_numbers(0) → []
    """
    if n < 0:
        raise ValueError("n must be non-negative")

    # Yield control for large sequences
    if n > 1000:
        await asyncio.sleep(0)

    result = []
    for i in range(1, n + 1):
        result.append(i * i)
        # Yield control every 1000 terms for very large sequences
        if i % 1000 == 0 and n > 1000:
            await asyncio.sleep(0)

    return result


@mcp_function(
    description="Generate the first n cube numbers (1, 8, 27, 64, 125, ...). nth cube number = n³.",
    namespace="arithmetic",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    examples=[
        {
            "input": {"n": 5},
            "output": [1, 8, 27, 64, 125],
            "description": "First 5 cube numbers",
        },
        {"input": {"n": 1}, "output": [1], "description": "First cube number"},
        {
            "input": {"n": 6},
            "output": [1, 8, 27, 64, 125, 216],
            "description": "First 6 cube numbers",
        },
        {"input": {"n": 0}, "output": [], "description": "Empty sequence"},
    ],
)
async def cube_numbers(n: int) -> List[int]:
    """
    Generate the first n cube numbers.

    Args:
        n: Number of cube numbers to generate (non-negative)

    Returns:
        List of the first n cube numbers

    Examples:
        await cube_numbers(5) → [1, 8, 27, 64, 125]
        await cube_numbers(1) → [1]
        await cube_numbers(0) → []
    """
    if n < 0:
        raise ValueError("n must be non-negative")

    # Yield control for large sequences
    if n > 1000:
        await asyncio.sleep(0)

    result = []
    for i in range(1, n + 1):
        result.append(i**3)
        # Yield control every 1000 terms for very large sequences
        if i % 1000 == 0 and n > 1000:
            await asyncio.sleep(0)

    return result


@mcp_function(
    description="Calculate the nth partial sum of the harmonic series: 1 + 1/2 + 1/3 + ... + 1/n.",
    namespace="arithmetic",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    estimated_cpu_usage="medium",
    examples=[
        {"input": {"n": 1}, "output": 1.0, "description": "H₁ = 1"},
        {"input": {"n": 2}, "output": 1.5, "description": "H₂ = 1 + 1/2"},
        {
            "input": {"n": 4},
            "output": 2.083333333333333,
            "description": "H₄ = 1 + 1/2 + 1/3 + 1/4",
        },
        {"input": {"n": 10}, "output": 2.9289682539682538, "description": "H₁₀"},
    ],
)
async def harmonic_series(n: int) -> float:
    """
    Calculate the nth partial sum of the harmonic series.

    Args:
        n: Positive integer indicating how many terms to sum

    Returns:
        The nth harmonic number H_n

    Raises:
        ValueError: If n is not positive

    Examples:
        await harmonic_series(1) → 1.0
        await harmonic_series(2) → 1.5
        await harmonic_series(4) → 2.083333333333333
    """
    if n <= 0:
        raise ValueError("n must be positive")

    # Yield control for large calculations
    if n > 10000:
        await asyncio.sleep(0)

    total = 0.0
    for i in range(1, n + 1):
        total += 1.0 / i
        # Yield control every 10000 terms for very large calculations
        if i % 10000 == 0 and n > 10000:
            await asyncio.sleep(0)

    return total


@mcp_function(
    description="Calculate the sum of a power series: x⁰ + x¹ + x² + ... + xⁿ for given x and n.",
    namespace="arithmetic",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    examples=[
        {
            "input": {"x": 2, "n": 4},
            "output": 31,
            "description": "1 + 2 + 4 + 8 + 16 = 31",
        },
        {
            "input": {"x": 0.5, "n": 3},
            "output": 1.875,
            "description": "1 + 0.5 + 0.25 + 0.125 = 1.875",
        },
        {
            "input": {"x": 1, "n": 5},
            "output": 6,
            "description": "1 + 1 + 1 + 1 + 1 + 1 = 6",
        },
        {"input": {"x": -1, "n": 3}, "output": 0, "description": "1 - 1 + 1 - 1 = 0"},
    ],
)
async def power_series_sum(x: Number, n: int) -> Number:
    """
    Calculate the sum of a power series from x⁰ to xⁿ.

    Args:
        x: The base value
        n: The highest power (non-negative)

    Returns:
        Sum of x⁰ + x¹ + x² + ... + xⁿ

    Raises:
        ValueError: If n is negative

    Examples:
        await power_series_sum(2, 4) → 31
        await power_series_sum(0.5, 3) → 1.875
        await power_series_sum(1, 5) → 6
    """
    if n < 0:
        raise ValueError("n must be non-negative")

    if x == 1:
        return n + 1

    # Use geometric series formula: (1 - x^(n+1)) / (1 - x)
    return (1 - x ** (n + 1)) / (1 - x)


@mcp_function(
    description="Find the differences between consecutive terms in a sequence. Useful for pattern analysis.",
    namespace="arithmetic",
    execution_modes=["local", "remote"],
    examples=[
        {
            "input": {"sequence": [1, 4, 9, 16, 25]},
            "output": [3, 5, 7, 9],
            "description": "Differences in square numbers",
        },
        {
            "input": {"sequence": [2, 5, 8, 11, 14]},
            "output": [3, 3, 3, 3],
            "description": "Constant differences (arithmetic)",
        },
        {
            "input": {"sequence": [1, 2, 4, 8, 16]},
            "output": [1, 2, 4, 8],
            "description": "Differences in powers of 2",
        },
        {
            "input": {"sequence": [10]},
            "output": [],
            "description": "Single element sequence",
        },
    ],
)
async def find_differences(sequence: List[Number]) -> List[Number]:
    """
    Find the differences between consecutive terms in a sequence.

    Args:
        sequence: List of numbers

    Returns:
        List of differences between consecutive terms

    Examples:
        await find_differences([1, 4, 9, 16, 25]) → [3, 5, 7, 9]
        await find_differences([2, 5, 8, 11, 14]) → [3, 3, 3, 3]
    """
    if len(sequence) < 2:
        return []

    # Yield control for large sequences
    if len(sequence) > 1000:
        await asyncio.sleep(0)

    result = []
    for i in range(len(sequence) - 1):
        result.append(sequence[i + 1] - sequence[i])
        # Yield control every 1000 differences for very large sequences
        if i % 1000 == 999 and len(sequence) > 1000:
            await asyncio.sleep(0)

    return result


@mcp_function(
    description="Check if a sequence is arithmetic (constant difference between consecutive terms).",
    namespace="arithmetic",
    execution_modes=["local", "remote"],
    examples=[
        {
            "input": {"sequence": [2, 5, 8, 11, 14]},
            "output": True,
            "description": "Arithmetic sequence with diff 3",
        },
        {
            "input": {"sequence": [1, 4, 9, 16, 25]},
            "output": False,
            "description": "Square numbers (not arithmetic)",
        },
        {
            "input": {"sequence": [10, 10, 10]},
            "output": True,
            "description": "Constant sequence (diff 0)",
        },
        {
            "input": {"sequence": [1]},
            "output": True,
            "description": "Single element (trivially arithmetic)",
        },
    ],
)
async def is_arithmetic(sequence: List[Number], tolerance: float = 1e-9) -> bool:
    """
    Check if a sequence is arithmetic.

    Args:
        sequence: List of numbers to check
        tolerance: Tolerance for floating-point comparison

    Returns:
        True if the sequence is arithmetic, False otherwise

    Examples:
        await is_arithmetic([2, 5, 8, 11, 14]) → True
        await is_arithmetic([1, 4, 9, 16, 25]) → False
        await is_arithmetic([10, 10, 10]) → True
    """
    if len(sequence) < 2:
        return True

    differences = await find_differences(sequence)

    if not differences:
        return True

    first_diff = differences[0]
    return all(abs(diff - first_diff) <= tolerance for diff in differences)


@mcp_function(
    description="Check if a sequence is geometric (constant ratio between consecutive terms).",
    namespace="arithmetic",
    execution_modes=["local", "remote"],
    examples=[
        {
            "input": {"sequence": [2, 6, 18, 54]},
            "output": True,
            "description": "Geometric sequence with ratio 3",
        },
        {
            "input": {"sequence": [1, 2, 4, 7]},
            "output": False,
            "description": "Not geometric",
        },
        {
            "input": {"sequence": [5, 5, 5]},
            "output": True,
            "description": "Constant sequence (ratio 1)",
        },
        {
            "input": {"sequence": [1, -2, 4, -8]},
            "output": True,
            "description": "Alternating geometric sequence",
        },
    ],
)
async def is_geometric(sequence: List[Number], tolerance: float = 1e-9) -> bool:
    """
    Check if a sequence is geometric.

    Args:
        sequence: List of numbers to check
        tolerance: Tolerance for floating-point comparison

    Returns:
        True if the sequence is geometric, False otherwise

    Examples:
        await is_geometric([2, 6, 18, 54]) → True
        await is_geometric([1, 2, 4, 7]) → False
        await is_geometric([5, 5, 5]) → True
    """
    if len(sequence) < 2:
        return True

    # Check for zeros (except possibly the first term)
    for i in range(1, len(sequence)):
        if abs(sequence[i - 1]) <= tolerance:
            return False

    if abs(sequence[0]) <= tolerance:
        # First term is zero, all subsequent terms must be zero
        return all(abs(term) <= tolerance for term in sequence[1:])

    # Calculate ratios
    ratios = []
    for i in range(1, len(sequence)):
        if abs(sequence[i - 1]) <= tolerance:
            return False
        ratios.append(sequence[i] / sequence[i - 1])

    if not ratios:
        return True

    first_ratio = ratios[0]
    return all(abs(ratio - first_ratio) <= tolerance for ratio in ratios)


# Export all sequence functions
__all__ = [
    "arithmetic_sequence",
    "arithmetic_sum",
    "geometric_sequence",
    "geometric_sum",
    "triangular_numbers",
    "square_numbers",
    "cube_numbers",
    "harmonic_series",
    "power_series_sum",
    "find_differences",
    "is_arithmetic",
    "is_geometric",
]

if __name__ == "__main__":
    import asyncio

    async def test_sequence_functions():
        # Test the sequence functions
        print("🔢 Sequences and Series Functions Test (Async Native)")
        print("=" * 50)

        # Test arithmetic sequences
        print(f"arithmetic_sequence(2, 3, 5) = {await arithmetic_sequence(2, 3, 5)}")
        print(f"arithmetic_sum(2, 3, 5) = {await arithmetic_sum(2, 3, 5)}")

        # Test geometric sequences
        print(f"geometric_sequence(2, 3, 4) = {await geometric_sequence(2, 3, 4)}")
        print(f"geometric_sum(2, 3, 4) = {await geometric_sum(2, 3, 4)}")
        print(f"geometric_sum(1, 0.5, None) = {await geometric_sum(1, 0.5, None)}")

        # Test special sequences
        print(f"triangular_numbers(5) = {await triangular_numbers(5)}")
        print(f"square_numbers(5) = {await square_numbers(5)}")
        print(f"cube_numbers(4) = {await cube_numbers(4)}")

        # Test series
        print(f"harmonic_series(4) = {await harmonic_series(4)}")
        print(f"power_series_sum(2, 4) = {await power_series_sum(2, 4)}")

        # Test pattern analysis
        seq = [2, 5, 8, 11, 14]
        print(f"find_differences({seq}) = {await find_differences(seq)}")
        print(f"is_arithmetic({seq}) = {await is_arithmetic(seq)}")

        geo_seq = [2, 6, 18, 54]
        print(f"is_geometric({geo_seq}) = {await is_geometric(geo_seq)}")

        print("\n✅ All async sequence functions working correctly!")

    asyncio.run(test_sequence_functions())
