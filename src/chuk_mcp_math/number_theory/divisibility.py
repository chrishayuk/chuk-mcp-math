#!/usr/bin/env python3
# chuk_mcp_math/number_theory/divisibility.py
"""
Divisibility Operations - Async Native

Functions for working with divisibility, GCD, LCM, and divisor operations.

Functions:
- gcd, lcm, divisors, is_divisible
- is_even, is_odd, extended_gcd
"""

import math
import asyncio
from typing import List, Tuple
from chuk_mcp_math.mcp_decorator import mcp_function


@mcp_function(
    description="Calculate the Greatest Common Divisor (GCD) of two integers using Euclidean algorithm.",
    namespace="arithmetic",
    category="number_theory",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    examples=[
        {"input": {"a": 48, "b": 18}, "output": 6, "description": "GCD of 48 and 18"},
        {
            "input": {"a": 17, "b": 13},
            "output": 1,
            "description": "GCD of coprime numbers",
        },
        {
            "input": {"a": 100, "b": 25},
            "output": 25,
            "description": "GCD when one divides the other",
        },
        {"input": {"a": 0, "b": 5}, "output": 5, "description": "GCD with zero"},
    ],
)
async def gcd(a: int, b: int) -> int:
    """
    Calculate the Greatest Common Divisor (GCD) of two integers.

    Args:
        a: First integer
        b: Second integer

    Returns:
        The GCD of a and b

    Examples:
        await gcd(48, 18) → 6
        await gcd(17, 13) → 1
        await gcd(100, 25) → 25
    """
    return math.gcd(abs(a), abs(b))


@mcp_function(
    description="Calculate the Least Common Multiple (LCM) of two integers.",
    namespace="arithmetic",
    category="number_theory",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    examples=[
        {"input": {"a": 12, "b": 18}, "output": 36, "description": "LCM of 12 and 18"},
        {
            "input": {"a": 7, "b": 13},
            "output": 91,
            "description": "LCM of coprime numbers",
        },
        {
            "input": {"a": 10, "b": 5},
            "output": 10,
            "description": "LCM when one divides the other",
        },
        {
            "input": {"a": 4, "b": 6},
            "output": 12,
            "description": "LCM of small numbers",
        },
    ],
)
async def lcm(a: int, b: int) -> int:
    """
    Calculate the Least Common Multiple (LCM) of two integers.

    Args:
        a: First integer
        b: Second integer

    Returns:
        The LCM of a and b

    Examples:
        await lcm(12, 18) → 36
        await lcm(7, 13) → 91
        await lcm(10, 5) → 10
    """
    if a == 0 or b == 0:
        return 0
    gcd_result = await gcd(a, b)
    return abs(a * b) // gcd_result


@mcp_function(
    description="Find all positive divisors of a number in sorted order.",
    namespace="arithmetic",
    category="number_theory",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    estimated_cpu_usage="medium",
    examples=[
        {
            "input": {"n": 12},
            "output": [1, 2, 3, 4, 6, 12],
            "description": "Divisors of 12",
        },
        {
            "input": {"n": 17},
            "output": [1, 17],
            "description": "Divisors of prime number",
        },
        {"input": {"n": 1}, "output": [1], "description": "Divisors of 1"},
        {
            "input": {"n": 36},
            "output": [1, 2, 3, 4, 6, 9, 12, 18, 36],
            "description": "Divisors of perfect square",
        },
    ],
)
async def divisors(n: int) -> List[int]:
    """
    Find all positive divisors of a number.

    Args:
        n: Positive integer

    Returns:
        Sorted list of all positive divisors

    Examples:
        await divisors(12) → [1, 2, 3, 4, 6, 12]
        await divisors(17) → [1, 17]
        await divisors(1) → [1]
    """
    if n <= 0:
        return []

    result = []
    sqrt_n = int(math.sqrt(n))

    # Yield control for large numbers
    if sqrt_n > 10000:
        await asyncio.sleep(0)

    for i in range(1, sqrt_n + 1):
        if n % i == 0:
            result.append(i)
            if i != n // i:
                result.append(n // i)

        # Yield control every 1000 iterations for large numbers
        if i % 1000 == 0 and sqrt_n > 10000:
            await asyncio.sleep(0)

    return sorted(result)


@mcp_function(
    description="Check if the first number is divisible by the second number (remainder is zero).",
    namespace="arithmetic",
    category="number_theory",
    execution_modes=["local", "remote"],
    examples=[
        {
            "input": {"a": 20, "b": 4},
            "output": True,
            "description": "20 is divisible by 4",
        },
        {
            "input": {"a": 17, "b": 3},
            "output": False,
            "description": "17 is not divisible by 3",
        },
        {
            "input": {"a": 0, "b": 5},
            "output": True,
            "description": "0 is divisible by any non-zero number",
        },
        {
            "input": {"a": 15, "b": 1},
            "output": True,
            "description": "Any number is divisible by 1",
        },
    ],
)
async def is_divisible(a: int, b: int) -> bool:
    """
    Check if a is divisible by b.

    Args:
        a: Dividend
        b: Divisor (cannot be zero)

    Returns:
        True if a is divisible by b, False otherwise

    Raises:
        ValueError: If b is zero

    Examples:
        await is_divisible(20, 4) → True
        await is_divisible(17, 3) → False
        await is_divisible(0, 5) → True
    """
    if b == 0:
        raise ValueError("Cannot check divisibility by zero")
    return a % b == 0


@mcp_function(
    description="Check if a number is even (divisible by 2).",
    namespace="arithmetic",
    category="number_theory",
    execution_modes=["local", "remote"],
    examples=[
        {"input": {"n": 4}, "output": True, "description": "4 is even"},
        {"input": {"n": 7}, "output": False, "description": "7 is odd"},
        {"input": {"n": 0}, "output": True, "description": "0 is even"},
        {"input": {"n": -2}, "output": True, "description": "Negative even number"},
    ],
)
async def is_even(n: int) -> bool:
    """
    Check if a number is even.

    Args:
        n: Integer to check

    Returns:
        True if n is even, False otherwise

    Examples:
        await is_even(4) → True
        await is_even(7) → False
        await is_even(0) → True
    """
    return n % 2 == 0


@mcp_function(
    description="Check if a number is odd (not divisible by 2).",
    namespace="arithmetic",
    category="number_theory",
    execution_modes=["local", "remote"],
    examples=[
        {"input": {"n": 7}, "output": True, "description": "7 is odd"},
        {"input": {"n": 4}, "output": False, "description": "4 is even"},
        {"input": {"n": 1}, "output": True, "description": "1 is odd"},
        {"input": {"n": -3}, "output": True, "description": "Negative odd number"},
    ],
)
async def is_odd(n: int) -> bool:
    """
    Check if a number is odd.

    Args:
        n: Integer to check

    Returns:
        True if n is odd, False otherwise

    Examples:
        await is_odd(7) → True
        await is_odd(4) → False
        await is_odd(1) → True
    """
    return n % 2 != 0


@mcp_function(
    description="Extended Euclidean algorithm. Returns gcd(a,b) and coefficients x,y such that ax + by = gcd(a,b).",
    namespace="arithmetic",
    category="number_theory",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    examples=[
        {
            "input": {"a": 30, "b": 18},
            "output": [6, -1, 2],
            "description": "gcd=6, 30×(-1) + 18×2 = 6",
        },
        {
            "input": {"a": 35, "b": 15},
            "output": [5, 1, -2],
            "description": "gcd=5, 35×1 + 15×(-2) = 5",
        },
        {
            "input": {"a": 17, "b": 13},
            "output": [1, -3, 4],
            "description": "gcd=1, 17×(-3) + 13×4 = 1",
        },
        {
            "input": {"a": 0, "b": 5},
            "output": [5, 0, 1],
            "description": "gcd=5, 0×0 + 5×1 = 5",
        },
    ],
)
async def extended_gcd(a: int, b: int) -> Tuple[int, int, int]:
    """
    Extended Euclidean algorithm.

    Finds integers x, y such that ax + by = gcd(a, b).

    Args:
        a: First integer
        b: Second integer

    Returns:
        Tuple (gcd, x, y) where gcd = gcd(a, b) and ax + by = gcd

    Examples:
        await extended_gcd(30, 18) → (6, -1, 2)
        await extended_gcd(35, 15) → (5, 1, -2)
        await extended_gcd(17, 13) → (1, -3, 4)
    """
    if a == 0:
        return b, 0, 1

    gcd_val, x1, y1 = await extended_gcd(b % a, a)
    x = y1 - (b // a) * x1
    y = x1

    return gcd_val, x, y


@mcp_function(
    description="Count the number of positive divisors of a number (divisor function τ(n)).",
    namespace="arithmetic",
    category="number_theory",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    estimated_cpu_usage="medium",
    examples=[
        {
            "input": {"n": 12},
            "output": 6,
            "description": "12 has 6 divisors: 1,2,3,4,6,12",
        },
        {
            "input": {"n": 17},
            "output": 2,
            "description": "Prime numbers have 2 divisors",
        },
        {"input": {"n": 1}, "output": 1, "description": "1 has 1 divisor"},
        {"input": {"n": 36}, "output": 9, "description": "36 has 9 divisors"},
    ],
)
async def divisor_count(n: int) -> int:
    """
    Count the number of positive divisors of a number.

    Args:
        n: Positive integer

    Returns:
        Number of positive divisors of n

    Examples:
        await divisor_count(12) → 6  # divisors: 1,2,3,4,6,12
        await divisor_count(17) → 2  # divisors: 1,17
        await divisor_count(1) → 1   # divisors: 1
    """
    if n <= 0:
        return 0

    count = 0
    sqrt_n = int(math.sqrt(n))

    # Yield control for large numbers
    if sqrt_n > 10000:
        await asyncio.sleep(0)

    for i in range(1, sqrt_n + 1):
        if n % i == 0:
            count += 1  # Count divisor i
            if i != n // i:
                count += 1  # Count divisor n//i (if different)

        # Yield control every 1000 iterations for large numbers
        if i % 1000 == 0 and sqrt_n > 10000:
            await asyncio.sleep(0)

    return count


@mcp_function(
    description="Calculate the sum of all positive divisors of a number (divisor function σ(n)).",
    namespace="arithmetic",
    category="number_theory",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    estimated_cpu_usage="medium",
    examples=[
        {"input": {"n": 12}, "output": 28, "description": "Sum: 1+2+3+4+6+12 = 28"},
        {"input": {"n": 6}, "output": 12, "description": "Sum: 1+2+3+6 = 12"},
        {"input": {"n": 17}, "output": 18, "description": "Sum: 1+17 = 18 (prime)"},
        {"input": {"n": 1}, "output": 1, "description": "Sum: 1 = 1"},
    ],
)
async def divisor_sum(n: int) -> int:
    """
    Calculate the sum of all positive divisors of a number.

    Args:
        n: Positive integer

    Returns:
        Sum of all positive divisors of n

    Examples:
        await divisor_sum(12) → 28  # 1+2+3+4+6+12
        await divisor_sum(6) → 12   # 1+2+3+6
        await divisor_sum(17) → 18  # 1+17
    """
    if n <= 0:
        return 0

    total = 0
    sqrt_n = int(math.sqrt(n))

    # Yield control for large numbers
    if sqrt_n > 10000:
        await asyncio.sleep(0)

    for i in range(1, sqrt_n + 1):
        if n % i == 0:
            total += i  # Add divisor i
            if i != n // i:
                total += n // i  # Add divisor n//i (if different)

        # Yield control every 1000 iterations for large numbers
        if i % 1000 == 0 and sqrt_n > 10000:
            await asyncio.sleep(0)

    return total


# Export all divisibility functions
__all__ = [
    "gcd",
    "lcm",
    "divisors",
    "is_divisible",
    "is_even",
    "is_odd",
    "extended_gcd",
    "divisor_count",
    "divisor_sum",
]

if __name__ == "__main__":
    import asyncio

    async def test_divisibility_operations():
        """Test all divisibility operations."""
        print("🔢 Divisibility Operations Test")
        print("=" * 35)

        # Test basic divisibility functions
        print(f"gcd(48, 18) = {await gcd(48, 18)}")
        print(f"lcm(12, 18) = {await lcm(12, 18)}")
        print(f"divisors(12) = {await divisors(12)}")
        print(f"is_divisible(20, 4) = {await is_divisible(20, 4)}")
        print(f"is_divisible(17, 3) = {await is_divisible(17, 3)}")

        # Test parity functions
        print(f"is_even(4) = {await is_even(4)}")
        print(f"is_odd(7) = {await is_odd(7)}")

        # Test extended GCD
        gcd_val, x, y = await extended_gcd(30, 18)
        print(f"extended_gcd(30, 18) = ({gcd_val}, {x}, {y})")
        print(f"Verification: 30×{x} + 18×{y} = {30 * x + 18 * y}")

        # Test divisor functions
        print(f"divisor_count(12) = {await divisor_count(12)}")
        print(f"divisor_sum(12) = {await divisor_sum(12)}")

        print("\n✅ All divisibility operations working!")

    asyncio.run(test_divisibility_operations())
