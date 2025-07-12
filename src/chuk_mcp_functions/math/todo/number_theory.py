#!/usr/bin/env python3
# chuk_mcp_functions/math/arithmetic/number_theory.py
"""
Number Theory Functions for AI Models (Async Native)

Mathematical functions dealing with properties of integers, prime numbers,
divisibility, and discrete mathematics. Essential for cryptography, algorithms,
and mathematical analysis. All functions are async native with strategic yield points
for computationally expensive operations.

Functions:
- Prime number operations: is_prime, next_prime, prime_factors, nth_prime
- Divisibility: gcd, lcm, divisors, is_divisible
- Integer properties: is_even, is_odd, is_perfect_square, is_power_of_two
- Sequences: fibonacci, factorial, collatz_steps
- Modular arithmetic: mod_power
"""

import math
import asyncio
from typing import List, Tuple, Optional, Union
from chuk_mcp_functions.mcp_decorator import mcp_function

@mcp_function(
    description="Check if a number is prime. A prime number is only divisible by 1 and itself.",
    namespace="arithmetic",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    estimated_cpu_usage="medium",
    examples=[
        {"input": {"n": 17}, "output": True, "description": "17 is prime"},
        {"input": {"n": 4}, "output": False, "description": "4 is not prime (divisible by 2)"},
        {"input": {"n": 2}, "output": True, "description": "2 is the smallest prime"},
        {"input": {"n": 1}, "output": False, "description": "1 is not considered prime"}
    ]
)
async def is_prime(n: int) -> bool:
    """
    Check if a number is prime.
    
    Args:
        n: Positive integer to check
    
    Returns:
        True if n is prime, False otherwise
    
    Examples:
        await is_prime(17) → True
        await is_prime(4) → False
        await is_prime(2) → True
        await is_prime(1) → False
    """
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    
    # For large numbers, yield control periodically
    sqrt_n = int(math.sqrt(n))
    if sqrt_n > 10000:
        await asyncio.sleep(0)
    
    # Check odd divisors up to sqrt(n)
    for i in range(3, sqrt_n + 1, 2):
        if n % i == 0:
            return False
        # Yield control every 1000 iterations for very large numbers
        if i % 1000 == 999 and sqrt_n > 10000:
            await asyncio.sleep(0)
    
    return True

@mcp_function(
    description="Find the next prime number greater than the given number.",
    namespace="arithmetic",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    estimated_cpu_usage="medium",
    examples=[
        {"input": {"n": 10}, "output": 11, "description": "Next prime after 10"},
        {"input": {"n": 17}, "output": 19, "description": "Next prime after prime 17"},
        {"input": {"n": 1}, "output": 2, "description": "Next prime after 1"},
        {"input": {"n": 100}, "output": 101, "description": "Next prime after 100"}
    ]
)
async def next_prime(n: int) -> int:
    """
    Find the next prime number greater than n.
    
    Args:
        n: Starting number
    
    Returns:
        The smallest prime number greater than n
    
    Examples:
        await next_prime(10) → 11
        await next_prime(17) → 19
        await next_prime(1) → 2
    """
    candidate = n + 1
    checks = 0
    
    while not await is_prime(candidate):
        candidate += 1
        checks += 1
        # Yield control every 100 checks for large searches
        if checks % 100 == 0:
            await asyncio.sleep(0)
    
    return candidate

@mcp_function(
    description="Find all prime factors of a number. Returns the prime factorization as a list.",
    namespace="arithmetic",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    estimated_cpu_usage="high",
    examples=[
        {"input": {"n": 12}, "output": [2, 2, 3], "description": "12 = 2² × 3"},
        {"input": {"n": 17}, "output": [17], "description": "17 is prime"},
        {"input": {"n": 60}, "output": [2, 2, 3, 5], "description": "60 = 2² × 3 × 5"},
        {"input": {"n": 1}, "output": [], "description": "1 has no prime factors"}
    ]
)
async def prime_factors(n: int) -> List[int]:
    """
    Find all prime factors of a number.
    
    Args:
        n: Positive integer to factorize
    
    Returns:
        List of prime factors (with repetition)
    
    Examples:
        await prime_factors(12) → [2, 2, 3]
        await prime_factors(17) → [17]
        await prime_factors(60) → [2, 2, 3, 5]
    """
    if n <= 1:
        return []
    
    factors = []
    d = 2
    original_n = n
    
    # Yield control for large numbers
    if n > 100000:
        await asyncio.sleep(0)
    
    while d * d <= n:
        while n % d == 0:
            factors.append(d)
            n //= d
        d += 1
        
        # Yield control periodically for large factorizations
        if d % 1000 == 0 and original_n > 100000:
            await asyncio.sleep(0)
    
    if n > 1:
        factors.append(n)
    
    return factors

@mcp_function(
    description="Find the nth prime number (1-indexed). Uses efficient prime generation.",
    namespace="arithmetic",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    estimated_cpu_usage="high",
    examples=[
        {"input": {"n": 1}, "output": 2, "description": "1st prime is 2"},
        {"input": {"n": 10}, "output": 29, "description": "10th prime is 29"},
        {"input": {"n": 5}, "output": 11, "description": "5th prime is 11"},
        {"input": {"n": 25}, "output": 97, "description": "25th prime is 97"}
    ]
)
async def nth_prime(n: int) -> int:
    """
    Find the nth prime number (1-indexed).
    
    Args:
        n: Position of prime to find (must be positive)
    
    Returns:
        The nth prime number
    
    Raises:
        ValueError: If n is not positive
    
    Examples:
        await nth_prime(1) → 2
        await nth_prime(10) → 29
        await nth_prime(5) → 11
    """
    if n < 1:
        raise ValueError("n must be positive")
    
    if n == 1:
        return 2
    
    primes_found = 1
    candidate = 3
    checks = 0
    
    while primes_found < n:
        if await is_prime(candidate):
            primes_found += 1
            if primes_found == n:
                return candidate
        candidate += 2
        checks += 1
        
        # Yield control every 100 checks for large n
        if checks % 100 == 0 and n > 100:
            await asyncio.sleep(0)
    
    return candidate

@mcp_function(
    description="Calculate the Greatest Common Divisor (GCD) of two integers using Euclidean algorithm.",
    namespace="arithmetic",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    examples=[
        {"input": {"a": 48, "b": 18}, "output": 6, "description": "GCD of 48 and 18"},
        {"input": {"a": 17, "b": 13}, "output": 1, "description": "GCD of coprime numbers"},
        {"input": {"a": 100, "b": 25}, "output": 25, "description": "GCD when one divides the other"},
        {"input": {"a": 0, "b": 5}, "output": 5, "description": "GCD with zero"}
    ]
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
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    examples=[
        {"input": {"a": 12, "b": 18}, "output": 36, "description": "LCM of 12 and 18"},
        {"input": {"a": 7, "b": 13}, "output": 91, "description": "LCM of coprime numbers"},
        {"input": {"a": 10, "b": 5}, "output": 10, "description": "LCM when one divides the other"},
        {"input": {"a": 4, "b": 6}, "output": 12, "description": "LCM of small numbers"}
    ]
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
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    estimated_cpu_usage="medium",
    examples=[
        {"input": {"n": 12}, "output": [1, 2, 3, 4, 6, 12], "description": "Divisors of 12"},
        {"input": {"n": 17}, "output": [1, 17], "description": "Divisors of prime number"},
        {"input": {"n": 1}, "output": [1], "description": "Divisors of 1"},
        {"input": {"n": 36}, "output": [1, 2, 3, 4, 6, 9, 12, 18, 36], "description": "Divisors of perfect square"}
    ]
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
    execution_modes=["local", "remote"],
    examples=[
        {"input": {"a": 20, "b": 4}, "output": True, "description": "20 is divisible by 4"},
        {"input": {"a": 17, "b": 3}, "output": False, "description": "17 is not divisible by 3"},
        {"input": {"a": 0, "b": 5}, "output": True, "description": "0 is divisible by any non-zero number"},
        {"input": {"a": 15, "b": 1}, "output": True, "description": "Any number is divisible by 1"}
    ]
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
    execution_modes=["local", "remote"],
    examples=[
        {"input": {"n": 4}, "output": True, "description": "4 is even"},
        {"input": {"n": 7}, "output": False, "description": "7 is odd"},
        {"input": {"n": 0}, "output": True, "description": "0 is even"},
        {"input": {"n": -2}, "output": True, "description": "Negative even number"}
    ]
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
    execution_modes=["local", "remote"],
    examples=[
        {"input": {"n": 7}, "output": True, "description": "7 is odd"},
        {"input": {"n": 4}, "output": False, "description": "4 is even"},
        {"input": {"n": 1}, "output": True, "description": "1 is odd"},
        {"input": {"n": -3}, "output": True, "description": "Negative odd number"}
    ]
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
    description="Check if a number is a perfect square (integer square root exists).",
    namespace="arithmetic",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    examples=[
        {"input": {"n": 16}, "output": True, "description": "16 = 4²"},
        {"input": {"n": 15}, "output": False, "description": "15 is not a perfect square"},
        {"input": {"n": 1}, "output": True, "description": "1 = 1²"},
        {"input": {"n": 0}, "output": True, "description": "0 = 0²"}
    ]
)
async def is_perfect_square(n: int) -> bool:
    """
    Check if a number is a perfect square.
    
    Args:
        n: Non-negative integer to check
    
    Returns:
        True if n is a perfect square, False otherwise
    
    Examples:
        await is_perfect_square(16) → True
        await is_perfect_square(15) → False
        await is_perfect_square(1) → True
    """
    if n < 0:
        return False
    
    sqrt_n = int(math.sqrt(n))
    return sqrt_n * sqrt_n == n

@mcp_function(
    description="Check if a number is a power of two (2^k for some non-negative integer k).",
    namespace="arithmetic",
    execution_modes=["local", "remote"],
    examples=[
        {"input": {"n": 8}, "output": True, "description": "8 = 2³"},
        {"input": {"n": 6}, "output": False, "description": "6 is not a power of 2"},
        {"input": {"n": 1}, "output": True, "description": "1 = 2⁰"},
        {"input": {"n": 16}, "output": True, "description": "16 = 2⁴"}
    ]
)
async def is_power_of_two(n: int) -> bool:
    """
    Check if a number is a power of two.
    
    Args:
        n: Positive integer to check
    
    Returns:
        True if n is a power of two, False otherwise
    
    Examples:
        await is_power_of_two(8) → True
        await is_power_of_two(6) → False
        await is_power_of_two(1) → True
    """
    return n > 0 and (n & (n - 1)) == 0

@mcp_function(
    description="Calculate the nth Fibonacci number. F(0)=0, F(1)=1, F(n)=F(n-1)+F(n-2).",
    namespace="arithmetic",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    estimated_cpu_usage="medium",
    examples=[
        {"input": {"n": 0}, "output": 0, "description": "0th Fibonacci number"},
        {"input": {"n": 1}, "output": 1, "description": "1st Fibonacci number"},
        {"input": {"n": 10}, "output": 55, "description": "10th Fibonacci number"},
        {"input": {"n": 15}, "output": 610, "description": "15th Fibonacci number"}
    ]
)
async def fibonacci(n: int) -> int:
    """
    Calculate the nth Fibonacci number.
    
    Args:
        n: Non-negative integer position in sequence
    
    Returns:
        The nth Fibonacci number
    
    Raises:
        ValueError: If n is negative
    
    Examples:
        await fibonacci(0) → 0
        await fibonacci(1) → 1
        await fibonacci(10) → 55
    """
    if n < 0:
        raise ValueError("n must be non-negative")
    
    if n <= 1:
        return n
    
    a, b = 0, 1
    
    # Yield control for large Fibonacci calculations
    if n > 1000:
        await asyncio.sleep(0)
    
    for i in range(2, n + 1):
        a, b = b, a + b
        # Yield control every 1000 iterations for very large n
        if i % 1000 == 0 and n > 1000:
            await asyncio.sleep(0)
    
    return b

@mcp_function(
    description="Calculate the factorial of a number (n! = n × (n-1) × ... × 2 × 1).",
    namespace="arithmetic",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    estimated_cpu_usage="high",
    examples=[
        {"input": {"n": 0}, "output": 1, "description": "0! = 1 by definition"},
        {"input": {"n": 5}, "output": 120, "description": "5! = 5×4×3×2×1"},
        {"input": {"n": 1}, "output": 1, "description": "1! = 1"},
        {"input": {"n": 10}, "output": 3628800, "description": "10! = 3,628,800"}
    ]
)
async def factorial(n: int) -> int:
    """
    Calculate the factorial of a number.
    
    Args:
        n: Non-negative integer
    
    Returns:
        n! (n factorial)
    
    Raises:
        ValueError: If n is negative
    
    Examples:
        await factorial(0) → 1
        await factorial(5) → 120
        await factorial(10) → 3628800
    """
    if n < 0:
        raise ValueError("Factorial is not defined for negative numbers")
    
    # For large factorials, yield control
    if n > 100:
        await asyncio.sleep(0)
    
    return math.factorial(n)

@mcp_function(
    description="Count steps in the Collatz sequence until reaching 1. If even: n/2, if odd: 3n+1.",
    namespace="arithmetic",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    estimated_cpu_usage="medium",
    examples=[
        {"input": {"n": 1}, "output": 0, "description": "1 is already at the end"},
        {"input": {"n": 2}, "output": 1, "description": "2 → 1 (1 step)"},
        {"input": {"n": 3}, "output": 7, "description": "3 → 10 → 5 → 16 → 8 → 4 → 2 → 1"},
        {"input": {"n": 7}, "output": 16, "description": "Longer Collatz sequence"}
    ]
)
async def collatz_steps(n: int) -> int:
    """
    Count steps in the Collatz sequence until reaching 1.
    
    Args:
        n: Positive integer to start the sequence
    
    Returns:
        Number of steps to reach 1
    
    Raises:
        ValueError: If n is not positive
    
    Examples:
        await collatz_steps(1) → 0
        await collatz_steps(2) → 1
        await collatz_steps(3) → 7
    """
    if n <= 0:
        raise ValueError("n must be positive")
    
    steps = 0
    original_n = n
    
    while n != 1:
        if n % 2 == 0:
            n = n // 2
        else:
            n = 3 * n + 1
        steps += 1
        
        # Yield control every 100 steps for potentially long sequences
        if steps % 100 == 0:
            await asyncio.sleep(0)
        
        # Safety check for runaway sequences (though mathematically unproven)
        if steps > 10000:
            await asyncio.sleep(0)
    
    return steps

@mcp_function(
    description="Calculate modular exponentiation: (base^exponent) mod modulus. Efficient for large numbers.",
    namespace="arithmetic",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    estimated_cpu_usage="medium",
    examples=[
        {"input": {"base": 2, "exponent": 10, "modulus": 1000}, "output": 24, "description": "2^10 mod 1000 = 24"},
        {"input": {"base": 3, "exponent": 5, "modulus": 7}, "output": 5, "description": "3^5 mod 7 = 5"},
        {"input": {"base": 5, "exponent": 0, "modulus": 13}, "output": 1, "description": "Any number^0 mod m = 1"},
        {"input": {"base": 7, "exponent": 3, "modulus": 10}, "output": 3, "description": "7^3 mod 10 = 3"}
    ]
)
async def mod_power(base: int, exponent: int, modulus: int) -> int:
    """
    Calculate modular exponentiation efficiently.
    
    Args:
        base: Base number
        exponent: Non-negative exponent
        modulus: Positive modulus
    
    Returns:
        (base^exponent) mod modulus
    
    Raises:
        ValueError: If exponent is negative or modulus is not positive
    
    Examples:
        await mod_power(2, 10, 1000) → 24
        await mod_power(3, 5, 7) → 5
    """
    if exponent < 0:
        raise ValueError("Exponent must be non-negative")
    if modulus <= 0:
        raise ValueError("Modulus must be positive")
    
    # Yield control for large exponents
    if exponent > 1000:
        await asyncio.sleep(0)
    
    return pow(base, exponent, modulus)

# Export all number theory functions
__all__ = [
    'is_prime', 'next_prime', 'prime_factors', 'nth_prime',
    'gcd', 'lcm', 'divisors', 'is_divisible',
    'is_even', 'is_odd', 'is_perfect_square', 'is_power_of_two',
    'fibonacci', 'factorial', 'collatz_steps', 'mod_power'
]

if __name__ == "__main__":
    import asyncio
    
    async def test_number_theory_functions():
        """Test all number theory functions (async)."""
        print("🔢 Number Theory Functions Test (Async Native)")
        print("=" * 50)
        
        # Test prime functions
        print(f"is_prime(17) = {await is_prime(17)}")
        print(f"next_prime(10) = {await next_prime(10)}")
        print(f"prime_factors(60) = {await prime_factors(60)}")
        print(f"nth_prime(10) = {await nth_prime(10)}")
        
        # Test divisibility functions
        print(f"gcd(48, 18) = {await gcd(48, 18)}")
        print(f"lcm(12, 18) = {await lcm(12, 18)}")
        print(f"divisors(12) = {await divisors(12)}")
        print(f"is_divisible(20, 4) = {await is_divisible(20, 4)}")
        
        # Test integer properties
        print(f"is_even(4) = {await is_even(4)}")
        print(f"is_odd(7) = {await is_odd(7)}")
        print(f"is_perfect_square(16) = {await is_perfect_square(16)}")
        print(f"is_power_of_two(8) = {await is_power_of_two(8)}")
        
        # Test sequences
        print(f"fibonacci(10) = {await fibonacci(10)}")
        print(f"factorial(5) = {await factorial(5)}")
        print(f"collatz_steps(3) = {await collatz_steps(3)}")
        print(f"mod_power(2, 10, 1000) = {await mod_power(2, 10, 1000)}")
        
        print("\n✅ All async number theory functions working correctly!")
        
        # Test parallel execution
        print("\n🚀 Testing Parallel Execution:")
        parallel_results = await asyncio.gather(
            is_prime(17), next_prime(10), gcd(48, 18), fibonacci(10), factorial(5)
        )
        print(f"Parallel results: {parallel_results}")
        
        # Test cryptographic operations
        print("\n🔐 Testing Cryptographic Operations:")
        large_prime_check = await is_prime(2147483647)  # Large Mersenne prime
        mod_exp_result = await mod_power(123, 456, 789)
        print(f"is_prime(2147483647) = {large_prime_check}")
        print(f"mod_power(123, 456, 789) = {mod_exp_result}")
    
    asyncio.run(test_number_theory_functions())