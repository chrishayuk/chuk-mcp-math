#!/usr/bin/env python3
# chuk_mcp_functions/math/arithmetic/number_theory/special_numbers.py
"""
Special Prime Numbers and Arithmetic Functions - Async Native

Functions for working with special types of prime numbers and arithmetic functions.
Includes Mersenne primes, Fermat primes, Sophie Germain primes, twin primes,
Wilson's theorem, Carmichael numbers, and various arithmetic functions.

Functions:
- Mersenne primes: is_mersenne_prime, mersenne_prime_exponents, lucas_lehmer_test
- Fermat primes: is_fermat_prime, fermat_numbers, known_fermat_primes
- Sophie Germain & Safe primes: is_sophie_germain_prime, is_safe_prime, safe_prime_pairs
- Twin primes: is_twin_prime, twin_prime_pairs, cousin_primes, sexy_primes
- Wilson's theorem: wilson_theorem_test, wilson_factorial_mod
- Pseudoprimes: is_carmichael_number, is_fermat_pseudoprime, fermat_test
- Arithmetic functions: euler_totient, mobius_function, omega_functions
- Prime gaps: prime_gap, largest_prime_gap, twin_prime_gaps
"""

import math
import asyncio
from typing import List, Tuple, Optional, Dict, Any
from chuk_mcp_functions.mcp_decorator import mcp_function

# Import from other modules
from .primes import is_prime, next_prime, nth_prime
from .divisibility import gcd

# ============================================================================
# BASIC SPECIAL NUMBERS
# ============================================================================

@mcp_function(
    description="Check if a number is a perfect square.",
    namespace="arithmetic",
    category="special_numbers",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    examples=[
        {"input": {"n": 16}, "output": True, "description": "16 = 4² is a perfect square"},
        {"input": {"n": 25}, "output": True, "description": "25 = 5² is a perfect square"},
        {"input": {"n": 15}, "output": False, "description": "15 is not a perfect square"},
        {"input": {"n": 0}, "output": True, "description": "0 = 0² is a perfect square"}
    ]
)
async def is_perfect_square(n: int) -> bool:
    """
    Check if a number is a perfect square.
    
    A perfect square is an integer that is the square of another integer.
    
    Args:
        n: Non-negative integer to check
    
    Returns:
        True if n is a perfect square, False otherwise
    
    Examples:
        await is_perfect_square(16) → True   # 16 = 4²
        await is_perfect_square(25) → True   # 25 = 5²
        await is_perfect_square(15) → False  # No integer squared equals 15
        await is_perfect_square(0) → True    # 0 = 0²
    """
    if n < 0:
        return False
    
    if n == 0:
        return True
    
    # Use integer square root to check
    sqrt_n = int(math.sqrt(n))
    return sqrt_n * sqrt_n == n

@mcp_function(
    description="Check if a number is a power of two.",
    namespace="arithmetic", 
    category="special_numbers",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    examples=[
        {"input": {"n": 8}, "output": True, "description": "8 = 2³ is a power of two"},
        {"input": {"n": 16}, "output": True, "description": "16 = 2⁴ is a power of two"},
        {"input": {"n": 15}, "output": False, "description": "15 is not a power of two"},
        {"input": {"n": 1}, "output": True, "description": "1 = 2⁰ is a power of two"}
    ]
)
async def is_power_of_two(n: int) -> bool:
    """
    Check if a number is a power of two.
    
    A power of two is a number of the form 2^k where k ≥ 0.
    
    Args:
        n: Positive integer to check
    
    Returns:
        True if n is a power of two, False otherwise
    
    Examples:
        await is_power_of_two(8) → True    # 8 = 2³
        await is_power_of_two(16) → True   # 16 = 2⁴
        await is_power_of_two(15) → False  # Not a power of two
        await is_power_of_two(1) → True    # 1 = 2⁰
    """
    if n <= 0:
        return False
    
    # Efficient bit manipulation: n is power of 2 iff n & (n-1) == 0
    return (n & (n - 1)) == 0

@mcp_function(
    description="Calculate the nth Fibonacci number using efficient matrix exponentiation.",
    namespace="arithmetic",
    category="special_numbers", 
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    estimated_cpu_usage="medium",
    examples=[
        {"input": {"n": 10}, "output": 55, "description": "10th Fibonacci number"},
        {"input": {"n": 0}, "output": 0, "description": "0th Fibonacci number"},
        {"input": {"n": 1}, "output": 1, "description": "1st Fibonacci number"},
        {"input": {"n": 20}, "output": 6765, "description": "20th Fibonacci number"}
    ]
)
async def fibonacci(n: int) -> int:
    """
    Calculate the nth Fibonacci number.
    
    Uses efficient matrix exponentiation for large n.
    Fibonacci sequence: 0, 1, 1, 2, 3, 5, 8, 13, 21, 34, ...
    
    Args:
        n: Non-negative integer (position in sequence)
    
    Returns:
        The nth Fibonacci number
    
    Raises:
        ValueError: If n is negative
    
    Examples:
        await fibonacci(0) → 0     # F₀ = 0
        await fibonacci(1) → 1     # F₁ = 1
        await fibonacci(10) → 55   # F₁₀ = 55
        await fibonacci(20) → 6765 # F₂₀ = 6765
    """
    if n < 0:
        raise ValueError("Fibonacci number position must be non-negative")
    
    if n <= 1:
        return n
    
    # For small n, use simple iteration
    if n <= 100:
        a, b = 0, 1
        for i in range(2, n + 1):
            a, b = b, a + b
            # Yield control every 10 iterations
            if i % 10 == 0:
                await asyncio.sleep(0)
        return b
    
    # For large n, use matrix exponentiation: [[1,1],[1,0]]^n
    def matrix_mult(A, B):
        """Multiply two 2x2 matrices."""
        return [
            [A[0][0] * B[0][0] + A[0][1] * B[1][0], A[0][0] * B[0][1] + A[0][1] * B[1][1]],
            [A[1][0] * B[0][0] + A[1][1] * B[1][0], A[1][0] * B[0][1] + A[1][1] * B[1][1]]
        ]
    
    async def matrix_power(matrix, power):
        """Calculate matrix^power using fast exponentiation."""
        if power == 1:
            return matrix
        
        result = [[1, 0], [0, 1]]  # Identity matrix
        base = matrix
        exp = power
        iterations = 0
        
        while exp > 0:
            if exp % 2 == 1:
                result = matrix_mult(result, base)
            base = matrix_mult(base, base)
            exp //= 2
            iterations += 1
            
            # Yield control every few iterations for very large n
            if iterations % 10 == 0:
                await asyncio.sleep(0)
        
        return result
    
    # [[1,1],[1,0]]^(n-1) gives [[F_n, F_{n-1}], [F_{n-1}, F_{n-2}]]
    fib_matrix = [[1, 1], [1, 0]]
    result_matrix = await matrix_power(fib_matrix, n)
    return result_matrix[0][1]

@mcp_function(
    description="Calculate factorial n! = n × (n-1) × ... × 2 × 1.",
    namespace="arithmetic",
    category="special_numbers",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    estimated_cpu_usage="medium",
    examples=[
        {"input": {"n": 5}, "output": 120, "description": "5! = 5×4×3×2×1 = 120"},
        {"input": {"n": 0}, "output": 1, "description": "0! = 1 by definition"},
        {"input": {"n": 10}, "output": 3628800, "description": "10! = 3,628,800"},
        {"input": {"n": 1}, "output": 1, "description": "1! = 1"}
    ]
)
async def factorial(n: int) -> int:
    """
    Calculate the factorial of n.
    
    n! = n × (n-1) × (n-2) × ... × 2 × 1
    By definition: 0! = 1
    
    Args:
        n: Non-negative integer
    
    Returns:
        n! (factorial of n)
    
    Raises:
        ValueError: If n is negative
    
    Examples:
        await factorial(0) → 1        # 0! = 1
        await factorial(5) → 120      # 5! = 120
        await factorial(10) → 3628800 # 10! = 3,628,800
    """
    if n < 0:
        raise ValueError("Factorial is not defined for negative numbers")
    
    if n <= 1:
        return 1
    
    result = 1
    for i in range(2, n + 1):
        result *= i
        
        # Yield control every 1000 iterations for large factorials
        if i % 1000 == 0 and n > 1000:
            await asyncio.sleep(0)
    
    return result

@mcp_function(
    description="Generate the first n Fibonacci numbers.",
    namespace="arithmetic",
    category="special_numbers",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    estimated_cpu_usage="medium",
    examples=[
        {"input": {"n": 10}, "output": [0, 1, 1, 2, 3, 5, 8, 13, 21, 34], "description": "First 10 Fibonacci numbers"},
        {"input": {"n": 5}, "output": [0, 1, 1, 2, 3], "description": "First 5 Fibonacci numbers"},
        {"input": {"n": 0}, "output": [], "description": "No Fibonacci numbers"}
    ]
)
async def fibonacci_sequence(n: int) -> List[int]:
    """
    Generate the first n Fibonacci numbers.
    
    Args:
        n: Number of Fibonacci numbers to generate (non-negative)
    
    Returns:
        List of the first n Fibonacci numbers
    
    Examples:
        await fibonacci_sequence(10) → [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
        await fibonacci_sequence(5) → [0, 1, 1, 2, 3]
        await fibonacci_sequence(0) → []
    """
    if n <= 0:
        return []
    
    if n == 1:
        return [0]
    
    if n == 2:
        return [0, 1]
    
    result = [0, 1]
    
    for i in range(2, n):
        result.append(result[i-1] + result[i-2])
        
        # Yield control every 1000 iterations for large sequences
        if i % 1000 == 0 and n > 1000:
            await asyncio.sleep(0)
    
    return result

@mcp_function(
    description="Check if a number is a Fibonacci number.",
    namespace="arithmetic",
    category="special_numbers",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    examples=[
        {"input": {"n": 13}, "output": True, "description": "13 is the 7th Fibonacci number"},
        {"input": {"n": 55}, "output": True, "description": "55 is the 10th Fibonacci number"},
        {"input": {"n": 12}, "output": False, "description": "12 is not a Fibonacci number"},
        {"input": {"n": 0}, "output": True, "description": "0 is the 0th Fibonacci number"}
    ]
)
async def is_fibonacci_number(n: int) -> bool:
    """
    Check if a number is a Fibonacci number.
    
    Uses the mathematical property: a positive integer n is a Fibonacci number
    if and only if one of (5n²+4) or (5n²-4) is a perfect square.
    
    Args:
        n: Non-negative integer to check
    
    Returns:
        True if n is a Fibonacci number, False otherwise
    
    Examples:
        await is_fibonacci_number(13) → True   # F₇ = 13
        await is_fibonacci_number(55) → True   # F₁₀ = 55
        await is_fibonacci_number(12) → False  # Not in sequence
        await is_fibonacci_number(0) → True    # F₀ = 0
    """
    if n < 0:
        return False
    
    if n == 0:
        return True
    
    # Use the property: n is Fibonacci iff (5n²+4) or (5n²-4) is a perfect square
    def is_perfect_square_helper(x):
        if x < 0:
            return False
        sqrt_x = int(math.sqrt(x))
        return sqrt_x * sqrt_x == x
    
    n_squared = n * n
    return (is_perfect_square_helper(5 * n_squared + 4) or 
            is_perfect_square_helper(5 * n_squared - 4))

# ============================================================================
# MERSENNE PRIMES
# ============================================================================

@mcp_function(
    description="Check if a number is a Mersenne prime (prime of form 2^p - 1 where p is prime).",
    namespace="arithmetic",
    category="special_primes",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    estimated_cpu_usage="medium",
    examples=[
        {"input": {"n": 31}, "output": True, "description": "31 = 2^5 - 1 is Mersenne prime"},
        {"input": {"n": 127}, "output": True, "description": "127 = 2^7 - 1 is Mersenne prime"},
        {"input": {"n": 15}, "output": False, "description": "15 = 2^4 - 1 is not prime"},
        {"input": {"n": 17}, "output": False, "description": "17 is prime but not of Mersenne form"}
    ]
)
async def is_mersenne_prime(n: int) -> bool:
    """
    Check if a number is a Mersenne prime.
    
    A Mersenne prime is a prime number of the form 2^p - 1 where p is also prime.
    
    Args:
        n: Number to check
    
    Returns:
        True if n is a Mersenne prime, False otherwise
    
    Examples:
        await is_mersenne_prime(31) → True   # 2^5 - 1
        await is_mersenne_prime(127) → True  # 2^7 - 1
        await is_mersenne_prime(15) → False  # 2^4 - 1, not prime
    """
    if n <= 1:
        return False
    
    # Check if n is prime first
    if not await is_prime(n):
        return False
    
    # Check if n is of the form 2^p - 1
    temp = n + 1
    if temp & (temp - 1) != 0:  # Check if temp is power of 2
        return False
    
    # Find the exponent p
    p = temp.bit_length() - 1
    
    # Check if p is prime
    return await is_prime(p)

@mcp_function(
    description="Get known Mersenne prime exponents up to a limit.",
    namespace="arithmetic", 
    category="special_primes",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    examples=[
        {"input": {"limit": 20}, "output": [2, 3, 5, 7, 13, 17, 19], "description": "Mersenne exponents ≤ 20"},
        {"input": {"limit": 10}, "output": [2, 3, 5, 7], "description": "Mersenne exponents ≤ 10"}
    ]
)
async def mersenne_prime_exponents(limit: int) -> List[int]:
    """
    Get known Mersenne prime exponents up to a limit.
    
    Returns exponents p such that 2^p - 1 is prime, for p ≤ limit.
    
    Args:
        limit: Maximum exponent to check
    
    Returns:
        List of Mersenne prime exponents
    
    Examples:
        await mersenne_prime_exponents(20) → [2, 3, 5, 7, 13, 17, 19]
        await mersenne_prime_exponents(10) → [2, 3, 5, 7]
    """
    # Known Mersenne prime exponents (first 50)
    known_exponents = [
        2, 3, 5, 7, 13, 17, 19, 31, 61, 89, 107, 127, 521, 607, 1279, 2203, 2281,
        3217, 4253, 4423, 9689, 9941, 11213, 19937, 21701, 23209, 44497, 86243,
        110503, 132049, 216091, 756839, 859433, 1257787, 1398269, 2976221, 3021377,
        6972593, 13466917, 20996011, 24036583, 25964951, 30402457, 32582657,
        37156667, 42643801, 43112609, 57885161, 74207281, 77232917
    ]
    
    return [p for p in known_exponents if p <= limit]

@mcp_function(
    description="Perform Lucas-Lehmer primality test for Mersenne numbers.",
    namespace="arithmetic",
    category="special_primes", 
    execution_modes=["local", "remote"],
    estimated_cpu_usage="high",
    examples=[
        {"input": {"p": 5}, "output": True, "description": "2^5 - 1 = 31 is prime"},
        {"input": {"p": 7}, "output": True, "description": "2^7 - 1 = 127 is prime"},
        {"input": {"p": 11}, "output": False, "description": "2^11 - 1 = 2047 is composite"}
    ]
)
async def lucas_lehmer_test(p: int) -> bool:
    """
    Lucas-Lehmer primality test for Mersenne numbers 2^p - 1.
    
    This is the most efficient test for Mersenne number primality.
    
    Args:
        p: Exponent (must be odd prime > 2)
    
    Returns:
        True if 2^p - 1 is prime, False otherwise
    
    Examples:
        await lucas_lehmer_test(5) → True   # 2^5 - 1 = 31 is prime
        await lucas_lehmer_test(7) → True   # 2^7 - 1 = 127 is prime
        await lucas_lehmer_test(11) → False # 2^11 - 1 = 2047 is composite
    """
    if p == 2:
        return True  # 2^2 - 1 = 3 is prime
    
    if not await is_prime(p) or p <= 2:
        return False
    
    # Lucas-Lehmer sequence: s_0 = 4, s_{i+1} = s_i^2 - 2
    s = 4
    mersenne = (1 << p) - 1  # 2^p - 1
    
    # Yield control for large computations
    if p > 1000:
        await asyncio.sleep(0)
    
    for i in range(p - 2):
        s = (s * s - 2) % mersenne
        
        # Yield control every 1000 iterations for very large p
        if i % 1000 == 0 and p > 10000:
            await asyncio.sleep(0)
    
    return s == 0

# ============================================================================
# FERMAT PRIMES
# ============================================================================

@mcp_function(
    description="Check if a number is a Fermat prime (prime of form 2^(2^n) + 1).",
    namespace="arithmetic",
    category="special_primes",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    examples=[
        {"input": {"n": 3}, "output": True, "description": "3 = 2^(2^0) + 1 is Fermat prime"},
        {"input": {"n": 5}, "output": True, "description": "5 = 2^(2^1) + 1 is Fermat prime"},
        {"input": {"n": 17}, "output": True, "description": "17 = 2^(2^2) + 1 is Fermat prime"},
        {"input": {"n": 7}, "output": False, "description": "7 is prime but not Fermat form"}
    ]
)
async def is_fermat_prime(n: int) -> bool:
    """
    Check if a number is a Fermat prime.
    
    A Fermat prime is a prime number of the form 2^(2^k) + 1 for some k ≥ 0.
    
    Args:
        n: Number to check
    
    Returns:
        True if n is a Fermat prime, False otherwise
    
    Examples:
        await is_fermat_prime(3) → True    # F_0 = 2^(2^0) + 1 = 3
        await is_fermat_prime(5) → True    # F_1 = 2^(2^1) + 1 = 5
        await is_fermat_prime(17) → True   # F_2 = 2^(2^2) + 1 = 17
    """
    if n <= 2:
        return False
    
    # Check if n is prime first
    if not await is_prime(n):
        return False
    
    # Check if n is of the form 2^(2^k) + 1
    temp = n - 1
    
    # temp should be a power of 2
    if temp & (temp - 1) != 0:
        return False
    
    # The exponent should also be a power of 2
    exponent = temp.bit_length() - 1
    return exponent & (exponent - 1) == 0

@mcp_function(
    description="Generate Fermat numbers F_n = 2^(2^n) + 1 up to index limit.",
    namespace="arithmetic",
    category="special_primes",
    execution_modes=["local", "remote"],
    examples=[
        {"input": {"limit": 4}, "output": [3, 5, 17, 257, 65537], "description": "First 5 Fermat numbers"},
        {"input": {"limit": 2}, "output": [3, 5, 17], "description": "First 3 Fermat numbers"}
    ]
)
async def fermat_numbers(limit: int) -> List[int]:
    """
    Generate Fermat numbers F_n = 2^(2^n) + 1.
    
    Args:
        limit: Maximum index n to generate
    
    Returns:
        List of Fermat numbers F_0 through F_limit
    
    Examples:
        await fermat_numbers(4) → [3, 5, 17, 257, 65537]
        await fermat_numbers(2) → [3, 5, 17]
    """
    if limit < 0:
        return []
    
    result = []
    for n in range(limit + 1):
        if n > 10:  # Avoid computing extremely large numbers
            break
        fermat_n = (1 << (1 << n)) + 1  # 2^(2^n) + 1
        result.append(fermat_n)
        
        # Yield control for large computations
        if n > 5:
            await asyncio.sleep(0)
    
    return result

@mcp_function(
    description="Get the five known Fermat primes.",
    namespace="arithmetic",
    category="special_primes",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    examples=[
        {"input": {}, "output": [3, 5, 17, 257, 65537], "description": "The 5 known Fermat primes"}
    ]
)
async def known_fermat_primes() -> List[int]:
    """
    Get the five known Fermat primes.
    
    Returns:
        List of the five known Fermat primes: F_0, F_1, F_2, F_3, F_4
    
    Examples:
        await known_fermat_primes() → [3, 5, 17, 257, 65537]
    """
    return [3, 5, 17, 257, 65537]

# ============================================================================
# SOPHIE GERMAIN AND SAFE PRIMES
# ============================================================================

@mcp_function(
    description="Check if a prime p is a Sophie Germain prime (2p + 1 is also prime).",
    namespace="arithmetic",
    category="special_primes",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    examples=[
        {"input": {"p": 11}, "output": True, "description": "11 is Sophie Germain: 2×11+1=23 is prime"},
        {"input": {"p": 23}, "output": True, "description": "23 is Sophie Germain: 2×23+1=47 is prime"},
        {"input": {"p": 13}, "output": False, "description": "13 is not Sophie Germain: 2×13+1=27 is composite"}
    ]
)
async def is_sophie_germain_prime(p: int) -> bool:
    """
    Check if a prime p is a Sophie Germain prime.
    
    A Sophie Germain prime is a prime p such that 2p + 1 is also prime.
    
    Args:
        p: Number to check
    
    Returns:
        True if p is a Sophie Germain prime, False otherwise
    
    Examples:
        await is_sophie_germain_prime(11) → True  # 2×11+1 = 23 is prime
        await is_sophie_germain_prime(23) → True  # 2×23+1 = 47 is prime
        await is_sophie_germain_prime(13) → False # 2×13+1 = 27 is composite
    """
    if not await is_prime(p):
        return False
    
    safe_prime = 2 * p + 1
    return await is_prime(safe_prime)

@mcp_function(
    description="Check if a prime q is a safe prime (q = 2p + 1 where p is prime).",
    namespace="arithmetic",
    category="special_primes",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    examples=[
        {"input": {"q": 23}, "output": True, "description": "23 = 2×11+1 where 11 is prime"},
        {"input": {"q": 47}, "output": True, "description": "47 = 2×23+1 where 23 is prime"},
        {"input": {"q": 29}, "output": False, "description": "29 = 2×14+1 where 14 is not prime"}
    ]
)
async def is_safe_prime(q: int) -> bool:
    """
    Check if a prime q is a safe prime.
    
    A safe prime is a prime q such that (q-1)/2 is also prime.
    
    Args:
        q: Number to check
    
    Returns:
        True if q is a safe prime, False otherwise
    
    Examples:
        await is_safe_prime(23) → True   # (23-1)/2 = 11 is prime
        await is_safe_prime(47) → True   # (47-1)/2 = 23 is prime
        await is_safe_prime(29) → False  # (29-1)/2 = 14 is not prime
    """
    if not await is_prime(q):
        return False
    
    if q == 2 or q == 3:
        return False  # Special cases
    
    if (q - 1) % 2 != 0:
        return False  # q-1 must be even
    
    sophie_germain = (q - 1) // 2
    return await is_prime(sophie_germain)

@mcp_function(
    description="Find Sophie Germain and safe prime pairs up to limit.",
    namespace="arithmetic",
    category="special_primes",
    execution_modes=["local", "remote"],
    estimated_cpu_usage="medium",
    examples=[
        {"input": {"limit": 50}, "output": [(2, 5), (3, 7), (5, 11), (11, 23), (23, 47)], "description": "Sophie Germain pairs ≤ 50"},
        {"input": {"limit": 25}, "output": [(2, 5), (3, 7), (5, 11), (11, 23)], "description": "Sophie Germain pairs ≤ 25"}
    ]
)
async def safe_prime_pairs(limit: int) -> List[Tuple[int, int]]:
    """
    Find Sophie Germain and safe prime pairs up to limit.
    
    Returns pairs (p, q) where p is Sophie Germain prime and q = 2p + 1 is safe prime.
    
    Args:
        limit: Upper limit for Sophie Germain primes
    
    Returns:
        List of (sophie_germain_prime, safe_prime) pairs
    
    Examples:
        await safe_prime_pairs(50) → [(2, 5), (3, 7), (5, 11), (11, 23), (23, 47)]
    """
    pairs = []
    checks = 0
    
    for p in range(2, limit + 1):
        if await is_prime(p):
            safe = 2 * p + 1
            if await is_prime(safe):
                pairs.append((p, safe))
        
        checks += 1
        # Yield control every 1000 checks for large limits
        if checks % 1000 == 0 and limit > 1000:
            await asyncio.sleep(0)
    
    return pairs

# ============================================================================
# TWIN PRIMES AND RELATED
# ============================================================================

@mcp_function(
    description="Check if a number is part of a twin prime pair (p, p+2).",
    namespace="arithmetic",
    category="special_primes",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    examples=[
        {"input": {"p": 3}, "output": True, "description": "3 and 5 are twin primes"},
        {"input": {"p": 5}, "output": True, "description": "3 and 5 are twin primes"},
        {"input": {"p": 13}, "output": True, "description": "11 and 13 are twin primes"},
        {"input": {"p": 7}, "output": False, "description": "7 is not part of twin prime pair"}
    ]
)
async def is_twin_prime(p: int) -> bool:
    """
    Check if a number is part of a twin prime pair.
    
    Twin primes are pairs of primes (p, p+2) or (p-2, p).
    
    Args:
        p: Number to check
    
    Returns:
        True if p is part of a twin prime pair, False otherwise
    
    Examples:
        await is_twin_prime(3) → True   # (3, 5) are twin primes
        await is_twin_prime(13) → True  # (11, 13) are twin primes
        await is_twin_prime(7) → False  # Neither (5, 7) nor (7, 9) are both prime
    """
    if not await is_prime(p):
        return False
    
    # Check if p+2 is prime or p-2 is prime
    plus_two_prime = await is_prime(p + 2)
    minus_two_prime = p > 2 and await is_prime(p - 2)
    
    return plus_two_prime or minus_two_prime

@mcp_function(
    description="Find twin prime pairs up to limit.",
    namespace="arithmetic",
    category="special_primes",
    execution_modes=["local", "remote"],
    estimated_cpu_usage="medium",
    examples=[
        {"input": {"limit": 50}, "output": [(3, 5), (5, 7), (11, 13), (17, 19), (29, 31), (41, 43)], "description": "Twin primes ≤ 50"},
        {"input": {"limit": 20}, "output": [(3, 5), (5, 7), (11, 13), (17, 19)], "description": "Twin primes ≤ 20"}
    ]
)
async def twin_prime_pairs(limit: int) -> List[Tuple[int, int]]:
    """
    Find twin prime pairs up to limit.
    
    Args:
        limit: Upper limit for the smaller twin prime
    
    Returns:
        List of (p, p+2) twin prime pairs
    
    Examples:
        await twin_prime_pairs(50) → [(3, 5), (5, 7), (11, 13), (17, 19), (29, 31), (41, 43)]
    """
    pairs = []
    checks = 0
    
    for p in range(3, limit + 1, 2):  # Only check odd numbers (except 2)
        if await is_prime(p) and await is_prime(p + 2):
            pairs.append((p, p + 2))
        
        checks += 1
        # Yield control every 1000 checks for large limits
        if checks % 1000 == 0 and limit > 1000:
            await asyncio.sleep(0)
    
    return pairs

@mcp_function(
    description="Find cousin prime pairs (p, p+4) up to limit.",
    namespace="arithmetic",
    category="special_primes",
    execution_modes=["local", "remote"],
    estimated_cpu_usage="medium",
    examples=[
        {"input": {"limit": 50}, "output": [(3, 7), (7, 11), (13, 17), (19, 23), (37, 41), (43, 47)], "description": "Cousin primes ≤ 50"},
        {"input": {"limit": 20}, "output": [(3, 7), (7, 11), (13, 17), (19, 23)], "description": "Cousin primes ≤ 20"}
    ]
)
async def cousin_primes(limit: int) -> List[Tuple[int, int]]:
    """
    Find cousin prime pairs (primes that differ by 4).
    
    Args:
        limit: Upper limit for the smaller cousin prime
    
    Returns:
        List of (p, p+4) cousin prime pairs
    
    Examples:
        await cousin_primes(50) → [(3, 7), (7, 11), (13, 17), (19, 23), (37, 41), (43, 47)]
    """
    pairs = []
    checks = 0
    
    for p in range(3, limit + 1):
        if await is_prime(p) and await is_prime(p + 4):
            pairs.append((p, p + 4))
        
        checks += 1
        # Yield control every 1000 checks for large limits
        if checks % 1000 == 0 and limit > 1000:
            await asyncio.sleep(0)
    
    return pairs

@mcp_function(
    description="Find sexy prime pairs (p, p+6) up to limit.",
    namespace="arithmetic",
    category="special_primes",
    execution_modes=["local", "remote"],
    estimated_cpu_usage="medium",
    examples=[
        {"input": {"limit": 50}, "output": [(5, 11), (7, 13), (13, 19), (17, 23), (31, 37), (37, 43), (41, 47)], "description": "Sexy primes ≤ 50"},
        {"input": {"limit": 25}, "output": [(5, 11), (7, 13), (13, 19), (17, 23)], "description": "Sexy primes ≤ 25"}
    ]
)
async def sexy_primes(limit: int) -> List[Tuple[int, int]]:
    """
    Find sexy prime pairs (primes that differ by 6).
    
    Args:
        limit: Upper limit for the smaller sexy prime
    
    Returns:
        List of (p, p+6) sexy prime pairs
    
    Examples:
        await sexy_primes(50) → [(5, 11), (7, 13), (13, 19), (17, 23), (31, 37), (37, 43), (41, 47)]
    """
    pairs = []
    checks = 0
    
    for p in range(5, limit + 1):  # Start from 5 since smaller primes don't work
        if await is_prime(p) and await is_prime(p + 6):
            pairs.append((p, p + 6))
        
        checks += 1
        # Yield control every 1000 checks for large limits
        if checks % 1000 == 0 and limit > 1000:
            await asyncio.sleep(0)
    
    return pairs

# ============================================================================
# WILSON'S THEOREM
# ============================================================================

@mcp_function(
    description="Check Wilson's theorem: p is prime iff (p-1)! ≡ -1 (mod p).",
    namespace="arithmetic",
    category="primality_tests",
    execution_modes=["local", "remote"],
    estimated_cpu_usage="high",
    examples=[
        {"input": {"n": 7}, "output": True, "description": "7 is prime: 6! ≡ -1 (mod 7)"},
        {"input": {"n": 11}, "output": True, "description": "11 is prime: 10! ≡ -1 (mod 11)"},
        {"input": {"n": 8}, "output": False, "description": "8 is composite: 7! ≢ -1 (mod 8)"}
    ]
)
async def wilson_theorem_check(n: int) -> bool:
    """
    Check Wilson's theorem for primality.
    
    Wilson's theorem: p is prime if and only if (p-1)! ≡ -1 (mod p).
    
    Args:
        n: Number to check
    
    Returns:
        True if n satisfies Wilson's theorem, False otherwise
    
    Examples:
        await wilson_theorem_check(7) → True   # 6! ≡ -1 (mod 7)
        await wilson_theorem_check(11) → True  # 10! ≡ -1 (mod 11)
        await wilson_theorem_check(8) → False  # 8 is composite
    """
    if n <= 1:
        return False
    if n == 2:
        return True
    
    # Calculate (n-1)! mod n
    factorial_mod = await wilson_factorial_mod(n - 1, n)
    
    # Check if (n-1)! ≡ -1 (mod n), i.e., ≡ n-1 (mod n)
    return factorial_mod == n - 1

@mcp_function(
    description="Calculate k! mod m efficiently for Wilson's theorem.",
    namespace="arithmetic",
    category="primality_tests",
    execution_modes=["local", "remote"],
    estimated_cpu_usage="medium",
    examples=[
        {"input": {"k": 6, "m": 7}, "output": 6, "description": "6! mod 7 = 720 mod 7 = 6"},
        {"input": {"k": 10, "m": 11}, "output": 10, "description": "10! mod 11 = 10"},
        {"input": {"k": 4, "m": 5}, "output": 4, "description": "4! mod 5 = 24 mod 5 = 4"}
    ]
)
async def wilson_factorial_mod(k: int, m: int) -> int:
    """
    Calculate k! mod m efficiently.
    
    Args:
        k: Factorial number
        m: Modulus
    
    Returns:
        k! mod m
    
    Examples:
        await wilson_factorial_mod(6, 7) → 6   # 6! mod 7
        await wilson_factorial_mod(10, 11) → 10 # 10! mod 11
    """
    if k >= m:
        return 0  # k! is divisible by m
    
    result = 1
    for i in range(1, k + 1):
        result = (result * i) % m
        
        # Yield control every 1000 iterations for large k
        if i % 1000 == 0 and k > 1000:
            await asyncio.sleep(0)
    
    return result

# ============================================================================
# PSEUDOPRIMES AND CARMICHAEL NUMBERS  
# ============================================================================

@mcp_function(
    description="Check if n is a Fermat pseudoprime to base a.",
    namespace="arithmetic",
    category="pseudoprimes",
    execution_modes=["local", "remote"],
    estimated_cpu_usage="medium",
    examples=[
        {"input": {"n": 341, "a": 2}, "output": True, "description": "341 is pseudoprime base 2"},
        {"input": {"n": 341, "a": 3}, "output": False, "description": "341 fails base 3 check"},
        {"input": {"n": 561, "a": 2}, "output": True, "description": "561 is Carmichael number"}
    ]
)
async def is_fermat_pseudoprime(n: int, a: int) -> bool:
    """
    Check if n is a Fermat pseudoprime to base a.
    
    A composite number n is a Fermat pseudoprime to base a if:
    gcd(a, n) = 1 and a^(n-1) ≡ 1 (mod n).
    
    Args:
        n: Number to check
        a: Base for the check
    
    Returns:
        True if n is a Fermat pseudoprime to base a, False otherwise
    
    Examples:
        await is_fermat_pseudoprime(341, 2) → True   # 341 is base-2 pseudoprime
        await is_fermat_pseudoprime(341, 3) → False  # 341 fails base-3 check
    """
    if n <= 1 or await is_prime(n):
        return False
    
    if await gcd(a, n) != 1:
        return False  # a and n must be coprime
    
    # Check if a^(n-1) ≡ 1 (mod n)
    return pow(a, n - 1, n) == 1

@mcp_function(
    description="Perform Fermat primality check for base a.",
    namespace="arithmetic",
    category="pseudoprimes",
    execution_modes=["local", "remote"],
    estimated_cpu_usage="medium",
    examples=[
        {"input": {"n": 17, "a": 2}, "output": True, "description": "17 passes Fermat check base 2"},
        {"input": {"n": 15, "a": 2}, "output": False, "description": "15 fails Fermat check base 2"},
        {"input": {"n": 341, "a": 2}, "output": True, "description": "341 falsely passes (pseudoprime)"}
    ]
)
async def fermat_primality_check(n: int, a: int) -> bool:
    """
    Perform Fermat primality check.
    
    Checks if a^(n-1) ≡ 1 (mod n) where gcd(a, n) = 1.
    If this fails, n is definitely composite.
    If this passes, n is probably prime (or a pseudoprime).
    
    Args:
        n: Number to check
        a: Base for the check
    
    Returns:
        True if n passes the check, False if n definitely composite
    
    Examples:
        await fermat_primality_check(17, 2) → True   # 17 is prime
        await fermat_primality_check(15, 2) → False  # 15 is composite
        await fermat_primality_check(341, 2) → True  # 341 is pseudoprime (false positive)
    """
    if n <= 1:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    
    if await gcd(a, n) != 1:
        return False  # Check requires gcd(a, n) = 1
    
    return pow(a, n - 1, n) == 1

@mcp_function(
    description="Check if n is a Carmichael number (absolute Fermat pseudoprime).",
    namespace="arithmetic",
    category="pseudoprimes",
    execution_modes=["local", "remote"],
    estimated_cpu_usage="high",
    examples=[
        {"input": {"n": 561}, "output": True, "description": "561 is smallest Carmichael number"},
        {"input": {"n": 1105}, "output": True, "description": "1105 is Carmichael number"},
        {"input": {"n": 341}, "output": False, "description": "341 is pseudoprime but not Carmichael"}
    ]
)
async def is_carmichael_number(n: int) -> bool:
    """
    Check if n is a Carmichael number.
    
    A Carmichael number is a composite number that is a Fermat pseudoprime
    to every base a coprime to n.
    
    Uses Korselt's criterion: n is Carmichael iff:
    1. n is composite and square-free
    2. For every prime p dividing n: (p-1) divides (n-1)
    
    Args:
        n: Number to test
    
    Returns:
        True if n is a Carmichael number, False otherwise
    
    Examples:
        await is_carmichael_number(561) → True   # First Carmichael number
        await is_carmichael_number(1105) → True  # Second Carmichael number
        await is_carmichael_number(341) → False  # Pseudoprime but not Carmichael
    """
    if n <= 2 or await is_prime(n):
        return False
    
    # Get prime factorization to check Korselt's criterion
    from .primes import prime_factors
    factors = await prime_factors(n)
    
    if not factors:
        return False
    
    # Check if square-free (no repeated prime factors)
    unique_factors = list(set(factors))
    if len(factors) != len(unique_factors):
        return False
    
    # Must have at least 3 distinct prime factors
    if len(unique_factors) < 3:
        return False
    
    # Check Korselt's criterion: for each prime p | n, (p-1) | (n-1)
    for p in unique_factors:
        if (n - 1) % (p - 1) != 0:
            return False
    
    return True

# ============================================================================
# ARITHMETIC FUNCTIONS
# ============================================================================

@mcp_function(
    description="Calculate Euler's totient function φ(n) - count of integers ≤ n coprime to n.",
    namespace="arithmetic",
    category="arithmetic_functions",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    estimated_cpu_usage="medium",
    examples=[
        {"input": {"n": 12}, "output": 4, "description": "φ(12) = 4: numbers 1,5,7,11 are coprime to 12"},
        {"input": {"n": 9}, "output": 6, "description": "φ(9) = 6: numbers 1,2,4,5,7,8 are coprime to 9"},
        {"input": {"n": 17}, "output": 16, "description": "φ(17) = 16: all numbers 1-16 coprime to prime 17"}
    ]
)
async def euler_totient(n: int) -> int:
    """
    Calculate Euler's totient function φ(n).
    
    φ(n) counts the number of integers from 1 to n that are coprime to n.
    
    Args:
        n: Positive integer
    
    Returns:
        φ(n) - count of integers ≤ n that are coprime to n
    
    Examples:
        await euler_totient(12) → 4   # 1, 5, 7, 11 are coprime to 12
        await euler_totient(9) → 6    # 1, 2, 4, 5, 7, 8 are coprime to 9
        await euler_totient(17) → 16  # All 1-16 are coprime to prime 17
    """
    if n <= 0:
        return 0
    if n == 1:
        return 1
    
    # Use the formula: φ(n) = n * ∏(1 - 1/p) for all prime p dividing n
    from .primes import prime_factors
    factors = await prime_factors(n)
    
    if not factors:
        return 1
    
    result = n
    unique_primes = set(factors)
    
    for p in unique_primes:
        result = result * (p - 1) // p
    
    return result

@mcp_function(
    description="Calculate the Möbius function μ(n).",
    namespace="arithmetic",
    category="arithmetic_functions",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    examples=[
        {"input": {"n": 6}, "output": 1, "description": "μ(6) = 1: 6 = 2×3 (2 distinct primes)"},
        {"input": {"n": 12}, "output": 0, "description": "μ(12) = 0: 12 = 2²×3 (has square factor)"},
        {"input": {"n": 30}, "output": -1, "description": "μ(30) = -1: 30 = 2×3×5 (3 distinct primes)"}
    ]
)
async def mobius_function(n: int) -> int:
    """
    Calculate the Möbius function μ(n).
    
    μ(n) = 1 if n is square-free with even number of prime factors
    μ(n) = -1 if n is square-free with odd number of prime factors  
    μ(n) = 0 if n has a squared prime factor
    
    Args:
        n: Positive integer
    
    Returns:
        μ(n) ∈ {-1, 0, 1}
    
    Examples:
        await mobius_function(6) → 1    # 6 = 2×3 (2 distinct primes)
        await mobius_function(12) → 0   # 12 = 2²×3 (has square factor)
        await mobius_function(30) → -1  # 30 = 2×3×5 (3 distinct primes)
    """
    if n <= 0:
        return 0
    if n == 1:
        return 1
    
    from .primes import prime_factors
    factors = await prime_factors(n)
    
    if not factors:
        return 1
    
    # Check if square-free
    unique_factors = set(factors)
    if len(factors) != len(unique_factors):
        return 0  # Has repeated prime factor
    
    # Return (-1)^k where k is number of distinct prime factors
    k = len(unique_factors)
    return (-1) ** k

@mcp_function(
    description="Calculate ω(n) - number of distinct prime factors.",
    namespace="arithmetic",
    category="arithmetic_functions",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    examples=[
        {"input": {"n": 12}, "output": 2, "description": "ω(12) = 2: prime factors are 2, 3"},
        {"input": {"n": 30}, "output": 3, "description": "ω(30) = 3: prime factors are 2, 3, 5"},
        {"input": {"n": 17}, "output": 1, "description": "ω(17) = 1: only prime factor is 17"}
    ]
)
async def little_omega(n: int) -> int:
    """
    Calculate ω(n) - number of distinct prime factors.
    
    Args:
        n: Positive integer
    
    Returns:
        Number of distinct prime factors of n
    
    Examples:
        await little_omega(12) → 2   # 12 = 2²×3, distinct primes: 2, 3
        await little_omega(30) → 3   # 30 = 2×3×5, distinct primes: 2, 3, 5
        await little_omega(17) → 1   # 17 is prime
    """
    if n <= 1:
        return 0
    
    from .primes import prime_factors
    factors = await prime_factors(n)
    
    return len(set(factors))

@mcp_function(
    description="Calculate Ω(n) - total number of prime factors (with multiplicity).",
    namespace="arithmetic",
    category="arithmetic_functions",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    examples=[
        {"input": {"n": 12}, "output": 3, "description": "Ω(12) = 3: prime factors 2, 2, 3"},
        {"input": {"n": 30}, "output": 3, "description": "Ω(30) = 3: prime factors 2, 3, 5"},
        {"input": {"n": 8}, "output": 3, "description": "Ω(8) = 3: prime factors 2, 2, 2"}
    ]
)
async def big_omega(n: int) -> int:
    """
    Calculate Ω(n) - total number of prime factors counting multiplicity.
    
    Args:
        n: Positive integer
    
    Returns:
        Total number of prime factors of n (with repetition)
    
    Examples:
        await big_omega(12) → 3   # 12 = 2²×3, factors: 2, 2, 3
        await big_omega(30) → 3   # 30 = 2×3×5, factors: 2, 3, 5  
        await big_omega(8) → 3    # 8 = 2³, factors: 2, 2, 2
    """
    if n <= 1:
        return 0
    
    from .primes import prime_factors
    factors = await prime_factors(n)
    
    return len(factors)

# ============================================================================
# PRIME GAPS
# ============================================================================

@mcp_function(
    description="Calculate the gap between a prime and the next prime.",
    namespace="arithmetic",
    category="prime_gaps",
    execution_modes=["local", "remote"],
    estimated_cpu_usage="medium",
    examples=[
        {"input": {"p": 7}, "output": 4, "description": "Gap from 7 to next prime 11 is 4"},
        {"input": {"p": 23}, "output": 6, "description": "Gap from 23 to next prime 29 is 6"},
        {"input": {"p": 2}, "output": 1, "description": "Gap from 2 to next prime 3 is 1"}
    ]
)
async def prime_gap(p: int) -> int:
    """
    Calculate the gap between prime p and the next prime.
    
    Args:
        p: A prime number
    
    Returns:
        The gap to the next prime
    
    Raises:
        ValueError: If p is not prime
    
    Examples:
        await prime_gap(7) → 4   # Next prime after 7 is 11, gap = 4
        await prime_gap(23) → 6  # Next prime after 23 is 29, gap = 6
        await prime_gap(2) → 1   # Next prime after 2 is 3, gap = 1
    """
    if not await is_prime(p):
        raise ValueError(f"{p} is not prime")
    
    next_p = await next_prime(p)
    return next_p - p

# Export all functions
__all__ = [
    # Basic special numbers
    'is_perfect_square', 'is_power_of_two', 'fibonacci', 'factorial', 
    'fibonacci_sequence', 'is_fibonacci_number',
    
    # Mersenne primes
    'is_mersenne_prime', 'mersenne_prime_exponents', 'lucas_lehmer_test',
    
    # Fermat primes  
    'is_fermat_prime', 'fermat_numbers', 'known_fermat_primes',
    
    # Sophie Germain and safe primes
    'is_sophie_germain_prime', 'is_safe_prime', 'safe_prime_pairs',
    
    # Twin primes and related
    'is_twin_prime', 'twin_prime_pairs', 'cousin_primes', 'sexy_primes',
    
    # Wilson's theorem
    'wilson_theorem_check', 'wilson_factorial_mod',
    
    # Pseudoprimes and Carmichael numbers
    'is_fermat_pseudoprime', 'fermat_primality_check', 'is_carmichael_number',
    
    # Arithmetic functions
    'euler_totient', 'mobius_function', 'little_omega', 'big_omega',
    
    # Prime gaps
    'prime_gap'
]

if __name__ == "__main__":
    import asyncio
    
    async def test_special_numbers():
        """Test special number functions."""
        print("🔢 Special Numbers Functions Test")
        print("=" * 40)
        
        # Test basic special numbers
        print("Basic Special Numbers:")
        print(f"  is_perfect_square(16) = {await is_perfect_square(16)}")
        print(f"  is_power_of_two(8) = {await is_power_of_two(8)}")
        print(f"  fibonacci(10) = {await fibonacci(10)}")
        print(f"  factorial(5) = {await factorial(5)}")
        print(f"  is_fibonacci_number(13) = {await is_fibonacci_number(13)}")
        
        # Test Mersenne primes
        print("Mersenne Primes:")
        print(f"  is_mersenne_prime(31) = {await is_mersenne_prime(31)}")
        print(f"  is_mersenne_prime(127) = {await is_mersenne_prime(127)}")
        print(f"  lucas_lehmer_test(5) = {await lucas_lehmer_test(5)}")
        print(f"  mersenne_prime_exponents(20) = {await mersenne_prime_exponents(20)}")
        
        # Test Fermat primes
        print("\nFermat Primes:")
        print(f"  is_fermat_prime(17) = {await is_fermat_prime(17)}")
        print(f"  known_fermat_primes() = {await known_fermat_primes()}")
        
        # Test Sophie Germain primes
        print("\nSophie Germain Primes:")
        print(f"  is_sophie_germain_prime(11) = {await is_sophie_germain_prime(11)}")
        print(f"  is_safe_prime(23) = {await is_safe_prime(23)}")
        print(f"  safe_prime_pairs(25) = {await safe_prime_pairs(25)}")
        
        # Test twin primes
        print("\nTwin Primes:")
        print(f"  is_twin_prime(13) = {await is_twin_prime(13)}")
        print(f"  twin_prime_pairs(30) = {await twin_prime_pairs(30)}")
        print(f"  cousin_primes(20) = {await cousin_primes(20)}")
        
        # Test Wilson's theorem
        print("\nWilson's Theorem:")
        print(f"  wilson_theorem_check(7) = {await wilson_theorem_check(7)}")
        print(f"  wilson_theorem_check(8) = {await wilson_theorem_check(8)}")
        
        # Test pseudoprimes
        print("\nPseudoprimes:")
        print(f"  is_fermat_pseudoprime(341, 2) = {await is_fermat_pseudoprime(341, 2)}")
        print(f"  is_carmichael_number(561) = {await is_carmichael_number(561)}")
        
        # Test arithmetic functions
        print("\nArithmetic Functions:")
        print(f"  euler_totient(12) = {await euler_totient(12)}")
        print(f"  mobius_function(30) = {await mobius_function(30)}")
        print(f"  little_omega(12) = {await little_omega(12)}")
        print(f"  big_omega(12) = {await big_omega(12)}")
        
        # Test prime gaps
        print("\nPrime Gaps:")
        print(f"  prime_gap(7) = {await prime_gap(7)}")
        print(f"  prime_gap(23) = {await prime_gap(23)}")
        
        print("\n✅ All special number functions working!")
    
    asyncio.run(test_special_numbers())