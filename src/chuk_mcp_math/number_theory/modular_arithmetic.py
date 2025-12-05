#!/usr/bin/env python3
# chuk_mcp_math/number_theory/modular_arithmetic.py
"""
Extended Modular Arithmetic - Async Native

Functions for advanced modular arithmetic including Chinese Remainder Theorem,
quadratic residues, Legendre symbols, primitive roots, and discrete logarithms.

Functions:
- Chinese Remainder Theorem: crt_solve, crt_system, generalized_crt
- Quadratic residues: is_quadratic_residue, quadratic_residues, tonelli_shanks
- Legendre symbols: legendre_symbol, jacobi_symbol, kronecker_symbol
- Primitive roots: primitive_root, all_primitive_roots, order_modulo
- Discrete logarithms: discrete_log_naive, baby_step_giant_step, pohlig_hellman
- Advanced modular: modular_sqrt, carmichael_function, euler_criterion
"""

import math
import asyncio
from typing import List, Tuple, Optional
from chuk_mcp_math.mcp_decorator import mcp_function

# Import dependencies
from .primes import is_prime, prime_factors
from .divisibility import gcd, extended_gcd

# ============================================================================
# CHINESE REMAINDER THEOREM
# ============================================================================


@mcp_function(
    description="Solve system of congruences using Chinese Remainder Theorem.",
    namespace="arithmetic",
    category="modular_arithmetic",
    execution_modes=["local", "remote"],
    estimated_cpu_usage="medium",
    examples=[
        {
            "input": {"remainders": [2, 3, 2], "moduli": [3, 5, 7]},
            "output": [23, 105],
            "description": "x ≡ 2 (mod 3), x ≡ 3 (mod 5), x ≡ 2 (mod 7)",
        },
        {
            "input": {"remainders": [1, 4], "moduli": [5, 6]},
            "output": None,
            "description": "No solution when moduli not coprime",
        },
        {
            "input": {"remainders": [0, 0], "moduli": [4, 6]},
            "output": [0, 12],
            "description": "x ≡ 0 (mod 4), x ≡ 0 (mod 6)",
        },
    ],
)
async def crt_solve(
    remainders: List[int], moduli: List[int]
) -> Optional[Tuple[int, int]]:
    """
    Solve system of congruences using Chinese Remainder Theorem.

    Args:
        remainders: List of remainders [a1, a2, ..., ak]
        moduli: List of moduli [m1, m2, ..., mk]

    Returns:
        Tuple (solution, modulus) where solution is the unique solution
        modulo the product of pairwise coprime moduli, or None if no solution

    Examples:
        await crt_solve([2, 3, 2], [3, 5, 7]) → (23, 105)
        await crt_solve([1, 4], [5, 6]) → None  # gcd(5,6) = 1 but 1 ≢ 4 (mod 1)
    """
    if len(remainders) != len(moduli) or len(remainders) == 0:
        return None

    if len(remainders) == 1:
        return (remainders[0] % moduli[0], moduli[0])

    # Start with first congruence
    x, m = remainders[0], moduli[0]

    for i in range(1, len(remainders)):
        a, n = remainders[i], moduli[i]

        # Solve x ≡ a (mod n) where x ≡ current_x (mod m)
        # This means x = current_x + k*m = a + j*n for some integers k, j
        # So current_x + k*m ≡ a (mod n)
        # k*m ≡ (a - current_x) (mod n)

        g = await gcd(m, n)
        if (a - x) % g != 0:
            return None  # No solution exists

        # Use extended Euclidean algorithm
        gcd_val, u, v = await extended_gcd(m, n)

        if gcd_val != g:
            return None

        # Scale the solution
        diff = a - x
        k = (diff // g) * u

        # Update solution
        x = x + k * m
        m = m * n // g
        x = x % m

        # Yield control for large systems
        if i % 10 == 0:
            await asyncio.sleep(0)

    return (x, m)


@mcp_function(
    description="Solve generalized CRT system allowing non-coprime moduli.",
    namespace="arithmetic",
    category="modular_arithmetic",
    execution_modes=["local", "remote"],
    estimated_cpu_usage="medium",
    examples=[
        {
            "input": {"congruences": [[2, 6], [3, 10], [2, 15]]},
            "output": [2, 30],
            "description": "Handle overlapping moduli",
        },
        {
            "input": {"congruences": [[1, 4], [3, 6]]},
            "output": [9, 12],
            "description": "Non-coprime moduli case",
        },
    ],
)
async def generalized_crt(congruences: List[List[int]]) -> Optional[Tuple[int, int]]:
    """
    Solve generalized CRT system that may have non-coprime moduli.

    Args:
        congruences: List of [remainder, modulus] pairs

    Returns:
        Tuple (solution, lcm_modulus) or None if inconsistent

    Examples:
        await generalized_crt([[2, 6], [3, 10], [2, 15]]) → (2, 30)
        await generalized_crt([[1, 4], [3, 6]]) → (9, 12)
    """
    if not congruences:
        return None

    remainders = [c[0] for c in congruences]
    moduli = [c[1] for c in congruences]

    return await crt_solve(remainders, moduli)


# ============================================================================
# QUADRATIC RESIDUES
# ============================================================================


@mcp_function(
    description="Check if a is a quadratic residue modulo n.",
    namespace="arithmetic",
    category="modular_arithmetic",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    examples=[
        {"input": {"a": 2, "n": 7}, "output": True, "description": "2 ≡ 3² (mod 7)"},
        {
            "input": {"a": 3, "n": 7},
            "output": False,
            "description": "3 is not a quadratic residue mod 7",
        },
        {"input": {"a": 1, "n": 8}, "output": True, "description": "1 ≡ 1² (mod 8)"},
    ],
)
async def is_quadratic_residue(a: int, n: int) -> bool:
    """
    Check if a is a quadratic residue modulo n.

    Args:
        a: Integer to test
        n: Modulus (should be > 1)

    Returns:
        True if there exists x such that x² ≡ a (mod n)

    Examples:
        await is_quadratic_residue(2, 7) → True   # 3² ≡ 2 (mod 7)
        await is_quadratic_residue(3, 7) → False  # No solution
    """
    if n <= 1:
        return False

    a = a % n
    if a == 0:
        return True

    # For small n, check all possible squares
    for x in range(n):
        if (x * x) % n == a:
            return True

        # Yield control every 100 iterations for large n
        if x % 100 == 0 and n > 1000:
            await asyncio.sleep(0)

    return False


@mcp_function(
    description="Find all quadratic residues modulo n.",
    namespace="arithmetic",
    category="modular_arithmetic",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    examples=[
        {
            "input": {"n": 7},
            "output": [0, 1, 2, 4],
            "description": "Quadratic residues mod 7",
        },
        {
            "input": {"n": 8},
            "output": [0, 1, 4],
            "description": "Quadratic residues mod 8",
        },
        {
            "input": {"n": 11},
            "output": [0, 1, 3, 4, 5, 9],
            "description": "Quadratic residues mod 11",
        },
    ],
)
async def quadratic_residues(n: int) -> List[int]:
    """
    Find all quadratic residues modulo n.

    Args:
        n: Positive integer modulus

    Returns:
        Sorted list of all quadratic residues modulo n

    Examples:
        await quadratic_residues(7) → [0, 1, 2, 4]
        await quadratic_residues(8) → [0, 1, 4]
    """
    if n <= 1:
        return []

    residues = set()

    for x in range(n):
        residue = (x * x) % n
        residues.add(residue)

        # Yield control every 100 iterations
        if x % 100 == 0 and n > 1000:
            await asyncio.sleep(0)

    return sorted(list(residues))


@mcp_function(
    description="Solve x² ≡ a (mod p) using Tonelli-Shanks algorithm.",
    namespace="arithmetic",
    category="modular_arithmetic",
    execution_modes=["local", "remote"],
    estimated_cpu_usage="medium",
    examples=[
        {
            "input": {"a": 2, "p": 7},
            "output": [3, 4],
            "description": "3² ≡ 4² ≡ 2 (mod 7)",
        },
        {
            "input": {"a": 3, "p": 7},
            "output": None,
            "description": "No square root exists",
        },
        {
            "input": {"a": 1, "p": 11},
            "output": [1, 10],
            "description": "1² ≡ 10² ≡ 1 (mod 11)",
        },
    ],
)
async def tonelli_shanks(a: int, p: int) -> Optional[List[int]]:
    """
    Find square roots of a modulo prime p using Tonelli-Shanks algorithm.

    Args:
        a: Integer to find square root of
        p: Prime modulus

    Returns:
        List of solutions [x, p-x] if they exist, None otherwise

    Examples:
        await tonelli_shanks(2, 7) → [3, 4]  # 3² ≡ 4² ≡ 2 (mod 7)
        await tonelli_shanks(3, 7) → None    # No solution
    """
    if not await is_prime(p) or p == 2:
        return None

    a = a % p
    if a == 0:
        return [0]

    # Check if a is a quadratic residue using Legendre symbol
    legendre = await legendre_symbol(a, p)
    if legendre != 1:
        return None

    # Special case: p ≡ 3 (mod 4)
    if p % 4 == 3:
        x = pow(a, (p + 1) // 4, p)
        return sorted([x, p - x])

    # General case: Tonelli-Shanks algorithm
    # Find Q and S such that p - 1 = Q * 2^S with Q odd
    q = p - 1
    s = 0
    while q % 2 == 0:
        q //= 2
        s += 1

    # Find a quadratic non-residue z
    z = 2
    while await legendre_symbol(z, p) != -1:
        z += 1
        if z >= p:
            return None  # Shouldn't happen for prime p

    # Initialize variables
    m = s
    c = pow(z, q, p)
    t = pow(a, q, p)
    r = pow(a, (q + 1) // 2, p)

    while t != 1:
        # Find the smallest i such that t^(2^i) ≡ 1 (mod p)
        i = 1
        temp = (t * t) % p
        while temp != 1 and i < m:
            temp = (temp * temp) % p
            i += 1

        if i == m:
            return None  # No solution found

        # Update variables
        b = pow(c, 1 << (m - i - 1), p)
        m = i
        c = (b * b) % p
        t = (t * c) % p
        r = (r * b) % p

        # Yield control in long loops
        await asyncio.sleep(0)

    return sorted([r, p - r])


# ============================================================================
# LEGENDRE AND JACOBI SYMBOLS
# ============================================================================


@mcp_function(
    description="Compute Legendre symbol (a/p) for prime p.",
    namespace="arithmetic",
    category="modular_arithmetic",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    examples=[
        {
            "input": {"a": 2, "p": 7},
            "output": 1,
            "description": "2 is a quadratic residue mod 7",
        },
        {
            "input": {"a": 3, "p": 7},
            "output": -1,
            "description": "3 is not a quadratic residue mod 7",
        },
        {"input": {"a": 7, "p": 7}, "output": 0, "description": "7 ≡ 0 (mod 7)"},
    ],
)
async def legendre_symbol(a: int, p: int) -> int:
    """
    Compute the Legendre symbol (a/p).

    Args:
        a: Integer
        p: Odd prime

    Returns:
        1 if a is a quadratic residue mod p
        -1 if a is a quadratic non-residue mod p
        0 if a ≡ 0 (mod p)

    Examples:
        await legendre_symbol(2, 7) → 1   # 2 is QR mod 7
        await legendre_symbol(3, 7) → -1  # 3 is not QR mod 7
    """
    if not await is_prime(p) or p == 2:
        raise ValueError("p must be an odd prime")

    a = a % p
    if a == 0:
        return 0

    # Use Euler's criterion: (a/p) ≡ a^((p-1)/2) (mod p)
    result = pow(a, (p - 1) // 2, p)
    return -1 if result == p - 1 else result


@mcp_function(
    description="Compute Jacobi symbol (a/n) for odd n.",
    namespace="arithmetic",
    category="modular_arithmetic",
    execution_modes=["local", "remote"],
    estimated_cpu_usage="medium",
    examples=[
        {
            "input": {"a": 2, "n": 15},
            "output": 1,
            "description": "Jacobi symbol (2/15)",
        },
        {
            "input": {"a": 5, "n": 21},
            "output": -1,
            "description": "Jacobi symbol (5/21)",
        },
        {"input": {"a": 3, "n": 9}, "output": 0, "description": "gcd(3,9) > 1"},
    ],
)
async def jacobi_symbol(a: int, n: int) -> int:
    """
    Compute the Jacobi symbol (a/n) using quadratic reciprocity.

    Args:
        a: Integer
        n: Odd positive integer

    Returns:
        Jacobi symbol value (-1, 0, or 1)

    Examples:
        await jacobi_symbol(2, 15) → 1
        await jacobi_symbol(5, 21) → -1
    """
    if n <= 0 or n % 2 == 0:
        raise ValueError("n must be odd and positive")

    a = a % n
    result = 1

    while a != 0:
        # Remove factors of 2 from a
        while a % 2 == 0:
            a //= 2
            # (2/n) = (-1)^((n²-1)/8)
            if n % 8 in [3, 5]:
                result = -result

        # Swap a and n
        a, n = n, a

        # Quadratic reciprocity: (a/n)(n/a) = (-1)^((a-1)(n-1)/4)
        if a % 4 == 3 and n % 4 == 3:
            result = -result

        a = a % n

        # Yield control periodically
        await asyncio.sleep(0)

    return result if n == 1 else 0


# ============================================================================
# PRIMITIVE ROOTS
# ============================================================================


@mcp_function(
    description="Find the smallest primitive root modulo n.",
    namespace="arithmetic",
    category="modular_arithmetic",
    execution_modes=["local", "remote"],
    estimated_cpu_usage="high",
    examples=[
        {"input": {"n": 7}, "output": 3, "description": "3 is primitive root mod 7"},
        {"input": {"n": 11}, "output": 2, "description": "2 is primitive root mod 11"},
        {
            "input": {"n": 8},
            "output": None,
            "description": "No primitive root exists mod 8",
        },
    ],
)
async def primitive_root(n: int) -> Optional[int]:
    """
    Find the smallest primitive root modulo n.

    Args:
        n: Positive integer

    Returns:
        Smallest primitive root modulo n, or None if none exists

    Examples:
        await primitive_root(7) → 3   # 3 is primitive root mod 7
        await primitive_root(11) → 2  # 2 is primitive root mod 11
        await primitive_root(8) → None # No primitive root mod 8
    """
    if n <= 1:
        return None

    # Primitive roots exist only for n = 1, 2, 4, p^k, 2*p^k where p is odd prime
    if not await _has_primitive_root(n):
        return None

    phi_n = await _euler_totient(n)

    # Find prime factors of φ(n)
    factors = await prime_factors(phi_n)
    prime_divisors = list(set(factors))

    for g in range(2, n):
        if await gcd(g, n) != 1:
            continue

        # Check if g is a primitive root
        is_primitive = True
        for p in prime_divisors:
            if pow(g, phi_n // p, n) == 1:
                is_primitive = False
                break

        if is_primitive:
            return g

        # Yield control every 100 candidates
        if g % 100 == 0:
            await asyncio.sleep(0)

    return None


@mcp_function(
    description="Find all primitive roots modulo n.",
    namespace="arithmetic",
    category="modular_arithmetic",
    execution_modes=["local", "remote"],
    estimated_cpu_usage="high",
    examples=[
        {
            "input": {"n": 7},
            "output": [3, 5],
            "description": "All primitive roots mod 7",
        },
        {
            "input": {"n": 11},
            "output": [2, 6, 7, 8],
            "description": "All primitive roots mod 11",
        },
        {"input": {"n": 12}, "output": [], "description": "No primitive roots mod 12"},
    ],
)
async def all_primitive_roots(n: int) -> List[int]:
    """
    Find all primitive roots modulo n.

    Args:
        n: Positive integer

    Returns:
        Sorted list of all primitive roots modulo n

    Examples:
        await all_primitive_roots(7) → [3, 5]
        await all_primitive_roots(11) → [2, 6, 7, 8]
    """
    if n <= 1 or not await _has_primitive_root(n):
        return []

    phi_n = await _euler_totient(n)
    factors = await prime_factors(phi_n)
    prime_divisors = list(set(factors))

    roots = []

    for g in range(2, n):
        if await gcd(g, n) != 1:
            continue

        # Check if g is a primitive root
        is_primitive = True
        for p in prime_divisors:
            if pow(g, phi_n // p, n) == 1:
                is_primitive = False
                break

        if is_primitive:
            roots.append(g)

        # Yield control every 50 candidates
        if g % 50 == 0:
            await asyncio.sleep(0)

    return roots


@mcp_function(
    description="Find the multiplicative order of a modulo n.",
    namespace="arithmetic",
    category="modular_arithmetic",
    execution_modes=["local", "remote"],
    estimated_cpu_usage="medium",
    examples=[
        {"input": {"a": 3, "n": 7}, "output": 6, "description": "ord₇(3) = 6"},
        {"input": {"a": 2, "n": 7}, "output": 3, "description": "ord₇(2) = 3"},
        {"input": {"a": 4, "n": 6}, "output": None, "description": "gcd(4,6) ≠ 1"},
    ],
)
async def order_modulo(a: int, n: int) -> Optional[int]:
    """
    Find the multiplicative order of a modulo n.

    Args:
        a: Integer
        n: Positive integer modulus

    Returns:
        Smallest positive integer k such that a^k ≡ 1 (mod n),
        or None if gcd(a,n) ≠ 1

    Examples:
        await order_modulo(3, 7) → 6  # 3⁶ ≡ 1 (mod 7)
        await order_modulo(2, 7) → 3  # 2³ ≡ 1 (mod 7)
    """
    if n <= 1 or await gcd(a, n) != 1:
        return None

    a = a % n
    if a == 1:
        return 1

    phi_n = await _euler_totient(n)

    # The order must divide φ(n)
    await prime_factors(phi_n)

    # Try all divisors of φ(n)
    divisors = await _get_divisors(phi_n)

    for d in sorted(divisors):
        if pow(a, d, n) == 1:
            return d

        # Yield control every 10 divisors
        if len([x for x in divisors if x <= d]) % 10 == 0:
            await asyncio.sleep(0)

    return None


# ============================================================================
# DISCRETE LOGARITHMS
# ============================================================================


@mcp_function(
    description="Solve discrete logarithm g^x ≡ h (mod n) using naive method.",
    namespace="arithmetic",
    category="modular_arithmetic",
    execution_modes=["local", "remote"],
    estimated_cpu_usage="high",
    examples=[
        {
            "input": {"g": 3, "h": 2, "n": 7},
            "output": 2,
            "description": "3² ≡ 2 (mod 7)",
        },
        {
            "input": {"g": 2, "h": 3, "n": 5},
            "output": None,
            "description": "No solution exists",
        },
        {
            "input": {"g": 5, "h": 1, "n": 11},
            "output": 0,
            "description": "5⁰ ≡ 1 (mod 11)",
        },
    ],
)
async def discrete_log_naive(
    g: int, h: int, n: int, max_exp: Optional[int] = None
) -> Optional[int]:
    """
    Solve discrete logarithm g^x ≡ h (mod n) using brute force.

    Args:
        g: Base
        h: Target value
        n: Modulus
        max_exp: Maximum exponent to try (default: n-1)

    Returns:
        Smallest non-negative x such that g^x ≡ h (mod n), or None

    Examples:
        await discrete_log_naive(3, 2, 7) → 2  # 3² ≡ 2 (mod 7)
        await discrete_log_naive(2, 3, 5) → None
    """
    if n <= 1:
        return None

    g, h = g % n, h % n

    if max_exp is None:
        max_exp = n - 1

    current = 1
    for x in range(max_exp + 1):
        if current == h:
            return x
        current = (current * g) % n

        # Yield control every 1000 iterations
        if x % 1000 == 0:
            await asyncio.sleep(0)

    return None


@mcp_function(
    description="Solve discrete logarithm using Baby-step Giant-step algorithm.",
    namespace="arithmetic",
    category="modular_arithmetic",
    execution_modes=["local", "remote"],
    estimated_cpu_usage="high",
    examples=[
        {
            "input": {"g": 2, "h": 5, "n": 11},
            "output": 4,
            "description": "2⁴ ≡ 5 (mod 11)",
        },
        {
            "input": {"g": 3, "h": 4, "n": 13},
            "output": 4,
            "description": "3⁴ ≡ 4 (mod 13)",
        },
    ],
)
async def baby_step_giant_step(g: int, h: int, n: int) -> Optional[int]:
    """
    Solve discrete logarithm using Baby-step Giant-step algorithm.

    More efficient than naive approach with O(√n) time complexity.

    Args:
        g: Base (should be primitive root or have known order)
        h: Target value
        n: Modulus

    Returns:
        Solution x such that g^x ≡ h (mod n), or None

    Examples:
        await baby_step_giant_step(2, 5, 11) → 4  # 2⁴ ≡ 5 (mod 11)
    """
    if n <= 1:
        return None

    g, h = g % n, h % n

    if h == 1:
        return 0

    # Choose m ≈ √n
    m = int(math.sqrt(n)) + 1

    # Baby steps: store g^j for j = 0, 1, ..., m-1
    baby_steps = {}
    gamma = 1

    for j in range(m):
        if gamma == h:
            return j
        baby_steps[gamma] = j
        gamma = (gamma * g) % n

        # Yield control every 100 steps
        if j % 100 == 0:
            await asyncio.sleep(0)

    # Giant steps: look for h * (g^(-m))^i in baby_steps
    try:
        g_inv_m = pow(g, n - 1 - m, n)  # g^(-m) = g^(φ(n)-m) when gcd(g,n)=1
    except Exception:
        return None

    y = h
    for i in range(m):
        if y in baby_steps:
            x = i * m + baby_steps[y]
            # Verify solution
            if pow(g, x, n) == h:
                return x
        y = (y * g_inv_m) % n

        # Yield control every 100 steps
        if i % 100 == 0:
            await asyncio.sleep(0)

    return None


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


async def _has_primitive_root(n: int) -> bool:
    """Check if n has primitive roots."""
    if n <= 0:
        return False
    if n <= 2:
        return True
    if n == 4:
        return True

    # Check if n = p^k or n = 2*p^k where p is odd prime
    if n % 2 == 0:
        n //= 2

    # Check if n is a prime power
    factors = await prime_factors(n)
    return len(set(factors)) == 1


async def _euler_totient(n: int) -> int:
    """Compute Euler's totient function φ(n)."""
    if n <= 1:
        return 0

    result = n
    factors = await prime_factors(n)
    unique_primes = set(factors)

    for p in unique_primes:
        result = result * (p - 1) // p

    return result


async def _get_divisors(n: int) -> List[int]:
    """Get all positive divisors of n."""
    if n <= 0:
        return []

    divisors = []
    for i in range(1, int(math.sqrt(n)) + 1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)

        # Yield control every 100 iterations
        if i % 100 == 0:
            await asyncio.sleep(0)

    return sorted(divisors)


# Export all functions
__all__ = [
    # Chinese Remainder Theorem
    "crt_solve",
    "generalized_crt",
    # Quadratic residues
    "is_quadratic_residue",
    "quadratic_residues",
    "tonelli_shanks",
    # Legendre and Jacobi symbols
    "legendre_symbol",
    "jacobi_symbol",
    # Primitive roots
    "primitive_root",
    "all_primitive_roots",
    "order_modulo",
    # Discrete logarithms
    "discrete_log_naive",
    "baby_step_giant_step",
]
