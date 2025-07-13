#!/usr/bin/env python3
# tests/math/arithmetic/number_theory/test_arithmetic_functions.py
"""
Comprehensive pytest test suite for arithmetic_functions.py module.

Tests cover:
- Euler's totient function (φ) and Jordan's totient function
- Möbius function (μ) and multiplicative properties
- Omega functions (ω and Ω) for prime factor counting
- Divisor functions (σ_k) for sums of powers of divisors
- Von Mangoldt function (Λ) for prime powers
- Liouville function (λ) and complete multiplicativity
- Carmichael function (λ) for exponent of multiplicative group
- Perfect, abundant, and deficient number classification
- Mathematical properties and relationships
- Edge cases, error conditions, and performance testing
- Async behavior verification
"""

import pytest
import asyncio
import time
import math
from typing import List, Tuple

# Import the functions to test
from chuk_mcp_functions.math.arithmetic.number_theory.arithmetic_functions import (
    # Multiplicative functions
    euler_totient, jordan_totient, mobius_function,
    
    # Additive functions
    little_omega, big_omega,
    
    # Divisor functions
    divisor_power_sum,
    
    # Special functions
    von_mangoldt_function, liouville_function, carmichael_lambda,
    
    # Perfect number functions
    is_perfect_number, is_abundant_number, is_deficient_number, perfect_numbers_up_to
)

# ============================================================================
# EULER'S TOTIENT FUNCTION TESTS
# ============================================================================

class TestEulerTotient:
    """Test cases for Euler's totient function and Jordan's generalization."""
    
    @pytest.mark.asyncio
    async def test_mobius_function_square_free_products(self):
        """Test Möbius function on square-free products."""
        # Products of distinct primes should alternate sign based on number of factors
        square_free_products = [
            (6, 1),    # 2×3 → μ(6) = (-1)² = 1
            (10, 1),   # 2×5 → μ(10) = (-1)² = 1
            (14, 1),   # 2×7 → μ(14) = (-1)² = 1
            (15, 1),   # 3×5 → μ(15) = (-1)² = 1
            (21, 1),   # 3×7 → μ(21) = (-1)² = 1
            (22, 1),   # 2×11 → μ(22) = (-1)² = 1
            (30, -1),  # 2×3×5 → μ(30) = (-1)³ = -1
            (42, -1),  # 2×3×7 → μ(42) = (-1)³ = -1
            (66, -1),  # 2×3×11 → μ(66) = (-1)³ = -1
            (70, -1),  # 2×5×7 → μ(70) = (-1)³ = -1
            (105, -1), # 3×5×7 → μ(105) = (-1)³ = -1
            (210, 1),  # 2×3×5×7 → μ(210) = (-1)⁴ = 1
        ]
        
        for n, expected in square_free_products:
            result = await mobius_function(n)
            assert result == expected, f"μ({n}) should be {expected}"
    
    @pytest.mark.asyncio
    async def test_mobius_function_edge_cases(self):
        """Test edge cases for Möbius function."""
        assert await mobius_function(0) == 0
        assert await mobius_function(1) == 1
    
    @pytest.mark.asyncio
    async def test_mobius_function_properties(self):
        """Test mathematical properties of Möbius function."""
        # μ(n) ∈ {-1, 0, 1} for all n
        for n in range(1, 51):
            mu_n = await mobius_function(n)
            assert mu_n in [-1, 0, 1], f"μ({n}) should be in {{-1, 0, 1}}"

# ============================================================================
# OMEGA FUNCTIONS TESTS
# ============================================================================

class TestOmegaFunctions:
    """Test cases for omega functions (prime factor counting)."""
    
    @pytest.mark.asyncio
    async def test_little_omega_known_values(self):
        """Test ω(n) - number of distinct prime factors."""
        known_values = [
            (1, 0),    # ω(1) = 0 (no prime factors)
            (2, 1),    # ω(2) = 1 (one prime: 2)
            (3, 1),    # ω(3) = 1 (one prime: 3)
            (4, 1),    # ω(4) = 1 (one distinct prime: 2)
            (6, 2),    # ω(6) = 2 (two primes: 2, 3)
            (8, 1),    # ω(8) = 1 (one distinct prime: 2)
            (12, 2),   # ω(12) = 2 (two distinct primes: 2, 3)
            (15, 2),   # ω(15) = 2 (two primes: 3, 5)
            (30, 3),   # ω(30) = 3 (three primes: 2, 3, 5)
            (42, 3),   # ω(42) = 3 (three primes: 2, 3, 7)
            (60, 3),   # ω(60) = 3 (three distinct primes: 2, 3, 5)
            (105, 3),  # ω(105) = 3 (three primes: 3, 5, 7)
            (210, 4),  # ω(210) = 4 (four primes: 2, 3, 5, 7)
        ]
        
        for n, expected in known_values:
            result = await little_omega(n)
            assert result == expected, f"ω({n}) should be {expected}, got {result}"
    
    @pytest.mark.asyncio
    async def test_big_omega_known_values(self):
        """Test Ω(n) - number of prime factors with multiplicity."""
        known_values = [
            (1, 0),    # Ω(1) = 0 (no prime factors)
            (2, 1),    # Ω(2) = 1 (one prime: 2)
            (3, 1),    # Ω(3) = 1 (one prime: 3)
            (4, 2),    # Ω(4) = 2 (factors: 2, 2)
            (6, 2),    # Ω(6) = 2 (factors: 2, 3)
            (8, 3),    # Ω(8) = 3 (factors: 2, 2, 2)
            (12, 3),   # Ω(12) = 3 (factors: 2, 2, 3)
            (15, 2),   # Ω(15) = 2 (factors: 3, 5)
            (16, 4),   # Ω(16) = 4 (factors: 2, 2, 2, 2)
            (18, 3),   # Ω(18) = 3 (factors: 2, 3, 3)
            (20, 3),   # Ω(20) = 3 (factors: 2, 2, 5)
            (24, 4),   # Ω(24) = 4 (factors: 2, 2, 2, 3)
            (30, 3),   # Ω(30) = 3 (factors: 2, 3, 5)
            (60, 4),   # Ω(60) = 4 (factors: 2, 2, 3, 5)
        ]
        
        for n, expected in known_values:
            result = await big_omega(n)
            assert result == expected, f"Ω({n}) should be {expected}, got {result}"
    
    @pytest.mark.asyncio
    async def test_omega_relationship(self):
        """Test relationship between ω(n) and Ω(n)."""
        # ω(n) ≤ Ω(n) for all n > 1
        # ω(n) = Ω(n) iff n is square-free
        
        for n in range(2, 51):
            little_omega_n = await little_omega(n)
            big_omega_n = await big_omega(n)
            
            assert little_omega_n <= big_omega_n, f"ω({n}) should be ≤ Ω({n})"
    
    @pytest.mark.asyncio
    async def test_omega_prime_powers(self):
        """Test omega functions on prime powers."""
        # For p^k: ω(p^k) = 1, Ω(p^k) = k
        prime_powers = [
            (4, 1, 2),   # 2² → ω=1, Ω=2
            (8, 1, 3),   # 2³ → ω=1, Ω=3
            (9, 1, 2),   # 3² → ω=1, Ω=2
            (16, 1, 4),  # 2⁴ → ω=1, Ω=4
            (25, 1, 2),  # 5² → ω=1, Ω=2
            (27, 1, 3),  # 3³ → ω=1, Ω=3
            (32, 1, 5),  # 2⁵ → ω=1, Ω=5
        ]
        
        for n, expected_little, expected_big in prime_powers:
            little_result = await little_omega(n)
            big_result = await big_omega(n)
            
            assert little_result == expected_little, f"ω({n}) should be {expected_little}"
            assert big_result == expected_big, f"Ω({n}) should be {expected_big}"
    
    @pytest.mark.asyncio
    async def test_omega_edge_cases(self):
        """Test edge cases for omega functions."""
        assert await little_omega(1) == 0
        assert await big_omega(1) == 0
        assert await little_omega(0) == 0
        assert await big_omega(0) == 0

# ============================================================================
# DIVISOR FUNCTIONS TESTS
# ============================================================================

class TestDivisorFunctions:
    """Test cases for divisor power sum functions."""
    
    @pytest.mark.asyncio
    async def test_divisor_power_sum_tau_function(self):
        """Test σ₀(n) = τ(n) - divisor count function."""
        known_tau_values = [
            (1, 1),    # τ(1) = 1
            (2, 2),    # τ(2) = 2
            (3, 2),    # τ(3) = 2
            (4, 3),    # τ(4) = 3
            (5, 2),    # τ(5) = 2
            (6, 4),    # τ(6) = 4
            (8, 4),    # τ(8) = 4
            (9, 3),    # τ(9) = 3
            (10, 4),   # τ(10) = 4
            (12, 6),   # τ(12) = 6
            (16, 5),   # τ(16) = 5
            (18, 6),   # τ(18) = 6
            (20, 6),   # τ(20) = 6
            (24, 8),   # τ(24) = 8
        ]
        
        for n, expected in known_tau_values:
            result = await divisor_power_sum(n, 0)
            assert result == expected, f"τ({n}) = σ₀({n}) should be {expected}, got {result}"
    
    @pytest.mark.asyncio
    async def test_divisor_power_sum_sigma_function(self):
        """Test σ₁(n) = σ(n) - sum of divisors function."""
        known_sigma_values = [
            (1, 1),    # σ(1) = 1
            (2, 3),    # σ(2) = 1 + 2 = 3
            (3, 4),    # σ(3) = 1 + 3 = 4
            (4, 7),    # σ(4) = 1 + 2 + 4 = 7
            (5, 6),    # σ(5) = 1 + 5 = 6
            (6, 12),   # σ(6) = 1 + 2 + 3 + 6 = 12
            (8, 15),   # σ(8) = 1 + 2 + 4 + 8 = 15
            (9, 13),   # σ(9) = 1 + 3 + 9 = 13
            (10, 18),  # σ(10) = 1 + 2 + 5 + 10 = 18
            (12, 28),  # σ(12) = 1 + 2 + 3 + 4 + 6 + 12 = 28
        ]
        
        for n, expected in known_sigma_values:
            result = await divisor_power_sum(n, 1)
            assert result == expected, f"σ({n}) = σ₁({n}) should be {expected}, got {result}"
    
    @pytest.mark.asyncio
    async def test_divisor_power_sum_higher_powers(self):
        """Test σₖ(n) for k > 1."""
        test_cases = [
            (6, 2, 50),   # σ₂(6) = 1² + 2² + 3² + 6² = 1 + 4 + 9 + 36 = 50
            (8, 2, 85),   # σ₂(8) = 1² + 2² + 4² + 8² = 1 + 4 + 16 + 64 = 85
            (4, 3, 73),   # σ₃(4) = 1³ + 2³ + 4³ = 1 + 8 + 64 = 73
        ]
        
        for n, k, expected in test_cases:
            result = await divisor_power_sum(n, k)
            assert result == expected, f"σ_{k}({n}) should be {expected}, got {result}"
    
    @pytest.mark.asyncio
    async def test_divisor_power_sum_edge_cases(self):
        """Test edge cases for divisor power sum."""
        assert await divisor_power_sum(0, 0) == 0
        assert await divisor_power_sum(1, 0) == 1
        assert await divisor_power_sum(1, 1) == 1
        assert await divisor_power_sum(1, 2) == 1

# ============================================================================
# VON MANGOLDT FUNCTION TESTS
# ============================================================================

class TestVonMangoldtFunction:
    """Test cases for the von Mangoldt function."""
    
    @pytest.mark.asyncio
    async def test_von_mangoldt_function_primes(self):
        """Test von Mangoldt function on primes."""
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
        
        for p in primes:
            result = await von_mangoldt_function(p)
            expected = math.log(p)
            assert abs(result - expected) < 1e-10, f"Λ({p}) should be ln({p}) ≈ {expected}"
    
    @pytest.mark.asyncio
    async def test_von_mangoldt_function_prime_powers(self):
        """Test von Mangoldt function on prime powers."""
        prime_powers = [
            (4, 2),   # 2² → Λ(4) = ln(2)
            (8, 2),   # 2³ → Λ(8) = ln(2)
            (9, 3),   # 3² → Λ(9) = ln(3)
            (16, 2),  # 2⁴ → Λ(16) = ln(2)
            (25, 5),  # 5² → Λ(25) = ln(5)
            (27, 3),  # 3³ → Λ(27) = ln(3)
            (32, 2),  # 2⁵ → Λ(32) = ln(2)
        ]
        
        for pk, p in prime_powers:
            result = await von_mangoldt_function(pk)
            expected = math.log(p)
            assert abs(result - expected) < 1e-10, f"Λ({pk}) should be ln({p}) ≈ {expected}"
    
    @pytest.mark.asyncio
    async def test_von_mangoldt_function_composites(self):
        """Test von Mangoldt function on composite numbers (not prime powers)."""
        composites = [6, 10, 12, 14, 15, 18, 20, 21, 22, 24, 26, 28, 30]
        
        for n in composites:
            result = await von_mangoldt_function(n)
            assert abs(result) < 1e-10, f"Λ({n}) should be 0 for composite {n}"
    
    @pytest.mark.asyncio
    async def test_von_mangoldt_function_edge_cases(self):
        """Test edge cases for von Mangoldt function."""
        assert await von_mangoldt_function(0) == 0.0
        assert await von_mangoldt_function(1) == 0.0

# ============================================================================
# LIOUVILLE FUNCTION TESTS
# ============================================================================

class TestLiouvilleFunction:
    """Test cases for the Liouville function."""
    
    @pytest.mark.asyncio
    async def test_liouville_function_known_values(self):
        """Test Liouville function with known values."""
        known_values = [
            (1, 1),    # λ(1) = 1 (Ω(1) = 0, (-1)⁰ = 1)
            (2, -1),   # λ(2) = -1 (Ω(2) = 1, (-1)¹ = -1)
            (3, -1),   # λ(3) = -1 (Ω(3) = 1, (-1)¹ = -1)
            (4, 1),    # λ(4) = 1 (Ω(4) = 2, (-1)² = 1)
            (5, -1),   # λ(5) = -1 (Ω(5) = 1, (-1)¹ = -1)
            (6, 1),    # λ(6) = 1 (Ω(6) = 2, (-1)² = 1)
            (7, -1),   # λ(7) = -1 (Ω(7) = 1, (-1)¹ = -1)
            (8, -1),   # λ(8) = -1 (Ω(8) = 3, (-1)³ = -1)
            (9, 1),    # λ(9) = 1 (Ω(9) = 2, (-1)² = 1)
            (10, 1),   # λ(10) = 1 (Ω(10) = 2, (-1)² = 1)
            (12, -1),  # λ(12) = -1 (Ω(12) = 3, (-1)³ = -1)
            (15, 1),   # λ(15) = 1 (Ω(15) = 2, (-1)² = 1)
            (16, 1),   # λ(16) = 1 (Ω(16) = 4, (-1)⁴ = 1)
        ]
        
        for n, expected in known_values:
            result = await liouville_function(n)
            assert result == expected, f"λ({n}) should be {expected}, got {result}"
    
    @pytest.mark.asyncio
    async def test_liouville_function_properties(self):
        """Test mathematical properties of Liouville function."""
        # λ(n) ∈ {-1, 1} for all n ≥ 1
        for n in range(1, 51):
            lambda_n = await liouville_function(n)
            assert lambda_n in [-1, 1], f"λ({n}) should be in {{-1, 1}}"
    
    @pytest.mark.asyncio
    async def test_liouville_function_completely_multiplicative(self):
        """Test that Liouville function is completely multiplicative."""
        # λ(mn) = λ(m)λ(n) for all m, n
        test_pairs = [(2, 3), (3, 5), (4, 5), (2, 7), (3, 8), (5, 6)]
        
        for m, n in test_pairs:
            lambda_m = await liouville_function(m)
            lambda_n = await liouville_function(n)
            lambda_mn = await liouville_function(m * n)
            
            assert lambda_mn == lambda_m * lambda_n, f"λ({m}×{n}) should equal λ({m})×λ({n})"
    
    @pytest.mark.asyncio
    async def test_liouville_function_edge_cases(self):
        """Test edge cases for Liouville function."""
        assert await liouville_function(0) == 1
        assert await liouville_function(1) == 1

# ============================================================================
# CARMICHAEL FUNCTION TESTS
# ============================================================================

class TestCarmichaelFunction:
    """Test cases for the Carmichael lambda function."""
    
    @pytest.mark.asyncio
    async def test_carmichael_lambda_known_values(self):
        """Test Carmichael function with known values."""
        known_values = [
            (1, 1),     # λ(1) = 1
            (2, 1),     # λ(2) = 1
            (3, 2),     # λ(3) = 2
            (4, 2),     # λ(4) = 2
            (5, 4),     # λ(5) = 4
            (6, 2),     # λ(6) = lcm(λ(2), λ(3)) = lcm(1, 2) = 2
            (7, 6),     # λ(7) = 6
            (8, 2),     # λ(8) = 2^(3-2) = 2
            (9, 6),     # λ(9) = 3×3^(2-1) = 6
            (10, 4),    # λ(10) = lcm(λ(2), λ(5)) = lcm(1, 4) = 4
            (12, 2),    # λ(12) = lcm(λ(4), λ(3)) = lcm(2, 2) = 2
            (15, 4),    # λ(15) = lcm(λ(3), λ(5)) = lcm(2, 4) = 4
            (16, 4),    # λ(16) = 2^(4-2) = 4
        ]
        
        for n, expected in known_values:
            result = await carmichael_lambda(n)
            assert result == expected, f"λ({n}) should be {expected}, got {result}"
    
    @pytest.mark.asyncio
    async def test_carmichael_lambda_primes(self):
        """Test that λ(p) = p-1 for odd primes p."""
        odd_primes = [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
        
        for p in odd_primes:
            result = await carmichael_lambda(p)
            expected = p - 1
            assert result == expected, f"λ({p}) should be {expected} for prime {p}"
    
    @pytest.mark.asyncio
    async def test_carmichael_lambda_powers_of_2(self):
        """Test Carmichael function on powers of 2."""
        powers_of_2_cases = [
            (4, 2),     # λ(2²) = 2
            (8, 2),     # λ(2³) = 2
            (16, 4),    # λ(2⁴) = 4
            (32, 8),    # λ(2⁵) = 8
            (64, 16),   # λ(2⁶) = 16
        ]
        
        for n, expected in powers_of_2_cases:
            result = await carmichael_lambda(n)
            assert result == expected, f"λ({n}) should be {expected}"
    
    @pytest.mark.asyncio
    async def test_carmichael_lambda_odd_prime_powers(self):
        """Test Carmichael function on odd prime powers."""
        odd_prime_powers = [
            (9, 6),     # λ(3²) = 3¹(3-1) = 6
            (25, 20),   # λ(5²) = 5¹(5-1) = 20
            (27, 18),   # λ(3³) = 3²(3-1) = 18
            (49, 42),   # λ(7²) = 7¹(7-1) = 42
            (125, 100), # λ(5³) = 5²(5-1) = 100
        ]
        
        for n, expected in odd_prime_powers:
            result = await carmichael_lambda(n)
            assert result == expected, f"λ({n}) should be {expected}"
    
    @pytest.mark.asyncio
    async def test_carmichael_lambda_divides_euler_totient(self):
        """Test that λ(n) divides φ(n)."""
        for n in range(2, 51):
            lambda_n = await carmichael_lambda(n)
            phi_n = await euler_totient(n)
            
            assert phi_n % lambda_n == 0, f"λ({n}) should divide φ({n})"
    
    @pytest.mark.asyncio
    async def test_carmichael_lambda_edge_cases(self):
        """Test edge cases for Carmichael function."""
        assert await carmichael_lambda(0) == 0
        assert await carmichael_lambda(1) == 1
        assert await carmichael_lambda(2) == 1

# ============================================================================
# PERFECT NUMBER FUNCTIONS TESTS
# ============================================================================

class TestPerfectNumberFunctions:
    """Test cases for perfect, abundant, and deficient number functions."""
    
    @pytest.mark.asyncio
    async def test_is_perfect_number_known_values(self):
        """Test perfect number identification."""
        perfect_numbers = [6, 28, 496, 8128]
        non_perfect_numbers = [1, 2, 3, 4, 5, 7, 8, 9, 10, 12, 24, 30, 100, 1000]
        
        for n in perfect_numbers:
            assert await is_perfect_number(n) == True, f"{n} should be perfect"
        
        for n in non_perfect_numbers:
            assert await is_perfect_number(n) == False, f"{n} should not be perfect"
    
    @pytest.mark.asyncio
    async def test_is_abundant_number_known_values(self):
        """Test abundant number identification."""
        abundant_numbers = [12, 18, 20, 24, 30, 36, 40, 42, 48, 54, 56, 60, 66, 70, 72, 78, 80]
        non_abundant_numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 19, 21, 22, 23]
        
        for n in abundant_numbers:
            assert await is_abundant_number(n) == True, f"{n} should be abundant"
        
        for n in non_abundant_numbers:
            assert await is_abundant_number(n) == False, f"{n} should not be abundant"
    
    @pytest.mark.asyncio
    async def test_is_deficient_number_known_values(self):
        """Test deficient number identification."""
        deficient_numbers = [1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 19, 21, 22, 23]
        non_deficient_numbers = [6, 12, 18, 20, 24, 28, 30]  # Perfect or abundant
        
        for n in deficient_numbers:
            assert await is_deficient_number(n) == True, f"{n} should be deficient"
        
        for n in non_deficient_numbers:
            assert await is_deficient_number(n) == False, f"{n} should not be deficient"
    
    @pytest.mark.asyncio
    async def test_number_classification_completeness(self):
        """Test that every number is exactly one of perfect, abundant, or deficient."""
        for n in range(1, 101):
            is_perfect = await is_perfect_number(n)
            is_abundant = await is_abundant_number(n)
            is_deficient = await is_deficient_number(n)
            
            # Exactly one should be true
            total = sum([is_perfect, is_abundant, is_deficient])
            assert total == 1, f"{n} should be exactly one of perfect/abundant/deficient"
    
    @pytest.mark.asyncio
    async def test_perfect_numbers_up_to_small(self):
        """Test finding perfect numbers up to small limits."""
        result_100 = await perfect_numbers_up_to(100)
        assert result_100 == [6, 28], f"Perfect numbers ≤ 100 should be [6, 28]"
        
        result_10 = await perfect_numbers_up_to(10)
        assert result_10 == [6], f"Perfect numbers ≤ 10 should be [6]"
        
        result_5 = await perfect_numbers_up_to(5)
        assert result_5 == [], f"Perfect numbers ≤ 5 should be []"
    
    @pytest.mark.asyncio
    async def test_perfect_number_functions_edge_cases(self):
        """Test edge cases for perfect number functions."""
        assert await is_perfect_number(0) == False
        assert await is_perfect_number(1) == False
        assert await is_abundant_number(0) == False
        assert await is_abundant_number(1) == False
        assert await is_deficient_number(0) == False
        assert await is_deficient_number(1) == True  # 1 is deficient

# ============================================================================
# INTEGRATION AND MATHEMATICAL PROPERTIES TESTS
# ============================================================================

class TestIntegrationAndProperties:
    """Integration tests and mathematical property verification."""
    
    @pytest.mark.asyncio
    async def test_mobius_inversion_formula(self):
        """Test Möbius inversion formula applications."""
        # Sum of μ(d) over all divisors d of n should be 1 if n=1, 0 otherwise
        for n in range(1, 31):
            # Import divisors function
            from chuk_mcp_functions.math.arithmetic.number_theory.divisibility import divisors
            
            divisors_n = await divisors(n)
            mobius_sum = sum([await mobius_function(d) for d in divisors_n])
            
            expected = 1 if n == 1 else 0
            assert mobius_sum == expected, f"Sum of μ(d) for d|{n} should be {expected}"
    
    @pytest.mark.asyncio
    async def test_euler_totient_sum_formula(self):
        """Test that sum of φ(d) over divisors of n equals n."""
        # ∑_{d|n} φ(d) = n
        for n in range(1, 31):
            from chuk_mcp_functions.math.arithmetic.number_theory.divisibility import divisors
            
            divisors_n = await divisors(n)
            phi_sum = sum([await euler_totient(d) for d in divisors_n])
            
            assert phi_sum == n, f"Sum of φ(d) for d|{n} should equal {n}"
    
    @pytest.mark.asyncio
    async def test_divisor_function_relationships(self):
        """Test relationships between different divisor functions."""
        for n in range(1, 21):
            # σ₀(n) = τ(n) should equal actual divisor count
            from chuk_mcp_functions.math.arithmetic.number_theory.divisibility import divisor_count, divisor_sum
            
            sigma_0 = await divisor_power_sum(n, 0)
            tau_n = await divisor_count(n)
            assert sigma_0 == tau_n, f"σ₀({n}) should equal τ({n})"
            
            # σ₁(n) = σ(n) should equal actual divisor sum
            sigma_1 = await divisor_power_sum(n, 1)
            sigma_n = await divisor_sum(n)
            assert sigma_1 == sigma_n, f"σ₁({n}) should equal σ({n})"
    
    @pytest.mark.asyncio
    async def test_multiplicative_function_properties(self):
        """Test multiplicative properties of arithmetic functions."""
        # Test that φ, λ (Carmichael), τ, σ are multiplicative
        coprime_pairs = [(3, 4), (5, 7), (8, 9), (7, 11)]
        
        for a, b in coprime_pairs:
            # Euler totient is multiplicative
            phi_a = await euler_totient(a)
            phi_b = await euler_totient(b)
            phi_ab = await euler_totient(a * b)
            assert phi_ab == phi_a * phi_b, f"φ({a}×{b}) should equal φ({a})×φ({b})"
            
            # Divisor count is multiplicative
            tau_a = await divisor_power_sum(a, 0)
            tau_b = await divisor_power_sum(b, 0)
            tau_ab = await divisor_power_sum(a * b, 0)
            assert tau_ab == tau_a * tau_b, f"τ({a}×{b}) should equal τ({a})×τ({b})"

# ============================================================================
# PERFORMANCE AND ASYNC BEHAVIOR TESTS
# ============================================================================

class TestPerformance:
    """Performance and async behavior tests."""
    
    @pytest.mark.asyncio
    async def test_all_functions_are_async(self):
        """Test that all arithmetic functions are properly async."""
        operations = [
            euler_totient(12),
            jordan_totient(12, 2),
            mobius_function(30),
            little_omega(30),
            big_omega(30),
            divisor_power_sum(12, 1),
            von_mangoldt_function(8),
            liouville_function(12),
            carmichael_lambda(12),
            is_perfect_number(6),
            is_abundant_number(12),
            is_deficient_number(8)
        ]
        
        # Ensure all are coroutines
        for op in operations:
            assert asyncio.iscoroutine(op)
        
        # Run all operations concurrently
        results = await asyncio.gather(*operations)
        
        # Verify results have expected types and reasonable values
        assert all(isinstance(r, (int, float, bool)) for r in results)
        assert len(results) == len(operations)
    
    @pytest.mark.asyncio
    async def test_concurrent_execution(self):
        """Test that arithmetic operations can run concurrently."""
        start_time = time.time()
        
        # Run multiple operations concurrently
        tasks = []
        for i in range(1, 51):
            tasks.append(euler_totient(i))
            if i > 1:
                tasks.append(mobius_function(i))
                tasks.append(little_omega(i))
                tasks.append(big_omega(i))
                tasks.append(carmichael_lambda(i))
        
        results = await asyncio.gather(*tasks)
        duration = time.time() - start_time
        
        # Should complete quickly due to async nature
        assert duration < 2.0
        assert len(results) > 0
        
        # Check that Euler totient results are positive
        euler_totient_results = results[::5]  # Every 5th result should be Euler totient
        for result in euler_totient_results:
            if result is not None:
                assert result > 0, "Euler totient results should be positive"
    
    @pytest.mark.asyncio
    async def test_large_number_handling(self):
        """Test handling of moderately large numbers."""
        large_tests = [
            euler_totient(1000),
            mobius_function(997),      # Large prime
            little_omega(1024),        # Large power of 2
            big_omega(1000),
            carmichael_lambda(100),
            divisor_power_sum(100, 1),
            perfect_numbers_up_to(1000)
        ]
        
        results = await asyncio.gather(*large_tests)
        
        # Verify results are reasonable
        assert all(isinstance(r, (int, list)) for r in results)
        assert results[0] > 0        # φ(1000) > 0
        assert results[1] == -1      # μ(997) = -1 for prime
        assert results[2] == 1       # ω(1024) = 1 for power of 2
        assert results[3] > 0        # Ω(1000) > 0
        assert isinstance(results[6], list)  # Perfect numbers list

# ============================================================================
# ERROR HANDLING TESTS
# ============================================================================

class TestErrorHandling:
    """Test error handling and edge cases."""
    
    @pytest.mark.asyncio
    async def test_edge_case_handling(self):
        """Test edge case handling across all functions."""
        # Test with 0 and 1
        edge_cases = [0, 1]
        
        for n in edge_cases:
            # These should not raise exceptions
            await euler_totient(n)
            await jordan_totient(n, 1)
            await mobius_function(n)
            await little_omega(n)
            await big_omega(n)
            await divisor_power_sum(n, 0)
            await von_mangoldt_function(n)
            await liouville_function(n)
            await carmichael_lambda(n)
            await is_perfect_number(n)
            await is_abundant_number(n)
            await is_deficient_number(n)
    
    @pytest.mark.asyncio
    async def test_error_preserves_async_context(self):
        """Test that operations continue working after edge cases."""
        # Test edge cases don't break subsequent operations
        await euler_totient(0)
        result = await euler_totient(12)
        assert result == 4
        
        await mobius_function(0)
        result = await mobius_function(6)
        assert result == 1

# ============================================================================
# PARAMETRIZED TESTS
# ============================================================================

class TestParametrized:
    """Parametrized tests for comprehensive coverage."""
    
    @pytest.mark.asyncio
    @pytest.mark.parametrize("n,expected", [
        (1, 1), (2, 1), (3, 2), (4, 2), (5, 4), (6, 2), (8, 4), (9, 6), (10, 4), (12, 4)
    ])
    async def test_euler_totient_parametrized(self, n, expected):
        """Parametrized test for Euler totient calculation."""
        assert await euler_totient(n) == expected
    
    @pytest.mark.asyncio
    @pytest.mark.parametrize("n,expected", [
        (1, 1), (2, -1), (6, 1), (12, 0), (30, -1), (42, -1)
    ])
    async def test_mobius_function_parametrized(self, n, expected):
        """Parametrized test for Möbius function calculation."""
        assert await mobius_function(n) == expected
    
    @pytest.mark.asyncio
    @pytest.mark.parametrize("n,expected_little,expected_big", [
        (1, 0, 0), (6, 2, 2), (12, 2, 3), (18, 2, 3), (30, 3, 3), (60, 3, 4)
    ])
    async def test_omega_functions_parametrized(self, n, expected_little, expected_big):
        """Parametrized test for omega functions."""
        assert await little_omega(n) == expected_little
        assert await big_omega(n) == expected_big

if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v", "--tb=short", "--asyncio-mode=auto"])#!/usr/bin/env python3
# tests/math/arithmetic/number_theory/test_arithmetic_functions.py
"""
Comprehensive pytest test suite for arithmetic_functions.py module.

Tests cover:
- Euler's totient function (φ) and Jordan's totient function
- Möbius function (μ) and multiplicative properties
- Omega functions (ω and Ω) for prime factor counting
- Divisor functions (σ_k) for sums of powers of divisors
- Von Mangoldt function (Λ) for prime powers
- Liouville function (λ) and complete multiplicativity
- Carmichael function (λ) for exponent of multiplicative group
- Perfect, abundant, and deficient number classification
- Mathematical properties and relationships
- Edge cases, error conditions, and performance testing
- Async behavior verification
"""

import pytest
import asyncio
import time
import math
from typing import List, Tuple

# Import the functions to test
from chuk_mcp_functions.math.arithmetic.number_theory.arithmetic_functions import (
    # Multiplicative functions
    euler_totient, jordan_totient, mobius_function,
    
    # Additive functions
    little_omega, big_omega,
    
    # Divisor functions
    divisor_power_sum,
    
    # Special functions
    von_mangoldt_function, liouville_function, carmichael_lambda,
    
    # Perfect number functions
    is_perfect_number, is_abundant_number, is_deficient_number, perfect_numbers_up_to
)

# ============================================================================
# EULER'S TOTIENT FUNCTION TESTS
# ============================================================================

class TestEulerTotient:
    """Test cases for Euler's totient function and Jordan's generalization."""
    
    @pytest.mark.asyncio
    async def test_mobius_function_square_free_products(self):
        """Test Möbius function on square-free products."""
        # Products of distinct primes should alternate sign based on number of factors
        square_free_products = [
            (6, 1),    # 2×3 → μ(6) = (-1)² = 1
            (10, 1),   # 2×5 → μ(10) = (-1)² = 1
            (14, 1),   # 2×7 → μ(14) = (-1)² = 1
            (15, 1),   # 3×5 → μ(15) = (-1)² = 1
            (21, 1),   # 3×7 → μ(21) = (-1)² = 1
            (22, 1),   # 2×11 → μ(22) = (-1)² = 1
            (30, -1),  # 2×3×5 → μ(30) = (-1)³ = -1
            (42, -1),  # 2×3×7 → μ(42) = (-1)³ = -1
            (66, -1),  # 2×3×11 → μ(66) = (-1)³ = -1
            (70, -1),  # 2×5×7 → μ(70) = (-1)³ = -1
            (105, -1), # 3×5×7 → μ(105) = (-1)³ = -1
            (210, 1),  # 2×3×5×7 → μ(210) = (-1)⁴ = 1
        ]
        
        for n, expected in square_free_products:
            result = await mobius_function(n)
            assert result == expected, f"μ({n}) should be {expected}"
    
    @pytest.mark.asyncio
    async def test_mobius_function_edge_cases(self):
        """Test edge cases for Möbius function."""
        assert await mobius_function(0) == 0
        assert await mobius_function(1) == 1
    
    @pytest.mark.asyncio
    async def test_mobius_function_properties(self):
        """Test mathematical properties of Möbius function."""
        # μ(n) ∈ {-1, 0, 1} for all n
        for n in range(1, 51):
            mu_n = await mobius_function(n)
            assert mu_n in [-1, 0, 1], f"μ({n}) should be in {{-1, 0, 1}}"

# ============================================================================
# OMEGA FUNCTIONS TESTS
# ============================================================================

class TestOmegaFunctions:
    """Test cases for omega functions (prime factor counting)."""
    
    @pytest.mark.asyncio
    async def test_little_omega_known_values(self):
        """Test ω(n) - number of distinct prime factors."""
        known_values = [
            (1, 0),    # ω(1) = 0 (no prime factors)
            (2, 1),    # ω(2) = 1 (one prime: 2)
            (3, 1),    # ω(3) = 1 (one prime: 3)
            (4, 1),    # ω(4) = 1 (one distinct prime: 2)
            (6, 2),    # ω(6) = 2 (two primes: 2, 3)
            (8, 1),    # ω(8) = 1 (one distinct prime: 2)
            (12, 2),   # ω(12) = 2 (two distinct primes: 2, 3)
            (15, 2),   # ω(15) = 2 (two primes: 3, 5)
            (30, 3),   # ω(30) = 3 (three primes: 2, 3, 5)
            (42, 3),   # ω(42) = 3 (three primes: 2, 3, 7)
            (60, 3),   # ω(60) = 3 (three distinct primes: 2, 3, 5)
            (105, 3),  # ω(105) = 3 (three primes: 3, 5, 7)
            (210, 4),  # ω(210) = 4 (four primes: 2, 3, 5, 7)
        ]
        
        for n, expected in known_values:
            result = await little_omega(n)
            assert result == expected, f"ω({n}) should be {expected}, got {result}"
    
    @pytest.mark.asyncio
    async def test_big_omega_known_values(self):
        """Test Ω(n) - number of prime factors with multiplicity."""
        known_values = [
            (1, 0),    # Ω(1) = 0 (no prime factors)
            (2, 1),    # Ω(2) = 1 (one prime: 2)
            (3, 1),    # Ω(3) = 1 (one prime: 3)
            (4, 2),    # Ω(4) = 2 (factors: 2, 2)
            (6, 2),    # Ω(6) = 2 (factors: 2, 3)
            (8, 3),    # Ω(8) = 3 (factors: 2, 2, 2)
            (12, 3),   # Ω(12) = 3 (factors: 2, 2, 3)
            (15, 2),   # Ω(15) = 2 (factors: 3, 5)
            (16, 4),   # Ω(16) = 4 (factors: 2, 2, 2, 2)
            (18, 3),   # Ω(18) = 3 (factors: 2, 3, 3)
            (20, 3),   # Ω(20) = 3 (factors: 2, 2, 5)
            (24, 4),   # Ω(24) = 4 (factors: 2, 2, 2, 3)
            (30, 3),   # Ω(30) = 3 (factors: 2, 3, 5)
            (60, 4),   # Ω(60) = 4 (factors: 2, 2, 3, 5)
        ]
        
        for n, expected in known_values:
            result = await big_omega(n)
            assert result == expected, f"Ω({n}) should be {expected}, got {result}"
    
    @pytest.mark.asyncio
    async def test_omega_relationship(self):
        """Test relationship between ω(n) and Ω(n)."""
        # ω(n) ≤ Ω(n) for all n > 1
        # ω(n) = Ω(n) iff n is square-free
        
        for n in range(2, 51):
            little_omega_n = await little_omega(n)
            big_omega_n = await big_omega(n)
            
            assert little_omega_n <= big_omega_n, f"ω({n}) should be ≤ Ω({n})"
    
    @pytest.mark.asyncio
    async def test_omega_prime_powers(self):
        """Test omega functions on prime powers."""
        # For p^k: ω(p^k) = 1, Ω(p^k) = k
        prime_powers = [
            (4, 1, 2),   # 2² → ω=1, Ω=2
            (8, 1, 3),   # 2³ → ω=1, Ω=3
            (9, 1, 2),   # 3² → ω=1, Ω=2
            (16, 1, 4),  # 2⁴ → ω=1, Ω=4
            (25, 1, 2),  # 5² → ω=1, Ω=2
            (27, 1, 3),  # 3³ → ω=1, Ω=3
            (32, 1, 5),  # 2⁵ → ω=1, Ω=5
        ]
        
        for n, expected_little, expected_big in prime_powers:
            little_result = await little_omega(n)
            big_result = await big_omega(n)
            
            assert little_result == expected_little, f"ω({n}) should be {expected_little}"
            assert big_result == expected_big, f"Ω({n}) should be {expected_big}"
    
    @pytest.mark.asyncio
    async def test_omega_edge_cases(self):
        """Test edge cases for omega functions."""
        assert await little_omega(1) == 0
        assert await big_omega(1) == 0
        assert await little_omega(0) == 0
        assert await big_omega(0) == 0

# ============================================================================
# DIVISOR FUNCTIONS TESTS
# ============================================================================

class TestDivisorFunctions:
    """Test cases for divisor power sum functions."""
    
    @pytest.mark.asyncio
    async def test_divisor_power_sum_tau_function(self):
        """Test σ₀(n) = τ(n) - divisor count function."""
        known_tau_values = [
            (1, 1),    # τ(1) = 1
            (2, 2),    # τ(2) = 2
            (3, 2),    # τ(3) = 2
            (4, 3),    # τ(4) = 3
            (5, 2),    # τ(5) = 2
            (6, 4),    # τ(6) = 4
            (8, 4),    # τ(8) = 4
            (9, 3),    # τ(9) = 3
            (10, 4),   # τ(10) = 4
            (12, 6),   # τ(12) = 6
            (16, 5),   # τ(16) = 5
            (18, 6),   # τ(18) = 6
            (20, 6),   # τ(20) = 6
            (24, 8),   # τ(24) = 8
        ]
        
        for n, expected in known_tau_values:
            result = await divisor_power_sum(n, 0)
            assert result == expected, f"τ({n}) = σ₀({n}) should be {expected}, got {result}"
    
    @pytest.mark.asyncio
    async def test_divisor_power_sum_sigma_function(self):
        """Test σ₁(n) = σ(n) - sum of divisors function."""
        known_sigma_values = [
            (1, 1),    # σ(1) = 1
            (2, 3),    # σ(2) = 1 + 2 = 3
            (3, 4),    # σ(3) = 1 + 3 = 4
            (4, 7),    # σ(4) = 1 + 2 + 4 = 7
            (5, 6),    # σ(5) = 1 + 5 = 6
            (6, 12),   # σ(6) = 1 + 2 + 3 + 6 = 12
            (8, 15),   # σ(8) = 1 + 2 + 4 + 8 = 15
            (9, 13),   # σ(9) = 1 + 3 + 9 = 13
            (10, 18),  # σ(10) = 1 + 2 + 5 + 10 = 18
            (12, 28),  # σ(12) = 1 + 2 + 3 + 4 + 6 + 12 = 28
        ]
        
        for n, expected in known_sigma_values:
            result = await divisor_power_sum(n, 1)
            assert result == expected, f"σ({n}) = σ₁({n}) should be {expected}, got {result}"
    
    @pytest.mark.asyncio
    async def test_divisor_power_sum_higher_powers(self):
        """Test σₖ(n) for k > 1."""
        test_cases = [
            (6, 2, 50),   # σ₂(6) = 1² + 2² + 3² + 6² = 1 + 4 + 9 + 36 = 50
            (8, 2, 85),   # σ₂(8) = 1² + 2² + 4² + 8² = 1 + 4 + 16 + 64 = 85
            (4, 3, 73),   # σ₃(4) = 1³ + 2³ + 4³ = 1 + 8 + 64 = 73
        ]
        
        for n, k, expected in test_cases:
            result = await divisor_power_sum(n, k)
            assert result == expected, f"σ_{k}({n}) should be {expected}, got {result}"
    
    @pytest.mark.asyncio
    async def test_divisor_power_sum_edge_cases(self):
        """Test edge cases for divisor power sum."""
        assert await divisor_power_sum(0, 0) == 0
        assert await divisor_power_sum(1, 0) == 1
        assert await divisor_power_sum(1, 1) == 1
        assert await divisor_power_sum(1, 2) == 1

# ============================================================================
# VON MANGOLDT FUNCTION TESTS
# ============================================================================

class TestVonMangoldtFunction:
    """Test cases for the von Mangoldt function."""
    
    @pytest.mark.asyncio
    async def test_von_mangoldt_function_primes(self):
        """Test von Mangoldt function on primes."""
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
        
        for p in primes:
            result = await von_mangoldt_function(p)
            expected = math.log(p)
            assert abs(result - expected) < 1e-10, f"Λ({p}) should be ln({p}) ≈ {expected}"
    
    @pytest.mark.asyncio
    async def test_von_mangoldt_function_prime_powers(self):
        """Test von Mangoldt function on prime powers."""
        prime_powers = [
            (4, 2),   # 2² → Λ(4) = ln(2)
            (8, 2),   # 2³ → Λ(8) = ln(2)
            (9, 3),   # 3² → Λ(9) = ln(3)
            (16, 2),  # 2⁴ → Λ(16) = ln(2)
            (25, 5),  # 5² → Λ(25) = ln(5)
            (27, 3),  # 3³ → Λ(27) = ln(3)
            (32, 2),  # 2⁵ → Λ(32) = ln(2)
        ]
        
        for pk, p in prime_powers:
            result = await von_mangoldt_function(pk)
            expected = math.log(p)
            assert abs(result - expected) < 1e-10, f"Λ({pk}) should be ln({p}) ≈ {expected}"
    
    @pytest.mark.asyncio
    async def test_von_mangoldt_function_composites(self):
        """Test von Mangoldt function on composite numbers (not prime powers)."""
        composites = [6, 10, 12, 14, 15, 18, 20, 21, 22, 24, 26, 28, 30]
        
        for n in composites:
            result = await von_mangoldt_function(n)
            assert abs(result) < 1e-10, f"Λ({n}) should be 0 for composite {n}"
    
    @pytest.mark.asyncio
    async def test_von_mangoldt_function_edge_cases(self):
        """Test edge cases for von Mangoldt function."""
        assert await von_mangoldt_function(0) == 0.0
        assert await von_mangoldt_function(1) == 0.0

# ============================================================================
# LIOUVILLE FUNCTION TESTS
# ============================================================================

class TestLiouvilleFunction:
    """Test cases for the Liouville function."""
    
    @pytest.mark.asyncio
    async def test_liouville_function_known_values(self):
        """Test Liouville function with known values."""
        known_values = [
            (1, 1),    # λ(1) = 1 (Ω(1) = 0, (-1)⁰ = 1)
            (2, -1),   # λ(2) = -1 (Ω(2) = 1, (-1)¹ = -1)
            (3, -1),   # λ(3) = -1 (Ω(3) = 1, (-1)¹ = -1)
            (4, 1),    # λ(4) = 1 (Ω(4) = 2, (-1)² = 1)
            (5, -1),   # λ(5) = -1 (Ω(5) = 1, (-1)¹ = -1)
            (6, 1),    # λ(6) = 1 (Ω(6) = 2, (-1)² = 1)
            (7, -1),   # λ(7) = -1 (Ω(7) = 1, (-1)¹ = -1)
            (8, -1),   # λ(8) = -1 (Ω(8) = 3, (-1)³ = -1)
            (9, 1),    # λ(9) = 1 (Ω(9) = 2, (-1)² = 1)
            (10, 1),   # λ(10) = 1 (Ω(10) = 2, (-1)² = 1)
            (12, -1),  # λ(12) = -1 (Ω(12) = 3, (-1)³ = -1)
            (15, 1),   # λ(15) = 1 (Ω(15) = 2, (-1)² = 1)
            (16, 1),   # λ(16) = 1 (Ω(16) = 4, (-1)⁴ = 1)
        ]
        
        for n, expected in known_values:
            result = await liouville_function(n)
            assert result == expected, f"λ({n}) should be {expected}, got {result}"
    
    @pytest.mark.asyncio
    async def test_liouville_function_properties(self):
        """Test mathematical properties of Liouville function."""
        # λ(n) ∈ {-1, 1} for all n ≥ 1
        for n in range(1, 51):
            lambda_n = await liouville_function(n)
            assert lambda_n in [-1, 1], f"λ({n}) should be in {{-1, 1}}"
    
    @pytest.mark.asyncio
    async def test_liouville_function_completely_multiplicative(self):
        """Test that Liouville function is completely multiplicative."""
        # λ(mn) = λ(m)λ(n) for all m, n
        test_pairs = [(2, 3), (3, 5), (4, 5), (2, 7), (3, 8), (5, 6)]
        
        for m, n in test_pairs:
            lambda_m = await liouville_function(m)
            lambda_n = await liouville_function(n)
            lambda_mn = await liouville_function(m * n)
            
            assert lambda_mn == lambda_m * lambda_n, f"λ({m}×{n}) should equal λ({m})×λ({n})"
    
    @pytest.mark.asyncio
    async def test_liouville_function_edge_cases(self):
        """Test edge cases for Liouville function."""
        assert await liouville_function(0) == 1
        assert await liouville_function(1) == 1

# ============================================================================
# CARMICHAEL FUNCTION TESTS
# ============================================================================

class TestCarmichaelFunction:
    """Test cases for the Carmichael lambda function."""
    
    @pytest.mark.asyncio
    async def test_carmichael_lambda_known_values(self):
        """Test Carmichael function with known values."""
        known_values = [
            (1, 1),     # λ(1) = 1
            (2, 1),     # λ(2) = 1
            (3, 2),     # λ(3) = 2
            (4, 2),     # λ(4) = 2
            (5, 4),     # λ(5) = 4
            (6, 2),     # λ(6) = lcm(λ(2), λ(3)) = lcm(1, 2) = 2
            (7, 6),     # λ(7) = 6
            (8, 2),     # λ(8) = 2^(3-2) = 2
            (9, 6),     # λ(9) = 3×3^(2-1) = 6
            (10, 4),    # λ(10) = lcm(λ(2), λ(5)) = lcm(1, 4) = 4
            (12, 2),    # λ(12) = lcm(λ(4), λ(3)) = lcm(2, 2) = 2
            (15, 4),    # λ(15) = lcm(λ(3), λ(5)) = lcm(2, 4) = 4
            (16, 4),    # λ(16) = 2^(4-2) = 4
        ]
        
        for n, expected in known_values:
            result = await carmichael_lambda(n)
            assert result == expected, f"λ({n}) should be {expected}, got {result}"
    
    @pytest.mark.asyncio
    async def test_carmichael_lambda_primes(self):
        """Test that λ(p) = p-1 for odd primes p."""
        odd_primes = [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
        
        for p in odd_primes:
            result = await carmichael_lambda(p)
            expected = p - 1
            assert result == expected, f"λ({p}) should be {expected} for prime {p}"
    
    @pytest.mark.asyncio
    async def test_carmichael_lambda_powers_of_2(self):
        """Test Carmichael function on powers of 2."""
        powers_of_2_cases = [
            (4, 2),     # λ(2²) = 2
            (8, 2),     # λ(2³) = 2
            (16, 4),    # λ(2⁴) = 4
            (32, 8),    # λ(2⁵) = 8
            (64, 16),   # λ(2⁶) = 16
        ]
        
        for n, expected in powers_of_2_cases:
            result = await carmichael_lambda(n)
            assert result == expected, f"λ({n}) should be {expected}"
    
    @pytest.mark.asyncio
    async def test_carmichael_lambda_odd_prime_powers(self):
        """Test Carmichael function on odd prime powers."""
        odd_prime_powers = [
            (9, 6),     # λ(3²) = 3¹(3-1) = 6
            (25, 20),   # λ(5²) = 5¹(5-1) = 20
            (27, 18),   # λ(3³) = 3²(3-1) = 18
            (49, 42),   # λ(7²) = 7¹(7-1) = 42
            (125, 100), # λ(5³) = 5²(5-1) = 100
        ]
        
        for n, expected in odd_prime_powers:
            result = await carmichael_lambda(n)
            assert result == expected, f"λ({n}) should be {expected}"
    
    @pytest.mark.asyncio
    async def test_carmichael_lambda_divides_euler_totient(self):
        """Test that λ(n) divides φ(n)."""
        for n in range(2, 51):
            lambda_n = await carmichael_lambda(n)
            phi_n = await euler_totient(n)
            
            assert phi_n % lambda_n == 0, f"λ({n}) should divide φ({n})"
    
    @pytest.mark.asyncio
    async def test_carmichael_lambda_edge_cases(self):
        """Test edge cases for Carmichael function."""
        assert await carmichael_lambda(0) == 0
        assert await carmichael_lambda(1) == 1
        assert await carmichael_lambda(2) == 1

# ============================================================================
# PERFECT NUMBER FUNCTIONS TESTS
# ============================================================================

class TestPerfectNumberFunctions:
    """Test cases for perfect, abundant, and deficient number functions."""
    
    @pytest.mark.asyncio
    async def test_is_perfect_number_known_values(self):
        """Test perfect number identification."""
        perfect_numbers = [6, 28, 496, 8128]
        non_perfect_numbers = [1, 2, 3, 4, 5, 7, 8, 9, 10, 12, 24, 30, 100, 1000]
        
        for n in perfect_numbers:
            assert await is_perfect_number(n) == True, f"{n} should be perfect"
        
        for n in non_perfect_numbers:
            assert await is_perfect_number(n) == False, f"{n} should not be perfect"
    
    @pytest.mark.asyncio
    async def test_is_abundant_number_known_values(self):
        """Test abundant number identification."""
        abundant_numbers = [12, 18, 20, 24, 30, 36, 40, 42, 48, 54, 56, 60, 66, 70, 72, 78, 80]
        non_abundant_numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 19, 21, 22, 23]
        
        for n in abundant_numbers:
            assert await is_abundant_number(n) == True, f"{n} should be abundant"
        
        for n in non_abundant_numbers:
            assert await is_abundant_number(n) == False, f"{n} should not be abundant"
    
    @pytest.mark.asyncio
    async def test_is_deficient_number_known_values(self):
        """Test deficient number identification."""
        deficient_numbers = [1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 19, 21, 22, 23]
        non_deficient_numbers = [6, 12, 18, 20, 24, 28, 30]  # Perfect or abundant
        
        for n in deficient_numbers:
            assert await is_deficient_number(n) == True, f"{n} should be deficient"
        
        for n in non_deficient_numbers:
            assert await is_deficient_number(n) == False, f"{n} should not be deficient"
    
    @pytest.mark.asyncio
    async def test_number_classification_completeness(self):
        """Test that every number is exactly one of perfect, abundant, or deficient."""
        for n in range(1, 101):
            is_perfect = await is_perfect_number(n)
            is_abundant = await is_abundant_number(n)
            is_deficient = await is_deficient_number(n)
            
            # Exactly one should be true
            total = sum([is_perfect, is_abundant, is_deficient])
            assert total == 1, f"{n} should be exactly one of perfect/abundant/deficient"
    
    @pytest.mark.asyncio
    async def test_perfect_numbers_up_to_small(self):
        """Test finding perfect numbers up to small limits."""
        result_100 = await perfect_numbers_up_to(100)
        assert result_100 == [6, 28], f"Perfect numbers ≤ 100 should be [6, 28]"
        
        result_10 = await perfect_numbers_up_to(10)
        assert result_10 == [6], f"Perfect numbers ≤ 10 should be [6]"
        
        result_5 = await perfect_numbers_up_to(5)
        assert result_5 == [], f"Perfect numbers ≤ 5 should be []"
    
    @pytest.mark.asyncio
    async def test_perfect_number_functions_edge_cases(self):
        """Test edge cases for perfect number functions."""
        assert await is_perfect_number(0) == False
        assert await is_perfect_number(1) == False
        assert await is_abundant_number(0) == False
        assert await is_abundant_number(1) == False
        assert await is_deficient_number(0) == False
        assert await is_deficient_number(1) == True  # 1 is deficient

# ============================================================================
# INTEGRATION AND MATHEMATICAL PROPERTIES TESTS
# ============================================================================

class TestIntegrationAndProperties:
    """Integration tests and mathematical property verification."""
    
    @pytest.mark.asyncio
    async def test_mobius_inversion_formula(self):
        """Test Möbius inversion formula applications."""
        # Sum of μ(d) over all divisors d of n should be 1 if n=1, 0 otherwise
        for n in range(1, 31):
            # Import divisors function
            from chuk_mcp_functions.math.arithmetic.number_theory.divisibility import divisors
            
            divisors_n = await divisors(n)
            mobius_sum = sum([await mobius_function(d) for d in divisors_n])
            
            expected = 1 if n == 1 else 0
            assert mobius_sum == expected, f"Sum of μ(d) for d|{n} should be {expected}"
    
    @pytest.mark.asyncio
    async def test_euler_totient_sum_formula(self):
        """Test that sum of φ(d) over divisors of n equals n."""
        # ∑_{d|n} φ(d) = n
        for n in range(1, 31):
            from chuk_mcp_functions.math.arithmetic.number_theory.divisibility import divisors
            
            divisors_n = await divisors(n)
            phi_sum = sum([await euler_totient(d) for d in divisors_n])
            
            assert phi_sum == n, f"Sum of φ(d) for d|{n} should equal {n}"
    
    @pytest.mark.asyncio
    async def test_divisor_function_relationships(self):
        """Test relationships between different divisor functions."""
        for n in range(1, 21):
            # σ₀(n) = τ(n) should equal actual divisor count
            from chuk_mcp_functions.math.arithmetic.number_theory.divisibility import divisor_count, divisor_sum
            
            sigma_0 = await divisor_power_sum(n, 0)
            tau_n = await divisor_count(n)
            assert sigma_0 == tau_n, f"σ₀({n}) should equal τ({n})"
            
            # σ₁(n) = σ(n) should equal actual divisor sum
            sigma_1 = await divisor_power_sum(n, 1)
            sigma_n = await divisor_sum(n)
            assert sigma_1 == sigma_n, f"σ₁({n}) should equal σ({n})"
    
    @pytest.mark.asyncio
    async def test_multiplicative_function_properties(self):
        """Test multiplicative properties of arithmetic functions."""
        # Test that φ, λ (Carmichael), τ, σ are multiplicative
        coprime_pairs = [(3, 4), (5, 7), (8, 9), (7, 11)]
        
        for a, b in coprime_pairs:
            # Euler totient is multiplicative
            phi_a = await euler_totient(a)
            phi_b = await euler_totient(b)
            phi_ab = await euler_totient(a * b)
            assert phi_ab == phi_a * phi_b, f"φ({a}×{b}) should equal φ({a})×φ({b})"
            
            # Divisor count is multiplicative
            tau_a = await divisor_power_sum(a, 0)
            tau_b = await divisor_power_sum(b, 0)
            tau_ab = await divisor_power_sum(a * b, 0)
            assert tau_ab == tau_a * tau_b, f"τ({a}×{b}) should equal τ({a})×τ({b})"

# ============================================================================
# PERFORMANCE AND ASYNC BEHAVIOR TESTS
# ============================================================================

class TestPerformance:
    """Performance and async behavior tests."""
    
    @pytest.mark.asyncio
    async def test_all_functions_are_async(self):
        """Test that all arithmetic functions are properly async."""
        operations = [
            euler_totient(12),
            jordan_totient(12, 2),
            mobius_function(30),
            little_omega(30),
            big_omega(30),
            divisor_power_sum(12, 1),
            von_mangoldt_function(8),
            liouville_function(12),
            carmichael_lambda(12),
            is_perfect_number(6),
            is_abundant_number(12),
            is_deficient_number(8)
        ]
        
        # Ensure all are coroutines
        for op in operations:
            assert asyncio.iscoroutine(op)
        
        # Run all operations concurrently
        results = await asyncio.gather(*operations)
        
        # Verify results have expected types and reasonable values
        assert all(isinstance(r, (int, float, bool)) for r in results)
        assert len(results) == len(operations)
    
    @pytest.mark.asyncio
    async def test_concurrent_execution(self):
        """Test that arithmetic operations can run concurrently."""
        start_time = time.time()
        
        # Run multiple operations concurrently
        tasks = []
        for i in range(1, 51):
            tasks.append(euler_totient(i))
            if i > 1:
                tasks.append(mobius_function(i))
                tasks.append(little_omega(i))
                tasks.append(big_omega(i))
                tasks.append(carmichael_lambda(i))
        
        results = await asyncio.gather(*tasks)
        duration = time.time() - start_time
        
        # Should complete quickly due to async nature
        assert duration < 2.0
        assert len(results) > 0
        
        # Check that Euler totient results are positive
        euler_totient_results = results[::5]  # Every 5th result should be Euler totient
        for result in euler_totient_results:
            if result is not None:
                assert result > 0, "Euler totient results should be positive"
    
    @pytest.mark.asyncio
    async def test_large_number_handling(self):
        """Test handling of moderately large numbers."""
        large_tests = [
            euler_totient(1000),
            mobius_function(997),      # Large prime
            little_omega(1024),        # Large power of 2
            big_omega(1000),
            carmichael_lambda(100),
            divisor_power_sum(100, 1),
            perfect_numbers_up_to(1000)
        ]
        
        results = await asyncio.gather(*large_tests)
        
        # Verify results are reasonable
        assert all(isinstance(r, (int, list)) for r in results)
        assert results[0] > 0        # φ(1000) > 0
        assert results[1] == -1      # μ(997) = -1 for prime
        assert results[2] == 1       # ω(1024) = 1 for power of 2
        assert results[3] > 0        # Ω(1000) > 0
        assert isinstance(results[6], list)  # Perfect numbers list

# ============================================================================
# ERROR HANDLING TESTS
# ============================================================================

class TestErrorHandling:
    """Test error handling and edge cases."""
    
    @pytest.mark.asyncio
    async def test_edge_case_handling(self):
        """Test edge case handling across all functions."""
        # Test with 0 and 1
        edge_cases = [0, 1]
        
        for n in edge_cases:
            # These should not raise exceptions
            await euler_totient(n)
            await jordan_totient(n, 1)
            await mobius_function(n)
            await little_omega(n)
            await big_omega(n)
            await divisor_power_sum(n, 0)
            await von_mangoldt_function(n)
            await liouville_function(n)
            await carmichael_lambda(n)
            await is_perfect_number(n)
            await is_abundant_number(n)
            await is_deficient_number(n)
    
    @pytest.mark.asyncio
    async def test_error_preserves_async_context(self):
        """Test that operations continue working after edge cases."""
        # Test edge cases don't break subsequent operations
        await euler_totient(0)
        result = await euler_totient(12)
        assert result == 4
        
        await mobius_function(0)
        result = await mobius_function(6)
        assert result == 1

# ============================================================================
# PARAMETRIZED TESTS
# ============================================================================

class TestParametrized:
    """Parametrized tests for comprehensive coverage."""
    
    @pytest.mark.asyncio
    @pytest.mark.parametrize("n,expected", [
        (1, 1), (2, 1), (3, 2), (4, 2), (5, 4), (6, 2), (8, 4), (9, 6), (10, 4), (12, 4)
    ])
    async def test_euler_totient_parametrized(self, n, expected):
        """Parametrized test for Euler totient calculation."""
        assert await euler_totient(n) == expected
    
    @pytest.mark.asyncio
    @pytest.mark.parametrize("n,expected", [
        (1, 1), (2, -1), (6, 1), (12, 0), (30, -1), (42, -1)
    ])
    async def test_mobius_function_parametrized(self, n, expected):
        """Parametrized test for Möbius function calculation."""
        assert await mobius_function(n) == expected
    
    @pytest.mark.asyncio
    @pytest.mark.parametrize("n,expected_little,expected_big", [
        (1, 0, 0), (6, 2, 2), (12, 2, 3), (18, 2, 3), (30, 3, 3), (60, 3, 4)
    ])
    async def test_omega_functions_parametrized(self, n, expected_little, expected_big):
        """Parametrized test for omega functions."""
        assert await little_omega(n) == expected_little
        assert await big_omega(n) == expected_big

if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v", "--tb=short", "--asyncio-mode=auto"])
    async def test_euler_totient_known_values(self):
        """Test Euler's totient function with known values."""
        known_values = [
            (1, 1),    # φ(1) = 1
            (2, 1),    # φ(2) = 1
            (3, 2),    # φ(3) = 2
            (4, 2),    # φ(4) = 2
            (5, 4),    # φ(5) = 4
            (6, 2),    # φ(6) = 2
            (7, 6),    # φ(7) = 6
            (8, 4),    # φ(8) = 4
            (9, 6),    # φ(9) = 6
            (10, 4),   # φ(10) = 4
            (12, 4),   # φ(12) = 4
            (15, 8),   # φ(15) = 8
            (16, 8),   # φ(16) = 8
            (18, 6),   # φ(18) = 6
            (20, 8),   # φ(20) = 8
            (24, 8),   # φ(24) = 8
            (30, 8),   # φ(30) = 8
        ]
        
        for n, expected in known_values:
            result = await euler_totient(n)
            assert result == expected, f"φ({n}) should be {expected}, got {result}"
    
    @pytest.mark.asyncio
    async def test_euler_totient_prime_property(self):
        """Test that φ(p) = p-1 for prime p."""
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
        
        for p in primes:
            result = await euler_totient(p)
            expected = p - 1
            assert result == expected, f"φ({p}) should be {expected} for prime {p}"
    
    @pytest.mark.asyncio
    async def test_euler_totient_prime_power_property(self):
        """Test that φ(p^k) = p^k - p^(k-1) = p^(k-1)(p-1)."""
        test_cases = [
            (4, 2),   # φ(2²) = 2²⁻¹(2-1) = 2
            (8, 4),   # φ(2³) = 2²(2-1) = 4
            (9, 6),   # φ(3²) = 3¹(3-1) = 6
            (16, 8),  # φ(2⁴) = 2³(2-1) = 8
            (25, 20), # φ(5²) = 5¹(5-1) = 20
            (27, 18), # φ(3³) = 3²(3-1) = 18
            (32, 16), # φ(2⁵) = 2⁴(2-1) = 16
        ]
        
        for n, expected in test_cases:
            result = await euler_totient(n)
            assert result == expected, f"φ({n}) should be {expected}"
    
    @pytest.mark.asyncio
    async def test_euler_totient_multiplicative_property(self):
        """Test that φ(mn) = φ(m)φ(n) when gcd(m,n) = 1."""
        coprime_pairs = [(3, 4), (5, 7), (8, 9), (7, 11), (9, 16), (5, 8)]
        
        for m, n in coprime_pairs:
            phi_m = await euler_totient(m)
            phi_n = await euler_totient(n)
            phi_mn = await euler_totient(m * n)
            
            assert phi_mn == phi_m * phi_n, f"φ({m}×{n}) should equal φ({m})×φ({n})"
    
    @pytest.mark.asyncio
    async def test_euler_totient_edge_cases(self):
        """Test edge cases for Euler's totient function."""
        assert await euler_totient(0) == 0
        assert await euler_totient(1) == 1
    
    @pytest.mark.asyncio
    async def test_jordan_totient_known_values(self):
        """Test Jordan's totient function with known values."""
        known_values = [
            ((4, 1), 2),    # J₁(4) = φ(4) = 2
            ((4, 2), 12),   # J₂(4) = 16 × (3/4) = 12
            ((6, 1), 2),    # J₁(6) = φ(6) = 2
            ((6, 2), 24),   # J₂(6) = 36 × (3/4) × (8/9) = 24
            ((8, 1), 4),    # J₁(8) = φ(8) = 4
            ((8, 2), 48),   # J₂(8) = 64 × (3/4) = 48
            ((9, 1), 6),    # J₁(9) = φ(9) = 6
            ((9, 2), 72),   # J₂(9) = 81 × (8/9) = 72
            ((12, 1), 4),   # J₁(12) = φ(12) = 4
        ]
        
        for (n, k), expected in known_values:
            result = await jordan_totient(n, k)
            assert result == expected, f"J_{k}({n}) should be {expected}, got {result}"
    
    @pytest.mark.asyncio
    async def test_jordan_totient_reduces_to_euler(self):
        """Test that J₁(n) = φ(n)."""
        test_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 16, 18, 20]
        
        for n in test_values:
            jordan_1 = await jordan_totient(n, 1)
            euler_phi = await euler_totient(n)
            assert jordan_1 == euler_phi, f"J₁({n}) should equal φ({n})"
    
    @pytest.mark.asyncio
    async def test_jordan_totient_edge_cases(self):
        """Test edge cases for Jordan's totient function."""
        assert await jordan_totient(0, 1) == 0
        assert await jordan_totient(1, 1) == 1
        assert await jordan_totient(1, 2) == 1
        assert await jordan_totient(0, 2) == 0

# ============================================================================
# MÖBIUS FUNCTION TESTS
# ============================================================================

class TestMobiusFunction:
    """Test cases for the Möbius function."""
    
    @pytest.mark.asyncio
    async def test_mobius_function_known_values(self):
        """Test Möbius function with known values."""
        known_values = [
            (1, 1),     # μ(1) = 1
            (2, -1),    # μ(2) = -1 (1 prime)
            (3, -1),    # μ(3) = -1 (1 prime)
            (4, 0),     # μ(4) = 0 (2² has repeated prime)
            (5, -1),    # μ(5) = -1 (1 prime)
            (6, 1),     # μ(6) = 1 (2×3, 2 distinct primes)
            (7, -1),    # μ(7) = -1 (1 prime)
            (8, 0),     # μ(8) = 0 (2³ has repeated prime)
            (9, 0),     # μ(9) = 0 (3² has repeated prime)
            (10, 1),    # μ(10) = 1 (2×5, 2 distinct primes)
            (12, 0),    # μ(12) = 0 (2²×3 has repeated prime)
            (15, 1),    # μ(15) = 1 (3×5, 2 distinct primes)
            (30, -1),   # μ(30) = -1 (2×3×5, 3 distinct primes)
            (42, -1),   # μ(42) = -1 (2×3×7, 3 distinct primes)
        ]
        
        for n, expected in known_values:
            result = await mobius_function(n)
            assert result == expected, f"μ({n}) should be {expected}, got {result}"
    
    @pytest.mark.asyncio
    async def test_mobius_function_prime_powers(self):
        """Test Möbius function on prime powers."""
        # μ(p^k) = 0 for k > 1, μ(p) = -1 for prime p
        prime_powers = [4, 8, 16, 32, 9, 27, 25, 125, 49, 121]
        
        for pk in prime_powers:
            result = await mobius_function(pk)
            assert result == 0, f"μ({pk}) should be 0 for prime power > 1"
    
    @pytest.mark.asyncio
    async def test_euler_totient_known_values(self):
        """Test Euler's totient function with known values."""
        known_values = [
            (1, 1),    # φ(1) = 1
            (2, 1),    # φ(2) = 1
            (3, 2),    # φ(3) = 2
            (4, 2),    # φ(4) = 2
            (5, 4),    # φ(5) = 4
            (6, 2),    # φ(6) = 2
            (7, 6),    # φ(7) = 6
            (8, 4),    # φ(8) = 4
            (9, 6),    # φ(9) = 6
            (10, 4),   # φ(10) = 4
            (12, 4),   # φ(12) = 4
            (15, 8),   # φ(15) = 8
            (16, 8),   # φ(16) = 8
            (18, 6),   # φ(18) = 6
            (20, 8),   # φ(20) = 8
            (24, 8),   # φ(24) = 8
            (30, 8),   # φ(30) = 8
        ]
        
        for n, expected in known_values:
            result = await euler_totient(n)
            assert result == expected, f"φ({n}) should be {expected}, got {result}"
    
    @pytest.mark.asyncio
    async def test_euler_totient_prime_property(self):
        """Test that φ(p) = p-1 for prime p."""
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
        
        for p in primes:
            result = await euler_totient(p)
            expected = p - 1
            assert result == expected, f"φ({p}) should be {expected} for prime {p}"
    
    @pytest.mark.asyncio
    async def test_euler_totient_prime_power_property(self):
        """Test that φ(p^k) = p^k - p^(k-1) = p^(k-1)(p-1)."""
        test_cases = [
            (4, 2),   # φ(2²) = 2²⁻¹(2-1) = 2
            (8, 4),   # φ(2³) = 2²(2-1) = 4
            (9, 6),   # φ(3²) = 3¹(3-1) = 6
            (16, 8),  # φ(2⁴) = 2³(2-1) = 8
            (25, 20), # φ(5²) = 5¹(5-1) = 20
            (27, 18), # φ(3³) = 3²(3-1) = 18
            (32, 16), # φ(2⁵) = 2⁴(2-1) = 16
        ]
        
        for n, expected in test_cases:
            result = await euler_totient(n)
            assert result == expected, f"φ({n}) should be {expected}"
    
    @pytest.mark.asyncio
    async def test_euler_totient_multiplicative_property(self):
        """Test that φ(mn) = φ(m)φ(n) when gcd(m,n) = 1."""
        coprime_pairs = [(3, 4), (5, 7), (8, 9), (7, 11), (9, 16), (5, 8)]
        
        for m, n in coprime_pairs:
            phi_m = await euler_totient(m)
            phi_n = await euler_totient(n)
            phi_mn = await euler_totient(m * n)
            
            assert phi_mn == phi_m * phi_n, f"φ({m}×{n}) should equal φ({m})×φ({n})"
    
    @pytest.mark.asyncio
    async def test_euler_totient_edge_cases(self):
        """Test edge cases for Euler's totient function."""
        assert await euler_totient(0) == 0
        assert await euler_totient(1) == 1
    
    @pytest.mark.asyncio
    async def test_jordan_totient_known_values(self):
        """Test Jordan's totient function with known values."""
        known_values = [
            ((4, 1), 2),    # J₁(4) = φ(4) = 2
            ((4, 2), 12),   # J₂(4) = 16 × (3/4) = 12
            ((6, 1), 2),    # J₁(6) = φ(6) = 2
            ((6, 2), 24),   # J₂(6) = 36 × (3/4) × (8/9) = 24
            ((8, 1), 4),    # J₁(8) = φ(8) = 4
            ((8, 2), 48),   # J₂(8) = 64 × (3/4) = 48
            ((9, 1), 6),    # J₁(9) = φ(9) = 6
            ((9, 2), 72),   # J₂(9) = 81 × (8/9) = 72
            ((12, 1), 4),   # J₁(12) = φ(12) = 4
        ]
        
        for (n, k), expected in known_values:
            result = await jordan_totient(n, k)
            assert result == expected, f"J_{k}({n}) should be {expected}, got {result}"
    
    @pytest.mark.asyncio
    async def test_jordan_totient_reduces_to_euler(self):
        """Test that J₁(n) = φ(n)."""
        test_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 16, 18, 20]
        
        for n in test_values:
            jordan_1 = await jordan_totient(n, 1)
            euler_phi = await euler_totient(n)
            assert jordan_1 == euler_phi, f"J₁({n}) should equal φ({n})"
    
    @pytest.mark.asyncio
    async def test_jordan_totient_edge_cases(self):
        """Test edge cases for Jordan's totient function."""
        assert await jordan_totient(0, 1) == 0
        assert await jordan_totient(1, 1) == 1
        assert await jordan_totient(1, 2) == 1
        assert await jordan_totient(0, 2) == 0

# ============================================================================
# MÖBIUS FUNCTION TESTS
# ============================================================================

class TestMobiusFunction:
    """Test cases for the Möbius function."""
    
    @pytest.mark.asyncio
    async def test_mobius_function_known_values(self):
        """Test Möbius function with known values."""
        known_values = [
            (1, 1),     # μ(1) = 1
            (2, -1),    # μ(2) = -1 (1 prime)
            (3, -1),    # μ(3) = -1 (1 prime)
            (4, 0),     # μ(4) = 0 (2² has repeated prime)
            (5, -1),    # μ(5) = -1 (1 prime)
            (6, 1),     # μ(6) = 1 (2×3, 2 distinct primes)
            (7, -1),    # μ(7) = -1 (1 prime)
            (8, 0),     # μ(8) = 0 (2³ has repeated prime)
            (9, 0),     # μ(9) = 0 (3² has repeated prime)
            (10, 1),    # μ(10) = 1 (2×5, 2 distinct primes)
            (12, 0),    # μ(12) = 0 (2²×3 has repeated prime)
            (15, 1),    # μ(15) = 1 (3×5, 2 distinct primes)
            (30, -1),   # μ(30) = -1 (2×3×5, 3 distinct primes)
            (42, -1),   # μ(42) = -1 (2×3×7, 3 distinct primes)
        ]
        
        for n, expected in known_values:
            result = await mobius_function(n)
            assert result == expected, f"μ({n}) should be {expected}, got {result}"
    
    @pytest.mark.asyncio
    async def test_mobius_function_prime_powers(self):
        """Test Möbius function on prime powers."""
        # μ(p^k) = 0 for k > 1, μ(p) = -1 for prime p
        prime_powers = [4, 8, 16, 32, 9, 27, 25, 125, 49, 121]
        
        for pk in prime_powers:
            result = await mobius_function(pk)
            assert result == 0, f"μ({pk}) should be 0 for prime power > 1"
    