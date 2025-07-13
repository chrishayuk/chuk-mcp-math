#!/usr/bin/env python3
# tests/math/arithmetic/number_theory/test_primes.py
"""
Comprehensive pytest unit tests for prime number operations.

Tests cover:
- Prime testing with various inputs including edge cases
- Prime generation (next_prime, nth_prime, first_n_primes)
- Prime factorization with different number types
- Prime counting function
- Coprimality testing
- Performance and async behavior
- Edge cases and error conditions
- Mathematical properties and relationships
"""

import pytest
import math
import asyncio
import time
from typing import List

# Import the functions to test
from chuk_mcp_functions.math.arithmetic.number_theory.primes import (
    is_prime, next_prime, nth_prime, prime_factors,
    prime_count, is_coprime, first_n_primes
)

class TestIsPrime:
    """Test cases for the is_prime function."""
    
    @pytest.mark.asyncio
    async def test_is_prime_basic_cases(self):
        """Test prime checking with basic cases."""
        assert await is_prime(17) == True
        assert await is_prime(4) == False
        assert await is_prime(2) == True
        assert await is_prime(3) == True
        assert await is_prime(5) == True
        assert await is_prime(7) == True
        assert await is_prime(11) == True
        assert await is_prime(13) == True
    
    @pytest.mark.asyncio
    async def test_is_prime_composite_numbers(self):
        """Test prime checking with composite numbers."""
        assert await is_prime(4) == False   # 2²
        assert await is_prime(6) == False   # 2×3
        assert await is_prime(8) == False   # 2³
        assert await is_prime(9) == False   # 3²
        assert await is_prime(10) == False  # 2×5
        assert await is_prime(12) == False  # 2²×3
        assert await is_prime(15) == False  # 3×5
        assert await is_prime(21) == False  # 3×7
        assert await is_prime(25) == False  # 5²
        assert await is_prime(27) == False  # 3³
    
    @pytest.mark.asyncio
    async def test_is_prime_edge_cases(self):
        """Test prime checking with edge cases."""
        assert await is_prime(1) == False   # 1 is not prime by definition
        assert await is_prime(0) == False   # 0 is not prime
        assert await is_prime(-1) == False  # Negative numbers are not prime
        assert await is_prime(-5) == False  # Negative prime-like number
        assert await is_prime(-17) == False # Negative prime
    
    @pytest.mark.asyncio
    async def test_is_prime_small_primes(self):
        """Test the first 25 prime numbers."""
        first_25_primes = [
            2, 3, 5, 7, 11, 13, 17, 19, 23, 29,
            31, 37, 41, 43, 47, 53, 59, 61, 67, 71,
            73, 79, 83, 89, 97
        ]
        
        for p in first_25_primes:
            assert await is_prime(p) == True, f"{p} should be prime"
    
    @pytest.mark.asyncio
    async def test_is_prime_large_primes(self):
        """Test with some larger known primes."""
        large_primes = [101, 103, 107, 109, 113, 127, 131, 137, 139, 149,
                       151, 157, 163, 167, 173, 179, 181, 191, 193, 197,
                       199, 211, 223, 227, 229, 233, 239, 241, 251, 257]
        
        for p in large_primes:
            assert await is_prime(p) == True, f"{p} should be prime"
    
    @pytest.mark.asyncio
    async def test_is_prime_large_composites(self):
        """Test with larger composite numbers."""
        large_composites = [
            100, 102, 104, 105, 106, 108, 110, 111, 112, 114,
            115, 116, 117, 118, 119, 120, 121, 122, 123, 124,
            125, 126, 128, 129, 130, 132, 133, 134, 135, 136
        ]
        
        for c in large_composites:
            assert await is_prime(c) == False, f"{c} should not be prime"
    
    @pytest.mark.asyncio
    async def test_is_prime_perfect_squares(self):
        """Test that perfect squares (except 1) are not prime."""
        squares = [4, 9, 16, 25, 36, 49, 64, 81, 100, 121, 144, 169, 196, 225, 256]
        
        for square in squares:
            assert await is_prime(square) == False, f"{square} = {int(math.sqrt(square))}² should not be prime"
    
    @pytest.mark.asyncio
    async def test_is_prime_mersenne_numbers(self):
        """Test some Mersenne numbers (2^n - 1)."""
        # Some Mersenne primes
        mersenne_primes = [3, 7, 31, 127]  # 2^2-1, 2^3-1, 2^5-1, 2^7-1
        
        for mp in mersenne_primes:
            assert await is_prime(mp) == True, f"Mersenne number {mp} should be prime"
        
        # Some composite Mersenne numbers
        mersenne_composites = [15, 63, 255, 511]  # 2^4-1, 2^6-1, 2^8-1, 2^9-1
        
        for mc in mersenne_composites:
            assert await is_prime(mc) == False, f"Mersenne number {mc} should be composite"

class TestNextPrime:
    """Test cases for the next_prime function."""
    
    @pytest.mark.asyncio
    async def test_next_prime_basic_cases(self):
        """Test finding next prime with basic cases."""
        assert await next_prime(10) == 11
        assert await next_prime(17) == 19
        assert await next_prime(1) == 2
        assert await next_prime(2) == 3
        assert await next_prime(3) == 5
        assert await next_prime(5) == 7
        assert await next_prime(7) == 11
    
    @pytest.mark.asyncio
    async def test_next_prime_after_primes(self):
        """Test finding next prime after known primes."""
        primes_and_next = [
            (2, 3), (3, 5), (5, 7), (7, 11), (11, 13),
            (13, 17), (17, 19), (19, 23), (23, 29), (29, 31),
            (31, 37), (37, 41), (41, 43), (43, 47), (47, 53)
        ]
        
        for prime, expected_next in primes_and_next:
            assert await next_prime(prime) == expected_next
    
    @pytest.mark.asyncio
    async def test_next_prime_after_composites(self):
        """Test finding next prime after composite numbers."""
        composites_and_next = [
            (4, 5), (6, 7), (8, 11), (9, 11), (10, 11),
            (12, 13), (14, 17), (15, 17), (16, 17), (18, 19),
            (20, 23), (21, 23), (22, 23), (24, 29), (25, 29)
        ]
        
        for composite, expected_next in composites_and_next:
            assert await next_prime(composite) == expected_next
    
    @pytest.mark.asyncio
    async def test_next_prime_large_gaps(self):
        """Test next prime for numbers with larger prime gaps."""
        # Test some numbers before larger prime gaps
        assert await next_prime(113) == 127  # Gap of 14
        assert await next_prime(126) == 127
        assert await next_prime(199) == 211  # Gap of 12
        assert await next_prime(210) == 211
    
    @pytest.mark.asyncio
    async def test_next_prime_edge_cases(self):
        """Test next prime with edge cases."""
        assert await next_prime(0) == 2
        assert await next_prime(-1) == 2
        assert await next_prime(-10) == 2
    
    @pytest.mark.asyncio
    async def test_next_prime_sequential(self):
        """Test that calling next_prime repeatedly gives correct sequence."""
        current = 1
        expected_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
        
        for expected in expected_primes:
            current = await next_prime(current)
            assert current == expected

class TestNthPrime:
    """Test cases for the nth_prime function."""
    
    @pytest.mark.asyncio
    async def test_nth_prime_basic_cases(self):
        """Test finding nth prime with basic cases."""
        assert await nth_prime(1) == 2
        assert await nth_prime(2) == 3
        assert await nth_prime(3) == 5
        assert await nth_prime(4) == 7
        assert await nth_prime(5) == 11
        assert await nth_prime(10) == 29
    
    @pytest.mark.asyncio
    async def test_nth_prime_first_25(self):
        """Test the first 25 primes by position."""
        expected_primes = [
            2, 3, 5, 7, 11, 13, 17, 19, 23, 29,
            31, 37, 41, 43, 47, 53, 59, 61, 67, 71,
            73, 79, 83, 89, 97
        ]
        
        for i, expected in enumerate(expected_primes, 1):
            result = await nth_prime(i)
            assert result == expected, f"The {i}th prime should be {expected}, got {result}"
    
    @pytest.mark.asyncio
    async def test_nth_prime_larger_positions(self):
        """Test nth prime for larger positions."""
        # Some known nth primes
        known_nth_primes = [
            (25, 97), (50, 229), (100, 541)
        ]
        
        for n, expected in known_nth_primes:
            result = await nth_prime(n)
            assert result == expected, f"The {n}th prime should be {expected}, got {result}"
    
    @pytest.mark.asyncio
    async def test_nth_prime_invalid_input(self):
        """Test nth_prime with invalid input."""
        with pytest.raises(ValueError, match="n must be positive"):
            await nth_prime(0)
        
        with pytest.raises(ValueError, match="n must be positive"):
            await nth_prime(-1)
        
        with pytest.raises(ValueError, match="n must be positive"):
            await nth_prime(-5)
    
    @pytest.mark.asyncio
    async def test_nth_prime_consistency_with_sequence(self):
        """Test that nth_prime is consistent with prime sequence."""
        # Generate first 10 primes using nth_prime
        primes_by_nth = []
        for i in range(1, 11):
            primes_by_nth.append(await nth_prime(i))
        
        # They should be in ascending order
        assert primes_by_nth == sorted(primes_by_nth)
        
        # Each should be prime
        for p in primes_by_nth:
            assert await is_prime(p) == True

class TestPrimeFactors:
    """Test cases for the prime_factors function."""
    
    @pytest.mark.asyncio
    async def test_prime_factors_basic_cases(self):
        """Test prime factorization with basic cases."""
        assert await prime_factors(12) == [2, 2, 3]
        assert await prime_factors(17) == [17]  # Prime number
        assert await prime_factors(60) == [2, 2, 3, 5]
        assert await prime_factors(2) == [2]
        assert await prime_factors(3) == [3]
        assert await prime_factors(4) == [2, 2]
        assert await prime_factors(6) == [2, 3]
    
    @pytest.mark.asyncio
    async def test_prime_factors_edge_cases(self):
        """Test prime factorization with edge cases."""
        assert await prime_factors(1) == []  # 1 has no prime factors
        assert await prime_factors(0) == []  # 0 has no prime factors
        assert await prime_factors(-5) == [] # Negative numbers handled
    
    @pytest.mark.asyncio
    async def test_prime_factors_powers_of_primes(self):
        """Test factorization of prime powers."""
        assert await prime_factors(8) == [2, 2, 2]     # 2³
        assert await prime_factors(9) == [3, 3]        # 3²
        assert await prime_factors(16) == [2, 2, 2, 2] # 2⁴
        assert await prime_factors(25) == [5, 5]       # 5²
        assert await prime_factors(27) == [3, 3, 3]    # 3³
        assert await prime_factors(32) == [2, 2, 2, 2, 2] # 2⁵
    
    @pytest.mark.asyncio
    async def test_prime_factors_products_of_primes(self):
        """Test factorization of products of distinct primes."""
        assert await prime_factors(6) == [2, 3]        # 2×3
        assert await prime_factors(10) == [2, 5]       # 2×5
        assert await prime_factors(14) == [2, 7]       # 2×7
        assert await prime_factors(15) == [3, 5]       # 3×5
        assert await prime_factors(21) == [3, 7]       # 3×7
        assert await prime_factors(35) == [5, 7]       # 5×7
        assert await prime_factors(30) == [2, 3, 5]    # 2×3×5
        assert await prime_factors(42) == [2, 3, 7]    # 2×3×7
        assert await prime_factors(70) == [2, 5, 7]    # 2×5×7
        assert await prime_factors(105) == [3, 5, 7]   # 3×5×7
    
    @pytest.mark.asyncio
    async def test_prime_factors_mixed_cases(self):
        """Test factorization of numbers with repeated and distinct factors."""
        assert await prime_factors(18) == [2, 3, 3]    # 2×3²
        assert await prime_factors(20) == [2, 2, 5]    # 2²×5
        assert await prime_factors(24) == [2, 2, 2, 3] # 2³×3
        assert await prime_factors(36) == [2, 2, 3, 3] # 2²×3²
        assert await prime_factors(40) == [2, 2, 2, 5] # 2³×5
        assert await prime_factors(45) == [3, 3, 5]    # 3²×5
        assert await prime_factors(48) == [2, 2, 2, 2, 3] # 2⁴×3
        assert await prime_factors(50) == [2, 5, 5]    # 2×5²
        assert await prime_factors(72) == [2, 2, 2, 3, 3] # 2³×3²
    
    @pytest.mark.asyncio
    async def test_prime_factors_large_numbers(self):
        """Test factorization of larger numbers."""
        # Test some larger numbers with known factorizations
        assert await prime_factors(100) == [2, 2, 5, 5]     # 2²×5²
        assert await prime_factors(120) == [2, 2, 2, 3, 5]  # 2³×3×5
        assert await prime_factors(150) == [2, 3, 5, 5]     # 2×3×5²
        assert await prime_factors(180) == [2, 2, 3, 3, 5]  # 2²×3²×5
        assert await prime_factors(200) == [2, 2, 2, 5, 5]  # 2³×5²
        assert await prime_factors(210) == [2, 3, 5, 7]     # 2×3×5×7
    
    @pytest.mark.asyncio
    async def test_prime_factors_verification(self):
        """Test that prime factors multiply back to original number."""
        test_numbers = [12, 18, 24, 30, 36, 42, 48, 60, 72, 84, 90, 96]
        
        for n in test_numbers:
            factors = await prime_factors(n)
            if factors:  # Skip empty factor lists
                product = 1
                for factor in factors:
                    product *= factor
                assert product == n, f"Product of factors {factors} should equal {n}, got {product}"
    
    @pytest.mark.asyncio
    async def test_prime_factors_all_prime(self):
        """Test that all returned factors are actually prime."""
        test_numbers = [12, 18, 24, 30, 36, 42, 48, 60, 72, 84, 90, 96, 120, 150, 180, 210]
        
        for n in test_numbers:
            factors = await prime_factors(n)
            for factor in factors:
                assert await is_prime(factor) == True, f"Factor {factor} of {n} should be prime"
    
    @pytest.mark.asyncio
    async def test_prime_factors_sorted_order(self):
        """Test that prime factors are returned in sorted order."""
        test_numbers = [12, 18, 24, 30, 36, 42, 48, 60, 72, 84, 90, 96, 120, 150, 180, 210]
        
        for n in test_numbers:
            factors = await prime_factors(n)
            assert factors == sorted(factors), f"Factors {factors} should be sorted"

class TestPrimeCount:
    """Test cases for the prime_count function."""
    
    @pytest.mark.asyncio
    async def test_prime_count_basic_cases(self):
        """Test prime counting with basic cases."""
        assert await prime_count(10) == 4   # 2, 3, 5, 7
        assert await prime_count(20) == 8   # 2, 3, 5, 7, 11, 13, 17, 19
        assert await prime_count(2) == 1    # 2
        assert await prime_count(1) == 0    # No primes ≤ 1
        assert await prime_count(0) == 0    # No primes ≤ 0
    
    @pytest.mark.asyncio
    async def test_prime_count_edge_cases(self):
        """Test prime counting with edge cases."""
        assert await prime_count(-1) == 0   # No primes ≤ -1
        assert await prime_count(-10) == 0  # No primes ≤ -10
    
    @pytest.mark.asyncio
    async def test_prime_count_known_values(self):
        """Test prime counting for known values."""
        # Known values of π(n) for various n
        known_counts = [
            (10, 4), (20, 8), (30, 10), (50, 15), (100, 25)
        ]
        
        for n, expected_count in known_counts:
            result = await prime_count(n)
            assert result == expected_count, f"π({n}) should be {expected_count}, got {result}"
    
    @pytest.mark.asyncio
    async def test_prime_count_incremental(self):
        """Test that prime count is non-decreasing."""
        counts = []
        for n in range(1, 31):
            count = await prime_count(n)
            counts.append(count)
            if len(counts) > 1:
                assert count >= counts[-2], f"π({n}) = {count} should be ≥ π({n-1}) = {counts[-2]}"
    
    @pytest.mark.asyncio
    async def test_prime_count_at_primes(self):
        """Test prime count exactly at prime numbers."""
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
        
        for i, p in enumerate(primes, 1):
            count = await prime_count(p)
            assert count >= i, f"π({p}) should be at least {i} since {p} is the {i}th prime"
            
            # Check that count increases exactly by 1 when we hit a new prime
            if p > 2:
                prev_count = await prime_count(p - 1)
                assert count == prev_count + 1, f"π({p}) should be π({p-1}) + 1"
    
    @pytest.mark.asyncio
    async def test_prime_count_consistency_with_is_prime(self):
        """Test that prime_count is consistent with is_prime results."""
        # Count primes manually using is_prime and compare
        for limit in [10, 20, 30, 50]:
            manual_count = 0
            for i in range(2, limit + 1):
                if await is_prime(i):
                    manual_count += 1
            
            auto_count = await prime_count(limit)
            assert manual_count == auto_count, f"Manual count {manual_count} should equal π({limit}) = {auto_count}"

class TestIsCoprime:
    """Test cases for the is_coprime function."""
    
    @pytest.mark.asyncio
    async def test_is_coprime_basic_cases(self):
        """Test coprimality with basic cases."""
        assert await is_coprime(8, 15) == True   # gcd(8, 15) = 1
        assert await is_coprime(12, 18) == False # gcd(12, 18) = 6
        assert await is_coprime(7, 13) == True   # Two primes are always coprime
        assert await is_coprime(1, 100) == True  # 1 is coprime with any number
        assert await is_coprime(14, 15) == True  # gcd(14, 15) = 1
    
    @pytest.mark.asyncio
    async def test_is_coprime_with_one(self):
        """Test that 1 is coprime with all numbers."""
        test_numbers = [1, 2, 3, 4, 5, 10, 17, 25, 100, 101]
        
        for n in test_numbers:
            assert await is_coprime(1, n) == True
            assert await is_coprime(n, 1) == True
    
    @pytest.mark.asyncio
    async def test_is_coprime_prime_pairs(self):
        """Test that distinct primes are always coprime."""
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
        
        for i, p1 in enumerate(primes):
            for p2 in primes[i+1:]:  # Only test each pair once
                assert await is_coprime(p1, p2) == True, f"Primes {p1} and {p2} should be coprime"
    
    @pytest.mark.asyncio
    async def test_is_coprime_same_number(self):
        """Test coprimality of a number with itself."""
        # Only 1 is coprime with itself
        assert await is_coprime(1, 1) == True
        
        # All other numbers share all their factors with themselves
        test_numbers = [2, 3, 4, 5, 6, 10, 12, 15, 17, 20]
        for n in test_numbers:
            assert await is_coprime(n, n) == False, f"{n} should not be coprime with itself"
    
    @pytest.mark.asyncio
    async def test_is_coprime_even_numbers(self):
        """Test that even numbers are not coprime with each other."""
        even_numbers = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
        
        for i, a in enumerate(even_numbers):
            for b in even_numbers[i+1:]:
                assert await is_coprime(a, b) == False, f"Even numbers {a} and {b} should not be coprime"
    
    @pytest.mark.asyncio
    async def test_is_coprime_consecutive_integers(self):
        """Test that consecutive integers are always coprime."""
        for n in range(1, 50):
            assert await is_coprime(n, n + 1) == True, f"Consecutive integers {n} and {n+1} should be coprime"
    
    @pytest.mark.asyncio
    async def test_is_coprime_powers_of_same_prime(self):
        """Test that powers of the same prime are not coprime."""
        # Powers of 2
        assert await is_coprime(4, 8) == False   # 2² and 2³
        assert await is_coprime(2, 16) == False  # 2¹ and 2⁴
        assert await is_coprime(8, 32) == False  # 2³ and 2⁵
        
        # Powers of 3
        assert await is_coprime(3, 9) == False   # 3¹ and 3²
        assert await is_coprime(9, 27) == False  # 3² and 3³
        
        # Powers of 5
        assert await is_coprime(5, 25) == False  # 5¹ and 5²
    
    @pytest.mark.asyncio
    async def test_is_coprime_negative_numbers(self):
        """Test coprimality with negative numbers."""
        assert await is_coprime(-8, 15) == True
        assert await is_coprime(8, -15) == True
        assert await is_coprime(-8, -15) == True
        assert await is_coprime(-12, 18) == False
        assert await is_coprime(12, -18) == False
        assert await is_coprime(-12, -18) == False
    
    @pytest.mark.asyncio
    async def test_is_coprime_symmetry(self):
        """Test that coprimality is symmetric."""
        test_pairs = [(8, 15), (12, 18), (7, 13), (14, 15), (9, 16), (25, 36)]
        
        for a, b in test_pairs:
            result_ab = await is_coprime(a, b)
            result_ba = await is_coprime(b, a)
            assert result_ab == result_ba, f"is_coprime({a}, {b}) should equal is_coprime({b}, {a})"

class TestFirstNPrimes:
    """Test cases for the first_n_primes function."""
    
    @pytest.mark.asyncio
    async def test_first_n_primes_basic_cases(self):
        """Test generating first n primes with basic cases."""
        assert await first_n_primes(5) == [2, 3, 5, 7, 11]
        assert await first_n_primes(10) == [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
        assert await first_n_primes(1) == [2]
        assert await first_n_primes(0) == []
    
    @pytest.mark.asyncio
    async def test_first_n_primes_edge_cases(self):
        """Test edge cases for first_n_primes."""
        assert await first_n_primes(0) == []
        assert await first_n_primes(-1) == []
        assert await first_n_primes(-5) == []
    
    @pytest.mark.asyncio
    async def test_first_n_primes_known_sequences(self):
        """Test known prime sequences."""
        # First 25 primes
        expected_25 = [
            2, 3, 5, 7, 11, 13, 17, 19, 23, 29,
            31, 37, 41, 43, 47, 53, 59, 61, 67, 71,
            73, 79, 83, 89, 97
        ]
        
        result_25 = await first_n_primes(25)
        assert result_25 == expected_25
        
        # Test smaller subsets
        for n in [1, 2, 3, 5, 10, 15, 20]:
            result = await first_n_primes(n)
            assert result == expected_25[:n]
    
    @pytest.mark.asyncio
    async def test_first_n_primes_all_prime(self):
        """Test that all returned numbers are actually prime."""
        for n in [5, 10, 15, 20, 25]:
            primes = await first_n_primes(n)
            for p in primes:
                assert await is_prime(p) == True, f"Generated number {p} should be prime"
    
    @pytest.mark.asyncio
    async def test_first_n_primes_sorted_order(self):
        """Test that primes are returned in ascending order."""
        for n in [5, 10, 15, 20, 25, 30]:
            primes = await first_n_primes(n)
            assert primes == sorted(primes), f"Primes {primes} should be in sorted order"
    
    @pytest.mark.asyncio
    async def test_first_n_primes_correct_count(self):
        """Test that exactly n primes are returned."""
        for n in range(0, 26):
            primes = await first_n_primes(n)
            assert len(primes) == n, f"Should return exactly {n} primes, got {len(primes)}"
    
    @pytest.mark.asyncio
    async def test_first_n_primes_no_duplicates(self):
        """Test that no duplicate primes are returned."""
        for n in [5, 10, 15, 20, 25, 30]:
            primes = await first_n_primes(n)
            assert len(primes) == len(set(primes)), f"Primes {primes} should have no duplicates"
    
    @pytest.mark.asyncio
    async def test_first_n_primes_consistency_with_nth_prime(self):
        """Test consistency between first_n_primes and nth_prime."""
        for n in [1, 5, 10, 15, 20]:
            primes = await first_n_primes(n)
            if primes:  # Skip empty list
                # Last prime should equal nth_prime(n)
                last_prime = primes[-1]
                nth_result = await nth_prime(n)
                assert last_prime == nth_result, f"Last of first {n} primes should equal nth_prime({n})"
    
    @pytest.mark.asyncio
    async def test_first_n_primes_large_n(self):
        """Test first_n_primes with moderately large n."""
        # Test with n=50
        primes_50 = await first_n_primes(50)
        assert len(primes_50) == 50
        assert primes_50[0] == 2
        assert primes_50[-1] == 229  # 50th prime is 229
        
        # All should be prime
        for p in primes_50:
            assert await is_prime(p) == True

class TestIntegration:
    """Integration tests for prime number operations."""
    
    @pytest.mark.asyncio
    async def test_prime_function_consistency(self):
        """Test consistency between different prime functions."""
        # Test that prime_count matches length of first_n_primes at prime positions
        limit = 30
        count = await prime_count(limit)
        primes_up_to_limit = await first_n_primes(count)
        
        # All primes should be ≤ limit
        for p in primes_up_to_limit:
            assert p <= limit, f"Prime {p} should be ≤ {limit}"
        
        # The last prime should be ≤ limit, and there should be no prime between it and limit
        if primes_up_to_limit:
            last_prime = primes_up_to_limit[-1]
            next_prime_after_last = await next_prime(last_prime)
            assert next_prime_after_last > limit, f"Next prime {next_prime_after_last} after {last_prime} should be > {limit}"
    
    @pytest.mark.asyncio
    async def test_factorization_roundtrip(self):
        """Test that factorization and reconstruction work correctly."""
        test_numbers = [12, 18, 24, 30, 36, 42, 48, 60, 72, 84, 90, 96, 120]
        
        for n in test_numbers:
            factors = await prime_factors(n)
            
            # Reconstruct number from factors
            reconstructed = 1
            for factor in factors:
                reconstructed *= factor
            
            assert reconstructed == n, f"Reconstructed {reconstructed} should equal original {n}"
            
            # All factors should be prime
            for factor in factors:
                assert await is_prime(factor) == True, f"Factor {factor} should be prime"
    
    @pytest.mark.asyncio
    async def test_coprimality_and_gcd_relationship(self):
        """Test relationship between coprimality and GCD."""
        # Import gcd for verification
        from chuk_mcp_functions.math.arithmetic.number_theory.divisibility import gcd
        
        test_pairs = [(8, 15), (12, 18), (7, 13), (14, 15), (9, 16), (25, 36)]
        
        for a, b in test_pairs:
            coprime = await is_coprime(a, b)
            gcd_result = await gcd(a, b)
            
            # Numbers are coprime if and only if their GCD is 1
            assert coprime == (gcd_result == 1), f"is_coprime({a}, {b}) = {coprime} should match gcd({a}, {b}) == 1"
    
    @pytest.mark.asyncio
    async def test_prime_generation_methods_consistency(self):
        """Test that different prime generation methods give consistent results."""
        n = 15
        
        # Method 1: first_n_primes
        primes_method1 = await first_n_primes(n)
        
        # Method 2: repeated next_prime calls
        primes_method2 = []
        current = 1
        for _ in range(n):
            current = await next_prime(current)
            primes_method2.append(current)
        
        # Method 3: nth_prime for each position
        primes_method3 = []
        for i in range(1, n + 1):
            primes_method3.append(await nth_prime(i))
        
        # All methods should give same result
        assert primes_method1 == primes_method2, "first_n_primes and next_prime should give same result"
        assert primes_method1 == primes_method3, "first_n_primes and nth_prime should give same result"

class TestPerformance:
    """Performance and async behavior tests."""
    
    @pytest.mark.asyncio
    async def test_all_operations_are_async(self):
        """Test that all prime operations are properly async."""
        operations = [
            is_prime(17),
            next_prime(10),
            nth_prime(10),
            prime_factors(60),
            prime_count(20),
            is_coprime(8, 15),
            first_n_primes(10)
        ]
        
        # Ensure all are coroutines
        for op in operations:
            assert asyncio.iscoroutine(op)
        
        # Run all operations concurrently
        results = await asyncio.gather(*operations)
        expected = [True, 11, 29, [2, 2, 3, 5], 8, True, [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]]
        
        assert results == expected
    
    @pytest.mark.asyncio
    async def test_concurrent_execution(self):
        """Test that prime operations can run concurrently."""
        import time
        
        start_time = time.time()
        
        # Run multiple prime checks concurrently
        tasks = []
        test_numbers = list(range(100, 200))  # Check numbers 100-199
        for n in test_numbers:
            tasks.append(is_prime(n))
        
        results = await asyncio.gather(*tasks)
        duration = time.time() - start_time
        
        # Verify some known results
        assert results[1] == True   # 101 is prime
        assert results[3] == True   # 103 is prime
        assert results[7] == True   # 107 is prime
        
        # Should complete quickly due to async nature
        assert duration < 2.0
    
    @pytest.mark.asyncio
    async def test_async_yielding_behavior(self):
        """Test async yielding for large computations."""
        import time
        
        # Test with larger numbers that should trigger async yielding
        start_time = time.time()
        
        # These operations should yield control during execution
        tasks = [
            prime_factors(5040),  # Highly composite number
            first_n_primes(100), # Generate many primes
            prime_count(1000),   # Count many primes
        ]
        
        results = await asyncio.gather(*tasks)
        duration = time.time() - start_time
        
        # Should complete in reasonable time
        assert duration < 5.0
        assert len(results) == 3
    
    @pytest.mark.asyncio
    async def test_prime_operations_with_timeout(self):
        """Test that prime operations complete within reasonable time."""
        # Test operations that might take longer
        operations_with_timeout = [
            (is_prime(982451653), 1.0),    # Large prime
            (prime_factors(123456), 2.0),  # Moderately large factorization
            (first_n_primes(200), 3.0),   # Generate many primes
            (prime_count(2000), 3.0),      # Count many primes
        ]
        
        for operation, timeout in operations_with_timeout:
            try:
                result = await asyncio.wait_for(operation, timeout=timeout)
                # If we get here, the operation completed within timeout
                assert result is not None
            except asyncio.TimeoutError:
                pytest.fail(f"Operation {operation} took longer than {timeout} seconds")

# Parametrized tests for comprehensive coverage
class TestParametrized:
    """Parametrized tests for comprehensive coverage."""
    
    @pytest.mark.asyncio
    @pytest.mark.parametrize("n,expected", [
        (1, False), (2, True), (3, True), (4, False), (5, True),
        (6, False), (7, True), (8, False), (9, False), (10, False),
        (11, True), (12, False), (13, True), (14, False), (15, False),
        (16, False), (17, True), (18, False), (19, True), (20, False)
    ])
    async def test_is_prime_parametrized(self, n, expected):
        """Parametrized test for is_prime function."""
        assert await is_prime(n) == expected
    
    @pytest.mark.asyncio
    @pytest.mark.parametrize("n,expected_factors", [
        (1, []), (2, [2]), (3, [3]), (4, [2, 2]), (6, [2, 3]),
        (8, [2, 2, 2]), (9, [3, 3]), (10, [2, 5]), (12, [2, 2, 3]),
        (15, [3, 5]), (16, [2, 2, 2, 2]), (18, [2, 3, 3]), (20, [2, 2, 5])
    ])
    async def test_prime_factors_parametrized(self, n, expected_factors):
        """Parametrized test for prime_factors function."""
        assert await prime_factors(n) == expected_factors
    
    @pytest.mark.asyncio
    @pytest.mark.parametrize("a,b,expected_coprime", [
        (1, 5, True), (2, 3, True), (4, 6, False), (8, 15, True),
        (9, 16, True), (12, 18, False), (7, 11, True), (10, 15, False),
        (13, 17, True), (14, 21, False)
    ])
    async def test_is_coprime_parametrized(self, a, b, expected_coprime):
        """Parametrized test for is_coprime function."""
        assert await is_coprime(a, b) == expected_coprime

# Error handling and edge cases
class TestErrorHandling:
    """Test error handling and edge cases."""
    
    @pytest.mark.asyncio
    async def test_nth_prime_error_conditions(self):
        """Test error conditions for nth_prime."""
        with pytest.raises(ValueError):
            await nth_prime(0)
        
        with pytest.raises(ValueError):
            await nth_prime(-1)
        
        with pytest.raises(ValueError):
            await nth_prime(-10)
    
    @pytest.mark.asyncio
    async def test_edge_case_handling(self):
        """Test edge case handling across all functions."""
        # All functions should handle these edge cases gracefully
        edge_cases = [0, 1, -1, -5]
        
        for n in edge_cases:
            # These should not raise exceptions
            await is_prime(n)
            await prime_factors(n)
            await prime_count(n)
            
            if n > 0:
                await next_prime(n)
            
            # first_n_primes with non-positive n should return empty list
            result = await first_n_primes(n)
            if n <= 0:
                assert result == []
    
    @pytest.mark.asyncio
    async def test_error_preserves_async_context(self):
        """Test that errors are properly raised in async context."""
        try:
            await nth_prime(-1)
            assert False, "Should have raised ValueError"
        except ValueError:
            # Should be able to continue with async operations
            result = await is_prime(17)
            assert result == True

if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v", "--tb=short"])