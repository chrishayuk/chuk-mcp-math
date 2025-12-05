#!/usr/bin/env python3
# tests/math/number_theory/test_divisibility.py
"""
Comprehensive pytest unit tests for divisibility operations.

Tests cover:
- GCD and LCM operations with various inputs
- Divisor finding and counting functions
- Parity checking (even/odd)
- Extended Euclidean algorithm
- Divisor sum calculations
- Edge cases and error conditions
- Async behavior and performance
"""

import pytest
import asyncio

# Import the functions to test
from chuk_mcp_math.number_theory.divisibility import (
    gcd,
    lcm,
    divisors,
    is_divisible,
    is_even,
    is_odd,
    extended_gcd,
    divisor_count,
    divisor_sum,
)


class TestGcd:
    """Test cases for the gcd function."""

    @pytest.mark.asyncio
    async def test_gcd_basic_cases(self):
        """Test GCD with basic cases."""
        assert await gcd(48, 18) == 6
        assert await gcd(17, 13) == 1  # Coprime numbers
        assert await gcd(100, 25) == 25  # One divides the other
        assert await gcd(12, 8) == 4
        assert await gcd(21, 14) == 7

    @pytest.mark.asyncio
    async def test_gcd_with_zero(self):
        """Test GCD with zero values."""
        assert await gcd(0, 5) == 5
        assert await gcd(5, 0) == 5
        assert await gcd(0, 0) == 0

    @pytest.mark.asyncio
    async def test_gcd_with_one(self):
        """Test GCD with one."""
        assert await gcd(1, 5) == 1
        assert await gcd(5, 1) == 1
        assert await gcd(1, 1) == 1

    @pytest.mark.asyncio
    async def test_gcd_negative_numbers(self):
        """Test GCD with negative numbers."""
        assert await gcd(-48, 18) == 6
        assert await gcd(48, -18) == 6
        assert await gcd(-48, -18) == 6
        assert await gcd(-12, -8) == 4

    @pytest.mark.asyncio
    async def test_gcd_same_numbers(self):
        """Test GCD of identical numbers."""
        assert await gcd(7, 7) == 7
        assert await gcd(12, 12) == 12
        assert await gcd(-5, -5) == 5

    @pytest.mark.asyncio
    async def test_gcd_large_numbers(self):
        """Test GCD with large numbers."""
        assert await gcd(1071, 462) == 21
        assert await gcd(12345, 6789) == 3
        assert await gcd(987654321, 123456789) == 9

    @pytest.mark.asyncio
    async def test_gcd_order_independence(self):
        """Test that GCD is commutative."""
        test_pairs = [(48, 18), (17, 13), (100, 25), (12, 8)]

        for a, b in test_pairs:
            assert await gcd(a, b) == await gcd(b, a)


class TestLcm:
    """Test cases for the lcm function."""

    @pytest.mark.asyncio
    async def test_lcm_basic_cases(self):
        """Test LCM with basic cases."""
        assert await lcm(12, 18) == 36
        assert await lcm(7, 13) == 91  # Coprime numbers
        assert await lcm(10, 5) == 10  # One divides the other
        assert await lcm(4, 6) == 12
        assert await lcm(15, 20) == 60

    @pytest.mark.asyncio
    async def test_lcm_with_zero(self):
        """Test LCM with zero values."""
        assert await lcm(0, 5) == 0
        assert await lcm(5, 0) == 0
        assert await lcm(0, 0) == 0

    @pytest.mark.asyncio
    async def test_lcm_with_one(self):
        """Test LCM with one."""
        assert await lcm(1, 5) == 5
        assert await lcm(5, 1) == 5
        assert await lcm(1, 1) == 1

    @pytest.mark.asyncio
    async def test_lcm_negative_numbers(self):
        """Test LCM with negative numbers."""
        assert await lcm(-12, 18) == 36
        assert await lcm(12, -18) == 36
        assert await lcm(-12, -18) == 36
        assert await lcm(-4, -6) == 12

    @pytest.mark.asyncio
    async def test_lcm_same_numbers(self):
        """Test LCM of identical numbers."""
        assert await lcm(7, 7) == 7
        assert await lcm(12, 12) == 12
        assert await lcm(-5, -5) == 5

    @pytest.mark.asyncio
    async def test_lcm_order_independence(self):
        """Test that LCM is commutative."""
        test_pairs = [(12, 18), (7, 13), (10, 5), (4, 6)]

        for a, b in test_pairs:
            assert await lcm(a, b) == await lcm(b, a)

    @pytest.mark.asyncio
    async def test_lcm_gcd_relationship(self):
        """Test the relationship LCM(a,b) * GCD(a,b) = |a * b|."""
        test_pairs = [(12, 18), (15, 20), (7, 13), (48, 18)]

        for a, b in test_pairs:
            if a != 0 and b != 0:  # Skip zero cases
                lcm_result = await lcm(a, b)
                gcd_result = await gcd(a, b)
                assert lcm_result * gcd_result == abs(a * b)


class TestDivisors:
    """Test cases for the divisors function."""

    @pytest.mark.asyncio
    async def test_divisors_basic_cases(self):
        """Test divisors with basic cases."""
        assert await divisors(12) == [1, 2, 3, 4, 6, 12]
        assert await divisors(17) == [1, 17]  # Prime number
        assert await divisors(1) == [1]
        assert await divisors(36) == [1, 2, 3, 4, 6, 9, 12, 18, 36]

    @pytest.mark.asyncio
    async def test_divisors_perfect_squares(self):
        """Test divisors of perfect squares."""
        assert await divisors(4) == [1, 2, 4]
        assert await divisors(9) == [1, 3, 9]
        assert await divisors(16) == [1, 2, 4, 8, 16]
        assert await divisors(25) == [1, 5, 25]

    @pytest.mark.asyncio
    async def test_divisors_prime_numbers(self):
        """Test divisors of prime numbers."""
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

        for p in primes:
            divs = await divisors(p)
            assert divs == [1, p]
            assert len(divs) == 2

    @pytest.mark.asyncio
    async def test_divisors_composite_numbers(self):
        """Test divisors of composite numbers."""
        # Test specific composite numbers with known divisors
        assert await divisors(6) == [1, 2, 3, 6]
        assert await divisors(8) == [1, 2, 4, 8]
        assert await divisors(10) == [1, 2, 5, 10]
        assert await divisors(15) == [1, 3, 5, 15]
        assert await divisors(20) == [1, 2, 4, 5, 10, 20]

    @pytest.mark.asyncio
    async def test_divisors_large_numbers(self):
        """Test divisors with larger numbers."""
        # Test a moderately large number
        divs_60 = await divisors(60)
        expected_60 = [1, 2, 3, 4, 5, 6, 10, 12, 15, 20, 30, 60]
        assert divs_60 == expected_60

        # Test 100
        divs_100 = await divisors(100)
        expected_100 = [1, 2, 4, 5, 10, 20, 25, 50, 100]
        assert divs_100 == expected_100

    @pytest.mark.asyncio
    async def test_divisors_edge_cases(self):
        """Test divisors with edge cases."""
        assert await divisors(0) == []  # No positive divisors of 0
        assert await divisors(-5) == []  # No positive divisors of negative numbers

    @pytest.mark.asyncio
    async def test_divisors_sorted_order(self):
        """Test that divisors are returned in sorted order."""
        test_numbers = [12, 24, 30, 48, 60]

        for n in test_numbers:
            divs = await divisors(n)
            assert divs == sorted(divs)  # Should be sorted
            assert all(n % d == 0 for d in divs)  # All should be divisors

    @pytest.mark.asyncio
    async def test_divisors_async_yielding(self):
        """Test that divisors function yields control for large numbers."""
        import time

        # Test with a number that has many divisors
        start_time = time.time()
        divs = await divisors(5040)  # Highly composite number
        duration = time.time() - start_time

        # Should complete quickly even with async yielding
        assert duration < 1.0
        assert len(divs) > 10  # 5040 has many divisors


class TestIsDivisible:
    """Test cases for the is_divisible function."""

    @pytest.mark.asyncio
    async def test_is_divisible_basic_cases(self):
        """Test basic divisibility cases."""
        assert await is_divisible(20, 4)
        assert not await is_divisible(17, 3)
        assert await is_divisible(15, 1)  # Any number divisible by 1
        assert await is_divisible(0, 5)  # 0 divisible by any non-zero

    @pytest.mark.asyncio
    async def test_is_divisible_negative_numbers(self):
        """Test divisibility with negative numbers."""
        assert await is_divisible(-20, 4)
        assert await is_divisible(20, -4)
        assert await is_divisible(-20, -4)
        assert not await is_divisible(-17, 3)

    @pytest.mark.asyncio
    async def test_is_divisible_same_numbers(self):
        """Test divisibility of identical numbers."""
        assert await is_divisible(7, 7)
        assert await is_divisible(-5, -5)

        # Note: is_divisible(0, 0) should raise an error, not return False
        with pytest.raises(ValueError, match="Cannot check divisibility by zero"):
            await is_divisible(0, 0)

    @pytest.mark.asyncio
    async def test_is_divisible_division_by_zero(self):
        """Test that division by zero raises ValueError."""
        with pytest.raises(ValueError, match="Cannot check divisibility by zero"):
            await is_divisible(5, 0)

        with pytest.raises(ValueError, match="Cannot check divisibility by zero"):
            await is_divisible(0, 0)

    @pytest.mark.asyncio
    async def test_is_divisible_powers_of_two(self):
        """Test divisibility by powers of 2."""
        powers_of_two = [1, 2, 4, 8, 16, 32, 64]

        for power in powers_of_two:
            assert await is_divisible(64, power)
            assert await is_divisible(63, power) == (power == 1)


class TestIsEven:
    """Test cases for the is_even function."""

    @pytest.mark.asyncio
    async def test_is_even_basic_cases(self):
        """Test basic even number cases."""
        assert await is_even(4)
        assert not await is_even(7)
        assert await is_even(0)
        assert await is_even(2)
        assert not await is_even(1)

    @pytest.mark.asyncio
    async def test_is_even_negative_numbers(self):
        """Test even check with negative numbers."""
        assert await is_even(-2)
        assert not await is_even(-3)
        assert await is_even(-4)
        assert not await is_even(-1)

    @pytest.mark.asyncio
    async def test_is_even_large_numbers(self):
        """Test even check with large numbers."""
        assert await is_even(1000000)
        assert not await is_even(1000001)
        assert await is_even(-1000000)
        assert await is_even(2**31 - 2)
        assert not await is_even(2**31 - 1)


class TestIsOdd:
    """Test cases for the is_odd function."""

    @pytest.mark.asyncio
    async def test_is_odd_basic_cases(self):
        """Test basic odd number cases."""
        assert await is_odd(7)
        assert not await is_odd(4)
        assert await is_odd(1)
        assert not await is_odd(0)
        assert not await is_odd(2)

    @pytest.mark.asyncio
    async def test_is_odd_negative_numbers(self):
        """Test odd check with negative numbers."""
        assert await is_odd(-3)
        assert not await is_odd(-2)
        assert await is_odd(-1)
        assert not await is_odd(-4)

    @pytest.mark.asyncio
    async def test_is_odd_large_numbers(self):
        """Test odd check with large numbers."""
        assert await is_odd(1000001)
        assert not await is_odd(1000000)
        assert await is_odd(-1000001)
        assert await is_odd(2**31 - 1)
        assert not await is_odd(2**31 - 2)

    @pytest.mark.asyncio
    async def test_even_odd_complementary(self):
        """Test that even and odd are complementary."""
        test_numbers = [0, 1, 2, 3, 4, 5, -1, -2, -3, 100, 101, -100, -101]

        for n in test_numbers:
            even_result = await is_even(n)
            odd_result = await is_odd(n)
            assert even_result != odd_result  # Exactly one should be true


class TestExtendedGcd:
    """Test cases for the extended_gcd function."""

    @pytest.mark.asyncio
    async def test_extended_gcd_basic_cases(self):
        """Test extended GCD with basic cases."""
        gcd_val, x, y = await extended_gcd(30, 18)
        assert gcd_val == 6
        assert 30 * x + 18 * y == gcd_val

        gcd_val, x, y = await extended_gcd(35, 15)
        assert gcd_val == 5
        assert 35 * x + 15 * y == gcd_val

    @pytest.mark.asyncio
    async def test_extended_gcd_coprime_numbers(self):
        """Test extended GCD with coprime numbers."""
        gcd_val, x, y = await extended_gcd(17, 13)
        assert gcd_val == 1
        assert 17 * x + 13 * y == 1

        gcd_val, x, y = await extended_gcd(7, 11)
        assert gcd_val == 1
        assert 7 * x + 11 * y == 1

    @pytest.mark.asyncio
    async def test_extended_gcd_with_zero(self):
        """Test extended GCD with zero values."""
        gcd_val, x, y = await extended_gcd(0, 5)
        assert gcd_val == 5
        assert 0 * x + 5 * y == 5
        assert y == 1

        gcd_val, x, y = await extended_gcd(7, 0)
        assert gcd_val == 7
        assert 7 * x + 0 * y == 7
        assert x == 1

    @pytest.mark.asyncio
    async def test_extended_gcd_verification(self):
        """Test that extended GCD results satisfy the equation."""
        test_pairs = [(48, 18), (100, 25), (12, 8), (21, 14), (1071, 462)]

        for a, b in test_pairs:
            gcd_val, x, y = await extended_gcd(a, b)

            # Verify that gcd is correct
            assert gcd_val == await gcd(a, b)

            # Verify the equation ax + by = gcd(a, b)
            assert a * x + b * y == gcd_val

    @pytest.mark.asyncio
    async def test_extended_gcd_negative_numbers(self):
        """Test extended GCD with negative numbers."""
        gcd_val, x, y = await extended_gcd(-30, 18)
        # The implementation may return negative GCD for negative inputs
        # What matters is that the equation holds and the absolute value matches regular GCD
        expected_gcd = await gcd(-30, 18)  # This should be positive (6)

        assert abs(gcd_val) == expected_gcd  # Absolute value should match
        assert (-30) * x + 18 * y == gcd_val  # Equation should hold

    @pytest.mark.asyncio
    async def test_extended_gcd_same_numbers(self):
        """Test extended GCD with identical numbers."""
        gcd_val, x, y = await extended_gcd(12, 12)
        assert gcd_val == 12
        assert 12 * x + 12 * y == 12
        assert x + y == 1  # Since 12(x + y) = 12


class TestDivisorCount:
    """Test cases for the divisor_count function."""

    @pytest.mark.asyncio
    async def test_divisor_count_basic_cases(self):
        """Test divisor count with basic cases."""
        assert await divisor_count(12) == 6  # 1,2,3,4,6,12
        assert await divisor_count(17) == 2  # 1,17 (prime)
        assert await divisor_count(1) == 1  # 1
        assert await divisor_count(36) == 9  # 1,2,3,4,6,9,12,18,36

    @pytest.mark.asyncio
    async def test_divisor_count_prime_numbers(self):
        """Test divisor count for prime numbers."""
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

        for p in primes:
            assert await divisor_count(p) == 2

    @pytest.mark.asyncio
    async def test_divisor_count_perfect_squares(self):
        """Test divisor count for perfect squares."""
        # Perfect squares have odd number of divisors
        assert await divisor_count(4) == 3  # 1,2,4
        assert await divisor_count(9) == 3  # 1,3,9
        assert await divisor_count(16) == 5  # 1,2,4,8,16
        assert await divisor_count(25) == 3  # 1,5,25

    @pytest.mark.asyncio
    async def test_divisor_count_powers_of_primes(self):
        """Test divisor count for powers of primes."""
        # For p^k, divisor count is k+1
        assert await divisor_count(8) == 4  # 2^3, divisors: 1,2,4,8
        assert await divisor_count(27) == 4  # 3^3, divisors: 1,3,9,27
        assert await divisor_count(32) == 6  # 2^5, divisors: 1,2,4,8,16,32

    @pytest.mark.asyncio
    async def test_divisor_count_edge_cases(self):
        """Test divisor count with edge cases."""
        assert await divisor_count(0) == 0  # No positive divisors
        assert await divisor_count(-5) == 0  # No positive divisors

    @pytest.mark.asyncio
    async def test_divisor_count_vs_divisors_length(self):
        """Test that divisor_count matches length of divisors list."""
        test_numbers = [1, 6, 12, 17, 24, 30, 36, 48, 60]

        for n in test_numbers:
            count = await divisor_count(n)
            divs = await divisors(n)
            assert count == len(divs)


class TestDivisorSum:
    """Test cases for the divisor_sum function."""

    @pytest.mark.asyncio
    async def test_divisor_sum_basic_cases(self):
        """Test divisor sum with basic cases."""
        assert await divisor_sum(12) == 28  # 1+2+3+4+6+12
        assert await divisor_sum(6) == 12  # 1+2+3+6
        assert await divisor_sum(17) == 18  # 1+17 (prime)
        assert await divisor_sum(1) == 1  # 1

    @pytest.mark.asyncio
    async def test_divisor_sum_perfect_numbers(self):
        """Test divisor sum for perfect numbers."""
        # Perfect numbers equal the sum of their proper divisors
        # σ(n) - n = n for perfect numbers, so σ(n) = 2n
        assert await divisor_sum(6) == 12  # 6 is perfect: 1+2+3+6 = 12
        assert await divisor_sum(28) == 56  # 28 is perfect: sum = 56

    @pytest.mark.asyncio
    async def test_divisor_sum_prime_numbers(self):
        """Test divisor sum for prime numbers."""
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

        for p in primes:
            assert await divisor_sum(p) == p + 1  # 1 + p

    @pytest.mark.asyncio
    async def test_divisor_sum_powers_of_primes(self):
        """Test divisor sum for powers of primes."""
        # For p^k, σ(p^k) = (p^(k+1) - 1) / (p - 1)
        assert await divisor_sum(4) == 7  # 2^2: 1+2+4 = 7
        assert await divisor_sum(8) == 15  # 2^3: 1+2+4+8 = 15
        assert await divisor_sum(9) == 13  # 3^2: 1+3+9 = 13
        assert await divisor_sum(27) == 40  # 3^3: 1+3+9+27 = 40

    @pytest.mark.asyncio
    async def test_divisor_sum_edge_cases(self):
        """Test divisor sum with edge cases."""
        assert await divisor_sum(0) == 0  # No positive divisors
        assert await divisor_sum(-5) == 0  # No positive divisors

    @pytest.mark.asyncio
    async def test_divisor_sum_vs_divisors_sum(self):
        """Test that divisor_sum matches sum of divisors list."""
        test_numbers = [1, 6, 12, 17, 24, 30, 36, 48]

        for n in test_numbers:
            sum_result = await divisor_sum(n)
            divs = await divisors(n)
            manual_sum = sum(divs)
            assert sum_result == manual_sum


class TestIntegration:
    """Integration tests for divisibility operations."""

    @pytest.mark.asyncio
    async def test_gcd_lcm_relationship(self):
        """Test the fundamental relationship between GCD and LCM."""
        test_pairs = [(12, 18), (15, 20), (7, 13), (48, 18), (100, 25)]

        for a, b in test_pairs:
            if a != 0 and b != 0:
                gcd_result = await gcd(a, b)
                lcm_result = await lcm(a, b)
                assert gcd_result * lcm_result == abs(a * b)

    @pytest.mark.asyncio
    async def test_divisibility_consistency(self):
        """Test consistency between different divisibility functions."""
        test_numbers = [12, 18, 24, 30, 36]

        for n in test_numbers:
            divs = await divisors(n)
            count = await divisor_count(n)
            sum_divs = await divisor_sum(n)

            # Count should match length of divisors
            assert count == len(divs)

            # Sum should match sum of divisors
            assert sum_divs == sum(divs)

            # All divisors should actually divide n
            for d in divs:
                assert await is_divisible(n, d)

    @pytest.mark.asyncio
    async def test_extended_gcd_consistency(self):
        """Test consistency between gcd and extended_gcd."""
        test_pairs = [(30, 18), (35, 15), (17, 13), (48, 18)]

        for a, b in test_pairs:
            gcd_simple = await gcd(a, b)
            gcd_extended, x, y = await extended_gcd(a, b)

            # GCD values should match
            assert gcd_simple == gcd_extended

            # Extended GCD equation should hold
            assert a * x + b * y == gcd_extended

    @pytest.mark.asyncio
    async def test_parity_divisibility_relationship(self):
        """Test relationship between parity and divisibility by 2."""
        test_numbers = list(range(-10, 11))

        for n in test_numbers:
            even_result = await is_even(n)
            odd_result = await is_odd(n)
            div_by_2 = await is_divisible(n, 2)

            # Even should match divisible by 2
            assert even_result == div_by_2

            # Odd should be opposite of even
            assert odd_result == (not even_result)


class TestPerformance:
    """Performance and async behavior tests."""

    @pytest.mark.asyncio
    async def test_all_operations_are_async(self):
        """Test that all divisibility operations are properly async."""
        operations = [
            gcd(48, 18),
            lcm(12, 18),
            divisors(12),
            is_divisible(20, 4),
            is_even(4),
            is_odd(7),
            extended_gcd(30, 18),
            divisor_count(12),
            divisor_sum(12),
        ]

        # Ensure all are coroutines
        for op in operations:
            assert asyncio.iscoroutine(op)

        # Run all operations concurrently
        results = await asyncio.gather(*operations)
        expected = [6, 36, [1, 2, 3, 4, 6, 12], True, True, True, (6, -1, 2), 6, 28]

        # Check basic results (extended_gcd might have different x,y values)
        assert results[0] == expected[0]  # gcd
        assert results[1] == expected[1]  # lcm
        assert results[2] == expected[2]  # divisors
        assert results[3] == expected[3]  # is_divisible
        assert results[4] == expected[4]  # is_even
        assert results[5] == expected[5]  # is_odd
        assert results[6][0] == expected[6][0]  # extended_gcd gcd value
        assert results[7] == expected[7]  # divisor_count
        assert results[8] == expected[8]  # divisor_sum

    @pytest.mark.asyncio
    async def test_concurrent_execution(self):
        """Test that divisibility operations can run concurrently."""
        import time

        start_time = time.time()

        # Run multiple operations concurrently
        tasks = []
        for i in range(2, 52):  # Test numbers 2 through 51
            tasks.append(gcd(i, 12))

        results = await asyncio.gather(*tasks)
        duration = time.time() - start_time

        # Verify some results
        assert results[0] == 2  # gcd(2, 12) = 2
        assert results[4] == 6  # gcd(6, 12) = 6
        assert results[10] == 12  # gcd(12, 12) = 12

        # Should complete quickly due to async nature
        assert duration < 1.0

    @pytest.mark.asyncio
    async def test_large_number_performance(self):
        """Test performance with moderately large numbers."""
        import time

        # Test with numbers that require some computation
        large_numbers = [5040, 7560, 9240]  # Highly composite numbers

        start_time = time.time()

        # Test multiple operations
        tasks = []
        for n in large_numbers:
            tasks.extend([divisors(n), divisor_count(n), divisor_sum(n)])

        results = await asyncio.gather(*tasks)
        duration = time.time() - start_time

        # Should handle moderately large numbers efficiently
        assert duration < 2.0
        assert len(results) == len(tasks)


# Parametrized tests for comprehensive coverage
class TestParametrized:
    """Parametrized tests for comprehensive coverage."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "a,b,expected_gcd,expected_lcm",
        [
            (12, 18, 6, 36),
            (15, 20, 5, 60),
            (7, 13, 1, 91),
            (100, 25, 25, 100),
            (48, 18, 6, 144),
            (8, 12, 4, 24),
        ],
    )
    async def test_gcd_lcm_parametrized(self, a, b, expected_gcd, expected_lcm):
        """Parametrized test for GCD and LCM functions."""
        assert await gcd(a, b) == expected_gcd
        assert await lcm(a, b) == expected_lcm

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "n,expected_even,expected_odd",
        [
            (0, True, False),
            (1, False, True),
            (2, True, False),
            (3, False, True),
            (4, True, False),
            (-1, False, True),
            (-2, True, False),
            (100, True, False),
            (101, False, True),
        ],
    )
    async def test_parity_parametrized(self, n, expected_even, expected_odd):
        """Parametrized test for parity functions."""
        assert await is_even(n) == expected_even
        assert await is_odd(n) == expected_odd

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "n,expected_count,expected_sum",
        [
            (1, 1, 1),  # 1: divisors = [1]
            (6, 4, 12),  # 6: divisors = [1,2,3,6]
            (12, 6, 28),  # 12: divisors = [1,2,3,4,6,12]
            (17, 2, 18),  # 17: divisors = [1,17] (prime)
            (20, 6, 42),  # 20: divisors = [1,2,4,5,10,20]
        ],
    )
    async def test_divisor_functions_parametrized(
        self, n, expected_count, expected_sum
    ):
        """Parametrized test for divisor counting and sum functions."""
        assert await divisor_count(n) == expected_count
        assert await divisor_sum(n) == expected_sum


# Error handling tests
class TestErrorHandling:
    """Test error handling and edge cases."""

    @pytest.mark.asyncio
    async def test_division_by_zero_error(self):
        """Test that division by zero raises appropriate errors."""
        with pytest.raises(ValueError, match="Cannot check divisibility by zero"):
            await is_divisible(10, 0)

    @pytest.mark.asyncio
    async def test_negative_input_handling(self):
        """Test handling of negative inputs where appropriate."""
        # Functions that should handle negatives gracefully
        assert await gcd(-12, 18) == 6
        assert await lcm(-12, 18) == 36
        assert await is_even(-4)
        assert await is_odd(-3)
        assert await is_divisible(-20, 4)

        # Functions that return empty/zero for non-positive inputs
        assert await divisors(-5) == []
        assert await divisor_count(-5) == 0
        assert await divisor_sum(-5) == 0

    @pytest.mark.asyncio
    async def test_error_preserves_async_context(self):
        """Test that errors are properly raised in async context."""
        try:
            await is_divisible(5, 0)
            assert False, "Should have raised ValueError"
        except ValueError:
            # Should be able to continue with async operations
            result = await gcd(12, 8)
            assert result == 4


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v", "--tb=short"])
