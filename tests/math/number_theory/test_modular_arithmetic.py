#!/usr/bin/env python3
# tests/math/number_theory/test_modular_arithmetic.py
"""
Comprehensive pytest test suite for modular_arithmetic.py module.

Tests cover:
- Chinese Remainder Theorem (CRT) solving and generalized systems
- Quadratic residues and Tonelli-Shanks algorithm
- Legendre and Jacobi symbols
- Primitive roots and multiplicative order
- Discrete logarithms (naive and baby-step giant-step)
- Mathematical properties and relationships
- Edge cases, error conditions, and performance testing
- Async behavior verification
"""

import pytest
import asyncio
import time
import math
from typing import List, Tuple, Optional

# Import the functions to test
from chuk_mcp_functions.math.number_theory.modular_arithmetic import (
    # Chinese Remainder Theorem
    crt_solve, generalized_crt,
    
    # Quadratic residues
    is_quadratic_residue, quadratic_residues, tonelli_shanks,
    
    # Legendre and Jacobi symbols
    legendre_symbol, jacobi_symbol,
    
    # Primitive roots
    primitive_root, all_primitive_roots, order_modulo,
    
    # Discrete logarithms
    discrete_log_naive, baby_step_giant_step
)

# ============================================================================
# CHINESE REMAINDER THEOREM TESTS
# ============================================================================

class TestChineseRemainderTheorem:
    """Test cases for Chinese Remainder Theorem functions."""
    
    @pytest.mark.asyncio
    async def test_crt_solve_basic_cases(self):
        """Test basic CRT solving with known examples."""
        # Example: x ≡ 2 (mod 3), x ≡ 3 (mod 5), x ≡ 2 (mod 7)
        # Solution: x ≡ 23 (mod 105)
        remainders = [2, 3, 2]
        moduli = [3, 5, 7]
        result = await crt_solve(remainders, moduli)
        
        assert result is not None, "CRT should have solution"
        x, m = result
        assert x == 23, f"Solution should be 23, got {x}"
        assert m == 105, f"Combined modulus should be 105, got {m}"
        
        # Verify the solution
        assert x % 3 == 2, "x ≡ 2 (mod 3) should hold"
        assert x % 5 == 3, "x ≡ 3 (mod 5) should hold"
        assert x % 7 == 2, "x ≡ 2 (mod 7) should hold"
    
    @pytest.mark.asyncio
    async def test_crt_solve_pairwise_coprime(self):
        """Test CRT with pairwise coprime moduli."""
        test_cases = [
            # Simple case
            ([1, 2], [2, 3], (5, 6)),
            # Three moduli - corrected expected value
            ([0, 0, 1], [2, 3, 5], (6, 30)),  # Was (25, 30), corrected to (6, 30)
            # Larger moduli - corrected expected value
            ([1, 4, 6], [7, 11, 13], (708, 1001)),  # Was (895, 1001), corrected to (708, 1001)
        ]
        
        for remainders, moduli, expected in test_cases:
            result = await crt_solve(remainders, moduli)
            assert result is not None, f"CRT should have solution for {remainders}, {moduli}"
            
            x, m = result
            expected_x, expected_m = expected
            assert x == expected_x, f"Solution should be {expected_x}, got {x}"
            assert m == expected_m, f"Combined modulus should be {expected_m}, got {m}"
            
            # Verify solution satisfies all congruences
            for i, (r, mod) in enumerate(zip(remainders, moduli)):
                assert x % mod == r, f"x ≡ {r} (mod {mod}) should hold"
    
    @pytest.mark.asyncio
    async def test_crt_solve_no_solution(self):
        """Test CRT cases with no solution."""
        # Use a system that truly has no solution
        remainders = [1, 5]
        moduli = [6, 9]  # gcd(6,9) = 3, and 1 ≢ 5 (mod 3)
        result = await crt_solve(remainders, moduli)
        
        assert result is None, "CRT should have no solution for inconsistent system"
    
    @pytest.mark.asyncio
    async def test_crt_solve_single_congruence(self):
        """Test CRT with single congruence."""
        remainders = [5]
        moduli = [7]
        result = await crt_solve(remainders, moduli)
        
        assert result is not None, "Single congruence should have solution"
        x, m = result
        assert x == 5, f"Solution should be 5, got {x}"
        assert m == 7, f"Modulus should be 7, got {m}"
    
    @pytest.mark.asyncio
    async def test_crt_solve_non_coprime_solvable(self):
        """Test CRT with non-coprime moduli that still has solution."""
        # x ≡ 0 (mod 4), x ≡ 0 (mod 6) → x ≡ 0 (mod 12)
        remainders = [0, 0]
        moduli = [4, 6]
        result = await crt_solve(remainders, moduli)
        
        assert result is not None, "Compatible non-coprime system should have solution"
        x, m = result
        assert x == 0, f"Solution should be 0, got {x}"
        assert m == 12, f"Combined modulus should be lcm(4,6)=12, got {m}"
    
    @pytest.mark.asyncio
    async def test_generalized_crt(self):
        """Test generalized CRT function."""
        congruences = [[2, 3], [3, 5], [2, 7]]  # [[remainder, modulus], ...]
        result = await generalized_crt(congruences)
        
        assert result is not None, "Generalized CRT should have solution"
        x, m = result
        assert x == 23, f"Solution should be 23, got {x}"
        assert m == 105, f"Combined modulus should be 105, got {m}"
    
    @pytest.mark.asyncio
    async def test_crt_edge_cases(self):
        """Test edge cases for CRT functions."""
        # Empty input
        result = await crt_solve([], [])
        assert result is None, "Empty CRT system should return None"
        
        result = await generalized_crt([])
        assert result is None, "Empty generalized CRT should return None"
        
        # Mismatched lengths - test graceful handling instead of exceptions
        try:
            result = await crt_solve([1, 2], [3])
            assert result is None or isinstance(result, tuple)
        except (ValueError, IndexError):
            pass  # Either exception or graceful handling is acceptable

# ============================================================================
# QUADRATIC RESIDUES TESTS  
# ============================================================================

class TestQuadraticResidues:
    """Test cases for quadratic residue functions."""
    
    @pytest.mark.asyncio
    async def test_is_quadratic_residue_known_cases(self):
        """Test quadratic residue recognition with known cases."""
        # Known quadratic residues mod 7: {0, 1, 2, 4}
        qr_mod_7 = [0, 1, 2, 4]
        non_qr_mod_7 = [3, 5, 6]
        
        for a in qr_mod_7:
            assert await is_quadratic_residue(a, 7), f"{a} should be QR mod 7"
        
        for a in non_qr_mod_7:
            assert not await is_quadratic_residue(a, 7), f"{a} should not be QR mod 7"
        
        # Known quadratic residues mod 11: {0, 1, 3, 4, 5, 9}
        qr_mod_11 = [0, 1, 3, 4, 5, 9]
        non_qr_mod_11 = [2, 6, 7, 8, 10]
        
        for a in qr_mod_11:
            assert await is_quadratic_residue(a, 11), f"{a} should be QR mod 11"
        
        for a in non_qr_mod_11:
            assert not await is_quadratic_residue(a, 11), f"{a} should not be QR mod 11"
    
    @pytest.mark.asyncio
    async def test_quadratic_residues_generation(self):
        """Test generation of all quadratic residues."""
        # Test mod 7
        qr_7 = await quadratic_residues(7)
        expected_7 = [0, 1, 2, 4]
        assert qr_7 == expected_7, f"QR mod 7 should be {expected_7}, got {qr_7}"
        
        # Test mod 11
        qr_11 = await quadratic_residues(11)
        expected_11 = [0, 1, 3, 4, 5, 9]
        assert qr_11 == expected_11, f"QR mod 11 should be {expected_11}, got {qr_11}"
        
        # Test mod 8 (composite)
        qr_8 = await quadratic_residues(8)
        expected_8 = [0, 1, 4]
        assert qr_8 == expected_8, f"QR mod 8 should be {expected_8}, got {qr_8}"
    
    @pytest.mark.asyncio
    async def test_tonelli_shanks_algorithm(self):
        """Test Tonelli-Shanks square root algorithm."""
        # Test cases where square roots exist
        test_cases = [
            (2, 7, [3, 4]),   # 3² ≡ 4² ≡ 2 (mod 7)
            (1, 11, [1, 10]), # 1² ≡ 10² ≡ 1 (mod 11)
            (4, 11, [2, 9]),  # 2² ≡ 9² ≡ 4 (mod 11)
            (9, 11, [3, 8]),  # 3² ≡ 8² ≡ 9 (mod 11)
        ]
        
        for a, p, expected in test_cases:
            result = await tonelli_shanks(a, p)
            assert result is not None, f"Square root of {a} mod {p} should exist"
            assert sorted(result) == expected, f"Square roots of {a} mod {p} should be {expected}, got {result}"
            
            # Verify the solutions
            for root in result:
                assert (root * root) % p == a % p, f"{root}² should ≡ {a} (mod {p})"
    
    @pytest.mark.asyncio
    async def test_tonelli_shanks_no_solution(self):
        """Test Tonelli-Shanks when no square root exists."""
        # 3 is not a quadratic residue mod 7
        result = await tonelli_shanks(3, 7)
        assert result is None, "Square root of 3 mod 7 should not exist"
        
        # 2 is not a quadratic residue mod 11
        result = await tonelli_shanks(2, 11)
        assert result is None, "Square root of 2 mod 11 should not exist"
    
    @pytest.mark.asyncio
    async def test_tonelli_shanks_special_cases(self):
        """Test special cases for Tonelli-Shanks."""
        # Test p ≡ 3 (mod 4) case
        result = await tonelli_shanks(4, 7)  # 7 ≡ 3 (mod 4)
        assert result is not None, "Should handle p ≡ 3 (mod 4) case"
        assert sorted(result) == [2, 5], "Square roots of 4 mod 7 should be [2, 5]"
        
        # Test a = 0
        result = await tonelli_shanks(0, 7)
        assert result == [0], "Square root of 0 should be [0]"
    
    @pytest.mark.asyncio
    async def test_quadratic_residues_edge_cases(self):
        """Test edge cases for quadratic residue functions."""
        # Test invalid moduli
        assert not await is_quadratic_residue(2, 1), "n ≤ 1 should return False"
        assert not await is_quadratic_residue(2, 0), "n = 0 should return False"
        
        # Test a = 0
        assert await is_quadratic_residue(0, 7), "0 should always be QR"
        
        # Test empty case
        empty_qr = await quadratic_residues(1)
        assert empty_qr == [], "QR mod 1 should be empty"
        
        # Test Tonelli-Shanks with non-prime
        result = await tonelli_shanks(1, 8)
        assert result is None, "Tonelli-Shanks should require prime modulus"

# ============================================================================
# LEGENDRE AND JACOBI SYMBOLS TESTS
# ============================================================================

class TestLegendreJacobiSymbols:
    """Test cases for Legendre and Jacobi symbol functions."""
    
    @pytest.mark.asyncio
    async def test_legendre_symbol_known_values(self):
        """Test Legendre symbol with known values."""
        # Test cases for various primes
        test_cases = [
            # (a, p, expected)
            (1, 7, 1),   # 1 is always QR
            (2, 7, 1),   # 2 is QR mod 7
            (3, 7, -1),  # 3 is not QR mod 7
            (4, 7, 1),   # 4 is QR mod 7
            (5, 7, -1),  # 5 is not QR mod 7
            (6, 7, -1),  # 6 is not QR mod 7
            (7, 7, 0),   # 7 ≡ 0 (mod 7)
            
            (1, 11, 1),  # 1 is always QR
            (2, 11, -1), # 2 is not QR mod 11
            (3, 11, 1),  # 3 is QR mod 11
            (4, 11, 1),  # 4 is QR mod 11
            (5, 11, 1),  # 5 is QR mod 11
            (11, 11, 0), # 11 ≡ 0 (mod 11)
        ]
        
        for a, p, expected in test_cases:
            result = await legendre_symbol(a, p)
            assert result == expected, f"({a}/{p}) should be {expected}, got {result}"
    
    @pytest.mark.asyncio
    async def test_legendre_symbol_properties(self):
        """Test mathematical properties of Legendre symbol."""
        # Test that (a/p) ∈ {-1, 0, 1}
        for p in [7, 11, 13]:
            for a in range(20):
                legendre = await legendre_symbol(a, p)
                assert legendre in [-1, 0, 1], f"Legendre symbol should be in {{-1,0,1}}"
        
        # Test multiplicativity: (ab/p) = (a/p)(b/p)
        p = 11
        test_pairs = [(2, 3), (3, 5), (4, 7), (2, 9)]
        
        for a, b in test_pairs:
            legendre_a = await legendre_symbol(a, p)
            legendre_b = await legendre_symbol(b, p)
            legendre_ab = await legendre_symbol((a * b) % p, p)
            
            expected = (legendre_a * legendre_b) % p
            if expected == p - 1:
                expected = -1  # Convert to standard form
            
            assert legendre_ab == expected, f"Multiplicativity should hold for ({a})({b})/{p}"
    
    @pytest.mark.asyncio
    async def test_jacobi_symbol_known_values(self):
        """Test Jacobi symbol with known values."""
        test_cases = [
            # (a, n, expected)
            (2, 15, 1),   # (2/15) = (2/3)(2/5) = (-1)(-1) = 1
            (5, 21, 1),   # Corrected: (5/21) = (5/3)(5/7) = (-1)(-1) = 1, was -1
            (3, 9, 0),    # gcd(3,9) > 1
            (1, 15, 1),   # 1 is always 1
            (8, 15, 1),   # Test composite a
        ]
        
        for a, n, expected in test_cases:
            result = await jacobi_symbol(a, n)
            assert result == expected, f"({a}/{n}) should be {expected}, got {result}"
    
    @pytest.mark.asyncio
    async def test_jacobi_symbol_properties(self):
        """Test mathematical properties of Jacobi symbol."""
        # Test that Jacobi symbol reduces to Legendre symbol for prime modulus
        primes = [7, 11, 13]
        
        for p in primes:
            for a in range(1, 15):
                jacobi = await jacobi_symbol(a, p)
                legendre = await legendre_symbol(a, p)
                assert jacobi == legendre, f"Jacobi should equal Legendre for prime {p}"
        
        # Test multiplicativity in the numerator
        n = 15
        test_pairs = [(2, 3), (4, 5), (7, 8)]
        
        for a, b in test_pairs:
            jacobi_a = await jacobi_symbol(a, n)
            jacobi_b = await jacobi_symbol(b, n)
            jacobi_ab = await jacobi_symbol((a * b) % n, n)
            
            expected = jacobi_a * jacobi_b
            if expected == 2:
                expected = -1
            elif expected == -2:
                expected = 1
            
            assert jacobi_ab == expected, f"Multiplicativity should hold for ({a})({b})/{n}"
    
    @pytest.mark.asyncio
    async def test_symbol_edge_cases(self):
        """Test edge cases for symbol functions."""
        # Test error conditions for Legendre symbol
        with pytest.raises(ValueError):
            await legendre_symbol(2, 4)  # Not prime
        
        with pytest.raises(ValueError):
            await legendre_symbol(2, 2)  # Even prime (should be odd)
        
        # Test error conditions for Jacobi symbol
        with pytest.raises(ValueError):
            await jacobi_symbol(2, 4)   # Even n
        
        with pytest.raises(ValueError):
            await jacobi_symbol(2, -3)  # Negative n
        
        with pytest.raises(ValueError):
            await jacobi_symbol(2, 0)   # Zero n

# ============================================================================
# PRIMITIVE ROOTS TESTS
# ============================================================================

class TestPrimitiveRoots:
    """Test cases for primitive root functions."""
    
    @pytest.mark.asyncio
    async def test_primitive_root_known_values(self):
        """Test primitive root calculation for known cases."""
        # Known primitive roots
        known_primitive_roots = [
            (7, 3),   # 3 is primitive root mod 7
            (11, 2),  # 2 is primitive root mod 11
            (13, 2),  # 2 is primitive root mod 13
            (17, 3),  # 3 is primitive root mod 17
            (19, 2),  # 2 is primitive root mod 19
        ]
        
        for n, expected in known_primitive_roots:
            result = await primitive_root(n)
            assert result == expected, f"Primitive root mod {n} should be {expected}, got {result}"
    
    @pytest.mark.asyncio
    async def test_primitive_root_no_primitive_root(self):
        """Test cases where no primitive root exists."""
        # Numbers that don't have primitive roots
        no_primitive_root = [8, 12, 15, 16, 20, 24]
        
        for n in no_primitive_root:
            result = await primitive_root(n)
            assert result is None, f"No primitive root should exist mod {n}"
    
    @pytest.mark.asyncio
    async def test_all_primitive_roots_known_cases(self):
        """Test finding all primitive roots for known cases."""
        # All primitive roots mod 7: {3, 5}
        roots_7 = await all_primitive_roots(7)
        assert roots_7 == [3, 5], f"All primitive roots mod 7 should be [3, 5], got {roots_7}"
        
        # All primitive roots mod 11: {2, 6, 7, 8}
        roots_11 = await all_primitive_roots(11)
        expected_11 = [2, 6, 7, 8]
        assert roots_11 == expected_11, f"All primitive roots mod 11 should be {expected_11}, got {roots_11}"
        
        # All primitive roots mod 5: {2, 3}
        roots_5 = await all_primitive_roots(5)
        expected_5 = [2, 3]
        assert roots_5 == expected_5, f"All primitive roots mod 5 should be {expected_5}, got {roots_5}"
    
    @pytest.mark.asyncio
    async def test_order_modulo_calculation(self):
        """Test multiplicative order calculation."""
        # Known orders
        test_cases = [
            # (a, n, expected_order)
            (3, 7, 6),    # ord_7(3) = 6 (primitive root)
            (2, 7, 3),    # ord_7(2) = 3
            (4, 7, 3),    # ord_7(4) = 3
            (6, 7, 2),    # ord_7(6) = 2
            
            (2, 11, 10),  # ord_11(2) = 10 (primitive root)
            (3, 11, 5),   # ord_11(3) = 5
            (4, 11, 5),   # ord_11(4) = 5
            (5, 11, 5),   # ord_11(5) = 5
        ]
        
        for a, n, expected in test_cases:
            result = await order_modulo(a, n)
            assert result == expected, f"ord_{n}({a}) should be {expected}, got {result}"
    
    @pytest.mark.asyncio
    async def test_order_modulo_properties(self):
        """Test properties of multiplicative order."""
        # Test that a^ord(a) ≡ 1 (mod n)
        test_cases = [(3, 7), (2, 11), (5, 13), (7, 17)]
        
        for a, n in test_cases:
            order = await order_modulo(a, n)
            if order is not None:
                assert pow(a, order, n) == 1, f"{a}^{order} should ≡ 1 (mod {n})"
                
                # Test that this is the smallest such exponent
                for k in range(1, order):
                    assert pow(a, k, n) != 1, f"{a}^{k} should not ≡ 1 (mod {n}) for k < {order}"
    
    @pytest.mark.asyncio
    async def test_primitive_root_verification(self):
        """Test that found primitive roots are actually primitive."""
        primes_with_primitive_roots = [5, 7, 11, 13, 17, 19]
        
        for p in primes_with_primitive_roots:
            root = await primitive_root(p)
            if root is not None:
                order = await order_modulo(root, p)
                assert order == p - 1, f"Primitive root {root} mod {p} should have order {p-1}, got {order}"
    
    @pytest.mark.asyncio
    async def test_primitive_root_edge_cases(self):
        """Test edge cases for primitive root functions."""
        # Test invalid inputs
        assert await primitive_root(1) is None, "No primitive root mod 1"
        assert await primitive_root(0) is None, "No primitive root mod 0"
        
        assert await all_primitive_roots(1) == [], "No primitive roots mod 1"
        assert await all_primitive_roots(8) == [], "No primitive roots mod 8"
        
        assert await order_modulo(2, 1) is None, "Order undefined for n=1"
        assert await order_modulo(0, 7) is None, "Order undefined for gcd(a,n) > 1"

# ============================================================================
# DISCRETE LOGARITHMS TESTS
# ============================================================================

class TestDiscreteLogarithms:
    """Test cases for discrete logarithm functions."""
    
    @pytest.mark.asyncio
    async def test_discrete_log_naive_known_cases(self):
        """Test naive discrete logarithm with known cases."""
        test_cases = [
            # (g, h, n, expected_x) where g^x ≡ h (mod n)
            (3, 2, 7, 2),   # 3² ≡ 2 (mod 7)
            (3, 4, 7, 4),   # 3⁴ ≡ 4 (mod 7)
            (2, 1, 5, 0),   # 2⁰ ≡ 1 (mod 5)
            (2, 3, 5, 3),   # 2³ ≡ 3 (mod 5)
            (5, 1, 11, 0),  # 5⁰ ≡ 1 (mod 11)
        ]
        
        for g, h, n, expected in test_cases:
            result = await discrete_log_naive(g, h, n)
            assert result == expected, f"log_{g}({h}) mod {n} should be {expected}, got {result}"
            
            # Verify the solution
            assert pow(g, result, n) == h % n, f"{g}^{result} should ≡ {h} (mod {n})"
    
    @pytest.mark.asyncio
    async def test_discrete_log_naive_no_solution(self):
        """Test discrete logarithm when no solution exists."""
        # Test case 1: 2^x ≡ 0 (mod 5) - impossible since gcd(2,5)=1
        result = await discrete_log_naive(2, 0, 5)
        assert result is None, "2^x cannot equal 0 mod 5"
        
        # Test case 2: 2^x ≡ 3 (mod 8) - impossible since 3 not in subgroup <2> mod 8
        result = await discrete_log_naive(2, 3, 8)
        assert result is None, "2^x cannot equal 3 mod 8"
    
    @pytest.mark.asyncio
    async def test_baby_step_giant_step_known_cases(self):
        """Test baby-step giant-step algorithm with known cases."""
        test_cases = [
            # (g, h, n, expected_x)
            (2, 5, 11, 4),   # 2⁴ ≡ 5 (mod 11)
            (2, 3, 5, 3),    # 2³ ≡ 3 (mod 5) - corrected test case
            (5, 7, 17, 15),  # 5¹⁵ ≡ 7 (mod 17) - corrected from 11 to 15
        ]
        
        for g, h, n, expected in test_cases:
            result = await baby_step_giant_step(g, h, n)
            assert result == expected, f"BSGS log_{g}({h}) mod {n} should be {expected}, got {result}"
            
            # Verify the solution
            assert pow(g, result, n) == h % n, f"{g}^{result} should ≡ {h} (mod {n})"
    
    @pytest.mark.asyncio
    async def test_baby_step_giant_step_vs_naive(self):
        """Test that baby-step giant-step agrees with naive method."""
        test_cases = [
            (2, 3, 7),
            (3, 5, 11),
            (5, 2, 13),
            (7, 3, 17),
        ]
        
        for g, h, n in test_cases:
            naive_result = await discrete_log_naive(g, h, n, max_exp=50)
            bsgs_result = await baby_step_giant_step(g, h, n)
            
            # Both should give same result (or both None)
            assert naive_result == bsgs_result, f"Naive and BSGS should agree for log_{g}({h}) mod {n}"
    
    @pytest.mark.asyncio
    async def test_discrete_log_special_cases(self):
        """Test special cases for discrete logarithm."""
        # Test h = 1 (should always give x = 0 if gcd(g,n) = 1)
        for n in [7, 11, 13]:
            for g in [2, 3, 5]:
                if math.gcd(g, n) == 1:
                    result = await discrete_log_naive(g, 1, n)
                    assert result == 0, f"log_{g}(1) mod {n} should be 0"
                    
                    result = await baby_step_giant_step(g, 1, n)
                    assert result == 0, f"BSGS log_{g}(1) mod {n} should be 0"
        
        # Test g = h (should give x = 1 if gcd(g,n) = 1)
        for n in [7, 11, 13]:
            for g in [2, 3, 5]:
                if math.gcd(g, n) == 1:
                    result = await discrete_log_naive(g, g, n)
                    assert result == 1, f"log_{g}({g}) mod {n} should be 1"
    
    @pytest.mark.asyncio
    async def test_discrete_log_edge_cases(self):
        """Test edge cases for discrete logarithm functions."""
        # Test invalid moduli
        assert await discrete_log_naive(2, 3, 1) is None, "n ≤ 1 should return None"
        assert await baby_step_giant_step(2, 3, 1) is None, "n ≤ 1 should return None"
        
        # Test with max_exp limit
        result = await discrete_log_naive(2, 3, 7, max_exp=2)
        # Since 2³ ≡ 1 (mod 7) and we need larger exponent, should return None
        assert result is None, "Should respect max_exp limit"

# ============================================================================
# INTEGRATION AND MATHEMATICAL PROPERTIES TESTS
# ============================================================================

class TestIntegrationAndProperties:
    """Integration tests and mathematical property verification."""
    
    @pytest.mark.asyncio
    async def test_legendre_quadratic_residue_consistency(self):
        """Test consistency between Legendre symbol and quadratic residue testing."""
        primes = [7, 11, 13, 17]
        
        for p in primes:
            qr_list = await quadratic_residues(p)
            
            for a in range(1, p):
                legendre = await legendre_symbol(a, p)
                is_qr = await is_quadratic_residue(a, p)
                
                if legendre == 1:
                    assert is_qr, f"{a} should be QR mod {p} (Legendre = 1)"
                    assert a in qr_list, f"{a} should be in QR list mod {p}"
                elif legendre == -1:
                    assert not is_qr, f"{a} should not be QR mod {p} (Legendre = -1)"
                    assert a not in qr_list, f"{a} should not be in QR list mod {p}"
    
    @pytest.mark.asyncio
    async def test_primitive_root_order_relationship(self):
        """Test relationship between primitive roots and orders."""
        primes = [5, 7, 11, 13]
        
        for p in primes:
            all_roots = await all_primitive_roots(p)
            
            for root in all_roots:
                order = await order_modulo(root, p)
                assert order == p - 1, f"Primitive root {root} mod {p} should have order {p-1}"
            
            # Test that elements of smaller order are not primitive roots
            for a in range(2, p):
                if math.gcd(a, p) == 1:
                    order = await order_modulo(a, p)
                    if order < p - 1:
                        assert a not in all_roots, f"{a} has order {order} < {p-1}, so not primitive root"
    
    @pytest.mark.asyncio
    async def test_tonelli_shanks_legendre_consistency(self):
        """Test that Tonelli-Shanks only finds roots when Legendre symbol = 1."""
        primes = [7, 11, 13, 17]
        
        for p in primes:
            for a in range(1, p):
                legendre = await legendre_symbol(a, p)
                roots = await tonelli_shanks(a, p)
                
                if legendre == 1:
                    assert roots is not None, f"Should find roots for {a} mod {p} (Legendre = 1)"
                    assert len(roots) == 2, f"Should find exactly 2 roots for {a} mod {p}"
                elif legendre == -1:
                    assert roots is None, f"Should not find roots for {a} mod {p} (Legendre = -1)"
    
    @pytest.mark.asyncio
    async def test_discrete_log_primitive_root_consistency(self):
        """Test discrete logs with primitive roots have full range."""
        primes = [7, 11, 13]
        
        for p in primes:
            prim_root = await primitive_root(p)
            if prim_root is not None:
                # Should be able to find discrete logs for all non-zero elements
                for h in range(1, p):
                    log_val = await discrete_log_naive(prim_root, h, p, max_exp=p-1)
                    assert log_val is not None, f"Should find log_{prim_root}({h}) mod {p}"
                    assert 0 <= log_val < p, f"Discrete log should be in range [0, {p})"

# ============================================================================
# PERFORMANCE AND ASYNC BEHAVIOR TESTS
# ============================================================================

class TestPerformance:
    """Performance and async behavior tests."""
    
    @pytest.mark.asyncio
    async def test_all_functions_are_async(self):
        """Test that all modular arithmetic functions are properly async."""
        operations = [
            crt_solve([2, 3], [3, 5]),
            is_quadratic_residue(2, 7),
            quadratic_residues(11),
            legendre_symbol(3, 7),
            jacobi_symbol(5, 21),
            primitive_root(11),
            order_modulo(3, 7),
            discrete_log_naive(2, 3, 7)
        ]
        
        # Ensure all are coroutines
        for op in operations:
            assert asyncio.iscoroutine(op)
        
        # Run all operations concurrently
        results = await asyncio.gather(*operations)
        
        # Verify results have expected types and reasonable values
        assert all(isinstance(r, (int, bool, list, tuple, type(None))) for r in results)
        assert len(results) == len(operations)
    
    @pytest.mark.asyncio
    async def test_concurrent_execution(self):
        """Test that modular arithmetic operations can run concurrently."""
        start_time = time.time()
        
        # Run multiple operations concurrently
        tasks = []
        
        # CRT tasks
        for i in range(2, 10):
            tasks.append(crt_solve([1, 2], [i, i+1]))
        
        # Quadratic residue tasks
        for p in [7, 11, 13, 17]:
            for a in range(1, 6):
                tasks.append(is_quadratic_residue(a, p))
        
        # Symbol tasks
        for p in [7, 11, 13]:
            for a in range(1, 5):
                tasks.append(legendre_symbol(a, p))
        
        # Primitive root tasks
        for n in [5, 7, 11, 13]:
            tasks.append(primitive_root(n))
        
        results = await asyncio.gather(*tasks)
        duration = time.time() - start_time
        
        # Should complete quickly due to async nature
        assert duration < 3.0
        assert len(results) > 0
    
    @pytest.mark.asyncio
    async def test_large_modulus_performance(self):
        """Test performance with moderately large moduli."""
        large_primes = [101, 103, 107, 109, 113]
        
        for p in large_primes:
            # Test quadratic residues
            qr_list = await quadratic_residues(p)
            assert len(qr_list) == (p + 1) // 2, f"Should have (p+1)/2 QRs mod {p}"
            
            # Test primitive root
            prim_root = await primitive_root(p)
            assert prim_root is not None, f"Should find primitive root mod {p}"
            
            # Test order calculation
            order = await order_modulo(prim_root, p)
            assert order == p - 1, f"Primitive root should have order {p-1}"
    
    @pytest.mark.asyncio
    async def test_tonelli_shanks_performance(self):
        """Test Tonelli-Shanks performance on various cases."""
        # Test with primes where p ≡ 3 (mod 4) (faster case)
        primes_3_mod_4 = [7, 11, 19, 23]
        
        for p in primes_3_mod_4:
            qr_list = await quadratic_residues(p)
            for a in qr_list[1:4]:  # Test first few non-zero QRs
                roots = await tonelli_shanks(a, p)
                assert roots is not None, f"Should find roots for QR {a} mod {p}"
        
        # Test with primes where p ≡ 1 (mod 4) (general case)
        primes_1_mod_4 = [13, 17, 29]
        
        for p in primes_1_mod_4:
            qr_list = await quadratic_residues(p)
            for a in qr_list[1:3]:  # Test first few non-zero QRs
                roots = await tonelli_shanks(a, p)
                assert roots is not None, f"Should find roots for QR {a} mod {p}"
    
    @pytest.mark.asyncio
    async def test_discrete_log_algorithms_comparison(self):
        """Compare performance of discrete log algorithms."""
        test_cases = [
            (2, 5, 11),
            (3, 4, 13),
            (5, 7, 17),
        ]
        
        for g, h, n in test_cases:
            # Test both algorithms
            start_time = time.time()
            naive_result = await discrete_log_naive(g, h, n, max_exp=n-1)
            naive_time = time.time() - start_time
            
            start_time = time.time()
            bsgs_result = await baby_step_giant_step(g, h, n)
            bsgs_time = time.time() - start_time
            
            # Results should match
            assert naive_result == bsgs_result, f"Algorithms should agree for log_{g}({h}) mod {n}"
            
            # For small cases, times should be reasonable
            assert naive_time < 1.0, "Naive algorithm should be fast for small cases"
            assert bsgs_time < 1.0, "BSGS algorithm should be fast for small cases"

# ============================================================================
# ERROR HANDLING TESTS
# ============================================================================

class TestErrorHandling:
    """Test error handling and edge cases."""
    
    @pytest.mark.asyncio
    async def test_invalid_crt_inputs(self):
        """Test handling of invalid CRT inputs."""
        # Test mismatched array lengths - graceful handling instead of exceptions
        try:
            result = await crt_solve([1, 2], [3])
            assert result is None or isinstance(result, tuple)
        except (ValueError, IndexError):
            pass  # Either exception or graceful handling is acceptable
        
        try:
            result = await crt_solve([1], [3, 5])
            assert result is None or isinstance(result, tuple)
        except (ValueError, IndexError):
            pass  # Either exception or graceful handling is acceptable
        
        # Test empty inputs
        result = await crt_solve([], [])
        assert result is None, "Empty CRT should return None"
        
        result = await generalized_crt([])
        assert result is None, "Empty generalized CRT should return None"
    
    @pytest.mark.asyncio
    async def test_invalid_symbol_inputs(self):
        """Test handling of invalid inputs for symbol functions."""
        # Legendre symbol with non-prime
        with pytest.raises(ValueError):
            await legendre_symbol(2, 4)
        
        with pytest.raises(ValueError):
            await legendre_symbol(2, 8)
        
        # Jacobi symbol with even n
        with pytest.raises(ValueError):
            await jacobi_symbol(2, 4)
        
        with pytest.raises(ValueError):
            await jacobi_symbol(2, 6)
        
        # Jacobi symbol with non-positive n
        with pytest.raises(ValueError):
            await jacobi_symbol(2, 0)
        
        with pytest.raises(ValueError):
            await jacobi_symbol(2, -3)
    
    @pytest.mark.asyncio
    async def test_invalid_tonelli_shanks_inputs(self):
        """Test handling of invalid inputs for Tonelli-Shanks."""
        # Non-prime modulus
        result = await tonelli_shanks(2, 8)
        assert result is None, "Tonelli-Shanks should return None for composite modulus"
        
        # Even prime (p = 2)
        result = await tonelli_shanks(1, 2)
        assert result is None, "Tonelli-Shanks should return None for p = 2"
        
        # Non-quadratic residue
        result = await tonelli_shanks(3, 7)
        assert result is None, "Should return None for non-QR"
    
    @pytest.mark.asyncio
    async def test_order_modulo_invalid_inputs(self):
        """Test handling of invalid inputs for order calculation."""
        # gcd(a, n) > 1
        result = await order_modulo(2, 4)
        assert result is None, "Order undefined when gcd(a, n) > 1"
        
        result = await order_modulo(6, 9)
        assert result is None, "Order undefined when gcd(a, n) > 1"
        
        # n ≤ 1
        result = await order_modulo(2, 1)
        assert result is None, "Order undefined for n ≤ 1"
        
        result = await order_modulo(2, 0)
        assert result is None, "Order undefined for n ≤ 1"
    
    @pytest.mark.asyncio
    async def test_discrete_log_invalid_inputs(self):
        """Test handling of invalid inputs for discrete logarithm."""
        # n ≤ 1
        result = await discrete_log_naive(2, 3, 1)
        assert result is None, "Discrete log undefined for n ≤ 1"
        
        result = await baby_step_giant_step(2, 3, 1)
        assert result is None, "BSGS undefined for n ≤ 1"
        
        # Test with max_exp = 0
        result = await discrete_log_naive(2, 3, 7, max_exp=0)
        assert result is None, "Should return None with max_exp=0 and h≠1"
    
    @pytest.mark.asyncio
    async def test_error_preserves_async_context(self):
        """Test that operations continue working after errors."""
        # Test that errors don't break subsequent operations
        try:
            await legendre_symbol(2, 4)  # Should raise ValueError
        except ValueError:
            pass
        
        result = await legendre_symbol(2, 7)
        assert result == 1
        
        try:
            await jacobi_symbol(2, 4)  # Should raise ValueError
        except ValueError:
            pass
        
        result = await jacobi_symbol(2, 15)
        assert result == 1
        
        try:
            await order_modulo(2, 4)  # Should return None
        except:
            pass
        
        result = await order_modulo(3, 7)
        assert result == 6

# ============================================================================
# PARAMETRIZED TESTS
# ============================================================================

class TestParametrized:
    """Parametrized tests for comprehensive coverage."""
    
    @pytest.mark.asyncio
    @pytest.mark.parametrize("remainders,moduli,expected", [
        ([2, 3, 2], [3, 5, 7], (23, 105)),
        ([1, 2], [2, 3], (5, 6)),
        ([0, 0], [4, 6], (0, 12)),
        ([1, 0], [3, 4], (4, 12)),  # Corrected: was (9, 12), should be (4, 12)
    ])
    async def test_crt_solve_parametrized(self, remainders, moduli, expected):
        """Parametrized test for CRT solving."""
        result = await crt_solve(remainders, moduli)
        assert result is not None, f"CRT should have solution for {remainders}, {moduli}"
        assert result == expected, f"CRT solution should be {expected}, got {result}"
    
    @pytest.mark.asyncio
    @pytest.mark.parametrize("a,p,expected", [
        (1, 7, True), (2, 7, True), (3, 7, False), (4, 7, True),
        (1, 11, True), (2, 11, False), (3, 11, True), (4, 11, True), (5, 11, True),
        (0, 7, True), (0, 11, True)  # 0 is always QR
    ])
    async def test_is_quadratic_residue_parametrized(self, a, p, expected):
        """Parametrized test for quadratic residue testing."""
        assert await is_quadratic_residue(a, p) == expected
    
    @pytest.mark.asyncio
    @pytest.mark.parametrize("a,p,expected", [
        (1, 7, 1), (2, 7, 1), (3, 7, -1), (4, 7, 1), (7, 7, 0),
        (1, 11, 1), (2, 11, -1), (3, 11, 1), (4, 11, 1), (11, 11, 0)
    ])
    async def test_legendre_symbol_parametrized(self, a, p, expected):
        """Parametrized test for Legendre symbol calculation."""
        assert await legendre_symbol(a, p) == expected
    
    @pytest.mark.asyncio
    @pytest.mark.parametrize("a,n,expected", [
        (2, 15, 1), (5, 21, 1), (3, 9, 0), (1, 15, 1), (8, 15, 1)  # Corrected (5, 21) to 1
    ])
    async def test_jacobi_symbol_parametrized(self, a, n, expected):
        """Parametrized test for Jacobi symbol calculation."""
        assert await jacobi_symbol(a, n) == expected
    
    @pytest.mark.asyncio
    @pytest.mark.parametrize("n,expected", [
        (5, 2), (7, 3), (11, 2), (13, 2), (17, 3), (19, 2)
    ])
    async def test_primitive_root_parametrized(self, n, expected):
        """Parametrized test for primitive root calculation."""
        assert await primitive_root(n) == expected
    
    @pytest.mark.asyncio
    @pytest.mark.parametrize("a,n,expected", [
        (3, 7, 6), (2, 7, 3), (4, 7, 3), (6, 7, 2),
        (2, 11, 10), (3, 11, 5), (4, 11, 5), (5, 11, 5)
    ])
    async def test_order_modulo_parametrized(self, a, n, expected):
        """Parametrized test for multiplicative order calculation."""
        assert await order_modulo(a, n) == expected

# ============================================================================
# SPECIAL MATHEMATICAL PROPERTIES TESTS
# ============================================================================

class TestSpecialProperties:
    """Test special mathematical properties and advanced relationships."""
    
    @pytest.mark.asyncio
    async def test_quadratic_reciprocity_law(self):
        """Test the quadratic reciprocity law."""
        # For distinct odd primes p, q: (p/q)(q/p) = (-1)^((p-1)(q-1)/4)
        prime_pairs = [(3, 5), (3, 7), (5, 7), (7, 11), (11, 13)]
        
        for p, q in prime_pairs:
            legendre_p_q = await legendre_symbol(p, q)
            legendre_q_p = await legendre_symbol(q, p)
            
            # Calculate (-1)^((p-1)(q-1)/4)
            exponent = ((p - 1) * (q - 1)) // 4
            expected_product = (-1) ** exponent
            
            actual_product = legendre_p_q * legendre_q_p
            
            assert actual_product == expected_product, f"Quadratic reciprocity fails for ({p}, {q})"
    
    @pytest.mark.asyncio
    async def test_euler_criterion_verification(self):
        """Test Euler's criterion: (a/p) ≡ a^((p-1)/2) (mod p)."""
        primes = [7, 11, 13, 17]
        
        for p in primes:
            for a in range(1, min(p, 10)):
                legendre = await legendre_symbol(a, p)
                euler_power = pow(a, (p - 1) // 2, p)
                
                # Convert to standard form (-1 becomes p-1)
                if euler_power == p - 1:
                    euler_power = -1
                
                assert legendre == euler_power, f"Euler criterion fails for a={a}, p={p}"
    
    @pytest.mark.asyncio
    async def test_wilson_theorem_relation(self):
        """Test Wilson's theorem: (p-1)! ≡ -1 (mod p) for prime p."""
        primes = [5, 7, 11, 13]
        
        for p in primes:
            factorial = 1
            for i in range(1, p):
                factorial = (factorial * i) % p
            
            # Wilson's theorem: (p-1)! ≡ -1 ≡ p-1 (mod p)
            assert factorial == p - 1, f"Wilson's theorem fails for p={p}"
    
    @pytest.mark.asyncio
    async def test_fermat_little_theorem_verification(self):
        """Test Fermat's Little Theorem: a^(p-1) ≡ 1 (mod p) for gcd(a,p)=1."""
        primes = [7, 11, 13]
        
        for p in primes:
            for a in range(2, min(p, 8)):
                if math.gcd(a, p) == 1:
                    result = pow(a, p - 1, p)
                    assert result == 1, f"Fermat's Little Theorem fails for a={a}, p={p}"
    
    @pytest.mark.asyncio
    async def test_primitive_root_count(self):
        """Test that number of primitive roots equals φ(φ(p)) = φ(p-1)."""
        primes = [5, 7, 11, 13]
        
        for p in primes:
            all_roots = await all_primitive_roots(p)
            
            # Calculate φ(p-1)
            phi_p_minus_1 = await _euler_totient_simple(p - 1)
            
            assert len(all_roots) == phi_p_minus_1, f"Number of primitive roots mod {p} should be φ({p-1}) = {phi_p_minus_1}"

# ============================================================================
# HELPER FUNCTIONS FOR TESTS
# ============================================================================

async def _euler_totient_simple(n: int) -> int:
    """Simple Euler totient calculation for test verification."""
    if n <= 1:
        return 0
    
    result = 0
    for i in range(1, n + 1):
        if math.gcd(i, n) == 1:
            result += 1
    
    return result

if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v", "--tb=short", "--asyncio-mode=auto"])