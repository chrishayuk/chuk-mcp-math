#!/usr/bin/env python3
"""
Comprehensive pytest test suite for diophantine_equations.py module.

Tests cover:
- Linear Diophantine equations: solve_linear_diophantine, count_solutions_diophantine, parametric_solutions_diophantine
- Pell's equation: solve_pell_equation, pell_solutions_generator, solve_negative_pell_equation
- Quadratic equations: pythagorean_triples, sum_of_two_squares_all, solve_quadratic_diophantine
- Special problems: frobenius_number, postage_stamp_problem
- Analysis utilities: diophantine_analysis
- Mathematical properties and relationships
- Edge cases, error conditions, and performance testing
- Async behavior verification

Run with: python -m pytest test_diophantine_equations.py -v --tb=short --asyncio-mode=auto
"""

import pytest
import asyncio
import time
import math
from typing import List, Dict, Any

# Import the functions to test
try:
    from chuk_mcp_functions.math.number_theory.diophantine_equations import (
        # Linear Diophantine equations
        solve_linear_diophantine, count_solutions_diophantine, parametric_solutions_diophantine,
        
        # Pell's equation
        solve_pell_equation, pell_solutions_generator, solve_negative_pell_equation,
        
        # Quadratic Diophantine equations
        pythagorean_triples, sum_of_two_squares_all, solve_quadratic_diophantine,
        
        # Special problems
        frobenius_number, postage_stamp_problem,
        
        # Analysis utilities
        diophantine_analysis
    )
except ImportError as e:
    pytest.skip(f"diophantine_equations module not available: {e}", allow_module_level=True)

# ============================================================================
# LINEAR DIOPHANTINE EQUATIONS TESTS
# ============================================================================

class TestLinearDiophantine:
    """Test cases for linear Diophantine equation functions."""
    
    @pytest.mark.asyncio
    async def test_solve_linear_diophantine_solvable(self):
        """Test solvable linear Diophantine equations."""
        # 3x + 5y = 1 (solvable since gcd(3,5) = 1 divides 1)
        result = await solve_linear_diophantine(3, 5, 1)
        
        assert result["solvable"] == True
        assert result["gcd"] == 1
        assert "particular" in result
        assert "general" in result
        
        # Verify the particular solution
        x0, y0 = result["particular"]
        assert 3 * x0 + 5 * y0 == 1
    
    @pytest.mark.asyncio
    async def test_solve_linear_diophantine_3x_5y_1(self):
        """Test specific case 3x + 5y = 1."""
        result = await solve_linear_diophantine(3, 5, 1)
        
        assert result["solvable"] == True
        x0, y0 = result["particular"]
        
        # One known solution is (-2, 1): 3(-2) + 5(1) = -6 + 5 = -1... wait
        # Actually: 3(2) + 5(-1) = 6 - 5 = 1, so (2, -1) is a solution
        # Or: 3(-3) + 5(2) = -9 + 10 = 1, so (-3, 2) is a solution
        assert 3 * x0 + 5 * y0 == 1
        
        # Check step sizes
        assert result["step_x"] == 5  # b/gcd = 5/1 = 5
        assert result["step_y"] == -3  # -a/gcd = -3/1 = -3
    
    @pytest.mark.asyncio
    async def test_solve_linear_diophantine_unsolvable(self):
        """Test unsolvable linear Diophantine equations."""
        # 6x + 9y = 7 (unsolvable since gcd(6,9) = 3 does not divide 7)
        result = await solve_linear_diophantine(6, 9, 7)
        
        assert result["solvable"] == False
        assert result["gcd"] == 3
        assert "reason" in result
        assert "3 does not divide 7" in result["reason"]
    
    @pytest.mark.asyncio
    async def test_solve_linear_diophantine_2x_3y_7(self):
        """Test case 2x + 3y = 7."""
        result = await solve_linear_diophantine(2, 3, 7)
        
        assert result["solvable"] == True
        assert result["gcd"] == 1
        
        x0, y0 = result["particular"]
        assert 2 * x0 + 3 * y0 == 7
        
        # Verify general solution format
        assert result["step_x"] == 3  # b/gcd = 3/1 = 3
        assert result["step_y"] == -2  # -a/gcd = -2/1 = -2
    
    @pytest.mark.asyncio
    async def test_solve_linear_diophantine_edge_cases(self):
        """Test edge cases for linear Diophantine equations."""
        # Both coefficients zero with c = 0 (infinitely many solutions)
        result = await solve_linear_diophantine(0, 0, 0)
        assert result["solvable"] == True
        assert "any integers" in result["general"]
        
        # Both coefficients zero with c ≠ 0 (no solutions)
        result = await solve_linear_diophantine(0, 0, 5)
        assert result["solvable"] == False
        assert "0x + 0y = c where c ≠ 0" in result["reason"]
    
    @pytest.mark.asyncio
    async def test_solve_linear_diophantine_negative_coefficients(self):
        """Test with negative coefficients."""
        result = await solve_linear_diophantine(-3, 5, 1)
        assert result["solvable"] == True
        
        x0, y0 = result["particular"]
        assert -3 * x0 + 5 * y0 == 1
    
    @pytest.mark.asyncio
    async def test_count_solutions_diophantine_basic(self):
        """Test counting solutions within bounds."""
        # 2x + 3y = 12 with bounds x∈[0,6], y∈[0,4]
        # Solutions: (0,4), (3,2), (6,0)
        count = await count_solutions_diophantine(2, 3, 12, 0, 6, 0, 4)
        assert count == 3
    
    @pytest.mark.asyncio
    async def test_count_solutions_diophantine_x_plus_y_10(self):
        """Test x + y = 10 with bounds."""
        # x + y = 10 with x∈[0,10], y∈[0,10]
        # Solutions: (0,10), (1,9), (2,8), ..., (10,0) = 11 solutions
        count = await count_solutions_diophantine(1, 1, 10, 0, 10, 0, 10)
        assert count == 11
    
    @pytest.mark.asyncio
    async def test_count_solutions_diophantine_no_solutions(self):
        """Test counting when no solutions exist."""
        # Unsolvable equation
        count = await count_solutions_diophantine(6, 9, 7, 0, 10, 0, 10)
        assert count == 0
        
        # Solvable but no solutions in bounds
        count = await count_solutions_diophantine(1, 1, 100, 0, 10, 0, 10)
        assert count == 0
    
    @pytest.mark.asyncio
    async def test_parametric_solutions_diophantine(self):
        """Test parametric solution generation."""
        # 3x + 5y = 1 for t ∈ [-2, 2]
        solutions = await parametric_solutions_diophantine(3, 5, 1, -2, 2)
        
        assert len(solutions) == 5  # t = -2, -1, 0, 1, 2
        
        # Verify each solution
        for x, y in solutions:
            assert 3 * x + 5 * y == 1
    
    @pytest.mark.asyncio
    async def test_parametric_solutions_diophantine_unsolvable(self):
        """Test parametric solutions for unsolvable equation."""
        solutions = await parametric_solutions_diophantine(6, 9, 7, 0, 5)
        assert solutions == []

# ============================================================================
# PELL'S EQUATION TESTS
# ============================================================================

class TestPellEquation:
    """Test cases for Pell's equation functions."""
    
    @pytest.mark.asyncio
    async def test_solve_pell_equation_n_2(self):
        """Test Pell's equation x² - 2y² = 1."""
        result = await solve_pell_equation(2)
        
        assert result["exists"] == True
        x, y = result["fundamental"]
        
        # Verify solution: x² - 2y² = 1
        assert x * x - 2 * y * y == 1
        
        # Known fundamental solution is (3, 2)
        assert x == 3 and y == 2
    
    @pytest.mark.asyncio
    async def test_solve_pell_equation_n_3(self):
        """Test Pell's equation x² - 3y² = 1."""
        result = await solve_pell_equation(3)
        
        assert result["exists"] == True
        x, y = result["fundamental"]
        assert x * x - 3 * y * y == 1
        
        # Known fundamental solution is (2, 1)
        assert x == 2 and y == 1
    
    @pytest.mark.asyncio
    async def test_solve_pell_equation_n_5(self):
        """Test Pell's equation x² - 5y² = 1."""
        result = await solve_pell_equation(5)
        
        assert result["exists"] == True
        x, y = result["fundamental"]
        assert x * x - 5 * y * y == 1
        
        # Known fundamental solution is (9, 4)
        assert x == 9 and y == 4
    
    @pytest.mark.asyncio
    async def test_solve_pell_equation_perfect_square(self):
        """Test Pell's equation with perfect square (no non-trivial solutions)."""
        result = await solve_pell_equation(4)
        
        assert result["exists"] == False
        assert "perfect square" in result["reason"]
    
    @pytest.mark.asyncio
    async def test_solve_pell_equation_edge_cases(self):
        """Test edge cases for Pell's equation."""
        # n = 0
        with pytest.raises(ValueError, match="n must be positive"):
            await solve_pell_equation(0)
        
        # n < 0
        with pytest.raises(ValueError, match="n must be positive"):
            await solve_pell_equation(-1)
    
    @pytest.mark.asyncio
    async def test_pell_solutions_generator_n_2(self):
        """Test generating multiple Pell solutions for n=2."""
        solutions = await pell_solutions_generator(2, 5)
        
        assert len(solutions) == 5
        
        # Known solutions for x² - 2y² = 1:
        # (3,2), (17,12), (99,70), (577,408), (3363,2378)
        expected = [(3, 2), (17, 12), (99, 70), (577, 408), (3363, 2378)]
        
        for i, (x, y) in enumerate(solutions):
            assert x * x - 2 * y * y == 1
            assert (x, y) == expected[i]
    
    @pytest.mark.asyncio
    async def test_pell_solutions_generator_n_3(self):
        """Test generating multiple Pell solutions for n=3."""
        solutions = await pell_solutions_generator(3, 4)
        
        assert len(solutions) == 4
        
        # Known solutions for x² - 3y² = 1:
        # (2,1), (7,4), (26,15), (97,56)
        expected = [(2, 1), (7, 4), (26, 15), (97, 56)]
        
        for i, (x, y) in enumerate(solutions):
            assert x * x - 3 * y * y == 1
            assert (x, y) == expected[i]
    
    @pytest.mark.asyncio
    async def test_pell_solutions_generator_edge_cases(self):
        """Test edge cases for Pell solutions generator."""
        # num_solutions = 0
        solutions = await pell_solutions_generator(2, 0)
        assert solutions == []
        
        # Perfect square n
        solutions = await pell_solutions_generator(4, 3)
        assert solutions == []
    
    @pytest.mark.asyncio
    async def test_solve_negative_pell_equation_n_2(self):
        """Test negative Pell's equation x² - 2y² = -1."""
        result = await solve_negative_pell_equation(2)
        
        assert result["exists"] == True
        x, y = result["fundamental"]
        
        # Verify solution: x² - 2y² = -1
        assert x * x - 2 * y * y == -1
        
        # Known fundamental solution is (1, 1)
        assert x == 1 and y == 1
    
    @pytest.mark.asyncio
    async def test_solve_negative_pell_equation_n_5(self):
        """Test negative Pell's equation x² - 5y² = -1."""
        result = await solve_negative_pell_equation(5)
        
        assert result["exists"] == True
        x, y = result["fundamental"]
        assert x * x - 5 * y * y == -1
        
        # Known fundamental solution is (2, 1)
        assert x == 2 and y == 1
    
    @pytest.mark.asyncio
    async def test_solve_negative_pell_equation_no_solution(self):
        """Test negative Pell's equation with no solutions."""
        # x² - 3y² = -1 has no solutions
        result = await solve_negative_pell_equation(3)
        
        assert result["exists"] == False
        assert "even" in result["reason"] or "No solutions exist" in result["reason"]
    
    @pytest.mark.asyncio
    async def test_solve_negative_pell_equation_perfect_square(self):
        """Test negative Pell's equation with perfect square."""
        result = await solve_negative_pell_equation(4)
        
        assert result["exists"] == False
        assert "perfect square" in result["reason"]

# ============================================================================
# QUADRATIC DIOPHANTINE EQUATIONS TESTS
# ============================================================================

class TestQuadraticDiophantine:
    """Test cases for quadratic Diophantine equation functions."""
    
    @pytest.mark.asyncio
    async def test_pythagorean_triples_small_limit(self):
        """Test Pythagorean triples with small limit."""
        triples = await pythagorean_triples(25)
        
        # Should include [3,4,5], [5,12,13], [8,15,17], [7,24,25] (note: lists not tuples)
        expected_triples = [[3, 4, 5], [5, 12, 13], [8, 15, 17], [7, 24, 25]]
        
        for triple in expected_triples:
            assert triple in triples
        
        # Verify each triple
        for a, b, c in triples:
            assert a * a + b * b == c * c
            assert c <= 25
    
    @pytest.mark.asyncio
    async def test_pythagorean_triples_primitive_only(self):
        """Test primitive Pythagorean triples only."""
        triples = await pythagorean_triples(50, primitive_only=True)
        
        # Verify each triple is primitive (gcd(a,b,c) = 1)
        for a, b, c in triples:
            assert a * a + b * b == c * c
            assert math.gcd(math.gcd(a, b), c) == 1
            assert c <= 50
        
        # Should include known primitive triples
        assert [3, 4, 5] in triples
        assert [5, 12, 13] in triples
        assert [8, 15, 17] in triples
    
    @pytest.mark.asyncio
    async def test_pythagorean_triples_includes_multiples(self):
        """Test that non-primitive mode includes multiples."""
        all_triples = await pythagorean_triples(30, primitive_only=False)
        primitive_triples = await pythagorean_triples(30, primitive_only=True)
        
        # Should have more triples when including multiples
        assert len(all_triples) >= len(primitive_triples)
        
        # Should include multiples like [6,8,10] = 2*[3,4,5]
        assert [6, 8, 10] in all_triples
        assert [9, 12, 15] in all_triples  # 3*[3,4,5]
    
    @pytest.mark.asyncio
    async def test_pythagorean_triples_sorted(self):
        """Test that triples are returned in sorted order."""
        triples = await pythagorean_triples(100)
        
        # Should be sorted by hypotenuse, then by first leg
        for i in range(1, len(triples)):
            a1, b1, c1 = triples[i-1]
            a2, b2, c2 = triples[i]
            
            assert c1 <= c2  # Sorted by hypotenuse
            if c1 == c2:
                assert a1 <= a2  # Then by first leg
    
    @pytest.mark.asyncio
    async def test_sum_of_two_squares_all_basic(self):
        """Test sum of two squares representation."""
        # 25 = 0² + 5² = 3² + 4²
        representations = await sum_of_two_squares_all(25)
        
        assert [0, 5] in representations
        assert [3, 4] in representations
        
        # Verify each representation
        for x, y in representations:
            assert x * x + y * y == 25
            assert x <= y  # Should be in canonical form
    
    @pytest.mark.asyncio
    async def test_sum_of_two_squares_all_13(self):
        """Test sum of two squares for 13."""
        # 13 = 2² + 3²
        representations = await sum_of_two_squares_all(13)
        
        assert [2, 3] in representations
        
        for x, y in representations:
            assert x * x + y * y == 13
            assert x <= y
    
    @pytest.mark.asyncio
    async def test_sum_of_two_squares_all_50(self):
        """Test sum of two squares for 50."""
        # 50 = 1² + 7² = 5² + 5²
        representations = await sum_of_two_squares_all(50)
        
        assert [1, 7] in representations
        assert [5, 5] in representations
        
        for x, y in representations:
            assert x * x + y * y == 50
            assert x <= y
    
    @pytest.mark.asyncio
    async def test_sum_of_two_squares_edge_cases(self):
        """Test edge cases for sum of two squares."""
        # n = 0
        result = await sum_of_two_squares_all(0)
        assert result == [[0, 0]]
        
        # n = 1
        result = await sum_of_two_squares_all(1)
        assert result == [[0, 1]]
        
        # Negative n
        result = await sum_of_two_squares_all(-5)
        assert result == []
    
    @pytest.mark.asyncio
    async def test_solve_quadratic_diophantine_circle(self):
        """Test quadratic Diophantine for circle x² + y² = 25."""
        # x² + y² = 25 → coeffs = [1, 0, 1, 0, 0, -25]
        solutions = await solve_quadratic_diophantine([1, 0, 1, 0, 0, -25], [-5, 5])
        
        # Should find solutions like [±3,±4], [±4,±3], [0,±5], [±5,0]
        expected_points = [
            [-5, 0], [-4, -3], [-4, 3], [-3, -4], [-3, 4],
            [0, -5], [0, 5], [3, -4], [3, 4], [4, -3], [4, 3], [5, 0]
        ]
        
        for point in expected_points:
            assert point in solutions
        
        # Verify each solution
        for x, y in solutions:
            assert x * x + y * y == 25
    
    @pytest.mark.asyncio
    async def test_solve_quadratic_diophantine_hyperbola(self):
        """Test quadratic Diophantine for hyperbola x² - y² = 0."""
        # x² - y² = 0 → coeffs = [1, 0, -1, 0, 0, 0]
        solutions = await solve_quadratic_diophantine([1, 0, -1, 0, 0, 0], [-3, 3])
        
        # Should find solutions where x = ±y
        expected_solutions = [
            [-3, -3], [-3, 3], [-2, -2], [-2, 2], [-1, -1], [-1, 1],
            [0, 0], [1, -1], [1, 1], [2, -2], [2, 2], [3, -3], [3, 3]
        ]
        
        for sol in expected_solutions:
            assert sol in solutions
        
        # Verify each solution
        for x, y in solutions:
            assert x * x - y * y == 0

# ============================================================================
# SPECIAL DIOPHANTINE PROBLEMS TESTS
# ============================================================================

class TestSpecialDiophantineProblems:
    """Test cases for special Diophantine problems."""
    
    @pytest.mark.asyncio
    async def test_frobenius_number_3_5(self):
        """Test Frobenius number for [3, 5]."""
        # Known result: Chicken McNugget problem
        frobenius = await frobenius_number([3, 5])
        assert frobenius == 7
        
        # 7 cannot be expressed as 3a + 5b with a,b ≥ 0
        # But 8 = 3(1) + 5(1), 9 = 3(3) + 5(0), 10 = 3(0) + 5(2), etc.
    
    @pytest.mark.asyncio
    async def test_frobenius_number_4_6_9(self):
        """Test Frobenius number for [4, 6, 9]."""
        frobenius = await frobenius_number([4, 6, 9])
        
        # Should be a finite number (since gcd(4,6,9) = gcd(4,gcd(6,9)) = gcd(4,3) = 1)
        assert isinstance(frobenius, int)
        assert frobenius >= 0
    
    @pytest.mark.asyncio
    async def test_frobenius_number_coprime_two_numbers(self):
        """Test Frobenius number formula for two coprime numbers."""
        # For coprime a, b: Frobenius number = ab - a - b
        a, b = 7, 11  # Coprime
        frobenius = await frobenius_number([a, b])
        expected = a * b - a - b  # 7*11 - 7 - 11 = 77 - 18 = 59
        assert frobenius == expected
    
    @pytest.mark.asyncio
    async def test_frobenius_number_non_coprime(self):
        """Test Frobenius number for non-coprime numbers."""
        # If gcd > 1, should return infinity
        frobenius = await frobenius_number([6, 9])  # gcd(6,9) = 3
        assert frobenius == float('inf')
    
    @pytest.mark.asyncio
    async def test_frobenius_number_edge_cases(self):
        """Test edge cases for Frobenius number."""
        # Empty list
        with pytest.raises(ValueError):
            await frobenius_number([])
        
        # Single number
        frobenius = await frobenius_number([5])
        assert frobenius == -1 or frobenius == float('inf')
        
        # With 1 (everything representable)
        frobenius = await frobenius_number([1, 3, 5])
        assert frobenius == -1
    
    @pytest.mark.asyncio
    async def test_postage_stamp_problem_solvable(self):
        """Test solvable postage stamp problems."""
        # 17 = 4*3 + 1*5
        result = await postage_stamp_problem(17, [3, 5])
        
        assert result["possible"] == True
        solution = result["solution"]
        assert len(solution) == 2  # Two denominations
        
        # Verify solution
        total = solution[0] * 3 + solution[1] * 5
        assert total == 17
        
        # Check breakdown
        assert "denomination_breakdown" in result
    
    @pytest.mark.asyncio
    async def test_postage_stamp_problem_43_with_5_9_20(self):
        """Test specific case: 43 with denominations [5, 9, 20]."""
        result = await postage_stamp_problem(43, [5, 9, 20])
        
        assert result["possible"] == True
        solution = result["solution"]
        
        # Verify solution: should sum to 43
        total = solution[0] * 5 + solution[1] * 9 + solution[2] * 20
        assert total == 43
    
    @pytest.mark.asyncio
    async def test_postage_stamp_problem_unsolvable(self):
        """Test unsolvable postage stamp problems."""
        # Let's check what 11 with [3, 5] actually returns
        result = await postage_stamp_problem(11, [3, 5])
        print(f"postage_stamp_problem(11, [3, 5]) = {result}")
        
        # 11 with denominations [3, 5] - let's verify manually:
        # 11 = 3a + 5b needs non-negative integers a, b
        # If b=0: 11 = 3a → a = 11/3 = 3.67 (not integer)
        # If b=1: 11 = 3a + 5 → 6 = 3a → a = 2 → 11 = 3*2 + 5*1 = 6 + 5 = 11 ✓
        # So actually 11 IS representable as 3*2 + 5*1 = 11
        
        # Let's try a truly unsolvable case: 1 with [3, 5]
        result = await postage_stamp_problem(1, [3, 5])
        assert result["possible"] == False
        
        # Or 2 with [3, 5]  
        result = await postage_stamp_problem(2, [3, 5])
        assert result["possible"] == False
    
    @pytest.mark.asyncio
    async def test_postage_stamp_problem_edge_cases(self):
        """Test edge cases for postage stamp problem."""
        # Amount = 0
        result = await postage_stamp_problem(0, [3, 5])
        assert result["possible"] == True
        assert result["solution"] == [0, 0]
        assert result["stamps_used"] == 0
        
        # Negative amount
        result = await postage_stamp_problem(-5, [3, 5])
        assert result["possible"] == False
        
        # No denominations
        result = await postage_stamp_problem(10, [])
        assert result["possible"] == False
    
    @pytest.mark.asyncio
    async def test_postage_stamp_problem_optimization(self):
        """Test that postage stamp problem finds minimum stamps."""
        # For amount that can be made in multiple ways, should use minimum stamps
        result = await postage_stamp_problem(15, [3, 5])
        
        assert result["possible"] == True
        # 15 = 5*3 (3 stamps) or 15 = 3*5 (5 stamps)
        # Should prefer 5*3 = 15 (3 stamps)
        assert result["stamps_used"] == 3

# ============================================================================
# DIOPHANTINE ANALYSIS TESTS
# ============================================================================

class TestDiophantineAnalysis:
    """Test cases for Diophantine analysis utilities."""
    
    @pytest.mark.asyncio
    async def test_diophantine_analysis_linear(self):
        """Test analysis of linear Diophantine equations."""
        result = await diophantine_analysis("linear", coefficients=[3, 5, 1])
        
        assert result["type"] == "linear"
        assert result["solvable"] == True
        assert result["infinite_solutions"] == True
        assert result["classification"] == "indefinite"
        assert result["gcd"] == 1
    
    @pytest.mark.asyncio
    async def test_diophantine_analysis_linear_unsolvable(self):
        """Test analysis of unsolvable linear equations."""
        result = await diophantine_analysis("linear", coefficients=[6, 9, 7])
        
        assert result["type"] == "linear"
        assert result["solvable"] == False
        assert result["classification"] == "inconsistent"
        assert result["gcd"] == 3
    
    @pytest.mark.asyncio
    async def test_diophantine_analysis_pell(self):
        """Test analysis of Pell's equation."""
        result = await diophantine_analysis("pell", n=2)
        
        assert result["type"] == "pell"
        assert result["has_solutions"] == True
        assert result["fundamental_solution"] == [3, 2]
        assert result["classification"] == "hyperbolic"
        assert result["infinite_solutions"] == True
    
    @pytest.mark.asyncio
    async def test_diophantine_analysis_negative_pell(self):
        """Test analysis of negative Pell's equation."""
        # n=2 has solutions for x² - 2y² = -1
        result = await diophantine_analysis("negative_pell", n=2)
        
        assert result["type"] == "negative_pell"
        assert result["has_solutions"] == True
        assert result["fundamental_solution"] == [1, 1]
        
        # n=3 has no solutions for x² - 3y² = -1
        result = await diophantine_analysis("negative_pell", n=3)
        
        assert result["type"] == "negative_pell"
        assert result["has_solutions"] == False
        assert result["classification"] == "no_solutions"
    
    @pytest.mark.asyncio
    async def test_diophantine_analysis_pythagorean(self):
        """Test analysis of Pythagorean triples."""
        result = await diophantine_analysis("pythagorean", limit=100)
        
        assert result["type"] == "pythagorean"
        assert result["classification"] == "elliptic"
        assert result["infinite_solutions"] == True
        assert "parametric_form" in result
        assert result["primitive_triples_found"] > 0
    
    @pytest.mark.asyncio
    async def test_diophantine_analysis_unknown_type(self):
        """Test analysis with unknown equation type."""
        result = await diophantine_analysis("unknown_type")
        
        assert result["type"] == "unknown_type"
        assert "error" in result
        assert "Unknown equation type" in result["error"]

# ============================================================================
# INTEGRATION AND MATHEMATICAL PROPERTIES TESTS
# ============================================================================

class TestIntegrationAndProperties:
    """Integration tests and mathematical property verification."""
    
    @pytest.mark.asyncio
    async def test_linear_diophantine_solution_verification(self):
        """Verify that all parametric solutions satisfy the equation."""
        # Test multiple equations
        test_cases = [
            (3, 5, 1),
            (2, 3, 7),
            (4, 6, 2),
            (-3, 7, 5)
        ]
        
        for a, b, c in test_cases:
            solution = await solve_linear_diophantine(a, b, c)
            if solution["solvable"]:
                # Generate parametric solutions
                solutions = await parametric_solutions_diophantine(a, b, c, -5, 5)
                
                # Verify each solution
                for x, y in solutions:
                    assert a * x + b * y == c
    
    @pytest.mark.asyncio
    async def test_pell_equation_recurrence_relation(self):
        """Test Pell equation recurrence relation."""
        # If (x₁,y₁) is fundamental solution, then
        # (xₖ,yₖ) = ((x₁ + y₁√n)^k + (x₁ - y₁√n)^k)/2, ((x₁ + y₁√n)^k - (x₁ - y₁√n)^k)/(2√n)
        
        solutions = await pell_solutions_generator(2, 3)
        
        # Check recurrence: x_{k+1} = x₁*xₖ + n*y₁*yₖ, y_{k+1} = x₁*yₖ + y₁*xₖ
        x1, y1 = solutions[0]  # Fundamental solution
        
        for k in range(1, len(solutions)):
            xk, yk = solutions[k-1]
            x_next, y_next = solutions[k]
            
            expected_x = x1 * xk + 2 * y1 * yk
            expected_y = x1 * yk + y1 * xk
            
            assert x_next == expected_x
            assert y_next == expected_y
    
    @pytest.mark.asyncio
    async def test_pythagorean_triple_properties(self):
        """Test mathematical properties of Pythagorean triples."""
        triples = await pythagorean_triples(100, primitive_only=True)
        
        for a, b, c in triples:
            # Basic Pythagorean property
            assert a * a + b * b == c * c
            
            # Primitive property
            assert math.gcd(math.gcd(a, b), c) == 1
            
            # For primitive triples: exactly one of a,b is even
            assert (a % 2 == 0) != (b % 2 == 0)  # XOR: exactly one is even
            
            # c is always odd for primitive triples
            assert c % 2 == 1
    
    @pytest.mark.asyncio
    async def test_sum_of_two_squares_theorem(self):
        """Test sum of two squares theorem properties."""
        # A prime p can be expressed as sum of two squares iff p = 2 or p ≡ 1 (mod 4)
        test_primes = [2, 5, 13, 17, 29, 37, 41, 53, 61, 73, 89, 97]  # p ≡ 1 (mod 4) or p = 2
        
        for p in test_primes:
            if p == 2 or p % 4 == 1:
                representations = await sum_of_two_squares_all(p)
                assert len(representations) > 0  # Should have at least one representation
                
                for x, y in representations:
                    assert x * x + y * y == p
    
    @pytest.mark.asyncio
    async def test_frobenius_postage_stamp_consistency(self):
        """Test consistency between Frobenius number and postage stamp problem."""
        denominations = [3, 5]
        frobenius = await frobenius_number(denominations)
        
        if frobenius != float('inf') and frobenius >= 0:
            # Frobenius number should not be representable
            result = await postage_stamp_problem(frobenius, denominations)
            assert result["possible"] == False
            
            # All numbers > Frobenius should be representable
            # Test a few numbers after Frobenius
            for amount in range(frobenius + 1, frobenius + 10):
                result = await postage_stamp_problem(amount, denominations)
                assert result["possible"] == True
    
    @pytest.mark.asyncio
    async def test_pell_equation_solver_verification(self):
        """Test that Pell and negative Pell solvers work correctly for specific known cases."""
        # Test n=2 (we know both work from earlier tests)
        neg_pell_2 = await solve_negative_pell_equation(2)
        pos_pell_2 = await solve_pell_equation(2)
        
        assert pos_pell_2["exists"] == True
        assert neg_pell_2["exists"] == True
        
        # Verify solutions for n=2
        x, y = pos_pell_2["fundamental"]
        assert x * x - 2 * y * y == 1
        assert (x, y) == (3, 2)  # Known fundamental solution
        
        x, y = neg_pell_2["fundamental"]
        assert x * x - 2 * y * y == -1
        assert (x, y) == (1, 1)  # Known fundamental solution
        
        # Test n=5 (we know both work from earlier tests)
        neg_pell_5 = await solve_negative_pell_equation(5)
        pos_pell_5 = await solve_pell_equation(5)
        
        assert pos_pell_5["exists"] == True
        assert neg_pell_5["exists"] == True
        
        # Verify solutions for n=5
        x, y = pos_pell_5["fundamental"]
        assert x * x - 5 * y * y == 1
        assert (x, y) == (9, 4)  # Known fundamental solution
        
        x, y = neg_pell_5["fundamental"]
        assert x * x - 5 * y * y == -1
        assert (x, y) == (2, 1)  # Known fundamental solution
        
        # Test n=3 (positive has solutions, negative doesn't)
        neg_pell_3 = await solve_negative_pell_equation(3)
        pos_pell_3 = await solve_pell_equation(3)
        
        assert pos_pell_3["exists"] == True  # x² - 3y² = 1 has solutions
        assert neg_pell_3["exists"] == False  # x² - 3y² = -1 has no solutions
        
        # Verify solution for n=3 positive case
        x, y = pos_pell_3["fundamental"]
        assert x * x - 3 * y * y == 1
        assert (x, y) == (2, 1)  # Known fundamental solution
# ============================================================================
# PERFORMANCE AND ASYNC BEHAVIOR TESTS
# ============================================================================

class TestPerformance:
    """Performance and async behavior tests."""
    
    @pytest.mark.asyncio
    async def test_all_functions_are_async(self):
        """Test that all functions are properly async."""
        operations = [
            solve_linear_diophantine(3, 5, 1),
            count_solutions_diophantine(2, 3, 12, 0, 6, 0, 4),
            parametric_solutions_diophantine(3, 5, 1, -2, 2),
            solve_pell_equation(2),
            pell_solutions_generator(2, 3),
            solve_negative_pell_equation(2),
            pythagorean_triples(50),
            sum_of_two_squares_all(25),
            solve_quadratic_diophantine([1, 0, 1, 0, 0, -25], [-5, 5]),
            frobenius_number([3, 5]),
            postage_stamp_problem(17, [3, 5]),
            diophantine_analysis("linear", coefficients=[3, 5, 1])
        ]
        
        # Ensure all are coroutines
        for op in operations:
            assert asyncio.iscoroutine(op)
        
        # Run all operations concurrently
        results = await asyncio.gather(*operations)
        
        # Verify results have expected types and properties
        assert isinstance(results[0], dict)   # solve_linear_diophantine
        assert isinstance(results[1], int)    # count_solutions_diophantine
        assert isinstance(results[2], list)   # parametric_solutions_diophantine
        assert isinstance(results[3], dict)   # solve_pell_equation
        assert isinstance(results[4], list)   # pell_solutions_generator
        assert isinstance(results[5], dict)   # solve_negative_pell_equation
        assert isinstance(results[6], list)   # pythagorean_triples
        assert isinstance(results[7], list)   # sum_of_two_squares_all
        assert isinstance(results[8], list)   # solve_quadratic_diophantine
        assert isinstance(results[9], (int, float))  # frobenius_number
        assert isinstance(results[10], dict)  # postage_stamp_problem
        assert isinstance(results[11], dict)  # diophantine_analysis
    
    @pytest.mark.asyncio
    async def test_concurrent_execution(self):
        """Test that operations can run concurrently."""
        start_time = time.time()
        
        # Run multiple operations concurrently
        tasks = [
            solve_linear_diophantine(3, 5, 1),
            solve_pell_equation(2),
            pythagorean_triples(50),
            frobenius_number([3, 5]),
            postage_stamp_problem(17, [3, 5])
        ]
        
        results = await asyncio.gather(*tasks)
        duration = time.time() - start_time
        
        # Should complete in reasonable time due to async nature
        assert duration < 5.0
        assert len(results) == 5
        
        # Verify we got valid results
        assert results[0]["solvable"] == True    # Linear Diophantine
        assert results[1]["exists"] == True      # Pell equation
        assert len(results[2]) > 0               # Pythagorean triples
        assert isinstance(results[3], (int, float))  # Frobenius number
        assert results[4]["possible"] == True   # Postage stamp
    
    @pytest.mark.asyncio
    async def test_large_input_handling(self):
        """Test handling of moderately large inputs."""
        large_tests = [
            pythagorean_triples(200),
            pell_solutions_generator(2, 10),
            sum_of_two_squares_all(1000),
            count_solutions_diophantine(1, 1, 50, 0, 50, 0, 50),
            parametric_solutions_diophantine(2, 3, 7, -20, 20)
        ]
        
        results = await asyncio.gather(*large_tests)
        
        # Verify results are reasonable
        assert isinstance(results[0], list)  # Pythagorean triples
        assert isinstance(results[1], list)  # Pell solutions
        assert isinstance(results[2], list)  # Sum of squares
        assert isinstance(results[3], int)   # Count solutions
        assert isinstance(results[4], list)  # Parametric solutions
        
        # Check some basic properties
        pythagorean_triples_result = results[0]
        for a, b, c in pythagorean_triples_result:
            assert a * a + b * b == c * c
            assert c <= 200
        
        pell_solutions_result = results[1]
        assert len(pell_solutions_result) == 10
        for x, y in pell_solutions_result:
            assert x * x - 2 * y * y == 1

# ============================================================================
# ERROR HANDLING TESTS
# ============================================================================

class TestErrorHandling:
    """Test error handling and edge cases."""
    
    @pytest.mark.asyncio
    async def test_invalid_inputs(self):
        """Test handling of invalid inputs."""
        # Negative n for Pell equation
        with pytest.raises(ValueError):
            await solve_pell_equation(-1)
        
        with pytest.raises(ValueError):
            await solve_negative_pell_equation(0)
        
        # Invalid coefficients length for quadratic
        with pytest.raises(ValueError):
            await solve_quadratic_diophantine([1, 2, 3], [-5, 5])  # Wrong length
        
        # Invalid bounds length
        with pytest.raises(ValueError):
            await solve_quadratic_diophantine([1, 0, 1, 0, 0, -25], [-5])  # Wrong bounds length
    
    @pytest.mark.asyncio
    async def test_empty_or_invalid_denominations(self):
        """Test handling of empty or invalid denominations."""
        # Empty denominations for Frobenius
        with pytest.raises(ValueError):
            await frobenius_number([])
        
        # Negative denominations
        with pytest.raises(ValueError):
            await frobenius_number([3, -5])
        
        # Zero denomination
        with pytest.raises(ValueError):
            await frobenius_number([0, 5])
    
    @pytest.mark.asyncio
    async def test_boundary_conditions(self):
        """Test boundary conditions."""
        # Empty range for parametric solutions
        solutions = await parametric_solutions_diophantine(3, 5, 1, 5, 2)  # t_min > t_max
        assert solutions == []
        
        # Zero limit for Pythagorean triples
        triples = await pythagorean_triples(0)
        assert triples == []
        
        # Negative limit
        triples = await pythagorean_triples(-10)
        assert triples == []

# ============================================================================
# PARAMETRIZED TESTS
# ============================================================================

class TestParametrized:
    """Parametrized tests for comprehensive coverage."""
    
    @pytest.mark.asyncio
    @pytest.mark.parametrize("a,b,c,expected_solvable", [
        (3, 5, 1, True),    # gcd(3,5)=1 divides 1
        (6, 9, 7, False),   # gcd(6,9)=3 does not divide 7
        (2, 3, 7, True),    # gcd(2,3)=1 divides 7
        (4, 6, 2, True),    # gcd(4,6)=2 divides 2
        (10, 15, 8, False), # gcd(10,15)=5 does not divide 8
    ])
    async def test_linear_diophantine_solvability(self, a, b, c, expected_solvable):
        """Parametrized test for linear Diophantine solvability."""
        result = await solve_linear_diophantine(a, b, c)
        assert result["solvable"] == expected_solvable
        
        if expected_solvable:
            x0, y0 = result["particular"]
            assert a * x0 + b * y0 == c
    
    @pytest.mark.asyncio
    @pytest.mark.parametrize("n,expected_fundamental", [
        (2, (3, 2)),
        (3, (2, 1)),
        (5, (9, 4)),
        (7, (8, 3)),
        (10, (19, 6)),
    ])
    async def test_pell_equation_fundamental_solutions(self, n, expected_fundamental):
        """Parametrized test for Pell equation fundamental solutions."""
        result = await solve_pell_equation(n)
        
        assert result["exists"] == True
        x, y = result["fundamental"]
        
        # Verify it's a solution
        assert x * x - n * y * y == 1
        
        # Check if it matches known fundamental solution
        assert (x, y) == expected_fundamental
    
    @pytest.mark.asyncio
    @pytest.mark.parametrize("n,has_negative_solution", [
        (2, True),   # (1,1) is solution to x² - 2y² = -1
        (3, False),  # No solutions to x² - 3y² = -1
        (5, True),   # (2,1) is solution to x² - 5y² = -1
        (6, False),  # No solutions to x² - 6y² = -1
        (10, True),  # Has solutions to x² - 10y² = -1
    ])
    async def test_negative_pell_existence(self, n, has_negative_solution):
        """Parametrized test for negative Pell equation existence."""
        result = await solve_negative_pell_equation(n)
        assert result["exists"] == has_negative_solution
        
        if has_negative_solution:
            x, y = result["fundamental"]
            assert x * x - n * y * y == -1
    
    @pytest.mark.asyncio
    @pytest.mark.parametrize("triple", [[3, 4, 5], [5, 12, 13], [8, 15, 17], [7, 24, 25]])
    async def test_known_pythagorean_triples(self, triple):
        """Parametrized test for known Pythagorean triples."""
        a, b, c = triple
        
        # Should be found in appropriate range
        triples = await pythagorean_triples(c + 5)
        assert triple in triples
        
        # Verify Pythagorean property
        assert a * a + b * b == c * c
    
    @pytest.mark.asyncio
    @pytest.mark.parametrize("denominations,expected_frobenius", [
        ([3, 5], 7),      # Chicken McNugget problem
        ([4, 7], 17),     # 4*7 - 4 - 7 = 28 - 11 = 17
        ([6, 10], float('inf')),  # gcd(6,10) = 2 > 1
    ])
    async def test_frobenius_numbers(self, denominations, expected_frobenius):
        """Parametrized test for Frobenius numbers."""
        frobenius = await frobenius_number(denominations)
        assert frobenius == expected_frobenius

if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v", "--tb=short", "--asyncio-mode=auto"])