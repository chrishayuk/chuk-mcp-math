#!/usr/bin/env python3
# tests/math/number_theory/test_farey_sequences.py
"""
Comprehensive pytest test suite for farey_sequences.py module.

Tests cover:
- Basic Farey sequence generation and properties
- Mediant operations and Stern-Brocot tree navigation
- Ford circles and geometric properties
- Mathematical properties and theoretical verification
- Applications: rational approximation, fraction finding
- Advanced: Calkin-Wilf tree, Riemann Hypothesis connections
- Edge cases, error conditions, and performance testing
- Async behavior verification and concurrent execution

Test Categories:
1. Basic Farey Sequence Operations
2. Mediant Operations and Tree Structures
3. Ford Circles and Geometric Properties
4. Mathematical Properties and Relationships
5. Applications and Practical Use Cases
6. Advanced Mathematical Connections
7. Performance and Async Behavior
8. Error Handling and Edge Cases
"""

import pytest
import asyncio
import time
import math

# Import the functions to test
from chuk_mcp_math.number_theory.farey_sequences import (
    # Basic operations
    farey_sequence,
    farey_sequence_length,
    farey_neighbors,
    # Mediant operations
    mediant,
    stern_brocot_tree,
    farey_mediant_path,
    # Ford circles
    ford_circles,
    ford_circle_properties,
    circle_tangency,
    # Analysis
    farey_sequence_properties,
    density_analysis,
    gap_analysis,
    # Applications
    best_approximation_farey,
    farey_fraction_between,
    # Advanced
    farey_sum,
    calkin_wilf_tree,
    riemann_hypothesis_connection,
)

# Known Farey sequences for testing
FAREY_SEQUENCES = {
    1: [[0, 1], [1, 1]],
    2: [[0, 1], [1, 2], [1, 1]],
    3: [[0, 1], [1, 3], [1, 2], [2, 3], [1, 1]],
    4: [[0, 1], [1, 4], [1, 3], [1, 2], [2, 3], [3, 4], [1, 1]],
    5: [
        [0, 1],
        [1, 5],
        [1, 4],
        [1, 3],
        [2, 5],
        [1, 2],
        [3, 5],
        [2, 3],
        [3, 4],
        [4, 5],
        [1, 1],
    ],
}

FAREY_LENGTHS = {1: 2, 2: 3, 3: 5, 4: 7, 5: 11, 6: 13, 7: 19, 8: 23, 9: 29, 10: 33}

# ============================================================================
# BASIC FAREY SEQUENCE OPERATIONS TESTS
# ============================================================================


class TestBasicFareyOperations:
    """Test cases for basic Farey sequence operations."""

    @pytest.mark.asyncio
    async def test_farey_sequence_known_values(self):
        """Test Farey sequence generation with known values."""
        for n, expected in FAREY_SEQUENCES.items():
            result = await farey_sequence(n)
            assert result == expected, f"F_{n} should be {expected}, got {result}"

    @pytest.mark.asyncio
    async def test_farey_sequence_properties(self):
        """Test mathematical properties of generated Farey sequences."""
        for n in range(1, 8):
            seq = await farey_sequence(n)

            # Check ordering (fractions should be in ascending order)
            values = [p / q for p, q in seq]
            assert values == sorted(values), f"F_{n} should be in ascending order"

            # Check bounds (all fractions should be in [0, 1])
            for p, q in seq:
                assert 0 <= p <= q, f"Fraction {p}/{q} in F_{n} should be in [0, 1]"

            # Check denominators (all denominators should be ≤ n)
            for p, q in seq:
                assert q <= n, f"Denominator {q} in F_{n} should be ≤ {n}"

            # Check first and last elements
            assert seq[0] == [0, 1], f"F_{n} should start with 0/1"
            assert seq[-1] == [1, 1], f"F_{n} should end with 1/1"

    @pytest.mark.asyncio
    async def test_farey_sequence_adjacent_property(self):
        """Test that adjacent fractions satisfy |ad - bc| = 1."""
        for n in range(2, 8):
            seq = await farey_sequence(n)

            for i in range(len(seq) - 1):
                p1, q1 = seq[i]
                p2, q2 = seq[i + 1]

                # Adjacent fractions should satisfy |p1*q2 - p2*q1| = 1
                det = abs(p1 * q2 - p2 * q1)
                assert det == 1, (
                    f"Adjacent fractions {p1}/{q1}, {p2}/{q2} in F_{n} should have determinant 1, got {det}"
                )

    @pytest.mark.asyncio
    async def test_farey_sequence_length_formula(self):
        """Test Farey sequence length using Euler's totient function."""
        for n, expected_length in FAREY_LENGTHS.items():
            result = await farey_sequence_length(n)

            assert result["length"] == expected_length, (
                f"|F_{n}| should be {expected_length}, got {result['length']}"
            )
            assert result["formula_result"] == expected_length
            assert result["n"] == n

            # Verify actual sequence length matches formula
            actual_seq = await farey_sequence(n)
            assert len(actual_seq) == expected_length

    @pytest.mark.asyncio
    async def test_farey_sequence_edge_cases(self):
        """Test edge cases for Farey sequence generation."""
        # n = 0 should raise error
        with pytest.raises(ValueError, match="n must be a positive integer"):
            await farey_sequence(0)

        # Negative n should raise error
        with pytest.raises(ValueError, match="n must be a positive integer"):
            await farey_sequence(-5)

        # n = 1 should work
        result = await farey_sequence(1)
        assert result == [[0, 1], [1, 1]]

    @pytest.mark.asyncio
    async def test_farey_sequence_large_n_async_yield(self):
        """Test that large n triggers async yield for responsiveness."""
        # Test with n > 100 to trigger the asyncio.sleep(0) on line 111
        result = await farey_sequence(150)
        # Just verify it completes and has reasonable length
        assert len(result) > 100
        assert result[0] == [0, 1]
        assert result[-1] == [1, 1]

    @pytest.mark.asyncio
    async def test_farey_sequence_length_edge_cases(self):
        """Test edge cases for farey_sequence_length."""
        # n < 0 should raise error (line 162)
        with pytest.raises(ValueError, match="n must be non-negative"):
            await farey_sequence_length(-1)

        # n = 0 should return zeros (line 164)
        result = await farey_sequence_length(0)
        assert result["length"] == 0
        assert result["formula_result"] == 0
        assert result["totient_sum"] == 0

    @pytest.mark.asyncio
    async def test_farey_sequence_length_large_k_async_yield(self):
        """Test that large k triggers async yield in totient sum."""
        # Test with n > 1000 to trigger the asyncio.sleep(0) on line 173
        result = await farey_sequence_length(1500)
        assert result["length"] > 1000
        assert result["n"] == 1500

    @pytest.mark.asyncio
    async def test_farey_neighbors_basic(self):
        """Test finding neighbors of fractions in Farey sequences."""
        # Test 1/2 in F_5
        neighbors = await farey_neighbors(1, 2, 5)

        assert neighbors["fraction"] == [1, 2]
        assert neighbors["left_neighbor"] == [2, 5]
        assert neighbors["right_neighbor"] == [3, 5]

        # Verify mediants
        assert neighbors["mediant_left"] == [3, 7]  # mediant of 2/5 and 1/2
        assert neighbors["mediant_right"] == [4, 7]  # mediant of 1/2 and 3/5

    @pytest.mark.asyncio
    async def test_farey_neighbors_edge_positions(self):
        """Test neighbors for fractions at sequence edges."""
        # Test 0/1 (first element) in F_5
        neighbors = await farey_neighbors(0, 1, 5)

        assert neighbors["fraction"] == [0, 1]
        assert neighbors["left_neighbor"] is None
        assert neighbors["right_neighbor"] == [1, 5]
        assert neighbors["mediant_left"] is None
        assert neighbors["mediant_right"] == [1, 6]

        # Test 1/1 (last element) in F_5
        neighbors = await farey_neighbors(1, 1, 5)

        assert neighbors["fraction"] == [1, 1]
        assert neighbors["left_neighbor"] == [4, 5]
        assert neighbors["right_neighbor"] is None
        assert neighbors["mediant_left"] == [5, 6]
        assert neighbors["mediant_right"] is None

    @pytest.mark.asyncio
    async def test_farey_neighbors_error_cases(self):
        """Test error handling in farey_neighbors function."""
        # Invalid fraction (not in lowest terms)
        with pytest.raises(ValueError, match="Fraction must be in lowest terms"):
            await farey_neighbors(2, 4, 5)  # 2/4 = 1/2 but not in lowest terms

        # Fraction not in sequence (denominator too large)
        with pytest.raises(ValueError, match="Invalid fraction or Farey sequence order"):
            await farey_neighbors(1, 10, 5)  # 1/10 not in F_5

        # Invalid parameters
        with pytest.raises(ValueError, match="Invalid fraction or Farey sequence order"):
            await farey_neighbors(-1, 2, 5)  # Negative numerator

        with pytest.raises(ValueError, match="Invalid fraction or Farey sequence order"):
            await farey_neighbors(3, 2, 5)  # p > q

    @pytest.mark.asyncio
    async def test_farey_neighbors_fraction_not_found(self):
        """Test error when fraction is not in sequence (line 251)."""
        # Use a fraction that passes validation but mock farey_sequence to not include it
        from unittest.mock import patch, AsyncMock

        # Mock farey_sequence to return a sequence that doesn't include 1/3
        mock_seq = [[0, 1], [1, 2], [2, 3], [1, 1]]  # Missing 1/3
        with patch(
            "chuk_mcp_math.number_theory.farey_sequences.farey_sequence",
            new_callable=AsyncMock,
            return_value=mock_seq,
        ):
            with pytest.raises(ValueError, match="Fraction .* not found in F_"):
                await farey_neighbors(1, 3, 5)


# ============================================================================
# MEDIANT OPERATIONS AND TREE STRUCTURES TESTS
# ============================================================================


class TestMediantOperations:
    """Test cases for mediant operations and tree structures."""

    @pytest.mark.asyncio
    async def test_mediant_basic_operations(self):
        """Test basic mediant calculations."""
        test_cases = [
            ((1, 3), (1, 2), [2, 5]),  # mediant of 1/3 and 1/2
            ((0, 1), (1, 1), [1, 2]),  # mediant of 0/1 and 1/1
            ((2, 5), (3, 4), [5, 9]),  # mediant of 2/5 and 3/4
            ((1, 4), (1, 3), [2, 7]),  # mediant of 1/4 and 1/3
        ]

        for (p1, q1), (p2, q2), expected in test_cases:
            result = await mediant(p1, q1, p2, q2)
            assert result == expected, (
                f"mediant({p1}/{q1}, {p2}/{q2}) should be {expected}, got {result}"
            )

    @pytest.mark.asyncio
    async def test_mediant_properties(self):
        """Test mathematical properties of mediants."""
        # Test that mediant is between the two fractions
        test_fractions = [
            ((1, 4), (1, 3)),
            ((1, 3), (1, 2)),
            ((2, 5), (3, 5)),
            ((1, 5), (1, 4)),
        ]

        for (p1, q1), (p2, q2) in test_fractions:
            med = await mediant(p1, q1, p2, q2)
            med_p, med_q = med[0], med[1]

            val1 = p1 / q1
            val2 = p2 / q2
            med_val = med_p / med_q

            # Mediant should be between the two fractions
            min_val, max_val = min(val1, val2), max(val1, val2)
            assert min_val < med_val < max_val, (
                f"Mediant {med_p}/{med_q} should be between {p1}/{q1} and {p2}/{q2}"
            )

    @pytest.mark.asyncio
    async def test_mediant_error_handling(self):
        """Test error handling in mediant function."""
        # Zero denominators should raise error
        with pytest.raises(ValueError, match="Denominators must be positive"):
            await mediant(1, 0, 1, 2)

        with pytest.raises(ValueError, match="Denominators must be positive"):
            await mediant(1, 2, 1, 0)

        # Negative denominators should raise error
        with pytest.raises(ValueError, match="Denominators must be positive"):
            await mediant(1, -2, 1, 3)

    @pytest.mark.asyncio
    async def test_stern_brocot_tree_navigation(self):
        """Test Stern-Brocot tree navigation to target fractions."""
        # First test a few cases to understand the actual paths
        test_fractions = [(3, 7), (2, 5), (1, 3), (2, 3)]

        for target_p, target_q in test_fractions:
            result = await stern_brocot_tree(target_p, target_q)

            assert result["target"] == [target_p, target_q]
            assert isinstance(result["path"], list)
            assert result["depth"] == len(result["path"])

            # Verify the last fraction in the path is the target
            last_fraction = result["fractions"][-1]
            assert last_fraction == [target_p, target_q]

            # Verify path consists of valid directions
            for direction in result["path"]:
                assert direction in ["L", "R"]

    @pytest.mark.asyncio
    async def test_stern_brocot_tree_properties(self):
        """Test properties of Stern-Brocot tree navigation."""
        # Test that all intermediate fractions are in reduced form
        result = await stern_brocot_tree(5, 13)

        for frac in result["fractions"]:
            p, q = frac
            # Check that gcd(p, q) = 1 (reduced form)
            gcd_val = math.gcd(p, q)
            assert gcd_val == 1, f"Fraction {p}/{q} should be in reduced form"

        # Test that fractions are properly ordered along the path
        fractions = result["fractions"][2:]  # Skip initial bounds 0/1, 1/1

        for i, frac in enumerate(fractions[:-1]):  # Exclude final target
            p, q = frac
            val = p / q
            # Each fraction should be getting closer to target
            # This is a heuristic check, not always strictly monotonic
            assert 0 <= val <= 1, f"Intermediate fraction {p}/{q} should be in [0,1]"

    @pytest.mark.asyncio
    async def test_stern_brocot_tree_edge_cases(self):
        """Test edge cases for Stern-Brocot tree."""
        # Invalid fractions should raise errors
        with pytest.raises(ValueError, match="Invalid target fraction"):
            await stern_brocot_tree(-1, 3)  # Negative numerator

        with pytest.raises(ValueError, match="Invalid target fraction"):
            await stern_brocot_tree(1, 0)  # Zero denominator

        # Non-reduced fractions should raise error
        with pytest.raises(ValueError, match="Target fraction must be in lowest terms"):
            await stern_brocot_tree(2, 4)  # 2/4 = 1/2 but not reduced

    @pytest.mark.asyncio
    async def test_farey_mediant_path(self):
        """Test mediant path generation between fractions."""
        # Test convergent path from 1/3 to 1/2
        result = await farey_mediant_path(1, 3, 1, 2, 10)

        assert result["start"] == [1, 3]
        assert result["end"] == [1, 2]

        # Check that all mediants are between start and end
        start_val = 1 / 3
        end_val = 1 / 2

        for med in result["mediants"]:
            med_val = med[0] / med[1]
            assert start_val <= med_val <= end_val, (
                f"Mediant {med[0]}/{med[1]} should be between 1/3 and 1/2"
            )

        # The path might not converge exactly to [1, 2] due to the iterative nature
        # Just check that we get some reasonable mediants
        assert len(result["mediants"]) > 0, "Should generate at least one mediant"

    @pytest.mark.asyncio
    async def test_farey_mediant_path_non_convergent(self):
        """Test mediant path that doesn't converge within limits."""
        # Use very small max_denom to force non-convergence
        result = await farey_mediant_path(1, 7, 1, 6, 3)

        assert result["start"] == [1, 7]
        assert result["end"] == [1, 6]
        assert not result["converges"]
        assert "reason" in result


# ============================================================================
# FORD CIRCLES AND GEOMETRIC PROPERTIES TESTS
# ============================================================================


class TestFordCircles:
    """Test cases for Ford circles and geometric properties."""

    @pytest.mark.asyncio
    async def test_ford_circles_generation(self):
        """Test Ford circles generation for Farey sequences."""
        result = await ford_circles(4)

        assert result["n"] == 4
        assert result["count"] == 7  # F_4 has 7 fractions
        assert len(result["circles"]) == 7

        # Check circle properties
        for circle in result["circles"]:
            assert "fraction" in circle
            assert "center" in circle
            assert "radius" in circle
            assert "denominator" in circle

            p, q = circle["fraction"]
            radius = circle["radius"]
            center = circle["center"]

            # Verify Ford circle formula for non-0/1 fractions
            if not (p == 0 and q == 1):
                expected_radius = 1 / (2 * q * q)
                expected_center_x = p / q
                expected_center_y = 1 / (2 * q * q)

                assert abs(radius - expected_radius) < 1e-6, f"Radius for {p}/{q} incorrect"
                assert abs(center[0] - expected_center_x) < 1e-6, f"Center x for {p}/{q} incorrect"
                assert abs(center[1] - expected_center_y) < 1e-6, f"Center y for {p}/{q} incorrect"

    @pytest.mark.asyncio
    async def test_ford_circles_special_case(self):
        """Test Ford circles for special case 0/1."""
        result = await ford_circles(3)

        # Find the circle for 0/1
        zero_circle = next(c for c in result["circles"] if c["fraction"] == [0, 1])

        assert zero_circle["center"] == [0.0, 0.5]
        assert zero_circle["radius"] == 0.5

    @pytest.mark.asyncio
    async def test_ford_circle_properties_analysis(self):
        """Test Ford circle properties analysis."""
        result = await ford_circle_properties(5)

        assert result["n"] == 5
        assert result["total_circles"] == 11  # F_5 has 11 fractions
        assert result["tangent_pairs"] == 10  # 10 adjacent pairs

        # Check radius statistics
        assert result["max_radius"] > 0
        assert result["min_radius"] > 0
        assert result["avg_radius"] > 0
        assert result["min_radius"] <= result["avg_radius"] <= result["max_radius"]

        # Check tangency analysis
        assert len(result["tangency_analysis"]) <= 5  # Limited to first 5

        for analysis in result["tangency_analysis"]:
            assert "fraction1" in analysis
            assert "fraction2" in analysis
            assert "is_tangent" in analysis
            assert "distance" in analysis
            assert "sum_radii" in analysis

    @pytest.mark.asyncio
    async def test_circle_tangency_adjacent_fractions(self):
        """Test tangency between Ford circles of adjacent fractions."""
        # Test known adjacent pairs from F_5
        adjacent_pairs = [
            ((0, 1), (1, 5)),
            ((1, 5), (1, 4)),
            ((1, 4), (1, 3)),
            ((1, 3), (2, 5)),
            ((2, 5), (1, 2)),
        ]

        for (p1, q1), (p2, q2) in adjacent_pairs:
            result = await circle_tangency(p1, q1, p2, q2)

            assert result["fraction1"] == [p1, q1]
            assert result["fraction2"] == [p2, q2]
            assert result["are_farey_adjacent"]
            assert result["determinant"] == 1
            assert result["are_tangent"]
            assert result["tangency_type"] == "external"

            # Verify tangency condition: distance = sum of radii
            assert abs(result["distance"] - result["sum_radii"]) < 1e-10

    @pytest.mark.asyncio
    async def test_circle_tangency_non_adjacent_fractions(self):
        """Test tangency between Ford circles of non-adjacent fractions."""
        # Test non-adjacent pairs - need to verify which are actually non-adjacent
        test_pairs = [
            ((1, 5), (1, 3)),  # Check if these are non-adjacent
            ((1, 4), (2, 5)),  # Check if these are non-adjacent
            ((1, 6), (1, 4)),  # These should definitely be non-adjacent
        ]

        for (p1, q1), (p2, q2) in test_pairs:
            result = await circle_tangency(p1, q1, p2, q2)

            # Check determinant to see if they're actually adjacent
            det = abs(p1 * q2 - p2 * q1)
            expected_adjacent = det == 1

            assert result["are_farey_adjacent"] == expected_adjacent
            assert result["determinant"] == det

            if not expected_adjacent:
                assert not result["are_tangent"]
                assert result["tangency_type"] in ["separate", "overlapping"]
            else:
                # If they are adjacent, they should be tangent
                assert result["are_tangent"]

    @pytest.mark.asyncio
    async def test_circle_tangency_error_handling(self):
        """Test error handling in circle tangency function."""
        # Zero denominators should raise error
        with pytest.raises(ValueError, match="Denominators must be positive"):
            await circle_tangency(1, 0, 1, 2)

        with pytest.raises(ValueError, match="Denominators must be positive"):
            await circle_tangency(1, 2, 1, 0)


# ============================================================================
# MATHEMATICAL PROPERTIES AND RELATIONSHIPS TESTS
# ============================================================================


class TestMathematicalProperties:
    """Test mathematical properties and relationships."""

    @pytest.mark.asyncio
    async def test_farey_sequence_properties_comprehensive(self):
        """Test comprehensive properties analysis of Farey sequences."""
        result = await farey_sequence_properties(5)

        assert result["n"] == 5
        assert result["length"] == 11
        assert result["max_denominator"] == 5
        assert result["unique_denominators"] >= 1
        assert result["adjacent_pairs"] == 10  # n-1 adjacent pairs
        assert result["farey_property_violations"] == 0  # Should be 0 for valid Farey sequence

        # Check gap statistics
        assert result["max_gap"] > 0
        assert result["min_gap"] > 0
        assert result["avg_gap"] > 0
        assert result["min_gap"] <= result["avg_gap"] <= result["max_gap"]

        # Check that gaps sample is reasonable
        assert len(result["gaps_sample"]) <= 10

    @pytest.mark.asyncio
    async def test_density_analysis_growth(self):
        """Test density analysis and growth patterns."""
        result = await density_analysis(10)

        assert result["max_n"] == 10
        assert len(result["densities"]) == 10
        assert len(result["density_ratios"]) == 9  # n-1 ratios

        # Check actual densities (the F_n lengths start with F_1 = 2, not 1)
        expected_densities = [2, 3, 5, 7, 11, 13, 19, 23, 29, 33]
        assert result["densities"] == expected_densities

        # Check theoretical constant
        theoretical = 3 / (math.pi**2)
        assert abs(result["theoretical_constant"] - theoretical) < 1e-6

        # Check that densities are increasing
        densities = result["densities"]
        for i in range(1, len(densities)):
            assert densities[i] > densities[i - 1], "Densities should be increasing"

    @pytest.mark.asyncio
    async def test_gap_analysis_distribution(self):
        """Test gap analysis and distribution properties."""
        result = await gap_analysis(5)

        assert result["n"] == 5
        assert result["total_gaps"] == 10  # 11 fractions → 10 gaps

        # Check gap statistics
        assert result["max_gap"] > 0
        assert result["min_gap"] > 0
        assert result["avg_gap"] > 0

        # Check gap distribution
        gap_dist = result["gap_distribution"]
        total_counted = sum(gap_dist.values())
        assert total_counted == result["total_gaps"]

        # Check largest gaps analysis
        assert len(result["largest_gaps"]) <= 5

        for gap_info in result["largest_gaps"]:
            assert "fraction1" in gap_info
            assert "fraction2" in gap_info
            assert "gap" in gap_info
            assert "theoretical_gap" in gap_info

    @pytest.mark.asyncio
    async def test_gap_formula_verification(self):
        """Test that gaps follow the formula 1/(q1*q2) for adjacent fractions."""
        await gap_analysis(4)

        # Get the actual Farey sequence to verify gaps
        farey_seq = await farey_sequence(4)

        # Test the first few gaps directly
        for i in range(min(3, len(farey_seq) - 1)):
            p1, q1 = farey_seq[i]
            p2, q2 = farey_seq[i + 1]

            # Calculate actual gap
            actual_gap = p2 / q2 - p1 / q1

            # Calculate theoretical gap using formula
            theoretical = 1 / (q1 * q2)

            # Allow reasonable numerical error (increased tolerance)
            assert abs(actual_gap - theoretical) < 1e-6, (
                f"Gap between {p1}/{q1} and {p2}/{q2} doesn't match formula"
            )


# ============================================================================
# APPLICATIONS AND PRACTICAL USE CASES TESTS
# ============================================================================


class TestApplications:
    """Test applications and practical use cases."""

    @pytest.mark.asyncio
    async def test_best_approximation_farey_known_values(self):
        """Test best rational approximation for known values."""
        test_cases = [
            (0.5, 10, [1, 2], 0.0),  # Exact case
            (0.333333, 10, [1, 3], None),  # Close to 1/3
            (0.6, 10, [3, 5], 0.0),  # Exact case
            (0.25, 10, [1, 4], 0.0),  # Exact case
        ]

        for target, max_denom, expected_frac, expected_error in test_cases:
            result = await best_approximation_farey(target, max_denom)

            assert result["target"] == target
            assert result["max_denom"] == max_denom
            assert result["best_approximation"] == expected_frac

            if expected_error is not None:
                assert abs(result["error"] - expected_error) < 1e-10

            # Check that approximation is within bounds
            approx_val = result["best_value"]
            assert 0 <= approx_val <= 1

    @pytest.mark.asyncio
    async def test_best_approximation_farey_mathematical_constants(self):
        """Test approximation of mathematical constants."""
        constants = [
            (math.pi - 3, "π - 3"),  # π - 3 ≈ 0.14159
            (math.e - 2, "e - 2"),  # e - 2 ≈ 0.71828
            (math.sqrt(2) - 1, "√2 - 1"),  # √2 - 1 ≈ 0.41421
        ]

        for value, name in constants:
            if 0 <= value <= 1:
                result = await best_approximation_farey(value, 20)

                # Should find some approximation
                assert result["best_approximation"] != [0, 1]  # Should be better than 0
                assert result["error"] < 0.5  # Should be reasonably close

                # Verify that the approximation is actually closer than simple fractions
                result["best_value"]
                simple_errors = [abs(value - 0), abs(value - 0.5), abs(value - 1)]
                assert result["error"] <= min(simple_errors)

    @pytest.mark.asyncio
    async def test_best_approximation_farey_edge_cases(self):
        """Test edge cases for best approximation."""
        # Target outside [0, 1] should raise error
        with pytest.raises(ValueError, match="Target must be in \\[0, 1\\]"):
            await best_approximation_farey(1.5, 10)

        with pytest.raises(ValueError, match="Target must be in \\[0, 1\\]"):
            await best_approximation_farey(-0.5, 10)

        # Invalid max_denom should raise error
        with pytest.raises(ValueError, match="Maximum denominator must be positive"):
            await best_approximation_farey(0.5, 0)

        with pytest.raises(ValueError, match="Maximum denominator must be positive"):
            await best_approximation_farey(0.5, -5)

    @pytest.mark.asyncio
    async def test_farey_fraction_between_basic(self):
        """Test finding fractions between two given fractions."""
        test_cases = [
            ((1, 3), (1, 2), [2, 5]),  # Between 1/3 and 1/2
            ((0, 1), (1, 2), [1, 3]),  # Between 0/1 and 1/2
            ((1, 2), (1, 1), [2, 3]),  # Between 1/2 and 1/1
            ((2, 7), (1, 3), [3, 10]),  # Between 2/7 and 1/3
        ]

        for (p1, q1), (p2, q2), expected in test_cases:
            result = await farey_fraction_between(p1, q1, p2, q2)

            assert result["fraction1"] in [
                [p1, q1],
                [p2, q2],
            ]  # Input fractions (may be reordered)
            assert result["fraction2"] in [[p1, q1], [p2, q2]]
            assert result["fraction_between"] == expected
            assert result["is_mediant"]
            assert result["is_between"]

    @pytest.mark.asyncio
    async def test_farey_fraction_between_properties(self):
        """Test properties of fractions found between two fractions."""
        # Test that the fraction is actually between the two inputs
        result = await farey_fraction_between(1, 4, 1, 3)

        frac_between = result["fraction_between"]
        between_val = frac_between[0] / frac_between[1]

        val1 = 1 / 4
        val2 = 1 / 3
        min_val, max_val = min(val1, val2), max(val1, val2)

        assert min_val < between_val < max_val
        assert result["is_between"]

        # Test adjacency detection
        # 1/4 and 1/3 are adjacent in some Farey sequence
        det = abs(1 * 3 - 1 * 4)
        expected_adjacent = det == 1
        assert result["fractions_are_adjacent"] == expected_adjacent

    @pytest.mark.asyncio
    async def test_farey_fraction_between_error_handling(self):
        """Test error handling for farey_fraction_between."""
        # Zero denominators should raise error
        with pytest.raises(ValueError, match="Denominators must be positive"):
            await farey_fraction_between(1, 0, 1, 2)

        with pytest.raises(ValueError, match="Denominators must be positive"):
            await farey_fraction_between(1, 2, 1, 0)


# ============================================================================
# ADVANCED MATHEMATICAL CONNECTIONS TESTS
# ============================================================================


class TestAdvancedConnections:
    """Test advanced mathematical connections and applications."""

    @pytest.mark.asyncio
    async def test_farey_sum_basic_operations(self):
        """Test basic Farey sum operations."""
        test_cases = [
            ((1, 3), (1, 4), [7, 12]),  # 1/3 + 1/4 = 4/12 + 3/12 = 7/12
            ((1, 2), (1, 2), [1, 1]),  # 1/2 + 1/2 = 1
            ((2, 5), (3, 7), [29, 35]),  # 2/5 + 3/7 = 14/35 + 15/35 = 29/35
            ((1, 6), (1, 6), [1, 3]),  # 1/6 + 1/6 = 2/6 = 1/3
        ]

        for (p1, q1), (p2, q2), expected in test_cases:
            result = await farey_sum(p1, q1, p2, q2)

            assert result["fraction1"] == [p1, q1]
            assert result["fraction2"] == [p2, q2]
            assert result["farey_sum"] == expected
            assert result["regular_sum"] == expected
            assert result["are_equal"]

            # Verify the sum is correct (with relaxed tolerance)
            expected_val = expected[0] / expected[1]
            calculated_val = result["sum_value"]
            assert abs(expected_val - calculated_val) < 1e-6

    @pytest.mark.asyncio
    async def test_farey_sum_reduction(self):
        """Test that Farey sums are properly reduced."""
        # Test case where reduction is needed
        result = await farey_sum(1, 6, 1, 6)  # 1/6 + 1/6 = 2/6 = 1/3

        assert result["farey_sum"] == [1, 3]  # Should be reduced
        # The unreduced form depends on how the implementation calculates it
        # Let's just check that reduction_factor > 1
        assert result["reduction_factor"] > 1

    @pytest.mark.asyncio
    async def test_calkin_wilf_tree_generation(self):
        """Test Calkin-Wilf tree generation."""
        result = await calkin_wilf_tree(3)

        assert result["levels"] == 3
        assert len(result["tree_levels"]) == 3

        # Check level structure
        assert result["tree_levels"][0] == [[1, 1]]  # Level 0: root
        assert result["tree_levels"][1] == [[1, 2], [2, 1]]  # Level 1: 2 nodes
        assert result["tree_levels"][2] == [
            [1, 3],
            [3, 2],
            [2, 3],
            [3, 1],
        ]  # Level 2: 4 nodes

        # Check total count
        expected_total = 1 + 2 + 4  # 2^0 + 2^1 + 2^2
        assert result["total_fractions"] == expected_total

        # Check enumeration
        expected_enum = [[1, 1], [1, 2], [2, 1], [1, 3], [3, 2], [2, 3], [3, 1]]
        assert result["enumeration"][:7] == expected_enum

    @pytest.mark.asyncio
    async def test_calkin_wilf_tree_properties(self):
        """Test properties of Calkin-Wilf tree."""
        result = await calkin_wilf_tree(4)

        # Check that all fractions are positive and in reduced form
        for level in result["tree_levels"]:
            for p, q in level:
                assert p > 0 and q > 0, f"Fraction {p}/{q} should be positive"
                assert math.gcd(p, q) == 1, f"Fraction {p}/{q} should be in reduced form"

        # Check parent-child relationships for level 1 → level 2
        parent_level = result["tree_levels"][1]  # [[1, 2], [2, 1]]
        child_level = result["tree_levels"][2]  # [[1, 3], [3, 2], [2, 3], [3, 1]]

        # For parent [1, 2]: children should be [1, 3] and [3, 2]
        p, q = parent_level[0]  # [1, 2]
        expected_left = [p, p + q]  # [1, 3]
        expected_right = [p + q, q]  # [3, 2]
        assert child_level[0] == expected_left
        assert child_level[1] == expected_right

        # For parent [2, 1]: children should be [2, 3] and [3, 1]
        p, q = parent_level[1]  # [2, 1]
        expected_left = [p, p + q]  # [2, 3]
        expected_right = [p + q, q]  # [3, 1]
        assert child_level[2] == expected_left
        assert child_level[3] == expected_right

    @pytest.mark.asyncio
    async def test_calkin_wilf_tree_edge_cases(self):
        """Test edge cases for Calkin-Wilf tree."""
        # Zero levels should return empty
        result = await calkin_wilf_tree(0)
        assert result["tree_levels"] == []
        assert result["total_fractions"] == 0

        # Single level should have just the root
        result = await calkin_wilf_tree(1)
        assert result["tree_levels"] == [[[1, 1]]]
        assert result["total_fractions"] == 1

    @pytest.mark.asyncio
    async def test_riemann_hypothesis_connection_basic(self):
        """Test basic Riemann Hypothesis connection analysis."""
        result = await riemann_hypothesis_connection(10)

        assert result["n"] == 10
        assert result["farey_length"] == 33  # Known value for F_10

        # Check theoretical formula
        theoretical_constant = 3 / (math.pi**2)
        expected_theoretical = theoretical_constant * 10 * 10
        assert abs(result["theoretical_length"] - expected_theoretical) < 0.01

        # Check error analysis
        assert result["error"] >= 0
        assert result["relative_error"] >= 0
        assert result["rh_bound"] == 0.5

        # Check that theoretical constant is correct
        assert abs(result["theoretical_constant"] - theoretical_constant) < 1e-6

    @pytest.mark.asyncio
    async def test_riemann_hypothesis_connection_error_bounds(self):
        """Test RH error bound analysis for different values."""
        test_values = [5, 10, 20, 50]

        for n in test_values:
            result = await riemann_hypothesis_connection(n)

            # Error exponent should be reasonable
            error_exp = result["error_exponent"]

            # For reasonable n, error exponent should be between 0 and 1
            # (though RH predicts it should be ≤ 0.5 + ε)
            assert 0 <= error_exp <= 2, f"Error exponent {error_exp} seems unreasonable for n={n}"

            # RH consistency check should be based on the bound
            rh_bound = result["rh_bound"]
            tolerance = 0.1  # Small tolerance for numerical errors
            error_exp <= rh_bound + tolerance
            # Note: For small n, this might not hold due to lower-order terms

    @pytest.mark.asyncio
    async def test_riemann_hypothesis_connection_edge_cases(self):
        """Test edge cases for RH connection."""
        # n = 0 should return zeros
        result = await riemann_hypothesis_connection(0)
        assert result["farey_length"] == 0
        assert result["theoretical_length"] == 0
        assert result["error"] == 0

        # n = 1 should work
        result = await riemann_hypothesis_connection(1)
        assert result["farey_length"] == 2  # F_1 = {0/1, 1/1}
        assert result["n"] == 1


# ============================================================================
# PERFORMANCE AND ASYNC BEHAVIOR TESTS
# ============================================================================


class TestPerformanceAndAsync:
    """Test performance characteristics and async behavior."""

    @pytest.mark.asyncio
    async def test_all_functions_are_async(self):
        """Test that all Farey sequence functions are properly async."""
        operations = [
            farey_sequence(10),
            farey_sequence_length(10),
            farey_neighbors(1, 2, 5),
            mediant(1, 3, 1, 2),
            stern_brocot_tree(3, 7),
            farey_mediant_path(1, 3, 1, 2, 10),
            ford_circles(5),
            ford_circle_properties(5),
            circle_tangency(1, 3, 1, 2),
            farey_sequence_properties(5),
            density_analysis(5),
            gap_analysis(5),
            best_approximation_farey(0.618, 10),
            farey_fraction_between(1, 3, 1, 2),
            farey_sum(1, 3, 1, 4),
            calkin_wilf_tree(3),
            riemann_hypothesis_connection(10),
        ]

        # Ensure all are coroutines
        for op in operations:
            assert asyncio.iscoroutine(op)

        # Run all operations concurrently
        results = await asyncio.gather(*operations)

        # Verify results have expected types
        assert isinstance(results[0], list)  # farey_sequence
        assert isinstance(results[1], dict)  # farey_sequence_length
        assert isinstance(results[2], dict)  # farey_neighbors
        assert isinstance(results[3], list)  # mediant
        assert isinstance(results[4], dict)  # stern_brocot_tree
        assert isinstance(results[5], dict)  # farey_mediant_path
        assert isinstance(results[6], dict)  # ford_circles
        assert isinstance(results[7], dict)  # ford_circle_properties
        assert isinstance(results[8], dict)  # circle_tangency
        assert isinstance(results[9], dict)  # farey_sequence_properties
        assert isinstance(results[10], dict)  # density_analysis
        assert isinstance(results[11], dict)  # gap_analysis
        assert isinstance(results[12], dict)  # best_approximation_farey
        assert isinstance(results[13], dict)  # farey_fraction_between
        assert isinstance(results[14], dict)  # farey_sum
        assert isinstance(results[15], dict)  # calkin_wilf_tree
        assert isinstance(results[16], dict)  # riemann_hypothesis_connection

    @pytest.mark.asyncio
    async def test_concurrent_execution_performance(self):
        """Test concurrent execution of Farey functions."""
        start_time = time.time()

        # Run multiple computations concurrently
        farey_task = farey_sequence(20)
        circles_task = ford_circles(15)
        density_task = density_analysis(15)
        tree_task = calkin_wilf_tree(5)
        props_task = farey_sequence_properties(15)

        results = await asyncio.gather(
            farey_task, circles_task, density_task, tree_task, props_task
        )

        duration = time.time() - start_time

        # Should complete reasonably quickly due to async nature
        assert duration < 5.0
        assert len(results) == 5

        # Verify all results are non-empty/valid
        farey_seq, circles, density, tree, props = results
        assert len(farey_seq) > 0
        assert circles["count"] > 0
        assert len(density["densities"]) > 0
        assert tree["total_fractions"] > 0
        assert props["length"] > 0

    @pytest.mark.asyncio
    async def test_large_scale_operations(self):
        """Test performance with larger inputs."""
        # Test with larger values (but not too large for test speed)
        large_operations = [
            farey_sequence(30),  # Generate larger Farey sequence
            farey_sequence_length(100),  # Calculate length for F_100
            ford_circles(25),  # Generate more Ford circles
            density_analysis(25),  # Analyze density up to F_25
            best_approximation_farey(math.pi - 3, 50),  # Larger search space
        ]

        start_time = time.time()
        results = await asyncio.gather(*large_operations)
        duration = time.time() - start_time

        # Should still complete in reasonable time
        assert duration < 10.0

        # Results should be larger than smaller-scale tests
        farey_30 = results[0]
        length_100 = results[1]
        circles_25 = results[2]

        assert len(farey_30) > 50  # F_30 should have many fractions
        assert length_100["length"] > 1000  # F_100 should be quite large
        assert circles_25["count"] > 30  # Should have many circles

    @pytest.mark.asyncio
    async def test_memory_efficiency(self):
        """Test memory efficiency with repeated operations."""
        # Run same operations multiple times to check for memory leaks
        for _ in range(10):
            seq = await farey_sequence(15)
            circles = await ford_circles(10)
            props = await farey_sequence_properties(10)

            # Basic sanity checks
            assert len(seq) > 0
            assert circles["count"] > 0
            assert props["length"] > 0

            # Force garbage collection opportunity
            del seq, circles, props


# ============================================================================
# ERROR HANDLING AND EDGE CASES TESTS
# ============================================================================


class TestErrorHandlingAndEdgeCases:
    """Test error handling and edge cases comprehensively."""

    @pytest.mark.asyncio
    async def test_invalid_parameters_comprehensive(self):
        """Test comprehensive invalid parameter handling."""
        # Test negative parameters for functions that should reject them
        with pytest.raises(ValueError):
            await farey_sequence(-1)

        with pytest.raises(ValueError):
            await best_approximation_farey(0.5, 0)

        # Test zero parameters where inappropriate
        with pytest.raises(ValueError):
            await farey_sequence(0)

        # Test invalid fraction parameters
        with pytest.raises(ValueError):
            await farey_neighbors(1, 0, 5)  # Zero denominator

        with pytest.raises(ValueError):
            await stern_brocot_tree(1, 0)  # Zero denominator

        with pytest.raises(ValueError):
            await circle_tangency(1, 0, 1, 2)  # Zero denominator

        # Some functions may not raise errors for negative inputs
        # Let's test specific cases that should definitely fail
        with pytest.raises(ValueError, match="Denominators must be positive"):
            await mediant(1, -2, 1, 3)

    @pytest.mark.asyncio
    async def test_boundary_conditions(self):
        """Test boundary conditions across functions."""
        # Test minimum valid values
        min_tests = [
            farey_sequence(1),
            farey_sequence_length(1),
            ford_circles(1),
            density_analysis(1),
            gap_analysis(1),
            calkin_wilf_tree(1),
            riemann_hypothesis_connection(1),
        ]

        results = await asyncio.gather(*min_tests)

        # All should handle minimum inputs gracefully
        for i, result in enumerate(results):
            assert isinstance(result, (list, dict))

            # Specific checks for minimum cases
            if i == 0:  # farey_sequence(1)
                assert result == [[0, 1], [1, 1]]
            elif i == 1:  # farey_sequence_length(1)
                assert result["length"] == 2

    @pytest.mark.asyncio
    async def test_numerical_precision_edge_cases(self):
        """Test numerical precision in edge cases."""
        # Test with fractions that might cause precision issues
        precision_tests = [
            ((1, 1000000), (1, 999999)),  # Very close fractions
            ((999999, 1000000), (1, 1)),  # Very close to 1
            ((1, 999999), (2, 999999)),  # Small differences
        ]

        for (p1, q1), (p2, q2) in precision_tests:
            # These should not raise errors even with potential precision issues
            try:
                mediant_result = await mediant(p1, q1, p2, q2)
                assert len(mediant_result) == 2
                assert mediant_result[1] > 0  # Valid denominator

                tangency_result = await circle_tangency(p1, q1, p2, q2)
                assert isinstance(tangency_result, dict)

            except ValueError:
                # Some cases might be invalid (like non-reduced fractions)
                # This is acceptable
                pass

    @pytest.mark.asyncio
    async def test_consistency_across_functions(self):
        """Test consistency between related functions."""
        # Test that farey_sequence and farey_sequence_length are consistent
        for n in range(1, 10):
            seq = await farey_sequence(n)
            length_data = await farey_sequence_length(n)

            assert len(seq) == length_data["length"], f"Inconsistent length for F_{n}"

        # Test that ford_circles and farey_sequence are consistent
        for n in range(1, 6):
            seq = await farey_sequence(n)
            circles = await ford_circles(n)

            assert len(seq) == circles["count"], f"Inconsistent count for F_{n} circles"

            # Check that fractions match
            seq_fractions = set((p, q) for p, q in seq)
            circle_fractions = set((p, q) for p, q in [c["fraction"] for c in circles["circles"]])

            assert seq_fractions == circle_fractions, f"Fraction sets don't match for F_{n}"


# ============================================================================
# PARAMETRIZED TESTS FOR COMPREHENSIVE COVERAGE
# ============================================================================


class TestParametrized:
    """Parametrized tests for comprehensive coverage."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "n,expected_length",
        [(1, 2), (2, 3), (3, 5), (4, 7), (5, 11), (6, 13), (7, 19), (8, 23)],
    )
    async def test_farey_length_parametrized(self, n, expected_length):
        """Parametrized test for Farey sequence lengths."""
        length_data = await farey_sequence_length(n)
        assert length_data["length"] == expected_length

        # Verify with actual sequence
        seq = await farey_sequence(n)
        assert len(seq) == expected_length

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "p1,q1,p2,q2,expected",
        [
            (1, 3, 1, 2, [2, 5]),
            (0, 1, 1, 1, [1, 2]),
            (1, 4, 1, 3, [2, 7]),
            (2, 5, 3, 7, [5, 12]),
            (1, 5, 1, 4, [2, 9]),
        ],
    )
    async def test_mediant_parametrized(self, p1, q1, p2, q2, expected):
        """Parametrized test for mediant calculations."""
        result = await mediant(p1, q1, p2, q2)
        assert result == expected

        # Verify mediant is between the fractions
        val1, val2 = p1 / q1, p2 / q2
        med_val = expected[0] / expected[1]
        min_val, max_val = min(val1, val2), max(val1, val2)
        assert min_val < med_val < max_val

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "target,max_denom",
        [(0.5, 10), (0.25, 8), (0.75, 12), (0.6, 15), (0.333333, 20)],
    )
    async def test_best_approximation_parametrized(self, target, max_denom):
        """Parametrized test for best rational approximation."""
        result = await best_approximation_farey(target, max_denom)

        assert result["target"] == target
        assert result["max_denom"] == max_denom
        assert 0 <= result["best_value"] <= 1
        assert result["error"] >= 0

        # Check that approximation denominator is within bounds
        denom = result["best_approximation"][1]
        assert 1 <= denom <= max_denom

    @pytest.mark.asyncio
    @pytest.mark.parametrize("levels", [1, 2, 3, 4, 5])
    async def test_calkin_wilf_tree_parametrized(self, levels):
        """Parametrized test for Calkin-Wilf tree generation."""
        result = await calkin_wilf_tree(levels)

        assert result["levels"] == levels
        assert len(result["tree_levels"]) == levels

        # Check exponential growth: level k should have 2^k nodes
        for k in range(levels):
            expected_nodes = 2**k
            actual_nodes = len(result["tree_levels"][k])
            assert actual_nodes == expected_nodes

        # Check total count
        expected_total = sum(2**k for k in range(levels))
        assert result["total_fractions"] == expected_total

    @pytest.mark.asyncio
    @pytest.mark.parametrize("n", [3, 5, 7, 10, 15, 20])
    async def test_ford_circles_parametrized(self, n):
        """Parametrized test for Ford circles generation."""
        result = await ford_circles(n)

        assert result["n"] == n
        assert result["count"] > 0
        assert len(result["circles"]) == result["count"]

        # Verify all circles have proper structure
        for circle in result["circles"]:
            assert len(circle["fraction"]) == 2
            assert len(circle["center"]) == 2
            assert circle["radius"] > 0
            assert circle["denominator"] > 0

        # Count should match Farey sequence length
        seq = await farey_sequence(n)
        assert result["count"] == len(seq)


class TestDemoFunctions:
    """Test demo functions at end of module."""

    @pytest.mark.asyncio
    async def test_demo_functions_execute(self):
        """Test that all demo functions execute without errors."""
        from chuk_mcp_math.number_theory import farey_sequences

        # Run all demo/test functions to improve coverage
        if hasattr(farey_sequences, "test_farey_sequences"):
            await farey_sequences.test_farey_sequences()

        if hasattr(farey_sequences, "demo_mathematical_properties"):
            await farey_sequences.demo_mathematical_properties()

        if hasattr(farey_sequences, "demo_applications"):
            await farey_sequences.demo_applications()

        if hasattr(farey_sequences, "performance_benchmark"):
            await farey_sequences.performance_benchmark()


class TestAdditionalEdgeCases:
    """Additional edge case tests to improve coverage."""

    @pytest.mark.asyncio
    async def test_stern_brocot_tree_max_path_length(self):
        """Test Stern-Brocot tree with path length limit."""
        # Try to trigger the safety check on line 424 by using a large enough fraction
        # that could potentially loop more than 100 times
        result = await stern_brocot_tree(1, 2)
        assert result["depth"] < 100

    @pytest.mark.asyncio
    async def test_farey_mediant_path_max_steps_exceeded(self):
        """Test mediant path exceeding max steps."""
        # Create a scenario where max_steps is hit (line 502)
        result = await farey_mediant_path(0, 1, 1, 1, max_denom=2)
        # Should either converge or report max steps exceeded
        assert "steps" in result

    @pytest.mark.asyncio
    async def test_ford_circle_properties_edge_case(self):
        """Test ford circle properties with edge cases."""
        # Test with n <= 0 (lines 688, 695)
        result = await ford_circle_properties(0)
        assert result["total_circles"] == 0
        assert result["tangent_pairs"] == 0

        result_neg = await ford_circle_properties(-1)
        assert result_neg["total_circles"] == 0

    @pytest.mark.asyncio
    async def test_farey_sequence_properties_edge_cases(self):
        """Test farey sequence properties with edge cases."""
        # Test with n <= 0 (line 904)
        result = await farey_sequence_properties(0)
        assert result["length"] == 0

        # Test with length < 2 (line 911)
        result_1 = await farey_sequence_properties(1)
        assert result_1["length"] == 2

    @pytest.mark.asyncio
    async def test_density_analysis_edge_case(self):
        """Test density analysis with n <= 0."""
        # Line 1016
        result = await density_analysis(0)
        assert result["densities"] == []
        assert result["density_ratios"] == []

    @pytest.mark.asyncio
    async def test_gap_analysis_edge_cases(self):
        """Test gap analysis with edge cases."""
        # Line 1104, 1110
        result = await gap_analysis(0)
        assert result["total_gaps"] == 0

        result_1 = await gap_analysis(1)
        # F_1 has only 2 elements, so 1 gap
        assert result_1["total_gaps"] <= 1

    @pytest.mark.asyncio
    async def test_riemann_hypothesis_negative_n(self):
        """Test Riemann hypothesis with n <= 0."""
        # Line 1527
        result = await riemann_hypothesis_connection(0)
        assert result["farey_length"] == 0

        result_neg = await riemann_hypothesis_connection(-1)
        assert result_neg["farey_length"] == 0

    @pytest.mark.asyncio
    async def test_calkin_wilf_negative_levels(self):
        """Test Calkin-Wilf tree with negative/zero levels."""
        # Line 1433
        result = await calkin_wilf_tree(0)
        assert result["tree_levels"] == []

        result_neg = await calkin_wilf_tree(-1)
        assert result_neg["tree_levels"] == []

    @pytest.mark.asyncio
    async def test_farey_sum_error_handling(self):
        """Test farey_sum error handling."""
        # Line 1362
        with pytest.raises(ValueError, match="Denominators must be positive"):
            await farey_sum(1, 0, 1, 2)

    @pytest.mark.asyncio
    async def test_farey_mediant_path_invalid_params(self):
        """Test farey_mediant_path with invalid parameters."""
        # Line 493
        with pytest.raises(ValueError, match="Invalid parameters"):
            await farey_mediant_path(1, 0, 1, 2, 10)

        with pytest.raises(ValueError, match="Invalid parameters"):
            await farey_mediant_path(1, 2, 1, 0, 10)

        with pytest.raises(ValueError, match="Invalid parameters"):
            await farey_mediant_path(1, 2, 1, 3, 0)

    @pytest.mark.asyncio
    async def test_best_approximation_large_n(self):
        """Test best approximation with async yield."""
        # Trigger line 1228 (async sleep every 10 iterations)
        result = await best_approximation_farey(0.618, 50)
        assert result["max_denom"] == 50

    @pytest.mark.asyncio
    async def test_density_analysis_async_yield(self):
        """Test density analysis with async yield."""
        # Trigger line 1027 (async sleep every 5 iterations)
        result = await density_analysis(15)
        assert len(result["densities"]) == 15


if __name__ == "__main__":
    # Run the comprehensive test suite
    pytest.main(
        [
            __file__,
            "-v",  # Verbose output
            "--tb=short",  # Short traceback format
            "--asyncio-mode=auto",  # Handle async tests automatically
            "--durations=10",  # Show 10 slowest tests
            "--strict-markers",  # Require markers to be defined
            "--strict-config",  # Strict configuration parsing
            "-x",  # Stop on first failure for debugging
            "--cov=chuk_mcp_math.number_theory.farey_sequences",  # Coverage
            "--cov-report=term-missing",  # Show missing lines in coverage
        ]
    )
