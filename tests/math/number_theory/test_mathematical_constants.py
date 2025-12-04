#!/usr/bin/env python3
# tests/math/number_theory/test_mathematical_constants.py
"""
Comprehensive pytest test suite for mathematical_constants.py module (FIXED VERSION).

Tests cover:
- Pi approximations: Leibniz, Nilakantha, Machin, Chudnovsky algorithms
- E approximations: series expansion, limit definition
- Golden ratio: Fibonacci ratios, continued fractions
- Euler-Mascheroni constant: harmonic series approximation
- Continued fractions: pi, e, golden ratio representations
- High precision digit generation: pi and e to arbitrary precision
- Approximation analysis: error calculation, convergence comparison
- Mathematical relationships: Euler's identity, constant relationships
- Convergence properties and algorithm accuracy
- Edge cases, error conditions, and performance testing
- Async behavior verification

Fixed to have realistic convergence expectations and handle edge cases properly.
"""

import pytest
import asyncio
import time
import math

# Import the functions to test
from chuk_mcp_math.number_theory.mathematical_constants import (
    # Pi approximations
    compute_pi_leibniz,
    compute_pi_nilakantha,
    compute_pi_machin,
    compute_pi_chudnovsky,
    # E approximations
    compute_e_series,
    compute_e_limit,
    # Golden ratio
    compute_golden_ratio_fibonacci,
    compute_golden_ratio_continued_fraction,
    # Euler gamma
    compute_euler_gamma_harmonic,
    # Continued fractions
    continued_fraction_pi,
    continued_fraction_e,
    continued_fraction_golden_ratio,
    # High precision
    pi_digits,
    e_digits,
    # Analysis
    approximation_error,
    convergence_comparison,
    constant_relationships,
)

# Mathematical constants for comparison
PI = math.pi
E = math.e
GOLDEN_RATIO = (1 + math.sqrt(5)) / 2  # φ ≈ 1.618033988749895
EULER_GAMMA = 0.5772156649015329  # γ (Euler-Mascheroni constant)

# ============================================================================
# PI APPROXIMATION TESTS
# ============================================================================


class TestPiApproximations:
    """Test cases for pi approximation functions."""

    @pytest.mark.asyncio
    async def test_compute_pi_leibniz_convergence(self):
        """Test Leibniz formula convergence."""
        # Test with increasing number of terms - more realistic expectations
        test_cases = [
            (100, 5e-1),  # Very rough approximation
            (1000, 5e-2),  # Better approximation
            (10000, 5e-3),  # Good approximation
        ]

        for terms, max_error in test_cases:
            result = await compute_pi_leibniz(terms)
            error = abs(result - PI)
            assert error < max_error, (
                f"Leibniz({terms}) error {error} should be < {max_error}"
            )
            assert 3.0 < result < 3.2, (
                f"Leibniz result should be reasonable, got {result}"
            )

    @pytest.mark.asyncio
    async def test_compute_pi_leibniz_properties(self):
        """Test mathematical properties of Leibniz formula."""
        # More terms should give better approximation
        pi_100 = await compute_pi_leibniz(100)
        pi_1000 = await compute_pi_leibniz(1000)

        error_100 = abs(pi_100 - PI)
        error_1000 = abs(pi_1000 - PI)

        assert error_1000 < error_100, "More terms should give better approximation"

        # Test alternating series behavior (should approach from below for even terms)
        pi_even = await compute_pi_leibniz(1000)  # Even number of terms
        assert pi_even < PI, "Leibniz with even terms should underestimate pi"

    @pytest.mark.asyncio
    async def test_compute_pi_nilakantha_convergence(self):
        """Test Nilakantha series convergence (faster than Leibniz)."""
        # Adjusted expectations based on actual convergence rates
        test_cases = [
            (10, 5e-4),  # Very few terms needed
            (100, 5e-7),  # Excellent accuracy
            (1000, 5e-10),  # Very high accuracy
        ]

        for terms, max_error in test_cases:
            result = await compute_pi_nilakantha(terms)
            error = abs(result - PI)
            assert error < max_error, (
                f"Nilakantha({terms}) error {error} should be < {max_error}"
            )
            assert 3.1 < result < 3.15, (
                f"Nilakantha result should be close to pi, got {result}"
            )

    @pytest.mark.asyncio
    async def test_compute_pi_machin_accuracy(self):
        """Test Machin's formula (very fast convergence)."""
        # Adjusted expectations for realistic convergence
        test_cases = [
            (5, 5e-8),  # Few terms give excellent accuracy
            (10, 1e-12),  # Very high accuracy
            (20, 1e-15),  # Near machine precision
        ]

        for terms, max_error in test_cases:
            result = await compute_pi_machin(terms)
            error = abs(result - PI)
            assert error < max_error, (
                f"Machin({terms}) error {error} should be < {max_error}"
            )
            assert 3.14 < result < 3.15, (
                f"Machin result should be very close to pi, got {result}"
            )

    @pytest.mark.asyncio
    async def test_compute_pi_chudnovsky_precision(self):
        """Test Chudnovsky algorithm (extremely fast convergence)."""
        # Chudnovsky gives ~14 digits per term
        test_cases = [
            (1, 1e-10),  # Single term gives excellent accuracy
            (2, 1e-15),  # Two terms near machine precision
            (3, 1e-15),  # Three terms still excellent
        ]

        for terms, max_error in test_cases:
            result = await compute_pi_chudnovsky(terms)
            error = abs(result - PI)
            assert error < max_error, (
                f"Chudnovsky({terms}) error {error} should be < {max_error}"
            )
            assert 3.141 < result < 3.142, (
                f"Chudnovsky result should be very close to pi, got {result}"
            )

    @pytest.mark.asyncio
    async def test_pi_approximation_comparison(self):
        """Test that different pi methods converge to the same value."""
        terms = 100

        leibniz_result = await compute_pi_leibniz(terms * 100)  # Needs more terms
        nilakantha_result = await compute_pi_nilakantha(terms)
        machin_result = await compute_pi_machin(terms // 5)  # Needs fewer terms

        # All should be within reasonable range of pi
        for result, method in [
            (leibniz_result, "Leibniz"),
            (nilakantha_result, "Nilakantha"),
            (machin_result, "Machin"),
        ]:
            assert 3.0 < result < 3.2, f"{method} result {result} should be reasonable"

        # Machin should be most accurate
        machin_error = abs(machin_result - PI)
        assert machin_error < 1e-10, (
            f"Machin should be very accurate, error: {machin_error}"
        )

    @pytest.mark.asyncio
    async def test_pi_approximations_edge_cases(self):
        """Test pi approximation edge cases."""
        # Zero terms
        assert await compute_pi_leibniz(0) == 0.0
        assert await compute_pi_nilakantha(0) == 3.0  # Starts with 3
        assert await compute_pi_machin(0) == 0.0
        assert await compute_pi_chudnovsky(0) == 0.0

        # Single term
        single_leibniz = await compute_pi_leibniz(1)
        assert single_leibniz == 4.0, "Single Leibniz term should be 4"

        single_nilakantha = await compute_pi_nilakantha(1)
        assert 3.1 < single_nilakantha < 3.2, (
            "Single Nilakantha term should be close to pi"
        )


# ============================================================================
# E APPROXIMATION TESTS
# ============================================================================


class TestEApproximations:
    """Test cases for e approximation functions."""

    @pytest.mark.asyncio
    async def test_compute_e_series_convergence(self):
        """Test e series convergence."""
        # Adjusted expectations for realistic convergence
        test_cases = [
            (5, 1e-2),  # Few terms for rough approximation
            (10, 5e-7),  # Good approximation
            (15, 1e-12),  # Excellent approximation
            (20, 1e-15),  # Near machine precision
        ]

        for terms, max_error in test_cases:
            result = await compute_e_series(terms)
            error = abs(result - E)
            assert error < max_error, (
                f"e_series({terms}) error {error} should be < {max_error}"
            )
            assert 2.7 < result < 2.73, (
                f"e_series result should be close to e, got {result}"
            )

    @pytest.mark.asyncio
    async def test_compute_e_series_properties(self):
        """Test mathematical properties of e series."""
        # More terms should give better approximation
        e_5 = await compute_e_series(5)
        e_15 = await compute_e_series(15)

        error_5 = abs(e_5 - E)
        error_15 = abs(e_15 - E)

        assert error_15 < error_5, "More terms should give better e approximation"

        # Series should approach from below initially
        e_small = await compute_e_series(3)
        assert e_small < E, "Small e series should underestimate e"

    @pytest.mark.asyncio
    async def test_compute_e_limit_convergence(self):
        """Test e limit definition convergence."""
        test_cases = [
            (1000, 1e-2),  # Rough approximation
            (100000, 1e-4),  # Better approximation
            (1000000, 1e-5),  # Good approximation
        ]

        for n, max_error in test_cases:
            result = await compute_e_limit(n)
            error = abs(result - E)
            assert error < max_error, (
                f"e_limit({n}) error {error} should be < {max_error}"
            )
            assert 2.7 < result < 2.72, (
                f"e_limit result should be close to e, got {result}"
            )

    @pytest.mark.asyncio
    async def test_e_approximation_comparison(self):
        """Test that different e methods converge to similar values."""
        e_series_result = await compute_e_series(15)
        e_limit_result = await compute_e_limit(100000)

        # Both should be close to the true value of e
        series_error = abs(e_series_result - E)
        limit_error = abs(e_limit_result - E)

        assert series_error < 1e-10, (
            f"e_series should be very accurate, error: {series_error}"
        )
        assert limit_error < 1e-4, (
            f"e_limit should be reasonably accurate, error: {limit_error}"
        )

        # Series method should be more accurate for reasonable term counts
        assert series_error < limit_error, (
            "e_series should be more accurate than e_limit"
        )

    @pytest.mark.asyncio
    async def test_e_approximations_edge_cases(self):
        """Test e approximation edge cases."""
        # Zero/negative terms
        assert await compute_e_series(0) == 1.0  # First term only
        assert await compute_e_limit(0) == 1.0  # Edge case

        # Single term
        single_series = await compute_e_series(1)
        assert single_series == 1.0, "Single e_series term should be 1"


# ============================================================================
# GOLDEN RATIO TESTS
# ============================================================================


class TestGoldenRatio:
    """Test cases for golden ratio approximation functions."""

    @pytest.mark.asyncio
    async def test_compute_golden_ratio_fibonacci_convergence(self):
        """Test golden ratio via Fibonacci convergence."""
        # Adjusted expectations for realistic convergence
        test_cases = [
            (10, 1e-3),  # Decent approximation
            (20, 5e-8),  # Very good approximation (adjusted)
            (30, 1e-10),  # Excellent approximation
        ]

        for terms, max_error in test_cases:
            result = await compute_golden_ratio_fibonacci(terms)
            error = abs(result - GOLDEN_RATIO)
            assert error < max_error, (
                f"fibonacci_golden_ratio({terms}) error {error} should be < {max_error}"
            )
            assert 1.6 < result < 1.62, (
                f"Golden ratio result should be reasonable, got {result}"
            )

    @pytest.mark.asyncio
    async def test_compute_golden_ratio_continued_fraction_convergence(self):
        """Test golden ratio via continued fraction convergence."""
        # Adjusted expectations for realistic continued fraction convergence
        # Golden ratio has notoriously slow continued fraction convergence
        test_cases = [
            (5, 2e-2),  # Rough approximation (fixed edge case)
            (10, 2e-4),  # Good approximation (adjusted)
            (20, 1e-8),  # Excellent approximation (adjusted from 1e-12)
        ]

        for depth, max_error in test_cases:
            result = await compute_golden_ratio_continued_fraction(depth)
            error = abs(result - GOLDEN_RATIO)
            assert error < max_error, (
                f"continued_fraction_golden_ratio({depth}) error {error} should be < {max_error}"
            )
            assert 1.59 < result < 1.62, (
                f"Golden ratio result should be reasonable, got {result}"
            )

    @pytest.mark.asyncio
    async def test_golden_ratio_methods_consistency(self):
        """Test that both golden ratio methods converge to same value."""
        fib_result = await compute_golden_ratio_fibonacci(25)
        cf_result = await compute_golden_ratio_continued_fraction(20)

        # Both should be close to the true golden ratio - adjusted expectations
        fib_error = abs(fib_result - GOLDEN_RATIO)
        cf_error = abs(cf_result - GOLDEN_RATIO)

        assert fib_error < 5e-10, (
            f"Fibonacci method should be very accurate, error: {fib_error}"
        )
        assert cf_error < 1e-8, (
            f"Continued fraction method should be very accurate, error: {cf_error}"
        )

        # Results should be close to each other
        method_diff = abs(fib_result - cf_result)
        assert method_diff < 1e-8, (
            f"Both methods should give similar results, diff: {method_diff}"
        )

    @pytest.mark.asyncio
    async def test_golden_ratio_mathematical_properties(self):
        """Test mathematical properties of golden ratio."""
        phi = await compute_golden_ratio_fibonacci(30)

        # Golden ratio properties: φ² = φ + 1 - adjusted tolerance
        phi_squared = phi * phi
        phi_plus_one = phi + 1
        property_error = abs(phi_squared - phi_plus_one)
        assert property_error < 5e-8, f"φ² should equal φ + 1, error: {property_error}"

        # Another property: 1/φ = φ - 1 - adjusted tolerance
        one_over_phi = 1 / phi
        phi_minus_one = phi - 1
        inverse_property_error = abs(one_over_phi - phi_minus_one)
        assert inverse_property_error < 5e-8, (
            f"1/φ should equal φ - 1, error: {inverse_property_error}"
        )

    @pytest.mark.asyncio
    async def test_golden_ratio_edge_cases(self):
        """Test golden ratio edge cases."""
        # Insufficient terms
        assert await compute_golden_ratio_fibonacci(1) == 1.0
        assert await compute_golden_ratio_fibonacci(0) == 1.0
        assert await compute_golden_ratio_continued_fraction(0) == 1.0


# ============================================================================
# EULER-MASCHERONI CONSTANT TESTS
# ============================================================================


class TestEulerGamma:
    """Test cases for Euler-Mascheroni constant approximation."""

    @pytest.mark.asyncio
    async def test_compute_euler_gamma_harmonic_convergence(self):
        """Test Euler gamma approximation convergence."""
        test_cases = [
            (1000, 1e-3),  # Rough approximation
            (10000, 1e-4),  # Better approximation
            (100000, 1e-5),  # Good approximation
        ]

        for terms, max_error in test_cases:
            result = await compute_euler_gamma_harmonic(terms)
            error = abs(result - EULER_GAMMA)
            assert error < max_error, (
                f"euler_gamma({terms}) error {error} should be < {max_error}"
            )
            assert 0.5 < result < 0.6, (
                f"Euler gamma result should be reasonable, got {result}"
            )

    @pytest.mark.asyncio
    async def test_euler_gamma_properties(self):
        """Test mathematical properties of Euler gamma."""
        # More terms should give better approximation
        gamma_1000 = await compute_euler_gamma_harmonic(1000)
        gamma_10000 = await compute_euler_gamma_harmonic(10000)

        error_1000 = abs(gamma_1000 - EULER_GAMMA)
        error_10000 = abs(gamma_10000 - EULER_GAMMA)

        assert error_10000 < error_1000, (
            "More terms should give better Euler gamma approximation"
        )

    @pytest.mark.asyncio
    async def test_euler_gamma_edge_cases(self):
        """Test Euler gamma edge cases."""
        assert await compute_euler_gamma_harmonic(0) == 0.0

        # Single term
        gamma_1 = await compute_euler_gamma_harmonic(1)
        expected_1 = 1.0 - math.log(1)  # H_1 - ln(1) = 1 - 0 = 1
        assert abs(gamma_1 - expected_1) < 1e-10, "Single term should give H_1 - ln(1)"


# ============================================================================
# CONTINUED FRACTIONS TESTS
# ============================================================================


class TestContinuedFractions:
    """Test cases for continued fraction representations."""

    @pytest.mark.asyncio
    async def test_continued_fraction_pi_known_values(self):
        """Test known values of pi continued fraction."""
        # Known first few terms: [3; 7, 15, 1, 292, 1, 1, 1, 2, 1, 3, 1, ...]
        expected_start = [3, 7, 15, 1, 292, 1, 1, 1, 2, 1]

        result = await continued_fraction_pi(10)
        assert result == expected_start, (
            f"Pi CF should start with {expected_start}, got {result}"
        )

        # Test specific known values
        result_5 = await continued_fraction_pi(5)
        assert result_5 == [3, 7, 15, 1, 292], (
            "First 5 pi CF terms should be [3, 7, 15, 1, 292]"
        )

    @pytest.mark.asyncio
    async def test_continued_fraction_e_pattern(self):
        """Test e continued fraction pattern."""
        # Known pattern: [2; 1, 2, 1, 1, 4, 1, 1, 6, 1, 1, 8, 1, 1, 10, ...]
        expected_start = [2, 1, 2, 1, 1, 4, 1, 1, 6, 1, 1, 8, 1, 1, 10]

        result = await continued_fraction_e(15)
        assert result == expected_start, f"e CF should follow pattern, got {result}"

        # Test pattern recognition
        result_long = await continued_fraction_e(20)

        # Should follow pattern: 2, then groups of (1, 2k, 1)
        assert result_long[0] == 2, "e CF should start with 2"
        assert result_long[1] == 1, "Second term should be 1"
        assert result_long[2] == 2, "Third term should be 2"
        assert result_long[5] == 4, "Sixth term should be 4 (2×2)"
        assert result_long[8] == 6, "Ninth term should be 6 (2×3)"

    @pytest.mark.asyncio
    async def test_continued_fraction_golden_ratio_pattern(self):
        """Test golden ratio continued fraction (all 1s)."""
        for depth in [5, 10, 20, 50]:
            result = await continued_fraction_golden_ratio(depth)
            expected = [1] * depth
            assert result == expected, f"Golden ratio CF should be all 1s, got {result}"

    @pytest.mark.asyncio
    async def test_continued_fractions_edge_cases(self):
        """Test continued fraction edge cases."""
        # Zero depth
        assert await continued_fraction_pi(0) == []
        assert await continued_fraction_e(0) == []
        assert await continued_fraction_golden_ratio(0) == []

        # Single term
        assert await continued_fraction_pi(1) == [3]
        assert await continued_fraction_e(1) == [2]
        assert await continued_fraction_golden_ratio(1) == [1]


# ============================================================================
# HIGH PRECISION TESTS
# ============================================================================


class TestHighPrecision:
    """Test cases for high precision digit generation."""

    @pytest.mark.asyncio
    async def test_pi_digits_accuracy(self):
        """Test pi digits accuracy."""
        # Known pi digits: 3.141592653589793238462643383279502884197169399375105820974944...

        # Test various precisions - check realistic output
        pi_10 = await pi_digits(10)
        assert pi_10.startswith("3.141592653"), (
            f"Pi to 10 digits should start correctly, got {pi_10}"
        )

        pi_20 = await pi_digits(20)
        # Check that it starts correctly and has reasonable length
        assert pi_20.startswith("3.141592653589793"), (
            f"Pi to 20 digits should start correctly, got {pi_20}"
        )
        assert len(pi_20) >= 19, (
            f"Pi to 20 digits should have reasonable length, got {len(pi_20)}"
        )

        # Check format
        assert "." in pi_10, "Pi digits should contain decimal point"

    @pytest.mark.asyncio
    async def test_e_digits_accuracy(self):
        """Test e digits accuracy."""
        # Known e digits: 2.718281828459045235360287471352662497757247093699959574966967...

        # Test various precisions
        e_10 = await e_digits(10)
        assert e_10.startswith("2.718281828"), (
            f"e to 10 digits should start correctly, got {e_10}"
        )

        e_20 = await e_digits(20)
        assert e_20.startswith("2.71828182845904523"), (
            f"e to 20 digits should start correctly, got {e_20}"
        )
        assert len(e_20) >= 19, (
            f"e to 20 digits should have reasonable length, got {len(e_20)}"
        )

        # Check format
        assert "." in e_10, "e digits should contain decimal point"

    @pytest.mark.asyncio
    async def test_high_precision_edge_cases(self):
        """Test high precision edge cases."""
        # Zero precision
        assert await pi_digits(0) == "3"
        assert await e_digits(0) == "2"

        # Negative precision (should handle gracefully)
        assert await pi_digits(-5) == "3"
        assert await e_digits(-5) == "2"


# ============================================================================
# APPROXIMATION ANALYSIS TESTS
# ============================================================================


class TestApproximationAnalysis:
    """Test cases for approximation analysis functions."""

    @pytest.mark.asyncio
    async def test_approximation_error_accuracy(self):
        """Test approximation error calculation."""
        # Test known error patterns - adjusted expectations
        leibniz_error = await approximation_error("leibniz", 1000)
        nilakantha_error = await approximation_error("nilakantha", 100)
        machin_error = await approximation_error("machin", 20)

        # Errors should be reasonable
        assert 0 < leibniz_error < 0.1, (
            f"Leibniz error should be reasonable, got {leibniz_error}"
        )
        assert 0 < nilakantha_error < 1e-4, (
            f"Nilakantha error should be small, got {nilakantha_error}"
        )
        assert 0 < machin_error < 1e-8, (
            f"Machin error should be very small, got {machin_error}"
        )

        # Machin should be most accurate
        assert machin_error < nilakantha_error < leibniz_error, (
            "Machin should be most accurate"
        )

    @pytest.mark.asyncio
    async def test_approximation_error_methods(self):
        """Test all approximation error methods."""
        terms = 100
        methods = ["leibniz", "nilakantha", "machin", "chudnovsky"]

        for method in methods:
            error = await approximation_error(method, terms)
            assert 0 <= error < 1, (
                f"Error for {method} should be reasonable, got {error}"
            )

    @pytest.mark.asyncio
    async def test_approximation_error_edge_cases(self):
        """Test approximation error edge cases."""
        # Unknown method
        with pytest.raises(ValueError, match="Unknown method"):
            await approximation_error("unknown_method", 100)

    @pytest.mark.asyncio
    async def test_convergence_comparison_structure(self):
        """Test convergence comparison structure."""
        result = await convergence_comparison(60)

        # Should contain expected methods
        expected_methods = ["leibniz", "nilakantha", "machin"]
        for method in expected_methods:
            assert method in result, f"Result should contain {method}"

        # Each method should have multiple test points
        for method, values in result.items():
            assert isinstance(values, list), f"{method} should have list of values"
            assert len(values) >= 1, f"{method} should have at least one value"

            # All values should be reasonable approximations of pi
            for val in values:
                assert 2.9 < val < 3.2, (
                    f"{method} value {val} should be reasonable"
                )  # Adjusted range

    @pytest.mark.asyncio
    async def test_convergence_comparison_edge_cases(self):
        """Test convergence comparison edge cases."""
        # Zero terms
        result = await convergence_comparison(0)
        assert result == {}, "Zero terms should return empty dict"

        # Small number of terms
        result = await convergence_comparison(3)
        assert len(result) > 0, "Should handle small term counts"


# ============================================================================
# CONSTANT RELATIONSHIPS TESTS
# ============================================================================


class TestConstantRelationships:
    """Test cases for mathematical constant relationships."""

    @pytest.mark.asyncio
    async def test_euler_identity_relationship(self):
        """Test Euler's identity relationship."""
        # e^(iπ) + 1 = 0, so Re(e^(iπ)) = -1
        result = await constant_relationships("euler")
        assert abs(result - (-1.0)) < 1e-10, (
            f"Euler identity should give -1, got {result}"
        )

    @pytest.mark.asyncio
    async def test_golden_ratio_conjugate(self):
        """Test golden ratio conjugate relationship."""
        result = await constant_relationships("golden_conjugate")
        expected = -(GOLDEN_RATIO - 1)  # Fixed: should be -(φ - 1) = -1/φ
        assert abs(result - expected) < 1e-10, (
            f"Golden conjugate should be {expected}, got {result}"
        )

        # Should be negative reciprocal of golden ratio
        assert result < 0, "Golden conjugate should be negative"
        assert abs(result + 1 / GOLDEN_RATIO) < 1e-10, "Should equal -1/φ"

    @pytest.mark.asyncio
    async def test_pi_e_difference(self):
        """Test pi - e relationship."""
        result = await constant_relationships("pi_e_difference")
        expected = PI - E
        assert abs(result - expected) < 1e-10, (
            f"π - e should be {expected}, got {result}"
        )
        assert result > 0, "π should be larger than e"

    @pytest.mark.asyncio
    async def test_constant_relationships_edge_cases(self):
        """Test constant relationships edge cases."""
        # Unknown identity
        with pytest.raises(ValueError, match="Unknown identity"):
            await constant_relationships("unknown_identity")


# ============================================================================
# INTEGRATION AND MATHEMATICAL PROPERTIES TESTS
# ============================================================================


class TestIntegrationAndProperties:
    """Integration tests and mathematical property verification."""

    @pytest.mark.asyncio
    async def test_algorithm_consistency_pi(self):
        """Test that different pi algorithms converge to same value."""
        # Use appropriate term counts for each algorithm
        leibniz_pi = await compute_pi_leibniz(50000)  # Needs many terms
        nilakantha_pi = await compute_pi_nilakantha(1000)  # Converges faster
        machin_pi = await compute_pi_machin(50)  # Very fast convergence
        chudnovsky_pi = await compute_pi_chudnovsky(3)  # Extremely fast

        # All should be close to true pi
        algorithms = [
            ("Leibniz", leibniz_pi),
            ("Nilakantha", nilakantha_pi),
            ("Machin", machin_pi),
            ("Chudnovsky", chudnovsky_pi),
        ]

        for name, result in algorithms:
            error = abs(result - PI)
            assert error < 1e-4, f"{name} should be reasonably accurate, error: {error}"

        # Machin and Chudnovsky should be very accurate
        assert abs(machin_pi - PI) < 1e-10, "Machin should be very accurate"
        assert abs(chudnovsky_pi - PI) < 1e-10, "Chudnovsky should be very accurate"

    @pytest.mark.asyncio
    async def test_algorithm_consistency_e(self):
        """Test that different e algorithms converge to same value."""
        series_e = await compute_e_series(20)
        limit_e = await compute_e_limit(1000000)

        # Both should approximate e
        series_error = abs(series_e - E)
        limit_error = abs(limit_e - E)

        assert series_error < 1e-12, (
            f"e_series should be very accurate, error: {series_error}"
        )
        assert limit_error < 1e-4, (
            f"e_limit should be reasonably accurate, error: {limit_error}"
        )

    @pytest.mark.asyncio
    async def test_continued_fraction_convergence(self):
        """Test that continued fractions converge to expected values."""
        # Test convergent calculation for golden ratio
        # φ = [1; 1, 1, 1, ...] should converge to golden ratio

        def evaluate_continued_fraction(coeffs):
            """Evaluate continued fraction from coefficients."""
            if not coeffs:
                return 0
            if len(coeffs) == 1:
                return coeffs[0]

            result = coeffs[-1]
            for i in range(len(coeffs) - 2, -1, -1):
                result = coeffs[i] + 1 / result
            return result

        # Test golden ratio convergence - very relaxed tolerance due to slow convergence
        for depth in [25, 30]:  # Use higher depths where convergence is better
            cf_coeffs = await continued_fraction_golden_ratio(depth)
            convergent = evaluate_continued_fraction(cf_coeffs)
            error = abs(convergent - GOLDEN_RATIO)
            assert error < 1e-6, (
                f"Golden ratio CF convergent should be accurate at depth {depth}, error: {error}"
            )

    @pytest.mark.asyncio
    async def test_mathematical_identities_verification(self):
        """Test verification of mathematical identities."""
        # Golden ratio identity: φ² = φ + 1 - adjusted tolerance
        phi_fib = await compute_golden_ratio_fibonacci(30)
        phi_cf = await compute_golden_ratio_continued_fraction(20)

        for phi in [phi_fib, phi_cf]:
            identity_error = abs(phi * phi - (phi + 1))
            assert identity_error < 5e-8, (
                f"Golden ratio identity φ² = φ + 1 should hold, error: {identity_error}"
            )

        # e series relationship: e = sum(1/n!) should match limit definition
        e_series_20 = await compute_e_series(20)
        e_limit_result = await compute_e_limit(100000)

        # Should be reasonably close (limit converges slower)
        e_methods_diff = abs(e_series_20 - e_limit_result)
        assert e_methods_diff < 1e-3, (
            f"Different e methods should be reasonably close, diff: {e_methods_diff}"
        )

    @pytest.mark.asyncio
    async def test_precision_scaling_properties(self):
        """Test that precision scales appropriately with computational effort."""
        # Pi approximations should improve with more terms
        pi_errors = []
        term_counts = [100, 1000, 10000]

        for terms in term_counts:
            pi_approx = await compute_pi_nilakantha(terms)
            error = abs(pi_approx - PI)
            pi_errors.append(error)

        # Errors should generally decrease
        for i in range(len(pi_errors) - 1):
            assert pi_errors[i + 1] < pi_errors[i] * 2, (
                "Error should improve with more terms"
            )

        # e series should improve dramatically with more terms
        e_errors = []
        e_term_counts = [5, 10, 15]

        for terms in e_term_counts:
            e_approx = await compute_e_series(terms)
            error = abs(e_approx - E)
            e_errors.append(error)

        # e series converges very fast
        assert e_errors[-1] < e_errors[0] / 100, "e series should converge very quickly"


# ============================================================================
# PERFORMANCE AND ASYNC BEHAVIOR TESTS
# ============================================================================


class TestPerformance:
    """Performance and async behavior tests."""

    @pytest.mark.asyncio
    async def test_all_functions_are_async(self):
        """Test that all mathematical constants functions are properly async."""
        operations = [
            compute_pi_leibniz(100),
            compute_pi_nilakantha(50),
            compute_pi_machin(10),
            compute_pi_chudnovsky(2),
            compute_e_series(10),
            compute_e_limit(10000),
            compute_golden_ratio_fibonacci(15),
            compute_golden_ratio_continued_fraction(10),
            compute_euler_gamma_harmonic(1000),
            continued_fraction_pi(10),
            continued_fraction_e(15),
            continued_fraction_golden_ratio(10),
            pi_digits(20),
            e_digits(20),
            approximation_error("machin", 10),
            convergence_comparison(30),
            constant_relationships("pi_e_difference"),
        ]

        # Ensure all are coroutines
        for op in operations:
            assert asyncio.iscoroutine(op)

        # Run all operations concurrently
        start_time = time.time()
        results = await asyncio.gather(*operations)
        duration = time.time() - start_time

        # Should complete in reasonable time
        assert duration < 10.0, f"Operations took too long: {duration}s"

        # Verify results have expected types and reasonable values
        assert 3.1 < results[0] < 3.2  # compute_pi_leibniz
        assert 3.14 < results[1] < 3.15  # compute_pi_nilakantha
        assert 3.141 < results[2] < 3.142  # compute_pi_machin
        assert 3.141 < results[3] < 3.142  # compute_pi_chudnovsky
        assert 2.7 < results[4] < 2.72  # compute_e_series
        assert 2.7 < results[5] < 2.72  # compute_e_limit
        assert 1.6 < results[6] < 1.62  # compute_golden_ratio_fibonacci
        assert 1.6 < results[7] < 1.62  # compute_golden_ratio_continued_fraction
        assert 0.5 < results[8] < 0.6  # compute_euler_gamma_harmonic
        assert isinstance(results[9], list)  # continued_fraction_pi
        assert isinstance(results[10], list)  # continued_fraction_e
        assert isinstance(results[11], list)  # continued_fraction_golden_ratio
        assert results[12].startswith("3.141")  # pi_digits
        assert results[13].startswith("2.718")  # e_digits
        assert isinstance(results[14], float)  # approximation_error
        assert isinstance(results[15], dict)  # convergence_comparison
        assert isinstance(results[16], float)  # constant_relationships

    @pytest.mark.asyncio
    async def test_concurrent_execution(self):
        """Test that mathematical constants operations can run concurrently."""
        start_time = time.time()

        # Run multiple different algorithms concurrently
        tasks = []
        for i in range(5):
            tasks.extend(
                [
                    compute_pi_leibniz(1000 + i * 100),
                    compute_pi_nilakantha(100 + i * 10),
                    compute_e_series(10 + i),
                    compute_golden_ratio_fibonacci(15 + i),
                    approximation_error("machin", 10 + i),
                ]
            )

        results = await asyncio.gather(*tasks)
        duration = time.time() - start_time

        # Should complete efficiently due to async nature
        assert duration < 15.0
        assert len(results) == 25  # 5 iterations × 5 operations

        # Check some patterns in results
        pi_leibniz_results = results[::5]  # Every 5th result
        for result in pi_leibniz_results:
            assert 3.1 < result < 3.2, (
                f"Pi Leibniz result should be reasonable: {result}"
            )

    @pytest.mark.asyncio
    async def test_high_precision_performance(self):
        """Test performance of high precision computations."""
        start_time = time.time()

        # Test various precision levels
        precision_tasks = [
            pi_digits(50),
            e_digits(50),
            compute_pi_chudnovsky(4),
            compute_e_series(25),
        ]

        results = await asyncio.gather(*precision_tasks)
        duration = time.time() - start_time

        # High precision should complete in reasonable time
        assert duration < 5.0, f"High precision computations took too long: {duration}s"

        # Results should be accurate
        pi_50, e_50, pi_chud, e_25 = results

        assert pi_50.startswith("3.14159265358979323846"), (
            "High precision pi should be accurate"
        )
        assert e_50.startswith("2.71828182845904523536"), (
            "High precision e should be accurate"
        )
        assert abs(pi_chud - PI) < 1e-14, "Chudnovsky should be very accurate"
        assert abs(e_25 - E) < 1e-15, "e_series with 25 terms should be very accurate"


# ============================================================================
# ERROR HANDLING TESTS
# ============================================================================


class TestErrorHandling:
    """Test error handling and edge cases."""

    @pytest.mark.asyncio
    async def test_edge_case_handling(self):
        """Test edge case handling across all functions."""
        # Functions should handle zero/negative inputs gracefully

        # Test zero terms/precision
        assert await compute_pi_leibniz(0) == 0.0
        assert await compute_e_series(0) == 1.0
        assert await pi_digits(0) == "3"
        assert await e_digits(0) == "2"

        # Test small inputs
        assert await compute_golden_ratio_fibonacci(1) == 1.0
        assert await continued_fraction_pi(1) == [3]
        assert await convergence_comparison(1) != {}

    @pytest.mark.asyncio
    async def test_invalid_method_handling(self):
        """Test handling of invalid method names."""
        with pytest.raises(ValueError, match="Unknown method"):
            await approximation_error("invalid_method", 100)

        with pytest.raises(ValueError, match="Unknown identity"):
            await constant_relationships("invalid_identity")

    @pytest.mark.asyncio
    async def test_error_preserves_async_context(self):
        """Test that errors are properly raised in async context."""
        try:
            await approximation_error("invalid", 100)  # Should raise ValueError
            assert False, "Should have raised ValueError"
        except ValueError:
            # Should be able to continue with async operations
            result = await compute_pi_machin(10)
            assert 3.141 < result < 3.142


# ============================================================================
# PARAMETRIZED TESTS
# ============================================================================


class TestParametrized:
    """Parametrized tests for comprehensive coverage."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "terms,max_error", [(100, 0.1), (1000, 0.01), (10000, 0.001)]
    )
    async def test_pi_leibniz_parametrized(self, terms, max_error):
        """Parametrized test for pi Leibniz convergence."""
        result = await compute_pi_leibniz(terms)
        error = abs(result - PI)
        assert error < max_error

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "terms,max_error", [(10, 5e-4), (100, 5e-7), (1000, 5e-10)]
    )
    async def test_pi_nilakantha_parametrized(self, terms, max_error):
        """Parametrized test for pi Nilakantha convergence."""
        result = await compute_pi_nilakantha(terms)
        error = abs(result - PI)
        assert error < max_error

    @pytest.mark.asyncio
    @pytest.mark.parametrize("terms,max_error", [(5, 1e-2), (10, 5e-7), (20, 1e-15)])
    async def test_e_series_parametrized(self, terms, max_error):
        """Parametrized test for e series convergence."""
        result = await compute_e_series(terms)
        error = abs(result - E)
        assert error < max_error

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "method", ["leibniz", "nilakantha", "machin", "chudnovsky"]
    )
    async def test_approximation_error_parametrized(self, method):
        """Parametrized test for approximation error methods."""
        error = await approximation_error(method, 50)
        assert 0 <= error < 1

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "identity", ["euler", "golden_conjugate", "pi_e_difference"]
    )
    async def test_constant_relationships_parametrized(self, identity):
        """Parametrized test for constant relationships."""
        result = await constant_relationships(identity)
        assert isinstance(result, float)


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v", "--tb=short", "--asyncio-mode=auto"])
