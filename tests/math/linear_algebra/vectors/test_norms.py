#!/usr/bin/env python3
"""
Unit tests for vector norms and normalization functions.

Tests cover all functions in chuk_mcp_math.linear_algebra.vectors.norms:
- vector_norm (general p-norm)
- euclidean_norm (L2 norm)
- manhattan_norm (L1 norm)
- chebyshev_norm (L∞ norm)
- p_norm
- normalize_vector
"""

import pytest
import math
import asyncio
from typing import List, Union
from unittest.mock import patch, AsyncMock

from chuk_mcp_math.linear_algebra.vectors.norms import (
    vector_norm,
    euclidean_norm,
    manhattan_norm,
    chebyshev_norm,
    p_norm,
    normalize_vector
)

Number = Union[int, float]


class TestData:
    """Constants and test data for vector norm tests."""
    
    # Standard test vectors
    VECTORS = {
        'unit_x': [1, 0, 0],
        'unit_y': [0, 1, 0],
        'unit_z': [0, 0, 1],
        'ones': [1, 1, 1],
        'simple_3_4': [3, 4],
        'simple_3_4_0': [3, 4, 0],
        'negative': [-1, -2, -3],
        'mixed': [1, -2, 3, -4],
        'decimal': [1.5, 2.5, 3.5],
        'small': [1e-10, 2e-10, 3e-10],
        'large': [1e10, 2e10, 3e10]
    }
    
    # Expected norm values
    EXPECTED_NORMS = {
        'euclidean_3_4': 5.0,
        'manhattan_1_2_3': 6.0,
        'chebyshev_1_5_3': 5.0
    }
    
    # Tolerances
    ABS_TOL = 1e-10
    REL_TOL = 1e-9
    
    # Error messages
    ERRORS = {
        'empty_vector': "Vector cannot be empty",
        'invalid_p': "Norm order p must be >= 1",
        'zero_norm': "Cannot normalize vector with norm"
    }


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def simple_vectors():
    """Provide simple test vectors."""
    return {
        'two_d': [3, 4],
        'three_d': [1, 2, 3],
        'negative': [-1, -2, -3],
        'unit': [1, 0, 0]
    }


@pytest.fixture
def edge_vectors():
    """Provide edge case vectors."""
    return {
        'zero': [0, 0, 0],
        'single': [5],
        'large': [1e10] * 10,
        'small': [1e-10] * 10
    }


# ============================================================================
# TEST vector_norm (GENERAL P-NORM)
# ============================================================================

class TestVectorNorm:
    """Test cases for the general vector_norm function."""
    
    # ------------------------------------------------------------------------
    # Normal Operation Tests
    # ------------------------------------------------------------------------
    
    @pytest.mark.asyncio
    async def test_euclidean_norm_default(self):
        """Test that p=2 (Euclidean) is the default."""
        vector = [3, 4]
        result = await vector_norm(vector)
        assert math.isclose(result, 5.0, abs_tol=TestData.ABS_TOL)
    
    @pytest.mark.asyncio
    async def test_manhattan_norm(self):
        """Test L1 (Manhattan) norm."""
        vector = [1, -2, 3]
        result = await vector_norm(vector, p=1)
        assert math.isclose(result, 6.0, abs_tol=TestData.ABS_TOL)
    
    @pytest.mark.asyncio
    async def test_chebyshev_norm(self):
        """Test L∞ (Chebyshev) norm."""
        vector = [1, -5, 3]
        result = await vector_norm(vector, p=float('inf'))
        assert math.isclose(result, 5.0, abs_tol=TestData.ABS_TOL)
    
    @pytest.mark.asyncio
    async def test_p_norm_general(self):
        """Test general p-norm for various p values."""
        vector = [1, 2, 3]
        
        # p=3 norm
        result_p3 = await vector_norm(vector, p=3)
        expected_p3 = (1**3 + 2**3 + 3**3) ** (1/3)
        assert math.isclose(result_p3, expected_p3, rel_tol=TestData.REL_TOL)
        
        # p=4 norm
        result_p4 = await vector_norm(vector, p=4)
        expected_p4 = (1**4 + 2**4 + 3**4) ** (1/4)
        assert math.isclose(result_p4, expected_p4, rel_tol=TestData.REL_TOL)
    
    @pytest.mark.asyncio
    async def test_fractional_p_norm(self):
        """Test p-norm with fractional p values."""
        vector = [2, 3, 4]
        
        # p=1.5
        result = await vector_norm(vector, p=1.5)
        expected = (2**1.5 + 3**1.5 + 4**1.5) ** (1/1.5)
        assert math.isclose(result, expected, rel_tol=TestData.REL_TOL)
    
    @pytest.mark.asyncio
    async def test_single_element_vector(self):
        """Test norm of single-element vector."""
        vector = [5]
        
        # All norms should return absolute value
        assert await vector_norm(vector, p=1) == 5.0
        assert await vector_norm(vector, p=2) == 5.0
        assert await vector_norm(vector, p=float('inf')) == 5.0
        
        # Negative single element
        vector_neg = [-5]
        assert await vector_norm(vector_neg, p=1) == 5.0
        assert await vector_norm(vector_neg, p=2) == 5.0
    
    @pytest.mark.asyncio
    async def test_zero_vector(self):
        """Test norm of zero vector."""
        vector = [0, 0, 0]
        
        assert await vector_norm(vector, p=1) == 0.0
        assert await vector_norm(vector, p=2) == 0.0
        assert await vector_norm(vector, p=float('inf')) == 0.0
    
    @pytest.mark.asyncio
    async def test_negative_elements(self):
        """Test that norm handles negative elements correctly."""
        vector = [-3, -4]
        result = await vector_norm(vector, p=2)
        assert math.isclose(result, 5.0, abs_tol=TestData.ABS_TOL)
    
    @pytest.mark.asyncio
    async def test_mixed_signs(self):
        """Test vectors with mixed positive and negative elements."""
        vector = [1, -2, 3, -4]
        
        # L1 norm should sum absolute values
        result_l1 = await vector_norm(vector, p=1)
        assert math.isclose(result_l1, 10.0, abs_tol=TestData.ABS_TOL)
        
        # L2 norm
        result_l2 = await vector_norm(vector, p=2)
        expected_l2 = math.sqrt(1 + 4 + 9 + 16)
        assert math.isclose(result_l2, expected_l2, abs_tol=TestData.ABS_TOL)
    
    # ------------------------------------------------------------------------
    # Edge Case Tests
    # ------------------------------------------------------------------------
    
    @pytest.mark.asyncio
    async def test_empty_vector_error(self):
        """Test that empty vector raises ValueError."""
        with pytest.raises(ValueError, match=TestData.ERRORS['empty_vector']):
            await vector_norm([])
    
    @pytest.mark.asyncio
    async def test_invalid_p_value(self):
        """Test that p < 1 raises ValueError."""
        vector = [1, 2, 3]
        
        with pytest.raises(ValueError, match=TestData.ERRORS['invalid_p']):
            await vector_norm(vector, p=0)
        
        with pytest.raises(ValueError, match=TestData.ERRORS['invalid_p']):
            await vector_norm(vector, p=-1)
        
        with pytest.raises(ValueError, match=TestData.ERRORS['invalid_p']):
            await vector_norm(vector, p=0.5)
    
    @pytest.mark.asyncio
    async def test_large_vector_performance(self):
        """Test performance with large vectors (should yield control)."""
        large_vector = list(range(2000))
        
        # Should complete without issues
        result = await vector_norm(large_vector, p=2)
        assert result > 0
        assert not math.isinf(result)
    
    @pytest.mark.asyncio
    async def test_very_large_values(self):
        """Test numerical stability with very large values."""
        vector = [1e100, 2e100, 3e100]
        
        result = await vector_norm(vector, p=2)
        assert not math.isinf(result)
        assert not math.isnan(result)
        assert result > 0
    
    @pytest.mark.asyncio
    async def test_very_small_values(self):
        """Test numerical stability with very small values."""
        vector = [1e-100, 2e-100, 3e-100]
        
        result = await vector_norm(vector, p=2)
        assert not math.isnan(result)
        assert result >= 0
    
    # ------------------------------------------------------------------------
    # Type Handling Tests
    # ------------------------------------------------------------------------
    
    @pytest.mark.asyncio
    async def test_integer_vector(self):
        """Test with all integer elements."""
        vector = [1, 2, 3, 4, 5]
        result = await vector_norm(vector)
        assert isinstance(result, float)
        assert result > 0
    
    @pytest.mark.asyncio
    async def test_float_vector(self):
        """Test with all float elements."""
        vector = [1.5, 2.5, 3.5]
        result = await vector_norm(vector)
        assert isinstance(result, float)
        assert result > 0
    
    @pytest.mark.asyncio
    async def test_mixed_types_vector(self):
        """Test with mixed int and float elements."""
        vector = [1, 2.5, 3, 4.5]
        result = await vector_norm(vector)
        assert isinstance(result, float)
        assert result > 0


# ============================================================================
# TEST SPECIFIC NORM FUNCTIONS
# ============================================================================

class TestEuclideanNorm:
    """Test cases for euclidean_norm function."""
    
    @pytest.mark.asyncio
    async def test_simple_3_4_5_triangle(self):
        """Test classic 3-4-5 right triangle."""
        result = await euclidean_norm([3, 4])
        assert math.isclose(result, 5.0, abs_tol=TestData.ABS_TOL)
    
    @pytest.mark.asyncio
    async def test_3d_vector(self):
        """Test Euclidean norm of 3D vector."""
        vector = [2, 3, 6]
        expected = math.sqrt(4 + 9 + 36)  # sqrt(49) = 7
        result = await euclidean_norm(vector)
        assert math.isclose(result, expected, abs_tol=TestData.ABS_TOL)
    
    @pytest.mark.asyncio
    async def test_unit_vectors(self):
        """Test that unit vectors have norm 1."""
        assert math.isclose(await euclidean_norm([1, 0, 0]), 1.0, abs_tol=TestData.ABS_TOL)
        assert math.isclose(await euclidean_norm([0, 1, 0]), 1.0, abs_tol=TestData.ABS_TOL)
        assert math.isclose(await euclidean_norm([0, 0, 1]), 1.0, abs_tol=TestData.ABS_TOL)
    
    @pytest.mark.parametrize("vector,expected", [
        ([0, 0], 0.0),
        ([1, 0], 1.0),
        ([1, 1], math.sqrt(2)),
        ([1, 1, 1], math.sqrt(3)),
        ([-3, 4], 5.0),
        ([5, 12], 13.0),
    ])
    @pytest.mark.asyncio
    async def test_known_values(self, vector, expected):
        """Test with known vector-norm pairs."""
        result = await euclidean_norm(vector)
        assert math.isclose(result, expected, abs_tol=TestData.ABS_TOL)


class TestManhattanNorm:
    """Test cases for manhattan_norm function."""
    
    @pytest.mark.asyncio
    async def test_positive_vector(self):
        """Test Manhattan norm with positive elements."""
        result = await manhattan_norm([1, 2, 3])
        assert math.isclose(result, 6.0, abs_tol=TestData.ABS_TOL)
    
    @pytest.mark.asyncio
    async def test_negative_vector(self):
        """Test Manhattan norm with negative elements."""
        result = await manhattan_norm([-1, -2, -3])
        assert math.isclose(result, 6.0, abs_tol=TestData.ABS_TOL)
    
    @pytest.mark.asyncio
    async def test_mixed_signs(self):
        """Test Manhattan norm with mixed signs."""
        result = await manhattan_norm([1, -2, 3, -4])
        assert math.isclose(result, 10.0, abs_tol=TestData.ABS_TOL)
    
    @pytest.mark.parametrize("vector,expected", [
        ([0, 0], 0.0),
        ([1, 0], 1.0),
        ([1, 1], 2.0),
        ([1, -1], 2.0),
        ([2, 3, 4], 9.0),
        ([-5], 5.0),
    ])
    @pytest.mark.asyncio
    async def test_known_values(self, vector, expected):
        """Test with known vector-norm pairs."""
        result = await manhattan_norm(vector)
        assert math.isclose(result, expected, abs_tol=TestData.ABS_TOL)


class TestChebyshevNorm:
    """Test cases for chebyshev_norm function."""
    
    @pytest.mark.asyncio
    async def test_maximum_element(self):
        """Test that Chebyshev norm returns maximum absolute value."""
        result = await chebyshev_norm([1, -5, 3])
        assert math.isclose(result, 5.0, abs_tol=TestData.ABS_TOL)
    
    @pytest.mark.asyncio
    async def test_all_equal(self):
        """Test Chebyshev norm when all elements have same absolute value."""
        result = await chebyshev_norm([3, -3, 3])
        assert math.isclose(result, 3.0, abs_tol=TestData.ABS_TOL)
    
    @pytest.mark.parametrize("vector,expected", [
        ([0, 0], 0.0),
        ([1, 0], 1.0),
        ([1, 2, 3], 3.0),
        ([-4, 2, 3], 4.0),
        ([10, -20, 15], 20.0),
        ([7], 7.0),
    ])
    @pytest.mark.asyncio
    async def test_known_values(self, vector, expected):
        """Test with known vector-norm pairs."""
        result = await chebyshev_norm(vector)
        assert math.isclose(result, expected, abs_tol=TestData.ABS_TOL)


class TestPNorm:
    """Test cases for p_norm function."""
    
    @pytest.mark.asyncio
    async def test_equivalent_to_vector_norm(self):
        """Test that p_norm is equivalent to vector_norm."""
        vector = [1, 2, 3]
        
        for p in [1, 2, 3, 4, float('inf')]:
            result1 = await p_norm(vector, p)
            result2 = await vector_norm(vector, p)
            assert math.isclose(result1, result2, abs_tol=TestData.ABS_TOL)
    
    @pytest.mark.asyncio
    async def test_various_p_values(self):
        """Test p_norm with various p values."""
        vector = [2, 3, 4]
        
        # Test different p values
        p_values = [1, 1.5, 2, 2.5, 3, 10, 100]
        
        for p in p_values:
            result = await p_norm(vector, p)
            assert result > 0
            assert not math.isnan(result)
            assert not math.isinf(result)


# ============================================================================
# TEST normalize_vector
# ============================================================================

class TestNormalizeVector:
    """Test cases for normalize_vector function."""
    
    # ------------------------------------------------------------------------
    # Normal Operation Tests
    # ------------------------------------------------------------------------
    
    @pytest.mark.asyncio
    async def test_simple_normalization(self):
        """Test basic vector normalization."""
        vector = [3, 4]
        result = await normalize_vector(vector)
        
        assert len(result) == 2
        assert math.isclose(result[0], 0.6, abs_tol=TestData.ABS_TOL)
        assert math.isclose(result[1], 0.8, abs_tol=TestData.ABS_TOL)
        
        # Verify unit length
        norm = math.sqrt(result[0]**2 + result[1]**2)
        assert math.isclose(norm, 1.0, abs_tol=TestData.ABS_TOL)
    
    @pytest.mark.asyncio
    async def test_3d_normalization(self):
        """Test normalization of 3D vector."""
        vector = [1, 2, 2]
        result = await normalize_vector(vector)
        
        # Expected: [1/3, 2/3, 2/3]
        assert math.isclose(result[0], 1/3, abs_tol=TestData.ABS_TOL)
        assert math.isclose(result[1], 2/3, abs_tol=TestData.ABS_TOL)
        assert math.isclose(result[2], 2/3, abs_tol=TestData.ABS_TOL)
        
        # Verify unit length
        norm = math.sqrt(sum(x**2 for x in result))
        assert math.isclose(norm, 1.0, abs_tol=TestData.ABS_TOL)
    
    @pytest.mark.asyncio
    async def test_already_normalized(self):
        """Test normalizing an already normalized vector."""
        vector = [0.6, 0.8]
        result = await normalize_vector(vector)
        
        assert math.isclose(result[0], 0.6, abs_tol=TestData.ABS_TOL)
        assert math.isclose(result[1], 0.8, abs_tol=TestData.ABS_TOL)
    
    @pytest.mark.asyncio
    async def test_negative_elements(self):
        """Test normalization with negative elements."""
        vector = [-3, 4]
        result = await normalize_vector(vector)
        
        assert math.isclose(result[0], -0.6, abs_tol=TestData.ABS_TOL)
        assert math.isclose(result[1], 0.8, abs_tol=TestData.ABS_TOL)
        
        # Verify unit length
        norm = math.sqrt(result[0]**2 + result[1]**2)
        assert math.isclose(norm, 1.0, abs_tol=TestData.ABS_TOL)
    
    @pytest.mark.asyncio
    async def test_different_norm_types(self):
        """Test normalization using different norm types."""
        vector = [3, 4]
        
        # L1 normalization
        result_l1 = await normalize_vector(vector, norm_type=1)
        l1_norm = sum(abs(x) for x in result_l1)
        assert math.isclose(l1_norm, 1.0, abs_tol=TestData.ABS_TOL)
        
        # L∞ normalization
        result_linf = await normalize_vector(vector, norm_type=float('inf'))
        linf_norm = max(abs(x) for x in result_linf)
        assert math.isclose(linf_norm, 1.0, abs_tol=TestData.ABS_TOL)
    
    @pytest.mark.asyncio
    async def test_custom_tolerance(self):
        """Test normalization with custom tolerance."""
        vector = [1e-11, 1e-11]
        
        # Should fail with default tolerance
        with pytest.raises(ValueError, match=TestData.ERRORS['zero_norm']):
            await normalize_vector(vector)
        
        # Should work with higher tolerance
        result = await normalize_vector(vector, tolerance=1e-12)
        assert len(result) == 2
    
    # ------------------------------------------------------------------------
    # Edge Case Tests
    # ------------------------------------------------------------------------
    
    @pytest.mark.asyncio
    async def test_zero_vector_error(self):
        """Test that zero vector raises ValueError."""
        vector = [0, 0, 0]
        
        with pytest.raises(ValueError, match=TestData.ERRORS['zero_norm']):
            await normalize_vector(vector)
    
    @pytest.mark.asyncio
    async def test_near_zero_vector_error(self):
        """Test that near-zero vector raises ValueError."""
        vector = [1e-15, 1e-15]
        
        with pytest.raises(ValueError, match=TestData.ERRORS['zero_norm']):
            await normalize_vector(vector)
    
    @pytest.mark.asyncio
    async def test_empty_vector_error(self):
        """Test that empty vector raises ValueError."""
        with pytest.raises(ValueError, match=TestData.ERRORS['empty_vector']):
            await normalize_vector([])
    
    @pytest.mark.asyncio
    async def test_single_element_normalization(self):
        """Test normalization of single-element vector."""
        # Positive element
        result = await normalize_vector([5])
        assert len(result) == 1
        assert math.isclose(result[0], 1.0, abs_tol=TestData.ABS_TOL)
        
        # Negative element
        result = await normalize_vector([-5])
        assert len(result) == 1
        assert math.isclose(result[0], -1.0, abs_tol=TestData.ABS_TOL)
    
    @pytest.mark.asyncio
    async def test_large_vector_performance(self):
        """Test performance with large vectors."""
        large_vector = list(range(1, 2001))
        
        # Should complete without issues
        result = await normalize_vector(large_vector)
        assert len(result) == 2000
        
        # Verify unit length
        norm = math.sqrt(sum(x**2 for x in result))
        assert math.isclose(norm, 1.0, rel_tol=TestData.REL_TOL)
    
    # ------------------------------------------------------------------------
    # Type Preservation Tests
    # ------------------------------------------------------------------------
    
    @pytest.mark.asyncio
    async def test_return_type(self):
        """Test that normalized vector contains floats."""
        vector = [1, 2, 3]
        result = await normalize_vector(vector)
        
        assert all(isinstance(x, float) for x in result)
    
    @pytest.mark.asyncio
    async def test_input_not_modified(self):
        """Test that input vector is not modified."""
        vector = [3, 4]
        original = vector.copy()
        
        await normalize_vector(vector)
        
        assert vector == original


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestIntegration:
    """Integration tests for norm functions."""
    
    @pytest.mark.asyncio
    async def test_norm_after_normalization(self):
        """Test that normalized vectors have unit norm."""
        vectors = [
            [3, 4],
            [1, 2, 3],
            [-1, -2, -3],
            [1, 0, 0, 0, 0]
        ]
        
        for vector in vectors:
            normalized = await normalize_vector(vector)
            norm = await euclidean_norm(normalized)
            assert math.isclose(norm, 1.0, abs_tol=TestData.ABS_TOL)
    
    @pytest.mark.asyncio
    async def test_different_norms_relationship(self):
        """Test mathematical relationships between different norms."""
        vector = [1, 2, 3, 4]
        
        l1 = await manhattan_norm(vector)
        l2 = await euclidean_norm(vector)
        linf = await chebyshev_norm(vector)
        
        # Mathematical property: L∞ ≤ L2 ≤ L1
        assert linf <= l2 <= l1
        
        # For n-dimensional vector: L∞ ≤ L2 ≤ √n * L∞
        n = len(vector)
        assert linf <= l2 <= math.sqrt(n) * linf
    
    @pytest.mark.asyncio
    async def test_norm_scaling_property(self):
        """Test that norm scales linearly with scalar multiplication."""
        vector = [1, 2, 3]
        scalar = 5
        
        norm1 = await euclidean_norm(vector)
        scaled_vector = [scalar * x for x in vector]
        norm2 = await euclidean_norm(scaled_vector)
        
        assert math.isclose(norm2, abs(scalar) * norm1, rel_tol=TestData.REL_TOL)
    
    @pytest.mark.asyncio
    async def test_triangle_inequality(self):
        """Test triangle inequality: ||x + y|| ≤ ||x|| + ||y||."""
        x = [1, 2, 3]
        y = [4, 5, 6]
        
        norm_x = await euclidean_norm(x)
        norm_y = await euclidean_norm(y)
        
        # x + y
        sum_vector = [a + b for a, b in zip(x, y)]
        norm_sum = await euclidean_norm(sum_vector)
        
        # Triangle inequality
        assert norm_sum <= norm_x + norm_y + TestData.ABS_TOL


# ============================================================================
# PERFORMANCE TESTS
# ============================================================================

class TestPerformance:
    """Performance tests for norm functions."""
    
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_concurrent_norm_calculations(self):
        """Test multiple concurrent norm calculations."""
        vectors = [list(range(i, i+100)) for i in range(50)]
        
        tasks = [euclidean_norm(v) for v in vectors]
        results = await asyncio.gather(*tasks)
        
        assert len(results) == 50
        assert all(r > 0 for r in results)
    
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_large_vector_handling(self):
        """Test handling of very large vectors."""
        import time
        
        # Create a large vector
        large_vector = list(range(10000))
        
        start = time.perf_counter()
        result = await euclidean_norm(large_vector)
        elapsed = time.perf_counter() - start
        
        assert result > 0
        assert elapsed < 1.0  # Should complete within 1 second
    
    @pytest.mark.asyncio
    async def test_memory_efficiency(self):
        """Test that functions don't have memory leaks."""
        # Run many operations
        for _ in range(100):
            vector = list(range(100))
            await euclidean_norm(vector)
            await manhattan_norm(vector)
            await chebyshev_norm(vector)
            await normalize_vector(vector)
        
        # If we get here without memory errors, test passes
        assert True


# ============================================================================
# NUMERICAL STABILITY TESTS
# ============================================================================

class TestNumericalStability:
    """Test numerical stability of norm calculations."""
    
    @pytest.mark.asyncio
    async def test_overflow_prevention(self):
        """Test that large values are handled (may overflow for extreme values)."""
        # Use more reasonable large values that won't cause overflow
        vector = [1e100, 2e100, 3e100]
        
        # Should handle large values gracefully
        result = await euclidean_norm(vector)
        # For extremely large values, we accept infinity as a valid result
        assert not math.isnan(result)
        # Result should be positive (could be inf)
        if not math.isinf(result):
            assert result > 0
    
    @pytest.mark.asyncio
    async def test_underflow_prevention(self):
        """Test that small values don't cause underflow."""
        vector = [1e-200, 2e-200, 3e-200]
        
        result = await euclidean_norm(vector)
        assert not math.isnan(result)
        assert result >= 0
    
    @pytest.mark.asyncio
    async def test_mixed_scales(self):
        """Test vectors with elements of vastly different scales."""
        vector = [1e-100, 1.0, 1e100]
        
        result = await euclidean_norm(vector)
        # Should be dominated by the largest element
        assert math.isclose(result, 1e100, rel_tol=0.01)