#!/usr/bin/env python3
"""
Comprehensive unit tests for vector operations module.

Tests all functions in chuk_mcp_math.linear_algebra.vectors.operations:
- dot_product
- cross_product
- scalar_multiply
- vector_add
- vector_subtract
- element_wise_multiply
- element_wise_divide
"""

import pytest
import math
import asyncio

from chuk_mcp_math.linear_algebra.vectors.operations import (
    dot_product,
    cross_product,
    scalar_multiply,
    vector_add,
    vector_subtract,
    element_wise_multiply,
    element_wise_divide,
)


# ============================================================================
# TEST DATA AND FIXTURES
# ============================================================================


class TestData:
    """Test data and constants."""

    ABS_TOL = 1e-10
    REL_TOL = 1e-9

    # Standard test vectors
    VECTORS = {
        "zero_2d": [0, 0],
        "zero_3d": [0, 0, 0],
        "unit_x": [1, 0, 0],
        "unit_y": [0, 1, 0],
        "unit_z": [0, 0, 1],
        "simple_2d": [3, 4],
        "simple_3d": [1, 2, 3],
        "negative": [-1, -2, -3],
        "mixed": [1, -2, 3],
    }


@pytest.fixture
def standard_vectors():
    """Provide standard test vectors."""
    return {
        "v1": [1, 2, 3],
        "v2": [4, 5, 6],
        "v3": [2, 0, -1],
    }


# ============================================================================
# TEST DOT PRODUCT
# ============================================================================


class TestDotProduct:
    """Test dot product function."""

    @pytest.mark.asyncio
    async def test_basic_dot_product(self):
        """Test basic dot product calculation."""
        result = await dot_product([1, 2, 3], [4, 5, 6])
        expected = 1 * 4 + 2 * 5 + 3 * 6  # 32
        assert math.isclose(result, expected, abs_tol=TestData.ABS_TOL)

    @pytest.mark.asyncio
    async def test_dot_product_example(self):
        """Test dot product with documented example."""
        result = await dot_product([1, 2, 3], [4, 5, 6])
        assert math.isclose(result, 32.0, abs_tol=TestData.ABS_TOL)

    @pytest.mark.asyncio
    async def test_dot_product_orthogonal(self):
        """Test dot product of orthogonal vectors equals zero."""
        result = await dot_product([1, 0], [0, 1])
        assert math.isclose(result, 0.0, abs_tol=TestData.ABS_TOL)

    @pytest.mark.asyncio
    async def test_dot_product_parallel(self):
        """Test dot product of parallel vectors."""
        result = await dot_product([2, 0, 0], [3, 0, 0])
        assert math.isclose(result, 6.0, abs_tol=TestData.ABS_TOL)

    @pytest.mark.asyncio
    async def test_dot_product_negative(self):
        """Test dot product with negative values."""
        result = await dot_product([1, -2, 3], [4, 5, -6])
        expected = 1 * 4 + (-2) * 5 + 3 * (-6)  # 4 - 10 - 18 = -24
        assert math.isclose(result, expected, abs_tol=TestData.ABS_TOL)

    @pytest.mark.asyncio
    async def test_dot_product_zero_vector(self):
        """Test dot product with zero vector."""
        result = await dot_product([0, 0, 0], [1, 2, 3])
        assert math.isclose(result, 0.0, abs_tol=TestData.ABS_TOL)

    @pytest.mark.asyncio
    async def test_dot_product_single_element(self):
        """Test dot product with single-element vectors."""
        result = await dot_product([5], [3])
        assert math.isclose(result, 15.0, abs_tol=TestData.ABS_TOL)

    @pytest.mark.asyncio
    async def test_dot_product_dimension_mismatch(self):
        """Test that dimension mismatch raises ValueError."""
        with pytest.raises(ValueError, match="same dimension"):
            await dot_product([1, 2], [1, 2, 3])

    @pytest.mark.asyncio
    async def test_dot_product_empty_vectors(self):
        """Test that empty vectors raise ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            await dot_product([], [])

    @pytest.mark.asyncio
    async def test_dot_product_commutative(self):
        """Test that dot product is commutative."""
        v1 = [1, 2, 3]
        v2 = [4, 5, 6]

        result1 = await dot_product(v1, v2)
        result2 = await dot_product(v2, v1)

        assert math.isclose(result1, result2, abs_tol=TestData.ABS_TOL)

    @pytest.mark.asyncio
    async def test_dot_product_return_type(self):
        """Test that result is always float."""
        result = await dot_product([1, 2], [3, 4])
        assert isinstance(result, float)

    @pytest.mark.asyncio
    async def test_dot_product_large_vector(self):
        """Test dot product with large vectors (async yield test)."""
        large_v = list(range(1, 1001))
        result = await dot_product(large_v, large_v)
        # Sum of squares from 1 to 1000
        expected = sum(i * i for i in range(1, 1001))
        assert math.isclose(result, expected, rel_tol=TestData.REL_TOL)

    @pytest.mark.asyncio
    async def test_dot_product_very_large_vector(self):
        """Test dot product with very large vectors (>1000 elements)."""
        large_v1 = list(range(1, 2001))
        large_v2 = list(range(2001, 4001))
        result = await dot_product(large_v1, large_v2)
        assert result > 0
        assert isinstance(result, float)


# ============================================================================
# TEST CROSS PRODUCT
# ============================================================================


class TestCrossProduct:
    """Test cross product function."""

    @pytest.mark.asyncio
    async def test_basic_cross_product(self):
        """Test basic cross product calculation."""
        result = await cross_product([1, 0, 0], [0, 1, 0])
        expected = [0.0, 0.0, 1.0]  # i × j = k
        assert all(math.isclose(r, e, abs_tol=TestData.ABS_TOL) for r, e in zip(result, expected))

    @pytest.mark.asyncio
    async def test_cross_product_example(self):
        """Test cross product with documented example."""
        result = await cross_product([1, 0, 0], [0, 1, 0])
        expected = [0.0, 0.0, 1.0]
        assert all(math.isclose(r, e, abs_tol=TestData.ABS_TOL) for r, e in zip(result, expected))

    @pytest.mark.asyncio
    async def test_cross_product_standard_basis(self):
        """Test cross product of standard basis vectors."""
        # i × j = k
        result_ij = await cross_product([1, 0, 0], [0, 1, 0])
        assert all(
            math.isclose(r, e, abs_tol=TestData.ABS_TOL) for r, e in zip(result_ij, [0, 0, 1])
        )

        # j × k = i
        result_jk = await cross_product([0, 1, 0], [0, 0, 1])
        assert all(
            math.isclose(r, e, abs_tol=TestData.ABS_TOL) for r, e in zip(result_jk, [1, 0, 0])
        )

        # k × i = j
        result_ki = await cross_product([0, 0, 1], [1, 0, 0])
        assert all(
            math.isclose(r, e, abs_tol=TestData.ABS_TOL) for r, e in zip(result_ki, [0, 1, 0])
        )

    @pytest.mark.asyncio
    async def test_cross_product_anticommutative(self):
        """Test that cross product is anticommutative."""
        v1 = [1, 2, 3]
        v2 = [4, 5, 6]

        result1 = await cross_product(v1, v2)
        result2 = await cross_product(v2, v1)

        # v1 × v2 = -(v2 × v1)
        assert all(
            math.isclose(r1, -r2, abs_tol=TestData.ABS_TOL) for r1, r2 in zip(result1, result2)
        )

    @pytest.mark.asyncio
    async def test_cross_product_parallel_vectors(self):
        """Test cross product of parallel vectors is zero."""
        result = await cross_product([2, 4, 6], [1, 2, 3])
        expected = [0.0, 0.0, 0.0]
        assert all(math.isclose(r, e, abs_tol=TestData.ABS_TOL) for r, e in zip(result, expected))

    @pytest.mark.asyncio
    async def test_cross_product_zero_vector(self):
        """Test cross product with zero vector."""
        result = await cross_product([0, 0, 0], [1, 2, 3])
        expected = [0.0, 0.0, 0.0]
        assert all(math.isclose(r, e, abs_tol=TestData.ABS_TOL) for r, e in zip(result, expected))

    @pytest.mark.asyncio
    async def test_cross_product_not_3d(self):
        """Test that non-3D vectors raise ValueError."""
        with pytest.raises(ValueError, match="3D vectors"):
            await cross_product([1, 2], [3, 4])

        with pytest.raises(ValueError, match="3D vectors"):
            await cross_product([1, 2, 3, 4], [5, 6, 7, 8])

    @pytest.mark.asyncio
    async def test_cross_product_return_type(self):
        """Test that result elements are floats."""
        result = await cross_product([1, 2, 3], [4, 5, 6])
        assert all(isinstance(x, float) for x in result)
        assert len(result) == 3


# ============================================================================
# TEST SCALAR MULTIPLY
# ============================================================================


class TestScalarMultiply:
    """Test scalar multiplication function."""

    @pytest.mark.asyncio
    async def test_basic_scalar_multiply(self):
        """Test basic scalar multiplication."""
        result = await scalar_multiply([1, 2, 3], 2)
        expected = [2.0, 4.0, 6.0]
        assert all(math.isclose(r, e, abs_tol=TestData.ABS_TOL) for r, e in zip(result, expected))

    @pytest.mark.asyncio
    async def test_scalar_multiply_example(self):
        """Test scalar multiplication with documented example."""
        result = await scalar_multiply([1, 2, 3], 2)
        expected = [2.0, 4.0, 6.0]
        assert all(math.isclose(r, e, abs_tol=TestData.ABS_TOL) for r, e in zip(result, expected))

    @pytest.mark.asyncio
    async def test_scalar_multiply_zero(self):
        """Test multiplication by zero."""
        result = await scalar_multiply([1, 2, 3], 0)
        expected = [0.0, 0.0, 0.0]
        assert all(math.isclose(r, e, abs_tol=TestData.ABS_TOL) for r, e in zip(result, expected))

    @pytest.mark.asyncio
    async def test_scalar_multiply_one(self):
        """Test multiplication by one."""
        vector = [1.5, 2.5, 3.5]
        result = await scalar_multiply(vector, 1)
        assert all(math.isclose(r, v, abs_tol=TestData.ABS_TOL) for r, v in zip(result, vector))

    @pytest.mark.asyncio
    async def test_scalar_multiply_negative(self):
        """Test multiplication by negative scalar."""
        result = await scalar_multiply([1, 2, 3], -2)
        expected = [-2.0, -4.0, -6.0]
        assert all(math.isclose(r, e, abs_tol=TestData.ABS_TOL) for r, e in zip(result, expected))

    @pytest.mark.asyncio
    async def test_scalar_multiply_fraction(self):
        """Test multiplication by fractional scalar."""
        result = await scalar_multiply([2, 4, 6], 0.5)
        expected = [1.0, 2.0, 3.0]
        assert all(math.isclose(r, e, abs_tol=TestData.ABS_TOL) for r, e in zip(result, expected))

    @pytest.mark.asyncio
    async def test_scalar_multiply_empty_vector(self):
        """Test that empty vector raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            await scalar_multiply([], 2)

    @pytest.mark.asyncio
    async def test_scalar_multiply_return_type(self):
        """Test that result elements are floats."""
        result = await scalar_multiply([1, 2, 3], 2)
        assert all(isinstance(x, float) for x in result)

    @pytest.mark.asyncio
    async def test_scalar_multiply_large_vector(self):
        """Test scalar multiplication with large vector."""
        large_v = list(range(1, 1001))
        result = await scalar_multiply(large_v, 2.5)
        assert len(result) == 1000
        assert math.isclose(result[0], 2.5, abs_tol=TestData.ABS_TOL)

    @pytest.mark.asyncio
    async def test_scalar_multiply_preserves_length(self):
        """Test that scalar multiplication preserves vector length."""
        vector = [1, 2, 3, 4, 5]
        result = await scalar_multiply(vector, 3)
        assert len(result) == len(vector)


# ============================================================================
# TEST VECTOR ADD
# ============================================================================


class TestVectorAdd:
    """Test vector addition function."""

    @pytest.mark.asyncio
    async def test_basic_vector_add(self):
        """Test basic vector addition."""
        result = await vector_add([1, 2, 3], [4, 5, 6])
        expected = [5.0, 7.0, 9.0]
        assert all(math.isclose(r, e, abs_tol=TestData.ABS_TOL) for r, e in zip(result, expected))

    @pytest.mark.asyncio
    async def test_vector_add_example(self):
        """Test vector addition with documented example."""
        result = await vector_add([1, 2, 3], [4, 5, 6])
        expected = [5.0, 7.0, 9.0]
        assert all(math.isclose(r, e, abs_tol=TestData.ABS_TOL) for r, e in zip(result, expected))

    @pytest.mark.asyncio
    async def test_vector_add_zero(self):
        """Test adding zero vector."""
        vector = [1, 2, 3]
        result = await vector_add(vector, [0, 0, 0])
        assert all(math.isclose(r, v, abs_tol=TestData.ABS_TOL) for r, v in zip(result, vector))

    @pytest.mark.asyncio
    async def test_vector_add_negative(self):
        """Test adding negative values."""
        result = await vector_add([1, 2, 3], [-1, -2, -3])
        expected = [0.0, 0.0, 0.0]
        assert all(math.isclose(r, e, abs_tol=TestData.ABS_TOL) for r, e in zip(result, expected))

    @pytest.mark.asyncio
    async def test_vector_add_commutative(self):
        """Test that addition is commutative."""
        v1 = [1, 2, 3]
        v2 = [4, 5, 6]

        result1 = await vector_add(v1, v2)
        result2 = await vector_add(v2, v1)

        assert all(
            math.isclose(r1, r2, abs_tol=TestData.ABS_TOL) for r1, r2 in zip(result1, result2)
        )

    @pytest.mark.asyncio
    async def test_vector_add_dimension_mismatch(self):
        """Test that dimension mismatch raises ValueError."""
        with pytest.raises(ValueError, match="same dimension"):
            await vector_add([1, 2], [1, 2, 3])

    @pytest.mark.asyncio
    async def test_vector_add_empty_vectors(self):
        """Test that empty vectors raise ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            await vector_add([], [])

    @pytest.mark.asyncio
    async def test_vector_add_return_type(self):
        """Test that result elements are floats."""
        result = await vector_add([1, 2], [3, 4])
        assert all(isinstance(x, float) for x in result)

    @pytest.mark.asyncio
    async def test_vector_add_large_vectors(self):
        """Test addition with large vectors."""
        large_v1 = list(range(1, 1001))
        large_v2 = list(range(1001, 2001))
        result = await vector_add(large_v1, large_v2)
        assert len(result) == 1000


# ============================================================================
# TEST VECTOR SUBTRACT
# ============================================================================


class TestVectorSubtract:
    """Test vector subtraction function."""

    @pytest.mark.asyncio
    async def test_basic_vector_subtract(self):
        """Test basic vector subtraction."""
        result = await vector_subtract([5, 7, 9], [1, 2, 3])
        expected = [4.0, 5.0, 6.0]
        assert all(math.isclose(r, e, abs_tol=TestData.ABS_TOL) for r, e in zip(result, expected))

    @pytest.mark.asyncio
    async def test_vector_subtract_example(self):
        """Test vector subtraction with documented example."""
        result = await vector_subtract([5, 7, 9], [1, 2, 3])
        expected = [4.0, 5.0, 6.0]
        assert all(math.isclose(r, e, abs_tol=TestData.ABS_TOL) for r, e in zip(result, expected))

    @pytest.mark.asyncio
    async def test_vector_subtract_zero(self):
        """Test subtracting zero vector."""
        vector = [1, 2, 3]
        result = await vector_subtract(vector, [0, 0, 0])
        assert all(math.isclose(r, v, abs_tol=TestData.ABS_TOL) for r, v in zip(result, vector))

    @pytest.mark.asyncio
    async def test_vector_subtract_self(self):
        """Test subtracting vector from itself."""
        vector = [1, 2, 3]
        result = await vector_subtract(vector, vector)
        expected = [0.0, 0.0, 0.0]
        assert all(math.isclose(r, e, abs_tol=TestData.ABS_TOL) for r, e in zip(result, expected))

    @pytest.mark.asyncio
    async def test_vector_subtract_negative(self):
        """Test subtraction with negative values."""
        result = await vector_subtract([1, 2, 3], [-1, -2, -3])
        expected = [2.0, 4.0, 6.0]
        assert all(math.isclose(r, e, abs_tol=TestData.ABS_TOL) for r, e in zip(result, expected))

    @pytest.mark.asyncio
    async def test_vector_subtract_dimension_mismatch(self):
        """Test that dimension mismatch raises ValueError."""
        with pytest.raises(ValueError, match="same dimension"):
            await vector_subtract([1, 2], [1, 2, 3])

    @pytest.mark.asyncio
    async def test_vector_subtract_empty_vectors(self):
        """Test that empty vectors raise ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            await vector_subtract([], [])

    @pytest.mark.asyncio
    async def test_vector_subtract_return_type(self):
        """Test that result elements are floats."""
        result = await vector_subtract([5, 4], [3, 2])
        assert all(isinstance(x, float) for x in result)

    @pytest.mark.asyncio
    async def test_vector_subtract_large_vectors(self):
        """Test subtraction with large vectors."""
        large_v1 = list(range(2001, 3001))
        large_v2 = list(range(1, 1001))
        result = await vector_subtract(large_v1, large_v2)
        assert len(result) == 1000


# ============================================================================
# TEST ELEMENT-WISE MULTIPLY
# ============================================================================


class TestElementWiseMultiply:
    """Test element-wise multiplication function."""

    @pytest.mark.asyncio
    async def test_basic_element_wise_multiply(self):
        """Test basic element-wise multiplication."""
        result = await element_wise_multiply([1, 2, 3], [4, 5, 6])
        expected = [4.0, 10.0, 18.0]
        assert all(math.isclose(r, e, abs_tol=TestData.ABS_TOL) for r, e in zip(result, expected))

    @pytest.mark.asyncio
    async def test_element_wise_multiply_example(self):
        """Test element-wise multiply with documented example."""
        result = await element_wise_multiply([1, 2, 3], [4, 5, 6])
        expected = [4.0, 10.0, 18.0]
        assert all(math.isclose(r, e, abs_tol=TestData.ABS_TOL) for r, e in zip(result, expected))

    @pytest.mark.asyncio
    async def test_element_wise_multiply_zero(self):
        """Test multiplication with zero vector."""
        result = await element_wise_multiply([1, 2, 3], [0, 0, 0])
        expected = [0.0, 0.0, 0.0]
        assert all(math.isclose(r, e, abs_tol=TestData.ABS_TOL) for r, e in zip(result, expected))

    @pytest.mark.asyncio
    async def test_element_wise_multiply_ones(self):
        """Test multiplication with ones vector."""
        vector = [2, 3, 4]
        result = await element_wise_multiply(vector, [1, 1, 1])
        assert all(math.isclose(r, v, abs_tol=TestData.ABS_TOL) for r, v in zip(result, vector))

    @pytest.mark.asyncio
    async def test_element_wise_multiply_negative(self):
        """Test multiplication with negative values."""
        result = await element_wise_multiply([1, -2, 3], [-4, 5, -6])
        expected = [-4.0, -10.0, -18.0]
        assert all(math.isclose(r, e, abs_tol=TestData.ABS_TOL) for r, e in zip(result, expected))

    @pytest.mark.asyncio
    async def test_element_wise_multiply_commutative(self):
        """Test that element-wise multiplication is commutative."""
        v1 = [1, 2, 3]
        v2 = [4, 5, 6]

        result1 = await element_wise_multiply(v1, v2)
        result2 = await element_wise_multiply(v2, v1)

        assert all(
            math.isclose(r1, r2, abs_tol=TestData.ABS_TOL) for r1, r2 in zip(result1, result2)
        )

    @pytest.mark.asyncio
    async def test_element_wise_multiply_dimension_mismatch(self):
        """Test that dimension mismatch raises ValueError."""
        with pytest.raises(ValueError, match="same dimension"):
            await element_wise_multiply([1, 2], [1, 2, 3])

    @pytest.mark.asyncio
    async def test_element_wise_multiply_empty_vectors(self):
        """Test that empty vectors raise ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            await element_wise_multiply([], [])

    @pytest.mark.asyncio
    async def test_element_wise_multiply_return_type(self):
        """Test that result elements are floats."""
        result = await element_wise_multiply([1, 2], [3, 4])
        assert all(isinstance(x, float) for x in result)

    @pytest.mark.asyncio
    async def test_element_wise_multiply_large_vectors(self):
        """Test multiplication with large vectors."""
        large_v = list(range(1, 1001))
        result = await element_wise_multiply(large_v, large_v)
        assert len(result) == 1000


# ============================================================================
# TEST ELEMENT-WISE DIVIDE
# ============================================================================


class TestElementWiseDivide:
    """Test element-wise division function."""

    @pytest.mark.asyncio
    async def test_basic_element_wise_divide(self):
        """Test basic element-wise division."""
        result = await element_wise_divide([10, 20, 30], [2, 4, 5])
        expected = [5.0, 5.0, 6.0]
        assert all(math.isclose(r, e, abs_tol=TestData.ABS_TOL) for r, e in zip(result, expected))

    @pytest.mark.asyncio
    async def test_element_wise_divide_example(self):
        """Test element-wise divide with documented example."""
        result = await element_wise_divide([10, 20, 30], [2, 4, 5])
        expected = [5.0, 5.0, 6.0]
        assert all(math.isclose(r, e, abs_tol=TestData.ABS_TOL) for r, e in zip(result, expected))

    @pytest.mark.asyncio
    async def test_element_wise_divide_by_one(self):
        """Test division by ones vector."""
        vector = [2, 3, 4]
        result = await element_wise_divide(vector, [1, 1, 1])
        assert all(math.isclose(r, v, abs_tol=TestData.ABS_TOL) for r, v in zip(result, vector))

    @pytest.mark.asyncio
    async def test_element_wise_divide_self(self):
        """Test dividing vector by itself."""
        vector = [2, 3, 4]
        result = await element_wise_divide(vector, vector)
        expected = [1.0, 1.0, 1.0]
        assert all(math.isclose(r, e, abs_tol=TestData.ABS_TOL) for r, e in zip(result, expected))

    @pytest.mark.asyncio
    async def test_element_wise_divide_negative(self):
        """Test division with negative values."""
        result = await element_wise_divide([10, -20, 30], [-2, 4, -5])
        expected = [-5.0, -5.0, -6.0]
        assert all(math.isclose(r, e, abs_tol=TestData.ABS_TOL) for r, e in zip(result, expected))

    @pytest.mark.asyncio
    async def test_element_wise_divide_zero_error(self):
        """Test that division by zero raises ZeroDivisionError."""
        with pytest.raises(ZeroDivisionError, match="divide by zero"):
            await element_wise_divide([1, 2, 3], [1, 0, 3])

    @pytest.mark.asyncio
    async def test_element_wise_divide_dimension_mismatch(self):
        """Test that dimension mismatch raises ValueError."""
        with pytest.raises(ValueError, match="same dimension"):
            await element_wise_divide([1, 2], [1, 2, 3])

    @pytest.mark.asyncio
    async def test_element_wise_divide_empty_vectors(self):
        """Test that empty vectors raise ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            await element_wise_divide([], [])

    @pytest.mark.asyncio
    async def test_element_wise_divide_return_type(self):
        """Test that result elements are floats."""
        result = await element_wise_divide([10, 20], [2, 4])
        assert all(isinstance(x, float) for x in result)

    @pytest.mark.asyncio
    async def test_element_wise_divide_large_vectors(self):
        """Test division with large vectors."""
        large_v1 = list(range(1, 1001))
        large_v2 = [2] * 1000
        result = await element_wise_divide(large_v1, large_v2)
        assert len(result) == 1000


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


class TestIntegration:
    """Integration tests combining multiple operations."""

    @pytest.mark.asyncio
    async def test_combined_operations(self):
        """Test combining multiple vector operations."""
        v1 = [1, 2, 3]
        v2 = [4, 5, 6]

        # (v1 + v2) * 2
        sum_vec = await vector_add(v1, v2)
        result = await scalar_multiply(sum_vec, 2)

        expected = [10.0, 14.0, 18.0]
        assert all(math.isclose(r, e, abs_tol=TestData.ABS_TOL) for r, e in zip(result, expected))

    @pytest.mark.asyncio
    async def test_dot_product_with_add(self):
        """Test dot product with vector addition."""
        v1 = [1, 2]
        v2 = [3, 4]
        v3 = [5, 6]

        sum_vec = await vector_add(v2, v3)
        result = await dot_product(v1, sum_vec)

        # v1 · (v2 + v3) = v1 · v2 + v1 · v3 (distributive property)
        dot1 = await dot_product(v1, v2)
        dot2 = await dot_product(v1, v3)
        expected = dot1 + dot2

        assert math.isclose(result, expected, abs_tol=TestData.ABS_TOL)

    @pytest.mark.asyncio
    async def test_concurrent_operations(self):
        """Test concurrent execution of multiple operations."""
        v1 = [1, 2, 3]
        v2 = [4, 5, 6]

        # Execute operations concurrently
        results = await asyncio.gather(
            dot_product(v1, v2),
            vector_add(v1, v2),
            vector_subtract(v2, v1),
            element_wise_multiply(v1, v2),
            scalar_multiply(v1, 2),
        )

        assert len(results) == 5
        assert isinstance(results[0], float)  # dot product
        assert isinstance(results[1], list)  # vector add
        assert isinstance(results[2], list)  # vector subtract
        assert isinstance(results[3], list)  # element-wise multiply
        assert isinstance(results[4], list)  # scalar multiply


# ============================================================================
# PERFORMANCE TESTS
# ============================================================================


class TestPerformance:
    """Performance and stress tests."""

    @pytest.mark.asyncio
    async def test_very_large_vectors(self):
        """Test operations on very large vectors."""
        size = 5000
        v1 = list(range(size))
        v2 = list(range(size, 2 * size))

        result = await dot_product(v1, v2)
        assert isinstance(result, float)
        assert result > 0

    @pytest.mark.asyncio
    async def test_many_small_operations(self):
        """Test many small operations."""
        operations = []
        for i in range(100):
            v1 = [i, i + 1, i + 2]
            v2 = [i + 3, i + 4, i + 5]
            operations.append(dot_product(v1, v2))

        results = await asyncio.gather(*operations)
        assert len(results) == 100
        assert all(isinstance(r, float) for r in results)
