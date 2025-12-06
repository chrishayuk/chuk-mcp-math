#!/usr/bin/env python3
"""
Comprehensive unit tests for vector projections module.

Tests all functions in chuk_mcp_math.linear_algebra.vectors.projections:
- vector_projection
- vector_rejection
- scalar_projection
- orthogonalize
- gram_schmidt
"""

import pytest
import math
import asyncio

from chuk_mcp_math.linear_algebra.vectors.projections import (
    vector_projection,
    vector_rejection,
    scalar_projection,
    orthogonalize,
    gram_schmidt,
)
from chuk_mcp_math.linear_algebra.vectors.norms import vector_norm
from chuk_mcp_math.linear_algebra.vectors.operations import dot_product


# ============================================================================
# TEST DATA AND FIXTURES
# ============================================================================


class TestData:
    """Test data and constants."""

    ABS_TOL = 1e-10
    REL_TOL = 1e-9

    # Standard test vectors
    VECTORS = {
        "unit_x": [1, 0, 0],
        "unit_y": [0, 1, 0],
        "unit_z": [0, 0, 1],
        "simple_2d": [3, 4],
        "simple_3d": [1, 2, 3],
    }


@pytest.fixture
def standard_vectors():
    """Provide standard test vectors."""
    return {
        "v1": [3, 4],
        "v2": [1, 0],
        "v3": [1, 2, 3],
        "v4": [4, 5, 6],
    }


# ============================================================================
# TEST VECTOR PROJECTION
# ============================================================================


class TestVectorProjection:
    """Test vector projection function."""

    @pytest.mark.asyncio
    async def test_basic_projection(self):
        """Test basic vector projection."""
        # Project [3, 4] onto x-axis [1, 0]
        result = await vector_projection([3, 4], [1, 0])
        expected = [3.0, 0.0]
        assert all(math.isclose(r, e, abs_tol=TestData.ABS_TOL) for r, e in zip(result, expected))

    @pytest.mark.asyncio
    async def test_projection_example(self):
        """Test projection with documented example."""
        result = await vector_projection([3, 4], [1, 0])
        expected = [3.0, 0.0]
        assert all(math.isclose(r, e, abs_tol=TestData.ABS_TOL) for r, e in zip(result, expected))

    @pytest.mark.asyncio
    async def test_projection_onto_self(self):
        """Test projecting vector onto itself."""
        vector = [3, 4, 5]
        result = await vector_projection(vector, vector)
        # Projection onto self should equal the vector
        assert all(math.isclose(r, v, abs_tol=TestData.ABS_TOL) for r, v in zip(result, vector))

    @pytest.mark.asyncio
    async def test_projection_orthogonal(self):
        """Test projection of orthogonal vectors."""
        # Project y-axis onto x-axis
        result = await vector_projection([0, 1], [1, 0])
        expected = [0.0, 0.0]
        assert all(math.isclose(r, e, abs_tol=TestData.ABS_TOL) for r, e in zip(result, expected))

    @pytest.mark.asyncio
    async def test_projection_parallel(self):
        """Test projection of parallel vectors."""
        # Project [2, 4] onto [1, 2]
        result = await vector_projection([2, 4], [1, 2])
        # Should be proportional to [1, 2]
        norm = await vector_norm(result)
        direction = [r / norm if norm > 0 else 0 for r in result]
        expected_dir = [1 / math.sqrt(5), 2 / math.sqrt(5)]
        assert all(
            math.isclose(d, e, abs_tol=TestData.ABS_TOL) for d, e in zip(direction, expected_dir)
        )

    @pytest.mark.asyncio
    async def test_projection_onto_unit_vector(self):
        """Test projection onto unit vector."""
        # Project onto x-axis unit vector
        result = await vector_projection([3, 4, 5], [1, 0, 0])
        expected = [3.0, 0.0, 0.0]
        assert all(math.isclose(r, e, abs_tol=TestData.ABS_TOL) for r, e in zip(result, expected))

    @pytest.mark.asyncio
    async def test_projection_3d(self):
        """Test 3D vector projection."""
        result = await vector_projection([1, 2, 3], [1, 1, 1])
        # Projection onto [1, 1, 1]
        # dot([1,2,3], [1,1,1]) = 6
        # dot([1,1,1], [1,1,1]) = 3
        # scalar = 6/3 = 2
        # result = 2 * [1, 1, 1] = [2, 2, 2]
        expected = [2.0, 2.0, 2.0]
        assert all(math.isclose(r, e, abs_tol=TestData.ABS_TOL) for r, e in zip(result, expected))

    @pytest.mark.asyncio
    async def test_projection_zero_vector_error(self):
        """Test that projecting onto zero vector raises error."""
        with pytest.raises(ValueError, match="zero vector"):
            await vector_projection([1, 2, 3], [0, 0, 0])

    @pytest.mark.asyncio
    async def test_projection_dimension_mismatch(self):
        """Test that dimension mismatch raises error."""
        with pytest.raises(ValueError, match="same dimension"):
            await vector_projection([1, 2], [1, 2, 3])

    @pytest.mark.asyncio
    async def test_projection_return_type(self):
        """Test that result elements are floats."""
        result = await vector_projection([1, 2], [3, 4])
        assert all(isinstance(x, float) for x in result)


# ============================================================================
# TEST VECTOR REJECTION
# ============================================================================


class TestVectorRejection:
    """Test vector rejection function."""

    @pytest.mark.asyncio
    async def test_basic_rejection(self):
        """Test basic vector rejection."""
        # Reject [3, 4] from x-axis [1, 0]
        result = await vector_rejection([3, 4], [1, 0])
        expected = [0.0, 4.0]
        assert all(math.isclose(r, e, abs_tol=TestData.ABS_TOL) for r, e in zip(result, expected))

    @pytest.mark.asyncio
    async def test_rejection_example(self):
        """Test rejection with documented example."""
        result = await vector_rejection([3, 4], [1, 0])
        expected = [0.0, 4.0]
        assert all(math.isclose(r, e, abs_tol=TestData.ABS_TOL) for r, e in zip(result, expected))

    @pytest.mark.asyncio
    async def test_rejection_from_self(self):
        """Test rejecting vector from itself."""
        vector = [3, 4, 5]
        result = await vector_rejection(vector, vector)
        # Rejection from self should be zero
        expected = [0.0, 0.0, 0.0]
        assert all(math.isclose(r, e, abs_tol=TestData.ABS_TOL) for r, e in zip(result, expected))

    @pytest.mark.asyncio
    async def test_rejection_orthogonal(self):
        """Test rejection of orthogonal vectors."""
        # Reject y-axis from x-axis
        result = await vector_rejection([0, 1], [1, 0])
        # Should equal the original vector since they're orthogonal
        expected = [0.0, 1.0]
        assert all(math.isclose(r, e, abs_tol=TestData.ABS_TOL) for r, e in zip(result, expected))

    @pytest.mark.asyncio
    async def test_rejection_parallel(self):
        """Test rejection of parallel vectors."""
        # Reject [2, 4] from [1, 2]
        result = await vector_rejection([2, 4], [1, 2])
        # Should be zero since they're parallel
        expected = [0.0, 0.0]
        assert all(math.isclose(r, e, abs_tol=TestData.ABS_TOL) for r, e in zip(result, expected))

    @pytest.mark.asyncio
    async def test_rejection_3d(self):
        """Test 3D vector rejection."""
        # Reject [1, 2, 3] from [1, 0, 0]
        result = await vector_rejection([1, 2, 3], [1, 0, 0])
        expected = [0.0, 2.0, 3.0]
        assert all(math.isclose(r, e, abs_tol=TestData.ABS_TOL) for r, e in zip(result, expected))

    @pytest.mark.asyncio
    async def test_projection_rejection_sum(self):
        """Test that projection + rejection = original vector."""
        v1 = [3, 4]
        v2 = [1, 1]

        proj = await vector_projection(v1, v2)
        rej = await vector_rejection(v1, v2)

        # proj + rej should equal v1
        result = [p + r for p, r in zip(proj, rej)]
        assert all(math.isclose(r, v, abs_tol=TestData.ABS_TOL) for r, v in zip(result, v1))

    @pytest.mark.asyncio
    async def test_rejection_return_type(self):
        """Test that result elements are floats."""
        result = await vector_rejection([1, 2], [3, 4])
        assert all(isinstance(x, float) for x in result)


# ============================================================================
# TEST SCALAR PROJECTION
# ============================================================================


class TestScalarProjection:
    """Test scalar projection function."""

    @pytest.mark.asyncio
    async def test_basic_scalar_projection(self):
        """Test basic scalar projection."""
        # Scalar projection of [3, 4] onto x-axis [1, 0]
        result = await scalar_projection([3, 4], [1, 0])
        expected = 3.0
        assert math.isclose(result, expected, abs_tol=TestData.ABS_TOL)

    @pytest.mark.asyncio
    async def test_scalar_projection_example(self):
        """Test scalar projection with documented example."""
        result = await scalar_projection([3, 4], [1, 0])
        expected = 3.0
        assert math.isclose(result, expected, abs_tol=TestData.ABS_TOL)

    @pytest.mark.asyncio
    async def test_scalar_projection_onto_unit_vector(self):
        """Test scalar projection onto unit vector."""
        # Should equal dot product when projecting onto unit vector
        v1 = [3, 4, 5]
        unit_x = [1, 0, 0]

        result = await scalar_projection(v1, unit_x)
        dot = await dot_product(v1, unit_x)

        assert math.isclose(result, dot, abs_tol=TestData.ABS_TOL)

    @pytest.mark.asyncio
    async def test_scalar_projection_orthogonal(self):
        """Test scalar projection of orthogonal vectors."""
        result = await scalar_projection([0, 1], [1, 0])
        expected = 0.0
        assert math.isclose(result, expected, abs_tol=TestData.ABS_TOL)

    @pytest.mark.asyncio
    async def test_scalar_projection_parallel(self):
        """Test scalar projection of parallel vectors."""
        # Project [2, 4] onto [1, 2]
        v1 = [2, 4]
        v2 = [1, 2]
        result = await scalar_projection(v1, v2)

        # Should be length of v1 since they're parallel
        v1_norm = await vector_norm(v1)
        assert math.isclose(abs(result), v1_norm, abs_tol=TestData.ABS_TOL)

    @pytest.mark.asyncio
    async def test_scalar_projection_negative(self):
        """Test scalar projection can be negative."""
        # Project [-3, 0] onto [1, 0]
        result = await scalar_projection([-3, 0], [1, 0])
        assert result < 0
        assert math.isclose(result, -3.0, abs_tol=TestData.ABS_TOL)

    @pytest.mark.asyncio
    async def test_scalar_projection_3d(self):
        """Test 3D scalar projection."""
        result = await scalar_projection([1, 2, 3], [1, 1, 1])
        # dot([1,2,3], [1,1,1]) = 6
        # norm([1,1,1]) = sqrt(3)
        # result = 6 / sqrt(3) = 2*sqrt(3)
        expected = 6.0 / math.sqrt(3)
        assert math.isclose(result, expected, abs_tol=TestData.ABS_TOL)

    @pytest.mark.asyncio
    async def test_scalar_projection_zero_vector_error(self):
        """Test that projecting onto zero vector raises error."""
        with pytest.raises(ValueError, match="zero vector"):
            await scalar_projection([1, 2, 3], [0, 0, 0])

    @pytest.mark.asyncio
    async def test_scalar_projection_return_type(self):
        """Test that result is float."""
        result = await scalar_projection([1, 2], [3, 4])
        assert isinstance(result, float)


# ============================================================================
# TEST ORTHOGONALIZE
# ============================================================================


class TestOrthogonalize:
    """Test orthogonalize function."""

    @pytest.mark.asyncio
    async def test_basic_orthogonalize(self):
        """Test basic orthogonalization."""
        # Orthogonalize [1, 1] against x-axis
        result = await orthogonalize([1, 1], [[1, 0]])
        expected = [0.0, 1.0]
        assert all(math.isclose(r, e, abs_tol=TestData.ABS_TOL) for r, e in zip(result, expected))

    @pytest.mark.asyncio
    async def test_orthogonalize_example(self):
        """Test orthogonalize with documented example."""
        result = await orthogonalize([1, 1], [[1, 0]])
        expected = [0.0, 1.0]
        assert all(math.isclose(r, e, abs_tol=TestData.ABS_TOL) for r, e in zip(result, expected))

    @pytest.mark.asyncio
    async def test_orthogonalize_already_orthogonal(self):
        """Test orthogonalizing already orthogonal vector."""
        result = await orthogonalize([0, 1], [[1, 0]])
        # Should remain unchanged
        expected = [0.0, 1.0]
        assert all(math.isclose(r, e, abs_tol=TestData.ABS_TOL) for r, e in zip(result, expected))

    @pytest.mark.asyncio
    async def test_orthogonalize_multiple_basis(self):
        """Test orthogonalizing against multiple basis vectors."""
        # Orthogonalize [1, 1, 1] against x and y axes
        result = await orthogonalize([1, 1, 1], [[1, 0, 0], [0, 1, 0]])
        expected = [0.0, 0.0, 1.0]
        assert all(math.isclose(r, e, abs_tol=TestData.ABS_TOL) for r, e in zip(result, expected))

    @pytest.mark.asyncio
    async def test_orthogonalize_3d(self):
        """Test 3D orthogonalization."""
        result = await orthogonalize([1, 2, 3], [[1, 0, 0]])
        expected = [0.0, 2.0, 3.0]
        assert all(math.isclose(r, e, abs_tol=TestData.ABS_TOL) for r, e in zip(result, expected))

    @pytest.mark.asyncio
    async def test_orthogonalize_result_orthogonal(self):
        """Test that result is orthogonal to basis vectors."""
        vector = [3, 4, 5]
        basis = [[1, 0, 0], [0, 1, 0]]

        result = await orthogonalize(vector, basis)

        # Check orthogonality
        for b in basis:
            dot = await dot_product(result, b)
            assert math.isclose(dot, 0.0, abs_tol=TestData.ABS_TOL)

    @pytest.mark.asyncio
    async def test_orthogonalize_empty_basis(self):
        """Test orthogonalize with empty basis list."""
        vector = [1, 2, 3]
        result = await orthogonalize(vector, [])
        # Should return copy of original vector
        assert all(math.isclose(r, v, abs_tol=TestData.ABS_TOL) for r, v in zip(result, vector))

    @pytest.mark.asyncio
    async def test_orthogonalize_return_type(self):
        """Test that result elements are floats."""
        result = await orthogonalize([1, 2], [[1, 0]])
        assert all(isinstance(x, float) for x in result)


# ============================================================================
# TEST GRAM-SCHMIDT
# ============================================================================


class TestGramSchmidt:
    """Test Gram-Schmidt orthogonalization."""

    @pytest.mark.asyncio
    async def test_basic_gram_schmidt(self):
        """Test basic Gram-Schmidt process."""
        vectors = [[1, 0], [1, 1]]
        result = await gram_schmidt(vectors, normalize=True)

        # Should produce orthonormal basis
        assert len(result) == 2

        # Check orthogonality
        dot = await dot_product(result[0], result[1])
        assert math.isclose(dot, 0.0, abs_tol=TestData.ABS_TOL)

        # Check normalization
        norm0 = await vector_norm(result[0])
        norm1 = await vector_norm(result[1])
        assert math.isclose(norm0, 1.0, abs_tol=TestData.ABS_TOL)
        assert math.isclose(norm1, 1.0, abs_tol=TestData.ABS_TOL)

    @pytest.mark.asyncio
    async def test_gram_schmidt_example(self):
        """Test Gram-Schmidt with documented example."""
        result = await gram_schmidt([[1, 0], [1, 1]])
        expected = [[1.0, 0.0], [0.0, 1.0]]

        # Check first vector
        assert all(
            math.isclose(r, e, abs_tol=TestData.ABS_TOL) for r, e in zip(result[0], expected[0])
        )
        # Check second vector
        assert all(
            math.isclose(r, e, abs_tol=TestData.ABS_TOL) for r, e in zip(result[1], expected[1])
        )

    @pytest.mark.asyncio
    async def test_gram_schmidt_3d(self):
        """Test Gram-Schmidt in 3D."""
        vectors = [[1, 0, 0], [1, 1, 0], [1, 1, 1]]
        result = await gram_schmidt(vectors, normalize=True)

        assert len(result) == 3

        # Check all pairs are orthogonal
        for i in range(len(result)):
            for j in range(i + 1, len(result)):
                dot = await dot_product(result[i], result[j])
                assert math.isclose(dot, 0.0, abs_tol=TestData.ABS_TOL)

        # Check all are normalized
        for vec in result:
            norm = await vector_norm(vec)
            assert math.isclose(norm, 1.0, abs_tol=TestData.ABS_TOL)

    @pytest.mark.asyncio
    async def test_gram_schmidt_without_normalization(self):
        """Test Gram-Schmidt without normalization."""
        vectors = [[1, 0], [1, 1]]
        result = await gram_schmidt(vectors, normalize=False)

        # Should be orthogonal but not normalized
        dot = await dot_product(result[0], result[1])
        assert math.isclose(dot, 0.0, abs_tol=TestData.ABS_TOL)

        # First vector unchanged
        assert all(
            math.isclose(r, v, abs_tol=TestData.ABS_TOL) for r, v in zip(result[0], vectors[0])
        )

    @pytest.mark.asyncio
    async def test_gram_schmidt_already_orthogonal(self):
        """Test Gram-Schmidt on already orthogonal vectors."""
        vectors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        result = await gram_schmidt(vectors, normalize=True)

        # Should remain essentially the same
        for i in range(3):
            # Allow for sign flips
            matches = all(
                math.isclose(abs(r), abs(v), abs_tol=TestData.ABS_TOL)
                for r, v in zip(result[i], vectors[i])
            )
            assert matches

    @pytest.mark.asyncio
    async def test_gram_schmidt_linearly_dependent(self):
        """Test that linearly dependent vectors raise error."""
        # Two identical vectors
        vectors = [[1, 0], [1, 0]]

        with pytest.raises(ValueError, match="linearly dependent"):
            await gram_schmidt(vectors)

    @pytest.mark.asyncio
    async def test_gram_schmidt_proportional_vectors(self):
        """Test that proportional vectors raise error."""
        # Vectors that are scalar multiples
        vectors = [[1, 2], [2, 4]]

        with pytest.raises(ValueError, match="linearly dependent"):
            await gram_schmidt(vectors)

    @pytest.mark.asyncio
    async def test_gram_schmidt_empty_input(self):
        """Test that empty input raises error."""
        with pytest.raises(ValueError, match="cannot be empty"):
            await gram_schmidt([])

    @pytest.mark.asyncio
    async def test_gram_schmidt_dimension_mismatch(self):
        """Test that vectors with different dimensions raise error."""
        vectors = [[1, 2], [1, 2, 3]]

        with pytest.raises(ValueError, match="same dimension"):
            await gram_schmidt(vectors)

    @pytest.mark.asyncio
    async def test_gram_schmidt_single_vector(self):
        """Test Gram-Schmidt with single vector."""
        vectors = [[3, 4]]
        result = await gram_schmidt(vectors, normalize=True)

        assert len(result) == 1
        norm = await vector_norm(result[0])
        assert math.isclose(norm, 1.0, abs_tol=TestData.ABS_TOL)

    @pytest.mark.asyncio
    async def test_gram_schmidt_return_type(self):
        """Test that result elements are floats."""
        vectors = [[1, 2], [3, 4]]
        result = await gram_schmidt(vectors)

        for vec in result:
            assert all(isinstance(x, float) for x in vec)

    @pytest.mark.asyncio
    async def test_gram_schmidt_large_set(self):
        """Test Gram-Schmidt with larger set of vectors."""
        # Create 5 random-ish linearly independent vectors
        vectors = [
            [1, 0, 0, 0, 0],
            [1, 1, 0, 0, 0],
            [1, 1, 1, 0, 0],
            [1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1],
        ]

        result = await gram_schmidt(vectors, normalize=True)

        assert len(result) == 5

        # Check all pairs are orthogonal
        for i in range(len(result)):
            for j in range(i + 1, len(result)):
                dot = await dot_product(result[i], result[j])
                assert math.isclose(dot, 0.0, abs_tol=TestData.ABS_TOL)


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


class TestIntegration:
    """Integration tests combining multiple projection operations."""

    @pytest.mark.asyncio
    async def test_projection_rejection_orthogonality(self):
        """Test that projection and rejection are orthogonal."""
        v1 = [3, 4, 5]
        v2 = [1, 1, 1]

        proj = await vector_projection(v1, v2)
        rej = await vector_rejection(v1, v2)

        # Projection and rejection should be orthogonal
        dot = await dot_product(proj, rej)
        assert math.isclose(dot, 0.0, abs_tol=TestData.ABS_TOL)

    @pytest.mark.asyncio
    async def test_scalar_and_vector_projection_relationship(self):
        """Test relationship between scalar and vector projection."""
        v1 = [3, 4, 5]
        v2 = [1, 2, 3]

        scalar_proj = await scalar_projection(v1, v2)
        vector_proj = await vector_projection(v1, v2)

        # |vector_proj| should equal |scalar_proj|
        vec_norm = await vector_norm(vector_proj)
        assert math.isclose(vec_norm, abs(scalar_proj), abs_tol=TestData.ABS_TOL)

    @pytest.mark.asyncio
    async def test_gram_schmidt_produces_orthonormal_basis(self):
        """Test that Gram-Schmidt produces a proper orthonormal basis."""
        vectors = [[1, 1, 0], [1, 0, 1], [0, 1, 1]]
        result = await gram_schmidt(vectors, normalize=True)

        # Check orthonormality
        for i in range(len(result)):
            # Check normalized
            norm = await vector_norm(result[i])
            assert math.isclose(norm, 1.0, abs_tol=TestData.ABS_TOL)

            # Check orthogonal to others
            for j in range(i + 1, len(result)):
                dot = await dot_product(result[i], result[j])
                assert math.isclose(dot, 0.0, abs_tol=TestData.ABS_TOL)

    @pytest.mark.asyncio
    async def test_orthogonalize_vs_gram_schmidt(self):
        """Test that orthogonalize and Gram-Schmidt give consistent results."""
        vectors = [[1, 0], [1, 1]]

        # Use Gram-Schmidt
        gs_result = await gram_schmidt(vectors, normalize=False)

        # Use orthogonalize
        ortho_result = await orthogonalize(vectors[1], [vectors[0]])

        # Second vector from GS should match orthogonalize result
        assert all(
            math.isclose(g, o, abs_tol=TestData.ABS_TOL) for g, o in zip(gs_result[1], ortho_result)
        )

    @pytest.mark.asyncio
    async def test_concurrent_projections(self):
        """Test concurrent execution of projection operations."""
        v1 = [3, 4, 5]
        v2 = [1, 0, 0]
        v3 = [0, 1, 0]

        results = await asyncio.gather(
            vector_projection(v1, v2),
            vector_projection(v1, v3),
            scalar_projection(v1, v2),
            vector_rejection(v1, v2),
        )

        assert len(results) == 4
        assert all(results[i] is not None for i in range(4))


# ============================================================================
# PERFORMANCE TESTS
# ============================================================================


class TestPerformance:
    """Performance and stress tests."""

    @pytest.mark.asyncio
    async def test_gram_schmidt_many_vectors(self):
        """Test Gram-Schmidt with many vectors."""
        # Create 20 linearly independent vectors in 20D space
        vectors = []
        for i in range(20):
            vec = [0] * 20
            for j in range(i + 1):
                vec[j] = 1
            vectors.append(vec)

        result = await gram_schmidt(vectors, normalize=True)

        assert len(result) == 20

        # Spot check orthonormality
        for i in range(0, len(result), 5):  # Check every 5th
            norm = await vector_norm(result[i])
            assert math.isclose(norm, 1.0, abs_tol=TestData.ABS_TOL)

    @pytest.mark.asyncio
    async def test_many_small_projections(self):
        """Test many small projection operations."""
        operations = []
        for i in range(100):
            v1 = [i, i + 1]
            v2 = [1, 0]
            operations.append(vector_projection(v1, v2))

        results = await asyncio.gather(*operations)
        assert len(results) == 100


# ============================================================================
# EDGE CASES
# ============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_projection_very_small_vector(self):
        """Test projection with very small vectors."""
        v1 = [1e-10, 2e-10]
        v2 = [1, 0]

        result = await vector_projection(v1, v2)
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_projection_very_large_vector(self):
        """Test projection with very large vectors."""
        v1 = [1e10, 2e10]
        v2 = [1, 0]

        result = await vector_projection(v1, v2)
        assert len(result) == 2
        assert not any(math.isnan(x) for x in result)

    @pytest.mark.asyncio
    async def test_gram_schmidt_near_dependent(self):
        """Test Gram-Schmidt with nearly dependent vectors."""
        # Create vectors that are almost but not quite dependent
        vectors = [[1, 0], [1, 1e-8]]

        # This should work but produce a very small second vector
        result = await gram_schmidt(vectors, normalize=True)
        assert len(result) == 2
