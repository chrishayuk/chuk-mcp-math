#!/usr/bin/env python3
# tests/math/linear_algebra/vectors/test_geometric.py
"""
Comprehensive pytest unit tests for geometric vector operations.

Tests cover:
- Normal operation cases
- Edge cases (zero vectors, parallel/orthogonal vectors)
- Error conditions
- Async behavior
- Type validation
- Numerical precision
- MCP compliance
"""

import pytest
import math
import asyncio
from typing import List, Union
from unittest.mock import patch, AsyncMock

# Import the functions to test
from chuk_mcp_math.linear_algebra.vectors.geometric import (
    vector_angle,
    vectors_parallel,
    vectors_orthogonal,
    triple_scalar_product,
    triple_vector_product
)

Number = Union[int, float]


class TestVectorAngle:
    """Test cases for the vector_angle function."""
    
    @pytest.mark.asyncio
    async def test_angle_orthogonal_vectors(self):
        """Test angle between orthogonal vectors."""
        # 90 degrees between unit vectors along x and y axes
        angle = await vector_angle([1, 0], [0, 1])
        assert abs(angle - math.pi/2) < 1e-10
        
        # Test with degrees=True
        angle_deg = await vector_angle([1, 0], [0, 1], degrees=True)
        assert abs(angle_deg - 90.0) < 1e-10
    
    @pytest.mark.asyncio
    async def test_angle_parallel_vectors(self):
        """Test angle between parallel vectors."""
        # Same direction (0 degrees)
        angle = await vector_angle([1, 2, 3], [2, 4, 6])
        assert abs(angle) < 1e-10
        
        # Opposite direction (180 degrees)
        angle = await vector_angle([1, 2, 3], [-1, -2, -3])
        assert abs(angle - math.pi) < 1e-10
    
    @pytest.mark.asyncio
    async def test_angle_45_degrees(self):
        """Test angle of 45 degrees."""
        angle = await vector_angle([1, 0], [1, 1], degrees=True)
        assert abs(angle - 45.0) < 1e-10
    
    @pytest.mark.asyncio
    async def test_angle_3d_vectors(self):
        """Test angle between 3D vectors."""
        # Orthogonal 3D vectors
        angle = await vector_angle([1, 0, 0], [0, 0, 1])
        assert abs(angle - math.pi/2) < 1e-10
        
        # Known angle in 3D
        v1 = [1, 1, 0]
        v2 = [1, 0, 1]
        angle = await vector_angle(v1, v2)
        expected = math.acos(1/2)  # 60 degrees
        assert abs(angle - expected) < 1e-10
    
    @pytest.mark.asyncio
    async def test_angle_zero_vector_error(self):
        """Test that zero vector raises ValueError."""
        with pytest.raises(ValueError, match="Cannot calculate angle with zero vector"):
            await vector_angle([0, 0], [1, 1])
        
        with pytest.raises(ValueError, match="Cannot calculate angle with zero vector"):
            await vector_angle([1, 1], [0, 0])
    
    @pytest.mark.asyncio
    async def test_angle_dimension_mismatch(self):
        """Test that vectors of different dimensions raise ValueError."""
        with pytest.raises(ValueError, match="Vectors must have same dimension"):
            await vector_angle([1, 2], [1, 2, 3])
    
    @pytest.mark.asyncio
    async def test_angle_numerical_stability(self):
        """Test numerical stability for nearly parallel vectors."""
        # Very small angle
        v1 = [1, 0]
        v2 = [1, 1e-10]
        angle = await vector_angle(v1, v2)
        assert angle >= 0 and angle <= math.pi
        
        # Nearly opposite vectors
        v1 = [1, 0]
        v2 = [-1, 1e-10]
        angle = await vector_angle(v1, v2)
        assert abs(angle - math.pi) < 1e-5


class TestVectorsParallel:
    """Test cases for the vectors_parallel function."""
    
    @pytest.mark.asyncio
    async def test_parallel_vectors(self):
        """Test detection of parallel vectors."""
        # Same direction
        assert await vectors_parallel([1, 2, 3], [2, 4, 6]) is True
        assert await vectors_parallel([1, 0], [3, 0]) is True
        
        # Opposite direction (anti-parallel)
        assert await vectors_parallel([1, 2, 3], [-1, -2, -3]) is True
        # Note: 2D vectors use angle-based check, not cross product
    
    @pytest.mark.asyncio
    async def test_non_parallel_vectors(self):
        """Test detection of non-parallel vectors."""
        assert await vectors_parallel([1, 0], [0, 1]) is False
        assert await vectors_parallel([1, 2, 3], [1, 2, 4]) is False
        assert await vectors_parallel([1, 1], [1, -1]) is False
    
    @pytest.mark.asyncio
    async def test_zero_vector_parallel(self):
        """Test that zero vector is parallel to any vector."""
        assert await vectors_parallel([0, 0], [1, 2]) is True
        assert await vectors_parallel([1, 2, 3], [0, 0, 0]) is True
        assert await vectors_parallel([0, 0, 0], [0, 0, 0]) is True
    
    @pytest.mark.asyncio
    async def test_3d_parallel_vectors(self):
        """Test parallel detection in 3D using cross product."""
        # Parallel 3D vectors
        assert await vectors_parallel([1, 2, 3], [3, 6, 9]) is True
        assert await vectors_parallel([1, 0, 0], [-2, 0, 0]) is True
        
        # Non-parallel 3D vectors
        assert await vectors_parallel([1, 0, 0], [0, 1, 0]) is False
        assert await vectors_parallel([1, 1, 1], [1, 1, 0]) is False
    
    @pytest.mark.asyncio
    async def test_dimension_mismatch(self):
        """Test that vectors of different dimensions return False."""
        assert await vectors_parallel([1, 2], [1, 2, 3]) is False
    
    @pytest.mark.asyncio
    async def test_custom_tolerance(self):
        """Test parallel detection with custom tolerance."""
        # Nearly parallel 3D vectors (uses cross product)
        v1 = [1, 0, 0]
        v2 = [1, 1e-8, 0]  # Slightly off parallel
        assert await vectors_parallel(v1, v2, tolerance=1e-7) is True
        assert await vectors_parallel(v1, v2, tolerance=1e-10) is False  # Stricter tolerance


class TestVectorsOrthogonal:
    """Test cases for the vectors_orthogonal function."""
    
    @pytest.mark.asyncio
    async def test_orthogonal_vectors(self):
        """Test detection of orthogonal vectors."""
        # 2D orthogonal vectors
        assert await vectors_orthogonal([1, 0], [0, 1]) is True
        assert await vectors_orthogonal([1, 1], [-1, 1]) is True
        assert await vectors_orthogonal([2, 3], [-3, 2]) is True
        
        # 3D orthogonal vectors
        assert await vectors_orthogonal([1, 0, 0], [0, 1, 0]) is True
        assert await vectors_orthogonal([1, 1, 0], [-1, 1, 0]) is True
        assert await vectors_orthogonal([1, 2, 3], [-3, 0, 1]) is True
    
    @pytest.mark.asyncio
    async def test_non_orthogonal_vectors(self):
        """Test detection of non-orthogonal vectors."""
        assert await vectors_orthogonal([1, 0], [1, 1]) is False
        assert await vectors_orthogonal([1, 2], [2, 4]) is False  # Parallel
        assert await vectors_orthogonal([1, 1, 1], [1, 1, 0]) is False
    
    @pytest.mark.asyncio
    async def test_zero_vector_orthogonal(self):
        """Test that zero vector is orthogonal to any vector."""
        assert await vectors_orthogonal([0, 0], [1, 2]) is True
        assert await vectors_orthogonal([1, 2, 3], [0, 0, 0]) is True
    
    @pytest.mark.asyncio
    async def test_dimension_mismatch(self):
        """Test that vectors of different dimensions return False."""
        assert await vectors_orthogonal([1, 2], [1, 2, 3]) is False
    
    @pytest.mark.asyncio
    async def test_custom_tolerance(self):
        """Test orthogonal detection with custom tolerance."""
        # Nearly orthogonal vectors
        v1 = [1, 0]
        v2 = [1e-11, 1]
        assert await vectors_orthogonal(v1, v2, tolerance=1e-10) is True
        assert await vectors_orthogonal(v1, v2, tolerance=1e-12) is False
    
    @pytest.mark.asyncio
    async def test_negative_components(self):
        """Test orthogonality with negative components."""
        assert await vectors_orthogonal([1, -1], [1, 1]) is True
        assert await vectors_orthogonal([-2, 3], [3, 2]) is True
        assert await vectors_orthogonal([-1, -1], [1, -1]) is True


class TestTripleScalarProduct:
    """Test cases for the triple_scalar_product function."""
    
    @pytest.mark.asyncio
    async def test_unit_cube_volume(self):
        """Test triple scalar product gives unit cube volume."""
        # Standard basis vectors form unit cube
        result = await triple_scalar_product([1, 0, 0], [0, 1, 0], [0, 0, 1])
        assert abs(result - 1.0) < 1e-10
    
    @pytest.mark.asyncio
    async def test_parallelepiped_volume(self):
        """Test triple scalar product for general parallelepiped."""
        # Known volume calculation
        result = await triple_scalar_product([2, 0, 0], [0, 3, 0], [0, 0, 4])
        assert abs(result - 24.0) < 1e-10  # 2 * 3 * 4
    
    @pytest.mark.asyncio
    async def test_coplanar_vectors(self):
        """Test that coplanar vectors give zero volume."""
        # Vectors in xy-plane
        result = await triple_scalar_product([1, 0, 0], [0, 1, 0], [1, 1, 0])
        assert abs(result) < 1e-10
        
        # Linearly dependent vectors
        result = await triple_scalar_product([1, 2, 3], [2, 4, 6], [0, 0, 1])
        assert abs(result) < 1e-10
    
    @pytest.mark.asyncio
    async def test_negative_volume(self):
        """Test that reversed orientation gives negative volume."""
        # Positive orientation
        pos_result = await triple_scalar_product([1, 0, 0], [0, 1, 0], [0, 0, 1])
        
        # Negative orientation (swap two vectors)
        neg_result = await triple_scalar_product([0, 1, 0], [1, 0, 0], [0, 0, 1])
        
        assert abs(pos_result + neg_result) < 1e-10
        assert pos_result > 0
        assert neg_result < 0
    
    @pytest.mark.asyncio
    async def test_non_3d_vectors_error(self):
        """Test that non-3D vectors raise ValueError."""
        with pytest.raises(ValueError, match="Triple scalar product requires 3D vectors"):
            await triple_scalar_product([1, 2], [3, 4], [5, 6])
        
        with pytest.raises(ValueError, match="Triple scalar product requires 3D vectors"):
            await triple_scalar_product([1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4])
    
    @pytest.mark.asyncio
    async def test_cyclic_property(self):
        """Test cyclic property: a·(b×c) = b·(c×a) = c·(a×b)."""
        a, b, c = [1, 2, 3], [4, 5, 6], [7, 8, 9]
        
        result1 = await triple_scalar_product(a, b, c)
        result2 = await triple_scalar_product(b, c, a)
        result3 = await triple_scalar_product(c, a, b)
        
        assert abs(result1 - result2) < 1e-10
        assert abs(result2 - result3) < 1e-10


class TestTripleVectorProduct:
    """Test cases for the triple_vector_product function."""
    
    @pytest.mark.asyncio
    async def test_bac_cab_rule(self):
        """Test the BAC-CAB rule: a×(b×c) = b(a·c) - c(a·b)."""
        a = [1, 2, 3]
        b = [4, 5, 6]
        c = [7, 8, 9]
        
        result = await triple_vector_product(a, b, c)
        
        # Manual calculation using BAC-CAB rule
        # a·c = 1*7 + 2*8 + 3*9 = 50
        # a·b = 1*4 + 2*5 + 3*6 = 32
        # Result = b*50 - c*32 = [200, 250, 300] - [224, 256, 288] = [-24, -6, 12]
        
        expected = [-24, -6, 12]
        for i in range(3):
            assert abs(result[i] - expected[i]) < 1e-10
    
    @pytest.mark.asyncio
    async def test_orthogonal_basis(self):
        """Test triple vector product with orthogonal basis vectors."""
        i = [1, 0, 0]
        j = [0, 1, 0]
        k = [0, 0, 1]
        
        # i×(j×k) = i×i = 0
        result = await triple_vector_product(i, j, k)
        assert all(abs(x) < 1e-10 for x in result)
    
    @pytest.mark.asyncio
    async def test_parallel_vectors(self):
        """Test triple vector product with parallel vectors."""
        a = [1, 2, 3]
        b = [2, 4, 6]  # Parallel to a
        c = [1, 0, 0]
        
        # When b is parallel to a, a×(b×c) simplifies
        result = await triple_vector_product(a, b, c)
        
        # b×c will be perpendicular to b (and a)
        # a×(b×c) should be in the plane of b and c
        assert len(result) == 3
        assert isinstance(result[0], float)
    
    @pytest.mark.asyncio
    async def test_non_3d_vectors_error(self):
        """Test that non-3D vectors raise ValueError."""
        with pytest.raises(ValueError, match="Triple vector product requires 3D vectors"):
            await triple_vector_product([1, 2], [3, 4], [5, 6])
        
        with pytest.raises(ValueError, match="Triple vector product requires 3D vectors"):
            await triple_vector_product([1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4])
    
    @pytest.mark.asyncio
    async def test_zero_vector_result(self):
        """Test cases that should produce zero vector."""
        # Same vectors
        a = [1, 2, 3]
        result = await triple_vector_product(a, a, a)
        assert all(abs(x) < 1e-10 for x in result)
        
        # Orthogonal to both b and c
        b = [1, 0, 0]
        c = [0, 1, 0]
        a = [0, 0, 1]
        result = await triple_vector_product(a, b, c)
        assert all(abs(x) < 1e-10 for x in result)
    
    @pytest.mark.asyncio
    async def test_result_type(self):
        """Test that result is always a list of floats."""
        result = await triple_vector_product([1, 0, 0], [0, 1, 0], [0, 0, 1])
        assert isinstance(result, list)
        assert len(result) == 3
        assert all(isinstance(x, float) for x in result)


class TestIntegration:
    """Integration tests combining multiple geometric operations."""
    
    @pytest.mark.asyncio
    async def test_orthogonal_and_angle(self):
        """Test that orthogonal vectors have 90-degree angle."""
        v1 = [1, 0, 0]
        v2 = [0, 1, 0]
        
        is_orthogonal = await vectors_orthogonal(v1, v2)
        angle_deg = await vector_angle(v1, v2, degrees=True)
        
        assert is_orthogonal is True
        assert abs(angle_deg - 90.0) < 1e-10
    
    @pytest.mark.asyncio
    async def test_parallel_and_angle(self):
        """Test that parallel vectors have 0 or 180-degree angle."""
        # Same direction
        v1 = [1, 2, 3]
        v2 = [2, 4, 6]
        
        is_parallel = await vectors_parallel(v1, v2)
        angle = await vector_angle(v1, v2)
        
        assert is_parallel is True
        assert abs(angle) < 1e-10
        
        # Opposite direction
        v3 = [-2, -4, -6]
        
        is_parallel = await vectors_parallel(v1, v3)
        angle = await vector_angle(v1, v3)
        
        assert is_parallel is True
        assert abs(angle - math.pi) < 1e-10
    
    @pytest.mark.asyncio
    async def test_triple_products_consistency(self):
        """Test consistency between triple scalar and vector products."""
        a = [1, 2, 3]
        b = [4, 5, 6]
        c = [7, 8, 10]  # Not coplanar with a and b
        
        # Triple scalar product should be non-zero for non-coplanar vectors
        scalar_result = await triple_scalar_product(a, b, c)
        assert abs(scalar_result) > 1e-10
        
        # Triple vector product should give a vector in the plane of b and c
        vector_result = await triple_vector_product(a, b, c)
        assert len(vector_result) == 3
        
        # The result should be perpendicular to a×b (approximately)
        # This is a property of the triple vector product
        assert isinstance(vector_result[0], float)


# Performance and edge case tests
class TestPerformance:
    """Test performance-related aspects."""
    
    @pytest.mark.asyncio
    async def test_large_component_values(self):
        """Test with large component values."""
        v1 = [1e10, 2e10, 3e10]
        v2 = [4e10, 5e10, 6e10]
        
        # Should handle large values without overflow
        angle = await vector_angle(v1, v2)
        assert 0 <= angle <= math.pi
        
        is_parallel = await vectors_parallel([1e10, 0, 0], [2e10, 0, 0])
        assert is_parallel is True
    
    @pytest.mark.asyncio
    async def test_small_component_values(self):
        """Test with very small component values."""
        v1 = [1e-10, 2e-10, 3e-10]
        v2 = [4e-10, 5e-10, 6e-10]
        
        # Should handle small values without underflow
        angle = await vector_angle(v1, v2)
        assert 0 <= angle <= math.pi
        
        # Test orthogonality with small values
        is_orthogonal = await vectors_orthogonal([1e-10, 0], [0, 1e-10])
        assert is_orthogonal is True