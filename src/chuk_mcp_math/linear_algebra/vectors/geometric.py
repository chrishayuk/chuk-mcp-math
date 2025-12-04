"""
Geometric Vector Operations

Functions for geometric operations on vectors including angles, parallelism,
orthogonality, and triple products.
"""

import math
from typing import List, Union
from .operations import dot_product, cross_product
from .norms import vector_norm
from ...mcp_decorator import mcp_function


@mcp_function(
    description="Calculate the angle between two vectors in radians",
    namespace="linear_algebra.vectors",
    cache_strategy="memory",
    estimated_cpu_usage="low",
)
async def vector_angle(
    vector_a: List[Union[int, float]],
    vector_b: List[Union[int, float]],
    degrees: bool = False,
) -> float:
    """
    Calculate the angle between two vectors.

    Formula:
        θ = arccos((a·b) / (||a|| × ||b||))

    Args:
        vector_a: First vector
        vector_b: Second vector
        degrees: If True, return angle in degrees (default: False for radians)

    Returns:
        Angle between vectors in radians (or degrees if specified)

    Raises:
        ValueError: If either vector is zero vector

    Examples:
        >>> await vector_angle([1, 0], [0, 1])
        1.5707...  # π/2 radians (90 degrees)
        >>> await vector_angle([1, 0], [1, 1], degrees=True)
        45.0
    """
    if len(vector_a) != len(vector_b):
        raise ValueError(
            f"Vectors must have same dimension. Got {len(vector_a)} and {len(vector_b)}"
        )

    # Calculate norms
    norm_a = await vector_norm(vector_a)
    norm_b = await vector_norm(vector_b)

    if norm_a < 1e-10 or norm_b < 1e-10:
        raise ValueError("Cannot calculate angle with zero vector")

    # Calculate dot product
    dot_ab = await dot_product(vector_a, vector_b)

    # Calculate cosine of angle
    cos_angle = dot_ab / (norm_a * norm_b)

    # Clamp to [-1, 1] to handle numerical errors
    cos_angle = max(-1.0, min(1.0, cos_angle))

    # Calculate angle in radians
    angle_rad = math.acos(cos_angle)

    if degrees:
        return math.degrees(angle_rad)
    return angle_rad


@mcp_function(
    description="Check if two vectors are parallel (or anti-parallel)",
    namespace="linear_algebra.vectors",
    cache_strategy="memory",
    estimated_cpu_usage="low",
)
async def vectors_parallel(
    vector_a: List[Union[int, float]],
    vector_b: List[Union[int, float]],
    tolerance: float = 1e-10,
) -> bool:
    """
    Check if two vectors are parallel or anti-parallel.

    Two vectors are parallel if one is a scalar multiple of the other.

    Args:
        vector_a: First vector
        vector_b: Second vector
        tolerance: Numerical tolerance for comparison

    Returns:
        True if vectors are parallel, False otherwise

    Examples:
        >>> await vectors_parallel([1, 2, 3], [2, 4, 6])
        True
        >>> await vectors_parallel([1, 2, 3], [-1, -2, -3])
        True  # Anti-parallel
    """
    if len(vector_a) != len(vector_b):
        return False

    # Check for zero vectors
    norm_a = await vector_norm(vector_a)
    norm_b = await vector_norm(vector_b)

    if norm_a < tolerance or norm_b < tolerance:
        return True  # Zero vector is parallel to any vector

    # For 2D and 3D vectors, use cross product
    if len(vector_a) == 3:
        cross = await cross_product(vector_a, vector_b)
        cross_norm = await vector_norm(cross)
        return cross_norm < tolerance

    # For general dimensions, check if angle is 0 or π
    angle = await vector_angle(vector_a, vector_b)
    return abs(angle) < tolerance or abs(angle - math.pi) < tolerance


@mcp_function(
    description="Check if two vectors are orthogonal (perpendicular)",
    namespace="linear_algebra.vectors",
    cache_strategy="memory",
    estimated_cpu_usage="low",
)
async def vectors_orthogonal(
    vector_a: List[Union[int, float]],
    vector_b: List[Union[int, float]],
    tolerance: float = 1e-10,
) -> bool:
    """
    Check if two vectors are orthogonal (perpendicular).

    Two vectors are orthogonal if their dot product is zero.

    Args:
        vector_a: First vector
        vector_b: Second vector
        tolerance: Numerical tolerance for comparison

    Returns:
        True if vectors are orthogonal, False otherwise

    Examples:
        >>> await vectors_orthogonal([1, 0], [0, 1])
        True
        >>> await vectors_orthogonal([1, 1], [-1, 1])
        True
    """
    if len(vector_a) != len(vector_b):
        return False

    dot = await dot_product(vector_a, vector_b)
    return abs(dot) < tolerance


@mcp_function(
    description="Calculate the triple scalar product of three 3D vectors",
    namespace="linear_algebra.vectors",
    cache_strategy="memory",
    estimated_cpu_usage="low",
)
async def triple_scalar_product(
    vector_a: List[Union[int, float]],
    vector_b: List[Union[int, float]],
    vector_c: List[Union[int, float]],
) -> float:
    """
    Calculate the triple scalar product a·(b×c).

    The triple scalar product gives the signed volume of the parallelepiped
    formed by the three vectors.

    Formula:
        a·(b×c) = |a₁ b₁ c₁|
                  |a₂ b₂ c₂|
                  |a₃ b₃ c₃|

    Args:
        vector_a: First 3D vector
        vector_b: Second 3D vector
        vector_c: Third 3D vector

    Returns:
        Scalar triple product value (signed volume)

    Raises:
        ValueError: If vectors are not 3-dimensional

    Examples:
        >>> await triple_scalar_product([1, 0, 0], [0, 1, 0], [0, 0, 1])
        1.0  # Volume of unit cube
    """
    if len(vector_a) != 3 or len(vector_b) != 3 or len(vector_c) != 3:
        raise ValueError("Triple scalar product requires 3D vectors")

    # Calculate b × c
    cross_bc = await cross_product(vector_b, vector_c)

    # Calculate a · (b × c)
    result = await dot_product(vector_a, cross_bc)

    return result


@mcp_function(
    description="Calculate the triple vector product a×(b×c)",
    namespace="linear_algebra.vectors",
    cache_strategy="memory",
    estimated_cpu_usage="low",
)
async def triple_vector_product(
    vector_a: List[Union[int, float]],
    vector_b: List[Union[int, float]],
    vector_c: List[Union[int, float]],
) -> List[float]:
    """
    Calculate the triple vector product a×(b×c).

    Uses the BAC-CAB rule:
        a×(b×c) = b(a·c) - c(a·b)

    Args:
        vector_a: First 3D vector
        vector_b: Second 3D vector
        vector_c: Third 3D vector

    Returns:
        Triple vector product result

    Raises:
        ValueError: If vectors are not 3-dimensional

    Examples:
        >>> await triple_vector_product([1, 0, 0], [0, 1, 0], [0, 0, 1])
        [0.0, 0.0, 0.0]
    """
    if len(vector_a) != 3 or len(vector_b) != 3 or len(vector_c) != 3:
        raise ValueError("Triple vector product requires 3D vectors")

    # Apply BAC-CAB rule: a×(b×c) = b(a·c) - c(a·b)
    from .operations import scalar_multiply, vector_subtract

    dot_ac = await dot_product(vector_a, vector_c)
    dot_ab = await dot_product(vector_a, vector_b)

    b_scaled = await scalar_multiply(vector_b, dot_ac)
    c_scaled = await scalar_multiply(vector_c, dot_ab)

    result = await vector_subtract(b_scaled, c_scaled)

    return result
