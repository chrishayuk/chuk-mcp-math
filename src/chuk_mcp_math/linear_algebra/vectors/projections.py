"""
Vector Projections and Orthogonalization

Functions for vector projections, rejections, and orthogonalization processes.
All functions are async-native with MCP decoration.
"""

import asyncio
from typing import List, Union
from .operations import dot_product, scalar_multiply, vector_subtract
from .norms import vector_norm
from ...mcp_decorator import mcp_function


@mcp_function(
    description="Project one vector onto another",
    namespace="linear_algebra.vectors",
    cache_strategy="memory",
    estimated_cpu_usage="low",
)
async def vector_projection(
    vector_a: List[Union[int, float]], vector_b: List[Union[int, float]]
) -> List[float]:
    """
    Project vector_a onto vector_b.

    Formula:
        proj_b(a) = ((a·b) / (b·b)) × b

    Args:
        vector_a: Vector to project
        vector_b: Vector to project onto

    Returns:
        Projection of vector_a onto vector_b

    Raises:
        ValueError: If vector_b is zero vector

    Examples:
        >>> await vector_projection([3, 4], [1, 0])
        [3.0, 0.0]  # Projects onto x-axis
    """
    if len(vector_a) != len(vector_b):
        raise ValueError(
            f"Vectors must have same dimension. Got {len(vector_a)} and {len(vector_b)}"
        )

    # Calculate dot products
    dot_ab = await dot_product(vector_a, vector_b)
    dot_bb = await dot_product(vector_b, vector_b)

    if abs(dot_bb) < 1e-10:
        raise ValueError("Cannot project onto zero vector")

    scalar = dot_ab / dot_bb
    return await scalar_multiply(vector_b, scalar)


@mcp_function(
    description="Calculate vector rejection (component perpendicular to another vector)",
    namespace="linear_algebra.vectors",
    cache_strategy="memory",
    estimated_cpu_usage="low",
)
async def vector_rejection(
    vector_a: List[Union[int, float]], vector_b: List[Union[int, float]]
) -> List[float]:
    """
    Calculate the rejection of vector_a from vector_b (perpendicular component).

    Formula:
        rej_b(a) = a - proj_b(a)

    Args:
        vector_a: Vector to reject
        vector_b: Vector to reject from

    Returns:
        Component of vector_a perpendicular to vector_b

    Examples:
        >>> await vector_rejection([3, 4], [1, 0])
        [0.0, 4.0]  # Component perpendicular to x-axis
    """
    projection = await vector_projection(vector_a, vector_b)
    return await vector_subtract(vector_a, projection)


@mcp_function(
    description="Calculate scalar projection of one vector onto another",
    namespace="linear_algebra.vectors",
    cache_strategy="memory",
    estimated_cpu_usage="low",
)
async def scalar_projection(
    vector_a: List[Union[int, float]], vector_b: List[Union[int, float]]
) -> float:
    """
    Calculate the scalar projection of vector_a onto vector_b.

    Formula:
        comp_b(a) = a·b / ||b||

    Args:
        vector_a: Vector to project
        vector_b: Vector to project onto

    Returns:
        Scalar projection value (signed length)

    Examples:
        >>> await scalar_projection([3, 4], [1, 0])
        3.0  # Length of projection onto x-axis
    """
    dot_ab = await dot_product(vector_a, vector_b)
    norm_b = await vector_norm(vector_b)

    if norm_b < 1e-10:
        raise ValueError("Cannot project onto zero vector")

    return dot_ab / norm_b


@mcp_function(
    description="Orthogonalize a vector with respect to a set of vectors",
    namespace="linear_algebra.vectors",
    cache_strategy="memory",
    estimated_cpu_usage="medium",
)
async def orthogonalize(
    vector: List[Union[int, float]], basis_vectors: List[List[Union[int, float]]]
) -> List[float]:
    """
    Orthogonalize a vector with respect to a set of basis vectors.

    Removes all components of the vector in the directions of the basis vectors.

    Args:
        vector: Vector to orthogonalize
        basis_vectors: List of vectors to orthogonalize against

    Returns:
        Orthogonalized vector

    Examples:
        >>> await orthogonalize([1, 1], [[1, 0]])
        [0.0, 1.0]  # Remove x-component
    """
    result = vector.copy()

    for basis in basis_vectors:
        projection = await vector_projection(result, basis)
        result = await vector_subtract(result, projection)

    return [float(x) for x in result]


@mcp_function(
    description="Apply Gram-Schmidt orthogonalization to a set of vectors",
    namespace="linear_algebra.vectors",
    cache_strategy="memory",
    estimated_cpu_usage="medium",
)
async def gram_schmidt(
    vectors: List[List[Union[int, float]]], normalize: bool = True
) -> List[List[float]]:
    """
    Apply Gram-Schmidt process to orthogonalize a set of vectors.

    Creates an orthogonal (or orthonormal if normalized) set of vectors
    that span the same space as the input vectors.

    Args:
        vectors: List of linearly independent vectors
        normalize: If True, return orthonormal vectors (default: True)

    Returns:
        List of orthogonal (or orthonormal) vectors

    Raises:
        ValueError: If vectors are linearly dependent

    Examples:
        >>> await gram_schmidt([[1, 0], [1, 1]])
        [[1.0, 0.0], [0.0, 1.0]]  # Orthonormal basis
    """
    if not vectors:
        raise ValueError("Input vectors list cannot be empty")

    # Check dimensions
    dim = len(vectors[0])
    if not all(len(v) == dim for v in vectors):
        raise ValueError("All vectors must have the same dimension")

    orthogonal: list[list[float]] = []

    for i, vector in enumerate(vectors):
        # Start with the original vector
        ortho_vector = vector.copy()

        # Remove projections onto all previous orthogonal vectors
        for j in range(i):
            projection = await vector_projection(ortho_vector, orthogonal[j])
            ortho_vector = await vector_subtract(ortho_vector, projection)

        # Check for linear dependence
        norm = await vector_norm(ortho_vector)
        if norm < 1e-10:
            raise ValueError(f"Vector {i} is linearly dependent on previous vectors")

        # Normalize if requested
        if normalize:
            from .norms import normalize_vector

            ortho_vector = await normalize_vector(ortho_vector)

        orthogonal.append([float(x) for x in ortho_vector])

        # Yield control for large sets
        if i % 10 == 0 and i > 0:
            await asyncio.sleep(0)

    return orthogonal
