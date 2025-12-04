"""
Vector Norms and Normalization

Functions for calculating various vector norms and normalizing vectors.
All functions are async-native with MCP decoration.
"""

import asyncio
from typing import List, Union
from ...mcp_decorator import mcp_function


@mcp_function(
    description="Calculate the norm (magnitude) of a vector",
    namespace="linear_algebra.vectors",
    cache_strategy="memory",
    estimated_cpu_usage="low",
)
async def vector_norm(
    vector: List[Union[int, float]], p: Union[int, float] = 2
) -> float:
    """
    Calculate the p-norm of a vector.

    Formula:
        ||v||_p = (Σ|vᵢ|^p)^(1/p)

    Special cases:
        p=1: Manhattan norm (L1)
        p=2: Euclidean norm (L2)
        p=∞: Chebyshev norm (L∞)

    Args:
        vector: Input vector
        p: Norm order (default: 2 for Euclidean)

    Returns:
        Vector norm value

    Raises:
        ValueError: If vector is empty or p < 1

    Examples:
        >>> await vector_norm([3, 4])  # Euclidean norm
        5.0
        >>> await vector_norm([1, -2, 3], p=1)  # Manhattan norm
        6.0
    """
    if not vector:
        raise ValueError("Vector cannot be empty")

    if p < 1:
        raise ValueError(f"Norm order p must be >= 1, got {p}")

    # Yield for large vectors
    if len(vector) > 1000:
        await asyncio.sleep(0)

    if p == float("inf"):
        return float(max(abs(v) for v in vector))

    sum_powers = sum(abs(v) ** p for v in vector)
    return float(sum_powers ** (1 / p))


@mcp_function(
    description="Calculate the Euclidean norm (L2 norm) of a vector",
    namespace="linear_algebra.vectors",
    cache_strategy="memory",
    estimated_cpu_usage="low",
)
async def euclidean_norm(vector: List[Union[int, float]]) -> float:
    """
    Calculate the Euclidean norm (L2 norm) of a vector.

    Formula:
        ||v||₂ = √(Σvᵢ²)

    Args:
        vector: Input vector

    Returns:
        Euclidean norm value

    Examples:
        >>> await euclidean_norm([3, 4])
        5.0
    """
    return await vector_norm(vector, p=2)


@mcp_function(
    description="Calculate the Manhattan norm (L1 norm) of a vector",
    namespace="linear_algebra.vectors",
    cache_strategy="memory",
    estimated_cpu_usage="low",
)
async def manhattan_norm(vector: List[Union[int, float]]) -> float:
    """
    Calculate the Manhattan norm (L1 norm) of a vector.

    Formula:
        ||v||₁ = Σ|vᵢ|

    Args:
        vector: Input vector

    Returns:
        Manhattan norm value

    Examples:
        >>> await manhattan_norm([1, -2, 3])
        6.0
    """
    return await vector_norm(vector, p=1)


@mcp_function(
    description="Calculate the Chebyshev norm (L∞ norm) of a vector",
    namespace="linear_algebra.vectors",
    cache_strategy="memory",
    estimated_cpu_usage="low",
)
async def chebyshev_norm(vector: List[Union[int, float]]) -> float:
    """
    Calculate the Chebyshev norm (L∞ norm, maximum norm) of a vector.

    Formula:
        ||v||∞ = max(|vᵢ|)

    Args:
        vector: Input vector

    Returns:
        Maximum absolute value in the vector

    Examples:
        >>> await chebyshev_norm([1, -5, 3])
        5.0
    """
    return await vector_norm(vector, p=float("inf"))


@mcp_function(
    description="Calculate the p-norm of a vector for any p >= 1",
    namespace="linear_algebra.vectors",
    cache_strategy="memory",
    estimated_cpu_usage="low",
)
async def p_norm(vector: List[Union[int, float]], p: Union[int, float]) -> float:
    """
    Calculate the p-norm of a vector for any p >= 1.

    Formula:
        ||v||_p = (Σ|vᵢ|^p)^(1/p)

    Args:
        vector: Input vector
        p: Norm order (must be >= 1)

    Returns:
        p-norm value

    Examples:
        >>> await p_norm([1, 2, 3], 3)
        3.302...
    """
    return await vector_norm(vector, p=p)


@mcp_function(
    description="Normalize a vector to unit length",
    namespace="linear_algebra.vectors",
    cache_strategy="memory",
    estimated_cpu_usage="low",
)
async def normalize_vector(
    vector: List[Union[int, float]],
    norm_type: Union[int, float] = 2,
    tolerance: float = 1e-10,
) -> List[float]:
    """
    Normalize a vector to unit length using specified norm.

    Formula:
        v_normalized = v / ||v||

    Args:
        vector: Input vector to normalize
        norm_type: Type of norm to use (default: 2 for Euclidean)
        tolerance: Minimum norm value to avoid division by zero

    Returns:
        Normalized vector with unit length

    Raises:
        ValueError: If vector has zero norm (within tolerance)

    Examples:
        >>> await normalize_vector([3, 4])
        [0.6, 0.8]
        >>> await normalize_vector([1, 1, 1])
        [0.577..., 0.577..., 0.577...]
    """
    if not vector:
        raise ValueError("Vector cannot be empty")

    # Calculate norm
    norm = await vector_norm(vector, p=norm_type)

    if norm < tolerance:
        raise ValueError(
            f"Cannot normalize vector with norm {norm} (below tolerance {tolerance}). "
            "Vector is too close to zero vector."
        )

    # Yield for large vectors
    if len(vector) > 1000:
        await asyncio.sleep(0)

    return [float(v / norm) for v in vector]
