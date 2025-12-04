"""
Vector Operations

Basic vector arithmetic operations including addition, subtraction, multiplication.
All functions are async-native with MCP decoration for AI model integration.
"""

import asyncio
from typing import List, Union
from ...mcp_decorator import mcp_function


@mcp_function(
    description="Calculate the dot product of two vectors (inner product)",
    namespace="linear_algebra.vectors",
    cache_strategy="memory",
    cache_ttl_seconds=3600,
    estimated_cpu_usage="low",
    examples=[
        {
            "input": {"vector_a": [1, 2, 3], "vector_b": [4, 5, 6]},
            "output": 32,
            "description": "Dot product of [1,2,3] and [4,5,6]",
        }
    ],
)
async def dot_product(
    vector_a: List[Union[int, float]], vector_b: List[Union[int, float]]
) -> float:
    """
    Calculate the dot product (inner product) of two vectors.

    Formula:
        a·b = Σ(aᵢ × bᵢ) for i = 1 to n

    Args:
        vector_a: First vector
        vector_b: Second vector

    Returns:
        Scalar dot product value

    Raises:
        ValueError: If vectors have different dimensions

    Examples:
        >>> await dot_product([1, 2, 3], [4, 5, 6])
        32.0
        >>> await dot_product([1, 0], [0, 1])  # Orthogonal vectors
        0.0
    """
    if len(vector_a) != len(vector_b):
        raise ValueError(
            f"Vectors must have same dimension. Got {len(vector_a)} and {len(vector_b)}"
        )

    if not vector_a:
        raise ValueError("Vectors cannot be empty")

    # For large vectors, yield control
    if len(vector_a) > 1000:
        await asyncio.sleep(0)

    result = sum(a * b for a, b in zip(vector_a, vector_b))
    return float(result)


@mcp_function(
    description="Calculate the cross product of two 3D vectors",
    namespace="linear_algebra.vectors",
    cache_strategy="memory",
    estimated_cpu_usage="low",
)
async def cross_product(
    vector_a: List[Union[int, float]], vector_b: List[Union[int, float]]
) -> List[float]:
    """
    Calculate the cross product of two 3D vectors.

    Formula:
        a × b = [a₂b₃ - a₃b₂, a₃b₁ - a₁b₃, a₁b₂ - a₂b₁]

    Args:
        vector_a: First 3D vector
        vector_b: Second 3D vector

    Returns:
        Cross product vector (perpendicular to both inputs)

    Raises:
        ValueError: If vectors are not 3-dimensional

    Examples:
        >>> await cross_product([1, 0, 0], [0, 1, 0])
        [0.0, 0.0, 1.0]  # z-axis
    """
    if len(vector_a) != 3 or len(vector_b) != 3:
        raise ValueError(
            f"Cross product requires 3D vectors. Got dimensions {len(vector_a)} and {len(vector_b)}"
        )

    result = [
        vector_a[1] * vector_b[2] - vector_a[2] * vector_b[1],
        vector_a[2] * vector_b[0] - vector_a[0] * vector_b[2],
        vector_a[0] * vector_b[1] - vector_a[1] * vector_b[0],
    ]

    return [float(x) for x in result]


@mcp_function(
    description="Multiply a vector by a scalar value",
    namespace="linear_algebra.vectors",
    cache_strategy="memory",
    estimated_cpu_usage="low",
)
async def scalar_multiply(
    vector: List[Union[int, float]], scalar: Union[int, float]
) -> List[float]:
    """
    Multiply each component of a vector by a scalar.

    Formula:
        c·v = [c×v₁, c×v₂, ..., c×vₙ]

    Args:
        vector: Input vector
        scalar: Scalar multiplier

    Returns:
        Scaled vector

    Examples:
        >>> await scalar_multiply([1, 2, 3], 2)
        [2.0, 4.0, 6.0]
    """
    if not vector:
        raise ValueError("Vector cannot be empty")

    # Yield for large vectors
    if len(vector) > 1000:
        await asyncio.sleep(0)

    return [float(scalar * v) for v in vector]


@mcp_function(
    description="Add two vectors element-wise",
    namespace="linear_algebra.vectors",
    cache_strategy="memory",
    estimated_cpu_usage="low",
)
async def vector_add(
    vector_a: List[Union[int, float]], vector_b: List[Union[int, float]]
) -> List[float]:
    """
    Add two vectors element-wise.

    Formula:
        a + b = [a₁+b₁, a₂+b₂, ..., aₙ+bₙ]

    Args:
        vector_a: First vector
        vector_b: Second vector

    Returns:
        Sum vector

    Raises:
        ValueError: If vectors have different dimensions

    Examples:
        >>> await vector_add([1, 2, 3], [4, 5, 6])
        [5.0, 7.0, 9.0]
    """
    if len(vector_a) != len(vector_b):
        raise ValueError(
            f"Vectors must have same dimension. Got {len(vector_a)} and {len(vector_b)}"
        )

    if not vector_a:
        raise ValueError("Vectors cannot be empty")

    # Yield for large vectors
    if len(vector_a) > 1000:
        await asyncio.sleep(0)

    return [float(a + b) for a, b in zip(vector_a, vector_b)]


@mcp_function(
    description="Subtract one vector from another element-wise",
    namespace="linear_algebra.vectors",
    cache_strategy="memory",
    estimated_cpu_usage="low",
)
async def vector_subtract(
    vector_a: List[Union[int, float]], vector_b: List[Union[int, float]]
) -> List[float]:
    """
    Subtract vector_b from vector_a element-wise.

    Formula:
        a - b = [a₁-b₁, a₂-b₂, ..., aₙ-bₙ]

    Args:
        vector_a: Vector to subtract from
        vector_b: Vector to subtract

    Returns:
        Difference vector

    Raises:
        ValueError: If vectors have different dimensions

    Examples:
        >>> await vector_subtract([5, 7, 9], [1, 2, 3])
        [4.0, 5.0, 6.0]
    """
    if len(vector_a) != len(vector_b):
        raise ValueError(
            f"Vectors must have same dimension. Got {len(vector_a)} and {len(vector_b)}"
        )

    if not vector_a:
        raise ValueError("Vectors cannot be empty")

    # Yield for large vectors
    if len(vector_a) > 1000:
        await asyncio.sleep(0)

    return [float(a - b) for a, b in zip(vector_a, vector_b)]


@mcp_function(
    description="Multiply two vectors element-wise (Hadamard product)",
    namespace="linear_algebra.vectors",
    cache_strategy="memory",
    estimated_cpu_usage="low",
)
async def element_wise_multiply(
    vector_a: List[Union[int, float]], vector_b: List[Union[int, float]]
) -> List[float]:
    """
    Multiply two vectors element-wise (Hadamard product).

    Formula:
        a ⊙ b = [a₁×b₁, a₂×b₂, ..., aₙ×bₙ]

    Args:
        vector_a: First vector
        vector_b: Second vector

    Returns:
        Element-wise product vector

    Raises:
        ValueError: If vectors have different dimensions

    Examples:
        >>> await element_wise_multiply([1, 2, 3], [4, 5, 6])
        [4.0, 10.0, 18.0]
    """
    if len(vector_a) != len(vector_b):
        raise ValueError(
            f"Vectors must have same dimension. Got {len(vector_a)} and {len(vector_b)}"
        )

    if not vector_a:
        raise ValueError("Vectors cannot be empty")

    # Yield for large vectors
    if len(vector_a) > 1000:
        await asyncio.sleep(0)

    return [float(a * b) for a, b in zip(vector_a, vector_b)]


@mcp_function(
    description="Divide two vectors element-wise",
    namespace="linear_algebra.vectors",
    cache_strategy="memory",
    estimated_cpu_usage="low",
)
async def element_wise_divide(
    vector_a: List[Union[int, float]], vector_b: List[Union[int, float]]
) -> List[float]:
    """
    Divide vector_a by vector_b element-wise.

    Formula:
        a ⊘ b = [a₁/b₁, a₂/b₂, ..., aₙ/bₙ]

    Args:
        vector_a: Numerator vector
        vector_b: Denominator vector

    Returns:
        Element-wise quotient vector

    Raises:
        ValueError: If vectors have different dimensions
        ZeroDivisionError: If any element in vector_b is zero

    Examples:
        >>> await element_wise_divide([10, 20, 30], [2, 4, 5])
        [5.0, 5.0, 6.0]
    """
    if len(vector_a) != len(vector_b):
        raise ValueError(
            f"Vectors must have same dimension. Got {len(vector_a)} and {len(vector_b)}"
        )

    if not vector_a:
        raise ValueError("Vectors cannot be empty")

    # Check for zero division
    if any(b == 0 for b in vector_b):
        raise ZeroDivisionError("Cannot divide by zero in vector_b")

    # Yield for large vectors
    if len(vector_a) > 1000:
        await asyncio.sleep(0)

    return [float(a / b) for a, b in zip(vector_a, vector_b)]
