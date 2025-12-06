"""
Matrix Operations

Basic matrix arithmetic operations including multiplication, transpose, and basic properties.
All functions are async-native with MCP decoration for AI model integration.
"""

import asyncio
from typing import List, Union
from ...mcp_decorator import mcp_function

# Type alias for matrices
Matrix = List[List[Union[int, float]]]


@mcp_function(
    description="Multiply two matrices (matrix-matrix multiplication)",
    namespace="linear_algebra.matrices",
    cache_strategy="memory",
    cache_ttl_seconds=3600,
    estimated_cpu_usage="medium",
    examples=[
        {
            "input": {"matrix_a": [[1, 2], [3, 4]], "matrix_b": [[5, 6], [7, 8]]},
            "output": [[19, 22], [43, 50]],
            "description": "Multiply two 2x2 matrices",
        }
    ],
)
async def matrix_multiply(matrix_a: Matrix, matrix_b: Matrix) -> Matrix:
    """
    Multiply two matrices using standard matrix multiplication.

    Formula:
        C[i][j] = Σ(A[i][k] × B[k][j]) for k = 1 to n

    Args:
        matrix_a: First matrix (m × n)
        matrix_b: Second matrix (n × p)

    Returns:
        Product matrix (m × p)

    Raises:
        ValueError: If matrices are incompatible for multiplication

    Examples:
        >>> await matrix_multiply([[1, 2], [3, 4]], [[5, 6], [7, 8]])
        [[19, 22], [43, 50]]
    """
    if not matrix_a or not matrix_a[0]:
        raise ValueError("First matrix cannot be empty")
    if not matrix_b or not matrix_b[0]:
        raise ValueError("Second matrix cannot be empty")

    rows_a = len(matrix_a)
    cols_a = len(matrix_a[0])
    rows_b = len(matrix_b)
    cols_b = len(matrix_b[0])

    if cols_a != rows_b:
        raise ValueError(
            f"Matrix dimensions incompatible for multiplication: "
            f"({rows_a}×{cols_a}) × ({rows_b}×{cols_b})"
        )

    # Yield for large matrices
    if rows_a * cols_b > 100:
        await asyncio.sleep(0)

    result = []
    for i in range(rows_a):
        row = []
        for j in range(cols_b):
            # Calculate dot product of row i from A and column j from B
            value = sum(matrix_a[i][k] * matrix_b[k][j] for k in range(cols_a))
            row.append(float(value))
        result.append(row)

    return result


@mcp_function(
    description="Transpose a matrix (swap rows and columns)",
    namespace="linear_algebra.matrices",
    cache_strategy="memory",
    estimated_cpu_usage="low",
)
async def matrix_transpose(matrix: Matrix) -> Matrix:
    """
    Transpose a matrix by swapping rows and columns.

    Formula:
        B[j][i] = A[i][j]

    Args:
        matrix: Input matrix (m × n)

    Returns:
        Transposed matrix (n × m)

    Raises:
        ValueError: If matrix is empty

    Examples:
        >>> await matrix_transpose([[1, 2, 3], [4, 5, 6]])
        [[1, 4], [2, 5], [3, 6]]
    """
    if not matrix or not matrix[0]:
        raise ValueError("Matrix cannot be empty")

    rows = len(matrix)
    cols = len(matrix[0])

    # Yield for large matrices
    if rows * cols > 1000:
        await asyncio.sleep(0)

    return [[float(matrix[i][j]) for i in range(rows)] for j in range(cols)]


@mcp_function(
    description="Calculate the determinant of a 2x2 matrix",
    namespace="linear_algebra.matrices",
    cache_strategy="memory",
    estimated_cpu_usage="low",
)
async def matrix_det_2x2(matrix: Matrix) -> float:
    """
    Calculate the determinant of a 2×2 matrix.

    Formula:
        det([[a, b], [c, d]]) = ad - bc

    Args:
        matrix: 2×2 matrix

    Returns:
        Determinant value

    Raises:
        ValueError: If matrix is not 2×2

    Examples:
        >>> await matrix_det_2x2([[1, 2], [3, 4]])
        -2.0
    """
    if len(matrix) != 2 or len(matrix[0]) != 2 or len(matrix[1]) != 2:
        raise ValueError(f"Expected 2×2 matrix, got {len(matrix)}×{len(matrix[0])}")

    return float(matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0])


@mcp_function(
    description="Calculate the determinant of a 3x3 matrix",
    namespace="linear_algebra.matrices",
    cache_strategy="memory",
    estimated_cpu_usage="low",
)
async def matrix_det_3x3(matrix: Matrix) -> float:
    """
    Calculate the determinant of a 3×3 matrix using the rule of Sarrus.

    Formula:
        det = a(ei−fh) − b(di−fg) + c(dh−eg)

    Args:
        matrix: 3×3 matrix

    Returns:
        Determinant value

    Raises:
        ValueError: If matrix is not 3×3

    Examples:
        >>> await matrix_det_3x3([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        0.0
    """
    if len(matrix) != 3 or any(len(row) != 3 for row in matrix):
        raise ValueError(f"Expected 3×3 matrix, got {len(matrix)} rows")

    a, b, c = matrix[0]
    d, e, f = matrix[1]
    g, h, i = matrix[2]

    det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
    return float(det)


@mcp_function(
    description="Multiply a matrix by a scalar value",
    namespace="linear_algebra.matrices",
    cache_strategy="memory",
    estimated_cpu_usage="low",
)
async def matrix_scalar_multiply(matrix: Matrix, scalar: Union[int, float]) -> Matrix:
    """
    Multiply every element of a matrix by a scalar.

    Formula:
        B[i][j] = c × A[i][j]

    Args:
        matrix: Input matrix
        scalar: Scalar multiplier

    Returns:
        Scaled matrix

    Raises:
        ValueError: If matrix is empty

    Examples:
        >>> await matrix_scalar_multiply([[1, 2], [3, 4]], 2)
        [[2.0, 4.0], [6.0, 8.0]]
    """
    if not matrix or not matrix[0]:
        raise ValueError("Matrix cannot be empty")

    # Yield for large matrices
    if len(matrix) * len(matrix[0]) > 1000:
        await asyncio.sleep(0)

    return [[float(scalar * val) for val in row] for row in matrix]


@mcp_function(
    description="Add two matrices element-wise",
    namespace="linear_algebra.matrices",
    cache_strategy="memory",
    estimated_cpu_usage="low",
)
async def matrix_add(matrix_a: Matrix, matrix_b: Matrix) -> Matrix:
    """
    Add two matrices element-wise.

    Formula:
        C[i][j] = A[i][j] + B[i][j]

    Args:
        matrix_a: First matrix
        matrix_b: Second matrix

    Returns:
        Sum matrix

    Raises:
        ValueError: If matrices have different dimensions

    Examples:
        >>> await matrix_add([[1, 2], [3, 4]], [[5, 6], [7, 8]])
        [[6.0, 8.0], [10.0, 12.0]]
    """
    if not matrix_a or not matrix_a[0]:
        raise ValueError("First matrix cannot be empty")
    if not matrix_b or not matrix_b[0]:
        raise ValueError("Second matrix cannot be empty")

    rows_a, cols_a = len(matrix_a), len(matrix_a[0])
    rows_b, cols_b = len(matrix_b), len(matrix_b[0])

    if rows_a != rows_b or cols_a != cols_b:
        raise ValueError(
            f"Matrices must have same dimensions for addition: "
            f"({rows_a}×{cols_a}) vs ({rows_b}×{cols_b})"
        )

    # Yield for large matrices
    if rows_a * cols_a > 1000:
        await asyncio.sleep(0)

    return [[float(matrix_a[i][j] + matrix_b[i][j]) for j in range(cols_a)] for i in range(rows_a)]


@mcp_function(
    description="Subtract one matrix from another element-wise",
    namespace="linear_algebra.matrices",
    cache_strategy="memory",
    estimated_cpu_usage="low",
)
async def matrix_subtract(matrix_a: Matrix, matrix_b: Matrix) -> Matrix:
    """
    Subtract matrix_b from matrix_a element-wise.

    Formula:
        C[i][j] = A[i][j] - B[i][j]

    Args:
        matrix_a: Matrix to subtract from
        matrix_b: Matrix to subtract

    Returns:
        Difference matrix

    Raises:
        ValueError: If matrices have different dimensions

    Examples:
        >>> await matrix_subtract([[5, 6], [7, 8]], [[1, 2], [3, 4]])
        [[4.0, 4.0], [4.0, 4.0]]
    """
    if not matrix_a or not matrix_a[0]:
        raise ValueError("First matrix cannot be empty")
    if not matrix_b or not matrix_b[0]:
        raise ValueError("Second matrix cannot be empty")

    rows_a, cols_a = len(matrix_a), len(matrix_a[0])
    rows_b, cols_b = len(matrix_b), len(matrix_b[0])

    if rows_a != rows_b or cols_a != cols_b:
        raise ValueError(
            f"Matrices must have same dimensions for subtraction: "
            f"({rows_a}×{cols_a}) vs ({rows_b}×{cols_b})"
        )

    # Yield for large matrices
    if rows_a * cols_a > 1000:
        await asyncio.sleep(0)

    return [[float(matrix_a[i][j] - matrix_b[i][j]) for j in range(cols_a)] for i in range(rows_a)]
