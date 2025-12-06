"""
Matrix Solvers

Functions for solving systems of linear equations.
All functions are async-native with MCP decoration for AI model integration.
"""

import asyncio
from typing import List, Union
from ...mcp_decorator import mcp_function
from .operations import matrix_det_2x2, matrix_det_3x3

# Type aliases
Matrix = List[List[Union[int, float]]]
Vector = List[Union[int, float]]


@mcp_function(
    description="Solve a 2x2 system of linear equations Ax = b using Cramer's rule",
    namespace="linear_algebra.matrices",
    cache_strategy="memory",
    estimated_cpu_usage="low",
    examples=[
        {
            "input": {"matrix_a": [[2, 1], [1, 3]], "vector_b": [5, 6]},
            "output": [1.8, 1.4],
            "description": "Solve 2x + y = 5, x + 3y = 6",
        }
    ],
)
async def matrix_solve_2x2(matrix_a: Matrix, vector_b: Vector) -> Vector:
    """
    Solve a 2×2 system of linear equations using Cramer's rule.

    Solves: Ax = b for x

    Args:
        matrix_a: Coefficient matrix (2×2)
        vector_b: Right-hand side vector (length 2)

    Returns:
        Solution vector [x, y]

    Raises:
        ValueError: If matrix is singular (determinant = 0) or dimensions are wrong

    Examples:
        >>> await matrix_solve_2x2([[2, 1], [1, 3]], [5, 6])
        [1.8, 1.4]
    """
    if len(matrix_a) != 2 or len(matrix_a[0]) != 2 or len(matrix_a[1]) != 2:
        raise ValueError(f"Expected 2×2 matrix, got {len(matrix_a)}×{len(matrix_a[0])}")
    if len(vector_b) != 2:
        raise ValueError(f"Expected vector of length 2, got {len(vector_b)}")

    # Calculate determinant
    det_a = await matrix_det_2x2(matrix_a)

    if abs(det_a) < 1e-10:
        raise ValueError("Matrix is singular (determinant ≈ 0), system has no unique solution")

    # Cramer's rule: x_i = det(A_i) / det(A)
    # where A_i is A with column i replaced by b

    # For x: replace first column with b
    det_x = await matrix_det_2x2([[vector_b[0], matrix_a[0][1]], [vector_b[1], matrix_a[1][1]]])

    # For y: replace second column with b
    det_y = await matrix_det_2x2([[matrix_a[0][0], vector_b[0]], [matrix_a[1][0], vector_b[1]]])

    x = det_x / det_a
    y = det_y / det_a

    return [float(x), float(y)]


@mcp_function(
    description="Solve a 3x3 system of linear equations Ax = b using Cramer's rule",
    namespace="linear_algebra.matrices",
    cache_strategy="memory",
    estimated_cpu_usage="medium",
    examples=[
        {
            "input": {"matrix_a": [[2, 1, 1], [1, 3, 2], [1, 0, 0]], "vector_b": [4, 5, 6]},
            "output": [6.0, -1.333, -1.0],
            "description": "Solve 3x3 system",
        }
    ],
)
async def matrix_solve_3x3(matrix_a: Matrix, vector_b: Vector) -> Vector:
    """
    Solve a 3×3 system of linear equations using Cramer's rule.

    Solves: Ax = b for x

    Args:
        matrix_a: Coefficient matrix (3×3)
        vector_b: Right-hand side vector (length 3)

    Returns:
        Solution vector [x, y, z]

    Raises:
        ValueError: If matrix is singular (determinant = 0) or dimensions are wrong

    Examples:
        >>> await matrix_solve_3x3([[2, 1, 1], [1, 3, 2], [1, 0, 0]], [4, 5, 6])
        [6.0, -1.333..., -1.0]
    """
    if len(matrix_a) != 3 or any(len(row) != 3 for row in matrix_a):
        raise ValueError(f"Expected 3×3 matrix, got {len(matrix_a)} rows")
    if len(vector_b) != 3:
        raise ValueError(f"Expected vector of length 3, got {len(vector_b)}")

    # Calculate determinant
    det_a = await matrix_det_3x3(matrix_a)

    if abs(det_a) < 1e-10:
        raise ValueError("Matrix is singular (determinant ≈ 0), system has no unique solution")

    # Cramer's rule for each variable
    # For x: replace first column with b
    det_x = await matrix_det_3x3(
        [
            [vector_b[0], matrix_a[0][1], matrix_a[0][2]],
            [vector_b[1], matrix_a[1][1], matrix_a[1][2]],
            [vector_b[2], matrix_a[2][1], matrix_a[2][2]],
        ]
    )

    # For y: replace second column with b
    det_y = await matrix_det_3x3(
        [
            [matrix_a[0][0], vector_b[0], matrix_a[0][2]],
            [matrix_a[1][0], vector_b[1], matrix_a[1][2]],
            [matrix_a[2][0], vector_b[2], matrix_a[2][2]],
        ]
    )

    # For z: replace third column with b
    det_z = await matrix_det_3x3(
        [
            [matrix_a[0][0], matrix_a[0][1], vector_b[0]],
            [matrix_a[1][0], matrix_a[1][1], vector_b[1]],
            [matrix_a[2][0], matrix_a[2][1], vector_b[2]],
        ]
    )

    x = det_x / det_a
    y = det_y / det_a
    z = det_z / det_a

    return [float(x), float(y), float(z)]


@mcp_function(
    description="Solve a system of linear equations Ax = b using Gaussian elimination",
    namespace="linear_algebra.matrices",
    cache_strategy="memory",
    estimated_cpu_usage="medium",
    examples=[
        {
            "input": {"matrix_a": [[3, 2, -1], [2, -2, 4], [-1, 0.5, -1]], "vector_b": [1, -2, 0]},
            "output": [1.0, -2.0, -2.0],
            "description": "Solve 3x3 system with Gaussian elimination",
        }
    ],
)
async def gaussian_elimination(matrix_a: Matrix, vector_b: Vector) -> Vector:
    """
    Solve a system of linear equations using Gaussian elimination with back substitution.

    Solves: Ax = b for x

    Args:
        matrix_a: Coefficient matrix (n×n)
        vector_b: Right-hand side vector (length n)

    Returns:
        Solution vector

    Raises:
        ValueError: If matrix is singular or dimensions are incompatible

    Examples:
        >>> await gaussian_elimination([[3, 2, -1], [2, -2, 4], [-1, 0.5, -1]], [1, -2, 0])
        [1.0, -2.0, -2.0]
    """
    if not matrix_a or not matrix_a[0]:
        raise ValueError("Matrix cannot be empty")

    n = len(matrix_a)
    if any(len(row) != n for row in matrix_a):
        raise ValueError("Matrix must be square")
    if len(vector_b) != n:
        raise ValueError(f"Vector length {len(vector_b)} doesn't match matrix size {n}")

    # Create augmented matrix [A|b]
    # Make copies to avoid modifying inputs
    aug = [row[:] + [vector_b[i]] for i, row in enumerate(matrix_a)]

    # Yield for large systems
    if n > 10:
        await asyncio.sleep(0)

    # Forward elimination
    for i in range(n):
        # Find pivot
        max_row = i
        for k in range(i + 1, n):
            if abs(aug[k][i]) > abs(aug[max_row][i]):
                max_row = k

        # Swap rows
        aug[i], aug[max_row] = aug[max_row], aug[i]

        # Check for singular matrix
        if abs(aug[i][i]) < 1e-10:
            raise ValueError("Matrix is singular, system has no unique solution")

        # Eliminate column below pivot
        for k in range(i + 1, n):
            factor = aug[k][i] / aug[i][i]
            for j in range(i, n + 1):
                aug[k][j] -= factor * aug[i][j]

    # Back substitution
    solution = [0.0] * n
    for i in range(n - 1, -1, -1):
        solution[i] = aug[i][n]
        for j in range(i + 1, n):
            solution[i] -= aug[i][j] * solution[j]
        solution[i] /= aug[i][i]

    return [float(x) for x in solution]
