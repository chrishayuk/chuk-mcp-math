"""
Matrix Operations Module

Core matrix operations including arithmetic, properties, decompositions, and special matrices.
All functions are async-native and MCP-decorated for AI model integration.
"""

from .operations import (
    matrix_add,
    matrix_subtract,
    matrix_multiply,
    matrix_scalar_multiply,
    matrix_transpose,
    matrix_power,
    element_wise_multiply,
    element_wise_divide,
)

from .properties import (
    matrix_determinant,
    matrix_trace,
    matrix_rank,
    matrix_inverse,
    is_square,
    is_symmetric,
    is_diagonal,
    is_orthogonal,
    is_identity,
    matrix_norm,
)

from .special import (
    identity_matrix,
    zero_matrix,
    ones_matrix,
    diagonal_matrix,
    random_matrix,
    rotation_matrix_2d,
    rotation_matrix_3d,
)

__all__ = [
    # Operations
    "matrix_add",
    "matrix_subtract",
    "matrix_multiply",
    "matrix_scalar_multiply",
    "matrix_transpose",
    "matrix_power",
    "element_wise_multiply",
    "element_wise_divide",
    # Properties
    "matrix_determinant",
    "matrix_trace",
    "matrix_rank",
    "matrix_inverse",
    "is_square",
    "is_symmetric",
    "is_diagonal",
    "is_orthogonal",
    "is_identity",
    "matrix_norm",
    # Special matrices
    "identity_matrix",
    "zero_matrix",
    "ones_matrix",
    "diagonal_matrix",
    "random_matrix",
    "rotation_matrix_2d",
    "rotation_matrix_3d",
]