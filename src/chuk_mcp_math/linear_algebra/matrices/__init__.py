"""
Matrix Operations Module

Core matrix operations including arithmetic, properties, decompositions, and special matrices.
All functions are async-native and MCP-decorated for AI model integration.
"""

# Matrix operations
from .operations import (
    matrix_add,
    matrix_subtract,
    matrix_multiply,
    matrix_scalar_multiply,
    matrix_transpose,
    matrix_det_2x2,
    matrix_det_3x3,
)

# Matrix solvers
from .solvers import (
    matrix_solve_2x2,
    matrix_solve_3x3,
    gaussian_elimination,
)

# TODO: Implement matrix properties
# from .properties import (
#     matrix_trace,
#     matrix_rank,
#     matrix_inverse,
#     is_square,
#     is_symmetric,
#     is_diagonal,
#     is_orthogonal,
#     is_identity,
#     matrix_norm,
# )

# TODO: Implement special matrices
# from .special import (
#     identity_matrix,
#     zero_matrix,
#     ones_matrix,
#     diagonal_matrix,
#     random_matrix,
#     rotation_matrix_2d,
#     rotation_matrix_3d,
# )

__all__: list[str] = [
    # Operations
    "matrix_add",
    "matrix_subtract",
    "matrix_multiply",
    "matrix_scalar_multiply",
    "matrix_transpose",
    "matrix_det_2x2",
    "matrix_det_3x3",
    # Solvers
    "matrix_solve_2x2",
    "matrix_solve_3x3",
    "gaussian_elimination",
    # Properties (planned)
    # "matrix_trace",
    # "matrix_rank",
    # "matrix_inverse",
    # "is_square",
    # "is_symmetric",
    # "is_diagonal",
    # "is_orthogonal",
    # "is_identity",
    # "matrix_norm",
    # Special matrices (planned)
    # "identity_matrix",
    # "zero_matrix",
    # "ones_matrix",
    # "diagonal_matrix",
    # "random_matrix",
    # "rotation_matrix_2d",
    # "rotation_matrix_3d",
]
