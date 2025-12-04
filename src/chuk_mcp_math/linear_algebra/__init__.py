"""
Linear Algebra Module for CHUK MCP Math

Comprehensive linear algebra operations including vectors, matrices,
decompositions, solvers, and transformations. All functions are async-native
and optimized for AI model execution.
"""

from .vectors import (
    # Basic operations
    dot_product,
    cross_product,
    scalar_multiply,
    vector_add,
    vector_subtract,
    # Norms
    vector_norm,
    normalize_vector,
    # Projections
    vector_projection,
    vector_rejection,
    # Angles
    vector_angle,
    vectors_parallel,
    vectors_orthogonal,
)

# from .matrices import (
#     # Basic operations
#     matrix_add,
#     matrix_subtract,
#     matrix_multiply,
#     matrix_scalar_multiply,
#     matrix_transpose,
#     # Properties
#     matrix_determinant,
#     matrix_trace,
#     matrix_rank,
#     matrix_inverse,
#     # Special matrices
#     identity_matrix,
#     zero_matrix,
#     diagonal_matrix,
#     is_symmetric,
#     is_orthogonal,
# )

__all__ = [
    # Vector operations
    "dot_product",
    "cross_product",
    "scalar_multiply",
    "vector_add",
    "vector_subtract",
    "vector_norm",
    "normalize_vector",
    "vector_projection",
    "vector_rejection",
    "vector_angle",
    "vectors_parallel",
    "vectors_orthogonal",
    # Matrix operations - commented out until implemented
    # "matrix_add",
    # "matrix_subtract",
    # "matrix_multiply",
    # "matrix_scalar_multiply",
    # "matrix_transpose",
    # "matrix_determinant",
    # "matrix_trace",
    # "matrix_rank",
    # "matrix_inverse",
    # "identity_matrix",
    # "zero_matrix",
    # "diagonal_matrix",
    # "is_symmetric",
    # "is_orthogonal",
]
