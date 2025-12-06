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

from .matrices import (
    # Basic operations
    matrix_add,
    matrix_subtract,
    matrix_multiply,
    matrix_scalar_multiply,
    matrix_transpose,
    matrix_det_2x2,
    matrix_det_3x3,
    # Solvers
    matrix_solve_2x2,
    matrix_solve_3x3,
    gaussian_elimination,
)

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
    # Matrix operations
    "matrix_add",
    "matrix_subtract",
    "matrix_multiply",
    "matrix_scalar_multiply",
    "matrix_transpose",
    "matrix_det_2x2",
    "matrix_det_3x3",
    # Matrix solvers
    "matrix_solve_2x2",
    "matrix_solve_3x3",
    "gaussian_elimination",
]
