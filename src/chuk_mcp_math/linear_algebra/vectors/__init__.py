"""
Vector Operations Module

Core vector operations including arithmetic, norms, projections, and geometric operations.
All functions are async-native and MCP-decorated for AI model integration.
"""

from .operations import (
    dot_product,
    cross_product,
    scalar_multiply,
    vector_add,
    vector_subtract,
    element_wise_multiply,
    element_wise_divide,
)

from .norms import (
    vector_norm,
    normalize_vector,
    euclidean_norm,
    manhattan_norm,
    chebyshev_norm,
    p_norm,
)

from .projections import (
    vector_projection,
    vector_rejection,
    scalar_projection,
    orthogonalize,
    gram_schmidt,
)

from .geometric import (
    vector_angle,
    vectors_parallel,
    vectors_orthogonal,
    triple_scalar_product,
    triple_vector_product,
)

__all__ = [
    # Operations
    "dot_product",
    "cross_product",
    "scalar_multiply",
    "vector_add",
    "vector_subtract",
    "element_wise_multiply",
    "element_wise_divide",
    # Norms
    "vector_norm",
    "normalize_vector",
    "euclidean_norm",
    "manhattan_norm",
    "chebyshev_norm",
    "p_norm",
    # Projections
    "vector_projection",
    "vector_rejection",
    "scalar_projection",
    "orthogonalize",
    "gram_schmidt",
    # Geometric
    "vector_angle",
    "vectors_parallel",
    "vectors_orthogonal",
    "triple_scalar_product",
    "triple_vector_product",
]
