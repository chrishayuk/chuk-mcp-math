"""Numerical methods and algorithms."""

from .interpolation import (
    linear_interpolate,
    linear_interpolate_sequence,
    lagrange_interpolate,
    newton_interpolate,
    cubic_spline_coefficients,
    cubic_spline_interpolate,
    bilinear_interpolate,
)
from .optimization import (
    gradient_descent,
    gradient_descent_momentum,
    adam_optimizer,
    golden_section_search,
    nelder_mead,
    coordinate_descent,
)
from .series import (
    taylor_series,
    power_series,
    horner_method,
    fourier_series_approximation,
    maclaurin_series,
    binomial_series,
    geometric_series,
    arithmetic_series,
    exp_series,
    sin_series,
    cos_series,
    ln_series,
)

__all__ = [
    # Interpolation
    "linear_interpolate",
    "linear_interpolate_sequence",
    "lagrange_interpolate",
    "newton_interpolate",
    "cubic_spline_coefficients",
    "cubic_spline_interpolate",
    "bilinear_interpolate",
    # Optimization
    "gradient_descent",
    "gradient_descent_momentum",
    "adam_optimizer",
    "golden_section_search",
    "nelder_mead",
    "coordinate_descent",
    # Series
    "taylor_series",
    "power_series",
    "horner_method",
    "fourier_series_approximation",
    "maclaurin_series",
    "binomial_series",
    "geometric_series",
    "arithmetic_series",
    "exp_series",
    "sin_series",
    "cos_series",
    "ln_series",
]
