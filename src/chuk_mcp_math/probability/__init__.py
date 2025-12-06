"""
Probability Module

Probability distributions, sampling, and statistical functions.
All functions are async-native and MCP-decorated for AI model integration.
"""

from .distributions import (
    normal_pdf,
    normal_cdf,
    normal_sample,
    uniform_sample,
)

from .discrete_distributions import (
    exponential_pdf,
    exponential_cdf,
    exponential_sample,
    binomial_pmf,
    binomial_cdf,
    binomial_sample,
)

__all__ = [
    # Normal distribution
    "normal_pdf",
    "normal_cdf",
    "normal_sample",
    # Uniform distribution
    "uniform_sample",
    # Exponential distribution
    "exponential_pdf",
    "exponential_cdf",
    "exponential_sample",
    # Binomial distribution
    "binomial_pmf",
    "binomial_cdf",
    "binomial_sample",
]
