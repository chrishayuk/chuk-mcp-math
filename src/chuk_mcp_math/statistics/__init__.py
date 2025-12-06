"""Statistics module: Descriptive and Inferential Statistics.

This module provides comprehensive statistical functions including:
- Descriptive statistics: mean, median, mode, variance, standard deviation, etc.
- Inferential statistics: hypothesis testing, confidence intervals, effect sizes, etc.
"""

# Import descriptive statistics
from .descriptive import (
    mean,
    median,
    mode,
    variance,
    standard_deviation,
    range_value,
    comprehensive_stats,
    percentile,
    quartiles,
    covariance,
    correlation,
    linear_regression,
    moving_average,
    z_scores,
    detect_outliers,
)

# Import inferential statistics
from .inference import (
    t_test_one_sample,
    t_test_two_sample,
    paired_t_test,
    z_test,
    chi_square_test,
    confidence_interval_mean,
    confidence_interval_proportion,
    cohens_d,
    effect_size_r,
    anova_one_way,
    proportion_test,
    sample_size_mean,
    power_analysis,
    mann_whitney_u,
    wilcoxon_signed_rank,
    kruskal_wallis,
    fishers_exact_test,
    permutation_test,
    bootstrap_confidence_interval,
    levenes_test,
)

__all__ = [
    # Descriptive statistics
    "mean",
    "median",
    "mode",
    "variance",
    "standard_deviation",
    "range_value",
    "comprehensive_stats",
    "percentile",
    "quartiles",
    "covariance",
    "correlation",
    "linear_regression",
    "moving_average",
    "z_scores",
    "detect_outliers",
    # Inferential statistics
    "t_test_one_sample",
    "t_test_two_sample",
    "paired_t_test",
    "z_test",
    "chi_square_test",
    "confidence_interval_mean",
    "confidence_interval_proportion",
    "cohens_d",
    "effect_size_r",
    "anova_one_way",
    "proportion_test",
    "sample_size_mean",
    "power_analysis",
    "mann_whitney_u",
    "wilcoxon_signed_rank",
    "kruskal_wallis",
    "fishers_exact_test",
    "permutation_test",
    "bootstrap_confidence_interval",
    "levenes_test",
]
