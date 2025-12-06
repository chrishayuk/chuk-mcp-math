#!/usr/bin/env python3
"""Inferential statistics for hypothesis testing and statistical inference.

This module provides comprehensive tools for statistical inference including
hypothesis testing, confidence intervals, effect sizes, and power analysis.
Essential for A/B testing, scientific research, and data-driven decision making.
"""

import asyncio
import math
from typing import Dict, List, Any


async def t_test_one_sample(
    data: List[float], population_mean: float, alpha: float = 0.05
) -> Dict[str, Any]:
    """
    One-sample t-test.

    Tests whether sample mean differs significantly from population mean.

    Args:
        data: Sample data
        population_mean: Hypothesized population mean
        alpha: Significance level (default 0.05)

    Returns:
        Dictionary with t_statistic, p_value, degrees_of_freedom, reject_null

    Raises:
        ValueError: If data has < 2 observations or alpha not in (0, 1)

    Example:
        >>> data = [23, 25, 27, 24, 26]
        >>> result = await t_test_one_sample(data, 20.0)
        >>> # Tests if sample mean differs from 20
    """
    if len(data) < 2:
        raise ValueError("data must have at least 2 observations")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")

    n = len(data)
    sample_mean = sum(data) / n
    sample_var = sum((x - sample_mean) ** 2 for x in data) / (n - 1)
    sample_std = math.sqrt(sample_var)

    # t-statistic
    se = sample_std / math.sqrt(n)
    if se == 0:
        # All values are identical
        if abs(sample_mean - population_mean) < 1e-10:
            t_stat = 0.0
        else:
            t_stat = float("inf") if sample_mean > population_mean else float("-inf")
    else:
        t_stat = (sample_mean - population_mean) / se
    df = n - 1

    # Two-tailed p-value approximation using normal distribution (for large n)
    # For small n, this is approximate; exact requires t-distribution
    p_value = 2 * (1 - _normal_cdf(abs(t_stat)))

    # Critical value approximation (using normal for simplicity)
    critical_value = _normal_inverse_cdf(1 - alpha / 2)

    await asyncio.sleep(0)

    return {
        "t_statistic": t_stat,
        "p_value": p_value,
        "degrees_of_freedom": df,
        "reject_null": abs(t_stat) > critical_value,
        "sample_mean": sample_mean,
        "critical_value": critical_value,
    }


async def t_test_two_sample(
    data1: List[float],
    data2: List[float],
    alpha: float = 0.05,
    equal_variance: bool = True,
) -> Dict[str, Any]:
    """
    Two-sample t-test (independent samples).

    Tests whether two sample means differ significantly.

    Args:
        data1: First sample
        data2: Second sample
        alpha: Significance level
        equal_variance: Assume equal variance (default True)

    Returns:
        Dictionary with t_statistic, p_value, degrees_of_freedom, reject_null

    Raises:
        ValueError: If either sample has < 2 observations

    Example:
        >>> control = [20, 22, 21, 23, 22]
        >>> treatment = [25, 27, 26, 28, 27]
        >>> result = await t_test_two_sample(control, treatment)
    """
    if len(data1) < 2 or len(data2) < 2:
        raise ValueError("both samples must have at least 2 observations")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")

    n1, n2 = len(data1), len(data2)
    mean1 = sum(data1) / n1
    mean2 = sum(data2) / n2
    var1 = sum((x - mean1) ** 2 for x in data1) / (n1 - 1)
    var2 = sum((x - mean2) ** 2 for x in data2) / (n2 - 1)

    df: float
    if equal_variance:
        # Pooled variance
        pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
        se = math.sqrt(pooled_var * (1 / n1 + 1 / n2))
        df = float(n1 + n2 - 2)
    else:
        # Welch's t-test (unequal variance)
        se = math.sqrt(var1 / n1 + var2 / n2)
        # Welch-Satterthwaite df
        df = (var1 / n1 + var2 / n2) ** 2 / (
            (var1 / n1) ** 2 / (n1 - 1) + (var2 / n2) ** 2 / (n2 - 1)
        )

    t_stat = (mean1 - mean2) / se
    p_value = 2 * (1 - _normal_cdf(abs(t_stat)))
    critical_value = _normal_inverse_cdf(1 - alpha / 2)

    await asyncio.sleep(0)

    return {
        "t_statistic": t_stat,
        "p_value": p_value,
        "degrees_of_freedom": df,
        "reject_null": abs(t_stat) > critical_value,
        "mean_difference": mean1 - mean2,
    }


async def paired_t_test(
    before: List[float], after: List[float], alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Paired t-test (dependent samples).

    Tests whether paired observations differ significantly.

    Args:
        before: Before measurements
        after: After measurements
        alpha: Significance level

    Returns:
        Dictionary with t_statistic, p_value, degrees_of_freedom, reject_null

    Raises:
        ValueError: If samples have different lengths or < 2 observations

    Example:
        >>> before = [80, 85, 82, 84, 83]
        >>> after = [85, 90, 87, 89, 88]
        >>> result = await paired_t_test(before, after)
    """
    if len(before) != len(after):
        raise ValueError("before and after must have same length")
    if len(before) < 2:
        raise ValueError("need at least 2 pairs")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")

    differences = [a - b for b, a in zip(before, after)]
    return await t_test_one_sample(differences, 0.0, alpha)


async def z_test(
    data: List[float],
    population_mean: float,
    population_std: float,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Z-test for known population standard deviation.

    Tests whether sample mean differs from population mean when σ is known.

    Args:
        data: Sample data
        population_mean: Hypothesized population mean
        population_std: Known population standard deviation
        alpha: Significance level

    Returns:
        Dictionary with z_statistic, p_value, reject_null

    Raises:
        ValueError: If population_std <= 0 or data is empty

    Example:
        >>> data = [105, 110, 108, 107, 109]
        >>> result = await z_test(data, 100.0, 5.0)
    """
    if not data:
        raise ValueError("data cannot be empty")
    if population_std <= 0:
        raise ValueError("population_std must be positive")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")

    n = len(data)
    sample_mean = sum(data) / n
    z_stat = (sample_mean - population_mean) / (population_std / math.sqrt(n))
    p_value = 2 * (1 - _normal_cdf(abs(z_stat)))
    critical_value = _normal_inverse_cdf(1 - alpha / 2)

    await asyncio.sleep(0)

    return {
        "z_statistic": z_stat,
        "p_value": p_value,
        "reject_null": abs(z_stat) > critical_value,
        "sample_mean": sample_mean,
    }


async def chi_square_test(
    observed: List[float], expected: List[float], alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Chi-square goodness-of-fit test.

    Tests whether observed frequencies match expected frequencies.

    Args:
        observed: Observed frequencies
        expected: Expected frequencies
        alpha: Significance level

    Returns:
        Dictionary with chi_square_statistic, p_value, degrees_of_freedom, reject_null

    Raises:
        ValueError: If observed and expected have different lengths or any expected <= 0

    Example:
        >>> observed = [30, 25, 20, 25]
        >>> expected = [25, 25, 25, 25]
        >>> result = await chi_square_test(observed, expected)
    """
    if len(observed) != len(expected):
        raise ValueError("observed and expected must have same length")
    if any(e <= 0 for e in expected):
        raise ValueError("all expected values must be positive")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")

    chi_square = sum((o - e) ** 2 / e for o, e in zip(observed, expected))
    df = len(observed) - 1

    # Approximate p-value using chi-square distribution
    # For simplicity, using a basic approximation
    p_value = 1 - _chi_square_cdf(chi_square, df)

    # Critical value at alpha level (approximation)
    critical_value = _chi_square_inverse(1 - alpha, df)

    await asyncio.sleep(0)

    return {
        "chi_square_statistic": chi_square,
        "p_value": p_value,
        "degrees_of_freedom": df,
        "reject_null": chi_square > critical_value,
    }


async def confidence_interval_mean(
    data: List[float], confidence_level: float = 0.95
) -> Dict[str, Any]:
    """
    Confidence interval for population mean.

    Args:
        data: Sample data
        confidence_level: Confidence level (default 0.95)

    Returns:
        Dictionary with lower, upper, mean, margin_of_error

    Raises:
        ValueError: If data has < 2 observations or confidence_level not in (0, 1)

    Example:
        >>> data = [23, 25, 27, 24, 26]
        >>> result = await confidence_interval_mean(data, 0.95)
    """
    if len(data) < 2:
        raise ValueError("data must have at least 2 observations")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be in (0, 1)")

    n = len(data)
    mean = sum(data) / n
    variance = sum((x - mean) ** 2 for x in data) / (n - 1)
    std = math.sqrt(variance)
    se = std / math.sqrt(n)

    # Using normal approximation for simplicity
    alpha = 1 - confidence_level
    z_critical = _normal_inverse_cdf(1 - alpha / 2)

    margin_of_error = z_critical * se
    lower = mean - margin_of_error
    upper = mean + margin_of_error

    await asyncio.sleep(0)

    return {
        "lower": lower,
        "upper": upper,
        "mean": mean,
        "margin_of_error": margin_of_error,
        "confidence_level": confidence_level,
    }


async def confidence_interval_proportion(
    successes: int, n: int, confidence_level: float = 0.95
) -> Dict[str, Any]:
    """
    Confidence interval for population proportion.

    Args:
        successes: Number of successes
        n: Total sample size
        confidence_level: Confidence level

    Returns:
        Dictionary with lower, upper, proportion, margin_of_error

    Raises:
        ValueError: If n < 1 or successes not in [0, n]

    Example:
        >>> result = await confidence_interval_proportion(60, 100, 0.95)
        >>> # 60 successes out of 100 trials
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    if not 0 <= successes <= n:
        raise ValueError("successes must be in [0, n]")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be in (0, 1)")

    p = successes / n
    alpha = 1 - confidence_level
    z_critical = _normal_inverse_cdf(1 - alpha / 2)

    # Standard error for proportion
    se = math.sqrt(p * (1 - p) / n)
    margin_of_error = z_critical * se

    lower = max(0, p - margin_of_error)
    upper = min(1, p + margin_of_error)

    await asyncio.sleep(0)

    return {
        "lower": lower,
        "upper": upper,
        "proportion": p,
        "margin_of_error": margin_of_error,
        "confidence_level": confidence_level,
    }


async def cohens_d(data1: List[float], data2: List[float]) -> float:
    """
    Cohen's d effect size for two samples.

    Measures standardized difference between two means.

    Args:
        data1: First sample
        data2: Second sample

    Returns:
        Cohen's d (effect size)

    Raises:
        ValueError: If either sample has < 2 observations

    Example:
        >>> control = [20, 22, 21, 23]
        >>> treatment = [25, 27, 26, 28]
        >>> d = await cohens_d(control, treatment)
    """
    if len(data1) < 2 or len(data2) < 2:
        raise ValueError("both samples must have at least 2 observations")

    n1, n2 = len(data1), len(data2)
    mean1 = sum(data1) / n1
    mean2 = sum(data2) / n2
    var1 = sum((x - mean1) ** 2 for x in data1) / (n1 - 1)
    var2 = sum((x - mean2) ** 2 for x in data2) / (n2 - 1)

    # Pooled standard deviation
    pooled_std = math.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    d = (mean1 - mean2) / pooled_std

    await asyncio.sleep(0)
    return d


async def effect_size_r(t_statistic: float, df: float) -> float:
    """
    Effect size r from t-statistic.

    Converts t-statistic to effect size r (correlation coefficient).

    Args:
        t_statistic: t-statistic from t-test
        df: Degrees of freedom

    Returns:
        Effect size r

    Raises:
        ValueError: If df <= 0

    Example:
        >>> r = await effect_size_r(2.5, 18)
    """
    if df <= 0:
        raise ValueError("df must be positive")

    r = math.sqrt(t_statistic**2 / (t_statistic**2 + df))
    await asyncio.sleep(0)
    return r


async def anova_one_way(groups: List[List[float]], alpha: float = 0.05) -> Dict[str, Any]:
    """
    One-way ANOVA (Analysis of Variance).

    Tests whether means of multiple groups differ significantly.

    Args:
        groups: List of groups (each group is a list of observations)
        alpha: Significance level

    Returns:
        Dictionary with F_statistic, p_value, between_variance, within_variance, reject_null

    Raises:
        ValueError: If < 2 groups or any group has < 2 observations

    Example:
        >>> group1 = [20, 22, 21, 23]
        >>> group2 = [25, 27, 26, 28]
        >>> group3 = [30, 32, 31, 33]
        >>> result = await anova_one_way([group1, group2, group3])
    """
    if len(groups) < 2:
        raise ValueError("need at least 2 groups")
    if any(len(g) < 2 for g in groups):
        raise ValueError("all groups must have at least 2 observations")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")

    # Overall mean
    all_data = [x for group in groups for x in group]
    grand_mean = sum(all_data) / len(all_data)

    # Between-group variance (SSB)
    ssb = sum(len(group) * (sum(group) / len(group) - grand_mean) ** 2 for group in groups)
    df_between = len(groups) - 1

    # Within-group variance (SSW)
    ssw = sum(sum((x - sum(group) / len(group)) ** 2 for x in group) for group in groups)
    df_within = len(all_data) - len(groups)

    # Mean squares
    msb = ssb / df_between
    msw = ssw / df_within

    # F-statistic
    f_stat = msb / msw if msw > 0 else float("inf")

    # Approximate p-value
    p_value = 1 - _f_cdf(f_stat, df_between, df_within)

    # Critical value
    critical_value = _f_inverse(1 - alpha, df_between, df_within)

    await asyncio.sleep(0)

    return {
        "F_statistic": f_stat,
        "p_value": p_value,
        "df_between": df_between,
        "df_within": df_within,
        "between_variance": msb,
        "within_variance": msw,
        "reject_null": f_stat > critical_value,
    }


async def proportion_test(
    successes1: int, n1: int, successes2: int, n2: int, alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Two-proportion z-test.

    Tests whether two population proportions differ significantly.

    Args:
        successes1: Successes in sample 1
        n1: Size of sample 1
        successes2: Successes in sample 2
        n2: Size of sample 2
        alpha: Significance level

    Returns:
        Dictionary with z_statistic, p_value, reject_null

    Raises:
        ValueError: If n1 or n2 < 1 or successes not in valid range

    Example:
        >>> result = await proportion_test(60, 100, 45, 100)
        >>> # Compare 60% vs 45% success rates
    """
    if n1 < 1 or n2 < 1:
        raise ValueError("sample sizes must be >= 1")
    if not 0 <= successes1 <= n1 or not 0 <= successes2 <= n2:
        raise ValueError("successes must be in valid range")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")

    p1 = successes1 / n1
    p2 = successes2 / n2

    # Pooled proportion
    p_pooled = (successes1 + successes2) / (n1 + n2)

    # Standard error
    se = math.sqrt(p_pooled * (1 - p_pooled) * (1 / n1 + 1 / n2))

    # Z-statistic
    z_stat = (p1 - p2) / se if se > 0 else float("inf")

    p_value = 2 * (1 - _normal_cdf(abs(z_stat)))
    critical_value = _normal_inverse_cdf(1 - alpha / 2)

    await asyncio.sleep(0)

    return {
        "z_statistic": z_stat,
        "p_value": p_value,
        "reject_null": abs(z_stat) > critical_value,
        "proportion_difference": p1 - p2,
    }


async def sample_size_mean(
    effect_size: float,
    alpha: float = 0.05,
    power: float = 0.8,
    alternative: str = "two-sided",
) -> int:
    """
    Sample size calculation for comparing means.

    Args:
        effect_size: Expected Cohen's d
        alpha: Significance level
        power: Desired statistical power
        alternative: "two-sided" or "one-sided"

    Returns:
        Required sample size per group

    Raises:
        ValueError: If parameters not in valid ranges

    Example:
        >>> n = await sample_size_mean(0.5, alpha=0.05, power=0.8)
    """
    if effect_size <= 0:
        raise ValueError("effect_size must be positive")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    if not 0 < power < 1:
        raise ValueError("power must be in (0, 1)")
    if alternative not in ["two-sided", "one-sided"]:
        raise ValueError("alternative must be 'two-sided' or 'one-sided'")

    # Z-scores
    if alternative == "two-sided":
        z_alpha = _normal_inverse_cdf(1 - alpha / 2)
    else:
        z_alpha = _normal_inverse_cdf(1 - alpha)

    z_beta = _normal_inverse_cdf(power)

    # Sample size formula
    n = 2 * ((z_alpha + z_beta) / effect_size) ** 2

    await asyncio.sleep(0)
    return max(2, int(math.ceil(n)))


async def power_analysis(
    n: int, effect_size: float, alpha: float = 0.05, alternative: str = "two-sided"
) -> float:
    """
    Statistical power calculation for t-test.

    Args:
        n: Sample size per group
        effect_size: Cohen's d
        alpha: Significance level
        alternative: "two-sided" or "one-sided"

    Returns:
        Statistical power (probability of detecting effect)

    Raises:
        ValueError: If parameters not in valid ranges

    Example:
        >>> power = await power_analysis(50, 0.5, alpha=0.05)
    """
    if n < 2:
        raise ValueError("n must be >= 2")
    if effect_size <= 0:
        raise ValueError("effect_size must be positive")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    if alternative not in ["two-sided", "one-sided"]:
        raise ValueError("alternative must be 'two-sided' or 'one-sided'")

    # Non-centrality parameter
    ncp = effect_size * math.sqrt(n / 2)

    # Critical value
    if alternative == "two-sided":
        z_critical = _normal_inverse_cdf(1 - alpha / 2)
    else:
        z_critical = _normal_inverse_cdf(1 - alpha)

    # Power = P(reject H0 | H1 is true)
    power = 1 - _normal_cdf(z_critical - ncp)

    await asyncio.sleep(0)
    return power


async def mann_whitney_u(
    data1: List[float], data2: List[float], alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Mann-Whitney U test (non-parametric alternative to t-test).

    Tests whether two independent samples come from same distribution.

    Args:
        data1: First sample
        data2: Second sample
        alpha: Significance level

    Returns:
        Dictionary with U_statistic, p_value, reject_null

    Raises:
        ValueError: If either sample is empty

    Example:
        >>> sample1 = [20, 22, 21, 23]
        >>> sample2 = [25, 27, 26, 28]
        >>> result = await mann_whitney_u(sample1, sample2)
    """
    if not data1 or not data2:
        raise ValueError("both samples must be non-empty")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")

    n1, n2 = len(data1), len(data2)

    # Rank all values
    combined = [(x, 1) for x in data1] + [(x, 2) for x in data2]
    combined.sort(key=lambda item: item[0])

    # Assign ranks
    ranks1 = []
    for i, (val, group) in enumerate(combined):
        if group == 1:
            ranks1.append(i + 1)

    # U statistic
    r1 = sum(ranks1)
    u1 = r1 - n1 * (n1 + 1) / 2
    u2 = n1 * n2 - u1
    u = min(u1, u2)

    # Normal approximation for large samples
    mean_u = n1 * n2 / 2
    std_u = math.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)

    z = (u - mean_u) / std_u if std_u > 0 else 0
    p_value = 2 * (1 - _normal_cdf(abs(z)))

    critical_value = _normal_inverse_cdf(1 - alpha / 2)

    await asyncio.sleep(0)

    return {
        "U_statistic": u,
        "p_value": p_value,
        "reject_null": abs(z) > critical_value,
    }


async def wilcoxon_signed_rank(
    before: List[float], after: List[float], alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Wilcoxon signed-rank test (non-parametric paired test).

    Tests whether paired samples differ significantly.

    Args:
        before: Before measurements
        after: After measurements
        alpha: Significance level

    Returns:
        Dictionary with W_statistic, p_value, reject_null

    Raises:
        ValueError: If samples have different lengths or are empty

    Example:
        >>> before = [80, 85, 82, 84]
        >>> after = [85, 90, 87, 89]
        >>> result = await wilcoxon_signed_rank(before, after)
    """
    if len(before) != len(after):
        raise ValueError("before and after must have same length")
    if not before:
        raise ValueError("samples cannot be empty")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")

    # Compute differences
    differences = [a - b for b, a in zip(before, after)]

    # Remove zeros
    nonzero_diffs = [(abs(d), 1 if d > 0 else -1) for d in differences if d != 0]

    if not nonzero_diffs:
        return {
            "W_statistic": 0.0,
            "p_value": 1.0,
            "reject_null": False,
        }

    # Rank absolute differences
    nonzero_diffs.sort(key=lambda x: x[0])
    ranks = []
    for i, (val, sign) in enumerate(nonzero_diffs):
        ranks.append((i + 1, sign))

    # Sum of positive ranks
    w_plus = sum(rank for rank, sign in ranks if sign > 0)

    n = len(nonzero_diffs)
    mean_w = n * (n + 1) / 4
    std_w = math.sqrt(n * (n + 1) * (2 * n + 1) / 24)

    # Normal approximation
    z = (w_plus - mean_w) / std_w if std_w > 0 else 0
    p_value = 2 * (1 - _normal_cdf(abs(z)))

    critical_value = _normal_inverse_cdf(1 - alpha / 2)

    await asyncio.sleep(0)

    return {
        "W_statistic": w_plus,
        "p_value": p_value,
        "reject_null": abs(z) > critical_value,
    }


async def kruskal_wallis(groups: List[List[float]], alpha: float = 0.05) -> Dict[str, Any]:
    """
    Kruskal-Wallis test (non-parametric ANOVA).

    Tests whether multiple independent samples come from same distribution.

    Args:
        groups: List of groups (each group is a list of observations)
        alpha: Significance level

    Returns:
        Dictionary with H_statistic, p_value, reject_null

    Raises:
        ValueError: If < 2 groups or any group is empty

    Example:
        >>> group1 = [20, 22, 21]
        >>> group2 = [25, 27, 26]
        >>> group3 = [30, 32, 31]
        >>> result = await kruskal_wallis([group1, group2, group3])
    """
    if len(groups) < 2:
        raise ValueError("need at least 2 groups")
    if any(not g for g in groups):
        raise ValueError("all groups must be non-empty")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")

    # Combine and rank all data
    combined = []
    for group_idx, group in enumerate(groups):
        combined.extend([(val, group_idx) for val in group])

    combined.sort(key=lambda x: x[0])

    # Assign ranks
    group_ranks: List[List[int]] = [[] for _ in groups]
    for i, (val, group_idx) in enumerate(combined):
        group_ranks[group_idx].append(i + 1)

    # Kruskal-Wallis H statistic
    n = len(combined)
    h = 0.0

    for group, ranks in zip(groups, group_ranks):
        n_i = len(group)
        r_i = sum(ranks)
        h += r_i**2 / n_i

    h = (12 / (n * (n + 1))) * h - 3 * (n + 1)

    # Approximate with chi-square distribution
    df = len(groups) - 1
    p_value = 1 - _chi_square_cdf(h, df)
    critical_value = _chi_square_inverse(1 - alpha, df)

    await asyncio.sleep(0)

    return {
        "H_statistic": h,
        "p_value": p_value,
        "degrees_of_freedom": df,
        "reject_null": h > critical_value,
    }


async def fishers_exact_test(a: int, b: int, c: int, d: int, alpha: float = 0.05) -> Dict[str, Any]:
    """
    Fisher's exact test for 2x2 contingency tables.

    Tests association between two categorical variables.

    Args:
        a, b, c, d: Counts in 2x2 table [[a, b], [c, d]]
        alpha: Significance level

    Returns:
        Dictionary with p_value, odds_ratio, reject_null

    Raises:
        ValueError: If any count is negative

    Example:
        >>> # Treatment vs Control, Success vs Failure
        >>> result = await fishers_exact_test(10, 5, 3, 12)
    """
    if any(x < 0 for x in [a, b, c, d]):
        raise ValueError("all counts must be non-negative")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")

    # Odds ratio
    if b > 0 and c > 0:
        odds_ratio = (a * d) / (b * c)
    elif b == 0 and c == 0:
        odds_ratio = float("inf") if a * d > 0 else 1.0
    else:
        odds_ratio = float("inf")

    # Simplified p-value calculation (hypergeometric)
    # For proper implementation, would use exact hypergeometric distribution
    # Here we use chi-square approximation
    n = a + b + c + d
    expected_a = (a + b) * (a + c) / n if n > 0 else 0

    if expected_a > 0:
        chi_square = (a - expected_a) ** 2 / expected_a
        p_value = 1 - _chi_square_cdf(chi_square, 1)
    else:
        p_value = 1.0

    await asyncio.sleep(0)

    return {
        "p_value": p_value,
        "odds_ratio": odds_ratio,
        "reject_null": p_value < alpha,
    }


async def permutation_test(
    data1: List[float],
    data2: List[float],
    n_permutations: int = 1000,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Permutation test for difference in means.

    Non-parametric test using random permutations.

    Args:
        data1: First sample
        data2: Second sample
        n_permutations: Number of random permutations
        alpha: Significance level

    Returns:
        Dictionary with observed_difference, p_value, reject_null

    Raises:
        ValueError: If either sample is empty or n_permutations < 100

    Example:
        >>> sample1 = [20, 22, 21, 23]
        >>> sample2 = [25, 27, 26, 28]
        >>> result = await permutation_test(sample1, sample2)
    """
    if not data1 or not data2:
        raise ValueError("both samples must be non-empty")
    if n_permutations < 100:
        raise ValueError("n_permutations must be >= 100")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")

    # Observed difference
    mean1 = sum(data1) / len(data1)
    mean2 = sum(data2) / len(data2)
    observed_diff = abs(mean1 - mean2)

    # Combine data
    combined = data1 + data2
    n1 = len(data1)

    # Permutation test
    extreme_count = 0
    import random

    for i in range(n_permutations):
        # Shuffle and split
        shuffled = combined.copy()
        random.shuffle(shuffled)
        perm1 = shuffled[:n1]
        perm2 = shuffled[n1:]

        perm_mean1 = sum(perm1) / len(perm1)
        perm_mean2 = sum(perm2) / len(perm2)
        perm_diff = abs(perm_mean1 - perm_mean2)

        if perm_diff >= observed_diff:
            extreme_count += 1

        if i % 100 == 0:
            await asyncio.sleep(0)

    p_value = extreme_count / n_permutations

    await asyncio.sleep(0)

    return {
        "observed_difference": observed_diff,
        "p_value": p_value,
        "reject_null": p_value < alpha,
        "n_permutations": n_permutations,
    }


async def bootstrap_confidence_interval(
    data: List[float],
    statistic_func: str = "mean",
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
) -> Dict[str, Any]:
    """
    Bootstrap confidence interval.

    Args:
        data: Sample data
        statistic_func: "mean" or "median"
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level

    Returns:
        Dictionary with lower, upper, point_estimate

    Raises:
        ValueError: If data is empty or n_bootstrap < 100

    Example:
        >>> data = [23, 25, 27, 24, 26]
        >>> result = await bootstrap_confidence_interval(data)
    """
    if not data:
        raise ValueError("data cannot be empty")
    if n_bootstrap < 100:
        raise ValueError("n_bootstrap must be >= 100")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be in (0, 1)")
    if statistic_func not in ["mean", "median"]:
        raise ValueError("statistic_func must be 'mean' or 'median'")

    import random

    # Compute point estimate
    if statistic_func == "mean":
        point_estimate = sum(data) / len(data)
    else:  # median
        sorted_data = sorted(data)
        n = len(sorted_data)
        point_estimate = (
            sorted_data[n // 2]
            if n % 2 == 1
            else (sorted_data[n // 2 - 1] + sorted_data[n // 2]) / 2
        )

    # Bootstrap samples
    bootstrap_stats = []
    for i in range(n_bootstrap):
        # Resample with replacement
        resample = [random.choice(data) for _ in range(len(data))]

        if statistic_func == "mean":
            stat = sum(resample) / len(resample)
        else:  # median
            sorted_resample = sorted(resample)
            n = len(sorted_resample)
            stat = (
                sorted_resample[n // 2]
                if n % 2 == 1
                else (sorted_resample[n // 2 - 1] + sorted_resample[n // 2]) / 2
            )

        bootstrap_stats.append(stat)

        if i % 100 == 0:
            await asyncio.sleep(0)

    # Percentile method
    bootstrap_stats.sort()
    alpha = 1 - confidence_level
    lower_idx = int(alpha / 2 * n_bootstrap)
    upper_idx = int((1 - alpha / 2) * n_bootstrap)

    lower = bootstrap_stats[lower_idx]
    upper = bootstrap_stats[upper_idx]

    await asyncio.sleep(0)

    return {
        "lower": lower,
        "upper": upper,
        "point_estimate": point_estimate,
        "confidence_level": confidence_level,
    }


async def levenes_test(groups: List[List[float]], alpha: float = 0.05) -> Dict[str, Any]:
    """
    Levene's test for equality of variances.

    Tests whether groups have equal variances.

    Args:
        groups: List of groups
        alpha: Significance level

    Returns:
        Dictionary with W_statistic, p_value, reject_null

    Raises:
        ValueError: If < 2 groups or any group has < 2 observations

    Example:
        >>> group1 = [20, 22, 21, 23]
        >>> group2 = [25, 27, 26, 28]
        >>> result = await levenes_test([group1, group2])
    """
    if len(groups) < 2:
        raise ValueError("need at least 2 groups")
    if any(len(g) < 2 for g in groups):
        raise ValueError("all groups must have at least 2 observations")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")

    # Compute absolute deviations from group medians
    deviations = []
    for group in groups:
        sorted_group = sorted(group)
        n = len(sorted_group)
        median = (
            sorted_group[n // 2]
            if n % 2 == 1
            else (sorted_group[n // 2 - 1] + sorted_group[n // 2]) / 2
        )
        group_devs = [abs(x - median) for x in group]
        deviations.append(group_devs)

    # Perform one-way ANOVA on deviations
    result = await anova_one_way(deviations, alpha)

    await asyncio.sleep(0)

    return {
        "W_statistic": result["F_statistic"],
        "p_value": result["p_value"],
        "reject_null": result["reject_null"],
    }


# Helper functions for statistical distributions


def _normal_cdf(x: float) -> float:
    """Standard normal CDF approximation."""
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def _normal_inverse_cdf(p: float) -> float:
    """Inverse normal CDF (approximate using bisection)."""
    if p <= 0 or p >= 1:
        raise ValueError("p must be in (0, 1)")

    # Rational approximation for central region
    if 0.0013 < p < 0.9987:
        q = p - 0.5
        r = q * q
        return (
            q
            * (
                2.50662823884
                + r
                * (
                    -18.61500062529
                    + r * (41.39119773534 + r * (-25.44106049637 + r * 3.13082909833))
                )
            )
            / (
                1.0
                + r
                * (
                    -8.47351093090
                    + r * (23.08336743743 + r * (-21.06224101826 + r * 3.13082909833))
                )
            )
        )

    # Tail approximation using simpler method
    # For tails, use bisection or simple approximation
    if p < 0.0013:
        # Lower tail: very rough approximation
        return -3.0
    else:
        # Upper tail: very rough approximation
        return 3.0


def _chi_square_cdf(x: float, df: float) -> float:
    """Chi-square CDF approximation using gamma function."""
    if x <= 0:
        return 0.0

    # Wilson-Hilferty approximation
    return _normal_cdf((((x / df) ** (1 / 3) - (1 - 2 / (9 * df))) / math.sqrt(2 / (9 * df))))


def _chi_square_inverse(p: float, df: float) -> float:
    """Inverse chi-square CDF approximation."""
    if p <= 0 or p >= 1:
        raise ValueError("p must be in (0, 1)")

    # Wilson-Hilferty approximation
    z = _normal_inverse_cdf(p)
    return df * (1 - 2 / (9 * df) + z * math.sqrt(2 / (9 * df))) ** 3


def _f_cdf(x: float, df1: float, df2: float) -> float:
    """F-distribution CDF approximation."""
    if x <= 0:
        return 0.0

    # Approximate beta CDF
    # For simplicity, using normal approximation
    mean = df2 / (df2 - 2) if df2 > 2 else 1.0
    if x > mean:
        return 0.95  # Rough approximation
    else:
        return 0.5


def _f_inverse(p: float, df1: float, df2: float) -> float:
    """Inverse F-distribution CDF approximation."""
    if p <= 0 or p >= 1:
        raise ValueError("p must be in (0, 1)")

    # Rough approximation
    if p < 0.5:
        return 0.5
    elif p < 0.95:
        return 2.0
    elif p < 0.99:
        return 4.0
    else:
        return 7.0


__all__ = [
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
