#!/usr/bin/env python3
"""Tests for inferential statistics module."""

import pytest
from chuk_mcp_math.statistics import inference


class TestTTests:
    """Tests for t-tests."""

    @pytest.mark.asyncio
    async def test_one_sample_t_test(self):
        """Test one-sample t-test."""
        data = [23, 25, 27, 24, 26]
        result = await inference.t_test_one_sample(data, 20.0)
        assert "t_statistic" in result
        assert "p_value" in result
        assert "degrees_of_freedom" in result
        assert result["degrees_of_freedom"] == 4
        assert result["sample_mean"] == 25.0

    @pytest.mark.asyncio
    async def test_two_sample_t_test(self):
        """Test two-sample t-test."""
        data1 = [20, 22, 21, 23, 22]
        data2 = [25, 27, 26, 28, 27]
        result = await inference.t_test_two_sample(data1, data2)
        assert "t_statistic" in result
        assert "mean_difference" in result
        assert result["mean_difference"] < 0  # data1 < data2

    @pytest.mark.asyncio
    async def test_paired_t_test(self):
        """Test paired t-test."""
        before = [80, 85, 82, 84, 83]
        after = [85, 90, 87, 89, 88]
        result = await inference.paired_t_test(before, after)
        assert "t_statistic" in result
        assert "p_value" in result

    @pytest.mark.asyncio
    async def test_z_test(self):
        """Test z-test."""
        data = [105, 110, 108, 107, 109]
        result = await inference.z_test(data, 100.0, 5.0)
        assert "z_statistic" in result
        assert result["z_statistic"] > 0  # Sample mean > population mean


class TestChiSquare:
    """Tests for chi-square tests."""

    @pytest.mark.asyncio
    async def test_chi_square_test(self):
        """Test chi-square goodness-of-fit."""
        observed = [30, 25, 20, 25]
        expected = [25, 25, 25, 25]
        result = await inference.chi_square_test(observed, expected)
        assert "chi_square_statistic" in result
        assert result["degrees_of_freedom"] == 3


class TestConfidenceIntervals:
    """Tests for confidence intervals."""

    @pytest.mark.asyncio
    async def test_confidence_interval_mean(self):
        """Test CI for mean."""
        data = [23, 25, 27, 24, 26]
        result = await inference.confidence_interval_mean(data, 0.95)
        assert result["lower"] < result["mean"] < result["upper"]
        assert result["confidence_level"] == 0.95

    @pytest.mark.asyncio
    async def test_confidence_interval_proportion(self):
        """Test CI for proportion."""
        result = await inference.confidence_interval_proportion(60, 100, 0.95)
        assert 0 <= result["lower"] <= result["proportion"] <= result["upper"] <= 1
        assert result["proportion"] == 0.6


class TestEffectSizes:
    """Tests for effect sizes."""

    @pytest.mark.asyncio
    async def test_cohens_d(self):
        """Test Cohen's d."""
        control = [20, 22, 21, 23]
        treatment = [25, 27, 26, 28]
        d = await inference.cohens_d(control, treatment)
        assert d != 0  # Should detect difference

    @pytest.mark.asyncio
    async def test_effect_size_r(self):
        """Test effect size r."""
        r = await inference.effect_size_r(2.5, 18)
        assert 0 <= abs(r) <= 1


class TestANOVA:
    """Tests for ANOVA."""

    @pytest.mark.asyncio
    async def test_anova_one_way(self):
        """Test one-way ANOVA."""
        group1 = [20, 22, 21, 23]
        group2 = [25, 27, 26, 28]
        group3 = [30, 32, 31, 33]
        result = await inference.anova_one_way([group1, group2, group3])
        assert "F_statistic" in result
        assert result["df_between"] == 2
        assert result["df_within"] == 9


class TestProportions:
    """Tests for proportion tests."""

    @pytest.mark.asyncio
    async def test_proportion_test(self):
        """Test two-proportion z-test."""
        result = await inference.proportion_test(60, 100, 45, 100)
        assert "z_statistic" in result
        assert "proportion_difference" in result
        assert abs(result["proportion_difference"] - 0.15) < 0.01


class TestPowerAnalysis:
    """Tests for power and sample size."""

    @pytest.mark.asyncio
    async def test_sample_size_mean(self):
        """Test sample size calculation."""
        n = await inference.sample_size_mean(0.5, alpha=0.05, power=0.8)
        assert n >= 2
        assert isinstance(n, int)

    @pytest.mark.asyncio
    async def test_power_analysis(self):
        """Test power calculation."""
        power = await inference.power_analysis(50, 0.5, alpha=0.05)
        assert 0 <= power <= 1


class TestNonParametric:
    """Tests for non-parametric tests."""

    @pytest.mark.asyncio
    async def test_mann_whitney_u(self):
        """Test Mann-Whitney U test."""
        sample1 = [20, 22, 21, 23]
        sample2 = [25, 27, 26, 28]
        result = await inference.mann_whitney_u(sample1, sample2)
        assert "U_statistic" in result
        assert "p_value" in result

    @pytest.mark.asyncio
    async def test_wilcoxon_signed_rank(self):
        """Test Wilcoxon signed-rank test."""
        before = [80, 85, 82, 84]
        after = [85, 90, 87, 89]
        result = await inference.wilcoxon_signed_rank(before, after)
        assert "W_statistic" in result

    @pytest.mark.asyncio
    async def test_kruskal_wallis(self):
        """Test Kruskal-Wallis test."""
        group1 = [20, 22, 21]
        group2 = [25, 27, 26]
        group3 = [30, 32, 31]
        result = await inference.kruskal_wallis([group1, group2, group3])
        assert "H_statistic" in result


class TestOtherTests:
    """Tests for other statistical tests."""

    @pytest.mark.asyncio
    async def test_fishers_exact_test(self):
        """Test Fisher's exact test."""
        result = await inference.fishers_exact_test(10, 5, 3, 12)
        assert "p_value" in result
        assert "odds_ratio" in result

    @pytest.mark.asyncio
    async def test_permutation_test(self):
        """Test permutation test."""
        sample1 = [20, 22, 21, 23]
        sample2 = [25, 27, 26, 28]
        result = await inference.permutation_test(sample1, sample2, n_permutations=100)
        assert "observed_difference" in result
        assert "p_value" in result

    @pytest.mark.asyncio
    async def test_bootstrap_confidence_interval(self):
        """Test bootstrap CI."""
        data = [23, 25, 27, 24, 26]
        result = await inference.bootstrap_confidence_interval(data, n_bootstrap=100)
        assert result["lower"] < result["point_estimate"] < result["upper"]

    @pytest.mark.asyncio
    async def test_levenes_test(self):
        """Test Levene's test."""
        group1 = [20, 22, 21, 23]
        group2 = [25, 27, 26, 28]
        result = await inference.levenes_test([group1, group2])
        assert "W_statistic" in result


class TestErrorHandling:
    """Test error handling."""

    @pytest.mark.asyncio
    async def test_t_test_errors(self):
        """Test t-test error handling."""
        with pytest.raises(ValueError, match="at least 2 observations"):
            await inference.t_test_one_sample([1], 0.0)

        with pytest.raises(ValueError, match="alpha must be in"):
            await inference.t_test_one_sample([1, 2], 0.0, alpha=1.5)

    @pytest.mark.asyncio
    async def test_z_test_errors(self):
        """Test z-test error handling."""
        with pytest.raises(ValueError, match="cannot be empty"):
            await inference.z_test([], 0.0, 1.0)

        with pytest.raises(ValueError, match="must be positive"):
            await inference.z_test([1, 2], 0.0, -1.0)

    @pytest.mark.asyncio
    async def test_chi_square_errors(self):
        """Test chi-square error handling."""
        with pytest.raises(ValueError, match="same length"):
            await inference.chi_square_test([1, 2], [1, 2, 3])

        with pytest.raises(ValueError, match="must be positive"):
            await inference.chi_square_test([1, 2], [0, 1])

    @pytest.mark.asyncio
    async def test_paired_t_test_errors(self):
        """Test paired t-test error handling."""
        with pytest.raises(ValueError, match="same length"):
            await inference.paired_t_test([1, 2], [1, 2, 3])

        with pytest.raises(ValueError, match="at least 2 pairs"):
            await inference.paired_t_test([1], [1])

    @pytest.mark.asyncio
    async def test_anova_errors(self):
        """Test ANOVA error handling."""
        with pytest.raises(ValueError, match="at least 2 groups"):
            await inference.anova_one_way([[1, 2]])

        with pytest.raises(ValueError, match="at least 2 observations"):
            await inference.anova_one_way([[1, 2], [3]])

    @pytest.mark.asyncio
    async def test_proportion_errors(self):
        """Test proportion error handling."""
        with pytest.raises(ValueError, match="must be >= 1"):
            await inference.confidence_interval_proportion(0, 0)

        with pytest.raises(ValueError, match="must be in"):
            await inference.confidence_interval_proportion(10, 5)

    @pytest.mark.asyncio
    async def test_sample_size_errors(self):
        """Test sample size calculation errors."""
        with pytest.raises(ValueError, match="must be positive"):
            await inference.sample_size_mean(-0.5)

        with pytest.raises(ValueError, match="must be 'two-sided' or 'one-sided'"):
            await inference.sample_size_mean(0.5, alternative="invalid")

    @pytest.mark.asyncio
    async def test_bootstrap_errors(self):
        """Test bootstrap error handling."""
        with pytest.raises(ValueError, match="cannot be empty"):
            await inference.bootstrap_confidence_interval([])

        with pytest.raises(ValueError, match="must be >= 100"):
            await inference.bootstrap_confidence_interval([1, 2], n_bootstrap=50)

        with pytest.raises(ValueError, match="must be 'mean' or 'median'"):
            await inference.bootstrap_confidence_interval([1, 2], statistic_func="invalid")


class TestEdgeCases:
    """Test edge cases."""

    @pytest.mark.asyncio
    async def test_wilcoxon_all_zeros(self):
        """Test Wilcoxon with no differences."""
        before = [5, 5, 5, 5]
        after = [5, 5, 5, 5]
        result = await inference.wilcoxon_signed_rank(before, after)
        assert result["W_statistic"] == 0.0
        assert result["p_value"] == 1.0

    @pytest.mark.asyncio
    async def test_fishers_exact_zeros(self):
        """Test Fisher's exact with zeros."""
        result = await inference.fishers_exact_test(0, 5, 5, 0)
        assert "p_value" in result

    @pytest.mark.asyncio
    async def test_bootstrap_median(self):
        """Test bootstrap with median statistic."""
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        result = await inference.bootstrap_confidence_interval(
            data, statistic_func="median", n_bootstrap=100
        )
        assert result["point_estimate"] == 5

    @pytest.mark.asyncio
    async def test_two_sample_unequal_variance(self):
        """Test two-sample t-test with unequal variance."""
        data1 = [1, 2, 3, 4, 5]
        data2 = [10, 20, 30, 40, 50]
        result = await inference.t_test_two_sample(data1, data2, equal_variance=False)
        assert "t_statistic" in result
        assert result["mean_difference"] < 0

    @pytest.mark.asyncio
    async def test_confidence_interval_extremes(self):
        """Test CI for proportion at extremes."""
        # All successes
        result = await inference.confidence_interval_proportion(100, 100)
        assert result["proportion"] == 1.0
        assert result["upper"] == 1.0

        # No successes
        result = await inference.confidence_interval_proportion(0, 100)
        assert result["proportion"] == 0.0
        assert result["lower"] == 0.0

    @pytest.mark.asyncio
    async def test_power_analysis_one_sided(self):
        """Test power analysis with one-sided test."""
        power = await inference.power_analysis(50, 0.5, alpha=0.05, alternative="one-sided")
        assert 0 <= power <= 1

    @pytest.mark.asyncio
    async def test_sample_size_one_sided(self):
        """Test sample size with one-sided test."""
        n = await inference.sample_size_mean(0.5, alpha=0.05, power=0.8, alternative="one-sided")
        assert n >= 2

    @pytest.mark.asyncio
    async def test_permutation_small_samples(self):
        """Test permutation test with small samples."""
        sample1 = [1, 2]
        sample2 = [3, 4]
        result = await inference.permutation_test(sample1, sample2, n_permutations=100)
        assert "observed_difference" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


class TestAdditionalCoverage:
    """Additional tests to improve coverage."""

    @pytest.mark.asyncio
    async def test_t_test_one_sample_zero_se(self):
        """Test t-test with zero standard error (all identical values)."""
        # All values equal to population mean
        data = [5.0, 5.0, 5.0, 5.0]
        result = await inference.t_test_one_sample(data, 5.0)
        assert result["t_statistic"] == 0.0

        # All values different from population mean
        result = await inference.t_test_one_sample(data, 10.0)
        assert result["t_statistic"] == float("-inf")

    @pytest.mark.asyncio
    async def test_two_sample_welch_edge_cases(self):
        """Test Welch's t-test edge cases."""
        data1 = [1, 2, 3]
        data2 = [10, 20, 30]
        result = await inference.t_test_two_sample(data1, data2, equal_variance=False)
        assert "degrees_of_freedom" in result

    @pytest.mark.asyncio
    async def test_proportion_test_edge_cases(self):
        """Test proportion test with extreme values."""
        # Zero standard error case
        result = await inference.proportion_test(50, 100, 50, 100)
        assert result["proportion_difference"] == 0.0

    @pytest.mark.asyncio
    async def test_normal_inverse_cdf_tails(self):
        """Test normal inverse CDF tail approximation."""
        # Test lower tail
        z_low = inference._normal_inverse_cdf(0.001)
        assert z_low < -2.0

        # Test upper tail
        z_high = inference._normal_inverse_cdf(0.999)
        assert z_high > 2.0

    @pytest.mark.asyncio
    async def test_fisher_exact_edge_cases(self):
        """Test Fisher's exact with various edge cases."""
        # Both b and c are zero
        result = await inference.fishers_exact_test(5, 0, 0, 5)
        assert result["odds_ratio"] in [1.0, float("inf")]

        # b is zero but c is not
        result = await inference.fishers_exact_test(5, 0, 3, 5)
        assert result["odds_ratio"] == float("inf")

        # c is zero but b is not
        result = await inference.fishers_exact_test(5, 3, 0, 5)
        assert result["odds_ratio"] == float("inf")

    @pytest.mark.asyncio
    async def test_chi_square_cdf_zero(self):
        """Test chi-square CDF with x <= 0."""
        cdf = inference._chi_square_cdf(0.0, 5)
        assert cdf == 0.0

        cdf = inference._chi_square_cdf(-1.0, 5)
        assert cdf == 0.0

    @pytest.mark.asyncio
    async def test_f_cdf_approximation(self):
        """Test F-distribution CDF approximation."""
        # Test with x <= 0
        cdf = inference._f_cdf(0.0, 5, 10)
        assert cdf == 0.0

        # Test with x > mean
        cdf = inference._f_cdf(5.0, 5, 10)
        assert cdf > 0.5

        # Test with x < mean
        cdf = inference._f_cdf(0.5, 5, 10)
        assert cdf < 0.95

    @pytest.mark.asyncio
    async def test_f_inverse_quantiles(self):
        """Test F-distribution inverse CDF at various quantiles."""
        # Lower quantile
        f_val = inference._f_inverse(0.3, 5, 10)
        assert f_val > 0

        # Medium quantile
        f_val = inference._f_inverse(0.9, 5, 10)
        assert f_val > 0

        # High quantile
        f_val = inference._f_inverse(0.98, 5, 10)
        assert f_val > 0
