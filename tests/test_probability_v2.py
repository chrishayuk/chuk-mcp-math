"""
Test Phase 2 Probability Distributions

Tests for exponential and binomial distributions.
"""

import pytest
import math


@pytest.mark.asyncio
class TestExponentialDistribution:
    """Test exponential distribution functions."""

    async def test_exponential_pdf(self):
        from chuk_mcp_math.probability.discrete_distributions import exponential_pdf

        # At x=1, λ=1: PDF = e^(-1)
        result = await exponential_pdf(1.0, 1.0)
        assert abs(result - math.exp(-1)) < 1e-10

        # At x=0, PDF = λ
        result_zero = await exponential_pdf(0.0, 2.0)
        assert abs(result_zero - 2.0) < 1e-10

    async def test_exponential_pdf_errors(self):
        from chuk_mcp_math.probability.discrete_distributions import exponential_pdf

        # Test negative x
        with pytest.raises(ValueError, match="x must be non-negative"):
            await exponential_pdf(-1.0, 1.0)

        # Test invalid rate
        with pytest.raises(ValueError, match="Rate must be positive"):
            await exponential_pdf(1.0, 0.0)

        with pytest.raises(ValueError, match="Rate must be positive"):
            await exponential_pdf(1.0, -1.0)

    async def test_exponential_cdf(self):
        from chuk_mcp_math.probability.discrete_distributions import exponential_cdf

        # At x=1, λ=1: CDF = 1 - e^(-1)
        result = await exponential_cdf(1.0, 1.0)
        assert abs(result - (1 - math.exp(-1))) < 1e-10

        # At x=0, CDF = 0
        result_zero = await exponential_cdf(0.0, 1.0)
        assert abs(result_zero) < 1e-10

    async def test_exponential_cdf_errors(self):
        from chuk_mcp_math.probability.discrete_distributions import exponential_cdf

        # Test negative x
        with pytest.raises(ValueError, match="x must be non-negative"):
            await exponential_cdf(-1.0, 1.0)

        # Test invalid rate
        with pytest.raises(ValueError, match="Rate must be positive"):
            await exponential_cdf(1.0, 0.0)

    async def test_exponential_sample(self):
        from chuk_mcp_math.probability.discrete_distributions import exponential_sample

        samples = await exponential_sample(1000, 1.0, seed=42)

        assert len(samples) == 1000
        assert all(s >= 0 for s in samples)  # All samples should be non-negative

        # Mean should be approximately 1/λ = 1
        mean_sample = sum(samples) / len(samples)
        assert 0.8 < mean_sample < 1.2  # Loose check for mean

    async def test_exponential_sample_large(self):
        from chuk_mcp_math.probability.discrete_distributions import exponential_sample

        # Test large sample (triggers async yield)
        samples = await exponential_sample(1500, 1.0, seed=42)
        assert len(samples) == 1500

    async def test_exponential_sample_errors(self):
        from chuk_mcp_math.probability.discrete_distributions import exponential_sample

        # Test invalid n
        with pytest.raises(ValueError, match="Number of samples must be positive"):
            await exponential_sample(0, 1.0)

        with pytest.raises(ValueError, match="Number of samples must be positive"):
            await exponential_sample(-10, 1.0)

        # Test invalid rate
        with pytest.raises(ValueError, match="Rate must be positive"):
            await exponential_sample(10, 0.0)


@pytest.mark.asyncio
class TestBinomialDistribution:
    """Test binomial distribution functions."""

    async def test_binomial_pmf(self):
        from chuk_mcp_math.probability.discrete_distributions import binomial_pmf

        # P(X=3) for Binomial(5, 0.5) = C(5,3) * 0.5^3 * 0.5^2 = 10 * 0.03125
        result = await binomial_pmf(3, 5, 0.5)
        assert abs(result - 0.3125) < 1e-10

        # P(X=0) for any binomial = (1-p)^n
        result_zero = await binomial_pmf(0, 10, 0.3)
        assert abs(result_zero - 0.7**10) < 1e-10

    async def test_binomial_pmf_errors(self):
        from chuk_mcp_math.probability.discrete_distributions import binomial_pmf

        # Test negative n
        with pytest.raises(ValueError, match="n must be non-negative"):
            await binomial_pmf(0, -1, 0.5)

        # Test k out of range
        with pytest.raises(ValueError, match="k must be between 0 and n"):
            await binomial_pmf(-1, 5, 0.5)

        with pytest.raises(ValueError, match="k must be between 0 and n"):
            await binomial_pmf(6, 5, 0.5)

        # Test p out of range
        with pytest.raises(ValueError, match="p must be between 0 and 1"):
            await binomial_pmf(2, 5, -0.1)

        with pytest.raises(ValueError, match="p must be between 0 and 1"):
            await binomial_pmf(2, 5, 1.5)

    async def test_binomial_cdf(self):
        from chuk_mcp_math.probability.discrete_distributions import binomial_cdf

        # P(X ≤ 3) for Binomial(5, 0.5)
        result = await binomial_cdf(3, 5, 0.5)
        # This is sum of PMFs for k=0,1,2,3
        # = 1/32 + 5/32 + 10/32 + 10/32 = 26/32 = 0.8125
        assert abs(result - 0.8125) < 1e-10

    async def test_binomial_cdf_errors(self):
        from chuk_mcp_math.probability.discrete_distributions import binomial_cdf

        # Test negative n
        with pytest.raises(ValueError, match="n must be non-negative"):
            await binomial_cdf(0, -1, 0.5)

        # Test k out of range
        with pytest.raises(ValueError, match="k must be between 0 and n"):
            await binomial_cdf(-1, 5, 0.5)

        # Test p out of range
        with pytest.raises(ValueError, match="p must be between 0 and 1"):
            await binomial_cdf(2, 5, -0.1)

    async def test_binomial_sample(self):
        from chuk_mcp_math.probability.discrete_distributions import binomial_sample

        samples = await binomial_sample(100, 10, 0.5, seed=42)

        assert len(samples) == 100
        assert all(0 <= s <= 10 for s in samples)  # All samples in valid range

        # Mean should be approximately n*p = 10 * 0.5 = 5
        mean_sample = sum(samples) / len(samples)
        assert 3 < mean_sample < 7  # Loose check

    async def test_binomial_sample_large(self):
        from chuk_mcp_math.probability.discrete_distributions import binomial_sample

        # Test large sample (triggers async yield)
        samples = await binomial_sample(1500, 10, 0.5, seed=42)
        assert len(samples) == 1500

    async def test_binomial_sample_errors(self):
        from chuk_mcp_math.probability.discrete_distributions import binomial_sample

        # Test invalid num_samples
        with pytest.raises(ValueError, match="Number of samples must be positive"):
            await binomial_sample(0, 10, 0.5)

        with pytest.raises(ValueError, match="Number of samples must be positive"):
            await binomial_sample(-10, 10, 0.5)

        # Test negative n
        with pytest.raises(ValueError, match="n must be non-negative"):
            await binomial_sample(10, -1, 0.5)

        # Test p out of range
        with pytest.raises(ValueError, match="p must be between 0 and 1"):
            await binomial_sample(10, 10, 1.5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
