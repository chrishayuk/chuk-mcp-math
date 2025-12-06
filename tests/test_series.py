#!/usr/bin/env python3
"""Tests for series expansions module."""

import pytest
import math
from chuk_mcp_math.numerical import series


class TestTaylorSeries:
    """Tests for taylor_series function."""

    @pytest.mark.asyncio
    async def test_exp_taylor_series(self):
        """Test Taylor series for e^x."""
        f = lambda x: math.exp(x)
        derivatives = [f] * 10  # All derivatives of e^x are e^x

        result = await series.taylor_series(f, derivatives, 0.0, 1.0, 9)
        assert abs(result - math.e) < 0.001

    @pytest.mark.asyncio
    async def test_sin_taylor_series(self):
        """Test Taylor series for sin(x)."""
        f = lambda x: math.sin(x)
        # Derivatives cycle: sin, cos, -sin, -cos
        derivatives = [
            lambda x: math.sin(x),
            lambda x: math.cos(x),
            lambda x: -math.sin(x),
            lambda x: -math.cos(x),
        ] * 3

        result = await series.taylor_series(f, derivatives, 0.0, math.pi / 4, 10)
        assert abs(result - math.sin(math.pi / 4)) < 0.01

    @pytest.mark.asyncio
    async def test_taylor_errors(self):
        """Test error handling."""
        f = lambda x: x**2
        derivatives = [f]

        with pytest.raises(ValueError, match="n must be >= 0"):
            await series.taylor_series(f, derivatives, 0.0, 1.0, -1)

        with pytest.raises(ValueError, match="Need at least"):
            await series.taylor_series(f, [f], 0.0, 1.0, 5)


class TestPowerSeries:
    """Tests for power_series function."""

    @pytest.mark.asyncio
    async def test_simple_polynomial(self):
        """Test 1 + 2x + 3x^2."""
        result = await series.power_series([1, 2, 3], 2.0)
        assert abs(result - 17.0) < 1e-10  # 1 + 4 + 12

    @pytest.mark.asyncio
    async def test_with_center(self):
        """Test power series with non-zero center."""
        # 1 + 2(x-1) + 3(x-1)^2 at x=2
        result = await series.power_series([1, 2, 3], 2.0, x0=1.0)
        assert abs(result - 6.0) < 1e-10  # 1 + 2 + 3

    @pytest.mark.asyncio
    async def test_error(self):
        """Test error handling."""
        with pytest.raises(ValueError, match="coefficients cannot be empty"):
            await series.power_series([], 1.0)


class TestHornerMethod:
    """Tests for horner_method function."""

    @pytest.mark.asyncio
    async def test_simple_polynomial(self):
        """Test Horner's method."""
        # 2 + 3x + 4x^2 at x=1
        result = await series.horner_method([2, 3, 4], 1.0)
        assert abs(result - 9.0) < 1e-10

    @pytest.mark.asyncio
    async def test_same_as_power_series(self):
        """Test Horner matches power series."""
        coeffs = [1, 2, 3, 4, 5]
        x = 1.5

        horner_result = await series.horner_method(coeffs, x)
        power_result = await series.power_series(coeffs, x)

        assert abs(horner_result - power_result) < 1e-10

    @pytest.mark.asyncio
    async def test_error(self):
        """Test error handling."""
        with pytest.raises(ValueError, match="coefficients cannot be empty"):
            await series.horner_method([], 1.0)


class TestFourierSeries:
    """Tests for fourier_series_approximation function."""

    @pytest.mark.asyncio
    async def test_square_wave(self):
        """Test Fourier series on square wave."""
        # Square wave with period 2
        f = lambda x: 1.0 if (x % 2) < 1 else -1.0

        result = await series.fourier_series_approximation(f, 2.0, 10, 0.5)
        # Should be close to 1.0
        assert 0.5 < result < 1.5

    @pytest.mark.asyncio
    async def test_errors(self):
        """Test error handling."""
        f = lambda x: x

        with pytest.raises(ValueError, match="period must be positive"):
            await series.fourier_series_approximation(f, -1.0, 5, 0.5)

        with pytest.raises(ValueError, match="n_terms must be >= 1"):
            await series.fourier_series_approximation(f, 1.0, 0, 0.5)

        with pytest.raises(ValueError, match="n_samples must be >= 1"):
            await series.fourier_series_approximation(f, 1.0, 5, 0.5, n_samples=0)


class TestMaclaurinSeries:
    """Tests for maclaurin_series function."""

    @pytest.mark.asyncio
    async def test_exp_maclaurin(self):
        """Test Maclaurin series for e^x."""
        f = lambda x: math.exp(x)
        derivatives = [f] * 10

        result = await series.maclaurin_series(derivatives, 1.0, 9)
        assert abs(result - math.e) < 0.001


class TestBinomialSeries:
    """Tests for binomial_series function."""

    @pytest.mark.asyncio
    async def test_sqrt_approximation(self):
        """Test (1+x)^0.5 approximation."""
        # sqrt(1.2) = (1 + 0.2)^0.5
        result = await series.binomial_series(0.2, 0.5, 20)
        assert abs(result - math.sqrt(1.2)) < 0.001

    @pytest.mark.asyncio
    async def test_error(self):
        """Test error handling."""
        with pytest.raises(ValueError, match="n must be >= 0"):
            await series.binomial_series(0.5, 2.0, -1)


class TestGeometricSeries:
    """Tests for geometric_series function."""

    @pytest.mark.asyncio
    async def test_geometric_sum(self):
        """Test geometric series sum."""
        # 1 + 2 + 4 + 8 + 16 = 31
        result = await series.geometric_series(1, 2, 5)
        assert abs(result - 31.0) < 1e-10

    @pytest.mark.asyncio
    async def test_r_equals_one(self):
        """Test when r=1."""
        # 5 + 5 + 5 = 15
        result = await series.geometric_series(5, 1, 3)
        assert abs(result - 15.0) < 1e-10

    @pytest.mark.asyncio
    async def test_error(self):
        """Test error handling."""
        with pytest.raises(ValueError, match="n must be >= 1"):
            await series.geometric_series(1, 2, 0)


class TestArithmeticSeries:
    """Tests for arithmetic_series function."""

    @pytest.mark.asyncio
    async def test_arithmetic_sum(self):
        """Test arithmetic series sum."""
        # 1 + 3 + 5 + 7 + 9 = 25
        result = await series.arithmetic_series(1, 2, 5)
        assert abs(result - 25.0) < 1e-10

    @pytest.mark.asyncio
    async def test_error(self):
        """Test error handling."""
        with pytest.raises(ValueError, match="n must be >= 1"):
            await series.arithmetic_series(1, 2, 0)


class TestExpSeries:
    """Tests for exp_series function."""

    @pytest.mark.asyncio
    async def test_exp_approximation(self):
        """Test exponential approximation."""
        result = await series.exp_series(1.0, 15)
        assert abs(result - math.e) < 1e-6

    @pytest.mark.asyncio
    async def test_exp_zero(self):
        """Test e^0 = 1."""
        result = await series.exp_series(0.0, 5)
        assert abs(result - 1.0) < 1e-10

    @pytest.mark.asyncio
    async def test_error(self):
        """Test error handling."""
        with pytest.raises(ValueError, match="n must be >= 0"):
            await series.exp_series(1.0, -1)


class TestSinSeries:
    """Tests for sin_series function."""

    @pytest.mark.asyncio
    async def test_sin_approximation(self):
        """Test sine approximation."""
        result = await series.sin_series(math.pi / 2, 15)
        assert abs(result - 1.0) < 1e-6

    @pytest.mark.asyncio
    async def test_sin_zero(self):
        """Test sin(0) = 0."""
        result = await series.sin_series(0.0, 5)
        assert abs(result) < 1e-10

    @pytest.mark.asyncio
    async def test_error(self):
        """Test error handling."""
        with pytest.raises(ValueError, match="n must be >= 0"):
            await series.sin_series(1.0, -1)


class TestCosSeries:
    """Tests for cos_series function."""

    @pytest.mark.asyncio
    async def test_cos_approximation(self):
        """Test cosine approximation."""
        result = await series.cos_series(0.0, 10)
        assert abs(result - 1.0) < 1e-10

    @pytest.mark.asyncio
    async def test_cos_pi(self):
        """Test cos(π) ≈ -1."""
        result = await series.cos_series(math.pi, 20)
        assert abs(result - (-1.0)) < 1e-6

    @pytest.mark.asyncio
    async def test_error(self):
        """Test error handling."""
        with pytest.raises(ValueError, match="n must be >= 0"):
            await series.cos_series(1.0, -1)


class TestLnSeries:
    """Tests for ln_series function."""

    @pytest.mark.asyncio
    async def test_ln_approximation(self):
        """Test natural log approximation."""
        result = await series.ln_series(1.5, 50)
        assert abs(result - math.log(1.5)) < 0.01

    @pytest.mark.asyncio
    async def test_errors(self):
        """Test error handling."""
        with pytest.raises(ValueError, match="x must be positive"):
            await series.ln_series(-1.0, 10)

        with pytest.raises(ValueError, match="x must be <= 2"):
            await series.ln_series(3.0, 10)

        with pytest.raises(ValueError, match="n must be >= 1"):
            await series.ln_series(1.5, 0)


class TestAsyncBehavior:
    """Test async execution behavior."""

    @pytest.mark.asyncio
    async def test_async_function_support(self):
        """Test that series functions work with async callables."""

        async def async_f(x):
            """Async function."""
            return math.exp(x)

        async def async_deriv(x):
            """Async derivative."""
            return math.exp(x)

        derivatives = [async_deriv] * 5
        result = await series.taylor_series(async_f, derivatives, 0.0, 1.0, 4)
        assert abs(result - math.e) < 0.1  # Rough approximation


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
