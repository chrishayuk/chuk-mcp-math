#!/usr/bin/env python3
"""Tests for time series analysis module."""

import pytest
from chuk_mcp_math.timeseries import analysis


class TestMovingAverages:
    """Tests for moving average functions."""

    @pytest.mark.asyncio
    async def test_simple_moving_average(self):
        """Test SMA calculation."""
        data = [1, 2, 3, 4, 5]
        result = await analysis.simple_moving_average(data, 3)
        assert len(result) == 3
        assert abs(result[0] - 2.0) < 1e-10
        assert abs(result[1] - 3.0) < 1e-10
        assert abs(result[2] - 4.0) < 1e-10

    @pytest.mark.asyncio
    async def test_exponential_moving_average(self):
        """Test EMA calculation."""
        data = [1, 2, 3, 4, 5]
        result = await analysis.exponential_moving_average(data, 0.5)
        assert len(result) == len(data)
        assert result[0] == 1.0  # First value

    @pytest.mark.asyncio
    async def test_weighted_moving_average(self):
        """Test WMA calculation."""
        data = [1, 2, 3, 4, 5]
        weights = [0.1, 0.2, 0.3, 0.4]
        result = await analysis.weighted_moving_average(data, weights)
        assert len(result) == 2  # len(data) - len(weights) + 1


class TestAutocorrelation:
    """Tests for autocorrelation functions."""

    @pytest.mark.asyncio
    async def test_autocorrelation(self):
        """Test ACF calculation."""
        data = [1, 2, 3, 4, 5, 6, 7, 8]
        acf_1 = await analysis.autocorrelation(data, 1)
        assert -1 <= acf_1 <= 1

    @pytest.mark.asyncio
    async def test_partial_autocorrelation(self):
        """Test PACF calculation."""
        data = [1, 2, 3, 4, 5, 6, 7, 8]
        pacf_1 = await analysis.partial_autocorrelation(data, 1)
        assert -1 <= pacf_1 <= 1


class TestDecomposition:
    """Tests for seasonal decomposition."""

    @pytest.mark.asyncio
    async def test_seasonal_decompose(self):
        """Test seasonal decomposition."""
        data = [10, 12, 15, 11, 10, 12, 15, 11, 10, 12, 15, 11]
        result = await analysis.seasonal_decompose(data, period=4)
        assert "trend" in result
        assert "seasonal" in result
        assert "residual" in result
        assert len(result["trend"]) == len(data)


class TestTrendDetection:
    """Tests for trend detection."""

    @pytest.mark.asyncio
    async def test_detect_trend(self):
        """Test trend detection."""
        data = [1, 2, 3, 4, 5]  # Perfect linear trend
        result = await analysis.detect_trend(data)
        assert result["slope"] > 0
        assert result["strength"] > 0.9

    @pytest.mark.asyncio
    async def test_detrend(self):
        """Test detrending."""
        data = [1, 2, 3, 4, 5]
        result = await analysis.detrend(data)
        assert len(result) == len(data)


class TestSeasonality:
    """Tests for seasonality detection."""

    @pytest.mark.asyncio
    async def test_detect_seasonality(self):
        """Test seasonality detection."""
        data = [1, 2, 1, 2, 1, 2, 1, 2]  # Period 2
        result = await analysis.detect_seasonality(data, max_period=4)
        assert result["period"] >= 0
        assert 0 <= result["strength"] <= 1

    @pytest.mark.asyncio
    async def test_deseasonalize(self):
        """Test deseasonalization."""
        data = [10, 12, 15, 11, 10, 12, 15, 11]
        result = await analysis.deseasonalize(data, period=4)
        assert len(result) == len(data)


class TestDifferencing:
    """Tests for differencing."""

    @pytest.mark.asyncio
    async def test_differencing(self):
        """Test differencing."""
        data = [1, 3, 6, 10, 15]
        result = await analysis.differencing(data, lag=1)
        assert result == [2, 3, 4, 5]


class TestForecasting:
    """Tests for forecasting methods."""

    @pytest.mark.asyncio
    async def test_holt_winters_forecast(self):
        """Test Holt-Winters forecast."""
        data = [10, 12, 15, 11] * 3
        result = await analysis.holt_winters_forecast(data, period=4, forecast_periods=2)
        assert "forecast" in result
        assert len(result["forecast"]) == 2

    @pytest.mark.asyncio
    async def test_exponential_smoothing(self):
        """Test exponential smoothing forecast."""
        data = [1, 2, 3, 4, 5]
        result = await analysis.exponential_smoothing(data, alpha=0.5, forecast_periods=2)
        assert "forecast" in result
        assert len(result["forecast"]) == 2

    @pytest.mark.asyncio
    async def test_moving_average_forecast(self):
        """Test moving average forecast."""
        data = [1, 2, 3, 4, 5]
        result = await analysis.moving_average_forecast(data, window=3, forecast_periods=2)
        assert len(result) == 2


class TestUtilities:
    """Tests for utility functions."""

    @pytest.mark.asyncio
    async def test_lagged_values(self):
        """Test lagged values."""
        data = [1, 2, 3, 4, 5]
        result = await analysis.lagged_values(data, [1, 2])
        assert 1 in result
        assert len(result[1]) == 4

    @pytest.mark.asyncio
    async def test_rolling_std(self):
        """Test rolling standard deviation."""
        data = [1, 2, 3, 4, 5]
        result = await analysis.rolling_std(data, window=3)
        assert len(result) == 3
        assert all(r >= 0 for r in result)

    @pytest.mark.asyncio
    async def test_rolling_variance(self):
        """Test rolling variance."""
        data = [1, 2, 3, 4, 5]
        result = await analysis.rolling_variance(data, window=3)
        assert len(result) == 3
        assert all(r >= 0 for r in result)


class TestStrength:
    """Tests for strength measures."""

    @pytest.mark.asyncio
    async def test_seasonal_strength(self):
        """Test seasonal strength."""
        data = [1, 2, 1, 2, 1, 2, 1, 2, 1, 2]
        strength = await analysis.seasonal_strength(data, period=2)
        assert 0 <= strength <= 1

    @pytest.mark.asyncio
    async def test_trend_strength(self):
        """Test trend strength."""
        data = [1, 2, 3, 4, 5]
        strength = await analysis.trend_strength(data)
        assert strength > 0.9  # Perfect linear trend


class TestStationarity:
    """Tests for stationarity."""

    @pytest.mark.asyncio
    async def test_stationarity_test(self):
        """Test stationarity test."""
        data = [1, 2, 1, 2, 1, 2, 1, 2]
        result = await analysis.stationarity_test(data)
        assert "is_stationary" in result
        assert isinstance(result["is_stationary"], bool)


class TestErrorHandling:
    """Test error handling."""

    @pytest.mark.asyncio
    async def test_sma_errors(self):
        """Test SMA error handling."""
        with pytest.raises(ValueError, match="window must be >= 1"):
            await analysis.simple_moving_average([1, 2, 3], 0)

        with pytest.raises(ValueError, match="cannot be larger than"):
            await analysis.simple_moving_average([1, 2], 5)

    @pytest.mark.asyncio
    async def test_ema_errors(self):
        """Test EMA error handling."""
        with pytest.raises(ValueError, match="alpha must be in"):
            await analysis.exponential_moving_average([1, 2, 3], 1.5)

    @pytest.mark.asyncio
    async def test_autocorrelation_errors(self):
        """Test autocorrelation error handling."""
        with pytest.raises(ValueError, match="lag must be >= 1"):
            await analysis.autocorrelation([1, 2, 3], 0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


class TestEdgeCases:
    """Test edge cases for better coverage."""

    @pytest.mark.asyncio
    async def test_wma_weight_sum_validation(self):
        """Test WMA validates weight sum."""
        with pytest.raises(ValueError, match="weights must sum to"):
            await analysis.weighted_moving_average([1, 2, 3], [0.5, 0.3])  # Sum = 0.8

    @pytest.mark.asyncio
    async def test_holt_winters_parameter_validation(self):
        """Test Holt-Winters parameter validation."""
        data = [1, 2, 3, 4, 5, 6, 7, 8]

        with pytest.raises(ValueError, match="alpha must be in"):
            await analysis.holt_winters_forecast(data, 2, alpha=1.5)

        with pytest.raises(ValueError, match="beta must be in"):
            await analysis.holt_winters_forecast(data, 2, beta=1.5)

        with pytest.raises(ValueError, match="gamma must be in"):
            await analysis.holt_winters_forecast(data, 2, gamma=1.5)

    @pytest.mark.asyncio
    async def test_seasonal_decompose_validation(self):
        """Test seasonal decompose validation."""
        with pytest.raises(ValueError, match="period must be >= 2"):
            await analysis.seasonal_decompose([1, 2, 3], period=1)

        with pytest.raises(ValueError, match="period too large"):
            await analysis.seasonal_decompose([1, 2, 3], period=5)

    @pytest.mark.asyncio
    async def test_exponential_smoothing_validation(self):
        """Test exponential smoothing validation."""
        with pytest.raises(ValueError, match="alpha must be in"):
            await analysis.exponential_smoothing([1, 2, 3], alpha=1.5)

        with pytest.raises(ValueError, match="forecast_periods must be >= 1"):
            await analysis.exponential_smoothing([1, 2, 3], forecast_periods=0)

    @pytest.mark.asyncio
    async def test_rolling_window_validation(self):
        """Test rolling window validation."""
        with pytest.raises(ValueError, match="window must be >= 2"):
            await analysis.rolling_std([1, 2, 3], window=1)

        with pytest.raises(ValueError, match="window cannot exceed"):
            await analysis.rolling_variance([1, 2], window=5)

    @pytest.mark.asyncio
    async def test_moving_average_forecast_validation(self):
        """Test moving average forecast validation."""
        with pytest.raises(ValueError, match="window must be >= 1"):
            await analysis.moving_average_forecast([1, 2, 3], window=0)
