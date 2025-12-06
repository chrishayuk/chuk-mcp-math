#!/usr/bin/env python3
"""Time series analysis and forecasting functions.

This module provides essential time series analysis tools for business analytics,
including moving averages, autocorrelation, trend detection, seasonal decomposition,
and forecasting methods.
"""

import asyncio
import math
from typing import List, Dict, Any, Optional


async def simple_moving_average(data: List[float], window: int) -> List[float]:
    """
    Simple Moving Average (SMA).

    Computes the average of the last 'window' values at each point.

    Args:
        data: Time series data
        window: Window size for averaging

    Returns:
        List of moving averages (length = len(data) - window + 1)

    Raises:
        ValueError: If window < 1 or window > len(data)

    Example:
        >>> data = [1, 2, 3, 4, 5]
        >>> result = await simple_moving_average(data, 3)
        >>> # [2.0, 3.0, 4.0] (averages of [1,2,3], [2,3,4], [3,4,5])
    """
    if window < 1:
        raise ValueError("window must be >= 1")
    if window > len(data):
        raise ValueError(f"window ({window}) cannot be larger than data length ({len(data)})")

    result = []
    for i in range(len(data) - window + 1):
        window_data = data[i : i + window]
        avg = sum(window_data) / window
        result.append(avg)

        if i % 10 == 0:
            await asyncio.sleep(0)

    return result


async def exponential_moving_average(
    data: List[float], alpha: float, initial: Optional[float] = None
) -> List[float]:
    """
    Exponential Moving Average (EMA).

    Gives more weight to recent observations.
    EMA_t = alpha * x_t + (1-alpha) * EMA_{t-1}

    Args:
        data: Time series data
        alpha: Smoothing factor (0 < alpha <= 1), higher = more weight to recent
        initial: Initial EMA value (default: first data point)

    Returns:
        List of exponential moving averages (same length as data)

    Raises:
        ValueError: If alpha not in (0, 1] or data is empty

    Example:
        >>> data = [1, 2, 3, 4, 5]
        >>> result = await exponential_moving_average(data, 0.5)
    """
    if not data:
        raise ValueError("data cannot be empty")
    if not 0 < alpha <= 1:
        raise ValueError("alpha must be in (0, 1]")

    result = []
    ema = initial if initial is not None else data[0]
    result.append(ema)

    for i in range(1, len(data)):
        ema = alpha * data[i] + (1 - alpha) * ema
        result.append(ema)

        if i % 10 == 0:
            await asyncio.sleep(0)

    return result


async def weighted_moving_average(data: List[float], weights: List[float]) -> List[float]:
    """
    Weighted Moving Average (WMA).

    Applies custom weights to recent observations.

    Args:
        data: Time series data
        weights: Weights for each position in window (should sum to 1.0)

    Returns:
        List of weighted moving averages

    Raises:
        ValueError: If weights is empty or weights don't sum to ~1.0

    Example:
        >>> data = [1, 2, 3, 4, 5]
        >>> weights = [0.1, 0.2, 0.3, 0.4]  # More weight to recent
        >>> result = await weighted_moving_average(data, weights)
    """
    if not weights:
        raise ValueError("weights cannot be empty")
    if abs(sum(weights) - 1.0) > 1e-6:
        raise ValueError(f"weights must sum to 1.0, got {sum(weights)}")

    window = len(weights)
    if window > len(data):
        raise ValueError("window size cannot exceed data length")

    result = []
    for i in range(len(data) - window + 1):
        window_data = data[i : i + window]
        wma = sum(w * x for w, x in zip(weights, window_data))
        result.append(wma)

        if i % 10 == 0:
            await asyncio.sleep(0)

    return result


async def autocorrelation(data: List[float], lag: int) -> float:
    """
    Compute autocorrelation at a given lag.

    Measures correlation between series and itself shifted by 'lag' periods.

    Args:
        data: Time series data
        lag: Lag period (must be < len(data))

    Returns:
        Autocorrelation coefficient (-1 to 1)

    Raises:
        ValueError: If lag < 1 or lag >= len(data)

    Example:
        >>> data = [1, 2, 3, 4, 5, 6, 7, 8]
        >>> acf_1 = await autocorrelation(data, 1)  # Correlation with 1-step lag
    """
    if lag < 1:
        raise ValueError("lag must be >= 1")
    if lag >= len(data):
        raise ValueError("lag must be < data length")

    n = len(data)
    mean = sum(data) / n

    # Compute numerator: sum of (x_t - mean)(x_{t-lag} - mean)
    numerator = sum((data[i] - mean) * (data[i - lag] - mean) for i in range(lag, n))

    # Compute denominator: sum of (x_t - mean)^2
    denominator = sum((x - mean) ** 2 for x in data)

    await asyncio.sleep(0)

    if abs(denominator) < 1e-10:
        return 0.0

    return numerator / denominator


async def partial_autocorrelation(data: List[float], lag: int) -> float:
    """
    Compute partial autocorrelation at a given lag.

    PACF measures correlation after removing effect of intermediate lags.

    Args:
        data: Time series data
        lag: Lag period

    Returns:
        Partial autocorrelation coefficient

    Raises:
        ValueError: If lag < 1 or lag >= len(data)

    Example:
        >>> data = [1, 2, 3, 4, 5, 6, 7, 8]
        >>> pacf_2 = await partial_autocorrelation(data, 2)
    """
    if lag < 1:
        raise ValueError("lag must be >= 1")
    if lag >= len(data):
        raise ValueError("lag must be < data length")

    # Simplified PACF using Yule-Walker equations
    # For exact PACF, would need to solve system of equations
    # This is an approximation

    acf_values = [await autocorrelation(data, k) for k in range(1, lag + 1)]

    await asyncio.sleep(0)

    # For lag 1, PACF = ACF
    if lag == 1:
        return acf_values[0]

    # Simplified approximation for higher lags
    return acf_values[-1]


async def seasonal_decompose(data: List[float], period: int) -> Dict[str, List[float]]:
    """
    Decompose time series into trend, seasonal, and residual components.

    Uses additive model: data = trend + seasonal + residual

    Args:
        data: Time series data
        period: Seasonal period (e.g., 12 for monthly data with yearly seasonality)

    Returns:
        Dictionary with 'trend', 'seasonal', 'residual' components

    Raises:
        ValueError: If period < 2 or period > len(data)/2

    Example:
        >>> data = [10, 12, 15, 11, 10, 12, 15, 11, 10, 12, 15, 11]
        >>> result = await seasonal_decompose(data, period=4)
    """
    if period < 2:
        raise ValueError("period must be >= 2")
    if period > len(data) // 2:
        raise ValueError("period too large for data length")

    n = len(data)

    # Compute trend using centered moving average
    trend: List[Optional[float]] = []
    half_window = period // 2

    for i in range(n):
        if i < half_window or i >= n - half_window:
            trend.append(None)  # No trend at edges
        else:
            window_data = data[i - half_window : i + half_window + 1]
            trend.append(sum(window_data) / len(window_data))

        if i % 10 == 0:
            await asyncio.sleep(0)

    # Compute detrended series
    detrended = [
        data[i] - trend[i] if trend[i] is not None else 0.0  # type: ignore[operator]
        for i in range(n)
    ]

    # Compute seasonal component (average for each period position)
    seasonal = []
    for i in range(n):
        period_pos = i % period
        # Average all values at this position in the period
        period_values = [detrended[j] for j in range(period_pos, n, period) if trend[j] is not None]
        if period_values:
            seasonal.append(sum(period_values) / len(period_values))
        else:
            seasonal.append(0.0)

    # Compute residual
    residual = []
    for i in range(n):
        if trend[i] is not None:
            residual.append(data[i] - trend[i] - seasonal[i])  # type: ignore[operator]
        else:
            residual.append(0.0)

    # Replace None with 0.0 in trend for consistent output
    trend_clean = [t if t is not None else 0.0 for t in trend]

    return {"trend": trend_clean, "seasonal": seasonal, "residual": residual}


async def detect_trend(data: List[float]) -> Dict[str, Any]:
    """
    Detect trend in time series using linear regression.

    Args:
        data: Time series data

    Returns:
        Dictionary with 'slope', 'intercept', 'strength' (R²)

    Raises:
        ValueError: If data has fewer than 2 points

    Example:
        >>> data = [1, 2, 3, 4, 5]  # Strong upward trend
        >>> result = await detect_trend(data)
        >>> # result['slope'] > 0, result['strength'] near 1.0
    """
    if len(data) < 2:
        raise ValueError("Need at least 2 data points")

    n = len(data)
    x = list(range(n))

    # Linear regression: y = slope * x + intercept
    x_mean = sum(x) / n
    y_mean = sum(data) / n

    numerator = sum((x[i] - x_mean) * (data[i] - y_mean) for i in range(n))
    denominator = sum((x[i] - x_mean) ** 2 for i in range(n))

    await asyncio.sleep(0)

    if abs(denominator) < 1e-10:
        return {"slope": 0.0, "intercept": y_mean, "strength": 0.0}

    slope = numerator / denominator
    intercept = y_mean - slope * x_mean

    # Compute R² (coefficient of determination)
    predictions = [slope * x[i] + intercept for i in range(n)]
    ss_tot = sum((data[i] - y_mean) ** 2 for i in range(n))
    ss_res = sum((data[i] - predictions[i]) ** 2 for i in range(n))

    r_squared = 1 - (ss_res / ss_tot) if abs(ss_tot) > 1e-10 else 0.0

    return {"slope": slope, "intercept": intercept, "strength": r_squared}


async def detect_seasonality(data: List[float], max_period: int = 12) -> Dict[str, Any]:
    """
    Detect seasonality by finding period with strongest autocorrelation.

    Args:
        data: Time series data
        max_period: Maximum period to check

    Returns:
        Dictionary with 'period', 'strength' (autocorrelation)

    Raises:
        ValueError: If max_period < 2

    Example:
        >>> data = [1, 2, 1, 2, 1, 2, 1, 2]  # Period 2 pattern
        >>> result = await detect_seasonality(data, max_period=4)
    """
    if max_period < 2:
        raise ValueError("max_period must be >= 2")

    max_period = min(max_period, len(data) // 2)

    best_period = 0
    best_strength = 0.0

    for period in range(2, max_period + 1):
        if period < len(data):
            acf = await autocorrelation(data, period)
            if acf > best_strength:
                best_strength = acf
                best_period = period

    return {"period": best_period, "strength": best_strength}


async def detrend(data: List[float]) -> List[float]:
    """
    Remove linear trend from time series.

    Args:
        data: Time series data

    Returns:
        Detrended series

    Example:
        >>> data = [1, 2, 3, 4, 5]  # Linear trend
        >>> result = await detrend(data)
        >>> # Result centered around 0
    """
    trend_info = await detect_trend(data)
    slope = trend_info["slope"]
    intercept = trend_info["intercept"]

    detrended = [data[i] - (slope * i + intercept) for i in range(len(data))]

    await asyncio.sleep(0)
    return detrended


async def deseasonalize(data: List[float], period: int) -> List[float]:
    """
    Remove seasonal component from time series.

    Args:
        data: Time series data
        period: Seasonal period

    Returns:
        Deseasonalized series

    Example:
        >>> data = [10, 12, 15, 11, 10, 12, 15, 11]
        >>> result = await deseasonalize(data, period=4)
    """
    decomp = await seasonal_decompose(data, period)
    seasonal = decomp["seasonal"]

    deseasonalized = [data[i] - seasonal[i] for i in range(len(data))]

    await asyncio.sleep(0)
    return deseasonalized


async def differencing(data: List[float], lag: int = 1) -> List[float]:
    """
    Apply differencing to make series stationary.

    Computes: diff[i] = data[i] - data[i-lag]

    Args:
        data: Time series data
        lag: Difference lag (default 1)

    Returns:
        Differenced series (length = len(data) - lag)

    Raises:
        ValueError: If lag < 1

    Example:
        >>> data = [1, 3, 6, 10, 15]
        >>> result = await differencing(data, lag=1)
        >>> # [2, 3, 4, 5] (differences)
    """
    if lag < 1:
        raise ValueError("lag must be >= 1")
    if lag >= len(data):
        raise ValueError("lag must be < data length")

    result = [data[i] - data[i - lag] for i in range(lag, len(data))]

    await asyncio.sleep(0)
    return result


async def holt_winters_forecast(
    data: List[float],
    period: int,
    alpha: float = 0.5,
    beta: float = 0.5,
    gamma: float = 0.5,
    forecast_periods: int = 1,
) -> Dict[str, Any]:
    """
    Holt-Winters exponential smoothing forecast (additive).

    Triple exponential smoothing with level, trend, and seasonality.

    Args:
        data: Historical time series data
        period: Seasonal period
        alpha: Level smoothing (0 < alpha < 1)
        beta: Trend smoothing (0 < beta < 1)
        gamma: Seasonal smoothing (0 < gamma < 1)
        forecast_periods: Number of periods to forecast

    Returns:
        Dictionary with 'forecast', 'level', 'trend', 'seasonal'

    Example:
        >>> data = [10, 12, 15, 11] * 3  # Repeating pattern
        >>> result = await holt_winters_forecast(data, period=4, forecast_periods=4)
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    if not 0 < beta < 1:
        raise ValueError("beta must be in (0, 1)")
    if not 0 < gamma < 1:
        raise ValueError("gamma must be in (0, 1)")
    if forecast_periods < 1:
        raise ValueError("forecast_periods must be >= 1")

    n = len(data)

    # Initialize components
    level = sum(data[:period]) / period
    trend = 0.0
    seasonal = [data[i] - level for i in range(period)]

    levels = [level]
    trends = [trend]

    # Update components for each observation
    for i in range(period, n):
        prev_level = level
        prev_trend = trend

        level = alpha * (data[i] - seasonal[i % period]) + (1 - alpha) * (prev_level + prev_trend)
        trend = beta * (level - prev_level) + (1 - beta) * prev_trend
        seasonal[i % period] = gamma * (data[i] - level) + (1 - gamma) * seasonal[i % period]

        levels.append(level)
        trends.append(trend)

        if i % 10 == 0:
            await asyncio.sleep(0)

    # Generate forecast
    forecast = []
    for i in range(forecast_periods):
        forecast_val = level + (i + 1) * trend + seasonal[i % period]
        forecast.append(forecast_val)

    return {
        "forecast": forecast,
        "level": level,
        "trend": trend,
        "seasonal": seasonal,
    }


async def exponential_smoothing(
    data: List[float], alpha: float = 0.3, forecast_periods: int = 1
) -> Dict[str, Any]:
    """
    Simple exponential smoothing forecast.

    Args:
        data: Historical data
        alpha: Smoothing factor (0 < alpha < 1)
        forecast_periods: Number of periods to forecast

    Returns:
        Dictionary with 'forecast' and 'smoothed' series

    Example:
        >>> data = [1, 2, 3, 4, 5]
        >>> result = await exponential_smoothing(data, alpha=0.5, forecast_periods=2)
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    if forecast_periods < 1:
        raise ValueError("forecast_periods must be >= 1")

    smoothed = await exponential_moving_average(data, alpha)

    # Forecast is just the last smoothed value (flat forecast)
    last_smoothed = smoothed[-1]
    forecast = [last_smoothed] * forecast_periods

    return {"forecast": forecast, "smoothed": smoothed}


async def moving_average_forecast(
    data: List[float], window: int, forecast_periods: int = 1
) -> List[float]:
    """
    Moving average forecast.

    Forecast is the average of the last 'window' observations.

    Args:
        data: Historical data
        window: Window size
        forecast_periods: Number of periods to forecast

    Returns:
        List of forecasts

    Example:
        >>> data = [1, 2, 3, 4, 5]
        >>> result = await moving_average_forecast(data, window=3, forecast_periods=2)
    """
    if window < 1:
        raise ValueError("window must be >= 1")
    if window > len(data):
        raise ValueError("window cannot exceed data length")
    if forecast_periods < 1:
        raise ValueError("forecast_periods must be >= 1")

    # Last window average
    last_window = data[-window:]
    forecast_value = sum(last_window) / window

    await asyncio.sleep(0)

    return [forecast_value] * forecast_periods


async def lagged_values(data: List[float], lags: List[int]) -> Dict[int, List[float]]:
    """
    Create lagged versions of time series.

    Args:
        data: Time series data
        lags: List of lag periods

    Returns:
        Dictionary mapping lag -> lagged series

    Example:
        >>> data = [1, 2, 3, 4, 5]
        >>> result = await lagged_values(data, [1, 2])
        >>> # {1: [2, 3, 4, 5], 2: [3, 4, 5]}
    """
    result = {}

    for lag in lags:
        if lag < 1 or lag >= len(data):
            continue
        result[lag] = data[lag:]

    await asyncio.sleep(0)
    return result


async def rolling_std(data: List[float], window: int) -> List[float]:
    """
    Rolling standard deviation.

    Args:
        data: Time series data
        window: Window size

    Returns:
        List of rolling standard deviations

    Example:
        >>> data = [1, 2, 3, 4, 5]
        >>> result = await rolling_std(data, window=3)
    """
    if window < 2:
        raise ValueError("window must be >= 2")
    if window > len(data):
        raise ValueError("window cannot exceed data length")

    result = []

    for i in range(len(data) - window + 1):
        window_data = data[i : i + window]
        mean = sum(window_data) / window
        variance = sum((x - mean) ** 2 for x in window_data) / window
        std = math.sqrt(variance)
        result.append(std)

        if i % 10 == 0:
            await asyncio.sleep(0)

    return result


async def rolling_variance(data: List[float], window: int) -> List[float]:
    """
    Rolling variance.

    Args:
        data: Time series data
        window: Window size

    Returns:
        List of rolling variances

    Example:
        >>> data = [1, 2, 3, 4, 5]
        >>> result = await rolling_variance(data, window=3)
    """
    if window < 2:
        raise ValueError("window must be >= 2")
    if window > len(data):
        raise ValueError("window cannot exceed data length")

    result = []

    for i in range(len(data) - window + 1):
        window_data = data[i : i + window]
        mean = sum(window_data) / window
        variance = sum((x - mean) ** 2 for x in window_data) / window
        result.append(variance)

        if i % 10 == 0:
            await asyncio.sleep(0)

    return result


async def seasonal_strength(data: List[float], period: int) -> float:
    """
    Measure strength of seasonality (0 to 1).

    Args:
        data: Time series data
        period: Seasonal period

    Returns:
        Seasonal strength (higher = stronger seasonality)

    Example:
        >>> data = [1, 2, 1, 2, 1, 2, 1, 2]  # Strong seasonality
        >>> strength = await seasonal_strength(data, period=2)
    """
    decomp = await seasonal_decompose(data, period)
    seasonal = decomp["seasonal"]
    residual = decomp["residual"]

    # Variance of seasonal component vs residual
    seasonal_var = sum(x**2 for x in seasonal) / len(seasonal)
    residual_var = sum(x**2 for x in residual) / len(residual)

    total_var = seasonal_var + residual_var

    await asyncio.sleep(0)

    if total_var < 1e-10:
        return 0.0

    return seasonal_var / total_var


async def trend_strength(data: List[float]) -> float:
    """
    Measure strength of trend (0 to 1).

    Uses R² from linear regression.

    Args:
        data: Time series data

    Returns:
        Trend strength (higher = stronger trend)

    Example:
        >>> data = [1, 2, 3, 4, 5]  # Perfect linear trend
        >>> strength = await trend_strength(data)
        >>> # Should be close to 1.0
    """
    trend_info = await detect_trend(data)
    return trend_info["strength"]


async def stationarity_test(data: List[float]) -> Dict[str, Any]:
    """
    Simple stationarity test.

    Checks if mean and variance are relatively constant over time.

    Args:
        data: Time series data

    Returns:
        Dictionary with 'is_stationary', 'mean_stable', 'variance_stable'

    Example:
        >>> data = [1, 2, 1, 2, 1, 2]  # Stationary
        >>> result = await stationarity_test(data)
    """
    if len(data) < 4:
        return {"is_stationary": False, "mean_stable": False, "variance_stable": False}

    # Split data into two halves
    mid = len(data) // 2
    first_half = data[:mid]
    second_half = data[mid:]

    # Compute means
    mean1 = sum(first_half) / len(first_half)
    mean2 = sum(second_half) / len(second_half)

    # Compute variances
    var1 = sum((x - mean1) ** 2 for x in first_half) / len(first_half)
    var2 = sum((x - mean2) ** 2 for x in second_half) / len(second_half)

    # Check stability (within 20% of each other)
    mean_stable = abs(mean1 - mean2) < 0.2 * max(abs(mean1), abs(mean2), 1.0)
    variance_stable = abs(var1 - var2) < 0.2 * max(var1, var2, 1.0)

    is_stationary = mean_stable and variance_stable

    await asyncio.sleep(0)

    return {
        "is_stationary": is_stationary,
        "mean_stable": mean_stable,
        "variance_stable": variance_stable,
    }


__all__ = [
    "simple_moving_average",
    "exponential_moving_average",
    "weighted_moving_average",
    "autocorrelation",
    "partial_autocorrelation",
    "seasonal_decompose",
    "detect_trend",
    "detect_seasonality",
    "detrend",
    "deseasonalize",
    "differencing",
    "holt_winters_forecast",
    "exponential_smoothing",
    "moving_average_forecast",
    "lagged_values",
    "rolling_std",
    "rolling_variance",
    "seasonal_strength",
    "trend_strength",
    "stationarity_test",
]
