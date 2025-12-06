#!/usr/bin/env python3
"""
CHUK MCP Math - Priority 2 Demo (Simple & Accurate)

Demonstrates the 40 new functions added in Priority 2:
- Time Series Analysis: 20 functions
- Inferential Statistics: 20 functions
"""

import asyncio


async def demo_time_series():
    """Demonstrate time series analysis."""
    print("\n" + "=" * 70)
    print("TIME SERIES ANALYSIS (20 functions)")
    print("=" * 70)

    from chuk_mcp_math.timeseries import (
        simple_moving_average,
        exponential_moving_average,
        autocorrelation,
        seasonal_decompose,
        holt_winters_forecast,
    )

    # Monthly sales data with seasonal pattern
    sales = [100, 105, 110, 115, 120, 125, 130, 135, 140, 145, 150, 155,
             200, 210, 220, 230, 110, 115, 120, 125]

    print("\n📊 Moving Averages")
    sma = await simple_moving_average(sales, window=3)
    print(f"Simple MA (last 3): {[round(x, 2) for x in sma[-3:]]}")

    ema = await exponential_moving_average(sales, alpha=0.3)
    print(f"Exponential MA (last 3): {[round(x, 2) for x in ema[-3:]]}")

    print("\n📈 Autocorrelation Analysis")
    acf_values = []
    for lag in range(1, 4):
        acf_val = await autocorrelation(sales, lag)
        acf_values.append(acf_val)
    print(f"ACF (lags 1-3): {[round(x, 3) for x in acf_values]}")

    print("\n🔬 Seasonal Decomposition")
    decomp = await seasonal_decompose(sales, period=4)
    print(f"Trend (last 3): {[round(x, 2) for x in decomp['trend'][-3:]]}")

    print("\n🔮 Holt-Winters Forecast")
    forecast = await holt_winters_forecast(sales, period=4, forecast_periods=3)
    print(f"Next 3 months: {[round(x, 2) for x in forecast['forecast']]}")

    print("\n✅ Time series analysis complete (5 of 20 functions shown)")


async def demo_inferential_stats():
    """Demonstrate inferential statistics."""
    print("\n" + "=" * 70)
    print("INFERENTIAL STATISTICS (20 functions)")
    print("=" * 70)

    from chuk_mcp_math.statistics import (
        t_test_two_sample,
        confidence_interval_mean,
        cohens_d,
        anova_one_way,
        chi_square_test,
    )

    # A/B Testing Example
    print("\n🧪 A/B Testing")
    control = [0.05, 0.06, 0.04, 0.05, 0.07]  # 5% conversion rate
    treatment = [0.08, 0.09, 0.07, 0.08, 0.10]  # 8% conversion rate

    result = await t_test_two_sample(control, treatment)
    print(f"t-statistic: {result['t_statistic']:.4f}")
    print(f"p-value: {result['p_value']:.4f}")
    print(f"Significant: {result['reject_null']}")

    effect = await cohens_d(control, treatment)
    print(f"Cohen's d: {effect:.4f} (effect size)")

    print("\n📊 Confidence Intervals")
    ci = await confidence_interval_mean(control, confidence_level=0.95)
    print(f"Control mean: {ci['mean']:.4f} ± {ci['margin_of_error']:.4f}")
    print(f"95% CI: [{ci['lower']:.4f}, {ci['upper']:.4f}]")

    print("\n🏭 ANOVA (3 Groups)")
    group_a = [95, 97, 96, 98, 95]
    group_b = [92, 94, 93, 95, 92]
    group_c = [97, 99, 98, 100, 97]

    anova = await anova_one_way([group_a, group_b, group_c])
    print(f"F-statistic: {anova['F_statistic']:.4f}")
    print(f"p-value: {anova['p_value']:.4f}")
    print(f"Significant: {anova['reject_null']}")

    print("\n🎲 Chi-Square Test")
    observed = [60, 30, 10]
    expected = [50, 40, 10]

    chi_sq = await chi_square_test(observed, expected)
    print(f"χ² statistic: {chi_sq['chi_square_statistic']:.4f}")
    print(f"p-value: {chi_sq['p_value']:.4f}")

    print("\n✅ Inferential statistics complete (5 of 20 functions shown)")


async def main():
    """Run demonstrations."""
    print("=" * 70)
    print("CHUK MCP MATH - PRIORITY 2 FEATURES (v0.3)")
    print("=" * 70)

    await demo_time_series()
    await demo_inferential_stats()

    print("\n" + "=" * 70)
    print("✨ PRIORITY 2 COMPLETE - 40 NEW FUNCTIONS")
    print("=" * 70)
    print("\nBusiness Applications:")
    print("  ✅ Sales forecasting & demand planning")
    print("  ✅ A/B testing & hypothesis testing")
    print("  ✅ Trend detection & seasonality analysis")
    print("  ✅ Statistical significance & effect sizes")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
