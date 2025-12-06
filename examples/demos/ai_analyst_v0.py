#!/usr/bin/env python3
"""
AI Analyst v0 - Phase 1 Demo

Demonstrates all Phase 1 capabilities of chuk-mcp-math:
- Linear Algebra (vectors & matrices)
- Calculus (derivatives, integration, root finding)
- Probability (distributions, sampling)
- Statistics (descriptive stats, regression, correlation)

This demo shows how an AI assistant could use these tools to perform real data analysis.
"""

import asyncio
import math

# Import all Phase 1 modules
from chuk_mcp_math.linear_algebra.vectors.operations import dot_product
from chuk_mcp_math.linear_algebra.vectors.norms import vector_norm, normalize_vector
from chuk_mcp_math.linear_algebra.matrices.operations import (
    matrix_multiply,
    matrix_det_2x2,
)
from chuk_mcp_math.linear_algebra.matrices.solvers import matrix_solve_2x2
from chuk_mcp_math.calculus.derivatives import derivative_central
from chuk_mcp_math.calculus.integration import integrate_trapezoid, integrate_simpson
from chuk_mcp_math.calculus.root_finding import root_find_bisection
from chuk_mcp_math.probability.distributions import normal_pdf, normal_cdf, normal_sample
from chuk_mcp_math.statistics import (
    standard_deviation,
    correlation,
    linear_regression,
    comprehensive_stats,
)


def print_section(title: str):
    """Print a formatted section header."""
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print(f"{'=' * 80}\n")


async def demo_linear_algebra():
    """Demonstrate linear algebra capabilities."""
    print_section("LINEAR ALGEBRA - Vectors & Matrices")

    # Vector operations
    print("📐 Vector Operations")
    v1 = [3, 4]
    v2 = [1, 2]

    print(f"  v1 = {v1}")
    print(f"  v2 = {v2}")

    norm_v1 = await vector_norm(v1)
    print(f"  ||v1|| = {norm_v1}")

    normalized = await normalize_vector(v1)
    print(f"  v1_normalized = [{normalized[0]:.4f}, {normalized[1]:.4f}]")

    dot = await dot_product(v1, v2)
    print(f"  v1 · v2 = {dot}")

    # Matrix operations
    print("\n🔲 Matrix Operations")
    A = [[2, 1], [1, 3]]
    B = [[1, 0], [0, 1]]

    print(f"  A = {A}")
    print(f"  B = {B}")

    det_A = await matrix_det_2x2(A)
    print(f"  det(A) = {det_A}")

    product = await matrix_multiply(A, B)
    print(f"  A × B = {product}")

    # Solving linear systems
    print("\n⚡ Solving Linear Systems")
    print("  System: 2x + y = 5")
    print("          x + 3y = 6")

    solution = await matrix_solve_2x2(A, [5, 6])
    print(f"  Solution: x = {solution[0]:.4f}, y = {solution[1]:.4f}")


async def demo_calculus():
    """Demonstrate calculus capabilities."""
    print_section("CALCULUS - Derivatives, Integration & Root Finding")

    # Derivatives
    print("📊 Numeric Derivatives")
    f = lambda x: x**2
    x_point = 3.0

    deriv = await derivative_central(f, x_point)
    print("  f(x) = x²")
    print(f"  f'({x_point}) ≈ {deriv:.6f}  (exact: {2 * x_point})")

    # Integration
    print("\n∫ Numeric Integration")
    g = lambda x: x**2
    a, b = 0.0, 1.0

    integral_trap = await integrate_trapezoid(g, a, b, 1000)
    integral_simp = await integrate_simpson(g, a, b, 1000)

    print("  ∫₀¹ x² dx")
    print(f"  Trapezoidal rule: {integral_trap:.6f}")
    print(f"  Simpson's rule:   {integral_simp:.6f}")
    print(f"  Exact value:      {1 / 3:.6f}")

    # Root finding
    print("\n🎯 Root Finding")
    h = lambda x: x**2 - 4  # Root at x = 2

    root = await root_find_bisection(h, 0.0, 3.0)
    print("  f(x) = x² - 4")
    print(f"  Root found: x = {root:.6f}  (exact: 2.0)")


async def demo_probability():
    """Demonstrate probability distributions."""
    print_section("PROBABILITY - Distributions & Sampling")

    # Normal distribution
    print("📈 Normal Distribution N(0,1)")

    pdf_0 = await normal_pdf(0.0, 0.0, 1.0)
    print(f"  PDF at x=0: {pdf_0:.6f}  (= 1/√(2π))")

    cdf_0 = await normal_cdf(0.0, 0.0, 1.0)
    print(f"  CDF at x=0: {cdf_0:.6f}  (= 0.5)")

    cdf_196 = await normal_cdf(1.96, 0.0, 1.0)
    print(f"  CDF at x=1.96: {cdf_196:.6f}  (97.5th percentile)")

    # Sampling
    print("\n🎲 Random Sampling")
    samples = await normal_sample(1000, 0.0, 1.0, seed=42)

    sample_mean = sum(samples) / len(samples)
    sample_std = math.sqrt(sum((x - sample_mean) ** 2 for x in samples) / (len(samples) - 1))

    print("  Generated 1000 samples from N(0,1)")
    print(f"  Sample mean: {sample_mean:.4f}  (expected: 0.0)")
    print(f"  Sample std:  {sample_std:.4f}  (expected: 1.0)")


async def demo_statistics_and_regression():
    """Demonstrate statistical analysis and linear regression."""
    print_section("STATISTICS & REGRESSION - Data Analysis")

    # Simulated sales data (months 1-12)
    months = list(range(1, 13))
    sales = [102, 115, 128, 135, 148, 155, 170, 178, 185, 195, 208, 220]

    print("📊 Sales Data Analysis")
    print(f"  Months: {months}")
    print(f"  Sales:  {sales}")

    # Descriptive statistics
    print("\n📈 Descriptive Statistics")
    stats = await comprehensive_stats(sales)

    print(f"  Count:    {stats['count']}")
    print(f"  Mean:     ${stats['mean']:.2f}")
    print(f"  Median:   ${stats['median']:.2f}")
    print(f"  Std Dev:  ${stats['std_dev']:.2f}")
    print(f"  Min:      ${stats['min']:.2f}")
    print(f"  Max:      ${stats['max']:.2f}")
    print(f"  Range:    ${stats['range']:.2f}")

    # Correlation analysis
    print("\n🔗 Correlation Analysis")
    corr = await correlation(months, sales)
    print(f"  Correlation(months, sales) = {corr:.4f}")
    print(f"  Interpretation: {'Strong positive' if corr > 0.8 else 'Moderate'} relationship")

    # Linear regression
    print("\n📉 Linear Regression: sales = m × month + b")
    reg_result = await linear_regression(months, sales)

    print(f"  Slope (m):     ${reg_result['slope']:.2f}/month")
    print(f"  Intercept (b): ${reg_result['intercept']:.2f}")
    print(f"  R² score:      {reg_result['r_squared']:.4f}  (goodness of fit)")

    # Forecast
    print("\n🔮 Sales Forecast")
    forecast_month = 15
    forecast_sales = reg_result["slope"] * forecast_month + reg_result["intercept"]
    print(f"  Predicted sales for month {forecast_month}: ${forecast_sales:.2f}")

    # Find break-even point (if we had costs)
    print("\n💰 Break-Even Analysis")
    # Assume: Fixed costs = $150, Variable costs = $5/unit
    # Revenue = sales (in $), Cost = 150 + 5*units
    # For simplicity, assume units ≈ sales/10

    # Find when sales = 200 using root finding
    sales_target = 200.0
    f_target = lambda m: reg_result["slope"] * m + reg_result["intercept"] - sales_target

    try:
        month_for_target = await root_find_bisection(f_target, 1.0, 24.0)
        print(f"  Month to reach $200 in sales: {month_for_target:.1f}")
    except ValueError as e:
        print(f"  Target not reachable in range: {e}")


async def demo_combined_analysis():
    """Demonstrate combining multiple tools for complex analysis."""
    print_section("COMBINED ANALYSIS - Real-World Scenario")

    print("🏢 Business Scenario: Revenue Growth Analysis\n")
    print("A company wants to understand their revenue growth pattern and optimize pricing.")

    # Revenue data (in thousands)
    time_periods = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    revenue = [50, 52, 55, 59, 64, 70, 77, 85, 94, 104]

    print(f"Time periods: {time_periods}")
    print(f"Revenue ($k): {revenue}")

    # Step 1: Analyze growth rate using calculus
    print("\n1️⃣  Growth Rate Analysis (Calculus)")
    reg = await linear_regression(time_periods, revenue)

    # Create a revenue function from regression
    revenue_func = lambda t: reg["slope"] * t + reg["intercept"]

    # Estimate instantaneous growth rate at period 5
    growth_rate = await derivative_central(revenue_func, 5.0)
    print(f"   Instantaneous growth rate at t=5: ${growth_rate:.2f}k per period")
    print(f"   Average growth rate: ${reg['slope']:.2f}k per period")

    # Step 2: Calculate total revenue using integration
    print("\n2️⃣  Total Revenue Calculation (Integration)")
    total_revenue = await integrate_simpson(revenue_func, 1.0, 10.0, 100)
    print(f"   Total revenue over 10 periods: ${total_revenue:.2f}k")

    # Step 3: Statistical confidence
    print("\n3️⃣  Statistical Confidence (Statistics)")
    print(f"   R² = {reg['r_squared']:.4f}")
    if reg["r_squared"] > 0.95:
        print("   ✅ Excellent fit - predictions are highly reliable")
    elif reg["r_squared"] > 0.8:
        print("   ✅ Good fit - predictions are reliable")
    else:
        print("   ⚠️  Moderate fit - use predictions with caution")

    # Step 4: Find target milestones
    print("\n4️⃣  Target Milestones (Root Finding)")
    target_revenue = 150.0  # $150k

    target_func = lambda t: revenue_func(t) - target_revenue
    try:
        target_period = await root_find_bisection(target_func, 1.0, 20.0)
        print(f"   Expected to reach ${target_revenue}k at period {target_period:.1f}")
    except ValueError:
        periods_to_150 = (target_revenue - reg["intercept"]) / reg["slope"]
        print(f"   Expected to reach ${target_revenue}k at period {periods_to_150:.1f}")

    # Step 5: Risk analysis using probability
    print("\n5️⃣  Risk Analysis (Probability)")
    # Assume revenue forecast has uncertainty modeled as N(μ, σ)
    pred_std = await standard_deviation(
        [revenue[i] - reg["predicted"][i] for i in range(len(revenue))], population=False
    )

    period_12 = 12
    pred_revenue_12 = revenue_func(period_12)

    print(f"   Forecast for period {period_12}: ${pred_revenue_12:.2f}k ± ${2 * pred_std:.2f}k")

    # Probability of exceeding $125k at period 12
    prob = await normal_cdf(125.0, pred_revenue_12, pred_std)
    prob_exceed = 1 - prob
    print(f"   Probability of revenue > $125k: {prob_exceed * 100:.1f}%")


async def main():
    """Run all demos."""
    print("\n" + "=" * 80)
    print("  🚀 AI ANALYST V0 - PHASE 1 DEMO")
    print("  chuk-mcp-math: Numeric Brain for CHUK")
    print("=" * 80)

    await demo_linear_algebra()
    await demo_calculus()
    await demo_probability()
    await demo_statistics_and_regression()
    await demo_combined_analysis()

    print("\n" + "=" * 80)
    print("  ✅ PHASE 1 COMPLETE!")
    print("  All Phase 1 capabilities demonstrated successfully.")
    print("=" * 80)
    print()

    # Summary
    print_section("PHASE 1 CAPABILITIES SUMMARY")
    capabilities = {
        "Linear Algebra": [
            "Vector operations (add, dot product, norms)",
            "Vector normalization",
            "Matrix operations (multiply, transpose, determinant)",
            "Solving linear systems (2x2, 3x3, n×n)",
        ],
        "Calculus": [
            "Numeric derivatives (central, forward, backward)",
            "Numeric integration (trapezoidal, Simpson's, midpoint)",
            "Root finding (bisection, Newton, secant)",
        ],
        "Probability": [
            "Normal distribution (PDF, CDF)",
            "Random sampling (normal, uniform)",
        ],
        "Statistics": [
            "Descriptive statistics (mean, median, variance, std dev)",
            "Covariance and correlation",
            "Linear regression with R²",
        ],
    }

    for category, items in capabilities.items():
        print(f"\n{category}:")
        for item in items:
            print(f"  ✓ {item}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
