# CHUK MCP Math Library

🧮 **Comprehensive Mathematical Functions Library for AI Models (Async Native)**

A cutting-edge collection of **657+ mathematical functions** organized by domain, designed specifically for AI model execution with async-native performance, MCP integration, and robust error handling.

## ✨ Key Features

- **🔬 Enhanced Numerical Engine**: Production-ready optimization, interpolation, series expansions, integration, and derivatives (100% coverage)
- **📊 Business Analytics Suite**: Time series analysis (forecasting, seasonality) + inferential statistics (hypothesis testing, A/B testing)
- **🚀 Async Native**: All 657+ functions built from the ground up for async/await patterns
- **🔢 Comprehensive Coverage**: 657+ functions across 28+ specialized mathematical domains
- **✅ 94% Test Coverage**: 4,578 tests passing with 94% code coverage
- **🎯 MCP Integration**: Model Context Protocol compatible with smart caching and performance optimization
- **📐 Mathematical Domains**: Number theory (340+ functions), trigonometry (120+ functions), linear algebra, calculus, statistics, geometry, probability, time series, numerical methods
- **🌊 Streaming Support**: Real-time computation with backpressure handling
- **💾 Smart Caching**: Async-optimized memory caching with TTL and LRU eviction
- **⚡ Performance Optimized**: Built-in performance metrics and concurrency control
- **🔒 Type Safe**: Complete type safety with mypy (0 errors)
- **📚 Educational Ready**: Rich examples, comprehensive demos, and educational applications
- **🛠️ Zero External Dependencies**: CLI uses stdlib argparse, no click required

## 🏗️ Architecture Overview

```
chuk_mcp_math/
├── arithmetic/           # Core arithmetic operations (100% coverage)
│   ├── core/            # Basic operations, rounding, modular
│   └── comparison/      # Relational, extrema, tolerance
├── calculus/            # Numerical methods (100% coverage)
│   ├── derivatives.py   # Central, forward, backward differences
│   ├── integration.py   # Trapezoidal, Simpson's, midpoint rules
│   └── root_finding.py  # Bisection, Newton-Raphson, secant methods
├── linear_algebra/      # Matrix & vector operations (100% coverage)
│   ├── matrices/        # Matrix ops, determinants, solvers
│   └── vectors/         # Vector operations, norms, projections
├── numerical/           # 🆕 Advanced numerical methods (95-100% coverage)
│   ├── interpolation.py # Linear, Lagrange, Newton, splines (7 functions)
│   ├── optimization.py  # Gradient descent, Nelder-Mead, golden section (6 functions)
│   └── series.py        # Taylor, Fourier, power series (12 functions)
├── timeseries/          # 🆕 Time series analysis (92% coverage)
│   └── analysis.py      # SMA, EMA, autocorrelation, forecasting (20 functions)
├── statistics/          # 🆕 Complete statistics suite (90-92% coverage)
│   ├── descriptive.py   # Mean, variance, correlation, regression (15 functions)
│   └── inference.py     # T-tests, ANOVA, CI, hypothesis testing (20 functions)
├── probability/         # Distributions & sampling (98-100% coverage)
│   ├── distributions.py # Normal, uniform distributions
│   └── additional_distributions.py # Exponential, binomial
├── geometry/            # Geometric calculations (92-98% coverage)
│   ├── distances.py     # Euclidean, Manhattan, great circle
│   ├── intersections.py # Line, circle, polygon intersections
│   └── shapes.py        # Area, perimeter, centroid calculations
├── number_theory/       # 18 specialized modules, 340+ functions
│   ├── primes/          # Prime operations and testing
│   ├── divisibility/    # GCD, LCM, divisors
│   ├── sequences/       # Fibonacci, Lucas, Catalan
│   ├── special_numbers/ # Perfect, abundant, amicable
│   ├── diophantine_equations/ # Linear, Pell's equation
│   ├── continued_fractions/   # CF expansions, convergents
│   ├── farey_sequences/       # Farey sequences, Ford circles
│   └── ...              # 11 more specialized modules
└── trigonometry/        # 8 modules, 120+ functions
    ├── basic_functions/ # sin, cos, tan (radians & degrees)
    ├── inverse_functions/ # asin, acos, atan, atan2
    ├── hyperbolic/      # sinh, cosh, tanh
    ├── wave_analysis/   # Amplitude, frequency, harmonics
    ├── applications/    # Navigation, physics, GPS
    └── ...              # 3 more modules
```

## 🚀 Quick Start

### Installation

```bash
# Basic installation (no external dependencies!)
pip install chuk-mcp-math

# With Pydantic validation support (recommended)
pip install chuk-mcp-math[pydantic]

# With development dependencies
pip install chuk-mcp-math[dev]

# Install all optional dependencies
pip install chuk-mcp-math[pydantic,dev]
```

### Basic Usage

```python
import asyncio
from chuk_mcp_math import number_theory, trigonometry, calculus, statistics

async def main():
    # Number theory operations
    is_prime_result = await number_theory.is_prime(17)
    fibonacci_result = await number_theory.fibonacci(10)
    gcd_result = await number_theory.gcd(48, 18)

    # Trigonometric operations
    sin_result = await trigonometry.sin(3.14159/4)
    distance = await trigonometry.distance_haversine(40.7128, -74.0060, 34.0522, -118.2437)

    # Calculus operations
    f = lambda x: x**2
    derivative = await calculus.derivative_central(f, 3.0)  # f'(3) = 6
    integral = await calculus.integrate_simpson(f, 0.0, 1.0)  # ∫₀¹ x² dx = 1/3

    # Statistics operations
    data = [1, 2, 3, 4, 5]
    mean = await statistics.mean(data)
    variance = await statistics.variance(data)

    print(f"is_prime(17): {is_prime_result}")
    print(f"fibonacci(10): {fibonacci_result}")
    print(f"gcd(48, 18): {gcd_result}")
    print(f"sin(π/4): {sin_result:.6f}")
    print(f"NYC to LA distance: {distance['distance_km']:.0f} km")
    print(f"derivative of x² at x=3: {derivative:.4f}")
    print(f"integral of x² from 0 to 1: {integral:.6f}")

asyncio.run(main())
```

### 🆕 New Capabilities (Priority 1 & 2)

#### Optimization & Numerical Methods

```python
from chuk_mcp_math.numerical import optimization, interpolation, series

# Gradient descent optimization
f = lambda x: x**2 - 4*x + 4  # Minimize (x-2)²
result = await optimization.gradient_descent(
    f, lambda x: [2*x[0] - 4], [0.0], learning_rate=0.1
)
print(f"Minimum at x = {result['x']}")  # [2.0]

# Interpolation for missing values
x_data = [0, 1, 2, 3]
y_data = [0, 1, 4, 9]
y_interp = await interpolation.lagrange_interpolate(1.5, x_data, y_data)
print(f"Interpolated value: {y_interp}")  # 2.25

# Series approximations
taylor_exp = await series.exp_series(1.0, n=10)
print(f"e ≈ {taylor_exp}")  # 2.718...
```

#### Time Series Analysis & Forecasting

```python
from chuk_mcp_math.timeseries import analysis

# Sales data
sales = [100, 120, 115, 140, 135, 160, 155, 180]

# Moving averages for trend
sma = await analysis.simple_moving_average(sales, window=3)
ema = await analysis.exponential_moving_average(sales, alpha=0.3)

# Detect trend and seasonality
trend = await analysis.detect_trend(sales)
print(f"Trend slope: {trend['slope']:.2f}")

# Forecast future values
forecast = await analysis.holt_winters_forecast(
    sales, period=4, forecast_periods=3
)
print(f"Next 3 periods: {forecast['forecast']}")
```

#### Inferential Statistics & Hypothesis Testing

```python
from chuk_mcp_math.statistics import (
    t_test_two_sample, anova_one_way,
    confidence_interval_mean, cohens_d
)

# A/B testing
control = [20, 22, 21, 23, 22]
treatment = [25, 27, 26, 28, 27]

# T-test for difference
result = await t_test_two_sample(control, treatment)
print(f"P-value: {result['p_value']:.4f}")
print(f"Reject null: {result['reject_null']}")

# Effect size
effect = await cohens_d(control, treatment)
print(f"Cohen's d: {effect:.2f}")

# Confidence interval
ci = await confidence_interval_mean(treatment, confidence_level=0.95)
print(f"95% CI: [{ci['lower']:.2f}, {ci['upper']:.2f}]")

# ANOVA for multiple groups
group1 = [20, 22, 21]
group2 = [25, 27, 26]
group3 = [30, 32, 31]
anova = await anova_one_way([group1, group2, group3])
print(f"F-statistic: {anova['F_statistic']:.2f}")
```

### CLI Usage

The library includes a powerful CLI with no external dependencies (uses stdlib argparse):

```bash
# List all available functions
python -m chuk_mcp_math.cli.main list

# Search for functions
python -m chuk_mcp_math.cli.main search prime

# Describe a function
python -m chuk_mcp_math.cli.main describe is_prime

# Call a function
python -m chuk_mcp_math.cli.main call is_prime 17

# Filter by module
python -m chuk_mcp_math.cli.main list --module number_theory

# Show detailed information
python -m chuk_mcp_math.cli.main list --detailed
```

### MCP Function Decorator

```python
from chuk_mcp_math.mcp_decorator import mcp_function

@mcp_function(
    description="Calculate compound interest with async optimization",
    cache_strategy="memory",
    cache_ttl_seconds=3600,
    estimated_cpu_usage="medium"
)
async def compound_interest(principal: float, rate: float, time: float, compounds_per_year: int = 12) -> float:
    """Calculate compound interest: A = P(1 + r/n)^(nt)"""
    import math
    return principal * math.pow(1 + rate/compounds_per_year, compounds_per_year * time)
```

## 🔬 Core Numerical Methods (Production Ready)

The library provides a complete numerical computing engine with production-grade implementations of fundamental algorithms. All methods are async-native with 100% test coverage.

### Root Finding
Solve equations of the form f(x) = 0 with multiple robust algorithms:

```python
from chuk_mcp_math.calculus import root_finding

# Find where f(x) = x² - 4 = 0 (root at x = 2)
f = lambda x: x**2 - 4
f_prime = lambda x: 2*x

# Bisection method (guaranteed convergence)
root_bisect = await root_finding.root_find_bisection(f, 0.0, 3.0, tolerance=1e-6)
print(f"Bisection: x = {root_bisect:.6f}")  # 2.000000

# Newton-Raphson (quadratic convergence)
root_newton = await root_finding.root_find_newton(f, f_prime, 1.0, tolerance=1e-10)
print(f"Newton-Raphson: x = {root_newton:.10f}")  # 2.0000000000

# Secant method (no derivative needed)
root_secant = await root_finding.root_find_secant(f, 1.0, 3.0, tolerance=1e-8)
print(f"Secant: x = {root_secant:.8f}")  # 2.00000000
```

**Applications**: Optimization, engineering design, financial modeling (IRR, break-even analysis)

### Numerical Integration
Compute definite integrals with adaptive precision:

```python
from chuk_mcp_math.calculus import integration

# Integrate f(x) = x² from 0 to 1 (exact value = 1/3)
f = lambda x: x**2

# Trapezoidal rule (good for smooth functions)
integral_trap = await integration.integrate_trapezoid(f, 0.0, 1.0, steps=1000)
print(f"Trapezoid: {integral_trap:.6f}")  # 0.333333

# Simpson's rule (higher accuracy)
integral_simp = await integration.integrate_simpson(f, 0.0, 1.0, steps=1000)
print(f"Simpson's: {integral_simp:.8f}")  # 0.33333333

# Midpoint rule (robust for oscillatory functions)
integral_mid = await integration.integrate_midpoint(f, 0.0, 1.0, steps=1000)
print(f"Midpoint: {integral_mid:.6f}")  # 0.333333
```

**Applications**: Physics simulations, area/volume calculations, probability (CDF), signal processing

### Numerical Derivatives
Compute derivatives with configurable precision and stability:

```python
from chuk_mcp_math.calculus import derivatives

f = lambda x: x**3  # f'(x) = 3x²

# Central difference (most accurate, O(h²))
deriv_central = await derivatives.derivative_central(f, 2.0, h=1e-5)
print(f"Central: f'(2) = {deriv_central:.6f}")  # 12.000000 (exact)

# Forward difference (O(h))
deriv_forward = await derivatives.derivative_forward(f, 2.0, h=1e-5)
print(f"Forward: f'(2) = {deriv_forward:.6f}")  # ~12.000030

# Backward difference (O(h))
deriv_backward = await derivatives.derivative_backward(f, 2.0, h=1e-5)
print(f"Backward: f'(2) = {deriv_backward:.6f}")  # ~11.999970
```

**Applications**: Optimization gradients, physics (velocity, acceleration), machine learning (gradient descent), sensitivity analysis

### Linear Systems
Solve systems of linear equations Ax = b:

```python
from chuk_mcp_math.linear_algebra import matrices

# Solve 2x2 system using Cramer's rule
A = [[3, 2], [1, 4]]
b = [7, 6]
solution = await matrices.matrix_solve_2x2(A, b)
print(f"Solution: x = {solution}")  # [1.4, 1.3]

# Gaussian elimination (scales to larger systems)
A_large = [[2, 1, -1], [-3, -1, 2], [-2, 1, 2]]
b_large = [8, -11, -3]
solution_ge = await matrices.gaussian_elimination(A_large, b_large)
print(f"Gaussian: x = {solution_ge}")  # [2, 3, -1]
```

**Applications**: Circuit analysis, structural engineering, economics (input-output models), computer graphics

### Why These Methods Matter

These aren't toy implementations—they're the **numerical backbone** that powers:
- **Business Analytics**: Break-even analysis, IRR calculation, demand forecasting
- **Engineering**: Circuit simulation, stress analysis, control systems
- **Finance**: Option pricing, risk models, portfolio optimization
- **Science**: Physics simulations, chemical kinetics, climate modeling
- **AI/ML**: Optimization algorithms, parameter estimation, feature engineering

All methods include comprehensive error handling, configurable precision, and production-tested edge case coverage.

## 📐 Additional Mathematical Domains

### Phase 1: Core Numerical Engine (100% Complete)

#### Linear Algebra (100% Coverage)
```python
from chuk_mcp_math.linear_algebra import matrices, vectors

# Matrix operations
A = [[1, 2], [3, 4]]
B = [[5, 6], [7, 8]]
product = await matrices.matrix_multiply(A, B)
transpose = await matrices.matrix_transpose(A)
det = await matrices.matrix_det_2x2(A)

# Solve linear systems
solution = await matrices.matrix_solve_2x2(A, [5, 6])  # Ax = b
gaussian = await matrices.gaussian_elimination(A, [5, 6])

# Vector operations
v1 = [1, 2, 3]
v2 = [4, 5, 6]
dot = await vectors.dot_product(v1, v2)
norm = await vectors.vector_norm(v1)
normalized = await vectors.normalize_vector(v1)
```

#### Calculus (100% Coverage)
```python
from chuk_mcp_math.calculus import derivatives, integration, root_finding

# Derivatives
f = lambda x: x**2
central = await derivatives.derivative_central(f, 3.0)  # Most accurate
forward = await derivatives.derivative_forward(f, 3.0)
backward = await derivatives.derivative_backward(f, 3.0)

# Integration
integral_trap = await integration.integrate_trapezoid(f, 0.0, 1.0, 1000)
integral_simp = await integration.integrate_simpson(f, 0.0, 1.0, 1000)  # More accurate
integral_mid = await integration.integrate_midpoint(f, 0.0, 1.0, 1000)

# Root finding
f_root = lambda x: x**2 - 4  # Root at x = 2
root_bisect = await root_finding.root_find_bisection(f_root, 0.0, 3.0)
root_newton = await root_finding.root_find_newton(f_root, lambda x: 2*x, 1.0)
root_secant = await root_finding.root_find_secant(f_root, 1.0, 3.0)
```

#### Probability & Statistics (98-100% Coverage)
```python
from chuk_mcp_math.probability import distributions, additional_distributions
from chuk_mcp_math import statistics

# Normal distribution
pdf = await distributions.normal_pdf(0.0, 0.0, 1.0)
cdf = await distributions.normal_cdf(0.0, 0.0, 1.0)
samples = await distributions.normal_sample(100, 0.0, 1.0, seed=42)

# Exponential & Binomial
exp_pdf = await additional_distributions.exponential_pdf(1.0, 1.0)
binom_pmf = await additional_distributions.binomial_pmf(3, 5, 0.5)

# Statistics
data_x = [1, 2, 3, 4, 5]
data_y = [2, 4, 6, 8, 10]
cov = await statistics.covariance(data_x, data_y)
corr = await statistics.correlation(data_x, data_y)
regression = await statistics.linear_regression(data_x, data_y)
print(f"Slope: {regression['slope']}, R²: {regression['r_squared']}")
```

### Phase 2: Scientific & Geometry Toolkit (100% Complete)

#### Geometry (92-98% Coverage)
```python
from chuk_mcp_math.geometry import distances, intersections, shapes

# Distance calculations
dist_2d = await distances.geom_distance((0, 0), (3, 4))  # Euclidean
dist_3d = await distances.geom_distance_3d((0, 0, 0), (1, 1, 1))
manhattan = await distances.geom_manhattan_distance((0, 0), (3, 4))

# GPS distance (Haversine formula)
nyc_to_la = await distances.geom_great_circle_distance(
    40.7128, -74.0060,  # NYC
    34.0522, -118.2437  # LA
)
print(f"Distance: {nyc_to_la['km']:.0f} km")

# Intersections
line_int = await intersections.geom_line_intersection(
    {"p1": (0, 0), "p2": (1, 1)},
    {"p1": (0, 1), "p2": (1, 0)}
)
circle_int = await intersections.geom_circle_intersection(
    {"x": 0, "y": 0, "r": 5},
    {"x": 5, "y": 0, "r": 5}
)

# Shapes
area = await shapes.geom_polygon_area([(0, 0), (4, 0), (4, 3), (0, 3)])
circle_area = await shapes.geom_circle_area(5.0)
triangle_area = await shapes.geom_triangle_area(3.0, 4.0, 5.0)
```

#### Advanced Statistics (90% Coverage)
```python
from chuk_mcp_math import statistics

# Moving average
data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
ma = await statistics.moving_average(data, window=3)

# Z-scores and outlier detection
z_scores = await statistics.z_scores(data)
outliers = await statistics.detect_outliers(data, method="zscore", threshold=2.0)
print(f"Outliers found: {outliers['num_outliers']}")
```

## 🎯 Production-Quality Demos

### 🆕 Priority 2 Demo - Business Analytics (v0.3)
NEW! Showcase of advanced business analytics and forecasting:

```bash
uv run python examples/demos/demo_priority2_simple.py
```

Features:
- **Time Series Analysis**: Moving averages, autocorrelation, seasonal decomposition, Holt-Winters forecasting
- **Inferential Statistics**: A/B testing, t-tests, ANOVA, chi-square, confidence intervals, effect sizes
- **Business Applications**: Sales forecasting, demand planning, hypothesis testing, statistical significance
- **40 new functions** demonstrating cutting-edge analytics capabilities

### AI Analyst Demo (Phase 1)
Comprehensive demonstration of numerical computing capabilities:

```bash
python examples/demos/ai_analyst_v0.py
```

Features:
- Linear algebra: Matrix operations, solving systems
- Calculus: Derivatives, integration, root finding
- Probability: Distribution analysis, sampling
- Statistics: Regression, correlation analysis
- Combined analysis: Business metrics prediction

### F1 Track Geometry Demo (Phase 2)
Advanced motorsport analytics with real-world applications:

```bash
python examples/demos/f1_track_geometry.py
```

Features:
- Track geometry analysis (Monaco GP circuit)
- Lap time modeling with sector breakdown
- Tire degradation and fuel strategy
- Statistical outlier detection
- GPS-based distance calculations

## 📊 Test Coverage & Quality

### Coverage Highlights

```
╔══════════════════════════════════════════════════════════════════╗
║  OVERALL COVERAGE: 94% (10,495 statements, 643 missed)         ║
║  TESTS PASSING: 4,429 ✅                                        ║
║  30 FILES WITH 100% COVERAGE ✅                                 ║
║  LINTING: PASSED ✅                                              ║
║  FORMATTING: PASSED ✅                                           ║
║  TYPE CHECKING: PASSED ✅                                        ║
║  SECURITY: PASSED ✅                                             ║
╚══════════════════════════════════════════════════════════════════╝
```

**Module-by-Module Coverage:**
- ✅ Linear Algebra: **100%** (matrices, vectors, solvers)
- ✅ Calculus: **100%** (derivatives, integration, root finding)
- ✅ Probability: **98-100%** (all distributions)
- ✅ Statistics: **90%** (descriptive, regression, outlier detection)
- ✅ Geometry: **92-98%** (distances, intersections, shapes)
- ✅ Number Theory: **90-100%** across all 18 modules
- ✅ Trigonometry: **90-100%** across all 8 modules
- ✅ Arithmetic: **100%**

### Quality Metrics

- **4,429 tests passing** (0 failures)
- **Zero external dependencies** for core functionality
- **Type-safe** with mypy (0 errors)
- **Security-audited** with bandit (0 issues)
- **Formatted** with ruff (153 files)
- **Comprehensive error handling** with edge case tests

## 🛠️ Development & Testing

### Running Tests

```bash
# Run all quality checks (lint, format, typecheck, security, tests)
make check

# Run unit tests
make test

# Run with coverage report
make test-cov

# Run specific test suites
pytest tests/test_phase1.py -v          # Phase 1 (linear algebra, calculus, probability, stats)
pytest tests/test_phase2_geometry.py -v # Phase 2 (geometry, advanced stats)
pytest tests/cli/ -v                    # CLI tests
```

### Running Examples

```bash
# Run Phase 1 demo (AI Analyst)
python examples/demos/ai_analyst_v0.py

# Run Phase 2 demo (F1 Track Geometry)
python examples/demos/f1_track_geometry.py

# Run all demos
make run-demos

# Run comprehensive number theory examples
python examples/applications/demo_number_theory.py

# Run comprehensive trigonometry examples
python examples/applications/demo_trigonometry.py
```

## 📚 Documentation

### Core Documentation
- [Architecture](./ARCHITECTURE.md) - Module structure and organization
- [Phase 1 Complete](./PHASE1_COMPLETE.md) - Linear algebra, calculus, probability, statistics
- [Phase 2 Complete](./PHASE2_COMPLETE.md) - Geometry toolkit, advanced statistics
- [Principles](./PRINCIPLES.md) - Design philosophy
- [Patterns](./PATTERNS.md) - Implementation patterns
- [Roadmap](./ROADMAP.md) - Development timeline

### Testing Documentation
- [Testing Guide](./docs/testing/TESTING.md) - Complete testing documentation
- [Unit Testing](./docs/testing/UNIT_TESTING.md) - Isolation and mocking
- [Math Testing](./docs/testing/MATH_PATTERNS.md) - Numerical testing patterns

## 🎓 Use Cases

### Educational
- **Mathematics Curricula**: Complete coverage of undergraduate topics
- **Research Projects**: Advanced algorithms for graduate-level research
- **Competitive Programming**: High-performance algorithms

### Professional
- **AI/ML Applications**: Mathematical functions optimized for AI execution
- **Engineering Simulations**: Physics, signal processing, control systems
- **Financial Modeling**: Statistical analysis, optimization, risk assessment
- **Scientific Computing**: Numerical methods, data analysis

### Research
- **Number Theory**: Prime distribution, Diophantine equations
- **Computational Mathematics**: High-precision calculations
- **Data Science**: Statistical analysis, machine learning preprocessing

## 🔗 Integration with CHUK Stack

This library is designed to integrate seamlessly with the broader CHUK ecosystem:

### MCP Server Integration
```python
# MCP server can expose functions via the protocol
from chuk_mcp_math.mcp_decorator import mcp_function

@mcp_function(
    description="Solve quadratic equations ax² + bx + c = 0",
    cache_strategy="memory",
    estimated_cpu_usage="low"
)
async def solve_quadratic(a: float, b: float, c: float) -> dict:
    """Find roots using quadratic formula."""
    from chuk_mcp_math.calculus import root_finding
    import math

    discriminant = b**2 - 4*a*c

    if discriminant < 0:
        return {"real_roots": [], "complex_roots": 2}
    elif discriminant == 0:
        root = -b / (2*a)
        return {"real_roots": [root], "complex_roots": 0}
    else:
        sqrt_disc = math.sqrt(discriminant)
        root1 = (-b + sqrt_disc) / (2*a)
        root2 = (-b - sqrt_disc) / (2*a)
        return {"real_roots": [root1, root2], "complex_roots": 0}
```

### Claude AI Integration
The library is optimized for AI model consumption:
- **Self-documenting**: Rich metadata in MCP function specs
- **Error resilient**: Comprehensive validation and error messages
- **Performance hints**: CPU/memory usage estimates guide model execution
- **Caching strategies**: Avoid redundant computation in conversation flows

### Business Intelligence
Combine numerical methods with statistical analysis:
```python
from chuk_mcp_math.calculus import root_finding, integration
from chuk_mcp_math import statistics

async def forecast_revenue(historical_data: list, months_ahead: int):
    """Business forecasting with numerical methods."""
    # Linear regression for trend
    x_data = list(range(len(historical_data)))
    regression = await statistics.linear_regression(x_data, historical_data)

    # Forecast future values
    predictions = []
    for month in range(len(historical_data), len(historical_data) + months_ahead):
        pred = regression['slope'] * month + regression['intercept']
        predictions.append(pred)

    # Calculate cumulative revenue (integration)
    f = lambda t: regression['slope'] * t + regression['intercept']
    total_revenue = await integration.integrate_simpson(
        f, 0, len(historical_data) + months_ahead, steps=100
    )

    return {
        "predictions": predictions,
        "total_revenue": total_revenue,
        "r_squared": regression['r_squared']
    }
```

## 📝 License

MIT License - see LICENSE file for details.

## 🤝 Contributing

Contributions welcome! Please see CONTRIBUTING.md for guidelines.

## 📊 Function Statistics

- **Total Functions**: 572+
- **Test Coverage**: 94% overall (4,429 tests)
- **Zero Failures**: All tests passing
- **Type Safety**: 0 mypy errors
- **Phase 1 Complete**: Linear Algebra, Calculus, Probability, Statistics (100% coverage)
- **Phase 2 Complete**: Geometry, Advanced Statistics (92-98% coverage)
- **Number Theory**: 340+ functions (90-100% coverage)
- **Trigonometry**: 120+ functions (90-100% coverage)
- **30 Files**: 100% test coverage

---

**Built with ❤️ for the mathematical computing community**

*Async-native • MCP-optimized • Educational-ready • Research-grade • Production-tested*
