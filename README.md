# CHUK MCP Math Library

🧮 **Comprehensive Mathematical Functions Library for AI Models (Async Native)**

A cutting-edge collection of 572+ mathematical functions organized by domain, designed specifically for AI model execution with async-native performance, MCP integration, and robust error handling.

## ✨ Key Features

- **🚀 Async Native**: All 572+ functions built from the ground up for async/await patterns
- **🔢 Comprehensive Coverage**: 572+ functions across 25+ specialized mathematical domains
- **✅ 94% Test Coverage**: 4,429 tests passing with 94% code coverage
- **🎯 MCP Integration**: Model Context Protocol compatible with smart caching and performance optimization
- **📐 Mathematical Domains**: Number theory (340+ functions), trigonometry (120+ functions), linear algebra, calculus, statistics, geometry, probability
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
├── probability/         # Distributions & sampling (98-100% coverage)
│   ├── distributions.py # Normal, uniform distributions
│   └── additional_distributions.py # Exponential, binomial
├── statistics/          # Statistical analysis (90% coverage)
│   └── statistics.py    # Mean, variance, correlation, regression
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

## 📐 New Features - Phase 1 & 2 Complete!

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
