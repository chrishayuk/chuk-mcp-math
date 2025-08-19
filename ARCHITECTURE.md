# CHUK MCP Math Library Architecture

## Overview

This document describes the structural organization and module hierarchy of the CHUK MCP Math library. The library is designed as a comprehensive, async-native mathematical toolkit with clear domain separation and logical module organization.

For related documentation:
- **[PRINCIPLES.md](./PRINCIPLES.md)** - Core design principles and philosophy
- **[PATTERNS.md](./PATTERNS.md)** - Implementation patterns and code examples  
- **[ROADMAP.md](./ROADMAP.md)** - Development timeline and version planning

## Current Architecture

### Directory Structure

```
chuk_mcp_math/
├── mcp_decorator.py         # Core async MCP function decorator
├── mcp_pydantic_base.py     # Pydantic base with MCP optimizations
├── arithmetic/              # ✅ Basic operations (30+ functions)
│   ├── core/               # Basic ops, rounding, modular
│   └── comparison/         # Relational, extrema, tolerance
├── number_theory/          # ✅ Extensive coverage (340+ functions)
│   ├── primes.py           # Prime number operations
│   ├── divisibility.py     # GCD, LCM, divisors
│   ├── sequences/          # Fibonacci, Lucas, Catalan
│   ├── diophantine_equations.py
│   ├── continued_fractions.py
│   ├── farey_sequences.py
│   └── [15+ more modules]
├── trigonometry/           # ✅ Complete trig (120+ functions)
│   ├── basic_functions.py  # sin, cos, tan
│   ├── inverse_functions.py # asin, acos, atan
│   ├── hyperbolic.py       # sinh, cosh, tanh
│   ├── applications.py     # Navigation, physics
│   └── wave_analysis.py    # Harmonics, amplitude
├── statistics.py           # 📝 Partial implementation
├── geometry.py             # 📝 Basic 2D shapes only
├── sequences.py            # 📝 Basic sequences
└── constants.py            # 📝 Mathematical constants
```

## Proposed Complete Architecture

### Phase 1: Core Mathematical Foundations (Priority 1)

#### 1. Linear Algebra Module
```
linear_algebra/
├── vectors/
│   ├── operations.py       # dot, cross, scalar multiplication
│   ├── norms.py           # L1, L2, L∞ norms
│   └── projections.py     # Vector projections, orthogonalization
├── matrices/
│   ├── operations.py      # Addition, multiplication, transpose
│   ├── properties.py      # Determinant, trace, rank
│   ├── special.py         # Identity, diagonal, symmetric
│   └── factorization.py   # LU, QR, Cholesky
├── decompositions/
│   ├── eigenvalues.py     # Eigenvalues, eigenvectors
│   ├── svd.py            # Singular Value Decomposition
│   └── jordan.py         # Jordan normal form
├── solvers/
│   ├── linear_systems.py  # Ax = b solvers
│   ├── least_squares.py   # Overdetermined systems
│   └── iterative.py      # Jacobi, Gauss-Seidel
└── transformations/
    ├── rotations.py       # 2D/3D rotations
    ├── reflections.py     # Mirror transformations
    └── projections.py     # Orthogonal projections
```

#### 2. Calculus Module
```
calculus/
├── derivatives/
│   ├── numerical.py       # Forward, backward, central differences
│   ├── symbolic.py        # Symbolic differentiation
│   ├── partial.py         # Partial derivatives, gradients
│   └── higher_order.py    # Second, third derivatives
├── integrals/
│   ├── definite.py        # Definite integrals
│   ├── indefinite.py      # Antiderivatives
│   ├── multiple.py        # Double, triple integrals
│   └── line_integrals.py  # Path integrals
├── differential_equations/
│   ├── ode_solvers.py     # Euler, RK4, adaptive methods
│   ├── pde_solvers.py     # Finite difference, finite element
│   └── boundary_value.py  # Shooting, relaxation methods
├── optimization/
│   ├── unconstrained.py   # Gradient descent, Newton's method
│   ├── constrained.py     # Lagrange multipliers, KKT
│   └── global.py          # Simulated annealing, genetic
└── series/
    ├── taylor.py          # Taylor series expansion
    ├── fourier.py         # Fourier series, FFT
    └── power_series.py    # Convergence, manipulation
```

#### 3. Probability Module
```
probability/
├── distributions/
│   ├── discrete.py        # Binomial, Poisson, Geometric
│   ├── continuous.py      # Normal, Exponential, Gamma
│   ├── multivariate.py    # Multivariate normal, Dirichlet
│   └── special.py         # Beta, Chi-square, Student-t
├── random/
│   ├── generators.py      # PRNGs, seeds
│   ├── sampling.py        # Uniform, weighted sampling
│   └── shuffling.py       # Fisher-Yates, permutations
├── bayesian/
│   ├── bayes_theorem.py   # Prior, likelihood, posterior
│   ├── inference.py       # MAP, MCMC
│   └── networks.py        # Bayesian networks
├── markov/
│   ├── chains.py          # Transition matrices
│   ├── processes.py       # Continuous-time Markov
│   └── hidden.py          # HMM algorithms
└── monte_carlo/
    ├── integration.py     # MC integration
    ├── simulation.py      # MC simulation
    └── importance.py      # Importance sampling
```

#### 4. Enhanced Statistics Module
```
statistics/
├── descriptive/
│   ├── central_tendency.py # Mean, median, mode
│   ├── dispersion.py       # Variance, std dev, IQR
│   ├── shape.py            # Skewness, kurtosis
│   └── summary.py          # Five-number summary
├── inferential/
│   ├── hypothesis_testing.py # t-test, chi-square, ANOVA
│   ├── confidence_intervals.py
│   ├── power_analysis.py
│   └── effect_size.py
├── regression/
│   ├── linear.py           # Simple, multiple linear
│   ├── polynomial.py       # Polynomial regression
│   ├── logistic.py         # Binary, multinomial
│   └── robust.py           # RANSAC, Theil-Sen
├── correlation/
│   ├── parametric.py       # Pearson correlation
│   ├── non_parametric.py   # Spearman, Kendall
│   └── partial.py          # Partial correlation
├── time_series/
│   ├── decomposition.py    # Trend, seasonality
│   ├── arima.py           # ARIMA models
│   ├── forecasting.py     # Exponential smoothing
│   └── stationarity.py    # ADF test, differencing
└── multivariate/
    ├── pca.py             # Principal Component Analysis
    ├── factor_analysis.py  # Factor extraction
    ├── lda.py             # Linear Discriminant Analysis
    └── clustering.py      # K-means, hierarchical
```

### Phase 2: Computational Mathematics (Priority 2)

#### 5. Numerical Methods Module
```
numerical/
├── interpolation/
│   ├── polynomial.py      # Lagrange, Newton
│   ├── splines.py        # Cubic, B-splines
│   └── rbf.py            # Radial basis functions
├── approximation/
│   ├── least_squares.py  # Linear, nonlinear
│   ├── chebyshev.py     # Chebyshev approximation
│   └── pade.py          # Padé approximants
├── root_finding/
│   ├── bracketing.py    # Bisection, false position
│   ├── open_methods.py  # Newton-Raphson, secant
│   └── polynomial.py    # Polynomial roots
├── quadrature/
│   ├── newton_cotes.py  # Trapezoidal, Simpson's
│   ├── gaussian.py      # Gauss-Legendre, Gauss-Hermite
│   └── adaptive.py      # Adaptive quadrature
└── stability/
    ├── condition.py     # Condition numbers
    ├── error_analysis.py # Round-off, truncation
    └── convergence.py   # Convergence rates
```

#### 6. Complex Analysis Module
```
complex/
├── basic/
│   ├── operations.py    # Addition, multiplication
│   ├── conjugate.py    # Complex conjugate
│   └── polar.py        # Polar form, arg, modulus
├── functions/
│   ├── exponential.py  # Complex exp, log
│   ├── trigonometric.py # Complex sin, cos
│   └── hyperbolic.py   # Complex sinh, cosh
├── analysis/
│   ├── cauchy_riemann.py # Analyticity
│   ├── residues.py      # Residue theorem
│   └── contour.py       # Contour integration
└── transforms/
    ├── fourier.py       # FFT, DFT
    ├── laplace.py       # Laplace transform
    └── z_transform.py   # Z-transform
```

#### 7. Discrete Mathematics Module
```
discrete_math/
├── graph_theory/
│   ├── representations.py # Adjacency matrix, list
│   ├── traversal.py      # BFS, DFS
│   ├── shortest_path.py  # Dijkstra, Floyd-Warshall
│   ├── spanning_tree.py  # Kruskal, Prim
│   └── coloring.py       # Graph coloring
├── combinatorics/
│   ├── counting.py       # Permutations, combinations
│   ├── generating.py     # Generate all permutations
│   ├── partitions.py     # Integer partitions
│   └── catalan.py        # Catalan numbers
├── logic/
│   ├── boolean.py        # Boolean algebra
│   ├── truth_tables.py   # Truth table generation
│   ├── satisfiability.py # SAT solving
│   └── predicates.py     # Predicate logic
├── sets/
│   ├── operations.py     # Union, intersection
│   ├── relations.py      # Equivalence, ordering
│   └── cardinality.py    # Finite, countable, uncountable
└── algorithms/
    ├── sorting.py        # Comparison, counting sorts
    ├── searching.py      # Binary, interpolation search
    ├── dynamic_prog.py   # DP algorithms
    └── greedy.py         # Greedy algorithms
```

### Phase 3: Applied Mathematics (Priority 3)

#### 8. Enhanced Geometry Module
```
geometry/
├── euclidean_2d/
│   ├── primitives.py     # Points, lines, rays
│   ├── polygons.py       # Triangles, quadrilaterals
│   ├── circles.py        # Circles, arcs, sectors
│   └── transformations.py # 2D transforms
├── euclidean_3d/
│   ├── primitives.py     # Points, lines, planes
│   ├── polyhedra.py      # Tetrahedron, cube, etc.
│   ├── spheres.py        # Spheres, ellipsoids
│   └── transformations.py # 3D transforms
├── analytical/
│   ├── conic_sections.py # Ellipse, parabola, hyperbola
│   ├── parametric.py     # Parametric curves
│   └── implicit.py       # Implicit curves/surfaces
├── computational/
│   ├── convex_hull.py    # 2D/3D convex hull
│   ├── voronoi.py        # Voronoi diagrams
│   ├── delaunay.py       # Delaunay triangulation
│   └── intersection.py   # Line/polygon intersection
└── differential/
    ├── curves.py         # Curvature, torsion
    ├── surfaces.py       # Gaussian, mean curvature
    └── geodesics.py      # Shortest paths on surfaces
```

#### 9. Special Functions Module
```
special_functions/
├── gamma/
│   ├── gamma.py          # Gamma function
│   ├── beta.py           # Beta function
│   ├── digamma.py        # Digamma, polygamma
│   └── incomplete.py     # Incomplete gamma, beta
├── bessel/
│   ├── first_kind.py     # J_n(x)
│   ├── second_kind.py    # Y_n(x)
│   ├── modified.py       # I_n(x), K_n(x)
│   └── spherical.py      # Spherical Bessel
├── hypergeometric/
│   ├── gauss.py          # 2F1
│   ├── confluent.py      # 1F1, Kummer's function
│   └── generalized.py    # pFq
├── elliptic/
│   ├── integrals.py      # K(k), E(k), Π(n,k)
│   ├── functions.py      # Jacobi elliptic
│   └── theta.py          # Theta functions
└── orthogonal/
    ├── legendre.py       # Legendre polynomials
    ├── hermite.py        # Hermite polynomials
    ├── laguerre.py       # Laguerre polynomials
    └── chebyshev.py      # Chebyshev polynomials
```

#### 10. Financial Mathematics Module
```
finance/
├── interest/
│   ├── simple.py         # Simple interest
│   ├── compound.py       # Compound interest
│   ├── continuous.py     # Continuous compounding
│   └── annuities.py      # Present/future value
├── options/
│   ├── black_scholes.py  # BS pricing model
│   ├── greeks.py         # Delta, gamma, theta, vega
│   ├── binomial.py       # Binomial tree model
│   └── monte_carlo.py    # MC option pricing
├── bonds/
│   ├── pricing.py        # Bond pricing
│   ├── yield.py          # YTM, current yield
│   ├── duration.py       # Macaulay, modified duration
│   └── convexity.py      # Bond convexity
├── risk/
│   ├── var.py            # Value at Risk
│   ├── cvar.py           # Conditional VaR
│   ├── sharpe.py         # Sharpe ratio
│   └── drawdown.py       # Maximum drawdown
└── portfolio/
    ├── optimization.py   # Markowitz optimization
    ├── efficient_frontier.py
    ├── capm.py          # Capital Asset Pricing Model
    └── performance.py    # Portfolio metrics
```

### Phase 4: Specialized Domains (Priority 4)

#### 11. Mathematical Physics Module
```
physics/
├── mechanics/
│   ├── kinematics.py     # Position, velocity, acceleration
│   ├── dynamics.py       # Forces, momentum, energy
│   ├── rotational.py     # Angular motion, torque
│   └── oscillations.py   # SHM, damped, forced
├── waves/
│   ├── wave_equation.py  # 1D, 2D, 3D wave equation
│   ├── interference.py   # Superposition, beats
│   ├── diffraction.py    # Single slit, double slit
│   └── doppler.py        # Doppler effect
├── quantum/
│   ├── operators.py      # Quantum operators
│   ├── wavefunctions.py  # Schrödinger equation
│   ├── uncertainty.py    # Heisenberg uncertainty
│   └── spin.py           # Spin matrices, Pauli
├── relativity/
│   ├── lorentz.py        # Lorentz transformations
│   ├── spacetime.py      # Minkowski space
│   └── energy_momentum.py # E=mc², four-vectors
└── thermodynamics/
    ├── ideal_gas.py      # Ideal gas law
    ├── entropy.py        # Entropy, free energy
    ├── phase_transitions.py
    └── statistical.py    # Partition functions
```

#### 12. Cryptographic Mathematics Module
```
cryptography/
├── prime_generation/
│   ├── primality_tests.py # Miller-Rabin, Solovay-Strassen
│   ├── prime_gen.py       # Safe primes, Sophie Germain
│   └── factorization.py   # Pollard rho, quadratic sieve
├── modular/
│   ├── exponentiation.py  # Fast modular exponentiation
│   ├── inverse.py         # Modular multiplicative inverse
│   └── chinese_remainder.py # CRT
├── elliptic_curves/
│   ├── operations.py      # Point addition, doubling
│   ├── scalar_mult.py     # Scalar multiplication
│   └── ecdh.py           # ECDH key exchange
├── hash_primitives/
│   ├── merkle_tree.py    # Merkle tree construction
│   ├── bloom_filter.py   # Bloom filter operations
│   └── cryptographic_hash.py
└── protocols/
    ├── diffie_hellman.py  # DH key exchange
    ├── rsa_math.py        # RSA encryption math
    ├── elgamal.py         # ElGamal encryption
    └── zero_knowledge.py  # ZK proof primitives
```

## Module Naming Conventions

- **Modules**: Snake_case, descriptive names (e.g., `linear_algebra`, `number_theory`)
- **Functions**: Snake_case, verb_noun format when applicable (e.g., `calculate_eigenvalues`)
- **Constants**: UPPER_CASE (e.g., `MAX_ITERATIONS`)
- **Classes**: PascalCase (e.g., `MatrixDecomposition`)

## File Organization

```
module_name/
├── __init__.py          # Module exports and initialization
├── core.py             # Core/basic functionality
├── advanced.py         # Advanced operations
├── utils.py            # Helper functions
└── submodule/          # Logical grouping of related functions
    ├── __init__.py
    └── operations.py
```

## Extension Points

The architecture supports extensibility through:

1. **Module Plugins**: New mathematical domains as self-contained modules
2. **Decorator Extensions**: Custom decorators extending `mcp_decorator.py`
3. **Backend Adapters**: Alternative computation backends (NumPy, JAX, etc.)
4. **Caching Strategies**: Custom cache implementations
5. **Type Systems**: Extended validation through Pydantic models

## Core Components

### MCP Decorator System
- `mcp_decorator.py`: Core async function decorator with caching and performance tracking
- `mcp_pydantic_base.py`: Pydantic integration for type validation

### Function Registry
- Global registry of all MCP functions
- Namespace-based organization
- Performance metrics collection
- Async semaphore for concurrency control

## Dependencies

### Core Runtime
- Python 3.11+ (async features)
- Built-in: asyncio, math, typing, functools

### Optional Enhancements
- `pydantic>=2.11.1`: Runtime validation
- `numpy`: Optimized array operations
- `scipy`: Scientific algorithms

### Development Tools
- `pytest` + `pytest-asyncio`: Testing
- `hypothesis`: Property-based tests
- `ruff`: Linting and formatting
- `mypy`: Static type checking