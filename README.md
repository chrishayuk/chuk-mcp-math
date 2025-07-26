# CHUK MCP Math Library

🧮 **Comprehensive Mathematical Functions Library for AI Models (Async Native)**

A cutting-edge collection of 400+ mathematical functions organized by domain, designed specifically for AI model execution with async-native performance, MCP integration, and robust error handling.

## ✨ Key Features

- **🚀 Async Native**: All functions built from the ground up for async/await patterns
- **🔢 Comprehensive Coverage**: 400+ functions across 20+ specialized mathematical domains
- **🎯 MCP Integration**: Model Context Protocol compatible with smart caching and performance optimization
- **📐 Mathematical Domains**: Number theory, trigonometry, arithmetic, sequences, special functions
- **🌊 Streaming Support**: Real-time computation with backpressure handling
- **💾 Smart Caching**: Async-optimized memory caching with TTL and LRU eviction
- **⚡ Performance Optimized**: Built-in performance metrics and concurrency control
- **🔒 Security Focused**: Sandboxed execution with trusted/untrusted function classification
- **📚 Educational Ready**: Rich examples, documentation, and educational applications

## 🏗️ Architecture Overview

```
chuk_mcp_math/
│   ├── arithmetic/           # Core arithmetic operations
│   │   ├── core/            # Basic operations, rounding, modular
│   │   └── comparison/      # Relational, extrema, tolerance
│   ├── number_theory/       # 18 specialized modules, 340+ functions
│   │   ├── primes/          # Prime operations and testing
│   │   ├── divisibility/    # GCD, LCM, divisors
│   │   ├── sequences/       # Fibonacci, Lucas, Catalan
│   │   ├── special_numbers/ # Perfect, abundant, amicable
│   │   ├── diophantine_equations/ # Linear, Pell's equation
│   │   ├── continued_fractions/   # CF expansions, convergents
│   │   ├── farey_sequences/       # Farey sequences, Ford circles
│   │   └── ...              # 11 more specialized modules
│   └── trigonometry/        # 8 modules, 120+ functions
│       ├── basic_functions/ # sin, cos, tan (radians & degrees)
│       ├── inverse_functions/ # asin, acos, atan, atan2
│       ├── hyperbolic/      # sinh, cosh, tanh
│       ├── wave_analysis/   # Amplitude, frequency, harmonics
│       ├── applications/    # Navigation, physics, GPS
│       └── ...              # 3 more modules
├── mcp_decorator.py         # Async-native MCP function decorator
└── mcp_pydantic_base.py     # Enhanced Pydantic base with MCP optimizations
```

## 🚀 Quick Start

### Installation

```bash
pip install chuk-mcp-math
```

### Basic Usage

```python
import asyncio
from chuk_mcp_math import number_theory, trigonometry

async def main():
    # Number theory operations
    is_prime_result = await number_theory.is_prime(17)
    fibonacci_result = await number_theory.fibonacci(10)
    gcd_result = await number_theory.gcd(48, 18)
    
    # Trigonometric operations
    sin_result = await trigonometry.sin(3.14159/4)
    distance = await trigonometry.distance_haversine(40.7128, -74.0060, 34.0522, -118.2437)
    
    print(f"is_prime(17): {is_prime_result}")
    print(f"fibonacci(10): {fibonacci_result}")
    print(f"gcd(48, 18): {gcd_result}")
    print(f"sin(π/4): {sin_result:.6f}")
    print(f"NYC to LA distance: {distance['distance_km']:.0f} km")

asyncio.run(main())
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

## 📐 Mathematical Domains

### Number Theory (340+ Functions)

The most comprehensive number theory library available, featuring:

#### 🔢 Core Operations
- **Primes**: `is_prime()`, `next_prime()`, `prime_factors()`, `twin_primes()`
- **Divisibility**: `gcd()`, `lcm()`, `divisors()`, `euler_totient()`
- **Sequences**: `fibonacci()`, `lucas_number()`, `catalan_number()`

#### 🧮 Advanced Modules
- **Diophantine Equations**: Linear, Pell's equation, Pythagorean triples
- **Continued Fractions**: CF expansions, convergents, rational approximations
- **Farey Sequences**: Ford circles, Stern-Brocot tree, mediants
- **Special Numbers**: Amicable pairs, vampire numbers, Keith numbers
- **Modular Arithmetic**: Chinese Remainder Theorem, quadratic residues

```python
# Advanced number theory examples
import asyncio
from chuk_mcp_math import number_theory

async def advanced_demo():
    # Solve Pell's equation x² - 2y² = 1
    pell_solution = await number_theory.solve_pell_equation(2)
    print(f"Pell equation solution: {pell_solution}")
    
    # Find continued fraction expansion of π
    pi_cf = await number_theory.continued_fraction_expansion(3.14159, 8)
    print(f"π continued fraction: {pi_cf}")
    
    # Generate Farey sequence F₅
    farey_5 = await number_theory.farey_sequence(5)
    print(f"Farey sequence F₅: {farey_5}")
    
    # Find amicable pairs up to 10000
    amicable = await number_theory.find_amicable_pairs(10000)
    print(f"Amicable pairs: {amicable}")

asyncio.run(advanced_demo())
```

### Trigonometry (120+ Functions)

Complete trigonometric capabilities for navigation, physics, and signal processing:

#### 📐 Core Functions
- **Basic**: `sin()`, `cos()`, `tan()` with radians/degrees variants
- **Inverse**: `asin()`, `acos()`, `atan()`, `atan2()` with full quadrant support
- **Hyperbolic**: `sinh()`, `cosh()`, `tanh()` and their inverses

#### 🌊 Applications
- **Navigation**: GPS distance calculation, bearing computation, triangulation
- **Wave Analysis**: Amplitude extraction, harmonic analysis, Fourier basics
- **Physics**: Pendulum motion, spring oscillations, damping analysis

```python
# Navigation and wave analysis examples
import asyncio
from chuk_mcp_math import trigonometry

async def navigation_demo():
    # Calculate great circle distance between cities
    nyc_to_london = await trigonometry.distance_haversine(
        40.7128, -74.0060,  # NYC coordinates
        51.5074, -0.1278    # London coordinates
    )
    print(f"NYC to London: {nyc_to_london['distance_km']:.0f} km")
    
    # Calculate bearing
    bearing = await trigonometry.bearing_calculation(
        40.7128, -74.0060, 51.5074, -0.1278
    )
    print(f"Bearing: {bearing['bearing_degrees']:.1f}° ({bearing['compass_direction']})")
    
    # Analyze wave with amplitude and phase
    wave_analysis = await trigonometry.amplitude_from_coefficients(3, 4)
    print(f"Wave amplitude: {wave_analysis['amplitude']:.3f}")
    print(f"Phase shift: {wave_analysis['phase_degrees']:.1f}°")

asyncio.run(navigation_demo())
```

### Arithmetic Operations

Reorganized structure with logical categorization:

#### 🔧 Core Operations
```python
from chuk_mcp_math.arithmetic.core import add, multiply, power, sqrt
from chuk_mcp_math.arithmetic.comparison import minimum, maximum, clamp

# Basic operations with async support
result = await add(5, 3)
product = await multiply(4, 7)
square_root = await sqrt(16)

# Comparison operations
min_val = await minimum(10, 20)
max_val = await maximum(10, 20)
clamped = await clamp(15, 5, 25)
```

## 🎯 Advanced Features

### Async-Native Performance

All functions built for async/await with:
- **Concurrency Control**: Configurable semaphores prevent resource exhaustion
- **Strategic Yielding**: Automatic yielding in long-running operations
- **Performance Metrics**: Built-in timing and execution statistics

### Smart Caching System

```python
@mcp_function(
    cache_strategy="memory",        # or "file", "hybrid", "async_lru"
    cache_ttl_seconds=3600,        # 1 hour TTL
    max_concurrent_executions=5     # Concurrency limit
)
async def expensive_calculation(n: int) -> int:
    # Expensive computation here
    await asyncio.sleep(1)  # Simulate work
    return n ** 2
```

### Streaming Support

```python
@mcp_function(
    supports_streaming=True,
    streaming_mode="chunked"
)
async def generate_primes(limit: int) -> AsyncIterator[int]:
    """Stream prime numbers up to limit."""
    for num in range(2, limit + 1):
        if await is_prime(num):
            yield num
```

### Educational Applications

```python
async def educational_demo():
    """Comprehensive number analysis for students."""
    n = 60
    
    # Analyze number properties
    factors = await number_theory.prime_factors(n)
    divisors = await number_theory.divisors(n)
    totient = await number_theory.euler_totient(n)
    
    print(f"Analysis of {n}:")
    print(f"Prime factorization: {' × '.join(map(str, factors))}")
    print(f"All divisors: {divisors}")
    print(f"Euler's totient φ({n}) = {totient}")
    
    # Check special properties
    is_abundant = await number_theory.is_abundant_number(n)
    is_harshad = await number_theory.is_harshad_number(n)
    print(f"Abundant: {is_abundant}, Harshad: {is_harshad}")
```

### Research Applications

```python
async def research_demo():
    """Research-level mathematical analysis."""
    
    # Prime distribution analysis
    gaps = await number_theory.prime_gaps_analysis(1000, 1100)
    print(f"Prime gaps 1000-1100: avg={gaps['avg_gap']}, max={gaps['max_gap']}")
    
    # Farey sequence density study
    density = await number_theory.density_analysis(15)
    print(f"Farey density constant: {density['estimated_constant']:.6f}")
    
    # Continued fraction convergence
    cf_analysis = await number_theory.cf_convergence_analysis(math.pi, 10)
    print(f"π convergence type: {cf_analysis['diophantine_type']}")
    
    # Cross-module relationships
    # Perfect numbers ↔ Mersenne primes
    for exp in [2, 3, 5, 7]:
        mersenne = 2**exp - 1
        if await number_theory.is_prime(mersenne):
            perfect = (2**(exp-1)) * mersenne
            print(f"Mersenne prime 2^{exp}-1 = {mersenne} → Perfect: {perfect}")
```

## 🔧 Performance & Optimization

### Built-in Metrics

```python
# Get performance statistics
stats = function.get_performance_stats()
print(f"Executions: {stats['execution_count']}")
print(f"Average duration: {stats['average_duration']:.4f}s")
print(f"Cache hit rate: {stats['cache_hit_rate']:.2%}")
print(f"Async yields: {stats['async_yields']}")
```

### Benchmarking

```python
async def benchmark_demo():
    import time
    
    # Benchmark large computations
    start = time.time()
    large_fib = await number_theory.fibonacci(1000)
    fib_time = time.time() - start
    print(f"fibonacci(1000): {fib_time:.4f}s")
    
    # Benchmark with caching
    await expensive_calculation.clear_cache()
    
    start = time.time()
    result1 = await expensive_calculation(100)  # Cache miss
    first_time = time.time() - start
    
    start = time.time()
    result2 = await expensive_calculation(100)  # Cache hit
    cached_time = time.time() - start
    
    print(f"First call: {first_time:.4f}s, Cached call: {cached_time:.6f}s")
    print(f"Speedup: {first_time/cached_time:.1f}x")
```

## 🛠️ Development & Testing

### Running the Demos

```bash
# Number theory comprehensive demo
python -m chuk_mcp_math.number_theory

# Trigonometry comprehensive demo  
python -m chuk_mcp_math.trigonometry

# Run specific demonstrations
python number_theory_demo.py
python trigonometry_demo.py
```

### Testing Individual Functions

```python
import asyncio
from chuk_mcp_math import number_theory

async def test_functions():
    # Test prime operations
    assert await number_theory.is_prime(17) == True
    assert await number_theory.is_prime(4) == False
    
    # Test Fibonacci
    assert await number_theory.fibonacci(10) == 55
    
    # Test GCD
    assert await number_theory.gcd(48, 18) == 6
    
    print("✅ All tests passed!")

asyncio.run(test_functions())
```

## 📊 Function Statistics

- **Total Functions**: 400+
- **Async Native**: 100%
- **Number Theory**: 340+ functions across 18 modules
- **Trigonometry**: 120+ functions across 8 modules
- **Arithmetic**: 30+ functions in reorganized structure
- **Performance Optimized**: Built-in caching, concurrency control
- **Educational Ready**: Comprehensive examples and documentation

## 🎓 Educational Use Cases

- **Mathematics Curricula**: Complete coverage of undergraduate number theory and trigonometry
- **Research Projects**: Advanced algorithms for graduate-level mathematical research
- **AI/ML Applications**: Mathematical functions optimized for AI model execution
- **Competitive Programming**: High-performance algorithms for contests
- **Professional Development**: Mathematical software for engineering applications

## 🔬 Research Applications

- **Number Theory Research**: Prime distribution, Diophantine equations, continued fractions
- **Cryptographic Analysis**: Modular arithmetic, quadratic residues, discrete logarithms
- **Mathematical Physics**: Trigonometric applications, oscillations, wave analysis
- **Computational Mathematics**: High-precision constants, approximation theory
- **Geometric Number Theory**: Farey sequences, Ford circles, lattice problems

## 🚀 Performance Highlights

- **Async-Native**: All functions built for async/await from the ground up
- **Smart Caching**: Memory-optimized caching with TTL and LRU eviction
- **Concurrency Control**: Configurable semaphores prevent resource exhaustion
- **Strategic Yielding**: Long operations yield control automatically
- **Batch Processing**: Optimized for processing large datasets
- **Memory Efficient**: Minimal memory footprint with cleanup strategies

## 📝 License

MIT License - see LICENSE file for details.

## 🤝 Contributing

Contributions welcome! Please see CONTRIBUTING.md for guidelines.

## 📚 Documentation

- Full API documentation available in docstrings
- Comprehensive examples in demo files
- Educational materials for classroom use
- Research applications and case studies

## 🔗 Links

- [GitHub Repository](https://github.com/yourusername/chuk-mcp-math)
- [Documentation](https://docs.example.com)
- [PyPI Package](https://pypi.org/project/chuk-mcp-math/)
- [Examples & Tutorials](https://examples.example.com)

---

**Built with ❤️ for the mathematical computing community**

*Async-native • MCP-optimized • Educational-ready • Research-grade*