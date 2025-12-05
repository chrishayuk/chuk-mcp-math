# CHUK MCP Math - Examples

This directory contains comprehensive examples and demonstrations of the CHUK MCP Math library.

## 📁 Directory Structure

```
examples/
├── demos/              # Quick demos and comprehensive tests
│   ├── DEMO.py                              # Main library demo (32 functions)
│   ├── comprehensive_demo_01_arithmetic.py  # Arithmetic demo (44 functions)
│   ├── quick_comprehensive_test.py          # Quick test (all 572 functions)
│   └── truly_comprehensive_test.py          # Complete test (533/533 functions)
│
└── applications/       # Real-world application examples
    ├── demo_number_theory.py                # Number theory (340+ functions)
    └── demo_trigonometry.py                 # Trigonometry (120+ functions)
```

## 🚀 Running Examples

### Run All Tests

```bash
# From project root
./RUN_ALL_DEMOS.sh        # Run all demo scripts
./RUN_ALL_EXAMPLES.sh     # Run all application examples
```

### Run Individual Demos

```bash
# Quick demos (no dependencies)
python3 examples/demos/DEMO.py
python3 examples/demos/comprehensive_demo_01_arithmetic.py
python3 examples/demos/quick_comprehensive_test.py
python3 examples/demos/truly_comprehensive_test.py
```

### Run Application Examples

```bash
# Comprehensive examples (require uv)
uv run python examples/applications/demo_number_theory.py
uv run python examples/applications/demo_trigonometry.py
```

## 📊 Demos Overview

### Quick Demos (`demos/`)

#### 1. DEMO.py - Main Library Demonstration
- **Functions Tested**: 32 core functions
- **Purpose**: Quick introduction to library capabilities
- **Topics**: Basic arithmetic, number theory, trigonometry, statistics
- **Run Time**: ~1 second

```bash
python3 examples/demos/DEMO.py
```

#### 2. comprehensive_demo_01_arithmetic.py - Arithmetic Operations
- **Functions Tested**: 44 arithmetic functions
- **Purpose**: Complete arithmetic operations showcase
- **Topics**: Basic ops, rounding, modular, comparison, tolerance
- **Run Time**: ~1 second

```bash
python3 examples/demos/comprehensive_demo_01_arithmetic.py
```

#### 3. quick_comprehensive_test.py - Quick Verification
- **Functions Tested**: 572 functions (sampled)
- **Purpose**: Fast verification that all functions work
- **Coverage**: Samples from all 25+ domains
- **Run Time**: ~2-3 seconds

```bash
python3 examples/demos/quick_comprehensive_test.py
```

#### 4. truly_comprehensive_test.py - Complete Testing
- **Functions Tested**: 533/533 functions individually
- **Purpose**: Exhaustive verification with real arguments
- **Coverage**: 100% of testable functions
- **Run Time**: ~5-10 seconds
- **Status**: ✅ 533/533 passing, 0 failures, 0 skipped

```bash
python3 examples/demos/truly_comprehensive_test.py
```

## 🎨 Application Examples (`applications/`)

### 1. demo_number_theory.py - Number Theory Applications

**Comprehensive demonstration of 340+ number theory functions across 16 sections:**

#### Sections Covered:
1. **Prime Numbers & Applications**
   - Basic primality testing
   - Prime factorization
   - Mersenne primes
   - Twin, cousin, and sexy primes
   - Prime triplets and patterns

2. **Cryptographic Applications**
   - RSA key generation
   - Chinese Remainder Theorem
   - Quadratic residues
   - Legendre symbols
   - Discrete logarithms
   - Primitive roots

3. **Diophantine Equations**
   - Linear Diophantine equations
   - Pell's equation
   - Pythagorean triples
   - Frobenius numbers
   - Postage stamp problem

4. **Special Number Categories**
   - Amicable pairs
   - Vampire numbers
   - Kaprekar numbers
   - Armstrong numbers
   - Keith numbers
   - Taxi numbers (Hardy-Ramanujan)

5. **Continued Fractions**
   - CF expansions of π, e, φ, √2
   - Convergent calculations
   - Rational approximations
   - Periodic continued fractions
   - Calendar approximations

6. **Farey Sequences**
   - Farey sequence generation
   - Mediant operations
   - Stern-Brocot tree
   - Ford circles
   - Circle tangency
   - Density analysis

7. **Mathematical Sequences**
   - Fibonacci, Lucas, Catalan
   - Bell numbers
   - Pell numbers
   - Tribonacci
   - Golden ratio approximations

8. **Figurate Numbers**
   - Polygonal numbers (triangular, square, pentagonal, etc.)
   - Centered polygonal numbers
   - 3D figurate numbers
   - Pronic and star numbers

9. **Advanced Prime Analysis**
   - Prime counting functions
   - Prime gap analysis
   - Bertrand's postulate
   - Prime distribution

10. **Number Properties**
    - Perfect, abundant, deficient
    - Palindromic numbers
    - Harshad numbers
    - Digital properties

11. **Iterative Sequences**
    - Collatz conjecture
    - Happy numbers
    - Narcissistic numbers

12. **Advanced Arithmetic Functions**
    - Euler's totient function
    - Möbius function
    - Divisor functions

13. **Mathematical Constants**
    - High-precision π (Leibniz, Machin)
    - Euler's number e
    - Golden ratio

14. **Partitions & Additive Theory**
    - Integer partitions
    - Goldbach conjecture
    - Sum of squares

15. **Cross-Module Relationships**
    - Perfect ↔ Mersenne primes
    - Continued fractions ↔ Pell equations
    - Farey sequences ↔ Continued fractions
    - Figurate numbers ↔ Diophantine equations

16. **Performance & Scale**
    - Large number computations
    - Batch operations
    - Performance benchmarks

**Run Time**: ~10-15 seconds

```bash
uv run python examples/applications/demo_number_theory.py
```

### 2. demo_trigonometry.py - Trigonometry Applications

**Comprehensive demonstration of 120+ trigonometry functions across 10 sections:**

#### Sections Covered:
1. **Basic Trigonometric Functions**
   - sin, cos, tan at key angles
   - Reciprocal functions (csc, sec, cot)
   - Degree variants

2. **Inverse Trigonometric Functions**
   - asin, acos, atan
   - atan2 with full quadrant coverage
   - Degree conversions

3. **Hyperbolic Functions**
   - sinh, cosh, tanh
   - Identity verification (cosh² - sinh² = 1)
   - Catenary curve applications

4. **Angle Conversions**
   - Degrees, radians, gradians
   - Angle normalization
   - Shortest angular distances

5. **Mathematical Identities**
   - Pythagorean identities
   - Sum and difference formulas
   - Double angle formulas

6. **Wave Analysis**
   - Amplitude and phase extraction
   - Beat frequency analysis
   - Harmonic analysis
   - Phase relationships

7. **Navigation Applications**
   - Great circle distances (Haversine)
   - Bearing calculations
   - GPS triangulation
   - Real city-to-city examples

8. **Physics Simulations**
   - Pendulum motion
   - Spring-mass systems
   - Oscillation analysis
   - Damping effects

9. **Educational Examples**
   - Unit circle exploration
   - Real-world problem solving
   - Building height calculations
   - Ship navigation

10. **Performance & Precision**
    - High-precision calculations
    - Numerical error analysis
    - Performance benchmarks

**Run Time**: ~5-8 seconds

```bash
uv run python examples/applications/demo_trigonometry.py
```

## 📈 Test Coverage

### Demo Scripts
- ✅ **DEMO.py**: 32 functions tested
- ✅ **comprehensive_demo_01_arithmetic.py**: 44 functions tested
- ✅ **quick_comprehensive_test.py**: 572 functions sampled
- ✅ **truly_comprehensive_test.py**: 533/533 functions individually tested

### Application Examples
- ✅ **demo_number_theory.py**: 340+ functions demonstrated
- ✅ **demo_trigonometry.py**: 120+ functions demonstrated

### Overall Statistics
```
Total Functions: 572
Individually Tested: 533/533 (100%)
Application Demos: 460+ functions
Test Failures: 0
Test Errors: 0
```

## 🎯 Use Cases

### For Learning
- **Beginners**: Start with `DEMO.py` for quick overview
- **Students**: Use application examples for educational demonstrations
- **Researchers**: Study cross-module relationships in number theory demo

### For Testing
- **Quick Check**: Run `quick_comprehensive_test.py` (~3 seconds)
- **Full Verification**: Run `truly_comprehensive_test.py` (~10 seconds)
- **Automated CI/CD**: Use `RUN_ALL_DEMOS.sh` for complete testing

### For Development
- **API Examples**: See application demos for usage patterns
- **Performance**: Study benchmarks in performance sections
- **Best Practices**: Learn from comprehensive examples

## 🔧 Requirements

### Demos
- Python 3.8+
- No external dependencies (uses library only)

### Application Examples
- Python 3.8+
- `uv` package manager (for running with project dependencies)

## 📚 Additional Resources

- **Main README**: [../README.md](../README.md)
- **Testing Guide**: [../docs/testing/TESTING.md](../docs/testing/TESTING.md)
- **Architecture**: [../ARCHITECTURE.md](../ARCHITECTURE.md)
- **API Documentation**: See function docstrings in source code

## ✨ Key Features Demonstrated

All examples showcase:
- ✅ **100% Async Native**: All functions use async/await
- ✅ **Type Safety**: Complete type hints throughout
- ✅ **Error Handling**: Comprehensive error messages
- ✅ **Real-World Applications**: Practical use cases
- ✅ **Educational Value**: Clear explanations and examples
- ✅ **Performance**: Optimized implementations

---

**Run all examples**: `./RUN_ALL_DEMOS.sh && ./RUN_ALL_EXAMPLES.sh`
