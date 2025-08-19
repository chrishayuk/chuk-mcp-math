# Testing Documentation

## Overview

Comprehensive testing documentation for the CHUK MCP Math library, organized by testing type and domain patterns.

## 🔑 Key Testing Considerations

### Numerical Stability
When testing mathematical functions, always consider:
- **Overflow/Underflow**: Test with very large (1e100) and very small (1e-100) values
- **Mixed Scales**: Test inputs with vastly different magnitudes in the same operation
- **Precision Loss**: Verify operations maintain appropriate precision levels
- **Special Values**: Always test with 0, 1, -1, and edge cases specific to the domain

### Error Message Consistency
- Define error message constants in test files for reusability
- Match error messages using regex patterns for flexibility
- Ensure error messages are descriptive and actionable

### Performance Awareness
- Mark slow tests with `@pytest.mark.slow` 
- Test async yielding for operations on large datasets (>1000 elements)
- Verify concurrent operations don't introduce race conditions

## 📚 Documentation Structure

### Core Testing Types

#### [UNIT_TESTING.md](./UNIT_TESTING.md)
- Test isolation principles
- Function-level testing
- Mock strategies
- Test organization
- Coverage requirements

#### [INTEGRATION_TESTING.md](./INTEGRATION_TESTING.md)
- Module interaction testing
- End-to-end workflows
- System integration patterns
- API testing strategies

#### [PERFORMANCE_TESTING.md](./PERFORMANCE_TESTING.md)
- Benchmark testing
- Load testing patterns
- Memory profiling
- Performance regression detection
- Optimization validation

### Domain-Specific Patterns

#### [MATH_PATTERNS.md](./MATH_PATTERNS.md)
- Numerical precision testing
- Mathematical property verification
- Algorithm correctness
- Statistical validation
- Special value handling

#### [ASYNC_PATTERNS.md](./ASYNC_PATTERNS.md)
- Async function testing
- Concurrency testing
- Event loop management
- Timeout handling
- Race condition detection

### General Patterns

#### [LIBRARY_PATTERNS.md](./LIBRARY_PATTERNS.md)
- Public API testing
- Backward compatibility
- Error message testing
- Documentation testing
- Example validation

#### [TEST_FUNDAMENTALS.md](./TEST_FUNDAMENTALS.md)
- Test naming conventions
- Assertion strategies
- Fixture patterns
- Parametrization
- Test data management

## 📁 Template Organization

All test templates are located in [`templates/`](./templates/):

```
templates/
├── unit_test_template.py       # Basic unit test template
├── integration_test_template.py # Integration test template
├── performance_test_template.py # Performance test template
├── math_test_template.py        # Mathematical function test template
└── async_test_template.py       # Async function test template
```

## 🚀 Quick Start Guide

### 1. Choose Your Testing Type

- **Writing a new math function?** → Start with [MATH_PATTERNS.md](./MATH_PATTERNS.md) and use [`templates/math_test_template.py`](./templates/math_test_template.py)
- **Testing async code?** → See [ASYNC_PATTERNS.md](./ASYNC_PATTERNS.md) and use [`templates/async_test_template.py`](./templates/async_test_template.py)
- **Performance testing?** → Follow [PERFORMANCE_TESTING.md](./PERFORMANCE_TESTING.md)
- **General unit testing?** → Begin with [UNIT_TESTING.md](./UNIT_TESTING.md)

### 2. Apply the Patterns

1. Copy the appropriate template from `templates/`
2. Follow the patterns in the relevant documentation
3. Ensure coverage meets requirements
4. Run the test suite

### 3. Validate Your Tests

```bash
# Run all tests
make test

# Run specific test type
pytest tests/unit/ -v
pytest tests/integration/ -v
pytest tests/performance/ -v

# Check coverage
pytest --cov=chuk_mcp_math --cov-report=html

# Run with markers
pytest -m unit
pytest -m integration
pytest -m performance
pytest -m math
```

## 📊 Test Organization

```
tests/
├── unit/                    # Unit tests
│   ├── math/               # Mathematical unit tests
│   ├── utils/              # Utility unit tests
│   └── core/               # Core functionality tests
├── integration/            # Integration tests
│   ├── workflows/          # Workflow tests
│   └── api/               # API integration tests
├── performance/           # Performance tests
│   ├── benchmarks/        # Benchmark tests
│   └── load/              # Load tests
├── templates/             # Test templates
│   └── *.py              # Various templates
└── conftest.py           # Shared fixtures
```

## 🎯 Testing Philosophy

### Core Principles

1. **Isolation**: Each test should be independent
2. **Clarity**: Test names describe what is being tested
3. **Completeness**: Cover normal, edge, and error cases
4. **Maintainability**: Keep tests simple and documented
5. **Performance**: Tests should run quickly

### Coverage Goals

| Test Type | Minimum Coverage | Target |
|-----------|-----------------|---------|
| Unit Tests | 80% | 90% |
| Integration Tests | 70% | 80% |
| Critical Paths | 95% | 100% |
| Error Handling | 90% | 95% |

## 🔧 Common Commands

```bash
# Development
make test                    # Run all tests
make test-unit              # Run unit tests only
make test-integration       # Run integration tests
make test-performance       # Run performance tests

# Coverage
make test-cov               # Run with coverage report
pytest --cov-fail-under=80  # Fail if coverage < 80%

# Debugging
pytest -x                   # Stop on first failure
pytest --pdb               # Drop to debugger on failure
pytest -vv                 # Very verbose output

# Filtering
pytest -k "pattern"        # Run tests matching pattern
pytest -m "not slow"       # Skip slow tests
pytest --lf                # Run last failed tests
```

## 📋 Testing Checklist

Before committing:
- [ ] Unit tests pass
- [ ] Integration tests pass (if applicable)
- [ ] Performance tests pass (if applicable)
- [ ] Coverage meets requirements
- [ ] No linting errors
- [ ] Documentation updated

## 🔗 Related Documentation

- [Project Architecture](../../ARCHITECTURE.md)
- [Design Principles](../../PRINCIPLES.md)
- [Implementation Patterns](../../PATTERNS.md)
- [Development Roadmap](../../ROADMAP.md)

## 📝 Contributing

When adding new test patterns:
1. Update the appropriate pattern document
2. Add a template if needed
3. Update this index
4. Ensure examples are provided