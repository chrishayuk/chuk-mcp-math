# Testing Patterns & Workflow

## 📍 Documentation Has Moved

The testing documentation has been reorganized into a comprehensive, modular structure for better clarity and maintainability.

## 📚 New Documentation Structure

All testing documentation is now located in **[`docs/testing/`](./docs/testing/)**:

### Main Index
- **[TESTING.md](./docs/testing/TESTING.md)** - Central hub for all testing documentation

### Testing Types
- **[UNIT_TESTING.md](./docs/testing/UNIT_TESTING.md)** - Unit testing patterns and isolation strategies
- **[INTEGRATION_TESTING.md](./docs/testing/INTEGRATION_TESTING.md)** - Module interaction and workflow testing
- **[PERFORMANCE_TESTING.md](./docs/testing/PERFORMANCE_TESTING.md)** - Benchmarking, profiling, and load testing

### Domain Patterns
- **[MATH_PATTERNS.md](./docs/testing/MATH_PATTERNS.md)** - Mathematical and numerical testing patterns
- **[ASYNC_PATTERNS.md](./docs/testing/ASYNC_PATTERNS.md)** - Async function testing strategies
- **[LIBRARY_PATTERNS.md](./docs/testing/LIBRARY_PATTERNS.md)** - Public API and library testing

### Templates
All test templates are in **[`docs/testing/templates/`](./docs/testing/templates/)**:
- `unit_test_template.py` - General unit test template
- `math_test_template.py` - Mathematical function test template
- Additional templates for specific test types

## 🚀 Quick Start

### For New Contributors
1. Start with [TESTING.md](./docs/testing/TESTING.md) for overview
2. Choose appropriate testing type documentation
3. Use relevant template from `docs/testing/templates/`

### For Testing Math Functions
1. Review [MATH_PATTERNS.md](./docs/testing/MATH_PATTERNS.md)
2. Use [`math_test_template.py`](./docs/testing/templates/math_test_template.py)
3. Follow numerical precision guidelines

### Common Commands
```bash
# Run all tests
make test

# Run specific test type
pytest tests/unit/ -v
pytest tests/integration/ -v  
pytest tests/performance/ -v

# Run with coverage
pytest --cov=chuk_mcp_math --cov-report=html

# Run by markers
pytest -m unit
pytest -m integration
pytest -m math
```

## 📝 Example Implementation

For a complete example implementing all patterns:
- [Geometric Vector Tests](./tests/math/linear_algebra/vectors/test_geometric.py)

---

*This file serves as a pointer to the comprehensive testing documentation. Please refer to the documents linked above for detailed patterns and workflows.*