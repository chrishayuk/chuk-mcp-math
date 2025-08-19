# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CHUK MCP Math is an async-native mathematical functions library with 400+ functions organized across specialized domains. It uses MCP (Model Context Protocol) decorators for AI model integration with smart caching and performance optimization.

## Documentation Structure

### Core Documentation
- **[ARCHITECTURE.md](./ARCHITECTURE.md)** - Module structure and organization
- **[PRINCIPLES.md](./PRINCIPLES.md)** - Core design principles and philosophy
- **[PATTERNS.md](./PATTERNS.md)** - Implementation patterns and code examples
- **[ROADMAP.md](./ROADMAP.md)** - Development timeline and expansion plans

### Testing Documentation
- **[docs/testing/TESTING.md](./docs/testing/TESTING.md)** - Testing documentation hub
- **[docs/testing/UNIT_TESTING.md](./docs/testing/UNIT_TESTING.md)** - Unit testing patterns
- **[docs/testing/MATH_PATTERNS.md](./docs/testing/MATH_PATTERNS.md)** - Mathematical testing patterns
- **[docs/testing/templates/](./docs/testing/templates/)** - Test templates

## Quick Reference

### Current Modules
- **Number Theory** (`number_theory/`) - 340+ functions, 18 modules
- **Trigonometry** (`trigonometry/`) - 120+ functions, 8 modules  
- **Arithmetic** (`arithmetic/`) - 30+ functions, core and comparison operations
- **Statistics** (`statistics.py`) - Basic descriptive statistics
- **Geometry** (`geometry.py`) - 2D shapes and calculations

### Key Patterns
All functions follow these patterns:
- Async-native: `async def` with `await` for all operations
- MCP decorated: `@mcp_function()` with caching and metrics
- Type-safe: Full type hints with Pydantic validation
- Well-documented: Mathematical formulas in docstrings

## Package Management

CHUK MCP Math uses **`uv`** for all package management. See [Package Management Guide](./docs/PACKAGE_MANAGEMENT.md) for detailed usage.

**Important**: Always use `uv` instead of `pip`:
```bash
# Install dependencies
uv sync --dev              # Install all dependencies including dev

# Add new packages
uv add package-name        # Add production dependency
uv add --dev pytest-cov    # Add development dependency

# Run commands
uv run pytest              # Run tests
uv run python script.py    # Run Python scripts
```

## Common Development Commands

```bash
# Run tests (all commands use uv internally)
make test                  # Runs pytest with uv
make test-cov             # Run tests with coverage report

# Code quality
make lint                 # Run ruff linter
make format              # Auto-format code with ruff
make typecheck           # Run mypy type checking
make check               # Run all checks (lint, typecheck, test)

# Build and publish
make build               # Build distribution packages
make publish             # Publish to PyPI
make publish-test        # Publish to test PyPI

# Cleanup
make clean               # Basic cleanup (pyc, build artifacts)
make clean-all           # Deep clean everything

# Installation
make install             # Install package with uv
make dev-install         # Install in development mode with uv

# Run specific module demos
uv run python -m chuk_mcp_math.number_theory    # Number theory demo
uv run python -m chuk_mcp_math.trigonometry     # Trigonometry demo
```

## Testing Strategy

Comprehensive testing documentation is available in [`docs/testing/`](./docs/testing/):
- **[Testing Overview](./docs/testing/TESTING.md)** - Complete testing guide
- **[Unit Testing](./docs/testing/UNIT_TESTING.md)** - Isolation and mocking patterns
- **[Math Testing](./docs/testing/MATH_PATTERNS.md)** - Numerical precision and mathematical properties
- **[Performance Testing](./docs/testing/PERFORMANCE_TESTING.md)** - Benchmarking and profiling

Quick facts:
- Tests located in `tests/` mirroring source structure
- Templates in `docs/testing/templates/`
- Markers: `asyncio`, `unit`, `integration`, `performance`, `math`, `property`, `slow`
- Run specific tests: `pytest tests/math/number_theory/test_primes.py`

## Development Workflow

When adding new mathematical functions:
1. Review [PRINCIPLES.md](./PRINCIPLES.md) for design philosophy
2. Follow patterns in [PATTERNS.md](./PATTERNS.md) for implementation
3. Place in appropriate module per [ARCHITECTURE.md](./ARCHITECTURE.md)
4. Use `@mcp_function()` decorator with appropriate settings
5. Write tests following [Testing Patterns](./docs/testing/TESTING.md):
   - Use templates from `docs/testing/templates/`
   - Follow [Math Testing Patterns](./docs/testing/MATH_PATTERNS.md) for numerical functions
   - Ensure proper [Unit Testing](./docs/testing/UNIT_TESTING.md) isolation
6. Run `make check` before committing

## Important Notes

- **Never use synchronous wrappers** - Keep everything async-native
- **Always validate inputs** - Use descriptive error messages
- **Include mathematical formulas** - Document the math in docstrings
- **Test edge cases** - Include property-based tests with Hypothesis
- **Check performance** - Use strategic yielding for long operations