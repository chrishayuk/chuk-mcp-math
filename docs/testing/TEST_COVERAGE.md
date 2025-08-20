# Test Coverage Guide

## Overview

Test coverage measures how much of your code is executed during testing. This guide covers coverage measurement, targets, and best practices for the CHUK MCP Math project.

## Coverage Tools

### Installation
```bash
# Install coverage tools using uv (preferred)
uv add --dev pytest-cov

# The tool is already included in pyproject.toml dev dependencies
```

### Running Coverage Reports

```bash
# Basic coverage report
uv run pytest --cov=src/chuk_mcp_math

# Detailed terminal report with missing lines
uv run pytest --cov=src/chuk_mcp_math --cov-report=term-missing

# Generate HTML coverage report
uv run pytest --cov=src/chuk_mcp_math --cov-report=html

# Coverage for specific module
uv run pytest --cov=src/chuk_mcp_math/arithmetic tests/math/arithmetic/

# Fail tests if coverage drops below threshold
uv run pytest --cov=src/chuk_mcp_math --cov-fail-under=80
```

## Coverage Targets

### Project Goals
- **Overall Coverage**: ≥ 80%
- **Core Modules**: ≥ 90%
- **New Code**: ≥ 95%
- **Critical Paths**: 100%

### Module-Specific Targets

| Module Category | Target Coverage | Priority |
|----------------|-----------------|----------|
| Core Operations (arithmetic) | 95% | Critical |
| Mathematical Functions | 90% | High |
| Utility Functions | 85% | Medium |
| Decorators/Infrastructure | 80% | Medium |
| Example/Demo Code | 50% | Low |

## Understanding Coverage Reports

### Terminal Output
```
Name                                              Stmts   Miss  Cover   Missing
--------------------------------------------------------------------------------
src/chuk_mcp_math/arithmetic/core/basic.py          43      0   100%
src/chuk_mcp_math/arithmetic/core/modular.py        40      0   100%
src/chuk_mcp_math/linear_algebra/vectors/norms.py   38      0   100%
src/chuk_mcp_math/trigonometry/basic.py             98     98     0%   19-495
--------------------------------------------------------------------------------
TOTAL                                              9663   4527    53%
```

- **Stmts**: Total number of statements
- **Miss**: Number of statements not executed
- **Cover**: Percentage of statements covered
- **Missing**: Line numbers not covered

### HTML Reports
```bash
# Generate HTML report
uv run pytest --cov=src/chuk_mcp_math --cov-report=html

# Open report (macOS)
open htmlcov/index.html

# Report location: htmlcov/index.html
```

HTML reports provide:
- Interactive line-by-line coverage visualization
- Sortable module list
- Coverage trends over time
- Branch coverage details

## Coverage Types

### Line Coverage
Basic metric showing which lines were executed:
```python
def calculate(x, y):
    result = x + y  # ✓ Covered
    if result > 100:
        return 100  # ✗ Not covered if result ≤ 100
    return result   # ✓ Covered
```

### Branch Coverage
Ensures all code paths are tested:
```python
def process(value):
    if value > 0:      # Need tests for both True and False
        return "positive"
    elif value < 0:    # Need tests for both True and False
        return "negative"
    else:
        return "zero"
```

### Statement Coverage vs Functional Coverage
```python
# High statement coverage but poor functional coverage
async def divide(a, b):
    # Test might cover the line but miss edge cases
    return a / b  # ✓ Line covered, but did we test b=0?
```

## Best Practices

### 1. Focus on Meaningful Coverage
```python
# Good: Test actual functionality
@pytest.mark.asyncio
async def test_norm_properties():
    """Test mathematical properties, not just lines."""
    vector = [3, 4]
    norm = await euclidean_norm(vector)
    assert math.isclose(norm, 5.0)  # Pythagorean triple
    
    # Test norm properties
    assert norm >= 0  # Non-negativity
    scaled = await euclidean_norm([6, 8])
    assert math.isclose(scaled, 2 * norm)  # Scaling property
```

### 2. Don't Chase 100% Coverage Blindly
```python
# Not worth testing
if __name__ == "__main__":
    # Demo code - low priority for coverage
    demo()

# Platform-specific code
if sys.platform == "win32":
    # Only test on relevant platform
    windows_specific_function()
```

### 3. Prioritize Critical Paths
```python
# High priority - core mathematical operations
async def vector_norm(vector, p=2):
    """Critical function - aim for 100% coverage."""
    # Every line and branch should be tested
    
# Lower priority - convenience wrapper
async def l2_norm(vector):
    """Simple wrapper - basic test sufficient."""
    return await vector_norm(vector, p=2)
```

### 4. Use Coverage to Find Gaps
```bash
# Identify untested modules
uv run pytest --cov=src/chuk_mcp_math --cov-report=term-missing | grep "0%"

# Find partially tested modules
uv run pytest --cov=src/chuk_mcp_math --cov-report=term-missing | grep -E "[0-9]{1,2}%"
```

## Improving Coverage

### Step-by-Step Approach

1. **Measure Baseline**
   ```bash
   uv run pytest --cov=src/chuk_mcp_math --cov-report=term > coverage_baseline.txt
   ```

2. **Identify Gaps**
   - Sort by coverage percentage
   - Focus on critical modules first
   - Look for easy wins (simple functions)

3. **Write Targeted Tests**
   ```python
   # Use coverage report to identify missing lines
   # Missing: lines 45-52 (error handling)
   @pytest.mark.asyncio
   async def test_error_conditions():
       """Target uncovered error paths."""
       with pytest.raises(ValueError):
           await function_that_needs_coverage(invalid_input)
   ```

4. **Verify Improvement**
   ```bash
   # Run coverage again and compare
   uv run pytest --cov=src/chuk_mcp_math --cov-report=term
   ```

## Coverage in CI/CD

### GitHub Actions
For GitHub Actions workflow configuration, see:
- **Template**: [github-actions-coverage.yaml](https://github.com/chrishayuk/vibe-coding-templates/blob/main/python/templates/cicd/workflows/github-actions-coverage.yaml)
- **Local Implementation**: [github-actions-coverage.yaml](../../templates/cicd/workflows/github-actions-coverage.yaml)

The workflow includes coverage reporting, Codecov integration, and artifact uploading.

### Pre-commit Hooks
For pre-commit hook configuration, see:
- **Template**: [pre-commit-coverage-hook.yaml](https://github.com/chrishayuk/vibe-coding-templates/blob/main/python/templates/cicd/hooks/pre-commit-coverage-hook.yaml)
- **Local Implementation**: [pre-commit-coverage-hook.yaml](../../templates/cicd/hooks/pre-commit-coverage-hook.yaml)

Quick setup:
```bash
# Install pre-commit
uv add --dev pre-commit

# Add hooks to .pre-commit-config.yaml from template

# Install hooks
pre-commit install

# Run coverage check
pre-commit run test-coverage --all-files
```

## Common Coverage Patterns

### Async Function Coverage
```python
@pytest.mark.asyncio
async def test_async_function():
    """Ensure async functions are properly covered."""
    result = await async_function()
    assert result is not None
    
    # Test async yielding for large inputs
    large_input = list(range(10000))
    result = await async_function(large_input)
    # Verify the function yielded control
```

### Error Path Coverage
```python
@pytest.mark.asyncio
async def test_error_paths():
    """Cover all error conditions."""
    # Invalid input type
    with pytest.raises(TypeError):
        await function("not a number")
    
    # Invalid input value
    with pytest.raises(ValueError, match="must be positive"):
        await function(-1)
    
    # Edge case errors
    with pytest.raises(ValueError, match="empty"):
        await function([])
```

### Branch Coverage
```python
@pytest.mark.parametrize("p,expected_type", [
    (1, "manhattan"),
    (2, "euclidean"),
    (float('inf'), "chebyshev"),
    (3, "general")
])
@pytest.mark.asyncio
async def test_all_branches(p, expected_type):
    """Ensure all conditional branches are covered."""
    result = await norm_function(vector, p=p)
    # Verify each branch behaves correctly
```

## Troubleshooting

### Coverage Not Detected
```bash
# Ensure test discovery is working
uv run pytest --collect-only

# Check source path is correct
uv run pytest --cov=src/chuk_mcp_math --cov-report=term

# Verify __init__.py files exist
find src -name "*.py" -type f | head
```

### Inconsistent Coverage
```bash
# Clear coverage cache
rm -rf .coverage .pytest_cache

# Run with fresh environment
uv run pytest --cov=src/chuk_mcp_math --no-cov-on-fail
```

### Missing Async Coverage
```python
# Ensure pytest-asyncio is installed
uv add --dev pytest-asyncio

# Use proper async test marking
@pytest.mark.asyncio  # Required for async tests
async def test_async():
    result = await async_function()
```

## Coverage Badges

Add coverage badges to README:
```markdown
![Coverage](https://img.shields.io/badge/coverage-53%25-yellow)
![Tests](https://img.shields.io/badge/tests-2419%20passed-green)
```

Or with dynamic coverage:
```markdown
[![codecov](https://codecov.io/gh/username/repo/branch/main/graph/badge.svg)](https://codecov.io/gh/username/repo)
```

## Related Documentation

- [Unit Testing](./UNIT_TESTING.md) - General unit testing practices
- [Testing Overview](./TESTING.md) - Complete testing guide
- [Math Patterns](./MATH_PATTERNS.md) - Mathematical testing patterns
- [Package Management](../PACKAGE_MANAGEMENT.md) - Using uv for dependencies

## Template Information

- **Source**: [vibe-coding-templates](https://github.com/chrishayuk/vibe-coding-templates/blob/main/python/docs/testing/TEST_COVERAGE.md)
- **Version**: 1.0.0
- **Date**: 2025-01-19
- **Author**: chrishayuk
- **Last Synced**: 2025-01-19