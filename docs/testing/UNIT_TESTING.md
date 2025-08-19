# Unit Testing Patterns

## Overview

Unit testing focuses on testing individual functions and methods in isolation. This document covers principles and patterns for effective unit testing.

## Core Principles

### Test Isolation
- Each unit test should be completely independent
- No shared state between tests
- Mock external dependencies
- Test one thing at a time

### Test Structure (AAA Pattern)
```python
@pytest.mark.asyncio
async def test_function_behavior():
    """Test specific behavior of function."""
    # Arrange - Set up test data and conditions
    input_data = prepare_test_data()
    expected_result = calculate_expected()
    
    # Act - Execute the function under test
    actual_result = await function_under_test(input_data)
    
    # Assert - Verify the result
    assert actual_result == expected_result
```

## Unit Test Organization

### File Structure
```
tests/unit/
├── test_<module_name>.py     # Mirror source structure
├── conftest.py               # Shared fixtures for unit tests
└── <module>/
    ├── test_<function>.py    # One file per complex function
    └── test_<feature>.py     # Group related simple functions
```

### Test Class Organization
```python
class TestFunctionName:
    """Unit tests for function_name."""
    
    def test_normal_operation(self):
        """Test expected behavior with valid input."""
        pass
    
    def test_edge_cases(self):
        """Test boundary conditions."""
        pass
    
    def test_error_conditions(self):
        """Test error handling."""
        pass
    
    def test_type_validation(self):
        """Test input type handling."""
        pass
```

## Mocking Strategies

### Basic Mocking
```python
from unittest.mock import Mock, patch

def test_with_mock_dependency():
    """Test function with mocked dependency."""
    # Create mock
    mock_dep = Mock()
    mock_dep.get_value.return_value = 42
    
    # Inject mock
    result = function_under_test(dependency=mock_dep)
    
    # Verify interaction
    mock_dep.get_value.assert_called_once()
    assert result == 42
```

### Patching Dependencies
```python
@patch('module.external_function')
def test_with_patched_function(mock_func):
    """Test with patched external function."""
    mock_func.return_value = "mocked_result"
    
    result = function_under_test()
    
    assert result == "processed_mocked_result"
    mock_func.assert_called_with(expected_args)
```

### Async Mocking
```python
from unittest.mock import AsyncMock

@pytest.mark.asyncio
async def test_async_dependency():
    """Test with async mock."""
    mock_service = AsyncMock()
    mock_service.fetch_data.return_value = {"key": "value"}
    
    result = await async_function(service=mock_service)
    
    mock_service.fetch_data.assert_awaited_once()
    assert result["key"] == "value"
```

## Testing Patterns

### Testing Pure Functions
```python
def test_pure_function():
    """Test deterministic function."""
    # Pure functions always return same output for same input
    assert add(2, 3) == 5
    assert add(2, 3) == 5  # Same result
    
    # Test properties
    assert add(0, x) == x  # Identity
    assert add(x, y) == add(y, x)  # Commutative
```

### Testing Stateful Functions
```python
class TestStatefulClass:
    """Test class with internal state."""
    
    def setup_method(self):
        """Reset state before each test."""
        self.obj = StatefulClass()
    
    def test_state_initialization(self):
        """Test initial state."""
        assert self.obj.count == 0
    
    def test_state_modification(self):
        """Test state changes."""
        self.obj.increment()
        assert self.obj.count == 1
        
        self.obj.increment()
        assert self.obj.count == 2
```

### Testing Error Conditions
```python
def test_input_validation():
    """Test that invalid input raises appropriate errors."""
    with pytest.raises(ValueError, match="must be positive"):
        sqrt(-1)
    
    with pytest.raises(TypeError, match="expected number"):
        sqrt("not a number")
```

### Testing Side Effects
```python
def test_function_with_side_effects(tmp_path):
    """Test function that writes to filesystem."""
    output_file = tmp_path / "output.txt"
    
    write_result(data="test", filepath=output_file)
    
    assert output_file.exists()
    assert output_file.read_text() == "test"
```

## Parametrized Testing

### Basic Parametrization
```python
@pytest.mark.parametrize("input,expected", [
    (0, 0),
    (1, 1),
    (4, 2),
    (9, 3),
    (16, 4),
])
def test_sqrt_values(input, expected):
    """Test sqrt with multiple values."""
    assert sqrt(input) == expected
```

### Complex Parametrization
```python
@pytest.mark.parametrize("a,b,operation,expected", [
    (5, 3, "add", 8),
    (5, 3, "subtract", 2),
    (5, 3, "multiply", 15),
    (6, 3, "divide", 2),
])
def test_calculator(a, b, operation, expected):
    """Test calculator operations."""
    result = calculate(a, b, operation)
    assert result == expected
```

## Fixtures

### Basic Fixtures
```python
@pytest.fixture
def sample_data():
    """Provide sample data for tests."""
    return {
        "id": 1,
        "name": "test",
        "value": 42
    }

def test_process_data(sample_data):
    """Test using fixture."""
    result = process(sample_data)
    assert result["id"] == 1
```

### Fixture Scopes
```python
@pytest.fixture(scope="function")  # Default - per test
def db_connection():
    conn = create_connection()
    yield conn
    conn.close()

@pytest.fixture(scope="class")  # Per test class
def shared_resource():
    return expensive_setup()

@pytest.fixture(scope="module")  # Per test module
def module_config():
    return load_config()
```

## Coverage Guidelines

### What to Test
- All public functions/methods
- Complex private methods
- Error handling paths
- Edge cases and boundaries
- Different input types
- State transitions

### What Not to Test
- Simple getters/setters
- Framework code
- Third-party libraries
- Trivial functions (unless critical)
- Generated code

### Coverage Metrics

For comprehensive coverage guidance, see [Test Coverage Guide](./TEST_COVERAGE.md).

```bash
# Check coverage (using uv)
uv run pytest tests/unit/ --cov=module --cov-report=term-missing

# Enforce minimum coverage
uv run pytest tests/unit/ --cov=module --cov-fail-under=80

# Generate HTML report
uv run pytest tests/unit/ --cov=module --cov-report=html
```

Target coverage levels:
- Overall: ≥ 80%
- Core modules: ≥ 90%
- New code: ≥ 95%

## Best Practices

### DO's
✅ Keep tests simple and focused  
✅ Use descriptive test names  
✅ Test behavior, not implementation  
✅ Use fixtures for common setup  
✅ Mock external dependencies  
✅ Test edge cases  
✅ Maintain test isolation  
✅ Write tests first (TDD)  

### DON'Ts
❌ Don't test multiple behaviors in one test  
❌ Don't use production data  
❌ Don't make tests dependent on order  
❌ Don't test private methods directly  
❌ Don't ignore test failures  
❌ Don't use hard-coded delays  
❌ Don't over-mock  
❌ Don't write brittle tests  

## Example: Complete Unit Test

```python
"""Unit tests for calculator.add function."""

import pytest
from unittest.mock import patch
from calculator import add

class TestAdd:
    """Test cases for add function."""
    
    @pytest.mark.parametrize("a,b,expected", [
        (2, 3, 5),
        (0, 0, 0),
        (-1, 1, 0),
        (1.5, 2.5, 4.0),
    ])
    def test_add_various_inputs(self, a, b, expected):
        """Test add with various numeric inputs."""
        assert add(a, b) == expected
    
    def test_add_type_error(self):
        """Test that non-numeric input raises TypeError."""
        with pytest.raises(TypeError):
            add("2", 3)
    
    @patch('calculator.logger')
    def test_add_logging(self, mock_logger):
        """Test that operations are logged."""
        result = add(2, 3)
        
        assert result == 5
        mock_logger.info.assert_called_once()
```

## Related Documentation
- [Test Coverage Guide](./TEST_COVERAGE.md)
- [Integration Testing](./INTEGRATION_TESTING.md)
- [Performance Testing](./PERFORMANCE_TESTING.md)
- [Test Fundamentals](./TEST_FUNDAMENTALS.md)
- [Testing Index](./TESTING.md)