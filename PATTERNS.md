# Implementation Patterns

## Function Implementation Patterns

### Basic Async Function Pattern

```python
from chuk_mcp_math.mcp_decorator import mcp_function
from typing import Union, List, Optional, Dict, Any
import asyncio

@mcp_function(
    description="Clear, concise description for AI models",
    namespace="module_name",
    cache_strategy="memory",
    cache_ttl_seconds=3600,
    estimated_cpu_usage="low"
)
async def function_name(
    param1: Union[int, float],
    param2: Optional[List[float]] = None
) -> float:
    """
    Brief description of what the function does.
    
    Formula:
        result = param1 * sum(param2) if param2 else param1
    
    Args:
        param1: Description of parameter 1
        param2: Optional list of values to process
        
    Returns:
        Computed result as float
        
    Raises:
        ValueError: If param1 is negative
        
    Examples:
        >>> await function_name(10, [1, 2, 3])
        60.0
    """
    # Input validation
    if param1 < 0:
        raise ValueError(f"param1 must be non-negative, got {param1}")
    
    # Handle optional parameters
    if param2 is None:
        return float(param1)
    
    # Strategic yielding for long operations
    if len(param2) > 1000:
        await asyncio.sleep(0)
    
    # Core computation
    result = param1 * sum(param2)
    
    return result
```

### Complex Return Pattern

```python
@mcp_function(
    description="Function returning structured results",
    namespace="analysis"
)
async def analyze_data(
    data: List[float]
) -> Dict[str, Any]:
    """
    Analyze data and return comprehensive results.
    
    Returns:
        Dictionary containing:
        - mean: Average value
        - std: Standard deviation
        - min: Minimum value
        - max: Maximum value
        - diagnostics: Additional information
    """
    if not data:
        raise ValueError("Data list cannot be empty")
    
    mean = sum(data) / len(data)
    variance = sum((x - mean) ** 2 for x in data) / len(data)
    
    return {
        "mean": mean,
        "std": variance ** 0.5,
        "min": min(data),
        "max": max(data),
        "diagnostics": {
            "n_samples": len(data),
            "variance": variance
        }
    }
```

### Streaming Pattern

```python
from typing import AsyncIterator

@mcp_function(
    description="Generate values in a stream",
    supports_streaming=True,
    streaming_mode="chunked"
)
async def generate_sequence(
    start: int,
    end: int,
    step: int = 1
) -> AsyncIterator[int]:
    """
    Generate sequence of numbers as a stream.
    
    Yields:
        Numbers from start to end by step
    """
    current = start
    while current < end:
        yield current
        current += step
        
        # Yield control periodically
        if current % 100 == 0:
            await asyncio.sleep(0)
```

### Batch Processing Pattern

```python
@mcp_function(
    description="Process data in batches",
    max_concurrent_executions=5
)
async def batch_process(
    items: List[Any],
    batch_size: int = 100
) -> List[Any]:
    """
    Process large lists in batches for efficiency.
    """
    results = []
    
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        
        # Process batch concurrently
        batch_results = await asyncio.gather(
            *[process_item(item) for item in batch]
        )
        results.extend(batch_results)
        
        # Yield control between batches
        await asyncio.sleep(0)
    
    return results

async def process_item(item: Any) -> Any:
    """Process individual item."""
    return item * 2  # Example processing
```

## Error Handling Patterns

### Validation with Helpful Messages

```python
async def validate_probability(p: float) -> float:
    """Validate probability with helpful error messages."""
    if not isinstance(p, (int, float)):
        raise TypeError(
            f"Probability must be a number, got {type(p).__name__}"
        )
    
    if not 0 <= p <= 1:
        if 0 <= p <= 100:
            raise ValueError(
                f"Probability must be between 0 and 1, got {p}. "
                f"Did you mean {p/100}?"
            )
        else:
            raise ValueError(
                f"Probability must be between 0 and 1, got {p}"
            )
    
    return float(p)
```

### Graceful Degradation

```python
async def robust_calculation(
    x: float,
    method: str = "auto"
) -> float:
    """Calculate with fallback methods."""
    try:
        if method == "auto":
            # Try fast method first
            return await fast_calculation(x)
    except (OverflowError, ValueError):
        # Fall back to slower but more stable method
        return await stable_calculation(x)
```

## Caching Patterns

### Conditional Caching

```python
@mcp_function(
    description="Expensive computation with smart caching",
    cache_strategy="memory",
    cache_ttl_seconds=3600
)
async def expensive_function(
    n: int,
    use_cache: bool = True
) -> int:
    """
    Expensive computation with optional caching.
    
    Args:
        n: Input value
        use_cache: Whether to use cached results
    """
    if not use_cache:
        # Bypass cache for this call
        return await _compute_expensive(n)
    
    # Normal cached execution
    return await _compute_expensive_cached(n)

@lru_cache(maxsize=128)
async def _compute_expensive_cached(n: int) -> int:
    return await _compute_expensive(n)

async def _compute_expensive(n: int) -> int:
    await asyncio.sleep(1)  # Simulate expensive operation
    return n ** 2
```

### Cache Warming

```python
async def warm_cache(values: List[int]) -> None:
    """Pre-compute and cache common values."""
    tasks = [expensive_function(v) for v in values]
    await asyncio.gather(*tasks)
```

## Testing Patterns

> **Note**: Comprehensive testing documentation is now available in [`docs/testing/`](./docs/testing/):
> - [Testing Overview](./docs/testing/TESTING.md) - Complete guide
> - [Unit Testing](./docs/testing/UNIT_TESTING.md) - Isolation patterns
> - [Math Testing](./docs/testing/MATH_PATTERNS.md) - Numerical precision
> - [Test Templates](./docs/testing/templates/) - Ready-to-use templates

### Basic Test Pattern

```python
import pytest
import asyncio
from math import isclose

@pytest.mark.asyncio
async def test_function_basic():
    """Test basic functionality."""
    result = await function_name(10, [1, 2, 3])
    assert isclose(result, 60.0, rel_tol=1e-9)

@pytest.mark.asyncio
async def test_function_edge_cases():
    """Test edge cases."""
    # Empty input
    result = await function_name(5)
    assert result == 5.0
    
    # Zero value
    result = await function_name(0, [1, 2, 3])
    assert result == 0.0
    
    # Large input
    large_list = list(range(10000))
    result = await function_name(1, large_list)
    assert result == sum(large_list)

@pytest.mark.asyncio
async def test_function_errors():
    """Test error conditions."""
    with pytest.raises(ValueError, match="non-negative"):
        await function_name(-1)
    
    with pytest.raises(TypeError):
        await function_name("invalid")
```

### Property-Based Testing

```python
from hypothesis import given, strategies as st

@pytest.mark.asyncio
@given(
    n=st.integers(min_value=0, max_value=1000),
    values=st.lists(
        st.floats(min_value=-1000, max_value=1000, allow_nan=False),
        min_size=0,
        max_size=100
    )
)
async def test_function_properties(n, values):
    """Test mathematical properties."""
    result = await function_name(n, values)
    
    # Test commutative property
    result2 = await function_name(n, list(reversed(values)))
    assert isclose(result, result2, rel_tol=1e-9)
    
    # Test distributive property
    if values:
        expected = n * sum(values)
        assert isclose(result, expected, rel_tol=1e-9)
```

### Performance Testing

```python
import time

@pytest.mark.asyncio
@pytest.mark.performance
async def test_function_performance():
    """Test performance characteristics."""
    # Test caching effectiveness
    start = time.time()
    result1 = await expensive_function(100)
    first_call = time.time() - start
    
    start = time.time()
    result2 = await expensive_function(100)
    cached_call = time.time() - start
    
    assert result1 == result2
    assert cached_call < first_call * 0.1  # 10x speedup expected
    
    # Test concurrent execution
    start = time.time()
    results = await asyncio.gather(
        *[expensive_function(i) for i in range(10)]
    )
    concurrent_time = time.time() - start
    
    assert len(results) == 10
    assert concurrent_time < first_call * 10  # Should be faster than serial
```

## Module Organization Patterns

### Module Structure

```python
# module/__init__.py
"""
Module description and overview.
"""

from .core import (
    basic_function,
    another_function
)
from .advanced import (
    complex_function,
    specialized_function  
)
from .utils import (
    helper_function
)

__all__ = [
    # Core functions
    "basic_function",
    "another_function",
    # Advanced functions
    "complex_function", 
    "specialized_function",
    # Utilities
    "helper_function"
]

# Optional: Module-level initialization
async def _initialize_module():
    """Initialize module resources."""
    pass
```

### Submodule Pattern

```python
# module/submodule/operations.py
"""
Specific operations within a mathematical domain.
"""

from typing import List, Dict, Any
from ...mcp_decorator import mcp_function

@mcp_function(
    namespace="module.submodule",
    description="Submodule operation"
)
async def submodule_function(x: float) -> float:
    """Perform submodule-specific operation."""
    return x * 2
```

## Performance Optimization Patterns

### Memoization Pattern

```python
from functools import lru_cache

@mcp_function(description="Function with internal memoization")
async def fibonacci(n: int) -> int:
    """Calculate Fibonacci number with memoization."""
    
    @lru_cache(maxsize=None)
    def _fib(n: int) -> int:
        if n <= 1:
            return n
        return _fib(n - 1) + _fib(n - 2)
    
    # Yield for large n
    if n > 30:
        await asyncio.sleep(0)
    
    return _fib(n)
```

### Vectorization Pattern

```python
@mcp_function(description="Vectorized operation")
async def vectorized_operation(
    data: List[float],
    operation: str = "square"
) -> List[float]:
    """Apply operation to all elements efficiently."""
    
    if operation == "square":
        # Use list comprehension for efficiency
        result = [x ** 2 for x in data]
    elif operation == "sqrt":
        import math
        result = [math.sqrt(abs(x)) for x in data]
    else:
        raise ValueError(f"Unknown operation: {operation}")
    
    # Yield for large datasets
    if len(data) > 10000:
        await asyncio.sleep(0)
    
    return result
```

### Concurrent Execution Pattern

```python
async def parallel_computation(
    inputs: List[Any],
    max_concurrent: int = 10
) -> List[Any]:
    """Process inputs with controlled concurrency."""
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_with_limit(item):
        async with semaphore:
            return await process_item(item)
    
    results = await asyncio.gather(
        *[process_with_limit(item) for item in inputs]
    )
    
    return results
```

## Documentation Patterns

### Comprehensive Docstring

```python
async def well_documented_function(
    x: float,
    y: float,
    method: str = "euclidean"
) -> float:
    """
    Calculate distance between two points.
    
    This function computes the distance between two points using
    the specified method. The default Euclidean distance is the
    straight-line distance between points.
    
    Mathematical Formula:
        Euclidean: d = √((x₂-x₁)² + (y₂-y₁)²)
        Manhattan: d = |x₂-x₁| + |y₂-y₁|
    
    Args:
        x: X-coordinate of the point
        y: Y-coordinate of the point
        method: Distance calculation method
            - "euclidean": Straight-line distance (default)
            - "manhattan": Grid-based distance
            - "chebyshev": Maximum coordinate difference
    
    Returns:
        float: The calculated distance
    
    Raises:
        ValueError: If method is not recognized
        TypeError: If coordinates are not numeric
    
    Examples:
        Calculate Euclidean distance:
        >>> await well_documented_function(0, 0, "euclidean")
        0.0
        
        Calculate Manhattan distance:
        >>> await well_documented_function(3, 4, "manhattan")  
        7.0
    
    Notes:
        - All methods return non-negative values
        - NaN inputs will propagate to the output
        - Infinity is handled correctly
    
    References:
        - Euclidean distance: https://en.wikipedia.org/wiki/Euclidean_distance
        - Manhattan distance: https://en.wikipedia.org/wiki/Taxicab_geometry
    
    See Also:
        - `calculate_3d_distance`: For 3D point distances
        - `geodesic_distance`: For distances on curved surfaces
    """
    pass
```

## Anti-Patterns to Avoid

### ❌ Synchronous Wrapper

```python
# BAD - Don't do this
def sync_wrapper(x: float) -> float:
    return asyncio.run(async_function(x))
```

### ❌ Missing Error Handling

```python
# BAD - No validation
async def bad_function(x: float) -> float:
    return 1 / x  # Will crash on x=0
```

### ❌ Inconsistent Returns

```python
# BAD - Sometimes returns float, sometimes int
async def inconsistent(x: float) -> float:
    if x > 0:
        return x
    return 0  # Should be 0.0
```

### ❌ Side Effects

```python
# BAD - Modifies global state
_global_cache = {}

async def impure_function(x: float) -> float:
    _global_cache[x] = x * 2  # Side effect
    return x * 2
```

### ❌ Poor Naming

```python
# BAD - Unclear names
async def calc(x, y, z):
    return x * y + z
```