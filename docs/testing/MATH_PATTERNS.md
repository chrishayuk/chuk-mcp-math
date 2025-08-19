# Mathematical Testing Patterns

## Overview

Specialized testing patterns for mathematical and numerical computing functions, focusing on precision, correctness, and mathematical properties.

## Numerical Precision

### Floating-Point Comparisons
```python
import math
import numpy as np

# Absolute tolerance for small values
assert abs(result - expected) < 1e-10

# Relative tolerance for large values  
assert abs((result - expected) / expected) < 1e-9

# Combined tolerance
assert math.isclose(result, expected, rel_tol=1e-9, abs_tol=1e-12)

# Array comparisons
np.testing.assert_allclose(actual_array, expected_array, rtol=1e-7, atol=1e-10)
```

### Numerical Stability Testing
```python
@pytest.mark.asyncio
async def test_numerical_stability():
    """Test stability with extreme values."""
    # Test with very large numbers
    large_result = await compute(1e308)
    assert not math.isinf(large_result)
    assert not math.isnan(large_result)
    
    # Test with very small numbers
    small_result = await compute(1e-308)
    assert small_result != 0  # No underflow
    
    # Test catastrophic cancellation
    a, b = 1.0000001, 1.0000000
    result = await stable_subtract(a, b)
    assert abs(result - 1e-7) < 1e-12

@pytest.mark.asyncio
async def test_mixed_scale_inputs():
    """Test with inputs of vastly different magnitudes."""
    # When combining very large and very small values
    inputs = [1e-100, 1.0, 1e100]
    result = await compute(inputs)
    
    # Result should handle scale differences appropriately
    assert not math.isnan(result)
    # For operations like sum/norm, large values dominate
    # For operations like product, check for under/overflow
```

## Mathematical Properties

### Algebraic Properties
```python
@pytest.mark.property
class TestAlgebraicProperties:
    """Test mathematical properties."""
    
    async def test_commutative(self):
        """a ⊕ b = b ⊕ a"""
        assert await op(a, b) == await op(b, a)
    
    async def test_associative(self):
        """(a ⊕ b) ⊕ c = a ⊕ (b ⊕ c)"""
        left = await op(await op(a, b), c)
        right = await op(a, await op(b, c))
        assert math.isclose(left, right)
    
    async def test_distributive(self):
        """a × (b + c) = (a × b) + (a × c)"""
        left = await multiply(a, await add(b, c))
        right = await add(await multiply(a, b), await multiply(a, c))
        assert math.isclose(left, right)
    
    async def test_identity(self):
        """a ⊕ e = a"""
        identity = await get_identity()
        assert await op(a, identity) == a
    
    async def test_inverse(self):
        """a ⊕ a⁻¹ = e"""
        inverse = await get_inverse(a)
        result = await op(a, inverse)
        assert math.isclose(result, identity)
```

### Mathematical Identities
```python
async def test_trigonometric_identities():
    """Test trig identities."""
    # sin²θ + cos²θ = 1
    sin_val = await sine(angle)
    cos_val = await cosine(angle)
    assert abs(sin_val**2 + cos_val**2 - 1.0) < 1e-10
    
    # tan θ = sin θ / cos θ
    tan_val = await tangent(angle)
    assert abs(tan_val - sin_val/cos_val) < 1e-10

async def test_logarithmic_identities():
    """Test log identities."""
    # log(a×b) = log(a) + log(b)
    log_product = await log(a * b)
    sum_logs = await log(a) + await log(b)
    assert abs(log_product - sum_logs) < 1e-10
```

## Vector and Matrix Testing

### Vector Properties
```python
async def test_vector_properties():
    """Test vector mathematical properties."""
    # Dot product commutative
    assert await dot(v1, v2) == await dot(v2, v1)
    
    # Cross product anti-commutative
    cross1 = await cross(v1, v2)
    cross2 = await cross(v2, v1)
    np.testing.assert_allclose(cross1, -cross2)
    
    # Orthogonal vectors
    if await is_orthogonal(v1, v2):
        assert abs(await dot(v1, v2)) < 1e-10
    
    # Unit vector
    unit = await normalize(v)
    assert abs(await norm(unit) - 1.0) < 1e-10
```

### Matrix Properties
```python
async def test_matrix_properties():
    """Test matrix properties."""
    # Transpose properties
    assert matrices_equal(await transpose(await transpose(A)), A)
    
    # Inverse properties
    A_inv = await inverse(A)
    I = await multiply(A, A_inv)
    assert is_identity_matrix(I, tolerance=1e-10)
    
    # Determinant properties
    det_AB = await determinant(await multiply(A, B))
    det_A_det_B = await determinant(A) * await determinant(B)
    assert abs(det_AB - det_A_det_B) < 1e-10
    
    # Eigenvalue properties
    eigenvals = await eigenvalues(A)
    trace_A = await trace(A)
    assert abs(sum(eigenvals) - trace_A) < 1e-10
```

## Return Type Consistency

### Type Preservation and Conversion
```python
@pytest.mark.asyncio
async def test_return_types():
    """Test that functions return consistent types."""
    # Integer inputs might return float for some operations
    int_result = await compute(5)
    assert isinstance(int_result, (int, float))
    
    # Float inputs should typically return float
    float_result = await compute(5.5)
    assert isinstance(float_result, float)
    
    # Collections should preserve structure
    list_input = [1, 2, 3]
    list_result = await process(list_input)
    assert isinstance(list_result, list)
    assert len(list_result) == len(list_input)
    
    # Ensure input is not modified (immutability)
    original = [1, 2, 3]
    copy = original.copy()
    await process(original)
    assert original == copy
```

## Special Values Testing

### Edge Cases
```python
@pytest.mark.edge_case
async def test_special_values():
    """Test with mathematical special values."""
    # Zero (additive identity)
    assert await func(0) == expected_for_zero
    
    # One (multiplicative identity)
    assert await func(1) == expected_for_one
    
    # Negative values
    assert await func(-1) handles_negative_correctly
    
    # Mathematical constants
    assert math.isfinite(await func(math.pi))
    assert math.isfinite(await func(math.e))
    assert math.isfinite(await func(math.tau))
    
    # Golden ratio
    phi = (1 + math.sqrt(5)) / 2
    assert math.isfinite(await func(phi))
```

### Boundary Conditions
```python
async def test_boundaries():
    """Test mathematical boundaries."""
    # Empty sequences
    with pytest.raises(ValueError):
        await mean([])
    
    # Single element
    assert await mean([5]) == 5
    
    # Powers of 2 (binary boundaries)
    for power in [2**n for n in range(1, 11)]:
        result = await func(power)
        assert math.isfinite(result)
    
    # Near-zero values
    for epsilon in [1e-10, 1e-100, 1e-300]:
        result = await func(epsilon)
        assert result != 0 or epsilon == 0
```

### Infinity and NaN Handling
```python
async def test_special_floats():
    """Test infinity and NaN handling."""
    # Infinity handling
    with pytest.raises(ValueError, match="infinite"):
        await func(float('inf'))
    
    # NaN handling
    with pytest.raises(ValueError, match="NaN"):
        await func(float('nan'))
    
    # Operations producing infinity
    result = await divide(1, 0)
    assert math.isinf(result) or raises_error
    
    # Operations producing NaN
    result = await sqrt(-1)
    assert math.isnan(result) or raises_error
```

## Statistical Testing

### Distribution Properties
```python
async def test_statistical_properties():
    """Test statistical computations."""
    data = generate_normal_distribution(1000)
    
    # Mean and median for symmetric distribution
    mean_val = await mean(data)
    median_val = await median(data)
    assert abs(mean_val - median_val) < 0.1
    
    # Variance is non-negative
    var = await variance(data)
    assert var >= 0
    
    # Standard deviation is sqrt of variance
    std = await std_dev(data)
    assert abs(std - math.sqrt(var)) < 1e-10
    
    # Correlation bounds
    corr = await correlation(x, y)
    assert -1.0 <= corr <= 1.0
```

## Algorithm Correctness

### Known Results
```python
@pytest.mark.parametrize("input,expected", [
    (0, 1),      # 0! = 1
    (1, 1),      # 1! = 1
    (5, 120),    # 5! = 120
    (10, 3628800), # 10! = 3628800
])
async def test_factorial(input, expected):
    """Test against known factorial values."""
    assert await factorial(input) == expected

async def test_known_primes():
    """Test prime number generation."""
    primes = await primes_up_to(30)
    expected = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
    assert primes == expected
```

### Convergence Testing
```python
async def test_iterative_convergence():
    """Test iterative algorithm convergence."""
    # Newton-Raphson for sqrt(2)
    result = await newton_raphson(
        f=lambda x: x**2 - 2,
        initial=1.0,
        tolerance=1e-10,
        max_iters=100
    )
    
    assert abs(result - math.sqrt(2)) < 1e-10
    assert converged_within_max_iters
    
    # Series convergence
    pi_approx = await calculate_pi_series(terms=10000)
    assert abs(pi_approx - math.pi) < 1e-4
```

## Test Data Generation

### Mathematical Test Data
```python
class MathTestData:
    """Generate mathematical test data."""
    
    @staticmethod
    def fibonacci_sequence(n):
        """Generate Fibonacci numbers."""
        if n <= 0: return []
        if n == 1: return [0]
        seq = [0, 1]
        for i in range(2, n):
            seq.append(seq[-1] + seq[-2])
        return seq
    
    @staticmethod
    def prime_numbers(n):
        """Generate first n primes."""
        primes = []
        candidate = 2
        while len(primes) < n:
            if all(candidate % p != 0 for p in primes):
                primes.append(candidate)
            candidate += 1
        return primes
    
    @staticmethod
    def special_matrices():
        """Generate special test matrices."""
        return {
            'identity': np.eye(3),
            'zero': np.zeros((3, 3)),
            'singular': np.array([[1, 2], [2, 4]]),
            'symmetric': np.array([[1, 2], [2, 1]]),
            'orthogonal': np.array([[0, -1], [1, 0]]),
            'positive_definite': np.array([[2, 1], [1, 2]])
        }
```

## Property-Based Testing

### Using Hypothesis
```python
from hypothesis import given, strategies as st

@given(
    st.floats(min_value=-1e6, max_value=1e6, allow_nan=False),
    st.floats(min_value=-1e6, max_value=1e6, allow_nan=False)
)
async def test_addition_properties(a, b):
    """Property-based test for addition."""
    # Commutative
    assert await add(a, b) == await add(b, a)
    
    # Identity
    assert await add(a, 0) == a
    
    # Inverse
    assert abs(await add(a, -a)) < 1e-10

@given(st.lists(st.floats(allow_nan=False), min_size=1))
async def test_mean_properties(data):
    """Property-based test for mean."""
    mean_val = await mean(data)
    
    # Mean is within data range
    assert min(data) <= mean_val <= max(data)
    
    # Mean of constant list
    if all(x == data[0] for x in data):
        assert mean_val == data[0]
```

## Best Practices

### DO's
✅ Always use appropriate tolerance for comparisons  
✅ Test mathematical properties and identities  
✅ Validate against known results  
✅ Test numerical stability  
✅ Handle special values properly  
✅ Test convergence of iterative methods  
✅ Use property-based testing  
✅ Check boundary conditions  

### DON'Ts
❌ Never use exact equality for floats  
❌ Don't ignore numerical precision issues  
❌ Don't skip testing special values  
❌ Don't assume commutativity  
❌ Don't forget about overflow/underflow  
❌ Don't test only happy paths  
❌ Don't ignore ill-conditioned problems  
❌ Don't use fixed seeds everywhere  

## Related Documentation
- [Unit Testing](./UNIT_TESTING.md)
- [Test Fundamentals](./TEST_FUNDAMENTALS.md)
- [Testing Index](./TESTING.md)