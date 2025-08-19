# Performance Testing Patterns

## Overview

Performance testing ensures that code meets speed, memory, and scalability requirements. This document covers patterns for benchmarking, profiling, and load testing.

## Core Principles

### Performance Metrics
- **Latency**: Response time for single operations
- **Throughput**: Operations per second
- **Memory Usage**: RAM consumption
- **CPU Usage**: Processor utilization
- **Scalability**: Performance under load

## Performance Test Organization

### Directory Structure
```
tests/performance/
├── benchmarks/         # Micro-benchmarks
├── load/              # Load and stress tests
├── memory/            # Memory profiling tests
├── regression/        # Performance regression tests
└── conftest.py        # Performance test fixtures
```

## Benchmarking

### Using pytest-benchmark
```python
import pytest

@pytest.mark.benchmark
def test_function_speed(benchmark):
    """Benchmark function execution time."""
    result = benchmark(expensive_function, arg1, arg2)
    
    # Assertions on result
    assert result is not None
    
    # Benchmark stats available in benchmark.stats
    assert benchmark.stats["mean"] < 0.1  # Mean time < 100ms

@pytest.mark.benchmark
def test_compare_algorithms(benchmark):
    """Compare performance of different algorithms."""
    
    def run_algorithm(algo, data):
        return algo(data)
    
    data = generate_test_data(1000)
    
    # Run different algorithms
    if request.config.getoption("--algorithm") == "fast":
        result = benchmark(run_algorithm, fast_algorithm, data)
    else:
        result = benchmark(run_algorithm, slow_algorithm, data)
    
    assert result is not None
```

### Custom Benchmarks
```python
import time
import statistics

class BenchmarkTest:
    """Custom benchmark framework."""
    
    def benchmark_function(self, func, *args, iterations=100):
        """Benchmark a function."""
        times = []
        
        # Warmup
        for _ in range(10):
            func(*args)
        
        # Actual benchmark
        for _ in range(iterations):
            start = time.perf_counter()
            result = func(*args)
            end = time.perf_counter()
            times.append(end - start)
        
        return {
            "min": min(times),
            "max": max(times),
            "mean": statistics.mean(times),
            "median": statistics.median(times),
            "stdev": statistics.stdev(times) if len(times) > 1 else 0,
            "result": result
        }
    
    def test_performance(self):
        """Test function performance."""
        stats = self.benchmark_function(process_data, test_data)
        
        assert stats["mean"] < 0.1  # Average < 100ms
        assert stats["max"] < 0.2   # Worst case < 200ms
        assert stats["stdev"] < 0.05  # Consistent performance
```

## Memory Profiling

### Using memory_profiler
```python
from memory_profiler import profile
import tracemalloc

@pytest.mark.memory
def test_memory_usage():
    """Test memory consumption."""
    tracemalloc.start()
    
    # Run function
    result = memory_intensive_function()
    
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    # Check memory usage
    peak_mb = peak / 1024 / 1024
    assert peak_mb < 100  # Peak usage < 100MB
    
    return result

@profile
def memory_intensive_function():
    """Function to profile."""
    large_list = [i for i in range(1000000)]
    processed = [x * 2 for x in large_list]
    return len(processed)
```

### Memory Leak Detection
```python
import gc
import weakref

@pytest.mark.memory
def test_no_memory_leak():
    """Test that objects are properly garbage collected."""
    # Track object creation
    objects_before = len(gc.get_objects())
    
    # Create objects in a scope
    weak_refs = []
    for _ in range(100):
        obj = LargeObject()
        weak_refs.append(weakref.ref(obj))
    
    # Objects should be collected
    gc.collect()
    
    # Check that objects were deleted
    alive = sum(1 for ref in weak_refs if ref() is not None)
    assert alive == 0, f"{alive} objects not garbage collected"
    
    objects_after = len(gc.get_objects())
    assert objects_after - objects_before < 10  # Minimal growth
```

## Load Testing

### Concurrent Load Testing
```python
import asyncio
import aiohttp

@pytest.mark.load
@pytest.mark.asyncio
async def test_concurrent_requests():
    """Test system under concurrent load."""
    
    async def make_request(session, url):
        """Make a single request."""
        start = time.time()
        async with session.get(url) as response:
            await response.text()
            return time.time() - start
    
    url = "http://localhost:8000/api/endpoint"
    concurrent_requests = 100
    
    async with aiohttp.ClientSession() as session:
        # Create concurrent requests
        tasks = [
            make_request(session, url) 
            for _ in range(concurrent_requests)
        ]
        
        # Execute concurrently
        response_times = await asyncio.gather(*tasks)
    
    # Analyze results
    avg_time = statistics.mean(response_times)
    max_time = max(response_times)
    p95_time = statistics.quantiles(response_times, n=20)[18]  # 95th percentile
    
    assert avg_time < 0.5  # Average response < 500ms
    assert p95_time < 1.0  # 95% of requests < 1s
    assert max_time < 2.0  # No request > 2s
```

### Stress Testing
```python
@pytest.mark.stress
@pytest.mark.asyncio
async def test_stress_limits():
    """Test system breaking point."""
    
    async def stress_test(rate):
        """Apply load at given rate."""
        failures = 0
        successes = 0
        
        for _ in range(rate):
            try:
                await make_request()
                successes += 1
            except Exception:
                failures += 1
            
            await asyncio.sleep(0.01)  # 10ms between requests
        
        return successes, failures
    
    # Gradually increase load
    for rate in [10, 50, 100, 500, 1000]:
        successes, failures = await stress_test(rate)
        
        failure_rate = failures / (successes + failures)
        
        if failure_rate > 0.01:  # 1% failure rate
            print(f"System degrades at {rate} requests/sec")
            break
    
    assert rate >= 100  # Should handle at least 100 req/sec
```

## Performance Regression Testing

### Baseline Comparison
```python
import json

@pytest.mark.regression
def test_performance_regression(benchmark, request):
    """Test against performance baseline."""
    
    # Run benchmark
    result = benchmark(function_to_test, test_data)
    
    # Load baseline
    baseline_file = "performance_baseline.json"
    if os.path.exists(baseline_file):
        with open(baseline_file) as f:
            baseline = json.load(f)
        
        # Compare with baseline
        current_time = benchmark.stats["mean"]
        baseline_time = baseline["mean"]
        
        # Allow 10% regression
        assert current_time < baseline_time * 1.1, \
            f"Performance regressed: {current_time:.3f}s vs {baseline_time:.3f}s"
    
    # Save new baseline if requested
    if request.config.getoption("--save-baseline"):
        with open(baseline_file, "w") as f:
            json.dump(benchmark.stats, f)
```

### Continuous Performance Monitoring
```python
@pytest.mark.performance
class TestPerformanceMonitoring:
    """Monitor performance over time."""
    
    def test_track_performance(self, benchmark, record_property):
        """Track performance metrics."""
        result = benchmark(function_under_test)
        
        # Record metrics for CI/CD
        record_property("performance_mean", benchmark.stats["mean"])
        record_property("performance_max", benchmark.stats["max"])
        record_property("performance_stdev", benchmark.stats["stddev"])
        
        # These can be graphed over time
        assert benchmark.stats["mean"] < 0.1
```

## Profiling

### CPU Profiling
```python
import cProfile
import pstats

@pytest.mark.profile
def test_cpu_profile():
    """Profile CPU usage."""
    profiler = cProfile.Profile()
    
    profiler.enable()
    result = cpu_intensive_function()
    profiler.disable()
    
    # Analyze profile
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    
    # Check specific function times
    function_stats = stats.stats
    
    # Verify optimization targets
    for func_name, stats in function_stats.items():
        if 'critical_function' in str(func_name):
            cumulative_time = stats[3]
            assert cumulative_time < 0.5  # Critical function < 500ms
```

### Line Profiling
```python
from line_profiler import LineProfiler

@pytest.mark.profile
def test_line_profile():
    """Profile line-by-line execution."""
    lp = LineProfiler()
    
    # Add function to profile
    lp.add_function(function_to_profile)
    
    # Run with profiler
    lp.enable()
    result = function_to_profile(test_data)
    lp.disable()
    
    # Get stats
    stats = lp.get_stats()
    
    # Analyze hot spots
    for key, timings in stats.timings.items():
        for line_no, hits, time in timings:
            if time > 1000000:  # Line takes > 1ms
                print(f"Hot spot at line {line_no}: {time/1000000:.2f}ms")
```

## Optimization Validation

### A/B Performance Testing
```python
@pytest.mark.parametrize("implementation", ["original", "optimized"])
def test_optimization_comparison(benchmark, implementation):
    """Compare original vs optimized implementation."""
    
    if implementation == "original":
        result = benchmark(original_function, test_data)
    else:
        result = benchmark(optimized_function, test_data)
    
    # Both should produce same result
    assert result == expected_result
    
    # Stats will be compared by pytest-benchmark
```

### Scalability Testing
```python
@pytest.mark.parametrize("size", [10, 100, 1000, 10000])
def test_scalability(benchmark, size):
    """Test performance scales appropriately."""
    data = generate_data(size)
    
    result = benchmark(process_data, data)
    
    # Check result
    assert len(result) == size
    
    # Performance should scale linearly or better
    # This will be visible in benchmark comparison
```

## Configuration

### Performance Test Settings
```python
# pytest.ini
[tool:pytest]
addopts = 
    --benchmark-only  # Run only benchmarks
    --benchmark-autosave  # Save results automatically
    --benchmark-compare  # Compare with saved results
    --benchmark-max-time=5  # Max time per benchmark

markers =
    benchmark: Benchmark tests
    load: Load tests
    stress: Stress tests
    profile: Profiling tests
    memory: Memory tests
```

### CI/CD Integration
```yaml
# .github/workflows/performance.yml
name: Performance Tests

on:
  pull_request:
    paths:
      - 'src/**'
      - 'tests/performance/**'

jobs:
  performance:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Run performance tests
        run: |
          pytest tests/performance/ \
            --benchmark-json=benchmark.json \
            --benchmark-compare
      
      - name: Upload results
        uses: actions/upload-artifact@v3
        with:
          name: benchmark-results
          path: benchmark.json
      
      - name: Comment on PR
        uses: benchmark-action/github-action-benchmark@v1
        with:
          tool: 'pytest'
          output-file-path: benchmark.json
          comment-on-alert: true
          alert-threshold: '150%'  # Alert if 50% slower
```

## Best Practices

### DO's
✅ Establish performance baselines  
✅ Test with realistic data sizes  
✅ Profile before optimizing  
✅ Test performance regressions  
✅ Monitor trends over time  
✅ Test under concurrent load  
✅ Clean up after tests  
✅ Use appropriate timeouts  

### DON'Ts
❌ Don't optimize prematurely  
❌ Don't test on development machines only  
❌ Don't ignore variance in results  
❌ Don't profile with tiny datasets  
❌ Don't mix debug and release builds  
❌ Don't forget warmup iterations  
❌ Don't test with caching disabled  
❌ Don't ignore memory usage  

## Related Documentation
- [Unit Testing](./UNIT_TESTING.md)
- [Integration Testing](./INTEGRATION_TESTING.md)
- [Test Fundamentals](./TEST_FUNDAMENTALS.md)
- [Testing Index](./TESTING.md)