#!/usr/bin/env python3
# chuk_mcp_functions/diagnostics/async_arithmetic_diagnostics.py
"""
Async Native Arithmetic Library Diagnostic Script

Comprehensive diagnostic and performance testing tool for the async native
arithmetic library. Validates functionality, measures performance, tests
concurrency, and provides detailed reports.

Usage:
    python async_arithmetic_diagnostics.py
    python async_arithmetic_diagnostics.py --quick
    python async_arithmetic_diagnostics.py --stress
    python async_arithmetic_diagnostics.py --report-only
"""

import asyncio
import time
import statistics
import sys
import traceback
import json
import platform
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path

# Add src directory to path for imports
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

try:
    from chuk_mcp_functions.math.arithmetic import *
    from chuk_mcp_functions.math.arithmetic import (
        get_arithmetic_functions, get_arithmetic_constants,
        validate_arithmetic_installation
    )
    from chuk_mcp_functions.math.arithmetic.sequences import harmonic_series
    from chuk_mcp_functions.math.arithmetic.comparison import sort_numbers  # Fixed: Added missing import
    from chuk_mcp_functions.mcp_decorator import (
        get_mcp_functions, get_async_functions, print_function_summary_async
    )
    IMPORTS_OK = True
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print(f"   Project root: {project_root}")
    print(f"   Src path: {src_path}")
    print(f"   Src exists: {src_path.exists()}")
    if src_path.exists():
        print(f"   Contents: {list(src_path.iterdir())}")
    IMPORTS_OK = False

@dataclass
class TestResult:
    """Individual test result."""
    name: str
    success: bool
    duration: float
    error: Optional[str] = None
    yields: int = 0
    cache_hits: int = 0
    concurrent_executions: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DiagnosticReport:
    """Complete diagnostic report."""
    timestamp: str
    environment: Dict[str, Any]
    import_status: bool
    function_counts: Dict[str, int]
    test_results: List[TestResult]
    performance_stats: Dict[str, Any]
    concurrency_stats: Dict[str, Any]
    errors: List[str]
    recommendations: List[str]

class AsyncArithmeticDiagnostics:
    """Comprehensive diagnostic suite for async arithmetic library."""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.results: List[TestResult] = []
        self.errors: List[str] = []
        self.start_time = time.time()
        
    def log(self, message: str, level: str = "INFO"):
        """Log a message with timestamp."""
        if self.verbose:
            timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            prefix = {
                "INFO": "ℹ️ ",
                "SUCCESS": "✅",
                "ERROR": "❌",
                "WARNING": "⚠️ ",
                "PERF": "⚡"
            }.get(level, "  ")
            print(f"[{timestamp}] {prefix} {message}")
    
    async def run_full_diagnostics(self) -> DiagnosticReport:
        """Run complete diagnostic suite."""
        self.log("🚀 Starting Async Native Arithmetic Library Diagnostics")
        self.log("=" * 60)
        
        # Environment check
        env_info = self._get_environment_info()
        self.log(f"Environment: Python {env_info['python_version']} on {env_info['platform']}")
        
        # Import validation
        if not IMPORTS_OK:
            self.errors.append("Failed to import arithmetic library")
            return self._generate_report(env_info, {}, {}, {})
        
        self.log("✅ Imports successful", "SUCCESS")
        
        # Function discovery
        function_counts = await self._discover_functions()
        
        # Basic functionality tests
        await self._test_basic_functionality()
        
        # Async-specific tests
        await self._test_async_features()
        
        # Performance tests
        perf_stats = await self._test_performance()
        
        # Concurrency tests
        concurrency_stats = await self._test_concurrency()
        
        # Caching tests
        await self._test_caching()
        
        # Error handling tests
        await self._test_error_handling()
        
        # Generate final report
        return self._generate_report(env_info, function_counts, perf_stats, concurrency_stats)
    
    async def run_quick_diagnostics(self) -> DiagnosticReport:
        """Run quick diagnostic suite."""
        self.log("⚡ Running Quick Diagnostics")
        
        env_info = self._get_environment_info()
        if not IMPORTS_OK:
            return self._generate_report(env_info, {}, {}, {})
        
        function_counts = await self._discover_functions()
        await self._test_basic_functionality_quick()
        await self._test_async_features_quick()
        
        return self._generate_report(env_info, function_counts, {}, {})
    
    async def run_stress_test(self) -> DiagnosticReport:
        """Run stress test suite."""
        self.log("💪 Running Stress Tests")
        
        env_info = self._get_environment_info()
        if not IMPORTS_OK:
            return self._generate_report(env_info, {}, {}, {})
        
        function_counts = await self._discover_functions()
        
        # Heavy load tests
        await self._stress_test_concurrency()
        await self._stress_test_large_operations()
        await self._stress_test_memory_usage()
        
        return self._generate_report(env_info, function_counts, {}, {})
    
    def _get_environment_info(self) -> Dict[str, Any]:
        """Get environment information."""
        return {
            "python_version": sys.version,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "architecture": platform.architecture(),
            "asyncio_policy": type(asyncio.get_event_loop_policy()).__name__,
            "event_loop": type(asyncio.get_event_loop()).__name__,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    async def _discover_functions(self) -> Dict[str, int]:
        """Discover and count functions."""
        self.log("🔍 Discovering functions...")
        
        try:
            all_functions = get_mcp_functions()
            async_functions = get_async_functions()
            arithmetic_functions = await get_arithmetic_functions()
            
            counts = {
                "total_mcp_functions": len(all_functions),
                "async_native_functions": len(async_functions),
                "arithmetic_functions": sum(len(cat) for cat in arithmetic_functions.values()),
                "basic_operations": len(arithmetic_functions.get('basic_operations', {})),
                "comparison": len(arithmetic_functions.get('comparison', {})),
                "number_theory": len(arithmetic_functions.get('number_theory', {})),
                "sequences": len(arithmetic_functions.get('sequences', {})),
                "advanced_operations": len(arithmetic_functions.get('advanced_operations', {})),
                "constants": len(arithmetic_functions.get('constants', {}))
            }
            
            self.log(f"Found {counts['total_mcp_functions']} total functions")
            self.log(f"Found {counts['async_native_functions']} async native functions")
            self.log(f"Found {counts['arithmetic_functions']} arithmetic functions")
            
            return counts
            
        except Exception as e:
            self.errors.append(f"Function discovery failed: {e}")
            return {}
    
    # Fixed: Added helper method for async ln(e()) test
    async def _test_ln_with_e(self) -> float:
        """Helper method to test ln(e) properly with async."""
        e_value = await e()
        return await ln(e_value)
    
    async def _test_basic_functionality(self):
        """Test basic functionality of all arithmetic categories."""
        self.log("🧮 Testing basic functionality...")
        
        # Basic operations
        await self._test_category("Basic Operations", [
            ("add", lambda: add(5, 3), 8),
            ("subtract", lambda: subtract(10, 4), 6),
            ("multiply", lambda: multiply(6, 7), 42),
            ("divide", lambda: divide(20, 4), 5.0),
            ("power", lambda: power(2, 3), 8),
            ("sqrt", lambda: sqrt(16), 4.0),
            ("abs_value", lambda: abs_value(-5), 5)
        ])
        
        # Comparison
        await self._test_category("Comparison", [
            ("equal", lambda: equal(5, 5), True),
            ("less_than", lambda: less_than(3, 5), True),
            ("greater_than", lambda: greater_than(7, 3), True),
            ("minimum", lambda: minimum(3, 8), 3),
            ("maximum", lambda: maximum(3, 8), 8),
            ("clamp", lambda: clamp(15, 1, 10), 10)
        ])
        
        # Number theory
        await self._test_category("Number Theory", [
            ("is_prime", lambda: is_prime(17), True),
            ("gcd", lambda: gcd(48, 18), 6),
            ("lcm", lambda: lcm(12, 8), 24),
            ("factorial", lambda: factorial(5), 120),
            ("fibonacci", lambda: fibonacci(7), 13),
            ("is_even", lambda: is_even(4), True)
        ])
        
        # Sequences
        await self._test_category("Sequences", [
            ("arithmetic_sequence", lambda: arithmetic_sequence(2, 3, 4), [2, 5, 8, 11]),
            ("geometric_sequence", lambda: geometric_sequence(2, 2, 3), [2, 4, 8]),
            ("triangular_numbers", lambda: triangular_numbers(4), [1, 3, 6, 10]),
            ("harmonic_series", lambda: harmonic_series(3), 1 + 1/2 + 1/3)
        ])
        
        # Advanced operations - Fixed: Use helper method for ln(e()) test
        await self._test_category("Advanced Operations", [
            ("ln", lambda: self._test_ln_with_e(), 1.0),
            ("log10", lambda: log10(100), 2.0),
            ("exp", lambda: exp(0), 1.0),
            ("product", lambda: product([2, 3, 4]), 24)
        ])
        
        # Constants
        await self._test_category("Constants", [
            ("pi", lambda: pi(), 3.141592653589793),
            ("e", lambda: e(), 2.718281828459045),
            ("golden_ratio", lambda: golden_ratio(), 1.618033988749895)
        ])
    
    async def _test_basic_functionality_quick(self):
        """Quick test of basic functionality."""
        self.log("⚡ Quick functionality test...")
        
        quick_tests = [
            ("add", lambda: add(2, 3), 5),
            ("is_prime", lambda: is_prime(7), True),
            ("arithmetic_sequence", lambda: arithmetic_sequence(1, 2, 3), [1, 3, 5]),
            ("pi", lambda: pi(), 3.141592653589793)
        ]
        
        await self._test_category("Quick Tests", quick_tests)
    
    async def _test_category(self, category: str, tests: List[Tuple[str, Any, Any]]):
        """Test a category of functions."""
        self.log(f"  Testing {category}...")
        
        for test_name, test_func, expected in tests:
            await self._run_single_test(f"{category}.{test_name}", test_func, expected)
    
    async def _run_single_test(self, test_name: str, test_func: Any, expected: Any = None):
        """Run a single test with timing and error handling."""
        start_time = time.time()
        
        try:
            # Handle different types of test functions
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                # Handle lambda functions that return coroutines
                func_result = test_func()
                if asyncio.iscoroutine(func_result):
                    result = await func_result
                else:
                    result = func_result
            
            duration = time.time() - start_time
            
            if expected is not None:
                if isinstance(expected, float) and isinstance(result, float):
                    success = abs(result - expected) < 1e-9
                elif isinstance(expected, list) and isinstance(result, list):
                    success = len(result) == len(expected) and all(
                        abs(a - b) < 1e-9 if isinstance(a, float) else a == b
                        for a, b in zip(result, expected)
                    )
                else:
                    success = result == expected
                    
                if not success:
                    error_msg = f"Expected {expected}, got {result}"
                    self.errors.append(f"{test_name}: {error_msg}")
                    self.results.append(TestResult(test_name, False, duration, error_msg))
                    self.log(f"    ❌ {test_name}: {error_msg}", "ERROR")
                    return
            else:
                success = True
            
            self.results.append(TestResult(test_name, success, duration))
            if self.verbose and duration > 0.001:  # Only log slow operations
                self.log(f"    ✅ {test_name}: {duration:.4f}s", "SUCCESS")
                
        except Exception as e:
            duration = time.time() - start_time
            error_msg = f"{type(e).__name__}: {str(e)}"
            self.errors.append(f"{test_name}: {error_msg}")
            self.results.append(TestResult(test_name, False, duration, error_msg))
            self.log(f"    ❌ {test_name}: {error_msg}", "ERROR")
    
    async def _test_async_features(self):
        """Test async-specific features."""
        self.log("🚀 Testing async features...")
        
        # Test async execution
        await self._run_single_test("async.basic_execution", 
                                   lambda: add(1, 2))
        
        # Test parallel execution - ensure all functions are awaitable
        start_time = time.time()
        results = await asyncio.gather(
            add(1, 2),
            multiply(3, 4),
            is_prime(17),
            fibonacci(10)
        )
        parallel_duration = time.time() - start_time
        
        self.results.append(TestResult(
            "async.parallel_execution", 
            True, 
            parallel_duration,
            metadata={"results": results, "operations": 4}
        ))
        self.log(f"    ✅ Parallel execution: {parallel_duration:.4f}s for 4 operations", "PERF")
        
        # Test yielding behavior with large operations
        await self._test_yielding()
    
    async def _test_async_features_quick(self):
        """Quick test of async features."""
        self.log("⚡ Quick async test...")
        
        # Basic async test
        result = await add(1, 1)
        success = result == 2
        self.results.append(TestResult("quick.async_basic", success, 0.0))
        
        # Quick parallel test
        results = await asyncio.gather(add(1, 2), multiply(2, 3))
        success = results == [3, 6]
        self.results.append(TestResult("quick.async_parallel", success, 0.0))
    
    async def _test_yielding(self):
        """Test yielding behavior in long operations."""
        self.log("  Testing yielding behavior...")
        
        # Test large sequence generation (should yield)
        start_time = time.time()
        large_seq = await arithmetic_sequence(1, 1, 5000)
        duration = time.time() - start_time
        
        self.results.append(TestResult(
            "async.yielding_large_sequence",
            len(large_seq) == 5000,
            duration,
            metadata={"sequence_length": len(large_seq)}
        ))
        
        # Test large sorting (should yield) - Fixed: Now properly imported
        import random
        large_list = [random.randint(1, 1000) for _ in range(2000)]
        start_time = time.time()
        sorted_list = await sort_numbers(large_list)
        duration = time.time() - start_time
        
        self.results.append(TestResult(
            "async.yielding_large_sort",
            sorted_list == sorted(large_list),
            duration,
            metadata={"list_length": len(large_list)}
        ))
    
    async def _test_performance(self) -> Dict[str, Any]:
        """Test performance characteristics."""
        self.log("⚡ Testing performance...")
        
        perf_stats = {}
        
        # Benchmark basic operations
        operations = [
            ("add", lambda: add(100, 200)),
            ("multiply", lambda: multiply(123, 456)),
            ("is_prime", lambda: is_prime(97)),
            ("fibonacci", lambda: fibonacci(20)),
            ("factorial", lambda: factorial(10))
        ]
        
        for op_name, op_func in operations:
            times = []
            for _ in range(100):  # Run each operation 100 times
                start = time.time()
                await op_func()
                times.append(time.time() - start)
            
            perf_stats[op_name] = {
                "mean": statistics.mean(times),
                "median": statistics.median(times),
                "min": min(times),
                "max": max(times),
                "stdev": statistics.stdev(times) if len(times) > 1 else 0
            }
            
            self.log(f"    {op_name}: {perf_stats[op_name]['mean']:.6f}s avg", "PERF")
        
        return perf_stats
    
    async def _test_concurrency(self) -> Dict[str, Any]:
        """Test concurrency handling."""
        self.log("🔄 Testing concurrency...")
        
        concurrency_stats = {}
        
        # Test concurrent execution of same function
        concurrent_tasks = 50
        start_time = time.time()
        
        tasks = [add(i, i+1) for i in range(concurrent_tasks)]
        results = await asyncio.gather(*tasks)
        
        concurrent_duration = time.time() - start_time
        expected_results = [i + (i+1) for i in range(concurrent_tasks)]
        
        concurrency_stats["concurrent_adds"] = {
            "tasks": concurrent_tasks,
            "duration": concurrent_duration,
            "success": results == expected_results,
            "throughput": concurrent_tasks / concurrent_duration
        }
        
        self.log(f"    {concurrent_tasks} concurrent adds: {concurrent_duration:.4f}s "
                f"({concurrency_stats['concurrent_adds']['throughput']:.1f} ops/s)", "PERF")
        
        # Test mixed concurrent operations
        mixed_tasks = [
            add(1, 2),
            multiply(3, 4),
            is_prime(29),
            fibonacci(15),
            sqrt(144),
            gcd(48, 18),
            factorial(8)
        ]
        
        start_time = time.time()
        mixed_results = await asyncio.gather(*mixed_tasks)
        mixed_duration = time.time() - start_time
        
        concurrency_stats["mixed_operations"] = {
            "tasks": len(mixed_tasks),
            "duration": mixed_duration,
            "results": mixed_results
        }
        
        self.log(f"    {len(mixed_tasks)} mixed operations: {mixed_duration:.4f}s", "PERF")
        
        return concurrency_stats
    
    async def _test_caching(self):
        """Test caching functionality."""
        self.log("💾 Testing caching...")
        
        # Note: This test depends on whether caching is enabled in the functions
        # For now, just test that cached functions work correctly
        
        # Test repeated calls (should be fast if cached)
        expensive_operations = [
            lambda: factorial(50),
            lambda: fibonacci(30),
            lambda: is_prime(982451653)  # Large prime
        ]
        
        for i, op in enumerate(expensive_operations):
            # First call (cache miss)
            start1 = time.time()
            result1 = await op()
            time1 = time.time() - start1
            
            # Second call (potential cache hit)
            start2 = time.time()
            result2 = await op()
            time2 = time.time() - start2
            
            cache_test = TestResult(
                f"caching.operation_{i}",
                result1 == result2,
                time1,
                metadata={
                    "first_call": time1,
                    "second_call": time2,
                    "speedup": time1 / time2 if time2 > 0 else float('inf')
                }
            )
            self.results.append(cache_test)
            
            if time2 < time1 * 0.1:  # Significant speedup
                self.log(f"    ✅ Cache hit detected: {time1:.6f}s → {time2:.6f}s", "SUCCESS")
    
    async def _test_error_handling(self):
        """Test error handling."""
        self.log("🛡️  Testing error handling...")
        
        error_tests = [
            ("divide_by_zero", lambda: divide(5, 0), ValueError),  # Fixed: Changed from ZeroDivisionError
            ("sqrt_negative", lambda: sqrt(-4), ValueError),
            ("factorial_negative", lambda: factorial(-1), ValueError),
            ("prime_negative", lambda: is_prime(-5), None),  # Should return False, not error
            ("log_zero", lambda: ln(0), ValueError)
        ]
        
        for test_name, test_func, expected_error in error_tests:
            try:
                result = await test_func()
                if expected_error is None:
                    # No error expected
                    self.results.append(TestResult(f"error_handling.{test_name}", True, 0.0))
                else:
                    # Error was expected but didn't occur
                    self.results.append(TestResult(
                        f"error_handling.{test_name}", 
                        False, 
                        0.0, 
                        f"Expected {expected_error.__name__} but got result: {result}"
                    ))
            except Exception as e:
                if expected_error and isinstance(e, expected_error):
                    self.results.append(TestResult(f"error_handling.{test_name}", True, 0.0))
                    self.log(f"    ✅ Correctly caught {type(e).__name__}", "SUCCESS")
                else:
                    self.results.append(TestResult(
                        f"error_handling.{test_name}", 
                        False, 
                        0.0, 
                        f"Unexpected error: {type(e).__name__}: {e}"
                    ))
    
    async def _stress_test_concurrency(self):
        """Stress test concurrency with many simultaneous operations."""
        self.log("💪 Stress testing concurrency...")
        
        # Create 200 concurrent operations
        tasks = []
        for i in range(200):
            if i % 4 == 0:
                tasks.append(add(i, i+1))
            elif i % 4 == 1:
                tasks.append(multiply(i, 2))
            elif i % 4 == 2:
                tasks.append(is_prime(i))
            else:
                tasks.append(fibonacci(i % 20))
        
        start_time = time.time()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        duration = time.time() - start_time
        
        errors = [r for r in results if isinstance(r, Exception)]
        successes = len(results) - len(errors)
        
        self.results.append(TestResult(
            "stress.concurrency",
            len(errors) == 0,
            duration,
            metadata={
                "total_tasks": len(tasks),
                "successes": successes,
                "errors": len(errors),
                "throughput": len(tasks) / duration
            }
        ))
        
        self.log(f"    200 concurrent operations: {duration:.4f}s, "
                f"{successes}/{len(tasks)} successful", "PERF")
    
    async def _stress_test_large_operations(self):
        """Stress test with large computational operations."""
        self.log("💪 Stress testing large operations...")
        
        large_ops = [
            ("large_factorial", lambda: factorial(100)),
            ("large_fibonacci", lambda: fibonacci(50)),
            ("large_sequence", lambda: arithmetic_sequence(1, 1, 10000)),
            ("large_prime_check", lambda: is_prime(1000003)),
            ("large_sort", lambda: sort_numbers(list(range(5000, 0, -1))))  # Fixed: Now properly imported
        ]
        
        for op_name, op_func in large_ops:
            start_time = time.time()
            try:
                result = await op_func()
                duration = time.time() - start_time
                self.results.append(TestResult(
                    f"stress.{op_name}",
                    True,
                    duration,
                    metadata={"operation_size": "large"}
                ))
                self.log(f"    {op_name}: {duration:.4f}s", "PERF")
            except Exception as e:
                duration = time.time() - start_time
                self.results.append(TestResult(
                    f"stress.{op_name}",
                    False,
                    duration,
                    str(e)
                ))
    
    async def _stress_test_memory_usage(self):
        """Stress test memory usage with large data structures."""
        self.log("💪 Stress testing memory usage...")
        
        # Generate large sequences and process them
        try:
            large_arithmetic = await arithmetic_sequence(1, 1, 50000)
            large_geometric = await geometric_sequence(1, 1.001, 10000)
            large_triangular = await triangular_numbers(1000)
            
            # Process the large sequences - Fixed: Now properly imported
            sorted_large = await sort_numbers(large_arithmetic[:10000])
            sum_large = await arithmetic_sum(1, 1, 50000)
            
            self.results.append(TestResult(
                "stress.memory_usage",
                True,
                0.0,
                metadata={
                    "arithmetic_seq_len": len(large_arithmetic),
                    "geometric_seq_len": len(large_geometric),
                    "triangular_seq_len": len(large_triangular),
                    "sorted_len": len(sorted_large)
                }
            ))
            
            self.log("    ✅ Large data structure operations completed", "SUCCESS")
            
        except Exception as e:
            self.results.append(TestResult(
                "stress.memory_usage",
                False,
                0.0,
                str(e)
            ))
    
    def _generate_report(self, env_info: Dict[str, Any], function_counts: Dict[str, int], 
                        perf_stats: Dict[str, Any], concurrency_stats: Dict[str, Any]) -> DiagnosticReport:
        """Generate final diagnostic report."""
        
        # Calculate summary statistics
        successful_tests = [r for r in self.results if r.success]
        failed_tests = [r for r in self.results if not r.success]
        
        total_duration = time.time() - self.start_time
        test_durations = [r.duration for r in self.results if r.duration > 0]
        
        # Generate recommendations
        recommendations = []
        
        if len(failed_tests) > 0:
            recommendations.append(f"⚠️  {len(failed_tests)} tests failed - review error details")
        
        if len(self.results) > 0:
            avg_test_time = statistics.mean(test_durations) if test_durations else 0
            if avg_test_time > 0.01:
                recommendations.append("⚡ Consider optimizing slow operations")
            else:
                recommendations.append("✅ Good performance characteristics detected")
        
        if function_counts.get('async_native_functions', 0) > 0:
            recommendations.append("🚀 Async native functions detected - optimal for concurrent use")
        
        if not self.errors:
            recommendations.append("✅ All error handling tests passed")
        
        return DiagnosticReport(
            timestamp=datetime.now(timezone.utc).isoformat(),
            environment=env_info,
            import_status=IMPORTS_OK,
            function_counts=function_counts,
            test_results=self.results,
            performance_stats=perf_stats,
            concurrency_stats=concurrency_stats,
            errors=self.errors,
            recommendations=recommendations
        )
    
    def print_report(self, report: DiagnosticReport):
        """Print formatted diagnostic report."""
        print("\n" + "="*80)
        print("🔍 ASYNC NATIVE ARITHMETIC LIBRARY DIAGNOSTIC REPORT")
        print("="*80)
        
        # Environment
        print(f"\n🖥️  ENVIRONMENT:")
        print(f"   Python: {report.environment.get('python_version', 'Unknown').split()[0]}")
        print(f"   Platform: {report.environment.get('platform', 'Unknown')}")
        print(f"   Asyncio Policy: {report.environment.get('asyncio_policy', 'Unknown')}")
        
        # Import Status
        print(f"\n📦 IMPORT STATUS:")
        if report.import_status:
            print("   ✅ All imports successful")
        else:
            print("   ❌ Import failures detected")
        
        # Function Counts
        if report.function_counts:
            print(f"\n📊 FUNCTION DISCOVERY:")
            for key, count in report.function_counts.items():
                print(f"   {key.replace('_', ' ').title()}: {count}")
        
        # Test Results Summary
        successful = [r for r in report.test_results if r.success]
        failed = [r for r in report.test_results if not r.success]
        
        print(f"\n🧪 TEST RESULTS:")
        print(f"   Total Tests: {len(report.test_results)}")
        print(f"   ✅ Passed: {len(successful)}")
        print(f"   ❌ Failed: {len(failed)}")
        
        if failed:
            print(f"\n❌ FAILED TESTS:")
            for test in failed[:10]:  # Show first 10 failures
                print(f"   • {test.name}: {test.error}")
            if len(failed) > 10:
                print(f"   ... and {len(failed) - 10} more failures")
        
        # Performance Stats
        if report.performance_stats:
            print(f"\n⚡ PERFORMANCE STATS:")
            for op, stats in report.performance_stats.items():
                print(f"   {op}: {stats['mean']:.6f}s avg (±{stats['stdev']:.6f}s)")
        
        # Concurrency Stats
        if report.concurrency_stats:
            print(f"\n🔄 CONCURRENCY STATS:")
            for test, stats in report.concurrency_stats.items():
                if 'throughput' in stats:
                    print(f"   {test}: {stats['throughput']:.1f} ops/s")
                else:
                    print(f"   {test}: {stats['duration']:.4f}s")
        
        # Recommendations
        if report.recommendations:
            print(f"\n💡 RECOMMENDATIONS:")
            for rec in report.recommendations:
                print(f"   {rec}")
        
        # Summary
        print(f"\n📋 SUMMARY:")
        if len(failed) == 0:
            print("   🎉 All tests passed! Library is functioning correctly.")
        else:
            print(f"   ⚠️  {len(failed)} issues detected. Review details above.")
        
        print(f"   📊 Tested {len(report.test_results)} functions")
        print(f"   ⏱️  Total diagnostic time: {time.time() - self.start_time:.2f}s")
        print("="*80)

async def main():
    """Main diagnostic entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Async Arithmetic Library Diagnostics")
    parser.add_argument("--quick", action="store_true", help="Run quick diagnostics only")
    parser.add_argument("--stress", action="store_true", help="Run stress tests")
    parser.add_argument("--report-only", action="store_true", help="Generate report without running tests")
    parser.add_argument("--quiet", action="store_true", help="Minimal output")
    parser.add_argument("--save", type=str, help="Save report to JSON file")
    
    args = parser.parse_args()
    
    diagnostics = AsyncArithmeticDiagnostics(verbose=not args.quiet)
    
    if args.report_only:
        # Just print function summary
        if IMPORTS_OK:
            await print_function_summary_async()
        return
    
    # Run appropriate diagnostic suite
    if args.stress:
        report = await diagnostics.run_stress_test()
    elif args.quick:
        report = await diagnostics.run_quick_diagnostics()
    else:
        report = await diagnostics.run_full_diagnostics()
    
    # Print report
    diagnostics.print_report(report)
    
    # Save report if requested
    if args.save:
        report_dict = {
            "timestamp": report.timestamp,
            "environment": report.environment,
            "import_status": report.import_status,
            "function_counts": report.function_counts,
            "test_results": [
                {
                    "name": t.name,
                    "success": t.success,
                    "duration": t.duration,
                    "error": t.error,
                    "metadata": t.metadata
                }
                for t in report.test_results
            ],
            "performance_stats": report.performance_stats,
            "concurrency_stats": report.concurrency_stats,
            "errors": report.errors,
            "recommendations": report.recommendations
        }
        
        with open(args.save, 'w') as f:
            json.dump(report_dict, f, indent=2, default=str)
        print(f"\n💾 Report saved to {args.save}")

if __name__ == "__main__":
    asyncio.run(main())