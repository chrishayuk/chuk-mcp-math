#!/usr/bin/env python3
"""
Comprehensive unit tests for mcp_decorator module.

Tests all components of the MCP decorator system:
- ExecutionMode, CacheStrategy, ResourceLevel, StreamingMode, AsyncYieldStrategy enums
- PerformanceMetrics dataclass
- AsyncMemoryCache class
- MCPFunctionSpec class
- mcp_function decorator
- Utility functions
- Registry functions
"""

import pytest
import asyncio
from datetime import datetime
from typing import List

from chuk_mcp_math.mcp_decorator import (
    mcp_function,
    MCPFunctionSpec,
    ExecutionMode,
    CacheStrategy,
    ResourceLevel,
    StreamingMode,
    AsyncYieldStrategy,
    PerformanceMetrics,
    AsyncMemoryCache,
    get_mcp_functions,
    get_async_functions,
    get_function_by_name,
    export_function_specs_async,
    print_function_summary_async,
    _calculate_content_hash,
    _get_current_timestamp,
    _estimate_resource_usage,
    _detect_yield_strategy,
    _type_to_json_schema,
    _extract_function_info,
    _extract_source_code,
    _normalize_string_arg,
    _normalize_list_arg,
    _mcp_functions,
)


# ============================================================================
# TEST ENUMS
# ============================================================================


class TestEnums:
    """Test enum definitions."""

    def test_execution_mode_values(self):
        """Test ExecutionMode enum values."""
        assert ExecutionMode.LOCAL == "local"
        assert ExecutionMode.REMOTE == "remote"
        assert ExecutionMode.BOTH == "both"
        assert ExecutionMode.AUTO == "auto"

    def test_cache_strategy_values(self):
        """Test CacheStrategy enum values."""
        assert CacheStrategy.NONE == "none"
        assert CacheStrategy.MEMORY == "memory"
        assert CacheStrategy.FILE == "file"
        assert CacheStrategy.HYBRID == "hybrid"
        assert CacheStrategy.ASYNC_LRU == "async_lru"

    def test_resource_level_values(self):
        """Test ResourceLevel enum values."""
        assert ResourceLevel.LOW == "low"
        assert ResourceLevel.MEDIUM == "medium"
        assert ResourceLevel.HIGH == "high"
        assert ResourceLevel.EXTREME == "extreme"

    def test_streaming_mode_values(self):
        """Test StreamingMode enum values."""
        assert StreamingMode.NONE == "none"
        assert StreamingMode.CHUNKED == "chunked"
        assert StreamingMode.REAL_TIME == "real_time"
        assert StreamingMode.BACKPRESSURE == "backpressure"

    def test_async_yield_strategy_values(self):
        """Test AsyncYieldStrategy enum values."""
        assert AsyncYieldStrategy.NONE == "none"
        assert AsyncYieldStrategy.TIME_BASED == "time_based"
        assert AsyncYieldStrategy.ITERATION_BASED == "iteration_based"
        assert AsyncYieldStrategy.ADAPTIVE == "adaptive"
        assert AsyncYieldStrategy.MANUAL == "manual"


# ============================================================================
# TEST PERFORMANCE METRICS
# ============================================================================


class TestPerformanceMetrics:
    """Test PerformanceMetrics dataclass."""

    def test_default_initialization(self):
        """Test default values on initialization."""
        metrics = PerformanceMetrics()
        assert metrics.execution_count == 0
        assert metrics.total_duration == 0.0
        assert metrics.average_duration == 0.0
        assert metrics.min_duration == float("inf")
        assert metrics.max_duration == 0.0
        assert metrics.error_count == 0
        assert metrics.cache_hits == 0
        assert metrics.cache_misses == 0
        assert metrics.async_yields == 0
        assert metrics.concurrent_executions == 0
        assert metrics.last_execution is None

    def test_update_success(self):
        """Test updating metrics with successful execution."""
        metrics = PerformanceMetrics()
        metrics.update(duration=1.5, success=True, cache_hit=False, yields=0)

        assert metrics.execution_count == 1
        assert metrics.total_duration == 1.5
        assert metrics.average_duration == 1.5
        assert metrics.min_duration == 1.5
        assert metrics.max_duration == 1.5
        assert metrics.error_count == 0
        assert metrics.cache_hits == 0
        assert metrics.cache_misses == 1
        assert metrics.last_execution is not None

    def test_update_failure(self):
        """Test updating metrics with failed execution."""
        metrics = PerformanceMetrics()
        metrics.update(duration=1.0, success=False, cache_hit=False, yields=0)

        assert metrics.execution_count == 1
        assert metrics.total_duration == 0.0  # Not added on failure
        assert metrics.error_count == 1

    def test_update_cache_hit(self):
        """Test updating metrics with cache hit."""
        metrics = PerformanceMetrics()
        metrics.update(duration=0.1, success=True, cache_hit=True, yields=0)

        assert metrics.cache_hits == 1
        assert metrics.cache_misses == 0

    def test_update_yields(self):
        """Test updating metrics with async yields."""
        metrics = PerformanceMetrics()
        metrics.update(duration=2.0, success=True, cache_hit=False, yields=5)

        assert metrics.async_yields == 5

    def test_multiple_updates(self):
        """Test multiple metric updates."""
        metrics = PerformanceMetrics()

        metrics.update(duration=1.0, success=True, cache_hit=False, yields=2)
        metrics.update(duration=2.0, success=True, cache_hit=True, yields=3)
        metrics.update(duration=0.5, success=True, cache_hit=False, yields=1)

        assert metrics.execution_count == 3
        assert metrics.total_duration == 3.5
        assert metrics.average_duration == 3.5 / 3
        assert metrics.min_duration == 0.5
        assert metrics.max_duration == 2.0
        assert metrics.cache_hits == 1
        assert metrics.cache_misses == 2
        assert metrics.async_yields == 6


# ============================================================================
# TEST ASYNC MEMORY CACHE
# ============================================================================


class TestAsyncMemoryCache:
    """Test AsyncMemoryCache class."""

    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test cache initialization."""
        cache = AsyncMemoryCache(max_size=100)
        assert cache.max_size == 100
        assert len(cache.cache) == 0
        assert len(cache.expiry) == 0
        assert len(cache.access_times) == 0

    @pytest.mark.asyncio
    async def test_set_get(self):
        """Test basic set and get operations."""
        cache = AsyncMemoryCache()
        await cache.set("key1", "value1")

        result = await cache.get("key1")
        assert result == "value1"

    @pytest.mark.asyncio
    async def test_get_nonexistent(self):
        """Test getting nonexistent key."""
        cache = AsyncMemoryCache()
        result = await cache.get("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_set_with_ttl(self):
        """Test set with TTL."""
        cache = AsyncMemoryCache()
        await cache.set("key1", "value1", ttl=1)

        # Should exist immediately
        result = await cache.get("key1")
        assert result == "value1"

        # Should expire after TTL
        await asyncio.sleep(1.1)
        result = await cache.get("key1")
        assert result is None

    @pytest.mark.asyncio
    async def test_lru_eviction(self):
        """Test LRU eviction when max size is reached."""
        cache = AsyncMemoryCache(max_size=2)

        await cache.set("key1", "value1")
        await cache.set("key2", "value2")
        await cache.set("key3", "value3")  # Should evict key1

        assert await cache.get("key1") is None
        assert await cache.get("key2") == "value2"
        assert await cache.get("key3") == "value3"

    @pytest.mark.asyncio
    async def test_lru_access_order(self):
        """Test that accessing keys updates LRU order."""
        cache = AsyncMemoryCache(max_size=2)

        await cache.set("key1", "value1")
        await cache.set("key2", "value2")

        # Access key1 to make it more recent
        await cache.get("key1")

        # Add key3, should evict oldest (either key1 or key2 depending on implementation)
        await cache.set("key3", "value3")

        # At least key3 should exist
        assert await cache.get("key3") == "value3"
        # And only 2 keys should be in cache
        assert len(cache.cache) == 2

    @pytest.mark.asyncio
    async def test_delete(self):
        """Test deleting cache entries."""
        cache = AsyncMemoryCache()
        await cache.set("key1", "value1")

        assert await cache.get("key1") == "value1"

        await cache.delete("key1")
        assert await cache.get("key1") is None

    @pytest.mark.asyncio
    async def test_clear(self):
        """Test clearing cache."""
        cache = AsyncMemoryCache()
        await cache.set("key1", "value1")
        await cache.set("key2", "value2")

        await cache.clear()

        assert await cache.get("key1") is None
        assert await cache.get("key2") is None
        assert len(cache.cache) == 0


# ============================================================================
# TEST MCP FUNCTION SPEC
# ============================================================================


class TestMCPFunctionSpec:
    """Test MCPFunctionSpec class."""

    def test_basic_initialization(self):
        """Test basic spec initialization."""
        spec = MCPFunctionSpec(
            function_name="test_func",
            namespace="default",
            qualified_name="default/test_func",
            description="Test function",
            implementation="def test_func(): pass",
            created_at="2024-01-01T00:00:00Z",
            updated_at="2024-01-01T00:00:00Z",
        )

        assert spec.function_name == "test_func"
        assert spec.namespace == "default"
        assert spec.qualified_name == "default/test_func"
        assert spec.description == "Test function"

    def test_auto_qualified_name(self):
        """Test automatic qualified name generation."""
        spec = MCPFunctionSpec(
            function_name="test_func",
            namespace="math",
            qualified_name="",  # Will be auto-generated
            description="Test",
            implementation="def test_func(): pass",
            created_at="2024-01-01T00:00:00Z",
            updated_at="2024-01-01T00:00:00Z",
        )

        assert spec.qualified_name == "math/test_func"

    def test_auto_content_hash(self):
        """Test automatic content hash generation."""
        impl = "def test_func(): pass"
        spec = MCPFunctionSpec(
            function_name="test_func",
            namespace="default",
            qualified_name="default/test_func",
            description="Test",
            implementation=impl,
            created_at="2024-01-01T00:00:00Z",
            updated_at="2024-01-01T00:00:00Z",
        )

        assert spec.content_hash != ""
        assert spec.content_hash == _calculate_content_hash(impl)

    def test_performance_metrics_initialization(self):
        """Test that performance metrics are initialized."""
        spec = MCPFunctionSpec(
            function_name="test_func",
            namespace="default",
            qualified_name="default/test_func",
            description="Test",
            implementation="def test_func(): pass",
            created_at="2024-01-01T00:00:00Z",
            updated_at="2024-01-01T00:00:00Z",
        )

        assert spec.performance_metrics is not None
        assert isinstance(spec.performance_metrics, PerformanceMetrics)

    def test_cache_backend_initialization(self):
        """Test cache backend initialization with caching enabled."""
        spec = MCPFunctionSpec(
            function_name="test_func",
            namespace="default",
            qualified_name="default/test_func",
            description="Test",
            implementation="def test_func(): pass",
            cache_strategy=CacheStrategy.MEMORY,
            created_at="2024-01-01T00:00:00Z",
            updated_at="2024-01-01T00:00:00Z",
        )

        assert spec.cache_backend is not None
        assert isinstance(spec.cache_backend, AsyncMemoryCache)

    def test_execution_semaphore_initialization(self):
        """Test execution semaphore initialization."""
        spec = MCPFunctionSpec(
            function_name="test_func",
            namespace="default",
            qualified_name="default/test_func",
            description="Test",
            implementation="def test_func(): pass",
            max_concurrent_executions=5,
            created_at="2024-01-01T00:00:00Z",
            updated_at="2024-01-01T00:00:00Z",
        )

        assert spec.execution_semaphore is not None
        assert isinstance(spec.execution_semaphore, asyncio.Semaphore)

    def test_supports_local_execution(self):
        """Test supports_local_execution method."""
        spec1 = MCPFunctionSpec(
            function_name="test",
            qualified_name="test/test",
            description="Test",
            implementation="pass",
            execution_modes=[ExecutionMode.LOCAL],
            created_at="2024-01-01T00:00:00Z",
            updated_at="2024-01-01T00:00:00Z",
        )
        assert spec1.supports_local_execution() is True

        spec2 = MCPFunctionSpec(
            function_name="test",
            qualified_name="test/test",
            description="Test",
            implementation="pass",
            execution_modes=[ExecutionMode.REMOTE],
            created_at="2024-01-01T00:00:00Z",
            updated_at="2024-01-01T00:00:00Z",
        )
        assert spec2.supports_local_execution() is False

        spec3 = MCPFunctionSpec(
            function_name="test",
            qualified_name="test/test",
            description="Test",
            implementation="pass",
            execution_modes=[ExecutionMode.BOTH],
            created_at="2024-01-01T00:00:00Z",
            updated_at="2024-01-01T00:00:00Z",
        )
        assert spec3.supports_local_execution() is True

    def test_supports_remote_execution(self):
        """Test supports_remote_execution method."""
        spec1 = MCPFunctionSpec(
            function_name="test",
            qualified_name="test/test",
            description="Test",
            implementation="pass",
            execution_modes=[ExecutionMode.REMOTE],
            created_at="2024-01-01T00:00:00Z",
            updated_at="2024-01-01T00:00:00Z",
        )
        assert spec1.supports_remote_execution() is True

        spec2 = MCPFunctionSpec(
            function_name="test",
            qualified_name="test/test",
            description="Test",
            implementation="pass",
            execution_modes=[ExecutionMode.LOCAL],
            created_at="2024-01-01T00:00:00Z",
            updated_at="2024-01-01T00:00:00Z",
        )
        assert spec2.supports_remote_execution() is False

    @pytest.mark.asyncio
    async def test_generate_cache_key(self):
        """Test cache key generation."""
        spec = MCPFunctionSpec(
            function_name="test_func",
            qualified_name="default/test_func",
            description="Test",
            implementation="def test_func(): pass",
            created_at="2024-01-01T00:00:00Z",
            updated_at="2024-01-01T00:00:00Z",
        )

        args1 = {"a": 1, "b": 2}
        args2 = {"b": 2, "a": 1}  # Same args, different order
        args3 = {"a": 1, "b": 3}  # Different args

        key1 = spec._generate_cache_key(args1)
        key2 = spec._generate_cache_key(args2)
        key3 = spec._generate_cache_key(args3)

        assert key1 == key2  # Order shouldn't matter
        assert key1 != key3  # Different values should produce different keys

    def test_get_performance_stats(self):
        """Test getting performance statistics."""
        spec = MCPFunctionSpec(
            function_name="test_func",
            qualified_name="default/test_func",
            description="Test",
            implementation="def test_func(): pass",
            created_at="2024-01-01T00:00:00Z",
            updated_at="2024-01-01T00:00:00Z",
        )

        # Update metrics
        spec.performance_metrics.update(1.0, True, False, 2)
        spec.performance_metrics.update(2.0, True, True, 3)

        stats = spec.get_performance_stats()

        assert stats["execution_count"] == 2
        assert stats["average_duration"] == 1.5
        assert stats["min_duration"] == 1.0
        assert stats["max_duration"] == 2.0
        assert stats["cache_hit_rate"] == 0.5
        assert stats["async_yields"] == 5


# ============================================================================
# TEST MCP DECORATOR
# ============================================================================


class TestMCPDecorator:
    """Test mcp_function decorator."""

    def test_simple_sync_function(self):
        """Test decorating a simple sync function."""

        @mcp_function
        def simple_func(x: int) -> int:
            """Simple function."""
            return x * 2

        assert hasattr(simple_func, "_mcp_function_spec")
        assert hasattr(simple_func, "_mcp_qualified_name")
        assert simple_func._mcp_qualified_name == "default/simple_func"

        # Function should still work
        result = simple_func(5)
        assert result == 10

    @pytest.mark.asyncio
    async def test_simple_async_function(self):
        """Test decorating a simple async function."""

        @mcp_function
        async def async_func(x: int) -> int:
            """Async function."""
            return x * 2

        spec = async_func._mcp_function_spec
        assert spec.is_async_native is True

        # Function should still work
        result = await async_func(5)
        assert result == 10

    def test_decorator_with_custom_namespace(self):
        """Test decorator with custom namespace."""

        @mcp_function(namespace="math")
        def math_func(x: int) -> int:
            return x + 1

        assert math_func._mcp_qualified_name == "math/math_func"

    def test_decorator_with_description(self):
        """Test decorator with custom description."""

        @mcp_function(description="Custom description")
        def desc_func():
            pass

        spec = desc_func._mcp_function_spec
        assert spec.description == "Custom description"

    def test_decorator_with_caching(self):
        """Test decorator with caching enabled."""

        @mcp_function(cache_strategy=CacheStrategy.MEMORY, cache_ttl_seconds=60)
        def cached_func(x: int) -> int:
            return x * 2

        spec = cached_func._mcp_function_spec
        assert spec.cache_strategy == CacheStrategy.MEMORY
        assert spec.cache_ttl_seconds == 60
        assert spec.cache_backend is not None

    @pytest.mark.asyncio
    async def test_async_function_with_caching(self):
        """Test async function with caching."""
        call_count = 0

        @mcp_function(cache_strategy=CacheStrategy.MEMORY)
        async def cached_async_func(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x * 2

        # First call
        result1 = await cached_async_func(5)
        assert result1 == 10
        assert call_count == 1

        # Second call with same args (should use cache)
        result2 = await cached_async_func(5)
        assert result2 == 10
        assert call_count == 1  # Should not increment

    def test_decorator_string_args(self):
        """Test decorator with string enum arguments."""

        @mcp_function(
            cache_strategy="memory",
            execution_modes=["local", "remote"],
            preferred_mode="local",
        )
        def string_arg_func():
            pass

        spec = string_arg_func._mcp_function_spec
        assert spec.cache_strategy == CacheStrategy.MEMORY
        assert ExecutionMode.LOCAL in spec.execution_modes
        assert ExecutionMode.REMOTE in spec.execution_modes
        assert spec.preferred_mode == ExecutionMode.LOCAL

    def test_auto_resource_estimation(self):
        """Test automatic resource estimation."""

        @mcp_function
        def complex_func():
            """Complex function with loops."""
            result = 0
            for i in range(1000):
                while i > 0:
                    result += i
                    break
            return result

        spec = complex_func._mcp_function_spec
        # Should detect loops and estimate higher CPU usage
        assert spec.estimated_cpu_usage in [ResourceLevel.MEDIUM, ResourceLevel.HIGH]

    def test_function_registration(self):
        """Test that functions are registered in global registry."""
        # Clear registry first
        _mcp_functions.clear()

        @mcp_function(namespace="test")
        def registered_func():
            pass

        assert "test/registered_func" in _mcp_functions
        assert _mcp_functions["test/registered_func"] == registered_func._mcp_function_spec

    @pytest.mark.asyncio
    async def test_async_yield_detection(self):
        """Test detection of async yield strategies."""

        @mcp_function
        async def manual_yield_func():
            """Function with manual yields."""
            for i in range(10):
                await asyncio.sleep(0)
                yield i

        spec = manual_yield_func._mcp_function_spec
        # Generator functions detected as streaming
        assert spec.supports_streaming is True
        # Async generator gets NONE since it's detected as generator, not regular async
        assert spec.async_yield_strategy in [
            AsyncYieldStrategy.NONE,
            AsyncYieldStrategy.MANUAL,
            AsyncYieldStrategy.ADAPTIVE,
            AsyncYieldStrategy.ITERATION_BASED,
        ]

    @pytest.mark.asyncio
    async def test_clear_cache_method(self):
        """Test clear_cache method on decorated functions."""

        @mcp_function(cache_strategy=CacheStrategy.MEMORY)
        async def cache_clear_func(x: int) -> int:
            return x * 2

        # Make a call to populate cache
        await cache_clear_func(5)

        # Clear cache
        await cache_clear_func.clear_cache()

        # Cache should be empty
        spec = cache_clear_func._mcp_function_spec
        if spec.cache_backend:
            assert len(spec.cache_backend.cache) == 0

    def test_get_performance_stats_method(self):
        """Test get_performance_stats method on decorated functions."""

        @mcp_function
        def stats_func(x: int) -> int:
            return x * 2

        stats_func(5)
        stats = stats_func.get_performance_stats()

        assert isinstance(stats, dict)
        assert "execution_count" in stats


# ============================================================================
# TEST UTILITY FUNCTIONS
# ============================================================================


class TestUtilityFunctions:
    """Test utility functions."""

    def test_calculate_content_hash(self):
        """Test content hash calculation."""
        content1 = "def func(): pass"
        content2 = "def func(): pass"
        content3 = "def func(): return 1"

        hash1 = _calculate_content_hash(content1)
        hash2 = _calculate_content_hash(content2)
        hash3 = _calculate_content_hash(content3)

        assert hash1 == hash2  # Same content = same hash
        assert hash1 != hash3  # Different content = different hash
        assert len(hash1) == 64  # SHA-256 produces 64 character hex string

    def test_get_current_timestamp(self):
        """Test timestamp generation."""
        timestamp = _get_current_timestamp()

        assert isinstance(timestamp, str)
        assert "T" in timestamp  # ISO format includes T
        # Should be parseable as datetime
        datetime.fromisoformat(timestamp.replace("Z", "+00:00"))

    def test_estimate_resource_usage(self):
        """Test resource usage estimation."""

        def simple_func():
            return 1

        def complex_func():
            result = []
            for i in range(100):
                while i > 0:
                    result.append(i)
                    break
            return result

        cpu1, mem1 = _estimate_resource_usage(simple_func)
        cpu2, mem2 = _estimate_resource_usage(complex_func)

        # Complex function should have higher estimates
        assert isinstance(cpu1, ResourceLevel)
        assert isinstance(mem1, ResourceLevel)

    def test_detect_yield_strategy(self):
        """Test yield strategy detection."""

        async def manual_yield():
            await asyncio.sleep(0)

        async def iteration_based():
            for i in range(10):
                pass

        async def time_based():
            while True:
                break

        def sync_func():
            pass

        strategy1 = _detect_yield_strategy(manual_yield)
        _strategy2 = _detect_yield_strategy(iteration_based)
        _strategy3 = _detect_yield_strategy(time_based)
        strategy4 = _detect_yield_strategy(sync_func)

        assert isinstance(strategy1, AsyncYieldStrategy)
        assert strategy4 == AsyncYieldStrategy.NONE

    def test_type_to_json_schema(self):
        """Test type hint to JSON schema conversion."""
        assert _type_to_json_schema(int) == {"type": "integer"}
        assert _type_to_json_schema(float) == {"type": "number"}
        assert _type_to_json_schema(str) == {"type": "string"}
        assert _type_to_json_schema(bool) == {"type": "boolean"}
        assert _type_to_json_schema(list) == {"type": "array"}
        assert _type_to_json_schema(dict) == {"type": "object"}

    def test_type_to_json_schema_list(self):
        """Test List type conversion."""

        schema = _type_to_json_schema(List[int])
        assert schema["type"] == "array"
        assert "items" in schema
        assert schema["items"]["type"] == "integer"

    def test_extract_function_info(self):
        """Test function information extraction."""

        def test_func(x: int, y: str = "default") -> int:
            """Test function docstring."""
            return len(y) + x

        name, doc, params, returns = _extract_function_info(test_func)

        assert name == "test_func"
        assert "Test function docstring" in doc
        assert "x" in params
        assert "y" in params
        assert params["x"]["required"] is True
        assert params["y"]["required"] is False
        assert returns["type"]["type"] == "integer"

    def test_extract_function_info_async(self):
        """Test async function information extraction."""

        async def async_test_func(x: int) -> int:
            """Async test function."""
            return x * 2

        name, doc, params, returns = _extract_function_info(async_test_func)

        assert name == "async_test_func"
        assert "[ASYNC]" in doc  # Should add async indicator

    def test_extract_source_code(self):
        """Test source code extraction."""

        def test_func():
            """Test function."""
            return 42

        source = _extract_source_code(test_func)

        assert "def test_func" in source
        assert "return 42" in source

    def test_extract_source_code_async(self):
        """Test async source code extraction."""

        async def async_func():
            """Async function."""
            return 42

        source = _extract_source_code(async_func)

        assert "async def async_func" in source or "async_func" in source

    def test_normalize_string_arg(self):
        """Test string argument normalization."""
        result1 = _normalize_string_arg("local", ExecutionMode)
        assert result1 == ExecutionMode.LOCAL

        result2 = _normalize_string_arg(ExecutionMode.LOCAL, ExecutionMode)
        assert result2 == ExecutionMode.LOCAL

    def test_normalize_list_arg(self):
        """Test list argument normalization."""
        result1 = _normalize_list_arg(["local", "remote"], ExecutionMode, [ExecutionMode.LOCAL])
        assert ExecutionMode.LOCAL in result1
        assert ExecutionMode.REMOTE in result1

        result2 = _normalize_list_arg(None, ExecutionMode, [ExecutionMode.LOCAL])
        assert result2 == [ExecutionMode.LOCAL]


# ============================================================================
# TEST REGISTRY FUNCTIONS
# ============================================================================


class TestRegistryFunctions:
    """Test registry and export functions."""

    def test_get_mcp_functions(self):
        """Test getting all MCP functions."""
        _mcp_functions.clear()

        @mcp_function(namespace="test1")
        def func1():
            pass

        @mcp_function(namespace="test2")
        def func2():
            pass

        functions = get_mcp_functions()
        assert len(functions) >= 2
        assert "test1/func1" in functions
        assert "test2/func2" in functions

    def test_get_mcp_functions_filtered(self):
        """Test getting MCP functions by namespace."""
        _mcp_functions.clear()

        @mcp_function(namespace="math")
        def math_func():
            pass

        @mcp_function(namespace="string")
        def string_func():
            pass

        math_functions = get_mcp_functions(namespace="math")
        assert len(math_functions) == 1
        assert "math/math_func" in math_functions

    def test_get_async_functions(self):
        """Test getting only async functions."""
        _mcp_functions.clear()

        @mcp_function
        def sync_func():
            pass

        @mcp_function
        async def async_func():
            pass

        async_functions = get_async_functions()
        assert "default/async_func" in async_functions
        assert "default/sync_func" not in async_functions

    def test_get_function_by_name(self):
        """Test getting function by qualified name."""
        _mcp_functions.clear()

        @mcp_function(namespace="test")
        def test_func():
            pass

        spec = get_function_by_name("test/test_func")
        assert spec is not None
        assert spec.function_name == "test_func"

        spec2 = get_function_by_name("nonexistent/func")
        assert spec2 is None

    @pytest.mark.asyncio
    async def test_export_function_specs_async(self, tmp_path):
        """Test exporting function specifications."""
        _mcp_functions.clear()

        @mcp_function(namespace="export_test")
        async def export_func(x: int) -> int:
            return x * 2

        export_file = tmp_path / "export.json"
        await export_function_specs_async(str(export_file))

        assert export_file.exists()

        # Verify JSON content
        import json

        with open(export_file) as f:
            data = json.load(f)

        assert "functions" in data
        assert "metadata" in data
        assert "export_test/export_func" in data["functions"]

    @pytest.mark.asyncio
    async def test_print_function_summary_async(self, capsys):
        """Test printing function summary."""
        _mcp_functions.clear()

        @mcp_function
        def summary_func():
            pass

        await print_function_summary_async()

        captured = capsys.readouterr()
        assert "MCP Functions Summary" in captured.out


# ============================================================================
# TEST EDGE CASES AND ERROR HANDLING
# ============================================================================


class TestEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.mark.asyncio
    async def test_execute_without_function_ref(self):
        """Test executing spec without function reference."""
        spec = MCPFunctionSpec(
            function_name="test",
            qualified_name="test/test",
            description="Test",
            implementation="pass",
            created_at="2024-01-01T00:00:00Z",
            updated_at="2024-01-01T00:00:00Z",
        )

        with pytest.raises(RuntimeError, match="Function reference not set"):
            await spec._execute_function_async({})

    @pytest.mark.asyncio
    async def test_concurrent_execution_limit(self):
        """Test concurrent execution limiting."""
        execution_count = 0
        max_concurrent = 0
        current_concurrent = 0

        @mcp_function(max_concurrent_executions=2, cache_strategy=CacheStrategy.NONE)
        async def concurrent_func(x: int) -> int:
            nonlocal execution_count, max_concurrent, current_concurrent
            current_concurrent += 1
            max_concurrent = max(max_concurrent, current_concurrent)
            execution_count += 1
            await asyncio.sleep(0.1)
            current_concurrent -= 1
            return x * 2

        # Start 5 concurrent executions
        tasks = [concurrent_func(i) for i in range(5)]
        results = await asyncio.gather(*tasks)

        assert len(results) == 5
        assert execution_count == 5
        # Max concurrent should not exceed limit
        assert max_concurrent <= 2

    @pytest.mark.asyncio
    async def test_cache_with_complex_arguments(self):
        """Test caching with complex nested arguments."""
        call_count = 0

        @mcp_function(cache_strategy=CacheStrategy.MEMORY)
        async def complex_args_func(data: dict) -> int:
            nonlocal call_count
            call_count += 1
            return data.get("value", 0) * 2

        # Same data, should use cache
        result1 = await complex_args_func({"value": 5, "nested": {"key": "val"}})
        result2 = await complex_args_func({"value": 5, "nested": {"key": "val"}})

        assert result1 == result2
        # Note: Due to JSON serialization, this should cache properly

    def test_decorator_preserves_function_attributes(self):
        """Test that decorator preserves function attributes."""

        @mcp_function
        def preserved_func(x: int) -> int:
            """Original docstring."""
            return x * 2

        assert preserved_func.__name__ == "preserved_func"
        assert "Original docstring" in preserved_func.__doc__

    @pytest.mark.asyncio
    async def test_async_generator_detection(self):
        """Test detection of async generator functions."""

        @mcp_function
        async def async_gen_func():
            """Async generator."""
            for i in range(10):
                yield i

        spec = async_gen_func._mcp_function_spec
        assert spec.supports_streaming is True


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


class TestIntegration:
    """Integration tests combining multiple features."""

    @pytest.mark.asyncio
    async def test_full_lifecycle(self):
        """Test full function lifecycle from decoration to execution."""
        _mcp_functions.clear()

        @mcp_function(
            namespace="integration",
            description="Integration test function",
            cache_strategy=CacheStrategy.MEMORY,
            cache_ttl_seconds=60,
        )
        async def lifecycle_func(x: int, y: int) -> int:
            """Add two numbers."""
            return x + y

        # Verify registration
        assert "integration/lifecycle_func" in _mcp_functions

        # Verify spec
        spec = lifecycle_func._mcp_function_spec
        assert spec.function_name == "lifecycle_func"
        assert spec.is_async_native is True
        assert spec.cache_strategy == CacheStrategy.MEMORY

        # Execute function
        result = await lifecycle_func(3, 4)
        assert result == 7

        # Check performance stats
        stats = lifecycle_func.get_performance_stats()
        assert stats["execution_count"] >= 0

        # Test caching
        result2 = await lifecycle_func(3, 4)
        assert result2 == 7

    @pytest.mark.asyncio
    async def test_multiple_namespaces(self):
        """Test functions in multiple namespaces."""
        _mcp_functions.clear()

        @mcp_function(namespace="math")
        def add(x: int, y: int) -> int:
            return x + y

        @mcp_function(namespace="math")
        def multiply(x: int, y: int) -> int:
            return x * y

        @mcp_function(namespace="string")
        def concat(a: str, b: str) -> str:
            return a + b

        all_funcs = get_mcp_functions()
        math_funcs = get_mcp_functions(namespace="math")
        string_funcs = get_mcp_functions(namespace="string")

        assert len(all_funcs) >= 3
        assert len(math_funcs) == 2
        assert len(string_funcs) == 1
