#!/usr/bin/env python3
# chuk_mcp_math/mcp_decorator.py
"""
Unified MCP Function Decorator - Async Native Optimized
"""

import inspect
import json
import textwrap
import hashlib
import time
import asyncio
from datetime import datetime, timezone
from typing import (
    Dict,
    Any,
    List,
    Optional,
    Callable,
    get_type_hints,
    Union,
    TypeVar,
    Coroutine,
)
from functools import wraps
from enum import Enum
from dataclasses import dataclass
from collections import defaultdict

# Import Pydantic with fallback
from .mcp_pydantic_base import McpPydanticBase, Field

# Global registries
_mcp_functions: dict[str, Any] = {}
_mcp_workflows: dict[str, Any] = {}
_performance_metrics: defaultdict[str, list[float]] = defaultdict(list)
_async_semaphore = asyncio.Semaphore(100)  # Global async concurrency limit

# Type aliases
T = TypeVar("T")


class ExecutionMode(str, Enum):
    """Supported execution modes for MCP functions."""

    LOCAL = "local"
    REMOTE = "remote"
    BOTH = "both"
    AUTO = "auto"


class CacheStrategy(str, Enum):
    """Cache strategies (async-optimized)."""

    NONE = "none"
    MEMORY = "memory"
    FILE = "file"
    HYBRID = "hybrid"
    ASYNC_LRU = "async_lru"  # New: async-optimized LRU cache


class ResourceLevel(str, Enum):
    """Resource usage levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTREME = "extreme"


class StreamingMode(str, Enum):
    """Streaming modes."""

    NONE = "none"
    CHUNKED = "chunked"
    REAL_TIME = "real_time"
    BACKPRESSURE = "backpressure"


class AsyncYieldStrategy(str, Enum):
    """Async yielding strategies for long-running operations."""

    NONE = "none"
    TIME_BASED = "time_based"
    ITERATION_BASED = "iteration_based"
    ADAPTIVE = "adaptive"
    MANUAL = "manual"


@dataclass
class PerformanceMetrics:
    """Performance tracking with async-specific metrics."""

    execution_count: int = 0
    total_duration: float = 0.0
    average_duration: float = 0.0
    min_duration: float = float("inf")
    max_duration: float = 0.0
    error_count: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    async_yields: int = 0  # New: track async yields
    concurrent_executions: int = 0  # New: track concurrent runs
    last_execution: Optional[datetime] = None

    def update(
        self,
        duration: float,
        success: bool = True,
        cache_hit: bool = False,
        yields: int = 0,
    ):
        self.execution_count += 1
        if success:
            self.total_duration += duration
            self.average_duration = self.total_duration / self.execution_count
            self.min_duration = min(self.min_duration, duration)
            self.max_duration = max(self.max_duration, duration)
        else:
            self.error_count += 1

        if cache_hit:
            self.cache_hits += 1
        else:
            self.cache_misses += 1

        self.async_yields += yields
        self.last_execution = datetime.now(timezone.utc)


# Enhanced async-native cache implementation
class AsyncMemoryCache:
    """Async-optimized memory cache for MCP functions."""

    def __init__(self, max_size: int = 1000):
        self.cache: dict[str, Any] = {}
        self.expiry: dict[str, float] = {}
        self.max_size = max_size
        self.access_times: dict[str, float] = {}
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> Optional[Any]:
        async with self._lock:
            current_time = time.time()

            if key in self.expiry and self.expiry[key] < current_time:
                await self._delete_unlocked(key)
                return None

            if key in self.cache:
                self.access_times[key] = current_time
                return self.cache[key]
            return None

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        async with self._lock:
            current_time = time.time()

            # LRU eviction if needed
            if len(self.cache) >= self.max_size:
                if self.access_times:
                    oldest_key = min(
                        self.access_times.keys(), key=lambda k: self.access_times[k]
                    )
                    await self._delete_unlocked(oldest_key)

            self.cache[key] = value
            self.access_times[key] = current_time

            if ttl:
                self.expiry[key] = current_time + ttl

    async def _delete_unlocked(self, key: str) -> None:
        """Delete without acquiring lock (internal use)."""
        self.cache.pop(key, None)
        self.access_times.pop(key, None)
        self.expiry.pop(key, None)

    async def delete(self, key: str) -> None:
        async with self._lock:
            await self._delete_unlocked(key)

    async def clear(self) -> None:
        async with self._lock:
            self.cache.clear()
            self.access_times.clear()
            self.expiry.clear()


class MCPFunctionSpec(McpPydanticBase):
    """Enhanced MCP function specification optimized for async native functions."""

    # Add model configuration for Pydantic v2
    model_config = {"arbitrary_types_allowed": True, "extra": "allow"}

    # Core identification (required)
    function_name: str = Field(description="Function name")
    namespace: str = Field(default="default", description="Function namespace")
    qualified_name: str = Field(description="Fully qualified name")

    # Documentation (smart defaults)
    description: str = Field(description="Function description")
    version: str = Field(default="1.0.0", description="Semantic version")
    category: str = Field(default="general", description="Category")

    # Function signature (auto-detected)
    parameters: Dict[str, Any] = Field(
        default_factory=dict, description="Parameter specifications"
    )
    returns: Dict[str, Any] = Field(
        default_factory=dict, description="Return specification"
    )

    # Implementation (auto-extracted)
    implementation: str = Field(description="Complete source code")
    content_hash: str = Field(default="", description="SHA-256 hash of implementation")

    # Execution (smart defaults)
    execution_modes: List[ExecutionMode] = Field(
        default_factory=lambda: [ExecutionMode.LOCAL], description="Execution modes"
    )
    preferred_mode: ExecutionMode = Field(
        default=ExecutionMode.LOCAL, description="Preferred execution mode"
    )

    # Async-specific features
    is_async_native: bool = Field(default=False, description="Is async native function")
    async_yield_strategy: AsyncYieldStrategy = Field(
        default=AsyncYieldStrategy.NONE, description="Async yielding strategy"
    )
    max_concurrent_executions: int = Field(
        default=10, description="Max concurrent executions"
    )

    # Dependencies (optional)
    dependencies: List[str] = Field(
        default_factory=list, description="Required packages"
    )

    # Usage examples (optional)
    examples: List[Dict[str, Any]] = Field(
        default_factory=list, description="Usage examples"
    )

    # Enhanced features (optional, lazy defaults)
    cache_strategy: CacheStrategy = Field(
        default=CacheStrategy.NONE, description="Caching strategy"
    )
    cache_ttl_seconds: Optional[int] = Field(default=None, description="Cache TTL")
    estimated_cpu_usage: ResourceLevel = Field(
        default=ResourceLevel.LOW, description="CPU usage estimate"
    )
    estimated_memory_usage: ResourceLevel = Field(
        default=ResourceLevel.LOW, description="Memory usage estimate"
    )
    supports_streaming: bool = Field(default=False, description="Streaming support")
    streaming_mode: StreamingMode = Field(
        default=StreamingMode.NONE, description="Streaming mode"
    )
    timeout_seconds: Optional[float] = Field(
        default=None, description="Execution timeout"
    )

    # Security (safe defaults)
    trusted: bool = Field(default=False, description="Trusted source")
    safe_for_local: bool = Field(default=True, description="Safe for local execution")
    requires_sandbox: bool = Field(default=True, description="Requires sandbox")

    # Workflow (enabled by default)
    workflow_compatible: bool = Field(default=True, description="Workflow compatible")

    # Metadata (extensible)
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )
    created_at: str = Field(description="Creation timestamp")
    updated_at: str = Field(description="Update timestamp")

    # Runtime (excluded from serialization) - FIXED: No leading underscores, proper exclusion
    function_ref: Optional[Callable] = Field(default=None, exclude=True)
    performance_metrics: Optional[PerformanceMetrics] = Field(
        default=None, exclude=True
    )
    cache_backend: Optional[AsyncMemoryCache] = Field(default=None, exclude=True)
    execution_semaphore: Optional[asyncio.Semaphore] = Field(default=None, exclude=True)

    def model_post_init(self, __context):
        """Post-initialization setup with async-native optimizations."""
        if not self.qualified_name:
            self.qualified_name = f"{self.namespace}/{self.function_name}"

        if not self.content_hash:
            self.content_hash = _calculate_content_hash(self.implementation)

        # Initialize performance metrics
        if not self.performance_metrics:
            self.performance_metrics = PerformanceMetrics()

        # Initialize async-optimized cache if needed
        if self.cache_strategy != CacheStrategy.NONE and not self.cache_backend:
            self.cache_backend = AsyncMemoryCache()

        # Initialize execution semaphore for concurrency control
        if not self.execution_semaphore:
            self.execution_semaphore = asyncio.Semaphore(self.max_concurrent_executions)

    async def execute_with_caching(self, arguments: Dict[str, Any]) -> Any:
        """Execute function with optional caching and async optimizations."""
        if self.cache_strategy == CacheStrategy.NONE or not self.cache_backend:
            return await self._execute_function_async(arguments)

        # Generate cache key
        cache_key = self._generate_cache_key(arguments)

        # Try cache first
        cached_result = await self.cache_backend.get(cache_key)
        if cached_result is not None:
            if self.performance_metrics:
                self.performance_metrics.update(0, True, cache_hit=True)
            return cached_result

        # Execute function with concurrency control
        semaphore = self.execution_semaphore or asyncio.Semaphore(1)
        async with semaphore:
            start_time = time.time()
            yields = 0
            try:
                result = await self._execute_function_async(arguments)
                duration = time.time() - start_time

                # Cache result
                await self.cache_backend.set(cache_key, result, self.cache_ttl_seconds)
                if self.performance_metrics:
                    self.performance_metrics.update(
                        duration, True, cache_hit=False, yields=yields
                    )
                return result

            except Exception:
                duration = time.time() - start_time
                if self.performance_metrics:
                    self.performance_metrics.update(
                        duration, False, cache_hit=False, yields=yields
                    )
                raise

    async def _execute_function_async(self, arguments: Dict[str, Any]) -> Any:
        """Execute the actual function with async optimizations."""
        if not self.function_ref:
            raise RuntimeError(f"Function reference not set for {self.qualified_name}")

        if self.is_async_native:
            # Execute async function
            if inspect.iscoroutinefunction(self.function_ref):
                return await self.function_ref(**arguments)
            else:
                # Sync function wrapped in async context
                return self.function_ref(**arguments)
        else:
            # Legacy sync function
            return self.function_ref(**arguments)

    def _generate_cache_key(self, arguments: Dict[str, Any]) -> str:
        """Generate cache key from arguments."""
        sorted_args = json.dumps(arguments, sort_keys=True, default=str)
        key_data = f"{self.qualified_name}:{sorted_args}"
        return hashlib.sha256(key_data.encode()).hexdigest()

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics with async-specific metrics."""
        if not self.performance_metrics:
            return {}

        metrics = self.performance_metrics
        cache_hit_rate = 0.0
        if metrics.cache_hits + metrics.cache_misses > 0:
            cache_hit_rate = metrics.cache_hits / (
                metrics.cache_hits + metrics.cache_misses
            )

        return {
            "execution_count": metrics.execution_count,
            "average_duration": metrics.average_duration,
            "min_duration": metrics.min_duration
            if metrics.min_duration != float("inf")
            else 0,
            "max_duration": metrics.max_duration,
            "error_rate": metrics.error_count / max(metrics.execution_count, 1),
            "cache_hit_rate": cache_hit_rate,
            "async_yields": metrics.async_yields,
            "concurrent_executions": metrics.concurrent_executions,
            "is_async_native": self.is_async_native,
            "yield_strategy": self.async_yield_strategy.value,
            "last_execution": metrics.last_execution.isoformat()
            if metrics.last_execution
            else None,
        }

    def supports_local_execution(self) -> bool:
        """Check if function can be executed locally."""
        return (
            ExecutionMode.LOCAL in self.execution_modes
            or ExecutionMode.BOTH in self.execution_modes
        )

    def supports_remote_execution(self) -> bool:
        """Check if function can be executed remotely."""
        return (
            ExecutionMode.REMOTE in self.execution_modes
            or ExecutionMode.BOTH in self.execution_modes
        )


# Utility functions (enhanced for async)
def _calculate_content_hash(content: str) -> str:
    """Calculate SHA-256 hash."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _get_current_timestamp() -> str:
    """Get current timestamp."""
    return datetime.now(timezone.utc).isoformat()


def _estimate_resource_usage(func: Callable) -> tuple:
    """Auto-estimate resource usage from function code with async considerations."""
    try:
        source = inspect.getsource(func)
    except OSError:
        return ResourceLevel.LOW, ResourceLevel.LOW

    # Enhanced heuristics including async patterns
    cpu_indicators = [
        "for ",
        "while ",
        "numpy",
        "pandas",
        "math.",
        "complex",
        "await ",
        "asyncio",
    ]
    memory_indicators = [
        "list(",
        "dict(",
        "array",
        "DataFrame",
        "load",
        "read",
        "cache",
    ]

    cpu_score = sum(source.count(indicator) for indicator in cpu_indicators)
    memory_score = sum(source.count(indicator) for indicator in memory_indicators)

    # Async functions get slight CPU boost due to overhead
    if inspect.iscoroutinefunction(func):
        cpu_score += 1

    cpu_level = ResourceLevel.LOW
    if cpu_score > 5:
        cpu_level = ResourceLevel.HIGH
    elif cpu_score > 2:
        cpu_level = ResourceLevel.MEDIUM

    memory_level = ResourceLevel.LOW
    if memory_score > 3:
        memory_level = ResourceLevel.HIGH
    elif memory_score > 1:
        memory_level = ResourceLevel.MEDIUM

    return cpu_level, memory_level


def _detect_yield_strategy(func: Callable) -> AsyncYieldStrategy:
    """Auto-detect optimal yield strategy for async functions."""
    if not inspect.iscoroutinefunction(func):
        return AsyncYieldStrategy.NONE

    try:
        source = inspect.getsource(func)
    except OSError:
        return AsyncYieldStrategy.ADAPTIVE

    # Look for yield patterns
    if "asyncio.sleep(0)" in source:
        return AsyncYieldStrategy.MANUAL
    elif "for " in source and "range(" in source:
        return AsyncYieldStrategy.ITERATION_BASED
    elif "while " in source:
        return AsyncYieldStrategy.TIME_BASED
    else:
        return AsyncYieldStrategy.ADAPTIVE


def _type_to_json_schema(type_hint) -> Dict[str, Any]:
    """Convert type hint to JSON Schema (enhanced for async types)."""
    if type_hint is int:
        return {"type": "integer"}
    elif type_hint is float:
        return {"type": "number"}
    elif type_hint is str:
        return {"type": "string"}
    elif type_hint is bool:
        return {"type": "boolean"}
    elif type_hint is list:
        return {"type": "array"}
    elif type_hint is dict:
        return {"type": "object"}

    # Handle typing module types
    origin = getattr(type_hint, "__origin__", None)
    args = getattr(type_hint, "__args__", ())

    # Handle Coroutine types
    if origin is Coroutine or (
        hasattr(type_hint, "__name__") and "Coroutine" in str(type_hint)
    ):
        if args:
            return _type_to_json_schema(args[-1])  # Return type
        return {"type": "object", "description": "Coroutine result"}

    if origin is list or (hasattr(type_hint, "__name__") and "List" in str(type_hint)):
        schema: Dict[str, Any] = {"type": "array"}
        if args:
            schema["items"] = _type_to_json_schema(args[0])
        return schema
    elif origin is Union:
        non_none_args = [arg for arg in args if arg is not type(None)]
        if len(non_none_args) == 1:
            schema = _type_to_json_schema(non_none_args[0])
            return {"anyOf": [schema, {"type": "null"}]}
        else:
            return {"anyOf": [_type_to_json_schema(arg) for arg in args]}

    return {"type": "object", "description": f"Python type: {str(type_hint)}"}


def _extract_function_info(func: Callable) -> tuple:
    """Extract function information with async-specific enhancements."""
    func_name = func.__name__
    func_doc = func.__doc__ or f"Function: {func_name}"

    # Add async indicator to description
    if inspect.iscoroutinefunction(func):
        func_doc = f"[ASYNC] {func_doc}"

    try:
        type_hints = get_type_hints(func)
    except Exception:
        type_hints = {}

    sig = inspect.signature(func)
    parameters = {}

    for param_name, param in sig.parameters.items():
        param_type = type_hints.get(param_name, Any)
        param_spec = {
            "type": _type_to_json_schema(param_type),
            "required": param.default == param.empty,
            "description": f"Parameter: {param_name}",
            "default": param.default if param.default != param.empty else None,
        }
        parameters[param_name] = param_spec

    return_type = type_hints.get("return", Any)
    returns = {
        "type": _type_to_json_schema(return_type),
        "description": f"Return value of {func_name}",
    }

    return func_name, func_doc, parameters, returns


def _extract_source_code(func: Callable) -> str:
    """Extract function source code with async-specific handling."""
    try:
        source = inspect.getsource(func)
        # Clean up the source code
        lines = source.split("\n")

        # Find function start (handle both def and async def)
        func_start = 0
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith("def ") or stripped.startswith("async def "):
                func_start = i
                break

        # Get function lines and dedent
        func_lines = lines[func_start:]
        if func_lines:
            return textwrap.dedent("\n".join(func_lines))

        return source
    except Exception:
        async_prefix = "async " if inspect.iscoroutinefunction(func) else ""
        return f'''{async_prefix}def {func.__name__}():
    """Source code not available"""
    pass
'''


# Enhanced normalization functions
def _normalize_string_arg(value: Union[str, Enum], enum_class: type) -> Enum:
    """Normalize string argument to enum."""
    if isinstance(value, str):
        return enum_class(value)
    return value


def _normalize_list_arg(
    value: Union[List[Any], None], enum_class: type, default: List[Enum]
) -> List[Enum]:
    """Normalize list of string/enum arguments."""
    if value is None:
        return default

    result: List[Enum] = []
    for item in value:
        if isinstance(item, str):
            result.append(enum_class(item))
        else:
            result.append(item)
    return result


# The Enhanced Async-Native Decorator
def mcp_function(
    # Can be called with no arguments for maximum laziness
    func: Optional[Callable] = None,
    *,
    # Core parameters (smart defaults)
    description: str = "",
    version: str = "1.0.0",
    category: str = "general",
    namespace: str = "default",
    # Execution (simple defaults)
    execution_modes: Optional[List[Union[ExecutionMode, str]]] = None,
    preferred_mode: Union[ExecutionMode, str] = ExecutionMode.LOCAL,
    # Async-specific parameters
    async_yield_strategy: Union[AsyncYieldStrategy, str] = AsyncYieldStrategy.ADAPTIVE,
    max_concurrent_executions: int = 10,
    # Dependencies (optional)
    dependencies: Optional[List[str]] = None,
    # Examples (optional)
    examples: Optional[List[Dict[str, Any]]] = None,
    # Enhanced features (lazy defaults - only used when specified)
    cache_strategy: Union[CacheStrategy, str] = CacheStrategy.NONE,
    cache_ttl_seconds: Optional[int] = None,
    estimated_cpu_usage: Optional[Union[ResourceLevel, str]] = None,
    estimated_memory_usage: Optional[Union[ResourceLevel, str]] = None,
    supports_streaming: bool = False,
    streaming_mode: Union[StreamingMode, str] = StreamingMode.NONE,
    timeout_seconds: Optional[float] = None,
    # Security (safe defaults)
    trusted: bool = False,
    safe_for_local: bool = True,
    requires_sandbox: bool = True,
    # Workflow (enabled by default)
    workflow_compatible: bool = True,
    # Metadata (extensible)
    metadata: Optional[Dict[str, Any]] = None,
):
    """
    Enhanced async-native MCP function decorator.

    Automatically detects async functions and optimizes for async execution.
    """

    def decorator(f: Callable) -> Callable:
        func_name, func_doc, parameters, returns = _extract_function_info(f)
        implementation = _extract_source_code(f)

        # Detect if function is async native
        is_async = inspect.iscoroutinefunction(f)

        # Smart defaults and auto-detection
        final_description = description or func_doc.strip()

        # Auto-detect streaming and yielding
        is_generator = inspect.isgeneratorfunction(f)
        is_async_generator = inspect.isasyncgenfunction(f)
        actual_streaming = supports_streaming or is_generator or is_async_generator

        # Auto-detect yield strategy for async functions
        if is_async:
            detected_yield_strategy = _detect_yield_strategy(f)
            final_yield_strategy = _normalize_string_arg(
                async_yield_strategy, AsyncYieldStrategy
            )
            if final_yield_strategy == AsyncYieldStrategy.ADAPTIVE:
                final_yield_strategy = detected_yield_strategy
        else:
            final_yield_strategy = AsyncYieldStrategy.NONE

        # Auto-estimate resource usage if not specified
        if estimated_cpu_usage is None or estimated_memory_usage is None:
            cpu_est, mem_est = _estimate_resource_usage(f)
            final_cpu_usage = (
                _normalize_string_arg(estimated_cpu_usage, ResourceLevel)
                if estimated_cpu_usage
                else cpu_est
            )
            final_memory_usage = (
                _normalize_string_arg(estimated_memory_usage, ResourceLevel)
                if estimated_memory_usage
                else mem_est
            )
        else:
            final_cpu_usage = _normalize_string_arg(estimated_cpu_usage, ResourceLevel)
            final_memory_usage = _normalize_string_arg(
                estimated_memory_usage, ResourceLevel
            )

        # Normalize execution modes
        final_execution_modes = _normalize_list_arg(
            execution_modes, ExecutionMode, [ExecutionMode.LOCAL]
        )
        final_preferred_mode = _normalize_string_arg(preferred_mode, ExecutionMode)

        # Ensure preferred mode is in execution modes
        if final_preferred_mode not in final_execution_modes:
            final_preferred_mode = final_execution_modes[0]

        # Normalize other enums
        final_cache_strategy = _normalize_string_arg(cache_strategy, CacheStrategy)
        final_streaming_mode = (
            _normalize_string_arg(streaming_mode, StreamingMode)
            if actual_streaming
            else StreamingMode.NONE
        )

        current_time = _get_current_timestamp()
        qualified_name = f"{namespace}/{func_name}"

        # Create function spec with async enhancements
        function_spec = MCPFunctionSpec(  # type: ignore[arg-type]
            function_name=func_name,
            namespace=namespace,
            qualified_name=qualified_name,
            description=final_description,
            version=version,
            category=category,
            parameters=parameters,
            returns=returns,
            implementation=implementation,
            execution_modes=final_execution_modes,  # type: ignore[arg-type]
            preferred_mode=final_preferred_mode,  # type: ignore[arg-type]
            is_async_native=is_async,
            async_yield_strategy=final_yield_strategy,  # type: ignore[arg-type]
            max_concurrent_executions=max_concurrent_executions,
            dependencies=dependencies or [],
            examples=examples or [],
            cache_strategy=final_cache_strategy,  # type: ignore[arg-type]
            cache_ttl_seconds=cache_ttl_seconds,
            estimated_cpu_usage=final_cpu_usage,  # type: ignore[arg-type]
            estimated_memory_usage=final_memory_usage,  # type: ignore[arg-type]
            supports_streaming=actual_streaming,
            streaming_mode=final_streaming_mode,  # type: ignore[arg-type]
            timeout_seconds=timeout_seconds,
            trusted=trusted,
            safe_for_local=safe_for_local,
            requires_sandbox=requires_sandbox,
            workflow_compatible=workflow_compatible,
            metadata=metadata or {},
            created_at=current_time,
            updated_at=current_time,
            function_ref=f,
        )

        # Register function
        _mcp_functions[qualified_name] = function_spec

        # Create optimized wrapper for async functions
        if is_async:

            @wraps(f)
            async def async_wrapper(*args, **kwargs):
                # Convert positional args to kwargs if needed
                if function_spec.cache_strategy != CacheStrategy.NONE:
                    sig = inspect.signature(f)
                    bound_args = sig.bind(*args, **kwargs)
                    bound_args.apply_defaults()
                    full_kwargs = dict(bound_args.arguments)
                    return await function_spec.execute_with_caching(full_kwargs)
                else:
                    # Direct async execution
                    async with function_spec.execution_semaphore:
                        function_spec.performance_metrics.concurrent_executions += 1
                        try:
                            return await f(*args, **kwargs)
                        finally:
                            function_spec.performance_metrics.concurrent_executions -= 1

            wrapper = async_wrapper
        else:

            @wraps(f)
            def sync_wrapper(*args, **kwargs):
                # For sync functions, execute directly without async caching
                # Since the original function is sync, just call it directly
                return f(*args, **kwargs)

            wrapper = sync_wrapper

        # Attach MCP attributes
        wrapper._mcp_function_spec = function_spec  # type: ignore[attr-defined]
        wrapper._mcp_qualified_name = qualified_name  # type: ignore[attr-defined]
        wrapper.get_performance_stats = function_spec.get_performance_stats  # type: ignore[attr-defined]

        async def clear_cache():
            if function_spec.cache_backend:
                await function_spec.cache_backend.clear()

        wrapper.clear_cache = clear_cache  # type: ignore[attr-defined]

        return wrapper

    # Support both @mcp_function and @mcp_function() syntax
    if func is None:
        return decorator
    else:
        return decorator(func)


# Enhanced utility functions for async
def get_mcp_functions(namespace: Optional[str] = None) -> Dict[str, MCPFunctionSpec]:
    """Get registered MCP functions with optional namespace filtering."""
    if namespace:
        return {
            name: spec
            for name, spec in _mcp_functions.items()
            if spec.namespace == namespace
        }
    return _mcp_functions.copy()


def get_async_functions() -> Dict[str, MCPFunctionSpec]:
    """Get only async native functions."""
    return {name: spec for name, spec in _mcp_functions.items() if spec.is_async_native}


def get_function_by_name(qualified_name: str) -> Optional[MCPFunctionSpec]:
    """Get function by qualified name."""
    return _mcp_functions.get(qualified_name)


async def export_function_specs_async(filename: str, namespace: Optional[str] = None):
    """Export function specifications to JSON file (async version)."""
    functions = get_mcp_functions(namespace)

    # Gather performance stats asynchronously
    stats = {}
    for name, spec in functions.items():
        if spec.is_async_native:
            stats[name] = spec.get_performance_stats()

    export_data = {
        "mcp_version": "2024-11-05",
        "async_native": True,
        "functions": {name: spec.model_dump() for name, spec in functions.items()},
        "performance_stats": stats,
        "metadata": {
            "export_timestamp": _get_current_timestamp(),
            "total_functions": len(functions),
            "async_functions": len(
                [s for s in functions.values() if s.is_async_native]
            ),
            "namespace": namespace,
        },
    }

    # Async file writing would require aiofiles, use sync for now
    with open(filename, "w") as f:
        json.dump(export_data, f, indent=2, default=str)


async def print_function_summary_async():
    """Print summary of registered functions (async version)."""
    functions = get_mcp_functions()

    if not functions:
        print("No MCP functions registered.")
        return

    async_funcs = [spec for spec in functions.values() if spec.is_async_native]
    local_funcs = [
        spec for spec in functions.values() if spec.supports_local_execution()
    ]
    remote_funcs = [
        spec for spec in functions.values() if spec.supports_remote_execution()
    ]
    cached_funcs = [
        spec for spec in functions.values() if spec.cache_strategy != CacheStrategy.NONE
    ]
    streaming_funcs = [spec for spec in functions.values() if spec.supports_streaming]

    print("🚀 MCP Functions Summary (Async Native Optimized)")
    print("=" * 50)
    print(f"Total Functions: {len(functions)}")
    print(f"🚀 Async Native: {len(async_funcs)}")
    print(f"📦 Local Executable: {len(local_funcs)}")
    print(f"🛠️  Remote Callable: {len(remote_funcs)}")
    print(f"💾 Cached Functions: {len(cached_funcs)}")
    print(f"🌊 Streaming Functions: {len(streaming_funcs)}")
    print()

    # Group by namespace
    namespaces = {}
    for name, spec in functions.items():
        if spec.namespace not in namespaces:
            namespaces[spec.namespace] = []
        namespaces[spec.namespace].append(spec)

    for namespace, specs in namespaces.items():
        async_count = len([s for s in specs if s.is_async_native])
        print(f"📁 {namespace} ({async_count}/{len(specs)} async native):")
        for spec in sorted(specs, key=lambda s: s.function_name):
            modes = "/".join([mode.value for mode in spec.execution_modes])
            preferred = f"({spec.preferred_mode.value})"
            features = ""
            if spec.is_async_native:
                features += " 🚀"
            if spec.cache_strategy != CacheStrategy.NONE:
                features += " 💾"
            if spec.supports_streaming:
                features += " 🌊"

            print(
                f"   • {spec.function_name} v{spec.version} - {modes} {preferred}{features}"
            )
            if spec.description:
                desc = spec.description.replace("[ASYNC] ", "")[:60] + "..."
                print(f"     {desc}")


# Export the enhanced components
__all__ = [
    "mcp_function",
    "MCPFunctionSpec",
    "ExecutionMode",
    "CacheStrategy",
    "ResourceLevel",
    "StreamingMode",
    "AsyncYieldStrategy",
    "get_mcp_functions",
    "get_async_functions",
    "get_function_by_name",
    "export_function_specs_async",
    "print_function_summary_async",
]

# Legacy compatibility
def export_function_specs(filename, namespace=None):
    return asyncio.run(
    export_function_specs_async(filename, namespace)
)
def print_function_summary():
    return asyncio.run(print_function_summary_async())
