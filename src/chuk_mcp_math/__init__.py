#!/usr/bin/env python3
# chuk_mcp_math/__init__.py
"""
Chuk MCP Functions - Comprehensive Mathematical Function Library for AI Models (Async Native)

A modular collection of MCP-compatible functions designed specifically for AI model execution.
Includes mathematical operations, data processing, utilities, and more with async native support.

Key Features:
- Full MCP compliance with resource and tool specifications
- Async native execution for optimal performance
- Smart caching and performance optimization
- Comprehensive error handling and validation
- Local and remote execution support
- Streaming capabilities where appropriate
- Rich documentation and examples for AI understanding

Mathematical Domains:
- arithmetic: Basic operations (reorganized structure) - ASYNC NATIVE ✅
- number_theory: Prime numbers, divisibility, sequences, cryptographic functions - ASYNC NATIVE ✅

Modules:
- math: Complete mathematical operations library (async native)
- data: Data processing and manipulation functions
- text: String and text processing utilities
- datetime: Date and time operations
- file: File system operations
- network: Network and API utilities
- conversion: Unit and format conversions
"""

from typing import Dict, List, Any
import logging
import math
import asyncio

# Import core MCP functionality
from .mcp_pydantic_base import McpPydanticBase, Field, ValidationError
from .mcp_decorator import (
    mcp_function,
    MCPFunctionSpec,
    ExecutionMode,
    CacheStrategy,
    ResourceLevel,
    StreamingMode,
    get_mcp_functions,
    get_function_by_name,
    export_function_specs,
    print_function_summary,
)

# Package metadata
__version__ = "1.0.0"
__author__ = "Chuk MCP Functions"
__description__ = (
    "Comprehensive MCP function library for AI models with async native math support"
)

# Configure logging
logger = logging.getLogger(__name__)

# Check if MCP decorator is available
try:
    from .mcp_decorator import get_mcp_functions

    _mcp_decorator_available = True
except ImportError:
    _mcp_decorator_available = False


def get_all_functions() -> Dict[str, MCPFunctionSpec]:
    """Get all registered MCP functions across all modules."""
    return get_mcp_functions()


def get_functions_by_category(category: str) -> Dict[str, MCPFunctionSpec]:
    """Get functions filtered by category."""
    all_funcs = get_mcp_functions()
    return {name: spec for name, spec in all_funcs.items() if spec.category == category}


def get_functions_by_namespace(namespace: str) -> Dict[str, MCPFunctionSpec]:
    """Get functions filtered by namespace."""
    return get_mcp_functions(namespace)


async def get_math_functions() -> Dict[str, Any]:
    """Get all mathematical functions organized by domain (async)."""
    if not _mcp_decorator_available:
        return {"arithmetic": {}, "number_theory": {}}

    all_funcs = get_mcp_functions()

    math_domains: list[dict[str, Any]] = {  # type: ignore[assignment]
        "arithmetic": {},
        "number_theory": {},
    }

    # Organize functions by their namespace
    for name, spec in all_funcs.items():
        domain = spec.namespace
        if domain in math_domains:
            math_domains[domain][spec.function_name] = spec  # type: ignore[call-overload]

    return math_domains  # type: ignore[return-value]


def get_math_constants() -> Dict[str, float]:
    """Get all mathematical constants."""
    return {
        "pi": math.pi,
        "e": math.e,
        "tau": math.tau,
        "inf": math.inf,
        "nan": math.nan,
        "golden_ratio": (1 + math.sqrt(5)) / 2,
        "euler_gamma": 0.5772156649015329,
        "sqrt2": math.sqrt(2),
        "sqrt3": math.sqrt(3),
        "ln2": math.log(2),
        "ln10": math.log(10),
        "log2e": math.log2(math.e),
        "log10e": math.log10(math.e),
    }


def get_execution_stats() -> Dict[str, Any]:
    """Get comprehensive execution statistics across all functions."""
    if not _mcp_decorator_available:
        return {
            "total_functions": 0,
            "execution_modes": {
                "local_capable": 0,
                "remote_capable": 0,
                "both_capable": 0,
            },
            "features": {
                "cached_functions": 0,
                "streaming_functions": 0,
                "workflow_compatible": 0,
            },
            "distribution": {"namespaces": {}, "categories": {}},
            "performance": {
                "total_executions": 0,
                "error_rate": 0.0,
                "cache_hit_rate": 0.0,
                "total_cache_hits": 0,
                "total_cache_misses": 0,
            },
        }

    all_funcs = get_mcp_functions()

    total_functions = len(all_funcs)
    local_count = sum(
        1 for spec in all_funcs.values() if spec.supports_local_execution()
    )
    remote_count = sum(
        1 for spec in all_funcs.values() if spec.supports_remote_execution()
    )
    cached_count = sum(
        1 for spec in all_funcs.values() if spec.cache_strategy != CacheStrategy.NONE
    )
    streaming_count = sum(1 for spec in all_funcs.values() if spec.supports_streaming)

    # Namespace distribution
    namespaces: dict[str, Any] = {}
    categories: dict[str, list[str]] = {}
    for spec in all_funcs.values():
        namespaces[spec.namespace] = namespaces.get(spec.namespace, 0) + 1
        categories[spec.category] = categories.get(spec.category, 0) + 1  # type: ignore[assignment,operator]

    # Performance metrics
    total_executions = 0
    total_errors = 0
    total_cache_hits = 0
    total_cache_misses = 0

    for spec in all_funcs.values():
        if hasattr(spec, "_performance_metrics") and spec._performance_metrics:
            total_executions += spec._performance_metrics.execution_count
            total_errors += spec._performance_metrics.error_count
            total_cache_hits += spec._performance_metrics.cache_hits
            total_cache_misses += spec._performance_metrics.cache_misses

    cache_hit_rate = 0.0
    if total_cache_hits + total_cache_misses > 0:
        cache_hit_rate = total_cache_hits / (total_cache_hits + total_cache_misses)

    error_rate = 0.0
    if total_executions > 0:
        error_rate = total_errors / total_executions

    return {
        "total_functions": total_functions,
        "execution_modes": {
            "local_capable": local_count,
            "remote_capable": remote_count,
            "both_capable": sum(
                1
                for spec in all_funcs.values()
                if spec.supports_local_execution() and spec.supports_remote_execution()
            ),
        },
        "features": {
            "cached_functions": cached_count,
            "streaming_functions": streaming_count,
            "workflow_compatible": sum(
                1 for spec in all_funcs.values() if spec.workflow_compatible
            ),
        },
        "distribution": {"namespaces": namespaces, "categories": categories},
        "performance": {
            "total_executions": total_executions,
            "error_rate": error_rate,
            "cache_hit_rate": cache_hit_rate,
            "total_cache_hits": total_cache_hits,
            "total_cache_misses": total_cache_misses,
        },
    }


async def get_async_performance_stats() -> Dict[str, Any]:
    """Get performance statistics for async functions."""
    if not _mcp_decorator_available:
        return {
            "total_async_functions": 0,
            "cached_functions": 0,
            "streaming_functions": 0,
            "high_performance_functions": 0,
            "domains_converted": 2,  # arithmetic + number_theory
        }

    math_funcs = await get_math_functions()

    stats = {
        "total_async_functions": 0,
        "cached_functions": 0,
        "streaming_functions": 0,
        "high_performance_functions": 0,
        "domains_converted": 0,
    }

    for domain_name, functions in math_funcs.items():
        if functions:  # Domain has functions
            stats["domains_converted"] += 1

        for func_name, spec in functions.items():
            stats["total_async_functions"] += 1

            if spec.cache_strategy.value != "none":
                stats["cached_functions"] += 1

            if spec.supports_streaming:
                stats["streaming_functions"] += 1

            if spec.estimated_cpu_usage.value == "high":
                stats["high_performance_functions"] += 1

    return stats


def get_function_recommendations(operation_type: str) -> List[str]:
    """Get function recommendations based on operation type."""
    recommendations = {
        # Arithmetic operations
        "basic": ["add", "subtract", "multiply", "divide", "power", "sqrt"],
        "comparison": [
            "equal",
            "less_than",
            "greater_than",
            "minimum",
            "maximum",
            "clamp",
        ],
        "rounding": ["round_number", "floor", "ceil"],
        "modular": ["modulo", "mod_power", "quotient"],
        # Number theory operations
        "primes": [
            "is_prime",
            "next_prime",
            "nth_prime",
            "prime_factors",
            "is_coprime",
        ],
        "divisibility": ["gcd", "lcm", "divisors", "is_even", "is_odd", "extended_gcd"],
        "sequences": [
            "fibonacci",
            "lucas_number",
            "catalan_number",
            "triangular_number",
        ],
        "special_numbers": [
            "is_perfect_number",
            "is_abundant_number",
            "is_palindromic_number",
        ],
        "cryptographic": [
            "discrete_log_naive",
            "primitive_root",
            "legendre_symbol",
            "crt_solve",
        ],
        "figurate": [
            "polygonal_number",
            "centered_triangular_number",
            "pronic_number",
            "star_number",
        ],
        "digital": ["digit_sum", "digital_root", "is_harshad_number", "digit_reversal"],
        "constants": [
            "compute_pi_machin",
            "compute_e_series",
            "compute_golden_ratio_fibonacci",
        ],
    }

    return recommendations.get(operation_type.lower(), [])


def validate_math_domain(domain: str) -> bool:
    """Validate if a mathematical domain exists."""
    valid_domains = {"arithmetic", "number_theory"}
    return domain.lower() in valid_domains


async def print_math_summary():
    """Print a summary of all mathematical functions by domain (async)."""
    print("🧮 Mathematical Functions Library (Async Native)")
    print("=" * 50)

    print("📊 Available Domains:")
    print("📐 arithmetic - Reorganized structure with core, comparison, number_theory")
    print(
        "🔢 number_theory - Primes, divisibility, sequences, special numbers, cryptographic functions"
    )
    print()

    # Check what's available in arithmetic
    try:
        from . import arithmetic as arith_module
        if hasattr(arith_module, "print_reorganized_status"):
            arith_module.print_reorganized_status()
    except ImportError:
        pass

    print()

    # Show number theory capabilities
    print("🔢 Number Theory Capabilities:")
    print("   • Prime operations: is_prime, next_prime, prime_factors, is_coprime")
    print("   • Divisibility: gcd, lcm, divisors, extended_gcd, euler_totient")
    print("   • Special sequences: fibonacci, lucas, catalan, bell numbers")
    print("   • Figurate numbers: polygonal, centered, pronic, pyramidal")
    print("   • Modular arithmetic: CRT, quadratic residues, primitive roots")
    print("   • Cryptographic functions: discrete log, legendre symbols")
    print("   • Digital operations: digit sums, palindromes, harshad numbers")
    print("   • Egyptian fractions: unit fractions, harmonic series")
    print("   • Mathematical constants: high-precision pi, e, golden ratio")


def print_comprehensive_summary():
    """Print a comprehensive summary of all registered functions."""
    stats = get_execution_stats()

    print("🚀 Chuk MCP Functions - Comprehensive Summary (Async Native)")
    print("=" * 60)
    print(f"📊 Total Functions: {stats['total_functions']}")
    print(f"📦 Local Executable: {stats['execution_modes']['local_capable']}")
    print(f"🛠️  Remote Callable: {stats['execution_modes']['remote_capable']}")
    print(f"🔄 Dual Mode: {stats['execution_modes']['both_capable']}")
    print(f"💾 Cached: {stats['features']['cached_functions']}")
    print(f"🌊 Streaming: {stats['features']['streaming_functions']}")
    print(f"🔗 Workflow Ready: {stats['features']['workflow_compatible']}")
    print()

    print("📁 By Namespace:")
    for namespace, count in sorted(stats["distribution"]["namespaces"].items()):
        print(f"   • {namespace}: {count} functions")
    print()

    print("🏷️  By Category:")
    for category, count in sorted(stats["distribution"]["categories"].items()):
        print(f"   • {category}: {count} functions")
    print()

    if stats["performance"]["total_executions"] > 0:
        print("⚡ Performance Metrics:")
        print(f"   • Total Executions: {stats['performance']['total_executions']:,}")
        print(f"   • Error Rate: {stats['performance']['error_rate']:.2%}")
        print(f"   • Cache Hit Rate: {stats['performance']['cache_hit_rate']:.2%}")
        print()


def math_quick_reference() -> str:
    """Generate a quick reference guide for mathematical functions."""
    reference = """
🧮 Mathematical Functions Quick Reference (Async Native)

🚀 REORGANIZED ARITHMETIC STRUCTURE:
   
📐 CORE OPERATIONS (use await):
   await add(a, b), await subtract(a, b), await multiply(a, b)
   await divide(a, b), await power(base, exp), await sqrt(x)
   await round_number(x, decimals), await floor(x), await ceil(x)
   await modulo(a, b), await mod_power(base, exp, mod)

🔍 COMPARISON OPERATIONS (use await):
   await equal(a, b), await less_than(a, b), await greater_than(a, b)
   await minimum(a, b), await maximum(a, b), await clamp(val, min, max)
   await sort_numbers(list), await approximately_equal(a, b, tol)

🔢 NUMBER THEORY OPERATIONS (use await):
   
   PRIMES & DIVISIBILITY:
   await is_prime(n), await next_prime(n), await prime_factors(n)
   await gcd(a, b), await lcm(a, b), await divisors(n)
   await is_even(n), await is_odd(n), await extended_gcd(a, b)
   
   SEQUENCES & SPECIAL NUMBERS:
   await fibonacci(n), await lucas_number(n), await catalan_number(n)
   await triangular_number(n), await factorial(n), await bell_number(n)
   await is_perfect_number(n), await euler_totient(n)
   
   FIGURATE NUMBERS:
   await polygonal_number(n, sides), await centered_triangular_number(n)
   await pronic_number(n), await star_number(n), await octahedral_number(n)
   
   MODULAR ARITHMETIC & CRYPTOGRAPHY:
   await crt_solve(remainders, moduli), await primitive_root(p)
   await is_quadratic_residue(a, p), await legendre_symbol(a, p)
   await discrete_log_naive(base, target, mod)
   
   DIGITAL OPERATIONS:
   await digit_sum(n), await digital_root(n), await is_palindromic_number(n)
   await is_harshad_number(n), await digit_reversal(n)
   
   MATHEMATICAL CONSTANTS:
   await compute_pi_machin(precision), await compute_e_series(terms)
   await compute_golden_ratio_fibonacci(n)

🎯 IMPORT PATTERNS:
   # Arithmetic (reorganized structure)
   from chuk_mcp_math.arithmetic.core import add, multiply
   from chuk_mcp_math.arithmetic.comparison import minimum
   
   # Number theory (comprehensive modules)
   from chuk_mcp_math.number_theory import is_prime, gcd
   from chuk_mcp_math.number_theory.primes import next_prime
   from chuk_mcp_math.number_theory.modular_arithmetic import crt_solve
   
   # Or use submodules
   from chuk_mcp_math import arithmetic, number_theory
   result = await arithmetic.core.add(5, 3)
   prime_check = await number_theory.is_prime(17)
   crt_result = await number_theory.crt_solve([1, 2], [3, 5])
"""
    return reference.strip()


def export_all_specs(filename: str = "mcp_functions_complete.json"):
    """Export all function specifications to a JSON file."""
    if _mcp_decorator_available:
        export_function_specs(filename)
        print(f"📤 Exported all function specifications to {filename}")
    else:
        print("⚠️  MCP decorator not available, cannot export specifications")


def clear_all_caches():
    """Clear all function caches."""
    if not _mcp_decorator_available:
        print("⚠️  MCP decorator not available, cannot clear caches")
        return

    cleared_count = 0
    for spec in get_mcp_functions().values():
        if hasattr(spec, "_cache_backend") and spec._cache_backend:
            spec._cache_backend.clear()
            cleared_count += 1

    print(f"🗑️  Cleared {cleared_count} function caches")


# Export main components
__all__ = [
    # Core MCP components
    "McpPydanticBase",
    "Field",
    "ValidationError",
    "mcp_function",
    "MCPFunctionSpec",
    "ExecutionMode",
    "CacheStrategy",
    "ResourceLevel",
    "StreamingMode",
    # Function management
    "get_mcp_functions",
    "get_function_by_name",
    "get_all_functions",
    "get_functions_by_category",
    "get_functions_by_namespace",
    # Math-specific functions (async native)
    "get_math_functions",
    "get_math_constants",
    "print_math_summary",
    "get_function_recommendations",
    "validate_math_domain",
    "get_async_performance_stats",
    "math_quick_reference",
    # Statistics and management
    "get_execution_stats",
    "print_function_summary",
    "print_comprehensive_summary",
    "export_function_specs",
    "export_all_specs",
    "clear_all_caches",
    # Math modules (async native)
    "arithmetic",  # Reorganized structure with core, comparison
    "number_theory",  # Comprehensive number theory functions
    # Future math modules (commented out until implemented)
    # 'trigonometry', 'logarithmic', 'statistical',
    # 'algebraic', 'financial', 'geometric', 'combinatorial', 'constants',
    # Package info
    "__version__",
    "__author__",
    "__description__",
]


# Initialize logging for the package
def setup_logging(level: str = "INFO"):
    """Setup package-wide logging."""
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {level}")

    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logger.info(f"Chuk MCP Functions v{__version__} initialized (Async Native)")
    if _mcp_decorator_available:
        logger.info(f"Loaded {len(get_mcp_functions())} functions")
    else:
        logger.warning("MCP decorator not available - some functionality limited")


# Auto-setup logging at import
setup_logging()

# DO NOT import specific functions here to avoid circular import issues
# Users should import from the reorganized structure directly:
# from chuk_mcp_math.arithmetic.core.basic_operations import add
# from chuk_mcp_math.number_theory.primes import is_prime
# from chuk_mcp_math.number_theory.modular_arithmetic import crt_solve

if __name__ == "__main__":
    import asyncio

    async def main():
        # Print comprehensive summary with both sync and async capabilities
        print_comprehensive_summary()
        print("\n" + "=" * 50)
        await print_math_summary()
        print("\n" + "=" * 50)
        print(math_quick_reference())

        # Test both domains if available
        print("\n🧪 Testing Both Domains:")

        # Test arithmetic if available
        try:
            from .arithmetic.core.basic_operations import add

            result = await add(5, 3)
            print(f"✅ Arithmetic test: 5 + 3 = {result}")
        except Exception as e:
            print(f"⚠️  Arithmetic test failed: {e}")

        # Test number theory if available
        try:
            from .number_theory import is_prime

            result = await is_prime(17)
            print(f"✅ Number theory test: is_prime(17) = {result}")
        except Exception as e:
            print(f"⚠️  Number theory test failed: {e}")

        # Show async performance stats
        async_stats = await get_async_performance_stats()
        print(
            f"\n📈 Async Performance: {async_stats['total_async_functions']} async functions, {async_stats['domains_converted']} domains converted"
        )

    asyncio.run(main())
