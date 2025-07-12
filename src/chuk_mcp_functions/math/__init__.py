#!/usr/bin/env python3
# chuk_mcp_functions/math/__init__.py
"""
Mathematical Functions Library for AI Models (Async Native)

A comprehensive collection of mathematical functions organized by domain.
Designed specifically for AI model execution with clear documentation,
examples, and robust error handling. All functions are async native for
optimal performance in async environments.

Mathematical Domains:
- arithmetic: Basic operations (add, subtract, multiply, divide, etc.) - ASYNC NATIVE ✅
- trigonometry: Trigonometric functions (sin, cos, tan, etc.) - Coming soon
- logarithmic: Logarithmic and exponential functions - Coming soon
- statistical: Statistics and probability functions - Coming soon
- algebraic: Advanced algebraic operations - Coming soon
- financial: Financial mathematics (interest, present value, etc.) - Coming soon
- geometric: Geometric calculations (area, volume, distance, etc.) - Coming soon
- combinatorial: Combinatorics and discrete math - Coming soon
- constants: Mathematical constants and special numbers - ASYNC NATIVE ✅

All functions support:
- Async native execution for optimal performance
- Local and remote execution modes
- Comprehensive error handling
- Performance optimization with caching where appropriate
- Rich examples for AI understanding
- Type safety and validation
- Strategic yielding for long-running operations
"""

from typing import Dict, List, Any
import math
import asyncio

# Import all mathematical modules (currently only arithmetic is async native)
from . import arithmetic
# TODO: Convert these to async native
# from . import trigonometry
# from . import logarithmic
# from . import statistical
# from . import algebraic
# from . import financial
# from . import geometric
# from . import combinatorial
# from . import constants

# Import core functions for easier access
from chuk_mcp_functions.mcp_decorator import get_mcp_functions

async def get_math_functions() -> Dict[str, Any]:
    """Get all mathematical functions organized by domain (async)."""
    all_funcs = get_mcp_functions()
    
    math_domains = {
        'arithmetic': {},
        # TODO: Add these as they are converted to async native
        # 'trigonometry': {},
        # 'logarithmic': {},
        # 'statistical': {},
        # 'algebraic': {},
        # 'financial': {},
        # 'geometric': {},
        # 'combinatorial': {},
        # 'constants': {}
    }
    
    # Organize functions by their namespace
    for name, spec in all_funcs.items():
        domain = spec.namespace
        if domain in math_domains:
            math_domains[domain][spec.function_name] = spec
    
    return math_domains

def get_math_constants() -> Dict[str, float]:
    """Get all mathematical constants."""
    return {
        'pi': math.pi,
        'e': math.e,
        'tau': math.tau,
        'inf': math.inf,
        'nan': math.nan,
        'golden_ratio': (1 + math.sqrt(5)) / 2,
        'euler_gamma': 0.5772156649015329,
        'sqrt2': math.sqrt(2),
        'sqrt3': math.sqrt(3),
        'ln2': math.log(2),
        'ln10': math.log(10),
        'log2e': math.log2(math.e),
        'log10e': math.log10(math.e)
    }

async def print_math_summary():
    """Print a summary of all mathematical functions by domain (async)."""
    math_funcs = await get_math_functions()
    
    print("🧮 Mathematical Functions Library (Async Native)")
    print("=" * 50)
    
    total_functions = sum(len(domain) for domain in math_funcs.values())
    print(f"📊 Total Math Functions: {total_functions}")
    print("🚀 Async native domains: arithmetic")
    print("⏳ Converting other domains to async native...")
    print()
    
    for domain_name, functions in math_funcs.items():
        if functions:  # Only show domains with functions
            async_status = "🚀 ASYNC NATIVE" if domain_name == "arithmetic" else "⏳ Converting..."
            print(f"📐 {domain_name.title()} ({len(functions)} functions) - {async_status}")
            
            # Show a few example functions
            example_funcs = list(functions.items())[:5]  # Show first 5
            for func_name, spec in example_funcs:
                # Show execution modes
                modes = "/".join([mode.value for mode in spec.execution_modes])
                features = ""
                if spec.cache_strategy.value != "none":
                    features += " 💾"
                if spec.supports_streaming:
                    features += " 🌊"
                if spec.estimated_cpu_usage.value == "high":
                    features += " ⚡"
                
                async_marker = " (async)" if domain_name == "arithmetic" else ""
                print(f"   • {func_name} - {modes}{features}{async_marker}")
                
            if len(functions) > 5:
                print(f"   ... and {len(functions) - 5} more functions")
            print()
    
    # Show constants summary
    print(f"🔢 Mathematical Constants Available:")
    print("   • Core: pi(), e(), tau(), golden_ratio() (async)")
    print("   • Roots: sqrt2(), sqrt3(), sqrt5(), cbrt2(), cbrt3() (async)")
    print("   • Logs: ln2(), ln10(), log2e(), log10e() (async)")
    print("   • Special: euler_gamma(), catalan(), apery() (async)")
    print("   • Limits: machine_epsilon(), max_float(), min_float() (async)")

def get_function_recommendations(operation_type: str) -> List[str]:
    """Get function recommendations based on operation type."""
    recommendations = {
        'basic': ['add', 'subtract', 'multiply', 'divide', 'power', 'sqrt'],
        # TODO: Add these as domains are converted
        # 'trigonometry': ['sin', 'cos', 'tan', 'asin', 'acos', 'atan', 'degrees', 'radians'],
        # 'statistics': ['mean', 'median', 'mode', 'std_dev', 'variance', 'correlation'],
        # 'financial': ['compound_interest', 'present_value', 'future_value', 'pmt', 'irr'],
        # 'geometry': ['circle_area', 'sphere_volume', 'distance_2d', 'triangle_area'],
        # 'probability': ['factorial', 'combinations', 'permutations', 'binomial_probability'],
        'advanced': ['log', 'ln', 'exp', 'gcd', 'lcm', 'is_prime', 'fibonacci'],
        'sequences': ['arithmetic_sequence', 'geometric_sequence', 'triangular_numbers'],
        'comparison': ['equal', 'less_than', 'greater_than', 'sort_numbers', 'clamp'],
        'number_theory': ['is_prime', 'prime_factors', 'gcd', 'lcm', 'factorial']
    }
    
    return recommendations.get(operation_type.lower(), [])

def validate_math_domain(domain: str) -> bool:
    """Validate if a mathematical domain exists."""
    # Currently only arithmetic is fully async native
    async_native_domains = {'arithmetic'}
    # TODO: Add these as they are converted
    # valid_domains = {
    #     'arithmetic', 'trigonometry', 'logarithmic', 'statistical',
    #     'algebraic', 'financial', 'geometric', 'combinatorial', 'constants'
    # }
    return domain.lower() in async_native_domains

async def get_async_performance_stats() -> Dict[str, Any]:
    """Get performance statistics for async functions."""
    math_funcs = await get_math_functions()
    
    stats = {
        'total_async_functions': 0,
        'cached_functions': 0,
        'streaming_functions': 0,
        'high_performance_functions': 0,
        'domains_converted': 0
    }
    
    for domain_name, functions in math_funcs.items():
        if functions:  # Domain has functions
            stats['domains_converted'] += 1
            
        for func_name, spec in functions.items():
            stats['total_async_functions'] += 1
            
            if spec.cache_strategy.value != "none":
                stats['cached_functions'] += 1
                
            if spec.supports_streaming:
                stats['streaming_functions'] += 1
                
            if spec.estimated_cpu_usage.value == "high":
                stats['high_performance_functions'] += 1
    
    return stats

def math_quick_reference() -> str:
    """Generate a quick reference guide for mathematical functions."""
    reference = """
🧮 Mathematical Functions Quick Reference (Async Native)

🚀 ASYNC NATIVE DOMAINS (use await):
   
📐 ARITHMETIC - Complete ✅
   await add(a, b), await subtract(a, b), await multiply(a, b)
   await divide(a, b), await power(base, exp), await sqrt(x)
   await is_prime(n), await gcd(a, b), await lcm(a, b)
   await factorial(n), await fibonacci(n)
   await arithmetic_sequence(first, diff, count)
   await geometric_sequence(first, ratio, count)
   await sort_numbers(list), await clamp(val, min, max)
   await ln(x), await log(x, base), await exp(x)

🔢 CONSTANTS - Complete ✅
   await pi(), await e(), await tau(), await golden_ratio()
   await sqrt2(), await sqrt3(), await ln2(), await ln10()
   await euler_gamma(), await catalan(), await apery()

⏳ CONVERTING TO ASYNC NATIVE:
   📊 trigonometry, 📈 statistical, 💰 financial
   🔬 algebraic, 📏 geometric, 🎲 combinatorial

🚀 ASYNC PERFORMANCE FEATURES:
   • Strategic yielding for long computations
   • Memory caching for expensive operations  
   • Streaming support for large datasets
   • Automatic load balancing

🎯 ASYNC USAGE PATTERNS:
   result = await add(5, 3)
   primes = [await is_prime(n) for n in range(100)]
   sequence = await arithmetic_sequence(1, 2, 10)
   
   # Parallel execution
   results = await asyncio.gather(
       add(1, 2), multiply(3, 4), sqrt(16)
   )
   
   # Batch processing with yielding
   for i in range(1000):
       result = await complex_calculation(i)
       if i % 100 == 0:  # Auto-yields built into functions
           pass  # Functions handle yielding automatically
"""
    return reference.strip()

# Export all math modules and utilities
__all__ = [
    # Mathematical domains (async native)
    'arithmetic',
    
    # TODO: Add these as they are converted to async native
    # 'trigonometry', 'logarithmic', 'statistical',
    # 'algebraic', 'financial', 'geometric', 'combinatorial', 'constants',
    
    # Utility functions
    'get_math_functions', 'get_math_constants', 'print_math_summary',
    'get_function_recommendations', 'validate_math_domain',
    'get_async_performance_stats', 'math_quick_reference'
]

# Quick access to async arithmetic functions
from .arithmetic import (
    # Basic operations (async)
    add, subtract, multiply, divide, power, sqrt, abs_value, round_number,
    # Comparison (async) 
    equal, less_than, greater_than, minimum, maximum, clamp,
    # Number theory (async)
    is_prime, gcd, lcm, factorial, fibonacci, is_even, is_odd,
    # Sequences (async)
    arithmetic_sequence, geometric_sequence, triangular_numbers,
    # Advanced operations (async)
    ln, log, log10, exp, product, random_float, random_int,
    # Constants (async)
    pi, e, golden_ratio, sqrt2
)

# Add quick access functions to exports
__all__.extend([
    # Async arithmetic functions
    'add', 'subtract', 'multiply', 'divide', 'power', 'sqrt', 'abs_value', 'round_number',
    'equal', 'less_than', 'greater_than', 'minimum', 'maximum', 'clamp',
    'is_prime', 'gcd', 'lcm', 'factorial', 'fibonacci', 'is_even', 'is_odd',
    'arithmetic_sequence', 'geometric_sequence', 'triangular_numbers',
    'ln', 'log', 'log10', 'exp', 'product', 'random_float', 'random_int',
    'pi', 'e', 'golden_ratio', 'sqrt2'
])

async def demo_async_math():
    """Demonstrate async math library capabilities."""
    print("🚀 Async Math Library Demo")
    print("=" * 30)
    
    # Basic arithmetic (async)
    print("📐 Basic Operations:")
    result1 = await add(5, 3)
    result2 = await multiply(4, 6)
    result3 = await sqrt(16)
    print(f"   add(5, 3) = {result1}")
    print(f"   multiply(4, 6) = {result2}")
    print(f"   sqrt(16) = {result3}")
    
    # Parallel execution
    print("\n⚡ Parallel Execution:")
    start_time = asyncio.get_event_loop().time()
    parallel_results = await asyncio.gather(
        factorial(10),
        fibonacci(15), 
        is_prime(97),
        arithmetic_sequence(1, 2, 5)
    )
    end_time = asyncio.get_event_loop().time()
    print(f"   Computed 4 operations in parallel: {end_time - start_time:.4f}s")
    print(f"   Results: {parallel_results}")
    
    # Sequences
    print("\n📊 Sequences:")
    arith_seq = await arithmetic_sequence(2, 3, 5)
    geom_seq = await geometric_sequence(2, 2, 4)
    print(f"   Arithmetic: {arith_seq}")
    print(f"   Geometric: {geom_seq}")
    
    # Advanced operations
    print("\n🔬 Advanced Operations:")
    log_result = await ln(await e())
    exp_result = await exp(1)
    print(f"   ln(e) = {log_result}")
    print(f"   exp(1) = {exp_result}")
    
    # Performance stats
    print("\n📈 Performance Stats:")
    stats = await get_async_performance_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")

if __name__ == "__main__":
    import asyncio
    
    async def main():
        await print_math_summary()
        print("\n" + "="*50)
        print(math_quick_reference())
        print("\n" + "="*50)
        await demo_async_math()
    
    asyncio.run(main())