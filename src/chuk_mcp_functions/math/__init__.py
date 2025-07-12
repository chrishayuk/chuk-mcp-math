#!/usr/bin/env python3
# chuk_mcp_functions/math/__init__.py
"""
Mathematical Functions Library for AI Models (Async Native)

A comprehensive collection of mathematical functions organized by domain.
Designed specifically for AI model execution with clear documentation,
examples, and robust error handling. All functions are async native for
optimal performance in async environments.

Mathematical Domains:
- arithmetic: Basic operations (reorganized structure) - ASYNC NATIVE ✅

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

# Import arithmetic module (reorganized structure)
from . import arithmetic

# Import core functions for easier access
try:
    from chuk_mcp_functions.mcp_decorator import get_mcp_functions
    _mcp_decorator_available = True
except ImportError:
    _mcp_decorator_available = False

async def get_math_functions() -> Dict[str, Any]:
    """Get all mathematical functions organized by domain (async)."""
    if not _mcp_decorator_available:
        return {'arithmetic': {}}
        
    all_funcs = get_mcp_functions()
    
    math_domains = {
        'arithmetic': {},
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
    print("🧮 Mathematical Functions Library (Async Native)")
    print("=" * 50)
    
    print("📊 Available Domains:")
    print("📐 arithmetic - Reorganized structure with core, comparison, number_theory")
    print()
    
    # Check what's available in arithmetic
    if hasattr(arithmetic, 'print_reorganized_status'):
        arithmetic.print_reorganized_status()

def get_function_recommendations(operation_type: str) -> List[str]:
    """Get function recommendations based on operation type."""
    recommendations = {
        'basic': ['add', 'subtract', 'multiply', 'divide', 'power', 'sqrt'],
        'comparison': ['equal', 'less_than', 'greater_than', 'minimum', 'maximum', 'clamp'],
        'number_theory': ['is_prime', 'gcd', 'lcm', 'is_even', 'is_odd'],
        'rounding': ['round_number', 'floor', 'ceil'],
        'modular': ['modulo', 'mod_power', 'quotient']
    }
    
    return recommendations.get(operation_type.lower(), [])

def validate_math_domain(domain: str) -> bool:
    """Validate if a mathematical domain exists."""
    valid_domains = {'arithmetic'}
    return domain.lower() in valid_domains

async def get_async_performance_stats() -> Dict[str, Any]:
    """Get performance statistics for async functions."""
    if not _mcp_decorator_available:
        return {
            'total_async_functions': 0,
            'cached_functions': 0,
            'streaming_functions': 0,
            'high_performance_functions': 0,
            'domains_converted': 1
        }
    
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

🔢 NUMBER THEORY (use await):
   await is_prime(n), await gcd(a, b), await lcm(a, b)
   await is_even(n), await is_odd(n), await divisors(n)

🎯 IMPORT PATTERNS:
   # Reorganized structure imports
   from chuk_mcp_functions.math.arithmetic.core import add, multiply
   from chuk_mcp_functions.math.arithmetic.number_theory import is_prime
   from chuk_mcp_functions.math.arithmetic.comparison import minimum
   
   # Or use submodules
   from chuk_mcp_functions.math.arithmetic import core, number_theory
   result = await core.add(5, 3)
   prime_check = await number_theory.is_prime(17)
"""
    return reference.strip()

# Export main components - only what exists
__all__ = [
    # Mathematical domains (async native)
    'arithmetic',
    
    # Utility functions
    'get_math_functions', 'get_math_constants', 'print_math_summary',
    'get_function_recommendations', 'validate_math_domain',
    'get_async_performance_stats', 'math_quick_reference'
]

# DO NOT import specific functions here to avoid circular import issues
# Users should import from the reorganized structure directly:
# from chuk_mcp_functions.math.arithmetic.core.basic_operations import add
# from chuk_mcp_functions.math.arithmetic.number_theory.primes import is_prime

if __name__ == "__main__":
    import asyncio
    
    async def main():
        await print_math_summary()
        print("\n" + "="*50)
        print(math_quick_reference())
    
    asyncio.run(main())