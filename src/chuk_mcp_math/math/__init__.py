#!/usr/bin/env python3
# chuk_mcp_math/__init__.py
"""
Mathematical Functions Library for AI Models (Async Native)

A comprehensive collection of mathematical functions organized by domain.
Designed specifically for AI model execution with clear documentation,
examples, and robust error handling. All functions are async native for
optimal performance in async environments.

Mathematical Domains:
- arithmetic: Basic operations (reorganized structure) - ASYNC NATIVE ✅
- number_theory: Prime numbers, divisibility, sequences, cryptographic functions - ASYNC NATIVE ✅

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

# Import number_theory module (comprehensive number theory functions)
from . import number_theory

# Import core functions for easier access
try:
    from chuk_mcp_math.mcp_decorator import get_mcp_functions
    _mcp_decorator_available = True
except ImportError:
    _mcp_decorator_available = False

async def get_math_functions() -> Dict[str, Any]:
    """Get all mathematical functions organized by domain (async)."""
    if not _mcp_decorator_available:
        return {
            'arithmetic': {},
            'number_theory': {}
        }
        
    all_funcs = get_mcp_functions()
    
    math_domains = {
        'arithmetic': {},
        'number_theory': {},
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
    print("🔢 number_theory - Primes, divisibility, sequences, special numbers, cryptographic functions")
    print()
    
    # Check what's available in arithmetic
    if hasattr(arithmetic, 'print_reorganized_status'):
        arithmetic.print_reorganized_status()
    
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

def get_function_recommendations(operation_type: str) -> List[str]:
    """Get function recommendations based on operation type."""
    recommendations = {
        # Arithmetic operations
        'basic': ['add', 'subtract', 'multiply', 'divide', 'power', 'sqrt'],
        'comparison': ['equal', 'less_than', 'greater_than', 'minimum', 'maximum', 'clamp'],
        'rounding': ['round_number', 'floor', 'ceil'],
        'modular': ['modulo', 'mod_power', 'quotient'],
        
        # Number theory operations
        'primes': ['is_prime', 'next_prime', 'nth_prime', 'prime_factors', 'is_coprime'],
        'divisibility': ['gcd', 'lcm', 'divisors', 'is_even', 'is_odd', 'extended_gcd'],
        'sequences': ['fibonacci', 'lucas_number', 'catalan_number', 'triangular_number'],
        'special_numbers': ['is_perfect_number', 'is_abundant_number', 'is_palindromic_number'],
        'cryptographic': ['discrete_log_naive', 'primitive_root', 'legendre_symbol', 'crt_solve'],
        'figurate': ['polygonal_number', 'centered_triangular_number', 'pronic_number', 'star_number'],
        'digital': ['digit_sum', 'digital_root', 'is_harshad_number', 'digit_reversal'],
        'constants': ['compute_pi_machin', 'compute_e_series', 'compute_golden_ratio_fibonacci']
    }
    
    return recommendations.get(operation_type.lower(), [])

def validate_math_domain(domain: str) -> bool:
    """Validate if a mathematical domain exists."""
    valid_domains = {'arithmetic', 'number_theory'}
    return domain.lower() in valid_domains

async def get_async_performance_stats() -> Dict[str, Any]:
    """Get performance statistics for async functions."""
    if not _mcp_decorator_available:
        return {
            'total_async_functions': 0,
            'cached_functions': 0,
            'streaming_functions': 0,
            'high_performance_functions': 0,
            'domains_converted': 2  # arithmetic + number_theory
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

# Export main components - both existing and new
__all__ = [
    # Mathematical domains (async native)
    'arithmetic',
    'number_theory',  # Added number_theory
    
    # Utility functions
    'get_math_functions', 'get_math_constants', 'print_math_summary',
    'get_function_recommendations', 'validate_math_domain',
    'get_async_performance_stats', 'math_quick_reference'
]

# DO NOT import specific functions here to avoid circular import issues
# Users should import from the reorganized structure directly:
# from chuk_mcp_math.arithmetic.core.basic_operations import add
# from chuk_mcp_math.number_theory.primes import is_prime
# from chuk_mcp_math.number_theory.modular_arithmetic import crt_solve

if __name__ == "__main__":
    import asyncio
    
    async def main():
        await print_math_summary()
        print("\n" + "="*50)
        print(math_quick_reference())
        
        # Test both domains
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
    
    asyncio.run(main())