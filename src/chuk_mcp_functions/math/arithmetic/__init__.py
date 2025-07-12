#!/usr/bin/env python3
# chuk_mcp_functions/math/arithmetic/__init__.py
"""
Arithmetic Functions Module for AI Models (Async Native)

A comprehensive collection of arithmetic functions organized into logical groupings.
This module provides fundamental mathematical operations that AI models need for
numerical computation, pattern recognition, and mathematical reasoning.

All functions are async native for optimal performance in async environments.

Submodules:
- basic_operations: Core arithmetic (add, subtract, multiply, divide, power, etc.)
- comparison: Comparison and ordering functions (equal, min/max, sorting, etc.)
- number_theory: Prime numbers, divisibility, GCD/LCM, integer properties
- sequences: Arithmetic/geometric sequences, series, and pattern analysis
- advanced_operations: Logarithms, exponentials, base conversions, etc.
- constants: Mathematical constants and special numbers

All functions are designed with:
- Async native implementations for optimal performance
- Clear documentation for AI understanding
- Comprehensive error handling
- Performance optimization with caching and yielding
- Rich examples and test cases
- Type safety and validation
"""

from typing import Dict, List, Any
import math
import asyncio

# Import all arithmetic submodules
from . import basic_operations
from . import comparison  
from . import number_theory
from . import sequences
from . import advanced_operations
from . import constants

# Import core functions for easier access
from chuk_mcp_functions.mcp_decorator import get_mcp_functions

async def get_arithmetic_functions(namespace: str = None) -> Dict[str, Any]:
    """Get all arithmetic functions organized by category (async)."""
    all_funcs = get_mcp_functions()
    
    # Filter for arithmetic namespace functions
    arithmetic_funcs = {
        name: spec for name, spec in all_funcs.items()
        if spec.namespace == "arithmetic"
    }
    
    if namespace:
        arithmetic_funcs = {
            name: spec for name, spec in arithmetic_funcs.items()
            if spec.category == namespace
        }
    
    # Organize by submodule/category
    categories = {
        'basic_operations': {},
        'comparison': {},
        'number_theory': {},
        'sequences': {},
        'advanced_operations': {},
        'constants': {}
    }
    
    # Basic operations
    basic_ops = [
        'add', 'subtract', 'multiply', 'divide', 'power', 'sqrt', 'nth_root',
        'modulo', 'divmod_operation', 'abs_value', 'sign', 'negate',
        'round_number', 'floor', 'ceil', 'truncate'
    ]
    
    # Comparison operations  
    comparison_ops = [
        'equal', 'not_equal', 'less_than', 'less_than_or_equal',
        'greater_than', 'greater_than_or_equal', 'minimum', 'maximum', 'clamp',
        'approximately_equal', 'close_to_zero', 'sort_numbers', 'rank_numbers',
        'in_range', 'between'
    ]
    
    # Number theory operations
    number_theory_ops = [
        'is_prime', 'next_prime', 'prime_factors', 'nth_prime',
        'gcd', 'lcm', 'divisors', 'is_divisible',
        'is_even', 'is_odd', 'is_perfect_square', 'is_power_of_two',
        'fibonacci', 'factorial', 'collatz_steps', 'mod_power'
    ]
    
    # Sequence operations
    sequence_ops = [
        'arithmetic_sequence', 'arithmetic_sum', 'geometric_sequence', 'geometric_sum',
        'triangular_numbers', 'square_numbers', 'cube_numbers',
        'harmonic_series', 'power_series_sum',
        'find_differences', 'is_arithmetic', 'is_geometric'
    ]
    
    # Advanced operations
    advanced_ops = [
        'ln', 'log', 'log10', 'log2', 'exp',
        'ceiling_multiple', 'floor_multiple', 'mround',
        'decimal_to_base', 'base_to_decimal',
        'quotient', 'double_factorial', 'multinomial',
        'sum_product', 'sum_squares', 'product',
        'random_float', 'random_int', 'random_array',
        'arabic_to_roman', 'roman_to_arabic', 'series_sum'
    ]
    
    # Constants
    constant_ops = [
        'pi', 'e', 'tau', 'infinity', 'nan',
        'golden_ratio', 'silver_ratio', 'plastic_number',
        'sqrt2', 'sqrt3', 'sqrt5', 'cbrt2', 'cbrt3',
        'ln2', 'ln10', 'log2e', 'log10e',
        'euler_gamma', 'catalan', 'apery', 'khinchin', 'glaisher',
        'machine_epsilon', 'max_float', 'min_float'
    ]
    
    # Categorize functions
    for name, spec in arithmetic_funcs.items():
        func_name = spec.function_name
        
        if func_name in basic_ops:
            categories['basic_operations'][func_name] = spec
        elif func_name in comparison_ops:
            categories['comparison'][func_name] = spec
        elif func_name in number_theory_ops:
            categories['number_theory'][func_name] = spec
        elif func_name in sequence_ops:
            categories['sequences'][func_name] = spec
        elif func_name in advanced_ops:
            categories['advanced_operations'][func_name] = spec
        elif func_name in constant_ops:
            categories['constants'][func_name] = spec
    
    return categories

def get_arithmetic_constants() -> Dict[str, float]:
    """Get arithmetic-related mathematical constants."""
    return {
        'pi': math.pi,
        'e': math.e,
        'tau': math.tau,
        'golden_ratio': (1 + math.sqrt(5)) / 2,
        'sqrt2': math.sqrt(2),
        'sqrt3': math.sqrt(3),
        'sqrt5': math.sqrt(5),
        'ln2': math.log(2),
        'ln10': math.log(10),
        'euler_gamma': 0.5772156649015329,  # Euler-Mascheroni constant
        'catalan': 0.9159655941772190,      # Catalan's constant
        'apery': 1.2020569031595942,        # Apéry's constant (ζ(3))
        'khinchin': 2.6854520010653062,     # Khinchin's constant
        'glaisher': 1.2824271291006226,     # Glaisher-Kinkelin constant
    }

async def print_arithmetic_summary():
    """Print a comprehensive summary of all arithmetic functions (async)."""
    categories = await get_arithmetic_functions()
    constants_dict = get_arithmetic_constants()
    
    print("🔢 Arithmetic Functions Module (Async Native)")
    print("=" * 45)
    
    total_functions = sum(len(cat) for cat in categories.values())
    print(f"📊 Total Arithmetic Functions: {total_functions}")
    print("🚀 All functions are async native for optimal performance")
    print()
    
    for category_name, functions in categories.items():
        if functions:
            print(f"📐 {category_name.replace('_', ' ').title()} ({len(functions)} functions):")
            
            for func_name, spec in sorted(functions.items()):
                # Show execution modes and features
                modes = "/".join([mode.value for mode in spec.execution_modes])
                features = ""
                if spec.cache_strategy.value != "none":
                    features += " 💾"
                if spec.supports_streaming:
                    features += " 🌊"
                if spec.estimated_cpu_usage.value == "high":
                    features += " ⚡"
                
                print(f"   • {func_name} - {modes}{features} (async)")
                
                # Show brief description
                if spec.description:
                    desc = spec.description[:50] + "..." if len(spec.description) > 50 else spec.description
                    print(f"     {desc}")
            print()
    
    print(f"🔢 Mathematical Constants ({len(constants_dict)}):")
    for name, value in sorted(constants_dict.items()):
        if name in ['pi', 'e', 'golden_ratio']:
            print(f"   • {name}: {value:.10f} ⭐")
        else:
            print(f"   • {name}: {value:.10f}")

def get_recommended_functions(use_case: str) -> List[str]:
    """Get recommended arithmetic functions for specific use cases."""
    recommendations = {
        'basic_math': [
            'add', 'subtract', 'multiply', 'divide', 'power', 'sqrt',
            'abs_value', 'round_number', 'minimum', 'maximum'
        ],
        'data_analysis': [
            'arithmetic_sum', 'harmonic_series', 'sort_numbers', 'rank_numbers',
            'find_differences', 'is_arithmetic', 'is_geometric'
        ],
        'number_properties': [
            'is_prime', 'prime_factors', 'gcd', 'lcm', 'divisors',
            'is_even', 'is_odd', 'is_perfect_square', 'factorial'
        ],
        'sequences': [
            'arithmetic_sequence', 'geometric_sequence', 'fibonacci',
            'triangular_numbers', 'square_numbers', 'cube_numbers'
        ],
        'comparisons': [
            'equal', 'less_than', 'greater_than', 'approximately_equal',
            'clamp', 'in_range', 'between', 'close_to_zero'
        ],
        'advanced_arithmetic': [
            'nth_root', 'mod_power', 'collatz_steps', 'geometric_sum',
            'power_series_sum', 'next_prime', 'nth_prime'
        ],
        'cryptography': [
            'mod_power', 'gcd', 'is_prime', 'next_prime', 'prime_factors'
        ],
        'algorithm_design': [
            'factorial', 'fibonacci', 'gcd', 'lcm', 'is_power_of_two',
            'modulo', 'divmod_operation'
        ]
    }
    
    return recommendations.get(use_case.lower(), [])

def arithmetic_quick_reference() -> str:
    """Generate a quick reference guide for arithmetic functions."""
    reference = """
🔢 Arithmetic Functions Quick Reference (Async Native)

📐 BASIC OPERATIONS (await required)
   await add(a, b)           - Addition: a + b
   await subtract(a, b)      - Subtraction: a - b  
   await multiply(a, b)      - Multiplication: a × b
   await divide(a, b)        - Division: a ÷ b
   await power(base, exp)    - Exponentiation: base^exp
   await sqrt(x)             - Square root: √x
   await abs_value(x)        - Absolute value: |x|

🔍 COMPARISONS (await required)
   await equal(a, b)         - Check if a == b
   await less_than(a, b)     - Check if a < b
   await minimum(a, b)       - Return smaller value
   await maximum(a, b)       - Return larger value
   await clamp(val, min, max)- Constrain value to range

🔢 NUMBER THEORY (await required)
   await is_prime(n)         - Check if n is prime
   await gcd(a, b)           - Greatest common divisor
   await lcm(a, b)           - Least common multiple
   await factorial(n)        - n! = n × (n-1) × ... × 1
   await fibonacci(n)        - nth Fibonacci number

📊 SEQUENCES (await required)
   await arithmetic_sequence(a, d, n)  - Generate arithmetic sequence
   await geometric_sequence(a, r, n)   - Generate geometric sequence
   await triangular_numbers(n)         - First n triangular numbers
   await harmonic_series(n)            - Sum 1 + 1/2 + ... + 1/n

🔬 ADVANCED OPERATIONS (await required)
   await ln(x), await log(x, base) - Natural and custom base logarithms
   await log10(x), await log2(x)   - Common and binary logarithms  
   await exp(x)              - e^x exponential function
   await product(list)       - Multiply all numbers in list
   await sum_squares(list)   - Sum of squares: Σx²

🎲 RANDOM FUNCTIONS (await required)
   await random_float()      - Random number 0 ≤ x < 1
   await random_int(min, max)- Random integer in range
   await random_array(r, c)  - 2D array of random numbers

🔢 CONSTANTS (await required for functions)
   await pi(), await e()           - Mathematical constants π and e
   await golden_ratio()      - Golden ratio φ ≈ 1.618
   await sqrt2(), await sqrt3()    - Square roots of 2 and 3
   await ln2(), await ln10()       - Natural logs of 2 and 10

🚀 ASYNC PERFORMANCE TIPS
   • All functions use await for async execution
   • Large computations automatically yield control
   • Caching available for expensive operations
   • Streaming support for data-heavy functions
   
🎯 COMMON ASYNC PATTERNS
   await round_number(x, decimals)     - Round to decimal places
   await floor(x), await ceil(x)       - Round down/up to integer
   await is_even(n), await is_odd(n)   - Check parity
   await sort_numbers(list)            - Sort number list
   """
    return reference.strip()

async def validate_arithmetic_installation():
    """Validate that all arithmetic functions are properly installed and working (async)."""
    categories = await get_arithmetic_functions()
    
    issues = []
    total_functions = 0
    
    for category_name, functions in categories.items():
        total_functions += len(functions)
        
        for func_name, spec in functions.items():
            # Check if function reference exists
            if not spec._function_ref:
                issues.append(f"{category_name}.{func_name}: Missing function reference")
            
            # Check basic properties
            if not spec.description:
                issues.append(f"{category_name}.{func_name}: Missing description")
            
            if not spec.examples:
                issues.append(f"{category_name}.{func_name}: Missing examples")
    
    print("🔍 Arithmetic Module Validation (Async Native)")
    print("=" * 45)
    print(f"📊 Total Functions Checked: {total_functions}")
    
    if issues:
        print(f"⚠️  Issues Found: {len(issues)}")
        for issue in issues[:10]:  # Show first 10 issues
            print(f"   • {issue}")
        if len(issues) > 10:
            print(f"   ... and {len(issues) - 10} more issues")
        return False
    else:
        print("✅ All arithmetic functions validated successfully!")
        print("🚀 All functions are async native and ready for high-performance execution")
        return True

async def get_function_examples(category: str = None) -> Dict[str, List[Dict]]:
    """Get usage examples for arithmetic functions (async)."""
    categories = await get_arithmetic_functions()
    
    examples = {}
    
    target_categories = [category] if category else categories.keys()
    
    for cat_name in target_categories:
        if cat_name in categories:
            cat_examples = {}
            for func_name, spec in categories[cat_name].items():
                if spec.examples:
                    cat_examples[func_name] = spec.examples
            if cat_examples:
                examples[cat_name] = cat_examples
    
    return examples

# Export all arithmetic submodules and utilities
__all__ = [
    # Submodules
    'basic_operations', 'comparison', 'number_theory', 'sequences', 
    'advanced_operations', 'constants',
    
    # Utility functions
    'get_arithmetic_functions', 'get_arithmetic_constants', 
    'print_arithmetic_summary', 'get_function_examples',
    'validate_arithmetic_installation', 'get_recommended_functions',
    'arithmetic_quick_reference'
]

# Quick function access - import commonly used functions at module level
from .basic_operations import (
    add, subtract, multiply, divide, power, sqrt, abs_value, round_number
)
from .comparison import (
    equal, less_than, greater_than, minimum, maximum, clamp
)
from .number_theory import (
    is_prime, gcd, lcm, factorial, fibonacci, is_even, is_odd
)
from .sequences import (
    arithmetic_sequence, geometric_sequence, triangular_numbers
)
from .advanced_operations import (
    ln, log, log10, exp, product, random_float, random_int
)
from .constants import (
    pi, e, golden_ratio, sqrt2
)

# Add commonly used functions to __all__
__all__.extend([
    # Quick access functions (all async)
    'add', 'subtract', 'multiply', 'divide', 'power', 'sqrt', 'abs_value', 'round_number',
    'equal', 'less_than', 'greater_than', 'minimum', 'maximum', 'clamp',
    'is_prime', 'gcd', 'lcm', 'factorial', 'fibonacci', 'is_even', 'is_odd',
    'arithmetic_sequence', 'geometric_sequence', 'triangular_numbers',
    'ln', 'log', 'log10', 'exp', 'product', 'random_float', 'random_int',
    'pi', 'e', 'golden_ratio', 'sqrt2'
])

if __name__ == "__main__":
    import asyncio
    
    async def main():
        await print_arithmetic_summary()
        print("\n" + "="*50)
        print(arithmetic_quick_reference())
        print("\n" + "="*50)
        await validate_arithmetic_installation()
    
    asyncio.run(main())