# #!/usr/bin/env python3
# # chuk_mcp_functions/math/arithmetic/__init__.py
# """
# Arithmetic Functions Module for AI Models (Async Native) - REORGANIZED

# A comprehensive collection of arithmetic functions organized into logical groupings.
# This module provides fundamental mathematical operations that AI models need for
# numerical computation, pattern recognition, and mathematical reasoning.

# All functions are async native for optimal performance in async environments.

# NEW STRUCTURE:
# ├── core/                    - Fundamental operations
# │   ├── basic_operations.py  - add, subtract, multiply, divide, power, sqrt, abs_value, sign, negate
# │   ├── rounding.py         - round_number, floor, ceil, truncate, mround, ceiling_multiple, floor_multiple
# │   └── modular.py          - modulo, divmod_operation, mod_power, quotient, remainder, fmod
# ├── comparison/              - Comparison and ordering
# │   ├── relational.py       - equal, less_than, greater_than, in_range, between
# │   ├── extrema.py          - minimum, maximum, clamp, sort_numbers, rank_numbers
# │   └── tolerance.py        - approximately_equal, close_to_zero, is_finite, is_nan, is_infinite
# ├── number_theory/           - Integer properties and prime numbers
# │   ├── primes.py           - is_prime, next_prime, nth_prime, prime_factors, prime_count
# │   ├── divisibility.py     - gcd, lcm, divisors, is_divisible, is_even, is_odd
# │   ├── special_numbers.py  - is_perfect_square, is_power_of_two, fibonacci, factorial
# │   └── sequences.py        - collatz_steps, triangular_numbers, perfect_numbers
# ├── sequences/               - Mathematical sequences and series
# │   ├── arithmetic.py       - arithmetic_sequence, arithmetic_sum
# │   ├── geometric.py        - geometric_sequence, geometric_sum
# │   ├── series.py           - harmonic_series, power_series_sum, series_sum
# │   ├── polygonal.py        - triangular_numbers, square_numbers, cube_numbers
# │   └── analysis.py         - find_differences, is_arithmetic, is_geometric
# ├── advanced/                - Advanced mathematical operations
# │   ├── logarithmic.py      - ln, log, log10, log2, exp
# │   ├── base_conversion.py  - decimal_to_base, base_to_decimal, arabic_to_roman, roman_to_arabic
# │   └── floating_point.py   - hypot, fma, ldexp, frexp, copysign
# ├── statistics/              - Statistical functions
# │   ├── descriptive.py      - mean, median, mode, variance, std_dev
# │   ├── aggregation.py      - sum_product, sum_squares, product
# │   └── distributions.py    - normal_pdf, binomial_pmf
# ├── random/                  - Random number generation
# │   ├── generators.py       - random_float, random_int, random_array
# │   ├── sampling.py         - random_choice, random_sample, random_shuffle
# │   └── distributions.py    - random_gaussian, random_uniform
# ├── constants/               - Mathematical constants
# │   ├── fundamental.py      - pi, e, tau, golden_ratio, silver_ratio
# │   ├── roots.py           - sqrt2, sqrt3, sqrt5, cbrt2, cbrt3
# │   ├── logarithmic.py     - ln2, ln10, log2e, log10e
# │   ├── special.py         - euler_gamma, catalan, apery, khinchin, glaisher
# │   └── limits.py          - infinity, nan, machine_epsilon, max_float, min_float
# └── utilities/              - Utility functions
#     ├── validation.py      - type checking, range validation
#     ├── precision.py       - decimal precision utilities
#     └── formatting.py      - number formatting, display utilities
#!/usr/bin/env python3
# src/chuk_mcp_functions/math/arithmetic/__init__.py
"""
Arithmetic Functions Library - REORGANIZED STRUCTURE ONLY

Mathematical arithmetic operations organized into logical categories.
This version ONLY imports from the reorganized structure:
- core/
- comparison/  
- number_theory/

Ignores old flat files to avoid conflicts.
"""
#!/usr/bin/env python3
# src/chuk_mcp_functions/math/arithmetic/__init__.py
"""
Arithmetic Functions Library - REORGANIZED STRUCTURE ONLY

Mathematical arithmetic operations organized into logical categories.
This version ONLY imports from the reorganized structure:
- core/
- comparison/  
- number_theory/

Ignores old flat files to avoid conflicts.
"""

# Import ONLY from reorganized structure to avoid conflicts with old files
try:
    from . import core
    _core_available = True
except ImportError as e:
    print(f"Warning: Could not import core: {e}")
    _core_available = False

try:
    from . import comparison
    _comparison_available = True
except ImportError as e:
    print(f"Warning: Could not import comparison: {e}")
    _comparison_available = False

try:
    from . import number_theory
    _number_theory_available = True
except ImportError as e:
    print(f"Warning: Could not import number_theory: {e}")
    _number_theory_available = False

# Import specific functions for backward compatibility - ONLY from reorganized modules
functions_imported = []

# Core functions
if _core_available:
    try:
        from .core.basic_operations import add, subtract, multiply, divide, power, sqrt, abs_value, sign, negate
        functions_imported.extend(['add', 'subtract', 'multiply', 'divide', 'power', 'sqrt', 'abs_value', 'sign', 'negate'])
    except ImportError as e:
        print(f"Warning: Could not import core.basic_operations: {e}")
    
    try:
        from .core.rounding import round_number, floor, ceil, truncate, mround
        functions_imported.extend(['round_number', 'floor', 'ceil', 'truncate', 'mround'])
    except ImportError as e:
        print(f"Warning: Could not import core.rounding: {e}")
    
    try:
        from .core.modular import modulo, mod_power, quotient
        functions_imported.extend(['modulo', 'mod_power', 'quotient'])
    except ImportError as e:
        print(f"Warning: Could not import core.modular: {e}")

# Comparison functions
if _comparison_available:
    try:
        from .comparison.relational import equal, less_than, greater_than, in_range, between
        functions_imported.extend(['equal', 'less_than', 'greater_than', 'in_range', 'between'])
    except ImportError as e:
        print(f"Warning: Could not import comparison.relational: {e}")
    
    try:
        from .comparison.extrema import minimum, maximum, clamp, sort_numbers
        functions_imported.extend(['minimum', 'maximum', 'clamp', 'sort_numbers'])
    except ImportError as e:
        print(f"Warning: Could not import comparison.extrema: {e}")
    
    try:
        from .comparison.tolerance import approximately_equal, is_finite, is_nan
        functions_imported.extend(['approximately_equal', 'is_finite', 'is_nan'])
    except ImportError as e:
        print(f"Warning: Could not import comparison.tolerance: {e}")

# Number theory functions
if _number_theory_available:
    try:
        from .number_theory.primes import is_prime, next_prime, prime_factors
        functions_imported.extend(['is_prime', 'next_prime', 'prime_factors'])
    except ImportError as e:
        print(f"Warning: Could not import number_theory.primes: {e}")
    
    try:
        from .number_theory.divisibility import gcd, lcm, is_even, is_odd, divisors
        functions_imported.extend(['gcd', 'lcm', 'is_even', 'is_odd', 'divisors'])
    except ImportError as e:
        print(f"Warning: Could not import number_theory.divisibility: {e}")

# Build __all__ with only available modules and functions
__all__ = []

# Add available modules
if _core_available:
    __all__.append('core')
if _comparison_available:
    __all__.append('comparison')
if _number_theory_available:
    __all__.append('number_theory')

# Add successfully imported functions
__all__.extend(functions_imported)

def print_reorganized_status():
    """Print the status of the reorganized arithmetic library."""
    print("🔢 Arithmetic Library - REORGANIZED STRUCTURE ONLY")
    print("=" * 50)
    print(f"📐 Core Operations: {_core_available}")
    print(f"🔍 Comparison Operations: {_comparison_available}")
    print(f"🔢 Number Theory Operations: {_number_theory_available}")
    print(f"📊 Available modules: {len([m for m in ['core', 'comparison', 'number_theory'] if m in __all__])}")
    print(f"📊 Available functions: {len(functions_imported)}")
    print()
    
    if functions_imported:
        print(f"✅ Successfully imported functions:")
        for func in functions_imported[:10]:  # Show first 10
            print(f"   • {func}")
        if len(functions_imported) > 10:
            print(f"   ... and {len(functions_imported) - 10} more")
    
    print()
    print("📁 Structure:")
    if _core_available:
        print("   📐 core/ - basic_operations, rounding, modular")
    if _comparison_available:
        print("   🔍 comparison/ - relational, extrema, tolerance")
    if _number_theory_available:
        print("   🔢 number_theory/ - primes, divisibility")

def get_reorganized_modules():
    """Get list of available reorganized modules."""
    available = []
    if _core_available:
        available.append('core')
    if _comparison_available:
        available.append('comparison')
    if _number_theory_available:
        available.append('number_theory')
    return available

# For debugging - show what was imported
if __name__ == "__main__":
    print_reorganized_status()