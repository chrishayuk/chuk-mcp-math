#!/usr/bin/env python3
# src/chuk_mcp_math/math/arithmetic/__init__.py
# ruff: noqa: F401
"""
Arithmetic Functions Library - REORGANIZED STRUCTURE ONLY

Mathematical arithmetic operations organized into logical categories.
This version ONLY imports from the reorganized structure:
- core/
- comparison/

Note: number_theory is a separate module at the math level, not part of arithmetic.
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

# REMOVED: number_theory import - it's not part of arithmetic, it's a separate math module
# This was causing the circular import warning:
# try:
#     from . import number_theory  # ❌ WRONG - causes circular import
#     _number_theory_available = True
# except ImportError as e:
#     print(f"Warning: Could not import number_theory: {e}")
#     _number_theory_available = False

# Import specific functions for backward compatibility - ONLY from reorganized modules
functions_imported = []

# Core functions
if _core_available:
    try:
        from .core.basic_operations import (
            add,
            subtract,
            multiply,
            divide,
            power,
            sqrt,
            abs_value,
            sign,
            negate,
        )

        functions_imported.extend(
            [
                "add",
                "subtract",
                "multiply",
                "divide",
                "power",
                "sqrt",
                "abs_value",
                "sign",
                "negate",
            ]
        )
    except ImportError as e:
        print(f"Warning: Could not import core.basic_operations: {e}")

    try:
        from .core.rounding import round_number, floor, ceil, truncate, mround

        functions_imported.extend(
            ["round_number", "floor", "ceil", "truncate", "mround"]
        )
    except ImportError as e:
        print(f"Warning: Could not import core.rounding: {e}")

    try:
        from .core.modular import modulo, mod_power, quotient

        functions_imported.extend(["modulo", "mod_power", "quotient"])
    except ImportError as e:
        print(f"Warning: Could not import core.modular: {e}")

# Comparison functions
if _comparison_available:
    try:
        from .comparison.relational import (
            equal,
            less_than,
            greater_than,
            in_range,
            between,
        )

        functions_imported.extend(
            ["equal", "less_than", "greater_than", "in_range", "between"]
        )
    except ImportError as e:
        print(f"Warning: Could not import comparison.relational: {e}")

    try:
        from .comparison.extrema import minimum, maximum, clamp, sort_numbers

        functions_imported.extend(["minimum", "maximum", "clamp", "sort_numbers"])
    except ImportError as e:
        print(f"Warning: Could not import comparison.extrema: {e}")

    try:
        from .comparison.tolerance import approximately_equal, is_finite, is_nan

        functions_imported.extend(["approximately_equal", "is_finite", "is_nan"])
    except ImportError as e:
        print(f"Warning: Could not import comparison.tolerance: {e}")

# Build __all__ with only available modules and functions
__all__ = []

# Add available modules (only core and comparison are part of arithmetic)
if _core_available:
    __all__.append("core")
if _comparison_available:
    __all__.append("comparison")

# Add successfully imported functions
__all__.extend(functions_imported)


def print_reorganized_status():
    """Print the status of the reorganized arithmetic library."""
    print("🔢 Arithmetic Library - REORGANIZED STRUCTURE ONLY")
    print("=" * 50)
    print(f"📐 Core Operations: {_core_available}")
    print(f"🔍 Comparison Operations: {_comparison_available}")
    print(
        f"📊 Available modules: {len([m for m in ['core', 'comparison'] if m in __all__])}"
    )
    print(f"📊 Available functions: {len(functions_imported)}")
    print()

    if functions_imported:
        print("✅ Successfully imported functions:")
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

    print()
    print("📝 Note: number_theory is a separate module at math level:")
    print("   Use: from chuk_mcp_math import number_theory")
    print("   Not: from chuk_mcp_math.arithmetic import number_theory")


def get_reorganized_modules():
    """Get list of available reorganized modules."""
    available = []
    if _core_available:
        available.append("core")
    if _comparison_available:
        available.append("comparison")
    return available


def get_module_info():
    """Get information about this arithmetic module."""
    return {
        "name": "arithmetic",
        "description": "Core arithmetic operations organized into logical categories",
        "available_modules": get_reorganized_modules(),
        "function_count": len(functions_imported),
        "core_available": _core_available,
        "comparison_available": _comparison_available,
        "note": "number_theory is a separate math module, not part of arithmetic",
    }


# For debugging - show what was imported
if __name__ == "__main__":
    print_reorganized_status()
    print("\n" + "=" * 50)
    print("🔍 Module Information:")
    info = get_module_info()
    for key, value in info.items():
        print(f"  {key}: {value}")
