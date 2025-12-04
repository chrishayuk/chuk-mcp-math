#!/usr/bin/env python3
# diagnostics/diagnose_reorganized_structure.py
"""
Test the reorganized arithmetic structure from the diagnostics folder.
"""

import asyncio
import sys
from pathlib import Path

# We're in diagnostics/, so project root is one level up
project_root = Path(__file__).parent.parent
src_path = project_root / "src"

print("🔍 PATH SETUP FROM DIAGNOSTICS:")
print(f"   Diagnostics folder: {Path(__file__).parent}")
print(f"   Project root: {project_root}")
print(f"   src path: {src_path}")
print(f"   src exists: {src_path.exists()}")

# Verify we have the right structure
if src_path.exists():
    chuk_path = src_path / "chuk_mcp_math"
    print(f"   chuk_mcp_math exists: {chuk_path.exists()}")

    if chuk_path.exists():
        # Check key files
        init_file = chuk_path / "__init__.py"
        math_dir = chuk_path / "math"
        arith_dir = math_dir / "arithmetic" if math_dir.exists() else None

        print(f"   chuk_mcp_math/__init__.py: {init_file.exists()}")
        print(f"   chuk_mcp_math/math/: {math_dir.exists() if math_dir else False}")
        print(
            f"   chuk_mcp_math/math/arithmetic/: {arith_dir.exists() if arith_dir else False}"
        )

        if arith_dir and arith_dir.exists():
            # Check reorganized structure
            core_dir = arith_dir / "core"
            comp_dir = arith_dir / "comparison"
            nt_dir = arith_dir / "number_theory"

            print(f"   📁 core/: {core_dir.exists()}")
            print(f"   📁 comparison/: {comp_dir.exists()}")
            print(f"   📁 number_theory/: {nt_dir.exists()}")

            # Check for key files in reorganized structure
            if core_dir.exists():
                basic_ops = core_dir / "basic_operations.py"
                print(f"      core/basic_operations.py: {basic_ops.exists()}")

# Add src to Python path
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))
    print(f"✅ Added {src_path} to Python path")

print()


async def test_imports():
    """Test the imports step by step."""
    print("🧪 STEP-BY-STEP IMPORT TEST")
    print("=" * 30)

    # Step 1: Import root package
    try:
        import chuk_mcp_math

        print("✅ Step 1: import chuk_mcp_math")
        print(f"   Location: {chuk_mcp_math.__file__}")
    except Exception as e:
        print(f"❌ Step 1 failed: {e}")
        return False

    # Step 2: Import math
    try:
        import chuk_mcp_math

        print("✅ Step 2: import chuk_mcp_math")
    except Exception as e:
        print(f"❌ Step 2 failed: {e}")
        return False

    # Step 3: Import arithmetic
    try:
        import chuk_mcp_math.arithmetic as arithmetic

        print("✅ Step 3: import chuk_mcp_math.arithmetic")

        # Show what's available
        attrs = [attr for attr in dir(arithmetic) if not attr.startswith("_")]
        print(f"   Available: {attrs[:10]}...")  # Show first 10

        # Test status function if available
        if hasattr(arithmetic, "print_reorganized_status"):
            print("\n📊 Arithmetic Module Status:")
            arithmetic.print_reorganized_status()

    except Exception as e:
        print(f"❌ Step 3 failed: {e}")
        print("   This is likely the __init__.py issue")
        import traceback

        traceback.print_exc()
        return False

    # Step 4: Test reorganized imports
    print("\n🏗️  Testing reorganized structure:")

    reorganized_tests = [
        ("core", "chuk_mcp_math.arithmetic.core"),
        ("core.basic_operations", "chuk_mcp_math.arithmetic.core.basic_operations"),
        ("comparison", "chuk_mcp_math.arithmetic.comparison"),
        ("number_theory", "chuk_mcp_math.number_theory"),
    ]

    working_modules = []

    for name, module_path in reorganized_tests:
        try:
            exec(f"import {module_path}")
            print(f"   ✅ {name}")
            working_modules.append(name)
        except Exception as e:
            print(f"   ❌ {name}: {e}")

    # Step 5: Test function calls
    if "core.basic_operations" in working_modules:
        print("\n🔧 Testing function calls:")
        try:
            from chuk_mcp_math.arithmetic.core.basic_operations import add

            print("   ✅ Imported add function")

            # Test the function
            result = add(5, 3)
            if asyncio.iscoroutine(result):
                result = await result

            print(f"   ✅ add(5, 3) = {result}")

        except Exception as e:
            print(f"   ❌ Function test failed: {e}")
            import traceback

            traceback.print_exc()

    return len(working_modules) > 0


async def main():
    """Main test."""
    success = await test_imports()

    print("\n📊 RESULTS:")
    if success:
        print("✅ Some parts of the reorganized structure are working!")
        print("🎯 You can now work on fixing any remaining issues")
    else:
        print("❌ Basic imports are failing")
        print("🔧 The main issue is likely in the arithmetic __init__.py file")
        print("💡 Try replacing it with the clean version provided")


if __name__ == "__main__":
    asyncio.run(main())
