#!/usr/bin/env python3
# simple_diagnostic.py
"""
Simple diagnostic to see what's actually in the current directory
"""

from pathlib import Path


def check_directory_structure():
    """Check the actual directory structure."""
    print("🔍 DIRECTORY STRUCTURE DIAGNOSTIC")
    print("=" * 40)

    current_dir = Path.cwd()
    print(f"📁 Current directory: {current_dir}")
    print()

    print("📂 Contents of current directory:")
    for item in sorted(current_dir.iterdir()):
        if item.is_dir():
            print(f"  📁 {item.name}/")
        else:
            print(f"  📄 {item.name}")
    print()

    # Check for src directory
    src_dir = current_dir / "src"
    if src_dir.exists():
        print("✅ Found src/ directory")
        print("📂 Contents of src/:")
        for item in sorted(src_dir.iterdir()):
            if item.is_dir():
                print(f"  📁 {item.name}/")
            else:
                print(f"  📄 {item.name}")

        # Check for chuk_mcp_math
        chuk_dir = src_dir / "chuk_mcp_math"
        if chuk_dir.exists():
            print("\n✅ Found src/chuk_mcp_math/")
            print("📂 Contents of src/chuk_mcp_math/:")
            for item in sorted(chuk_dir.iterdir()):
                if item.is_dir():
                    print(f"  📁 {item.name}/")
                else:
                    print(f"  📄 {item.name}")

            # Check for math/arithmetic
            math_dir = chuk_dir / "math"
            if math_dir.exists():
                print("\n✅ Found src/chuk_mcp_math/math/")
                arith_dir = math_dir / "arithmetic"
                if arith_dir.exists():
                    print("✅ Found src/chuk_mcp_math/math/arithmetic/")
                    print("📂 Contents of src/chuk_mcp_math/math/arithmetic/:")
                    for item in sorted(arith_dir.iterdir()):
                        if item.is_dir():
                            print(f"  📁 {item.name}/")
                            # Check if it has __init__.py
                            init_file = item / "__init__.py"
                            if init_file.exists():
                                print("      ✅ Has __init__.py")
                            else:
                                print("      ❌ No __init__.py")
                        else:
                            print(f"  📄 {item.name}")
                else:
                    print("❌ No src/chuk_mcp_math/math/arithmetic/")
            else:
                print("❌ No src/chuk_mcp_math/math/")
        else:
            print("❌ No src/chuk_mcp_math/")
    else:
        print("❌ No src/ directory found")

        # Check if chuk_mcp_math is directly in current directory
        direct_chuk = current_dir / "chuk_mcp_math"
        if direct_chuk.exists():
            print("✅ Found chuk_mcp_math/ directly in current directory")
            print("📂 Contents of chuk_mcp_math/:")
            for item in sorted(direct_chuk.iterdir()):
                if item.is_dir():
                    print(f"  📁 {item.name}/")
                else:
                    print(f"  📄 {item.name}")
        else:
            print("❌ No chuk_mcp_math/ in current directory either")


def try_imports():
    """Try different import approaches."""
    print("\n🧪 TESTING IMPORTS")
    print("=" * 20)

    import sys

    current_dir = Path.cwd()

    # Try adding current directory to path
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))
        print("✅ Added current directory to Python path")

    # Try adding src to path
    src_dir = current_dir / "src"
    if src_dir.exists() and str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
        print("✅ Added src/ to Python path")

    print(f"🐍 Python path (first 5): {sys.path[:5]}")

    # Try importing
    try:
        import chuk_mcp_math

        print("✅ Successfully imported chuk_mcp_math")
        print(f"   Location: {chuk_mcp_math.__file__}")

        try:
            import chuk_mcp_math

            print("✅ Successfully imported chuk_mcp_math")

            try:
                import chuk_mcp_math.arithmetic

                print("✅ Successfully imported chuk_mcp_math.arithmetic")

                # Check what's available
                arith = chuk_mcp_math.arithmetic
                attrs = [attr for attr in dir(arith) if not attr.startswith("_")]
                print(f"   Available attributes: {attrs[:10]}...")  # Show first 10

            except Exception as e:
                print(f"❌ Failed to import arithmetic: {e}")
        except Exception as e:
            print(f"❌ Failed to import math: {e}")
    except Exception as e:
        print(f"❌ Failed to import chuk_mcp_math: {e}")


def main():
    """Main diagnostic function."""
    check_directory_structure()
    try_imports()

    print("\n💡 RECOMMENDATIONS:")
    print("1. Make sure the chuk_mcp_math package exists")
    print("2. Make sure all directories have __init__.py files")
    print("3. Check that the reorganized structure is in place")


if __name__ == "__main__":
    main()
