#!/usr/bin/env python3
"""
Systematic debugging script to find the exact issue in trigonometry module
"""

import sys
import traceback
import ast
import os

def test_file_syntax(filepath):
    """Test if a Python file has valid syntax."""
    print(f"Testing syntax of {filepath}...")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            code = f.read()
        
        # Check for common issues
        if 'null' in code:
            print(f"❌ Found 'null' in {filepath} - should be 'None'")
            # Find line numbers with null
            lines = code.split('\n')
            for i, line in enumerate(lines, 1):
                if 'null' in line:
                    print(f"   Line {i}: {line.strip()}")
            return False
        
        # Parse AST
        ast.parse(code)
        print(f"✅ {filepath} syntax is valid")
        return True
        
    except SyntaxError as e:
        print(f"❌ Syntax Error in {filepath}: {e}")
        print(f"   Line {e.lineno}: {e.text}")
        return False
    except FileNotFoundError:
        print(f"❌ File not found: {filepath}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error in {filepath}: {e}")
        return False

def test_module_import(module_path):
    """Test importing a module step by step."""
    print(f"\nTesting import of {module_path}...")
    
    try:
        # Try importing the module
        module = __import__(module_path, fromlist=[''])
        print(f"✅ Successfully imported {module_path}")
        
        # List available attributes
        attrs = [attr for attr in dir(module) if not attr.startswith('_')]
        print(f"   Available attributes: {attrs[:10]}{'...' if len(attrs) > 10 else ''}")
        
        return True, module
        
    except ImportError as e:
        print(f"❌ ImportError in {module_path}: {e}")
        return False, None
    except Exception as e:
        print(f"❌ Unexpected error importing {module_path}: {e}")
        traceback.print_exc()
        return False, None

def find_trigonometry_files():
    """Find all trigonometry Python files."""
    base_path = "src/chuk_mcp_functions/math/trigonometry"
    
    if not os.path.exists(base_path):
        # Try alternative paths
        alt_paths = [
            "chuk_mcp_functions/math/trigonometry",
            "math/trigonometry",
            "trigonometry"
        ]
        
        for alt_path in alt_paths:
            if os.path.exists(alt_path):
                base_path = alt_path
                break
        else:
            print(f"❌ Could not find trigonometry directory in any of: {alt_paths}")
            return []
    
    files = []
    for file in os.listdir(base_path):
        if file.endswith('.py') and not file.startswith('__'):
            files.append(os.path.join(base_path, file))
    
    return files

def main():
    print("🔍 Systematic Trigonometry Module Debug")
    print("=" * 50)
    
    # Step 1: Find and test syntax of all trigonometry files
    print("\n1. Testing syntax of all trigonometry files...")
    trig_files = find_trigonometry_files()
    
    if not trig_files:
        print("❌ No trigonometry files found!")
        return
    
    syntax_ok = True
    for file in trig_files:
        if not test_file_syntax(file):
            syntax_ok = False
    
    if not syntax_ok:
        print("\n❌ Syntax errors found! Fix these before proceeding.")
        return
    
    print("\n✅ All files have valid syntax!")
    
    # Step 2: Test basic imports
    print("\n2. Testing basic imports...")
    
    basic_imports = [
        "math",
        "asyncio", 
        "typing",
    ]
    
    for module in basic_imports:
        try:
            __import__(module)
            print(f"✅ {module}")
        except ImportError as e:
            print(f"❌ {module}: {e}")
    
    # Step 3: Test mcp_decorator
    print("\n3. Testing mcp_decorator...")
    try:
        from chuk_mcp_functions.mcp_decorator import mcp_function
        print("✅ mcp_decorator import successful")
    except ImportError as e:
        print(f"❌ mcp_decorator failed: {e}")
        print("This might be the root cause!")
        return
    
    # Step 4: Test individual trigonometry modules
    print("\n4. Testing individual trigonometry modules...")
    
    modules_to_test = [
        "chuk_mcp_functions.math.trigonometry.basic_functions",
        "chuk_mcp_functions.math.trigonometry.inverse_functions", 
        "chuk_mcp_functions.math.trigonometry.hyperbolic",
        "chuk_mcp_functions.math.trigonometry.inverse_hyperbolic",
        "chuk_mcp_functions.math.trigonometry.angle_conversion",
        "chuk_mcp_functions.math.trigonometry.identities",
        "chuk_mcp_functions.math.trigonometry.wave_analysis",
        "chuk_mcp_functions.math.trigonometry.applications"
    ]
    
    failed_modules = []
    for module_path in modules_to_test:
        success, module = test_module_import(module_path)
        if not success:
            failed_modules.append(module_path)
    
    if failed_modules:
        print(f"\n❌ Failed modules: {failed_modules}")
        print("These are likely the root cause of the import issues.")
    else:
        print("\n✅ All individual modules imported successfully!")
    
    # Step 5: Test the main trigonometry module
    print("\n5. Testing main trigonometry module...")
    success, module = test_module_import("chuk_mcp_functions.math.trigonometry")
    
    if success:
        print("✅ Main trigonometry module imported successfully!")
        
        # Test specific function imports
        print("\n6. Testing specific function imports...")
        functions_to_test = ['sin', 'cos', 'tan', 'asin', 'acos', 'atan']
        
        for func_name in functions_to_test:
            if hasattr(module, func_name):
                print(f"✅ {func_name} available")
            else:
                print(f"❌ {func_name} not available")
    else:
        print("❌ Main trigonometry module failed to import!")
    
    print("\n" + "=" * 50)
    print("Debug complete!")

if __name__ == "__main__":
    main()