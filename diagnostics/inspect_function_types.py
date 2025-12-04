#!/usr/bin/env python3
# diagnostics/inspect_function_types.py
"""
Function Type Inspector

Inspects what types of functions we actually have and how they behave.
"""

import asyncio
import inspect
import sys
from pathlib import Path

# Setup path
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))


async def inspect_function_types():
    """Inspect the actual function types we have."""
    print("🔍 Function Type Inspector")
    print("=" * 50)

    try:
        # Import some key functions
        from chuk_mcp_math.arithmetic.basic_operations import add, subtract, multiply
        from chuk_mcp_math.number_theory import is_prime, factorial, fibonacci
        from chuk_mcp_math.arithmetic.constants import pi, e

        functions_to_test = {
            "add": add,
            "subtract": subtract,
            "multiply": multiply,
            "is_prime": is_prime,
            "factorial": factorial,
            "fibonacci": fibonacci,
            "pi": pi,
            "e": e,
        }

        print(f"📊 Analyzing {len(functions_to_test)} functions:")

        for name, func in functions_to_test.items():
            print(f"\n🔧 {name}:")
            print(f"   Type: {type(func)}")
            print(f"   Is coroutine function: {inspect.iscoroutinefunction(func)}")
            print(f"   Is function: {inspect.isfunction(func)}")
            print(f"   Is method: {inspect.ismethod(func)}")
            print(f"   Is builtin: {inspect.isbuiltin(func)}")

            # Check if it has MCP attributes
            if hasattr(func, "_mcp_function_spec"):
                spec = func._mcp_function_spec
                print("   Has MCP spec: ✅")
                print(f"   Is async native: {spec.is_async_native}")
                print(f"   Execution modes: {[m.value for m in spec.execution_modes]}")
            else:
                print("   Has MCP spec: ❌")

            # Test what happens when we call it
            try:
                if name == "add":
                    result = func(2, 3)
                elif name == "subtract":
                    result = func(5, 2)
                elif name == "multiply":
                    result = func(3, 4)
                elif name == "is_prime":
                    result = func(7)
                elif name == "factorial":
                    result = func(5)
                elif name == "fibonacci":
                    result = func(7)
                elif name in ["pi", "e"]:
                    result = func()
                else:
                    result = "not tested"

                print(f"   Call result type: {type(result)}")
                print(f"   Is coroutine: {inspect.iscoroutine(result)}")

                if inspect.iscoroutine(result):
                    awaited_result = await result
                    print(f"   Result (awaited): {awaited_result}")
                else:
                    print(f"   Result (direct): {result}")

            except Exception as e:
                print(f"   Call failed: {e}")

    except ImportError as e:
        print(f"❌ Import failed: {e}")


async def test_async_calls():
    """Test making async calls to functions."""
    print("\n🚀 Testing Async Calls")
    print("=" * 30)

    try:
        from chuk_mcp_math.arithmetic.basic_operations import add, multiply
        from chuk_mcp_math.number_theory import is_prime, fibonacci

        functions_and_args = [
            (add, (2, 3), "add(2, 3)"),
            (multiply, (4, 5), "multiply(4, 5)"),
            (is_prime, (17,), "is_prime(17)"),
            (fibonacci, (10,), "fibonacci(10)"),
        ]

        print("Testing individual async calls:")
        for func, args, desc in functions_and_args:
            try:
                result = func(*args)
                print(
                    f"   {desc}: type={type(result)}, coroutine={inspect.iscoroutine(result)}"
                )

                if inspect.iscoroutine(result):
                    awaited_result = await result
                    print(f"      Awaited result: {awaited_result}")
                else:
                    print(f"      Direct result: {result}")

            except Exception as e:
                print(f"   {desc}: ERROR - {e}")

        print("\nTesting asyncio.gather:")

        # Try to create coroutines for gather
        coroutines = []
        for func, args, desc in functions_and_args:
            result = func(*args)
            if inspect.iscoroutine(result):
                coroutines.append(result)
                print(f"   ✅ {desc} - added to gather")
            else:
                print(f"   ❌ {desc} - not a coroutine: {type(result)}")

        if coroutines:
            print(f"   Gathering {len(coroutines)} coroutines...")
            results = await asyncio.gather(*coroutines)
            print(f"   Gather results: {results}")
        else:
            print("   ❌ No coroutines to gather!")

    except Exception as e:
        print(f"❌ Async test failed: {e}")


async def main():
    """Main inspector."""
    await inspect_function_types()
    await test_async_calls()

    print("\n💡 Summary:")
    print("If functions are not coroutines, they need to be fixed in the MCP decorator")
    print("or the async wrapper isn't working properly.")


if __name__ == "__main__":
    asyncio.run(main())
