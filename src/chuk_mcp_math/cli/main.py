#!/usr/bin/env python3
"""Unified CLI entry point with subcommand routing."""

import argparse
import sys
import importlib
import inspect
from typing import Dict, Any

# Suppress info logs for cleaner CLI output
import logging

logging.getLogger("chuk_mcp_math").setLevel(logging.WARNING)


def discover_all_functions() -> Dict[str, Any]:
    """Discover all available math functions."""
    functions = {}

    # List of modules to scan (expand this as needed)
    modules_to_scan = [
        "chuk_mcp_math.number_theory.primes",
        "chuk_mcp_math.arithmetic.core.basic_operations",
        "chuk_mcp_math.trigonometry.basic_functions",
    ]

    for module_name in modules_to_scan:
        try:
            module = importlib.import_module(module_name)
            for name, obj in inspect.getmembers(module):
                if callable(obj):
                    # Check for MCP function spec or metadata
                    metadata = {}
                    if hasattr(obj, "_mcp_function_spec"):
                        spec = obj._mcp_function_spec
                        metadata = {
                            "description": spec.description,
                            "examples": spec.examples,
                            "category": spec.category,
                            "namespace": spec.namespace,
                        }

                    if metadata or inspect.isfunction(obj):
                        full_name = f"{module_name}.{name}"
                        functions[full_name] = {"function": obj, "metadata": metadata}
        except ImportError:
            continue

    return functions


def cmd_version(args):
    """Show version information."""
    from .. import __version__

    print(f"chuk-mcp-math version {__version__}")
    return 0


def cmd_list(args):
    """List all available functions."""
    functions = discover_all_functions()

    if args.module:
        # Filter by module
        functions = {k: v for k, v in functions.items() if args.module in k}

    if not functions:
        print("No functions found.")
        return 1

    print(f"\nFound {len(functions)} functions:\n")

    for name, info in sorted(functions.items()):
        if args.detailed and info["metadata"]:
            desc = info["metadata"].get("description", "No description")
            print(f"  • {name}")
            print(f"    {desc}")
            print()
        else:
            print(f"  • {name}")

    return 0


def cmd_search(args):
    """Search for functions by keyword."""
    functions = discover_all_functions()
    keyword = args.keyword.lower()

    matches = {}
    for name, info in functions.items():
        if keyword in name.lower():
            matches[name] = info
        elif info["metadata"]:
            desc = info["metadata"].get("description", "").lower()
            if keyword in desc:
                matches[name] = info

    if not matches:
        print(f"No functions found matching '{args.keyword}'")
        return 1

    print(f"\nFound {len(matches)} matches for '{args.keyword}':\n")
    for name, info in sorted(matches.items()):
        desc = (
            info["metadata"].get("description", "No description")
            if info["metadata"]
            else "No description"
        )
        print(f"  • {name}")
        print(f"    {desc}")
        print()

    return 0


def cmd_describe(args):
    """Describe a specific function."""
    functions = discover_all_functions()

    # Try to find the function
    func_info = None
    for name, info in functions.items():
        if args.function in name or name.endswith(f".{args.function}"):
            func_info = info
            break

    if not func_info:
        print(f"Function '{args.function}' not found.")
        return 1

    print(f"\nFunction: {args.function}")
    print("=" * 60)

    if func_info["metadata"]:
        metadata = func_info["metadata"]
        print(f"\nDescription: {metadata.get('description', 'N/A')}")
        print(f"Category: {metadata.get('category', 'N/A')}")
        print(f"Namespace: {metadata.get('namespace', 'N/A')}")

        if metadata.get("examples"):
            print("\nExamples:")
            for ex in metadata["examples"]:
                print(f"  Input: {ex.get('input', 'N/A')}")
                print(f"  Output: {ex.get('output', 'N/A')}")
                if ex.get("description"):
                    print(f"  Description: {ex['description']}")
                print()

    # Show function signature
    func = func_info["function"]
    sig = inspect.signature(func)
    print(f"\nSignature: {args.function}{sig}")

    # Show docstring
    if func.__doc__:
        print("\nDocumentation:")
        print(func.__doc__)

    return 0


def cmd_call(args):
    """Call a function with arguments."""
    functions = discover_all_functions()

    # Try to find the function
    func_info = None
    for name, info in functions.items():
        if args.function in name or name.endswith(f".{args.function}"):
            func_info = info
            break

    if not func_info:
        print(f"Function '{args.function}' not found.")
        return 1

    func = func_info["function"]

    # Parse arguments
    try:
        # Simple argument parsing (could be improved)
        import json

        parsed_args = []
        for arg in args.args:
            try:
                # Try to parse as JSON first
                parsed_args.append(json.loads(arg))
            except (json.JSONDecodeError, ValueError):
                # Otherwise treat as string
                parsed_args.append(arg)

        # Call the function
        result = func(*parsed_args)

        # Handle async functions
        if inspect.iscoroutine(result):
            import asyncio

            result = asyncio.run(result)

        print(f"\nResult: {result}")
        return 0

    except Exception as e:
        print(f"Error calling function: {e}")
        return 1


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="chuk-mcp-math",
        description="CHUK MCP Math - Mathematical functions from the command line",
    )

    parser.add_argument("--version", action="version", version="%(prog)s 0.1.3")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # List command
    list_parser = subparsers.add_parser("list", help="List all available functions")
    list_parser.add_argument("--module", "-m", help="Filter by module name")
    list_parser.add_argument(
        "--detailed", "-d", action="store_true", help="Show detailed information"
    )
    list_parser.set_defaults(func=cmd_list)

    # Search command
    search_parser = subparsers.add_parser("search", help="Search for functions")
    search_parser.add_argument("keyword", help="Keyword to search for")
    search_parser.set_defaults(func=cmd_search)

    # Describe command
    describe_parser = subparsers.add_parser("describe", help="Describe a function")
    describe_parser.add_argument("function", help="Function name to describe")
    describe_parser.set_defaults(func=cmd_describe)

    # Call command
    call_parser = subparsers.add_parser("call", help="Call a function")
    call_parser.add_argument("function", help="Function name to call")
    call_parser.add_argument("args", nargs="*", help="Arguments to pass to the function")
    call_parser.set_defaults(func=cmd_call)

    # Parse arguments
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 0

    # Call the appropriate command handler
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
