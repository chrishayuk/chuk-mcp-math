#!/usr/bin/env python3
"""Unified CLI entry point with subcommand routing."""

import click
import sys
import importlib
import inspect
from typing import Optional, Dict, Any, List, Tuple

# Suppress info logs for cleaner CLI output
import logging

logging.getLogger("chuk_mcp_math").setLevel(logging.WARNING)


@click.group()
def cli():
    """CHUK MCP Math - Command Line Interface

    Mathematical functions accessible from the command line.
    """
    pass


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
                    elif hasattr(obj, "_mcp_metadata"):
                        metadata = obj._mcp_metadata

                    if metadata or hasattr(obj, "_mcp_function_spec"):
                        func_key = f"{module_name.split('.')[-1]}.{name}"
                        functions[func_key] = {
                            "function": obj,
                            "module": module_name,
                            "metadata": metadata,
                            "signature": str(inspect.signature(obj)),
                        }
        except ImportError:
            # Debug: print import errors
            # print(f"Failed to import {module_name}: {e}", file=sys.stderr)
            continue

    return functions


@cli.command()
@click.argument("function_name")
@click.argument("args", nargs=-1)
@click.option("--json-output", is_flag=True, help="Output as JSON")
def call(function_name: str, args, json_output: bool):
    """Call a math function by name.

    Examples:
        chuk call primes.is_prime 17
        chuk call basic_operations.add 5 3
    """
    functions = discover_all_functions()

    # Try direct match first
    func_info = functions.get(function_name)

    # Try without module prefix
    if not func_info:
        for key, info in functions.items():
            if key.endswith(f".{function_name}"):
                func_info = info
                break

    if not func_info:
        click.echo(f"Function '{function_name}' not found", err=True)
        click.echo("Use 'chuk list' to see available functions", err=True)
        sys.exit(1)

    func = func_info["function"]
    from chuk_mcp_math.cli import CLIWrapper

    wrapper = CLIWrapper(func, function_name)
    result_code = wrapper.run(list(args))
    sys.exit(result_code)


@cli.command()
@click.option("--module", help="Filter by module")
@click.option("--detailed", is_flag=True, help="Show detailed information")
def list(module: Optional[str], detailed: bool):
    """List available functions."""
    functions = discover_all_functions()

    if not functions:
        click.echo("No functions found. Make sure modules are properly installed.")
        return

    # Group by module
    by_module: Dict[str, List[Tuple[str, Dict[str, Any]]]] = {}
    for func_name, info in functions.items():
        module_name = info["module"].split(".")[-1]
        if module and module != module_name:
            continue
        if module_name not in by_module:
            by_module[module_name] = []
        by_module[module_name].append((func_name, info))

    for module_name in sorted(by_module.keys()):
        click.echo(f"\n{module_name}:")
        for func_name, info in sorted(by_module[module_name]):
            short_name = func_name.split(".")[-1]
            if detailed:
                click.echo(f"  {short_name}{info['signature']}")
                if info["metadata"].get("description"):
                    click.echo(f"    {info['metadata']['description']}")
            else:
                desc = info["metadata"].get("description", "No description")
                # Truncate long descriptions
                if len(desc) > 60:
                    desc = desc[:57] + "..."
                click.echo(f"  {short_name}: {desc}")


@cli.command()
@click.argument("query")
def search(query: str):
    """Search functions by keyword."""
    functions = discover_all_functions()
    query_lower = query.lower()

    results = []
    for func_name, info in functions.items():
        score = 0
        short_name = func_name.split(".")[-1]

        # Check function name
        if query_lower in short_name.lower():
            score += 3

        # Check description
        desc = info["metadata"].get("description", "").lower()
        if query_lower in desc:
            score += 2

        # Check module name
        if query_lower in info["module"].lower():
            score += 1

        if score > 0:
            results.append((func_name, info, score))

    if not results:
        click.echo(f"No functions found matching '{query}'")
        return

    # Sort by score (descending) and name
    results.sort(key=lambda x: (-x[2], x[0]))

    click.echo(f"Functions matching '{query}':\n")
    for func_name, info, score in results:
        short_name = func_name.split(".")[-1]
        desc = info["metadata"].get("description", "No description")
        if len(desc) > 50:
            desc = desc[:47] + "..."
        click.echo(f"  {short_name}: {desc}")


@cli.command()
@click.argument("function_name")
def describe(function_name: str):
    """Show detailed information about a function."""
    functions = discover_all_functions()

    # Try direct match first
    func_info = functions.get(function_name)

    # Try without module prefix
    if not func_info:
        for key, info in functions.items():
            if key.endswith(f".{function_name}"):
                func_info = info
                break

    if not func_info:
        click.echo(f"Function '{function_name}' not found", err=True)
        sys.exit(1)

    short_name = function_name.split(".")[-1]
    click.echo(f"Function: {short_name}")
    click.echo(f"Module: {func_info['module']}")
    click.echo(f"Signature: {short_name}{func_info['signature']}")

    if func_info["metadata"].get("description"):
        click.echo("\nDescription:")
        click.echo(f"  {func_info['metadata']['description']}")

    if func_info["metadata"].get("examples"):
        click.echo("\nExamples:")
        for example in func_info["metadata"]["examples"]:
            args = " ".join(str(v) for v in example["input"].values())
            click.echo(f"  $ chuk call {short_name} {args}")
            click.echo(f"    → {example['output']}")
            if example.get("description"):
                click.echo(f"    # {example['description']}")


@cli.command()
def version():
    """Show version information."""
    click.echo("CHUK MCP Math CLI v1.0.0")
    click.echo("Async-native mathematical functions library")


if __name__ == "__main__":
    cli()
