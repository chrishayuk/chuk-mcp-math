"""CLI wrapper infrastructure for bash-executable math functions."""
# ruff: noqa: F401

import asyncio
import json
import sys
import inspect
from typing import Any, Dict, List, Optional, get_type_hints
from pathlib import Path


class CLIWrapper:
    """Base wrapper for exposing async functions to CLI."""

    def __init__(self, func, name: str):
        self.func = func
        self.name = name
        # Try different metadata attributes
        if hasattr(func, "_mcp_function_spec"):
            self.metadata = (
                func._mcp_function_spec.model_dump()
                if hasattr(func._mcp_function_spec, "model_dump")
                else {}
            )
        else:
            self.metadata = getattr(func, "_mcp_metadata", {})
        self.signature = inspect.signature(func)
        self.type_hints = get_type_hints(func)

    def parse_args(self, args: List[str]) -> Dict[str, Any]:
        """Parse command-line arguments into function parameters."""
        params = {}
        list(self.signature.parameters.keys())

        for i, (param_name, param) in enumerate(self.signature.parameters.items()):
            if i < len(args):
                arg_value = args[i]
                param_type = self.type_hints.get(param_name, str)

                # Type conversion
                if param_type is int:
                    params[param_name] = int(arg_value)  # type: ignore[assignment]
                elif param_type is float:
                    params[param_name] = float(arg_value)  # type: ignore[assignment]
                elif param_type is bool:
                    params[param_name] = arg_value.lower() in ("true", "1", "yes")
                elif param_type is list or param_type is List:
                    # Try to parse as JSON array
                    try:
                        params[param_name] = json.loads(arg_value)  # type: ignore[assignment]
                    except json.JSONDecodeError:
                        # Fall back to comma-separated values
                        params[param_name] = arg_value.split(",")  # type: ignore[assignment]
                else:
                    params[param_name] = arg_value  # type: ignore[assignment]
            elif param.default != inspect.Parameter.empty:
                # Use default value if available
                params[param_name] = param.default
            else:
                raise ValueError(f"Missing required argument: {param_name}")

        return params

    def format_output(self, result: Any) -> str:
        """Format function output for CLI display."""
        if isinstance(result, (dict, list)):
            return json.dumps(result, indent=2)
        elif isinstance(result, bool):
            return "true" if result else "false"
        else:
            return str(result)

    def show_help(self) -> None:
        """Display help information for the function."""
        print(f"Function: {self.name}")
        if self.func.__doc__:
            print(f"\nDescription:\n{self.func.__doc__.strip()}")

        print(f"\nSignature: {self.name}{self.signature}")

        examples = self.metadata.get("examples", [])
        if examples:
            print("\nExamples:")
            for example in examples:
                input_str = " ".join(str(v) for v in example["input"].values())
                print(f"  $ chuk-{self.name.replace('_', '-')} {input_str}")
                print(f"    → {example['output']}")
                if example.get("description"):
                    print(f"    # {example['description']}")

    def run(self, args: List[str]) -> int:
        """Execute the wrapped function with CLI args."""
        # Check for help flag
        if "--help" in args or "-h" in args:
            self.show_help()
            return 0

        # Remove any flags from args for parsing
        clean_args = [arg for arg in args if not arg.startswith("-")]

        try:
            params = self.parse_args(clean_args)

            # Handle async functions
            if asyncio.iscoroutinefunction(self.func):
                result = asyncio.run(self.func(**params))
            else:
                result = self.func(**params)

            print(self.format_output(result))
            return 0
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1


def create_cli_wrapper(func, name: Optional[str] = None) -> CLIWrapper:
    """Create a CLI wrapper for a function."""
    if name is None:
        name = func.__name__
    return CLIWrapper(func, name)
