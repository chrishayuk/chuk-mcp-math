#!/usr/bin/env python3
"""Generate individual bash executables for each math function."""

import sys
import stat
from pathlib import Path
import inspect

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

EXECUTABLE_TEMPLATE = """#!/usr/bin/env python3
# Auto-generated CLI wrapper for {function_name}
import sys
import os
import logging

# Suppress INFO logs for clean CLI output
logging.getLogger('chuk_mcp_math').setLevel(logging.WARNING)

sys.path.insert(0, "{src_path}")

from chuk_mcp_math.cli import CLIWrapper
from {module_import} import {function_name}

if __name__ == "__main__":
    wrapper = CLIWrapper({function_name}, "{function_name}")
    sys.exit(wrapper.run(sys.argv[1:]))
"""


def discover_functions(module_path: Path):
    """Discover all functions with @mcp_function decorator."""
    functions = {}

    # Import the main module
    import chuk_mcp_math.number_theory.primes as primes_module

    # Get all functions with mcp metadata
    for name, obj in inspect.getmembers(primes_module):
        if callable(obj) and hasattr(obj, "_mcp_metadata"):
            functions[name] = {
                "function": obj,
                "module": "chuk_mcp_math.number_theory.primes",
                "metadata": obj._mcp_metadata,
            }

    return functions


def generate_executables(output_dir: Path, src_path: Path):
    """Generate individual executable scripts."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # For now, just create executables for prime functions as a demo
    prime_functions = [
        "is_prime",
        "next_prime",
        "nth_prime",
        "prime_factors",
        "prime_count",
        "is_coprime",
    ]

    for func_name in prime_functions:
        script_name = f"chuk-{func_name.replace('_', '-')}"
        script_path = output_dir / script_name

        script_content = EXECUTABLE_TEMPLATE.format(
            function_name=func_name,
            module_import="chuk_mcp_math.number_theory.primes",
            src_path=str(src_path.absolute()),
        )

        script_path.write_text(script_content)
        # Make executable
        script_path.chmod(script_path.stat().st_mode | stat.S_IEXEC)
        print(f"Generated: {script_path}")

    print(f"\nGenerated {len(prime_functions)} executables in {output_dir}")
    print("\nTo use these commands:")
    print(f"  1. Add {output_dir} to your PATH:")
    print(f'     export PATH="$PATH:{output_dir}"')
    print("  2. Run commands like:")
    print("     chuk-is-prime 17")
    print("     chuk-prime-factors 100")


if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    src_path = project_root / "src"
    output_dir = project_root / "bin" / "executables"

    generate_executables(output_dir, src_path)
