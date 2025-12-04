#!/usr/bin/env python3
import sys

sys.path.insert(0, "src")

from chuk_mcp_math.cli.main import cli

# Test the CLI
if __name__ == "__main__":
    # Simulate command line args
    sys.argv = ["chuk", "call", "is_prime", "17"]
    cli()
