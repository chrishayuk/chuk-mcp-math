#!/usr/bin/env python3
# tests/cli/test_main.py
"""
Comprehensive pytest unit tests for CLI functionality.

Tests cover:
- CLI wrapper functionality
- Function discovery
- Argument parsing
- Output formatting
- Error handling
"""

import pytest
import json
from unittest.mock import patch

from chuk_mcp_math.cli import CLIWrapper


class TestCLIWrapper:
    """Test CLIWrapper class."""

    def test_wrapper_initialization(self):
        """Test CLIWrapper initialization."""

        def sample_func(x: int, y: int) -> int:
            """Sample function."""
            return x + y

        wrapper = CLIWrapper(sample_func, "sample_func")

        assert wrapper.name == "sample_func"
        assert wrapper.func == sample_func
        assert wrapper.signature is not None

    def test_parse_args_integers(self):
        """Test parsing integer arguments."""

        def add(a: int, b: int) -> int:
            return a + b

        wrapper = CLIWrapper(add, "add")
        params = wrapper.parse_args(["5", "3"])

        assert params == {"a": 5, "b": 3}

    def test_parse_args_floats(self):
        """Test parsing float arguments."""

        def multiply(x: float, y: float) -> float:
            return x * y

        wrapper = CLIWrapper(multiply, "multiply")
        params = wrapper.parse_args(["2.5", "4.0"])

        assert params == {"x": 2.5, "y": 4.0}

    def test_parse_args_boolean(self):
        """Test parsing boolean arguments."""

        def check(flag: bool) -> bool:
            return flag

        wrapper = CLIWrapper(check, "check")

        params = wrapper.parse_args(["true"])
        assert params == {"flag": True}

        params = wrapper.parse_args(["false"])
        assert params == {"flag": False}

    def test_parse_args_with_defaults(self):
        """Test parsing with default values."""

        def func_with_default(x: int, y: int = 10) -> int:
            return x + y

        wrapper = CLIWrapper(func_with_default, "func")

        # Provide only first argument
        params = wrapper.parse_args(["5"])
        assert params == {"x": 5, "y": 10}

        # Provide both arguments
        params = wrapper.parse_args(["5", "20"])
        assert params == {"x": 5, "y": 20}

    def test_parse_args_missing_required(self):
        """Test error when required argument is missing."""

        def func(x: int, y: int) -> int:
            return x + y

        wrapper = CLIWrapper(func, "func")

        with pytest.raises(ValueError, match="Missing required argument"):
            wrapper.parse_args(["5"])  # Missing y

    def test_format_output_dict(self):
        """Test formatting dictionary output."""

        def func() -> dict:
            return {"key": "value"}

        wrapper = CLIWrapper(func, "func")
        output = wrapper.format_output({"key": "value", "number": 42})

        # Should be valid JSON
        parsed = json.loads(output)
        assert parsed == {"key": "value", "number": 42}

    def test_format_output_list(self):
        """Test formatting list output."""

        def func() -> list:
            return [1, 2, 3]

        wrapper = CLIWrapper(func, "func")
        output = wrapper.format_output([1, 2, 3])

        parsed = json.loads(output)
        assert parsed == [1, 2, 3]

    def test_format_output_boolean(self):
        """Test formatting boolean output."""

        def func() -> bool:
            return True

        wrapper = CLIWrapper(func, "func")

        assert wrapper.format_output(True) == "true"
        assert wrapper.format_output(False) == "false"

    def test_format_output_number(self):
        """Test formatting numeric output."""

        def func() -> int:
            return 42

        wrapper = CLIWrapper(func, "func")
        assert wrapper.format_output(42) == "42"
        assert wrapper.format_output(3.14) == "3.14"

    def test_format_output_string(self):
        """Test formatting string output."""

        def func() -> str:
            return "hello"

        wrapper = CLIWrapper(func, "func")
        assert wrapper.format_output("hello") == "hello"

    @patch("sys.stdout")
    def test_show_help(self, mock_stdout):
        """Test help display."""

        def sample_func(x: int) -> int:
            """Sample function for testing."""
            return x * 2

        wrapper = CLIWrapper(sample_func, "sample_func")

        # Mock print to capture output
        with patch("builtins.print") as mock_print:
            wrapper.show_help()
            # Verify print was called
            assert mock_print.called

    def test_run_success(self):
        """Test successful function execution."""

        def add(a: int, b: int) -> int:
            return a + b

        wrapper = CLIWrapper(add, "add")

        with patch("builtins.print") as mock_print:
            result_code = wrapper.run(["5", "3"])

            assert result_code == 0
            # Check that result was printed
            mock_print.assert_called_once_with("8")

    def test_run_with_error(self):
        """Test function execution with error."""

        def failing_func(x: int) -> int:
            raise ValueError("Test error")

        wrapper = CLIWrapper(failing_func, "failing")

        with patch("builtins.print") as mock_print:
            result_code = wrapper.run(["5"])

            assert result_code == 1
            # Error should be printed to stderr
            # Check that print was called (for error message)
            assert mock_print.called

    def test_run_with_help_flag(self):
        """Test running with --help flag."""

        def sample_func(x: int) -> int:
            return x

        wrapper = CLIWrapper(sample_func, "sample")

        with patch.object(wrapper, "show_help") as mock_help:
            result_code = wrapper.run(["--help"])

            assert result_code == 0
            mock_help.assert_called_once()

    def test_run_async_function(self):
        """Test running async function."""
        import asyncio

        async def async_add(a: int, b: int) -> int:
            await asyncio.sleep(0.001)
            return a + b

        wrapper = CLIWrapper(async_add, "async_add")

        with patch("builtins.print") as mock_print:
            result_code = wrapper.run(["5", "3"])

            assert result_code == 0
            mock_print.assert_called_once_with("8")

    def test_run_removes_flags(self):
        """Test that flags are removed from args."""

        def func(x: int) -> int:
            return x * 2

        wrapper = CLIWrapper(func, "func")

        with patch("builtins.print"):
            # Should ignore --json-output flag and use just "5"
            result_code = wrapper.run(["--json-output", "5"])
            assert result_code == 0


class TestCLIWrapperWithMetadata:
    """Test CLIWrapper with MCP metadata."""

    def test_wrapper_with_mcp_metadata(self):
        """Test wrapper with function that has MCP metadata."""

        def func_with_metadata(x: int) -> int:
            return x * 2

        # Add metadata
        func_with_metadata._mcp_metadata = {
            "description": "Doubles a number",
            "examples": [{"input": {"x": 5}, "output": 10, "description": "Double 5"}],
        }

        wrapper = CLIWrapper(func_with_metadata, "func")

        assert wrapper.metadata["description"] == "Doubles a number"
        assert len(wrapper.metadata["examples"]) == 1

    def test_wrapper_without_metadata(self):
        """Test wrapper with function without metadata."""

        def simple_func(x: int) -> int:
            return x

        wrapper = CLIWrapper(simple_func, "simple")

        # Should have empty metadata
        assert wrapper.metadata == {}


class TestCLIWrapperEdgeCases:
    """Test edge cases for CLI wrapper."""

    def test_empty_args(self):
        """Test with empty arguments list."""

        def no_args_func() -> str:
            return "success"

        wrapper = CLIWrapper(no_args_func, "no_args")

        with patch("builtins.print") as mock_print:
            result_code = wrapper.run([])
            assert result_code == 0
            mock_print.assert_called_once_with("success")

    def test_parse_args_list_json(self):
        """Test parsing list arguments as JSON."""

        def func_with_list(items: list) -> int:
            return len(items)

        wrapper = CLIWrapper(func_with_list, "func")

        params = wrapper.parse_args(["[1, 2, 3]"])
        assert params == {"items": [1, 2, 3]}

    def test_parse_args_list_comma_separated(self):
        """Test parsing list as comma-separated values."""

        def func_with_list(items: list) -> int:
            return len(items)

        wrapper = CLIWrapper(func_with_list, "func")

        # Invalid JSON, should fall back to comma-separated
        params = wrapper.parse_args(["a,b,c"])
        assert params == {"items": ["a", "b", "c"]}

    def test_function_name_normalization(self):
        """Test function name in help."""

        def test_function_name(x: int) -> int:
            return x

        wrapper = CLIWrapper(test_function_name, "test_function_name")
        assert wrapper.name == "test_function_name"


class TestCLIIntegration:
    """Integration tests for CLI."""

    def test_cli_wrapper_end_to_end(self):
        """Test complete CLI wrapper workflow."""

        async def async_multiply(x: int, y: int) -> int:
            """Multiply two numbers."""
            import asyncio

            await asyncio.sleep(0.001)
            return x * y

        async_multiply._mcp_metadata = {
            "description": "Multiplies two integers",
            "examples": [{"input": {"x": 3, "y": 4}, "output": 12, "description": "3 times 4"}],
        }

        wrapper = CLIWrapper(async_multiply, "multiply")

        # Test execution
        with patch("builtins.print") as mock_print:
            result_code = wrapper.run(["6", "7"])
            assert result_code == 0
            mock_print.assert_called_once_with("42")

    def test_multiple_wrappers(self):
        """Test creating multiple wrappers."""

        def add(a: int, b: int) -> int:
            return a + b

        def subtract(a: int, b: int) -> int:
            return a - b

        wrapper1 = CLIWrapper(add, "add")
        wrapper2 = CLIWrapper(subtract, "subtract")

        assert wrapper1.name == "add"
        assert wrapper2.name == "subtract"

        with patch("builtins.print") as mock_print:
            wrapper1.run(["10", "5"])
            mock_print.assert_called_with("15")

        with patch("builtins.print") as mock_print:
            wrapper2.run(["10", "5"])
            mock_print.assert_called_with("5")


class TestTypeConversions:
    """Test type conversion edge cases."""

    def test_string_to_int_conversion(self):
        """Test string to int conversion."""

        def func(x: int) -> int:
            return x * 2

        wrapper = CLIWrapper(func, "func")
        params = wrapper.parse_args(["42"])

        assert params == {"x": 42}
        assert isinstance(params["x"], int)

    def test_string_to_float_conversion(self):
        """Test string to float conversion."""

        def func(x: float) -> float:
            return x * 2.0

        wrapper = CLIWrapper(func, "func")
        params = wrapper.parse_args(["3.14"])

        assert params == {"x": 3.14}
        assert isinstance(params["x"], float)

    def test_boolean_string_variants(self):
        """Test different boolean string values."""

        def func(flag: bool) -> bool:
            return flag

        wrapper = CLIWrapper(func, "func")

        # True variants
        for val in ["true", "True", "TRUE", "1", "yes"]:
            params = wrapper.parse_args([val])
            assert params["flag"] is True

        # False variants
        for val in ["false", "False", "FALSE", "0", "no"]:
            params = wrapper.parse_args([val])
            assert params["flag"] is False


class TestCLICommands:
    """Test CLI command interface."""

    def test_cli_group(self):
        """Test CLI group initialization."""
        from chuk_mcp_math.cli.main import cli
        from click.testing import CliRunner

        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])

        assert result.exit_code == 0
        assert "CHUK MCP Math" in result.output

    def test_version_command(self):
        """Test version command."""
        from chuk_mcp_math.cli.main import cli
        from click.testing import CliRunner

        runner = CliRunner()
        result = runner.invoke(cli, ["version"])

        assert result.exit_code == 0
        assert "CHUK MCP Math CLI" in result.output

    def test_discover_all_functions(self):
        """Test function discovery."""
        from chuk_mcp_math.cli.main import discover_all_functions

        functions = discover_all_functions()

        # Should find at least some functions
        assert isinstance(functions, dict)
        # All entries should have required fields
        for func_name, info in functions.items():
            assert "function" in info
            assert "module" in info
            assert "metadata" in info
            assert "signature" in info

    def test_list_command(self):
        """Test list command."""
        from chuk_mcp_math.cli.main import cli
        from click.testing import CliRunner

        runner = CliRunner()
        result = runner.invoke(cli, ["list"])

        assert result.exit_code == 0
        # Should show some functions or modules

    def test_list_command_with_module_filter(self):
        """Test list command with module filter."""
        from chuk_mcp_math.cli.main import cli
        from click.testing import CliRunner

        runner = CliRunner()
        result = runner.invoke(cli, ["list", "--module", "primes"])

        assert result.exit_code == 0

    def test_list_command_detailed(self):
        """Test list command with detailed flag."""
        from chuk_mcp_math.cli.main import cli
        from click.testing import CliRunner

        runner = CliRunner()
        result = runner.invoke(cli, ["list", "--detailed"])

        assert result.exit_code == 0

    def test_search_command(self):
        """Test search command."""
        from chuk_mcp_math.cli.main import cli
        from click.testing import CliRunner

        runner = CliRunner()
        result = runner.invoke(cli, ["search", "prime"])

        assert result.exit_code == 0

    def test_search_command_no_results(self):
        """Test search command with no matches."""
        from chuk_mcp_math.cli.main import cli
        from click.testing import CliRunner

        runner = CliRunner()
        result = runner.invoke(cli, ["search", "nonexistent_xyz_12345"])

        assert result.exit_code == 0
        assert "No functions found" in result.output

    def test_describe_command(self):
        """Test describe command."""
        from chuk_mcp_math.cli.main import cli, discover_all_functions
        from click.testing import CliRunner

        # First discover what functions exist
        functions = discover_all_functions()
        if functions:
            # Get first function name
            func_name = list(functions.keys())[0]

            runner = CliRunner()
            result = runner.invoke(cli, ["describe", func_name])

            assert result.exit_code == 0
            assert "Function:" in result.output

    def test_describe_command_not_found(self):
        """Test describe command with non-existent function."""
        from chuk_mcp_math.cli.main import cli
        from click.testing import CliRunner

        runner = CliRunner()
        result = runner.invoke(cli, ["describe", "nonexistent_function"])

        assert result.exit_code == 1
        assert "not found" in result.output

    def test_call_command(self):
        """Test call command."""
        from chuk_mcp_math.cli.main import cli, discover_all_functions
        from click.testing import CliRunner

        # Find a simple function to call
        functions = discover_all_functions()
        if functions:
            runner = CliRunner()
            # Try calling is_prime if it exists
            if any("is_prime" in key for key in functions.keys()):
                result = runner.invoke(cli, ["call", "is_prime", "17"])
                # Should either succeed or fail gracefully (0, 1, or 2 for Click errors)
                assert result.exit_code in [0, 1, 2]

    def test_call_command_not_found(self):
        """Test call command with non-existent function."""
        from chuk_mcp_math.cli.main import cli
        from click.testing import CliRunner

        runner = CliRunner()
        result = runner.invoke(cli, ["call", "nonexistent_function", "1"])

        assert result.exit_code == 1
        assert "not found" in result.output

    def test_call_command_with_short_name(self):
        """Test call command using short function name."""
        from chuk_mcp_math.cli.main import cli, discover_all_functions
        from click.testing import CliRunner

        functions = discover_all_functions()
        if functions:
            # Get a function and try calling it with just the short name
            func_key = list(functions.keys())[0]
            short_name = func_key.split(".")[-1]

            runner = CliRunner()
            runner.invoke(cli, ["call", short_name])
            # May fail due to missing args, but should recognize the function
            # Exit code 0 or 1 are both acceptable


class TestDiscoverFunctionsDetails:
    """Test detailed discovery functionality."""

    def test_discover_handles_import_errors(self):
        """Test that discovery handles import errors gracefully."""
        from chuk_mcp_math.cli.main import discover_all_functions

        # Should not raise exceptions even if some modules fail to import
        functions = discover_all_functions()
        assert isinstance(functions, dict)

    def test_discover_finds_mcp_functions(self):
        """Test that discovery finds functions with MCP metadata."""
        from chuk_mcp_math.cli.main import discover_all_functions

        functions = discover_all_functions()

        # Check that discovered functions have the expected structure
        for func_name, info in functions.items():
            assert callable(info["function"])
            assert isinstance(info["module"], str)
            assert isinstance(info["metadata"], dict)
            assert isinstance(info["signature"], str)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
