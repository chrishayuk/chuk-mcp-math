#!/usr/bin/env python3
# tests/math/arithmetic/core/test_init.py
"""
Comprehensive pytest unit tests for src/chuk_mcp_math/arithmetic/core/__init__.py

Tests cover:
- Submodule imports (basic_operations, rounding, modular)
- Function imports from each submodule
- __all__ exports
- test_core_functions async test function
- Module documentation
"""

import pytest
import asyncio

# Import the module to test
from chuk_mcp_math.arithmetic import core


class TestSubmoduleImports:
    """Test that submodules are imported correctly."""

    def test_basic_operations_module_imported(self):
        """Test that basic_operations module is imported."""
        assert hasattr(core, "basic_operations")
        assert core.basic_operations is not None

    def test_rounding_module_imported(self):
        """Test that rounding module is imported."""
        assert hasattr(core, "rounding")
        assert core.rounding is not None

    def test_modular_module_imported(self):
        """Test that modular module is imported."""
        assert hasattr(core, "modular")
        assert core.modular is not None


class TestBasicOperationsFunctions:
    """Test that basic operations functions are imported."""

    def test_add_imported(self):
        """Test that add function is imported."""
        assert hasattr(core, "add")
        assert callable(core.add)

    def test_subtract_imported(self):
        """Test that subtract function is imported."""
        assert hasattr(core, "subtract")
        assert callable(core.subtract)

    def test_multiply_imported(self):
        """Test that multiply function is imported."""
        assert hasattr(core, "multiply")
        assert callable(core.multiply)

    def test_divide_imported(self):
        """Test that divide function is imported."""
        assert hasattr(core, "divide")
        assert callable(core.divide)

    def test_power_imported(self):
        """Test that power function is imported."""
        assert hasattr(core, "power")
        assert callable(core.power)

    def test_sqrt_imported(self):
        """Test that sqrt function is imported."""
        assert hasattr(core, "sqrt")
        assert callable(core.sqrt)

    def test_abs_value_imported(self):
        """Test that abs_value function is imported."""
        assert hasattr(core, "abs_value")
        assert callable(core.abs_value)

    def test_sign_imported(self):
        """Test that sign function is imported."""
        assert hasattr(core, "sign")
        assert callable(core.sign)

    def test_negate_imported(self):
        """Test that negate function is imported."""
        assert hasattr(core, "negate")
        assert callable(core.negate)


class TestRoundingFunctions:
    """Test that rounding functions are imported."""

    def test_round_number_imported(self):
        """Test that round_number function is imported."""
        assert hasattr(core, "round_number")
        assert callable(core.round_number)

    def test_floor_imported(self):
        """Test that floor function is imported."""
        assert hasattr(core, "floor")
        assert callable(core.floor)

    def test_ceil_imported(self):
        """Test that ceil function is imported."""
        assert hasattr(core, "ceil")
        assert callable(core.ceil)

    def test_truncate_imported(self):
        """Test that truncate function is imported."""
        assert hasattr(core, "truncate")
        assert callable(core.truncate)

    def test_mround_imported(self):
        """Test that mround function is imported."""
        assert hasattr(core, "mround")
        assert callable(core.mround)

    def test_ceiling_multiple_imported(self):
        """Test that ceiling_multiple function is imported."""
        assert hasattr(core, "ceiling_multiple")
        assert callable(core.ceiling_multiple)

    def test_floor_multiple_imported(self):
        """Test that floor_multiple function is imported."""
        assert hasattr(core, "floor_multiple")
        assert callable(core.floor_multiple)


class TestModularFunctions:
    """Test that modular functions are imported."""

    def test_modulo_imported(self):
        """Test that modulo function is imported."""
        assert hasattr(core, "modulo")
        assert callable(core.modulo)

    def test_divmod_operation_imported(self):
        """Test that divmod_operation function is imported."""
        assert hasattr(core, "divmod_operation")
        assert callable(core.divmod_operation)

    def test_mod_power_imported(self):
        """Test that mod_power function is imported."""
        assert hasattr(core, "mod_power")
        assert callable(core.mod_power)

    def test_quotient_imported(self):
        """Test that quotient function is imported."""
        assert hasattr(core, "quotient")
        assert callable(core.quotient)

    def test_remainder_imported(self):
        """Test that remainder function is imported."""
        assert hasattr(core, "remainder")
        assert callable(core.remainder)

    def test_fmod_imported(self):
        """Test that fmod function is imported."""
        assert hasattr(core, "fmod")
        assert callable(core.fmod)


class TestAllExports:
    """Test __all__ exports."""

    def test_all_defined(self):
        """Test that __all__ is defined."""
        assert hasattr(core, "__all__")
        assert isinstance(core.__all__, list)
        assert len(core.__all__) > 0

    def test_all_contains_submodules(self):
        """Test that __all__ contains submodule names."""
        assert "basic_operations" in core.__all__
        assert "rounding" in core.__all__
        assert "modular" in core.__all__

    def test_all_contains_basic_operations(self):
        """Test that __all__ contains basic operation functions."""
        basic_ops = [
            "add",
            "subtract",
            "multiply",
            "divide",
            "power",
            "sqrt",
            "abs_value",
            "sign",
            "negate",
        ]

        for func in basic_ops:
            assert func in core.__all__, f"{func} not in __all__"

    def test_all_contains_rounding_operations(self):
        """Test that __all__ contains rounding functions."""
        rounding_ops = [
            "round_number",
            "floor",
            "ceil",
            "truncate",
            "mround",
            "ceiling_multiple",
            "floor_multiple",
        ]

        for func in rounding_ops:
            assert func in core.__all__, f"{func} not in __all__"

    def test_all_contains_modular_operations(self):
        """Test that __all__ contains modular functions."""
        modular_ops = ["modulo", "divmod_operation", "mod_power", "quotient", "remainder", "fmod"]

        for func in modular_ops:
            assert func in core.__all__, f"{func} not in __all__"

    def test_all_items_exist(self):
        """Test that all items in __all__ actually exist."""
        for item in core.__all__:
            assert hasattr(core, item), f"{item} in __all__ but not in module"


class TestAsyncFunctionality:
    """Test async functionality of imported functions."""

    @pytest.mark.asyncio
    async def test_add_function(self):
        """Test add function works correctly."""
        result = await core.add(5, 3)
        assert result == 8

    @pytest.mark.asyncio
    async def test_subtract_function(self):
        """Test subtract function works correctly."""
        result = await core.subtract(10, 4)
        assert result == 6

    @pytest.mark.asyncio
    async def test_multiply_function(self):
        """Test multiply function works correctly."""
        result = await core.multiply(6, 7)
        assert result == 42

    @pytest.mark.asyncio
    async def test_divide_function(self):
        """Test divide function works correctly."""
        result = await core.divide(20, 4)
        assert result == 5.0

    @pytest.mark.asyncio
    async def test_power_function(self):
        """Test power function works correctly."""
        result = await core.power(2, 3)
        assert result == 8

    @pytest.mark.asyncio
    async def test_sqrt_function(self):
        """Test sqrt function works correctly."""
        result = await core.sqrt(16)
        assert result == 4.0

    @pytest.mark.asyncio
    async def test_round_number_function(self):
        """Test round_number function works correctly."""
        result = await core.round_number(3.14159, 2)
        assert result == 3.14

    @pytest.mark.asyncio
    async def test_floor_function(self):
        """Test floor function works correctly."""
        result = await core.floor(3.7)
        assert result == 3

    @pytest.mark.asyncio
    async def test_ceil_function(self):
        """Test ceil function works correctly."""
        result = await core.ceil(3.2)
        assert result == 4

    @pytest.mark.asyncio
    async def test_modulo_function(self):
        """Test modulo function works correctly."""
        result = await core.modulo(17, 5)
        assert result == 2

    @pytest.mark.asyncio
    async def test_quotient_function(self):
        """Test quotient function works correctly."""
        result = await core.quotient(17, 5)
        assert result == 3

    @pytest.mark.asyncio
    async def test_mod_power_function(self):
        """Test mod_power function works correctly."""
        result = await core.mod_power(2, 10, 1000)
        assert result == 24


class TestTestCoreFunction:
    """Test the test_core_functions async function."""

    @pytest.mark.asyncio
    async def test_test_core_functions_exists(self):
        """Test that test_core_functions function exists."""
        assert hasattr(core, "_test_core_functions")
        assert callable(core._test_core_functions)

    @pytest.mark.asyncio
    async def test_test_core_functions_executes(self, capsys):
        """Test that test_core_functions executes without error."""
        await core._test_core_functions()
        captured = capsys.readouterr()

        # Should produce output
        assert len(captured.out) > 0
        assert "Core Arithmetic" in captured.out or "Functions Test" in captured.out


class TestModuleDocumentation:
    """Test module documentation."""

    def test_module_has_docstring(self):
        """Test that core module has a docstring."""
        assert core.__doc__ is not None
        assert len(core.__doc__) > 0

    def test_docstring_mentions_core_operations(self):
        """Test that docstring mentions core operations."""
        docstring = core.__doc__.upper()
        assert "CORE" in docstring or "ARITHMETIC" in docstring

    def test_docstring_mentions_submodules(self):
        """Test that docstring mentions submodules."""
        docstring = core.__doc__.lower()
        # Should mention at least one of the submodules
        assert "basic_operations" in docstring or "rounding" in docstring or "modular" in docstring


class TestFunctionCount:
    """Test that all expected functions are present."""

    def test_basic_operations_count(self):
        """Test that we have all basic operations."""
        basic_ops = [
            "add",
            "subtract",
            "multiply",
            "divide",
            "power",
            "sqrt",
            "abs_value",
            "sign",
            "negate",
        ]
        count = sum(1 for op in basic_ops if hasattr(core, op))
        assert count == len(basic_ops)

    def test_rounding_operations_count(self):
        """Test that we have all rounding operations."""
        rounding_ops = [
            "round_number",
            "floor",
            "ceil",
            "truncate",
            "mround",
            "ceiling_multiple",
            "floor_multiple",
        ]
        count = sum(1 for op in rounding_ops if hasattr(core, op))
        assert count == len(rounding_ops)

    def test_modular_operations_count(self):
        """Test that we have all modular operations."""
        modular_ops = ["modulo", "divmod_operation", "mod_power", "quotient", "remainder", "fmod"]
        count = sum(1 for op in modular_ops if hasattr(core, op))
        assert count == len(modular_ops)


class TestConcurrentExecution:
    """Test concurrent execution of core functions."""

    @pytest.mark.asyncio
    async def test_concurrent_basic_operations(self):
        """Test concurrent execution of basic operations."""
        tasks = [
            core.add(1, 2),
            core.subtract(10, 5),
            core.multiply(3, 4),
            core.divide(20, 4),
            core.power(2, 3),
        ]

        results = await asyncio.gather(*tasks)
        expected = [3, 5, 12, 5.0, 8]

        assert results == expected

    @pytest.mark.asyncio
    async def test_concurrent_rounding_operations(self):
        """Test concurrent execution of rounding operations."""
        tasks = [
            core.round_number(3.7, 0),
            core.floor(3.7),
            core.ceil(3.2),
        ]

        results = await asyncio.gather(*tasks)
        expected = [4.0, 3, 4]

        assert results == expected


class TestMainExecution:
    """Test main execution block behavior."""

    def test_main_block_does_not_execute_on_import(self):
        """Test that __name__ == '__main__' block doesn't run on import."""
        from chuk_mcp_math.arithmetic import core as test_core

        assert test_core is not None


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v", "--tb=short"])
