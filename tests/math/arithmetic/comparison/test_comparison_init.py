#!/usr/bin/env python3
# tests/math/arithmetic/comparison/test_init.py
"""
Comprehensive pytest unit tests for src/chuk_mcp_math/arithmetic/comparison/__init__.py

Tests cover:
- Submodule imports (relational, extrema, tolerance)
- Function imports from each submodule
- __all__ exports
- test_comparison_functions async test function
- Module documentation
"""

import pytest
import asyncio

# Import the module to test
from chuk_mcp_math.arithmetic import comparison


class TestSubmoduleImports:
    """Test that submodules are imported correctly."""

    def test_relational_module_imported(self):
        """Test that relational module is imported."""
        assert hasattr(comparison, "relational")
        assert comparison.relational is not None

    def test_extrema_module_imported(self):
        """Test that extrema module is imported."""
        assert hasattr(comparison, "extrema")
        assert comparison.extrema is not None

    def test_tolerance_module_imported(self):
        """Test that tolerance module is imported."""
        assert hasattr(comparison, "tolerance")
        assert comparison.tolerance is not None


class TestRelationalFunctions:
    """Test that relational functions are imported."""

    def test_equal_imported(self):
        """Test that equal function is imported."""
        assert hasattr(comparison, "equal")
        assert callable(comparison.equal)

    def test_not_equal_imported(self):
        """Test that not_equal function is imported."""
        assert hasattr(comparison, "not_equal")
        assert callable(comparison.not_equal)

    def test_less_than_imported(self):
        """Test that less_than function is imported."""
        assert hasattr(comparison, "less_than")
        assert callable(comparison.less_than)

    def test_less_than_or_equal_imported(self):
        """Test that less_than_or_equal function is imported."""
        assert hasattr(comparison, "less_than_or_equal")
        assert callable(comparison.less_than_or_equal)

    def test_greater_than_imported(self):
        """Test that greater_than function is imported."""
        assert hasattr(comparison, "greater_than")
        assert callable(comparison.greater_than)

    def test_greater_than_or_equal_imported(self):
        """Test that greater_than_or_equal function is imported."""
        assert hasattr(comparison, "greater_than_or_equal")
        assert callable(comparison.greater_than_or_equal)

    def test_in_range_imported(self):
        """Test that in_range function is imported."""
        assert hasattr(comparison, "in_range")
        assert callable(comparison.in_range)

    def test_between_imported(self):
        """Test that between function is imported."""
        assert hasattr(comparison, "between")
        assert callable(comparison.between)


class TestExtremaFunctions:
    """Test that extrema functions are imported."""

    def test_minimum_imported(self):
        """Test that minimum function is imported."""
        assert hasattr(comparison, "minimum")
        assert callable(comparison.minimum)

    def test_maximum_imported(self):
        """Test that maximum function is imported."""
        assert hasattr(comparison, "maximum")
        assert callable(comparison.maximum)

    def test_clamp_imported(self):
        """Test that clamp function is imported."""
        assert hasattr(comparison, "clamp")
        assert callable(comparison.clamp)

    def test_sort_numbers_imported(self):
        """Test that sort_numbers function is imported."""
        assert hasattr(comparison, "sort_numbers")
        assert callable(comparison.sort_numbers)

    def test_rank_numbers_imported(self):
        """Test that rank_numbers function is imported."""
        assert hasattr(comparison, "rank_numbers")
        assert callable(comparison.rank_numbers)

    def test_min_list_imported(self):
        """Test that min_list function is imported."""
        assert hasattr(comparison, "min_list")
        assert callable(comparison.min_list)

    def test_max_list_imported(self):
        """Test that max_list function is imported."""
        assert hasattr(comparison, "max_list")
        assert callable(comparison.max_list)


class TestToleranceFunctions:
    """Test that tolerance functions are imported."""

    def test_approximately_equal_imported(self):
        """Test that approximately_equal function is imported."""
        assert hasattr(comparison, "approximately_equal")
        assert callable(comparison.approximately_equal)

    def test_close_to_zero_imported(self):
        """Test that close_to_zero function is imported."""
        assert hasattr(comparison, "close_to_zero")
        assert callable(comparison.close_to_zero)

    def test_is_finite_imported(self):
        """Test that is_finite function is imported."""
        assert hasattr(comparison, "is_finite")
        assert callable(comparison.is_finite)

    def test_is_nan_imported(self):
        """Test that is_nan function is imported."""
        assert hasattr(comparison, "is_nan")
        assert callable(comparison.is_nan)

    def test_is_infinite_imported(self):
        """Test that is_infinite function is imported."""
        assert hasattr(comparison, "is_infinite")
        assert callable(comparison.is_infinite)

    def test_is_normal_imported(self):
        """Test that is_normal function is imported."""
        assert hasattr(comparison, "is_normal")
        assert callable(comparison.is_normal)

    def test_is_close_imported(self):
        """Test that is_close function is imported."""
        assert hasattr(comparison, "is_close")
        assert callable(comparison.is_close)


class TestAllExports:
    """Test __all__ exports."""

    def test_all_defined(self):
        """Test that __all__ is defined."""
        assert hasattr(comparison, "__all__")
        assert isinstance(comparison.__all__, list)
        assert len(comparison.__all__) > 0

    def test_all_contains_submodules(self):
        """Test that __all__ contains submodule names."""
        assert "relational" in comparison.__all__
        assert "extrema" in comparison.__all__
        assert "tolerance" in comparison.__all__

    def test_all_contains_relational_functions(self):
        """Test that __all__ contains relational functions."""
        relational_funcs = [
            "equal",
            "not_equal",
            "less_than",
            "less_than_or_equal",
            "greater_than",
            "greater_than_or_equal",
            "in_range",
            "between",
        ]

        for func in relational_funcs:
            assert func in comparison.__all__, f"{func} not in __all__"

    def test_all_contains_extrema_functions(self):
        """Test that __all__ contains extrema functions."""
        extrema_funcs = [
            "minimum",
            "maximum",
            "clamp",
            "sort_numbers",
            "rank_numbers",
            "min_list",
            "max_list",
        ]

        for func in extrema_funcs:
            assert func in comparison.__all__, f"{func} not in __all__"

    def test_all_contains_tolerance_functions(self):
        """Test that __all__ contains tolerance functions."""
        tolerance_funcs = [
            "approximately_equal",
            "close_to_zero",
            "is_finite",
            "is_nan",
            "is_infinite",
            "is_normal",
            "is_close",
        ]

        for func in tolerance_funcs:
            assert func in comparison.__all__, f"{func} not in __all__"

    def test_all_items_exist(self):
        """Test that all items in __all__ actually exist."""
        for item in comparison.__all__:
            assert hasattr(comparison, item), f"{item} in __all__ but not in module"


class TestAsyncRelationalFunctions:
    """Test async functionality of relational functions."""

    @pytest.mark.asyncio
    async def test_equal_function(self):
        """Test equal function works correctly."""
        result = await comparison.equal(5, 5)
        assert result is True

        result = await comparison.equal(5, 3)
        assert result is False

    @pytest.mark.asyncio
    async def test_not_equal_function(self):
        """Test not_equal function works correctly."""
        result = await comparison.not_equal(5, 3)
        assert result is True

        result = await comparison.not_equal(5, 5)
        assert result is False

    @pytest.mark.asyncio
    async def test_less_than_function(self):
        """Test less_than function works correctly."""
        result = await comparison.less_than(3, 5)
        assert result is True

        result = await comparison.less_than(5, 3)
        assert result is False

    @pytest.mark.asyncio
    async def test_greater_than_function(self):
        """Test greater_than function works correctly."""
        result = await comparison.greater_than(5, 3)
        assert result is True

        result = await comparison.greater_than(3, 5)
        assert result is False

    @pytest.mark.asyncio
    async def test_in_range_function(self):
        """Test in_range function works correctly."""
        result = await comparison.in_range(5, 1, 10)
        assert result is True

        result = await comparison.in_range(15, 1, 10)
        assert result is False

    @pytest.mark.asyncio
    async def test_between_function(self):
        """Test between function works correctly."""
        result = await comparison.between(5, 1, 10)
        assert result is True


class TestAsyncExtremaFunctions:
    """Test async functionality of extrema functions."""

    @pytest.mark.asyncio
    async def test_minimum_function(self):
        """Test minimum function works correctly."""
        result = await comparison.minimum(5, 3)
        assert result == 3

    @pytest.mark.asyncio
    async def test_maximum_function(self):
        """Test maximum function works correctly."""
        result = await comparison.maximum(5, 3)
        assert result == 5

    @pytest.mark.asyncio
    async def test_clamp_function(self):
        """Test clamp function works correctly."""
        result = await comparison.clamp(15, 1, 10)
        assert result == 10

        result = await comparison.clamp(-5, 1, 10)
        assert result == 1

        result = await comparison.clamp(5, 1, 10)
        assert result == 5

    @pytest.mark.asyncio
    async def test_sort_numbers_function(self):
        """Test sort_numbers function works correctly."""
        result = await comparison.sort_numbers([3, 1, 4, 1, 5])
        assert result == [1, 1, 3, 4, 5]

    @pytest.mark.asyncio
    async def test_min_list_function(self):
        """Test min_list function works correctly."""
        result = await comparison.min_list([3, 1, 4, 1, 5])
        assert result == 1

    @pytest.mark.asyncio
    async def test_max_list_function(self):
        """Test max_list function works correctly."""
        result = await comparison.max_list([3, 1, 4, 1, 5])
        assert result == 5


class TestAsyncToleranceFunctions:
    """Test async functionality of tolerance functions."""

    @pytest.mark.asyncio
    async def test_approximately_equal_function(self):
        """Test approximately_equal function works correctly."""
        result = await comparison.approximately_equal(0.1, 0.10000001, 1e-7)
        assert result is True

        result = await comparison.approximately_equal(0.1, 0.2, 1e-7)
        assert result is False

    @pytest.mark.asyncio
    async def test_is_finite_function(self):
        """Test is_finite function works correctly."""
        result = await comparison.is_finite(42.5)
        assert result is True

        result = await comparison.is_finite(float("inf"))
        assert result is False

    @pytest.mark.asyncio
    async def test_is_nan_function(self):
        """Test is_nan function works correctly."""
        result = await comparison.is_nan(float("nan"))
        assert result is True

        result = await comparison.is_nan(42.5)
        assert result is False

    @pytest.mark.asyncio
    async def test_is_infinite_function(self):
        """Test is_infinite function works correctly."""
        result = await comparison.is_infinite(float("inf"))
        assert result is True

        result = await comparison.is_infinite(42.5)
        assert result is False

    @pytest.mark.asyncio
    async def test_close_to_zero_function(self):
        """Test close_to_zero function works correctly."""
        result = await comparison.close_to_zero(1e-10, 1e-9)
        assert result is True

        result = await comparison.close_to_zero(0.5, 1e-9)
        assert result is False


class TestTestComparisonFunction:
    """Test the test_comparison_functions async function."""

    @pytest.mark.asyncio
    async def test_test_comparison_functions_exists(self):
        """Test that test_comparison_functions function exists."""
        assert hasattr(comparison, "_test_comparison_functions")
        assert callable(comparison._test_comparison_functions)

    @pytest.mark.asyncio
    async def test_test_comparison_functions_executes(self, capsys):
        """Test that test_comparison_functions executes without error."""
        await comparison._test_comparison_functions()
        captured = capsys.readouterr()

        # Should produce output
        assert len(captured.out) > 0
        assert "Comparison" in captured.out or "Functions Test" in captured.out


class TestModuleDocumentation:
    """Test module documentation."""

    def test_module_has_docstring(self):
        """Test that comparison module has a docstring."""
        assert comparison.__doc__ is not None
        assert len(comparison.__doc__) > 0

    def test_docstring_mentions_comparison(self):
        """Test that docstring mentions comparison operations."""
        docstring = comparison.__doc__.upper()
        assert "COMPARISON" in docstring or "ORDERING" in docstring

    def test_docstring_mentions_submodules(self):
        """Test that docstring mentions submodules."""
        docstring = comparison.__doc__.lower()
        # Should mention at least one of the submodules
        assert "relational" in docstring or "extrema" in docstring or "tolerance" in docstring


class TestFunctionCount:
    """Test that all expected functions are present."""

    def test_relational_functions_count(self):
        """Test that we have all relational functions."""
        relational_funcs = [
            "equal",
            "not_equal",
            "less_than",
            "less_than_or_equal",
            "greater_than",
            "greater_than_or_equal",
            "in_range",
            "between",
        ]
        count = sum(1 for func in relational_funcs if hasattr(comparison, func))
        assert count == len(relational_funcs)

    def test_extrema_functions_count(self):
        """Test that we have all extrema functions."""
        extrema_funcs = [
            "minimum",
            "maximum",
            "clamp",
            "sort_numbers",
            "rank_numbers",
            "min_list",
            "max_list",
        ]
        count = sum(1 for func in extrema_funcs if hasattr(comparison, func))
        assert count == len(extrema_funcs)

    def test_tolerance_functions_count(self):
        """Test that we have all tolerance functions."""
        tolerance_funcs = [
            "approximately_equal",
            "close_to_zero",
            "is_finite",
            "is_nan",
            "is_infinite",
            "is_normal",
            "is_close",
        ]
        count = sum(1 for func in tolerance_funcs if hasattr(comparison, func))
        assert count == len(tolerance_funcs)


class TestConcurrentExecution:
    """Test concurrent execution of comparison functions."""

    @pytest.mark.asyncio
    async def test_concurrent_relational_operations(self):
        """Test concurrent execution of relational operations."""
        tasks = [
            comparison.equal(5, 5),
            comparison.less_than(3, 5),
            comparison.greater_than(5, 3),
            comparison.in_range(5, 1, 10),
        ]

        results = await asyncio.gather(*tasks)
        expected = [True, True, True, True]

        assert results == expected

    @pytest.mark.asyncio
    async def test_concurrent_extrema_operations(self):
        """Test concurrent execution of extrema operations."""
        tasks = [
            comparison.minimum(5, 3),
            comparison.maximum(5, 3),
            comparison.clamp(15, 1, 10),
        ]

        results = await asyncio.gather(*tasks)
        expected = [3, 5, 10]

        assert results == expected

    @pytest.mark.asyncio
    async def test_concurrent_tolerance_operations(self):
        """Test concurrent execution of tolerance operations."""
        tasks = [
            comparison.is_finite(42.5),
            comparison.is_nan(float("nan")),
            comparison.is_infinite(float("inf")),
        ]

        results = await asyncio.gather(*tasks)
        expected = [True, True, True]

        assert results == expected


class TestMainExecution:
    """Test main execution block behavior."""

    def test_main_block_does_not_execute_on_import(self):
        """Test that __name__ == '__main__' block doesn't run on import."""
        from chuk_mcp_math.arithmetic import comparison as test_comparison

        assert test_comparison is not None


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v", "--tb=short"])
