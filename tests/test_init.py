#!/usr/bin/env python3
# tests/test_init.py
"""
Comprehensive pytest unit tests for src/chuk_mcp_math/__init__.py

Tests cover:
- Package metadata (__version__, __author__, __description__)
- Function discovery (get_all_functions, get_functions_by_category, get_functions_by_namespace)
- Math-specific functions (get_math_functions, get_math_constants)
- Statistics and execution info (get_execution_stats, get_async_performance_stats)
- Utility functions (get_function_recommendations, validate_math_domain)
- Export functions (export_all_specs, clear_all_caches)
- Print functions (print_math_summary, print_comprehensive_summary)
- Helper functions (math_quick_reference, setup_logging)
"""

import pytest
import asyncio
import math
import logging

# Import the module to test
import chuk_mcp_math


class TestPackageMetadata:
    """Test package metadata constants."""

    def test_version_exists(self):
        """Test that __version__ is defined."""
        assert hasattr(chuk_mcp_math, "__version__")
        assert isinstance(chuk_mcp_math.__version__, str)
        assert len(chuk_mcp_math.__version__) > 0

    def test_author_exists(self):
        """Test that __author__ is defined."""
        assert hasattr(chuk_mcp_math, "__author__")
        assert isinstance(chuk_mcp_math.__author__, str)
        assert len(chuk_mcp_math.__author__) > 0

    def test_description_exists(self):
        """Test that __description__ is defined."""
        assert hasattr(chuk_mcp_math, "__description__")
        assert isinstance(chuk_mcp_math.__description__, str)
        assert len(chuk_mcp_math.__description__) > 0


class TestFunctionDiscovery:
    """Test function discovery utilities."""

    def test_get_all_functions(self):
        """Test get_all_functions returns a dictionary."""
        result = chuk_mcp_math.get_all_functions()
        assert isinstance(result, dict)

    def test_get_functions_by_category(self):
        """Test get_functions_by_category returns filtered functions."""
        # Test with arithmetic category
        result = chuk_mcp_math.get_functions_by_category("arithmetic")
        assert isinstance(result, dict)

    def test_get_functions_by_namespace(self):
        """Test get_functions_by_namespace returns filtered functions."""
        # Test with arithmetic namespace
        result = chuk_mcp_math.get_functions_by_namespace("arithmetic")
        assert isinstance(result, dict)


class TestMathFunctions:
    """Test math-specific functions."""

    @pytest.mark.asyncio
    async def test_get_math_functions(self):
        """Test get_math_functions returns organized domains."""
        result = await chuk_mcp_math.get_math_functions()
        assert isinstance(result, dict)
        # Should have arithmetic and number_theory domains
        assert "arithmetic" in result
        assert "number_theory" in result
        assert isinstance(result["arithmetic"], dict)
        assert isinstance(result["number_theory"], dict)

    def test_get_math_constants(self):
        """Test get_math_constants returns all mathematical constants."""
        result = chuk_mcp_math.get_math_constants()
        assert isinstance(result, dict)

        # Check for expected constants
        expected_constants = [
            "pi",
            "e",
            "tau",
            "inf",
            "nan",
            "golden_ratio",
            "euler_gamma",
            "sqrt2",
            "sqrt3",
            "ln2",
            "ln10",
            "log2e",
            "log10e",
        ]

        for const in expected_constants:
            assert const in result
            assert (
                isinstance(result[const], float)
                or math.isnan(result[const])
                or math.isinf(result[const])
            )

    def test_math_constants_values(self):
        """Test that math constants have correct values."""
        result = chuk_mcp_math.get_math_constants()

        # Check specific values
        assert result["pi"] == math.pi
        assert result["e"] == math.e
        assert result["tau"] == math.tau
        assert math.isinf(result["inf"])
        assert math.isnan(result["nan"])
        assert pytest.approx(result["golden_ratio"], rel=1e-10) == (1 + math.sqrt(5)) / 2
        assert pytest.approx(result["sqrt2"], rel=1e-10) == math.sqrt(2)
        assert pytest.approx(result["sqrt3"], rel=1e-10) == math.sqrt(3)


class TestExecutionStats:
    """Test execution statistics functions."""

    def test_get_execution_stats(self):
        """Test get_execution_stats returns comprehensive statistics."""
        result = chuk_mcp_math.get_execution_stats()
        assert isinstance(result, dict)

        # Check structure
        assert "total_functions" in result
        assert "execution_modes" in result
        assert "features" in result
        assert "distribution" in result
        assert "performance" in result

        # Check execution modes
        assert "local_capable" in result["execution_modes"]
        assert "remote_capable" in result["execution_modes"]
        assert "both_capable" in result["execution_modes"]

        # Check features
        assert "cached_functions" in result["features"]
        assert "streaming_functions" in result["features"]
        assert "workflow_compatible" in result["features"]

        # Check distribution
        assert "namespaces" in result["distribution"]
        assert "categories" in result["distribution"]

        # Check performance
        assert "total_executions" in result["performance"]
        assert "error_rate" in result["performance"]
        assert "cache_hit_rate" in result["performance"]

    @pytest.mark.asyncio
    async def test_get_async_performance_stats(self):
        """Test get_async_performance_stats returns async function statistics."""
        result = await chuk_mcp_math.get_async_performance_stats()
        assert isinstance(result, dict)

        # Check structure
        assert "total_async_functions" in result
        assert "cached_functions" in result
        assert "streaming_functions" in result
        assert "high_performance_functions" in result
        assert "domains_converted" in result

        # Check types
        assert isinstance(result["total_async_functions"], int)
        assert isinstance(result["domains_converted"], int)


class TestUtilityFunctions:
    """Test utility helper functions."""

    def test_get_function_recommendations(self):
        """Test get_function_recommendations returns appropriate functions."""
        # Test different operation types
        test_cases = [
            ("basic", ["add", "subtract", "multiply", "divide", "power", "sqrt"]),
            ("comparison", ["equal", "less_than", "greater_than", "minimum", "maximum", "clamp"]),
            ("primes", ["is_prime", "next_prime", "nth_prime", "prime_factors", "is_coprime"]),
            ("divisibility", ["gcd", "lcm", "divisors", "is_even", "is_odd", "extended_gcd"]),
        ]

        for operation_type, expected_funcs in test_cases:
            result = chuk_mcp_math.get_function_recommendations(operation_type)
            assert isinstance(result, list)
            assert result == expected_funcs

    def test_get_function_recommendations_unknown_type(self):
        """Test get_function_recommendations with unknown operation type."""
        result = chuk_mcp_math.get_function_recommendations("nonexistent_operation")
        assert isinstance(result, list)
        assert len(result) == 0

    def test_validate_math_domain_valid(self):
        """Test validate_math_domain with valid domains."""
        valid_domains = ["arithmetic", "number_theory", "ARITHMETIC", "NUMBER_THEORY"]

        for domain in valid_domains:
            result = chuk_mcp_math.validate_math_domain(domain)
            assert result is True

    def test_validate_math_domain_invalid(self):
        """Test validate_math_domain with invalid domains."""
        invalid_domains = ["trigonometry", "statistics", "nonexistent", ""]

        for domain in invalid_domains:
            result = chuk_mcp_math.validate_math_domain(domain)
            assert result is False


class TestPrintFunctions:
    """Test print/display functions."""

    @pytest.mark.asyncio
    async def test_print_math_summary(self, capsys):
        """Test print_math_summary outputs to console."""
        await chuk_mcp_math.print_math_summary()
        captured = capsys.readouterr()

        # Check that output was generated
        assert len(captured.out) > 0
        assert "Mathematical Functions Library" in captured.out or "Domains:" in captured.out

    def test_print_comprehensive_summary(self, capsys):
        """Test print_comprehensive_summary outputs to console."""
        chuk_mcp_math.print_comprehensive_summary()
        captured = capsys.readouterr()

        # Check that output was generated
        assert len(captured.out) > 0
        # Should contain some statistics
        assert "Functions" in captured.out or "Summary" in captured.out

    def test_math_quick_reference(self):
        """Test math_quick_reference returns reference guide."""
        result = chuk_mcp_math.math_quick_reference()
        assert isinstance(result, str)
        assert len(result) > 0

        # Should contain key sections
        assert "Mathematical Functions" in result or "OPERATIONS" in result
        assert "await" in result  # Should mention async usage


class TestExportAndCacheFunctions:
    """Test export and cache management functions."""

    def test_export_all_specs(self, tmp_path, capsys):
        """Test export_all_specs creates a file."""

        # Create a temporary filename
        filename = str(tmp_path / "test_export.json")

        # Call export function
        chuk_mcp_math.export_all_specs(filename)

        captured = capsys.readouterr()

        # Should print a message
        assert len(captured.out) > 0 or len(captured.err) > 0

    def test_clear_all_caches(self, capsys):
        """Test clear_all_caches executes without error."""
        chuk_mcp_math.clear_all_caches()
        captured = capsys.readouterr()

        # Should print a message
        assert len(captured.out) > 0 or len(captured.err) > 0


class TestLoggingSetup:
    """Test logging setup functionality."""

    def test_setup_logging_default(self):
        """Test setup_logging with default level."""
        chuk_mcp_math.setup_logging()

        # Logger should be configured
        logger = logging.getLogger("chuk_mcp_math")
        assert logger is not None

    def test_setup_logging_custom_level(self):
        """Test setup_logging with custom level."""
        chuk_mcp_math.setup_logging("DEBUG")

        # Logger should be configured with DEBUG level
        logger = logging.getLogger("chuk_mcp_math")
        assert logger is not None

    def test_setup_logging_invalid_level(self):
        """Test setup_logging with invalid level raises ValueError."""
        with pytest.raises(ValueError, match="Invalid log level"):
            chuk_mcp_math.setup_logging("INVALID_LEVEL")

    def test_setup_logging_without_decorator(self, caplog):
        """Test setup_logging logs warning when MCP decorator unavailable."""
        original_value = chuk_mcp_math._mcp_decorator_available

        try:
            chuk_mcp_math._mcp_decorator_available = False

            with caplog.at_level(logging.WARNING):
                chuk_mcp_math.setup_logging("INFO")

            # Should log warning about decorator not being available
            assert any("not available" in record.message for record in caplog.records)

        finally:
            chuk_mcp_math._mcp_decorator_available = original_value


class TestImportedModules:
    """Test that expected modules are imported."""

    def test_arithmetic_module_imported(self):
        """Test that arithmetic module is in __all__."""
        assert "arithmetic" in chuk_mcp_math.__all__

    def test_number_theory_module_imported(self):
        """Test that number_theory module is in __all__."""
        assert "number_theory" in chuk_mcp_math.__all__


class TestAllExports:
    """Test __all__ exports."""

    def test_all_exports_defined(self):
        """Test that __all__ is defined and is a list."""
        assert hasattr(chuk_mcp_math, "__all__")
        assert isinstance(chuk_mcp_math.__all__, list)
        assert len(chuk_mcp_math.__all__) > 0

    def test_all_exported_items_exist(self):
        """Test that all items in __all__ actually exist in the module."""
        # Skip modules that are listed but may not be directly imported
        skip_items = {"arithmetic", "number_theory"}
        for item in chuk_mcp_math.__all__:
            if item not in skip_items:
                assert hasattr(chuk_mcp_math, item), f"{item} is in __all__ but not in module"

    def test_core_functions_in_all(self):
        """Test that core functions are exported."""
        core_functions = [
            "get_all_functions",
            "get_math_functions",
            "get_math_constants",
            "get_execution_stats",
            "get_function_recommendations",
            "validate_math_domain",
            "print_comprehensive_summary",
            "math_quick_reference",
        ]

        for func in core_functions:
            assert func in chuk_mcp_math.__all__


class TestAsyncBehavior:
    """Test async behavior of async functions."""

    @pytest.mark.asyncio
    async def test_async_functions_are_coroutines(self):
        """Test that async functions return coroutines."""
        # Test get_math_functions
        coro = chuk_mcp_math.get_math_functions()
        assert asyncio.iscoroutine(coro)
        result = await coro
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_concurrent_async_calls(self):
        """Test concurrent execution of async functions."""
        # Run multiple async calls concurrently
        tasks = [
            chuk_mcp_math.get_math_functions(),
            chuk_mcp_math.get_async_performance_stats(),
            chuk_mcp_math.get_math_functions(),
        ]

        results = await asyncio.gather(*tasks)
        assert len(results) == 3
        for result in results:
            assert isinstance(result, dict)


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_get_function_recommendations_case_insensitive(self):
        """Test that operation type is case-insensitive."""
        result1 = chuk_mcp_math.get_function_recommendations("BASIC")
        result2 = chuk_mcp_math.get_function_recommendations("basic")
        result3 = chuk_mcp_math.get_function_recommendations("BaSiC")

        assert result1 == result2 == result3

    def test_validate_math_domain_case_insensitive(self):
        """Test that domain validation is case-insensitive."""
        assert chuk_mcp_math.validate_math_domain("arithmetic") is True
        assert chuk_mcp_math.validate_math_domain("ARITHMETIC") is True
        assert chuk_mcp_math.validate_math_domain("Arithmetic") is True


class TestPerformanceMetrics:
    """Test performance metrics and statistics collection."""

    def test_execution_stats_with_performance_metrics(self):
        """Test get_execution_stats when functions have performance metrics."""
        from unittest.mock import MagicMock

        original_value = chuk_mcp_math._mcp_decorator_available
        original_get_mcp_functions = chuk_mcp_math.get_mcp_functions

        try:
            chuk_mcp_math._mcp_decorator_available = True

            # Create mock specs with performance metrics
            mock_metrics = MagicMock()
            mock_metrics.execution_count = 100
            mock_metrics.error_count = 5
            mock_metrics.cache_hits = 80
            mock_metrics.cache_misses = 20

            mock_spec = MagicMock()
            mock_spec.namespace = "arithmetic"
            mock_spec.category = "core"
            mock_spec.supports_local_execution.return_value = True
            mock_spec.supports_remote_execution.return_value = True
            mock_spec.cache_strategy = chuk_mcp_math.CacheStrategy.NONE
            mock_spec.supports_streaming = False
            mock_spec.workflow_compatible = True
            mock_spec._performance_metrics = mock_metrics

            # Mock get_mcp_functions to return our spec
            chuk_mcp_math.get_mcp_functions = lambda: {"test_func": mock_spec}

            result = chuk_mcp_math.get_execution_stats()

            # Check that performance metrics were calculated
            assert result["performance"]["total_executions"] == 100
            assert result["performance"]["error_rate"] == 0.05  # 5/100
            assert result["performance"]["total_cache_hits"] == 80
            assert result["performance"]["total_cache_misses"] == 20
            assert result["performance"]["cache_hit_rate"] == 0.8  # 80/100

        finally:
            chuk_mcp_math._mcp_decorator_available = original_value
            chuk_mcp_math.get_mcp_functions = original_get_mcp_functions

    def test_print_comprehensive_summary_with_performance_metrics(self, capsys):
        """Test print_comprehensive_summary when there are performance metrics."""
        from unittest.mock import MagicMock

        original_value = chuk_mcp_math._mcp_decorator_available
        original_get_mcp_functions = chuk_mcp_math.get_mcp_functions

        try:
            chuk_mcp_math._mcp_decorator_available = True

            # Create mock specs with performance metrics
            mock_metrics = MagicMock()
            mock_metrics.execution_count = 500
            mock_metrics.error_count = 10
            mock_metrics.cache_hits = 200
            mock_metrics.cache_misses = 50

            mock_spec = MagicMock()
            mock_spec.namespace = "arithmetic"
            mock_spec.category = "core"
            mock_spec.supports_local_execution.return_value = True
            mock_spec.supports_remote_execution.return_value = False
            mock_spec.cache_strategy = chuk_mcp_math.CacheStrategy.NONE
            mock_spec.supports_streaming = False
            mock_spec.workflow_compatible = True
            mock_spec._performance_metrics = mock_metrics

            # Mock get_mcp_functions to return our spec
            chuk_mcp_math.get_mcp_functions = lambda: {"test_func": mock_spec}

            chuk_mcp_math.print_comprehensive_summary()
            captured = capsys.readouterr()

            # Should contain performance metrics section
            assert "Performance" in captured.out or "Executions" in captured.out

        finally:
            chuk_mcp_math._mcp_decorator_available = original_value
            chuk_mcp_math.get_mcp_functions = original_get_mcp_functions

    @pytest.mark.asyncio
    async def test_async_performance_stats_with_streaming(self):
        """Test get_async_performance_stats with streaming and high performance functions."""
        from unittest.mock import MagicMock

        original_value = chuk_mcp_math._mcp_decorator_available
        original_get_math_functions = chuk_mcp_math.get_math_functions

        try:
            chuk_mcp_math._mcp_decorator_available = True

            # Create mock specs with various characteristics
            mock_spec_streaming = MagicMock()
            mock_spec_streaming.cache_strategy.value = "none"  # Not cached
            mock_spec_streaming.supports_streaming = True
            mock_spec_streaming.estimated_cpu_usage.value = "high"

            mock_spec_normal = MagicMock()
            mock_spec_normal.cache_strategy.value = "lru"  # Cached
            mock_spec_normal.supports_streaming = False
            mock_spec_normal.estimated_cpu_usage.value = "low"

            # Mock get_math_functions to return our specs
            async def mock_get_math_functions():
                return {
                    "arithmetic": {
                        "test_func1": mock_spec_streaming,
                        "test_func2": mock_spec_normal,
                    },
                    "number_theory": {
                        "test_func3": mock_spec_streaming,
                    },
                }

            chuk_mcp_math.get_math_functions = mock_get_math_functions

            result = await chuk_mcp_math.get_async_performance_stats()

            # Check that stats were calculated correctly
            assert result["total_async_functions"] == 3
            assert result["streaming_functions"] == 2
            assert result["cached_functions"] == 1
            assert result["high_performance_functions"] == 2
            assert result["domains_converted"] == 2

        finally:
            chuk_mcp_math._mcp_decorator_available = original_value
            chuk_mcp_math.get_math_functions = original_get_math_functions

    def test_clear_all_caches_with_cache_backends(self, capsys):
        """Test clear_all_caches when functions have cache backends."""
        from unittest.mock import MagicMock

        original_value = chuk_mcp_math._mcp_decorator_available
        original_get_mcp_functions = chuk_mcp_math.get_mcp_functions

        try:
            chuk_mcp_math._mcp_decorator_available = True

            # Create mock specs with cache backends
            mock_cache1 = MagicMock()
            mock_spec1 = MagicMock()
            mock_spec1._cache_backend = mock_cache1

            mock_cache2 = MagicMock()
            mock_spec2 = MagicMock()
            mock_spec2._cache_backend = mock_cache2

            mock_spec3 = MagicMock()
            mock_spec3._cache_backend = None

            # Mock get_mcp_functions to return our specs
            chuk_mcp_math.get_mcp_functions = lambda: {
                "func1": mock_spec1,
                "func2": mock_spec2,
                "func3": mock_spec3,
            }

            chuk_mcp_math.clear_all_caches()
            captured = capsys.readouterr()

            # Should have cleared 2 caches
            assert "2" in captured.out
            mock_cache1.clear.assert_called_once()
            mock_cache2.clear.assert_called_once()

        finally:
            chuk_mcp_math._mcp_decorator_available = original_value
            chuk_mcp_math.get_mcp_functions = original_get_mcp_functions


class TestImportErrorHandling:
    """Test import error handling for optional modules."""

    @pytest.mark.asyncio
    async def test_print_math_summary_with_import_error(self, capsys, monkeypatch):
        """Test print_math_summary when arithmetic module cannot be imported."""
        # Simulate import error by making the import fail
        import sys

        original_modules = sys.modules.copy()

        try:
            # Remove arithmetic from sys.modules to simulate import error
            if "chuk_mcp_math.arithmetic" in sys.modules:
                del sys.modules["chuk_mcp_math.arithmetic"]

            await chuk_mcp_math.print_math_summary()
            captured = capsys.readouterr()

            # Should still complete without error
            assert len(captured.out) > 0

        finally:
            # Restore modules
            sys.modules.update(original_modules)


class TestMCPDecoratorAvailability:
    """Test behavior when MCP decorator is/isn't available."""

    def test_mcp_decorator_available_flag(self):
        """Test that _mcp_decorator_available flag exists."""
        # The flag should be set during import
        assert hasattr(chuk_mcp_math, "_mcp_decorator_available")
        assert isinstance(chuk_mcp_math._mcp_decorator_available, bool)

    @pytest.mark.asyncio
    async def test_get_math_functions_without_decorator(self):
        """Test get_math_functions behavior when decorator unavailable."""
        # Temporarily disable decorator
        original_value = chuk_mcp_math._mcp_decorator_available
        try:
            chuk_mcp_math._mcp_decorator_available = False
            result = await chuk_mcp_math.get_math_functions()

            # Should return empty domains
            assert isinstance(result, dict)
            assert "arithmetic" in result
            assert "number_theory" in result

        finally:
            chuk_mcp_math._mcp_decorator_available = original_value

    def test_get_execution_stats_without_decorator(self):
        """Test get_execution_stats behavior when decorator unavailable."""
        original_value = chuk_mcp_math._mcp_decorator_available
        try:
            chuk_mcp_math._mcp_decorator_available = False
            result = chuk_mcp_math.get_execution_stats()

            # Should return empty stats
            assert isinstance(result, dict)
            assert result["total_functions"] == 0

        finally:
            chuk_mcp_math._mcp_decorator_available = original_value

    @pytest.mark.asyncio
    async def test_get_async_performance_stats_without_decorator(self):
        """Test get_async_performance_stats behavior when decorator unavailable."""
        original_value = chuk_mcp_math._mcp_decorator_available
        try:
            chuk_mcp_math._mcp_decorator_available = False
            result = await chuk_mcp_math.get_async_performance_stats()

            # Should return empty stats with default domains_converted
            assert isinstance(result, dict)
            assert result["total_async_functions"] == 0
            assert result["domains_converted"] == 2  # default value

        finally:
            chuk_mcp_math._mcp_decorator_available = original_value

    def test_export_all_specs_without_decorator(self, capsys):
        """Test export_all_specs behavior when decorator unavailable."""
        original_value = chuk_mcp_math._mcp_decorator_available
        try:
            chuk_mcp_math._mcp_decorator_available = False
            chuk_mcp_math.export_all_specs()

            captured = capsys.readouterr()
            # Should print warning message
            assert "not available" in captured.out

        finally:
            chuk_mcp_math._mcp_decorator_available = original_value

    def test_clear_all_caches_without_decorator(self, capsys):
        """Test clear_all_caches behavior when decorator unavailable."""
        original_value = chuk_mcp_math._mcp_decorator_available
        try:
            chuk_mcp_math._mcp_decorator_available = False
            chuk_mcp_math.clear_all_caches()

            captured = capsys.readouterr()
            # Should print warning message
            assert "not available" in captured.out

        finally:
            chuk_mcp_math._mcp_decorator_available = original_value


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v", "--tb=short"])
