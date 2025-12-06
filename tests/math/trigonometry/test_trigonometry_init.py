#!/usr/bin/env python3
# tests/math/trigonometry/test_init.py
"""
Comprehensive pytest unit tests for src/chuk_mcp_math/trigonometry/__init__.py

Tests cover:
- Submodule imports (basic_functions, inverse_functions, hyperbolic, etc.)
- Function imports from each submodule
- __all__ exports
- Helper functions (get_trigonometry_functions, get_trig_constants, etc.)
- Print functions (print_trigonometry_summary, trigonometry_quick_reference)
- Utility functions (get_function_recommendations, validate_trig_domain)
- Module documentation
"""

import pytest
import math

# Import the module to test
from chuk_mcp_math import trigonometry


class TestSubmoduleImports:
    """Test that all submodules are imported correctly."""

    def test_basic_functions_module_imported(self):
        """Test that basic_functions module is imported."""
        assert hasattr(trigonometry, "basic_functions")
        assert trigonometry.basic_functions is not None

    def test_inverse_functions_module_imported(self):
        """Test that inverse_functions module is imported."""
        assert hasattr(trigonometry, "inverse_functions")
        assert trigonometry.inverse_functions is not None

    def test_hyperbolic_module_imported(self):
        """Test that hyperbolic module is imported."""
        assert hasattr(trigonometry, "hyperbolic")
        assert trigonometry.hyperbolic is not None

    def test_inverse_hyperbolic_module_imported(self):
        """Test that inverse_hyperbolic module is imported."""
        assert hasattr(trigonometry, "inverse_hyperbolic")
        assert trigonometry.inverse_hyperbolic is not None

    def test_angle_conversion_module_imported(self):
        """Test that angle_conversion module is imported."""
        assert hasattr(trigonometry, "angle_conversion")
        assert trigonometry.angle_conversion is not None

    def test_identities_module_imported(self):
        """Test that identities module is imported."""
        assert hasattr(trigonometry, "identities")
        assert trigonometry.identities is not None

    def test_wave_analysis_module_imported(self):
        """Test that wave_analysis module is imported."""
        assert hasattr(trigonometry, "wave_analysis")
        assert trigonometry.wave_analysis is not None

    def test_applications_module_imported(self):
        """Test that applications module is imported."""
        assert hasattr(trigonometry, "applications")
        assert trigonometry.applications is not None


class TestBasicTrigFunctions:
    """Test that basic trigonometric functions are imported."""

    def test_sin_imported(self):
        """Test that sin function is imported."""
        assert hasattr(trigonometry, "sin")
        assert callable(trigonometry.sin)

    def test_cos_imported(self):
        """Test that cos function is imported."""
        assert hasattr(trigonometry, "cos")
        assert callable(trigonometry.cos)

    def test_tan_imported(self):
        """Test that tan function is imported."""
        assert hasattr(trigonometry, "tan")
        assert callable(trigonometry.tan)

    def test_csc_imported(self):
        """Test that csc function is imported."""
        assert hasattr(trigonometry, "csc")
        assert callable(trigonometry.csc)

    def test_sec_imported(self):
        """Test that sec function is imported."""
        assert hasattr(trigonometry, "sec")
        assert callable(trigonometry.sec)

    def test_cot_imported(self):
        """Test that cot function is imported."""
        assert hasattr(trigonometry, "cot")
        assert callable(trigonometry.cot)

    def test_sin_degrees_imported(self):
        """Test that sin_degrees function is imported."""
        assert hasattr(trigonometry, "sin_degrees")
        assert callable(trigonometry.sin_degrees)

    def test_cos_degrees_imported(self):
        """Test that cos_degrees function is imported."""
        assert hasattr(trigonometry, "cos_degrees")
        assert callable(trigonometry.cos_degrees)

    def test_tan_degrees_imported(self):
        """Test that tan_degrees function is imported."""
        assert hasattr(trigonometry, "tan_degrees")
        assert callable(trigonometry.tan_degrees)


class TestInverseTrigFunctions:
    """Test that inverse trigonometric functions are imported."""

    def test_asin_imported(self):
        """Test that asin function is imported."""
        assert hasattr(trigonometry, "asin")
        assert callable(trigonometry.asin)

    def test_acos_imported(self):
        """Test that acos function is imported."""
        assert hasattr(trigonometry, "acos")
        assert callable(trigonometry.acos)

    def test_atan_imported(self):
        """Test that atan function is imported."""
        assert hasattr(trigonometry, "atan")
        assert callable(trigonometry.atan)

    def test_atan2_imported(self):
        """Test that atan2 function is imported."""
        assert hasattr(trigonometry, "atan2")
        assert callable(trigonometry.atan2)

    def test_acsc_imported(self):
        """Test that acsc function is imported."""
        assert hasattr(trigonometry, "acsc")
        assert callable(trigonometry.acsc)

    def test_asec_imported(self):
        """Test that asec function is imported."""
        assert hasattr(trigonometry, "asec")
        assert callable(trigonometry.asec)

    def test_acot_imported(self):
        """Test that acot function is imported."""
        assert hasattr(trigonometry, "acot")
        assert callable(trigonometry.acot)


class TestHyperbolicFunctions:
    """Test that hyperbolic functions are imported."""

    def test_sinh_imported(self):
        """Test that sinh function is imported."""
        assert hasattr(trigonometry, "sinh")
        assert callable(trigonometry.sinh)

    def test_cosh_imported(self):
        """Test that cosh function is imported."""
        assert hasattr(trigonometry, "cosh")
        assert callable(trigonometry.cosh)

    def test_tanh_imported(self):
        """Test that tanh function is imported."""
        assert hasattr(trigonometry, "tanh")
        assert callable(trigonometry.tanh)

    def test_csch_imported(self):
        """Test that csch function is imported."""
        assert hasattr(trigonometry, "csch")
        assert callable(trigonometry.csch)

    def test_sech_imported(self):
        """Test that sech function is imported."""
        assert hasattr(trigonometry, "sech")
        assert callable(trigonometry.sech)

    def test_coth_imported(self):
        """Test that coth function is imported."""
        assert hasattr(trigonometry, "coth")
        assert callable(trigonometry.coth)


class TestInverseHyperbolicFunctions:
    """Test that inverse hyperbolic functions are imported."""

    def test_asinh_imported(self):
        """Test that asinh function is imported."""
        assert hasattr(trigonometry, "asinh")
        assert callable(trigonometry.asinh)

    def test_acosh_imported(self):
        """Test that acosh function is imported."""
        assert hasattr(trigonometry, "acosh")
        assert callable(trigonometry.acosh)

    def test_atanh_imported(self):
        """Test that atanh function is imported."""
        assert hasattr(trigonometry, "atanh")
        assert callable(trigonometry.atanh)

    def test_acsch_imported(self):
        """Test that acsch function is imported."""
        assert hasattr(trigonometry, "acsch")
        assert callable(trigonometry.acsch)

    def test_asech_imported(self):
        """Test that asech function is imported."""
        assert hasattr(trigonometry, "asech")
        assert callable(trigonometry.asech)

    def test_acoth_imported(self):
        """Test that acoth function is imported."""
        assert hasattr(trigonometry, "acoth")
        assert callable(trigonometry.acoth)


class TestAngleConversionFunctions:
    """Test that angle conversion functions are imported."""

    def test_degrees_to_radians_imported(self):
        """Test that degrees_to_radians function is imported."""
        assert hasattr(trigonometry, "degrees_to_radians")
        assert callable(trigonometry.degrees_to_radians)

    def test_radians_to_degrees_imported(self):
        """Test that radians_to_degrees function is imported."""
        assert hasattr(trigonometry, "radians_to_degrees")
        assert callable(trigonometry.radians_to_degrees)

    def test_gradians_to_radians_imported(self):
        """Test that gradians_to_radians function is imported."""
        assert hasattr(trigonometry, "gradians_to_radians")
        assert callable(trigonometry.gradians_to_radians)

    def test_normalize_angle_imported(self):
        """Test that normalize_angle function is imported."""
        assert hasattr(trigonometry, "normalize_angle")
        assert callable(trigonometry.normalize_angle)

    def test_angle_difference_imported(self):
        """Test that angle_difference function is imported."""
        assert hasattr(trigonometry, "angle_difference")
        assert callable(trigonometry.angle_difference)


class TestIdentityFunctions:
    """Test that identity functions are imported."""

    def test_pythagorean_identity_imported(self):
        """Test that pythagorean_identity function is imported."""
        assert hasattr(trigonometry, "pythagorean_identity")
        assert callable(trigonometry.pythagorean_identity)

    def test_sum_difference_formulas_imported(self):
        """Test that sum_difference_formulas function is imported."""
        assert hasattr(trigonometry, "sum_difference_formulas")
        assert callable(trigonometry.sum_difference_formulas)

    def test_double_angle_formulas_imported(self):
        """Test that double_angle_formulas function is imported."""
        assert hasattr(trigonometry, "double_angle_formulas")
        assert callable(trigonometry.double_angle_formulas)

    def test_verify_identity_imported(self):
        """Test that verify_identity function is imported."""
        assert hasattr(trigonometry, "verify_identity")
        assert callable(trigonometry.verify_identity)


class TestWaveAnalysisFunctions:
    """Test that wave analysis functions are imported."""

    def test_amplitude_from_coefficients_imported(self):
        """Test that amplitude_from_coefficients function is imported."""
        assert hasattr(trigonometry, "amplitude_from_coefficients")
        assert callable(trigonometry.amplitude_from_coefficients)

    def test_frequency_from_period_imported(self):
        """Test that frequency_from_period function is imported."""
        assert hasattr(trigonometry, "frequency_from_period")
        assert callable(trigonometry.frequency_from_period)

    def test_phase_shift_analysis_imported(self):
        """Test that phase_shift_analysis function is imported."""
        assert hasattr(trigonometry, "phase_shift_analysis")
        assert callable(trigonometry.phase_shift_analysis)


class TestApplicationFunctions:
    """Test that application functions are imported."""

    def test_distance_haversine_imported(self):
        """Test that distance_haversine function is imported."""
        assert hasattr(trigonometry, "distance_haversine")
        assert callable(trigonometry.distance_haversine)

    def test_bearing_calculation_imported(self):
        """Test that bearing_calculation function is imported."""
        assert hasattr(trigonometry, "bearing_calculation")
        assert callable(trigonometry.bearing_calculation)

    def test_triangulation_imported(self):
        """Test that triangulation function is imported."""
        assert hasattr(trigonometry, "triangulation")
        assert callable(trigonometry.triangulation)


class TestAllExports:
    """Test __all__ exports."""

    def test_all_defined(self):
        """Test that __all__ is defined."""
        assert hasattr(trigonometry, "__all__")
        assert isinstance(trigonometry.__all__, list)
        assert len(trigonometry.__all__) > 0

    def test_all_contains_submodules(self):
        """Test that __all__ contains submodule names."""
        expected_modules = [
            "basic_functions",
            "inverse_functions",
            "hyperbolic",
            "inverse_hyperbolic",
            "angle_conversion",
            "identities",
            "wave_analysis",
            "applications",
        ]

        for module in expected_modules:
            assert module in trigonometry.__all__, f"{module} not in __all__"

    def test_all_contains_basic_functions(self):
        """Test that __all__ contains basic trig functions."""
        basic_funcs = ["sin", "cos", "tan", "csc", "sec", "cot"]

        for func in basic_funcs:
            assert func in trigonometry.__all__, f"{func} not in __all__"

    def test_all_items_exist(self):
        """Test that all items in __all__ actually exist."""
        for item in trigonometry.__all__:
            assert hasattr(trigonometry, item), f"{item} in __all__ but not in module"


class TestHelperFunctions:
    """Test helper/utility functions."""

    @pytest.mark.asyncio
    async def test_get_trigonometry_functions(self):
        """Test get_trigonometry_functions returns organized domains."""
        result = await trigonometry.get_trigonometry_functions()
        assert isinstance(result, dict)

        # Should have all 8 domains
        expected_domains = [
            "basic_functions",
            "inverse_functions",
            "hyperbolic",
            "inverse_hyperbolic",
            "angle_conversion",
            "identities",
            "wave_analysis",
            "applications",
        ]

        for domain in expected_domains:
            assert domain in result

    def test_get_trig_constants(self):
        """Test get_trig_constants returns all constants."""
        result = trigonometry.get_trig_constants()
        assert isinstance(result, dict)

        # Check for expected constants
        expected_constants = [
            "pi",
            "tau",
            "e",
            "pi_2",
            "pi_4",
            "pi_3",
            "pi_6",
            "sqrt_2",
            "sqrt_3",
            "golden_ratio",
            "degrees_per_radian",
            "radians_per_degree",
        ]

        for const in expected_constants:
            assert const in result

    def test_trig_constants_values(self):
        """Test that trig constants have correct values."""
        result = trigonometry.get_trig_constants()

        assert result["pi"] == math.pi
        assert result["tau"] == math.tau
        assert result["e"] == math.e
        assert pytest.approx(result["pi_2"], rel=1e-10) == math.pi / 2
        assert pytest.approx(result["sqrt_2"], rel=1e-10) == math.sqrt(2)

    def test_get_function_recommendations(self):
        """Test get_function_recommendations returns appropriate functions."""
        # Test different operation types
        test_cases = [
            ("basic", ["sin", "cos", "tan", "sin_degrees", "cos_degrees", "tan_degrees"]),
            ("hyperbolic", ["sinh", "cosh", "tanh", "csch", "sech", "coth"]),
            ("conversion", ["degrees_to_radians", "radians_to_degrees", "normalize_angle"]),
        ]

        for operation_type, expected_funcs in test_cases:
            result = trigonometry.get_function_recommendations(operation_type)
            assert isinstance(result, list)
            assert result == expected_funcs

    def test_get_function_recommendations_unknown_type(self):
        """Test get_function_recommendations with unknown operation type."""
        result = trigonometry.get_function_recommendations("nonexistent_operation")
        assert isinstance(result, list)
        assert len(result) == 0

    def test_validate_trig_domain_valid(self):
        """Test validate_trig_domain with valid domains."""
        valid_domains = [
            "basic_functions",
            "inverse_functions",
            "hyperbolic",
            "angle_conversion",
            "identities",
            "wave_analysis",
        ]

        for domain in valid_domains:
            result = trigonometry.validate_trig_domain(domain)
            assert result is True

    def test_validate_trig_domain_invalid(self):
        """Test validate_trig_domain with invalid domains."""
        invalid_domains = ["arithmetic", "number_theory", "nonexistent", ""]

        for domain in invalid_domains:
            result = trigonometry.validate_trig_domain(domain)
            assert result is False

    def test_validate_trig_domain_case_insensitive(self):
        """Test that domain validation is case-insensitive."""
        assert trigonometry.validate_trig_domain("basic_functions") is True
        assert trigonometry.validate_trig_domain("BASIC_FUNCTIONS") is True
        assert trigonometry.validate_trig_domain("Basic_Functions") is True


class TestPrintFunctions:
    """Test print/display functions."""

    @pytest.mark.asyncio
    async def test_print_trigonometry_summary(self, capsys):
        """Test print_trigonometry_summary outputs to console."""
        await trigonometry.print_trigonometry_summary()
        captured = capsys.readouterr()

        # Check that output was generated
        assert len(captured.out) > 0
        assert "Trigonometric" in captured.out or "Functions" in captured.out

    def test_trigonometry_quick_reference(self):
        """Test trigonometry_quick_reference returns reference guide."""
        result = trigonometry.trigonometry_quick_reference()
        assert isinstance(result, str)
        assert len(result) > 0

        # Should contain key sections
        assert "Trigonometric Functions" in result or "FUNCTIONS" in result
        assert "await" in result  # Should mention async usage


class TestAsyncPerformanceStats:
    """Test async performance statistics."""

    @pytest.mark.asyncio
    async def test_get_async_performance_stats(self):
        """Test get_async_performance_stats returns stats."""
        result = await trigonometry.get_async_performance_stats()
        assert isinstance(result, dict)

        # Check structure
        assert "total_async_functions" in result
        assert "cached_functions" in result
        assert "streaming_functions" in result
        assert "high_performance_functions" in result
        assert "domains_implemented" in result

        # Check types
        assert isinstance(result["total_async_functions"], int)
        assert isinstance(result["domains_implemented"], int)


class TestModuleDocumentation:
    """Test module documentation."""

    def test_module_has_docstring(self):
        """Test that trigonometry module has a docstring."""
        assert trigonometry.__doc__ is not None
        assert len(trigonometry.__doc__) > 0

    def test_docstring_mentions_trigonometry(self):
        """Test that docstring mentions trigonometry."""
        docstring = trigonometry.__doc__.upper()
        assert "TRIGONOMETRY" in docstring or "TRIGONOMETRIC" in docstring

    def test_docstring_mentions_async(self):
        """Test that docstring mentions async functionality."""
        docstring = trigonometry.__doc__.lower()
        assert "async" in docstring

    def test_docstring_lists_domains(self):
        """Test that docstring lists trigonometric domains."""
        docstring = trigonometry.__doc__.lower()
        # Should mention at least some domains
        assert "basic_functions" in docstring or "hyperbolic" in docstring


class TestMCPDecoratorAvailability:
    """Test behavior related to MCP decorator availability."""

    def test_mcp_decorator_available_flag(self):
        """Test that _mcp_decorator_available flag exists."""
        assert hasattr(trigonometry, "_mcp_decorator_available")
        assert isinstance(trigonometry._mcp_decorator_available, bool)

    @pytest.mark.asyncio
    async def test_get_trigonometry_functions_without_decorator(self):
        """Test get_trigonometry_functions when decorator unavailable."""
        original_value = trigonometry._mcp_decorator_available
        try:
            trigonometry._mcp_decorator_available = False
            result = await trigonometry.get_trigonometry_functions()

            # Should return empty domains
            assert isinstance(result, dict)
            assert all(isinstance(v, dict) for v in result.values())

        finally:
            trigonometry._mcp_decorator_available = original_value

    @pytest.mark.asyncio
    async def test_get_async_performance_stats_without_decorator(self):
        """Test get_async_performance_stats when decorator unavailable."""
        original_value = trigonometry._mcp_decorator_available
        try:
            trigonometry._mcp_decorator_available = False
            result = await trigonometry.get_async_performance_stats()

            # Should return default stats
            assert isinstance(result, dict)
            assert result["total_async_functions"] == 0
            assert result["domains_implemented"] == 8

        finally:
            trigonometry._mcp_decorator_available = original_value


class TestFunctionCount:
    """Test that all expected functions are present."""

    def test_basic_functions_count(self):
        """Test that we have basic trig functions."""
        basic_funcs = ["sin", "cos", "tan", "csc", "sec", "cot"]
        count = sum(1 for func in basic_funcs if hasattr(trigonometry, func))
        assert count == len(basic_funcs)

    def test_inverse_functions_count(self):
        """Test that we have inverse trig functions."""
        inverse_funcs = ["asin", "acos", "atan", "atan2", "acsc", "asec", "acot"]
        count = sum(1 for func in inverse_funcs if hasattr(trigonometry, func))
        assert count == len(inverse_funcs)

    def test_hyperbolic_functions_count(self):
        """Test that we have hyperbolic functions."""
        hyperbolic_funcs = ["sinh", "cosh", "tanh", "csch", "sech", "coth"]
        count = sum(1 for func in hyperbolic_funcs if hasattr(trigonometry, func))
        assert count == len(hyperbolic_funcs)


class TestMainExecution:
    """Test main execution block behavior."""

    def test_main_block_does_not_execute_on_import(self):
        """Test that __name__ == '__main__' block doesn't run on import."""
        from chuk_mcp_math import trigonometry as test_trig

        assert test_trig is not None


class TestImportErrorHandling:
    """Test import error handling and edge cases."""

    @pytest.mark.asyncio
    async def test_import_error_path(self):
        """Test behavior when mcp_decorator import fails (line 50-51)."""
        # Save original value
        import chuk_mcp_math.trigonometry as trig_module

        original_value = trig_module._mcp_decorator_available

        try:
            # Simulate import failure
            trig_module._mcp_decorator_available = False

            # Test get_trigonometry_functions returns empty domains
            result = await trig_module.get_trigonometry_functions()
            assert isinstance(result, dict)
            assert all(v == {} for v in result.values())

            # Test get_async_performance_stats returns default values
            stats = await trig_module.get_async_performance_stats()
            assert stats["total_async_functions"] == 0
            assert stats["domains_implemented"] == 8
        finally:
            # Restore original value
            trig_module._mcp_decorator_available = original_value

    @pytest.mark.asyncio
    async def test_get_trigonometry_functions_with_domain_check(self):
        """Test line 160 - domain check in get_trigonometry_functions."""
        import chuk_mcp_math.trigonometry as trig_module

        # This tests the domain filtering logic at line 159-160
        result = await trig_module.get_trigonometry_functions()

        # Verify domains are correctly organized
        assert isinstance(result, dict)
        for domain in result:
            assert isinstance(result[domain], dict)

    @pytest.mark.asyncio
    async def test_get_async_performance_stats_detailed(self):
        """Test lines 315, 318-327 - async performance stats calculation."""
        import chuk_mcp_math.trigonometry as trig_module

        if trig_module._mcp_decorator_available:
            # Test the detailed stat calculation with real functions
            stats = await trig_module.get_async_performance_stats()

            # These lines (315-327) iterate through functions and check properties
            assert "domains_implemented" in stats
            assert "total_async_functions" in stats
            assert "cached_functions" in stats
            assert "streaming_functions" in stats
            assert "high_performance_functions" in stats

            # Verify counts are reasonable
            assert stats["domains_implemented"] >= 0
            assert stats["total_async_functions"] >= 0


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v", "--tb=short"])
