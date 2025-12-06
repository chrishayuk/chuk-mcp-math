#!/usr/bin/env python3
"""Test matrices __init__.py module."""

import pytest


class TestMatricesInit:
    """Test matrices module initialization."""

    def test_matrices_module_import(self):
        """Test that matrices module can be imported."""
        import chuk_mcp_math.linear_algebra.matrices as matrices_module

        assert hasattr(matrices_module, "__all__")
        assert isinstance(matrices_module.__all__, list)

    def test_matrices_all_has_functions(self):
        """Test that __all__ contains the implemented functions."""
        from chuk_mcp_math.linear_algebra.matrices import __all__

        # Should contain the matrix functions we implemented
        expected_functions = [
            "matrix_add",
            "matrix_subtract",
            "matrix_multiply",
            "matrix_transpose",
            "matrix_scalar_multiply",
            "matrix_det_2x2",
            "matrix_det_3x3",
            "matrix_solve_2x2",
            "matrix_solve_3x3",
            "gaussian_elimination",
        ]

        assert len(__all__) > 0
        # All expected functions should be in __all__
        for func in expected_functions:
            assert func in __all__


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
