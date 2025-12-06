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

    def test_matrices_all_is_empty(self):
        """Test that __all__ is empty (no functions implemented yet)."""
        from chuk_mcp_math.linear_algebra.matrices import __all__

        # Should be empty as functions are not yet implemented
        assert __all__ == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
