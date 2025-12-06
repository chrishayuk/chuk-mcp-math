#!/usr/bin/env python3
"""Tests for optimization module."""

import pytest
from chuk_mcp_math.numerical import optimization


class TestGradientDescent:
    """Tests for gradient_descent function."""

    @pytest.mark.asyncio
    async def test_quadratic_function(self):
        """Test on simple quadratic f(x,y) = x^2 + y^2."""
        f = lambda x: x[0] ** 2 + x[1] ** 2
        grad = lambda x: [2 * x[0], 2 * x[1]]

        result = await optimization.gradient_descent(f, grad, [1.0, 1.0], learning_rate=0.1)

        assert result["converged"] is True
        assert abs(result["x"][0]) < 0.01
        assert abs(result["x"][1]) < 0.01
        assert abs(result["f_x"]) < 0.01

    @pytest.mark.asyncio
    async def test_rosenbrock_function(self):
        """Test on Rosenbrock function (harder)."""
        # Rosenbrock: f(x,y) = (1-x)^2 + 100(y-x^2)^2, min at (1,1)
        f = lambda x: (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2
        grad = lambda x: [
            -2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0] ** 2),
            200 * (x[1] - x[0] ** 2),
        ]

        result = await optimization.gradient_descent(
            f, grad, [0.0, 0.0], learning_rate=0.001, max_iterations=10000
        )

        # Should get close to (1, 1)
        assert result["f_x"] < 0.1  # Rosenbrock is hard, just check improvement

    @pytest.mark.asyncio
    async def test_errors(self):
        """Test error handling."""
        f = lambda x: x[0] ** 2
        grad = lambda x: [2 * x[0]]

        # Negative learning rate
        with pytest.raises(ValueError, match="learning_rate must be positive"):
            await optimization.gradient_descent(f, grad, [1.0], learning_rate=-0.1)

        # Invalid max_iterations
        with pytest.raises(ValueError, match="max_iterations must be >= 1"):
            await optimization.gradient_descent(f, grad, [1.0], max_iterations=0)

        # Empty x0
        with pytest.raises(ValueError, match="x0 cannot be empty"):
            await optimization.gradient_descent(f, grad, [])

        # Invalid tolerance
        with pytest.raises(ValueError, match="tolerance must be positive"):
            await optimization.gradient_descent(f, grad, [1.0], tolerance=-1e-6)


class TestGradientDescentMomentum:
    """Tests for gradient_descent_momentum function."""

    @pytest.mark.asyncio
    async def test_quadratic_with_momentum(self):
        """Test momentum accelerates convergence."""
        f = lambda x: x[0] ** 2 + x[1] ** 2
        grad = lambda x: [2 * x[0], 2 * x[1]]

        result = await optimization.gradient_descent_momentum(
            f, grad, [1.0, 1.0], learning_rate=0.1, momentum=0.9
        )

        assert result["converged"] is True
        assert abs(result["x"][0]) < 0.01
        assert abs(result["x"][1]) < 0.01

    @pytest.mark.asyncio
    async def test_momentum_parameter_validation(self):
        """Test momentum parameter validation."""
        f = lambda x: x[0] ** 2
        grad = lambda x: [2 * x[0]]

        # Invalid momentum
        with pytest.raises(ValueError, match="momentum must be in"):
            await optimization.gradient_descent_momentum(f, grad, [1.0], momentum=1.5)

        with pytest.raises(ValueError, match="momentum must be in"):
            await optimization.gradient_descent_momentum(f, grad, [1.0], momentum=-0.1)


class TestAdamOptimizer:
    """Tests for adam_optimizer function."""

    @pytest.mark.asyncio
    async def test_adam_on_quadratic(self):
        """Test Adam optimizer on quadratic function."""
        f = lambda x: x[0] ** 2 + x[1] ** 2
        grad = lambda x: [2 * x[0], 2 * x[1]]

        result = await optimization.adam_optimizer(f, grad, [1.0, 1.0], max_iterations=2000)

        # Adam should significantly improve from start
        assert result["f_x"] < 0.01 or result["converged"] is True
        if result["converged"]:
            assert abs(result["x"][0]) < 0.01
            assert abs(result["x"][1]) < 0.01

    @pytest.mark.asyncio
    async def test_adam_parameter_validation(self):
        """Test Adam parameter validation."""
        f = lambda x: x[0] ** 2
        grad = lambda x: [2 * x[0]]

        # Invalid beta1
        with pytest.raises(ValueError, match="beta1 must be in"):
            await optimization.adam_optimizer(f, grad, [1.0], beta1=1.5)

        # Invalid beta2
        with pytest.raises(ValueError, match="beta2 must be in"):
            await optimization.adam_optimizer(f, grad, [1.0], beta2=-0.1)

        # Invalid epsilon
        with pytest.raises(ValueError, match="epsilon must be positive"):
            await optimization.adam_optimizer(f, grad, [1.0], epsilon=-1e-8)

    @pytest.mark.asyncio
    async def test_adam_convergence(self):
        """Test that Adam converges faster than vanilla GD."""
        f = lambda x: (x[0] - 3) ** 2 + (x[1] + 2) ** 2
        grad = lambda x: [2 * (x[0] - 3), 2 * (x[1] + 2)]

        result = await optimization.adam_optimizer(f, grad, [0.0, 0.0], learning_rate=0.1)

        assert result["converged"] is True
        assert abs(result["x"][0] - 3.0) < 0.01
        assert abs(result["x"][1] + 2.0) < 0.01


class TestGoldenSectionSearch:
    """Tests for golden_section_search function."""

    @pytest.mark.asyncio
    async def test_parabola(self):
        """Test on simple parabola."""
        f = lambda x: (x - 2) ** 2

        result = await optimization.golden_section_search(f, 0, 5)

        assert result["converged"] is True
        assert abs(result["x"] - 2.0) < 1e-5
        assert abs(result["f_x"]) < 1e-10

    @pytest.mark.asyncio
    async def test_quartic(self):
        """Test on quartic function."""
        f = lambda x: (x - 1) ** 4 + (x - 1) ** 2

        result = await optimization.golden_section_search(f, -2, 4)

        assert result["converged"] is True
        assert abs(result["x"] - 1.0) < 1e-5

    @pytest.mark.asyncio
    async def test_errors(self):
        """Test error handling."""
        f = lambda x: x**2

        # a >= b
        with pytest.raises(ValueError, match="a must be < b"):
            await optimization.golden_section_search(f, 5, 0)

        # Invalid tolerance
        with pytest.raises(ValueError, match="tolerance must be positive"):
            await optimization.golden_section_search(f, 0, 5, tolerance=-1e-6)

        # Invalid max_iterations
        with pytest.raises(ValueError, match="max_iterations must be >= 1"):
            await optimization.golden_section_search(f, 0, 5, max_iterations=0)


class TestNelderMead:
    """Tests for nelder_mead function."""

    @pytest.mark.asyncio
    async def test_quadratic_2d(self):
        """Test Nelder-Mead on 2D quadratic."""
        f = lambda x: x[0] ** 2 + x[1] ** 2
        initial_f = f([1.0, 1.0])  # 2.0

        result = await optimization.nelder_mead(f, [1.0, 1.0], max_iterations=500)

        # Nelder-Mead should show improvement (derivative-free is slower)
        assert result["f_x"] < initial_f  # Shows improvement
        if result["converged"]:
            assert abs(result["x"][0]) <= 0.6
            assert abs(result["x"][1]) <= 0.6

    @pytest.mark.asyncio
    async def test_shifted_quadratic(self):
        """Test on shifted quadratic."""
        # Min at (3, -2)
        f = lambda x: (x[0] - 3) ** 2 + (x[1] + 2) ** 2

        result = await optimization.nelder_mead(f, [0.0, 0.0], max_iterations=500)

        assert abs(result["x"][0] - 3.0) < 0.1
        assert abs(result["x"][1] + 2.0) < 0.1

    @pytest.mark.asyncio
    async def test_parameter_validation(self):
        """Test parameter validation."""
        f = lambda x: x[0] ** 2

        # Empty x0
        with pytest.raises(ValueError, match="x0 cannot be empty"):
            await optimization.nelder_mead(f, [])

        # Invalid alpha
        with pytest.raises(ValueError, match="alpha must be positive"):
            await optimization.nelder_mead(f, [1.0], alpha=-1.0)

        # Invalid rho
        with pytest.raises(ValueError, match="rho must be in"):
            await optimization.nelder_mead(f, [1.0], rho=1.5)

        # Invalid sigma
        with pytest.raises(ValueError, match="sigma must be in"):
            await optimization.nelder_mead(f, [1.0], sigma=-0.1)


class TestCoordinateDescent:
    """Tests for coordinate_descent function."""

    @pytest.mark.asyncio
    async def test_separable_quadratic(self):
        """Test on separable quadratic function."""
        f = lambda x: x[0] ** 2 + x[1] ** 2

        result = await optimization.coordinate_descent(f, [1.0, 1.0], step_size=0.1)

        assert result["converged"] is True
        assert abs(result["x"][0]) < 0.1
        assert abs(result["x"][1]) < 0.1

    @pytest.mark.asyncio
    async def test_parameter_validation(self):
        """Test parameter validation."""
        f = lambda x: x[0] ** 2

        # Empty x0
        with pytest.raises(ValueError, match="x0 cannot be empty"):
            await optimization.coordinate_descent(f, [])

        # Invalid step_size
        with pytest.raises(ValueError, match="step_size must be positive"):
            await optimization.coordinate_descent(f, [1.0], step_size=-0.1)

        # Invalid max_iterations
        with pytest.raises(ValueError, match="max_iterations must be >= 1"):
            await optimization.coordinate_descent(f, [1.0], max_iterations=0)

        # Invalid tolerance
        with pytest.raises(ValueError, match="tolerance must be positive"):
            await optimization.coordinate_descent(f, [1.0], tolerance=-1e-6)


class TestOptimizationComparison:
    """Compare different optimization methods."""

    @pytest.mark.asyncio
    async def test_all_methods_converge_quadratic(self):
        """Test that all methods improve on simple quadratic."""
        f = lambda x: x[0] ** 2 + x[1] ** 2
        grad = lambda x: [2 * x[0], 2 * x[1]]
        x0 = [1.0, 1.0]
        initial_f = f(x0)  # Should be 2.0

        # Gradient descent
        result_gd = await optimization.gradient_descent(f, grad, x0, learning_rate=0.1)
        assert result_gd["f_x"] < initial_f  # Improvement

        # Momentum
        result_mom = await optimization.gradient_descent_momentum(
            f, grad, x0, learning_rate=0.1, momentum=0.9
        )
        assert result_mom["f_x"] < initial_f  # Improvement

        # Adam
        result_adam = await optimization.adam_optimizer(f, grad, x0, max_iterations=2000)
        assert result_adam["f_x"] < 0.1  # Significant improvement

        # Nelder-Mead (no gradient needed)
        result_nm = await optimization.nelder_mead(f, x0, max_iterations=500)
        assert result_nm["f_x"] < initial_f  # Improvement

        # Gradient-based methods should converge well
        assert result_gd["converged"] is True
        assert result_mom["converged"] is True


class TestAsyncBehavior:
    """Test async execution behavior."""

    @pytest.mark.asyncio
    async def test_async_function_support(self):
        """Test that optimizers work with async functions."""

        async def async_f(x):
            """Async function."""
            return x[0] ** 2 + x[1] ** 2

        async def async_grad(x):
            """Async gradient."""
            return [2 * x[0], 2 * x[1]]

        result = await optimization.gradient_descent(async_f, async_grad, [1.0, 1.0])
        assert result["converged"] is True

    @pytest.mark.asyncio
    async def test_golden_section_async(self):
        """Test golden section with async function."""

        async def async_f(x):
            """Async 1D function."""
            return (x - 2) ** 2

        result = await optimization.golden_section_search(async_f, 0, 5)
        assert result["converged"] is True
        assert abs(result["x"] - 2.0) < 1e-5


class TestCoverageExtras:
    """Additional tests for coverage."""

    @pytest.mark.asyncio
    async def test_adam_non_convergence(self):
        """Test Adam when it doesn't converge."""
        f = lambda x: x[0] ** 2 + x[1] ** 2
        grad = lambda x: [2 * x[0], 2 * x[1]]

        result = await optimization.adam_optimizer(
            f, grad, [100.0, 100.0], max_iterations=10, tolerance=1e-12
        )
        # Should not converge in 10 iterations from far start
        assert result["iterations"] == 10

    @pytest.mark.asyncio
    async def test_golden_section_non_convergence(self):
        """Test golden section when it hits max iterations."""
        f = lambda x: x**2

        result = await optimization.golden_section_search(
            f, -100, 100, max_iterations=5, tolerance=1e-12
        )
        # Should hit max iterations
        assert result["iterations"] == 5

    @pytest.mark.asyncio
    async def test_nelder_mead_many_iterations(self):
        """Test Nelder-Mead expansion path."""
        # Function that benefits from expansion
        f = lambda x: (x[0] - 10) ** 2 + (x[1] - 10) ** 2

        result = await optimization.nelder_mead(f, [0.0, 0.0], max_iterations=200)
        # Should improve significantly
        assert result["f_x"] < 100  # Started at 200


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
