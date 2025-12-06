"""
Test Phase 1 Implementation

Tests for all Phase 1 functions:
- Linear Algebra (vectors + matrices)
- Calculus (derivatives, integration, root finding)
- Probability (normal distribution, sampling)
- Statistics (descriptive stats, regression)
"""

import pytest


@pytest.mark.asyncio
class TestLinearAlgebraVectors:
    """Test vector operations."""

    async def test_vector_add(self):
        from chuk_mcp_math.linear_algebra.vectors.operations import vector_add

        result = await vector_add([1, 2, 3], [4, 5, 6])
        assert result == [5.0, 7.0, 9.0]

    async def test_vector_subtract(self):
        from chuk_mcp_math.linear_algebra.vectors.operations import vector_subtract

        result = await vector_subtract([5, 7, 9], [1, 2, 3])
        assert result == [4.0, 5.0, 6.0]

    async def test_dot_product(self):
        from chuk_mcp_math.linear_algebra.vectors.operations import dot_product

        result = await dot_product([1, 2, 3], [4, 5, 6])
        assert result == 32.0

    async def test_vector_norm(self):
        from chuk_mcp_math.linear_algebra.vectors.norms import vector_norm

        result = await vector_norm([3, 4])
        assert result == 5.0

    async def test_normalize_vector(self):
        from chuk_mcp_math.linear_algebra.vectors.norms import normalize_vector

        result = await normalize_vector([3, 4])
        assert abs(result[0] - 0.6) < 1e-10
        assert abs(result[1] - 0.8) < 1e-10


@pytest.mark.asyncio
class TestLinearAlgebraMatrices:
    """Test matrix operations."""

    async def test_matrix_multiply(self):
        from chuk_mcp_math.linear_algebra.matrices.operations import matrix_multiply

        A = [[1, 2], [3, 4]]
        B = [[5, 6], [7, 8]]
        result = await matrix_multiply(A, B)
        assert result == [[19.0, 22.0], [43.0, 50.0]]

    async def test_matrix_multiply_errors(self):
        from chuk_mcp_math.linear_algebra.matrices.operations import matrix_multiply

        # Test empty matrix errors
        with pytest.raises(ValueError, match="First matrix cannot be empty"):
            await matrix_multiply([], [[1, 2]])

        with pytest.raises(ValueError, match="First matrix cannot be empty"):
            await matrix_multiply([[]], [[1, 2]])

        with pytest.raises(ValueError, match="Second matrix cannot be empty"):
            await matrix_multiply([[1, 2]], [])

        with pytest.raises(ValueError, match="Second matrix cannot be empty"):
            await matrix_multiply([[1, 2]], [[]])

        # Test dimension mismatch
        with pytest.raises(ValueError, match="dimensions incompatible"):
            await matrix_multiply([[1, 2]], [[1, 2], [3, 4], [5, 6]])

    async def test_matrix_multiply_large(self):
        from chuk_mcp_math.linear_algebra.matrices.operations import matrix_multiply

        # Test large matrix (triggers async yield)
        A = [[1] * 15 for _ in range(15)]  # 15x15 matrix
        B = [[1] * 15 for _ in range(15)]
        result = await matrix_multiply(A, B)
        assert len(result) == 15
        assert len(result[0]) == 15

    async def test_matrix_transpose(self):
        from chuk_mcp_math.linear_algebra.matrices.operations import matrix_transpose

        A = [[1, 2, 3], [4, 5, 6]]
        result = await matrix_transpose(A)
        assert result == [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]

    async def test_matrix_transpose_errors(self):
        from chuk_mcp_math.linear_algebra.matrices.operations import matrix_transpose

        # Test empty matrix
        with pytest.raises(ValueError, match="Matrix cannot be empty"):
            await matrix_transpose([])

        with pytest.raises(ValueError, match="Matrix cannot be empty"):
            await matrix_transpose([[]])

    async def test_matrix_transpose_large(self):
        from chuk_mcp_math.linear_algebra.matrices.operations import matrix_transpose

        # Test large matrix (triggers async yield)
        A = [[i * 50 + j for j in range(50)] for i in range(50)]  # 50x50 matrix
        result = await matrix_transpose(A)
        assert len(result) == 50
        assert len(result[0]) == 50

    async def test_matrix_det_2x2(self):
        from chuk_mcp_math.linear_algebra.matrices.operations import matrix_det_2x2

        A = [[1, 2], [3, 4]]
        result = await matrix_det_2x2(A)
        assert result == -2.0

    async def test_matrix_det_2x2_errors(self):
        from chuk_mcp_math.linear_algebra.matrices.operations import matrix_det_2x2

        # Test wrong dimensions
        with pytest.raises(ValueError, match="Expected 2×2 matrix"):
            await matrix_det_2x2([[1, 2, 3], [4, 5, 6]])

    async def test_matrix_det_3x3(self):
        from chuk_mcp_math.linear_algebra.matrices.operations import matrix_det_3x3

        A = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        result = await matrix_det_3x3(A)
        assert abs(result) < 1e-10  # Should be zero (singular matrix)

    async def test_matrix_det_3x3_errors(self):
        from chuk_mcp_math.linear_algebra.matrices.operations import matrix_det_3x3

        # Test wrong dimensions
        with pytest.raises(ValueError, match="Expected 3×3 matrix"):
            await matrix_det_3x3([[1, 2], [3, 4]])

    async def test_matrix_scalar_multiply(self):
        from chuk_mcp_math.linear_algebra.matrices.operations import matrix_scalar_multiply

        A = [[1, 2], [3, 4]]
        result = await matrix_scalar_multiply(A, 3)
        assert result == [[3.0, 6.0], [9.0, 12.0]]

    async def test_matrix_scalar_multiply_errors(self):
        from chuk_mcp_math.linear_algebra.matrices.operations import matrix_scalar_multiply

        # Test empty matrix
        with pytest.raises(ValueError, match="Matrix cannot be empty"):
            await matrix_scalar_multiply([], 2)

        with pytest.raises(ValueError, match="Matrix cannot be empty"):
            await matrix_scalar_multiply([[]], 2)

    async def test_matrix_scalar_multiply_large(self):
        from chuk_mcp_math.linear_algebra.matrices.operations import matrix_scalar_multiply

        # Test large matrix (triggers async yield)
        A = [[1] * 50 for _ in range(50)]  # 50x50 matrix
        result = await matrix_scalar_multiply(A, 2)
        assert result[0][0] == 2.0

    async def test_matrix_add(self):
        from chuk_mcp_math.linear_algebra.matrices.operations import matrix_add

        A = [[1, 2], [3, 4]]
        B = [[5, 6], [7, 8]]
        result = await matrix_add(A, B)
        assert result == [[6.0, 8.0], [10.0, 12.0]]

    async def test_matrix_add_errors(self):
        from chuk_mcp_math.linear_algebra.matrices.operations import matrix_add

        # Test empty matrices
        with pytest.raises(ValueError, match="First matrix cannot be empty"):
            await matrix_add([], [[1, 2]])

        with pytest.raises(ValueError, match="Second matrix cannot be empty"):
            await matrix_add([[1, 2]], [])

        # Test dimension mismatch
        with pytest.raises(ValueError, match="same dimensions"):
            await matrix_add([[1, 2], [3, 4]], [[1, 2, 3]])

    async def test_matrix_add_large(self):
        from chuk_mcp_math.linear_algebra.matrices.operations import matrix_add

        # Test large matrix (triggers async yield)
        A = [[1] * 50 for _ in range(50)]
        B = [[1] * 50 for _ in range(50)]
        result = await matrix_add(A, B)
        assert result[0][0] == 2.0

    async def test_matrix_subtract(self):
        from chuk_mcp_math.linear_algebra.matrices.operations import matrix_subtract

        A = [[5, 6], [7, 8]]
        B = [[1, 2], [3, 4]]
        result = await matrix_subtract(A, B)
        assert result == [[4.0, 4.0], [4.0, 4.0]]

    async def test_matrix_subtract_errors(self):
        from chuk_mcp_math.linear_algebra.matrices.operations import matrix_subtract

        # Test empty matrices
        with pytest.raises(ValueError, match="First matrix cannot be empty"):
            await matrix_subtract([], [[1, 2]])

        with pytest.raises(ValueError, match="Second matrix cannot be empty"):
            await matrix_subtract([[1, 2]], [])

        # Test dimension mismatch
        with pytest.raises(ValueError, match="same dimensions"):
            await matrix_subtract([[1, 2], [3, 4]], [[1, 2, 3]])

    async def test_matrix_subtract_large(self):
        from chuk_mcp_math.linear_algebra.matrices.operations import matrix_subtract

        # Test large matrix (triggers async yield)
        A = [[2] * 50 for _ in range(50)]
        B = [[1] * 50 for _ in range(50)]
        result = await matrix_subtract(A, B)
        assert result[0][0] == 1.0

    async def test_matrix_solve_2x2(self):
        from chuk_mcp_math.linear_algebra.matrices.solvers import matrix_solve_2x2

        A = [[2, 1], [1, 3]]
        b = [5, 6]
        result = await matrix_solve_2x2(A, b)
        # Solution: x = 1.8, y = 1.4
        assert abs(result[0] - 1.8) < 1e-10
        assert abs(result[1] - 1.4) < 1e-10

    async def test_matrix_solve_2x2_errors(self):
        from chuk_mcp_math.linear_algebra.matrices.solvers import matrix_solve_2x2

        # Test wrong matrix dimensions
        with pytest.raises(ValueError, match="Expected 2×2 matrix"):
            await matrix_solve_2x2([[1, 2, 3]], [1, 2])

        # Test wrong vector dimensions
        with pytest.raises(ValueError, match="Expected vector of length 2"):
            await matrix_solve_2x2([[1, 2], [3, 4]], [1, 2, 3])

        # Test singular matrix
        with pytest.raises(ValueError, match="singular"):
            await matrix_solve_2x2([[1, 2], [2, 4]], [1, 2])  # Rows are linearly dependent

    async def test_matrix_solve_3x3(self):
        from chuk_mcp_math.linear_algebra.matrices.solvers import matrix_solve_3x3

        A = [[2, 1, 1], [1, 3, 2], [1, 0, 0]]
        b = [4, 5, 6]
        result = await matrix_solve_3x3(A, b)
        # Verify dimensions
        assert len(result) == 3

    async def test_matrix_solve_3x3_errors(self):
        from chuk_mcp_math.linear_algebra.matrices.solvers import matrix_solve_3x3

        # Test wrong matrix dimensions
        with pytest.raises(ValueError, match="Expected 3×3 matrix"):
            await matrix_solve_3x3([[1, 2], [3, 4]], [1, 2, 3])

        # Test wrong vector dimensions
        with pytest.raises(ValueError, match="Expected vector of length 3"):
            await matrix_solve_3x3([[1, 2, 3], [4, 5, 6], [7, 8, 9]], [1, 2])

        # Test singular matrix
        with pytest.raises(ValueError, match="singular"):
            await matrix_solve_3x3([[1, 2, 3], [2, 4, 6], [3, 6, 9]], [1, 2, 3])

    async def test_gaussian_elimination(self):
        from chuk_mcp_math.linear_algebra.matrices.solvers import gaussian_elimination

        A = [[3, 2, -1], [2, -2, 4], [-1, 0.5, -1]]
        b = [1, -2, 0]
        result = await gaussian_elimination(A, b)
        # Verify solution
        assert abs(result[0] - 1.0) < 1e-6
        assert abs(result[1] + 2.0) < 1e-6
        assert abs(result[2] + 2.0) < 1e-6

    async def test_gaussian_elimination_errors(self):
        from chuk_mcp_math.linear_algebra.matrices.solvers import gaussian_elimination

        # Test empty matrix
        with pytest.raises(ValueError, match="Matrix cannot be empty"):
            await gaussian_elimination([], [])

        with pytest.raises(ValueError, match="Matrix cannot be empty"):
            await gaussian_elimination([[]], [])

        # Test non-square matrix
        with pytest.raises(ValueError, match="Matrix must be square"):
            await gaussian_elimination([[1, 2, 3], [4, 5, 6]], [1, 2])

        # Test dimension mismatch
        with pytest.raises(ValueError, match="doesn't match matrix size"):
            await gaussian_elimination([[1, 2], [3, 4]], [1, 2, 3])

        # Test singular matrix
        with pytest.raises(ValueError, match="singular"):
            await gaussian_elimination([[1, 2], [2, 4]], [1, 2])

    async def test_gaussian_elimination_large(self):
        from chuk_mcp_math.linear_algebra.matrices.solvers import gaussian_elimination

        # Test large system (triggers async yield)
        n = 15
        A = [[1 if i == j else 0.1 for j in range(n)] for i in range(n)]  # Diagonal dominant
        b = [1.0] * n
        result = await gaussian_elimination(A, b)
        assert len(result) == n


@pytest.mark.asyncio
class TestCalculusDerivatives:
    """Test numeric derivatives."""

    async def test_derivative_central(self):
        from chuk_mcp_math.calculus.derivatives import derivative_central

        # f(x) = x^2, f'(x) = 2x, f'(3) = 6
        f = lambda x: x**2
        result = await derivative_central(f, 3.0)
        assert abs(result - 6.0) < 1e-4

    async def test_derivative_forward(self):
        from chuk_mcp_math.calculus.derivatives import derivative_forward

        f = lambda x: x**2
        result = await derivative_forward(f, 3.0)
        assert abs(result - 6.0) < 1e-3  # Less accurate than central

    async def test_derivative_backward(self):
        from chuk_mcp_math.calculus.derivatives import derivative_backward

        f = lambda x: x**2
        result = await derivative_backward(f, 3.0)
        assert abs(result - 6.0) < 1e-3  # Less accurate than central


@pytest.mark.asyncio
class TestCalculusIntegration:
    """Test numeric integration."""

    async def test_integrate_trapezoid(self):
        from chuk_mcp_math.calculus.integration import integrate_trapezoid

        # ∫₀¹ x² dx = 1/3
        f = lambda x: x**2
        result = await integrate_trapezoid(f, 0.0, 1.0, 1000)
        assert abs(result - 1 / 3) < 1e-4

    async def test_integrate_trapezoid_large(self):
        from chuk_mcp_math.calculus.integration import integrate_trapezoid

        # Test with large n_steps (triggers async yield)
        f = lambda x: x**2
        result = await integrate_trapezoid(f, 0.0, 1.0, 1500)
        assert abs(result - 1 / 3) < 1e-4

        # Test with very large n_steps (triggers periodic yield)
        result2 = await integrate_trapezoid(f, 0.0, 1.0, 15000)
        assert abs(result2 - 1 / 3) < 1e-5

    async def test_integrate_trapezoid_errors(self):
        from chuk_mcp_math.calculus.integration import integrate_trapezoid

        f = lambda x: x**2

        # Test invalid n_steps
        with pytest.raises(ValueError, match="Number of steps must be >= 1"):
            await integrate_trapezoid(f, 0.0, 1.0, 0)

        # Test invalid limits
        with pytest.raises(ValueError, match="Lower limit.*must be <= upper limit"):
            await integrate_trapezoid(f, 1.0, 0.0, 100)

    async def test_integrate_simpson(self):
        from chuk_mcp_math.calculus.integration import integrate_simpson

        # ∫₀¹ x² dx = 1/3
        f = lambda x: x**2
        result = await integrate_simpson(f, 0.0, 1.0, 1000)
        assert abs(result - 1 / 3) < 1e-6  # Simpson is more accurate

    async def test_integrate_simpson_large(self):
        from chuk_mcp_math.calculus.integration import integrate_simpson

        # Test with large n_steps (triggers async yield)
        f = lambda x: x**2
        result = await integrate_simpson(f, 0.0, 1.0, 1500)
        assert abs(result - 1 / 3) < 1e-6

        # Test with very large n_steps (triggers periodic yield)
        result2 = await integrate_simpson(f, 0.0, 1.0, 15000)
        assert abs(result2 - 1 / 3) < 1e-7

    async def test_integrate_simpson_errors(self):
        from chuk_mcp_math.calculus.integration import integrate_simpson

        f = lambda x: x**2

        # Test n_steps < 2
        with pytest.raises(ValueError, match="Number of steps must be >= 2"):
            await integrate_simpson(f, 0.0, 1.0, 1)

        # Test odd n_steps
        with pytest.raises(ValueError, match="Number of steps must be even"):
            await integrate_simpson(f, 0.0, 1.0, 101)

        # Test invalid limits
        with pytest.raises(ValueError, match="Lower limit.*must be <= upper limit"):
            await integrate_simpson(f, 1.0, 0.0, 100)

    async def test_integrate_midpoint(self):
        from chuk_mcp_math.calculus.integration import integrate_midpoint

        # ∫₀¹ x² dx = 1/3
        f = lambda x: x**2
        result = await integrate_midpoint(f, 0.0, 1.0, 1000)
        assert abs(result - 1 / 3) < 1e-4

    async def test_integrate_midpoint_large(self):
        from chuk_mcp_math.calculus.integration import integrate_midpoint

        # Test with large n_steps (triggers async yield)
        f = lambda x: x**2
        result = await integrate_midpoint(f, 0.0, 1.0, 1500)
        assert abs(result - 1 / 3) < 1e-4

        # Test with very large n_steps (triggers periodic yield)
        result2 = await integrate_midpoint(f, 0.0, 1.0, 15000)
        assert abs(result2 - 1 / 3) < 1e-5

    async def test_integrate_midpoint_errors(self):
        from chuk_mcp_math.calculus.integration import integrate_midpoint

        f = lambda x: x**2

        # Test invalid n_steps
        with pytest.raises(ValueError, match="Number of steps must be >= 1"):
            await integrate_midpoint(f, 0.0, 1.0, 0)

        # Test invalid limits
        with pytest.raises(ValueError, match="Lower limit.*must be <= upper limit"):
            await integrate_midpoint(f, 1.0, 0.0, 100)


@pytest.mark.asyncio
class TestCalculusRootFinding:
    """Test root finding algorithms."""

    async def test_bisection(self):
        from chuk_mcp_math.calculus.root_finding import root_find_bisection

        # f(x) = x² - 4, root at x = 2
        f = lambda x: x**2 - 4
        result = await root_find_bisection(f, 0.0, 3.0)
        assert abs(result - 2.0) < 1e-6

    async def test_bisection_swap_bounds(self):
        from chuk_mcp_math.calculus.root_finding import root_find_bisection

        # Test with a > b (should swap automatically)
        f = lambda x: x**2 - 4
        result = await root_find_bisection(f, 3.0, 0.0)
        assert abs(result - 2.0) < 1e-6

    async def test_bisection_errors(self):
        from chuk_mcp_math.calculus.root_finding import root_find_bisection

        # Test same sign error
        f = lambda x: x**2 + 1  # Always positive
        with pytest.raises(ValueError, match="must have opposite signs"):
            await root_find_bisection(f, 0.0, 3.0)

        # Test max iterations exceeded
        f_slow = lambda x: x**2 - 4
        with pytest.raises(ValueError, match="failed to converge"):
            await root_find_bisection(f_slow, 0.0, 3.0, tol=1e-20, max_iter=2)

    async def test_newton(self):
        from chuk_mcp_math.calculus.root_finding import root_find_newton

        # f(x) = x² - 4, f'(x) = 2x, root at x = 2
        f = lambda x: x**2 - 4
        fp = lambda x: 2 * x
        result = await root_find_newton(f, fp, 1.0)
        assert abs(result - 2.0) < 1e-6

    async def test_newton_errors(self):
        from chuk_mcp_math.calculus.root_finding import root_find_newton

        # Test zero derivative error
        f = lambda x: x**2 - 4
        fp_zero = lambda x: 0.0  # Derivative always zero
        with pytest.raises(ValueError, match="Derivative too close to zero"):
            await root_find_newton(f, fp_zero, 1.0)

        # Test max iterations exceeded
        f_slow = lambda x: x**2 - 4
        fp = lambda x: 2 * x
        with pytest.raises(ValueError, match="failed to converge"):
            await root_find_newton(f_slow, fp, 1.0, tol=1e-20, max_iter=2)

    async def test_secant(self):
        from chuk_mcp_math.calculus.root_finding import root_find_secant

        # f(x) = x² - 4, root at x = 2
        f = lambda x: x**2 - 4
        result = await root_find_secant(f, 1.0, 3.0)
        assert abs(result - 2.0) < 1e-6

    async def test_secant_errors(self):
        from chuk_mcp_math.calculus.root_finding import root_find_secant

        # Test when function values are too close
        f_flat = lambda x: 1.0  # Constant function
        with pytest.raises(ValueError, match="Function values too close"):
            await root_find_secant(f_flat, 1.0, 2.0)

        # Test max iterations exceeded
        f_slow = lambda x: x**2 - 4
        with pytest.raises(ValueError, match="failed to converge"):
            await root_find_secant(f_slow, 1.0, 3.0, tol=1e-20, max_iter=2)


@pytest.mark.asyncio
class TestProbability:
    """Test probability distributions."""

    async def test_normal_pdf(self):
        from chuk_mcp_math.probability.distributions import normal_pdf

        # Standard normal at x=0 should be 1/sqrt(2π) ≈ 0.3989
        result = await normal_pdf(0.0, 0.0, 1.0)
        assert abs(result - 0.3989422804) < 1e-6

    async def test_normal_pdf_errors(self):
        from chuk_mcp_math.probability.distributions import normal_pdf

        # Test invalid std
        with pytest.raises(ValueError, match="Standard deviation must be positive"):
            await normal_pdf(0.0, 0.0, 0.0)

        with pytest.raises(ValueError, match="Standard deviation must be positive"):
            await normal_pdf(0.0, 0.0, -1.0)

    async def test_normal_cdf(self):
        from chuk_mcp_math.probability.distributions import normal_cdf

        # Standard normal CDF at x=0 should be 0.5
        result = await normal_cdf(0.0, 0.0, 1.0)
        assert abs(result - 0.5) < 1e-10

    async def test_normal_sample(self):
        from chuk_mcp_math.probability.distributions import normal_sample

        samples = await normal_sample(100, 0.0, 1.0, seed=42)
        assert len(samples) == 100
        # Check that samples are roughly normal (loose test)
        mean_sample = sum(samples) / len(samples)
        assert abs(mean_sample) < 0.5  # Should be close to 0

    async def test_normal_sample_large(self):
        from chuk_mcp_math.probability.distributions import normal_sample

        # Test large sample (triggers async yield)
        samples = await normal_sample(1500, 0.0, 1.0, seed=42)
        assert len(samples) == 1500

    async def test_normal_sample_errors(self):
        from chuk_mcp_math.probability.distributions import normal_sample

        # Test invalid n
        with pytest.raises(ValueError, match="Number of samples must be positive"):
            await normal_sample(0, 0.0, 1.0)

        # Test invalid std
        with pytest.raises(ValueError, match="Standard deviation must be positive"):
            await normal_sample(10, 0.0, 0.0)

    async def test_uniform_sample(self):
        from chuk_mcp_math.probability.distributions import uniform_sample

        samples = await uniform_sample(100, 0.0, 1.0, seed=42)
        assert len(samples) == 100
        assert all(0 <= s <= 1 for s in samples)

    async def test_uniform_sample_large(self):
        from chuk_mcp_math.probability.distributions import uniform_sample

        # Test large sample (triggers async yield)
        samples = await uniform_sample(1500, 0.0, 1.0, seed=42)
        assert len(samples) == 1500

    async def test_uniform_sample_errors(self):
        from chuk_mcp_math.probability.distributions import uniform_sample

        # Test invalid n
        with pytest.raises(ValueError, match="Number of samples must be positive"):
            await uniform_sample(0, 0.0, 1.0)

        # Test invalid range
        with pytest.raises(ValueError, match="Lower bound.*must be less than upper bound"):
            await uniform_sample(10, 1.0, 0.0)


@pytest.mark.asyncio
class TestStatistics:
    """Test statistics functions."""

    async def test_covariance(self):
        from chuk_mcp_math.statistics import covariance

        xs = [1, 2, 3, 4, 5]
        ys = [2, 4, 6, 8, 10]
        result = await covariance(xs, ys)
        assert abs(result - 5.0) < 1e-10  # Sample covariance

    async def test_correlation(self):
        from chuk_mcp_math.statistics import correlation

        xs = [1, 2, 3, 4, 5]
        ys = [2, 4, 6, 8, 10]
        result = await correlation(xs, ys)
        assert abs(result - 1.0) < 1e-10  # Perfect positive correlation

    async def test_linear_regression(self):
        from chuk_mcp_math.statistics import linear_regression

        xs = [1, 2, 3, 4, 5]
        ys = [2, 4, 6, 8, 10]
        result = await linear_regression(xs, ys)
        assert abs(result["slope"] - 2.0) < 1e-10
        assert abs(result["intercept"] - 0.0) < 1e-10
        assert abs(result["r_squared"] - 1.0) < 1e-10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
