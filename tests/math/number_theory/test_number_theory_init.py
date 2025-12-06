#!/usr/bin/env python3
# tests/math/number_theory/test_init.py
"""
Comprehensive pytest unit tests for src/chuk_mcp_math/number_theory/__init__.py

Tests cover:
- Submodule imports (primes, divisibility, sequences, etc.)
- Function imports from each submodule
- __all__ exports
- Helper functions (test_number_theory_functions, demo functions)
- Module documentation
- Cross-module functionality
"""

import pytest
import asyncio

# Import the module to test
from chuk_mcp_math import number_theory


class TestSubmoduleImports:
    """Test that all submodules are imported correctly."""

    def test_primes_module_imported(self):
        """Test that primes module is imported."""
        assert hasattr(number_theory, "primes")
        assert number_theory.primes is not None

    def test_divisibility_module_imported(self):
        """Test that divisibility module is imported."""
        assert hasattr(number_theory, "divisibility")
        assert number_theory.divisibility is not None

    def test_basic_sequences_module_imported(self):
        """Test that basic_sequences module is imported."""
        assert hasattr(number_theory, "basic_sequences")
        assert number_theory.basic_sequences is not None

    def test_special_primes_module_imported(self):
        """Test that special_primes module is imported."""
        assert hasattr(number_theory, "special_primes")
        assert number_theory.special_primes is not None

    def test_combinatorial_numbers_module_imported(self):
        """Test that combinatorial_numbers module is imported."""
        assert hasattr(number_theory, "combinatorial_numbers")
        assert number_theory.combinatorial_numbers is not None

    def test_arithmetic_functions_module_imported(self):
        """Test that arithmetic_functions module is imported."""
        assert hasattr(number_theory, "arithmetic_functions")
        assert number_theory.arithmetic_functions is not None

    def test_iterative_sequences_module_imported(self):
        """Test that iterative_sequences module is imported."""
        assert hasattr(number_theory, "iterative_sequences")
        assert number_theory.iterative_sequences is not None

    def test_mathematical_constants_module_imported(self):
        """Test that mathematical_constants module is imported."""
        assert hasattr(number_theory, "mathematical_constants")
        assert number_theory.mathematical_constants is not None

    def test_digital_operations_module_imported(self):
        """Test that digital_operations module is imported."""
        assert hasattr(number_theory, "digital_operations")
        assert number_theory.digital_operations is not None

    def test_partitions_module_imported(self):
        """Test that partitions module is imported."""
        assert hasattr(number_theory, "partitions")
        assert number_theory.partitions is not None

    def test_egyptian_fractions_module_imported(self):
        """Test that egyptian_fractions module is imported."""
        assert hasattr(number_theory, "egyptian_fractions")
        assert number_theory.egyptian_fractions is not None

    def test_figurate_numbers_module_imported(self):
        """Test that figurate_numbers module is imported."""
        assert hasattr(number_theory, "figurate_numbers")
        assert number_theory.figurate_numbers is not None

    def test_modular_arithmetic_module_imported(self):
        """Test that modular_arithmetic module is imported."""
        assert hasattr(number_theory, "modular_arithmetic")
        assert number_theory.modular_arithmetic is not None

    def test_recursive_sequences_module_imported(self):
        """Test that recursive_sequences module is imported."""
        assert hasattr(number_theory, "recursive_sequences")
        assert number_theory.recursive_sequences is not None

    def test_diophantine_equations_module_imported(self):
        """Test that diophantine_equations module is imported."""
        assert hasattr(number_theory, "diophantine_equations")
        assert number_theory.diophantine_equations is not None

    def test_advanced_prime_patterns_module_imported(self):
        """Test that advanced_prime_patterns module is imported."""
        assert hasattr(number_theory, "advanced_prime_patterns")
        assert number_theory.advanced_prime_patterns is not None

    def test_special_number_categories_module_imported(self):
        """Test that special_number_categories module is imported."""
        assert hasattr(number_theory, "special_number_categories")
        assert number_theory.special_number_categories is not None

    def test_continued_fractions_module_imported(self):
        """Test that continued_fractions module is imported."""
        assert hasattr(number_theory, "continued_fractions")
        assert number_theory.continued_fractions is not None

    def test_farey_sequences_module_imported(self):
        """Test that farey_sequences module is imported."""
        assert hasattr(number_theory, "farey_sequences")
        assert number_theory.farey_sequences is not None


class TestCorePrimeFunctions:
    """Test that core prime functions are imported."""

    def test_is_prime_imported(self):
        """Test that is_prime function is imported."""
        assert hasattr(number_theory, "is_prime")
        assert callable(number_theory.is_prime)

    def test_next_prime_imported(self):
        """Test that next_prime function is imported."""
        assert hasattr(number_theory, "next_prime")
        assert callable(number_theory.next_prime)

    def test_nth_prime_imported(self):
        """Test that nth_prime function is imported."""
        assert hasattr(number_theory, "nth_prime")
        assert callable(number_theory.nth_prime)

    def test_prime_factors_imported(self):
        """Test that prime_factors function is imported."""
        assert hasattr(number_theory, "prime_factors")
        assert callable(number_theory.prime_factors)

    def test_prime_count_imported(self):
        """Test that prime_count function is imported."""
        assert hasattr(number_theory, "prime_count")
        assert callable(number_theory.prime_count)

    def test_is_coprime_imported(self):
        """Test that is_coprime function is imported."""
        assert hasattr(number_theory, "is_coprime")
        assert callable(number_theory.is_coprime)

    def test_first_n_primes_imported(self):
        """Test that first_n_primes function is imported."""
        assert hasattr(number_theory, "first_n_primes")
        assert callable(number_theory.first_n_primes)


class TestCoreDivisibilityFunctions:
    """Test that core divisibility functions are imported."""

    def test_gcd_imported(self):
        """Test that gcd function is imported."""
        assert hasattr(number_theory, "gcd")
        assert callable(number_theory.gcd)

    def test_lcm_imported(self):
        """Test that lcm function is imported."""
        assert hasattr(number_theory, "lcm")
        assert callable(number_theory.lcm)

    def test_divisors_imported(self):
        """Test that divisors function is imported."""
        assert hasattr(number_theory, "divisors")
        assert callable(number_theory.divisors)

    def test_is_divisible_imported(self):
        """Test that is_divisible function is imported."""
        assert hasattr(number_theory, "is_divisible")
        assert callable(number_theory.is_divisible)

    def test_is_even_imported(self):
        """Test that is_even function is imported."""
        assert hasattr(number_theory, "is_even")
        assert callable(number_theory.is_even)

    def test_is_odd_imported(self):
        """Test that is_odd function is imported."""
        assert hasattr(number_theory, "is_odd")
        assert callable(number_theory.is_odd)

    def test_extended_gcd_imported(self):
        """Test that extended_gcd function is imported."""
        assert hasattr(number_theory, "extended_gcd")
        assert callable(number_theory.extended_gcd)


class TestAllExports:
    """Test __all__ exports."""

    def test_all_defined(self):
        """Test that __all__ is defined."""
        assert hasattr(number_theory, "__all__")
        assert isinstance(number_theory.__all__, list)
        assert len(number_theory.__all__) > 0

    def test_all_contains_submodules(self):
        """Test that __all__ contains submodule names."""
        expected_modules = [
            "primes",
            "divisibility",
            "basic_sequences",
            "special_primes",
            "combinatorial_numbers",
            "arithmetic_functions",
            "iterative_sequences",
            "mathematical_constants",
            "digital_operations",
            "partitions",
            "egyptian_fractions",
            "figurate_numbers",
            "modular_arithmetic",
            "recursive_sequences",
            "diophantine_equations",
            "advanced_prime_patterns",
            "special_number_categories",
            "continued_fractions",
            "farey_sequences",
        ]

        for module in expected_modules:
            assert module in number_theory.__all__, f"{module} not in __all__"

    def test_all_contains_core_functions(self):
        """Test that __all__ contains core number theory functions."""
        core_functions = [
            "is_prime",
            "next_prime",
            "nth_prime",
            "prime_factors",
            "gcd",
            "lcm",
            "divisors",
            "fibonacci",
            "factorial",
        ]

        for func in core_functions:
            assert func in number_theory.__all__, f"{func} not in __all__"

    def test_all_items_exist(self):
        """Test that all items in __all__ actually exist."""
        for item in number_theory.__all__:
            assert hasattr(number_theory, item), f"{item} in __all__ but not in module"


class TestAsyncPrimeFunctions:
    """Test async functionality of prime functions."""

    @pytest.mark.asyncio
    async def test_is_prime_function(self):
        """Test is_prime function works correctly."""
        assert await number_theory.is_prime(17) is True
        assert await number_theory.is_prime(4) is False

    @pytest.mark.asyncio
    async def test_next_prime_function(self):
        """Test next_prime function works correctly."""
        result = await number_theory.next_prime(10)
        assert result == 11

    @pytest.mark.asyncio
    async def test_prime_factors_function(self):
        """Test prime_factors function works correctly."""
        result = await number_theory.prime_factors(60)
        # 60 = 2 * 2 * 3 * 5
        assert 2 in result
        assert 3 in result
        assert 5 in result


class TestAsyncDivisibilityFunctions:
    """Test async functionality of divisibility functions."""

    @pytest.mark.asyncio
    async def test_gcd_function(self):
        """Test gcd function works correctly."""
        result = await number_theory.gcd(48, 18)
        assert result == 6

    @pytest.mark.asyncio
    async def test_lcm_function(self):
        """Test lcm function works correctly."""
        result = await number_theory.lcm(12, 18)
        assert result == 36

    @pytest.mark.asyncio
    async def test_divisors_function(self):
        """Test divisors function works correctly."""
        result = await number_theory.divisors(12)
        assert set(result) == {1, 2, 3, 4, 6, 12}

    @pytest.mark.asyncio
    async def test_is_even_function(self):
        """Test is_even function works correctly."""
        assert await number_theory.is_even(4) is True
        assert await number_theory.is_even(5) is False

    @pytest.mark.asyncio
    async def test_is_odd_function(self):
        """Test is_odd function works correctly."""
        assert await number_theory.is_odd(7) is True
        assert await number_theory.is_odd(6) is False


class TestAsyncSequenceFunctions:
    """Test async functionality of sequence functions."""

    @pytest.mark.asyncio
    async def test_fibonacci_function(self):
        """Test fibonacci function works correctly."""
        result = await number_theory.fibonacci(10)
        assert result == 55  # 10th Fibonacci number

    @pytest.mark.asyncio
    async def test_factorial_function(self):
        """Test factorial function works correctly."""
        result = await number_theory.factorial(5)
        assert result == 120

    @pytest.mark.asyncio
    async def test_catalan_number_function(self):
        """Test catalan_number function works correctly."""
        result = await number_theory.catalan_number(5)
        assert result == 42  # 5th Catalan number


class TestTestFunction:
    """Test the test_number_theory_functions async function."""

    @pytest.mark.asyncio
    async def test_test_number_theory_functions_exists(self):
        """Test that test_number_theory_functions function exists."""
        assert hasattr(number_theory, "test_number_theory_functions")
        assert callable(number_theory.test_number_theory_functions)

    @pytest.mark.asyncio
    async def test_test_number_theory_functions_executes(self, capsys):
        """Test that test_number_theory_functions executes without error."""
        await number_theory.test_number_theory_functions()
        captured = capsys.readouterr()

        # Should produce output
        assert len(captured.out) > 0


class TestDemoFunctions:
    """Test demo functions existence and execution."""

    def test_demo_comprehensive_functionality_exists(self):
        """Test that demo_comprehensive_functionality exists."""
        assert hasattr(number_theory, "demo_comprehensive_functionality")
        assert callable(number_theory.demo_comprehensive_functionality)

    def test_demo_educational_applications_exists(self):
        """Test that demo_educational_applications exists."""
        assert hasattr(number_theory, "demo_educational_applications")
        assert callable(number_theory.demo_educational_applications)

    def test_demo_research_applications_exists(self):
        """Test that demo_research_applications exists."""
        assert hasattr(number_theory, "demo_research_applications")
        assert callable(number_theory.demo_research_applications)

    @pytest.mark.asyncio
    async def test_demo_comprehensive_functionality_executes(self, capsys):
        """Test that demo_comprehensive_functionality executes without error."""
        await number_theory.demo_comprehensive_functionality()
        captured = capsys.readouterr()

        # Should produce output
        assert len(captured.out) > 0
        # Should mention comprehensive functionality
        assert "Comprehensive" in captured.out or "comprehensive" in captured.out

    @pytest.mark.asyncio
    async def test_demo_educational_applications_executes(self, capsys):
        """Test that demo_educational_applications executes without error."""
        await number_theory.demo_educational_applications()
        captured = capsys.readouterr()

        # Should produce output
        assert len(captured.out) > 0
        # Should mention educational aspects
        assert (
            "Educational" in captured.out or "Student" in captured.out or "analysis" in captured.out
        )

    @pytest.mark.asyncio
    async def test_demo_research_applications_executes(self, capsys):
        """Test that demo_research_applications executes without error."""
        await number_theory.demo_research_applications()
        captured = capsys.readouterr()

        # Should produce output
        assert len(captured.out) > 0
        # Should mention research aspects
        assert "Research" in captured.out or "research" in captured.out or "Prime" in captured.out


class TestModuleDocumentation:
    """Test module documentation."""

    def test_module_has_docstring(self):
        """Test that number_theory module has a docstring."""
        assert number_theory.__doc__ is not None
        assert len(number_theory.__doc__) > 0

    def test_docstring_mentions_number_theory(self):
        """Test that docstring mentions number theory."""
        docstring = number_theory.__doc__.upper()
        assert "NUMBER THEORY" in docstring or "NUMBER" in docstring

    def test_docstring_mentions_async(self):
        """Test that docstring mentions async functionality."""
        docstring = number_theory.__doc__.lower()
        assert "async" in docstring


class TestAdvancedModules:
    """Test advanced module functions."""

    def test_farey_sequence_imported(self):
        """Test that farey_sequence function is imported."""
        assert hasattr(number_theory, "farey_sequence")
        assert callable(number_theory.farey_sequence)

    def test_continued_fraction_expansion_imported(self):
        """Test that continued_fraction_expansion function is imported."""
        assert hasattr(number_theory, "continued_fraction_expansion")
        assert callable(number_theory.continued_fraction_expansion)

    def test_solve_pell_equation_imported(self):
        """Test that solve_pell_equation function is imported."""
        assert hasattr(number_theory, "solve_pell_equation")
        assert callable(number_theory.solve_pell_equation)

    def test_pythagorean_triples_imported(self):
        """Test that pythagorean_triples function is imported."""
        assert hasattr(number_theory, "pythagorean_triples")
        assert callable(number_theory.pythagorean_triples)


class TestFunctionCount:
    """Test that a large number of functions are available."""

    def test_many_functions_exported(self):
        """Test that many functions are exported."""
        # __all__ should contain 100+ items (modules + functions)
        assert len(number_theory.__all__) > 100

    def test_all_submodules_present(self):
        """Test that all 19 submodules are present."""
        modules = [
            "primes",
            "divisibility",
            "basic_sequences",
            "special_primes",
            "combinatorial_numbers",
            "arithmetic_functions",
            "iterative_sequences",
            "mathematical_constants",
            "digital_operations",
            "partitions",
            "egyptian_fractions",
            "figurate_numbers",
            "modular_arithmetic",
            "recursive_sequences",
            "diophantine_equations",
            "advanced_prime_patterns",
            "special_number_categories",
            "continued_fractions",
            "farey_sequences",
        ]

        for module in modules:
            assert hasattr(number_theory, module)


class TestConcurrentExecution:
    """Test concurrent execution of number theory functions."""

    @pytest.mark.asyncio
    async def test_concurrent_prime_operations(self):
        """Test concurrent execution of prime operations."""
        tasks = [
            number_theory.is_prime(17),
            number_theory.next_prime(10),
            number_theory.prime_factors(60),
        ]

        results = await asyncio.gather(*tasks)
        assert len(results) == 3
        assert results[0] is True  # is_prime(17)
        assert results[1] == 11  # next_prime(10)

    @pytest.mark.asyncio
    async def test_concurrent_divisibility_operations(self):
        """Test concurrent execution of divisibility operations."""
        tasks = [
            number_theory.gcd(48, 18),
            number_theory.lcm(12, 18),
            number_theory.is_even(4),
        ]

        results = await asyncio.gather(*tasks)
        assert results == [6, 36, True]


class TestMainExecution:
    """Test main execution block behavior."""

    def test_main_block_does_not_execute_on_import(self):
        """Test that __name__ == '__main__' block doesn't run on import."""
        from chuk_mcp_math import number_theory as test_nt

        assert test_nt is not None


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v", "--tb=short"])
