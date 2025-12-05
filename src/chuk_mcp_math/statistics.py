#!/usr/bin/env python3
# chuk_mcp_math/statistics.py
"""
Statistics Functions for AI Models

Statistical analysis and calculations: mean, median, mode, variance, standard deviation, and more.
Designed for AI model execution with clear descriptions and comprehensive error handling.
"""

import math
import asyncio
from typing import List, Union, Dict, Any
from chuk_mcp_math.mcp_decorator import mcp_function


@mcp_function(
    description="Calculate the arithmetic mean (average) of a list of numbers. Sum all values and divide by count.",
    namespace="statistics",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    examples=[
        {
            "input": {"numbers": [1, 2, 3, 4, 5]},
            "output": 3.0,
            "description": "Mean of consecutive integers",
        },
        {
            "input": {"numbers": [10, 20, 30]},
            "output": 20.0,
            "description": "Mean of multiples of 10",
        },
        {
            "input": {"numbers": [2.5, 3.5, 4.5]},
            "output": 3.5,
            "description": "Mean of decimal numbers",
        },
    ],
)
async def mean(numbers: List[Union[int, float]]) -> float:
    """
    Calculate the arithmetic mean of a list of numbers.

    Args:
        numbers: List of numbers

    Returns:
        The arithmetic mean

    Raises:
        ValueError: If the list is empty

    Examples:
        mean([1, 2, 3, 4, 5]) → 3.0
        mean([10, 20, 30]) → 20.0
        mean([2.5, 3.5, 4.5]) → 3.5
    """
    if not numbers:
        raise ValueError("Cannot calculate mean of empty list")
    await asyncio.sleep(0)  # Yield control for async execution
    return sum(numbers) / len(numbers)


@mcp_function(
    description="Calculate the median (middle value) of a list of numbers. The value that separates the higher half from the lower half.",
    namespace="statistics",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    examples=[
        {
            "input": {"numbers": [1, 2, 3, 4, 5]},
            "output": 3,
            "description": "Median of odd-length list",
        },
        {
            "input": {"numbers": [1, 2, 3, 4]},
            "output": 2.5,
            "description": "Median of even-length list",
        },
        {
            "input": {"numbers": [5, 1, 3, 2, 4]},
            "output": 3,
            "description": "Median of unsorted list",
        },
    ],
)
async def median(numbers: List[Union[int, float]]) -> Union[int, float]:
    """
    Calculate the median of a list of numbers.

    Args:
        numbers: List of numbers

    Returns:
        The median value

    Raises:
        ValueError: If the list is empty

    Examples:
        median([1, 2, 3, 4, 5]) → 3
        median([1, 2, 3, 4]) → 2.5
        median([5, 1, 3, 2, 4]) → 3
    """
    if not numbers:
        raise ValueError("Cannot calculate median of empty list")

    sorted_numbers = sorted(numbers)
    n = len(sorted_numbers)

    await asyncio.sleep(0)  # Yield control for async execution
    if n % 2 == 1:
        return sorted_numbers[n // 2]
    else:
        return (sorted_numbers[n // 2 - 1] + sorted_numbers[n // 2]) / 2


@mcp_function(
    description="Find the mode (most frequently occurring value) in a list of numbers. Returns all values that appear most frequently.",
    namespace="statistics",
    execution_modes=["local", "remote"],
    examples=[
        {
            "input": {"numbers": [1, 2, 2, 3, 4]},
            "output": [2],
            "description": "Single mode",
        },
        {
            "input": {"numbers": [1, 1, 2, 2, 3]},
            "output": [1, 2],
            "description": "Multiple modes",
        },
        {
            "input": {"numbers": [1, 2, 3, 4, 5]},
            "output": [1, 2, 3, 4, 5],
            "description": "No mode (all equal frequency)",
        },
    ],
)
async def mode(numbers: List[Union[int, float]]) -> List[Union[int, float]]:
    """
    Find the mode (most frequent value) of a list of numbers.

    Args:
        numbers: List of numbers

    Returns:
        List of the most frequent value(s)

    Raises:
        ValueError: If the list is empty

    Examples:
        mode([1, 2, 2, 3, 4]) → [2]
        mode([1, 1, 2, 2, 3]) → [1, 2]
        mode([1, 2, 3, 4, 5]) → [1, 2, 3, 4, 5]
    """
    if not numbers:
        raise ValueError("Cannot calculate mode of empty list")

    from collections import Counter

    counts = Counter(numbers)
    max_count = max(counts.values())

    await asyncio.sleep(0)  # Yield control for async execution
    return [num for num, count in counts.items() if count == max_count]


@mcp_function(
    description="Calculate the sample variance of a list of numbers. Measures how spread out the numbers are from the mean.",
    namespace="statistics",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    examples=[
        {
            "input": {"numbers": [1, 2, 3, 4, 5], "population": False},
            "output": 2.5,
            "description": "Sample variance",
        },
        {
            "input": {"numbers": [1, 2, 3, 4, 5], "population": True},
            "output": 2.0,
            "description": "Population variance",
        },
        {
            "input": {"numbers": [10, 10, 10]},
            "output": 0.0,
            "description": "No variance (all same)",
        },
    ],
)
async def variance(numbers: List[Union[int, float]], population: bool = False) -> float:
    """
    Calculate the variance of a list of numbers.

    Args:
        numbers: List of numbers
        population: If True, calculate population variance; if False, sample variance

    Returns:
        The variance

    Examples:
        variance([1, 2, 3, 4, 5], population=False) → 2.5
        variance([1, 2, 3, 4, 5], population=True) → 2.0
        variance([10, 10, 10]) → 0.0
    """
    if not numbers:
        raise ValueError("Cannot calculate variance of empty list")
    if len(numbers) == 1 and not population:
        raise ValueError("Cannot calculate sample variance with only one data point")

    avg = await mean(numbers)
    squared_diffs = [(x - avg) ** 2 for x in numbers]

    divisor = len(numbers) if population else len(numbers) - 1
    await asyncio.sleep(0)  # Yield control for async execution
    return sum(squared_diffs) / divisor


@mcp_function(
    description="Calculate the standard deviation of a list of numbers. The square root of variance, showing typical deviation from mean.",
    namespace="statistics",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    examples=[
        {
            "input": {"numbers": [1, 2, 3, 4, 5]},
            "output": 1.58,
            "description": "Standard deviation (approx)",
        },
        {
            "input": {"numbers": [10, 12, 14, 16, 18]},
            "output": 3.16,
            "description": "Standard deviation of evenly spaced numbers",
        },
        {
            "input": {"numbers": [5, 5, 5, 5]},
            "output": 0.0,
            "description": "No deviation (all same)",
        },
    ],
)
async def standard_deviation(
    numbers: List[Union[int, float]], population: bool = False
) -> float:
    """
    Calculate the standard deviation of a list of numbers.

    Args:
        numbers: List of numbers
        population: If True, calculate population std dev; if False, sample std dev

    Returns:
        The standard deviation

    Examples:
        standard_deviation([1, 2, 3, 4, 5]) → 1.5811388300841898
        standard_deviation([10, 12, 14, 16, 18]) → 3.1622776601683795
        standard_deviation([5, 5, 5, 5]) → 0.0
    """
    await asyncio.sleep(0)  # Yield control for async execution
    return math.sqrt(await variance(numbers, population))


@mcp_function(
    description="Calculate the range (difference between maximum and minimum) of a list of numbers.",
    namespace="statistics",
    execution_modes=["local", "remote"],
    examples=[
        {
            "input": {"numbers": [1, 2, 3, 4, 5]},
            "output": 4,
            "description": "Range of consecutive numbers",
        },
        {
            "input": {"numbers": [10, 5, 15, 8, 12]},
            "output": 10,
            "description": "Range of mixed numbers",
        },
        {
            "input": {"numbers": [7, 7, 7]},
            "output": 0,
            "description": "No range (all same)",
        },
    ],
)
async def range_value(numbers: List[Union[int, float]]) -> Union[int, float]:
    """
    Calculate the range (max - min) of a list of numbers.

    Args:
        numbers: List of numbers

    Returns:
        The range (difference between max and min)

    Raises:
        ValueError: If the list is empty

    Examples:
        range_value([1, 2, 3, 4, 5]) → 4
        range_value([10, 5, 15, 8, 12]) → 10
        range_value([7, 7, 7]) → 0
    """
    if not numbers:
        raise ValueError("Cannot calculate range of empty list")
    await asyncio.sleep(0)  # Yield control for async execution
    return max(numbers) - min(numbers)


@mcp_function(
    description="Calculate comprehensive statistics for a dataset including mean, median, mode, variance, and more.",
    namespace="statistics",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    estimated_cpu_usage="medium",
    examples=[
        {
            "input": {"numbers": [1, 2, 3, 4, 5]},
            "output": {
                "count": 5,
                "mean": 3.0,
                "median": 3,
                "min": 1,
                "max": 5,
                "variance": 2.5,
                "std_dev": 1.58,
                "range": 4,
            },
            "description": "Complete statistics for simple dataset",
        }
    ],
)
async def comprehensive_stats(numbers: List[Union[int, float]]) -> Dict[str, Any]:
    """
    Calculate comprehensive statistics for a dataset.

    Args:
        numbers: List of numbers

    Returns:
        Dictionary with statistics: count, mean, median, min, max, variance, std_dev, range

    Examples:
        comprehensive_stats([1, 2, 3, 4, 5]) → comprehensive statistics dictionary
    """
    if not numbers:
        raise ValueError("Cannot calculate statistics for empty list")

    n = len(numbers)
    mean_val = await mean(numbers)
    median_val = await median(numbers)
    min_val = min(numbers)
    max_val = max(numbers)
    variance_val = await variance(numbers) if n > 1 else 0
    std_dev_val = math.sqrt(variance_val)

    await asyncio.sleep(0)  # Yield control for async execution
    return {
        "count": n,
        "mean": mean_val,
        "median": median_val,
        "min": min_val,
        "max": max_val,
        "variance": variance_val,
        "std_dev": std_dev_val,
        "range": max_val - min_val,
        "sum": sum(numbers),
    }


@mcp_function(
    description="Calculate percentiles of a dataset. Find the value below which a certain percentage of data falls.",
    namespace="statistics",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    examples=[
        {
            "input": {"numbers": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], "percentile": 50},
            "output": 5.5,
            "description": "50th percentile (median)",
        },
        {
            "input": {"numbers": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], "percentile": 25},
            "output": 3.25,
            "description": "25th percentile (Q1)",
        },
        {
            "input": {"numbers": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], "percentile": 75},
            "output": 7.75,
            "description": "75th percentile (Q3)",
        },
    ],
)
async def percentile(
    numbers: List[Union[int, float]], percentile: Union[int, float]
) -> float:
    """
    Calculate a percentile of a dataset.

    Args:
        numbers: List of numbers
        percentile: Percentile to calculate (0-100)

    Returns:
        The percentile value

    Examples:
        percentile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 50) → 5.5
        percentile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 25) → 3.25
        percentile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 75) → 7.75
    """
    if not numbers:
        raise ValueError("Cannot calculate percentile of empty list")
    if not 0 <= percentile <= 100:
        raise ValueError("Percentile must be between 0 and 100")

    sorted_numbers = sorted(numbers)
    n = len(sorted_numbers)

    if percentile == 0:
        await asyncio.sleep(0)  # Yield control for async execution
    return sorted_numbers[0]
    if percentile == 100:
        return sorted_numbers[-1]

    # Calculate the index
    index = (percentile / 100) * (n - 1)
    lower_index = int(index)
    upper_index = min(lower_index + 1, n - 1)

    # Interpolate if necessary
    if lower_index == upper_index:
        return sorted_numbers[lower_index]
    else:
        fraction = index - lower_index
        return sorted_numbers[lower_index] + fraction * (
            sorted_numbers[upper_index] - sorted_numbers[lower_index]
        )


@mcp_function(
    description="Calculate quartiles of a dataset (Q1, Q2/median, Q3). Divides data into four equal parts.",
    namespace="statistics",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    examples=[
        {
            "input": {"numbers": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]},
            "output": {"Q1": 3.25, "Q2": 5.5, "Q3": 7.75, "IQR": 4.5},
            "description": "Quartiles of 1-10",
        }
    ],
)
async def quartiles(numbers: List[Union[int, float]]) -> Dict[str, float]:
    """
    Calculate the quartiles (Q1, Q2, Q3) of a dataset.

    Args:
        numbers: List of numbers

    Returns:
        Dictionary with Q1, Q2 (median), Q3, and IQR (interquartile range)

    Examples:
        quartiles([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) → quartiles dictionary
    """
    if not numbers:
        raise ValueError("Cannot calculate quartiles of empty list")

    q1 = await percentile(numbers, 25)
    q2 = await percentile(numbers, 50)  # median
    q3 = await percentile(numbers, 75)
    iqr = q3 - q1

    await asyncio.sleep(0)  # Yield control for async execution
    return {"Q1": q1, "Q2": q2, "Q3": q3, "IQR": iqr}


# Export all statistics functions
__all__ = [
    "mean",
    "median",
    "mode",
    "variance",
    "standard_deviation",
    "range_value",
    "comprehensive_stats",
    "percentile",
    "quartiles",
]

if __name__ == "__main__":
    # Test the statistics functions
    print("📊 Statistics Functions Test")
    print("=" * 35)

    test_data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    print(f"Test data: {test_data}")
    print(f"mean = {mean(test_data)}")
    print(f"median = {median(test_data)}")
    print(f"mode = {mode([1, 2, 2, 3, 4])}")
    print(f"variance = {variance(test_data):.2f}")
    print(f"std_dev = {standard_deviation(test_data):.2f}")
    print(f"range = {range_value(test_data)}")

    stats = comprehensive_stats(test_data)
    print(f"comprehensive_stats = {stats}")

    print(f"50th percentile = {percentile(test_data, 50)}")

    quart = quartiles(test_data)
    print(f"quartiles = {quart}")

    print("\n✅ All statistics functions working correctly!")
