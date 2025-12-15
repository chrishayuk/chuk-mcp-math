"""
Probability Distributions

Functions for common probability distributions including PDF, CDF, and sampling.
All functions are async-native with MCP decoration for AI model integration.
"""

import asyncio
import math
import random
from typing import List, Optional
from ..mcp_decorator import mcp_function


@mcp_function(
    description="Calculate the probability density function (PDF) of the normal distribution",
    namespace="probability",
    cache_strategy="memory",
    estimated_cpu_usage="low",
    examples=[
        {
            "input": {"x": 0.0, "mean": 0.0, "std": 1.0},
            "output": 0.3989,
            "description": "PDF of standard normal at x=0",
        }
    ],
)
async def normal_pdf(x: float, mean: float = 0.0, std: float = 1.0) -> float:
    """
    Calculate the probability density function of the normal distribution.

    Formula:
        f(x) = (1 / (σ√(2π))) × exp(-((x-μ)²) / (2σ²))

    Args:
        x: Point at which to evaluate PDF
        mean: Mean (μ) of the distribution (default: 0.0)
        std: Standard deviation (σ) of the distribution (default: 1.0)

    Returns:
        PDF value at x

    Raises:
        ValueError: If std <= 0

    Examples:
        >>> await normal_pdf(0.0, 0.0, 1.0)
        0.3989...  # 1/sqrt(2π)
        >>> await normal_pdf(1.0, 0.0, 1.0)
        0.2419...
    """
    if std <= 0:
        raise ValueError(f"Standard deviation must be positive, got {std}")

    await asyncio.sleep(0)

    coefficient = 1.0 / (std * math.sqrt(2 * math.pi))
    exponent = -0.5 * ((x - mean) / std) ** 2
    return float(coefficient * math.exp(exponent))


@mcp_function(
    description="Calculate the cumulative distribution function (CDF) of the normal distribution",
    namespace="probability",
    cache_strategy="memory",
    estimated_cpu_usage="low",
    examples=[
        {
            "input": {"x": 0.0, "mean": 0.0, "std": 1.0},
            "output": 0.5,
            "description": "CDF of standard normal at x=0 is 0.5",
        }
    ],
)
async def normal_cdf(x: float, mean: float = 0.0, std: float = 1.0) -> float:
    """
    Calculate the cumulative distribution function of the normal distribution.

    Gives P(X <= x) for a normally distributed random variable X.

    Uses the error function (erf) approximation.

    Args:
        x: Point at which to evaluate CDF
        mean: Mean (μ) of the distribution (default: 0.0)
        std: Standard deviation (σ) of the distribution (default: 1.0)

    Returns:
        CDF value at x (probability that X <= x)

    Raises:
        ValueError: If std <= 0

    Examples:
        >>> await normal_cdf(0.0, 0.0, 1.0)
        0.5
        >>> await normal_cdf(1.96, 0.0, 1.0)
        0.975  # ~97.5th percentile
    """
    if std <= 0:
        raise ValueError(f"Standard deviation must be positive, got {std}")

    await asyncio.sleep(0)

    # Standardize: z = (x - μ) / σ
    z = (x - mean) / std

    # CDF = 0.5 × (1 + erf(z / sqrt(2)))
    return float(0.5 * (1 + math.erf(z / math.sqrt(2))))


@mcp_function(
    description="Generate random samples from the normal distribution",
    namespace="probability",
    cache_strategy="none",  # Don't cache random samples
    estimated_cpu_usage="low",
    examples=[
        {
            "input": {"n": 5, "mean": 0.0, "std": 1.0},
            "output": [-0.123, 0.456, -0.789, 1.234, -0.567],
            "description": "Five random samples from standard normal",
        }
    ],
)
async def normal_sample(
    n: int, mean: float = 0.0, std: float = 1.0, seed: Optional[int] = None
) -> List[float]:
    """
    Generate random samples from the normal distribution.

    Uses the Box-Muller transform to generate normally distributed samples.

    Args:
        n: Number of samples to generate
        mean: Mean (μ) of the distribution (default: 0.0)
        std: Standard deviation (σ) of the distribution (default: 1.0)
        seed: Optional random seed for reproducibility

    Returns:
        List of n random samples

    Raises:
        ValueError: If n <= 0 or std <= 0

    Examples:
        >>> samples = await normal_sample(1000, 0.0, 1.0)
        >>> len(samples)
        1000
        >>> # Mean should be close to 0, std close to 1
    """
    if n <= 0:
        raise ValueError(f"Number of samples must be positive, got {n}")
    if std <= 0:
        raise ValueError(f"Standard deviation must be positive, got {std}")

    if seed is not None:
        random.seed(seed)

    # Yield for large samples
    if n > 1000:
        await asyncio.sleep(0)

    samples = []
    for i in range(n):
        # Box-Muller transform
        u1 = random.random()
        u2 = random.random()

        # Convert to standard normal
        z0 = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)

        # Scale and shift to desired mean and std
        sample = mean + std * z0
        samples.append(sample)

        # Yield periodically for very large samples
        if i % 10000 == 0:
            await asyncio.sleep(0)

    return samples


@mcp_function(
    description="Generate random samples from the uniform distribution",
    namespace="probability",
    cache_strategy="none",  # Don't cache random samples
    estimated_cpu_usage="low",
    examples=[
        {
            "input": {"n": 5, "a": 0.0, "b": 1.0},
            "output": [0.123, 0.456, 0.789, 0.234, 0.567],
            "description": "Five random samples from uniform(0, 1)",
        }
    ],
)
async def uniform_sample(
    n: int, a: float = 0.0, b: float = 1.0, seed: Optional[int] = None
) -> List[float]:
    """
    Generate random samples from the uniform distribution on [a, b].

    Args:
        n: Number of samples to generate
        a: Lower bound of the distribution (default: 0.0)
        b: Upper bound of the distribution (default: 1.0)
        seed: Optional random seed for reproducibility

    Returns:
        List of n random samples uniformly distributed on [a, b]

    Raises:
        ValueError: If n <= 0 or a >= b

    Examples:
        >>> samples = await uniform_sample(1000, 0.0, 1.0)
        >>> len(samples)
        1000
        >>> all(0 <= s <= 1 for s in samples)
        True
    """
    if n <= 0:
        raise ValueError(f"Number of samples must be positive, got {n}")
    if a >= b:
        raise ValueError(f"Lower bound a={a} must be less than upper bound b={b}")

    if seed is not None:
        random.seed(seed)

    # Yield for large samples
    if n > 1000:
        await asyncio.sleep(0)

    samples = []
    for i in range(n):
        # Generate uniform sample on [a, b]
        sample = a + (b - a) * random.random()
        samples.append(sample)

        # Yield periodically for very large samples
        if i % 10000 == 0:
            await asyncio.sleep(0)

    return samples


@mcp_function(
    description="Calculate stockout probability for inventory under lead time demand uncertainty. Returns mu_LT, sigma_LT, Z-score, and P(stockout).",
    namespace="probability",
    cache_strategy="memory",
    estimated_cpu_usage="low",
    examples=[
        {
            "input": {
                "inventory": 900,
                "daily_demand": 37,
                "lead_time_days": 18,
                "daily_demand_std": 15,
            },
            "output": {
                "mu_LT": 666,
                "sigma_LT": 63.64,
                "z_score": 3.68,
                "p_stockout": 0.00012,
                "service_level": 0.99988,
            },
            "description": "Stockout probability with 900 units, 37/day demand, 18 day lead time",
        }
    ],
)
async def stockout_probability(
    inventory: float,
    daily_demand: float,
    lead_time_days: float,
    daily_demand_std: float,
) -> dict:
    """
    Calculate the probability of stockout during lead time.

    This function computes all intermediate values for inventory risk analysis:
    - mu_LT = daily_demand * lead_time_days (mean demand during lead time)
    - sigma_LT = daily_demand_std * sqrt(lead_time_days) (std dev during lead time)
    - Z = (inventory - mu_LT) / sigma_LT (safety factor)
    - P(stockout) = 1 - Phi(Z) (probability demand exceeds inventory)

    Args:
        inventory: Current inventory level (I)
        daily_demand: Average daily demand rate (d)
        lead_time_days: Lead time in days (L)
        daily_demand_std: Standard deviation of daily demand (sigma_d)

    Returns:
        Dictionary with:
        - mu_LT: Mean lead time demand
        - sigma_LT: Std dev of lead time demand
        - z_score: Number of std devs inventory is above mean demand
        - p_stockout: Probability of stockout (demand > inventory)
        - service_level: 1 - p_stockout (probability of meeting demand)

    Raises:
        ValueError: If daily_demand_std <= 0 or lead_time_days <= 0

    Examples:
        >>> result = await stockout_probability(900, 37, 18, 15)
        >>> result['p_stockout']
        0.00012  # Very low stockout risk
    """
    if daily_demand_std <= 0:
        raise ValueError(f"Daily demand std must be positive, got {daily_demand_std}")
    if lead_time_days <= 0:
        raise ValueError(f"Lead time must be positive, got {lead_time_days}")

    await asyncio.sleep(0)

    # Calculate lead time demand distribution parameters
    mu_LT = daily_demand * lead_time_days
    sigma_LT = daily_demand_std * math.sqrt(lead_time_days)

    # Calculate Z-score (safety factor)
    z = (inventory - mu_LT) / sigma_LT

    # Calculate P(stockout) = P(demand > inventory) = 1 - Phi(Z)
    phi_z = 0.5 * (1 + math.erf(z / math.sqrt(2)))
    p_stockout = 1.0 - phi_z

    return {
        "mu_LT": float(mu_LT),
        "sigma_LT": float(sigma_LT),
        "z_score": float(z),
        "p_stockout": float(p_stockout),
        "service_level": float(phi_z),
    }
