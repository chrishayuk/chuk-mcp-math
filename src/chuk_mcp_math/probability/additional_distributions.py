"""
Additional Probability Distributions

Exponential and binomial distributions with PDF, CDF, and sampling functions.
All functions are async-native with MCP decoration for AI model integration.
"""

import asyncio
import math
import random
from typing import List, Optional
from ..mcp_decorator import mcp_function


@mcp_function(
    description="Calculate the probability density function (PDF) of the exponential distribution",
    namespace="probability",
    cache_strategy="memory",
    estimated_cpu_usage="low",
    examples=[
        {
            "input": {"x": 1.0, "rate": 1.0},
            "output": 0.3679,
            "description": "PDF of exponential(1) at x=1",
        }
    ],
)
async def exponential_pdf(x: float, rate: float = 1.0) -> float:
    """
    Calculate the probability density function of the exponential distribution.

    The exponential distribution models the time between events in a Poisson process.

    Formula:
        f(x) = λ × exp(-λx) for x ≥ 0

    Args:
        x: Point at which to evaluate PDF (must be ≥ 0)
        rate: Rate parameter (λ) (must be > 0, default: 1.0)

    Returns:
        PDF value at x

    Raises:
        ValueError: If x < 0 or rate <= 0

    Examples:
        >>> await exponential_pdf(1.0, 1.0)
        0.3678...  # e^(-1)
        >>> await exponential_pdf(0.0, 2.0)
        2.0  # At x=0, PDF = λ
    """
    if x < 0:
        raise ValueError(f"x must be non-negative, got {x}")
    if rate <= 0:
        raise ValueError(f"Rate must be positive, got {rate}")

    await asyncio.sleep(0)

    return float(rate * math.exp(-rate * x))


@mcp_function(
    description="Calculate the cumulative distribution function (CDF) of the exponential distribution",
    namespace="probability",
    cache_strategy="memory",
    estimated_cpu_usage="low",
)
async def exponential_cdf(x: float, rate: float = 1.0) -> float:
    """
    Calculate the cumulative distribution function of the exponential distribution.

    Formula:
        F(x) = 1 - exp(-λx) for x ≥ 0

    Args:
        x: Point at which to evaluate CDF (must be ≥ 0)
        rate: Rate parameter (λ) (must be > 0, default: 1.0)

    Returns:
        CDF value at x (probability that X ≤ x)

    Raises:
        ValueError: If x < 0 or rate <= 0

    Examples:
        >>> await exponential_cdf(1.0, 1.0)
        0.6321...  # 1 - e^(-1)
    """
    if x < 0:
        raise ValueError(f"x must be non-negative, got {x}")
    if rate <= 0:
        raise ValueError(f"Rate must be positive, got {rate}")

    await asyncio.sleep(0)

    return float(1 - math.exp(-rate * x))


@mcp_function(
    description="Generate random samples from the exponential distribution",
    namespace="probability",
    cache_strategy="none",
    estimated_cpu_usage="low",
)
async def exponential_sample(n: int, rate: float = 1.0, seed: Optional[int] = None) -> List[float]:
    """
    Generate random samples from the exponential distribution.

    Uses inverse transform sampling: X = -ln(U) / λ where U ~ Uniform(0,1)

    Args:
        n: Number of samples to generate
        rate: Rate parameter (λ) (must be > 0, default: 1.0)
        seed: Optional random seed for reproducibility

    Returns:
        List of n random samples

    Raises:
        ValueError: If n <= 0 or rate <= 0

    Examples:
        >>> samples = await exponential_sample(1000, 1.0, seed=42)
        >>> len(samples)
        1000
    """
    if n <= 0:
        raise ValueError(f"Number of samples must be positive, got {n}")
    if rate <= 0:
        raise ValueError(f"Rate must be positive, got {rate}")

    if seed is not None:
        random.seed(seed)

    # Yield for large samples
    if n > 1000:
        await asyncio.sleep(0)

    samples = []
    for i in range(n):
        u = random.random()
        # Inverse transform: X = -ln(U) / λ
        sample = -math.log(u) / rate
        samples.append(sample)

        # Yield periodically for very large samples
        if i % 10000 == 0:
            await asyncio.sleep(0)

    return samples


@mcp_function(
    description="Calculate the probability mass function (PMF) of the binomial distribution",
    namespace="probability",
    cache_strategy="memory",
    estimated_cpu_usage="low",
    examples=[
        {
            "input": {"k": 3, "n": 5, "p": 0.5},
            "output": 0.3125,
            "description": "P(X=3) for Binomial(5, 0.5)",
        }
    ],
)
async def binomial_pmf(k: int, n: int, p: float) -> float:
    """
    Calculate the probability mass function of the binomial distribution.

    Models the number of successes in n independent trials with probability p.

    Formula:
        P(X = k) = C(n,k) × p^k × (1-p)^(n-k)

    Args:
        k: Number of successes (0 ≤ k ≤ n)
        n: Number of trials (n ≥ 0)
        p: Probability of success (0 ≤ p ≤ 1)

    Returns:
        Probability of exactly k successes

    Raises:
        ValueError: If parameters are out of valid range

    Examples:
        >>> await binomial_pmf(3, 5, 0.5)
        0.3125  # Probability of 3 heads in 5 coin flips
    """
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")
    if not 0 <= k <= n:
        raise ValueError(f"k must be between 0 and n, got k={k}, n={n}")
    if not 0 <= p <= 1:
        raise ValueError(f"p must be between 0 and 1, got {p}")

    await asyncio.sleep(0)

    # Calculate binomial coefficient C(n, k)
    def binomial_coeff(n: int, k: int) -> int:
        if k > n - k:
            k = n - k
        result = 1
        for i in range(k):
            result = result * (n - i) // (i + 1)
        return result

    coeff = binomial_coeff(n, k)
    prob = coeff * (p**k) * ((1 - p) ** (n - k))

    return float(prob)


@mcp_function(
    description="Calculate the cumulative distribution function (CDF) of the binomial distribution",
    namespace="probability",
    cache_strategy="memory",
    estimated_cpu_usage="medium",
)
async def binomial_cdf(k: int, n: int, p: float) -> float:
    """
    Calculate the cumulative distribution function of the binomial distribution.

    Formula:
        F(k) = P(X ≤ k) = Σ(i=0 to k) C(n,i) × p^i × (1-p)^(n-i)

    Args:
        k: Number of successes (0 ≤ k ≤ n)
        n: Number of trials (n ≥ 0)
        p: Probability of success (0 ≤ p ≤ 1)

    Returns:
        Probability of k or fewer successes

    Raises:
        ValueError: If parameters are out of valid range

    Examples:
        >>> await binomial_cdf(3, 5, 0.5)
        0.8125  # P(X ≤ 3) for Binomial(5, 0.5)
    """
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")
    if not 0 <= k <= n:
        raise ValueError(f"k must be between 0 and n, got k={k}, n={n}")
    if not 0 <= p <= 1:
        raise ValueError(f"p must be between 0 and 1, got {p}")

    await asyncio.sleep(0)

    # Sum PMF values from 0 to k
    cdf_value = 0.0
    for i in range(k + 1):
        cdf_value += await binomial_pmf(i, n, p)

    return float(cdf_value)


@mcp_function(
    description="Generate random samples from the binomial distribution",
    namespace="probability",
    cache_strategy="none",
    estimated_cpu_usage="medium",
)
async def binomial_sample(
    num_samples: int, n: int, p: float, seed: Optional[int] = None
) -> List[int]:
    """
    Generate random samples from the binomial distribution.

    Each sample represents the number of successes in n trials.

    Args:
        num_samples: Number of samples to generate
        n: Number of trials per sample
        p: Probability of success (0 ≤ p ≤ 1)
        seed: Optional random seed for reproducibility

    Returns:
        List of random samples (each is count of successes)

    Raises:
        ValueError: If parameters are out of valid range

    Examples:
        >>> samples = await binomial_sample(100, 10, 0.5, seed=42)
        >>> len(samples)
        100
        >>> all(0 <= s <= 10 for s in samples)
        True
    """
    if num_samples <= 0:
        raise ValueError(f"Number of samples must be positive, got {num_samples}")
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")
    if not 0 <= p <= 1:
        raise ValueError(f"p must be between 0 and 1, got {p}")

    if seed is not None:
        random.seed(seed)

    # Yield for large samples
    if num_samples > 1000:
        await asyncio.sleep(0)

    samples = []
    for i in range(num_samples):
        # Count successes in n trials
        successes = sum(1 for _ in range(n) if random.random() < p)
        samples.append(successes)

        # Yield periodically
        if i % 1000 == 0:
            await asyncio.sleep(0)

    return samples
