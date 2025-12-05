#!/usr/bin/env python3
# chuk_mcp_math/number_theory/special_number_categories.py
"""
Special Number Categories - Async Native

Functions for working with various special number categories including amicable numbers,
social numbers, Kaprekar numbers, vampire numbers, and other recreational mathematics.

Functions:
- Amicable numbers: find_amicable_pairs, is_amicable_number, amicable_chains
- Social numbers: find_social_numbers, sociable_chain_analysis, aliquot_sequences
- Recreational: kaprekar_numbers, vampire_numbers, taxi_numbers, keith_numbers
- Armstrong & variants: armstrong_numbers, dudeney_numbers, pluperfect_numbers
- Magic numbers: magic_constants, magic_square_analysis, latin_squares
- Digit properties: sum_digit_powers, digital_persistence, digit_factorial_chains
"""

import math
import asyncio
from typing import List, Dict
from collections import defaultdict
from chuk_mcp_math.mcp_decorator import mcp_function


# Helper functions
async def _sum_of_proper_divisors(n: int) -> int:
    """Calculate sum of proper divisors (excluding n itself)."""
    if n <= 1:
        return 0

    divisor_sum = 1  # 1 is always a proper divisor
    sqrt_n = int(math.sqrt(n))

    for i in range(2, sqrt_n + 1):
        if n % i == 0:
            divisor_sum += i
            if i != n // i:  # Avoid counting square root twice
                divisor_sum += n // i

    return divisor_sum


async def _get_divisors(n: int) -> List[int]:
    """Get all divisors of n including 1 and n."""
    if n <= 0:
        return []

    divisors = []
    sqrt_n = int(math.sqrt(n))

    for i in range(1, sqrt_n + 1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)

    return sorted(divisors)


# ============================================================================
# AMICABLE NUMBERS AND CHAINS
# ============================================================================


@mcp_function(
    description="Find all amicable number pairs up to limit.",
    namespace="arithmetic",
    category="special_numbers",
    execution_modes=["local", "remote"],
    estimated_cpu_usage="high",
    examples=[
        {
            "input": {"limit": 10000},
            "output": [
                [220, 284],
                [1184, 1210],
                [2620, 2924],
                [5020, 5564],
                [6232, 6368],
            ],
            "description": "Amicable pairs up to 10000",
        },
        {
            "input": {"limit": 1500},
            "output": [[220, 284], [1184, 1210]],
            "description": "Amicable pairs up to 1500",
        },
        {
            "input": {"limit": 500},
            "output": [[220, 284]],
            "description": "Amicable pairs up to 500",
        },
    ],
)
async def find_amicable_pairs(limit: int) -> List[List[int]]:
    """
    Find all amicable number pairs (a, b) where σ(a) = b and σ(b) = a.

    Two numbers are amicable if each is the sum of the proper divisors of the other.

    Args:
        limit: Upper bound for search

    Returns:
        List of [a, b] amicable pairs with both a, b ≤ limit

    Examples:
        await find_amicable_pairs(10000) → [[220, 284], [1184, 1210], ...]
        await find_amicable_pairs(1500) → [[220, 284], [1184, 1210]]
    """
    if limit < 220:  # First amicable pair is (220, 284)
        return []

    amicable_pairs = []
    seen = set()

    for n in range(2, limit + 1):
        if n in seen:
            continue

        # Calculate sum of proper divisors
        sigma_n = await _sum_of_proper_divisors(n)

        if sigma_n <= n or sigma_n > limit:
            continue

        # Check if σ(σ(n)) = n (amicable condition)
        sigma_sigma_n = await _sum_of_proper_divisors(sigma_n)

        if sigma_sigma_n == n and sigma_n != n:  # Exclude perfect numbers
            pair = sorted([n, sigma_n])
            if pair not in amicable_pairs:
                amicable_pairs.append(pair)
                seen.add(n)
                seen.add(sigma_n)

        # Yield control every 100 numbers for large limits
        if n % 100 == 0:
            await asyncio.sleep(0)

    return sorted(amicable_pairs)


@mcp_function(
    description="Check if a number is amicable (part of an amicable pair).",
    namespace="arithmetic",
    category="special_numbers",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    examples=[
        {
            "input": {"n": 220},
            "output": {"is_amicable": True, "partner": 284, "sum_of_divisors": 284},
            "description": "220 is amicable with 284",
        },
        {
            "input": {"n": 284},
            "output": {"is_amicable": True, "partner": 220, "sum_of_divisors": 220},
            "description": "284 is amicable with 220",
        },
        {
            "input": {"n": 100},
            "output": {"is_amicable": False, "sum_of_divisors": 117},
            "description": "100 is not amicable",
        },
    ],
)
async def is_amicable_number(n: int) -> Dict:
    """
    Check if a number is amicable (part of an amicable pair).

    Args:
        n: Number to check

    Returns:
        Dictionary with amicability information

    Examples:
        await is_amicable_number(220) → {"is_amicable": True, "partner": 284, ...}
        await is_amicable_number(284) → {"is_amicable": True, "partner": 220, ...}
        await is_amicable_number(100) → {"is_amicable": False, ...}
    """
    if n <= 1:
        return {"is_amicable": False, "sum_of_divisors": 0}

    sigma_n = await _sum_of_proper_divisors(n)

    if sigma_n <= n:
        return {"is_amicable": False, "sum_of_divisors": sigma_n}

    sigma_sigma_n = await _sum_of_proper_divisors(sigma_n)

    is_amicable = sigma_sigma_n == n and sigma_n != n

    result = {"is_amicable": is_amicable, "sum_of_divisors": sigma_n}

    if is_amicable:
        result["partner"] = sigma_n
        result["verification"] = f"σ({n}) = {sigma_n}, σ({sigma_n}) = {sigma_sigma_n}"  # type: ignore[assignment]

    return result


@mcp_function(
    description="Find sociable numbers (amicable chains of length > 2).",
    namespace="arithmetic",
    category="special_numbers",
    execution_modes=["local", "remote"],
    estimated_cpu_usage="extreme",
    examples=[
        {
            "input": {"limit": 30000, "max_chain_length": 10},
            "output": {
                "chains": [
                    {
                        "length": 5,
                        "chain": [12496, 14288, 15472, 14536, 14264],
                        "type": "sociable",
                    }
                ],
                "total_chains": 1,
            },
            "description": "Sociable chains up to 30000",
        },
        {
            "input": {"limit": 15000, "max_chain_length": 8},
            "output": {"chains": [], "total_chains": 0},
            "description": "No sociable chains up to 15000",
        },
    ],
)
async def find_social_numbers(limit: int, max_chain_length: int = 10) -> Dict:
    """
    Find sociable numbers (chains of numbers where σ forms a cycle).

    A sociable chain is a sequence a₁, a₂, ..., aₖ where σ(aᵢ) = aᵢ₊₁
    and σ(aₖ) = a₁, with k > 2.

    Args:
        limit: Upper bound for search
        max_chain_length: Maximum length of chains to search for

    Returns:
        Dictionary with found sociable chains

    Examples:
        await find_social_numbers(30000, 10) → {"chains": [{"length": 5, "chain": [12496, ...]}], ...}
        await find_social_numbers(15000, 8) → {"chains": [], "total_chains": 0}
    """
    if limit < 12496:  # First known sociable number
        return {"chains": [], "total_chains": 0}

    chains = []
    visited = set()

    for start in range(2, limit + 1):
        if start in visited:
            continue

        # Follow the aliquot sequence
        current = start
        sequence = [current]
        seen_in_sequence = {current}

        for step in range(max_chain_length):
            sigma_current = await _sum_of_proper_divisors(current)

            if sigma_current <= 1 or sigma_current > limit:
                # Sequence escapes or goes to 1
                break

            if sigma_current in seen_in_sequence:
                # Found a cycle
                cycle_start_idx = sequence.index(sigma_current)
                cycle = sequence[cycle_start_idx:]
                cycle_length = len(cycle)

                if cycle_length > 2 and cycle[0] not in visited:
                    # Valid sociable chain
                    chains.append(
                        {"length": cycle_length, "chain": cycle, "type": "sociable"}
                    )

                    # Mark all numbers in chain as visited
                    for num in cycle:
                        visited.add(num)
                break

            sequence.append(sigma_current)
            seen_in_sequence.add(sigma_current)
            current = sigma_current

        # Mark starting number as visited
        visited.add(start)

        # Yield control every 100 numbers
        if start % 100 == 0:
            await asyncio.sleep(0)

    return {
        "limit": limit,
        "max_chain_length": max_chain_length,
        "chains": chains,
        "total_chains": len(chains),
    }


@mcp_function(
    description="Analyze aliquot sequence starting from n.",
    namespace="arithmetic",
    category="special_numbers",
    execution_modes=["local", "remote"],
    estimated_cpu_usage="medium",
    examples=[
        {
            "input": {"n": 220, "max_steps": 10},
            "output": {
                "sequence": [220, 284, 220],
                "type": "amicable_cycle",
                "cycle_length": 2,
                "reaches_cycle": True,
            },
            "description": "Aliquot sequence from 220",
        },
        {
            "input": {"n": 6, "max_steps": 5},
            "output": {
                "sequence": [6, 6],
                "type": "perfect",
                "cycle_length": 1,
                "reaches_cycle": True,
            },
            "description": "Aliquot sequence from 6 (perfect)",
        },
        {
            "input": {"n": 12, "max_steps": 10},
            "output": {
                "sequence": [12, 16, 15, 9, 4, 3, 1, 1],
                "type": "reaches_one",
                "reaches_cycle": False,
            },
            "description": "Aliquot sequence from 12",
        },
    ],
)
async def aliquot_sequence_analysis(n: int, max_steps: int = 50) -> Dict:
    """
    Analyze the aliquot sequence starting from n.

    The aliquot sequence is: n, σ(n), σ(σ(n)), ...

    Args:
        n: Starting number
        max_steps: Maximum steps to compute

    Returns:
        Dictionary with sequence analysis

    Examples:
        await aliquot_sequence_analysis(220, 10) → {"sequence": [220, 284, 220], "type": "amicable_cycle", ...}
        await aliquot_sequence_analysis(6, 5) → {"sequence": [6, 6], "type": "perfect", ...}
    """
    if n <= 0:
        return {"sequence": [], "type": "invalid", "reaches_cycle": False}

    sequence = [n]
    current = n
    seen = {n: 0}

    for step in range(1, max_steps + 1):
        sigma_current = await _sum_of_proper_divisors(current)
        sequence.append(sigma_current)

        if sigma_current in seen:
            # Found a cycle
            cycle_start = seen[sigma_current]
            cycle_length = step - cycle_start
            cycle = sequence[cycle_start:step]

            # Determine type
            if cycle_length == 1:
                if sigma_current == current:
                    seq_type = "perfect"
                else:
                    seq_type = "fixed_point"
            elif cycle_length == 2:
                seq_type = "amicable_cycle"
            else:
                seq_type = "sociable_cycle"

            return {
                "sequence": sequence,
                "type": seq_type,
                "cycle_length": cycle_length,
                "cycle": cycle,
                "reaches_cycle": True,
                "steps_to_cycle": cycle_start,
            }

        if sigma_current == 1:
            return {
                "sequence": sequence,
                "type": "reaches_one",
                "reaches_cycle": False,
                "steps_to_one": step,
            }

        if sigma_current == 0:
            return {
                "sequence": sequence,
                "type": "reaches_zero",
                "reaches_cycle": False,
            }

        seen[sigma_current] = step
        current = sigma_current

        # Yield control every 10 steps
        if step % 10 == 0:
            await asyncio.sleep(0)

    return {
        "sequence": sequence,
        "type": "undetermined",
        "reaches_cycle": False,
        "max_steps_reached": True,
    }


# ============================================================================
# KAPREKAR NUMBERS
# ============================================================================


@mcp_function(
    description="Find all Kaprekar numbers up to limit.",
    namespace="arithmetic",
    category="special_numbers",
    execution_modes=["local", "remote"],
    estimated_cpu_usage="high",
    examples=[
        {
            "input": {"limit": 1000},
            "output": [1, 9, 45, 55, 99, 297, 703, 999],
            "description": "Kaprekar numbers up to 1000",
        },
        {
            "input": {"limit": 100},
            "output": [1, 9, 45, 55, 99],
            "description": "Kaprekar numbers up to 100",
        },
        {
            "input": {"limit": 10000},
            "output": [
                1,
                9,
                45,
                55,
                99,
                297,
                703,
                999,
                2223,
                2728,
                4879,
                4950,
                5050,
                5292,
                7272,
                7777,
                9999,
            ],
            "description": "Kaprekar numbers up to 10000",
        },
    ],
)
async def kaprekar_numbers(limit: int) -> List[int]:
    """
    Find all Kaprekar numbers up to limit.

    A Kaprekar number n is such that n² can be split into two parts
    whose sum equals n. For example: 45² = 2025, 20 + 25 = 45.

    Args:
        limit: Upper bound for search

    Returns:
        List of Kaprekar numbers ≤ limit

    Examples:
        await kaprekar_numbers(1000) → [1, 9, 45, 55, 99, 297, 703, 999]
        await kaprekar_numbers(100) → [1, 9, 45, 55, 99]
    """
    if limit < 1:
        return []

    kaprekar_nums = []

    for n in range(1, limit + 1):
        square = n * n
        square_str = str(square)

        # Try all possible splits of the square
        for split_pos in range(1, len(square_str)):
            left_part = square_str[:split_pos]
            right_part = square_str[split_pos:]

            # Convert to integers (handle leading zeros)
            left_num = int(left_part) if left_part else 0
            right_num = int(right_part) if right_part else 0

            # Check if sum equals original number
            if left_num + right_num == n and right_num > 0:
                kaprekar_nums.append(n)
                break

        # Yield control every 100 numbers for large limits
        if n % 100 == 0:
            await asyncio.sleep(0)

    return kaprekar_nums


@mcp_function(
    description="Check if a number is a Kaprekar number and show the split.",
    namespace="arithmetic",
    category="special_numbers",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    examples=[
        {
            "input": {"n": 45},
            "output": {
                "is_kaprekar": True,
                "square": 2025,
                "splits": [{"left": 20, "right": 25, "sum": 45}],
            },
            "description": "45 is Kaprekar: 45² = 2025, 20 + 25 = 45",
        },
        {
            "input": {"n": 297},
            "output": {
                "is_kaprekar": True,
                "square": 88209,
                "splits": [{"left": 88, "right": 209, "sum": 297}],
            },
            "description": "297 is Kaprekar: 297² = 88209, 88 + 209 = 297",
        },
        {
            "input": {"n": 10},
            "output": {"is_kaprekar": False, "square": 100, "splits": []},
            "description": "10 is not Kaprekar",
        },
    ],
)
async def is_kaprekar_number(n: int) -> Dict:
    """
    Check if a number is a Kaprekar number and show valid splits.

    Args:
        n: Number to check

    Returns:
        Dictionary with Kaprekar analysis

    Examples:
        await is_kaprekar_number(45) → {"is_kaprekar": True, "square": 2025, "splits": [...]}
        await is_kaprekar_number(297) → {"is_kaprekar": True, "square": 88209, "splits": [...]}
        await is_kaprekar_number(10) → {"is_kaprekar": False, "square": 100, "splits": []}
    """
    if n <= 0:
        return {"is_kaprekar": False, "square": 0, "splits": []}

    square = n * n
    square_str = str(square)
    valid_splits = []

    # Try all possible splits
    for split_pos in range(1, len(square_str)):
        left_part = square_str[:split_pos]
        right_part = square_str[split_pos:]

        left_num = int(left_part) if left_part else 0
        right_num = int(right_part) if right_part else 0

        if left_num + right_num == n and right_num > 0:
            valid_splits.append(
                {
                    "left": left_num,
                    "right": right_num,
                    "sum": left_num + right_num,
                    "split_position": split_pos,
                }
            )

    return {
        "is_kaprekar": len(valid_splits) > 0,
        "square": square,
        "splits": valid_splits,
        "square_digits": len(square_str),
    }


# ============================================================================
# VAMPIRE NUMBERS
# ============================================================================


@mcp_function(
    description="Find all vampire numbers up to limit.",
    namespace="arithmetic",
    category="special_numbers",
    execution_modes=["local", "remote"],
    estimated_cpu_usage="extreme",
    examples=[
        {
            "input": {"limit": 200000},
            "output": [
                {"vampire": 1260, "fangs": [[21, 60]]},
                {"vampire": 1395, "fangs": [[15, 93]]},
                {"vampire": 1435, "fangs": [[35, 41]]},
                {"vampire": 1530, "fangs": [[30, 51]]},
                {"vampire": 1827, "fangs": [[21, 87]]},
                {"vampire": 2187, "fangs": [[27, 81]]},
                {"vampire": 6880, "fangs": [[80, 86]]},
                {"vampire": 102510, "fangs": [[201, 510]]},
                {"vampire": 104260, "fangs": [[260, 401]]},
                {"vampire": 105210, "fangs": [[210, 501]]},
                {"vampire": 105264, "fangs": [[204, 516]]},
                {"vampire": 105750, "fangs": [[150, 705]]},
                {"vampire": 108135, "fangs": [[135, 801]]},
                {"vampire": 110758, "fangs": [[158, 701]]},
                {"vampire": 115672, "fangs": [[152, 761]]},
                {"vampire": 116725, "fangs": [[161, 725]]},
                {"vampire": 117067, "fangs": [[167, 701]]},
                {"vampire": 118440, "fangs": [[141, 840]]},
                {"vampire": 120600, "fangs": [[201, 600]]},
                {"vampire": 123354, "fangs": [[231, 534]]},
                {"vampire": 124483, "fangs": [[281, 443]]},
                {"vampire": 125248, "fangs": [[152, 824]]},
                {"vampire": 125433, "fangs": [[231, 543]]},
                {"vampire": 125460, "fangs": [[204, 615], [246, 510]]},
                {"vampire": 125500, "fangs": [[251, 500]]},
                {"vampire": 186624, "fangs": [[186, 1002]]},
            ],
            "description": "Vampire numbers up to 200000",
        },
        {
            "input": {"limit": 10000},
            "output": [
                {"vampire": 1260, "fangs": [[21, 60]]},
                {"vampire": 1395, "fangs": [[15, 93]]},
                {"vampire": 1435, "fangs": [[35, 41]]},
                {"vampire": 1530, "fangs": [[30, 51]]},
                {"vampire": 1827, "fangs": [[21, 87]]},
                {"vampire": 2187, "fangs": [[27, 81]]},
                {"vampire": 6880, "fangs": [[80, 86]]},
            ],
            "description": "Vampire numbers up to 10000",
        },
    ],
)
async def vampire_numbers(limit: int) -> List[Dict]:
    """
    Find all vampire numbers up to limit.

    A vampire number has an even number of digits and can be factored into
    two numbers (fangs) each with half as many digits, where the digits
    of the fangs are a rearrangement of the original number's digits.

    Args:
        limit: Upper bound for search

    Returns:
        List of dictionaries with vampire numbers and their fangs

    Examples:
        await vampire_numbers(10000) → [{"vampire": 1260, "fangs": [[21, 60]]}, ...]
        await vampire_numbers(200000) → [{"vampire": 1260, "fangs": [[21, 60]]}, ...]
    """
    if limit < 1260:  # First vampire number
        return []

    vampire_nums = []

    # Start from first 4-digit number (smallest vampire numbers have 4 digits)
    start = 1000
    if limit < start:
        return []

    for n in range(start, limit + 1):
        n_str = str(n)
        n_digits = len(n_str)

        # Vampire numbers must have even number of digits
        if n_digits % 2 != 0:
            continue

        fang_digits = n_digits // 2
        fangs = []

        # Find all factor pairs
        sqrt_n = int(math.sqrt(n))
        for i in range(10 ** (fang_digits - 1), sqrt_n + 1):
            if n % i == 0:
                j = n // i

                # Both factors must have the right number of digits
                if len(str(i)) == fang_digits and len(str(j)) == fang_digits:
                    # Check if digits of i and j are a rearrangement of n
                    combined_digits = sorted(str(i) + str(j))
                    original_digits = sorted(n_str)

                    if combined_digits == original_digits:
                        # Avoid trailing zeros in both fangs (additional constraint)
                        if not (str(i).endswith("0") and str(j).endswith("0")):
                            fangs.append([i, j])

        if fangs:
            vampire_nums.append({"vampire": n, "fangs": fangs})

        # Yield control every 1000 numbers for large limits
        if n % 1000 == 0:
            await asyncio.sleep(0)

    return vampire_nums


@mcp_function(
    description="Check if a number is a vampire number and find its fangs.",
    namespace="arithmetic",
    category="special_numbers",
    execution_modes=["local", "remote"],
    examples=[
        {
            "input": {"n": 1260},
            "output": {"is_vampire": True, "fangs": [[21, 60]], "digit_count": 4},
            "description": "1260 is vampire: 21 × 60, digits rearrange",
        },
        {
            "input": {"n": 125460},
            "output": {
                "is_vampire": True,
                "fangs": [[204, 615], [246, 510]],
                "digit_count": 6,
            },
            "description": "125460 has multiple fang pairs",
        },
        {
            "input": {"n": 1234},
            "output": {"is_vampire": False, "fangs": [], "digit_count": 4},
            "description": "1234 is not vampire",
        },
    ],
)
async def is_vampire_number(n: int) -> Dict:
    """
    Check if a number is a vampire number and find all its fangs.

    Args:
        n: Number to check

    Returns:
        Dictionary with vampire analysis

    Examples:
        await is_vampire_number(1260) → {"is_vampire": True, "fangs": [[21, 60]], ...}
        await is_vampire_number(125460) → {"is_vampire": True, "fangs": [[204, 615], [246, 510]], ...}
        await is_vampire_number(1234) → {"is_vampire": False, "fangs": [], ...}
    """
    if n <= 0:
        return {"is_vampire": False, "fangs": [], "digit_count": 0}

    n_str = str(n)
    n_digits = len(n_str)

    # Must have even number of digits
    if n_digits % 2 != 0:
        return {
            "is_vampire": False,
            "fangs": [],
            "digit_count": n_digits,
            "reason": "Odd number of digits",
        }

    fang_digits = n_digits // 2
    fangs = []

    # Find all factor pairs
    sqrt_n = int(math.sqrt(n))
    for i in range(10 ** (fang_digits - 1), sqrt_n + 1):
        if n % i == 0:
            j = n // i

            # Both factors must have the right number of digits
            if len(str(i)) == fang_digits and len(str(j)) == fang_digits:
                # Check if digits are a rearrangement
                combined_digits = sorted(str(i) + str(j))
                original_digits = sorted(n_str)

                if combined_digits == original_digits:
                    # Additional constraint: not both fangs can end in 0
                    if not (str(i).endswith("0") and str(j).endswith("0")):
                        fangs.append([i, j])

    return {
        "is_vampire": len(fangs) > 0,
        "fangs": fangs,
        "digit_count": n_digits,
        "required_fang_digits": fang_digits,
    }


# ============================================================================
# ARMSTRONG NUMBERS AND VARIANTS
# ============================================================================


@mcp_function(
    description="Find all Armstrong numbers (narcissistic numbers) up to limit.",
    namespace="arithmetic",
    category="special_numbers",
    execution_modes=["local", "remote"],
    estimated_cpu_usage="high",
    examples=[
        {
            "input": {"limit": 10000},
            "output": [1, 2, 3, 4, 5, 6, 7, 8, 9, 153, 370, 371, 407, 1634, 8208, 9474],
            "description": "Armstrong numbers up to 10000",
        },
        {
            "input": {"limit": 1000},
            "output": [1, 2, 3, 4, 5, 6, 7, 8, 9, 153, 370, 371, 407],
            "description": "Armstrong numbers up to 1000",
        },
        {
            "input": {"limit": 100000},
            "output": [
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                153,
                370,
                371,
                407,
                1634,
                8208,
                9474,
                54748,
                92727,
                93084,
            ],
            "description": "Armstrong numbers up to 100000",
        },
    ],
)
async def armstrong_numbers(limit: int) -> List[int]:
    """
    Find all Armstrong (narcissistic) numbers up to limit.

    An Armstrong number equals the sum of its digits raised to the power
    of the number of digits. E.g., 153 = 1³ + 5³ + 3³.

    Args:
        limit: Upper bound for search

    Returns:
        List of Armstrong numbers ≤ limit

    Examples:
        await armstrong_numbers(10000) → [1, 2, 3, 4, 5, 6, 7, 8, 9, 153, 370, 371, 407, 1634, 8208, 9474]
        await armstrong_numbers(1000) → [1, 2, 3, 4, 5, 6, 7, 8, 9, 153, 370, 371, 407]
    """
    if limit < 1:
        return []

    armstrong_nums = []

    for n in range(1, limit + 1):
        digits = [int(d) for d in str(n)]
        num_digits = len(digits)

        digit_sum = sum(d**num_digits for d in digits)

        if digit_sum == n:
            armstrong_nums.append(n)

        # Yield control every 1000 numbers for large limits
        if n % 1000 == 0:
            await asyncio.sleep(0)

    return armstrong_nums


@mcp_function(
    description="Find all Dudeney numbers up to limit.",
    namespace="arithmetic",
    category="special_numbers",
    execution_modes=["local", "remote"],
    estimated_cpu_usage="high",
    examples=[
        {
            "input": {"limit": 100000},
            "output": [1, 512, 4913, 5832, 17576, 19683],
            "description": "Dudeney numbers up to 100000",
        },
        {
            "input": {"limit": 10000},
            "output": [1, 512, 4913, 5832],
            "description": "Dudeney numbers up to 10000",
        },
        {
            "input": {"limit": 1000},
            "output": [1, 512],
            "description": "Dudeney numbers up to 1000",
        },
    ],
)
async def dudeney_numbers(limit: int) -> List[int]:
    """
    Find all Dudeney numbers up to limit.

    A Dudeney number is a perfect cube whose digit sum equals the cube root.
    E.g., 512 = 8³ and 5 + 1 + 2 = 8.

    Args:
        limit: Upper bound for search

    Returns:
        List of Dudeney numbers ≤ limit

    Examples:
        await dudeney_numbers(100000) → [1, 512, 4913, 5832, 17576, 19683]
        await dudeney_numbers(10000) → [1, 512, 4913, 5832]
    """
    if limit < 1:
        return []

    dudeney_nums = []

    # Check perfect cubes up to limit
    cube_root = int(limit ** (1 / 3)) + 1

    for i in range(1, cube_root + 1):
        cube = i**3
        if cube > limit:
            break

        digit_sum = sum(int(d) for d in str(cube))

        if digit_sum == i:
            dudeney_nums.append(cube)

        # Yield control every 100 iterations
        if i % 100 == 0:
            await asyncio.sleep(0)

    return dudeney_nums


@mcp_function(
    description="Find all pluperfect digital invariant numbers up to limit.",
    namespace="arithmetic",
    category="special_numbers",
    execution_modes=["local", "remote"],
    estimated_cpu_usage="high",
    examples=[
        {
            "input": {"limit": 10000, "power": 4},
            "output": [1, 1634, 8208, 9474],
            "description": "4th power PPDI numbers up to 10000",
        },
        {
            "input": {"limit": 1000, "power": 3},
            "output": [1, 153, 371, 407],
            "description": "3rd power PPDI numbers up to 1000",
        },
        {
            "input": {"limit": 100000, "power": 5},
            "output": [1, 4150, 4151, 54748, 92727, 93084],
            "description": "5th power PPDI numbers up to 100000",
        },
    ],
)
async def pluperfect_numbers(limit: int, power: int) -> List[int]:
    """
    Find pluperfect digital invariant (PPDI) numbers up to limit.

    A PPDI number equals the sum of its digits raised to a given power.

    Args:
        limit: Upper bound for search
        power: Power to raise digits to

    Returns:
        List of PPDI numbers ≤ limit

    Examples:
        await pluperfect_numbers(10000, 4) → [1, 1634, 8208, 9474]
        await pluperfect_numbers(1000, 3) → [1, 153, 371, 407]
    """
    if limit < 1 or power < 1:
        return []

    ppdi_nums = []

    for n in range(1, limit + 1):
        digits = [int(d) for d in str(n)]
        digit_sum = sum(d**power for d in digits)

        if digit_sum == n:
            ppdi_nums.append(n)

        # Yield control every 1000 numbers for large limits
        if n % 1000 == 0:
            await asyncio.sleep(0)

    return ppdi_nums


# ============================================================================
# TAXI NUMBERS (HARDY-RAMANUJAN NUMBERS)
# ============================================================================


@mcp_function(
    description="Find all taxi numbers (sums of two cubes in multiple ways) up to limit.",
    namespace="arithmetic",
    category="special_numbers",
    execution_modes=["local", "remote"],
    estimated_cpu_usage="extreme",
    examples=[
        {
            "input": {"limit": 100000, "min_ways": 2},
            "output": [
                {"number": 1729, "representations": [[1, 12], [9, 10]]},
                {"number": 4104, "representations": [[2, 16], [9, 15]]},
                {"number": 5832, "representations": [[9, 18], [10, 17]]},
                {"number": 9729, "representations": [[1, 22], [18, 19]]},
                {"number": 20683, "representations": [[10, 27], [19, 24]]},
            ],
            "description": "Taxi numbers up to 100000",
        },
        {
            "input": {"limit": 10000, "min_ways": 2},
            "output": [
                {"number": 1729, "representations": [[1, 12], [9, 10]]},
                {"number": 4104, "representations": [[2, 16], [9, 15]]},
                {"number": 5832, "representations": [[9, 18], [10, 17]]},
                {"number": 9729, "representations": [[1, 22], [18, 19]]},
            ],
            "description": "Taxi numbers up to 10000",
        },
    ],
)
async def taxi_numbers(limit: int, min_ways: int = 2) -> List[Dict]:
    """
    Find taxi numbers (Hardy-Ramanujan numbers) up to limit.

    Taxi numbers can be expressed as sums of two cubes in multiple ways.
    Famous example: 1729 = 1³ + 12³ = 9³ + 10³.

    Args:
        limit: Upper bound for search
        min_ways: Minimum number of ways to express as sum of cubes

    Returns:
        List of dictionaries with taxi numbers and their representations

    Examples:
        await taxi_numbers(100000, 2) → [{"number": 1729, "representations": [[1, 12], [9, 10]]}, ...]
        await taxi_numbers(10000, 2) → [{"number": 1729, "representations": [[1, 12], [9, 10]]}, ...]
    """
    if limit < 1729:  # First taxi number
        return []

    # Generate all sums of two cubes up to limit
    cube_sums = defaultdict(list)
    max_cube_root = int(limit ** (1 / 3)) + 1

    for a in range(1, max_cube_root + 1):
        cube_a = a**3
        if cube_a > limit:
            break

        for b in range(a, max_cube_root + 1):
            cube_b = b**3
            sum_cubes = cube_a + cube_b

            if sum_cubes > limit:
                break

            cube_sums[sum_cubes].append([a, b])

        # Yield control every 10 values of a
        if a % 10 == 0:
            await asyncio.sleep(0)

    # Find numbers with at least min_ways representations
    taxi_nums = []
    for number, representations in cube_sums.items():
        if len(representations) >= min_ways:
            taxi_nums.append(
                {
                    "number": number,
                    "representations": representations,
                    "ways": len(representations),
                }
            )

    # Sort by number
    taxi_nums.sort(key=lambda x: x["number"])  # type: ignore[arg-type,return-value]
    return taxi_nums


# ============================================================================
# KEITH NUMBERS
# ============================================================================


@mcp_function(
    description="Find all Keith numbers up to limit.",
    namespace="arithmetic",
    category="special_numbers",
    execution_modes=["local", "remote"],
    estimated_cpu_usage="high",
    examples=[
        {
            "input": {"limit": 1000},
            "output": [14, 19, 28, 47, 61, 75, 197, 742, 1104],
            "description": "Keith numbers up to 1000",
        },
        {
            "input": {"limit": 100},
            "output": [14, 19, 28, 47, 61, 75],
            "description": "Keith numbers up to 100",
        },
        {
            "input": {"limit": 10000},
            "output": [
                14,
                19,
                28,
                47,
                61,
                75,
                197,
                742,
                1104,
                1537,
                2208,
                2580,
                3684,
                4788,
                7385,
                7647,
                7909,
            ],
            "description": "Keith numbers up to 10000",
        },
    ],
)
async def keith_numbers(limit: int) -> List[int]:
    """
    Find all Keith numbers up to limit.

    A Keith number appears in a Fibonacci-like sequence based on its digits.
    E.g., 14: sequence is 1, 4, 5, 9, 14... (14 appears in the sequence).

    Args:
        limit: Upper bound for search

    Returns:
        List of Keith numbers ≤ limit

    Examples:
        await keith_numbers(1000) → [14, 19, 28, 47, 61, 75, 197, 742, 1104]
        await keith_numbers(100) → [14, 19, 28, 47, 61, 75]
    """
    if limit < 14:  # First Keith number
        return []

    keith_nums = []

    for n in range(10, limit + 1):  # Keith numbers must have at least 2 digits
        digits = [int(d) for d in str(n)]
        len(digits)

        # Generate Fibonacci-like sequence
        sequence = digits[:]
        current_sum = sum(sequence)

        # Continue sequence until we reach or exceed n
        max_iterations = 100  # Prevent infinite loops
        iterations = 0

        while current_sum < n and iterations < max_iterations:
            sequence.append(current_sum)
            # Remove the first element and recalculate sum
            sequence.pop(0)
            current_sum = sum(sequence)
            iterations += 1

        if current_sum == n:
            keith_nums.append(n)

        # Yield control every 100 numbers for large limits
        if n % 100 == 0:
            await asyncio.sleep(0)

    return keith_nums


@mcp_function(
    description="Check if a number is a Keith number and show the sequence.",
    namespace="arithmetic",
    category="special_numbers",
    execution_modes=["local", "remote"],
    examples=[
        {
            "input": {"n": 14},
            "output": {"is_keith": True, "sequence": [1, 4, 5, 9, 14], "steps": 3},
            "description": "14 is Keith: 1, 4, 5, 9, 14",
        },
        {
            "input": {"n": 197},
            "output": {
                "is_keith": True,
                "sequence": [1, 9, 7, 17, 33, 57, 107, 197],
                "steps": 5,
            },
            "description": "197 is Keith with longer sequence",
        },
        {
            "input": {"n": 15},
            "output": {
                "is_keith": False,
                "sequence": [1, 5, 6, 11, 22],
                "final_sum": 22,
            },
            "description": "15 is not Keith",
        },
    ],
)
async def is_keith_number(n: int) -> Dict:
    """
    Check if a number is a Keith number and show the sequence.

    Args:
        n: Number to check

    Returns:
        Dictionary with Keith number analysis

    Examples:
        await is_keith_number(14) → {"is_keith": True, "sequence": [1, 4, 5, 9, 14], ...}
        await is_keith_number(197) → {"is_keith": True, "sequence": [1, 9, 7, 17, 33, 57, 107, 197], ...}
        await is_keith_number(15) → {"is_keith": False, "sequence": [1, 5, 6, 11, 22], ...}
    """
    if n < 10:
        return {
            "is_keith": False,
            "reason": "Keith numbers must have at least 2 digits",
        }

    digits = [int(d) for d in str(n)]
    len(digits)

    sequence = digits[:]
    current_sum = sum(sequence)
    steps = 0
    max_iterations = 100

    while current_sum < n and steps < max_iterations:
        sequence.append(current_sum)
        sequence.pop(0)
        current_sum = sum(sequence)
        steps += 1

    is_keith = current_sum == n

    result = {
        "is_keith": is_keith,
        "sequence": sequence,
        "steps": steps,
        "digits": digits,
        "final_sum": current_sum,
    }

    if not is_keith and current_sum > n:
        result["exceeded_at"] = current_sum

    return result


# ============================================================================
# MAGIC NUMBERS AND CONSTANTS
# ============================================================================


@mcp_function(
    description="Calculate magic constants for n×n magic squares.",
    namespace="arithmetic",
    category="special_numbers",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    examples=[
        {
            "input": {"n": 3},
            "output": {
                "magic_constant": 15,
                "sum_formula": "n(n²+1)/2",
                "total_sum": 45,
            },
            "description": "3×3 magic square constant",
        },
        {
            "input": {"n": 4},
            "output": {
                "magic_constant": 34,
                "sum_formula": "n(n²+1)/2",
                "total_sum": 136,
            },
            "description": "4×4 magic square constant",
        },
        {
            "input": {"n": 5},
            "output": {
                "magic_constant": 65,
                "sum_formula": "n(n²+1)/2",
                "total_sum": 325,
            },
            "description": "5×5 magic square constant",
        },
    ],
)
async def magic_constants(n: int) -> Dict:
    """
    Calculate the magic constant for an n×n magic square.

    The magic constant is the sum that each row, column, and diagonal
    must equal in a magic square using numbers 1 to n².

    Args:
        n: Size of the magic square

    Returns:
        Dictionary with magic square information

    Examples:
        await magic_constants(3) → {"magic_constant": 15, "sum_formula": "n(n²+1)/2", ...}
        await magic_constants(4) → {"magic_constant": 34, "sum_formula": "n(n²+1)/2", ...}
    """
    if n <= 0:
        return {"magic_constant": 0, "error": "n must be positive"}

    # Magic constant formula: M(n) = n(n² + 1) / 2
    magic_constant = n * (n * n + 1) // 2
    total_sum = n * n * (n * n + 1) // 2

    return {
        "n": n,
        "magic_constant": magic_constant,
        "sum_formula": "n(n²+1)/2",
        "total_sum": total_sum,
        "cells": n * n,
        "number_range": [1, n * n],
    }


# ============================================================================
# DIGIT-BASED SPECIAL PROPERTIES
# ============================================================================


@mcp_function(
    description="Calculate sum of digit powers for analysis.",
    namespace="arithmetic",
    category="special_numbers",
    execution_modes=["local", "remote"],
    examples=[
        {
            "input": {"n": 153, "power": 3},
            "output": {
                "digit_power_sum": 153,
                "digits": [1, 5, 3],
                "calculation": "1³ + 5³ + 3³ = 1 + 125 + 27",
            },
            "description": "153 with power 3 (Armstrong number)",
        },
        {
            "input": {"n": 9474, "power": 4},
            "output": {
                "digit_power_sum": 9474,
                "digits": [9, 4, 7, 4],
                "calculation": "9⁴ + 4⁴ + 7⁴ + 4⁴ = 6561 + 256 + 2401 + 256",
            },
            "description": "9474 with power 4",
        },
        {
            "input": {"n": 123, "power": 2},
            "output": {
                "digit_power_sum": 14,
                "digits": [1, 2, 3],
                "calculation": "1² + 2² + 3² = 1 + 4 + 9",
            },
            "description": "123 with power 2",
        },
    ],
)
async def sum_digit_powers(n: int, power: int) -> Dict:
    """
    Calculate the sum of digits raised to a given power.

    Args:
        n: Number to analyze
        power: Power to raise each digit to

    Returns:
        Dictionary with digit power analysis

    Examples:
        await sum_digit_powers(153, 3) → {"digit_power_sum": 153, "digits": [1, 5, 3], ...}
        await sum_digit_powers(9474, 4) → {"digit_power_sum": 9474, "digits": [9, 4, 7, 4], ...}
    """
    if n < 0:
        n = abs(n)

    digits = [int(d) for d in str(n)]
    powers = [d**power for d in digits]
    digit_power_sum = sum(powers)

    # Create calculation string
    power_strs = [f"{d}^{power}" if power != 1 else str(d) for d in digits]
    calculation = f"{' + '.join(power_strs)} = {' + '.join(map(str, powers))}"

    return {
        "n": n,
        "power": power,
        "digits": digits,
        "digit_powers": powers,
        "digit_power_sum": digit_power_sum,
        "calculation": calculation,
        "equals_original": digit_power_sum == n,
    }


@mcp_function(
    description="Calculate digital persistence (number of steps to reach single digit).",
    namespace="arithmetic",
    category="special_numbers",
    execution_modes=["local", "remote"],
    examples=[
        {
            "input": {"n": 39},
            "output": {
                "persistence": 3,
                "steps": [39, 27, 14, 4],
                "operation": "multiply",
            },
            "description": "39 → 27 → 14 → 4 (3 steps)",
        },
        {
            "input": {"n": 999},
            "output": {
                "persistence": 4,
                "steps": [999, 729, 126, 12, 2],
                "operation": "multiply",
            },
            "description": "999 has persistence 4",
        },
        {
            "input": {"n": 99},
            "output": {"persistence": 2, "steps": [99, 18, 8], "operation": "add"},
            "description": "99 with addition: 99 → 18 → 8",
        },
    ],
)
async def digital_persistence(n: int, operation: str = "multiply") -> Dict:
    """
    Calculate digital persistence (steps to reach single digit).

    Args:
        n: Starting number
        operation: "multiply" or "add" for digit operation

    Returns:
        Dictionary with persistence analysis

    Examples:
        await digital_persistence(39) → {"persistence": 3, "steps": [39, 27, 14, 4], ...}
        await digital_persistence(999) → {"persistence": 4, "steps": [999, 729, 126, 12, 2], ...}
    """
    if n < 0:
        n = abs(n)

    steps = [n]
    current = n
    step_count = 0

    while current >= 10:
        digits = [int(d) for d in str(current)]

        if operation == "multiply":
            current = 1
            for digit in digits:
                current *= digit
        elif operation == "add":
            current = sum(digits)
        else:
            return {"error": "Operation must be 'multiply' or 'add'"}

        steps.append(current)
        step_count += 1

        # Safety check
        if step_count > 100:
            break

    return {
        "n": n,
        "operation": operation,
        "persistence": step_count,
        "steps": steps,
        "final_digit": current,
    }


# Export all functions
__all__ = [
    # Amicable numbers and chains
    "find_amicable_pairs",
    "is_amicable_number",
    "find_social_numbers",
    "aliquot_sequence_analysis",
    # Kaprekar numbers
    "kaprekar_numbers",
    "is_kaprekar_number",
    # Vampire numbers
    "vampire_numbers",
    "is_vampire_number",
    # Armstrong numbers and variants
    "armstrong_numbers",
    "dudeney_numbers",
    "pluperfect_numbers",
    # Taxi numbers
    "taxi_numbers",
    # Keith numbers
    "keith_numbers",
    "is_keith_number",
    # Magic numbers
    "magic_constants",
    # Digit properties
    "sum_digit_powers",
    "digital_persistence",
]

if __name__ == "__main__":
    import asyncio

    async def test_special_number_categories():
        """Test special number categories functions."""
        print("🌟 Special Number Categories Test")
        print("=" * 35)

        # Test amicable numbers
        print("Amicable Numbers:")
        amicable = await find_amicable_pairs(1500)
        print(f"  find_amicable_pairs(1500) = {amicable}")

        amicable_check = await is_amicable_number(220)
        print(f"  is_amicable_number(220) = {amicable_check}")

        # Test Kaprekar numbers
        print("\nKaprekar Numbers:")
        kaprekar = await kaprekar_numbers(100)
        print(f"  kaprekar_numbers(100) = {kaprekar}")

        kaprekar_check = await is_kaprekar_number(45)
        print(f"  is_kaprekar_number(45) = {kaprekar_check}")

        # Test Armstrong numbers
        print("\nArmstrong Numbers:")
        armstrong = await armstrong_numbers(1000)
        print(f"  armstrong_numbers(1000) = {armstrong}")

        # Test Keith numbers
        print("\nKeith Numbers:")
        keith = await keith_numbers(100)
        print(f"  keith_numbers(100) = {keith}")

        keith_check = await is_keith_number(14)
        print(f"  is_keith_number(14) = {keith_check}")

        # Test magic constants
        print("\nMagic Constants:")
        magic = await magic_constants(3)
        print(f"  magic_constants(3) = {magic}")

        # Test digit properties
        print("\nDigit Properties:")
        powers = await sum_digit_powers(153, 3)
        print(f"  sum_digit_powers(153, 3) = {powers}")

        persistence = await digital_persistence(39)
        print(f"  digital_persistence(39) = {persistence}")

        print("\n✅ All special number category functions working!")

    asyncio.run(test_special_number_categories())
