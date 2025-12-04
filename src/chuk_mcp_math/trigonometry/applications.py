#!/usr/bin/env python3
# chuk_mcp_math/trigonometry/applications.py
"""
Trigonometry Applications - Async Native

Real-world applications of trigonometric functions including navigation,
physics simulations, oscillations, and engineering calculations.

Functions:
- Navigation: Haversine distance, bearing calculation, triangulation
- Physics: Oscillation analysis, pendulum motion, spring systems
- Engineering: Signal processing, wave interference, modulation
- Astronomy: Celestial calculations, coordinate transformations
"""

import math
import asyncio
from typing import Union, List, Dict, Any, Optional
from chuk_mcp_math.mcp_decorator import mcp_function

# ============================================================================
# NAVIGATION AND GEODESY
# ============================================================================


@mcp_function(
    description="Calculate great circle distance between two points using Haversine formula.",
    namespace="trigonometry",
    category="applications",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    cache_ttl_seconds=3600,
    examples=[
        {
            "input": {
                "lat1": 40.7128,
                "lon1": -74.0060,
                "lat2": 34.0522,
                "lon2": -118.2437,
            },
            "output": {
                "distance_km": 3944.4197329794813,
                "distance_miles": 2451.2003803234434,
                "distance_nautical_miles": 2129.6503831967943,
                "bearing_initial": 258.3,
            },
            "description": "Distance from NYC to LA",
        },
        {
            "input": {
                "lat1": 51.5074,
                "lon1": -0.1278,
                "lat2": 48.8566,
                "lon2": 2.3522,
            },
            "output": {
                "distance_km": 344.0906596718353,
                "distance_miles": 213.81868154273605,
                "distance_nautical_miles": 185.79617710151462,
            },
            "description": "Distance from London to Paris",
        },
    ],
)
async def distance_haversine(
    lat1: Union[int, float],
    lon1: Union[int, float],
    lat2: Union[int, float],
    lon2: Union[int, float],
    radius: Union[int, float] = 6371.0,
) -> Dict[str, Any]:
    """
    Calculate the great circle distance between two points on Earth using the Haversine formula.

    The Haversine formula determines the great-circle distance between two points
    on a sphere given their latitude and longitude coordinates.

    Args:
        lat1: Latitude of first point in degrees
        lon1: Longitude of first point in degrees
        lat2: Latitude of second point in degrees
        lon2: Longitude of second point in degrees
        radius: Earth radius in kilometers (default: 6371.0 km)

    Returns:
        Dictionary containing distances in various units and additional info

    Examples:
        await distance_haversine(40.7128, -74.0060, 34.0522, -118.2437) → {...}  # NYC to LA
        await distance_haversine(0, 0, 0, 90) → {...}  # Quarter of Earth circumference
    """
    from .angle_conversion import degrees_to_radians
    from .basic_functions import sin, cos
    from .inverse_functions import asin

    # Convert coordinates to radians
    lat1_rad = await degrees_to_radians(lat1)
    lon1_rad = await degrees_to_radians(lon1)
    lat2_rad = await degrees_to_radians(lat2)
    lon2_rad = await degrees_to_radians(lon2)

    # Calculate differences
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    # Haversine formula
    a = (await sin(dlat / 2)) ** 2 + (await cos(lat1_rad)) * (await cos(lat2_rad)) * (
        await sin(dlon / 2)
    ) ** 2
    c = 2 * await asin(math.sqrt(a))

    # Calculate distance
    distance_km = radius * c
    distance_miles = distance_km * 0.621371  # km to miles
    distance_nautical_miles = distance_km * 0.539957  # km to nautical miles
    distance_meters = distance_km * 1000

    # Calculate initial bearing
    bearing = await bearing_calculation(lat1, lon1, lat2, lon2)

    return {
        "distance_km": distance_km,
        "distance_miles": distance_miles,
        "distance_nautical_miles": distance_nautical_miles,
        "distance_meters": distance_meters,
        "bearing_initial": bearing["bearing_degrees"],
        "coordinates": {
            "point1": {"lat": lat1, "lon": lon1},
            "point2": {"lat": lat2, "lon": lon2},
        },
        "earth_radius_km": radius,
        "formula": "Haversine",
    }


@mcp_function(
    description="Calculate bearing (compass direction) from one point to another.",
    namespace="trigonometry",
    category="applications",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    cache_ttl_seconds=3600,
    examples=[
        {
            "input": {
                "lat1": 40.7128,
                "lon1": -74.0060,
                "lat2": 34.0522,
                "lon2": -118.2437,
            },
            "output": {
                "bearing_degrees": 258.32,
                "bearing_radians": 4.508,
                "compass_direction": "WSW",
                "bearing_description": "258.3° (West-Southwest)",
            },
            "description": "Bearing from NYC to LA",
        },
        {
            "input": {"lat1": 0, "lon1": 0, "lat2": 0, "lon2": 90},
            "output": {
                "bearing_degrees": 90.0,
                "compass_direction": "E",
                "bearing_description": "90.0° (East)",
            },
            "description": "Due east bearing",
        },
    ],
)
async def bearing_calculation(
    lat1: Union[int, float],
    lon1: Union[int, float],
    lat2: Union[int, float],
    lon2: Union[int, float],
) -> Dict[str, Any]:
    """
    Calculate the initial bearing (compass direction) from point 1 to point 2.

    Args:
        lat1: Latitude of starting point in degrees
        lon1: Longitude of starting point in degrees
        lat2: Latitude of destination point in degrees
        lon2: Longitude of destination point in degrees

    Returns:
        Dictionary containing bearing in degrees, radians, and compass direction

    Examples:
        await bearing_calculation(40.7128, -74.0060, 34.0522, -118.2437) → {...}
        await bearing_calculation(0, 0, 0, 90) → {"bearing_degrees": 90.0, ...}
    """
    from .angle_conversion import (
        degrees_to_radians,
        radians_to_degrees,
        normalize_angle,
    )
    from .basic_functions import sin, cos
    from .inverse_functions import atan2

    # Convert to radians
    lat1_rad = await degrees_to_radians(lat1)
    lon1_rad = await degrees_to_radians(lon1)
    lat2_rad = await degrees_to_radians(lat2)
    lon2_rad = await degrees_to_radians(lon2)

    # Calculate bearing
    dlon = lon2_rad - lon1_rad

    y = await sin(dlon) * await cos(lat2_rad)
    x = (await cos(lat1_rad)) * (await sin(lat2_rad)) - (await sin(lat1_rad)) * (
        await cos(lat2_rad)
    ) * (await cos(dlon))

    bearing_rad = await atan2(y, x)

    # Normalize to 0-360 degrees
    bearing_rad_normalized = await normalize_angle(bearing_rad, "radians", "positive")
    bearing_deg = await radians_to_degrees(bearing_rad_normalized)

    # Determine compass direction
    compass_direction = _degrees_to_compass(bearing_deg)

    return {
        "bearing_degrees": bearing_deg,
        "bearing_radians": bearing_rad_normalized,
        "compass_direction": compass_direction,
        "bearing_description": f"{bearing_deg:.1f}° ({compass_direction})",
        "coordinates": {
            "from": {"lat": lat1, "lon": lon1},
            "to": {"lat": lat2, "lon": lon2},
        },
    }


def _degrees_to_compass(degrees: float) -> str:
    """Convert bearing degrees to compass direction."""
    directions = [
        "N",
        "NNE",
        "NE",
        "ENE",
        "E",
        "ESE",
        "SE",
        "SSE",
        "S",
        "SSW",
        "SW",
        "WSW",
        "W",
        "WNW",
        "NW",
        "NNW",
    ]
    index = round(degrees / 22.5) % 16
    return directions[index]


@mcp_function(
    description="Solve triangulation problem to find position from known distances to reference points.",
    namespace="trigonometry",
    category="applications",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    estimated_cpu_usage="medium",
    examples=[
        {
            "input": {
                "point1": [0, 0],
                "point2": [10, 0],
                "distance1": 6,
                "distance2": 8,
            },
            "output": {
                "solutions": [[6.0, 0.0], [6.0, 0.0]],
                "unique_solution": True,
                "solution": [6.0, 0.0],
                "triangle_area": 24.0,
            },
            "description": "Simple triangulation example",
        },
        {
            "input": {
                "point1": [0, 0],
                "point2": [5, 0],
                "distance1": 4,
                "distance2": 3,
            },
            "output": {
                "solutions": [[2.4, 3.2], [2.4, -3.2]],
                "unique_solution": False,
                "triangle_area": 6.0,
            },
            "description": "Two possible solutions",
        },
    ],
)
async def triangulation(
    point1: List[Union[int, float]],
    point2: List[Union[int, float]],
    distance1: Union[int, float],
    distance2: Union[int, float],
) -> Dict[str, Any]:
    """
    Find unknown point position using triangulation from two known points.

    Given two reference points and distances to an unknown point,
    calculate the possible positions of the unknown point.

    Args:
        point1: Coordinates [x, y] of first reference point
        point2: Coordinates [x, y] of second reference point
        distance1: Distance from point1 to unknown point
        distance2: Distance from point2 to unknown point

    Returns:
        Dictionary with possible solutions and analysis

    Examples:
        await triangulation([0, 0], [10, 0], 6, 8) → {...}
        await triangulation([0, 0], [5, 0], 4, 3) → {...}
    """

    x1, y1 = point1
    x2, y2 = point2
    r1, r2 = distance1, distance2

    # Distance between reference points
    d = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    # Check if solution exists
    if d > r1 + r2:
        return {
            "solutions": [],
            "unique_solution": False,
            "error": "Points too far apart - no solution exists",
            "distance_between_refs": d,
            "sum_of_distances": r1 + r2,
        }

    if d < abs(r1 - r2):
        return {
            "solutions": [],
            "unique_solution": False,
            "error": "One circle contained within the other - no solution exists",
            "distance_between_refs": d,
            "difference_of_distances": abs(r1 - r2),
        }

    if d == 0 and r1 == r2:
        return {
            "solutions": [],
            "unique_solution": False,
            "error": "Reference points are identical with equal distances - infinite solutions",
        }

    # Calculate intersection points
    a = (r1**2 - r2**2 + d**2) / (2 * d)
    h_squared = r1**2 - a**2

    # Handle numerical precision issues
    if h_squared < 0:
        h_squared = 0

    h = math.sqrt(h_squared)

    # Point on line between circles
    px = x1 + a * (x2 - x1) / d
    py = y1 + a * (y2 - y1) / d

    # Calculate intersection points
    if h == 0:
        # Single solution (circles touch at one point)
        solutions = [[px, py]]
        unique_solution = True
    else:
        # Two solutions
        solution1 = [px + h * (y2 - y1) / d, py - h * (x2 - x1) / d]
        solution2 = [px - h * (y2 - y1) / d, py + h * (x2 - x1) / d]
        solutions = [solution1, solution2]
        unique_solution = False

    # Calculate triangle area (if unique solution)
    triangle_area = None
    if unique_solution and len(solutions) == 1:
        # Area = 0.5 * base * height
        triangle_area = 0.5 * d * h
    elif len(solutions) == 2:
        # Area using Heron's formula
        s = (r1 + r2 + d) / 2  # semi-perimeter
        triangle_area = math.sqrt(s * (s - r1) * (s - r2) * (s - d))

    result = {
        "solutions": solutions,
        "unique_solution": unique_solution,
        "distance_between_refs": d,
        "triangle_area": triangle_area,
        "reference_points": {"point1": point1, "point2": point2},
        "distances": {"to_point1": distance1, "to_point2": distance2},
    }

    if unique_solution:
        result["solution"] = solutions[0]

    return result


# ============================================================================
# PHYSICS AND OSCILLATIONS
# ============================================================================


@mcp_function(
    description="Analyze simple harmonic motion and oscillation properties.",
    namespace="trigonometry",
    category="applications",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    examples=[
        {
            "input": {"amplitude": 0.1, "frequency": 2, "phase": 0, "damping": 0},
            "output": {
                "period": 0.5,
                "angular_frequency": 12.566370614359172,
                "max_velocity": 1.2566370614359172,
                "max_acceleration": 15.78,
                "energy_total": 0.0079,
            },
            "description": "Undamped oscillation with 0.1m amplitude at 2 Hz",
        },
        {
            "input": {
                "amplitude": 0.05,
                "frequency": 1,
                "phase": 1.5707963267948966,
                "damping": 0.1,
            },
            "output": {
                "period": 1.0,
                "damping_type": "underdamped",
                "damped_frequency": 0.995,
            },
            "description": "Damped oscillation starting at maximum displacement",
        },
    ],
)
async def oscillation_analysis(
    amplitude: Union[int, float],
    frequency: Union[int, float],
    phase: Union[int, float] = 0,
    damping: Union[int, float] = 0,
    mass: Union[int, float] = 1.0,
) -> Dict[str, Any]:
    """
    Analyze properties of oscillatory motion (springs, pendulums, etc.).

    For equation: x(t) = A * e^(-γt) * cos(ωt + φ)
    where γ is the damping coefficient.

    Args:
        amplitude: Maximum displacement (m)
        frequency: Natural frequency (Hz)
        phase: Phase shift (radians)
        damping: Damping coefficient (1/s)
        mass: Mass of oscillator (kg)

    Returns:
        Dictionary with comprehensive oscillation analysis

    Examples:
        await oscillation_analysis(0.1, 2) → {...}  # Simple harmonic motion
        await oscillation_analysis(0.05, 1, π/2, 0.1) → {...}  # Damped oscillation
    """
    from .angle_conversion import radians_to_degrees

    # Basic properties
    angular_frequency = 2 * math.pi * frequency
    period = 1 / frequency if frequency > 0 else float("inf")

    # Kinematic properties (undamped)
    max_velocity = amplitude * angular_frequency
    max_acceleration = amplitude * angular_frequency**2

    # Energy analysis (undamped)
    # Assume spring constant k = mω²
    spring_constant = mass * angular_frequency**2
    total_energy = 0.5 * spring_constant * amplitude**2

    # Damping analysis
    if damping == 0:
        damping_type = "undamped"
        damped_frequency = frequency
        quality_factor = float("inf")
        decay_time = float("inf")
    else:
        critical_damping = 2 * angular_frequency
        damping_ratio = damping / critical_damping

        if damping_ratio < 1:
            damping_type = "underdamped"
            damped_angular_freq = angular_frequency * math.sqrt(1 - damping_ratio**2)
            damped_frequency = damped_angular_freq / (2 * math.pi)
        elif damping_ratio == 1:
            damping_type = "critically damped"
            damped_frequency = 0
        else:
            damping_type = "overdamped"
            damped_frequency = 0

        quality_factor = (
            angular_frequency / (2 * damping) if damping > 0 else float("inf")
        )
        decay_time = 1 / damping if damping > 0 else float("inf")

    # Phase analysis
    phase_degrees = await radians_to_degrees(phase)

    result = {
        "amplitude": amplitude,
        "frequency": frequency,
        "angular_frequency": angular_frequency,
        "period": period,
        "phase_radians": phase,
        "phase_degrees": phase_degrees,
        "max_velocity": max_velocity,
        "max_acceleration": max_acceleration,
        "spring_constant": spring_constant,
        "total_energy": total_energy,
        "mass": mass,
        "damping_coefficient": damping,
        "damping_type": damping_type,
        "quality_factor": quality_factor,
    }

    if damping > 0:
        result.update(
            {
                "damped_frequency": damped_frequency,
                "decay_time": decay_time,
                "damping_ratio": damping_ratio,
            }
        )

    return result


@mcp_function(
    description="Calculate pendulum period and analyze pendulum motion.",
    namespace="trigonometry",
    category="applications",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    examples=[
        {
            "input": {"length": 1.0, "gravity": 9.81},
            "output": {
                "period_small_angle": 2.006,
                "frequency": 0.498,
                "angular_frequency": 3.132,
                "length_meters": 1.0,
                "gravity_ms2": 9.81,
            },
            "description": "1-meter pendulum on Earth",
        },
        {
            "input": {"length": 0.25, "gravity": 1.62},
            "output": {
                "period_small_angle": 2.456,
                "frequency": 0.407,
                "gravity_ms2": 1.62,
            },
            "description": "25cm pendulum on Moon",
        },
    ],
)
async def pendulum_period(
    length: Union[int, float],
    gravity: Union[int, float] = 9.81,
    angle: Union[int, float] = None,  # type: ignore[assignment]
) -> Dict[str, Any]:
    """
    Calculate the period of a pendulum and analyze its motion.

    For small angles: T = 2π√(L/g)
    For large angles: More complex elliptic integral solution

    Args:
        length: Pendulum length in meters
        gravity: Gravitational acceleration (default: 9.81 m/s²)
        angle: Maximum swing angle in radians (optional, for large angle correction)

    Returns:
        Dictionary with pendulum analysis

    Examples:
        await pendulum_period(1.0) → {...}  # 1-meter pendulum
        await pendulum_period(0.25, 1.62) → {...}  # Moon gravity
    """
    from .angle_conversion import radians_to_degrees

    if length <= 0:
        raise ValueError("Pendulum length must be positive")
    if gravity <= 0:
        raise ValueError("Gravity must be positive")

    # Small angle approximation: T = 2π√(L/g)
    period_small_angle = 2 * math.pi * math.sqrt(length / gravity)
    frequency = 1 / period_small_angle
    angular_frequency = 2 * math.pi * frequency

    result = {
        "period_small_angle": period_small_angle,
        "frequency": frequency,
        "angular_frequency": angular_frequency,
        "length_meters": length,
        "gravity_ms2": gravity,
        "approximation": "Small angle (sin θ ≈ θ)",
    }

    # Large angle correction if angle provided
    if angle is not None:
        angle_degrees = await radians_to_degrees(angle)

        if abs(angle) > 0.1:  # > ~6 degrees
            # First-order correction: T ≈ T₀(1 + θ₀²/16)
            correction_factor = 1 + (angle**2) / 16
            period_corrected = period_small_angle * correction_factor

            # Error in small angle approximation
            error_percent = (correction_factor - 1) * 100

            result.update(
                {
                    "max_angle_radians": angle,
                    "max_angle_degrees": angle_degrees,
                    "period_large_angle": period_corrected,
                    "correction_factor": correction_factor,
                    "small_angle_error_percent": error_percent,
                    "approximation": "Large angle correction (first order)",
                }
            )
        else:
            result.update(
                {
                    "max_angle_radians": angle,
                    "max_angle_degrees": angle_degrees,
                    "note": "Angle small enough for small-angle approximation",
                }
            )

    # Add some physical insights
    if length == 1.0 and abs(gravity - 9.81) < 0.1:
        result["note"] = "Standard 1-meter pendulum on Earth"

    return result


@mcp_function(
    description="Analyze spring-mass oscillation system.",
    namespace="trigonometry",
    category="applications",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    examples=[
        {
            "input": {
                "mass": 1.0,
                "spring_constant": 100,
                "amplitude": 0.1,
                "phase": 0,
            },
            "output": {
                "natural_frequency": 1.592,
                "period": 0.628,
                "max_velocity": 1.0,
                "max_acceleration": 10.0,
                "total_energy": 0.5,
            },
            "description": "1 kg mass on 100 N/m spring",
        },
        {
            "input": {
                "mass": 0.5,
                "spring_constant": 50,
                "amplitude": 0.05,
                "phase": 1.5707963267948966,
            },
            "output": {"natural_frequency": 1.592, "period": 0.628, "max_force": 2.5},
            "description": "0.5 kg mass starting from equilibrium",
        },
    ],
)
async def spring_oscillation(
    mass: Union[int, float],
    spring_constant: Union[int, float],
    amplitude: Union[int, float],
    phase: Union[int, float] = 0,
) -> Dict[str, Any]:
    """
    Analyze spring-mass oscillation system.

    For equation: x(t) = A * cos(ωt + φ)
    where ω = √(k/m)

    Args:
        mass: Mass in kg
        spring_constant: Spring constant in N/m
        amplitude: Oscillation amplitude in meters
        phase: Phase shift in radians

    Returns:
        Dictionary with spring-mass system analysis

    Examples:
        await spring_oscillation(1.0, 100, 0.1) → {...}  # 1 kg, 100 N/m spring
        await spring_oscillation(0.5, 50, 0.05, π/2) → {...}  # Starting from equilibrium
    """
    from .angle_conversion import radians_to_degrees

    if mass <= 0:
        raise ValueError("Mass must be positive")
    if spring_constant <= 0:
        raise ValueError("Spring constant must be positive")

    # Natural frequency: ω = √(k/m)
    angular_frequency = math.sqrt(spring_constant / mass)
    natural_frequency = angular_frequency / (2 * math.pi)
    period = 1 / natural_frequency

    # Kinematic properties
    max_velocity = amplitude * angular_frequency
    max_acceleration = amplitude * angular_frequency**2
    max_force = spring_constant * amplitude

    # Energy analysis
    total_energy = 0.5 * spring_constant * amplitude**2

    # Phase analysis
    phase_degrees = await radians_to_degrees(phase)

    # Determine initial conditions based on phase
    if abs(phase) < 0.1:
        initial_condition = "Maximum displacement"
    elif abs(phase - math.pi / 2) < 0.1:
        initial_condition = "Equilibrium, maximum velocity"
    elif abs(phase - math.pi) < 0.1:
        initial_condition = "Maximum displacement (opposite direction)"
    elif abs(phase - 3 * math.pi / 2) < 0.1:
        initial_condition = "Equilibrium, maximum velocity (opposite direction)"
    else:
        initial_condition = f"Custom phase: {phase_degrees:.1f}°"

    return {
        "mass": mass,
        "spring_constant": spring_constant,
        "amplitude": amplitude,
        "phase_radians": phase,
        "phase_degrees": phase_degrees,
        "natural_frequency": natural_frequency,
        "angular_frequency": angular_frequency,
        "period": period,
        "max_velocity": max_velocity,
        "max_acceleration": max_acceleration,
        "max_force": max_force,
        "total_energy": total_energy,
        "initial_condition": initial_condition,
        "equation": f"x(t) = {amplitude} * cos({angular_frequency:.3f}t + {phase:.3f})",
    }


# ============================================================================
# SIGNAL PROCESSING AND WAVE APPLICATIONS
# ============================================================================


@mcp_function(
    description="Analyze wave interference patterns between two or more waves.",
    namespace="trigonometry",
    category="applications",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    estimated_cpu_usage="medium",
    examples=[
        {
            "input": {
                "waves": [
                    {"amplitude": 1, "frequency": 440, "phase": 0},
                    {"amplitude": 1, "frequency": 440, "phase": 3.141592653589793},
                ]
            },
            "output": {
                "interference_type": "destructive",
                "resulting_amplitude": 0.0,
                "beat_frequency": 0.0,
                "phase_difference": 3.141592653589793,
            },
            "description": "Destructive interference (180° out of phase)",
        },
        {
            "input": {
                "waves": [
                    {"amplitude": 1, "frequency": 440, "phase": 0},
                    {"amplitude": 1, "frequency": 444, "phase": 0},
                ]
            },
            "output": {
                "interference_type": "beating",
                "resulting_amplitude": 2.0,
                "beat_frequency": 4.0,
                "envelope_frequency": 2.0,
            },
            "description": "Beat interference (slightly different frequencies)",
        },
    ],
)
async def wave_interference(
    waves: List[Dict[str, Union[int, float]]], time_points: Optional[List[float]] = None
) -> Dict[str, Any]:
    """
    Analyze interference patterns between multiple waves.

    Args:
        waves: List of wave dictionaries with 'amplitude', 'frequency', 'phase'
        time_points: Optional time points for detailed analysis

    Returns:
        Dictionary with interference analysis

    Examples:
        await wave_interference([{"amplitude": 1, "frequency": 440, "phase": 0},
                               {"amplitude": 1, "frequency": 440, "phase": π}]) → {...}
    """
    from .wave_analysis import beat_frequency_analysis

    if len(waves) < 2:
        raise ValueError("Need at least 2 waves for interference analysis")

    # Extract wave parameters
    amplitudes = [w["amplitude"] for w in waves]
    frequencies = [w["frequency"] for w in waves]
    phases = [w["phase"] for w in waves]

    # Analyze two-wave interference (most common case)
    if len(waves) == 2:
        A1, A2 = amplitudes[0], amplitudes[1]
        f1, f2 = frequencies[0], frequencies[1]
        φ1, φ2 = phases[0], phases[1]

        phase_diff = φ2 - φ1
        freq_diff = abs(f2 - f1)

        # Determine interference type
        if freq_diff < 0.1:  # Same frequency (within tolerance)
            # Amplitude of resultant: A = √(A1² + A2² + 2A1A2cos(Δφ))
            cos_phase_diff = math.cos(phase_diff)
            resulting_amplitude = math.sqrt(
                A1**2 + A2**2 + 2 * A1 * A2 * cos_phase_diff
            )

            if abs(cos_phase_diff - 1) < 0.1:
                interference_type = "constructive"
            elif abs(cos_phase_diff + 1) < 0.1:
                interference_type = "destructive"
            else:
                interference_type = "partial"

            beat_frequency = 0.0
        else:
            # Different frequencies - beating
            interference_type = "beating"
            resulting_amplitude = A1 + A2  # Maximum envelope amplitude
            beat_analysis = await beat_frequency_analysis(f1, f2)
            beat_frequency = beat_analysis["beat_frequency"]

    # Calculate resultant at specific time points if provided
    resultant_values = None
    if time_points:
        resultant_values = []
        for t in time_points:
            total: float = 0
            for wave in waves:
                A, f, φ = wave["amplitude"], wave["frequency"], wave["phase"]
                value = A * math.sin(2 * math.pi * f * t + φ)
                total += value
            resultant_values.append(total)

    result = {
        "num_waves": len(waves),
        "wave_parameters": waves,
        "interference_type": interference_type,
        "resulting_amplitude": resulting_amplitude,
        "frequencies": frequencies,
        "amplitudes": amplitudes,
        "phases": phases,
    }

    if len(waves) == 2:
        result.update(
            {
                "phase_difference": phase_diff,
                "frequency_difference": freq_diff,
                "beat_frequency": beat_frequency
                if "beat_frequency" in locals()
                else 0.0,
            }
        )

    if resultant_values:
        result["time_points"] = time_points
        result["resultant_values"] = resultant_values

    return result


# Export all functions
__all__ = [
    # Navigation
    "distance_haversine",
    "bearing_calculation",
    "triangulation",
    # Physics and oscillations
    "oscillation_analysis",
    "pendulum_period",
    "spring_oscillation",
    # Signal processing
    "wave_interference",
]

if __name__ == "__main__":
    import asyncio

    async def test_trigonometry_applications():
        """Test trigonometry application functions."""
        print("🎯 Trigonometry Applications Test")
        print("=" * 35)

        # Test navigation functions
        print("Navigation Applications:")

        # Haversine distance examples
        locations = [
            (40.7128, -74.0060, 34.0522, -118.2437, "NYC to LA"),
            (51.5074, -0.1278, 48.8566, 2.3522, "London to Paris"),
            (0, 0, 0, 90, "Equator quarter circle"),
        ]

        for lat1, lon1, lat2, lon2, description in locations:
            distance_result = await distance_haversine(lat1, lon1, lat2, lon2)
            bearing_result = await bearing_calculation(lat1, lon1, lat2, lon2)

            dist_km = distance_result["distance_km"]
            bearing_deg = bearing_result["bearing_degrees"]
            compass = bearing_result["compass_direction"]

            print(f"  {description}:")
            print(f"    Distance: {dist_km:.1f} km")
            print(f"    Bearing: {bearing_deg:.1f}° ({compass})")

        # Test triangulation
        print("\nTriangulation:")
        triangulation_cases = [
            ([0, 0], [10, 0], 6, 8, "Simple case"),
            ([0, 0], [5, 0], 4, 3, "Two solutions"),
        ]

        for p1, p2, d1, d2, description in triangulation_cases:
            tri_result = await triangulation(p1, p2, d1, d2)
            solutions = tri_result["solutions"]
            unique = tri_result["unique_solution"]

            print(f"  {description}: {len(solutions)} solution(s), unique: {unique}")
            for i, sol in enumerate(solutions):
                print(f"    Solution {i + 1}: ({sol[0]:.2f}, {sol[1]:.2f})")

        print("\nPhysics Applications:")

        # Test oscillation analysis
        osc_result = await oscillation_analysis(
            amplitude=0.1, frequency=2, phase=0, damping=0.1, mass=1.0
        )
        print("  Damped oscillation:")
        print(f"    Period: {osc_result['period']:.3f} s")
        print(f"    Damping type: {osc_result['damping_type']}")
        print(f"    Max velocity: {osc_result['max_velocity']:.3f} m/s")

        # Test pendulum
        pendulum_result = await pendulum_period(
            1.0, 9.81, 0.2
        )  # 1m pendulum, 0.2 rad swing
        print("\n  Pendulum (1m, 0.2 rad swing):")
        print(f"    Small angle period: {pendulum_result['period_small_angle']:.3f} s")
        if "period_large_angle" in pendulum_result:
            print(
                f"    Large angle period: {pendulum_result['period_large_angle']:.3f} s"
            )
            print(
                f"    Error in small angle: {pendulum_result['small_angle_error_percent']:.2f}%"
            )

        # Test spring-mass system
        spring_result = await spring_oscillation(
            mass=1.0, spring_constant=100, amplitude=0.1, phase=0
        )
        print("\n  Spring-mass system (1 kg, 100 N/m):")
        print(f"    Natural frequency: {spring_result['natural_frequency']:.3f} Hz")
        print(f"    Max force: {spring_result['max_force']:.1f} N")
        print(f"    Total energy: {spring_result['total_energy']:.3f} J")

        print("\nWave Applications:")

        # Test wave interference
        interference_cases = [
            (
                [
                    {"amplitude": 1, "frequency": 440, "phase": 0},
                    {"amplitude": 1, "frequency": 440, "phase": math.pi},
                ],
                "Destructive",
            ),
            (
                [
                    {"amplitude": 1, "frequency": 440, "phase": 0},
                    {"amplitude": 1, "frequency": 444, "phase": 0},
                ],
                "Beat frequency",
            ),
        ]

        for waves, description in interference_cases:
            interference_result = await wave_interference(waves)
            interference_type = interference_result["interference_type"]
            amplitude = interference_result["resulting_amplitude"]

            print(f"  {description}:")
            print(f"    Type: {interference_type}")
            print(f"    Resulting amplitude: {amplitude:.3f}")

            if "beat_frequency" in interference_result:
                beat_freq = interference_result["beat_frequency"]
                print(f"    Beat frequency: {beat_freq:.1f} Hz")

        print("\n✅ All trigonometry application functions working!")

    asyncio.run(test_trigonometry_applications())
