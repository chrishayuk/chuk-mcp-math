#!/usr/bin/env python3
"""
Comprehensive Trigonometry Library Demonstration Script

A complete showcase of the chuk_mcp_math trigonometry capabilities.
This script demonstrates real-world applications, mathematical relationships,
advanced wave analysis, and cutting-edge trigonometric research applications
in an educational format.

Features:
- 120+ functions across 8 specialized modules
- Navigation and GPS applications
- Physics simulations and oscillations
- Wave analysis and signal processing
- Mathematical identity verification
- Cross-domain mathematical relationships
- Research-level demonstrations

Run with: python trigonometry_demo.py
"""

import asyncio
import time
import math

# Import the comprehensive trigonometry library
from chuk_mcp_math.trigonometry.basic_functions import (
    sin,
    cos,
    tan,
    csc,
    sec,
    cot,
    sin_degrees,
    cos_degrees,
)
from chuk_mcp_math.trigonometry.inverse_functions import asin, acos, atan, atan2
from chuk_mcp_math.trigonometry.hyperbolic import sinh, cosh, tanh
from chuk_mcp_math.trigonometry.angle_conversion import (
    degrees_to_radians,
    radians_to_degrees,
    normalize_angle,
    angle_difference,
)
from chuk_mcp_math.trigonometry.applications import (
    distance_haversine,
    bearing_calculation,
    triangulation,
    oscillation_analysis,
    pendulum_period,
    spring_oscillation,
)
from chuk_mcp_math.trigonometry.wave_analysis import (
    amplitude_from_coefficients,
    beat_frequency_analysis,
    harmonic_analysis,
    phase_shift_analysis,
)


async def print_header(title: str, char: str = "="):
    """Print a formatted section header."""
    print(f"\n{char * 70}")
    print(f"📐 {title}")
    print(f"{char * 70}")


async def print_subheader(title: str):
    """Print a formatted subsection header."""
    print(f"\n📊 {title}")
    print("-" * 50)


async def demo_basic_trigonometry():
    """Demonstrate basic trigonometric functions and their properties."""
    await print_header("Basic Trigonometric Functions")

    await print_subheader("Primary Functions (Radians)")

    # Test at key angles
    key_angles = [
        (0, "0°"),
        (math.pi / 6, "30°"),
        (math.pi / 4, "45°"),
        (math.pi / 3, "60°"),
        (math.pi / 2, "90°"),
        (math.pi, "180°"),
    ]

    print("  Angle    |   sin     |   cos     |   tan     ")
    print("  ---------|-----------|-----------|----------")

    for angle, description in key_angles:
        sin_val = await sin(angle)
        cos_val = await cos(angle)

        # Handle tan at π/2 (90°)
        if abs(angle - math.pi / 2) < 1e-10:
            tan_str = "undefined"
        else:
            tan_val = await tan(angle)
            tan_str = f"{tan_val:9.6f}"

        print(f"  {description:8s} | {sin_val:9.6f} | {cos_val:9.6f} | {tan_str}")

    await print_subheader("Reciprocal Functions")

    reciprocal_angles = [
        (math.pi / 6, "30°"),
        (math.pi / 4, "45°"),
        (math.pi / 3, "60°"),
    ]

    for angle, description in reciprocal_angles:
        csc_val = await csc(angle)
        sec_val = await sec(angle)
        cot_val = await cot(angle)
        print(f"  {description}: csc = {csc_val:.6f}, sec = {sec_val:.6f}, cot = {cot_val:.6f}")

    await print_subheader("Degree Variants")

    print("  Using degree input functions:")
    degree_angles = [0, 30, 45, 60, 90]
    for deg in degree_angles:
        sin_deg = await sin_degrees(deg)
        cos_deg = await cos_degrees(deg)
        print(f"    sin({deg}°) = {sin_deg:.6f}, cos({deg}°) = {cos_deg:.6f}")


async def demo_inverse_functions():
    """Demonstrate inverse trigonometric functions."""
    await print_header("Inverse Trigonometric Functions")

    await print_subheader("Standard Inverse Functions")

    # Test inverse functions with known values
    inverse_tests = [
        (0, "0"),
        (0.5, "0.5"),
        (math.sqrt(2) / 2, "√2/2"),
        (math.sqrt(3) / 2, "√3/2"),
        (1, "1"),
    ]

    print("  Value  |  asin (°) |  acos (°) |  atan (°)")
    print("  -------|-----------|-----------|----------")

    for value, description in inverse_tests:
        asin_rad = await asin(value)
        acos_rad = await acos(value)
        atan_rad = await atan(value)

        asin_deg = await radians_to_degrees(asin_rad)
        acos_deg = await radians_to_degrees(acos_rad)
        atan_deg = await radians_to_degrees(atan_rad)

        print(f"  {description:6s} | {asin_deg:9.1f} | {acos_deg:9.1f} | {atan_deg:8.1f}")

    await print_subheader("atan2 - Full Quadrant Coverage")

    # Demonstrate atan2 in all quadrants
    quadrant_tests = [
        (1, 1, "Q1: (+, +)"),
        (1, -1, "Q2: (+, -)"),
        (-1, -1, "Q3: (-, -)"),
        (-1, 1, "Q4: (-, +)"),
        (1, 0, "Positive Y-axis"),
        (0, 1, "Positive X-axis"),
    ]

    for y, x, description in quadrant_tests:
        atan2_rad = await atan2(y, x)
        atan2_deg = await radians_to_degrees(atan2_rad)
        print(f"  atan2({y:2d}, {x:2d}) = {atan2_deg:6.1f}° ({description})")


async def demo_hyperbolic_functions():
    """Demonstrate hyperbolic functions and their applications."""
    await print_header("Hyperbolic Functions & Applications")

    await print_subheader("Basic Hyperbolic Functions")

    test_values = [0, 0.5, 1, 2, -1]

    print("    x   |   sinh    |   cosh    |   tanh")
    print("  ------|-----------|-----------|----------")

    for x in test_values:
        sinh_val = await sinh(x)
        cosh_val = await cosh(x)
        tanh_val = await tanh(x)
        print(f"  {x:5.1f} | {sinh_val:9.6f} | {cosh_val:9.6f} | {tanh_val:8.6f}")

    await print_subheader("Hyperbolic Identity Verification")

    print("  Verifying cosh²(x) - sinh²(x) = 1:")
    for x in [0.5, 1, 2, 5]:
        sinh_val = await sinh(x)
        cosh_val = await cosh(x)
        identity_value = cosh_val**2 - sinh_val**2
        error = abs(identity_value - 1.0)
        holds = error <= 1e-12
        print(f"    x = {x}: Identity holds: {holds} (error: {error:.2e})")

    await print_subheader("Catenary Curve (Hanging Chain)")

    print("  Catenary curve y = a*cosh(x/a):")
    catenary_points = [(1, 0), (1, 1), (2, 1), (0.5, 0.5)]
    for a, x in catenary_points:
        x_over_a = x / a
        cosh_val = await cosh(x_over_a)
        sinh_val = await sinh(x_over_a)

        y = a * cosh_val
        slope = sinh_val
        arc_length = a * sinh_val

        print(f"    a={a}, x={x}: y={y:.3f}, slope={slope:.3f}, arc_length={arc_length:.3f}")


async def demo_angle_conversions():
    """Demonstrate comprehensive angle conversion capabilities."""
    await print_header("Angle Conversions & Normalization")

    await print_subheader("Multi-Unit Conversions")

    # Test conversions between degrees, radians, and gradians
    test_angles = [0, 30, 45, 90, 180, 270, 360]

    print("  Degrees | Radians  | Gradians")
    print("  --------|----------|----------")

    for deg in test_angles:
        rad = await degrees_to_radians(deg)
        grad = deg * 400.0 / 360.0  # Simple conversion for gradians
        print(f"  {deg:7.0f} | {rad:8.4f} | {grad:8.1f}")

    await print_subheader("Angle Normalization")

    # Test angle normalization
    weird_angles = [370, -45, 450, -180, 720]

    for angle in weird_angles:
        norm_pos = await normalize_angle(angle, "degrees", "positive")
        norm_sym = await normalize_angle(angle, "degrees", "symmetric")
        print(f"  {angle:4d}° → [0°, 360°): {norm_pos:6.1f}°, [-180°, 180°): {norm_sym:6.1f}°")

    await print_subheader("Angle Differences")

    # Test shortest angular distances
    angle_pairs = [(350, 10), (10, 350), (180, 0), (45, 135)]

    for a1, a2 in angle_pairs:
        diff = await angle_difference(a1, a2, "degrees")
        print(f"  From {a1:3d}° to {a2:3d}°: {diff:6.1f}° (shortest path)")


async def demo_wave_analysis():
    """Demonstrate wave analysis capabilities."""
    await print_header("Wave Analysis & Signal Processing")

    await print_subheader("Amplitude and Phase Extraction")

    # Use the existing amplitude_from_coefficients function
    coefficient_examples = [(3, 4), (1, 1), (5, 0), (0, 3)]

    for a, b in coefficient_examples:
        result = await amplitude_from_coefficients(a, b)
        amplitude = result["amplitude"]
        phase_deg = result["phase_degrees"]
        print(f"  {a}cos(θ) + {b}sin(θ) = {amplitude:.3f}cos(θ - {phase_deg:.1f}°)")

    await print_subheader("Beat Frequency Analysis")

    # Use the existing beat_frequency_analysis function
    beat_examples = [
        (440, 444, "A4 slightly out of tune"),
        (100, 103, "Low frequency beat"),
        (1000, 1000, "Perfect unison"),
    ]

    for f1, f2, description in beat_examples:
        beat_result = await beat_frequency_analysis(f1, f2)
        beat_freq = beat_result["beat_frequency"]
        audibility = beat_result["beat_audibility"]
        print(f"  {description}: {f1} Hz + {f2} Hz")
        print(f"    Beat frequency: {beat_freq:.1f} Hz ({audibility})")

    await print_subheader("Harmonic Analysis")

    # Use the existing harmonic_analysis function
    harmonic_result = await harmonic_analysis(
        fundamental_freq=100, harmonics=[1, 2, 3, 4], amplitudes=[1.0, 0.5, 0.25, 0.125]
    )

    thd = harmonic_result["thd_percent"]
    total_rms = harmonic_result["total_rms"]
    print("  Rich harmonic content (100 Hz fundamental):")
    print(f"    Total RMS: {total_rms:.3f}, THD: {thd:.1f}%")

    for h in harmonic_result["harmonic_analysis"][:3]:  # First 3 harmonics
        freq = h["frequency"]
        power = h["power_percent"]
        print(f"      H{h['harmonic_number']}: {freq} Hz ({power:.1f}% power)")

    await print_subheader("Phase Relationships")

    # Use the existing phase_shift_analysis function
    phase_examples = [
        0,
        math.pi / 4,
        math.pi / 2,
        3 * math.pi / 4,
        math.pi,
        5 * math.pi / 4,
        3 * math.pi / 2,
        7 * math.pi / 4,
    ]

    print("  Phase Analysis for 440 Hz waves:")
    for phase in phase_examples:
        phase_result = await phase_shift_analysis(phase, 440)
        phase_deg = phase_result["phase_degrees"]
        relationship = phase_result["phase_relationship"]
        amplitude_factor = phase_result["amplitude_factor"]

        if "time_delay" in phase_result:
            time_delay = phase_result["time_delay"]
            print(f"    {phase_deg:5.1f}°: {relationship}")
            print(
                f"             Amplitude factor: {amplitude_factor:5.3f}, Time delay: {time_delay * 1000:.3f} ms"
            )
        else:
            print(
                f"    {phase_deg:5.1f}°: {relationship}, Amplitude factor: {amplitude_factor:5.3f}"
            )


async def demo_navigation_applications():
    """Demonstrate navigation and GPS applications."""
    await print_header("Navigation & GPS Applications")

    await print_subheader("Great Circle Distances")

    # Famous city pairs
    city_pairs = [
        (40.7128, -74.0060, 34.0522, -118.2437, "New York", "Los Angeles"),
        (51.5074, -0.1278, 48.8566, 2.3522, "London", "Paris"),
        (35.6762, 139.6503, -33.8688, 151.2093, "Tokyo", "Sydney"),
        (55.7558, 37.6176, 59.9311, 30.3609, "Moscow", "St. Petersburg"),
    ]

    for lat1, lon1, lat2, lon2, city1, city2 in city_pairs:
        distance_result = await distance_haversine(lat1, lon1, lat2, lon2)
        bearing_result = await bearing_calculation(lat1, lon1, lat2, lon2)

        dist_km = distance_result["distance_km"]
        dist_miles = distance_result["distance_miles"]
        bearing = bearing_result["bearing_degrees"]
        compass = bearing_result["compass_direction"]

        print(f"  {city1} → {city2}:")
        print(f"    Distance: {dist_km:.0f} km ({dist_miles:.0f} miles)")
        print(f"    Initial bearing: {bearing:.1f}° ({compass})")

    await print_subheader("GPS Triangulation")

    # Simulate GPS triangulation scenarios
    triangulation_examples = [
        ([0, 0], [100, 0], 60, 80, "Simple triangulation"),
        ([10, 10], [50, 20], 45, 35, "Real-world GPS scenario"),
        ([0, 0], [10, 0], 8, 6, "Close reference points"),
    ]

    for point1, point2, dist1, dist2, description in triangulation_examples:
        tri_result = await triangulation(point1, point2, dist1, dist2)

        if tri_result["solutions"]:
            num_solutions = len(tri_result["solutions"])
            unique = tri_result["unique_solution"]

            print(f"  {description}:")
            print(f"    Reference points: {point1}, {point2}")
            print(f"    Distances: {dist1} km, {dist2} km")
            print(f"    Solutions: {num_solutions} ({'unique' if unique else 'ambiguous'})")

            for i, solution in enumerate(tri_result["solutions"]):
                print(f"      Position {i + 1}: ({solution[0]:.2f}, {solution[1]:.2f})")
        else:
            error = tri_result.get("error", "Unknown error")
            print(f"  {description}: {error}")


async def demo_physics_simulations():
    """Demonstrate physics applications and simulations."""
    await print_header("Physics Simulations & Oscillations")

    await print_subheader("Pendulum Motion")

    # Various pendulum configurations
    pendulum_examples = [
        (1.0, 9.81, None, "Standard 1m pendulum (Earth)"),
        (0.25, 1.62, None, "25cm pendulum (Moon gravity)"),
        (1.0, 9.81, 0.3, "1m pendulum, 17° swing"),
        (2.0, 9.81, 0.1, "2m pendulum, small swing"),
    ]

    for length, gravity, max_angle, description in pendulum_examples:
        pendulum_result = await pendulum_period(length, gravity, max_angle)

        period = pendulum_result["period_small_angle"]
        freq = pendulum_result["frequency"]

        print(f"  {description}:")
        print(f"    Period: {period:.3f} s, Frequency: {freq:.3f} Hz")

        if "period_large_angle" in pendulum_result:
            large_period = pendulum_result["period_large_angle"]
            error = pendulum_result["small_angle_error_percent"]
            angle_deg = pendulum_result["max_angle_degrees"]
            print(f"    Large angle ({angle_deg:.1f}°): {large_period:.3f} s")
            print(f"    Small angle error: {error:.2f}%")

    await print_subheader("Spring-Mass Systems")

    # Different spring-mass configurations
    spring_examples = [
        (1.0, 100, 0.1, 0, "Standard spring-mass"),
        (0.5, 200, 0.05, math.pi / 2, "Stiffer spring, release from equilibrium"),
        (2.0, 50, 0.2, 0, "Heavy mass, soft spring"),
        (0.1, 1000, 0.01, 0, "Light mass, very stiff spring"),
    ]

    for mass, k, amplitude, phase, description in spring_examples:
        spring_result = await spring_oscillation(mass, k, amplitude, phase)

        freq = spring_result["natural_frequency"]
        period = spring_result["period"]
        max_vel = spring_result["max_velocity"]
        max_force = spring_result["max_force"]
        energy = spring_result["total_energy"]
        initial = spring_result["initial_condition"]

        print(f"  {description} (m={mass}kg, k={k}N/m, A={amplitude}m):")
        print(f"    Natural frequency: {freq:.2f} Hz, Period: {period:.3f} s")
        print(f"    Max velocity: {max_vel:.3f} m/s, Max force: {max_force:.1f} N")
        print(f"    Total energy: {energy:.4f} J")
        print(f"    Initial condition: {initial}")

    await print_subheader("Oscillation Analysis")

    # Complex oscillation with damping
    damping_examples = [
        (0.1, 2, 0, 0, 1.0, "Undamped oscillation"),
        (0.1, 2, 0, 0.5, 1.0, "Light damping"),
        (0.1, 2, 0, 2.0, 1.0, "Heavy damping"),
        (0.1, 2, 0, 4 * math.pi * 2, 1.0, "Critical damping"),
    ]

    for amplitude, freq, phase, damping, mass, description in damping_examples:
        osc_result = await oscillation_analysis(amplitude, freq, phase, damping, mass)

        damping_type = osc_result["damping_type"]
        quality = osc_result["quality_factor"]

        print(f"  {description}:")
        print(f"    Damping type: {damping_type}")
        if quality == float("inf"):
            print("    Quality factor: ∞")
        else:
            print(f"    Quality factor: {quality:.2f}")

        if "decay_time" in osc_result:
            decay = osc_result["decay_time"]
            if decay != float("inf"):
                print(f"    Decay time: {decay:.3f} s")


async def demo_mathematical_identities():
    """Demonstrate trigonometric identity verification."""
    await print_header("Mathematical Identity Verification")

    await print_subheader("Pythagorean Identities")

    test_angles = [math.pi / 6, math.pi / 4, math.pi / 3, math.pi / 2, math.pi]

    for angle in test_angles:
        sin_val = await sin(angle)
        cos_val = await cos(angle)
        sin_cos_identity = sin_val**2 + cos_val**2
        sin_cos_error = abs(sin_cos_identity - 1.0)
        sin_cos_ok = sin_cos_error <= 1e-12

        angle_deg = math.degrees(angle)
        print(
            f"  sin²({angle_deg:3.0f}°) + cos²({angle_deg:3.0f}°) = 1: {sin_cos_ok} (error: {sin_cos_error:.2e})"
        )

        # Check other identities if not at singularities
        if abs(cos_val) > 1e-10:  # tan is defined
            tan_val = await tan(angle)
            sec_val = await sec(angle)
            sec_tan_identity = 1 + tan_val**2
            sec_squared = sec_val**2
            sec_tan_error = abs(sec_tan_identity - sec_squared)
            sec_tan_ok = sec_tan_error <= 1e-12
            print(f"    1 + tan²({angle_deg:3.0f}°) = sec²({angle_deg:3.0f}°): {sec_tan_ok}")

    await print_subheader("Sum and Difference Formulas")

    # Test classic angle combinations
    formula_tests = [
        (math.pi / 4, math.pi / 6, "add", "45° + 30° = 75°"),
        (math.pi / 3, math.pi / 6, "subtract", "60° - 30° = 30°"),
        (math.pi / 2, math.pi / 4, "subtract", "90° - 45° = 45°"),
    ]

    for a, b, operation, description in formula_tests:
        sin_a = await sin(a)
        cos_a = await cos(a)
        sin_b = await sin(b)
        cos_b = await cos(b)

        if operation == "add":
            sin_result = sin_a * cos_b + cos_a * sin_b
            cos_result = cos_a * cos_b - sin_a * sin_b
            result_angle = a + b
        else:
            sin_result = sin_a * cos_b - cos_a * sin_b
            cos_result = cos_a * cos_b + sin_a * sin_b
            result_angle = a - b

        # Verify with direct calculation
        sin_direct = await sin(result_angle)
        cos_direct = await cos(result_angle)

        sin_error = abs(sin_result - sin_direct)
        cos_error = abs(cos_result - cos_direct)
        verified = sin_error <= 1e-12 and cos_error <= 1e-12
        max_error = max(sin_error, cos_error)

        print(f"  {description}:")
        print(f"    sin = {sin_result:.6f}, cos = {cos_result:.6f}")
        print(f"    Verified: {verified} (max error: {max_error:.2e})")

    await print_subheader("Double Angle Formulas")

    double_test_angles = [math.pi / 6, math.pi / 4, math.pi / 3]

    for angle in double_test_angles:
        angle_deg = math.degrees(angle)

        # Double angle formulas
        sin_val = await sin(angle)
        cos_val = await cos(angle)

        double_sin_formula = 2 * sin_val * cos_val
        double_cos_formula = cos_val**2 - sin_val**2

        # Verify with direct calculation
        double_angle = 2 * angle
        double_sin_direct = await sin(double_angle)
        double_cos_direct = await cos(double_angle)

        print(f"  Double angle from {angle_deg:.0f}°:")
        print(
            f"    sin(2×{angle_deg:.0f}°) = {double_sin_formula:.6f} (direct: {double_sin_direct:.6f})"
        )
        print(
            f"    cos(2×{angle_deg:.0f}°) = {double_cos_formula:.6f} (direct: {double_cos_direct:.6f})"
        )


async def demo_educational_examples():
    """Demonstrate educational applications."""
    await print_header("Educational Applications & Examples")

    await print_subheader("Unit Circle Exploration")

    print("  Complete unit circle tour:")
    print("  Angle | Coordinates | Quadrant | Reference Angle")
    print("  ------|-------------|----------|----------------")

    unit_circle_angles = [
        0,
        30,
        45,
        60,
        90,
        120,
        135,
        150,
        180,
        210,
        225,
        240,
        270,
        300,
        315,
        330,
    ]

    for angle_deg in unit_circle_angles:
        angle_rad = await degrees_to_radians(angle_deg)
        x = await cos(angle_rad)
        y = await sin(angle_rad)

        # Determine quadrant
        if 0 <= angle_deg < 90:
            quadrant = "I"
        elif 90 <= angle_deg < 180:
            quadrant = "II"
        elif 180 <= angle_deg < 270:
            quadrant = "III"
        else:
            quadrant = "IV"

        # Calculate reference angle
        if angle_deg <= 90:
            ref_angle = angle_deg
        elif angle_deg <= 180:
            ref_angle = 180 - angle_deg
        elif angle_deg <= 270:
            ref_angle = angle_deg - 180
        else:
            ref_angle = 360 - angle_deg

        print(
            f"  {angle_deg:3d}°  | ({x:6.3f}, {y:6.3f}) |    {quadrant}     |     {ref_angle:2.0f}°"
        )

    await print_subheader("Real-World Problem Solving")

    # Classic trigonometry word problems
    print("  Classic problem: Finding height of a building")
    print("  Given: 50m from building, angle of elevation = 60°")

    distance = 50  # meters
    angle_deg = 60
    angle_rad = await degrees_to_radians(angle_deg)
    height = distance * await tan(angle_rad)

    print(f"    Building height = {distance}m × tan({angle_deg}°) = {height:.1f}m")

    print("\n  Navigation problem: Ship heading")
    print("  Ship at (0,0) needs to reach (100, 173) - what heading?")

    target_x, target_y = 100, 173
    bearing_rad = await atan2(target_x, target_y)  # Note: atan2(x,y) for navigation
    bearing_deg = await radians_to_degrees(bearing_rad)
    distance = math.sqrt(target_x**2 + target_y**2)

    print(f"    Distance: {distance:.1f} units")
    print(f"    Heading: {bearing_deg:.1f}° from north")


async def demo_performance_and_precision():
    """Demonstrate performance and numerical precision."""
    await print_header("Performance & Numerical Precision")

    await print_subheader("High-Precision Calculations")

    # Test precision at critical points
    precision_tests = [
        (0, "sin(0) should be exactly 0"),
        (math.pi / 2, "sin(π/2) should be exactly 1"),
        (math.pi, "sin(π) should be exactly 0"),
        (math.pi / 4, "sin(π/4) = cos(π/4) = √2/2"),
    ]

    for angle, description in precision_tests:
        sin_val = await sin(angle)
        cos_val = await cos(angle)

        print(f"  {description}:")
        print(f"    sin({math.degrees(angle):3.0f}°) = {sin_val}")
        print(f"    cos({math.degrees(angle):3.0f}°) = {cos_val}")

        # Check precision
        if angle == 0:
            error = abs(sin_val)
        elif angle == math.pi / 2:
            error = abs(sin_val - 1)
        elif angle == math.pi:
            error = abs(sin_val)
        elif angle == math.pi / 4:
            expected = math.sqrt(2) / 2
            error = max(abs(sin_val - expected), abs(cos_val - expected))

        print(f"    Precision error: {error:.2e}")

    await print_subheader("Performance Benchmarks")

    # Performance tests
    print("  Performance benchmarks:")

    # Basic trigonometry benchmark
    start_time = time.time()
    for _ in range(1000):
        await sin(math.pi / 4)
    end_time = time.time()
    avg_time = (end_time - start_time) / 1000 * 1000000  # Convert to microseconds
    print(f"    Basic trigonometry: {avg_time:.2f} μs per call")

    # Inverse functions benchmark
    start_time = time.time()
    for _ in range(1000):
        await asin(0.7071067811865476)
    end_time = time.time()
    avg_time = (end_time - start_time) / 1000 * 1000000
    print(f"    Inverse functions: {avg_time:.2f} μs per call")

    # Hyperbolic functions benchmark
    start_time = time.time()
    for _ in range(1000):
        await sinh(1.0)
    end_time = time.time()
    avg_time = (end_time - start_time) / 1000 * 1000000
    print(f"    Hyperbolic functions: {avg_time:.2f} μs per call")

    # Angle conversion benchmark
    start_time = time.time()
    for _ in range(1000):
        await degrees_to_radians(45)
    end_time = time.time()
    avg_time = (end_time - start_time) / 1000 * 1000000
    print(f"    Angle conversion: {avg_time:.2f} μs per call")


async def main():
    """Main demonstration function."""
    print("📐 COMPREHENSIVE TRIGONOMETRY LIBRARY DEMONSTRATION")
    print("=" * 70)
    print("Welcome to the chuk_mcp_math trigonometry showcase!")
    print("This script demonstrates the extensive capabilities of our")
    print("async-native trigonometry library with 120+ functions across")
    print("8 specialized modules, covering everything from basic trig")
    print("to advanced signal processing and navigation applications.")

    # Record start time
    start_time = time.time()

    # Run all demonstrations
    demos = [
        demo_basic_trigonometry,
        demo_inverse_functions,
        demo_hyperbolic_functions,
        demo_angle_conversions,
        demo_mathematical_identities,
        demo_wave_analysis,
        demo_navigation_applications,
        demo_physics_simulations,
        demo_educational_examples,
        demo_performance_and_precision,
    ]

    for demo in demos:
        await demo()

    # Show performance summary
    end_time = time.time()
    await print_header("Performance Summary", "=")
    print("✅ Demonstration completed!")
    print(f"📊 Successfully ran: {len(demos)}/{len(demos)} demonstrations")
    print(f"⏱️  Total execution time: {end_time - start_time:.2f} seconds")
    print("🚀 All functions executed asynchronously for optimal performance")

    print("\n💡 This trigonometry library demonstrates:")
    features = [
        "Complete async/await implementation for optimal performance",
        "Comprehensive trigonometric capabilities across 8 modules",
        "Real-world applications in navigation, physics, and signal processing",
        "Educational value with clear mathematical explanations",
        "High-precision calculations with identity verification",
        "Professional-grade error handling and robust implementations",
    ]

    for feature in features:
        print(f"   • {feature}")

    print("\n📈 Module Status:")
    modules = [
        ("Core trigonometry functions", "✅ Working"),
        ("Inverse functions", "✅ Working"),
        ("Hyperbolic functions", "✅ Working"),
        ("Angle conversions", "✅ Working"),
        ("Mathematical identities", "✅ Working"),
        ("Wave analysis", "✅ Working"),
        ("Navigation applications", "✅ Working"),
        ("Physics simulations", "✅ Working"),
        ("Educational examples", "✅ Working"),
        ("Performance benchmarks", "✅ Working"),
    ]

    for module, status in modules:
        print(f"   📐 {module}: {status}")


if __name__ == "__main__":
    # Run the comprehensive trigonometry demonstration
    asyncio.run(main())
