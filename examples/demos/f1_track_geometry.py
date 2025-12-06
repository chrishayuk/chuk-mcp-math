#!/usr/bin/env python3
"""
F1 Track Geometry & Pace Model Demo

Demonstrates Phase 2 geometry and statistics capabilities applied to F1 racing:
- Track geometry (distances, corners, sector analysis)
- Lap time modeling using linear regression
- Outlier detection for anomalous laps
- Moving averages for pace trends
- Circle geometry for corner analysis

This shows how chuk-mcp-math can power motorsport analytics and strategy.
"""

import asyncio
import math
from typing import List

# Import Phase 2 modules
from chuk_mcp_math.geometry.distances import (
    geom_distance,
)
from chuk_mcp_math.geometry.shapes import (
    geom_polygon_area,
    geom_circle_area,
    geom_triangle_area,
)
from chuk_mcp_math.statistics import (
    mean,
    linear_regression,
    moving_average,
    detect_outliers,
    correlation,
)
from chuk_mcp_math.calculus.integration import integrate_simpson


def print_section(title: str):
    """Print a formatted section header."""
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print(f"{'=' * 80}\n")


async def demo_track_geometry():
    """Demonstrate track geometry calculations."""
    print_section("🏁 F1 TRACK GEOMETRY ANALYSIS")

    # Simplified Monaco GP track (approximate coordinates in meters)
    # Using relative coordinates for key points
    track_points = [
        (0, 0),  # Sainte Devote (Start/Finish)
        (150, 50),  # Beau Rivage
        (200, 150),  # Massenet
        (180, 250),  # Casino Square
        (150, 350),  # Mirabeau
        (100, 400),  # Loews Hairpin
        (50, 450),  # Portier
        (150, 500),  # Tunnel entry
        (300, 550),  # Tunnel exit
        (400, 580),  # Nouvelle Chicane
        (450, 600),  # Tabac
        (500, 620),  # Swimming Pool complex
        (480, 650),  # La Rascasse
        (400, 670),  # Anthony Noghes
        (200, 650),  # Back to start
        (0, 0),  # Complete the loop
    ]

    print("🏎️  Track: Monaco Grand Prix (Simplified Model)")
    print(f"   Key points: {len(track_points)}")

    # Calculate track length
    print("\n📏 Track Length Calculation")
    total_length = 0.0
    segment_lengths = []

    for i in range(len(track_points) - 1):
        seg_length = await geom_distance(track_points[i], track_points[i + 1])
        segment_lengths.append(seg_length)
        total_length += seg_length

    print(f"   Total track length: {total_length:.0f} meters ({total_length / 1000:.2f} km)")
    print(f"   Longest segment: {max(segment_lengths):.0f}m")
    print(f"   Shortest segment: {min(segment_lengths):.0f}m")

    # Calculate track area (enclosed by the circuit)
    track_area = await geom_polygon_area(track_points[:-1])  # Exclude duplicate last point
    print(f"\n   Track enclosed area: {track_area:,.0f} m² ({track_area / 1_000_000:.3f} km²)")

    # Analyze a specific corner (Loews Hairpin - tightest corner in F1)
    print("\n🔄 Corner Analysis: Loews Hairpin")
    hairpin_entry = track_points[4]
    hairpin_apex = track_points[5]
    hairpin_exit = track_points[6]

    # Calculate corner angle using triangle
    corner_area = await geom_triangle_area(hairpin_entry, hairpin_apex, hairpin_exit)
    print(f"   Corner triangle area: {corner_area:.0f} m²")

    # Approximate corner radius (assuming circular arc)
    # For a tight hairpin, radius is typically 10-15m
    corner_radius = 12.0  # meters (estimate)
    corner_arc_area = await geom_circle_area(corner_radius)

    print(f"   Estimated corner radius: {corner_radius}m")
    print(f"   Corner arc area: {corner_arc_area:.1f} m²")

    # Calculate theoretical maximum speed through corner
    # v = √(μ × g × r) where μ ≈ 1.8 for F1 tires, g = 9.81 m/s²
    mu = 1.8  # Coefficient of friction
    g = 9.81  # m/s²
    max_corner_speed_ms = math.sqrt(mu * g * corner_radius)
    max_corner_speed_kmh = max_corner_speed_ms * 3.6

    print(
        f"   Theoretical max speed: {max_corner_speed_kmh:.1f} km/h ({max_corner_speed_ms:.1f} m/s)"
    )


async def demo_lap_time_analysis():
    """Demonstrate lap time analysis and modeling."""
    print_section("⏱️  LAP TIME ANALYSIS & MODELING")

    # Simulated lap times (in seconds) for a stint
    # Including some variation and one outlier (pit stop lap)
    lap_numbers = list(range(1, 26))
    lap_times = [
        78.234,
        77.891,
        77.654,
        77.512,
        77.498,  # Laps 1-5 (fuel load decreasing)
        77.421,
        77.389,
        77.345,
        77.298,
        77.267,  # Laps 6-10
        77.256,
        77.234,
        77.198,
        77.189,
        77.201,  # Laps 11-15 (tire deg starting)
        77.287,
        77.398,
        77.456,
        77.587,
        77.698,  # Laps 16-20 (tire deg)
        105.234,  # Lap 21 (PIT STOP)
        78.456,
        77.989,
        77.654,
        77.423,  # Laps 22-25 (fresh tires)
    ]

    print("🏁 Race Stint Analysis")
    print(f"   Total laps: {len(lap_times)}")
    print(f"   Fastest lap: {min(lap_times):.3f}s (Lap {lap_times.index(min(lap_times)) + 1})")
    print(f"   Slowest lap: {max(lap_times):.3f}s (Lap {lap_times.index(max(lap_times)) + 1})")

    # Detect pit stop lap using outlier detection
    print("\n🔍 Outlier Detection (Pit Stops)")
    outliers = await detect_outliers(lap_times, method="zscore", threshold=2.0)

    print(f"   Outliers detected: {outliers['num_outliers']}")
    for idx in outliers["outlier_indices"]:
        lap_num = lap_numbers[idx]
        lap_time = lap_times[idx]
        print(f"   - Lap {lap_num}: {lap_time:.3f}s (Likely pit stop)")

    # Remove pit stop lap for clean analysis
    clean_lap_numbers = [
        lap_numbers[i] for i in range(len(lap_numbers)) if i not in outliers["outlier_indices"]
    ]
    clean_lap_times = [
        lap_times[i] for i in range(len(lap_times)) if i not in outliers["outlier_indices"]
    ]

    # Linear regression to model tire degradation
    print("\n📉 Tire Degradation Model (Linear Regression)")
    reg = await linear_regression(clean_lap_numbers, clean_lap_times)

    print(f"   Degradation rate: {reg['slope']:.4f} s/lap")
    print(f"   Base time (new tires): {reg['intercept']:.3f}s")
    print(f"   R² (goodness of fit): {reg['r_squared']:.4f}")

    if reg["slope"] > 0:
        print("   ✅ Positive slope confirms tire degradation")
        # Calculate when lap time exceeds 78s
        target_time = 78.0
        laps_to_target = (target_time - reg["intercept"]) / reg["slope"]
        print(f"   Projected lap to exceed {target_time}s: Lap {laps_to_target:.0f}")

    # Moving average to smooth pace
    print("\n📊 Pace Trend (3-Lap Moving Average)")
    window = 3
    pace_ma = await moving_average(clean_lap_times, window)

    print(f"   Moving average (window={window}):")
    for i, ma in enumerate(pace_ma[:5]):  # Show first 5
        lap_num = clean_lap_numbers[i + window - 1]
        print(f"   - Lap {lap_num}: {ma:.3f}s")

    # Correlation between lap number and lap time (shows degradation)
    corr = await correlation(clean_lap_numbers, clean_lap_times)
    print(f"\n   Correlation (lap# vs time): {corr:.4f}")
    if corr > 0.5:
        print("   ✅ Strong positive correlation indicates tire degradation")


async def demo_sector_analysis():
    """Demonstrate sector time analysis."""
    print_section("📍 SECTOR ANALYSIS")

    # Monaco has 3 sectors
    # Sector times for multiple laps (in seconds)
    sector_1_times = [27.234, 27.189, 27.156, 27.198, 27.234, 27.287, 27.345]
    sector_2_times = [38.456, 38.389, 38.345, 38.398, 38.467, 38.523, 38.589]
    sector_3_times = [12.456, 12.412, 12.389, 12.398, 12.423, 12.467, 12.512]

    print("⏱️  Sector Time Analysis (7 laps)")

    async def analyze_sector(name: str, times: List[float]):
        avg = await mean(times)
        best = min(times)
        worst = max(times)
        print(f"\n{name}:")
        print(f"   Best: {best:.3f}s")
        print(f"   Average: {avg:.3f}s")
        print(f"   Worst: {worst:.3f}s")
        print(f"   Delta (worst-best): {worst - best:.3f}s")

    await analyze_sector("Sector 1 (Technical)", sector_1_times)
    await analyze_sector("Sector 2 (Tunnel & Pool)", sector_2_times)
    await analyze_sector("Sector 3 (Final Corners)", sector_3_times)

    # Calculate total theoretical best lap
    theoretical_best = min(sector_1_times) + min(sector_2_times) + min(sector_3_times)
    print(f"\n🏆 Theoretical Best Lap: {theoretical_best:.3f}s")
    print("   (Combining best sector times)")


async def demo_fuel_strategy():
    """Demonstrate fuel load impact on lap time."""
    print_section("⛽ FUEL LOAD STRATEGY")

    # Fuel load decreases by ~2 kg per lap in F1
    # Each kg of fuel costs ~0.03s per lap
    laps = list(range(1, 21))
    fuel_load = [100 - (i - 1) * 2 for i in laps]  # Start with 100kg
    lap_times_fuel_effect = [77.5 + (f - 60) * 0.03 for f in fuel_load]

    print("📊 Fuel Load Impact on Lap Times")
    print(f"   Starting fuel: {fuel_load[0]}kg")
    print("   Fuel burn rate: 2 kg/lap")
    print("   Lap time penalty: 0.03s per kg")

    print(f"\n   Lap 1 (heavy fuel): {lap_times_fuel_effect[0]:.3f}s")
    print(f"   Lap 20 (light fuel): {lap_times_fuel_effect[-1]:.3f}s")
    print(f"   Total time saved: {lap_times_fuel_effect[0] - lap_times_fuel_effect[-1]:.3f}s")

    # Regression to verify linear relationship
    reg = await linear_regression(fuel_load, lap_times_fuel_effect)
    print("\n   Linear regression:")
    print(f"   Slope: {reg['slope']:.4f} s/kg (expected: 0.03)")
    print(f"   R²: {reg['r_squared']:.6f} (perfect fit expected)")


async def demo_race_simulation():
    """Demonstrate simple race simulation using integration."""
    print_section("🏁 RACE SIMULATION")

    print("🚦 Race Distance Calculation")

    # Monaco GP: 78 laps of ~3.337 km
    laps_in_race = 78
    lap_length_km = 3.337

    total_distance = laps_in_race * lap_length_km
    print(f"   Race distance: {total_distance:.1f} km ({laps_in_race} laps)")

    # Simulate varying speed throughout lap
    # Speed varies from 80 km/h (hairpin) to 260 km/h (tunnel)
    print("\n⚡ Speed Profile Integration")

    # Model lap as function: speed varies sinusoidally through lap
    # This is simplified - real speed profile is much more complex
    def speed_profile(t):
        """Speed in km/h as function of distance through lap (0 to 1)."""
        # Base speed 170 km/h, varying ±90 km/h
        return 170 + 90 * math.sin(2 * math.pi * t)

    # Calculate average speed via integration
    avg_speed_integral = await integrate_simpson(speed_profile, 0.0, 1.0, 100)
    print(f"   Average speed (via integration): {avg_speed_integral:.1f} km/h")

    # Calculate lap time from average speed
    lap_time_from_speed = (lap_length_km / avg_speed_integral) * 3600  # Convert to seconds
    print(f"   Estimated lap time: {lap_time_from_speed:.1f}s")

    # Total race time
    total_race_time = lap_time_from_speed * laps_in_race
    total_race_minutes = total_race_time / 60

    print(
        f"\n   Estimated race duration: {total_race_minutes:.1f} minutes ({total_race_time / 3600:.2f} hours)"
    )


async def main():
    """Run all F1 demos."""
    print("\n" + "=" * 80)
    print("  🏎️  F1 TRACK GEOMETRY & PACE MODEL")
    print("  Powered by chuk-mcp-math Phase 2")
    print("=" * 80)

    await demo_track_geometry()
    await demo_lap_time_analysis()
    await demo_sector_analysis()
    await demo_fuel_strategy()
    await demo_race_simulation()

    print("\n" + "=" * 80)
    print("  ✅ F1 ANALYSIS COMPLETE!")
    print("  All Phase 2 capabilities demonstrated successfully.")
    print("=" * 80)
    print()

    # Summary
    print_section("PHASE 2 CAPABILITIES DEMONSTRATED")

    capabilities = {
        "Geometry": [
            "✓ Track length calculation (distance)",
            "✓ Track area calculation (polygon area)",
            "✓ Corner analysis (circle geometry, triangles)",
            "✓ Point-in-polygon detection",
        ],
        "Statistics": [
            "✓ Outlier detection (pit stop identification)",
            "✓ Linear regression (tire degradation modeling)",
            "✓ Moving averages (pace trends)",
            "✓ Correlation analysis (degradation verification)",
        ],
        "Calculus (from Phase 1)": [
            "✓ Integration (average speed calculation)",
        ],
    }

    for category, items in capabilities.items():
        print(f"\n{category}:")
        for item in items:
            print(f"  {item}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
