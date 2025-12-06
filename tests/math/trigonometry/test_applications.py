#!/usr/bin/env python3
"""Tests for trigonometry applications module."""

import pytest
import math
from chuk_mcp_math.trigonometry.applications import (
    distance_haversine,
    bearing_calculation,
    triangulation,
    oscillation_analysis,
    pendulum_period,
    spring_oscillation,
    wave_interference,
)


class TestNavigation:
    @pytest.mark.asyncio
    async def test_haversine_distance(self):
        result = await distance_haversine(40.7128, -74.0060, 34.0522, -118.2437)
        assert result["distance_km"] > 3900
        assert result["distance_km"] < 4000

    @pytest.mark.asyncio
    async def test_bearing_calculation(self):
        result = await bearing_calculation(0, 0, 0, 90)
        assert abs(result["bearing_degrees"] - 90) < 1
        assert result["compass_direction"] == "E"

    @pytest.mark.asyncio
    async def test_triangulation(self):
        result = await triangulation([0, 0], [10, 0], 6, 8)
        assert len(result["solutions"]) >= 1

    @pytest.mark.asyncio
    async def test_triangulation_too_far(self):
        """Test error case when points are too far apart - line 333"""
        result = await triangulation([0, 0], [10, 0], 2, 3)
        assert result["solutions"] == []
        assert result["unique_solution"] is False
        assert "too far apart" in result["error"]
        assert result["distance_between_refs"] == 10.0
        assert result["sum_of_distances"] == 5.0

    @pytest.mark.asyncio
    async def test_triangulation_one_circle_inside(self):
        """Test error case when one circle is inside the other - line 342"""
        result = await triangulation([0, 0], [2, 0], 10, 3)
        assert result["solutions"] == []
        assert result["unique_solution"] is False
        assert "contained within" in result["error"]
        assert result["distance_between_refs"] == 2.0
        assert result["difference_of_distances"] == 7.0

    @pytest.mark.asyncio
    async def test_triangulation_identical_points(self):
        """Test error case when reference points are identical - line 351"""
        result = await triangulation([0, 0], [0, 0], 5, 5)
        assert result["solutions"] == []
        assert result["unique_solution"] is False
        assert "infinite solutions" in result["error"]

    @pytest.mark.asyncio
    async def test_triangulation_numerical_precision(self):
        """Test handling of numerical precision issues - line 363"""
        # Case where h_squared might be slightly negative due to floating point
        # Using distances that almost reach each other
        result = await triangulation([0, 0], [10.0, 0], 5.0000001, 4.9999999)
        assert len(result["solutions"]) >= 1
        # This should result in a unique solution at the midpoint

    @pytest.mark.asyncio
    async def test_triangulation_single_solution(self):
        """Test unique solution case - lines 374-375, 387, 403"""
        # When circles touch at exactly one point
        result = await triangulation([0, 0], [10, 0], 5, 5)
        assert result["unique_solution"] is True
        assert len(result["solutions"]) == 1
        assert "solution" in result  # Line 403
        assert result["triangle_area"] is not None  # Line 387


class TestPhysics:
    @pytest.mark.asyncio
    async def test_oscillation_analysis(self):
        result = await oscillation_analysis(0.1, 2, 0, 0, 1.0)
        assert result["period"] == 0.5
        assert result["damping_type"] == "undamped"

    @pytest.mark.asyncio
    async def test_oscillation_analysis_underdamped(self):
        """Test underdamped oscillation - lines 496-511"""
        result = await oscillation_analysis(0.1, 2, 0, 0.5, 1.0)
        assert result["damping_type"] == "underdamped"
        assert "damped_frequency" in result  # Line 534
        assert "decay_time" in result  # Line 534
        assert "damping_ratio" in result  # Line 534
        assert result["damping_ratio"] < 1

    @pytest.mark.asyncio
    async def test_oscillation_analysis_critically_damped(self):
        """Test critically damped oscillation - lines 503-505"""
        angular_freq = 2 * math.pi * 2
        critical_damping = 2 * angular_freq
        result = await oscillation_analysis(0.1, 2, 0, critical_damping, 1.0)
        assert result["damping_type"] == "critically damped"
        assert "damped_frequency" in result  # Line 534
        assert result["damped_frequency"] == 0

    @pytest.mark.asyncio
    async def test_oscillation_analysis_overdamped(self):
        """Test overdamped oscillation - lines 507-508"""
        angular_freq = 2 * math.pi * 2
        critical_damping = 2 * angular_freq
        overdamping = critical_damping * 2
        result = await oscillation_analysis(0.1, 2, 0, overdamping, 1.0)
        assert result["damping_type"] == "overdamped"
        assert "damped_frequency" in result  # Line 534
        assert result["damped_frequency"] == 0

    @pytest.mark.asyncio
    async def test_pendulum_period(self):
        result = await pendulum_period(1.0, 9.81)
        assert abs(result["period_small_angle"] - 2.006) < 0.01

    @pytest.mark.asyncio
    async def test_pendulum_period_invalid_length(self):
        """Test error case for invalid length - line 600"""
        with pytest.raises(ValueError, match="Pendulum length must be positive"):
            await pendulum_period(-1.0, 9.81)

        with pytest.raises(ValueError, match="Pendulum length must be positive"):
            await pendulum_period(0, 9.81)

    @pytest.mark.asyncio
    async def test_pendulum_period_invalid_gravity(self):
        """Test error case for invalid gravity - line 602"""
        with pytest.raises(ValueError, match="Gravity must be positive"):
            await pendulum_period(1.0, -9.81)

        with pytest.raises(ValueError, match="Gravity must be positive"):
            await pendulum_period(1.0, 0)

    @pytest.mark.asyncio
    async def test_pendulum_period_large_angle(self):
        """Test large angle correction - lines 620-641"""
        result = await pendulum_period(1.0, 9.81, 0.5)  # 0.5 radians ~ 28 degrees
        assert "max_angle_radians" in result  # Line 620
        assert "max_angle_degrees" in result
        assert "period_large_angle" in result
        assert "correction_factor" in result
        assert "small_angle_error_percent" in result
        assert result["approximation"] == "Large angle correction (first order)"

    @pytest.mark.asyncio
    async def test_pendulum_period_small_angle_with_angle_param(self):
        """Test small angle with angle parameter - lines 640-641"""
        result = await pendulum_period(2.0, 9.81, 0.05)  # Small angle, not standard 1m pendulum
        assert "max_angle_radians" in result
        assert "max_angle_degrees" in result
        assert "note" in result
        assert "small enough" in result["note"]

    @pytest.mark.asyncio
    async def test_spring_oscillation(self):
        result = await spring_oscillation(1.0, 100, 0.1, 0)
        assert pytest.approx(result["total_energy"], rel=1e-10) == 0.5

    @pytest.mark.asyncio
    async def test_spring_oscillation_invalid_mass(self):
        """Test error case for invalid mass - line 719"""
        with pytest.raises(ValueError, match="Mass must be positive"):
            await spring_oscillation(-1.0, 100, 0.1, 0)

        with pytest.raises(ValueError, match="Mass must be positive"):
            await spring_oscillation(0, 100, 0.1, 0)

    @pytest.mark.asyncio
    async def test_spring_oscillation_invalid_spring_constant(self):
        """Test error case for invalid spring constant - line 721"""
        with pytest.raises(ValueError, match="Spring constant must be positive"):
            await spring_oscillation(1.0, -100, 0.1, 0)

        with pytest.raises(ValueError, match="Spring constant must be positive"):
            await spring_oscillation(1.0, 0, 0.1, 0)

    @pytest.mark.asyncio
    async def test_spring_oscillation_phase_conditions(self):
        """Test different phase conditions - lines 742-749"""
        # Test maximum displacement - line 742
        result = await spring_oscillation(1.0, 100, 0.1, 0)
        assert result["initial_condition"] == "Maximum displacement"

        # Test equilibrium maximum velocity - line 743
        result = await spring_oscillation(1.0, 100, 0.1, math.pi / 2)
        assert result["initial_condition"] == "Equilibrium, maximum velocity"

        # Test opposite direction displacement - line 745
        result = await spring_oscillation(1.0, 100, 0.1, math.pi)
        assert result["initial_condition"] == "Maximum displacement (opposite direction)"

        # Test opposite direction velocity - line 747
        result = await spring_oscillation(1.0, 100, 0.1, 3 * math.pi / 2)
        assert result["initial_condition"] == "Equilibrium, maximum velocity (opposite direction)"

        # Test custom phase - line 749
        result = await spring_oscillation(1.0, 100, 0.1, 1.0)
        assert "Custom phase" in result["initial_condition"]


class TestWaveApplications:
    @pytest.mark.asyncio
    async def test_wave_interference_destructive(self):
        waves = [
            {"amplitude": 1, "frequency": 440, "phase": 0},
            {"amplitude": 1, "frequency": 440, "phase": math.pi},
        ]
        result = await wave_interference(waves)
        assert result["interference_type"] == "destructive"
        assert abs(result["resulting_amplitude"]) < 0.1

    @pytest.mark.asyncio
    async def test_wave_interference_insufficient_waves(self):
        """Test error case for insufficient waves - line 834"""
        with pytest.raises(ValueError, match="Need at least 2 waves"):
            await wave_interference([{"amplitude": 1, "frequency": 440, "phase": 0}])

    @pytest.mark.asyncio
    async def test_wave_interference_constructive(self):
        """Test constructive interference - line 857"""
        waves = [
            {"amplitude": 1, "frequency": 440, "phase": 0},
            {"amplitude": 1, "frequency": 440, "phase": 0},
        ]
        result = await wave_interference(waves)
        assert result["interference_type"] == "constructive"
        assert abs(result["resulting_amplitude"] - 2.0) < 0.1

    @pytest.mark.asyncio
    async def test_wave_interference_partial(self):
        """Test partial interference - line 861"""
        waves = [
            {"amplitude": 1, "frequency": 440, "phase": 0},
            {"amplitude": 1, "frequency": 440, "phase": math.pi / 4},
        ]
        result = await wave_interference(waves)
        assert result["interference_type"] == "partial"
        assert result["resulting_amplitude"] > 0.1
        assert result["resulting_amplitude"] < 2.0

    @pytest.mark.asyncio
    async def test_wave_interference_beating(self):
        """Test beating interference - lines 866-869"""
        waves = [
            {"amplitude": 1, "frequency": 440, "phase": 0},
            {"amplitude": 1, "frequency": 444, "phase": 0},
        ]
        result = await wave_interference(waves)
        assert result["interference_type"] == "beating"
        assert "beat_frequency" in result
        assert result["beat_frequency"] == 4.0

    @pytest.mark.asyncio
    async def test_wave_interference_with_time_points(self):
        """Test wave interference with time points - lines 874-881, 903-904"""
        waves = [
            {"amplitude": 1, "frequency": 440, "phase": 0},
            {"amplitude": 1, "frequency": 440, "phase": math.pi},
        ]
        time_points = [0, 0.001, 0.002]
        result = await wave_interference(waves, time_points=time_points)

        assert "time_points" in result  # Line 903
        assert "resultant_values" in result  # Line 904
        assert result["time_points"] == time_points
        assert len(result["resultant_values"]) == len(time_points)
        # Verify calculation happens in lines 874-881
        assert isinstance(result["resultant_values"], list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
