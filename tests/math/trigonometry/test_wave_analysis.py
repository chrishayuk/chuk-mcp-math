#!/usr/bin/env python3
"""Tests for wave_analysis module."""

import pytest
import math
from chuk_mcp_math.trigonometry.wave_analysis import (
    amplitude_from_coefficients,
    wave_amplitude_analysis,
    frequency_from_period,
    beat_frequency_analysis,
    phase_shift_analysis,
    wave_equation,
    harmonic_analysis,
    fourier_coefficients_basic,
)


class TestAmplitudeAnalysis:
    @pytest.mark.asyncio
    async def test_amplitude_from_coefficients(self):
        result = await amplitude_from_coefficients(3, 4)
        assert abs(result["amplitude"] - 5.0) < 1e-10

    @pytest.mark.asyncio
    async def test_wave_amplitude_analysis(self):
        result = await wave_amplitude_analysis(2.5, 440, 0)
        assert result["peak_amplitude"] == 2.5
        assert result["frequency_hz"] == 440
        assert abs(result["rms_amplitude"] - 2.5 / math.sqrt(2)) < 1e-10


class TestFrequencyAnalysis:
    @pytest.mark.asyncio
    async def test_frequency_from_period(self):
        result = await frequency_from_period(0.02)
        assert abs(result["frequency"] - 50.0) < 1e-10

    @pytest.mark.asyncio
    async def test_frequency_from_period_negative_error(self):
        """Test that negative period raises ValueError (line 238)."""
        with pytest.raises(ValueError, match="Period must be positive"):
            await frequency_from_period(-0.5)

    @pytest.mark.asyncio
    async def test_frequency_from_period_zero_error(self):
        """Test that zero period raises ValueError (line 238)."""
        with pytest.raises(ValueError, match="Period must be positive"):
            await frequency_from_period(0)

    @pytest.mark.asyncio
    async def test_frequency_from_period_infrasonic(self):
        """Test infrasonic frequency category (line 269)."""
        result = await frequency_from_period(0.1)  # 10 Hz
        assert result["frequency_category"] == "Infrasonic (< 20 Hz)"
        assert "Outside typical audible range" in result["note_info"]  # line 252

    @pytest.mark.asyncio
    async def test_frequency_from_period_ultrasonic(self):
        """Test ultrasonic frequency category (lines 272-273)."""
        result = await frequency_from_period(1 / 25000)  # 25 kHz
        assert result["frequency_category"] == "Ultrasonic (> 20 kHz)"
        assert "Outside typical audible range" in result["note_info"]  # line 252

    @pytest.mark.asyncio
    async def test_frequency_from_period_radio(self):
        """Test radio frequency category (lines 274-275)."""
        result = await frequency_from_period(1 / 1e10)  # 10 GHz
        assert result["frequency_category"] == "Radio frequency"

    @pytest.mark.asyncio
    async def test_frequency_from_period_microwave(self):
        """Test microwave/higher frequency category (lines 276-277)."""
        result = await frequency_from_period(1 / 1e12)  # 1 THz
        assert result["frequency_category"] == "Microwave/Higher frequency"

    @pytest.mark.asyncio
    async def test_beat_frequency(self):
        result = await beat_frequency_analysis(440, 444)
        assert abs(result["beat_frequency"] - 4.0) < 1e-10
        assert abs(result["avg_frequency"] - 442.0) < 1e-10

    @pytest.mark.asyncio
    async def test_beat_frequency_identical(self):
        """Test beat frequency with identical frequencies (lines 333-334)."""
        result = await beat_frequency_analysis(440, 440)
        assert result["beat_frequency"] == 0
        assert result["beat_audibility"] == "No beat (identical frequencies)"

    @pytest.mark.asyncio
    async def test_beat_frequency_very_slow(self):
        """Test very slow beat (lines 335-336)."""
        result = await beat_frequency_analysis(440, 440.5)
        assert result["beat_frequency"] == 0.5
        assert result["beat_audibility"] == "Very slow beat (< 1 Hz)"

    @pytest.mark.asyncio
    async def test_beat_frequency_fast(self):
        """Test fast beat (lines 339-340)."""
        result = await beat_frequency_analysis(440, 455)
        assert result["beat_frequency"] == 15
        assert result["beat_audibility"] == "Fast beat, may sound rough"

    @pytest.mark.asyncio
    async def test_beat_frequency_too_fast(self):
        """Test beat too fast to perceive (lines 341-342)."""
        result = await beat_frequency_analysis(440, 480)
        assert result["beat_frequency"] == 40
        assert result["beat_audibility"] == "Beat too fast to perceive individually"


class TestPhaseAnalysis:
    @pytest.mark.asyncio
    async def test_phase_shift_quadrature(self):
        result = await phase_shift_analysis(math.pi / 2)
        assert abs(result["phase_degrees"] - 90) < 1e-10
        assert "Quadrature" in result["phase_relationship"]

    @pytest.mark.asyncio
    async def test_phase_shift_antiphase(self):
        result = await phase_shift_analysis(math.pi)
        assert abs(result["phase_degrees"] - 180) < 1e-10
        assert "Antiphase" in result["phase_relationship"]

    @pytest.mark.asyncio
    async def test_phase_shift_in_phase(self):
        """Test in phase relationship (lines 414-415)."""
        result = await phase_shift_analysis(0.05)
        assert "In phase" in result["phase_relationship"]

    @pytest.mark.asyncio
    async def test_phase_shift_negative_quadrature(self):
        """Test -90 degree quadrature (lines 420-421)."""
        result = await phase_shift_analysis(3 * math.pi / 2)
        assert "Quadrature (-90° out of phase)" in result["phase_relationship"]

    @pytest.mark.asyncio
    async def test_phase_shift_leading_first_quadrant(self):
        """Test leading phase in first quadrant (lines 422-423)."""
        result = await phase_shift_analysis(math.pi / 4)
        assert "Leading by" in result["phase_relationship"]
        assert result["phase_degrees"] < 90

    @pytest.mark.asyncio
    async def test_phase_shift_leading_second_quadrant(self):
        """Test leading phase in second quadrant (lines 424-425)."""
        result = await phase_shift_analysis(2 * math.pi / 3)
        assert "Leading by" in result["phase_relationship"]
        assert result["phase_degrees"] > 90

    @pytest.mark.asyncio
    async def test_phase_shift_lagging_third_quadrant(self):
        """Test lagging phase in third quadrant (lines 426-427)."""
        result = await phase_shift_analysis(4 * math.pi / 3)
        assert "Lagging by" in result["phase_relationship"]

    @pytest.mark.asyncio
    async def test_phase_shift_lagging_fourth_quadrant(self):
        """Test lagging phase in fourth quadrant (lines 428-429)."""
        result = await phase_shift_analysis(5 * math.pi / 3)
        assert "Lagging by" in result["phase_relationship"]

    @pytest.mark.asyncio
    async def test_phase_shift_with_frequency(self):
        """Test phase shift with frequency parameter (lines 447-452)."""
        result = await phase_shift_analysis(math.pi / 2, frequency=100)
        assert "time_delay" in result
        assert "frequency" in result
        assert result["frequency"] == 100
        assert "period" in result
        assert abs(result["period"] - 0.01) < 1e-10


class TestWaveEquation:
    @pytest.mark.asyncio
    async def test_wave_equation_generation(self):
        result = await wave_equation(1, 1, 0, 1, 10, "sine")
        assert len(result["time_points"]) == 10
        assert len(result["wave_values"]) == 10
        assert result["wave_properties"]["amplitude"] == 1
        assert result["wave_properties"]["frequency"] == 1

    @pytest.mark.asyncio
    async def test_wave_equation_cosine(self):
        """Test cosine wave type (line 542)."""
        result = await wave_equation(1, 1, 0, 1, 10, "cosine")
        assert result["wave_properties"]["wave_type"] == "cosine"
        # Cosine at t=0 should be amplitude (1.0)
        assert abs(result["wave_values"][0] - 1.0) < 1e-10

    @pytest.mark.asyncio
    async def test_wave_equation_large_samples(self):
        """Test wave equation with many samples to trigger asyncio.sleep (line 550)."""
        result = await wave_equation(1, 1, 0, 1, 250, "sine")
        assert len(result["wave_values"]) == 250
        # Verify the async sleep is triggered by checking we got all samples
        assert len(result["time_points"]) == 250


class TestHarmonicAnalysis:
    @pytest.mark.asyncio
    async def test_harmonic_analysis(self):
        result = await harmonic_analysis(100, [1, 2, 3], [1.0, 0.5, 0.25])
        assert len(result["harmonic_frequencies"]) == 3
        assert result["harmonic_frequencies"][0] == 100
        assert result["harmonic_frequencies"][1] == 200

    @pytest.mark.asyncio
    async def test_harmonic_analysis_mismatch_error(self):
        """Test error when harmonics and amplitudes don't match (line 644)."""
        with pytest.raises(ValueError, match="Number of harmonics must match number of amplitudes"):
            await harmonic_analysis(100, [1, 2, 3], [1.0, 0.5])

    @pytest.mark.asyncio
    async def test_harmonic_analysis_single_harmonic(self):
        """Test harmonic analysis with single harmonic (line 660)."""
        result = await harmonic_analysis(100, [1], [1.0])
        assert result["thd_percent"] == 0.0
        assert len(result["harmonic_frequencies"]) == 1

    @pytest.mark.asyncio
    async def test_fourier_square_wave(self):
        result = await fourier_coefficients_basic("square", 5)
        assert result["waveform_type"] == "square"
        assert len(result["fourier_terms"]) >= 1

    @pytest.mark.asyncio
    async def test_fourier_sawtooth(self):
        result = await fourier_coefficients_basic("sawtooth", 4)
        assert result["waveform_type"] == "sawtooth"
        assert len(result["fourier_terms"]) == 4

    @pytest.mark.asyncio
    async def test_fourier_triangle_wave(self):
        """Test triangle wave Fourier coefficients (lines 782-803)."""
        result = await fourier_coefficients_basic("triangle", 5)
        assert result["waveform_type"] == "triangle"
        # Triangle wave only has odd harmonics
        assert all(term["n"] % 2 == 1 for term in result["fourier_terms"])
        # Verify coefficient type is cosine
        assert all(term["coefficient_type"] == "cosine" for term in result["fourier_terms"])

    @pytest.mark.asyncio
    async def test_fourier_unsupported_waveform(self):
        """Test error for unsupported waveform type."""
        with pytest.raises(ValueError, match="Unsupported waveform type"):
            await fourier_coefficients_basic("invalid_waveform", 5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
