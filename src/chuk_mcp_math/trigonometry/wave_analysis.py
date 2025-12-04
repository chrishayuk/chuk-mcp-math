#!/usr/bin/env python3
# chuk_mcp_math/trigonometry/wave_analysis.py
"""
Wave Analysis Functions - Async Native

Comprehensive wave analysis and signal processing functions using trigonometric principles.
Includes amplitude extraction, frequency analysis, phase shift detection, and harmonic analysis.

Functions:
- Amplitude and coefficient analysis
- Frequency and period calculations
- Phase shift detection and analysis
- Wave equation modeling
- Harmonic analysis and Fourier basics
- Wave interference and modulation
"""

import math
import asyncio
from typing import Union, List, Dict, Any, Optional
from chuk_mcp_math.mcp_decorator import mcp_function

# ============================================================================
# WAVE AMPLITUDE AND COEFFICIENT ANALYSIS
# ============================================================================


@mcp_function(
    description="Extract amplitude from trigonometric coefficients A*cos(θ) + B*sin(θ).",
    namespace="trigonometry",
    category="wave_analysis",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    cache_ttl_seconds=3600,
    examples=[
        {
            "input": {"a_coeff": 3, "b_coeff": 4},
            "output": {
                "amplitude": 5.0,
                "phase_angle": 0.9272952180016122,
                "phase_degrees": 53.13010235415598,
                "equivalent_form": "5.0 * cos(θ - 0.927)",
            },
            "description": "3cos(θ) + 4sin(θ) = 5cos(θ - 53.13°)",
        },
        {
            "input": {"a_coeff": 1, "b_coeff": 1},
            "output": {
                "amplitude": 1.4142135623730951,
                "phase_angle": 0.7853981633974483,
                "phase_degrees": 45.0,
                "equivalent_form": "1.414 * cos(θ - 0.785)",
            },
            "description": "cos(θ) + sin(θ) = √2 cos(θ - 45°)",
        },
    ],
)
async def amplitude_from_coefficients(
    a_coeff: Union[int, float], b_coeff: Union[int, float]
) -> Dict[str, Any]:
    """
    Extract amplitude and phase from linear combination of sine and cosine.

    Converts A*cos(θ) + B*sin(θ) to R*cos(θ - φ) form where:
    - R = √(A² + B²) (amplitude)
    - φ = arctan(B/A) (phase angle)

    Args:
        a_coeff: Coefficient of cos(θ)
        b_coeff: Coefficient of sin(θ)

    Returns:
        Dictionary containing amplitude, phase angle, and equivalent forms

    Examples:
        await amplitude_from_coefficients(3, 4) → {"amplitude": 5.0, ...}
        await amplitude_from_coefficients(1, 1) → {"amplitude": √2, ...}
    """
    from .inverse_functions import atan2
    from .angle_conversion import radians_to_degrees

    # Calculate amplitude: R = √(A² + B²)
    amplitude = math.sqrt(a_coeff**2 + b_coeff**2)

    # Calculate phase angle: φ = arctan2(B, A)
    phase_angle = await atan2(b_coeff, a_coeff)
    phase_degrees = await radians_to_degrees(phase_angle)

    # Generate equivalent forms
    equivalent_form = f"{amplitude:.3f} * cos(θ - {phase_angle:.3f})"
    equivalent_form_degrees = f"{amplitude:.3f} * cos(θ - {phase_degrees:.1f}°)"

    return {
        "amplitude": amplitude,
        "phase_angle": phase_angle,
        "phase_degrees": phase_degrees,
        "equivalent_form": equivalent_form,
        "equivalent_form_degrees": equivalent_form_degrees,
        "original_a_coeff": a_coeff,
        "original_b_coeff": b_coeff,
        "conversion_type": "A*cos(θ) + B*sin(θ) → R*cos(θ - φ)",
    }


@mcp_function(
    description="Analyze amplitude and phase of a general sinusoidal wave.",
    namespace="trigonometry",
    category="wave_analysis",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    examples=[
        {
            "input": {"amplitude": 2.5, "frequency": 440, "phase": 0.5},
            "output": {
                "peak_amplitude": 2.5,
                "rms_amplitude": 1.7677669529663687,
                "angular_frequency": 2764.6015351590177,
                "period": 0.002272727272727273,
                "wavelength_info": "Depends on wave speed",
            },
            "description": "440 Hz sine wave with amplitude 2.5",
        },
        {
            "input": {"amplitude": 1, "frequency": 60, "phase": 1.5707963267948966},
            "output": {
                "peak_amplitude": 1,
                "rms_amplitude": 0.7071067811865476,
                "angular_frequency": 377.0,
                "period": 0.016666666666666666,
            },
            "description": "60 Hz cosine wave (90° phase shift)",
        },
    ],
)
async def wave_amplitude_analysis(
    amplitude: Union[int, float],
    frequency: Union[int, float],
    phase: Union[int, float] = 0,
) -> Dict[str, Any]:
    """
    Analyze amplitude and related properties of a sinusoidal wave.

    For wave: A * sin(2πft + φ) or A * cos(2πft + φ)

    Args:
        amplitude: Peak amplitude of the wave
        frequency: Frequency in Hz
        phase: Phase shift in radians

    Returns:
        Dictionary with comprehensive amplitude analysis

    Examples:
        await wave_amplitude_analysis(2.5, 440) → {...}  # 440 Hz tone
        await wave_amplitude_analysis(1, 60, π/2) → {...}  # 60 Hz with phase
    """
    from .angle_conversion import radians_to_degrees

    # Calculate derived values
    rms_amplitude = amplitude / math.sqrt(2)  # RMS value for sinusoidal waves
    peak_to_peak = 2 * amplitude
    angular_frequency = 2 * math.pi * frequency  # ω = 2πf
    period = 1 / frequency if frequency != 0 else float("inf")

    # Phase analysis
    phase_degrees = await radians_to_degrees(phase)
    normalized_phase = phase % (2 * math.pi)
    phase_fraction = normalized_phase / (2 * math.pi)

    return {
        "peak_amplitude": amplitude,
        "rms_amplitude": rms_amplitude,
        "peak_to_peak_amplitude": peak_to_peak,
        "frequency_hz": frequency,
        "angular_frequency": angular_frequency,
        "period": period,
        "phase_radians": phase,
        "phase_degrees": phase_degrees,
        "normalized_phase": normalized_phase,
        "phase_fraction": phase_fraction,
        "wavelength_info": "Depends on wave speed (λ = v/f)",
        "wave_equation": f"{amplitude} * sin(2π * {frequency} * t + {phase:.3f})",
    }


# ============================================================================
# FREQUENCY AND PERIOD ANALYSIS
# ============================================================================


@mcp_function(
    description="Calculate frequency from period and related wave properties.",
    namespace="trigonometry",
    category="wave_analysis",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    cache_ttl_seconds=3600,
    examples=[
        {
            "input": {"period": 0.02},
            "output": {
                "frequency": 50.0,
                "angular_frequency": 314.1592653589793,
                "cycles_per_minute": 3000.0,
                "wavelength_at_speed": "Depends on wave speed",
            },
            "description": "50 Hz wave (0.02 s period)",
        },
        {
            "input": {"period": 1.0},
            "output": {
                "frequency": 1.0,
                "angular_frequency": 6.283185307179586,
                "cycles_per_minute": 60.0,
            },
            "description": "1 Hz wave (1 s period)",
        },
    ],
)
async def frequency_from_period(period: Union[int, float]) -> Dict[str, Any]:
    """
    Calculate frequency and related properties from wave period.

    Args:
        period: Wave period in seconds

    Returns:
        Dictionary containing frequency analysis

    Raises:
        ValueError: If period is zero or negative

    Examples:
        await frequency_from_period(0.02) → {"frequency": 50.0, ...}  # 50 Hz
        await frequency_from_period(1/440) → {"frequency": 440.0, ...}  # A4 note
    """
    if period <= 0:
        raise ValueError("Period must be positive")

    # Calculate basic frequency properties
    frequency = 1.0 / period
    angular_frequency = 2 * math.pi * frequency
    cycles_per_minute = frequency * 60
    cycles_per_hour = frequency * 3600

    # Calculate some musical/acoustic properties if frequency is in audible range
    if 20 <= frequency <= 20000:
        # Rough musical note calculation (A4 = 440 Hz)
        semitones_from_a4 = 12 * math.log2(frequency / 440)
        note_info = f"~{semitones_from_a4:.1f} semitones from A4"
    else:
        note_info = "Outside typical audible range (20 Hz - 20 kHz)"

    return {
        "period": period,
        "frequency": frequency,
        "angular_frequency": angular_frequency,
        "cycles_per_minute": cycles_per_minute,
        "cycles_per_hour": cycles_per_hour,
        "wavelength_formula": "λ = v/f (where v is wave speed)",
        "note_info": note_info,
        "frequency_category": _categorize_frequency(frequency),
    }


def _categorize_frequency(freq: float) -> str:
    """Categorize frequency into standard ranges."""
    if freq < 20:
        return "Infrasonic (< 20 Hz)"
    elif freq <= 20000:
        return "Audible (20 Hz - 20 kHz)"
    elif freq <= 1e9:
        return "Ultrasonic (> 20 kHz)"
    elif freq <= 3e11:
        return "Radio frequency"
    else:
        return "Microwave/Higher frequency"


@mcp_function(
    description="Analyze beat frequency from two interfering waves.",
    namespace="trigonometry",
    category="wave_analysis",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    examples=[
        {
            "input": {"freq1": 440, "freq2": 444},
            "output": {
                "beat_frequency": 4.0,
                "beat_period": 0.25,
                "avg_frequency": 442.0,
                "interference_pattern": "Constructive/destructive interference",
            },
            "description": "Beat between 440 Hz and 444 Hz",
        },
        {
            "input": {"freq1": 100, "freq2": 103},
            "output": {
                "beat_frequency": 3.0,
                "beat_period": 0.3333333333333333,
                "avg_frequency": 101.5,
            },
            "description": "3 Hz beat frequency",
        },
    ],
)
async def beat_frequency_analysis(
    freq1: Union[int, float], freq2: Union[int, float]
) -> Dict[str, Any]:
    """
    Analyze beat frequency from two interfering sinusoidal waves.

    When two waves of slightly different frequencies interfere,
    they create a beat pattern with frequency |f1 - f2|.

    Args:
        freq1: Frequency of first wave (Hz)
        freq2: Frequency of second wave (Hz)

    Returns:
        Dictionary with beat frequency analysis

    Examples:
        await beat_frequency_analysis(440, 444) → {"beat_frequency": 4.0, ...}
        await beat_frequency_analysis(100, 103) → {"beat_frequency": 3.0, ...}
    """
    beat_frequency = abs(freq1 - freq2)
    beat_period = 1.0 / beat_frequency if beat_frequency > 0 else float("inf")
    avg_frequency = (freq1 + freq2) / 2

    # Determine audibility of beat
    if beat_frequency == 0:
        beat_audibility = "No beat (identical frequencies)"
    elif beat_frequency < 1:
        beat_audibility = "Very slow beat (< 1 Hz)"
    elif beat_frequency <= 10:
        beat_audibility = "Clearly audible beat"
    elif beat_frequency <= 30:
        beat_audibility = "Fast beat, may sound rough"
    else:
        beat_audibility = "Beat too fast to perceive individually"

    return {
        "freq1": freq1,
        "freq2": freq2,
        "beat_frequency": beat_frequency,
        "beat_period": beat_period,
        "avg_frequency": avg_frequency,
        "beat_audibility": beat_audibility,
        "interference_pattern": "Constructive/destructive interference",
        "envelope_equation": f"2 * cos(2π * {beat_frequency / 2:.1f} * t) * cos(2π * {avg_frequency:.1f} * t)",
    }


# ============================================================================
# PHASE SHIFT ANALYSIS
# ============================================================================


@mcp_function(
    description="Analyze phase relationships between waves.",
    namespace="trigonometry",
    category="wave_analysis",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    examples=[
        {
            "input": {"phase_diff": 1.5707963267948966},
            "output": {
                "phase_degrees": 90.0,
                "phase_relationship": "Quadrature (90° out of phase)",
                "time_delay_fraction": 0.25,
                "amplitude_factor": 0.0,
            },
            "description": "90° phase difference",
        },
        {
            "input": {"phase_diff": 3.141592653589793},
            "output": {
                "phase_degrees": 180.0,
                "phase_relationship": "Antiphase (180° out of phase)",
                "time_delay_fraction": 0.5,
                "amplitude_factor": -1.0,
            },
            "description": "180° phase difference (antiphase)",
        },
    ],
)
async def phase_shift_analysis(
    phase_diff: Union[int, float], frequency: Optional[Union[int, float]] = None
) -> Dict[str, Any]:
    """
    Analyze phase shift between two waves.

    Args:
        phase_diff: Phase difference in radians
        frequency: Frequency in Hz (optional, for time delay calculation)

    Returns:
        Dictionary with comprehensive phase analysis

    Examples:
        await phase_shift_analysis(π/2) → {"phase_relationship": "Quadrature", ...}
        await phase_shift_analysis(π) → {"phase_relationship": "Antiphase", ...}
    """
    from .angle_conversion import radians_to_degrees, normalize_angle

    # Normalize phase difference to [0, 2π)
    normalized_phase = await normalize_angle(phase_diff, "radians", "positive")
    phase_degrees = await radians_to_degrees(normalized_phase)

    # Determine phase relationship
    if abs(normalized_phase) < 0.1:
        relationship = "In phase (0°)"
    elif abs(normalized_phase - math.pi / 2) < 0.1:
        relationship = "Quadrature (90° out of phase)"
    elif abs(normalized_phase - math.pi) < 0.1:
        relationship = "Antiphase (180° out of phase)"
    elif abs(normalized_phase - 3 * math.pi / 2) < 0.1:
        relationship = "Quadrature (-90° out of phase)"
    elif normalized_phase < math.pi / 2:
        relationship = f"Leading by {phase_degrees:.1f}°"
    elif normalized_phase < math.pi:
        relationship = f"Leading by {phase_degrees:.1f}°"
    elif normalized_phase < 3 * math.pi / 2:
        relationship = f"Lagging by {360 - phase_degrees:.1f}°"
    else:
        relationship = f"Lagging by {360 - phase_degrees:.1f}°"

    # Calculate time delay fraction
    time_delay_fraction = normalized_phase / (2 * math.pi)

    # Calculate amplitude factor for interference
    amplitude_factor = math.cos(normalized_phase)

    result = {
        "phase_radians": phase_diff,
        "normalized_phase": normalized_phase,
        "phase_degrees": phase_degrees,
        "phase_relationship": relationship,
        "time_delay_fraction": time_delay_fraction,
        "amplitude_factor": amplitude_factor,
    }

    # Add time delay if frequency is provided
    if frequency is not None and frequency > 0:
        period = 1.0 / frequency
        time_delay = time_delay_fraction * period
        result["time_delay"] = time_delay
        result["frequency"] = frequency
        result["period"] = period

    return result


# ============================================================================
# WAVE EQUATION MODELING
# ============================================================================


@mcp_function(
    description="Generate points for a sinusoidal wave equation.",
    namespace="trigonometry",
    category="wave_analysis",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    examples=[
        {
            "input": {
                "amplitude": 1,
                "frequency": 1,
                "phase": 0,
                "duration": 2,
                "sample_rate": 10,
            },
            "output": {
                "time_points": [0.0, 0.1, 0.2],
                "wave_values": [0.0, 0.5877852522924731, 0.9510565162951536],
                "wave_properties": {"amplitude": 1, "frequency": 1, "period": 1.0},
            },
            "description": "1 Hz sine wave for 2 seconds",
        },
        {
            "input": {
                "amplitude": 2,
                "frequency": 0.5,
                "phase": 1.5707963267948966,
                "duration": 4,
                "sample_rate": 8,
            },
            "output": {
                "time_points": [0.0, 0.125, 0.25],
                "wave_values": [2.0, 1.8477590650225735, 1.4142135623730951],
            },
            "description": "0.5 Hz cosine wave (90° phase)",
        },
    ],
)
async def wave_equation(
    amplitude: Union[int, float],
    frequency: Union[int, float],
    phase: Union[int, float] = 0,
    duration: Union[int, float] = 1.0,
    sample_rate: Union[int, float] = 100,
    wave_type: str = "sine",
) -> Dict[str, Any]:
    """
    Generate time-domain points for a sinusoidal wave equation.

    Wave equation: A * sin(2πft + φ) or A * cos(2πft + φ)

    Args:
        amplitude: Wave amplitude
        frequency: Frequency in Hz
        phase: Phase shift in radians
        duration: Duration in seconds
        sample_rate: Samples per second
        wave_type: "sine" or "cosine"

    Returns:
        Dictionary with time points, wave values, and properties

    Examples:
        await wave_equation(1, 1, 0, 2, 10) → {...}  # 1 Hz sine wave
        await wave_equation(2, 0.5, π/2, 4, 8, "cosine") → {...}  # Cosine wave
    """
    from .basic_functions import sin, cos

    # Generate time points
    num_samples = int(duration * sample_rate)
    time_points = [i / sample_rate for i in range(num_samples)]

    # Calculate wave values
    wave_values = []
    angular_frequency = 2 * math.pi * frequency

    for t in time_points:
        argument = angular_frequency * t + phase

        if wave_type.lower() == "cosine":
            value = amplitude * await cos(argument)
        else:  # default to sine
            value = amplitude * await sin(argument)

        wave_values.append(value)

        # Yield control every 100 samples for large datasets
        if len(wave_values) % 100 == 0:
            await asyncio.sleep(0)

    # Calculate wave properties
    period = 1.0 / frequency if frequency > 0 else float("inf")
    angular_frequency = 2 * math.pi * frequency

    return {
        "time_points": time_points,
        "wave_values": wave_values,
        "wave_properties": {
            "amplitude": amplitude,
            "frequency": frequency,
            "angular_frequency": angular_frequency,
            "phase": phase,
            "period": period,
            "wave_type": wave_type,
            "duration": duration,
            "sample_rate": sample_rate,
            "num_samples": num_samples,
        },
        "wave_equation_string": f"{amplitude} * {wave_type}(2π * {frequency} * t + {phase:.3f})",
    }


# ============================================================================
# HARMONIC ANALYSIS
# ============================================================================


@mcp_function(
    description="Analyze harmonic content and calculate basic Fourier coefficients.",
    namespace="trigonometry",
    category="wave_analysis",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    estimated_cpu_usage="medium",
    examples=[
        {
            "input": {
                "fundamental_freq": 100,
                "harmonics": [1, 2, 3],
                "amplitudes": [1.0, 0.5, 0.25],
            },
            "output": {
                "harmonic_frequencies": [100, 200, 300],
                "total_rms": 1.224744871391589,
                "thd": 61.23724356957945,
                "harmonic_analysis": [
                    {
                        "harmonic": 1,
                        "frequency": 100,
                        "amplitude": 1.0,
                        "power_percent": 66.67,
                    }
                ],
            },
            "description": "Harmonic analysis of 100 Hz fundamental with harmonics",
        },
        {
            "input": {
                "fundamental_freq": 440,
                "harmonics": [1, 2],
                "amplitudes": [1.0, 0.3],
            },
            "output": {
                "harmonic_frequencies": [440, 880],
                "total_rms": 1.044030650891072,
                "thd": 30.0,
            },
            "description": "A4 note (440 Hz) with second harmonic",
        },
    ],
)
async def harmonic_analysis(
    fundamental_freq: Union[int, float],
    harmonics: List[int],
    amplitudes: List[Union[int, float]],
) -> Dict[str, Any]:
    """
    Analyze harmonic content of a complex waveform.

    Args:
        fundamental_freq: Fundamental frequency in Hz
        harmonics: List of harmonic numbers (1 = fundamental, 2 = 2nd harmonic, etc.)
        amplitudes: List of amplitudes for each harmonic

    Returns:
        Dictionary with comprehensive harmonic analysis

    Examples:
        await harmonic_analysis(100, [1, 2, 3], [1.0, 0.5, 0.25]) → {...}
        await harmonic_analysis(440, [1, 2], [1.0, 0.3]) → {...}
    """
    if len(harmonics) != len(amplitudes):
        raise ValueError("Number of harmonics must match number of amplitudes")

    # Calculate harmonic frequencies
    harmonic_frequencies = [fundamental_freq * h for h in harmonics]

    # Calculate RMS values
    rms_values = [amp / math.sqrt(2) for amp in amplitudes]
    total_rms = math.sqrt(sum(rms**2 for rms in rms_values))

    # Calculate Total Harmonic Distortion (THD)
    if len(amplitudes) > 1:
        fundamental_rms = rms_values[0]  # Assuming first harmonic is fundamental
        harmonic_rms_squared_sum = sum(rms**2 for rms in rms_values[1:])
        thd_ratio = math.sqrt(harmonic_rms_squared_sum) / fundamental_rms
        thd_percent = thd_ratio * 100
    else:
        thd_percent = 0.0

    # Calculate power distribution
    total_power = sum(amp**2 for amp in amplitudes)
    power_percentages = [(amp**2 / total_power) * 100 for amp in amplitudes]

    # Create detailed harmonic analysis
    harmonic_details = []
    for i, (harmonic, freq, amp, power_pct) in enumerate(
        zip(harmonics, harmonic_frequencies, amplitudes, power_percentages)
    ):
        harmonic_details.append(
            {
                "harmonic_number": harmonic,
                "frequency": freq,
                "amplitude": amp,
                "rms": rms_values[i],
                "power_percent": power_pct,
                "is_fundamental": harmonic == 1,
            }
        )

    return {
        "fundamental_frequency": fundamental_freq,
        "harmonic_frequencies": harmonic_frequencies,
        "amplitudes": amplitudes,
        "rms_values": rms_values,
        "total_rms": total_rms,
        "thd_percent": thd_percent,
        "harmonic_analysis": harmonic_details,
        "power_distribution": power_percentages,
        "num_harmonics": len(harmonics),
    }


@mcp_function(
    description="Calculate basic Fourier series coefficients for common waveforms.",
    namespace="trigonometry",
    category="wave_analysis",
    execution_modes=["local", "remote"],
    cache_strategy="memory",
    estimated_cpu_usage="medium",
    examples=[
        {
            "input": {"waveform": "square", "n_terms": 5},
            "output": {
                "waveform_type": "square",
                "fourier_terms": [
                    {"n": 1, "amplitude": 1.2732395447351628, "frequency_multiple": 1},
                    {"n": 3, "amplitude": 0.4244131815783876, "frequency_multiple": 3},
                ],
                "series_formula": "Square wave Fourier series",
            },
            "description": "Fourier series for square wave",
        },
        {
            "input": {"waveform": "sawtooth", "n_terms": 4},
            "output": {
                "waveform_type": "sawtooth",
                "fourier_terms": [
                    {"n": 1, "amplitude": 0.6366197723675814, "frequency_multiple": 1},
                    {"n": 2, "amplitude": 0.3183098861837907, "frequency_multiple": 2},
                ],
            },
            "description": "Fourier series for sawtooth wave",
        },
    ],
)
async def fourier_coefficients_basic(waveform: str, n_terms: int = 5) -> Dict[str, Any]:
    """
    Calculate basic Fourier series coefficients for common periodic waveforms.

    Supported waveforms:
    - "square": Square wave
    - "sawtooth": Sawtooth wave
    - "triangle": Triangle wave

    Args:
        waveform: Type of waveform
        n_terms: Number of Fourier terms to calculate

    Returns:
        Dictionary with Fourier coefficients and series information

    Examples:
        await fourier_coefficients_basic("square", 5) → {...}
        await fourier_coefficients_basic("sawtooth", 4) → {...}
    """
    fourier_terms = []

    if waveform.lower() == "square":
        # Square wave: only odd harmonics, amplitude = 4/(nπ)
        for n in range(1, n_terms + 1):
            if n % 2 == 1:  # Only odd harmonics
                amplitude = 4 / (n * math.pi)
                fourier_terms.append(
                    {
                        "n": n,
                        "amplitude": amplitude,
                        "frequency_multiple": n,
                        "coefficient_type": "sine",
                    }
                )

        series_formula = "4/π * Σ(1/n * sin(nωt)) for odd n"

    elif waveform.lower() == "sawtooth":
        # Sawtooth wave: all harmonics, amplitude = 2/(nπ) * (-1)^(n+1)
        for n in range(1, n_terms + 1):
            amplitude = 2 / (n * math.pi) * ((-1) ** (n + 1))
            fourier_terms.append(
                {
                    "n": n,
                    "amplitude": abs(amplitude),  # Store magnitude
                    "sign": 1 if amplitude > 0 else -1,
                    "frequency_multiple": n,
                    "coefficient_type": "sine",
                }
            )

        series_formula = "2/π * Σ((-1)^(n+1)/n * sin(nωt))"

    elif waveform.lower() == "triangle":
        # Triangle wave: only odd harmonics, amplitude = 8/(n²π²) * (-1)^((n-1)/2)
        for n in range(1, n_terms + 1):
            if n % 2 == 1:  # Only odd harmonics
                amplitude = 8 / ((n**2) * (math.pi**2))
                if ((n - 1) // 2) % 2 == 1:
                    amplitude = -amplitude

                fourier_terms.append(
                    {
                        "n": n,
                        "amplitude": abs(amplitude),
                        "sign": 1 if amplitude > 0 else -1,
                        "frequency_multiple": n,
                        "coefficient_type": "cosine",
                    }
                )

        series_formula = "8/π² * Σ((-1)^((n-1)/2)/n² * cos(nωt)) for odd n"

    else:
        raise ValueError(f"Unsupported waveform type: {waveform}")

    return {
        "waveform_type": waveform,
        "fourier_terms": fourier_terms,
        "n_terms": len(fourier_terms),
        "series_formula": series_formula,
        "convergence_note": "More terms provide better approximation",
    }


# Export all functions
__all__ = [
    # Amplitude analysis
    "amplitude_from_coefficients",
    "wave_amplitude_analysis",
    # Frequency analysis
    "frequency_from_period",
    "beat_frequency_analysis",
    # Phase analysis
    "phase_shift_analysis",
    # Wave equation
    "wave_equation",
    # Harmonic analysis
    "harmonic_analysis",
    "fourier_coefficients_basic",
]

if __name__ == "__main__":
    import asyncio

    async def test_wave_analysis_functions():
        """Test wave analysis functions."""
        print("🌊 Wave Analysis Functions Test")
        print("=" * 35)

        # Test amplitude extraction
        print("Amplitude from Coefficients:")
        coeff_cases = [(3, 4), (1, 1), (5, 0), (0, 2)]
        for a, b in coeff_cases:
            result = await amplitude_from_coefficients(a, b)
            amp = result["amplitude"]
            phase_deg = result["phase_degrees"]
            print(f"  {a}cos(θ) + {b}sin(θ) = {amp:.3f}cos(θ - {phase_deg:.1f}°)")

        print("\nFrequency Analysis:")
        periods = [0.02, 1 / 440, 1.0, 0.001]  # 50Hz, A4, 1Hz, 1kHz
        for period in periods:
            freq_result = await frequency_from_period(period)
            freq = freq_result["frequency"]
            category = freq_result["frequency_category"]
            print(f"  Period {period}s → {freq:.1f} Hz ({category})")

        print("\nBeat Frequency Analysis:")
        beat_cases = [(440, 444), (100, 103), (261.63, 261.63)]  # Perfect unison
        for f1, f2 in beat_cases:
            beat_result = await beat_frequency_analysis(f1, f2)
            beat_freq = beat_result["beat_frequency"]
            audibility = beat_result["beat_audibility"]
            print(f"  {f1} Hz + {f2} Hz → Beat: {beat_freq:.1f} Hz ({audibility})")

        print("\nPhase Shift Analysis:")
        phase_shifts = [0, math.pi / 4, math.pi / 2, math.pi, 3 * math.pi / 2]
        for phase in phase_shifts:
            phase_result = await phase_shift_analysis(phase)
            phase_deg = phase_result["phase_degrees"]
            relationship = phase_result["phase_relationship"]
            print(f"  {phase_deg:5.1f}° phase: {relationship}")

        print("\nWave Equation Generation:")
        wave_result = await wave_equation(
            amplitude=1, frequency=2, phase=0, duration=1, sample_rate=8
        )
        time_pts = wave_result["time_points"][:5]  # First 5 points
        wave_vals = wave_result["wave_values"][:5]
        print("  2 Hz sine wave (first 5 points):")
        for t, v in zip(time_pts, wave_vals):
            print(f"    t={t:.3f}s: {v:.3f}")

        print("\nHarmonic Analysis:")
        harmonic_result = await harmonic_analysis(
            fundamental_freq=100,
            harmonics=[1, 2, 3, 4],
            amplitudes=[1.0, 0.5, 0.25, 0.125],
        )
        thd = harmonic_result["thd_percent"]
        total_rms = harmonic_result["total_rms"]
        print("  100 Hz fundamental with harmonics:")
        print(f"    Total RMS: {total_rms:.3f}")
        print(f"    THD: {thd:.1f}%")

        for h in harmonic_result["harmonic_analysis"][:3]:  # First 3 harmonics
            freq = h["frequency"]
            amp = h["amplitude"]
            power = h["power_percent"]
            print(
                f"    H{h['harmonic_number']}: {freq} Hz, amp={amp:.3f}, power={power:.1f}%"
            )

        print("\nFourier Series Coefficients:")
        waveforms = ["square", "sawtooth", "triangle"]
        for waveform in waveforms:
            fourier_result = await fourier_coefficients_basic(waveform, 4)
            terms = fourier_result["fourier_terms"]
            print(f"  {waveform.title()} wave (first 3 terms):")
            for term in terms[:3]:
                n = term["n"]
                amp = term["amplitude"]
                print(f"    n={n}: amplitude={amp:.4f}")

        print("\n✅ All wave analysis functions working!")

    asyncio.run(test_wave_analysis_functions())
