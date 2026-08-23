"""Stage 2: phase-vocoder pitch shifting with true instantaneous-frequency
phase reconstruction (not naive waveform resampling).

Approach (the standard phase-vocoder pitch shifter):
  1. Time-stretch the signal by the pitch ratio using a phase vocoder
     whose synthesis phase is advanced by the bin's *measured*
     instantaneous frequency (analysis phase advance corrected by the
     wrapped phase deviation from the expected bin frequency), not by
     just re-using the analysis phase or the nominal bin frequency.
  2. Resample the time-stretched signal by 1/ratio, which restores the
     original duration and shifts the pitch by `ratio` because
     resampling changes playback speed.
"""

from __future__ import annotations

import numpy as np

from .stft import istft, periodic_hann, stft

TWO_PI = 2 * np.pi


def semitones_to_ratio(semitones: float) -> float:
    return 2.0 ** (semitones / 12.0)


def _wrap_phase(phase: np.ndarray) -> np.ndarray:
    return np.mod(phase + np.pi, TWO_PI) - np.pi


def phase_vocoder_stretch(
    x: np.ndarray,
    stretch_ratio: float,
    frame_size: int = 2048,
    analysis_hop: int = 512,
) -> np.ndarray:
    """Time-stretch `x` by `stretch_ratio` (output is stretch_ratio times
    longer) while preserving pitch, via true instantaneous-frequency
    phase-vocoder reconstruction.
    """
    if stretch_ratio <= 0:
        raise ValueError("stretch_ratio must be positive")

    synthesis_hop = max(1, int(round(analysis_hop * stretch_ratio)))
    window = periodic_hann(frame_size)

    pad = frame_size // 2
    xp = np.concatenate([np.zeros(pad), x, np.zeros(pad + frame_size)])
    spec = stft(xp, frame_size, analysis_hop, window)
    n_frames, n_bins = spec.shape

    mag = np.abs(spec)
    phase = np.angle(spec)

    # Expected phase advance per analysis hop for each bin, under the
    # assumption the bin's true frequency equals its center frequency.
    bin_index = np.arange(n_bins)
    expected_advance = TWO_PI * bin_index * analysis_hop / frame_size

    out_spec = np.empty_like(spec)
    synth_phase = phase[0].copy()
    out_spec[0] = mag[0] * np.exp(1j * synth_phase)

    for i in range(1, n_frames):
        # Measured phase advance vs. expected, wrapped to (-pi, pi] to
        # recover the true instantaneous frequency deviation per bin.
        delta = phase[i] - phase[i - 1] - expected_advance
        delta = _wrap_phase(delta)
        true_freq_per_sample = (expected_advance + delta) / analysis_hop
        synth_phase = synth_phase + true_freq_per_sample * synthesis_hop
        out_spec[i] = mag[i] * np.exp(1j * synth_phase)

    out_length = frame_size + (n_frames - 1) * synthesis_hop
    y = istft(out_spec, synthesis_hop, window, length=out_length)

    out_pad = int(round(pad * stretch_ratio))
    target_len = int(round(len(x) * stretch_ratio))
    y = y[out_pad:out_pad + target_len]
    if len(y) < target_len:
        y = np.concatenate([y, np.zeros(target_len - len(y))])
    return y


def _resample_linear(x: np.ndarray, out_length: int, rate: float) -> np.ndarray:
    """Resample `x` by reading it at `rate`x speed, producing `out_length`
    samples via linear interpolation.
    """
    src_positions = np.arange(out_length) * rate
    src_positions = np.clip(src_positions, 0, len(x) - 1)
    return np.interp(src_positions, np.arange(len(x)), x)


def pitch_shift(
    x: np.ndarray,
    semitones: float,
    frame_size: int = 2048,
    analysis_hop: int = 512,
) -> np.ndarray:
    """Shift the pitch of `x` by `semitones`, preserving duration.

    Positive semitones raise pitch, negative lower it. Formant/spectral
    envelope is *not* corrected here by design — that independent
    control is added in Stage 3.
    """
    if semitones == 0:
        return np.array(x, dtype=np.float64, copy=True)

    ratio = semitones_to_ratio(semitones)
    stretched = phase_vocoder_stretch(x, ratio, frame_size, analysis_hop)
    out_length = len(x)
    return _resample_linear(stretched, out_length, ratio)
