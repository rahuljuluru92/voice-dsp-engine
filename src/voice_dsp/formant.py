"""Stage 3: independent formant control via cepstral-envelope warping.

The spectral envelope (formants) and the excitation (pitch harmonics)
are separated using real-cepstrum liftering: the log-magnitude
spectrum's cepstrum concentrates slowly-varying envelope information
at low quefrency and pitch-periodicity information at high quefrency,
so a low-quefrency lifter recovers a smooth envelope estimate.

Formant shifting warps that envelope along the frequency axis and
reapplies it to the (unwarped) excitation, moving formant positions
without touching pitch. Pitch shifting (Stage 2) can optionally be
composed with an inverse formant warp to cancel out the frequency-axis
scaling that phase-vocoder-plus-resample pitch shifting otherwise
imposes on the envelope, so pitch and formants stay independently
controllable.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks

from .pitch import phase_vocoder_stretch, semitones_to_ratio
from .stft import istft, periodic_hann, stft

EPS = 1e-10


def harmonicity_confidence(
    magnitude: np.ndarray,
    min_peaks: int = 3,
    height_frac: float = 0.1,
    min_separation_bins: int = 5,
) -> float:
    """Confidence (0..1) that `magnitude` has enough distinct spectral
    peaks for cepstral envelope/excitation separation to be meaningful.

    The technique assumes a harmonic excitation source (many partials);
    on a signal with only one dominant spectral component (e.g. a pure
    sine), the "envelope" it extracts is really just a smoothed version
    of that single peak, and warping+reapplying it can misplace the
    peak entirely rather than correctly relocating a real envelope over
    real harmonic content (see DECISIONS.md, "Stage 6 benchmark
    debugging" and "Considered but out of scope" entries -- this
    resolves that limitation rather than just documenting it).

    `min_separation_bins` requires counted peaks to be at least that
    many FFT bins apart (default 5 bins, ~117Hz at the project's
    default frame_size=2048/sr=48000). This matters: a single tone that
    has gone through phase-vocoder stretch + resample can pick up
    spectral leakage sidelobes that `find_peaks` would otherwise count
    as extra "harmonics" a few bins from the true peak -- confirmed
    directly (a resampled 220Hz->+12st pure tone showed 3 "peaks" only
    ~4 bins apart before this constraint, 1 after). Real voice harmonics
    are spaced by the fundamental period (at least ~80Hz for adult
    speech, typically much more), well beyond this distance, so the
    constraint doesn't affect real multi-partial content.

    Returns 0.0 for a single dominant peak, ramping linearly up to 1.0
    once there are at least `min_peaks` significant, separated peaks
    (typical of real voice). The linear ramp (rather than a hard on/off
    threshold) matters for real audio: an abrupt full-strength/no
    -strength switch between frames would itself be a new discontinuity
    of exactly the kind this whole project has spent so much effort
    eliminating elsewhere.
    """
    if min_peaks <= 1:
        return 1.0
    peak_max = magnitude.max()
    if peak_max <= 0:
        return 1.0
    peaks, _ = find_peaks(
        magnitude, height=peak_max * height_frac, distance=max(1, min_separation_bins)
    )
    n_peaks = len(peaks)
    return float(np.clip((n_peaks - 1) / (min_peaks - 1), 0.0, 1.0))


def cepstral_envelope(magnitude: np.ndarray, cutoff_quefrency: int) -> np.ndarray:
    """Smooth log-magnitude spectral envelope via cepstral liftering.

    `magnitude` is one STFT frame's magnitude spectrum, shape
    (n_bins,) with n_bins = frame_size // 2 + 1. Returns the envelope
    magnitude, same shape.
    """
    n_bins = magnitude.shape[-1]
    frame_size = (n_bins - 1) * 2
    log_mag = np.log(magnitude + EPS)
    cepstrum = np.fft.irfft(log_mag, n=frame_size)
    liftered = cepstrum.copy()
    liftered[cutoff_quefrency:frame_size - cutoff_quefrency] = 0.0
    envelope_log_mag = np.fft.rfft(liftered).real
    return np.exp(envelope_log_mag)


def warp_envelope(envelope: np.ndarray, warp_ratio: float) -> np.ndarray:
    """Scale `envelope` along the frequency axis by `warp_ratio`.

    warp_ratio > 1 pushes formants up in frequency (e.g. smaller vocal
    tract); < 1 pushes them down. Implemented as envelope_new(f) =
    envelope_old(f / warp_ratio) via linear interpolation on bin index,
    holding the edge value beyond the available range.
    """
    if warp_ratio <= 0:
        raise ValueError("warp_ratio must be positive")
    n_bins = len(envelope)
    bins = np.arange(n_bins)
    src_bins = bins / warp_ratio
    src_bins = np.clip(src_bins, 0, n_bins - 1)
    return np.interp(src_bins, bins, envelope)


def shift_formants(
    x: np.ndarray,
    formant_semitones: float,
    frame_size: int = 2048,
    hop_size: int = 512,
    cutoff_quefrency: int | None = None,
) -> np.ndarray:
    """Shift the formant envelope of `x` by `formant_semitones` while
    leaving the fundamental frequency / harmonic positions unchanged.
    """
    if cutoff_quefrency is None:
        cutoff_quefrency = frame_size // 16

    warp_ratio = semitones_to_ratio(formant_semitones)
    window = periodic_hann(frame_size)
    pad = frame_size
    xp = np.concatenate([np.zeros(pad), x, np.zeros(pad)])
    spec = stft(xp, frame_size, hop_size, window)

    out_spec = np.empty_like(spec)
    for i in range(spec.shape[0]):
        mag = np.abs(spec[i])
        phase = np.angle(spec[i])
        envelope = cepstral_envelope(mag, cutoff_quefrency)
        excitation = mag / (envelope + EPS)
        new_envelope = warp_envelope(envelope, warp_ratio)
        confidence = harmonicity_confidence(mag)
        effective_envelope = envelope + confidence * (new_envelope - envelope)
        new_mag = excitation * effective_envelope
        out_spec[i] = new_mag * np.exp(1j * phase)

    y = istft(out_spec, hop_size, window, length=len(xp))
    return y[pad:pad + len(x)]


def pitch_shift_formant_preserving(
    x: np.ndarray,
    pitch_semitones: float,
    frame_size: int = 2048,
    analysis_hop: int = 512,
    cutoff_quefrency: int | None = None,
) -> np.ndarray:
    """Pitch-shift `x` while correcting the formant envelope back to its
    original position, so pitch changes independently of formants.
    """
    if pitch_semitones == 0:
        return np.array(x, dtype=np.float64, copy=True)

    ratio = semitones_to_ratio(pitch_semitones)
    stretched = phase_vocoder_stretch(x, ratio, frame_size, analysis_hop)

    n_out = int(round(len(stretched) / ratio))
    src_positions = np.clip(np.arange(n_out) * ratio, 0, len(stretched) - 1)
    resampled = np.interp(src_positions, np.arange(len(stretched)), stretched)
    resampled = resampled[: len(x)]
    if len(resampled) < len(x):
        resampled = np.concatenate([resampled, np.zeros(len(x) - len(resampled))])

    # Resampling scales every frequency (including formants) by `ratio`.
    # Warping the envelope by 1/ratio cancels that out.
    return shift_formants(
        resampled,
        formant_semitones=-pitch_semitones,
        frame_size=frame_size,
        hop_size=analysis_hop,
        cutoff_quefrency=cutoff_quefrency,
    )


def process(
    x: np.ndarray,
    pitch_semitones: float = 0.0,
    formant_semitones: float = 0.0,
    frame_size: int = 2048,
    analysis_hop: int = 512,
    cutoff_quefrency: int | None = None,
) -> np.ndarray:
    """Apply independent pitch and formant shifts to `x`."""
    y = pitch_shift_formant_preserving(
        x, pitch_semitones, frame_size, analysis_hop, cutoff_quefrency
    )
    if formant_semitones != 0:
        y = shift_formants(y, formant_semitones, frame_size, analysis_hop, cutoff_quefrency)
    return y
