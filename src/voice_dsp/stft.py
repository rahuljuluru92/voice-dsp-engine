"""Stage 2: STFT framing/windowing, inverse STFT, and overlap-add.

Uses a periodic (DFT-even) Hann window, which is the variant intended
for overlap-add resynthesis (as opposed to the symmetric Hann window
used for spectral analysis/display). With hop = frame_size / 4 this
window satisfies the constant-overlap-add condition for weighted OLA,
which is what makes near-exact STFT round-trip reconstruction possible.
"""

from __future__ import annotations

import numpy as np


def periodic_hann(frame_size: int) -> np.ndarray:
    n = np.arange(frame_size)
    return 0.5 - 0.5 * np.cos(2 * np.pi * n / frame_size)


def stft(x: np.ndarray, frame_size: int, hop_size: int, window: np.ndarray) -> np.ndarray:
    """Framed, windowed STFT of a 1-D real signal.

    Returns a complex array of shape (n_frames, frame_size // 2 + 1).
    `x` must already be long enough for at least one full frame; callers
    that need edge padding should pad before calling.
    """
    n = len(x)
    if n < frame_size:
        raise ValueError("signal shorter than frame_size; pad before calling stft()")
    n_frames = 1 + (n - frame_size) // hop_size
    frames = np.empty((n_frames, frame_size), dtype=np.float64)
    for i in range(n_frames):
        start = i * hop_size
        frames[i] = x[start:start + frame_size] * window
    return np.fft.rfft(frames, axis=1)


def istft(spec: np.ndarray, hop_size: int, window: np.ndarray, length: int) -> np.ndarray:
    """Weighted overlap-add inverse STFT.

    Applies the synthesis window again and normalizes by the running sum
    of squared window values, which is the standard WOLA reconstruction
    formula and is robust to any window/hop combination (not just exact
    COLA ones), degrading gracefully rather than blowing up.
    """
    n_frames, n_bins = spec.shape
    frame_size = (n_bins - 1) * 2
    out = np.zeros(length, dtype=np.float64)
    win_sum = np.zeros(length, dtype=np.float64)
    win_sq = window ** 2

    for i in range(n_frames):
        frame = np.fft.irfft(spec[i], n=frame_size)
        frame = frame * window
        start = i * hop_size
        end = start + frame_size
        if end > length:
            frame = frame[: length - start]
            end = length
        out[start:end] += frame
        win_sum[start:end] += win_sq[: end - start]

    nonzero = win_sum > 1e-8
    out[nonzero] /= win_sum[nonzero]
    return out


def num_frames(padded_length: int, frame_size: int, hop_size: int) -> int:
    return 1 + (padded_length - frame_size) // hop_size


def roundtrip(x: np.ndarray, frame_size: int = 2048, hop_size: int = 512) -> np.ndarray:
    """Analyze then resynthesize `x`, returning a signal the same length as `x`.

    Pads by a full frame on each side so every real sample gets full
    window overlap, then crops the padding back off after synthesis.
    """
    window = periodic_hann(frame_size)
    pad = frame_size
    xp = np.concatenate([np.zeros(pad), x, np.zeros(pad)])
    spec = stft(xp, frame_size, hop_size, window)
    y = istft(spec, hop_size, window, length=len(xp))
    return y[pad:pad + len(x)]
