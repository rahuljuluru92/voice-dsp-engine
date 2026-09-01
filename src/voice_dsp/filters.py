"""Stage 4: biquad high-pass filter (RBJ Audio EQ Cookbook design)."""

from __future__ import annotations

import numpy as np
from scipy.signal import freqz, lfilter


def design_highpass(cutoff_hz: float, sr: float, q: float = 0.707) -> tuple[np.ndarray, np.ndarray]:
    """RBJ cookbook biquad high-pass coefficients, normalized so a0 == 1."""
    w0 = 2 * np.pi * cutoff_hz / sr
    cos_w0 = np.cos(w0)
    sin_w0 = np.sin(w0)
    alpha = sin_w0 / (2 * q)

    b0 = (1 + cos_w0) / 2
    b1 = -(1 + cos_w0)
    b2 = (1 + cos_w0) / 2
    a0 = 1 + alpha
    a1 = -2 * cos_w0
    a2 = 1 - alpha

    b = np.array([b0, b1, b2]) / a0
    a = np.array([1.0, a1 / a0, a2 / a0])
    return b, a


class BiquadHighpass:
    """Streaming biquad high-pass filter that keeps state across process() calls."""

    def __init__(self, cutoff_hz: float, sr: float, q: float = 0.707):
        self.cutoff_hz = cutoff_hz
        self.sr = sr
        self.q = q
        self.b, self.a = design_highpass(cutoff_hz, sr, q)
        self.zi = np.zeros(2)

    def process(self, x: np.ndarray) -> np.ndarray:
        y, self.zi = lfilter(self.b, self.a, x, zi=self.zi)
        return y

    def reset(self) -> None:
        self.zi = np.zeros(2)

    def frequency_response_db(self, freqs_hz: np.ndarray) -> np.ndarray:
        w = 2 * np.pi * np.asarray(freqs_hz) / self.sr
        _, h = freqz(self.b, self.a, worN=w)
        return 20 * np.log10(np.abs(h) + 1e-20)
