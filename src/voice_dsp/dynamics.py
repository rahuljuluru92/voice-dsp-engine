"""Stage 4: noise gate and limiter, both streaming (state carried across
process() calls) so they can run inside a real-time block callback."""

from __future__ import annotations

import numpy as np


def _time_constant_coef(time_s: float, sr: float) -> float:
    if time_s <= 0:
        return 0.0
    return float(np.exp(-1.0 / (time_s * sr)))


class NoiseGate:
    """Attenuates the signal toward silence when its envelope falls below
    `threshold_db`, with separate attack/release smoothing on both the
    envelope follower and the applied gain to avoid audible clicks.
    """

    def __init__(
        self,
        threshold_db: float = -50.0,
        attack_s: float = 0.005,
        release_s: float = 0.15,
        sr: float = 48000,
    ):
        self.threshold_lin = 10 ** (threshold_db / 20)
        self.attack_coef = _time_constant_coef(attack_s, sr)
        self.release_coef = _time_constant_coef(release_s, sr)
        self.envelope = 0.0
        self.gain = 0.0

    def process(self, x: np.ndarray) -> np.ndarray:
        y = np.empty_like(x)
        env = self.envelope
        gain = self.gain
        for i in range(len(x)):
            rectified = abs(x[i])
            coef = self.attack_coef if rectified > env else self.release_coef
            env = coef * env + (1 - coef) * rectified

            target_gain = 1.0 if env >= self.threshold_lin else 0.0
            gain_coef = self.attack_coef if target_gain > gain else self.release_coef
            gain = gain_coef * gain + (1 - gain_coef) * target_gain

            y[i] = x[i] * gain
        self.envelope = env
        self.gain = gain
        return y

    def reset(self) -> None:
        self.envelope = 0.0
        self.gain = 0.0


class Limiter:
    """Peak limiter: smoothed gain reduction keeps the signal at or below
    `threshold`, followed by an unconditional hard clip to [-threshold,
    threshold] as a guarantee of bounded, finite output regardless of
    how extreme the input or how the smoothing responds to it.
    """

    def __init__(
        self,
        threshold: float = 0.98,
        attack_s: float = 0.001,
        release_s: float = 0.05,
        sr: float = 48000,
    ):
        self.threshold = threshold
        self.attack_coef = _time_constant_coef(attack_s, sr)
        self.release_coef = _time_constant_coef(release_s, sr)
        self.gain = 1.0

    def process(self, x: np.ndarray) -> np.ndarray:
        x = np.nan_to_num(x, nan=0.0, posinf=self.threshold, neginf=-self.threshold)
        y = np.empty_like(x)
        gain = self.gain
        for i in range(len(x)):
            peak = abs(x[i])
            desired_gain = 1.0 if peak <= self.threshold else self.threshold / peak
            coef = self.attack_coef if desired_gain < gain else self.release_coef
            gain = coef * gain + (1 - coef) * desired_gain
            y[i] = x[i] * gain
        self.gain = gain
        return np.clip(y, -self.threshold, self.threshold)

    def reset(self) -> None:
        self.gain = 1.0
