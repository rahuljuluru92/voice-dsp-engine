"""Stage 4: the runtime signal-processing chain — high-pass filter, then
noise gate, then limiter — applied in that order to each audio block.
"""

from __future__ import annotations

import numpy as np

from .dynamics import Limiter, NoiseGate
from .filters import BiquadHighpass


class SignalChain:
    def __init__(
        self,
        sr: float = 48000,
        hpf_cutoff_hz: float = 80.0,
        hpf_q: float = 0.707,
        gate_threshold_db: float = -50.0,
        gate_attack_s: float = 0.005,
        gate_release_s: float = 0.15,
        limiter_threshold: float = 0.98,
        limiter_attack_s: float = 0.001,
        limiter_release_s: float = 0.05,
    ):
        self.hpf = BiquadHighpass(hpf_cutoff_hz, sr, hpf_q)
        self.gate = NoiseGate(gate_threshold_db, gate_attack_s, gate_release_s, sr)
        self.limiter = Limiter(limiter_threshold, limiter_attack_s, limiter_release_s, sr)

    def process(self, x: np.ndarray) -> np.ndarray:
        y = self.hpf.process(x)
        y = self.gate.process(y)
        y = self.limiter.process(y)
        return y

    def reset(self) -> None:
        self.hpf.reset()
        self.gate.reset()
        self.limiter.reset()
