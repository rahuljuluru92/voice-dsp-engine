"""Continuous-phase streaming pitch/formant processor.

Unlike `formant.process()` (Stage 2/3, batch: analyzes a whole array at
once, restarting phase from frame 0 every call), this processes audio
incrementally as it arrives and never resets its internal state -- the
per-bin synthesis-phase accumulator, the previous-frame analysis phase,
and the overlap-add buffer all persist for the lifetime of the
instance. That continuity is what the chunked `VoiceEngine` architecture
could not provide even with the overlap-window crossfade (see
DECISIONS.md): crossfading smooths the amplitude at a chunk boundary,
but two independently-vocoded chunks are not guaranteed to agree in
phase. Here there are no chunk boundaries in the phase-vocoder math at
all -- only in how the caller happens to hand it audio.

Pitch and formant math: same algorithm as `pitch.py`/`formant.py`
(phase-vocoder time-stretch with true instantaneous-frequency phase
reconstruction, then resample to restore duration; cepstral-envelope
warping for formants), restructured to run one analysis hop at a time
instead of over a whole pre-loaded array, and with the formant warp
applied directly to the pre-resample envelope (mathematically
equivalent to the offline two-pass stretch-then-correct design -- see
DECISIONS.md for the derivation) so the whole pipeline is one
continuous pass instead of two independent STFT passes.
"""

from __future__ import annotations

import numpy as np

from .formant import cepstral_envelope, warp_envelope
from .stft import periodic_hann

TWO_PI = 2 * np.pi
EPS = 1e-10


def semitones_to_ratio(semitones: float) -> float:
    return 2.0 ** (semitones / 12.0)


def _wrap_phase(phase: np.ndarray) -> np.ndarray:
    return np.mod(phase + np.pi, TWO_PI) - np.pi


class _StreamPhaseVocoder:
    """Continuous phase-vocoder time-stretch + cepstral envelope warp,
    fed one analysis hop at a time. Never resets phase/OLA state.
    """

    def __init__(
        self,
        frame_size: int,
        analysis_hop: int,
        synthesis_hop: int,
        envelope_warp_ratio: float,
        cutoff_quefrency: int | None = None,
    ):
        self.frame_size = frame_size
        self.analysis_hop = analysis_hop
        self.synthesis_hop = synthesis_hop
        self.envelope_warp_ratio = envelope_warp_ratio
        self.cutoff_quefrency = cutoff_quefrency if cutoff_quefrency is not None else frame_size // 16

        self.window = periodic_hann(frame_size)
        n_bins = frame_size // 2 + 1
        bin_index = np.arange(n_bins)
        self.expected_advance = TWO_PI * bin_index * analysis_hop / frame_size

        self._input_hist = np.zeros(frame_size)
        self._pending = np.zeros(0)
        self._prev_phase: np.ndarray | None = None
        self._synth_phase: np.ndarray | None = None
        self._ola = np.zeros(frame_size)
        self._ola_winsum = np.zeros(frame_size)
        self._prime()

    def _prime(self) -> None:
        """Warm up the overlap-add window-sum before any real output is
        released, equivalent to the offline algorithm's zero-padding.

        Without this, the first ~frame_size samples of output are
        normalized by an artificially small accumulated window-sum
        (only the first frame or two have contributed so far, not the
        full overlap a steady-state position gets), causing a large
        amplitude spike. Not fixable with a single precomputed
        normalization constant either: for hop values that aren't an
        exact divisor of frame_size (e.g. synthesis_hop == frame_size/2
        at a +12 semitone shift), the true steady-state window-sum
        genuinely varies by position, it just isn't small. Priming with
        silent frames (discarding their output, keeping only their
        window-sum contribution) reproduces the same window-sum pattern
        a real interior position sees, before real audio is released.
        """
        n_priming = max(0, -(-self.frame_size // self.synthesis_hop) - 1)
        for _ in range(n_priming):
            self._process_frame()

    def feed(self, new_samples: np.ndarray) -> np.ndarray:
        self._pending = np.concatenate([self._pending, new_samples])
        out_frames = []
        while len(self._pending) >= self.analysis_hop:
            hop_samples = self._pending[: self.analysis_hop]
            self._pending = self._pending[self.analysis_hop:]
            self._input_hist = np.concatenate([self._input_hist[self.analysis_hop:], hop_samples])
            out_frames.append(self._process_frame())
        if not out_frames:
            return np.zeros(0)
        return np.concatenate(out_frames)

    def _process_frame(self) -> np.ndarray:
        windowed = self._input_hist * self.window
        spec = np.fft.rfft(windowed)
        mag = np.abs(spec)
        phase = np.angle(spec)

        if self._prev_phase is None:
            self._synth_phase = phase.copy()
        else:
            delta = phase - self._prev_phase - self.expected_advance
            delta = _wrap_phase(delta)
            true_freq_per_sample = (self.expected_advance + delta) / self.analysis_hop
            self._synth_phase = self._synth_phase + true_freq_per_sample * self.synthesis_hop
        self._prev_phase = phase

        if self.envelope_warp_ratio != 1.0:
            envelope = cepstral_envelope(mag, self.cutoff_quefrency)
            excitation = mag / (envelope + EPS)
            new_envelope = warp_envelope(envelope, self.envelope_warp_ratio)
            mag = excitation * new_envelope

        out_spec = mag * np.exp(1j * self._synth_phase)
        frame_time = np.fft.irfft(out_spec, n=self.frame_size) * self.window

        self._ola[: self.frame_size] += frame_time
        self._ola_winsum[: self.frame_size] += self.window ** 2

        hop = self.synthesis_hop
        release = self._ola[:hop].copy()
        release_winsum = self._ola_winsum[:hop]
        nonzero = release_winsum > 1e-8
        release[nonzero] /= release_winsum[nonzero]

        self._ola = np.concatenate([self._ola[hop:], np.zeros(hop)])
        self._ola_winsum = np.concatenate([self._ola_winsum[hop:], np.zeros(hop)])

        return release


class _StreamResampler:
    """Continuous linear-interpolation resampler: reads its input stream
    at `rate`x speed. rate > 1 speeds up (raises pitch), < 1 slows down.
    """

    def __init__(self, rate: float):
        self.rate = rate
        self._buffer = np.zeros(0)
        self._pos = 0.0

    def feed(self, new_samples: np.ndarray) -> np.ndarray:
        if self.rate == 1.0:
            return new_samples
        self._buffer = np.concatenate([self._buffer, new_samples])
        out = []
        while True:
            i0 = int(np.floor(self._pos))
            if i0 + 1 >= len(self._buffer):
                break
            frac = self._pos - i0
            out.append(self._buffer[i0] * (1 - frac) + self._buffer[i0 + 1] * frac)
            self._pos += self.rate

        consumed = int(np.floor(self._pos))
        if consumed > 0:
            self._buffer = self._buffer[consumed:]
            self._pos -= consumed

        return np.array(out) if out else np.zeros(0)


class StreamingVoiceProcessor:
    """Continuous pitch + formant processor: feed it audio incrementally
    (any block size, called repeatedly), get back however many output
    samples are ready each call. No independent-chunk restarts.
    """

    def __init__(
        self,
        sr: int = 48000,
        frame_size: int = 2048,
        analysis_hop: int = 512,
        pitch_semitones: float = 0.0,
        formant_semitones: float = 0.0,
    ):
        self.sr = sr
        self.frame_size = frame_size
        self.analysis_hop = analysis_hop
        self.pitch_semitones = pitch_semitones
        self.formant_semitones = formant_semitones

        stretch_ratio = semitones_to_ratio(pitch_semitones)
        formant_ratio = semitones_to_ratio(formant_semitones)
        synthesis_hop = max(1, int(round(analysis_hop * stretch_ratio)))
        # Applying formant_ratio / stretch_ratio to the pre-resample
        # envelope is mathematically equivalent to the offline
        # stretch -> resample -> inverse-formant-warp -> explicit-formant-warp
        # pipeline (formant.py), since resampling later scales every
        # frequency -- including this pre-warped envelope -- by
        # stretch_ratio. See DECISIONS.md for the derivation.
        envelope_warp_ratio = formant_ratio / stretch_ratio

        self._pv = _StreamPhaseVocoder(
            frame_size, analysis_hop, synthesis_hop, envelope_warp_ratio
        )
        self._resampler = _StreamResampler(stretch_ratio)

    def process(self, new_samples: np.ndarray) -> np.ndarray:
        stretched = self._pv.feed(new_samples)
        if len(stretched) == 0:
            return np.zeros(0)
        return self._resampler.feed(stretched)
