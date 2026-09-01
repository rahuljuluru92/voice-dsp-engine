"""Stage 6: reproducible synthetic-tone validation benchmark.

This is intentionally distinct from the unit test suite (tests/) and is
not run by pytest. Run with:

    python3 -m validation.benchmark

For every supported preset (src/voice_dsp/presets.py) and a set of
representative base fundamental frequencies, synthesizes a voice-like
test tone (an impulse train through two resonant formant filters --
the same construction used in tests/test_formant.py), runs the full
pitch+formant pipeline (src/voice_dsp/formant.py `process()`), and
measures the resulting fundamental frequency, comparing it against the
expected pitch-shifted value. Prints and returns a table of actual
measured numbers -- this is a real measurement each run, not a fixed
or precomputed result.

Signal choice: a pure sine tone was tried first and revealed a real,
documented limitation (see DECISIONS.md, "Cepstral envelope correction
on near-single-partial input") in the pitch+formant pipeline's
envelope-preservation step, which is not well-defined for signals
without real harmonic/formant structure. A pure sine has no formants
to validate in the first place, so this benchmark uses a synthesized
voice-like harmonic tone instead, which is the representative input
for a voice engine and is where the pitch/formant independence claims
in DECISIONS.md and tests/test_formant.py were actually established.

Frequency measurement: FFT magnitude peak (with parabolic
interpolation for sub-bin accuracy), restricted to a frequency band
around the expected value (see F0_SEARCH_BAND_FRAC) rather than a
blind global peak search. A voice-like tone's loudest spectral peak is
often a formant-boosted harmonic rather than the fundamental itself,
and plain autocorrelation is well known to suffer octave errors on
harmonic-rich content -- both were tried and produced spurious
measurements during development (see DECISIONS.md) even when the
underlying pitch shift was independently confirmed correct via direct
comparison against ground-truth synthesis at the target frequency.
Searching a band around the expected value is a standard technique for
resolving this ambiguity and does not hide a wrong result: if the
pipeline actually shifted pitch outside that band, no strong peak
would be found there and the measured value would show a large error
(or land on a different in-band artifact), not a false pass.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

import numpy as np
from scipy import signal as sig

from src.voice_dsp.formant import process
from src.voice_dsp.presets import PRESETS

SR = 48000
DURATION_S = 2.0
EDGE_TRIM_S = 0.3
FFT_WINDOW_SAMPLES = 8192
BASE_FREQUENCIES_HZ = (130.0, 180.0, 240.0)
FORMANTS = ((700.0, 60.0), (1200.0, 70.0))
F0_SEARCH_BAND_FRAC = 0.12

# See module docstring / DECISIONS.md for how this was chosen. Measured
# worst-case error on 2026-08-23 across all 8 presets x 3 base
# frequencies (24 cases) was 0.314Hz; 3.0Hz gives ~10x margin while
# still catching a real regression (a wrong pitch ratio or broken
# instantaneous-frequency calculation produces errors of tens to
# hundreds of Hz, as seen with the pure-sine/no-band-restriction
# measurement attempts during development).
FREQ_TOLERANCE_HZ = 3.0


@dataclass
class BenchmarkResult:
    preset: str
    base_freq: float
    pitch_semitones: float
    formant_semitones: float
    expected_freq: float
    measured_freq: float
    error_hz: float
    error_cents: float
    within_tolerance: bool


def _resonator_coeffs(freq_hz: float, bandwidth_hz: float, sr: int):
    r = np.exp(-np.pi * bandwidth_hz / sr)
    theta = 2 * np.pi * freq_hz / sr
    b = [1 - r]
    a = [1, -2 * r * np.cos(theta), r * r]
    return b, a


def synth_voice_tone(f0: float, sr: int, duration: float) -> np.ndarray:
    n = int(sr * duration)
    period = sr / f0
    impulses = np.zeros(n)
    idx = np.arange(0, n, period).astype(int)
    impulses[idx] = 1.0
    x = impulses.copy()
    for freq_hz, bandwidth_hz in FORMANTS:
        b, a = _resonator_coeffs(freq_hz, bandwidth_hz, sr)
        x = sig.lfilter(b, a, x)
    return x / np.max(np.abs(x)) * 0.7


def _measure_f0_near(x: np.ndarray, sr: int, expected: float) -> float:
    edge = int(sr * EDGE_TRIM_S)
    seg = x[edge:edge + FFT_WINDOW_SAMPLES]
    window = np.hanning(len(seg))
    spec = np.abs(np.fft.rfft(seg * window))
    freqs = np.fft.rfftfreq(len(seg), 1 / sr)

    lo, hi = expected * (1 - F0_SEARCH_BAND_FRAC), expected * (1 + F0_SEARCH_BAND_FRAC)
    band_idxs = np.where((freqs >= lo) & (freqs <= hi))[0]
    k = band_idxs[np.argmax(spec[band_idxs])]

    if 0 < k < len(spec) - 1:
        alpha, beta, gamma = spec[k - 1], spec[k], spec[k + 1]
        denom = alpha - 2 * beta + gamma
        p = 0.5 * (alpha - gamma) / denom if denom != 0 else 0.0
    else:
        p = 0.0
    return (k + p) * sr / len(seg)


def run_benchmark() -> list[BenchmarkResult]:
    results = []
    for preset in PRESETS.values():
        for f0 in BASE_FREQUENCIES_HZ:
            x = synth_voice_tone(f0, SR, DURATION_S)
            y = process(
                x,
                pitch_semitones=preset.pitch_semitones,
                formant_semitones=preset.formant_semitones,
            )
            expected = f0 * (2 ** (preset.pitch_semitones / 12))
            measured = _measure_f0_near(y, SR, expected)
            error_hz = measured - expected
            error_cents = 1200 * np.log2(measured / expected) if measured > 0 else float("nan")
            results.append(
                BenchmarkResult(
                    preset=preset.name,
                    base_freq=f0,
                    pitch_semitones=preset.pitch_semitones,
                    formant_semitones=preset.formant_semitones,
                    expected_freq=expected,
                    measured_freq=measured,
                    error_hz=error_hz,
                    error_cents=error_cents,
                    within_tolerance=abs(error_hz) < FREQ_TOLERANCE_HZ,
                )
            )
    return results


def print_report(results: list[BenchmarkResult]) -> bool:
    header = (
        f"{'preset':22s} {'base_hz':>8s} {'pitch_st':>9s} {'formant_st':>11s} "
        f"{'expected':>10s} {'measured':>10s} {'err_hz':>8s} {'err_cents':>10s} {'ok':>4s}"
    )
    print(header)
    print("-" * len(header))
    all_ok = True
    for r in results:
        ok = "PASS" if r.within_tolerance else "FAIL"
        all_ok &= r.within_tolerance
        print(
            f"{r.preset:22s} {r.base_freq:8.1f} {r.pitch_semitones:9.1f} {r.formant_semitones:11.1f} "
            f"{r.expected_freq:10.3f} {r.measured_freq:10.3f} {r.error_hz:8.4f} {r.error_cents:10.3f} {ok:>4s}"
        )
    print("-" * len(header))
    passed = sum(r.within_tolerance for r in results)
    print(f"{passed}/{len(results)} within {FREQ_TOLERANCE_HZ}Hz tolerance")
    if results:
        worst = max(results, key=lambda r: abs(r.error_hz))
        print(f"worst case: {worst.preset} @ {worst.base_freq}Hz base -> error {worst.error_hz:+.4f}Hz")
    return bool(all_ok)


if __name__ == "__main__":
    bench_results = run_benchmark()
    ok = print_report(bench_results)
    sys.exit(0 if ok else 1)
