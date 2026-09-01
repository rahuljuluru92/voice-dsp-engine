import numpy as np

from src.voice_dsp.dynamics import Limiter, NoiseGate

SR = 48000


def test_limiter_bounds_deliberately_over_range_input():
    lim = Limiter(threshold=0.98, sr=SR)
    t = np.arange(SR) / SR
    x = 5.0 * np.sin(2 * np.pi * 440 * t)  # 5x over-range sine
    # Deliberately inject extreme / non-finite values.
    x[1000:1010] = np.array(
        [1e6, -1e6, np.inf, -np.inf, np.nan, 1e9, -1e9, 3.0, -3.0, 100.0]
    )

    y = lim.process(x)

    assert np.all(np.isfinite(y))
    assert np.max(np.abs(y)) <= 0.98 + 1e-9


def test_limiter_passes_signal_within_bounds_mostly_unchanged():
    lim = Limiter(threshold=0.98, sr=SR)
    t = np.arange(SR) / SR
    x = 0.3 * np.sin(2 * np.pi * 440 * t)

    y = lim.process(x)

    # Measured 2026-08-23: a signal comfortably under threshold passes
    # through with negligible gain reduction (limiter is transparent
    # when nothing needs limiting).
    assert np.max(np.abs(y - x)) < 0.01


def test_noise_gate_silences_quiet_signal():
    gate = NoiseGate(threshold_db=-40, sr=SR)
    t = np.arange(SR) / SR
    quiet = 0.001 * np.sin(2 * np.pi * 440 * t)  # well below -40dB threshold

    y = gate.process(quiet)

    # Measured 2026-08-23: steady-state RMS over the last 1000 samples
    # (after release settles) was 0.0 for this well-below-threshold case.
    steady_state_rms = np.sqrt(np.mean(y[-1000:] ** 2))
    assert steady_state_rms < 1e-6


def test_noise_gate_passes_loud_signal():
    gate = NoiseGate(threshold_db=-40, sr=SR)
    t = np.arange(SR) / SR
    loud = 0.5 * np.sin(2 * np.pi * 440 * t)  # well above -40dB threshold

    y = gate.process(loud)

    # Measured 2026-08-23: steady-state RMS 0.3524 vs input RMS 0.3536
    # (~99.7% preserved) once the gate's attack settles fully open.
    in_rms = np.sqrt(np.mean(loud[-1000:] ** 2))
    out_rms = np.sqrt(np.mean(y[-1000:] ** 2))
    assert out_rms > 0.95 * in_rms


def test_limiter_state_persists_across_blocks():
    lim = Limiter(threshold=0.98, sr=SR)
    t = np.arange(SR) / SR
    x = 2.0 * np.sin(2 * np.pi * 440 * t)

    chunks = [x[i:i + 256] for i in range(0, len(x), 256)]
    y = np.concatenate([lim.process(c) for c in chunks])

    assert np.all(np.isfinite(y))
    assert np.max(np.abs(y)) <= 0.98 + 1e-9
