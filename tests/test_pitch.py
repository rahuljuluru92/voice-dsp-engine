import numpy as np

from src.voice_dsp.pitch import pitch_shift

# Empirically measured on 2026-08-23 for a 220Hz sine, sr=48000,
# frame_size=2048, analysis_hop=512, over +3/+7/-5/+12 semitones: worst
# observed error was ~0.033Hz (~0.35 cents). 1.0Hz absolute tolerance
# gives ~30x margin above that measurement while still catching a real
# regression (e.g. a wrong ratio or a broken instantaneous-frequency
# calculation, which produce multi-Hz to multi-hundred-Hz errors).
FREQ_TOLERANCE_HZ = 1.0


def _measure_freq(x: np.ndarray, sr: int) -> float:
    n = len(x)
    win = np.hanning(n)
    spec = np.fft.rfft(x * win)
    mag = np.abs(spec)
    k = int(np.argmax(mag))
    if 0 < k < len(mag) - 1:
        alpha, beta, gamma = mag[k - 1], mag[k], mag[k + 1]
        denom = alpha - 2 * beta + gamma
        p = 0.5 * (alpha - gamma) / denom if denom != 0 else 0.0
    else:
        p = 0.0
    return (k + p) * sr / n


def _make_tone(freq: float, sr: int, duration: float) -> np.ndarray:
    t = np.arange(int(sr * duration)) / sr
    return 0.8 * np.sin(2 * np.pi * freq * t)


def _measured_shifted_freq(f0: float, semitones: float, sr: int = 48000) -> tuple[float, float]:
    x = _make_tone(f0, sr, duration=2.0)
    y = pitch_shift(x, semitones, frame_size=2048, analysis_hop=512)

    edge = int(sr * 0.2)
    y_trimmed = y[edge:-edge]
    measured = _measure_freq(y_trimmed, sr)
    expected = f0 * (2 ** (semitones / 12))
    return measured, expected


def test_pitch_shift_up_perfect_fifth():
    measured, expected = _measured_shifted_freq(220.0, semitones=7)
    assert abs(measured - expected) < FREQ_TOLERANCE_HZ


def test_pitch_shift_up_octave():
    measured, expected = _measured_shifted_freq(220.0, semitones=12)
    assert abs(measured - expected) < FREQ_TOLERANCE_HZ


def test_pitch_shift_down_minor_third():
    measured, expected = _measured_shifted_freq(220.0, semitones=-5)
    assert abs(measured - expected) < FREQ_TOLERANCE_HZ


def test_pitch_shift_zero_semitones_is_identity():
    x = _make_tone(220.0, 48000, duration=1.0)
    y = pitch_shift(x, 0)
    assert np.allclose(x, y)


def test_pitch_shift_preserves_duration():
    x = _make_tone(220.0, 48000, duration=1.5)
    y = pitch_shift(x, 4)
    assert len(y) == len(x)
