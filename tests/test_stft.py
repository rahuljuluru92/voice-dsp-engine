import numpy as np

from src.voice_dsp.stft import roundtrip

# Empirically measured on 2026-08-23: reconstructing a 1s, 220Hz sine at
# sr=48000, frame_size=2048, hop_size=512 gave a max abs error of
# ~3.3e-16 (float64 machine precision territory) since periodic-Hann at
# 75% overlap satisfies constant-overlap-add. 1e-9 gives ample margin
# above noise while still catching a real reconstruction bug.
ROUNDTRIP_TOLERANCE = 1e-9


def test_stft_roundtrip_sine_within_tolerance():
    sr = 48000
    t = np.arange(sr) / sr
    x = 0.5 * np.sin(2 * np.pi * 220.0 * t)

    y = roundtrip(x, frame_size=2048, hop_size=512)

    assert len(y) == len(x)
    max_err = np.max(np.abs(x - y))
    assert max_err < ROUNDTRIP_TOLERANCE


def test_stft_roundtrip_multitone_within_tolerance():
    sr = 48000
    t = np.arange(sr) / sr
    x = (
        0.3 * np.sin(2 * np.pi * 110.0 * t)
        + 0.2 * np.sin(2 * np.pi * 880.0 * t)
        + 0.1 * np.sin(2 * np.pi * 3300.0 * t)
    )

    y = roundtrip(x, frame_size=2048, hop_size=512)

    max_err = np.max(np.abs(x - y))
    assert max_err < ROUNDTRIP_TOLERANCE


def test_stft_roundtrip_silence():
    x = np.zeros(48000)
    y = roundtrip(x, frame_size=2048, hop_size=512)
    assert np.max(np.abs(y)) < ROUNDTRIP_TOLERANCE
