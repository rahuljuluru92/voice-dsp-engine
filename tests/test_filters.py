import numpy as np

from src.voice_dsp.filters import BiquadHighpass

# Reference response measured on 2026-08-23 for cutoff=80Hz, Q=0.707,
# sr=48000 (RBJ cookbook biquad high-pass, see design_highpass()):
#   20Hz: -24.100 dB   40Hz: -12.305 dB   80Hz: -3.012 dB
#   160Hz: -0.264 dB   500Hz: -0.003 dB   1000Hz: -0.000 dB
# The 80Hz (cutoff) value matches the analytically expected -3.01dB for
# an RBJ biquad at its corner frequency.
SR = 48000
CUTOFF_HZ = 80.0


def test_highpass_attenuates_below_cutoff():
    hpf = BiquadHighpass(cutoff_hz=CUTOFF_HZ, sr=SR, q=0.707)
    resp = hpf.frequency_response_db(np.array([20.0]))
    assert resp[0] < -20.0  # measured -24.1dB


def test_highpass_minus_3db_at_cutoff():
    hpf = BiquadHighpass(cutoff_hz=CUTOFF_HZ, sr=SR, q=0.707)
    resp = hpf.frequency_response_db(np.array([CUTOFF_HZ]))
    assert abs(resp[0] - (-3.01)) < 0.1


def test_highpass_passes_high_frequencies():
    hpf = BiquadHighpass(cutoff_hz=CUTOFF_HZ, sr=SR, q=0.707)
    resp = hpf.frequency_response_db(np.array([1000.0, 5000.0]))
    assert np.all(np.abs(resp) < 0.1)


def test_highpass_streaming_matches_single_call():
    hpf_a = BiquadHighpass(cutoff_hz=CUTOFF_HZ, sr=SR, q=0.707)
    hpf_b = BiquadHighpass(cutoff_hz=CUTOFF_HZ, sr=SR, q=0.707)

    rng = np.random.default_rng(0)
    x = rng.uniform(-1, 1, size=4096)

    y_whole = hpf_a.process(x)

    chunks = [x[i:i + 256] for i in range(0, len(x), 256)]
    y_chunks = np.concatenate([hpf_b.process(c) for c in chunks])

    assert np.allclose(y_whole, y_chunks, atol=1e-10)
