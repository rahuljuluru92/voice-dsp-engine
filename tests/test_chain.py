import numpy as np

from src.voice_dsp.chain import SignalChain

SR = 48000


def test_chain_stays_finite_and_bounded_on_extreme_blockwise_input():
    chain = SignalChain(sr=SR)
    t = np.arange(SR) / SR
    x = 8.0 * np.sin(2 * np.pi * 440 * t)
    x[500:510] = [np.inf, -np.inf, np.nan, 1e12, -1e12, 50, -50, 3, -3, 0]

    chunks = [x[i:i + 256] for i in range(0, len(x), 256)]
    y = np.concatenate([chain.process(c) for c in chunks])

    assert np.all(np.isfinite(y))
    assert np.max(np.abs(y)) <= 0.98 + 1e-9
