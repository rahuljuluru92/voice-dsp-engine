import numpy as np

from src.voice_dsp.bypass import BypassGuard


def test_bypass_outputs_silence_when_processing_raises():
    def failing_process(x):
        raise RuntimeError("forced failure for test")

    guard = BypassGuard(failing_process)
    x = 0.5 * np.ones(256)

    y = guard.safe_process(x)

    assert np.all(y == 0.0)
    assert not np.array_equal(y, x)  # not raw mic input
    assert guard.bypass_count == 1


def test_bypass_outputs_silence_on_non_finite_result():
    def broken_process(x):
        y = x.copy()
        y[0] = np.nan
        return y

    guard = BypassGuard(broken_process)
    x = 0.5 * np.ones(256)

    y = guard.safe_process(x)

    assert np.all(y == 0.0)
    assert guard.bypass_count == 1


def test_bypass_outputs_silence_when_processing_exceeds_deadline():
    import time

    def slow_process(x):
        time.sleep(0.02)
        return x

    guard = BypassGuard(slow_process, max_process_seconds=0.005)
    x = 0.5 * np.ones(256)

    y = guard.safe_process(x)

    assert np.all(y == 0.0)
    assert guard.bypass_count == 1


def test_bypass_passes_through_healthy_processing():
    def double(x):
        return x * 2

    guard = BypassGuard(double)
    x = 0.25 * np.ones(256)

    y = guard.safe_process(x)

    assert np.allclose(y, 0.5)
    assert guard.bypass_count == 0
