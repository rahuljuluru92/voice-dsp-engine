import time

import numpy as np

from src.voice_dsp.engine import VoiceEngine


class _Status:
    pass


def _feed_synthetic_audio(engine, n_blocks=300, blocksize=256, sr=48000, freq=220.0):
    outputs = []
    # Sleep the real block duration (blocksize/sr), not an arbitrary
    # fixed amount: sleeping faster than real-time (e.g. a flat 1ms
    # regardless of blocksize) starves the worker thread relative to
    # how fast the callback is demanding output, producing genuine
    # silence-padding underruns that are an artifact of this test's
    # pacing, not of the engine -- confirmed 2026-08-23 by checking
    # input_queue.dropped_count == 0 (nothing was actually dropped)
    # while underrun_count was nonzero under the faster-than-real-time
    # version of this loop.
    block_duration = blocksize / sr
    for i in range(n_blocks):
        samples = np.arange(blocksize) + i * blocksize
        indata = (0.3 * np.sin(2 * np.pi * freq * samples / sr)).reshape(-1, 1).astype(np.float32)
        outdata = np.zeros((blocksize, 1), dtype=np.float32)
        engine.audio_callback(indata, outdata, blocksize, None, _Status())
        outputs.append(outdata.copy())
        time.sleep(block_duration)
    return np.concatenate(outputs).flatten()


def test_engine_end_to_end_produces_finite_bounded_output():
    engine = VoiceEngine(
        sr=48000,
        frame_size=2048,
        analysis_hop=512,
        pitch_semitones=0.0,
        formant_semitones=0.0,
        queue_capacity=8,
        max_process_seconds=1.0,
    )
    engine.start()
    try:
        y = _feed_synthetic_audio(engine)
    finally:
        engine.stop()

    assert np.all(np.isfinite(y))
    assert np.max(np.abs(y)) <= 0.98 + 1e-6
    # The continuous streaming processor releases output in small
    # increments as soon as each analysis hop is ready, rather than
    # waiting for a whole multi-thousand-sample chunk, so real audio
    # should appear quickly and dominate the run.
    nonzero_fraction = np.mean(y != 0.0)
    assert nonzero_fraction > 0.5
    assert engine.guard.bypass_count == 0


def test_engine_with_pitch_and_formant_preset_stays_finite():
    engine = VoiceEngine(
        sr=48000,
        frame_size=2048,
        analysis_hop=512,
        pitch_semitones=5.0,
        formant_semitones=-3.0,
        queue_capacity=8,
        max_process_seconds=1.0,
    )
    engine.start()
    try:
        y = _feed_synthetic_audio(engine, n_blocks=200)
    finally:
        engine.stop()

    assert np.all(np.isfinite(y))
    assert np.max(np.abs(y)) <= 0.98 + 1e-6


def test_engine_underruns_stay_rare_under_realtime_pacing():
    # The phase-continuity property itself (no chunk-boundary
    # discontinuities from the DSP) is proven directly and
    # deterministically in tests/test_streaming.py, independent of any
    # timing simulation. This test instead checks the queue/threading
    # layer's real-time behavior: measured 2026-08-23, a
    # time.sleep()-per-block Python simulation of real-time pacing
    # produced 13 underruns / 778 samples (~16ms) over a 2.13s run with
    # zero queue drops -- confirmed to be Python thread-scheduling
    # jitter, not lost or corrupted audio (real PortAudio hardware
    # testing the same session showed a comparable or better rate: 19
    # underruns / 101ms over a full 30s live run -- see DECISIONS.md).
    # Assert underruns stay in that same rare, bounded ballpark rather
    # than growing unboundedly, without asserting an exact zero this
    # synthetic harness can't reliably guarantee.
    engine = VoiceEngine(sr=48000, pitch_semitones=4.0, formant_semitones=0.0, queue_capacity=8)
    engine.start()
    try:
        y = _feed_synthetic_audio(engine, n_blocks=400, freq=180.0)
    finally:
        engine.stop()

    assert np.all(np.isfinite(y))
    assert engine.input_queue.dropped_count == 0
    assert engine.output_queue.dropped_count == 0
    # Generous margin above the measured 778/98560 (~0.8%) baseline.
    assert engine.underrun_samples < len(y) * 0.1


def test_engine_callback_never_blocks_indefinitely():
    engine = VoiceEngine(sr=48000, queue_capacity=8)
    # Do NOT start the worker thread: the callback must still return
    # promptly (with silence, via the leftover/queue-empty path) rather
    # than hang waiting for processed audio that will never arrive.
    blocksize = 256
    indata = np.zeros((blocksize, 1), dtype=np.float32)
    outdata = np.zeros((blocksize, 1), dtype=np.float32)

    start = time.perf_counter()
    engine.audio_callback(indata, outdata, blocksize, None, _Status())
    elapsed = time.perf_counter() - start

    assert elapsed < 0.5
    assert np.all(outdata == 0.0)
