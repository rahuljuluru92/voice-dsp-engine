import time

import numpy as np

from src.voice_dsp.engine import VoiceEngine


class _Status:
    pass


def _feed_synthetic_audio(engine, n_blocks=300, blocksize=256, sr=48000, freq=220.0):
    outputs = []
    for i in range(n_blocks):
        samples = np.arange(blocksize) + i * blocksize
        indata = (0.3 * np.sin(2 * np.pi * freq * samples / sr)).reshape(-1, 1).astype(np.float32)
        outdata = np.zeros((blocksize, 1), dtype=np.float32)
        engine.audio_callback(indata, outdata, blocksize, None, _Status())
        outputs.append(outdata.copy())
        time.sleep(0.001)
    return np.concatenate(outputs).flatten()


def test_engine_end_to_end_produces_finite_bounded_output():
    engine = VoiceEngine(
        sr=48000,
        chunk_size=4096,
        frame_size=2048,
        analysis_hop=512,
        pitch_semitones=0.0,
        formant_semitones=0.0,
        queue_capacity=4,
        max_process_seconds=1.0,
    )
    engine.start()
    try:
        y = _feed_synthetic_audio(engine)
    finally:
        engine.stop()

    assert np.all(np.isfinite(y))
    assert np.max(np.abs(y)) <= 0.98 + 1e-6
    # Measured 2026-08-23: after the initial chunk-buffering delay,
    # ~94% of frames (72192/76800) carried real processed audio rather
    # than the startup silence; require a comfortable majority so this
    # test fails if the pipeline stalls or bypasses everything.
    nonzero_fraction = np.mean(y != 0.0)
    assert nonzero_fraction > 0.5
    assert engine.guard.bypass_count == 0


def test_engine_with_pitch_and_formant_preset_stays_finite():
    engine = VoiceEngine(
        sr=48000,
        chunk_size=4096,
        frame_size=2048,
        analysis_hop=512,
        pitch_semitones=5.0,
        formant_semitones=-3.0,
        queue_capacity=4,
        max_process_seconds=1.0,
    )
    engine.start()
    try:
        y = _feed_synthetic_audio(engine, n_blocks=200)
    finally:
        engine.stop()

    assert np.all(np.isfinite(y))
    assert np.max(np.abs(y)) <= 0.98 + 1e-6


def test_engine_callback_never_blocks_indefinitely():
    engine = VoiceEngine(sr=48000, chunk_size=4096, queue_capacity=4)
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
