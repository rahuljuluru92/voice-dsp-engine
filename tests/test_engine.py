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


def test_emit_with_crossfade_conserves_all_samples():
    # Regression test: an earlier version of the overlap crossfade
    # blended two *unrelated* chunks (no shared input), which silently
    # discarded overlap_samples of real audio every cycle. This drives
    # _dsp_pipeline + _emit_with_crossfade directly (no threading) the
    # same way audio_callback's chunking does, and checks that total
    # samples emitted equals total samples consumed -- exactly the
    # invariant that broke before.
    sr = 48000
    engine = VoiceEngine(sr=sr, pitch_semitones=4.0, formant_semitones=0.0)

    freq = 180.0
    duration = 2.0
    n = int(sr * duration)
    x = 0.5 * np.sin(2 * np.pi * freq * np.arange(n) / sr)

    chunk_size = engine.chunk_size
    n_ov = engine.overlap_samples
    chunks_in = [x[i:i + chunk_size] for i in range(0, len(x) - chunk_size + 1, chunk_size)]
    total_in = sum(len(c) for c in chunks_in)

    emitted = []
    prev_tail = np.zeros(0)
    for c in chunks_in:
        windowed = np.concatenate([prev_tail, c]) if len(prev_tail) == n_ov else c
        prev_tail = c[-n_ov:].copy()
        processed = engine._dsp_pipeline(windowed)
        emitted.append(engine._emit_with_crossfade(processed))
    if engine._held_tail is not None:
        emitted.append(engine._held_tail)

    total_out = sum(len(e) for e in emitted)
    assert total_out == total_in


def test_overlap_crossfade_reduces_chunk_boundary_discontinuities():
    # Measured 2026-08-23: on a 3s 180Hz tone, no crossfade produced 41
    # discontinuities (peak magnitude 0.51 on a 0.5-amplitude signal);
    # the overlap crossfade reduced this to 17 (peak magnitude 0.17).
    # Assert the crossfaded version is a real improvement, not just
    # equal, so a future regression (like the sample-losing version)
    # would be caught even if it happened to conserve samples.
    sr = 48000
    freq = 180.0
    duration = 2.0
    n = int(sr * duration)
    x = 0.5 * np.sin(2 * np.pi * freq * np.arange(n) / sr)

    def render(overlap_samples):
        engine = VoiceEngine(
            sr=sr, pitch_semitones=4.0, formant_semitones=0.0, overlap_samples=overlap_samples
        )
        chunk_size = engine.chunk_size
        chunks_in = [x[i:i + chunk_size] for i in range(0, len(x) - chunk_size + 1, chunk_size)]
        if overlap_samples == 0:
            return np.concatenate([engine._dsp_pipeline(c) for c in chunks_in])
        emitted = []
        prev_tail = np.zeros(0)
        for c in chunks_in:
            windowed = np.concatenate([prev_tail, c]) if len(prev_tail) == overlap_samples else c
            prev_tail = c[-overlap_samples:].copy()
            processed = engine._dsp_pipeline(windowed)
            emitted.append(engine._emit_with_crossfade(processed))
        if engine._held_tail is not None:
            emitted.append(engine._held_tail)
        return np.concatenate(emitted)

    def max_discontinuity(y):
        d = np.abs(np.diff(y))
        return np.max(d) if len(d) else 0.0

    peak_no_crossfade = max_discontinuity(render(0))
    peak_with_crossfade = max_discontinuity(render(240))

    assert peak_no_crossfade > 0.3  # sanity: the baseline artifact is real and large
    assert peak_with_crossfade < peak_no_crossfade * 0.5  # meaningful reduction


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
