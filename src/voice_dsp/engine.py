"""Integration: the live voice engine, wiring the Stage 2/3 pitch and
formant DSP and the Stage 4 signal chain into the Stage 5 bounded-queue
/ worker-thread / bypass-guard runtime, driven by a PortAudio callback.

Architecture: the audio callback itself does only cheap buffer slicing
and queue push/pop — no FFT work happens on the real-time audio thread.
Incoming audio is accumulated into fixed-size chunks and handed to a
bounded, drop-oldest input queue; a background worker thread pulls
chunks, runs the (batch, whole-chunk) pitch/formant/chain pipeline
through a BypassGuard, and pushes results to a bounded output queue.
The callback pulls processed audio from an internal leftover buffer
fed by that output queue, and outputs silence if not enough processed
audio is ready yet (never raw passthrough).

This keeps the real-time thread safe by construction, at the cost of
one chunk's worth of latency before processing can begin. Each chunk
is still DSP-processed independently (no continuous phase-vocoder
state across the whole session), which causes a phase-vocoder analysis
discontinuity at each chunk boundary; consecutive chunks are given a
small amount of genuinely overlapping input context and their outputs
crossfaded in that overlap region (see `_emit_with_crossfade`) to
reduce (not eliminate) the audible effect of that discontinuity,
without discarding audio the way a naive crossfade of non-overlapping
chunks would (see DECISIONS.md for the earlier, reverted attempt that
did exactly that). See DECISIONS.md / ASSUMPTIONS.md for measured
latency and boundary-artifact numbers.
"""

from __future__ import annotations

import threading
import time

import numpy as np
import sounddevice as sd

from . import formant
from .bypass import BypassGuard
from .chain import SignalChain
from .queues import BoundedDropOldestQueue


class VoiceEngine:
    def __init__(
        self,
        sr: int = 48000,
        chunk_size: int = 4096,
        frame_size: int = 2048,
        analysis_hop: int = 512,
        pitch_semitones: float = 0.0,
        formant_semitones: float = 0.0,
        hpf_cutoff_hz: float = 80.0,
        gate_threshold_db: float = -50.0,
        limiter_threshold: float = 0.98,
        queue_capacity: int = 4,
        max_process_seconds: float | None = 1.0,
        overlap_samples: int = 240,
    ):
        self.sr = sr
        self.chunk_size = chunk_size
        self.frame_size = frame_size
        self.analysis_hop = analysis_hop
        self.pitch_semitones = pitch_semitones
        self.formant_semitones = formant_semitones
        self.overlap_samples = overlap_samples

        self.chain = SignalChain(
            sr=sr,
            hpf_cutoff_hz=hpf_cutoff_hz,
            gate_threshold_db=gate_threshold_db,
            limiter_threshold=limiter_threshold,
        )
        self.input_queue = BoundedDropOldestQueue(queue_capacity)
        self.output_queue = BoundedDropOldestQueue(queue_capacity)
        self.guard = BypassGuard(self._dsp_pipeline, max_process_seconds)

        self._input_accum = np.zeros(0, dtype=np.float64)
        self._output_leftover = np.zeros(0, dtype=np.float64)
        self._prev_raw_tail = np.zeros(0, dtype=np.float64)
        self._held_tail: np.ndarray | None = None
        self._fade_in = (
            0.5 - 0.5 * np.cos(np.linspace(0, np.pi, overlap_samples))
            if overlap_samples > 0 else np.zeros(0)
        )
        self.underrun_count = 0
        self.underrun_samples = 0
        self.callback_count = 0
        self._stop_event = threading.Event()
        self._worker_thread: threading.Thread | None = None

    def _dsp_pipeline(self, chunk: np.ndarray) -> np.ndarray:
        y = formant.process(
            chunk,
            pitch_semitones=self.pitch_semitones,
            formant_semitones=self.formant_semitones,
            frame_size=self.frame_size,
            analysis_hop=self.analysis_hop,
        )
        return self.chain.process(y)

    def _emit_with_crossfade(self, processed: np.ndarray) -> np.ndarray:
        """Turn one processed (possibly overlap-extended) chunk into the
        `chunk_size` samples of output to actually emit this iteration.

        Each chunk fed to `_dsp_pipeline` includes `overlap_samples` of
        genuine input context reused from the tail of the previous raw
        input chunk (see `audio_callback`), so -- unlike the earlier,
        reverted crossfade attempt -- the overlapping region here really
        does represent the same underlying audio processed twice, not
        two different moments. Blending it is therefore smoothing, not
        discarding: the last `overlap_samples` of this chunk's output
        are held back and blended into the *start* of the next chunk's
        own overlap region next iteration, and steady-state emission is
        exactly `chunk_size` samples per call (verified in
        tests/test_engine.py), unlike the earlier version which lost
        `overlap_samples` of real audio every cycle.
        """
        n = self.overlap_samples
        if n <= 0:
            return processed

        if self._held_tail is None:
            # First chunk: no held-back tail to blend with yet, and (per
            # audio_callback) this chunk has no prefix context either
            # (length == chunk_size, not chunk_size + n).
            emit = processed[: self.chunk_size - n]
        else:
            # Non-first chunk: length == chunk_size + n. Layout:
            #   [0:n]              overlap region, shared with the
            #                      previous chunk's held-back tail
            #   [n:chunk_size]     unambiguous new content (length
            #                      chunk_size - n), emitted as-is
            #   [chunk_size:+n]    new overlap region, held back for
            #                      next iteration's blend
            fade_in = self._fade_in
            fade_out = 1.0 - fade_in
            blended = self._held_tail * fade_out + processed[:n] * fade_in
            emit = np.concatenate([blended, processed[n:self.chunk_size]])

        tail_start = len(processed) - n
        self._held_tail = processed[tail_start:].copy()
        return emit

    def _worker_loop(self) -> None:
        while not self._stop_event.is_set():
            chunk = self.input_queue.get()
            if chunk is None:
                time.sleep(0.001)
                continue
            processed = self.guard.safe_process(chunk)
            emit = self._emit_with_crossfade(processed)
            if len(emit):
                self.output_queue.put(emit)

    def start(self) -> None:
        self._stop_event.clear()
        self._prev_raw_tail = np.zeros(0, dtype=np.float64)
        self._held_tail = None
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._worker_thread is not None:
            self._worker_thread.join(timeout=2.0)
            self._worker_thread = None
        # Safe to touch _held_tail now that the worker thread (the only
        # other writer) has fully exited.
        if self._held_tail is not None and len(self._held_tail):
            self.output_queue.put(self._held_tail)
            self._held_tail = None

    def _pull_output(self, n_needed: int) -> np.ndarray:
        while len(self._output_leftover) < n_needed:
            more = self.output_queue.get()
            if more is None:
                break
            self._output_leftover = np.concatenate([self._output_leftover, more])

        if len(self._output_leftover) >= n_needed:
            out = self._output_leftover[:n_needed]
            self._output_leftover = self._output_leftover[n_needed:]
            return out

        # Not enough processed audio ready: emit silence, not raw input.
        # Counted (cheap integer increments only, no I/O) so a live run
        # can report afterward how often this actually happened.
        available = self._output_leftover
        missing = n_needed - len(available)
        self.underrun_count += 1
        self.underrun_samples += missing
        self._output_leftover = np.zeros(0, dtype=np.float64)
        return np.concatenate([available, np.zeros(missing)])

    def audio_callback(self, indata, outdata, frames, time_info, status) -> None:
        self.callback_count += 1
        self._input_accum = np.concatenate([self._input_accum, indata[:, 0].astype(np.float64)])
        n = self.overlap_samples
        while len(self._input_accum) >= self.chunk_size:
            new_part = self._input_accum[: self.chunk_size]
            self._input_accum = self._input_accum[self.chunk_size:]
            if n > 0 and len(self._prev_raw_tail) == n:
                # Prepend genuine overlap context (the same raw input
                # samples the previous chunk ended with), so the two
                # chunks' outputs share real underlying audio in their
                # overlap region instead of being unrelated. See
                # _emit_with_crossfade.
                windowed = np.concatenate([self._prev_raw_tail, new_part])
            else:
                windowed = new_part
            if n > 0:
                self._prev_raw_tail = new_part[-n:].copy()
            self.input_queue.put(windowed)

        outdata[:, 0] = self._pull_output(frames)

    def run(self, duration: float | None = None, blocksize: int = 256, channels: int = 1) -> None:
        self.start()
        try:
            with sd.Stream(
                samplerate=self.sr,
                blocksize=blocksize,
                channels=channels,
                dtype="float32",
                latency="low",
                callback=self.audio_callback,
            ):
                if duration is None:
                    while True:
                        time.sleep(0.1)
                else:
                    # Deliberately silent during the run: unlike the
                    # zero-DSP Stage 1 passthrough, this callback and its
                    # worker thread are doing real FFT-based work with
                    # much less slack per block, and periodic print()
                    # calls from the main thread were confirmed (via
                    # live testing) to cause audible clicks by briefly
                    # holding the GIL at the wrong moment. Progress here
                    # is "are you hearing your voice shifted correctly",
                    # not a printed counter.
                    time.sleep(duration)
        finally:
            self.stop()


if __name__ == "__main__":
    import sys

    from .presets import PRESETS

    preset_name = sys.argv[1] if len(sys.argv) > 1 else "identity"
    dur = float(sys.argv[2]) if len(sys.argv) > 2 else 30.0
    preset = PRESETS[preset_name]

    print(f"Running live engine with preset '{preset.name}' "
          f"(pitch={preset.pitch_semitones:+.1f}st, formant={preset.formant_semitones:+.1f}st) "
          f"for {dur:.1f}s ...")
    engine = VoiceEngine(pitch_semitones=preset.pitch_semitones, formant_semitones=preset.formant_semitones)
    engine.run(duration=dur)
    print(f"bypass_count: {engine.guard.bypass_count}")
    print(f"input queue dropped: {engine.input_queue.dropped_count}")
    print(f"output queue dropped: {engine.output_queue.dropped_count}")
    print(f"callback_count: {engine.callback_count}")
    print(f"underrun_count: {engine.underrun_count} "
          f"(times the callback had to pad with silence -- not enough processed audio was ready)")
    print(f"underrun_samples: {engine.underrun_samples} "
          f"({engine.underrun_samples / engine.sr * 1000:.1f} ms of inserted silence total)")
