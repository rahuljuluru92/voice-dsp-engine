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
one chunk's worth of latency before processing can begin, and (since
each chunk is processed independently) a phase-vocoder analysis
discontinuity at each chunk boundary — see DECISIONS.md /
ASSUMPTIONS.md for the measured latency and the boundary-artifact
tradeoff.
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
    ):
        self.sr = sr
        self.chunk_size = chunk_size
        self.frame_size = frame_size
        self.analysis_hop = analysis_hop
        self.pitch_semitones = pitch_semitones
        self.formant_semitones = formant_semitones

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

    def _worker_loop(self) -> None:
        while not self._stop_event.is_set():
            chunk = self.input_queue.get()
            if chunk is None:
                time.sleep(0.001)
                continue
            processed = self.guard.safe_process(chunk)
            self.output_queue.put(processed)

    def start(self) -> None:
        self._stop_event.clear()
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._worker_thread is not None:
            self._worker_thread.join(timeout=2.0)
            self._worker_thread = None

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
        available = self._output_leftover
        self._output_leftover = np.zeros(0, dtype=np.float64)
        return np.concatenate([available, np.zeros(n_needed - len(available))])

    def audio_callback(self, indata, outdata, frames, time_info, status) -> None:
        self._input_accum = np.concatenate([self._input_accum, indata[:, 0].astype(np.float64)])
        while len(self._input_accum) >= self.chunk_size:
            chunk = self._input_accum[: self.chunk_size]
            self._input_accum = self._input_accum[self.chunk_size:]
            self.input_queue.put(chunk)

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
