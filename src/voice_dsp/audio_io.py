"""Stage 1: raw microphone-to-speaker passthrough through PortAudio.

No DSP happens here. The callback exists to prove the low-latency
duplex audio path works before any processing is layered on top.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import sounddevice as sd

DEFAULT_SAMPLERATE = 48000
DEFAULT_BLOCKSIZE = 256
DEFAULT_CHANNELS = 1


@dataclass
class PassthroughStats:
    """Measured behavior of a passthrough run, filled in as the stream runs."""

    callback_count: int = 0
    max_callback_seconds: float = 0.0
    total_callback_seconds: float = 0.0
    reported_input_latency: float = 0.0
    reported_output_latency: float = 0.0
    frames_processed: int = 0
    underflows: int = 0
    overflows: int = 0
    started_at: float = field(default_factory=time.perf_counter)

    @property
    def mean_callback_seconds(self) -> float:
        if self.callback_count == 0:
            return 0.0
        return self.total_callback_seconds / self.callback_count


def make_passthrough_callback(stats: PassthroughStats):
    """Build a sounddevice callback that copies input straight to output.

    Wall-clock time spent inside the callback is measured directly, since
    that is what determines whether we keep up with the audio clock.
    """

    def callback(indata, outdata, frames, time_info, status):
        t0 = time.perf_counter()
        if status.input_underflow:
            stats.underflows += 1
        if status.output_underflow:
            stats.underflows += 1
        if status.input_overflow or status.output_overflow:
            stats.overflows += 1

        outdata[:] = indata

        elapsed = time.perf_counter() - t0
        stats.callback_count += 1
        stats.frames_processed += frames
        stats.total_callback_seconds += elapsed
        if elapsed > stats.max_callback_seconds:
            stats.max_callback_seconds = elapsed

    return callback


def run_passthrough(
    duration: float | None = 30.0,
    samplerate: int = DEFAULT_SAMPLERATE,
    blocksize: int = DEFAULT_BLOCKSIZE,
    channels: int = DEFAULT_CHANNELS,
) -> PassthroughStats:
    """Run mic->speaker passthrough for `duration` seconds (None = forever).

    Returns measured stats after the stream closes.
    """
    stats = PassthroughStats()
    callback = make_passthrough_callback(stats)

    with sd.Stream(
        samplerate=samplerate,
        blocksize=blocksize,
        channels=channels,
        dtype="float32",
        latency="low",
        callback=callback,
    ) as stream:
        stats.reported_input_latency = stream.latency[0]
        stats.reported_output_latency = stream.latency[1]
        if duration is None:
            while True:
                time.sleep(0.1)
        else:
            # Tick visibly once a second rather than sleeping silently
            # for the whole duration, so a running (vs. hung) process
            # is obvious from the terminal.
            elapsed = 0.0
            while elapsed < duration:
                step = min(1.0, duration - elapsed)
                time.sleep(step)
                elapsed += step
                print(f"  ...{elapsed:.0f}s / {duration:.0f}s", flush=True)

    return stats


if __name__ == "__main__":
    import sys

    dur = float(sys.argv[1]) if len(sys.argv) > 1 else 30.0
    print(f"Running passthrough for {dur:.1f}s at {DEFAULT_SAMPLERATE} Hz, "
          f"blocksize={DEFAULT_BLOCKSIZE} ...")
    result = run_passthrough(duration=dur)
    print(f"callbacks: {result.callback_count}")
    print(f"frames processed: {result.frames_processed}")
    print(f"reported input latency: {result.reported_input_latency * 1000:.3f} ms")
    print(f"reported output latency: {result.reported_output_latency * 1000:.3f} ms")
    print(f"mean callback compute time: {result.mean_callback_seconds * 1e6:.1f} us")
    print(f"max callback compute time: {result.max_callback_seconds * 1e6:.1f} us")
    print(f"underflows: {result.underflows}, overflows: {result.overflows}")
