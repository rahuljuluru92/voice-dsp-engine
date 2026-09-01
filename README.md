# Voice DSP Engine

A real-time pitch- and formant-shifting voice engine, built from scratch in
Python. It captures microphone audio through PortAudio, shifts pitch using a
short-time Fourier transform (STFT) phase vocoder with true
instantaneous-frequency phase reconstruction, warps formants independently of
pitch via cepstral-envelope processing, and runs a high-pass filter / noise
gate / limiter chain on the output — all through a low-latency callback.

## What it does

- **STFT core** (`src/voice_dsp/stft.py`): framing, windowing (periodic
  Hann), and weighted overlap-add inverse STFT. Round-trips a signal to
  within machine precision.
- **Pitch shifting** (`src/voice_dsp/pitch.py`): phase-vocoder time-stretch
  using true instantaneous-frequency phase reconstruction (not naive
  resampling), followed by resampling to restore duration.
- **Independent formant control** (`src/voice_dsp/formant.py`): cepstral
  liftering extracts the spectral envelope; warping it along frequency shifts
  formants without touching pitch, and an inverse warp cancels the
  frequency-scaling that pitch-shift resampling would otherwise impose on
  formants — so pitch and formant are controlled independently.
- **Signal chain** (`src/voice_dsp/filters.py`, `dynamics.py`, `chain.py`):
  an RBJ-cookbook biquad high-pass filter, a noise gate, and a limiter that
  guarantees finite, bounded output even on adversarial input.
- **Runtime robustness** (`src/voice_dsp/queues.py`, `bypass.py`): bounded
  drop-oldest queues so latency can't grow unbounded, and a bypass guard that
  outputs silence (never raw microphone audio) if processing fails, produces
  non-finite output, or misses its deadline.
- **Continuous streaming processor** (`src/voice_dsp/streaming.py`): the
  pitch/formant math restructured to run incrementally with a persistent
  per-bin synthesis-phase accumulator that never resets, rather than
  independently re-analyzing fixed-size chunks — verified to produce
  identical output regardless of how the caller chunks its input (4096
  samples at a time vs. 1 sample at a time), with no boundary artifacts.
  See `DECISIONS.md` for why the earlier chunked approach (with or without
  a crossfade) couldn't fully fix this.
- **Live engine** (`src/voice_dsp/engine.py`): wires the above into a
  PortAudio duplex stream. The audio callback only moves data through
  bounded queues; the actual DSP runs on a background worker thread so the
  real-time callback stays cheap.

This project does not include a GUI, file import/export, or network
streaming — see `DECISIONS.md` for what was explicitly considered and kept
out of scope.

## Environment setup

Requires Python 3.12 (developed and tested against 3.12.8) and a working
PortAudio installation. On macOS, the `sounddevice` package's wheel bundles
what it needs and generally works out of the box; if it can't find
PortAudio, install it with `brew install portaudio`. On Linux, install
`libportaudio2` (e.g. `apt install libportaudio2`) first.

Create and activate a virtual environment, then install pinned dependencies:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` pins exactly the versions this project was built and
tested against: `numpy==2.5.2`, `scipy==1.18.1`, `sounddevice==0.5.6`,
`pytest==9.1.1`. See `DECISIONS.md` for why `sounddevice` was chosen over
`pyaudio`.

## Running the live engine

```bash
python3 -m src.voice_dsp.engine <preset> <duration_seconds>
```

`<preset>` is one of the names in `src/voice_dsp/presets.py`: `identity`,
`pitch_up_third`, `pitch_down_third`, `pitch_up_octave`, `formant_up`,
`formant_down`, `pitch_and_formant_up`, `pitch_up_formant_down`. Example:

```bash
python3 -m src.voice_dsp.engine pitch_up_third 30
```

This opens your microphone and speakers directly — **use headphones**, since
it is a live audio loop and can otherwise produce feedback. There is also a
zero-DSP passthrough mode (Stage 1) for isolating raw I/O behavior:

```bash
python3 src/voice_dsp/audio_io.py 30
```

## Running the unit tests

```bash
source venv/bin/activate
python3 -m pytest
```

All test tolerances are documented in `DECISIONS.md` alongside the actual
measurements that motivated them, not chosen arbitrarily.

## Running the validation suite

The Stage 6 validation benchmark is separate from the unit test suite and
not run by pytest:

```bash
source venv/bin/activate
python3 -m validation.benchmark
```

It synthesizes a voice-like harmonic test tone at three base frequencies
(130/180/240Hz) for each of the 8 presets (24 cases total), runs the full
pitch+formant pipeline, measures the actual output frequency, and reports
pass/fail against a documented tolerance. See `DECISIONS.md` for why a
voice-like tone is used rather than a pure sine (a pure sine has no formants
to validate, and exposed a real, documented limitation when tried first).

## Measured results

All numbers below are actual measurements recorded in `DECISIONS.md` as they
were produced, not placeholders.

**Stage 2 — STFT round-trip and pitch accuracy** (sr=48000, frame_size=2048,
hop_size=512): round-trip max absolute error 3.33e-16 on a 1s 220Hz sine.
Pitch-shift accuracy on a 220Hz sine: +3st error +0.027Hz, +7st error
-0.026Hz, -5st error +0.033Hz, +12st error 0.000Hz.

**Stage 3 — independent pitch/formant control** (synthetic two-formant
vowel, F0=180Hz, formants at 700/1200Hz): formant-only shifts left F0
exactly unchanged; pitch-only (formant-preserving) shifts changed F0 within
0.27Hz of expected while leaving formant peaks within 46.9Hz (worst case) of
their original location.

**Stage 4 — filter and dynamics** (cutoff=80Hz, Q=0.707, sr=48000): 20Hz
attenuated -24.100dB, 80Hz (cutoff) at -3.012dB (matches the analytic
-3.01dB), 1000Hz+ within 0.001dB of unity. Limiter on a 5x-over-range sine
with injected inf/nan/1e9 samples: output stayed fully finite, bounded
exactly to the 0.98 threshold.

**Stage 5 — runtime robustness**: an 8-item bounded queue fed 1000 items
while draining every third item never exceeded capacity; a 4-item queue fed
10 items with no draining retained exactly the 4 newest and reported
`dropped_count == 6`. The bypass guard emitted silence (not raw input) on a
forced exception, a NaN result, and a deadline overrun; it passed real
output through unchanged when processing succeeded.

**Stage 6 — validation benchmark** (24 cases: 8 presets x 3 base
frequencies): 24/24 within the 3.0Hz tolerance. Worst-case error +0.3141Hz
(`formant_down` preset, 180Hz base). Re-run `python3 -m validation.benchmark`
to reproduce.

**Stage 1 — live passthrough latency**: run live for the full 30 seconds on
real hardware (MacBook Air built-in microphone/speakers): 5670 callbacks,
1,451,520 frames processed, reported input latency 227.021ms, reported
output latency 31.854ms, mean callback compute time 2.5us, max 22.1us, 0
underflows, 0 overflows.

**Continuous streaming processor**: an earlier chunked architecture (with
independent per-chunk phase-vocoder analysis, later with an overlap-window
crossfade) produced an audible click at every chunk boundary. Replaced with
`src/voice_dsp/streaming.py`, which never resets its internal phase state.
Verified (`tests/test_streaming.py`) to produce identical output regardless
of how the caller chunks input (4096 samples/call vs. 1 sample/call: max
difference ~3e-12) and zero large discontinuities across tested pitch
ratios. Confirmed live on real hardware (`pitch_up_third` preset, 30s):
`underrun_count=68`, `underrun_samples=1202` (25.0ms total, 0.083% of the
run), 0 bypasses, 0 queue drops. Audible result reported directly by the
project owner: clicking reduced to "a very little" — a substantial
improvement over the pre-rewrite version, with the small remainder most
plausibly explained by brief real-time scheduling jitter (see "Known
limitations" below) rather than the phase discontinuity this rewrite
targeted, which testing shows is fully eliminated.

## Known limitations

- A small amount of clicking remains audible on real hardware (measured:
  ~25ms of brief silence-padding underruns over a 30s run, 0.083% of the
  audio). This is real-time queue/thread scheduling jitter between the
  audio callback and the DSP worker thread, not the phase-vocoder
  discontinuity the streaming rewrite targeted (which testing shows is
  fully eliminated). Increasing `queue_capacity` or `blocksize` would trade
  a small amount of latency for fewer underruns if this needs to be reduced
  further.
