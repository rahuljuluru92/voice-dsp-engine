# Voice DSP Engine

A real-time pitch- and formant-shifting voice engine, built entirely from
scratch in Python, with no third-party voice-changer library and no
black-box DSP. It listens to your microphone and reshapes your voice live:
higher, lower, bigger, smaller, or any combination, while staying clean,
low-latency, and safe against crashes and glitches.

Every stage of the signal path, from FFT framing through the phase vocoder,
the cepstral formant math, the biquad filter, the noise gate, the limiter,
and the runtime scheduling, is implemented directly on top of
`numpy`/`scipy`, not delegated to an existing pitch-shifting library.

## What it actually does

Speak into your microphone; the engine plays your voice back through your
speakers or headphones in real time, transformed according to whichever
preset you choose:

| Preset                  | Pitch     | Formant   | Effect                                   |
|--------------------------|-----------|-----------|-------------------------------------------|
| `identity`               | 0 st      | 0 st      | Passthrough, no change (sanity check)     |
| `pitch_up_third`         | +4 st     | 0 st      | Higher voice, same "size"                 |
| `pitch_down_third`       | -4 st     | 0 st      | Lower voice, same "size"                  |
| `pitch_up_octave`        | +12 st    | 0 st      | An octave up                              |
| `formant_up`             | 0 st      | +4 st     | Smaller-sounding voice, same pitch        |
| `formant_down`           | 0 st      | -4 st     | Bigger-sounding voice, same pitch         |
| `pitch_and_formant_up`   | +5 st     | +3 st     | Higher and smaller-sounding together      |
| `pitch_up_formant_down`  | +7 st     | -3 st     | Higher pitch, bigger-sounding voice       |

**Pitch** and **formant** are controlled independently. That's the core
technical claim of the project, and it's what separates this from a cheap
"speed up the tape" voice changer. Pitch is how high or low a note sounds;
formants are what make a voice sound like it's coming from a big or small
person, largely independent of pitch. Shifting pitch without correcting
formants makes everyone sound like a chipmunk or a demon; this engine keeps
them separate, so you can raise pitch *without* the chipmunk effect, or
change the apparent size of the speaker *without* changing the note.

## Core features

- **Real-time phase-vocoder pitch shifting** with true instantaneous-frequency
  phase reconstruction, not naive resampling (naive resampling would change
  duration and destroy phase coherence between harmonics).
- **Independent formant control** via cepstral-envelope warping: moves the
  spectral envelope (formants) without touching the fundamental frequency,
  and vice versa.
- **Continuous streaming DSP**: a persistent per-bin phase accumulator that
  never resets, so there are no chunk-boundary artifacts regardless of how
  audio happens to arrive from the OS.
- **A real signal chain**: high-pass filter, then noise gate, then limiter,
  all running on every block, with the limiter structurally guaranteeing
  finite, bounded output no matter what comes in.
- **Fails safe, not loud**: bounded queues prevent runaway latency, and a
  bypass guard emits silence (never raw, unprocessed microphone audio) if
  anything in the pipeline breaks or falls behind.
- **Backed by real measurements, not vibes**: every tolerance in the test
  suite and every claim in this README traces back to an actual number,
  recorded as it was produced, in `DECISIONS.md`.

## How it works

```
mic → [audio callback: cheap, real-time-safe]
        │
        ▼
  bounded input queue  ──►  worker thread (off the real-time thread)
                                 │
                                 ▼
                    StreamingVoiceProcessor
              (persistent phase-vocoder + cepstral
               formant warp, one continuous pass)
                                 │
                                 ▼
                SignalChain (HPF → gate → limiter)
                                 │
                                 ▼
                        bounded output queue
                                 │
                                 ▼
        [audio callback pulls ready audio, or outputs silence]
                                 │
                                 ▼
                             speakers
```

The real-time audio callback (the piece PortAudio calls on a tight, strict
schedule) does almost nothing: it just moves raw audio into a queue and
pulls processed audio out of another one. All the actual math (FFTs, phase
tracking, filtering) happens on a separate worker thread, so a slow DSP step
can never stall the audio hardware. If the worker ever falls behind, you get
silence, never a glitch or the raw unprocessed input.

### The DSP pipeline, module by module

| Module | Responsibility |
|---|---|
| `stft.py` | Framing, windowing (periodic Hann), and weighted-overlap-add reconstruction. The foundation everything else is built on, validated to reconstruct a signal to within machine precision. |
| `pitch.py` | Phase-vocoder time-stretch with true instantaneous-frequency tracking, followed by resampling to restore duration and shift pitch. |
| `formant.py` | Cepstral-envelope extraction and frequency-axis warping, used to move formants independently of pitch and to cancel the formant-shifting side effect that resampling would otherwise introduce. |
| `streaming.py` | The pitch and formant math re-derived to run incrementally, forever, with no restart points. This is what makes the live engine click-free. |
| `filters.py` | An RBJ-cookbook biquad high-pass filter, derived from closed-form coefficients, not a library black box. |
| `dynamics.py` | A noise gate (mutes silence/background hiss) and a limiter (structurally guarantees bounded output). |
| `chain.py` | Wires the filter, gate, and limiter into one processing chain. |
| `queues.py` | A thread-safe, capacity-bounded, drop-oldest queue: latency can never grow without bound. |
| `bypass.py` | Wraps the DSP pipeline; on any failure, non-finite output, or missed deadline, emits silence instead. |
| `engine.py` | The live engine: wires everything above into an actual PortAudio duplex stream. |
| `presets.py` | The named pitch/formant combinations listed in the table above. |

## Project structure

```
voice-dsp-engine/
├── src/voice_dsp/        # the engine itself (see table above)
├── tests/                 # pytest unit tests, one file per module
├── validation/            # the Stage 6 synthetic-tone benchmark (separate from pytest)
├── requirements.txt        # pinned dependency versions
├── DECISIONS.md             # every technical decision, with the reasoning and real measurements behind it
├── ASSUMPTIONS.md            # everything assumed because it wasn't explicitly specified
└── README.md                  # this file
```

`DECISIONS.md` and `ASSUMPTIONS.md` are worth reading if you want the full
story: they document the actual engineering process, including two dead
ends (a chunking approach that clicked at every boundary, and a crossfade
fix that turned out to silently discard audio) before arriving at the
current architecture, all with real measured numbers rather than
after-the-fact narrative.

## Getting started

### Requirements

- Python 3.12 (developed and tested against 3.12.8)
- A working PortAudio installation. On macOS this is usually bundled
  automatically; if not, `brew install portaudio`. On Linux,
  `apt install libportaudio2` (or your distro's equivalent) first.

### Install

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` pins exact versions this project was built and tested
against: `numpy==2.5.2`, `scipy==1.18.1`, `sounddevice==0.5.6`,
`pytest==9.1.1`.

### Run it

```bash
python3 -m src.voice_dsp.engine pitch_up_third 30
```

**Use headphones.** This opens a live duplex stream between your microphone
and speakers; without headphones you'll get feedback. Swap `pitch_up_third`
for any preset from the table above, and `30` for however many seconds you
want it to run.

There's also a zero-DSP raw passthrough mode, useful for isolating whether
any issue is in the audio I/O itself versus the DSP:

```bash
python3 src/voice_dsp/audio_io.py 30
```

### Run the tests

```bash
python3 -m pytest
```

43 tests covering every module, from bit-exact STFT reconstruction to full
threaded engine integration. Every tolerance in the suite is backed by a
real measurement recorded in `DECISIONS.md`, not picked arbitrarily.

### Run the validation benchmark

```bash
python3 -m validation.benchmark
```

A reproducible accuracy benchmark, separate from the unit tests: synthesizes
a voice-like harmonic test tone at three base frequencies for each of the 8
presets (24 cases), runs the full pipeline, and measures the actual output
frequency against what was expected.

## Measured results

Everything below is a real number, produced by actually running the code,
never a placeholder.

- **STFT round-trip**: 3.33e-16 max absolute error (essentially the limit of
  float64 precision) reconstructing a 1-second 220Hz sine.
- **Pitch-shift accuracy**: within 0.03Hz of the expected frequency across
  ±3 to +12 semitones on a 220Hz test tone.
- **Formant independence**: a formant-only shift leaves the fundamental
  frequency *exactly* unchanged; a pitch-only shift leaves formant peak
  locations within ~47Hz of their original position.
- **Filter response**: matches the analytically expected -3.01dB at the
  biquad's cutoff frequency, within 0.002dB.
- **Limiter safety**: fed a 5x-over-range signal with injected `inf`/`nan`
  values, output stayed fully finite and bounded exactly to the configured
  threshold.
- **Validation benchmark**: 24/24 cases (8 presets × 3 base frequencies)
  within a 3.0Hz tolerance; worst-case error 0.31Hz.
- **Live hardware, passthrough**: 30 real seconds, zero underflows/overflows.
- **Live hardware, full engine**: 30 real seconds with pitch and formant
  shifting active, 0 bypasses, 0 dropped audio, ~25ms of total scheduling
  jitter across the whole run (0.08%). Confirmed by ear: pitch shift sounds
  correct, and the chunk-boundary clicking that an earlier architecture had
  is reduced to nearly nothing.

The full, unabridged version of every measurement above, including the
things that didn't work on the first try, is in `DECISIONS.md`.

## Known limitations

- A very small amount of audio-scheduling jitter (~0.08% of runtime,
  measured) is still audible on real hardware as an occasional faint click.
  This is real-time thread scheduling noise, not a DSP correctness issue:
  the phase-vocoder discontinuity that used to cause audible clicking has
  been fully eliminated and verified with deterministic, chunk-size-invariant
  tests. Increasing `queue_capacity` or `blocksize` would trade a little
  more latency for less of this jitter, if needed.
- This project intentionally does not include a GUI, file import/export, or
  network streaming. See `DECISIONS.md` for what was considered and kept
  out of scope.
