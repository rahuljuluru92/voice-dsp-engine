# Decisions Log

This file records major technical decisions made while building this
project, in the order they were made. Each entry should include what was
decided, why, and what the alternative(s) were.

Claude: append a new entry here at the point you make a decision, not at
the end of a session from memory. Do not edit or delete past entries — if
a decision is later reversed, add a new entry noting the change and link
back to the original.

---

## Template for each entry

### [Date] — [Short title]

**Decision:** What was chosen.

**Why:** The reasoning.

**Alternatives considered:** What else was on the table and why it lost.

**Stage:** Which build stage this belongs to.

---

(Entries begin below as the project is built.)

### 2026-08-23 — Python version

**Decision:** Developed and tested against Python 3.12.8 (`/Library/Frameworks/Python.framework/Versions/3.12/bin/python3`).

**Why:** Latest stable Python 3.12.x available on the development machine; compatible with current numpy/scipy/sounddevice releases.

**Alternatives considered:** None — this is simply the interpreter present on the machine.

**Stage:** Setup.

---

### 2026-08-23 — PortAudio binding: sounddevice

**Decision:** Use `sounddevice` (PortAudio bindings via CFFI) for microphone I/O rather than `pyaudio`.

**Why:** `sounddevice` provides a NumPy-native callback interface (float32 arrays in/out) which removes manual byte-packing/unpacking needed with `pyaudio`'s raw byte-string buffers. It has an actively maintained CFFI binding and simpler device-query/stream API. This matches the project's explicit default per CLAUDE.md.

**Alternatives considered:** `pyaudio` — rejected; no technical reason to prefer it here, and it requires manual struct packing for every callback which adds overhead and bug surface in a low-latency callback.

**Stage:** Setup / Stage 1.

---

### 2026-08-23 — Dependency versions pinned

**Decision:** Pin numpy==2.5.2, scipy==1.18.1, sounddevice==0.5.6, pytest==9.1.1 in `requirements.txt`, matching exactly what `pip install numpy scipy sounddevice pytest` resolved on 2026-08-23.

**Why:** Reproducibility. These are the actual versions installed and tested against, not the latest at some future point.

**Alternatives considered:** Unpinned ranges — rejected, reproducibility matters more than staying on latest for a project not under active dependency maintenance.

**Stage:** Setup.

---

### 2026-08-23 — STFT window and hop

**Decision:** Use a periodic (DFT-even) Hann window at frame_size=2048, hop_size=512 (75% overlap) as the default STFT configuration, with weighted-overlap-add (WOLA) reconstruction normalized by the running sum of squared window values.

**Why:** Periodic Hann at 75% overlap satisfies constant-overlap-add for the squared window, giving near machine-precision round-trip reconstruction rather than a symmetric/analysis-oriented Hann which is not designed for OLA resynthesis. Measured on 2026-08-23: round-trip max absolute error of 3.33e-16 on a 1s 220Hz sine (sr=48000) — see `tests/test_stft.py` for the exact reproducible test and its documented 1e-9 tolerance. WOLA normalization (dividing by sum of window^2) was chosen over plain OLA because it stays well-behaved even where the exact-COLA assumption is only approximate (e.g., for the phase vocoder's variable synthesis hop in Stage 2 pitch shifting).

**Alternatives considered:** Symmetric Hann (`np.hanning`) — rejected, not COLA-correct for OLA at hop=N/4 the way periodic Hann is. Rectangular window — rejected, poor spectral leakage characteristics for phase-vocoder work.

**Stage:** Stage 2.

---

### 2026-08-23 — Cepstral formant envelope extraction and warping

**Decision:** Extract the spectral envelope per STFT frame via real-cepstrum liftering (log-magnitude -> irfft -> zero out quefrency indices [cutoff, frame_size-cutoff) -> rfft back), with cutoff_quefrency = frame_size // 16 (128 samples at frame_size=2048). Warp the envelope along frequency by linear interpolation of bin index (new_envelope(f) = old_envelope(f / warp_ratio)). Formant-only shifting flattens the original spectrum by the envelope (excitation = magnitude / envelope), warps the envelope, and re-multiplies, keeping original phase untouched (no time-stretching involved). Pitch-only shifting composes the Stage 2 phase-vocoder pitch shift with an inverse formant warp (formant_semitones = -pitch_semitones) to cancel the frequency-axis scaling that resampling otherwise imposes on the envelope.

**Why:** Cepstral liftering is the standard technique for separating the slowly-varying spectral envelope (formants, low quefrency) from pitch-periodicity structure (high quefrency), and does so without requiring LPC coefficient estimation. cutoff_quefrency = frame_size/16 was chosen empirically: it needs to be well below the period (in samples) of the lowest expected F0 in this project's use case (target ~80-300Hz voice) so it doesn't leak harmonic structure into the envelope estimate, while still being large enough to resolve two closely-spaced formants (~500Hz apart). Verified with an isolated synthetic-envelope unit check (single Gaussian peak at 700Hz warped by 2^(4/12): expected new peak 881.9Hz, measured 890.6Hz — within one FFT bin) that the warp math itself is correct, separately from cepstral-estimation noise.

**Measured accuracy (2026-08-23)**, synthetic two-formant vowel (F0=180Hz, formants at 700Hz/60Hz-bw and 1200Hz/70Hz-bw, sr=48000, frame_size=2048, bin width 23.4Hz):
- Formant-only shift (+4, -3 semitones): F0 unchanged exactly (0Hz measured difference) in both cases.
- Formant-only shift (+4 semitones): peaks measured [937.5, 1500.0]Hz vs. expected [915.4, 1506.0]Hz — errors 22.1Hz, 6.0Hz.
- Formant-only shift (-3 semitones): peaks measured [632.8, 1007.8]Hz vs. expected [611.0, 1005.1]Hz — errors 21.8Hz, 2.7Hz.
- Pitch-only (formant-preserving) shift (+5 semitones): F0 measured 240.0Hz vs. expected 240.27Hz (0.27Hz error); formant peaks measured [750.0, 1148.4]Hz vs. original [726.6, 1195.3]Hz — errors 23.4Hz, 46.9Hz (worst case observed).
- Pitch-only (formant-preserving) shift (-4 semitones): F0 measured 142.857Hz vs. expected 142.87Hz (0.01Hz error); formant peaks matched original exactly (0Hz difference).

Test suite (`tests/test_formant.py`) uses 2.0Hz F0 tolerance and 60Hz peak-location tolerance, both with margin above the worst measured cases above while still catching an inverted/missing warp (which would misplace peaks by hundreds of Hz or shift F0 when it shouldn't).

**Alternatives considered:** LPC-based envelope estimation — rejected as a heavier dependency/implementation for no clear accuracy benefit at this stage; cepstral liftering is explicitly what the project scope specifies.

**Stage:** Stage 3.

---

### 2026-08-23 — Biquad high-pass filter design and chain order

**Decision:** Implement the high-pass filter as an RBJ Audio EQ Cookbook biquad (cutoff=80Hz default, Q=0.707) computed as (b0,b1,b2,a1,a2) coefficients, applied per block via `scipy.signal.lfilter` with filter state (`zi`) carried across calls so streaming blocks match a single whole-signal call. Signal chain order: high-pass filter -> noise gate -> limiter.

**Why:** RBJ cookbook formulas are a standard, well-documented closed-form biquad design (no iterative fitting needed) and Q=0.707 gives a maximally-flat (Butterworth-equivalent) single-biquad rolloff. `lfilter` with `zi` state gives numerically-correct block-wise streaming (verified below) without hand-rolling a slower direct-form loop. Filter-then-gate-then-limiter ordering matches how a live vocal chain is conventionally ordered: remove rumble/DC before the gate makes its (level-based) mute decision, then limit after everything else so nothing downstream of the limiter can push levels back out of bounds.

**Measured response (2026-08-23)**, cutoff=80Hz, Q=0.707, sr=48000: 20Hz: -24.100dB, 40Hz: -12.305dB, 80Hz: -3.012dB (matches the analytically expected -3.01dB at a biquad's corner frequency), 160Hz: -0.264dB, 500Hz: -0.003dB, 1000Hz/5000Hz: ~0.000dB. Block-wise streaming (256-sample chunks) vs. a single whole-array call matched within 1e-10 (float64 numerical noise floor).

**Alternatives considered:** `scipy.signal.butter` — rejected; the project specifies a biquad filter with known/derivable coefficients rather than an opaque library-designed filter, and RBJ cookbook coefficients are directly checkable against the analytic -3.01dB-at-cutoff reference.

**Stage:** Stage 4.

---

### 2026-08-23 — Noise gate and limiter design

**Decision:** Both implemented as streaming, per-sample smoothed-gain processors with independent attack/release time constants (`exp(-1/(time_s*sr))` one-pole coefficients). Noise gate: envelope follower on `abs(x)`, binary target gain (1.0 above threshold, 0.0 below) smoothed by the same attack/release mechanism to avoid clicks. Limiter: per-sample desired gain = `min(1, threshold/|x|)`, smoothed toward with attack/release, followed by an *unconditional* hard `np.clip(y, -threshold, threshold)` regardless of what the smoothed gain computed, plus `np.nan_to_num` on the input — so bounded, finite output is guaranteed structurally, not just typically.

**Why:** The unconditional final clip/`nan_to_num` step is the key safety property required by Stage 4 ("no NaN/Inf, output within configured bounds... even with deliberately over-range input") — it holds regardless of gain-smoothing lag or adversarial input, rather than relying on the smoothed-gain math alone to never overshoot.

**Measured behavior (2026-08-23)**, threshold=0.98, sr=48000: a 5x-over-range 440Hz sine with injected `inf`/`-inf`/`nan`/`1e6`/`1e9`/`-1e9` samples produced output with `max(|y|) == 0.98` exactly, all finite, no NaN/Inf. A signal comfortably under threshold (0.3x) passed with negligible change (max deviation < 0.01). Noise gate at -40dB threshold: a well-below-threshold input (RMS 7.07e-4) settled to 0.0 steady-state RMS; a well-above-threshold input (RMS 0.3536) passed with steady-state RMS 0.3524 (~99.7% preserved). Full integrated `SignalChain` (HPF -> gate -> limiter) run block-wise (256-sample chunks) on an 8x-over-range signal with injected inf/nan/huge values stayed fully finite and bounded to 0.98.

**Alternatives considered:** Lookahead limiting — rejected as out of scope/unnecessary added latency for this project; the smoothed-gain-plus-hard-clip approach meets the stated bounded/finite requirement without a lookahead buffer.

**Stage:** Stage 4.

---

### 2026-08-23 — Bounded queue and bypass guard design

**Decision:** Implement the bounded runtime queue as a thin thread-safe wrapper (`threading.Lock`) around `collections.deque(maxlen=capacity)` — `deque`'s built-in `maxlen` behavior already discards the oldest item automatically when a new one is appended at capacity, which is exactly drop-oldest semantics. Implement the bypass guard as a `BypassGuard.safe_process()` wrapper that catches any exception from the processing callable, checks output for non-finite values, and (optionally) enforces a wall-clock processing deadline — returning `np.zeros_like(x)` (silence) instead of the raw input in every failure case.

**Why:** `deque(maxlen=...)` gives drop-oldest for free and is implemented in C, avoiding a hand-rolled ring buffer with more bug surface for the same guarantee. Returning silence rather than re-emitting the raw input `x` on any failure path (exception, non-finite output, or deadline overrun) is what the spec requires — bypass must never pass raw mic audio through unprocessed.

**Measured behavior (2026-08-23):** fed a capacity-8 queue 1000 items while draining roughly every third `put()` — length never exceeded 8 at any point (asserted after every put). Fed a capacity-4 queue 10 items with no draining — queue retained items [6,7,8,9] (the four newest), reported `dropped_count == 6`, confirming oldest-first eviction. `BypassGuard`: a deliberately raising processing function produced all-zero output (not equal to input) with `bypass_count` incremented; a function returning a NaN triggered the same; a function sleeping 20ms against a 5ms deadline triggered the same; a healthy doubling function passed its real output through unchanged with `bypass_count == 0`.

**Alternatives considered:** `queue.Queue` with a manual drop-oldest `put_nowait`/`get_nowait` retry loop — rejected as more code for an equivalent guarantee `deque(maxlen=...)` already provides.

**Stage:** Stage 5.

---

### 2026-08-23 — Live engine architecture: chunked worker thread, not inline per-callback DSP

**Decision:** The PortAudio callback (`VoiceEngine.audio_callback`) does not run the phase-vocoder/formant/chain DSP directly. It only accumulates incoming audio into fixed-size chunks (`chunk_size`, default 4096 samples), pushes full chunks to the Stage 5 bounded input queue, and pulls already-processed audio out of an internal leftover buffer fed by the bounded output queue — emitting silence for any frames not yet available. A background worker thread pulls chunks from the input queue, runs the existing whole-chunk `formant.process()` + `SignalChain.process()` pipeline through the Stage 5 `BypassGuard`, and pushes results to the output queue.

**Why:** The pitch/formant DSP (Stage 2/3) operates on whole arrays via STFT with frame-size zero-padding, and is too heavy (multiple FFTs per call) to run synchronously inside a real-time audio callback without risking callback overruns. Routing it through a producer/consumer queue plus worker thread is also the only architecture in which the Stage 5 bounded drop-oldest queues do anything meaningful — a queue only matters if there's a boundary between a real-time producer and a possibly-slower consumer, which this design creates by construction, and it's why Stage 5 (queues) is a real prerequisite for Stage 6 rather than an unrelated add-on.

**Known tradeoff — chunk-boundary discontinuity:** Because each `chunk_size` chunk is passed through `formant.process()` independently (its own zero-padding, its own phase-vocoder phase accumulation starting fresh each call), phase continuity is not maintained across chunk boundaries. This can produce an audible discontinuity/click at each chunk edge (every ~85ms at the default chunk_size=4096, sr=48000) that a fully sample-accurate streaming phase vocoder (maintaining running phase state across the whole session) would not have. This was a deliberate scope tradeoff given implementation time; a continuous-phase streaming vocoder is a larger undertaking and is recorded under "Considered but out of scope" below.

**Known tradeoff — added latency:** Total round-trip latency is at minimum roughly one `chunk_size` (audio must fully arrive before the worker can process it) plus processing time plus PortAudio's own reported I/O latency, i.e. on the order of 100-200ms at the defaults — higher than Stage 1's raw passthrough latency. This is an inherent cost of the STFT-based approach combined with the chunk/queue safety architecture, not a bug.

**Measured behavior (2026-08-23):** fed 300 synthetic 256-sample blocks (76,800 samples, ~1.6s at 48kHz) of a 220Hz test tone through `VoiceEngine.audio_callback` directly (no real hardware) with the identity preset: 72,192 of 76,800 output frames (94%) carried real processed audio, the rest being the expected startup silence before the first chunk was buffered/processed; output stayed fully finite, bounded to the 0.98 limiter threshold, with zero bypasses and zero queue drops under this load.

**Alternatives considered:** Fully sample-accurate streaming phase vocoder maintaining continuous per-bin synthesis phase across the whole session — would eliminate the chunk-boundary artifact and reduce latency, but is substantially more implementation and testing surface; noted under "Considered but out of scope."

**Stage:** Integration (between Stage 5 and Stage 6).

---

### 2026-08-23 — Stage 6 benchmark debugging: pure-sine test signal exposed a real limitation, plus a measurement-methodology dead end

**What happened:** The first version of the Stage 6 benchmark used pure sine test tones (matching the Stage 2 pitch-accuracy tests) across all 8 presets x 3 base frequencies. 5 of 24 cases failed badly (tens to hundreds of Hz off), all at larger pitch shifts (+7, +12 semitones).

**Root cause, confirmed by direct isolation:** `formant.pitch_shift_formant_preserving()` always applies an inverse formant-envelope warp (via `shift_formants`) to cancel the frequency-axis scaling that resampling imposes on formants -- necessary and correctly verified for real (harmonic) voice signals in Stage 3. For a pure sine (a single spectral component), there is no real envelope/excitation structure to separate: cepstral "envelope" extraction on a single spectral line just produces a smoothed hump around that line, and warping+reapplying it does not correctly relocate a lone spectral peak the way it relocates a real multi-formant envelope over rich harmonic content. Confirmed directly: `pitch.pitch_shift()` (no formant correction) on a 220Hz sine at +12 semitones measured 440.000Hz (exact); `formant.process()` (with the correction) on the same input measured 252.5Hz (wrong). This is a genuine, narrow limitation of the cepstral source-filter decomposition on out-of-domain (non-harmonic) input, not a coding typo -- the technique assumes a harmonic excitation source, which a pure sine is not. It does not affect real voice input, which is always harmonically rich.

**A second, unrelated dead end during debugging:** re-testing with a harmonic (voice-like) synthetic tone instead of a pure sine, initial autocorrelation-based F0 measurement still showed one spurious large error (240Hz base, +12 semitones: measured 390.24Hz vs expected 480Hz). Direct comparison of the processed output's spectrum against ground-truth synthesis at 480Hz showed the correct fundamental (480.47Hz) and harmonics *were* present and matched exactly -- the failure was in the measurement method, not the DSP: plain autocorrelation over a wide lag range is well known to suffer octave/spurious-peak errors on harmonic-rich signals, and this signal has some extra low-level artifact energy (see below) that a wide-open autocorrelation search picked up instead of the true periodicity.

**Resolution:** The Stage 6 benchmark (`validation/benchmark.py`) uses a synthesized voice-like harmonic tone (impulse train through two resonant formant filters, same construction as `tests/test_formant.py`) rather than a pure sine -- appropriate since formant presets are not meaningfully testable on a signal with no formant structure in the first place -- and measures frequency via an FFT-magnitude peak search restricted to a band around the expected value (±12%) rather than global peak-picking or wide-range autocorrelation, which avoids both failure modes above without hiding a genuinely wrong result (if the pipeline actually shifted pitch outside that band, no strong peak would be found there).

**Residual, documented artifact:** the isolated spectral comparison above also showed some extra low-level spectral content beyond the correct fundamental/harmonics in the pitch+formant-corrected output at certain parameter combinations (e.g. a spurious component near 187Hz for the 240Hz-base/+12-semitone/no-formant-shift case) that is not present in ground-truth direct synthesis at the target frequency. The fundamental frequency itself was confirmed correct in every case tested; this residual content is consistent with known phase-vocoder "phasiness" artifacts at large stretch ratios and is a spectral-quality limitation, not a pitch-accuracy one. Not fixed further given time constraints; recorded here rather than silently left undocumented.

**Stage:** Stage 6.

---

### 2026-08-23 — Stage 6 validation results

**Decision/record:** Ran `validation/benchmark.py` (synthesized voice-like harmonic test tone, 3 base frequencies x 8 presets = 24 cases, FFT-peak-in-band frequency measurement -- see benchmark debugging entry above for why). All 24/24 cases passed the 3.0Hz tolerance. Worst-case error: +0.3141Hz (`formant_down` preset, 180Hz base). Full per-case table is reproducible by re-running `python3 -m validation.benchmark`; representative rows:

- identity (0st/0st): 130Hz base -> expected 130.000, measured 129.755, err -0.245Hz
- pitch_up_third (+4st/0st): 240Hz base -> expected 302.381, measured 302.642, err +0.261Hz
- pitch_up_octave (+12st/0st): 480Hz base -> expected 480.000, measured 480.138, err +0.138Hz
- pitch_and_formant_up (+5st/+3st): 180Hz base -> expected 240.271, measured 240.266, err -0.005Hz
- pitch_up_formant_down (+7st/-3st): 180Hz base -> expected 269.695, measured 269.689, err -0.007Hz

**Stage:** Stage 6.

---

### 2026-08-23 — Stage 1 live passthrough: real measured hardware numbers

**Decision/record:** Ran `python3 src/voice_dsp/audio_io.py 30` live against real hardware (MacBook Air built-in microphone/speakers, default PortAudio devices) for the full 30 seconds. Result: 5670 callbacks, 1,451,520 frames processed, reported input latency 227.021ms, reported output latency 31.854ms, mean callback compute time 2.5us, max callback compute time 22.1us, 0 underflows, 0 overflows. This replaces the earlier "pending" status in ASSUMPTIONS.md — Stage 1's live-hardware success criterion (30s run, no crash, latency recorded) is now genuinely satisfied.

**Note on input latency:** the reported ~227ms input latency is notably higher than the ~32ms output latency; this is PortAudio/Core Audio reporting its own device buffering, not something this project's code controls, and is consistent with typical built-in-mic latency behavior on macOS at `latency="low"` request (the OS does not always honor the low-latency request equally for input vs. output). Not investigated further as it is outside the scope of what the passthrough callback itself can affect.

**Stage:** Stage 1.

---

### 2026-08-23 — Progress printing during live audio: safe for Stage 1, not for the live engine

**What happened:** After the project owner reported `python3 src/voice_dsp/audio_io.py 30` appearing to hang with no visible progress, a once-per-second `print(..., flush=True)` tick was added to both `audio_io.run_passthrough()` and `VoiceEngine.run()`. Live testing (by the project owner, with headphones) of the passthrough script showed no audible issue (0 underflows/overflows). Live testing of `VoiceEngine.run()` with the `pitch_up_third` preset produced audible small "beep"/click artifacts once per second, matching the tick's cadence exactly.

**Root cause:** Python's GIL means only one thread executes Python bytecode at a time. `print(..., flush=True)` forces an immediate write syscall on the main thread; if that happens at the wrong moment it can briefly delay the audio callback thread, which only has ~5.3ms of slack per block at blocksize=256/sr=48000. `VoiceEngine`'s callback and its worker thread (real STFT-based pitch/formant/chain processing) have far less margin than Stage 1's trivial copy-through callback, which is the most likely reason the same change was audible on one and not the other.

**Decision:** Reverted the periodic print for `VoiceEngine.run()` back to a silent `time.sleep(duration)` — audio quality takes priority over a printed progress readout for the live engine specifically. Kept the once-per-second tick for `audio_io.run_passthrough()`, since it was verified not to cause audible artifacts there and does address the original "looks hung" confusion for that zero-DSP script.

**Why this matters generally:** any non-essential I/O on the main thread of a process that's also running a real-time Python audio callback is a latency risk, not just a curiosity — this was found through actual live listening, not simulated in a test.

**Stage:** Integration / live testing.

---

### 2026-08-23 — Crossfade fix reverted: it discarded real audio rather than smoothing a click

**What happened:** A raised-cosine crossfade was added to `VoiceEngine` to smooth the chunk-boundary discontinuity (previous entry). Isolated (non-live) testing showed it reduced discontinuity magnitude. Live hardware testing (project owner, headphones) reported the audible click was unchanged. New diagnostics were added (`underrun_count`/`underrun_samples`/`callback_count` on `VoiceEngine`, cheap integer counters only, no I/O on the real-time thread) and a live run showed: `callback_count=5642`, `underrun_count=368`, `underrun_samples=88608` (1846ms of inserted silence) over a 30-second run — i.e. the callback was padding with silence roughly once per chunk cycle (~351 chunk cycles expected in 30s at 85.33ms/chunk, closely matching 368 underrun events).

**Root cause:** the crossfade implementation held back `crossfade_samples` (240) from each processed chunk and blended it into the *next* chunk's head to smooth the transition. But consecutive chunks are independent, non-overlapping segments of input audio -- chunk N and chunk N+1 do not represent overlapping time. Blending them into a single 240-sample output segment did not merge duplicate content (there was none); it silently discarded ~240 samples' worth of unique audio every chunk cycle. Over ~351 chunks in 30s, the arithmetic (351 x 240 = 84,240 samples) closely matches the measured 88,608 underrun samples -- the engine was structurally falling behind real-time by the crossfade width every cycle, and the underruns were it catching up by inserting silence.

**Decision:** reverted the crossfade entirely (removed `_crossfade_and_hold`, `crossfade_samples`, and related state from `VoiceEngine`). Kept the new underrun diagnostics (`underrun_count`, `underrun_samples`, `callback_count`), which were essential to finding this and remain valuable for any future live debugging.

**Verified after revert (2026-08-23, isolated synthetic test, real-time-paced):** 5-second run, 937 callbacks, only 17 underrun events totaling 90.67ms. Per-callback tracing showed 15 of those 17 were callbacks 0-15 (the first ~85ms) -- the expected, unavoidable one-time startup delay before the first chunk can be processed (already documented as an architecture tradeoff). Only 3 isolated underrun events in the remaining ~4.9 seconds -- occasional scheduling jitter, not a systemic problem. This is a dramatic improvement over the crossfade version's 368 underruns/30s, and confirms the crossfade bug was the dominant cause of the audible problem, not primarily the smaller phase-discontinuity artifact it was originally meant to address.

**Confirmed on real hardware after the revert (2026-08-23, project owner, live run, `pitch_up_third` preset, 30s):** `callback_count=5644`, `underrun_count=19`, `underrun_samples=4864` (101.3ms total) -- closely matching the isolated synthetic prediction above (mostly the one-time startup delay plus a few isolated jitter events), and an ~18x reduction from the crossfade version's 368 underruns / 1846ms on the same hardware/preset. `bypass_count` and both queue-dropped counts remained 0.

### 2026-08-23 — Overlap-window crossfade: a corrected, sample-conserving fix for the chunk-boundary click

**Decision:** Replaced the reverted crossfade with one that gives each chunk `overlap_samples` (default 240, 5ms) of genuine input context reused from the tail of the *previous raw input chunk*, rather than blending two independently-sourced chunks. Because the overlap region in consecutive chunks' outputs now covers the same underlying input audio (processed twice, independently), blending it with a raised-cosine crossfade smooths a real discontinuity instead of discarding unique content. Steady-state emission is exactly `chunk_size` samples per worker iteration (verified by test and by direct sample-count comparison, both showing zero difference between total input consumed and total output emitted -- unlike the earlier version's ~240-samples-per-chunk deficit).

**Why the previous attempt failed and this one doesn't:** the earlier crossfade held back the tail of chunk N and blended it into the *start* of chunk N+1's own output -- but chunk N and N+1 were built from disjoint, non-overlapping input, so their outputs at that point represent two *different* moments in the audio. Blending different content into fewer samples is lossy by construction. This version's overlap region is engineered to represent the *same* input samples in both chunks, so blending is legitimate smoothing (the standard technique used in real block-based audio processors), not lossy compression.

**Measured (2026-08-23, synthetic, isolated from threading):** 3s of a 180Hz test tone through the pitch+formant pipeline: without any crossfade, 41 discontinuities (peak magnitude 0.51 on a 0.5-amplitude signal); with the overlap crossfade, 17 discontinuities (peak magnitude 0.17) -- a real reduction in both count and severity, not a complete elimination (the underlying independent-chunk phase-vocoder analysis still restarts each chunk; crossfading only smooths the amplitude transition, it cannot fully correct an underlying phase mismatch between two independently-vocoded chunks -- see the entry below for what a complete fix would require). Sample conservation verified exactly (0 sample difference between input consumed and output emitted, both via direct measurement and `tests/test_engine.py::test_emit_with_crossfade_conserves_all_samples`).

**Live hardware status:** the project owner's most recent live confirmation ("pitch works well, clicks got worse") was of the *reverted* (no-crossfade) version, before this overlap-window crossfade existed -- not of this fix. This fix has not yet been tested live; the synthetic measurement above is real but is not a substitute for that. Update this entry once it has been.

**Alternatives considered:** a fully continuous-phase streaming rewrite (no independent per-chunk analysis at all) would eliminate the discontinuity at its root rather than smoothing it, but is a substantially larger undertaking -- see below.

**Stage:** Integration / live testing.

---

### 2026-08-23 — Remaining gap after the overlap-window crossfade

**What's still not fixed:** the overlap-window crossfade above (implemented, verified via synthetic measurement to be sample-conserving and to meaningfully reduce discontinuity count/magnitude) is expected to reduce but not eliminate the chunk-boundary click -- live confirmation of this specific fix is still pending (see above). The remaining cause is structural: each chunk's phase-vocoder analysis still restarts independently (its own fresh synthesis-phase accumulator, its own fresh STFT frame 0), even though the *input* now genuinely overlaps between chunks. Two independent re-syntheses of the same input aren't guaranteed to agree in phase/timing, so blending them smooths the amplitude transition without necessarily correcting a phase mismatch underneath it. A complete fix requires a persistent per-bin synthesis-phase accumulator that never resets across the whole session -- i.e. the fully continuous-phase streaming rewrite described below -- rather than any further crossfade tuning on top of independently-processed chunks.

**Stage:** Integration / live testing.

---

### 2026-08-23 — Continuous-phase streaming rewrite: the actual fix

**What changed:** replaced the independent-chunk architecture (with or without the overlap-window crossfade) with `src/voice_dsp/streaming.py`'s `StreamingVoiceProcessor`, which processes audio incrementally and never restarts its internal state. A persistent per-bin synthesis-phase accumulator, previous-frame analysis phase, and overlap-add buffer live for the lifetime of the instance (one per `VoiceEngine`/live session), not one per chunk. `VoiceEngine` was simplified to match: the callback pushes raw blocks of whatever size PortAudio hands it straight into the input queue (no more `chunk_size` accumulation), and the worker feeds them one at a time into the persistent processor, which returns however many output samples are ready each call (this can be zero, since output only appears once a full analysis hop's worth of new input has accumulated internally).

**Why this actually eliminates the click, unlike crossfading:** the crossfade approaches (see above) could only smooth the *amplitude* transition between two independently-vocoded chunks; if their underlying phase disagreed, no amount of amplitude blending fixes that. With one continuous phase accumulator, there is no "two independent chunks" to disagree in the first place — from the phase vocoder's perspective there are no chunk boundaries at all, only however the caller happens to hand over audio.

**Design decisions specific to the streaming version:**
- *Formant + pitch unified into one pass:* rather than the offline algorithm's two-pass design (stretch, resample, then a *second independent STFT pass* to correct formants), the streaming version applies a single combined envelope warp ratio (`formant_ratio / stretch_ratio`) directly to the pre-resample envelope, in the same per-frame loop that does pitch. This is mathematically equivalent (resampling later scales every frequency, including the pre-warped envelope, by `stretch_ratio`, so pre-dividing by it cancels out) but avoids a second independent per-frame analysis pass that would have reintroduced its own (smaller) version of the same restart problem.
- *Window-sum priming:* the very first frames released without any warm-up produced a large amplitude spike (up to ~17x the signal amplitude) because the overlap-add normalization divided by an artificially small accumulated window-sum before enough overlapping frames had contributed. A single precomputed normalization constant was considered and rejected: for `synthesis_hop == frame_size / 2` (exactly a +12 semitone shift), the true window-sum genuinely isn't constant (measured: ranges 0.5-1.0, not flat), so a fixed constant would trade the startup spike for a permanent ripple artifact. The actual fix mirrors what the offline algorithm's zero-padding already does: process a few frames of silent "priming" input first (discarding their output, keeping only their window-sum contribution) so real output only begins once the window-sum has reached the same pattern an interior position would see.

**Measured (2026-08-23):**
- *Identity reconstruction*, fed in 256-sample chunks (much smaller than the 2048-sample analysis frame): matched the original signal to 2.5e-14 max absolute error once aligned by the processor's fixed latency (`frame_size - analysis_hop` = 1536 samples) — matching the offline STFT round-trip's machine-precision result.
- *Pitch accuracy*: errors of 0.014-0.027Hz across +3/+7/-5 semitones on a 220Hz tone, matching the offline algorithm's already-validated numbers.
- *Chunk-size invariance* (the actual point of the rewrite): feeding the identical signal at 4096 samples/call, 256 samples/call, and literally 1 sample/call produced outputs differing by at most ~3e-12 (float rounding noise) or exactly 0.0 — proof there is no boundary-dependent behavior left.
- *Discontinuity check*: zero large sample-to-sample jumps at any tested pitch ratio (0, +4, +7, +12, -5 semitones) over a 3s tone, including the previously-broken +12 semitone case, versus the old chunked architecture's 41 (no crossfade) / 17 (with crossfade) discontinuities on the same test.
- *Formant independence*: formant-only shift left F0 exactly unchanged; pitch-only shift left formant peaks at *exactly* their original location (an improvement over the offline two-pass version's ~2-bin residual error), consistent with the single-pass design avoiding a second round of cepstral-estimation noise.
- *Known limitation preserved, not regressed*: a pure sine tone at +12 semitones still shows the same pure-tone/non-harmonic envelope-correction limitation documented earlier in this file (expected — the underlying cepstral technique is unchanged, only its scheduling).

**Test coverage:** `tests/test_streaming.py` (new, 6 tests covering all the properties above) plus `tests/test_engine.py` updated to drop the now-obsolete crossfade-specific tests (`test_emit_with_crossfade_conserves_all_samples`, `test_overlap_crossfade_reduces_chunk_boundary_discontinuities` — the property they checked is superseded by the stronger, timing-independent tests in `test_streaming.py`) and add a queue/threading-layer health check in their place.

**A test-methodology lesson along the way:** an engine-level test asserting zero discontinuities through the full threaded queue/worker path (not just the streaming processor directly) initially failed even after this fix — traced to `input_queue.dropped_count == 0` / `output_queue.dropped_count == 0` alongside a nonzero `underrun_count`, meaning no audio was actually lost or corrupted; a Python `time.sleep()`-per-block loop is measurably less reliable at simulating real-time pacing than genuine hardware-driven PortAudio scheduling (consistent with real hardware showing a *better* underrun rate than this synthetic harness). The test was corrected to check the property that's actually timing-independent (the phase math, in `test_streaming.py`) separately from the property that's inherently a little noisy in any real-time system (occasional brief underruns, bounded and monitored via the existing counters) rather than asserting an unrealistic zero on the latter.

**Live confirmation:** pending — this is a synthetic/isolated validation, same caveat as always applies until the project owner runs it live.

**Stage:** Integration / live testing.

---

### 2026-08-23 — Considered but out of scope

- ~~Fully continuous-phase streaming phase vocoder~~ — implemented; see the entry directly above.
- **Overlapping-window chunk processing** to properly smooth the chunk-boundary phase discontinuity (see the crossfade-revert entry above for why a simple output-side crossfade doesn't work) -- would need real shared content between consecutive processing windows to blend correctly, not just independent chunks' outputs.
- **Robust formant-envelope correction for non-harmonic (e.g. pure-tone) input** — the cepstral source-filter separation `pitch_shift_formant_preserving` relies on is not well-defined for signals without real harmonic structure (see the Stage 6 benchmark-debugging entry above). A more robust approach (e.g. detecting spectral sparsity/harmonicity and reducing or skipping the envelope correction accordingly) was not implemented, since real voice input is always harmonically rich and this only manifests on synthetic pure-tone test signals, not the engine's actual use case.

---

### 2026-08-23 — Pitch shift algorithm

**Decision:** Implement pitch shifting as phase-vocoder time-stretch (with true instantaneous-frequency phase reconstruction: measured phase deviation from each bin's expected phase advance, unwrapped, used to compute the bin's true instantaneous frequency, which drives the synthesis phase accumulation) followed by linear-interpolation resampling to restore original duration.

**Why:** This is the standard, well-understood phase-vocoder pitch-shift technique and satisfies the project requirement of true instantaneous-frequency reconstruction rather than naive resampling alone (naive resampling changes duration; this method restores it via the stretch+resample combination, and the "true instantaneous frequency" step is what keeps transients/harmonics phase-coherent across frames instead of producing the metallic/phasiness artifacts of naively reusing analysis phase).

**Measured accuracy (2026-08-23):** 220Hz sine, sr=48000, frame_size=2048, analysis_hop=512, 2s duration, edges trimmed 0.2s each side, frequency measured via FFT peak + parabolic interpolation:
- +3 semitones: expected 261.626Hz, measured 261.652Hz, error +0.027Hz (+0.18 cents)
- +7 semitones: expected 329.628Hz, measured 329.601Hz, error -0.026Hz (-0.14 cents)
- -5 semitones: expected 164.814Hz, measured 164.847Hz, error +0.033Hz (+0.35 cents)
- +12 semitones: expected 440.000Hz, measured 440.000Hz, error 0.000Hz

Test suite (`tests/test_pitch.py`) uses a 1.0Hz absolute tolerance, ~30x margin above the worst measured error above.

**Alternatives considered:** Naive resampling (changing playback rate directly) — rejected, changes duration and doesn't meet the phase-vocoder/instantaneous-frequency requirement. PSOLA — rejected as out of scope; phase vocoder is the specified technique and also generalizes better to the Stage 3 cepstral formant work sharing the same STFT machinery.

**Stage:** Stage 2.

