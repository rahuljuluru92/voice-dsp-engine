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

