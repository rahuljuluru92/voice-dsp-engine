# Real-Time Voice DSP Engine — Project Instructions

## Project scope

Build a real-time pitch- and formant-shifting voice engine from scratch in Python.

The engine must use a short-time Fourier transform (STFT) phase vocoder with true instantaneous-frequency phase reconstruction. Pitch and formant controls must be independent. Formant movement should use cepstral-envelope warping. The live processing chain must include a biquad high-pass filter, noise gate, and limiter, all running inside a low-latency microphone callback through PortAudio.

The project is intentionally developed over multiple working sessions from **August 15, 2026 through August 23, 2026**. The Git history should show a clear progression of implementation, testing, integration, validation, and documentation across that period.

Before beginning work in any session, read `ASSUMPTIONS.md` and `DECISIONS.md`. Update those files as decisions and assumptions are made; do not reconstruct them retrospectively.

## Development rules

1. Work through the stages in order. Do not begin a later stage until the previous stage is implemented, runs without crashing, has its required tests passing, and has been committed.

2. Each meaningful completed stage or milestone should have its own descriptive commit. Do not wait until the end of the project to create one large commit.

3. The project should reflect genuine multi-day development. The intended schedule runs from August 15 through August 23, with work distributed across those dates as described in the Git commit schedule below.

4. Never fabricate measurements or test results. If a validation script produces a number, record that number exactly in `DECISIONS.md` or the README. Do not change code merely to produce a more attractive result, and never describe a test as passing unless it was actually run.

5. Record every non-trivial technical decision in `DECISIONS.md` when the decision is made. This includes library selections, algorithm variants, parameter defaults, architecture decisions, and other implementation choices.

6. Record every assumption in `ASSUMPTIONS.md` when it is made. If an assumption later proves incorrect, retain the original entry and add a follow-up explaining what changed and why.

7. Standard scientific Python dependencies such as `numpy`, `scipy`, `sounddevice`, and `pytest` may be installed without additional approval. Ask before installing unusual dependencies.

8. Stay within the defined project scope. Do not add a GUI, web frontend, multi-format audio import/export, network streaming, or other unlisted features. If an out-of-scope idea would materially improve the project, document it under `Considered but out of scope` in `DECISIONS.md` and ask before implementing it.

9. If a stage's success criterion fails, do not weaken the criterion simply to make the stage pass. Debug the implementation first. If the problem cannot be resolved after reasonable effort, document the exact failure and what was attempted before deciding how to proceed.

10. At the end of each working session, update the `## Session log` section at the top of `ASSUMPTIONS.md` with what was completed, what remains, and the next intended step. The next session must begin by reading that note.

## Environment setup

Complete the following setup during the first working session:

1. Create and activate the virtual environment with `python3 -m venv venv`.

2. Install dependencies as they become necessary.

3. Pin the versions actually used by the project in `requirements.txt`. If `pip freeze` is used, review the resulting file and retain only dependencies that the project actually uses.

4. Record the Python version used in `DECISIONS.md`.

5. Use `sounddevice` for PortAudio access unless there is a documented technical reason to choose `pyaudio`. Record that decision in `DECISIONS.md`.

Do not re-run `git init`; the repository is already initialized.

## Testing requirements

From Stage 2 onward, every stage must have real automated tests using `pytest`. Tests belong under `tests/` and should mirror the source structure where practical.

Stage 2 must include an STFT round-trip test that reconstructs a source signal within a documented numerical tolerance. It must also include a pitch-shift test using a synthetic sine wave at a known frequency and verify that the measured output frequency is within the documented tolerance of the expected shifted frequency.

Stage 3 must include independent-control tests. A formant-only shift must leave the fundamental frequency unchanged, while a pitch-only shift must leave the formant-envelope peak locations unchanged. These tests are required to demonstrate that pitch and formant controls are actually independent.

Stage 4 must test the biquad filter against a known reference response, such as attenuation at a selected frequency. The limiter must also be tested with deliberately clipped or over-range synthetic input and must produce finite output within the configured bounds.

Stage 5 must test the bounded queue by feeding it faster than it is drained and verifying that it never exceeds its configured capacity and that the oldest entries are discarded as designed. The bypass guard must be tested by forcing the processing path to fail and verifying that the output is silence rather than raw microphone input.

Stage 6 must provide the broader end-to-end synthetic-tone validation benchmark. This is distinct from the unit tests. It must run across the supported presets and record actual measured output frequencies and errors.

Before every commit, run the relevant tests and the full `pytest` suite where practical. Do not commit with known failing tests. If a test is flaky, fix the underlying problem or remove it for a documented reason; never silently skip it.

## Build stages

### Stage 1 — Audio I/O passthrough

Open microphone input through PortAudio and route the input directly to output through a low-latency callback. There must be no DSP in this stage.

Success criteria:

- The live path runs for at least 30 seconds without crashing.
- There is no obvious audible glitching.
- The measured callback latency is recorded.
- The Python version and audio-library choice are recorded in `DECISIONS.md`.
- The completed stage is committed.

### Stage 2 — STFT phase-vocoder core

Implement STFT framing, windowing, overlap-add, and inverse-STFT reconstruction. Then implement pitch shifting using phase-vocoder processing with true instantaneous-frequency phase reconstruction rather than naive waveform resampling.

Formant processing remains untouched at this stage.

Success criteria:

- STFT round-trip reconstruction passes its numerical tolerance test.
- A known synthetic sine wave can be shifted by a known number of semitones.
- The measured output frequency falls within the documented tolerance.
- The complete test suite passes.
- The completed stage is committed.

### Stage 3 — Independent formant control

Add cepstral-envelope analysis and warping so that the formant envelope can be shifted independently of the fundamental frequency.

Success criteria:

- Formant-only processing preserves the fundamental frequency within the documented tolerance.
- Pitch-only processing preserves the relevant formant-envelope peak locations within the documented tolerance.
- The independence tests pass.
- The complete test suite passes.
- The completed stage is committed.

### Stage 4 — Signal-processing chain

Add the runtime signal chain:

- Biquad high-pass filter.
- Noise gate.
- Limiter.

Integrate the chain into the callback processing path without introducing unbounded or non-finite output.

Success criteria:

- The filter response matches its documented reference within tolerance.
- The limiter contains deliberately over-range input.
- No NaN or Inf values are produced.
- The complete test suite passes.
- The completed stage is committed.

### Stage 5 — Runtime robustness

Implement bounded drop-oldest input/output queues so latency cannot grow without bound when processing cannot keep up.

Add a bypass guard that emits silence rather than raw microphone input if the processing path raises an exception or falls behind beyond the configured safety threshold.

Success criteria:

- Queue capacity is strictly bounded.
- Oldest entries are dropped when capacity is exceeded.
- The bypass path outputs silence on processing failure.
- The relevant robustness tests pass.
- The complete test suite passes.
- The completed stage is committed.

### Stage 6 — Validation suite

Build a reproducible synthetic-tone benchmark covering all supported presets and relevant pitch/formant combinations.

For each case, measure the resulting output frequency and compare it with the expected value. Record actual numbers rather than only reporting pass/fail status.

Success criteria:

- Every supported preset is exercised.
- Expected and measured frequencies are recorded.
- Frequency errors and tolerances are documented.
- The validation command is reproducible.
- Unit tests and validation tests remain separately runnable.
- The validation work and verified results are committed.

### Stage 7 — README and final verification

Write the README last, after implementation and validation numbers are available.

The README must contain:

- What the project does.
- Environment setup instructions.
- Virtual-environment instructions.
- `requirements.txt` usage.
- Any required PortAudio system dependency.
- How to run the live engine.
- How to run the unit tests.
- How to run the validation suite.
- Actual measured validation results taken from `DECISIONS.md`.
- No placeholder measurements.

Success criteria:

- The README describes the implementation that actually exists.
- All commands documented in the README are verified.
- The complete test suite passes.
- The validation suite runs successfully.
- The final documentation commit is made on August 23, 2026.

## Git commit schedule: August 15–23, 2026

The Git history should communicate a credible development progression across the nine-day window. The dates below define the intended milestones.

**August 15 — Project setup and audio I/O**

Complete the repository/environment setup and Stage 1. Establish the audio callback, microphone passthrough, dependency file, and initial project documentation. Record the Python version, PortAudio/library decisions, and measured callback behavior. Commit the completed setup and passthrough work.

Suggested commit message: `Set up audio passthrough`

**August 16 — STFT foundation**

Implement signal framing, windowing, inverse STFT, overlap-add, and the reconstruction test. Verify the reconstruction before proceeding to pitch shifting.

Suggested commit message: `Add STFT reconstruction`

**August 17 — Pitch shifting**

Complete the phase-vocoder pitch-shifting implementation and the synthetic sine-wave frequency test. Verify the measured result against the expected frequency.

Suggested commit message: `Add phase-vocoder pitch shifting`

**August 18 — Formant control**

Implement cepstral-envelope analysis and formant warping. Add and run the tests proving that pitch and formant controls remain independent.

Suggested commit message: `Add independent formant control`

**August 19 — Signal chain**

Implement the high-pass filter, noise gate, and limiter. Add the required response and overload tests and integrate the chain into the processing path.

Suggested commit message: `Add signal processing chain`

**August 20 — Robust runtime behavior**

Implement bounded queues and the failure/bypass guard. Test queue capacity, drop-oldest behavior, exception handling, and silence-on-failure behavior.

Suggested commit message: `Add bounded runtime queues`

**August 21 — Integration and stabilization**

Integrate the completed stages into the live callback path. Run the full test suite, fix genuine integration issues, and make a focused stabilization commit if required.

Suggested commit message: `Stabilize live processing`

**August 22 — Validation**

Implement the end-to-end synthetic-tone validation suite, run it across the supported presets, and record the actual measurements in `DECISIONS.md`.

Suggested commit message: `Add validation benchmark`

**August 23 — Documentation and final verification**

Complete the README using verified implementation details and real measurements. Run the full unit-test suite and validation suite again. Review the repository for consistency and commit the final documentation and cleanup.

Suggested commit message: `Document verified project results`

If a day's work genuinely produces multiple independent milestones, more than one commit is acceptable. Conversely, do not manufacture commits simply to populate every date. The history must describe real work.

## Git history rules

Git metadata must represent when the work actually happened.

Do not:

- Set `GIT_AUTHOR_DATE`.
- Set `GIT_COMMITTER_DATE`.
- Use `git commit --date`.
- Rewrite existing commits solely to make their dates fit the schedule.
- Force-push rewritten history.
- Skip commit hooks.
- Add fake commits whose only purpose is to make the contribution history look busier.
- Claim that work was completed on a date when it was not actually completed.

Use the existing global `git config user.name` and `git config user.email`. Do not override identity on a per-commit basis.

Commit only the files relevant to the milestone. Avoid blanket `git add -A` when unrelated scratch files are present.

Commit messages must be plain, descriptive, and written in imperative mood. Do not include `Co-Authored-By`, AI-attribution text, generated-by text, or similar trailers.

## Session-log expectations

`ASSUMPTIONS.md` must maintain a chronological `## Session log` section.

Each entry should state:

- The date of the session.
- The stage or milestone worked on.
- What was completed.
- What remains incomplete.
- The next intended step.

The log must describe the actual state of the repository. It must not be used to manufacture evidence of work that did not occur.

## Final definition of done

The project is complete when the repository contains a verified real-time voice DSP engine that:

- Captures microphone audio through PortAudio.
- Performs real-time pitch shifting using a phase vocoder with instantaneous-frequency phase reconstruction.
- Provides independent formant control using cepstral-envelope warping.
- Applies the configured high-pass filter, noise gate, and limiter.
- Maintains bounded runtime queues.
- Fails safely to silence when the processing path cannot safely produce output.
- Has a passing automated test suite.
- Has a reproducible validation suite.
- Contains actual measured validation results.
- Has a README that accurately describes the verified implementation and commands.
- Has a Git history showing the real progression of the work without fabricated timestamps or rewritten historical activity.

The final verification and documentation milestone is targeted for **August 23, 2026**.
