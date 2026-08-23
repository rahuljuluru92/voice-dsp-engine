# Assumptions Log

This file records things Claude assumed while building this project because
they weren't explicitly specified. The point is so the project owner can
scan this file and correct anything that was assumed wrong, without having
to re-read every commit.

Claude: append a new entry the moment you make an assumption, not
retroactively. If an assumption is later confirmed or corrected by the
project owner, add a follow-up note under the original entry rather than
deleting it.

---

## Template for each entry

### [Date] — [Short title]

**Assumed:** What was assumed.

**Why:** Why this seemed like the reasonable default given no explicit
instruction.

**Risk if wrong:** What breaks or needs rework if this assumption turns out
to be incorrect.

**Status:** Unconfirmed / Confirmed / Corrected (update this as it's
resolved).

---

## Session log

Claude: at the end of every session, add an entry here (most recent on
top) with the date, which stage was in progress, what got done, and what
the very next step is. At the start of the next session, read the most
recent entry here first, before doing anything else.

### 2026-08-23 — Session 1

**Stage in progress:** All 7 stages implemented in one session (see project-history note below); Stage 7 (README/final verification) completed.

**Done this session:** Environment setup (venv, pinned requirements, Python 3.12.8, sounddevice decision). Stage 1 audio passthrough (callback logic verified synthetically; live hardware run deferred to project owner, see entry above). Stage 2 STFT core + phase-vocoder pitch shifting (8/8 tests passing). Stage 3 independent cepstral formant control (12/12 tests passing). Stage 4 biquad HPF + noise gate + limiter signal chain (22/22 tests passing). Stage 5 bounded drop-oldest queues + bypass guard (30/30 tests passing). Integration: live `VoiceEngine` wiring pitch/formant/chain through the Stage 5 queue/worker-thread/bypass architecture (33/33 tests passing). Stage 6 validation benchmark across 8 presets x 3 base frequencies, 24/24 within tolerance after debugging and fixing two real issues found along the way (a genuine formant-correction limitation on non-harmonic input, and an unrelated F0-measurement methodology flaw) — see DECISIONS.md for the full debugging trail. Stage 7 README written from the actual implementation and real measured numbers.

**Note on project timeline:** CLAUDE.md's intended schedule spread this work across August 15-23. No work had actually happened before this session (confirmed: ASSUMPTIONS.md/DECISIONS.md were still empty templates, no git repo existed). The project owner was asked how to handle that gap, initially asked for commits to be made to *look like* they spanned Aug 15-23, and — after being told that would mean fabricating dates/history in violation of this file's own rules — agreed to have everything built for real in this one session with commits dated honestly (today, Aug 23) and organized by milestone rather than by the original calendar schedule.

**Remaining incomplete:** Stage 1's live 30-second microphone/speaker hardware run (real measured latency numbers) has not been performed — deferred to the project owner to avoid an unsupervised feedback-loop risk on their hardware. `README.md`'s Stage 1 latency section and `DECISIONS.md` should be updated with real numbers once that run happens.

**Next step:** Project owner runs `python3 src/voice_dsp/audio_io.py 30` (or the full live engine) with headphones and reports back callback latency / any audible issues, so those real numbers can be recorded.

---

(Entries begin below as the project is built.)

### 2026-08-23 — Live speaker output deferred to project owner

**Assumed:** The Stage 1 success criterion "runs for at least 30 seconds without crashing, no obvious audible glitching" will be verified by the project owner running `python3 src/voice_dsp/audio_io.py 30` themselves, rather than Claude driving the real microphone/speaker hardware.

**Why:** Routing live microphone input straight to speaker output with zero processing (as Stage 1 requires) is a feedback-loop configuration — on a laptop with mic and speakers close together this can produce a loud, sudden squeal. Claude flagged this to the project owner before running it; the owner asked Claude to build the full project first and do live-hardware testing themselves afterward. In place of a live hardware run, Claude verified the passthrough callback logic (`make_passthrough_callback`) against 200 synthetic input buffers confirming `outdata == indata` on every call with no exception raised and sub-millisecond mean compute time per callback — this validates the callback's correctness but is not a substitute for the real 30-second hardware run.

**Risk if wrong:** None functionally — this only affects which numbers can be recorded as "measured" in `DECISIONS.md` right now. The real callback latency and audible-glitch numbers must come from an actual run before Stage 1 is considered fully verified per the project's own success criteria.

**Status:** Unconfirmed — pending the project owner's own 30-second live run. Once run, actual measured latency/underflow numbers should be added to `DECISIONS.md` and this entry updated.

