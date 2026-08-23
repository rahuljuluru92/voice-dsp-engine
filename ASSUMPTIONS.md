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

### [Date] — Session N

**Stage in progress:** —

**Done this session:** —

**Next step:** —

---

(Entries begin below as the project is built.)

### 2026-08-23 — Live speaker output deferred to project owner

**Assumed:** The Stage 1 success criterion "runs for at least 30 seconds without crashing, no obvious audible glitching" will be verified by the project owner running `python3 src/voice_dsp/audio_io.py 30` themselves, rather than Claude driving the real microphone/speaker hardware.

**Why:** Routing live microphone input straight to speaker output with zero processing (as Stage 1 requires) is a feedback-loop configuration — on a laptop with mic and speakers close together this can produce a loud, sudden squeal. Claude flagged this to the project owner before running it; the owner asked Claude to build the full project first and do live-hardware testing themselves afterward. In place of a live hardware run, Claude verified the passthrough callback logic (`make_passthrough_callback`) against 200 synthetic input buffers confirming `outdata == indata` on every call with no exception raised and sub-millisecond mean compute time per callback — this validates the callback's correctness but is not a substitute for the real 30-second hardware run.

**Risk if wrong:** None functionally — this only affects which numbers can be recorded as "measured" in `DECISIONS.md` right now. The real callback latency and audible-glitch numbers must come from an actual run before Stage 1 is considered fully verified per the project's own success criteria.

**Status:** Unconfirmed — pending the project owner's own 30-second live run. Once run, actual measured latency/underflow numbers should be added to `DECISIONS.md` and this entry updated.

