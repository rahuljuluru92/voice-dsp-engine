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

