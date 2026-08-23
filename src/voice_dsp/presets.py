"""Named pitch/formant presets used by the live engine and the Stage 6
validation benchmark."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Preset:
    name: str
    pitch_semitones: float
    formant_semitones: float


PRESETS: dict[str, Preset] = {
    "identity": Preset("identity", 0.0, 0.0),
    "pitch_up_third": Preset("pitch_up_third", 4.0, 0.0),
    "pitch_down_third": Preset("pitch_down_third", -4.0, 0.0),
    "pitch_up_octave": Preset("pitch_up_octave", 12.0, 0.0),
    "formant_up": Preset("formant_up", 0.0, 4.0),
    "formant_down": Preset("formant_down", 0.0, -4.0),
    "pitch_and_formant_up": Preset("pitch_and_formant_up", 5.0, 3.0),
    "pitch_up_formant_down": Preset("pitch_up_formant_down", 7.0, -3.0),
}
