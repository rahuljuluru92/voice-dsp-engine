import numpy as np
from scipy import signal as sig

from src.voice_dsp.formant import cepstral_envelope, shift_formants, pitch_shift_formant_preserving
from src.voice_dsp.stft import periodic_hann, stft

SR = 48000
FRAME_SIZE = 2048
HOP = 512

# Empirically measured on 2026-08-23 with a synthetic two-formant vowel
# (F0=180Hz, formants at 700Hz/1200Hz), sr=48000, frame_size=2048
# (bin width 23.4Hz): formant-only shifts left F0 exactly unchanged (0Hz
# observed difference); pitch-only (formant-preserving) shifts left F0
# within ~0.3Hz of the expected ratio-shifted value; formant peak
# locations after either operation matched the expected/original
# position within 2 FFT bins (worst observed: 46.9Hz on a 1200Hz peak
# after a +5 semitone formant-preserving pitch shift). Tolerances below
# keep ~1.3x margin above that worst case while still catching a real
# bug (e.g. a missing/inverted warp, which would misplace peaks by
# hundreds of Hz or shift F0 instead of leaving it alone).
F0_TOLERANCE_HZ = 2.0
PEAK_TOLERANCE_HZ = 60.0


def _resonator(freq_hz, bw_hz, sr):
    r = np.exp(-np.pi * bw_hz / sr)
    theta = 2 * np.pi * freq_hz / sr
    b = [1 - r]
    a = [1, -2 * r * np.cos(theta), r * r]
    return b, a


def _synth_vowel(f0, formants, sr, duration):
    n = int(sr * duration)
    period = sr / f0
    impulses = np.zeros(n)
    idx = np.arange(0, n, period).astype(int)
    impulses[idx] = 1.0
    x = impulses.copy()
    for freq_hz, bw_hz in formants:
        b, a = _resonator(freq_hz, bw_hz, sr)
        x = sig.lfilter(b, a, x)
    return x / np.max(np.abs(x)) * 0.7


def _measure_f0_autocorr(x, sr, fmin=60, fmax=400):
    x = x - np.mean(x)
    corr = np.correlate(x, x, mode="full")[len(x) - 1:]
    min_lag = int(sr / fmax)
    max_lag = int(sr / fmin)
    seg = corr[min_lag:max_lag]
    peak = int(np.argmax(seg)) + min_lag
    return sr / peak


def _measure_envelope_peaks(x, sr, fmin=300, fmax=2000, n_peaks=2):
    window = periodic_hann(FRAME_SIZE)
    pad = FRAME_SIZE
    xp = np.concatenate([np.zeros(pad), x, np.zeros(pad)])
    spec = stft(xp, FRAME_SIZE, HOP, window)
    mid = spec.shape[0] // 2
    mags = np.abs(spec[mid - 3:mid + 3])
    cutoff = FRAME_SIZE // 16
    env = np.mean([cepstral_envelope(m, cutoff) for m in mags], axis=0)
    freqs = np.fft.rfftfreq(FRAME_SIZE, 1 / sr)
    mask = (freqs >= fmin) & (freqs <= fmax)
    idxs = np.where(mask)[0]
    peaks, _ = sig.find_peaks(env[idxs])
    peak_freqs = freqs[idxs][peaks]
    peak_vals = env[idxs][peaks]
    order = np.argsort(-peak_vals)
    return sorted(peak_freqs[order][:n_peaks])


def _test_vowel(f0=180.0, formants=((700, 60), (1200, 70)), duration=1.5):
    return _synth_vowel(f0, formants, SR, duration), f0


def test_formant_only_shift_preserves_fundamental():
    x, f0 = _test_vowel()
    f0_before = _measure_f0_autocorr(x, SR)

    y = shift_formants(x, formant_semitones=4, frame_size=FRAME_SIZE, hop_size=HOP)
    f0_after = _measure_f0_autocorr(y, SR)

    assert abs(f0_after - f0_before) < F0_TOLERANCE_HZ


def test_formant_only_shift_moves_envelope_peaks():
    x, f0 = _test_vowel()
    peaks_before = _measure_envelope_peaks(x, SR)

    semitones = 4
    ratio = 2 ** (semitones / 12)
    y = shift_formants(x, formant_semitones=semitones, frame_size=FRAME_SIZE, hop_size=HOP)
    peaks_after = _measure_envelope_peaks(y, SR)

    expected = [p * ratio for p in peaks_before]
    for measured, exp in zip(peaks_after, expected):
        assert abs(measured - exp) < PEAK_TOLERANCE_HZ


def test_pitch_only_shift_preserves_formant_peaks():
    x, f0 = _test_vowel()
    peaks_before = _measure_envelope_peaks(x, SR)

    y = pitch_shift_formant_preserving(
        x, pitch_semitones=-4, frame_size=FRAME_SIZE, analysis_hop=HOP
    )
    peaks_after = _measure_envelope_peaks(y, SR)

    for measured, before in zip(peaks_after, peaks_before):
        assert abs(measured - before) < PEAK_TOLERANCE_HZ


def test_pitch_only_shift_changes_fundamental():
    x, f0 = _test_vowel()
    f0_before = _measure_f0_autocorr(x, SR)

    semitones = 5
    y = pitch_shift_formant_preserving(
        x, pitch_semitones=semitones, frame_size=FRAME_SIZE, analysis_hop=HOP
    )
    f0_after = _measure_f0_autocorr(y, SR)
    expected_f0 = f0_before * 2 ** (semitones / 12)

    assert abs(f0_after - expected_f0) < F0_TOLERANCE_HZ
