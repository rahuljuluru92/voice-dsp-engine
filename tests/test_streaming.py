import numpy as np
from scipy import signal as sig

from src.voice_dsp.formant import cepstral_envelope
from src.voice_dsp.stft import periodic_hann, stft
from src.voice_dsp.streaming import StreamingVoiceProcessor

SR = 48000

# Empirically measured on 2026-08-23 (see DECISIONS.md): identity
# reconstruction fed in 256-sample chunks matched the original signal
# to 2.5e-14 max absolute error once correctly time-aligned by the
# processor's fixed latency (frame_size - analysis_hop samples).
IDENTITY_LATENCY_SAMPLES = 2048 - 512
IDENTITY_TOLERANCE = 1e-9

# Pitch-shift accuracy matched the offline algorithm's already-validated
# numbers (errors under 0.03Hz); reuse the same generous documented
# tolerance as tests/test_pitch.py.
FREQ_TOLERANCE_HZ = 1.0


def _feed_in_chunks(processor, x, chunk_size):
    out = []
    for i in range(0, len(x), chunk_size):
        out.append(processor.process(x[i:i + chunk_size]))
    return np.concatenate(out) if out else np.zeros(0)


def _measure_freq(x, sr):
    n = len(x)
    win = np.hanning(n)
    spec = np.fft.rfft(x * win)
    mag = np.abs(spec)
    k = int(np.argmax(mag))
    if 0 < k < len(mag) - 1:
        a, b, g = mag[k - 1], mag[k], mag[k + 1]
        denom = a - 2 * b + g
        p = 0.5 * (a - g) / denom if denom != 0 else 0.0
    else:
        p = 0.0
    return (k + p) * sr / n


def test_identity_reconstruction_fed_in_small_chunks():
    sp = StreamingVoiceProcessor(sr=SR, pitch_semitones=0.0, formant_semitones=0.0)
    t = np.arange(SR) / SR
    x = 0.5 * np.sin(2 * np.pi * 220.0 * t)

    y = _feed_in_chunks(sp, x, chunk_size=256)

    lat = IDENTITY_LATENCY_SAMPLES
    y_aligned = y[lat:lat + len(x) - lat]
    x_aligned = x[: len(y_aligned)]
    assert np.max(np.abs(x_aligned - y_aligned)) < IDENTITY_TOLERANCE


def test_pitch_shift_accuracy_matches_offline_algorithm():
    for semitones, expected_err_margin in [(3, 1.0), (7, 1.0), (-5, 1.0)]:
        sp = StreamingVoiceProcessor(sr=SR, pitch_semitones=semitones, formant_semitones=0.0)
        t = np.arange(int(SR * 2)) / SR
        x = 0.8 * np.sin(2 * np.pi * 220.0 * t)

        y = _feed_in_chunks(sp, x, chunk_size=256)

        edge = int(SR * 0.1)
        measured = _measure_freq(y[edge:-1000], SR)
        expected = 220.0 * (2 ** (semitones / 12))
        assert abs(measured - expected) < FREQ_TOLERANCE_HZ


def test_output_is_independent_of_caller_chunk_size():
    # The core property this module exists for: no boundary artifacts
    # regardless of how the caller happens to hand over audio.
    sp_a = StreamingVoiceProcessor(sr=SR, pitch_semitones=7.0, formant_semitones=0.0)
    sp_b = StreamingVoiceProcessor(sr=SR, pitch_semitones=7.0, formant_semitones=0.0)
    sp_c = StreamingVoiceProcessor(sr=SR, pitch_semitones=7.0, formant_semitones=0.0)

    t = np.arange(int(SR * 2)) / SR
    x = 0.8 * np.sin(2 * np.pi * 220.0 * t)

    y_256 = _feed_in_chunks(sp_a, x, chunk_size=256)
    y_4096 = _feed_in_chunks(sp_b, x, chunk_size=4096)
    y_1 = _feed_in_chunks(sp_c, x, chunk_size=1)

    n = min(len(y_256), len(y_4096), len(y_1))
    assert np.max(np.abs(y_256[:n] - y_4096[:n])) < 1e-9
    assert np.max(np.abs(y_256[:n] - y_1[:n])) < 1e-9


def test_no_large_discontinuities_at_any_pitch_ratio():
    # Empirically measured 2026-08-23: the equivalent chunked
    # architecture produced up to 41 discontinuities (peak magnitude
    # 0.51) in 3s at some ratios. This processor should have none,
    # including at the previously-problematic +12 semitones case
    # (synthesis_hop == frame_size / 2, a non-COLA window-sum case that
    # caused a large startup spike before the priming fix).
    t = np.arange(int(SR * 3)) / SR
    x = 0.5 * np.sin(2 * np.pi * 180.0 * t)

    for semitones in [0.0, 4.0, 7.0, 12.0, -5.0]:
        sp = StreamingVoiceProcessor(sr=SR, pitch_semitones=semitones, formant_semitones=0.0)
        y = _feed_in_chunks(sp, x, chunk_size=256)

        assert np.all(np.isfinite(y))
        d = np.abs(np.diff(y))
        typical = np.median(d[d > 0]) if np.any(d > 0) else 0.0
        threshold = max(0.05, typical * 20)
        spikes = np.where(d > threshold)[0]
        assert len(spikes) == 0, f"unexpected discontinuities at {semitones:+.0f}st: {d[spikes]}"


def _resonator_coeffs(freq_hz, bandwidth_hz, sr):
    r = np.exp(-np.pi * bandwidth_hz / sr)
    theta = 2 * np.pi * freq_hz / sr
    return [1 - r], [1, -2 * r * np.cos(theta), r * r]


def _synth_vowel(f0, formants, sr, duration):
    n = int(sr * duration)
    period = sr / f0
    impulses = np.zeros(n)
    idx = np.arange(0, n, period).astype(int)
    impulses[idx] = 1.0
    x = impulses.copy()
    for freq_hz, bandwidth_hz in formants:
        b, a = _resonator_coeffs(freq_hz, bandwidth_hz, sr)
        x = sig.lfilter(b, a, x)
    return x / np.max(np.abs(x)) * 0.7


def _measure_f0_autocorr(x, sr, expected, band_frac=0.2):
    x = x - np.mean(x)
    corr = np.correlate(x, x, mode="full")[len(x) - 1:]
    lo = max(1, int(sr / (expected * (1 + band_frac))))
    hi = int(sr / (expected * (1 - band_frac)))
    seg = corr[lo:hi]
    peak = int(np.argmax(seg)) + lo
    return sr / peak


def _measure_envelope_peaks(x, sr, frame_size=2048, hop=512, fmin=300, fmax=2000, n_peaks=2):
    window = periodic_hann(frame_size)
    pad = frame_size
    xp = np.concatenate([np.zeros(pad), x, np.zeros(pad)])
    spec = stft(xp, frame_size, hop, window)
    mid = spec.shape[0] // 2
    mags = np.abs(spec[mid - 3:mid + 3])
    env = np.mean([cepstral_envelope(m, frame_size // 16) for m in mags], axis=0)
    freqs = np.fft.rfftfreq(frame_size, 1 / sr)
    mask = (freqs >= fmin) & (freqs <= fmax)
    idxs = np.where(mask)[0]
    peaks, _ = sig.find_peaks(env[idxs])
    peak_freqs = freqs[idxs][peaks]
    peak_vals = env[idxs][peaks]
    order = np.argsort(-peak_vals)
    return sorted(peak_freqs[order][:n_peaks])


def test_formant_only_shift_preserves_fundamental_streaming():
    x = _synth_vowel(180.0, [(700, 60), (1200, 70)], SR, 2.0)
    f0_before = _measure_f0_autocorr(x, SR, 180.0)

    sp = StreamingVoiceProcessor(sr=SR, pitch_semitones=0.0, formant_semitones=4.0)
    y = _feed_in_chunks(sp, x, chunk_size=256)
    f0_after = _measure_f0_autocorr(y, SR, f0_before)

    assert abs(f0_after - f0_before) < 2.0


def test_pitch_only_shift_preserves_formant_peaks_streaming():
    x = _synth_vowel(180.0, [(700, 60), (1200, 70)], SR, 2.0)
    peaks_before = _measure_envelope_peaks(x, SR)

    sp = StreamingVoiceProcessor(sr=SR, pitch_semitones=-4.0, formant_semitones=0.0)
    y = _feed_in_chunks(sp, x, chunk_size=256)
    peaks_after = _measure_envelope_peaks(y, SR)

    for before, after in zip(peaks_before, peaks_after):
        assert abs(after - before) < 60.0
