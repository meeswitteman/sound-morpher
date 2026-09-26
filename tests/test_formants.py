"""Formant preservation in the Pitch Shift plugin.

Resampling-based shifting moves a sound's resonances along with its pitch:
the "chipmunk" effect. With formants preserved, the harmonics land on the new
pitch but their amplitudes still follow the source's own resonance curve.

Ground truth comes from a synthetic vowel: a harmonic source through a known
all-pole formant filter H. After a shift by r, harmonic k sits at k * f0 * r,
and with formants preserved its level should follow |H(k * f0 * r)|.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.signal import freqz, lfilter

from plugins.base import apply_formant_correction, formant_envelope

SR = 44100


def _formant_filter(freqs, bandwidths) -> np.ndarray:
    a = np.array([1.0])
    for f, bw in zip(freqs, bandwidths):
        r = np.exp(-np.pi * bw / SR)
        a = np.convolve(a, [1.0, -2 * r * np.cos(2 * np.pi * f / SR), r * r])
    return a


VOWEL_A = _formant_filter([730, 1090, 2440, 3400], [90, 110, 160, 250])
VOWEL_I = _formant_filter([270, 2290, 3010, 3700], [60, 100, 120, 200])


def _vowel(f0: float, filt: np.ndarray, seconds: float = 1.0) -> np.ndarray:
    phase = 2 * np.pi * f0 * np.arange(int(SR * seconds)) / SR
    source = sum(np.sin(k * phase) for k in range(1, int(8000 / f0)))
    y = lfilter([1.0], filt, source)
    return (0.3 * y / np.max(np.abs(y))).astype(np.float32).reshape(-1, 1)


def _formant_deviation_db(y: np.ndarray, f0: float, filt: np.ndarray) -> float:
    """RMS distance between harmonic levels and the filter curve, level-free."""
    y = y.ravel()[SR // 8 : -SR // 8]
    spec = np.abs(np.fft.rfft(y * np.hanning(len(y))))
    freqs = np.fft.rfftfreq(len(y), 1 / SR)
    harmonics = np.arange(1, int(5000 / f0)) * f0
    levels = 20 * np.log10([spec[np.argmin(np.abs(freqs - h))] + 1e-12 for h in harmonics])
    _, h = freqz([1.0], filt, worN=harmonics, fs=SR)
    d = levels - 20 * np.log10(np.abs(h))
    d -= d.mean()
    return float(np.sqrt(np.mean(d ** 2)))


def _fundamental(y: np.ndarray, lo: float, hi: float) -> float:
    y = y.ravel()
    spec = np.abs(np.fft.rfft(y * np.hanning(len(y))))
    freqs = np.fft.rfftfreq(len(y), 1 / SR)
    band = (freqs > lo) & (freqs < hi)
    return float(freqs[band][np.argmax(spec[band])])


@pytest.mark.parametrize("filt", [VOWEL_A, VOWEL_I], ids=["a", "i"])
@pytest.mark.parametrize("semitones", [-7.0, 5.0, 9.0])
def test_preserved_formants_follow_the_source_resonances(filt, semitones):
    from plugins.pitch_shift import _envelopes, _restore_formants, _shift_channels

    f0 = 150.0
    x = _vowel(f0, filt)
    ratio = 2 ** (semitones / 12)
    shifted = _shift_channels(x, SR, semitones, 1)
    kept = _restore_formants(shifted, _envelopes(x, SR, f0), ratio, SR)

    moved = _formant_deviation_db(shifted, f0 * ratio, filt)
    preserved = _formant_deviation_db(kept, f0 * ratio, filt)
    # Shifting alone puts the resonances 13-23 dB off; preserved, they sit
    # within a few dB of the source curve.
    assert preserved < moved - 6.0
    assert preserved < 11.0


def test_formant_correction_does_not_move_the_pitch():
    from plugins.pitch_shift import _envelopes, _restore_formants, _shift_channels

    x = _vowel(150.0, VOWEL_A)
    shifted = _shift_channels(x, SR, 7.0, 1)
    kept = _restore_formants(shifted, _envelopes(x, SR, 150.0), 2 ** (7 / 12), SR)
    target = 150.0 * 2 ** (7 / 12)
    # The strongest harmonic changes with the formants; check the fundamental.
    assert _fundamental(kept, target * 0.8, target * 1.2) == pytest.approx(target, rel=0.02)


def test_correction_is_a_no_op_without_a_shift_or_an_envelope():
    x = _vowel(150.0, VOWEL_A).ravel()
    env = formant_envelope(x, SR, 150.0)
    np.testing.assert_array_equal(apply_formant_correction(x, env, 1.0, SR), x)
    np.testing.assert_array_equal(apply_formant_correction(x, None, 1.3, SR), x)


def test_too_short_to_analyse_passes_through():
    x = _vowel(150.0, VOWEL_A, seconds=0.02).ravel()
    assert formant_envelope(x, SR, 150.0) is None
    np.testing.assert_array_equal(apply_formant_correction(x, None, 1.3, SR), x)


def test_formants_default_to_preserve():
    from plugins.pitch_shift import PitchShiftPlugin

    param = next(p for p in PitchShiftPlugin.parameters if p.name == "formants")
    assert param.default == "preserve"
    assert set(param.choices) == {"preserve", "shift"}


@pytest.mark.parametrize("tracking", ["median", "dynamic"])
def test_plugin_preserves_formants_in_both_tracking_modes(tracking):
    from plugins.pitch_shift import PitchShiftPlugin

    a = np.repeat(_vowel(140.0, VOWEL_A), 2, axis=1)
    b = np.repeat(_vowel(210.0, VOWEL_A), 2, axis=1)
    a[:, 1] *= 0.5

    plugin = PitchShiftPlugin()
    kept = plugin.morph(a, b, steps=3, sample_rate=SR, tracking=tracking)[1]
    moved = plugin.morph(a, b, steps=3, sample_rate=SR, tracking=tracking, formants="shift")[1]

    assert kept.shape == a.shape
    assert np.all(np.isfinite(kept))
    # Both sounds share one formant filter, so a preserved midpoint should sit
    # on that curve, at the geometric-mean pitch, better than a moved one.
    mid = np.sqrt(140.0 * 210.0)
    assert _formant_deviation_db(kept[:, 0], mid, VOWEL_A) < _formant_deviation_db(
        moved[:, 0], mid, VOWEL_A
    ) - 3.0



@pytest.mark.parametrize("semitones", [-7.0, 7.0])
def test_restoring_formants_keeps_level_and_stereo_image(semitones):
    """The correction reshapes the spectrum; it must not change loudness.

    Its average gain is not 0 dB on its own (about -2 dB shifting up a fifth,
    +1.5 dB down), which would tilt the A/B balance of the crossfade.
    """
    from plugins.pitch_shift import _envelopes, _restore_formants, _shift_channels

    x = _vowel(150.0, VOWEL_A)
    stereo = np.concatenate([x, 0.5 * x], axis=1)
    shifted = _shift_channels(stereo, SR, semitones, 2)
    kept = _restore_formants(shifted, _envelopes(stereo, SR, 150.0), 2 ** (semitones / 12), SR)

    def rms(s):
        return float(np.sqrt(np.mean(s.astype(np.float64) ** 2)))

    assert rms(kept) == pytest.approx(rms(shifted), rel=1e-3)
    assert rms(kept[:, 1]) / rms(kept[:, 0]) == pytest.approx(0.5, rel=1e-3)
