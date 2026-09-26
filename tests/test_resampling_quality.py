"""Band-limited resampling and phase-locked vocoding.

Two artefacts that sat in every pitch-changing path:
  * fractional-position reads were linear interpolation, which dulls the top
    octave and, when reading faster than 1.0, folds the spectrum back down as
    aliasing
  * the phase vocoder advanced every bin on its own, smearing each partial
    over its neighbouring bins: the hollow, "phasey" vocoder sound
"""
from __future__ import annotations

import numpy as np
import pytest

from plugins.base import _phase_vocoder, read_bandlimited

SR = 44100


def _level_db(x: np.ndarray, freq: float) -> float:
    """Level of the component at `freq` relative to a unit-amplitude sine."""
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    freqs = np.fft.rfftfreq(len(x), 1 / SR)
    return float(20 * np.log10(spec[np.argmin(np.abs(freqs - freq))] / (len(x) / 4) + 1e-15))


def _harmonic(f0: float, seconds: float, partials: int = 11) -> np.ndarray:
    t = np.arange(int(SR * seconds)) / SR
    return sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, partials + 1)).astype(
        np.float32
    )


def _harmonic_to_residue_db(y: np.ndarray, f0: float, partials: int = 11) -> float:
    """Energy on the harmonics against everything in between."""
    y = y[SR // 2 : -SR // 2]
    spec = np.abs(np.fft.rfft(y * np.hanning(len(y)))) ** 2
    freqs = np.fft.rfftfreq(len(y), 1 / SR)
    on = np.zeros(len(freqs), dtype=bool)
    for k in range(1, partials + 1):
        on |= np.abs(freqs - f0 * k) < 5.0
    return float(10 * np.log10(spec[on].sum() / spec[~on].sum()))


# ── read_bandlimited ──────────────────────────────────────────────────────────

def test_bandlimited_read_is_exact_on_integer_positions():
    rng = np.random.default_rng(0)
    sig = rng.standard_normal(4000)
    out = read_bandlimited(sig, np.arange(100, 3900, dtype=np.float64))
    np.testing.assert_allclose(out, sig[100:3900], atol=1e-5)


def test_bandlimited_read_keeps_the_top_octave():
    """Linear interpolation loses ~6.8 dB at 15 kHz on a half-sample read."""
    tone = np.sin(2 * np.pi * 15000 * np.arange(SR) / SR)
    pos = np.arange(1000, SR - 1000) + 0.5
    assert _level_db(read_bandlimited(tone, pos), 15000) > -1.0


def test_bandlimited_read_filters_out_aliasing_when_reading_faster():
    """18 kHz read at 1.5x lands at 27 kHz and would fold to 17.1 kHz."""
    tone = np.sin(2 * np.pi * 18000 * np.arange(SR) / SR)
    pos = np.arange(1000, int((SR - 2000) / 1.5)) * 1.5
    assert _level_db(read_bandlimited(tone, pos, 1.5), 17100) < -60.0


def test_bandlimited_read_handles_stereo_and_edges():
    sig = np.ones((500, 2), dtype=np.float32)
    out = read_bandlimited(sig, np.linspace(-10.0, 510.0, 300), 1.3)
    assert out.shape == (300, 2)
    np.testing.assert_allclose(out, 1.0, atol=1e-5)


# ── Phase locking ─────────────────────────────────────────────────────────────

def test_phase_locking_keeps_partials_coherent():
    import librosa

    sig = _harmonic(220.0, 2.0)
    D = librosa.stft(sig, n_fft=2048, hop_length=512)
    time_map = np.arange(0, D.shape[1] - 1, 1 / 1.5)

    def stretched(lock: bool) -> np.ndarray:
        out = _phase_vocoder(D, time_map, 512, 2048, phase_lock=lock)
        return librosa.istft(out, hop_length=512, n_fft=2048)

    plain = _harmonic_to_residue_db(stretched(False), 220.0)
    locked = _harmonic_to_residue_db(stretched(True), 220.0)
    assert locked > plain + 10.0


def test_constant_pitch_shift_is_cleaner_than_librosa():
    """The plugin used librosa.effects.pitch_shift, which has no phase locking."""
    import librosa

    from plugins.pitch_shift import _shift_channels

    sig = _harmonic(220.0, 2.0, partials=8)
    ours = _shift_channels(sig.reshape(-1, 1), SR, 7.0, 1).ravel()
    theirs = librosa.effects.pitch_shift(sig, sr=SR, n_steps=7.0)

    f0 = 220.0 * 2 ** (7 / 12)
    assert len(ours) == len(sig)
    assert _harmonic_to_residue_db(ours, f0, 8) > _harmonic_to_residue_db(theirs, f0, 8) + 6.0


# ── Granular ──────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("rate", [0.7, 1.12, 1.5])
def test_grain_read_matches_the_reference_reader(rate):
    from plugins.granular import _read_grain

    t = np.arange(3 * SR) / SR
    src = np.stack(
        [np.sin(2 * np.pi * 440 * t) + 0.3 * np.sin(2 * np.pi * 5100 * t),
         np.cos(2 * np.pi * 870 * t)],
        axis=1,
    ).astype(np.float32)
    start, length = SR, 3000
    grain = _read_grain(src, start, length, rate)
    assert grain.shape == (length, 2)

    # The grain may start up to one sample early; find that offset, then the
    # waveform itself must match the sinc reader closely.
    best = min(
        np.max(np.abs(grain - read_bandlimited(src, start - s + np.arange(length) * rate, rate)))
        for s in np.linspace(0.0, 1.0, 101)
    )
    assert best < 0.01


def test_grain_read_filters_out_aliasing():
    from plugins.granular import _read_grain

    tone = np.sin(2 * np.pi * 18000 * np.arange(SR) / SR).astype(np.float32).reshape(-1, 1)
    grain = _read_grain(tone, 2000, 20000, 1.5).ravel()
    assert _level_db(grain, 17100) < -60.0


def test_grain_read_survives_the_signal_edges():
    from plugins.granular import _read_grain

    src = np.ones((1000, 1), dtype=np.float32)
    for start in (-300, 0, 900, 1200):
        grain = _read_grain(src, start, 400, 1.4)
        assert grain.shape == (400, 1)
        assert np.all(np.isfinite(grain))
