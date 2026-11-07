"""WORLD vocoder envelope blending: no hiss from the floor, no lost tails.

WORLD's spectral envelope is a power spectrum. The log blend used to floor it
100 dB under its peak read as amplitude, which is only 50 dB of power. Real
recordings go far deeper in their pauses and top octaves, and WORLD rebuilt
the raised floor as broadband hiss: even step 0, A resynthesised with nothing
morphed, came out tens of dB noisier than A.
"""
from __future__ import annotations

import numpy as np
import pytest

SR = 44100


@pytest.fixture(scope="module")
def plugin():
    from plugins.world_vocoder import WorldVocoderPlugin, _ensure_pyworld

    try:
        _ensure_pyworld()
    except ImportError:
        pytest.skip("pyworld not installed")
    return WorldVocoderPlugin()


def _note(f0: float, seconds: float) -> np.ndarray:
    """A note with a steep spectral tilt, between stretches of silence.

    Both give the envelope the deep dynamic range a recording has: the upper
    harmonics end up 80 dB and more under the fundamental.
    """
    t = np.arange(int(SR * seconds)) / SR
    note = sum(np.sin(2 * np.pi * f0 * k * t) / k ** 4 for k in range(1, 40))
    note *= np.minimum(1.0, t / 0.02) * np.minimum(1.0, (t[-1] - t) / 0.02)
    gap = np.zeros(int(0.4 * SR))
    y = 0.4 * np.concatenate([gap, note, gap, note, gap])
    return np.stack([y, y], axis=1).astype(np.float32)


def _flatness_db(audio: np.ndarray) -> float:
    """Median spectral flatness of the sounding frames: high means noisy."""
    import librosa

    mono = audio.mean(axis=1).astype(np.float32)
    S = np.abs(librosa.stft(mono, n_fft=2048, hop_length=512))
    power = (S ** 2).sum(axis=0)
    flat = librosa.feature.spectral_flatness(S=S)[0]
    return float(10.0 * np.log10(np.median(flat[power > power.max() * 1e-2])))


def _band_db(audio: np.ndarray, lo: float, hi: float, frames: slice) -> float:
    import librosa

    mono = audio.mean(axis=1).astype(np.float32)
    S = np.abs(librosa.stft(mono, n_fft=2048, hop_length=512)) ** 2
    freqs = librosa.fft_frequencies(sr=SR, n_fft=2048)
    band = (freqs >= lo) & (freqs < hi)
    return float(10.0 * np.log10(S[band][:, frames].mean() + 1e-20))


def test_resynthesis_adds_no_hiss(plugin):
    a, b = _note(220.0, 0.6), _note(330.0, 0.6)
    # The synthetic sources are cleaner (about -107 dB) than any resynthesis
    # gets; WORLD lands near -80 dB. The raised floor put every step at about
    # -35 dB, and the real recordings this was found on at -14 to -22 dB.
    for step in plugin.morph(a, b, 3, SR):
        assert _flatness_db(step) < -60.0


def test_pauses_stay_quiet(plugin):
    a, b = _note(220.0, 0.6), _note(330.0, 0.6)
    mid = plugin.morph(a, b, 3, SR)[1]
    # The first 0.3 s are silence in both sources.
    pause = slice(2, int(0.3 * SR / 512))
    note = slice(int(0.5 * SR / 512), int(0.9 * SR / 512))
    assert _band_db(mid, 0, SR / 2, pause) < _band_db(mid, 0, SR / 2, note) - 50.0


def test_a_sounding_source_survives_where_the_other_is_silent(plugin):
    a = _note(220.0, 0.6)
    b = a.copy()
    b[len(b) // 2 :] = 0.0   # B stops halfway; A's second note has no partner
    mid = plugin.morph(a, b, 3, SR)[1]
    second = slice(int(1.5 * SR / 512), int(1.9 * SR / 512))
    # A crossfade at t = 0.5 keeps A's note 6 dB down. The geometric blend
    # against silence would have wiped it out.
    assert _band_db(mid, 150, 300, second) > _band_db(a, 150, 300, second) - 12.0
