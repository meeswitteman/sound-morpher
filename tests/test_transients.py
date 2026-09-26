"""Transient preservation in Spectral FFT and Griffin-Lim.

One FFT size cannot suit both halves of a sound: long frames smear attacks
into pre-echo, short ones cannot resolve partials. The plugins now split each
source into a tonal and a transient layer and morph each at its own size.
"""
from __future__ import annotations

import numpy as np
import pytest

from plugins.base import split_transients, transient_fft_size
from plugins.griffin_lim import GriffinLimPlugin
from plugins.spectral_fft import SpectralFftPlugin

SR = 44100
ONSETS = np.arange(0.25, 2.0, 0.5)
PLUGINS = [SpectralFftPlugin(), GriffinLimPlugin()]


def _hits(freq: float, seed: int, seconds: float = 2.0) -> np.ndarray:
    """Isolated drum-like hits in silence: anything before a hit is artefact."""
    rng = np.random.default_rng(seed)
    n = int(SR * seconds)
    y = np.zeros(n)
    for o in ONSETS[ONSETS < seconds]:
        s = int(o * SR)
        tt = np.arange(n - s) / SR
        y[s:] += np.sin(2 * np.pi * freq * tt) * np.exp(-tt * 25)
        y[s:] += 0.5 * rng.standard_normal(n - s) * np.exp(-tt * 60)
    return (0.3 * y).astype(np.float32).reshape(-1, 1)


def _pre_echo_db(y: np.ndarray) -> float:
    """Energy 3-40 ms before each hit relative to the first 30 ms of it."""
    y = y.ravel()
    ratios = []
    for o in ONSETS:
        s = int(o * SR)
        before = np.mean(y[s - int(0.04 * SR) : s - int(0.003 * SR)] ** 2)
        after = np.mean(y[s : s + int(0.03 * SR)] ** 2)
        ratios.append(before / after)
    return float(10 * np.log10(np.mean(ratios)))


def _chord(freqs, seconds: float = 1.0) -> np.ndarray:
    t = np.arange(int(SR * seconds)) / SR
    return (0.2 * sum(np.sin(2 * np.pi * f * t) for f in freqs)).astype(np.float32).reshape(-1, 1)


# ── The split ─────────────────────────────────────────────────────────────────

def test_layers_sum_back_to_the_input_exactly():
    rng = np.random.default_rng(0)
    x = (rng.standard_normal((SR, 2)) * 0.2).astype(np.float32) + _hits(90.0, 1)[:SR]
    tonal, transient = split_transients(x, SR)
    assert tonal.shape == transient.shape == x.shape
    np.testing.assert_allclose(tonal + transient, x, atol=1e-6)


def test_steady_tones_go_to_the_tonal_layer():
    x = _chord((220, 330, 440))
    tonal, transient = split_transients(x, SR)
    body = slice(SR // 8, -SR // 8)
    ratio = np.sum(transient[body] ** 2) / np.sum(tonal[body] ** 2)
    assert 10 * np.log10(ratio) < -25.0


def test_hits_go_to_the_transient_layer():
    x = _hits(2000.0, 2)
    tonal, transient = split_transients(x, SR)
    s = int(ONSETS[0] * SR)
    attack = slice(s, s + int(0.005 * SR))
    assert np.sum(transient[attack] ** 2) > np.sum(tonal[attack] ** 2)


def test_too_short_to_split_is_all_tonal():
    x = _chord((440,), seconds=0.01)
    tonal, transient = split_transients(x, SR)
    np.testing.assert_array_equal(tonal, x)
    assert not np.any(transient)


def test_transient_fft_scales_with_the_sample_rate():
    assert transient_fft_size(44100) == 256
    assert transient_fft_size(48000) == 256
    assert transient_fft_size(96000) == 512


# ── The plugins ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("plugin", PLUGINS, ids=lambda p: p.name)
def test_transients_default_to_preserve(plugin):
    param = next(p for p in plugin.parameters if p.name == "transients")
    assert param.default == "preserve"


@pytest.mark.parametrize("plugin", PLUGINS, ids=lambda p: p.name)
@pytest.mark.parametrize("fft_size", ["1024", "4096"])
def test_preserving_transients_removes_pre_echo(plugin, fft_size):
    a, b = _hits(80.0, 1), _hits(180.0, 2)
    smeared = plugin.morph(a, b, 3, SR, fft_size=fft_size, transients="smear")[1]
    kept = plugin.morph(a, b, 3, SR, fft_size=fft_size, transients="preserve")[1]
    # Measured: 12-26 dB less, landing around -42 to -49 dB.
    assert _pre_echo_db(kept) < _pre_echo_db(smeared) - 8.0
    assert _pre_echo_db(kept) < -38.0


@pytest.mark.parametrize("plugin", PLUGINS, ids=lambda p: p.name)
def test_steady_material_is_barely_touched(plugin):
    a, b = _chord((220, 330, 440)), _chord((247, 370, 494))
    smeared = plugin.morph(a, b, 3, SR, transients="smear")[1]
    kept = plugin.morph(a, b, 3, SR, transients="preserve")[1]
    body = slice(SR // 4, -SR // 4)
    diff = np.sum((smeared[body] - kept[body]) ** 2) / np.sum(smeared[body] ** 2)
    assert 10 * np.log10(diff) < -20.0


@pytest.mark.parametrize("plugin", PLUGINS, ids=lambda p: p.name)
def test_layered_morph_keeps_endpoints_and_stereo(plugin):
    a = np.repeat(_hits(80.0, 1, 1.0), 2, axis=1)
    b = np.repeat(_hits(180.0, 2, 1.0), 2, axis=1)
    a[:, 1] *= 0.5
    steps = plugin.morph(a, b, 3, SR)
    np.testing.assert_array_equal(steps[0], a)
    np.testing.assert_array_equal(steps[-1], b)
    assert steps[1].shape == a.shape
    assert np.all(np.isfinite(steps[1]))
