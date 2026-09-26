"""The first step is always the original A, the last always the original B.

Whole and untouched: same samples, same length. Before, several things could
change them: plugins that resynthesise their endpoints (LPC lost B's tail
entirely when A was shorter; WORLD rebuilt both), the zero-padding that
equalises lengths, Stretch to Fit, DTW Align, and the shared level trim and
limiter.
"""
from __future__ import annotations

import numpy as np
import pytest

from app.morph_engine import _Worker, with_original_endpoints
from plugins.registry import build_default_registry

SR = 44100
REGISTRY = build_default_registry()


def _decaying(f0: float, seconds: float, seed: int, peak: float = 0.3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(int(SR * seconds)) / SR
    y = sum(np.sin(2 * np.pi * f0 * k * t + rng.uniform(0, 6)) / k for k in range(1, 6))
    y = y * np.exp(-1.5 * t)
    y = y / np.max(np.abs(y)) * peak
    return np.stack([y, 0.8 * y], axis=1).astype(np.float32)


def _run(plugin_name: str, a, b, steps: int = 4, **flags) -> list[np.ndarray]:
    worker = _Worker(REGISTRY.get(plugin_name), a, b, steps, SR, {}, **flags)
    done, errors = [], []
    worker.signals.finished.connect(done.append)
    worker.signals.error.connect(errors.append)
    worker.setAutoDelete(False)
    worker.run()
    assert not errors, errors
    return done[0]


@pytest.mark.parametrize("plugin", REGISTRY.names())
@pytest.mark.parametrize(
    "a_len, b_len", [(1.0, 1.0), (0.6, 1.2), (1.2, 0.6)], ids=["equal", "A-shorter", "B-shorter"]
)
def test_endpoints_are_the_untouched_sources(plugin, a_len, b_len):
    a = _decaying(220.0, a_len, 1)
    b = _decaying(330.0, b_len, 2)
    steps = _run(plugin, a, b)
    np.testing.assert_array_equal(steps[0], a)
    np.testing.assert_array_equal(steps[-1], b)


@pytest.mark.parametrize(
    "flags",
    [
        {"stretch_to_fit": True},
        {"dtw": True},
        {"level_match": False},
        {"stretch_to_fit": True, "dtw": True, "level_match": False},
    ],
    ids=["stretch", "dtw", "no-level-match", "all"],
)
@pytest.mark.parametrize("plugin", ["Spectral FFT", "LPC / Source-Filter", "WORLD Vocoder"])
def test_endpoints_survive_every_engine_option(plugin, flags):
    a = _decaying(220.0, 0.6, 1)
    b = _decaying(330.0, 1.2, 2)
    steps = _run(plugin, a, b, **flags)
    np.testing.assert_array_equal(steps[0], a)
    np.testing.assert_array_equal(steps[-1], b)


def test_full_scale_sources_are_not_limited_or_trimmed():
    """Sources at 0 dBFS used to go through the shared trim and the limiter."""
    a = _decaying(220.0, 1.0, 1, peak=1.0)
    b = _decaying(330.0, 1.0, 2, peak=1.0)
    steps = _run("Spectral FFT", a, b)
    np.testing.assert_array_equal(steps[0], a)
    np.testing.assert_array_equal(steps[-1], b)
    for s in steps[1:-1]:
        assert float(np.max(np.abs(s))) <= 0.99 + 1e-6


def test_endpoints_keep_their_own_length():
    a = _decaying(220.0, 0.6, 1)
    b = _decaying(330.0, 1.2, 2)
    steps = _run("Crossfade", a, b)
    assert len(steps[0]) == len(a)
    assert len(steps[-1]) == len(b)
    assert all(len(s) == len(b) for s in steps[1:-1])


def test_helper_copies_rather_than_aliases():
    a = np.ones((10, 1), dtype=np.float32)
    b = np.zeros((10, 1), dtype=np.float32)
    out = with_original_endpoints([a * 0, a * 0.5, a * 0], a, b)
    out[0][0, 0] = 5.0
    assert a[0, 0] == 1.0
    np.testing.assert_array_equal(out[1], a * 0.5)
