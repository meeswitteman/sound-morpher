import numpy as np
import pytest

from plugins.base import MorphPlugin, PluginParam, match_lengths
from plugins.crossfade import CrossfadePlugin
from plugins.registry import PluginRegistry, build_default_registry


# ── match_lengths ──────────────────────────────────────────────────────

def test_match_lengths_equal():
    a = np.ones((100, 2), dtype=np.float32)
    b = np.ones((100, 2), dtype=np.float32)
    a2, b2 = match_lengths(a, b)
    assert len(a2) == len(b2) == 100


def test_match_lengths_pads_shorter():
    a = np.ones((80, 2), dtype=np.float32)
    b = np.ones((100, 2), dtype=np.float32)
    a2, b2 = match_lengths(a, b)
    assert len(a2) == len(b2) == 100
    # Padded section should be zero
    assert np.all(a2[80:] == 0.0)


# ── CrossfadePlugin ────────────────────────────────────────────────────

def _stereo(frames: int, value: float = 1.0) -> np.ndarray:
    return np.full((frames, 2), value, dtype=np.float32)


def test_crossfade_returns_correct_step_count():
    plugin = CrossfadePlugin()
    steps = plugin.morph(_stereo(100), _stereo(100), steps=8, sample_rate=44100)
    assert len(steps) == 8


def test_crossfade_first_step_is_a():
    plugin = CrossfadePlugin()
    a = _stereo(100, 1.0)
    b = _stereo(100, 0.0)
    steps = plugin.morph(a, b, steps=4, sample_rate=44100)
    np.testing.assert_allclose(steps[0], a, atol=1e-5)


def test_crossfade_last_step_is_b():
    plugin = CrossfadePlugin()
    a = _stereo(100, 1.0)
    b = _stereo(100, 0.0)
    steps = plugin.morph(a, b, steps=4, sample_rate=44100)
    np.testing.assert_allclose(steps[-1], b, atol=1e-5)


def test_crossfade_midpoint():
    plugin = CrossfadePlugin()
    a = _stereo(100, 2.0)
    b = _stereo(100, 0.0)
    steps = plugin.morph(a, b, steps=3, sample_rate=44100)
    # Middle step: t=0.5 → 0.5*2 + 0.5*0 = 1.0
    np.testing.assert_allclose(steps[1], np.full((100, 2), 1.0, dtype=np.float32), atol=1e-5)


def test_crossfade_equal_power():
    plugin = CrossfadePlugin()
    steps = plugin.morph(_stereo(100), _stereo(100), steps=5, sample_rate=44100, curve="equal-power")
    assert len(steps) == 5


def test_crossfade_different_lengths():
    plugin = CrossfadePlugin()
    a = _stereo(80)
    b = _stereo(120)
    steps = plugin.morph(a, b, steps=4, sample_rate=44100)
    assert all(len(s) == 120 for s in steps)


# ── PluginRegistry ─────────────────────────────────────────────────────

def test_registry_register_and_get():
    registry = PluginRegistry()
    registry.register(CrossfadePlugin())
    assert "Crossfade" in registry
    plugin = registry.get("Crossfade")
    assert isinstance(plugin, CrossfadePlugin)


def test_registry_unknown_raises():
    registry = PluginRegistry()
    with pytest.raises(KeyError):
        registry.get("NonExistent")


def test_build_default_registry():
    registry = build_default_registry()
    assert len(registry) >= 1
    assert "Crossfade" in registry
