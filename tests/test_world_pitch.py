"""WORLD vocoder pitch detection: Harvest by default, DIO on request.

DIO drops voiced frames in the middle of notes once there is some noise or
jitter, and WORLD rebuilds those frames from noise: audible crackle.
"""
from __future__ import annotations

import numpy as np
import pytest

SR = 44100


def _pw():
    from plugins.world_vocoder import _ensure_pyworld

    try:
        return _ensure_pyworld()
    except ImportError:
        pytest.skip("pyworld not installed")


def _noisy_note(seconds: float = 2.0, seed: int = 0) -> np.ndarray:
    """A vibrato note with a weak fundamental in a fair amount of noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(SR * seconds)) / SR
    wobble = np.cumsum(rng.standard_normal(len(t))) / np.sqrt(SR)
    f0 = 220.0 * 2 ** ((0.5 * np.sin(2 * np.pi * 5.5 * t) + wobble) / 12)
    phase = 2 * np.pi * np.cumsum(f0) / SR
    amps = [0.3] + [1 / k ** 1.1 for k in range(2, 30)]
    note = sum(a * np.sin(k * phase) for k, a in enumerate(amps, 1))
    return (0.2 * note + 0.1 * rng.standard_normal(len(t))).astype(np.float64)


def _missed_voiced_percent(detector: str) -> float:
    from plugins.world_vocoder import _estimate_f0

    pw = _pw()
    sig = _noisy_note()
    f0, tax = _estimate_f0(pw, sig, SR, 5.0, detector)
    inside = (tax > 0.1) & (tax < tax[-1] - 0.1)
    return float(np.mean(f0[inside] == 0.0) * 100)


def test_harvest_is_the_default():
    from plugins.world_vocoder import WorldVocoderPlugin

    param = next(p for p in WorldVocoderPlugin.parameters if p.name == "pitch_detector")
    assert param.default == "harvest"
    assert set(param.choices) == {"harvest", "dio"}


def test_harvest_keeps_a_noisy_note_voiced():
    harvest = _missed_voiced_percent("harvest")
    dio = _missed_voiced_percent("dio")
    assert harvest < 1.0
    # DIO is kept as the fast option, and this is the case it loses.
    assert dio > harvest + 3.0


@pytest.mark.parametrize("detector", ["harvest", "dio"])
def test_morph_runs_with_either_detector(detector, monkeypatch):
    pw = _pw()
    import plugins.world_vocoder as wv

    used = []
    for name in ("harvest", "dio"):
        real = getattr(pw, name)
        monkeypatch.setattr(
            pw, name, lambda *a, _real=real, _name=name, **k: used.append(_name) or _real(*a, **k)
        )

    a = _noisy_note(0.5, seed=1).astype(np.float32).reshape(-1, 1)
    b = _noisy_note(0.5, seed=2).astype(np.float32).reshape(-1, 1)
    steps = wv.WorldVocoderPlugin().morph(a, b, steps=3, sample_rate=SR, pitch_detector=detector)

    assert set(used) == {detector}
    assert len(steps) == 3
    for s in steps:
        assert s.shape == a.shape
        assert np.all(np.isfinite(s))
