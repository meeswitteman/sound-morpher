"""Steps between the endpoints keep the longer sound's tail.

With A and B of different lengths (and Stretch to Fit off) the shorter one is
zero-padded. Where it is silent, Spectral FFT and Griffin-Lim's geometric
magnitude blend multiplied the other sound away (up to 41 dB lost), and
LPC / Source-Filter, taking its excitation from a silent A, went fully
silent. Frames where one source is absent now fall back to a crossfade.
"""
from __future__ import annotations

import numpy as np
import pytest

from app.morph_engine import _Worker
from plugins.base import both_present
from plugins.registry import build_default_registry

SR = 44100
REGISTRY = build_default_registry()
FIXED = ["Spectral FFT", "Griffin-Lim", "LPC / Source-Filter"]


def _decaying(f0: float, seconds: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(int(SR * seconds)) / SR
    y = sum(np.sin(2 * np.pi * f0 * k * t + rng.uniform(0, 6)) / k for k in range(1, 8))
    y = y * np.exp(-1.5 * t) + 0.02 * rng.standard_normal(len(t)) * np.exp(-1.5 * t)
    y = 0.3 * y / np.max(np.abs(y))
    return np.stack([y, 0.9 * y], axis=1).astype(np.float32)


def _run(name: str, a, b, steps: int = 6) -> list[np.ndarray]:
    worker = _Worker(REGISTRY.get(name), a, b, steps, SR, {})
    done, errors = [], []
    worker.signals.finished.connect(done.append)
    worker.signals.error.connect(errors.append)
    worker.setAutoDelete(False)
    worker.run()
    assert not errors, errors
    return done[0]


def _tail_db(x: np.ndarray, start_s: float, end_s: float) -> float:
    seg = x[int(start_s * SR) : int(end_s * SR)].astype(np.float64)
    return float(10 * np.log10(np.mean(seg ** 2) + 1e-20))


# ── The presence measure ──────────────────────────────────────────────────────

def test_presence_ramps_between_40_and_60_db():
    e = 1.0
    assert both_present(e, e) == 1.0
    assert both_present(e, e * 10 ** -3.9) == 1.0          # -39 dB
    assert both_present(e, e * 10 ** -5.0) == pytest.approx(0.5)   # -50 dB
    assert both_present(e, e * 10 ** -6.1) == 0.0          # -61 dB
    assert both_present(e, 0.0) == 0.0
    assert both_present(0.0, 0.0) == 1.0                   # silence into silence


# ── The plugins ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("name", FIXED)
@pytest.mark.parametrize("shorter", ["A", "B"])
def test_intermediate_steps_keep_the_longer_sounds_tail(name, shorter):
    short, long_ = _decaying(220.0, 1.2, 1), _decaying(330.0, 2.4, 2)
    a, b = (short, long_) if shorter == "A" else (long_, short)
    window = (1.8, 2.38)                     # where only the longer one sounds

    fade = _run("Crossfade", a, b)
    steps = _run(name, a, b)
    for i in range(1, len(steps) - 1):
        diff = _tail_db(steps[i], *window) - _tail_db(fade[i], *window)
        # Was down to -41 dB (Spectral FFT), -30 dB (Griffin-Lim) and
        # silence (LPC); now within a few dB of a crossfade either way.
        assert -8.0 < diff < 8.0, f"step {i}: {diff:+.1f} dB against a crossfade"


@pytest.mark.parametrize("name", FIXED)
def test_equal_length_morphs_are_unchanged(name, monkeypatch):
    """Where both sources sound throughout, the fallback never engages."""
    import plugins.base as base
    import plugins.griffin_lim as gl
    import plugins.lpc_morph as lpc
    import plugins.spectral_fft as sf

    a, b = _decaying(220.0, 1.0, 1), _decaying(330.0, 1.0, 2)
    # Keep both well above the 40 dB threshold to the end.
    a = a + 0.02 * np.sin(2 * np.pi * 440 * np.arange(len(a)) / SR)[:, None].astype(np.float32)
    b = b + 0.02 * np.sin(2 * np.pi * 550 * np.arange(len(b)) / SR)[:, None].astype(np.float32)
    now = _run(name, a, b)

    always = lambda ea, eb, *args, **kw: np.ones_like(np.asarray(ea, dtype=np.float64))  # noqa: E731
    for mod in (base, gl, lpc, sf):
        monkeypatch.setattr(mod, "both_present", always, raising=False)
    pure = _run(name, a, b)

    for x, y in zip(now, pure):
        np.testing.assert_allclose(x, y, atol=1e-5)


def test_lpc_no_longer_goes_silent_when_a_is_shorter():
    a, b = _decaying(220.0, 1.2, 1), _decaying(330.0, 2.4, 2)
    steps = _run("LPC / Source-Filter", a, b)
    for i in range(1, len(steps) - 1):
        assert _tail_db(steps[i], 1.8, 2.38) > -60.0
