"""Sweep: the morph starts at one place in the sound and travels through it.

Step k of a sweep is B behind a moving front and A ahead of it, with the
plugin's intermediate levels across the front itself. Uniform is the plain
morph and must be left exactly as it was.
"""
from __future__ import annotations

import numpy as np
import pytest

from app.morph_engine import _Worker, sweep_steps
from plugins.registry import build_default_registry

SR = 44100
N = 5
LEN = SR  # one second


def _levels(stereo: bool = False) -> list[np.ndarray]:
    """Level i is a constant i / (n - 1): the morph amount can be read back."""
    shape = (LEN, 2) if stereo else (LEN,)
    return [np.full(shape, i / (N - 1), dtype=np.float32) for i in range(N)]


def _at(x: np.ndarray, frac: float) -> float:
    v = x[int(frac * (len(x) - 1))]
    return float(np.mean(v))


def test_uniform_is_untouched():
    levels = _levels()
    assert sweep_steps(levels, "uniform", 0.25, SR) is levels


def test_two_steps_are_untouched():
    levels = _levels()[:2]
    assert sweep_steps(levels, "forward", 0.25, SR) is levels


def test_unknown_direction_raises():
    with pytest.raises(ValueError):
        sweep_steps(_levels(), "sideways", 0.25, SR)


@pytest.mark.parametrize("direction", ["forward", "backward", "center_out"])
def test_first_and_last_step_are_pure(direction):
    out = sweep_steps(_levels(), direction, 0.25, SR)
    assert np.allclose(out[0], 0.0)
    assert np.allclose(out[-1], 1.0)


def test_forward_morphs_the_start_first():
    out = sweep_steps(_levels(), "forward", 0.1, SR)
    mid = out[N // 2]
    assert _at(mid, 0.05) == pytest.approx(1.0)   # start is already B
    assert _at(mid, 0.95) == pytest.approx(0.0)   # end is still A
    # The front keeps moving towards the end, step by step.
    fronts = [np.argmax(s < 0.5) for s in out[1:-1]]
    assert fronts == sorted(fronts) and len(set(fronts)) == len(fronts)


def test_backward_morphs_the_end_first():
    out = sweep_steps(_levels(), "backward", 0.1, SR)
    mid = out[N // 2]
    assert _at(mid, 0.05) == pytest.approx(0.0)
    assert _at(mid, 0.95) == pytest.approx(1.0)


def test_center_out_morphs_the_middle_first():
    out = sweep_steps(_levels(), "center_out", 0.1, SR)
    mid = out[N // 2]
    assert _at(mid, 0.5) == pytest.approx(1.0)
    assert _at(mid, 0.02) == pytest.approx(0.0)
    assert _at(mid, 0.98) == pytest.approx(0.0)
    assert np.allclose(mid, mid[::-1], atol=1e-3)   # symmetric around the middle


def test_front_passes_through_the_intermediate_levels():
    out = sweep_steps(_levels(stereo=True), "forward", 0.5, SR)
    values = np.unique(np.round(out[N // 2][:, 0], 2))
    # Not a splice of A and B: values in between are present, smoothly.
    assert ((values > 0.05) & (values < 0.95)).sum() > 20
    assert np.max(np.abs(np.diff(out[N // 2][:, 0]))) < 1e-3


def test_narrow_edge_is_kept_click_free():
    out = sweep_steps(_levels(), "forward", 0.0, SR)
    # At least 10 ms of front: no jump bigger than 1 / 441 per sample.
    assert np.max(np.abs(np.diff(out[N // 2]))) <= 1.0 / (0.010 * SR) + 1e-6


def test_uneven_lengths_are_padded():
    levels = _levels()
    levels[2] = levels[2][: LEN // 2]
    out = sweep_steps(levels, "forward", 0.25, SR)
    assert {len(s) for s in out} == {LEN}


def test_sweep_never_peaks_above_its_levels():
    rng = np.random.default_rng(0)
    levels = [(0.9 * np.clip(rng.standard_normal(LEN), -1, 1)).astype(np.float32)
              for _ in range(N)]
    for direction in ("forward", "backward", "center_out"):
        out = sweep_steps(levels, direction, 0.3, SR)
        assert max(float(np.max(np.abs(s))) for s in out) <= 0.9 + 1e-6


def test_worker_applies_sweep_and_keeps_endpoints():
    t = np.arange(LEN) / SR
    a = (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    b = (0.3 * np.sin(2 * np.pi * 330 * t)).astype(np.float32)
    worker = _Worker(
        build_default_registry().get("Crossfade"), a, b, N, SR, {},
        sweep="forward", sweep_edge=0.1,
    )
    done, errors = [], []
    worker.signals.finished.connect(done.append)
    worker.signals.error.connect(errors.append)
    worker.setAutoDelete(False)
    worker.run()
    assert not errors, errors
    steps = done[0]
    assert np.array_equal(steps[0], a) and np.array_equal(steps[-1], b)
    mid = steps[N // 2]
    head, tail = slice(0, SR // 10), slice(LEN - SR // 10, LEN)
    corr = lambda x, y: float(np.dot(x, y) / np.linalg.norm(x) / np.linalg.norm(y))
    assert corr(mid[head], b[head]) > 0.95   # start already turned into B
    assert corr(mid[tail], a[tail]) > 0.95   # end still A
