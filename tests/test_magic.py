"""Magic: analyse both sounds, choose the settings, keep the best-scoring morph.

The analysis has to tell the kinds of sound apart, the plan has to pick the
length handling that suits them, the score has to prefer a real morph over
one that wanders off or blurs the pitch, and the result has to be an ordinary
morph: the requested number of steps, with the untouched sources at the ends.
"""
from __future__ import annotations

import numpy as np
import pytest

from app.magic import (
    MagicCancelled,
    analyse,
    make_plan,
    run_magic,
    score_morph,
)
from plugins.registry import build_default_registry

SR = 44100
rng = np.random.default_rng(7)


def _tone(f0: float, dur: float, lead: float = 0.0) -> np.ndarray:
    t = np.arange(int(dur * SR)) / SR
    y = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 8))
    y *= np.minimum(1.0, t / 0.02)
    out = np.concatenate([np.zeros(int(lead * SR)), 0.3 * y])
    return np.stack([out, out], axis=1).astype(np.float32)


def _noise(dur: float) -> np.ndarray:
    y = 0.2 * rng.standard_normal(int(dur * SR))
    return np.stack([y, y], axis=1).astype(np.float32)


def _hit(dur: float = 0.4, decay: float = 15.0) -> np.ndarray:
    t = np.arange(int(dur * SR)) / SR
    y = 0.5 * rng.standard_normal(len(t)) * np.exp(-t * decay)
    return np.stack([y, y], axis=1).astype(np.float32)


def _bursts(n: int, gap: float = 0.4) -> np.ndarray:
    parts = []
    for _ in range(n):
        parts.append(_hit(0.1, 40.0))
        parts.append(np.zeros((int(gap * SR), 2), dtype=np.float32))
    return np.concatenate(parts)


# ── Analysis ──────────────────────────────────────────────────────────────────

def test_a_held_tone_is_pitched():
    p = analyse(_tone(220.0, 1.0), SR)
    assert p.kind == "pitched"
    assert p.f0 == pytest.approx(220.0, rel=0.03)
    assert p.onsets == 1


def test_noise_is_noisy_not_percussive():
    p = analyse(_noise(1.5), SR)
    assert p.kind == "noisy"
    # Noise flickers in level all the time; none of that is a new event.
    assert p.onsets <= 1


def test_a_decaying_burst_is_a_hit():
    p = analyse(_hit(), SR)
    assert p.hit
    assert p.kind == "percussive"


def test_separate_bursts_are_counted():
    assert analyse(_bursts(4), SR).onsets == 4


def test_lead_silence_is_measured():
    p = analyse(_tone(220.0, 1.0, lead=0.3), SR)
    assert p.lead == pytest.approx(0.3, abs=0.03)


# ── Plan ──────────────────────────────────────────────────────────────────────

def test_plan_lines_up_the_starts():
    pa = analyse(_tone(220.0, 1.0, lead=0.25), SR)
    pb = analyse(_tone(330.0, 1.0), SR)
    plan = make_plan(pa, pb)
    assert plan.trim_a[0] == pytest.approx(0.25, abs=0.03)
    assert plan.trim_b[0] == 0.0


def test_plan_stretches_sustained_sounds_of_different_length():
    plan = make_plan(analyse(_tone(220.0, 1.0), SR), analyse(_tone(330.0, 2.0), SR))
    assert plan.stretch_to_fit and not plan.dtw
    assert plan.pair == "pitched"
    assert plan.candidates[0].plugin == "Pitch Shift"


def test_plan_pads_two_hits():
    plan = make_plan(analyse(_hit(0.5, 10.0), SR), analyse(_hit(0.2, 30.0), SR))
    assert plan.pair == "percussive"
    assert not plan.stretch_to_fit


def test_plan_does_not_stretch_too_far():
    plan = make_plan(analyse(_tone(220.0, 0.5), SR), analyse(_tone(330.0, 3.0), SR))
    assert not plan.stretch_to_fit


def test_plan_uses_dtw_for_rhythmic_material():
    plan = make_plan(analyse(_bursts(4, 0.3), SR), analyse(_bursts(4, 0.45), SR))
    assert plan.dtw


# ── Score ─────────────────────────────────────────────────────────────────────

def _lowpassed(noise: np.ndarray, cutoff: float) -> np.ndarray:
    from scipy.signal import butter, sosfilt

    sos = butter(4, cutoff, "low", fs=SR, output="sos")
    return sosfilt(sos, noise, axis=0).astype(np.float32)


def test_score_prefers_a_straight_path_over_a_detour():
    base = _noise(1.0)
    # Cutoff sliding from dark to bright: every step between its neighbours.
    straight = [_lowpassed(base, f) for f in (500, 1000, 2000, 4000, 8000)]
    # Same endpoints, but the middle step is something else entirely.
    detour = list(straight)
    detour[2] = _tone(1500.0, 1.0)
    assert score_morph(straight, SR).total < score_morph(detour, SR).total


def test_pitch_term_prefers_a_glide_over_two_pitches_at_once():
    a, b = _tone(220.0, 1.0), _tone(330.0, 1.0)
    glide = [_tone(220.0 * 1.5 ** (i / 4), 1.0) for i in range(5)]
    mixed = [(1 - i / 4) * a + (i / 4) * b for i in range(5)]
    assert score_morph(glide, SR, pitched=True).pitch < 0.05
    assert score_morph(mixed, SR, pitched=True).pitch > score_morph(glide, SR, pitched=True).pitch


# ── Run ───────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def registry():
    return build_default_registry()


def test_run_magic_returns_an_ordinary_morph(registry):
    a, b = _tone(220.0, 0.6, lead=0.1), _tone(330.0, 0.9)
    result = run_magic(registry, a, b, 6, SR)
    assert len(result.steps) == 6
    assert np.array_equal(result.steps[0], a)
    assert np.array_equal(result.steps[-1], b)
    assert result.plugin in registry
    assert set(result.params) <= {p.name for p in registry.get(result.plugin).parameters}
    assert any(line.startswith("Chosen:") for line in result.report)
    for step in result.steps[1:-1]:
        assert np.all(np.isfinite(step))
        assert float(np.max(np.abs(step))) <= 1.0


def test_run_magic_reports_progress_up_to_100(registry):
    seen: list[int] = []
    run_magic(registry, _hit(0.3), _hit(0.2, 25.0), 4, SR, progress_cb=seen.append)
    assert seen[-1] == 100
    assert seen == sorted(seen)


def test_run_magic_stops_when_asked(registry):
    with pytest.raises(MagicCancelled):
        run_magic(registry, _tone(220.0, 0.5), _tone(330.0, 0.5), 4, SR, should_stop=lambda: True)
