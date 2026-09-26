"""BS.1770 loudness and the step level matching built on it.

References: ITU-R BS.1770-4 for the measure, EBU Tech 3341 for the test
signals with known answers.
"""
from __future__ import annotations

import numpy as np
import pytest

from plugins.base import integrated_loudness, k_weighting, match_step_loudness

SR = 48000


def _sine(freq: float, seconds: float, dbfs: float, sr: int = SR) -> np.ndarray:
    t = np.arange(int(sr * seconds)) / sr
    return 10 ** (dbfs / 20) * np.sin(2 * np.pi * freq * t)


def _stereo(*segments: tuple[float, float]) -> np.ndarray:
    """Concatenated 1 kHz stereo segments of (seconds, dBFS)."""
    mono = np.concatenate([_sine(1000.0, s, db) for s, db in segments])
    return np.stack([mono, mono], axis=1)


# ── The measure ───────────────────────────────────────────────────────────────

def test_k_weighting_matches_the_published_48k_coefficients():
    b, a = k_weighting(48000)
    shelf_b = [1.53512485958697, -2.69169618940638, 1.19839281085285]
    shelf_a = [1.0, -1.69065929318241, 0.73248077421585]
    hp_b = [1.0, -2.0, 1.0]
    hp_a = [1.0, -1.99004745483398, 0.99007225036621]
    np.testing.assert_allclose(b, np.convolve(shelf_b, hp_b), atol=1e-10)
    np.testing.assert_allclose(a, np.convolve(shelf_a, hp_a), atol=1e-10)


@pytest.mark.parametrize("sr", [44100, 48000, 96000])
def test_full_scale_sine_on_one_channel_reads_minus_3_lufs(sr):
    """BS.1770's calibration point: 0 dBFS, 997 Hz, one channel = -3.01 LUFS."""
    assert integrated_loudness(_sine(997.0, 3.0, 0.0, sr), sr) == pytest.approx(-3.01, abs=0.05)


def test_tech_3341_case_1_steady_tone():
    assert integrated_loudness(_stereo((20.0, -23.0)), SR) == pytest.approx(-23.0, abs=0.1)


def test_tech_3341_case_3_relative_gate_drops_the_quiet_parts():
    sig = _stereo((10.0, -36.0), (60.0, -23.0), (10.0, -36.0))
    assert integrated_loudness(sig, SR) == pytest.approx(-23.0, abs=0.1)


def test_tech_3341_case_4_absolute_gate_drops_near_silence():
    sig = _stereo((10.0, -72.0), (10.0, -36.0), (60.0, -23.0), (10.0, -36.0), (10.0, -72.0))
    assert integrated_loudness(sig, SR) == pytest.approx(-23.0, abs=0.1)


def test_tech_3341_case_5_mixed_levels_above_the_gate():
    sig = _stereo((20.0, -26.0), (20.1, -20.0), (20.0, -26.0))
    assert integrated_loudness(sig, SR) == pytest.approx(-23.0, abs=0.1)


def test_silence_has_no_loudness():
    assert integrated_loudness(np.zeros((SR, 2)), SR) == -np.inf
    assert integrated_loudness(np.zeros((0, 2)), SR) == -np.inf


def test_dc_barely_registers():
    """DC is inaudible. Only its switch-on thump is measured, far below a tone
    of the same RMS; plain RMS matching treated the two as equally loud."""
    dc = np.full((2 * SR, 1), 0.5)
    tone = _sine(1000.0, 2.0, 20 * np.log10(0.5 * np.sqrt(2))).reshape(-1, 1)
    assert integrated_loudness(dc, SR) < integrated_loudness(tone, SR) - 20.0


def test_sample_shorter_than_one_block_is_still_measured():
    short = _sine(1000.0, 0.1, -20.0)
    long = _sine(1000.0, 2.0, -20.0)
    assert integrated_loudness(short, SR) == pytest.approx(integrated_loudness(long, SR), abs=0.1)


def test_gating_ignores_silence_inside_a_step():
    tone = _sine(1000.0, 1.0, -20.0)
    sparse = np.concatenate([tone, np.zeros(3 * SR)])
    # RMS reads the sparse version 6 dB quieter. The gates drop the silence;
    # only the blocks straddling the tone's end still pull it down slightly.
    assert integrated_loudness(sparse, SR) == pytest.approx(integrated_loudness(tone, SR), abs=1.0)


# ── Level matching ────────────────────────────────────────────────────────────

def test_level_matching_weighs_frequencies_like_the_ear():
    """Equal RMS, unequal loudness: 3 kHz sits in the ear's sensitive range.

    RMS matching left both steps where they were. Loudness matching sets them
    to the same LUFS, which for the 3 kHz step means turning it down.
    """
    a = _sine(1000.0, 1.0, -20.0).reshape(-1, 1)
    b = _sine(1000.0, 1.0, -20.0).reshape(-1, 1)
    low = _sine(80.0, 1.0, -20.0).reshape(-1, 1)
    bright = _sine(3000.0, 1.0, -20.0).reshape(-1, 1)
    steps = [a, low, bright, b]

    matched = match_step_loudness(steps, a, b, sample_rate=SR)
    target = integrated_loudness(a, SR)
    for s in matched:
        assert integrated_loudness(s, SR) == pytest.approx(target, abs=0.05)

    def rms(x):
        return float(np.sqrt(np.mean(x ** 2)))

    assert rms(matched[2]) < rms(bright) * 0.95
    assert rms(matched[1]) > rms(low) * 1.05


def test_level_matching_fades_cleanly_to_a_silent_end():
    """A line in dB cannot reach silence; the target then runs in amplitude."""
    a = _sine(1000.0, 0.5, -20.0).reshape(-1, 1)
    b = np.zeros_like(a)
    steps = [a * (1 - t) for t in np.linspace(0.0, 1.0, 5)]

    matched = match_step_loudness(steps, a, b, sample_rate=SR)
    levels = [integrated_loudness(s, SR) for s in matched[:-1]]
    assert all(np.isfinite(levels))
    assert levels == sorted(levels, reverse=True)
    assert np.all(matched[-1] == 0.0)
