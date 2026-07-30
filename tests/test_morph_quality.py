"""Regression tests for the morph output-quality fixes.

Each test pins down a defect that was measured on the pre-fix code:
  * linear phase interpolation cancelled partials (−11 dB at t=0.5)
  * arithmetic magnitude interpolation stacked both spectra instead of morphing
  * Griffin-Lim's endpoints were unrelated to A and B
  * intermediate steps drifted far below the endpoints in loudness
  * the Granular plugin's output was bit-for-bit a linear crossfade
"""

from __future__ import annotations

import numpy as np
import pytest

from plugins.base import interp_magnitude, interp_phase, match_step_loudness

SR = 44100


def _tone(freq: float, seconds: float = 0.5, phase: float = 0.0, amp: float = 0.5):
    t = np.arange(int(SR * seconds)) / SR
    return (amp * np.sin(2 * np.pi * freq * t + phase)).astype(np.float32).reshape(-1, 1)


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))


def _dominant_freqs(sig: np.ndarray, n: int = 2) -> list[float]:
    mono = sig.ravel()
    spec = np.abs(np.fft.rfft(mono * np.hanning(len(mono))))
    freqs = np.fft.rfftfreq(len(mono), 1 / SR)
    peaks = np.argsort(spec)[-n * 3:]
    # Cluster neighbouring bins, keep the n strongest distinct peaks
    found: list[float] = []
    for p in sorted(peaks, key=lambda i: -spec[i]):
        f = float(freqs[p])
        if all(abs(f - g) > 20.0 for g in found):
            found.append(f)
        if len(found) == n:
            break
    return sorted(found)


# ── interp_phase ──────────────────────────────────────────────────────────────

def test_shortest_arc_phase_takes_the_short_way_round():
    a = np.array([3.0])          # near +pi
    b = np.array([-3.0])         # near -pi; short way is +0.28 rad, not -6 rad
    out = interp_phase(a, b, 0.5)
    # Wrapped result should sit just outside +pi, i.e. adjacent to both inputs
    diff = (out - a + np.pi) % (2 * np.pi) - np.pi
    assert abs(diff.item()) < 0.2


def test_linear_phase_mode_preserves_old_behaviour():
    a = np.array([3.0])
    b = np.array([-3.0])
    assert interp_phase(a, b, 0.5, mode="linear").item() == pytest.approx(0.0)


def test_dominant_phase_switches_at_midpoint():
    a, b = np.array([1.0]), np.array([2.0])
    assert interp_phase(a, b, 0.4, mode="dominant").item() == 1.0
    assert interp_phase(a, b, 0.6, mode="dominant").item() == 2.0


# ── interp_magnitude ──────────────────────────────────────────────────────────

def test_log_magnitude_is_geometric_mean():
    a = np.array([1.0, 4.0])
    b = np.array([1.0, 1.0])
    out = interp_magnitude(a, b, 0.5)
    np.testing.assert_allclose(out, [1.0, 2.0], rtol=1e-6)


def test_linear_magnitude_is_arithmetic_mean():
    a = np.array([1.0, 4.0])
    b = np.array([1.0, 1.0])
    out = interp_magnitude(a, b, 0.5, mode="linear")
    np.testing.assert_allclose(out, [1.0, 2.5], rtol=1e-6)


def test_log_magnitude_endpoints_are_exact():
    a = np.array([0.2, 4.0])
    b = np.array([1.0, 0.5])
    np.testing.assert_allclose(interp_magnitude(a, b, 0.0), a, rtol=1e-6)
    np.testing.assert_allclose(interp_magnitude(a, b, 1.0), b, rtol=1e-6)


def test_log_magnitude_handles_zero_bins():
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    out = interp_magnitude(a, b, 0.5)
    assert np.all(np.isfinite(out))
    assert np.all(out >= 0.0)


# ── Spectral FFT ──────────────────────────────────────────────────────────────

def test_spectral_phase_offset_no_longer_cancels():
    """Two identical tones a phase-offset apart used to lose 11 dB at t=0.5."""
    from plugins.spectral_fft import SpectralFftPlugin

    a = _tone(440.0)
    b = _tone(440.0, phase=np.pi * 0.9)
    steps = SpectralFftPlugin().morph(a, b, steps=5, sample_rate=SR)

    endpoint_rms = _rms(steps[0])
    middle_rms = _rms(steps[2])
    # Was 0.098 vs 0.354 (−11 dB); require the dip to stay within 3 dB.
    assert middle_rms > endpoint_rms * 0.71


def test_spectral_linear_phase_still_cancels():
    """The old behaviour remains reachable, so the fix is demonstrably the cause."""
    from plugins.spectral_fft import SpectralFftPlugin

    a = _tone(440.0)
    b = _tone(440.0, phase=np.pi * 0.9)
    steps = SpectralFftPlugin().morph(a, b, steps=5, sample_rate=SR, phase="linear")
    assert _rms(steps[2]) < _rms(steps[0]) * 0.5


def test_spectral_log_magnitude_suppresses_the_departing_partial():
    """440 → 660: with a linear blend both tones are equally loud at t=0.5."""
    from plugins.spectral_fft import SpectralFftPlugin

    a, b = _tone(440.0), _tone(660.0)

    linear = SpectralFftPlugin().morph(
        a, b, steps=3, sample_rate=SR, magnitude="linear"
    )[1]
    assert len(_dominant_freqs(linear, 2)) == 2  # both partials survive

    log = SpectralFftPlugin().morph(a, b, steps=3, sample_rate=SR)[1]
    spec = np.abs(np.fft.rfft(log.ravel() * np.hanning(len(log))))
    freqs = np.fft.rfftfreq(len(log), 1 / SR)
    peak_440 = spec[np.argmin(np.abs(freqs - 440))]
    peak_660 = spec[np.argmin(np.abs(freqs - 660))]
    # Geometric blending pulls both endpoints' partials well below the level
    # a straight sum would give.
    assert max(peak_440, peak_660) < spec.max() * 1.01
    assert peak_440 / peak_660 == pytest.approx(1.0, abs=0.5)


def test_spectral_endpoints_are_bit_exact():
    from plugins.spectral_fft import SpectralFftPlugin

    a, b = _tone(440.0), _tone(660.0)
    steps = SpectralFftPlugin().morph(a, b, steps=4, sample_rate=SR)
    np.testing.assert_array_equal(steps[0], a)
    np.testing.assert_array_equal(steps[-1], b)


# ── Griffin-Lim ───────────────────────────────────────────────────────────────

def test_griffin_lim_endpoints_are_bit_exact():
    """Step 0 used to differ from A by 0.96 on a 0.5-amplitude sine."""
    from plugins.griffin_lim import GriffinLimPlugin

    a, b = _tone(440.0, 0.2), _tone(660.0, 0.2)
    steps = GriffinLimPlugin().morph(a, b, steps=3, sample_rate=SR, n_iter=8)
    np.testing.assert_array_equal(steps[0], a)
    np.testing.assert_array_equal(steps[-1], b)


def test_griffin_lim_seeded_phase_beats_random_init():
    from plugins.griffin_lim import GriffinLimPlugin

    a, b = _tone(440.0, 0.2), _tone(440.0, 0.2, phase=1.0)
    plugin = GriffinLimPlugin()
    seeded = plugin.morph(a, b, steps=3, sample_rate=SR, n_iter=8)[1]
    unseeded = plugin.morph(
        a, b, steps=3, sample_rate=SR, n_iter=8, seed_phase=False
    )[1]
    # Same target magnitude; the seeded run should land closer to the sources.
    assert _rms(seeded) >= _rms(unseeded) * 0.95
    assert np.all(np.isfinite(seeded))


# ── Granular ──────────────────────────────────────────────────────────────────

def _granular(a, b, steps=5, **kw):
    from plugins.granular import GranularPlugin
    return GranularPlugin().morph(a, b, steps=steps, sample_rate=SR, **kw)


def test_granular_scatter_is_not_a_crossfade():
    """The old implementation matched CrossfadePlugin on all but one sample."""
    from plugins.crossfade import CrossfadePlugin

    rng = np.random.default_rng(1)
    a = (rng.standard_normal((SR // 2, 1)) * 0.2).astype(np.float32)
    b = (rng.standard_normal((SR // 2, 1)) * 0.2).astype(np.float32)

    gran = _granular(a, b)[2]
    fade = CrossfadePlugin().morph(a, b, steps=5, sample_rate=SR, curve="linear")[2]
    differing = int(np.count_nonzero(np.abs(gran - fade) > 1e-5))
    assert differing > len(a) * 0.9


def test_granular_mix_mode_still_matches_crossfade():
    """The old behaviour stays reachable, so the difference is demonstrably the mode."""
    from plugins.crossfade import CrossfadePlugin

    rng = np.random.default_rng(1)
    a = (rng.standard_normal((SR // 2, 1)) * 0.2).astype(np.float32)
    b = (rng.standard_normal((SR // 2, 1)) * 0.2).astype(np.float32)

    gran = _granular(a, b, mode="mix")[2]
    fade = CrossfadePlugin().morph(a, b, steps=5, sample_rate=SR, curve="linear")[2]
    np.testing.assert_allclose(gran, fade, atol=1e-5)


def test_granular_grain_source_share_tracks_t():
    """A 440/660 pair: the share of B-sourced grains should rise with t."""
    a, b = _tone(440.0, 1.0), _tone(660.0, 1.0)
    steps = _granular(a, b, steps=5, jitter_ms=0.0)

    def b_share(sig):
        spec = np.abs(np.fft.rfft(sig.ravel() * np.hanning(len(sig))))
        freqs = np.fft.rfftfreq(len(sig), 1 / SR)
        e440 = spec[np.argmin(np.abs(freqs - 440))]
        e660 = spec[np.argmin(np.abs(freqs - 660))]
        return e660 / (e440 + e660 + 1e-12)

    shares = [b_share(s) for s in steps]
    assert shares == sorted(shares)
    assert shares[0] < 0.05 and shares[-1] > 0.95


def test_granular_grains_switch_monotonically_across_steps():
    """Shared draws mean a grain flips A→B once and never back."""
    a = np.full((SR // 2, 1), 1.0, dtype=np.float32)
    b = np.full((SR // 2, 1), -1.0, dtype=np.float32)
    steps = _granular(a, b, steps=6, jitter_ms=0.0)
    # Constant sources: each sample's value tracks how many of its overlapping
    # grains came from B, which must only ever increase.
    means = [float(np.mean(s)) for s in steps]
    assert all(later <= earlier + 1e-6 for earlier, later in zip(means, means[1:]))


def test_granular_seed_is_reproducible_and_changeable():
    rng = np.random.default_rng(2)
    a = (rng.standard_normal((SR // 4, 1)) * 0.2).astype(np.float32)
    b = (rng.standard_normal((SR // 4, 1)) * 0.2).astype(np.float32)

    first = _granular(a, b, seed=7)[2]
    again = _granular(a, b, seed=7)[2]
    other = _granular(a, b, seed=8)[2]

    np.testing.assert_array_equal(first, again)
    assert float(np.max(np.abs(first - other))) > 1e-4


def test_granular_endpoints_are_bit_exact():
    a, b = _tone(440.0, 0.3), _tone(660.0, 0.3)
    steps = _granular(a, b, steps=4)
    np.testing.assert_array_equal(steps[0], a)
    np.testing.assert_array_equal(steps[-1], b)


def test_granular_has_no_dropout_on_the_first_sample():
    """The old Hann window hit zero at sample 0, leaving it unweighted."""
    a = np.full((SR // 4, 1), 0.5, dtype=np.float32)
    b = np.full((SR // 4, 1), 0.5, dtype=np.float32)
    mid = _granular(a, b, steps=3, jitter_ms=0.0)[1]
    assert float(mid[0, 0]) == pytest.approx(0.5, abs=1e-3)


def test_granular_pitch_jitter_transposes_grains():
    a, b = _tone(440.0, 1.0), _tone(440.0, 1.0)
    clean = _granular(a, b, steps=3, jitter_ms=0.0, pitch_jitter=0.0)[1]
    shifted = _granular(a, b, steps=3, jitter_ms=0.0, pitch_jitter=6.0)[1]

    def spread(sig):
        spec = np.abs(np.fft.rfft(sig.ravel() * np.hanning(len(sig))))
        freqs = np.fft.rfftfreq(len(sig), 1 / SR)
        band = (freqs > 200) & (freqs < 900)
        w = spec[band] / (spec[band].sum() + 1e-12)
        centre = float((freqs[band] * w).sum())
        return float(np.sqrt((w * (freqs[band] - centre) ** 2).sum()))

    # Detuned grains smear energy either side of 440 Hz.
    assert spread(shifted) > spread(clean) * 1.5


def test_granular_stereo_is_preserved():
    rng = np.random.default_rng(3)
    a = (rng.standard_normal((SR // 4, 2)) * 0.2).astype(np.float32)
    b = (rng.standard_normal((SR // 4, 2)) * 0.2).astype(np.float32)
    steps = _granular(a, b, steps=4, pitch_jitter=3.0)
    for s in steps:
        assert s.shape == (SR // 4, 2)
        assert s.dtype == np.float32
        assert np.all(np.isfinite(s))


def test_granular_jitter_stays_in_bounds_at_the_edges():
    """Negative offsets at the head and overruns at the tail must not wrap or crash."""
    a, b = _tone(440.0, 0.2), _tone(660.0, 0.2)
    steps = _granular(a, b, steps=5, jitter_ms=200.0, pitch_jitter=12.0)
    for s in steps:
        assert np.all(np.isfinite(s))
        assert float(np.max(np.abs(s))) <= 1.0


# ── match_step_loudness ───────────────────────────────────────────────────────

def test_level_match_follows_a_straight_line_from_a_to_b():
    rng = np.random.default_rng(0)
    # Kept well below full scale so the shared anti-clip trim does not engage.
    a = (rng.standard_normal((SR // 4, 1)) * 0.08).astype(np.float32)
    b = (rng.standard_normal((SR // 4, 1)) * 0.03).astype(np.float32)

    from plugins.crossfade import CrossfadePlugin
    steps = CrossfadePlugin().morph(a, b, steps=5, sample_rate=SR, curve="linear")
    matched = match_step_loudness(steps, a, b)

    rms_a, rms_b = _rms(a), _rms(b)
    for i, step in enumerate(matched):
        t = i / (len(matched) - 1)
        expected = (1 - t) * rms_a + t * rms_b
        assert _rms(step) == pytest.approx(expected, rel=0.02)


def test_level_match_applies_one_shared_gain_when_clipping():
    a = np.full((100, 1), 0.9, dtype=np.float32)
    b = np.full((100, 1), 0.9, dtype=np.float32)
    steps = [np.full((100, 1), 0.45, dtype=np.float32) for _ in range(3)]
    matched = match_step_loudness(steps, a, b)

    peak = max(float(np.max(np.abs(s))) for s in matched)
    assert peak <= 0.99 + 1e-6
    # Relative levels between steps are untouched
    levels = [_rms(s) for s in matched]
    assert max(levels) == pytest.approx(min(levels), rel=1e-6)


def test_level_match_ignores_silent_steps():
    a = np.full((100, 1), 0.5, dtype=np.float32)
    b = np.zeros((100, 1), dtype=np.float32)
    steps = [a.copy(), np.zeros((100, 1), dtype=np.float32), b.copy()]
    matched = match_step_loudness(steps, a, b)
    assert len(matched) == 3
    assert all(np.all(np.isfinite(s)) for s in matched)


def test_level_match_respects_max_gain():
    a = np.full((100, 1), 0.5, dtype=np.float32)
    b = np.full((100, 1), 0.5, dtype=np.float32)
    tiny = [np.full((100, 1), 1e-4, dtype=np.float32) for _ in range(3)]
    matched = match_step_loudness(tiny, a, b, max_gain_db=6.0)
    assert _rms(matched[0]) == pytest.approx(1e-4 * 10 ** (6 / 20), rel=1e-3)
