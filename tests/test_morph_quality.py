"""Regression tests for the morph output-quality fixes.

Each test pins down a defect that was measured on the pre-fix code:
  * linear phase interpolation cancelled partials (−11 dB at t=0.5)
  * arithmetic magnitude interpolation stacked both spectra instead of morphing
  * Griffin-Lim's endpoints were unrelated to A and B
  * intermediate steps drifted far below the endpoints in loudness
  * the Granular plugin's output was bit-for-bit a linear crossfade
  * the LPC and WORLD vocoders collapsed stereo input to mono
  * blending LPC coefficients directly produced unstable synthesis filters
  * DTW alignment resampled the signal, sliding its pitch with the warp path
  * 16-bit export quantised without dither
"""

from __future__ import annotations

import numpy as np
import pytest

from plugins.base import (
    dtw_align,
    interp_lpc,
    interp_magnitude,
    interp_phase,
    lpc_to_lsf,
    lsf_to_lpc,
    match_step_loudness,
)

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


# ── LPC ↔ LSF ─────────────────────────────────────────────────────────────────

def _formant_lpc(freqs: list[float], radius: float = 0.95) -> np.ndarray:
    """Build a stable all-pole filter with poles at the given frequencies."""
    poly = np.array([1.0])
    for f in freqs:
        w = 2 * np.pi * f / SR
        poly = np.convolve(poly, [1.0, -2 * radius * np.cos(w), radius ** 2])
    return poly


def _is_stable(lpc: np.ndarray) -> bool:
    return bool(np.all(np.abs(np.roots(lpc)) < 1.0))


def test_lsf_roundtrip_reconstructs_the_polynomial():
    # rtol, not atol: the LSFs come from a 512-point zero-crossing search, so
    # accuracy is relative to the grid resolution rather than absolute.
    lpc = _formant_lpc([500.0, 1500.0, 2500.0, 3500.0])
    np.testing.assert_allclose(lsf_to_lpc(lpc_to_lsf(lpc)), lpc, rtol=1e-3)


def test_lsf_of_a_stable_filter_is_ordered_and_in_range():
    lsf = lpc_to_lsf(_formant_lpc([300.0, 1200.0, 2400.0, 3600.0]))
    assert np.all(np.diff(lsf) > 0)
    assert np.all((lsf > 0) & (lsf < np.pi))


def test_direct_coefficient_blending_can_go_unstable():
    """The defect the LSF path exists to avoid."""
    a = _formant_lpc([300.0, 900.0, 2700.0, 3300.0], radius=0.99)
    b = _formant_lpc([1700.0, 2100.0, 3900.0, 4300.0], radius=0.99)
    unstable = [t for t in np.linspace(0.05, 0.95, 19)
                if not _is_stable((1 - t) * a + t * b)]
    assert unstable, "expected the naive blend to leave the unit circle somewhere"


def test_interp_lpc_is_stable_across_the_whole_sweep():
    a = _formant_lpc([300.0, 900.0, 2700.0, 3300.0], radius=0.99)
    b = _formant_lpc([1700.0, 2100.0, 3900.0, 4300.0], radius=0.99)
    for t in np.linspace(0.0, 1.0, 41):
        assert _is_stable(interp_lpc(a, b, float(t))), f"unstable at t={t:.2f}"


def test_interp_lpc_endpoints_match_the_sources():
    a = _formant_lpc([400.0, 1400.0, 2400.0, 3400.0])
    b = _formant_lpc([700.0, 1100.0, 2900.0, 3100.0])
    np.testing.assert_allclose(interp_lpc(a, b, 0.0), a, rtol=1e-3)
    np.testing.assert_allclose(interp_lpc(a, b, 1.0), b, rtol=1e-3)


def test_interp_lpc_moves_formants_between_the_sources():
    """A single resonance should land between A's and B's, not split into two."""
    a = _formant_lpc([500.0])
    b = _formant_lpc([2000.0])
    mid = interp_lpc(a, b, 0.5)
    w, h = __import__("scipy.signal", fromlist=["freqz"]).freqz([1.0], mid, worN=2048, fs=SR)
    peak = float(w[np.argmax(np.abs(h))])
    assert 700.0 < peak < 1800.0


# ── Vocoder plugins: stereo ───────────────────────────────────────────────────

def _require_pyworld():
    """pyworld needs the plugin's pkg_resources shim on Python 3.14+."""
    from plugins.world_vocoder import _ensure_pyworld
    try:
        return _ensure_pyworld()
    except ImportError:
        pytest.skip("pyworld not installed")


def _stereo_noise(frames: int, seed: int, amp: float = 0.2) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((frames, 2)) * amp).astype(np.float32)


def test_lpc_vocoder_preserves_stereo():
    from plugins.vocoder import VocoderPlugin

    a, b = _stereo_noise(SR // 4, 10), _stereo_noise(SR // 4, 11)
    steps = VocoderPlugin().morph(a, b, steps=3, sample_rate=SR)
    for s in steps:
        assert s.shape == (SR // 4, 2)
        assert np.all(np.isfinite(s))
    # Channels must not be identical — that would mean a downmix happened.
    mid = steps[1]
    assert float(np.max(np.abs(mid[:, 0] - mid[:, 1]))) > 1e-4


def test_lpc_vocoder_mono_mode_still_downmixes():
    from plugins.vocoder import VocoderPlugin

    a, b = _stereo_noise(SR // 4, 10), _stereo_noise(SR // 4, 11)
    steps = VocoderPlugin().morph(a, b, steps=3, sample_rate=SR, channels="mono")
    for s in steps:
        assert s.shape == (SR // 4, 1)


def test_lpc_vocoder_accepts_mono_input():
    from plugins.vocoder import VocoderPlugin

    rng = np.random.default_rng(12)
    a = (rng.standard_normal((SR // 4, 1)) * 0.2).astype(np.float32)
    b = (rng.standard_normal((SR // 4, 1)) * 0.2).astype(np.float32)
    steps = VocoderPlugin().morph(a, b, steps=3, sample_rate=SR)
    assert all(s.shape == (SR // 4, 1) for s in steps)


def test_lpc_vocoder_output_stays_bounded_on_a_hard_case():
    """Two very different resonant sources: the old coefficient blend could blow up."""
    from plugins.vocoder import VocoderPlugin
    from scipy.signal import lfilter

    rng = np.random.default_rng(13)
    exc = rng.standard_normal(SR // 2)
    a = lfilter([1.0], _formant_lpc([300.0, 900.0], 0.99), exc)
    b = lfilter([1.0], _formant_lpc([2200.0, 3400.0], 0.99), exc)
    a = (a / np.max(np.abs(a)) * 0.5).astype(np.float32).reshape(-1, 1)
    b = (b / np.max(np.abs(b)) * 0.5).astype(np.float32).reshape(-1, 1)

    steps = VocoderPlugin().morph(a, b, steps=7, sample_rate=SR)
    for i, s in enumerate(steps):
        assert np.all(np.isfinite(s)), f"non-finite output at step {i}"
        assert float(np.max(np.abs(s))) <= 1.0


def test_lpc_vocoder_tracks_the_input_level():
    """A stable filter is not a quiet one.

    LSF interpolation guarantees the poles stay inside the unit circle, but poles
    close to it still ring hard. With a bounded level correction the synthesis
    could not be pulled back down and landed ~17 dB hot, clipping most samples.
    """
    from plugins.vocoder import VocoderPlugin
    from scipy.signal import lfilter

    rng = np.random.default_rng(16)
    exc = rng.standard_normal(SR // 2)
    a = lfilter([1.0], _formant_lpc([400.0, 1100.0], 0.995), exc)
    b = lfilter([1.0], _formant_lpc([1900.0, 3100.0], 0.995), exc)
    a = (a / np.max(np.abs(a)) * 0.3).astype(np.float32).reshape(-1, 1)
    b = (b / np.max(np.abs(b)) * 0.3).astype(np.float32).reshape(-1, 1)

    steps = VocoderPlugin().morph(a, b, steps=5, sample_rate=SR)
    reference = max(_rms(a), _rms(b))
    for i, s in enumerate(steps):
        assert _rms(s) < reference * 2.0, f"step {i} is {_rms(s) / reference:.1f}x too loud"
        at_full_scale = int(np.count_nonzero(np.abs(s) >= 0.999))
        assert at_full_scale < len(s) * 0.001, f"step {i} clips {at_full_scale} samples"


def test_lpc_vocoder_odd_order_is_rounded_to_even():
    """LSF conversion is only defined for even order."""
    from plugins.vocoder import VocoderPlugin

    a, b = _stereo_noise(SR // 8, 14), _stereo_noise(SR // 8, 15)
    steps = VocoderPlugin().morph(a, b, steps=3, sample_rate=SR, lpc_order=17)
    assert all(np.all(np.isfinite(s)) for s in steps)


def test_world_vocoder_preserves_stereo():
    _require_pyworld()
    from plugins.world_vocoder import WorldVocoderPlugin

    a, b = _tone(220.0, 0.4), _tone(330.0, 0.4)
    a = np.repeat(a, 2, axis=1).copy()
    b = np.repeat(b, 2, axis=1).copy()
    a[:, 1] *= 0.4          # give the channels a different level
    b[:, 1] *= 0.7

    steps = WorldVocoderPlugin().morph(a, b, steps=3, sample_rate=SR)
    for s in steps:
        assert s.shape == (a.shape[0], 2)
        assert np.all(np.isfinite(s))
    mid = steps[1]
    # The level difference between channels must survive the round trip.
    assert _rms(mid[:, 1]) < _rms(mid[:, 0]) * 0.9


def test_world_vocoder_mono_mode_still_downmixes():
    _require_pyworld()
    from plugins.world_vocoder import WorldVocoderPlugin

    a, b = _tone(220.0, 0.3), _tone(330.0, 0.3)
    a, b = np.repeat(a, 2, axis=1).copy(), np.repeat(b, 2, axis=1).copy()
    steps = WorldVocoderPlugin().morph(a, b, steps=3, sample_rate=SR, channels="mono")
    assert all(s.shape == (a.shape[0], 1) for s in steps)


def test_world_vocoder_channels_share_one_pitch_track():
    """Per-channel F0 estimation would let L and R drift apart into a chorus."""
    _require_pyworld()
    from plugins.world_vocoder import WorldVocoderPlugin

    a, b = _tone(220.0, 0.4), _tone(330.0, 0.4)
    a, b = np.repeat(a, 2, axis=1).copy(), np.repeat(b, 2, axis=1).copy()
    a[:, 1] *= 0.5

    mid = WorldVocoderPlugin().morph(a, b, steps=3, sample_rate=SR)[1]

    def dominant(sig):
        spec = np.abs(np.fft.rfft(sig * np.hanning(len(sig))))
        return float(np.fft.rfftfreq(len(sig), 1 / SR)[np.argmax(spec)])

    assert dominant(mid[:, 0]) == pytest.approx(dominant(mid[:, 1]), abs=1.0)


# ── DTW alignment ─────────────────────────────────────────────────────────────

def _dominant(sig: np.ndarray) -> float:
    mono = sig.mean(axis=1) if sig.ndim == 2 else sig
    spec = np.abs(np.fft.rfft(mono * np.hanning(len(mono))))
    return float(np.fft.rfftfreq(len(mono), 1 / SR)[np.argmax(spec)])


def _ramped_tone(freq: float, seconds: float, rate: float) -> np.ndarray:
    """A steady tone whose *events* run at `rate`, so DTW has to stretch it."""
    n = int(SR * seconds)
    t = np.arange(n) / SR
    tone = np.sin(2 * np.pi * freq * t)
    # Amplitude bursts at a tempo scaled by `rate` give the MFCCs something to
    # align on without changing the pitch.
    env = 0.5 + 0.5 * np.sin(2 * np.pi * 3.0 * rate * t)
    return (0.5 * tone * env).astype(np.float32).reshape(-1, 1)


def test_dtw_stretch_preserves_pitch():
    """The old resampling path slid pitch wherever the warp left the diagonal."""
    a = _ramped_tone(440.0, 1.5, rate=1.0)
    b = _ramped_tone(440.0, 1.5, rate=1.7)

    warped_a, warped_b = dtw_align(a, b, SR)
    assert _dominant(warped_a) == pytest.approx(440.0, abs=12.0)
    assert _dominant(warped_b) == pytest.approx(440.0, abs=12.0)


def test_dtw_resample_mode_still_slides_pitch():
    """Kept reachable, so the fix is demonstrably the cause of the improvement."""
    a = _ramped_tone(440.0, 1.5, rate=1.0)
    b = _ramped_tone(440.0, 1.5, rate=1.7)

    stretched, _ = dtw_align(a, b, SR)
    resampled, _ = dtw_align(a, b, SR, mode="resample")

    def spread(sig):
        mono = sig.mean(axis=1) if sig.ndim == 2 else sig
        spec = np.abs(np.fft.rfft(mono * np.hanning(len(mono))))
        freqs = np.fft.rfftfreq(len(mono), 1 / SR)
        band = (freqs > 200) & (freqs < 900)
        w = spec[band] / (spec[band].sum() + 1e-12)
        centre = float((freqs[band] * w).sum())
        return float(np.sqrt((w * (freqs[band] - centre) ** 2).sum()))

    # Varispeed smears the 440 Hz partial across a band; the stretch does not.
    assert spread(resampled) > spread(stretched)


def test_dtw_output_lengths_match():
    a = _ramped_tone(330.0, 1.0, rate=1.0)
    b = _ramped_tone(330.0, 1.4, rate=1.5)
    warped_a, warped_b = dtw_align(a, b, SR)
    assert len(warped_a) == len(warped_b) == max(len(a), len(b))


def test_dtw_preserves_shape_and_channels():
    rng = np.random.default_rng(20)
    a = (rng.standard_normal((SR, 2)) * 0.2).astype(np.float32)
    b = (rng.standard_normal((SR, 2)) * 0.2).astype(np.float32)
    warped_a, warped_b = dtw_align(a, b, SR)
    assert warped_a.shape == warped_b.shape == (SR, 2)
    assert np.all(np.isfinite(warped_a)) and np.all(np.isfinite(warped_b))


def test_dtw_handles_signals_shorter_than_one_fft_frame():
    a = _tone(440.0, 0.01)
    b = _tone(660.0, 0.01)
    warped_a, warped_b = dtw_align(a, b, SR)
    assert np.all(np.isfinite(warped_a)) and np.all(np.isfinite(warped_b))


# ── Export dither ─────────────────────────────────────────────────────────────

def test_dither_is_one_lsb_of_the_target_depth():
    from app.export import apply_tpdf_dither

    silence = np.zeros((200000, 1), dtype=np.float32)
    dithered = apply_tpdf_dither(silence, 16, np.random.default_rng(0))
    lsb = 1.0 / 2 ** 15

    assert float(np.max(np.abs(dithered))) <= lsb * 1.001
    # Triangular distribution: variance is 2/12 of a uniform LSB draw.
    assert float(np.std(dithered)) == pytest.approx(lsb * np.sqrt(2 / 12), rel=0.05)


def test_dither_turns_quantisation_distortion_into_noise():
    """The point of dither, measured where it is audible: the spectrum.

    A quiet sine quantised without dither comes back as a staircase, whose error
    is a deterministic function of the signal. The spectrum is then nothing but
    discrete harmonics — distortion tones at 6.5 % of the fundamental with an
    empty floor between them. Dither trades those for an ordinary noise floor.
    """
    from app.export import apply_tpdf_dither

    n_samples, cycles = 16384, 61     # exact number of cycles: no leakage
    lsb = 1.0 / 2 ** 15
    phase = 2 * np.pi * cycles * np.arange(n_samples) / n_samples
    signal = (2.0 * lsb * np.sin(phase)).astype(np.float32).reshape(-1, 1)

    def quantise(x):
        return np.round(x * 2 ** 15) / 2 ** 15

    def measure(x):
        spec = np.abs(np.fft.rfft(quantise(x).ravel()))
        harmonics = max(spec[3 * cycles], spec[5 * cycles], spec[7 * cycles])
        return float(np.median(spec)), float(spec[cycles]), float(harmonics)

    plain_floor, plain_fund, plain_harm = measure(signal)
    dith_floor, dith_fund, dith_harm = measure(
        apply_tpdf_dither(signal, 16, np.random.default_rng(1))
    )

    # Undithered: no noise floor at all, just harmonic distortion.
    assert plain_floor < plain_fund * 1e-6
    assert plain_harm > plain_fund * 0.03

    # Dithered: a real floor, with the distortion tones buried in it and the
    # signal still well clear of it.
    assert dith_floor > 0.0
    assert dith_harm < dith_floor * 6.0
    assert dith_fund > dith_floor * 50.0


def test_dither_never_exceeds_full_scale():
    from app.export import apply_tpdf_dither

    hot = np.full((10000, 2), 1.0, dtype=np.float32)
    out = apply_tpdf_dither(hot, 16, np.random.default_rng(2))
    assert float(np.max(np.abs(out))) <= 1.0
    assert out.dtype == np.float32


def test_export_writes_dithered_and_undithered_files(tmp_path):
    import soundfile as sf
    from app.export import _ExportWorker

    silence = [np.zeros((20000, 1), dtype=np.float32)]

    plain = tmp_path / "plain"
    _ExportWorker(silence, plain, SR, 16, dither=False).run()
    data, _ = sf.read(str(plain / "morph_step_01.wav"), always_2d=True)
    assert float(np.max(np.abs(data))) == 0.0

    dithered = tmp_path / "dithered"
    _ExportWorker(silence, dithered, SR, 16, dither=True).run()
    data, _ = sf.read(str(dithered / "morph_step_01.wav"), always_2d=True)
    assert float(np.max(np.abs(data))) > 0.0


def test_export_skips_dither_at_24_bit(tmp_path):
    import soundfile as sf
    from app.export import _ExportWorker

    silence = [np.zeros((20000, 1), dtype=np.float32)]
    out = tmp_path / "wide"
    _ExportWorker(silence, out, SR, 24, dither=True).run()
    data, _ = sf.read(str(out / "morph_step_01.wav"), always_2d=True)
    assert float(np.max(np.abs(data))) == 0.0


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
