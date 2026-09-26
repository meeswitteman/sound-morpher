"""Input conditioning: hot float WAVs and DC removal.

Both used to cost quality before a morph even started: float sources over
full scale were clipped on load, and the 20 Hz DC filter took 1-3 dB off the
fundamentals of kicks and bass.
"""
from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

from app.audio_engine import AudioEngine, _remove_dc_offset

SR = 44100


def _sine(freq: float, seconds: float = 2.0, amp: float = 0.5) -> np.ndarray:
    t = np.arange(int(SR * seconds)) / SR
    return (amp * np.sin(2 * np.pi * freq * t)).reshape(-1, 1)


def _level_db(x: np.ndarray) -> float:
    body = x[SR // 4 : -SR // 4]
    return float(20 * np.log10(np.sqrt(np.mean(body.astype(np.float64) ** 2))))


# ── Hot float WAVs ────────────────────────────────────────────────────────────

def test_hot_float_wav_keeps_its_waveform(tmp_path):
    """A 1.6x sine scaled back is still a sine; clipped, it was squared off."""
    wav = tmp_path / "hot.wav"
    hot = _sine(220.0, 0.5, amp=1.6).astype(np.float32)
    sf.write(str(wav), hot, SR, subtype="FLOAT")

    engine = AudioEngine()
    audio, _ = engine.load_wav(str(wav))

    np.testing.assert_allclose(audio, hot / 1.6, atol=1e-6)
    assert float(np.max(np.abs(audio))) <= 1.0
    assert engine.last_load_trim_db == pytest.approx(-20 * np.log10(1.6), abs=1e-3)


def test_in_range_files_are_untouched(tmp_path):
    wav = tmp_path / "ok.wav"
    ok = _sine(220.0, 0.5, amp=0.9).astype(np.float32)
    sf.write(str(wav), ok, SR, subtype="FLOAT")

    engine = AudioEngine()
    engine.last_load_trim_db = -5.0      # stale value from an earlier load
    audio, _ = engine.load_wav(str(wav))

    np.testing.assert_array_equal(audio, ok)
    assert engine.last_load_trim_db == 0.0


# ── DC removal ────────────────────────────────────────────────────────────────

def test_dc_offset_is_removed():
    x = _sine(440.0) + 0.2
    out = _remove_dc_offset(x, SR)
    assert abs(float(out[SR // 4 : -SR // 4].mean())) < 1e-3


def test_offset_does_not_thump_at_the_start():
    """A constant offset must not turn into a decaying step at sample 0."""
    x = np.full((SR, 1), 0.3)
    out = _remove_dc_offset(x, SR)
    assert float(np.max(np.abs(out))) < 1e-3


@pytest.mark.parametrize("freq, max_loss_db", [(20.0, 0.6), (30.0, 0.3), (40.0, 0.2), (60.0, 0.1)])
def test_sub_bass_keeps_its_level(freq, max_loss_db):
    """The old filter lost 3.0 / 1.6 / 1.0 / 0.5 dB at these frequencies."""
    x = _sine(freq)
    loss = _level_db(x) - _level_db(_remove_dc_offset(x, SR))
    assert loss < max_loss_db


def test_filter_is_zero_phase():
    """A kick-like low sine must not be delayed or reshaped."""
    x = _sine(50.0)
    out = _remove_dc_offset(x, SR)
    body = slice(SR // 4, -SR // 4)
    corr = np.correlate(out[body].ravel(), x[body].ravel(), mode="full")
    lag = int(np.argmax(corr)) - (len(x[body]) - 1)
    assert lag == 0
    np.testing.assert_allclose(out[body], x[body] * (out[body].std() / x[body].std()), atol=1e-3)


def test_very_short_input_still_loses_its_offset():
    x = np.array([[0.5], [0.6], [0.4]])
    out = _remove_dc_offset(x, SR)
    assert out.shape == x.shape
    assert abs(float(out.mean())) < 1e-6
