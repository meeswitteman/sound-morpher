"""Phase 7 tests: additional plugins and PluginParamPanel."""
import numpy as np
import pytest


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _stereo(n=4410, val=0.0):
    return np.full((n, 2), val, dtype=np.float32)


# ── SpectralFftPlugin ─────────────────────────────────────────────────────────

def test_spectral_fft_import():
    from plugins.spectral_fft import SpectralFftPlugin
    assert SpectralFftPlugin.name == "Spectral FFT"


def test_spectral_fft_output_shape():
    from plugins.spectral_fft import SpectralFftPlugin
    plugin = SpectralFftPlugin()
    a = _stereo(4410, 0.1)
    b = _stereo(4410, 0.9)
    steps = plugin.morph(a, b, steps=4, sample_rate=44100)
    assert len(steps) == 4
    for s in steps:
        assert s.shape == (4410, 2)
        assert s.dtype == np.float32


def test_spectral_fft_endpoint_a():
    """Step 0 should be close to A."""
    from plugins.spectral_fft import SpectralFftPlugin
    plugin = SpectralFftPlugin()
    a = _stereo(4410, 0.2)
    b = _stereo(4410, 0.8)
    steps = plugin.morph(a, b, steps=4, sample_rate=44100)
    np.testing.assert_allclose(steps[0], a, atol=0.05)


def test_spectral_fft_endpoint_b():
    """Last step should be close to B."""
    from plugins.spectral_fft import SpectralFftPlugin
    plugin = SpectralFftPlugin()
    a = _stereo(4410, 0.2)
    b = _stereo(4410, 0.8)
    steps = plugin.morph(a, b, steps=4, sample_rate=44100)
    np.testing.assert_allclose(steps[-1], b, atol=0.05)


def test_spectral_fft_fft_size_param():
    from plugins.spectral_fft import SpectralFftPlugin
    plugin = SpectralFftPlugin()
    a = _stereo(4410, 0.3)
    b = _stereo(4410, 0.7)
    steps = plugin.morph(a, b, steps=3, sample_rate=44100, fft_size="512")
    assert len(steps) == 3


# ── PitchShiftPlugin ──────────────────────────────────────────────────────────

def test_pitch_shift_import():
    from plugins.pitch_shift import PitchShiftPlugin
    assert PitchShiftPlugin.name == "Pitch Shift"


def test_pitch_shift_output_shape():
    from plugins.pitch_shift import PitchShiftPlugin
    plugin = PitchShiftPlugin()
    a = _stereo(4410, 0.2)
    b = _stereo(4410, 0.5)
    steps = plugin.morph(a, b, steps=3, sample_rate=44100)
    assert len(steps) == 3
    for s in steps:
        assert s.shape == (4410, 2)
        assert s.dtype == np.float32


def test_pitch_shift_unpitched_fallback():
    """Silent audio has no detectable pitch — plugin should not crash."""
    from plugins.pitch_shift import PitchShiftPlugin
    plugin = PitchShiftPlugin()
    a = np.zeros((4410, 2), dtype=np.float32)
    b = np.zeros((4410, 2), dtype=np.float32)
    steps = plugin.morph(a, b, steps=2, sample_rate=44100)
    assert len(steps) == 2


# ── GranularPlugin ────────────────────────────────────────────────────────────

def test_granular_import():
    from plugins.granular import GranularPlugin
    assert GranularPlugin.name == "Granular"


def test_granular_output_shape():
    from plugins.granular import GranularPlugin
    plugin = GranularPlugin()
    a = _stereo(8820, 0.2)
    b = _stereo(8820, 0.8)
    steps = plugin.morph(a, b, steps=4, sample_rate=44100)
    assert len(steps) == 4
    for s in steps:
        assert s.shape == (8820, 2)
        assert s.dtype == np.float32


def test_granular_step0_close_to_a():
    from plugins.granular import GranularPlugin
    plugin = GranularPlugin()
    a = _stereo(8820, 0.5)
    b = _stereo(8820, 0.0)
    steps = plugin.morph(a, b, steps=4, sample_rate=44100)
    # Step 0 is 100% A: mean should be near 0.5
    assert abs(float(steps[0].mean()) - 0.5) < 0.05


def test_granular_last_step_close_to_b():
    from plugins.granular import GranularPlugin
    plugin = GranularPlugin()
    a = _stereo(8820, 0.0)
    b = _stereo(8820, 0.4)
    steps = plugin.morph(a, b, steps=4, sample_rate=44100)
    assert abs(float(steps[-1].mean()) - 0.4) < 0.05


def test_granular_grain_param():
    from plugins.granular import GranularPlugin
    plugin = GranularPlugin()
    a = _stereo(8820, 0.3)
    b = _stereo(8820, 0.7)
    steps = plugin.morph(a, b, steps=3, sample_rate=44100, grain_ms=40.0, overlap=0.6)
    assert len(steps) == 3


# ── Registry ─────────────────────────────────────────────────────────────────

def test_registry_has_all_plugins():
    from plugins.registry import build_default_registry
    reg = build_default_registry()
    assert "Crossfade" in reg
    assert "Spectral FFT" in reg
    assert "Pitch Shift" in reg
    assert "Granular" in reg
    assert "Vocoder" in reg
    assert "WORLD Vocoder" in reg
    assert len(reg) == 6


# ── PluginParamPanel ──────────────────────────────────────────────────────────

def test_param_panel_loads_crossfade(qt_app):
    from app.widgets.plugin_param_panel import PluginParamPanel
    from plugins.crossfade import CrossfadePlugin
    panel = PluginParamPanel()
    panel.load_plugin(CrossfadePlugin())
    params = panel.get_params()
    assert "curve" in params
    assert params["curve"] == "linear"


def test_param_panel_loads_spectral_fft(qt_app):
    from app.widgets.plugin_param_panel import PluginParamPanel
    from plugins.spectral_fft import SpectralFftPlugin
    panel = PluginParamPanel()
    panel.load_plugin(SpectralFftPlugin())
    params = panel.get_params()
    assert "fft_size" in params
    assert "overlap" in params
    assert params["fft_size"] == "1024"
    assert int(params["overlap"]) == 75


def test_param_panel_set_params(qt_app):
    from app.widgets.plugin_param_panel import PluginParamPanel
    from plugins.spectral_fft import SpectralFftPlugin
    panel = PluginParamPanel()
    panel.load_plugin(SpectralFftPlugin())
    panel.set_params({"fft_size": "512", "overlap": 60})
    params = panel.get_params()
    assert params["fft_size"] == "512"
    assert int(params["overlap"]) == 60


def test_param_panel_granular(qt_app):
    from app.widgets.plugin_param_panel import PluginParamPanel
    from plugins.granular import GranularPlugin
    panel = PluginParamPanel()
    panel.load_plugin(GranularPlugin())
    params = panel.get_params()
    assert "grain_ms" in params
    assert "overlap" in params


def test_param_panel_pitch_shift(qt_app):
    from app.widgets.plugin_param_panel import PluginParamPanel
    from plugins.pitch_shift import PitchShiftPlugin
    panel = PluginParamPanel()
    panel.load_plugin(PitchShiftPlugin())
    params = panel.get_params()
    assert "fmin" in params
    assert "fmax" in params


# ── ProjectState.algorithm_params ────────────────────────────────────────────

def test_project_state_algorithm_params_default():
    from app.project_state import ProjectState
    s = ProjectState()
    assert isinstance(s.algorithm_params, dict)
    assert s.algorithm_params == {}


# ── ProjectFile round-trip preserves algorithm_params ─────────────────────────

def test_project_file_roundtrip_algorithm_params(tmp_path):
    from app.project_file import ProjectFile
    from app.project_state import ProjectState
    state = ProjectState(
        sample_rate=44100,
        bit_depth=16,
        name_a="a.wav",
        name_b="b.wav",
        algorithm="Spectral FFT",
        algorithm_params={"fft_size": "512", "overlap": 60},
    )
    state.audio_a = np.zeros((4410, 2), dtype=np.float32)
    state.audio_b = np.ones((4410, 2), dtype=np.float32) * 0.5
    out = tmp_path / "test.smorph"
    ProjectFile.save(str(out), state)
    loaded = ProjectFile.load(str(out))
    assert loaded.algorithm == "Spectral FFT"
    assert loaded.algorithm_params["fft_size"] == "512"
    assert int(loaded.algorithm_params["overlap"]) == 60
