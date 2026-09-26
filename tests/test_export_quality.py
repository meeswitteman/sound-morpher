"""Output-quality guarantees around export depth, project storage and peaks.

Covers three regressions that cost quality without anything failing loudly:
  * export was locked to 16 bit with no way to choose 24 bit or float
  * the .smorph archive re-quantised all audio to the export depth (16 bit,
    undithered) on every save, so each save/load round trip lost resolution
  * the Vocoder hard-clipped its output before the engine's limiter saw it
"""
from __future__ import annotations

import io
import json
import zipfile

import numpy as np
import pytest
import soundfile as sf

SR = 44100


# ── Export bit depth ──────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "depth, subtype", [(16, "PCM_16"), (24, "PCM_24"), (32, "FLOAT")]
)
def test_bit_depth_maps_to_wav_subtype(depth, subtype):
    from app.export import subtype_for_bit_depth

    assert subtype_for_bit_depth(depth) == subtype


def test_float_export_is_bit_exact_and_undithered(tmp_path):
    from app.export import _ExportWorker

    rng = np.random.default_rng(0)
    step = (rng.standard_normal((8000, 2)) * 0.1).astype(np.float32)
    silence = np.zeros((8000, 2), dtype=np.float32)

    _ExportWorker([step, silence], tmp_path, SR, 32, dither=True).run()

    assert sf.info(str(tmp_path / "morph_step_01.wav")).subtype == "FLOAT"
    data, _ = sf.read(str(tmp_path / "morph_step_01.wav"), dtype="float32", always_2d=True)
    np.testing.assert_array_equal(data, step)
    quiet, _ = sf.read(str(tmp_path / "morph_step_02.wav"), always_2d=True)
    assert float(np.max(np.abs(quiet))) == 0.0


# ── Project archive ───────────────────────────────────────────────────────────

def _state_with_fine_detail():
    from app.project_state import ProjectState

    rng = np.random.default_rng(1)
    # Values far off the 16-bit grid, so any re-quantisation shows up.
    fine = lambda: (rng.standard_normal((4000, 2)) * 1e-3).astype(np.float32)  # noqa: E731
    state = ProjectState(sample_rate=SR, bit_depth=16)
    state.audio_a = fine()
    state.audio_b = fine()
    state.morph_steps = [fine() for _ in range(3)]
    return state


def test_project_round_trip_is_lossless_at_any_export_depth(tmp_path):
    from app.project_file import ProjectFile

    state = _state_with_fine_detail()
    path = tmp_path / "p.smorph"
    ProjectFile.save(path, state)
    loaded = ProjectFile.load(path)

    np.testing.assert_array_equal(loaded.audio_a, state.audio_a)
    np.testing.assert_array_equal(loaded.audio_b, state.audio_b)
    for got, want in zip(loaded.morph_steps, state.morph_steps):
        np.testing.assert_array_equal(got, want)
    # The export depth is still remembered; it just no longer governs storage.
    assert loaded.bit_depth == 16


def test_project_stores_embedded_audio_as_float(tmp_path):
    from app.project_file import ProjectFile

    path = tmp_path / "p.smorph"
    ProjectFile.save(path, _state_with_fine_detail())
    with zipfile.ZipFile(path) as zf:
        for name in zf.namelist():
            if name.endswith(".wav"):
                assert sf.info(io.BytesIO(zf.read(name))).subtype == "FLOAT", name


def test_old_16_bit_projects_still_load(tmp_path):
    from app.project_file import ProjectFile

    audio = (np.sin(np.arange(4000) / 20.0) * 0.5).astype(np.float32).reshape(-1, 1)
    buf = io.BytesIO()
    sf.write(buf, audio, SR, format="WAV", subtype="PCM_16")

    path = tmp_path / "old.smorph"
    meta = {"version": "1.0", "sample_rate": SR, "bit_depth": 16, "step_count": 0}
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("project.json", json.dumps(meta))
        zf.writestr("audio/source_a.wav", buf.getvalue())

    loaded = ProjectFile.load(path)
    np.testing.assert_allclose(loaded.audio_a, audio, atol=1.0 / 32768)


# ── Peak handling ─────────────────────────────────────────────────────────────

def test_vocoder_limits_overshoot_instead_of_clipping(monkeypatch):
    """A hot synthesis frame must come out gain-reduced, not squared off."""
    import plugins.vocoder as vocoder

    n = SR // 4
    tone = np.sin(2 * np.pi * 220.0 * np.arange(n) / SR)
    monkeypatch.setattr(vocoder, "_synthesise", lambda *a, **k: 1.6 * tone)

    a = np.zeros((n, 2), dtype=np.float32)
    steps = vocoder.VocoderPlugin().morph(a, a, steps=2, sample_rate=SR)

    for s in steps:
        assert float(np.max(np.abs(s))) <= 1.0
        # A 1.6x sine clipped at 1.0 sits at full scale for ~57 % of its samples;
        # a limited one only touches the ceiling at the tip of each crest.
        at_ceiling = np.count_nonzero(np.abs(s[:, 0]) >= 0.999)
        assert at_ceiling < n * 0.05
        # The limiter shares one gain across channels: the image cannot move.
        np.testing.assert_array_equal(s[:, 0], s[:, 1])


class _HotPlugin:
    name = "hot"

    def morph(self, a, b, steps, sample_rate, progress_cb=None, **_):
        tone = np.sin(2 * np.pi * 330.0 * np.arange(len(a)) / sample_rate)
        return [(1.5 * tone).astype(np.float32).reshape(-1, 1) for _ in range(steps)]


def test_engine_limits_peaks_even_without_level_matching():
    from app.morph_engine import _Worker

    a = np.zeros((SR // 4, 1), dtype=np.float32)
    worker = _Worker(_HotPlugin(), a, a, 3, SR, {}, level_match=False)
    results: list = []
    errors: list = []
    worker.signals.finished.connect(results.append)
    worker.signals.error.connect(errors.append)
    worker.setAutoDelete(False)
    worker.run()

    assert not errors
    assert len(results) == 1
    for s in results[0]:
        assert float(np.max(np.abs(s))) <= 0.99 + 1e-6


# ── UI ────────────────────────────────────────────────────────────────────────

@pytest.fixture
def window(tmp_path, monkeypatch, qt_app):
    """MainWindow with its QSettings redirected to a throwaway INI file."""
    from PySide6.QtCore import QSettings

    import app.main_window as mw

    ini = str(tmp_path / "settings.ini")
    monkeypatch.setattr(mw, "QSettings", lambda *_: QSettings(ini, QSettings.Format.IniFormat))
    win = mw.MainWindow()
    yield win
    win.close()
    win.deleteLater()


def test_bit_depth_menu_drives_project_and_dither(window):
    assert window.project.bit_depth == 16
    assert window._act_dither.isEnabled()

    window._bit_depth_actions[24].trigger()
    assert window.project.bit_depth == 24
    assert not window._act_dither.isEnabled()
    assert "24-bit" in window.lbl_status_sr.text()

    window._bit_depth_actions[32].trigger()
    assert window.project.bit_depth == 32
    assert "float" in window.lbl_status_sr.text()


def test_new_project_remembers_last_bit_depth(window, monkeypatch):
    window._bit_depth_actions[24].trigger()
    monkeypatch.setattr(window, "_confirm_discard", lambda: True)
    window._on_new()
    assert window.project.bit_depth == 24
    assert window._bit_depth_actions[24].isChecked()
