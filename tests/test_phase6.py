import io
import json
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pytest

from app.project_file import ProjectFile, ProjectFileError
from app.project_state import ProjectState


def _make_state() -> ProjectState:
    state = ProjectState(sample_rate=44100, bit_depth=16, name_a="a.wav", name_b="b.wav")
    state.audio_a = np.zeros((4410, 2), dtype=np.float32)
    state.audio_b = np.ones((4410, 2), dtype=np.float32) * 0.5
    state.algorithm = "Crossfade"
    state.bpm = 130
    state.beats_per_step = 2
    state.loop = True
    state.steps = 4
    state.morph_steps = [np.zeros((4410, 2), dtype=np.float32) for _ in range(4)]
    return state


def test_save_creates_smorph(tmp_path):
    state = _make_state()
    out = tmp_path / "test.smorph"
    ProjectFile.save(str(out), state)
    assert out.exists()
    assert zipfile.is_zipfile(out)


def test_save_appends_smorph_extension(tmp_path):
    state = _make_state()
    out = tmp_path / "noext"
    ProjectFile.save(str(out), state)
    assert (tmp_path / "noext.smorph").exists()


def test_smorph_contains_expected_entries(tmp_path):
    state = _make_state()
    out = tmp_path / "test.smorph"
    ProjectFile.save(str(out), state)
    with zipfile.ZipFile(out) as zf:
        names = set(zf.namelist())
    assert "project.json" in names
    assert "audio/source_a.wav" in names
    assert "audio/source_b.wav" in names
    for i in range(1, 5):
        assert f"audio/step_{i:02d}.wav" in names


def test_project_json_schema(tmp_path):
    state = _make_state()
    out = tmp_path / "test.smorph"
    ProjectFile.save(str(out), state)
    with zipfile.ZipFile(out) as zf:
        meta = json.loads(zf.read("project.json"))
    assert meta["sample_rate"] == 44100
    assert meta["algorithm"] == "Crossfade"
    assert meta["bpm"] == 130
    assert meta["beats_per_step"] == 2
    assert meta["loop"] is True
    assert meta["step_count"] == 4


def test_roundtrip(tmp_path):
    state = _make_state()
    out = tmp_path / "round.smorph"
    ProjectFile.save(str(out), state)
    loaded = ProjectFile.load(str(out))
    assert loaded.sample_rate == 44100
    assert loaded.bpm == 130
    assert loaded.beats_per_step == 2
    assert loaded.loop is True
    assert loaded.name_a == "a.wav"
    assert loaded.name_b == "b.wav"
    assert loaded.audio_a is not None
    assert loaded.audio_b is not None
    assert len(loaded.morph_steps) == 4
    np.testing.assert_allclose(loaded.audio_b, state.audio_b, atol=1e-3)


def test_load_missing_file():
    with pytest.raises(ProjectFileError, match="not found"):
        ProjectFile.load("/nonexistent/path.smorph")


def test_load_invalid_zip(tmp_path):
    bad = tmp_path / "bad.smorph"
    bad.write_bytes(b"this is not a zip file")
    with pytest.raises(ProjectFileError):
        ProjectFile.load(str(bad))


def test_state_file_path_set_after_save(tmp_path):
    state = _make_state()
    out = tmp_path / "p.smorph"
    ProjectFile.save(str(out), state)
    assert state.file_path == str(out)
