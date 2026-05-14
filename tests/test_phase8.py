"""Phase 8 tests: export morph steps as individual WAV files."""
import time
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from PySide6.QtCore import QCoreApplication


def _steps(n=4, frames=4410):
    return [
        np.full((frames, 2), i / (n - 1), dtype=np.float32)
        for i in range(n)
    ]


# ── ExportEngine unit tests ───────────────────────────────────────────────────

def test_export_engine_import():
    from app.export import ExportEngine
    assert ExportEngine is not None


def test_export_writes_wav_files(tmp_path):
    """ExportEngine writes one WAV per step with correct naming."""
    from app.export import ExportEngine

    engine = ExportEngine()
    steps = _steps(4)
    done: list[str] = []
    engine.finished.connect(done.append)

    engine.export(steps, tmp_path, sample_rate=44100, bit_depth=16)

    deadline = time.perf_counter() + 5.0
    while not done and time.perf_counter() < deadline:
        QCoreApplication.processEvents()
        time.sleep(0.01)

    assert done, "finished signal not emitted"

    for i in range(1, 5):
        f = tmp_path / f"morph_step_{i:02d}.wav"
        assert f.exists(), f"{f.name} not found"


def test_export_file_count(tmp_path):
    from app.export import ExportEngine

    engine = ExportEngine()
    n = 6
    steps = _steps(n)
    done: list[str] = []
    engine.finished.connect(done.append)
    engine.export(steps, tmp_path, sample_rate=44100, bit_depth=16)

    deadline = time.perf_counter() + 5.0
    while not done and time.perf_counter() < deadline:
        QCoreApplication.processEvents()
        time.sleep(0.01)

    wavs = list(tmp_path.glob("morph_step_*.wav"))
    assert len(wavs) == n


def test_export_wav_readable_and_correct_sr(tmp_path):
    """Exported WAV files must be readable and have the correct sample rate."""
    from app.export import ExportEngine

    engine = ExportEngine()
    steps = _steps(2, frames=8820)
    done: list[str] = []
    engine.finished.connect(done.append)
    engine.export(steps, tmp_path, sample_rate=48000, bit_depth=16)

    deadline = time.perf_counter() + 5.0
    while not done and time.perf_counter() < deadline:
        QCoreApplication.processEvents()
        time.sleep(0.01)

    audio, sr = sf.read(str(tmp_path / "morph_step_01.wav"))
    assert sr == 48000
    assert audio.shape == (8820, 2)


def test_export_24bit(tmp_path):
    """bit_depth=24 produces PCM_24 files (soundfile can read them)."""
    from app.export import ExportEngine

    engine = ExportEngine()
    steps = _steps(2)
    done: list[str] = []
    engine.finished.connect(done.append)
    engine.export(steps, tmp_path, sample_rate=44100, bit_depth=24)

    deadline = time.perf_counter() + 5.0
    while not done and time.perf_counter() < deadline:
        QCoreApplication.processEvents()
        time.sleep(0.01)

    f = tmp_path / "morph_step_01.wav"
    info = sf.info(str(f))
    assert "PCM_24" in info.subtype


def test_export_creates_output_dir(tmp_path):
    """ExportEngine creates the output directory if it doesn't exist yet."""
    from app.export import ExportEngine

    engine = ExportEngine()
    out = tmp_path / "new_subdir" / "export"
    done: list[str] = []
    engine.finished.connect(done.append)
    engine.export(_steps(2), out, sample_rate=44100, bit_depth=16)

    deadline = time.perf_counter() + 5.0
    while not done and time.perf_counter() < deadline:
        QCoreApplication.processEvents()
        time.sleep(0.01)

    assert out.exists()
    assert len(list(out.glob("*.wav"))) == 2


def test_export_emits_progress(tmp_path):
    """progress signal is emitted during export."""
    from app.export import ExportEngine

    engine = ExportEngine()
    progress_vals: list[int] = []
    engine.progress.connect(progress_vals.append)
    done: list[str] = []
    engine.finished.connect(done.append)

    engine.export(_steps(4), tmp_path, sample_rate=44100, bit_depth=16)

    deadline = time.perf_counter() + 5.0
    while not done and time.perf_counter() < deadline:
        QCoreApplication.processEvents()
        time.sleep(0.01)

    assert progress_vals, "no progress values emitted"
    assert 100 in progress_vals


def test_export_finished_path_matches_output_dir(tmp_path):
    from app.export import ExportEngine

    engine = ExportEngine()
    done: list[str] = []
    engine.finished.connect(done.append)
    engine.export(_steps(2), tmp_path, sample_rate=44100, bit_depth=16)

    deadline = time.perf_counter() + 5.0
    while not done and time.perf_counter() < deadline:
        QCoreApplication.processEvents()
        time.sleep(0.01)

    assert Path(done[0]).resolve() == tmp_path.resolve()


def test_export_error_on_invalid_path(qt_app):
    """ExportEngine emits error when the path is not writable."""
    from app.export import ExportEngine

    engine = ExportEngine()
    errors: list[str] = []
    engine.error.connect(errors.append)

    # Pass a path that is actually a file (not a directory)
    import tempfile, os
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as f:
        file_path = f.name

    try:
        # Try to write into a file as if it were a directory — will fail
        invalid_dir = Path(file_path) / "subdir"
        engine.export(_steps(2), invalid_dir, sample_rate=44100, bit_depth=16)

        deadline = time.perf_counter() + 3.0
        while not errors and time.perf_counter() < deadline:
            QCoreApplication.processEvents()
            time.sleep(0.01)

        assert errors, "error signal not emitted for invalid path"
    finally:
        os.unlink(file_path)
