import numpy as np
import pytest


def _stereo(frames: int, value: float = 0.5) -> np.ndarray:
    return np.full((frames, 2), value, dtype=np.float32)


def test_spectrogram_pixmap_returns_pixmap():
    from app.widgets.spectrogram import compute_spectrogram_pixmap
    audio = _stereo(22050)
    pixmap = compute_spectrogram_pixmap(audio, sample_rate=44100, width=140, height=110)
    assert not pixmap.isNull()
    assert pixmap.width() == 140
    assert pixmap.height() == 110


def test_spectrogram_mono():
    from app.widgets.spectrogram import compute_spectrogram_pixmap
    audio = np.random.rand(22050).astype(np.float32)
    pixmap = compute_spectrogram_pixmap(audio, 44100, 100, 80)
    assert not pixmap.isNull()


def test_spectrogram_short_audio():
    from app.widgets.spectrogram import compute_spectrogram_pixmap
    audio = _stereo(256)
    pixmap = compute_spectrogram_pixmap(audio, 44100, 60, 50)
    assert not pixmap.isNull()


def test_step_tile_import():
    from app.widgets.step_tile import StepTile, TILE_W, TILE_H
    assert TILE_W > 0
    assert TILE_H > 0


def test_step_grid_import():
    from app.widgets.step_grid import StepGrid
    assert StepGrid is not None
