import numpy as np
import pytest

from app.project_state import ProjectState
from app.audio_engine import AudioEngine


def test_project_state_defaults():
    p = ProjectState()
    assert p.sample_rate == 44100
    assert p.steps == 8
    assert not p.ready_to_morph


def test_project_state_ready_to_morph():
    p = ProjectState()
    p.audio_a = np.zeros((1000, 2), dtype=np.float32)
    assert not p.ready_to_morph
    p.audio_b = np.zeros((1000, 2), dtype=np.float32)
    assert p.ready_to_morph


def test_waveform_widget_import():
    from app.widgets.waveform_widget import WaveformWidget
    assert WaveformWidget is not None


def test_sound_slot_import():
    from app.widgets.sound_slot import SoundSlot
    assert SoundSlot is not None


def test_recording_panel_import():
    from app.widgets.recording_panel import RecordingPanel
    assert RecordingPanel is not None
