import numpy as np
import pytest

from app.audio_engine import AudioEngine


def test_normalize_resample():
    engine = AudioEngine()
    audio = np.ones((44100, 2), dtype=np.float32)
    resampled = engine.normalize_audio(audio, src_sr=44100, target_sr=48000, target_channels=2)
    assert resampled.shape[0] == 48000
    assert resampled.shape[1] == 2


def test_normalize_mono_to_stereo():
    engine = AudioEngine()
    audio = np.ones((1000, 1), dtype=np.float32)
    stereo = engine.normalize_audio(audio, src_sr=44100, target_sr=44100, target_channels=2)
    assert stereo.shape[1] == 2


def test_normalize_stereo_to_mono():
    engine = AudioEngine()
    audio = np.ones((1000, 2), dtype=np.float32)
    mono = engine.normalize_audio(audio, src_sr=44100, target_sr=44100, target_channels=1)
    assert mono.shape[1] == 1


def test_list_input_devices_returns_list():
    engine = AudioEngine()
    devices = engine.list_input_devices()
    assert isinstance(devices, list)
