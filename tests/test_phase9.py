"""Phase 9 tests: error handling, performance, and packaging."""
import io
import time
import tempfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf


# ── AudioEngine hardening ─────────────────────────────────────────────────────

def test_get_wav_info(tmp_path):
    """get_wav_info returns correct metadata without reading full audio."""
    from app.audio_engine import AudioEngine

    wav = tmp_path / "test.wav"
    data = np.zeros((4410, 2), dtype=np.float32)
    sf.write(str(wav), data, 44100, subtype="PCM_16")

    engine = AudioEngine()
    info = engine.get_wav_info(str(wav))

    assert info["samplerate"] == 44100
    assert info["channels"] == 2
    assert info["frames"] == 4410
    assert "PCM_16" in info["subtype"]


def test_float_wav_loads_and_clips(tmp_path):
    """Float32 WAVs with out-of-range values are clipped to [-1, 1]."""
    from app.audio_engine import AudioEngine

    wav = tmp_path / "float.wav"
    # Write a float WAV with values > 1
    data = np.array([[1.5, -1.8], [0.5, 0.3]], dtype=np.float32)
    sf.write(str(wav), data, 44100, subtype="FLOAT")

    engine = AudioEngine()
    audio, sr = engine.load_wav(str(wav))

    assert audio.max() <= 1.0
    assert audio.min() >= -1.0


def test_float_wav_subtype_detected(tmp_path):
    """get_wav_info correctly identifies float subtype."""
    from app.audio_engine import AudioEngine

    wav = tmp_path / "float.wav"
    sf.write(str(wav), np.zeros((100, 1), dtype=np.float32), 44100, subtype="FLOAT")

    engine = AudioEngine()
    info = engine.get_wav_info(str(wav))
    assert "FLOAT" in info["subtype"].upper()


def test_load_wav_resampling_via_normalize(tmp_path):
    """Loading a 48kHz file and normalizing to 44100 produces correct length."""
    from app.audio_engine import AudioEngine

    wav = tmp_path / "48k.wav"
    n = 48000  # 1 second at 48 kHz
    sf.write(str(wav), np.zeros((n, 1), dtype=np.float32), 48000, subtype="PCM_16")

    engine = AudioEngine()
    audio, sr = engine.load_wav(str(wav))
    normalized = engine.normalize_audio(audio, src_sr=sr, target_sr=44100)

    assert sr == 48000
    assert abs(len(normalized) - 44100) <= 10  # within 10 samples of 1s at 44100 Hz


# ── Duration mismatch (match_lengths) ────────────────────────────────────────

def test_crossfade_with_mismatched_lengths():
    """Crossfade must succeed when A and B have different lengths."""
    from plugins.crossfade import CrossfadePlugin

    a = np.zeros((8000, 2), dtype=np.float32)
    b = np.ones((12000, 2), dtype=np.float32)
    plugin = CrossfadePlugin()
    steps = plugin.morph(a, b, steps=4, sample_rate=44100)
    assert len(steps) == 4
    for s in steps:
        assert s.shape == (12000, 2)  # padded to longer


def test_spectral_fft_with_mismatched_lengths():
    from plugins.spectral_fft import SpectralFftPlugin

    a = np.zeros((4000, 2), dtype=np.float32)
    b = np.full((6000, 2), 0.5, dtype=np.float32)
    plugin = SpectralFftPlugin()
    steps = plugin.morph(a, b, steps=3, sample_rate=44100)
    assert len(steps) == 3
    for s in steps:
        assert s.shape[0] == 6000


# ── Performance ───────────────────────────────────────────────────────────────
# 8 steps × 5s stereo at 44100 Hz — must complete in < 10 s each

FRAMES_5S = 44100 * 5
STEREO = 2


def _perf_audio():
    rng = np.random.default_rng(42)
    a = rng.uniform(-0.5, 0.5, (FRAMES_5S, STEREO)).astype(np.float32)
    b = rng.uniform(-0.5, 0.5, (FRAMES_5S, STEREO)).astype(np.float32)
    return a, b


def test_crossfade_performance():
    from plugins.crossfade import CrossfadePlugin

    a, b = _perf_audio()
    plugin = CrossfadePlugin()
    t0 = time.perf_counter()
    steps = plugin.morph(a, b, steps=8, sample_rate=44100)
    elapsed = time.perf_counter() - t0

    assert len(steps) == 8
    assert elapsed < 10.0, f"Crossfade took {elapsed:.2f}s (limit: 10s)"


def test_spectral_fft_performance():
    from plugins.spectral_fft import SpectralFftPlugin

    a, b = _perf_audio()
    plugin = SpectralFftPlugin()
    t0 = time.perf_counter()
    steps = plugin.morph(a, b, steps=8, sample_rate=44100)
    elapsed = time.perf_counter() - t0

    assert len(steps) == 8
    assert elapsed < 10.0, f"Spectral FFT took {elapsed:.2f}s (limit: 10s)"


def test_granular_performance():
    from plugins.granular import GranularPlugin

    a, b = _perf_audio()
    plugin = GranularPlugin()
    t0 = time.perf_counter()
    steps = plugin.morph(a, b, steps=8, sample_rate=44100)
    elapsed = time.perf_counter() - t0

    assert len(steps) == 8
    assert elapsed < 10.0, f"Granular took {elapsed:.2f}s (limit: 10s)"


# Pitch shift uses librosa and is slower; allow up to 30s
def test_pitch_shift_performance():
    from plugins.pitch_shift import PitchShiftPlugin

    a, b = _perf_audio()
    plugin = PitchShiftPlugin()
    t0 = time.perf_counter()
    steps = plugin.morph(a, b, steps=8, sample_rate=44100)
    elapsed = time.perf_counter() - t0

    assert len(steps) == 8
    assert elapsed < 30.0, f"Pitch Shift took {elapsed:.2f}s (limit: 30s)"


# ── Project file robustness ───────────────────────────────────────────────────

def test_load_project_with_missing_optional_fields(tmp_path):
    """Older .smorph without algorithm_params loads without error."""
    import zipfile, json
    from app.project_file import ProjectFile

    meta = {
        "version": "1.0",
        "sample_rate": 44100,
        "bit_depth": 16,
        "name_a": "a.wav",
        "name_b": "b.wav",
        "steps": 4,
        "algorithm": "Crossfade",
        # algorithm_params intentionally omitted (old format)
        "bpm": 120,
        "beats_per_step": 4,
        "loop": False,
        "step_count": 0,
    }
    out = tmp_path / "legacy.smorph"
    with zipfile.ZipFile(out, "w") as zf:
        zf.writestr("project.json", json.dumps(meta))

    state = ProjectFile.load(str(out))
    assert state.algorithm == "Crossfade"
    assert state.algorithm_params == {}


# ── Packaging artefacts ───────────────────────────────────────────────────────

def test_pyproject_toml_exists():
    root = Path(__file__).parent.parent
    assert (root / "pyproject.toml").exists()


def test_pyinstaller_spec_exists():
    root = Path(__file__).parent.parent
    assert (root / "soundmorpher.spec").exists()


def test_app_icon_exists():
    root = Path(__file__).parent.parent
    assert (root / "resources" / "icons" / "app.svg").exists()
