from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf


class AudioEngine:
    """Load and play WAV audio. Thread-safe playback control."""

    def __init__(self) -> None:
        self._lock = threading.Lock()

    def load_wav(self, path: str | Path) -> tuple[np.ndarray, int]:
        """Read a WAV file and return (samples float32, sample_rate).

        Always returns a 2-D array of shape (frames, channels).
        Float-format WAVs are clipped to [-1, 1] after reading.
        """
        data, sr = sf.read(str(path), dtype="float32", always_2d=True)
        # Float WAVs may exceed [-1, 1] — clip silently
        if data.dtype == np.float32:
            np.clip(data, -1.0, 1.0, out=data)
        return data, sr

    def get_wav_info(self, path: str | Path) -> dict:
        """Return metadata for a WAV file without reading the full audio data."""
        info = sf.info(str(path))
        return {
            "samplerate": info.samplerate,
            "channels": info.channels,
            "frames": info.frames,
            "subtype": info.subtype,
            "duration": info.duration,
        }

    def normalize_audio(
        self,
        audio: np.ndarray,
        src_sr: int,
        target_sr: int,
        target_channels: int = 2,
    ) -> np.ndarray:
        """Resample and adjust channel count to match project settings."""
        import scipy.signal as sig

        # Resample if needed
        if src_sr != target_sr:
            num_samples = int(len(audio) * target_sr / src_sr)
            audio = sig.resample(audio, num_samples, axis=0)

        # Channel conversion
        if audio.shape[1] == 1 and target_channels == 2:
            audio = np.repeat(audio, 2, axis=1)
        elif audio.shape[1] == 2 and target_channels == 1:
            audio = audio.mean(axis=1, keepdims=True)

        return audio

    def play(self, audio: np.ndarray, sample_rate: int) -> None:
        """Play audio non-blocking. Stops any current playback first."""
        self.stop()
        sd.play(audio, samplerate=sample_rate)

    def stop(self) -> None:
        sd.stop()

    def is_playing(self) -> bool:
        try:
            return bool(sd.get_stream().active)
        except RuntimeError:
            return False

    def list_input_devices(self) -> list[dict]:
        """Return available input devices as list of {index, name} dicts."""
        devices = sd.query_devices()
        return [
            {"index": i, "name": d["name"]}
            for i, d in enumerate(devices)
            if d["max_input_channels"] > 0
        ]

    def default_input_device(self) -> int:
        return int(sd.default.device[0])
