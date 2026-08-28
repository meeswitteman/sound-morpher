from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf


def _remove_dc_offset(audio: np.ndarray, sample_rate: int, cutoff_hz: float = 20.0) -> np.ndarray:
    """High-pass at `cutoff_hz` to strip DC bias and sub-audio rumble.

    Recorded input especially tends to carry a DC offset. Left in, it eats into
    the export limiter's headroom and can click at step boundaries where two
    different DC levels butt up against each other across a morph.
    """
    import scipy.signal as sig

    b, a = sig.butter(1, cutoff_hz, btype="highpass", fs=sample_rate)
    return sig.lfilter(b, a, audio, axis=0).astype(np.float32)


def _resample(audio: np.ndarray, src_sr: int, target_sr: int) -> np.ndarray:
    """Sample-rate conversion, best available quality.

    scipy.signal.resample — the previous choice — works by zeroing out FFT bins,
    which assumes the signal is periodic over its whole length. Samples are not:
    the discontinuity between the last sample and the first rings out as a
    pre-echo at the head and a smear at the tail. soxr is a proper polyphase
    resampler with no such assumption, and it is already a librosa dependency.
    """
    try:
        import soxr
        return soxr.resample(audio, src_sr, target_sr, quality="VHQ").astype(np.float32)
    except ImportError:
        pass

    # Rational polyphase: still no periodicity assumption, just a coarser filter.
    from math import gcd
    import scipy.signal as sig

    divisor = gcd(int(src_sr), int(target_sr))
    return sig.resample_poly(
        audio, int(target_sr) // divisor, int(src_sr) // divisor, axis=0
    ).astype(np.float32)


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
        audio = _remove_dc_offset(audio, src_sr)
        if src_sr != target_sr:
            audio = _resample(audio, src_sr, target_sr)

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
