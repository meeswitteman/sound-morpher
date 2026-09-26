from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf


def _remove_dc_offset(audio: np.ndarray, sample_rate: int, cutoff_hz: float = 5.0) -> np.ndarray:
    """Strip DC bias with a zero-phase high-pass at `cutoff_hz`.

    Recorded input especially tends to carry a DC offset. Left in, it eats into
    the export limiter's headroom and can click at step boundaries where two
    different DC levels butt up against each other across a morph.

    Removing DC does not need a cutoff anywhere near the audible range. The
    previous 20 Hz first-order filter took 3 dB at 20 Hz, 1.6 dB at 30 Hz and
    still 1 dB at 40 Hz: the fundamental of kicks, 808s and low bass notes,
    exactly the material a sample pack is full of. It was also causal, so it
    shifted the phase of everything below ~100 Hz and changed the shape of
    every kick transient. Run forwards and backwards at 5 Hz instead, it is
    zero-phase and down only 0.5 dB at 20 Hz and 0.2 dB at 30 Hz, and it still
    has no gain at all at DC. Starting from the steady state of the first
    sample keeps an offset from turning into a thump at the start.
    """
    import scipy.signal as sig

    arr = np.asarray(audio, dtype=np.float64)
    b, a = sig.butter(1, cutoff_hz, btype="highpass", fs=sample_rate)
    if arr.shape[0] <= 3 * max(len(a), len(b)):
        # Too short for the filter's edge handling; the mean is the DC.
        return (arr - arr.mean(axis=0, keepdims=True)).astype(np.float32)
    return sig.filtfilt(b, a, arr, axis=0).astype(np.float32)


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


_OUT_CHANNELS = 2


class VoiceMixer:
    """Sums any number of sounds ("voices") into one output, block by block.

    Kept free of any audio device so it can be tested directly; the stream
    callback in AudioEngine only asks it for the next block. Each voice plays
    to its own end and then drops out.
    """

    def __init__(self, channels: int = _OUT_CHANNELS) -> None:
        self._channels = channels
        self._voices: list[list] = []      # [audio (n, channels), position]
        self._lock = threading.Lock()

    def add(self, audio: np.ndarray) -> None:
        arr = np.asarray(audio, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        if arr.shape[1] == 1 and self._channels > 1:
            arr = np.repeat(arr, self._channels, axis=1)
        elif arr.shape[1] > self._channels:
            arr = arr[:, : self._channels]
        if arr.shape[0] == 0:
            return
        with self._lock:
            self._voices.append([np.ascontiguousarray(arr), 0])

    def clear(self) -> None:
        with self._lock:
            self._voices.clear()

    @property
    def active(self) -> bool:
        with self._lock:
            return bool(self._voices)

    @property
    def voice_count(self) -> int:
        with self._lock:
            return len(self._voices)

    def render(self, frames: int) -> np.ndarray:
        out = np.zeros((frames, self._channels), dtype=np.float32)
        with self._lock:
            still_playing = []
            for voice in self._voices:
                audio, pos = voice
                chunk = audio[pos : pos + frames]
                out[: len(chunk)] += chunk
                voice[1] = pos + len(chunk)
                if voice[1] < len(audio):
                    still_playing.append(voice)
            self._voices = still_playing
        # A tail under the next attack can add up past full scale; clip
        # rather than let the D/A wrap or distort harder.
        np.clip(out, -1.0, 1.0, out=out)
        return out


class AudioEngine:
    """Load and play WAV audio. Thread-safe playback control."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._mixer = VoiceMixer(_OUT_CHANNELS)
        self._stream = None
        self._stream_sr = 0
        # Gain applied by the last load_wav() to bring it under full scale, in
        # dB (0.0 or negative).
        self.last_load_trim_db = 0.0

    def load_wav(self, path: str | Path) -> tuple[np.ndarray, int]:
        """Read a WAV file and return (samples float32, sample_rate).

        Always returns a 2-D array of shape (frames, channels).

        Float WAVs can legitimately peak above full scale; a DAW bounce with
        hot peaks is a normal source. Clipping those peaks, as this used to,
        squared off every overshoot into broadband distortion. The whole file
        is scaled down to a peak of exactly 1.0 instead, which keeps the
        waveform intact; `last_load_trim_db` reports how much, 0.0 when
        nothing needed scaling.
        """
        data, sr = sf.read(str(path), dtype="float32", always_2d=True)
        peak = float(np.max(np.abs(data))) if data.size else 0.0
        self.last_load_trim_db = 0.0
        if peak > 1.0:
            data /= peak
            # Division can land a hair over 1.0 in float32.
            np.clip(data, -1.0, 1.0, out=data)
            self.last_load_trim_db = -20.0 * np.log10(peak)
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
        self._ensure_stream(sample_rate)
        self._mixer.clear()
        self._mixer.add(audio)

    def play_overlapping(self, audio: np.ndarray, sample_rate: int) -> None:
        """Start audio on top of whatever is playing, letting that ring out.

        Play All uses this: each step starts on the beat while the previous
        one's tail keeps sounding underneath it, as on a sampler, instead of
        being cut off at the step boundary.
        """
        self._ensure_stream(sample_rate)
        self._mixer.add(audio)

    def stop(self) -> None:
        """Silence everything immediately."""
        self._mixer.clear()

    def is_playing(self) -> bool:
        return self._mixer.active

    def close(self) -> None:
        """Release the audio device."""
        self._mixer.clear()
        with self._lock:
            stream, self._stream = self._stream, None
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception:
                pass

    def _ensure_stream(self, sample_rate: int) -> None:
        """Open the shared output stream, or reopen it at a new sample rate.

        One stream stays open for the whole session and every sound is mixed
        into it. sd.play() opened a new stream per call and could only play
        one sound at a time, which is what cut each Play All step's tail off.
        """
        with self._lock:
            if self._stream is not None and self._stream_sr == sample_rate:
                return
            old, self._stream = self._stream, None
        if old is not None:
            try:
                old.stop()
                old.close()
            except Exception:
                pass
        self._mixer.clear()
        stream = sd.OutputStream(
            samplerate=sample_rate,
            channels=_OUT_CHANNELS,
            dtype="float32",
            callback=self._callback,
        )
        stream.start()
        with self._lock:
            self._stream = stream
            self._stream_sr = sample_rate

    def _callback(self, outdata, frames, _time, _status) -> None:
        outdata[:] = self._mixer.render(frames)

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
