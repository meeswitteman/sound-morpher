from __future__ import annotations

import numpy as np
from scipy.signal import stft, istft

from plugins.base import MorphPlugin, PluginParam, match_lengths

_FFT_CHOICES = ["256", "512", "1024", "2048"]


class SpectralFftPlugin(MorphPlugin):
    """Spectral morph: interpolate STFT magnitude and phase between A and B."""

    name = "Spectral FFT"
    description = (
        "Interpolates the magnitude spectrum (STFT) of A into B, "
        "then reconstructs audio via inverse STFT."
    )
    parameters = [
        PluginParam(
            name="fft_size",
            label="FFT Size",
            type="choice",
            default="1024",
            choices=_FFT_CHOICES,
            tooltip="Larger FFT = better frequency resolution, slower computation.",
        ),
        PluginParam(
            name="overlap",
            label="Overlap %",
            type="int",
            default=75,
            min_val=50,
            max_val=87,
            tooltip="STFT frame overlap percentage (50–87). Higher = smoother.",
        ),
    ]

    def morph(
        self,
        audio_a: np.ndarray,
        audio_b: np.ndarray,
        steps: int,
        sample_rate: int,
        fft_size: str = "1024",
        overlap: int = 75,
        **_: object,
    ) -> list[np.ndarray]:
        n_fft = int(fft_size)
        hop = max(1, int(n_fft * (1 - overlap / 100)))
        a, b = match_lengths(audio_a, audio_b)
        channels = a.shape[1] if a.ndim == 2 else 1
        result: list[np.ndarray] = []

        for i in range(steps):
            t = i / (steps - 1) if steps > 1 else 0.0
            mixed = _spectral_mix(a, b, t, sample_rate, n_fft, hop, channels)
            result.append(mixed)

        return result


def _spectral_mix(
    a: np.ndarray,
    b: np.ndarray,
    t: float,
    sample_rate: int,
    n_fft: int,
    hop: int,
    channels: int,
) -> np.ndarray:
    out_channels: list[np.ndarray] = []
    for ch in range(channels):
        sig_a = a[:, ch] if channels > 1 else a.ravel()
        sig_b = b[:, ch] if channels > 1 else b.ravel()

        _, _, Za = stft(sig_a, fs=sample_rate, nperseg=n_fft, noverlap=n_fft - hop)
        _, _, Zb = stft(sig_b, fs=sample_rate, nperseg=n_fft, noverlap=n_fft - hop)

        mag_a, phase_a = np.abs(Za), np.angle(Za)
        mag_b, phase_b = np.abs(Zb), np.angle(Zb)

        mag   = (1 - t) * mag_a   + t * mag_b
        phase = (1 - t) * phase_a + t * phase_b
        Z_mix = mag * np.exp(1j * phase)

        _, ch_out = istft(Z_mix, fs=sample_rate, nperseg=n_fft, noverlap=n_fft - hop)
        # Match length to input
        target_len = len(sig_a)
        if len(ch_out) >= target_len:
            ch_out = ch_out[:target_len]
        else:
            ch_out = np.pad(ch_out, (0, target_len - len(ch_out)))

        out_channels.append(ch_out.astype(np.float32))

    if channels == 1:
        return out_channels[0].reshape(-1, 1)
    return np.stack(out_channels, axis=1)
