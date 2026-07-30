from __future__ import annotations

import numpy as np
from scipy.signal import stft, istft

from plugins.base import (
    MorphPlugin,
    PluginParam,
    interp_magnitude,
    interp_phase,
    match_lengths,
)

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
        PluginParam(
            name="magnitude",
            label="Magnitude",
            type="choice",
            default="log",
            choices=["log", "linear"],
            tooltip=(
                "log: geometric blend — partials of A fade out as B's fade in "
                "(true morph).  "
                "linear: arithmetic blend — you hear both spectra at once "
                "(spectral crossfade)."
            ),
        ),
        PluginParam(
            name="phase",
            label="Phase",
            type="choice",
            default="shortest-arc",
            choices=["shortest-arc", "dominant", "linear"],
            tooltip=(
                "shortest-arc: rotate A's phase toward B the short way (smooth, "
                "no cancellation).  "
                "dominant: take phase from the louder-weighted source.  "
                "linear: naive average — causes phasey cancellation, kept for "
                "comparison."
            ),
        ),
    ]

    def morph(
        self,
        audio_a: np.ndarray,
        audio_b: np.ndarray,
        steps: int,
        sample_rate: int,
        progress_cb=None,
        fft_size: str = "1024",
        overlap: int = 75,
        magnitude: str = "log",
        phase: str = "shortest-arc",
        **_: object,
    ) -> list[np.ndarray]:
        n_fft = int(fft_size)
        hop = max(1, int(n_fft * (1 - overlap / 100)))
        a, b = match_lengths(audio_a, audio_b)
        channels = a.shape[1] if a.ndim == 2 else 1
        result: list[np.ndarray] = []

        for i in range(steps):
            t = i / (steps - 1) if steps > 1 else 0.0
            # Endpoints pass through untouched — an analysis/synthesis round trip
            # can only lose fidelity there.
            if t <= 0.0:
                result.append(a.astype(np.float32))
            elif t >= 1.0:
                result.append(b.astype(np.float32))
            else:
                result.append(
                    _spectral_mix(
                        a, b, t, sample_rate, n_fft, hop, channels, magnitude, phase
                    )
                )
            if progress_cb:
                progress_cb(i + 1)

        return result


def _spectral_mix(
    a: np.ndarray,
    b: np.ndarray,
    t: float,
    sample_rate: int,
    n_fft: int,
    hop: int,
    channels: int,
    magnitude: str,
    phase: str,
) -> np.ndarray:
    out_channels: list[np.ndarray] = []
    for ch in range(channels):
        sig_a = a[:, ch] if channels > 1 else a.ravel()
        sig_b = b[:, ch] if channels > 1 else b.ravel()

        _, _, Za = stft(sig_a, fs=sample_rate, nperseg=n_fft, noverlap=n_fft - hop)
        _, _, Zb = stft(sig_b, fs=sample_rate, nperseg=n_fft, noverlap=n_fft - hop)

        mag = interp_magnitude(np.abs(Za), np.abs(Zb), t, mode=magnitude)
        ang = interp_phase(np.angle(Za), np.angle(Zb), t, mode=phase)
        Z_mix = mag * np.exp(1j * ang)

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
