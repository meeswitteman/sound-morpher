from __future__ import annotations

import numpy as np
import librosa

from plugins.base import MorphPlugin, PluginParam, match_lengths

_FFT_CHOICES = ["256", "512", "1024", "2048"]


class GriffinLimPlugin(MorphPlugin):
    """Magnitude-only STFT morph reconstructed via the Griffin-Lim algorithm.

    Unlike Spectral FFT (which interpolates phase too), this plugin discards
    phase entirely and lets Griffin-Lim re-estimate it from the blended
    magnitude spectrum. Result: a smooth but distinctly robotic / sci-fi timbre.
    """

    name = "Griffin-Lim"
    description = (
        "Interpolates the magnitude spectrum of A and B, then reconstructs audio "
        "using the Griffin-Lim phase-estimation algorithm. Produces a smooth, "
        "robotic morph — best for experimental and sci-fi textures."
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
            name="n_iter",
            label="GL iterations",
            type="int",
            default=32,
            min_val=8,
            max_val=128,
            tooltip=(
                "Griffin-Lim iterations. More = cleaner audio, slower. "
                "32 is a good balance."
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
        n_iter: int = 32,
        **_: object,
    ) -> list[np.ndarray]:
        n_fft = int(fft_size)
        hop = n_fft // 4       # 75 % overlap — optimal for Griffin-Lim

        a, b = match_lengths(audio_a, audio_b)
        channels = a.shape[1] if a.ndim == 2 else 1

        result: list[np.ndarray] = []
        for i in range(steps):
            t = i / (steps - 1) if steps > 1 else 0.0
            result.append(_griffin_lim_mix(a, b, t, n_fft, hop, n_iter, channels))
            if progress_cb:
                progress_cb(i + 1)

        return result


def _griffin_lim_mix(
    a: np.ndarray,
    b: np.ndarray,
    t: float,
    n_fft: int,
    hop: int,
    n_iter: int,
    channels: int,
) -> np.ndarray:
    out_channels: list[np.ndarray] = []
    target_len = a.shape[0]

    for ch in range(channels):
        sig_a = (a[:, ch] if channels > 1 else a.ravel()).astype(np.float32)
        sig_b = (b[:, ch] if channels > 1 else b.ravel()).astype(np.float32)

        # STFT magnitude only — phase is discarded
        mag_a = np.abs(librosa.stft(sig_a, n_fft=n_fft, hop_length=hop))
        mag_b = np.abs(librosa.stft(sig_b, n_fft=n_fft, hop_length=hop))

        # Align frame counts (rounding differences for same-length input)
        n_frames = min(mag_a.shape[1], mag_b.shape[1])
        mag_a = mag_a[:, :n_frames]
        mag_b = mag_b[:, :n_frames]

        # Interpolate magnitude spectra
        mag_mix = (1.0 - t) * mag_a + t * mag_b

        # Griffin-Lim phase reconstruction
        ch_out = librosa.griffinlim(
            mag_mix,
            n_iter=n_iter,
            hop_length=hop,
            n_fft=n_fft,
        )

        # Trim or pad to match original length
        if len(ch_out) >= target_len:
            ch_out = ch_out[:target_len]
        else:
            ch_out = np.pad(ch_out, (0, target_len - len(ch_out)))

        # Griffin-Lim can produce peaks above 1.0; normalise to original level
        peak = np.max(np.abs(ch_out))
        ref  = max(np.max(np.abs(sig_a)), np.max(np.abs(sig_b)), 1e-8)
        if peak > ref:
            ch_out *= ref / peak

        out_channels.append(ch_out.astype(np.float32))

    if channels == 1:
        return out_channels[0].reshape(-1, 1)
    return np.stack(out_channels, axis=1)
