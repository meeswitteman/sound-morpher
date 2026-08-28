from __future__ import annotations

import numpy as np
import librosa

from plugins.base import (
    MorphPlugin,
    PluginParam,
    interp_magnitude,
    interp_phase,
    match_lengths,
)

_FFT_CHOICES = ["256", "512", "1024", "2048", "4096"]


class GriffinLimPlugin(MorphPlugin):
    """Magnitude-only STFT morph reconstructed via the Griffin-Lim algorithm.

    Unlike Spectral FFT (which interpolates phase too), this plugin re-estimates
    phase from the blended magnitude spectrum. Result: a smooth but distinctly
    robotic / sci-fi timbre.
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
        PluginParam(
            name="magnitude",
            label="Magnitude",
            type="choice",
            default="log",
            choices=["log", "linear"],
            tooltip=(
                "log: geometric blend — partials of A fade out as B's fade in "
                "(true morph).  "
                "linear: arithmetic blend — you hear both spectra at once."
            ),
        ),
        PluginParam(
            name="seed_phase",
            label="Seed phase",
            type="bool",
            default=True,
            tooltip=(
                "Start Griffin-Lim from the interpolated phase of A and B instead "
                "of noise. Converges faster and keeps transients much crisper; "
                "turn off for the classic smeared, fully synthetic character."
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
        magnitude: str = "log",
        seed_phase: bool = True,
        **_: object,
    ) -> list[np.ndarray]:
        n_fft = int(fft_size)
        hop = n_fft // 4       # 75 % overlap — optimal for Griffin-Lim

        a, b = match_lengths(audio_a, audio_b)
        channels = a.shape[1] if a.ndim == 2 else 1

        result: list[np.ndarray] = []
        for i in range(steps):
            t = i / (steps - 1) if steps > 1 else 0.0
            # Griffin-Lim invents phase, so its "100 % A" step is not A at all
            # (measured: 0.96 absolute error on a 0.5-amplitude sine). Pass the
            # endpoints through unchanged.
            if t <= 0.0:
                result.append(a.astype(np.float32))
            elif t >= 1.0:
                result.append(b.astype(np.float32))
            else:
                result.append(
                    _griffin_lim_mix(
                        a, b, t, n_fft, hop, n_iter, channels, magnitude, seed_phase
                    )
                )
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
    magnitude: str,
    seed_phase: bool,
) -> np.ndarray:
    out_channels: list[np.ndarray] = []
    target_len = a.shape[0]

    for ch in range(channels):
        sig_a = (a[:, ch] if channels > 1 else a.ravel()).astype(np.float32)
        sig_b = (b[:, ch] if channels > 1 else b.ravel()).astype(np.float32)

        Za = librosa.stft(sig_a, n_fft=n_fft, hop_length=hop)
        Zb = librosa.stft(sig_b, n_fft=n_fft, hop_length=hop)

        # Align frame counts (rounding differences for same-length input)
        n_frames = min(Za.shape[1], Zb.shape[1])
        Za, Zb = Za[:, :n_frames], Zb[:, :n_frames]

        mag_mix = interp_magnitude(np.abs(Za), np.abs(Zb), t, mode=magnitude)

        if seed_phase:
            init = interp_phase(np.angle(Za), np.angle(Zb), t)
            ch_out = _griffinlim_seeded(mag_mix, init, n_iter, n_fft, hop, target_len)
        else:
            ch_out = librosa.griffinlim(
                mag_mix, n_iter=n_iter, hop_length=hop, n_fft=n_fft
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


def _griffinlim_seeded(
    magnitude: np.ndarray,
    init_phase: np.ndarray,
    n_iter: int,
    n_fft: int,
    hop: int,
    length: int,
) -> np.ndarray:
    """Fast Griffin-Lim starting from a supplied phase estimate.

    librosa.griffinlim only offers random or zero initialisation; seeding with
    the sources' own interpolated phase starts far closer to a consistent STFT,
    so the usual metallic smearing largely disappears.
    """
    momentum = 0.99
    stft_mix = magnitude * np.exp(1j * init_phase)
    previous = np.zeros_like(stft_mix)

    for _ in range(max(1, n_iter)):
        inverse = librosa.istft(stft_mix, hop_length=hop, n_fft=n_fft, length=length)
        rebuilt = librosa.stft(inverse, n_fft=n_fft, hop_length=hop)
        rebuilt = rebuilt[:, : stft_mix.shape[1]]
        if rebuilt.shape[1] < stft_mix.shape[1]:
            rebuilt = np.pad(
                rebuilt, ((0, 0), (0, stft_mix.shape[1] - rebuilt.shape[1]))
            )

        # Momentum term (Perraudin et al., "A fast Griffin-Lim algorithm")
        estimate = rebuilt - (momentum / (1.0 + momentum)) * previous
        previous = rebuilt
        stft_mix = magnitude * np.exp(1j * np.angle(estimate))

    return librosa.istft(stft_mix, hop_length=hop, n_fft=n_fft, length=length)
