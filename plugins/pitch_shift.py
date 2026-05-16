from __future__ import annotations

import numpy as np

from plugins.base import MorphPlugin, PluginParam, match_lengths


class PitchShiftPlugin(MorphPlugin):
    """Pitch-shift morph: detect fundamental of each sound and interpolate pitch."""

    name = "Pitch Shift"
    description = (
        "Detects the fundamental frequency of A and B, then pitch-shifts each "
        "step toward B's pitch while crossfading amplitude."
    )
    parameters = [
        PluginParam(
            name="fmin",
            label="F-min (Hz)",
            type="float",
            default=60.0,
            min_val=20.0,
            max_val=500.0,
            tooltip="Minimum fundamental frequency for pitch detection.",
        ),
        PluginParam(
            name="fmax",
            label="F-max (Hz)",
            type="float",
            default=4000.0,
            min_val=500.0,
            max_val=20000.0,
            tooltip="Maximum fundamental frequency for pitch detection.",
        ),
    ]

    def morph(
        self,
        audio_a: np.ndarray,
        audio_b: np.ndarray,
        steps: int,
        sample_rate: int,
        progress_cb=None,
        fmin: float = 60.0,
        fmax: float = 4000.0,
        **_: object,
    ) -> list[np.ndarray]:
        import librosa

        a, b = match_lengths(audio_a, audio_b)
        channels = a.shape[1] if a.ndim == 2 else 1

        f0_a = _detect_f0(a, sample_rate, fmin, fmax)
        f0_b = _detect_f0(b, sample_rate, fmin, fmax)

        result: list[np.ndarray] = []
        for i in range(steps):
            t = i / (steps - 1) if steps > 1 else 0.0
            target_f0 = (1 - t) * f0_a + t * f0_b

            if f0_a > 0 and target_f0 > 0:
                semitones = 12 * np.log2(target_f0 / f0_a) if f0_a > 0 else 0.0
                shifted = _shift_channels(a, sample_rate, semitones, channels)
            else:
                shifted = a.copy()

            result.append(((1 - t) * shifted + t * b).astype(np.float32))
            if progress_cb:
                progress_cb(i + 1)

        return result


def _detect_f0(audio: np.ndarray, sr: int, fmin: float, fmax: float) -> float:
    """Return median detected fundamental frequency (0.0 if unpitched)."""
    import librosa

    channels = audio.shape[1] if audio.ndim == 2 else 1
    mono = audio[:, 0] if channels > 1 else audio.ravel()
    f0 = librosa.yin(mono.astype(np.float32), fmin=fmin, fmax=fmax, sr=sr)
    valid = f0[(f0 > fmin) & (f0 < fmax)]
    return float(np.median(valid)) if len(valid) else 0.0


def _shift_channels(
    audio: np.ndarray,
    sr: int,
    semitones: float,
    channels: int,
) -> np.ndarray:
    import librosa

    if abs(semitones) < 0.01:
        return audio.copy()

    out_chs: list[np.ndarray] = []
    for ch in range(channels):
        sig = audio[:, ch] if channels > 1 else audio.ravel()
        shifted = librosa.effects.pitch_shift(
            sig.astype(np.float32), sr=sr, n_steps=semitones
        )
        out_chs.append(shifted.astype(np.float32))

    if channels == 1:
        return out_chs[0].reshape(-1, 1)
    return np.stack(out_chs, axis=1)
