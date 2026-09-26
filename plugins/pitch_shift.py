from __future__ import annotations

import numpy as np

from plugins.base import (
    MorphPlugin,
    PluginParam,
    match_lengths,
    apply_formant_correction,
    formant_envelope,
    pitch_shift_varying,
)

_HOP = 512


class PitchShiftPlugin(MorphPlugin):
    """Pitch-shift morph: move both sounds onto a common interpolated pitch."""

    name = "Pitch Shift"
    description = (
        "Detects the pitch of A and B, then moves BOTH toward an interpolated "
        "target pitch before crossfading. Intermediate steps therefore sound as "
        "one pitch rather than two."
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
        PluginParam(
            name="tracking",
            label="Tracking",
            type="choice",
            default="median",
            choices=["median", "dynamic"],
            tooltip=(
                "median: one representative pitch per sound. Robust, fast, and "
                "the right choice for single-note samples.  "
                "dynamic: follow each sound's pitch contour over time, so a "
                "melody or vibrato morphs into the other's. Much slower, and it "
                "warbles on unpitched material where detection is unreliable."
            ),
        ),
        PluginParam(
            name="formants",
            label="Formants",
            type="choice",
            default="preserve",
            choices=["preserve", "shift"],
            tooltip=(
                "preserve: each sound keeps its own resonances while its pitch "
                "moves, so a voice stays the same size and an instrument keeps "
                "its body.  "
                "shift: resonances move with the pitch, the classic "
                "resampling 'chipmunk' sound. Faster."
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
        fmin: float = 60.0,
        fmax: float = 4000.0,
        tracking: str = "median",
        formants: str = "preserve",
        **_: object,
    ) -> list[np.ndarray]:
        a, b = match_lengths(audio_a, audio_b)
        channels = a.shape[1] if a.ndim == 2 else 1

        if tracking == "dynamic":
            track_a = _f0_track(_mono(a), sample_rate, fmin, fmax)
            track_b = _f0_track(_mono(b), sample_rate, fmin, fmax)
            usable = track_a is not None and track_b is not None
            if usable:
                # 95th percentile: a single octave glitch in the track should
                # not drag the lifter cutoff down for the whole file.
                top_a = float(np.percentile(track_a, 95))
                top_b = float(np.percentile(track_b, 95))
        else:
            f0_a = _detect_f0(a, sample_rate, fmin, fmax)
            f0_b = _detect_f0(b, sample_rate, fmin, fmax)
            usable = f0_a > 0.0 and f0_b > 0.0
            top_a, top_b = f0_a, f0_b

        # Each shifted sound gets its own source envelope back. The envelopes
        # do not depend on the step, so they are analysed once here, each with
        # a lifter set by that source's own pitch: the shifted envelope is
        # derived from it, never estimated from the shifted signal.
        env_a = env_b = None
        if usable and formants == "preserve":
            env_a = _envelopes(a, sample_rate, top_a)
            env_b = _envelopes(b, sample_rate, top_b)

        result: list[np.ndarray] = []
        for i in range(steps):
            t = i / (steps - 1) if steps > 1 else 0.0

            if t <= 0.0:
                result.append(a.astype(np.float32))
            elif t >= 1.0:
                result.append(b.astype(np.float32))
            elif not usable:
                # No pitch to work with (percussion, noise) — a crossfade is the
                # honest answer rather than shifting by a made-up interval.
                result.append(_crossfade(a, b, t))
            elif tracking == "dynamic":
                result.append(
                    _dynamic_step(
                        a, b, track_a, track_b, t, channels, len(a),
                        sample_rate, env_a, env_b,
                    )
                )
            else:
                result.append(
                    _median_step(
                        a, b, f0_a, f0_b, t, sample_rate, channels, env_a, env_b
                    )
                )

            if progress_cb:
                progress_cb(i + 1)

        return result


# ── Per-step ──────────────────────────────────────────────────────────────────

def _median_step(
    a: np.ndarray,
    b: np.ndarray,
    f0_a: float,
    f0_b: float,
    t: float,
    sr: int,
    channels: int,
    env_a: list | None = None,
    env_b: list | None = None,
) -> np.ndarray:
    # Target pitch interpolates in the log domain, where a fixed distance is a
    # fixed musical interval. Linear interpolation of Hz drifts flat.
    # ratio_a is then (f0_b/f0_a)^t, and ratio_b its mirror image.
    interval = np.log2(f0_b / f0_a)
    shifted_a = _shift_channels(a, sr, 12.0 * interval * t, channels)
    shifted_b = _shift_channels(b, sr, -12.0 * interval * (1.0 - t), channels)
    shifted_a = _restore_formants(shifted_a, env_a, 2.0 ** (interval * t), sr)
    shifted_b = _restore_formants(shifted_b, env_b, 2.0 ** (-interval * (1.0 - t)), sr)
    return _crossfade(shifted_a, shifted_b, t)


def _dynamic_step(
    a: np.ndarray,
    b: np.ndarray,
    track_a: np.ndarray,
    track_b: np.ndarray,
    t: float,
    channels: int,
    n_samples: int,
    sr: int = 44100,
    env_a: list | None = None,
    env_b: list | None = None,
) -> np.ndarray:
    interval = np.log(track_b) - np.log(track_a)      # per frame, log domain
    ratio_a = _to_samples(np.exp(interval * t), n_samples)
    ratio_b = _to_samples(np.exp(-interval * (1.0 - t)), n_samples)

    cols_a, cols_b = [], []
    for ch in range(channels):
        sig_a = a[:, ch] if channels > 1 else a.ravel()
        sig_b = b[:, ch] if channels > 1 else b.ravel()
        cols_a.append(pitch_shift_varying(sig_a, ratio_a, _HOP))
        cols_b.append(pitch_shift_varying(sig_b, ratio_b, _HOP))

    shifted_a = np.stack(cols_a, axis=1)
    shifted_b = np.stack(cols_b, axis=1)
    shifted_a = _restore_formants(shifted_a, env_a, ratio_a, sr)
    shifted_b = _restore_formants(shifted_b, env_b, ratio_b, sr)
    return _crossfade(shifted_a, shifted_b, t)


def _envelopes(audio: np.ndarray, sr: int, f0_high: float) -> list:
    """Source spectral envelope per channel, for _restore_formants()."""
    audio = audio.reshape(len(audio), -1)
    return [formant_envelope(audio[:, ch], sr, f0_high) for ch in range(audio.shape[1])]


def _restore_formants(
    shifted: np.ndarray,
    envelopes: list | None,
    ratio: float | np.ndarray,
    sr: int,
) -> np.ndarray:
    """Per channel, give `shifted` its source's spectral envelope back.

    `envelopes` None means formants are allowed to move with the pitch.
    """
    if envelopes is None:
        return shifted
    cols = [
        apply_formant_correction(shifted[:, ch], envelopes[ch], ratio, sr)
        for ch in range(shifted.shape[1])
    ]
    corrected = np.stack(cols, axis=1)
    # The correction redistributes energy across the spectrum and should not
    # change loudness, but its average gain is not exactly 0 dB (about -2 dB
    # shifting up a fifth, +1.5 dB shifting down). Left in, that tilts the
    # A/B balance of the crossfade. One gain for all channels restores the
    # level without touching the stereo image.
    before = float(np.sqrt(np.mean(shifted.astype(np.float64) ** 2)))
    after = float(np.sqrt(np.mean(corrected.astype(np.float64) ** 2)))
    if before > 1e-9 and after > 1e-9:
        corrected = corrected * (before / after)
    return corrected.astype(np.float32)


def _crossfade(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    """Equal-power blend — the two sides are now at the same pitch, so this is
    the only thing left to do."""
    angle = t * (np.pi / 2.0)
    return (np.cos(angle) * a + np.sin(angle) * b).astype(np.float32)


# ── Pitch detection ───────────────────────────────────────────────────────────

def _mono(audio: np.ndarray) -> np.ndarray:
    arr = audio.mean(axis=1) if audio.ndim == 2 else audio
    return arr.astype(np.float32)


def _detect_f0(audio: np.ndarray, sr: int, fmin: float, fmax: float) -> float:
    """Return median detected fundamental frequency (0.0 if unpitched)."""
    import librosa

    f0 = librosa.yin(_mono(audio), fmin=fmin, fmax=fmax, sr=sr)
    valid = f0[(f0 > fmin) & (f0 < fmax)]
    return float(np.median(valid)) if len(valid) else 0.0


def _f0_track(
    mono: np.ndarray,
    sr: int,
    fmin: float,
    fmax: float,
) -> np.ndarray | None:
    """Per-frame fundamental, gaps filled and octave jumps smoothed away.

    Returns None when nothing pitched was found.
    """
    import librosa
    from scipy.signal import medfilt

    f0 = librosa.yin(mono, fmin=fmin, fmax=fmax, sr=sr, hop_length=_HOP)
    valid = np.isfinite(f0) & (f0 > fmin) & (f0 < fmax)
    if not valid.any():
        return None

    idx = np.arange(len(f0), dtype=np.float64)
    # Interpolate across unvoiced frames in the log domain, then median-filter:
    # YIN drops the odd frame an octave out, and an octave error survives as an
    # audible jump in the shift ratio.
    log_f0 = np.interp(idx, idx[valid], np.log(f0[valid]))
    return np.exp(medfilt(log_f0, kernel_size=7))


def _to_samples(frame_values: np.ndarray, n_samples: int) -> np.ndarray:
    """Expand a per-frame curve to one value per sample."""
    frame_centres = np.arange(len(frame_values), dtype=np.float64) * _HOP
    return np.interp(
        np.arange(n_samples, dtype=np.float64), frame_centres, frame_values
    )


# ── Constant-rate shifting ────────────────────────────────────────────────────

def _shift_channels(
    audio: np.ndarray,
    sr: int,
    semitones: float,
    channels: int,
) -> np.ndarray:
    """Shift every channel by a fixed interval.

    Goes through the same path as the dynamic mode, not librosa's pitch_shift:
    librosa's phase vocoder has no phase locking, and its hollow, phasey smear
    was the dominant artefact of this plugin on tonal material.
    """
    if abs(semitones) < 0.01:
        return audio.copy()

    ratio = 2.0 ** (semitones / 12.0)
    out_chs: list[np.ndarray] = []
    for ch in range(channels):
        sig = audio[:, ch] if channels > 1 else audio.ravel()
        shifted = pitch_shift_varying(sig, np.full(len(sig), ratio), _HOP)
        out_chs.append(shifted.astype(np.float32))

    if channels == 1:
        return out_chs[0].reshape(-1, 1)
    return np.stack(out_chs, axis=1)
