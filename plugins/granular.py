from __future__ import annotations

import numpy as np

from plugins.base import MorphPlugin, PluginParam, match_lengths


class GranularPlugin(MorphPlugin):
    """Granular morph: rebuild the sound from grains drawn from A and from B."""

    name = "Granular"
    description = (
        "Splits both sources into short overlapping grains and reassembles the "
        "output grain by grain. As the morph progresses, more grains are drawn "
        "from B and fewer from A, so the sound audibly disintegrates and "
        "reassembles instead of simply fading over."
    )
    parameters = [
        PluginParam(
            name="grain_ms",
            label="Grain (ms)",
            type="float",
            default=80.0,
            min_val=10.0,
            max_val=500.0,
            tooltip="Length of each grain in milliseconds.",
        ),
        PluginParam(
            name="overlap",
            label="Overlap",
            type="float",
            default=0.5,
            min_val=0.1,
            max_val=0.9,
            tooltip="Fraction of grain length used as hop (lower = more overlap).",
        ),
        PluginParam(
            name="mode",
            label="Mode",
            type="choice",
            default="scatter",
            choices=["scatter", "mix"],
            tooltip=(
                "scatter: each grain is taken whole from either A or B, with the "
                "odds shifting from A to B across the sequence — the actual "
                "granular effect.  "
                "mix: blend both sources inside every grain. Mathematically "
                "identical to a linear crossfade; kept for comparison."
            ),
        ),
        PluginParam(
            name="jitter_ms",
            label="Jitter (ms)",
            type="float",
            default=15.0,
            min_val=0.0,
            max_val=200.0,
            tooltip=(
                "How far a grain may read away from its nominal position. "
                "0 = strictly time-aligned; higher = smeared, cloud-like."
            ),
        ),
        PluginParam(
            name="pitch_jitter",
            label="Pitch jitter",
            type="float",
            default=0.0,
            min_val=0.0,
            max_val=12.0,
            tooltip=(
                "Random per-grain pitch deviation in semitones (±). "
                "0 = off. Small values thicken, large values shimmer."
            ),
        ),
        PluginParam(
            name="seed",
            label="Seed",
            type="int",
            default=0,
            min_val=0,
            max_val=9999,
            tooltip=(
                "Random seed for grain selection and jitter. The same seed always "
                "produces the same morph; change it for a different scatter."
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
        grain_ms: float = 80.0,
        overlap: float = 0.5,
        mode: str = "scatter",
        jitter_ms: float = 15.0,
        pitch_jitter: float = 0.0,
        seed: int = 0,
        **_: object,
    ) -> list[np.ndarray]:
        a, b = match_lengths(audio_a, audio_b)
        grain_samples = max(16, int(grain_ms * sample_rate / 1000))
        hop = max(1, int(grain_samples * overlap))

        # One draw shared by every step: grain k switches from A to B at exactly
        # t == threshold[k], and its position/pitch offsets stay put. Drawing per
        # step instead would re-scatter the whole cloud on every step and destroy
        # the sense of a single sound progressing.
        n_grains = len(range(0, len(a), hop)) + 1
        rng = np.random.default_rng(seed)
        thresholds = rng.random(n_grains)
        jitter = int(jitter_ms * sample_rate / 1000)
        offsets = (
            rng.integers(-jitter, jitter + 1, n_grains)
            if jitter > 0
            else np.zeros(n_grains, dtype=np.int64)
        )
        rates = (
            2.0 ** (rng.uniform(-pitch_jitter, pitch_jitter, n_grains) / 12.0)
            if pitch_jitter > 0.0
            else np.ones(n_grains)
        )

        result: list[np.ndarray] = []
        for i in range(steps):
            t = i / (steps - 1) if steps > 1 else 0.0
            # Endpoints are the untouched sources — granulating them would smear
            # the two anchors of the sequence.
            if t <= 0.0:
                result.append(a.astype(np.float32))
            elif t >= 1.0:
                result.append(b.astype(np.float32))
            else:
                result.append(
                    _granular_mix(
                        a, b, t, grain_samples, hop, mode, thresholds, offsets, rates
                    )
                )
            if progress_cb:
                progress_cb(i + 1)

        return result


def _granular_mix(
    a: np.ndarray,
    b: np.ndarray,
    t: float,
    grain_samples: int,
    hop: int,
    mode: str,
    thresholds: np.ndarray,
    offsets: np.ndarray,
    rates: np.ndarray,
) -> np.ndarray:
    n_frames = len(a)
    channels = a.shape[1] if a.ndim == 2 else 1
    # Endpoints trimmed off the Hann window: a window that reaches exactly zero
    # leaves the first output sample with zero overlap weight, i.e. a dropout.
    window = np.hanning(grain_samples + 2)[1:-1].astype(np.float32)
    out = np.zeros((n_frames, channels), dtype=np.float32)
    weight = np.zeros(n_frames, dtype=np.float32)

    pos = 0
    k = 0
    while pos < n_frames:
        end = min(pos + grain_samples, n_frames)
        length = end - pos
        w = window[:length]

        if mode == "mix":
            grain = (1 - t) * a[pos:end] + t * b[pos:end]
        else:
            source = b if thresholds[k] < t else a
            grain = _read_grain(source, pos + int(offsets[k]), length, float(rates[k]))

        out[pos:end] += grain * w[:, np.newaxis]
        weight[pos:end] += w
        pos += hop
        k += 1

    # Normalize by overlap weight (avoid division by zero)
    mask = weight > 1e-8
    out[mask] /= weight[mask, np.newaxis]
    return out.astype(np.float32)


def _read_grain(
    src: np.ndarray,
    start: int,
    length: int,
    rate: float,
) -> np.ndarray:
    """Read `length` output frames from `src` at playback speed `rate`.

    Reading faster or slower than 1.0 transposes the grain, which is what makes
    pitch jitter possible without a separate pitch shifter.
    """
    n = len(src)
    if abs(rate - 1.0) < 1e-9:
        start = int(np.clip(start, 0, max(0, n - length)))
        seg = src[start : start + length]
        if len(seg) < length:
            seg = np.pad(seg, ((0, length - len(seg)), (0, 0)))
        return seg

    idx = np.clip(start + np.arange(length) * rate, 0.0, n - 1.0)
    lo = np.floor(idx).astype(np.int64)
    hi = np.minimum(lo + 1, n - 1)
    frac = (idx - lo)[:, np.newaxis].astype(np.float32)
    return src[lo] * (1.0 - frac) + src[hi] * frac
