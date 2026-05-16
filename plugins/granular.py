from __future__ import annotations

import numpy as np

from plugins.base import MorphPlugin, PluginParam, match_lengths


class GranularPlugin(MorphPlugin):
    """Granular morph: blend A→B by mixing windowed grains from both sources."""

    name = "Granular"
    description = (
        "Splits audio into short overlapping grains.  "
        "Steps blend the ratio of A-grains vs B-grains progressively."
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
        **_: object,
    ) -> list[np.ndarray]:
        a, b = match_lengths(audio_a, audio_b)
        grain_samples = max(16, int(grain_ms * sample_rate / 1000))
        hop = max(1, int(grain_samples * overlap))
        result: list[np.ndarray] = []

        for i in range(steps):
            t = i / (steps - 1) if steps > 1 else 0.0
            result.append(_granular_mix(a, b, t, grain_samples, hop))
            if progress_cb:
                progress_cb(i + 1)

        return result


def _granular_mix(
    a: np.ndarray,
    b: np.ndarray,
    t: float,
    grain_samples: int,
    hop: int,
) -> np.ndarray:
    n_frames = len(a)
    channels = a.shape[1] if a.ndim == 2 else 1
    window = np.hanning(grain_samples).astype(np.float32)
    out = np.zeros((n_frames, channels), dtype=np.float32)
    weight = np.zeros(n_frames, dtype=np.float32)

    pos = 0
    while pos < n_frames:
        end = min(pos + grain_samples, n_frames)
        length = end - pos
        w = window[:length]

        grain_a = a[pos:end] * w[:, np.newaxis]
        grain_b = b[pos:end] * w[:, np.newaxis]
        grain = (1 - t) * grain_a + t * grain_b

        out[pos:end] += grain
        weight[pos:end] += w
        pos += hop

    # Normalize by overlap weight (avoid division by zero)
    mask = weight > 1e-8
    out[mask] /= weight[mask, np.newaxis]
    return out.astype(np.float32)
