from __future__ import annotations

import numpy as np

from plugins.base import MorphPlugin, PluginParam, match_lengths


class CrossfadePlugin(MorphPlugin):
    """Linear amplitude crossfade from Sound A to Sound B."""

    name = "Crossfade"
    description = "Simple linear volume blend: A fades out while B fades in."
    parameters = [
        PluginParam(
            name="curve",
            label="Curve",
            type="choice",
            default="linear",
            choices=["linear", "equal-power"],
            tooltip=(
                "linear: straight volume ramp.  "
                "equal-power: sinusoidal ramp that preserves perceived loudness."
            ),
        )
    ]

    def morph(
        self,
        audio_a: np.ndarray,
        audio_b: np.ndarray,
        steps: int,
        sample_rate: int,
        progress_cb=None,
        curve: str = "linear",
        **_: object,
    ) -> list[np.ndarray]:
        a, b = match_lengths(audio_a, audio_b)
        result: list[np.ndarray] = []

        for i in range(steps):
            t = i / (steps - 1) if steps > 1 else 0.0
            t_a, t_b = _blend_factors(t, curve)
            result.append((t_a * a + t_b * b).astype(np.float32))
            if progress_cb:
                progress_cb(i + 1)

        return result


def _blend_factors(t: float, curve: str) -> tuple[float, float]:
    if curve == "equal-power":
        import math
        angle = t * (math.pi / 2)
        return math.cos(angle), math.sin(angle)
    # linear (default)
    return 1.0 - t, t
