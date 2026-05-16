from __future__ import annotations

import sys
import types
from typing import Any

import numpy as np

from plugins.base import MorphPlugin, PluginParam, match_lengths


def _ensure_pyworld():
    """Import pyworld, patching the pkg_resources dependency absent in Python 3.14+."""
    if 'pkg_resources' not in sys.modules:
        _m = types.ModuleType('pkg_resources')
        _m.get_distribution = lambda name: type('_Dist', (), {'version': '0.3.5'})()
        sys.modules['pkg_resources'] = _m
    import pyworld as pw
    return pw


class WorldVocoderPlugin(MorphPlugin):
    """WORLD vocoder morph: interpolates F0, spectral envelope, and aperiodicity."""

    name = "WORLD Vocoder"
    description = (
        "WORLD vocoder: decomposes audio into pitch (F0), spectral envelope, and "
        "aperiodicity, then interpolates all three between A and B. "
        "Best results with voices and monophonic melodic samples."
    )
    parameters = [
        PluginParam(
            name="f0_mode",
            label="Pitch",
            type="choice",
            default="interpolate",
            choices=["interpolate", "keep_a", "keep_b"],
            tooltip=(
                "interpolate: blend pitch smoothly from A to B.  "
                "keep_a: always use A's pitch.  "
                "keep_b: always use B's pitch."
            ),
        ),
        PluginParam(
            name="frame_ms",
            label="Frame (ms)",
            type="float",
            default=5.0,
            min_val=1.0,
            max_val=20.0,
            tooltip="Analysis frame period in ms. Lower = more temporal detail, slower.",
        ),
    ]

    def morph(
        self,
        audio_a: np.ndarray,
        audio_b: np.ndarray,
        steps: int,
        sample_rate: int,
        progress_cb=None,
        f0_mode: str = "interpolate",
        frame_ms: float = 5.0,
        **_: Any,
    ) -> list[np.ndarray]:
        pw = _ensure_pyworld()

        a, b = match_lengths(audio_a, audio_b)
        a_mono = _to_mono(a)
        b_mono = _to_mono(b)
        n_samples = len(a_mono)

        # WORLD analysis of both sources
        f0_a, sp_a, ap_a = pw.wav2world(a_mono, sample_rate, frame_period=frame_ms)
        f0_b, sp_b, ap_b = pw.wav2world(b_mono, sample_rate, frame_period=frame_ms)

        # Align frame counts (WORLD can return ±1 frame for same-length input)
        n_frames = min(len(f0_a), len(f0_b))
        f0_a, sp_a, ap_a = f0_a[:n_frames], sp_a[:n_frames], ap_a[:n_frames]
        f0_b, sp_b, ap_b = f0_b[:n_frames], sp_b[:n_frames], ap_b[:n_frames]

        result: list[np.ndarray] = []
        for i in range(steps):
            t = i / (steps - 1) if steps > 1 else 0.0

            sp_mix = (1.0 - t) * sp_a + t * sp_b
            ap_mix = np.clip((1.0 - t) * ap_a + t * ap_b, 0.0, 1.0)

            if f0_mode == "keep_a":
                f0_mix = f0_a.copy()
            elif f0_mode == "keep_b":
                f0_mix = f0_b.copy()
            else:
                f0_mix = _interpolate_f0(f0_a, f0_b, t)

            synth = pw.synthesize(
                f0_mix, sp_mix, ap_mix, sample_rate, frame_period=frame_ms
            )

            if len(synth) >= n_samples:
                synth = synth[:n_samples]
            else:
                synth = np.pad(synth, (0, n_samples - len(synth)))

            peak = np.max(np.abs(synth))
            if peak > 1.0:
                synth /= peak

            result.append(synth.reshape(-1, 1).astype(np.float32))
            if progress_cb:
                progress_cb(i + 1)

        return result


# ── Helpers ───────────────────────────────────────────────────────────────────

def _to_mono(audio: np.ndarray) -> np.ndarray:
    arr = audio.astype(np.float64)
    if arr.ndim == 2:
        arr = arr.mean(axis=1)
    return arr


def _interpolate_f0(f0_a: np.ndarray, f0_b: np.ndarray, t: float) -> np.ndarray:
    """Interpolate F0 in the log domain (perceptually linear pitch).

    Unvoiced frames (F0 == 0) are handled per case:
    - both voiced   → log-domain blend
    - A only voiced → keep A's pitch for t < 0.5, then unvoiced
    - B only voiced → unvoiced for t < 0.5, then use B's pitch
    - both unvoiced → stays unvoiced
    """
    voiced_a = f0_a > 0.0
    voiced_b = f0_b > 0.0
    f0_mix = np.zeros(len(f0_a), dtype=np.float64)

    both = voiced_a & voiced_b
    if np.any(both):
        f0_mix[both] = np.exp(
            (1.0 - t) * np.log(f0_a[both]) + t * np.log(f0_b[both])
        )

    a_only = voiced_a & ~voiced_b
    if np.any(a_only) and t < 0.5:
        f0_mix[a_only] = f0_a[a_only]

    b_only = ~voiced_a & voiced_b
    if np.any(b_only) and t >= 0.5:
        f0_mix[b_only] = f0_b[b_only]

    return f0_mix
