from __future__ import annotations

import sys
import types
from typing import Any

import numpy as np

from plugins.base import MorphPlugin, PluginParam, interp_magnitude, match_lengths


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
        PluginParam(
            name="envelope",
            label="Envelope",
            type="choice",
            default="log",
            choices=["log", "linear"],
            tooltip=(
                "log: formants of A slide into B's (true morph).  "
                "linear: both formant sets sound at once."
            ),
        ),
        PluginParam(
            name="channels",
            label="Channels",
            type="choice",
            default="stereo",
            choices=["stereo", "mono"],
            tooltip=(
                "stereo: per-channel envelope and aperiodicity on a shared pitch "
                "track, preserving the stereo image.  "
                "mono: downmix first — roughly twice as fast."
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
        f0_mode: str = "interpolate",
        frame_ms: float = 5.0,
        envelope: str = "log",
        channels: str = "stereo",
        **_: Any,
    ) -> list[np.ndarray]:
        pw = _ensure_pyworld()

        a, b = match_lengths(audio_a, audio_b)
        a, b = _as_2d(a), _as_2d(b)
        if channels == "mono":
            a, b = _downmix(a), _downmix(b)

        n_ch = a.shape[1]
        n_samples = a.shape[0]

        # F0 is estimated on the downmix and shared by every channel. Tracking
        # pitch per channel lets L and R drift apart by a few cents, which smears
        # the stereo image into a chorus.
        f0_a, tax_a = _estimate_f0(pw, _mono(a), sample_rate, frame_ms)
        f0_b, tax_b = _estimate_f0(pw, _mono(b), sample_rate, frame_ms)

        # Align frame counts (WORLD can return ±1 frame for same-length input)
        n_frames = min(len(f0_a), len(f0_b))
        f0_a, tax_a = f0_a[:n_frames], tax_a[:n_frames]
        f0_b, tax_b = f0_b[:n_frames], tax_b[:n_frames]

        # Envelope and aperiodicity stay per channel — that is where the stereo
        # information lives.
        sp_a, ap_a, sp_b, ap_b = [], [], [], []
        for ch in range(n_ch):
            ch_a = np.ascontiguousarray(a[:, ch])
            ch_b = np.ascontiguousarray(b[:, ch])
            sp_a.append(pw.cheaptrick(ch_a, f0_a, tax_a, sample_rate)[:n_frames])
            ap_a.append(pw.d4c(ch_a, f0_a, tax_a, sample_rate)[:n_frames])
            sp_b.append(pw.cheaptrick(ch_b, f0_b, tax_b, sample_rate)[:n_frames])
            ap_b.append(pw.d4c(ch_b, f0_b, tax_b, sample_rate)[:n_frames])

        result: list[np.ndarray] = []
        for i in range(steps):
            t = i / (steps - 1) if steps > 1 else 0.0

            if f0_mode == "keep_a":
                f0_mix = f0_a.copy()
            elif f0_mode == "keep_b":
                f0_mix = f0_b.copy()
            else:
                f0_mix = _interpolate_f0(f0_a, f0_b, t)

            cols = []
            for ch in range(n_ch):
                # Log-domain blend: the spectral envelope is a power spectrum, so
                # an arithmetic blend stacks A's formants on top of B's instead
                # of sliding one into the other.
                sp_mix = interp_magnitude(sp_a[ch], sp_b[ch], t, mode=envelope)
                ap_mix = np.clip(
                    interp_magnitude(ap_a[ch], ap_b[ch], t, mode=envelope), 0.0, 1.0
                )
                synth = pw.synthesize(
                    f0_mix, sp_mix, ap_mix, sample_rate, frame_period=frame_ms
                )
                if len(synth) >= n_samples:
                    synth = synth[:n_samples]
                else:
                    synth = np.pad(synth, (0, n_samples - len(synth)))
                cols.append(synth)

            step = np.stack(cols, axis=1)
            # One shared gain across channels, so a peak in L cannot shift the
            # stereo balance.
            peak = np.max(np.abs(step))
            if peak > 1.0:
                step = step / peak

            result.append(step.astype(np.float32))
            if progress_cb:
                progress_cb(i + 1)

        return result


# ── Helpers ───────────────────────────────────────────────────────────────────

def _as_2d(audio: np.ndarray) -> np.ndarray:
    arr = audio.astype(np.float64)
    return arr if arr.ndim == 2 else arr.reshape(-1, 1)


def _downmix(audio: np.ndarray) -> np.ndarray:
    return audio.mean(axis=1, keepdims=True)


def _mono(audio: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(audio.mean(axis=1))


def _estimate_f0(pw, mono: np.ndarray, sr: int, frame_ms: float):
    """DIO + StoneMask F0 track — the same pair wav2world uses internally."""
    f0, time_axis = pw.dio(mono, sr, frame_period=frame_ms)
    return pw.stonemask(mono, f0, time_axis, sr), time_axis


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
