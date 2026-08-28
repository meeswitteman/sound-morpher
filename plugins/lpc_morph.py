from __future__ import annotations

import numpy as np
from scipy.signal import lfilter

from plugins.base import (
    MorphPlugin,
    PluginParam,
    lpc_to_lsf,
    lsf_to_lpc,
    match_lengths,
)


class LpcMorphPlugin(MorphPlugin):
    """Source-filter morph: interpolates the LPC vocal-tract filter via LSFs."""

    name = "LPC / Source-Filter"
    description = (
        "Models each sound as source × filter. The LPC filter (vocal tract shape) "
        "is interpolated between A and B via Line Spectral Frequencies — which "
        "always produce a stable filter when interpolated. The excitation (source) "
        "can be taken from A, B, or a blend of both. Best on voiced speech and wind "
        "instruments; may sound buzzy on percussive material."
    )
    parameters = [
        PluginParam(
            name="lpc_order",
            label="LPC Order",
            type="int",
            default=16,
            min_val=4,
            max_val=32,
            tooltip=(
                "Number of LPC coefficients (must be even; odd values are rounded up). "
                "Rule of thumb: sr / 1000 + 2.  E.g. 16 for 8 kHz, 46 for 22 kHz. "
                "Higher = more spectral detail, slower."
            ),
        ),
        PluginParam(
            name="frame_ms",
            label="Frame (ms)",
            type="float",
            default=25.0,
            min_val=10.0,
            max_val=50.0,
            tooltip="Analysis frame length in milliseconds (20–30 ms typical for speech).",
        ),
        PluginParam(
            name="hop_ms",
            label="Hop (ms)",
            type="float",
            default=10.0,
            min_val=2.0,
            max_val=20.0,
            tooltip="Frame stride in milliseconds.",
        ),
        PluginParam(
            name="excitation",
            label="Excitation",
            type="choice",
            default="A",
            choices=["A", "B", "Blend"],
            tooltip=(
                "Source signal used to drive the morphed filter. "
                "'A' keeps A's voice character; 'B' keeps B's; "
                "'Blend' cross-fades the residuals."
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
        lpc_order: int = 16,
        frame_ms: float = 25.0,
        hop_ms: float = 10.0,
        excitation: str = "A",
        **_: object,
    ) -> list[np.ndarray]:
        if lpc_order % 2 != 0:
            lpc_order += 1

        a, b = match_lengths(audio_a, audio_b)
        channels = a.shape[1] if a.ndim == 2 else 1
        result: list[np.ndarray] = []

        for i in range(steps):
            t = i / (steps - 1) if steps > 1 else 0.0
            result.append(
                _lpc_morph(a, b, t, sample_rate, lpc_order, frame_ms, hop_ms, excitation, channels)
            )
            if progress_cb:
                progress_cb(i + 1)
        return result


# ── Per-step driver ────────────────────────────────────────────────────────────

def _lpc_morph(
    a: np.ndarray,
    b: np.ndarray,
    t: float,
    sr: int,
    order: int,
    frame_ms: float,
    hop_ms: float,
    excitation: str,
    channels: int,
) -> np.ndarray:
    out_ch = []
    for ch in range(channels):
        sig_a = a[:, ch] if channels > 1 else a.ravel()
        sig_b = b[:, ch] if channels > 1 else b.ravel()
        out_ch.append(_morph_mono(sig_a, sig_b, t, sr, order, frame_ms, hop_ms, excitation))
    if channels == 1:
        return out_ch[0].reshape(-1, 1)
    return np.stack(out_ch, axis=1)


def _morph_mono(
    sig_a: np.ndarray,
    sig_b: np.ndarray,
    t: float,
    sr: int,
    order: int,
    frame_ms: float,
    hop_ms: float,
    excitation: str,
) -> np.ndarray:
    import librosa

    # Frame length must be at least 2×order for reliable LPC estimation
    frame_len = max(order * 2 + 2, int(frame_ms * sr / 1000))
    hop_len   = max(1, int(hop_ms * sr / 1000))
    n = len(sig_a)

    out     = np.zeros(n, dtype=np.float64)
    weights = np.zeros(n, dtype=np.float64)
    window  = np.hanning(frame_len).astype(np.float64)

    pos = 0
    while pos + frame_len <= n:
        # Windowed before the LPC fit — an unwindowed (rectangular) frame leaks
        # energy across bins and gives a rougher formant estimate, which shows
        # up as buzz right at each frame boundary. vocoder.py already does this;
        # this plugin didn't.
        fa = sig_a[pos : pos + frame_len].astype(np.float64) * window
        fb = sig_b[pos : pos + frame_len].astype(np.float64) * window

        try:
            lpc_a = librosa.lpc(fa, order=order).astype(np.float64)
            lpc_b = librosa.lpc(fb, order=order).astype(np.float64)

            lsf_t = (1.0 - t) * lpc_to_lsf(lpc_a) + t * lpc_to_lsf(lpc_b)
            lpc_t = lsf_to_lpc(lsf_t)

            exc  = _get_excitation(fa, fb, lpc_a, lpc_b, t, excitation)
            synth = lfilter([1.0], lpc_t, exc)
        except Exception:
            # Silent frame or numerical failure → linear blend
            synth = (1.0 - t) * fa + t * fb

        # Frame is now windowed twice (once for analysis, once for the OLA
        # below), so the overlap-add normalises by the window's square — same
        # scheme as vocoder.py.
        out[pos : pos + frame_len]     += synth * window
        weights[pos : pos + frame_len] += window ** 2
        pos += hop_len

    # Remaining tail shorter than one full frame
    if pos < n:
        out[pos:]     += (1.0 - t) * sig_a[pos:].astype(np.float64) + t * sig_b[pos:].astype(np.float64)
        weights[pos:] += 1.0

    mask = weights > 1e-8
    out[mask] /= weights[mask]
    return out.astype(np.float32)


def _get_excitation(
    fa: np.ndarray,
    fb: np.ndarray,
    lpc_a: np.ndarray,
    lpc_b: np.ndarray,
    t: float,
    mode: str,
) -> np.ndarray:
    if mode == "B":
        return lfilter(lpc_b, [1.0], fb)
    if mode == "Blend":
        exc_a = lfilter(lpc_a, [1.0], fa)
        exc_b = lfilter(lpc_b, [1.0], fb)
        # Normalise each to unit RMS, then scale to interpolated energy
        rms_a = float(np.sqrt(np.mean(exc_a ** 2))) + 1e-12
        rms_b = float(np.sqrt(np.mean(exc_b ** 2))) + 1e-12
        rms_t = (1.0 - t) * rms_a + t * rms_b
        return ((1.0 - t) * exc_a / rms_a + t * exc_b / rms_b) * rms_t
    # Default: "A"
    return lfilter(lpc_a, [1.0], fa)
