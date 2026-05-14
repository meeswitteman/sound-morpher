from __future__ import annotations

import numpy as np
import scipy.signal as ss

from plugins.base import MorphPlugin, PluginParam, match_lengths


class VocoderPlugin(MorphPlugin):
    """LPC vocoder morph: blends the spectral envelopes of A and B frame-by-frame."""

    name = "Vocoder"
    description = (
        "LPC vocoder: analyses each sound's spectral envelope via linear prediction, "
        "then smoothly interpolates envelopes and excitation from A to B."
    )
    parameters = [
        PluginParam(
            name="lpc_order",
            label="LPC order",
            type="int",
            default=16,
            min_val=8,
            max_val=48,
            tooltip="Number of LPC coefficients. Higher = more spectral detail.",
        ),
        PluginParam(
            name="frame_ms",
            label="Frame (ms)",
            type="int",
            default=30,
            min_val=10,
            max_val=80,
            tooltip="Analysis/synthesis frame length in milliseconds.",
        ),
    ]

    def morph(
        self,
        audio_a: np.ndarray,
        audio_b: np.ndarray,
        steps: int,
        sample_rate: int,
        lpc_order: int = 16,
        frame_ms: int = 30,
        **_: object,
    ) -> list[np.ndarray]:
        a, b = match_lengths(audio_a, audio_b)
        a = _to_mono(a)
        b = _to_mono(b)

        result: list[np.ndarray] = []
        for i in range(steps):
            t = i / (steps - 1) if steps > 1 else 0.0
            mixed = _vocoder_mix(a, b, t, sample_rate, lpc_order, frame_ms)
            result.append(mixed.reshape(-1, 1).astype(np.float32))

        return result


# ── Implementation ────────────────────────────────────────────────────────────

def _to_mono(audio: np.ndarray) -> np.ndarray:
    arr = audio.astype(np.float32)
    if arr.ndim == 2:
        arr = arr.mean(axis=1)
    return arr


def _safe_lpc(frame: np.ndarray, order: int) -> np.ndarray:
    """Return LPC coefficients; fall back to all-pass on failure."""
    try:
        import librosa
        coeffs = librosa.lpc(frame, order=order)
        if not np.all(np.isfinite(coeffs)):
            raise ValueError("non-finite LPC")
        return coeffs.astype(np.float64)
    except Exception:
        c = np.zeros(order + 1, dtype=np.float64)
        c[0] = 1.0
        return c


def _bandwidth_expand(lpc: np.ndarray, gamma: float) -> np.ndarray:
    """Multiply lpc[k] by gamma^k — pulls all poles inward, guaranteeing stability."""
    k = np.arange(len(lpc), dtype=np.float64)
    return lpc * (gamma ** k)


def _vocoder_mix(
    a: np.ndarray,
    b: np.ndarray,
    t: float,
    sr: int,
    order: int,
    frame_ms: int,
) -> np.ndarray:
    frame_len = int(sr * frame_ms / 1000)
    hop = frame_len // 2
    n = len(a)

    a64 = a.astype(np.float64)
    b64 = b.astype(np.float64)

    output = np.zeros(n, dtype=np.float64)
    window = np.hanning(frame_len)
    norm = np.zeros(n, dtype=np.float64)

    # Equal-power weights for excitation blend
    w_a = np.cos(t * np.pi / 2.0)
    w_b = np.sin(t * np.pi / 2.0)

    pos = 0
    while pos + frame_len <= n:
        fa = a64[pos : pos + frame_len] * window
        fb = b64[pos : pos + frame_len] * window

        lpc_a = _safe_lpc(fa.astype(np.float32), order)
        lpc_b = _safe_lpc(fb.astype(np.float32), order)

        res_a = ss.lfilter(lpc_a, [1.0], fa)
        res_b = ss.lfilter(lpc_b, [1.0], fb)

        # Normalise residual RMS before blending so levels stay consistent
        rms_a = np.sqrt(np.mean(res_a ** 2) + 1e-12)
        rms_b = np.sqrt(np.mean(res_b ** 2) + 1e-12)
        target_rms = (1.0 - t) * rms_a + t * rms_b
        res_mixed = (w_a * res_a / rms_a + w_b * res_b / rms_b) * target_rms

        # Blend LPC envelopes and pull poles inward for guaranteed stability
        lpc_mixed = (1.0 - t) * lpc_a + t * lpc_b
        lpc_mixed = _bandwidth_expand(lpc_mixed, gamma=0.94)

        synth = ss.lfilter([1.0], lpc_mixed, res_mixed)

        # Convex combination of two stable LPC polynomials is NOT necessarily
        # stable; the blended filter can have poles with radius > 1/gamma=1.064
        # that bandwidth expansion does not pull inside the unit circle.
        # Fall back to a direct frame blend — no gain adjustment needed because
        # the blend is already at the natural input level.
        if not np.all(np.isfinite(synth)) or np.max(np.abs(synth)) > 50.0:
            synth = (1.0 - t) * fa + t * fb
        else:
            # Normalise synthesis level to match the blended INPUT frame RMS.
            # Using input_rms (not residual_rms) gives smooth dynamics because
            # input_rms tracks the audio envelope rather than the LPC fit quality.
            # Residual_rms is near-zero for harmonic frames (LPC fits well) and
            # large for noisy/transient frames, causing abrupt level jumps.
            input_rms = (
                (1.0 - t) * np.sqrt(np.mean(fa ** 2) + 1e-12)
                + t * np.sqrt(np.mean(fb ** 2) + 1e-12)
            )
            synth_rms = np.sqrt(np.mean(synth ** 2) + 1e-12)
            if synth_rms > 1e-10:
                synth *= np.clip(input_rms / synth_rms, 0.1, 2.0)

        output[pos : pos + frame_len] += synth * window
        norm[pos : pos + frame_len] += window ** 2

        pos += hop

    # OLA normalisation. Floor at 0.3 (≈ 40 % of the typical Hann 50 %-overlap
    # sum of 0.75) prevents division-by-near-zero at signal edges where only
    # one frame contributes and the window is close to zero.
    output /= np.maximum(norm, 0.3)

    return np.clip(output, -1.0, 1.0).astype(np.float32)
