from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.signal as ss

from plugins.base import (
    MorphPlugin,
    PluginParam,
    lpc_to_lsf,
    lsf_to_lpc,
    match_lengths,
)


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
            tooltip=(
                "Number of LPC coefficients (rounded up to even). "
                "Higher = more spectral detail."
            ),
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
        PluginParam(
            name="channels",
            label="Channels",
            type="choice",
            default="stereo",
            choices=["stereo", "mono"],
            tooltip=(
                "stereo: analyse and synthesise each channel, preserving the "
                "stereo image.  "
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
        lpc_order: int = 16,
        frame_ms: int = 30,
        channels: str = "stereo",
        **_: object,
    ) -> list[np.ndarray]:
        # LSF conversion is only defined for an even-order polynomial.
        if lpc_order % 2 != 0:
            lpc_order += 1

        a, b = match_lengths(audio_a, audio_b)
        a, b = _as_2d(a), _as_2d(b)
        if channels == "mono":
            a, b = _downmix(a), _downmix(b)

        n_ch = a.shape[1]
        frame_len = max(lpc_order * 2 + 2, int(sample_rate * frame_ms / 1000))
        hop = frame_len // 2

        # Analysis does not depend on t, so it runs once per channel instead of
        # once per channel per step. That is what pays for the extra channel.
        analyses = [
            _analyse(a[:, ch], b[:, ch], lpc_order, frame_len, hop)
            for ch in range(n_ch)
        ]

        result: list[np.ndarray] = []
        for i in range(steps):
            t = i / (steps - 1) if steps > 1 else 0.0
            cols = [
                _synthesise(analyses[ch], a[:, ch], b[:, ch], t, frame_len, hop)
                for ch in range(n_ch)
            ]
            result.append(np.stack(cols, axis=1).astype(np.float32))
            if progress_cb:
                progress_cb(i + 1)

        return result


# ── Helpers ───────────────────────────────────────────────────────────────────

@dataclass
class _Analysis:
    """t-independent LPC analysis of one channel of both sources."""

    positions: np.ndarray   # (n_frames,)   frame start offsets
    lpc_a: np.ndarray       # (n_frames, order+1)
    lpc_b: np.ndarray
    lsf_a: np.ndarray       # (n_frames, order)
    lsf_b: np.ndarray


def _as_2d(audio: np.ndarray) -> np.ndarray:
    arr = audio.astype(np.float64)
    return arr if arr.ndim == 2 else arr.reshape(-1, 1)


def _downmix(audio: np.ndarray) -> np.ndarray:
    return audio.mean(axis=1, keepdims=True)


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


def _analyse(
    sig_a: np.ndarray,
    sig_b: np.ndarray,
    order: int,
    frame_len: int,
    hop: int,
) -> _Analysis:
    """Per-frame LPC fit of both sources, kept as coefficients and as LSFs.

    Only the frame positions and these small coefficient sets are stored — a few
    hundred bytes per frame. The residuals are cheap to recompute and would
    otherwise cost as much memory as the signal itself.
    """
    window = np.hanning(frame_len)
    n = len(sig_a)

    positions: list[int] = []
    lpc_a: list[np.ndarray] = []
    lpc_b: list[np.ndarray] = []
    lsf_a: list[np.ndarray] = []
    lsf_b: list[np.ndarray] = []

    pos = 0
    while pos + frame_len <= n:
        fa = sig_a[pos : pos + frame_len] * window
        fb = sig_b[pos : pos + frame_len] * window

        coef_a = _safe_lpc(fa.astype(np.float32), order)
        coef_b = _safe_lpc(fb.astype(np.float32), order)

        positions.append(pos)
        lpc_a.append(coef_a)
        lpc_b.append(coef_b)
        lsf_a.append(lpc_to_lsf(coef_a))
        lsf_b.append(lpc_to_lsf(coef_b))
        pos += hop

    if not positions:
        return _Analysis(
            np.zeros(0, dtype=np.int64),
            np.zeros((0, order + 1)), np.zeros((0, order + 1)),
            np.zeros((0, order)), np.zeros((0, order)),
        )

    return _Analysis(
        np.array(positions, dtype=np.int64),
        np.array(lpc_a), np.array(lpc_b),
        np.array(lsf_a), np.array(lsf_b),
    )


def _synthesise(
    analysis: _Analysis,
    sig_a: np.ndarray,
    sig_b: np.ndarray,
    t: float,
    frame_len: int,
    hop: int,
) -> np.ndarray:
    n = len(sig_a)
    window = np.hanning(frame_len)
    output = np.zeros(n, dtype=np.float64)
    norm = np.zeros(n, dtype=np.float64)

    # Equal-power weights for excitation blend
    w_a = np.cos(t * np.pi / 2.0)
    w_b = np.sin(t * np.pi / 2.0)

    for k, pos in enumerate(analysis.positions):
        fa = sig_a[pos : pos + frame_len] * window
        fb = sig_b[pos : pos + frame_len] * window

        # Interpolating LSFs always yields a stable filter, so the blended
        # envelope can be used directly — no bandwidth expansion to force the
        # poles inside the unit circle, and no fallback to a plain frame blend
        # on the frames where that failed. Both used to colour the output:
        # expansion widened every formant, and the fallback made the character
        # jump between neighbouring frames.
        lpc_t = lsf_to_lpc((1.0 - t) * analysis.lsf_a[k] + t * analysis.lsf_b[k])

        res_a = ss.lfilter(analysis.lpc_a[k], [1.0], fa)
        res_b = ss.lfilter(analysis.lpc_b[k], [1.0], fb)

        # Normalise residual RMS before blending so levels stay consistent
        rms_a = np.sqrt(np.mean(res_a ** 2) + 1e-12)
        rms_b = np.sqrt(np.mean(res_b ** 2) + 1e-12)
        target_rms = (1.0 - t) * rms_a + t * rms_b
        res_mixed = (w_a * res_a / rms_a + w_b * res_b / rms_b) * target_rms

        synth = ss.lfilter([1.0], lpc_t, res_mixed)

        if not np.all(np.isfinite(synth)):
            synth = (1.0 - t) * fa + t * fb
        else:
            # Normalise synthesis level to match the blended INPUT frame RMS.
            # Using input_rms (not residual_rms) gives smooth dynamics because
            # input_rms tracks the audio envelope rather than the LPC fit quality.
            # Residual_rms is near-zero for harmonic frames (LPC fits well) and
            # large for noisy/transient frames, causing abrupt level jumps.
            #
            # The correction itself is deliberately unbounded. LSF interpolation
            # guarantees the blended filter is *stable*, but not that its gain is
            # modest: poles just inside the unit circle still ring enormously, and
            # a bounded correction cannot pull those frames back down — the output
            # then sits ~17 dB hot and clips. Because the target is a smooth
            # function of the input envelope, the size of the correction does not
            # affect how smooth the result is; clamping it is what would make the
            # level jump between frames.
            input_rms = (
                (1.0 - t) * np.sqrt(np.mean(fa ** 2) + 1e-12)
                + t * np.sqrt(np.mean(fb ** 2) + 1e-12)
            )
            synth_rms = np.sqrt(np.mean(synth ** 2) + 1e-12)
            if synth_rms > 1e-10:
                synth *= input_rms / synth_rms

        output[pos : pos + frame_len] += synth * window
        norm[pos : pos + frame_len] += window ** 2

    # OLA normalisation. Floor at 0.3 (≈ 40 % of the typical Hann 50 %-overlap
    # sum of 0.75) prevents division-by-near-zero at signal edges where only
    # one frame contributes and the window is close to zero.
    output /= np.maximum(norm, 0.3)

    return np.clip(output, -1.0, 1.0).astype(np.float32)
