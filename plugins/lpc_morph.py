from __future__ import annotations

import numpy as np
from scipy.signal import lfilter

from plugins.base import MorphPlugin, PluginParam, match_lengths

# Frequency grid for zero-crossing LSF search
_N_EVAL = 512


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
        lpc_order: int = 16,
        frame_ms: float = 25.0,
        hop_ms: float = 10.0,
        excitation: str = "A",
        **_: object,
    ) -> list[np.ndarray]:
        # Force even order — LSF math is simpler and well-defined
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
        fa = sig_a[pos : pos + frame_len].astype(np.float64)
        fb = sig_b[pos : pos + frame_len].astype(np.float64)

        try:
            lpc_a = librosa.lpc(fa, order=order).astype(np.float64)
            lpc_b = librosa.lpc(fb, order=order).astype(np.float64)

            lsf_a = _lpc_to_lsf(lpc_a)
            lsf_b = _lpc_to_lsf(lpc_b)
            lsf_t = (1.0 - t) * lsf_a + t * lsf_b
            lpc_t = _lsf_to_lpc(lsf_t)

            exc  = _get_excitation(fa, fb, lpc_a, lpc_b, t, excitation)
            synth = lfilter([1.0], lpc_t, exc)
        except Exception:
            # Silent frame or numerical failure → linear blend
            synth = (1.0 - t) * fa + t * fb

        out[pos : pos + frame_len]     += synth * window
        weights[pos : pos + frame_len] += window
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


# ── LPC ↔ LSF conversion ───────────────────────────────────────────────────────

def _lpc_to_lsf(a: np.ndarray) -> np.ndarray:
    """LPC polynomial a (a[0]=1, even order p) → LSF in (0, π), shape (p,).

    Uses zero-crossing search on P(e^jω) and Q(e^jω) — no root-finding needed,
    making this numerically robust even for near-singular frames.
    """
    p = len(a) - 1         # even order
    a_r = a[::-1]

    # P and Q polynomials in z^{-1} (length p+2)
    P = np.zeros(p + 2)
    Q = np.zeros(p + 2)
    P[: p + 1] += a;  P[1:] += a_r   # symmetric:     P[k] = P[p+1-k]
    Q[: p + 1] += a;  Q[1:] -= a_r   # antisymmetric: Q[k] = -Q[p+1-k]

    # Sample both polynomials along the unit circle z = e^{jω}
    ω = np.linspace(0.0, np.pi, _N_EVAL + 2)[1:-1]   # avoid exactly 0 and π
    k = np.arange(p + 2, dtype=np.float64)
    phase = np.outer(k, ω)                             # shape (p+2, N_EVAL)

    # P and Q each carry a phase factor e^{-j(p+1)ω/2} on the unit circle.
    # Cancelling it gives real-valued g_P and g_Q whose zero crossings are
    # the actual LSFs — without spurious zeros from the phase term.
    phase_comp = np.exp(1j * (p + 1) / 2.0 * ω)      # shape (N_EVAL,)
    P_z = P @ np.exp(-1j * phase)
    Q_z = Q @ np.exp(-1j * phase)

    P_vals = (P_z * phase_comp).real   # g_P(ω): real, zeros give P-type LSFs
    Q_vals = (Q_z * phase_comp).imag   # g_Q(ω): real (extra j in Q decomp)

    def _crossings(vals: np.ndarray) -> np.ndarray:
        idx = np.flatnonzero(np.sign(vals[:-1]) != np.sign(vals[1:]))
        if not len(idx):
            return np.empty(0)
        dω = ω[1] - ω[0]
        frac = vals[idx] / (vals[idx] - vals[idx + 1])
        return ω[idx] + np.clip(frac, 0.0, 1.0) * dω

    lsf = np.sort(np.concatenate([_crossings(P_vals), _crossings(Q_vals)]))

    if len(lsf) < p:
        # Fallback: uniformly spaced LSFs (stable, just inaccurate)
        fallback = np.linspace(0.05, np.pi - 0.05, p)
        lsf = np.sort(np.concatenate([lsf, fallback[len(lsf):]]))

    return lsf[:p].copy()


def _lsf_to_lpc(lsf: np.ndarray) -> np.ndarray:
    """LSF in (0, π), shape (even p,) → LPC polynomial, shape (p+1,), a[0]=1.

    Reconstructs P and Q from their conjugate-pair roots, then recovers
    A(z) = (P(z) + Q(z)) / 2.

    Convention (even p):
      even-indexed LSFs [0,2,4,…] → P-type roots  (P also has trivial root z=-1)
      odd-indexed  LSFs [1,3,5,…] → Q-type roots  (Q also has trivial root z=+1)
    """
    p = len(lsf)
    lsf_s = np.sort(lsf)

    p_freqs = lsf_s[0::2]   # P roots
    q_freqs = lsf_s[1::2]   # Q roots

    def _build(freqs: np.ndarray) -> np.ndarray:
        """Product of quadratic factors for complex-conjugate root pairs."""
        poly = np.array([1.0])
        for ω in freqs:
            poly = np.convolve(poly, [1.0, -2.0 * np.cos(ω), 1.0])
        return poly

    # Reattach trivial roots then recover A(z)
    P = np.convolve(_build(p_freqs), [1.0, 1.0])   # multiply by (1 + z^{-1})
    Q = np.convolve(_build(q_freqs), [1.0, -1.0])  # multiply by (1 - z^{-1})

    n = max(len(P), len(Q))
    P = np.pad(P, (0, n - len(P)))
    Q = np.pad(Q, (0, n - len(Q)))

    a = (P + Q) / 2.0
    a = a.real / a[0]          # normalise so a[0] = 1
    return a[: p + 1]
