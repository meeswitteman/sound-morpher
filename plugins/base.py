from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np


@dataclass
class PluginParam:
    """Declarative description of one plugin parameter."""

    name: str
    label: str
    type: Literal["float", "int", "bool", "choice"]
    default: Any
    min_val: Any = None
    max_val: Any = None
    choices: list[str] | None = None
    tooltip: str = ""


class MorphPlugin(ABC):
    """Base class for all morphing algorithm plugins."""

    name: str = ""
    description: str = ""
    parameters: list[PluginParam] = []

    @abstractmethod
    def morph(
        self,
        audio_a: np.ndarray,
        audio_b: np.ndarray,
        steps: int,
        sample_rate: int,
        progress_cb=None,
        **params: Any,
    ) -> list[np.ndarray]:
        """Return exactly `steps` audio arrays interpolating from A to B.

        Index 0 is 100% A, index steps-1 is 100% B.
        All returned arrays have the same shape as the (length-matched) inputs.
        """
        ...


# ── Spectral interpolation ────────────────────────────────────────────────────

def interp_magnitude(
    mag_a: np.ndarray,
    mag_b: np.ndarray,
    t: float,
    mode: str = "log",
    floor_db: float = -100.0,
) -> np.ndarray:
    """Interpolate two magnitude spectra.

    "log" (default) takes the geometric mean — exp((1-t)·ln|A| + t·ln|B|) — so a
    partial present in A and absent in B *fades*, instead of both partials being
    audible side by side as with the arithmetic mean. That is the difference
    between a morph and a spectral crossfade.

    Because the geometric mean is always ≤ the arithmetic mean, log mode makes
    intermediate steps quieter; pair it with match_step_loudness().

    "linear" is the plain arithmetic mean (pre-0.2 behaviour).
    """
    if mode == "linear":
        return (1.0 - t) * mag_a + t * mag_b

    ref = max(float(mag_a.max(initial=0.0)), float(mag_b.max(initial=0.0)), 1e-12)
    floor = ref * (10.0 ** (floor_db / 20.0))
    log_a = np.log(np.maximum(mag_a, floor))
    log_b = np.log(np.maximum(mag_b, floor))
    return np.exp((1.0 - t) * log_a + t * log_b)


def interp_phase(
    phase_a: np.ndarray,
    phase_b: np.ndarray,
    t: float,
    mode: str = "shortest-arc",
) -> np.ndarray:
    """Interpolate two phase spectra (radians).

    Phase is circular, so the naive (1-t)·∠A + t·∠B averages straight through the
    ±π wrap and lands on an unrelated angle — which cancels partials that should
    reinforce (measurably: two 440 Hz tones a phase-offset apart lose 11 dB at
    t=0.5). The modes here all avoid that:

    "shortest-arc" rotates ∠A toward ∠B the short way round the circle.
    "dominant"     uses the phase of whichever source is weighted higher.
    "linear"       the old, broken behaviour — kept for A/B comparison.
    """
    if mode == "linear":
        return (1.0 - t) * phase_a + t * phase_b
    if mode == "dominant":
        return phase_b if t >= 0.5 else phase_a

    delta = (phase_b - phase_a + np.pi) % (2.0 * np.pi) - np.pi
    return phase_a + t * delta


# ── LPC ↔ LSF ─────────────────────────────────────────────────────────────────

# Frequency grid for the zero-crossing LSF search
_N_EVAL = 512


def interp_lpc(lpc_a: np.ndarray, lpc_b: np.ndarray, t: float) -> np.ndarray:
    """Interpolate two LPC polynomials (a[0] = 1, even order) through the LSF domain.

    A convex combination of the coefficients themselves is *not* guaranteed to be
    stable: the blended all-pole filter can end up with poles outside the unit
    circle and the synthesis blows up. Line Spectral Frequencies do not have that
    problem — a filter is stable exactly when its LSFs are ordered and lie in
    (0, π), and linear interpolation of two such sets preserves both properties.
    """
    lsf = (1.0 - t) * lpc_to_lsf(lpc_a) + t * lpc_to_lsf(lpc_b)
    return lsf_to_lpc(lsf)


def lpc_to_lsf(a: np.ndarray) -> np.ndarray:
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


def lsf_to_lpc(lsf: np.ndarray) -> np.ndarray:
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


# ── Level matching ────────────────────────────────────────────────────────────

def match_step_loudness(
    steps: list[np.ndarray],
    audio_a: np.ndarray,
    audio_b: np.ndarray,
    max_gain_db: float = 12.0,
    ceiling: float = 0.99,
    max_trim_db: float = 3.0,
    sample_rate: int = 44100,
) -> list[np.ndarray]:
    """Scale each step so its RMS tracks a straight line from A's RMS to B's.

    Blending two uncorrelated signals costs ~3 dB in the middle of the sequence,
    and geometric-mean spectral interpolation costs more still, so intermediate
    steps arrive audibly thinner than the endpoints. This restores the intended
    loudness curve.

    Peaks are then handled in two stages. First a single shared gain, up to
    `max_trim_db`, because one gain across the whole set is transparent and keeps
    the relative levels intact. Beyond that a shared trim would be the wrong tool:
    a single overshooting transient — which spectral reconstruction readily
    produces — would drag the entire sequence down with it. Whatever still pokes
    over the ceiling is caught by a look-ahead limiter instead, which only acts
    where and when it has to.
    """
    if not steps:
        return steps

    rms_a = _rms(audio_a)
    rms_b = _rms(audio_b)
    max_gain = 10.0 ** (max_gain_db / 20.0)
    n = len(steps)

    scaled: list[np.ndarray] = []
    for i, step in enumerate(steps):
        t = i / (n - 1) if n > 1 else 0.0
        target = (1.0 - t) * rms_a + t * rms_b
        current = _rms(step)
        if current < 1e-9 or target < 1e-9:
            scaled.append(step)
            continue
        gain = np.clip(target / current, 1.0 / max_gain, max_gain)
        scaled.append((step * gain).astype(np.float32))

    peak = max((float(np.max(np.abs(s))) for s in scaled), default=0.0)
    if peak > ceiling:
        trim = max(ceiling / peak, 10.0 ** (-max_trim_db / 20.0))
        scaled = [(s * trim).astype(np.float32) for s in scaled]
        if peak * trim > ceiling:
            scaled = [limit_peaks(s, sample_rate, ceiling) for s in scaled]

    return scaled


def limit_peaks(
    audio: np.ndarray,
    sample_rate: int,
    ceiling: float = 0.99,
    window_ms: float = 5.0,
) -> np.ndarray:
    """Look-ahead peak limiter. Returns `audio` with nothing above `ceiling`.

    Hard clipping generates broadband harmonics at full level; riding a smooth
    gain curve instead makes the reduction inaudible. Two steps:

    1. A *running minimum* of the required gain over ±W samples. The reduction is
       therefore already in force when the transient arrives, instead of catching
       up after it — that is what "look-ahead" means here.
    2. Two boxcar passes, giving a triangular smoothing kernel of half-width ≤ W,
       to round off the corners the running minimum leaves behind.

    Keeping the smoothing half-width inside the minimum's is what makes the
    ceiling a guarantee rather than a target: every sample the kernel averages
    over is itself ≤ the gain required at the centre, so the smoothed gain cannot
    exceed it either. No clamp afterwards, and therefore no reintroduced corners.

    The curve is symmetric — equal attack and release. Real-time limiters use a
    longer release to avoid pumping, but at these window lengths, offline, there
    is nothing to hear.

    Gain is derived from the loudest channel, so limiting cannot shift the stereo
    image.
    """
    from scipy.ndimage import minimum_filter1d, uniform_filter1d

    arr = audio.astype(np.float32)
    flat = arr.ndim == 1
    if flat:
        arr = arr.reshape(-1, 1)

    envelope = np.max(np.abs(arr), axis=1)
    if envelope.size == 0 or float(envelope.max()) <= ceiling:
        return audio.astype(np.float32)

    required = np.minimum(1.0, ceiling / np.maximum(envelope, 1e-12))

    w = max(1, int(sample_rate * window_ms / 1000))
    gain = minimum_filter1d(required, size=2 * w + 1, mode="nearest")

    smooth = max(1, w // 2)
    gain = uniform_filter1d(gain, size=smooth, mode="nearest")
    gain = uniform_filter1d(gain, size=smooth, mode="nearest")

    out = arr * gain[:, np.newaxis]
    return (out.ravel() if flat else out).astype(np.float32)


def _rms(audio: np.ndarray) -> float:
    if audio.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))


# ── Utility shared across plugins ─────────────────────────────────────────────

def match_lengths(
    a: np.ndarray,
    b: np.ndarray,
    mode: str = "pad",
    hop_length: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    """Make both arrays the same number of frames.

    "pad" (default) zero-pads the shorter one. Cheap and lossless, but the morph
    then spends the difference in silence: a one-second A against a three-second
    B gives two seconds where there is nothing left of A to morph.

    "stretch" time-stretches the shorter one to fit, through the phase vocoder,
    so its whole gesture lines up against B's for the entire duration. Pitch is
    preserved; heavy ratios sound processed, as any time-stretch does.
    """
    len_a, len_b = len(a), len(b)
    if len_a == len_b:
        return a, b
    target = max(len_a, len_b)
    channels = a.shape[1] if a.ndim == 2 else 1

    def _pad(arr: np.ndarray) -> np.ndarray:
        padding = np.zeros((target - len(arr), channels), dtype=arr.dtype)
        return np.concatenate([arr, padding], axis=0)

    def _stretch(arr: np.ndarray) -> np.ndarray:
        src = np.linspace(0.0, len(arr) - 1.0, target)
        is_2d = arr.ndim == 2
        cols = [
            _stretch_to_time_map(arr[:, ch] if is_2d else arr, src, hop_length)
            for ch in range(channels)
        ]
        return np.stack(cols, axis=1) if is_2d else cols[0]

    fit = _stretch if mode == "stretch" else _pad
    if len_a < target:
        return fit(a), b
    return a, fit(b)


def dtw_align(
    a: np.ndarray,
    b: np.ndarray,
    sr: int,
    hop_length: int = 512,
    n_mfcc: int = 20,
    mode: str = "stretch",
) -> tuple[np.ndarray, np.ndarray]:
    """Return time-warped copies of A and B aligned to a common DTW timeline.

    Uses MFCC features for the alignment cost matrix, then time-warps both
    signals so that phonetically/spectrally similar moments line up.

    "stretch" (default) warps through a phase vocoder, which preserves pitch.
    "resample" reads the samples straight off the warp path — that is literally
    varispeed, so wherever the path departs from the diagonal the pitch slides
    with it. It is kept only for comparison.

    If librosa or scipy are unavailable, falls back to returning the originals.
    """
    try:
        import librosa
        from scipy.interpolate import interp1d
    except ImportError:
        return a, b

    is_2d = a.ndim == 2
    channels = a.shape[1] if is_2d else 1
    n_out = max(len(a), len(b))

    def _mono(x: np.ndarray) -> np.ndarray:
        return (x.mean(axis=1) if x.ndim == 2 else x).astype(np.float32)

    mfcc_a = librosa.feature.mfcc(y=_mono(a), sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)
    mfcc_b = librosa.feature.mfcc(y=_mono(b), sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)

    try:
        _, wp = librosa.sequence.dtw(X=mfcc_a, Y=mfcc_b)
        wp = wp[::-1]  # librosa returns path end→start; reverse to start→end
    except Exception:
        return a, b

    path_len = len(wp)
    if path_len < 2:
        return a, b

    path_idx = np.arange(path_len, dtype=np.float64)
    centers_a = wp[:, 0].astype(np.float64) * hop_length + hop_length * 0.5
    centers_b = wp[:, 1].astype(np.float64) * hop_length + hop_length * 0.5

    out_idx = np.linspace(0, path_len - 1, n_out)
    src_a = np.clip(np.interp(out_idx, path_idx, centers_a), 0, len(a) - 1)
    src_b = np.clip(np.interp(out_idx, path_idx, centers_b), 0, len(b) - 1)

    def _resample(signal: np.ndarray, src: np.ndarray) -> np.ndarray:
        x = np.arange(len(signal), dtype=np.float64)
        f = interp1d(x, signal.astype(np.float64),
                     bounds_error=False, fill_value=(float(signal[0]), float(signal[-1])))
        return f(src).astype(np.float32)

    _warp = _resample if mode == "resample" else (
        lambda signal, src: _stretch_to_time_map(signal, src, hop_length)
    )

    cols_a = [_warp(a[:, ch] if is_2d else a, src_a) for ch in range(channels)]
    cols_b = [_warp(b[:, ch] if is_2d else b, src_b) for ch in range(channels)]

    if is_2d:
        return np.stack(cols_a, axis=1), np.stack(cols_b, axis=1)
    return cols_a[0], cols_b[0]


def _stretch_to_time_map(
    signal: np.ndarray,
    src: np.ndarray,
    hop_length: int,
) -> np.ndarray:
    """Time-warp `signal` onto the sample map `src` without moving its pitch.

    `src[i]` is the source sample that output sample i should come from. Reading
    those samples directly is varispeed; instead this resamples the *STFT frames*
    along that map and re-integrates phase at each bin's own rate, so partials
    keep their frequencies however far the map departs from the diagonal.
    """
    import librosa

    n_out = len(src)
    n_fft = hop_length * 4
    sig = signal.astype(np.float32)

    if len(sig) < n_fft:
        # Too short for a meaningful STFT; the naive read is all that is left.
        idx = np.clip(src, 0, len(sig) - 1)
        lo = np.floor(idx).astype(np.int64)
        hi = np.minimum(lo + 1, len(sig) - 1)
        frac = idx - lo
        return (sig[lo] * (1.0 - frac) + sig[hi] * frac).astype(np.float32)

    D = librosa.stft(sig, n_fft=n_fft, hop_length=hop_length)
    n_frames = D.shape[1]

    # Source frame index for each output frame, from the per-sample map.
    out_frames = np.arange(0, n_out, hop_length, dtype=np.float64)
    time_map = np.interp(out_frames, np.arange(n_out, dtype=np.float64), src)
    time_map = np.clip(time_map / hop_length, 0.0, n_frames - 1.0)

    warped = _phase_vocoder(D, time_map, hop_length, n_fft)
    out = librosa.istft(warped, hop_length=hop_length, n_fft=n_fft, length=n_out)
    return out.astype(np.float32)


def pitch_shift_varying(
    signal: np.ndarray,
    ratio: np.ndarray,
    hop_length: int = 512,
) -> np.ndarray:
    """Pitch-shift a mono signal by a per-sample frequency `ratio`, keeping its length.

    librosa.effects.pitch_shift only takes a constant number of semitones, which
    is no use when the target pitch is itself a moving contour.

    Works the standard way round: playing the signal back at speed `ratio` gives
    the right pitch but the wrong duration, so the source is first time-stretched
    by the inverse amount — through the phase vocoder, which changes duration
    without touching pitch — and the varispeed read then lands back on the
    original timing.
    """
    n = len(signal)
    ratio = np.asarray(ratio, dtype=np.float64)
    if n < 2 or np.allclose(ratio, 1.0):
        return signal.astype(np.float32).copy()

    # Where a varispeed read would land after each output sample.
    read = np.concatenate([[0.0], np.cumsum(ratio)[:-1]])
    n_stretched = int(np.ceil(read[-1])) + 2
    if n_stretched < 2:
        return signal.astype(np.float32).copy()

    # Inverse map: which source sample each stretched sample should come from.
    src = np.interp(
        np.arange(n_stretched, dtype=np.float64), read, np.arange(n, dtype=np.float64)
    )
    stretched = _stretch_to_time_map(signal, src, hop_length)

    idx = np.clip(read, 0.0, len(stretched) - 1.0)
    lo = np.floor(idx).astype(np.int64)
    hi = np.minimum(lo + 1, len(stretched) - 1)
    frac = idx - lo
    return (stretched[lo] * (1.0 - frac) + stretched[hi] * frac).astype(np.float32)


def _phase_vocoder(
    D: np.ndarray,
    time_map: np.ndarray,
    hop_length: int,
    n_fft: int,
) -> np.ndarray:
    """Phase vocoder over an arbitrary time map.

    librosa.phase_vocoder only takes a constant rate; DTW produces a rate that
    changes frame by frame, so the frame stepping is driven by `time_map` here.
    """
    n_bins = D.shape[0]
    # Phase a bin is expected to advance over one hop if its frequency sits
    # exactly on the bin centre.
    expected = hop_length * 2.0 * np.pi * np.arange(n_bins) / n_fft

    padded = np.pad(D, ((0, 0), (0, 2)), mode="constant")
    mag = np.abs(padded)
    ang = np.angle(padded)

    out = np.zeros((n_bins, len(time_map)), dtype=np.complex64)
    phase = ang[:, 0].copy()

    for i, step in enumerate(time_map):
        k = int(step)
        frac = step - k
        out[:, i] = ((1.0 - frac) * mag[:, k] + frac * mag[:, k + 1]) * np.exp(1j * phase)

        # Deviation from the expected advance = the bin's true instantaneous
        # frequency; accumulate that so partials stay put under any stretch.
        delta = ang[:, k + 1] - ang[:, k] - expected
        delta -= 2.0 * np.pi * np.round(delta / (2.0 * np.pi))
        phase = phase + expected + delta

    return out
