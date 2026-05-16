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


# ── Utility shared across plugins ─────────────────────────────────────────────

def match_lengths(
    a: np.ndarray,
    b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Zero-pad the shorter array so both have the same number of frames."""
    len_a, len_b = len(a), len(b)
    if len_a == len_b:
        return a, b
    target = max(len_a, len_b)
    channels = a.shape[1] if a.ndim == 2 else 1

    def _pad(arr: np.ndarray, target_len: int) -> np.ndarray:
        pad_frames = target_len - len(arr)
        padding = np.zeros((pad_frames, channels), dtype=arr.dtype)
        return np.concatenate([arr, padding], axis=0)

    if len_a < target:
        a = _pad(a, target)
    else:
        b = _pad(b, target)
    return a, b


def dtw_align(
    a: np.ndarray,
    b: np.ndarray,
    sr: int,
    hop_length: int = 512,
    n_mfcc: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """Return time-warped copies of A and B aligned to a common DTW timeline.

    Uses MFCC features for the alignment cost matrix, then resamples both
    signals so that phonetically/spectrally similar moments line up.
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

    def _warp(signal: np.ndarray, src: np.ndarray) -> np.ndarray:
        x = np.arange(len(signal), dtype=np.float64)
        f = interp1d(x, signal.astype(np.float64),
                     bounds_error=False, fill_value=(float(signal[0]), float(signal[-1])))
        return f(src).astype(np.float32)

    cols_a = [_warp(a[:, ch] if is_2d else a, src_a) for ch in range(channels)]
    cols_b = [_warp(b[:, ch] if is_2d else b, src_b) for ch in range(channels)]

    if is_2d:
        return np.stack(cols_a, axis=1), np.stack(cols_b, axis=1)
    return cols_a[0], cols_b[0]
