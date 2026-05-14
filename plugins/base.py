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
