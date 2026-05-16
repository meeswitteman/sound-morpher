from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ProjectState:
    """In-memory representation of the current session."""

    sample_rate: int = 44100
    bit_depth: int = 16

    audio_a: np.ndarray | None = None
    audio_b: np.ndarray | None = None
    name_a: str = ""
    name_b: str = ""

    steps: int = 8
    algorithm: str = "Crossfade"
    algorithm_params: dict = field(default_factory=dict)
    bpm: int = 120
    beats_per_step: int = 4
    loop_mode: str = "off"   # "off" | "loop" | "pingpong"
    reverse: bool = False

    morph_steps: list[np.ndarray] = field(default_factory=list)

    # Path to .smorph file on disk (None = unsaved)
    file_path: str | None = None

    @property
    def ready_to_morph(self) -> bool:
        return self.audio_a is not None and self.audio_b is not None

    @property
    def has_morph_steps(self) -> bool:
        return len(self.morph_steps) > 0

    @property
    def is_dirty(self) -> bool:
        """True when there are unsaved changes (basic heuristic)."""
        return self.ready_to_morph
