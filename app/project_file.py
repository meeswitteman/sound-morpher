from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import numpy as np
import soundfile as sf

from app.project_state import ProjectState

_FORMAT_VERSION = "1.0"
_SMORPH_FILTER  = "Sound Morpher Project (*.smorph);;All files (*)"


class ProjectFileError(Exception):
    pass


class ProjectFile:
    """Read and write .smorph project archives (ZIP + JSON + WAV assets)."""

    # ── Save ───────────────────────────────────────────────────────────

    @staticmethod
    def save(path: str | Path, state: ProjectState) -> None:
        path = Path(path)
        if path.suffix.lower() != ".smorph":
            path = path.with_suffix(".smorph")

        meta = {
            "version":          _FORMAT_VERSION,
            "sample_rate":      state.sample_rate,
            "bit_depth":        state.bit_depth,
            "name_a":           state.name_a,
            "name_b":           state.name_b,
            "steps":            state.steps,
            "algorithm":        state.algorithm,
            "algorithm_params": state.algorithm_params,
            "bpm":              state.bpm,
            "beats_per_step":   state.beats_per_step,
            "loop":             state.loop,
            "step_count":       len(state.morph_steps),
        }

        try:
            with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                zf.writestr("project.json", json.dumps(meta, indent=2))

                if state.audio_a is not None:
                    zf.writestr(
                        "audio/source_a.wav",
                        _to_wav_bytes(state.audio_a, state.sample_rate, state.bit_depth),
                    )
                if state.audio_b is not None:
                    zf.writestr(
                        "audio/source_b.wav",
                        _to_wav_bytes(state.audio_b, state.sample_rate, state.bit_depth),
                    )
                for i, step in enumerate(state.morph_steps):
                    zf.writestr(
                        f"audio/step_{i + 1:02d}.wav",
                        _to_wav_bytes(step, state.sample_rate, state.bit_depth),
                    )
        except OSError as exc:
            raise ProjectFileError(f"Cannot write project file: {exc}") from exc

        state.file_path = str(path)

    # ── Load ───────────────────────────────────────────────────────────

    @staticmethod
    def load(path: str | Path) -> ProjectState:
        path = Path(path)
        if not path.exists():
            raise ProjectFileError(f"File not found: {path}")

        try:
            with zipfile.ZipFile(path, "r") as zf:
                names = set(zf.namelist())

                try:
                    meta = json.loads(zf.read("project.json").decode())
                except (KeyError, json.JSONDecodeError) as exc:
                    raise ProjectFileError(f"Invalid project file: {exc}") from exc

                state = ProjectState(
                    sample_rate=int(meta.get("sample_rate", 44100)),
                    bit_depth=int(meta.get("bit_depth", 16)),
                    name_a=meta.get("name_a", ""),
                    name_b=meta.get("name_b", ""),
                    steps=int(meta.get("steps", 8)),
                    algorithm=meta.get("algorithm", "Crossfade"),
                    algorithm_params=dict(meta.get("algorithm_params") or {}),
                    bpm=int(meta.get("bpm", 120)),
                    beats_per_step=int(meta.get("beats_per_step", 4)),
                    loop=bool(meta.get("loop", False)),
                    file_path=str(path),
                )

                if "audio/source_a.wav" in names:
                    state.audio_a = _from_wav_bytes(zf.read("audio/source_a.wav"))
                if "audio/source_b.wav" in names:
                    state.audio_b = _from_wav_bytes(zf.read("audio/source_b.wav"))

                step_count = int(meta.get("step_count", 0))
                steps: list[np.ndarray] = []
                for i in range(1, step_count + 1):
                    name = f"audio/step_{i:02d}.wav"
                    if name in names:
                        steps.append(_from_wav_bytes(zf.read(name)))
                state.morph_steps = steps

        except zipfile.BadZipFile as exc:
            raise ProjectFileError(f"Not a valid .smorph file: {exc}") from exc

        return state


# ── Helpers ────────────────────────────────────────────────────────────────────

def _to_wav_bytes(audio: np.ndarray, sr: int, bit_depth: int) -> bytes:
    subtype = "PCM_16" if bit_depth <= 16 else "PCM_24"
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV", subtype=subtype)
    return buf.getvalue()


def _from_wav_bytes(data: bytes) -> np.ndarray:
    buf = io.BytesIO(data)
    audio, _ = sf.read(buf, dtype="float32", always_2d=True)
    return audio


SMORPH_FILTER = _SMORPH_FILTER
