from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import soundfile as sf
from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal


class _Signals(QObject):
    progress = Signal(int)    # 0–100
    finished = Signal(str)    # output directory path
    error = Signal(str)


class _ExportWorker(QRunnable):
    def __init__(
        self,
        steps: list[np.ndarray],
        output_dir: Path,
        sample_rate: int,
        bit_depth: int,
    ) -> None:
        super().__init__()
        self.signals = _Signals()
        self._steps = steps
        self._output_dir = output_dir
        self._sample_rate = sample_rate
        self._bit_depth = bit_depth
        self.setAutoDelete(True)

    def run(self) -> None:
        total = len(self._steps)
        subtype = "PCM_16" if self._bit_depth <= 16 else "PCM_24"
        try:
            self._output_dir.mkdir(parents=True, exist_ok=True)
            for i, step in enumerate(self._steps):
                filename = self._output_dir / f"morph_step_{i + 1:02d}.wav"
                sf.write(str(filename), step, self._sample_rate, subtype=subtype)
                pct = int((i + 1) / total * 100)
                self.signals.progress.emit(pct)
        except Exception as exc:
            self.signals.error.emit(str(exc))
            return
        self.signals.finished.emit(str(self._output_dir))


class ExportEngine(QObject):
    """Writes morph steps as individual WAV files on a background thread."""

    progress = Signal(int)   # 0–100
    finished = Signal(str)   # output directory
    error = Signal(str)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._pool = QThreadPool.globalInstance()
        self._active = False

    @property
    def is_running(self) -> bool:
        return self._active

    def export(
        self,
        steps: list[np.ndarray],
        output_dir: str | Path,
        sample_rate: int,
        bit_depth: int,
    ) -> None:
        if self._active:
            return
        self._active = True
        worker = _ExportWorker(steps, Path(output_dir), sample_rate, bit_depth)
        worker.signals.progress.connect(self.progress)
        worker.signals.finished.connect(self._on_finished)
        worker.signals.error.connect(self._on_error)
        self._pool.start(worker)

    def _on_finished(self, path: str) -> None:
        self._active = False
        self.finished.emit(path)

    def _on_error(self, msg: str) -> None:
        self._active = False
        self.error.emit(msg)
