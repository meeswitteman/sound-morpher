from __future__ import annotations

from typing import Any

import numpy as np
from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal

from plugins.base import MorphPlugin, dtw_align


class _Signals(QObject):
    progress = Signal(int)        # 0–100
    finished = Signal(list)       # list[np.ndarray]
    error = Signal(str)
    cancelled = Signal()


class _Worker(QRunnable):
    def __init__(
        self,
        plugin: MorphPlugin,
        audio_a: np.ndarray,
        audio_b: np.ndarray,
        steps: int,
        sample_rate: int,
        params: dict[str, Any],
        dtw: bool = False,
    ) -> None:
        super().__init__()
        self.signals = _Signals()
        self._plugin = plugin
        self._audio_a = audio_a
        self._audio_b = audio_b
        self._steps = steps
        self._sample_rate = sample_rate
        self._params = params
        self._dtw = dtw
        self.setAutoDelete(True)

    def run(self) -> None:
        try:
            self.signals.progress.emit(0)
            a, b = self._audio_a, self._audio_b
            if self._dtw:
                a, b = dtw_align(a, b, self._sample_rate)
            result = self._plugin.morph(
                a,
                b,
                self._steps,
                self._sample_rate,
                **self._params,
            )
            if len(result) != self._steps:
                raise ValueError(
                    f"Plugin '{self._plugin.name}' returned {len(result)} steps, "
                    f"expected {self._steps}"
                )
            self.signals.progress.emit(100)
            self.signals.finished.emit(result)
        except Exception as exc:
            self.signals.error.emit(str(exc))


class MorphEngine(QObject):
    """Runs morph computation on a QThreadPool worker thread."""

    progress  = Signal(int)
    finished  = Signal(list)   # list[np.ndarray]
    error     = Signal(str)
    cancelled = Signal()

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._pool = QThreadPool.globalInstance()
        self._active = False
        self._generation = 0   # incremented each compute() and cancel()

    @property
    def is_running(self) -> bool:
        return self._active

    def compute(
        self,
        plugin: MorphPlugin,
        audio_a: np.ndarray,
        audio_b: np.ndarray,
        steps: int,
        sample_rate: int,
        params: dict[str, Any] | None = None,
        dtw: bool = False,
    ) -> None:
        """Start async morph computation. Emits finished() or error() when done."""
        if self._active:
            return

        self._active = True
        self._generation += 1
        gen = self._generation

        worker = _Worker(plugin, audio_a, audio_b, steps, sample_rate, params or {}, dtw=dtw)
        worker.signals.progress.connect(self.progress)
        worker.signals.finished.connect(lambda result, g=gen: self._on_finished(result, g))
        worker.signals.error.connect(lambda msg, g=gen: self._on_error(msg, g))
        self._pool.start(worker)

    def cancel(self) -> None:
        """Immediately free the engine for new work; stale result is discarded on arrival."""
        self._active = False
        self._generation += 1   # invalidate the running worker's generation

    def _on_finished(self, result: list, gen: int) -> None:
        if gen != self._generation:
            # Stale result from a cancelled or superseded computation
            self.cancelled.emit()
            return
        self._active = False
        self.finished.emit(result)

    def _on_error(self, msg: str, gen: int) -> None:
        if gen != self._generation:
            return
        self._active = False
        self.error.emit(msg)
