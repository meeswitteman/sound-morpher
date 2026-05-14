from __future__ import annotations

import numpy as np
from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal, Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QHBoxLayout,
    QScrollArea,
    QSizePolicy,
    QWidget,
)

from app.widgets.spectrogram import compute_spectrogram_pixmap
from app.widgets.step_tile import TILE_H, TILE_W, StepTile


# ── Async spectrogram worker ───────────────────────────────────────────────────

class _SpectroSignals(QObject):
    tile_ready = Signal(int, QPixmap)   # (step_index, pixmap)
    all_done   = Signal()


class _SpectroWorker(QRunnable):
    def __init__(
        self,
        steps: list[np.ndarray],
        sample_rate: int,
        tile_w: int,
        tile_h: int,
    ) -> None:
        super().__init__()
        self.signals    = _SpectroSignals()
        self._steps     = steps
        self._sr        = sample_rate
        self._tile_w    = tile_w
        self._tile_h    = tile_h
        self._cancelled = False
        # setAutoDelete(False): we manage the worker lifetime ourselves so that
        # the _SpectroSignals QObject is not GC'd before queued signals arrive.
        self.setAutoDelete(False)

    def cancel(self) -> None:
        self._cancelled = True

    def run(self) -> None:
        spec_w = self._tile_w - 16
        spec_h = self._tile_h - 26 - 22
        for i, audio in enumerate(self._steps):
            if self._cancelled:
                break
            pixmap = compute_spectrogram_pixmap(audio, self._sr, spec_w, spec_h)
            self.signals.tile_ready.emit(i, pixmap)
        self.signals.all_done.emit()


# ── StepGrid ──────────────────────────────────────────────────────────────────

class StepGrid(QWidget):
    """Horizontally scrollable row of StepTile widgets."""

    step_clicked = Signal(int)   # step_index (0-based)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._tiles: list[StepTile] = []
        self._active_idx: int = -1
        self._worker: _SpectroWorker | None = None
        # Generation counter: incremented each time we start a new worker.
        # Allows _on_tile_ready to drop signals from previous (cancelled) workers.
        self._worker_gen: int = 0
        # Keep cancelled workers alive (Python reference) until they emit all_done,
        # preventing the _SpectroSignals QObject from being GC'd mid-flight.
        self._retiring: list[_SpectroWorker] = []

        self._build()

    # ── Build ──────────────────────────────────────────────────────────

    def _build(self) -> None:
        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        self._scroll = QScrollArea()
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._scroll.setWidgetResizable(False)
        self._scroll.setFrameShape(self._scroll.Shape.NoFrame)

        self._container = QWidget()
        self._container.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self._row = QHBoxLayout(self._container)
        self._row.setContentsMargins(4, 4, 4, 4)
        self._row.setSpacing(6)
        self._row.addStretch()

        self._scroll.setWidget(self._container)
        outer.addWidget(self._scroll)

    # ── Public API ─────────────────────────────────────────────────────

    def load_steps(
        self,
        steps: list[np.ndarray],
        sample_rate: int,
    ) -> None:
        """Replace tiles with a new set of morph steps and start async spectrogram compute."""
        self._cancel_worker()
        self._clear_tiles()

        n = len(steps)
        for i, audio in enumerate(steps):
            tile = StepTile(
                step_index=i,
                audio=audio,
                sample_rate=sample_rate,
                is_first=(i == 0),
                is_last=(i == n - 1),
                parent=self._container,
            )
            tile.clicked.connect(self._on_tile_clicked)
            self._tiles.append(tile)
            self._row.insertWidget(self._row.count() - 1, tile)

        self._container.setFixedHeight(TILE_H + 12)
        self._scroll.setMinimumHeight(TILE_H + 16)
        self._container.adjustSize()

        if steps:
            self._start_spectrogram_worker(steps, sample_rate)

    def set_active(self, step_index: int) -> None:
        if self._active_idx >= 0 and self._active_idx < len(self._tiles):
            self._tiles[self._active_idx].set_active(False)
        self._active_idx = step_index
        if 0 <= step_index < len(self._tiles):
            tile = self._tiles[step_index]
            tile.set_active(True)
            self._scroll.ensureWidgetVisible(tile)

    def clear_active(self) -> None:
        self.set_active(-1)

    # ── Internal ───────────────────────────────────────────────────────

    def _clear_tiles(self) -> None:
        for tile in self._tiles:
            self._row.removeWidget(tile)
            tile.deleteLater()
        self._tiles.clear()
        self._active_idx = -1

    def _on_tile_clicked(self, idx: int) -> None:
        self.set_active(idx)
        self.step_clicked.emit(idx)

    def _cancel_worker(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
            # Move to retiring list so the Python object (and its _SpectroSignals)
            # stays alive until all_done fires.
            self._retiring.append(self._worker)
            self._worker = None
        # Bump generation: tile_ready from retiring workers will be ignored.
        self._worker_gen += 1

    def _start_spectrogram_worker(
        self,
        steps: list[np.ndarray],
        sample_rate: int,
    ) -> None:
        gen = self._worker_gen
        worker = _SpectroWorker(steps, sample_rate, TILE_W, TILE_H)
        self._worker = worker
        worker.signals.tile_ready.connect(
            lambda idx, pix, g=gen: self._on_tile_ready(idx, pix, g)
        )
        worker.signals.all_done.connect(
            lambda w=worker, g=gen: self._on_spectro_done(w, g)
        )
        QThreadPool.globalInstance().start(worker)

    def _on_tile_ready(self, idx: int, pixmap: QPixmap, gen: int) -> None:
        if gen != self._worker_gen:
            return  # stale signal from a cancelled worker — ignore
        if 0 <= idx < len(self._tiles):
            self._tiles[idx].set_spectrogram(pixmap)

    def _on_spectro_done(self, worker: _SpectroWorker, gen: int) -> None:
        # Release the worker reference (allows GC now that all signals are done)
        if worker in self._retiring:
            self._retiring.remove(worker)
        if self._worker is worker:
            self._worker = None
