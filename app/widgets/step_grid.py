from __future__ import annotations

import numpy as np
from PySide6.QtCore import Signal, Qt
from PySide6.QtWidgets import (
    QHBoxLayout,
    QScrollArea,
    QSizePolicy,
    QWidget,
)

from app.widgets.spectrogram import compute_spectrogram_pixmap
from app.widgets.step_tile import TILE_H, TILE_W, StepTile


class StepGrid(QWidget):
    """Horizontally scrollable row of StepTile widgets."""

    step_clicked = Signal(int)   # step_index (0-based)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._tiles: list[StepTile] = []
        self._active_idx: int = -1
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
        """Replace tiles with a new set of morph steps.

        Spectrograms are computed synchronously so every tile already has its
        image when it first appears on screen — no async timing issues.
        """
        self._clear_tiles()

        spec_w = TILE_W - 16
        spec_h = TILE_H - 26 - 22
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
            pixmap = compute_spectrogram_pixmap(audio, sample_rate, spec_w, spec_h)
            tile.set_spectrogram(pixmap)

            tile.clicked.connect(self._on_tile_clicked)
            self._tiles.append(tile)
            self._row.insertWidget(self._row.count() - 1, tile)

        # adjustSize() returns wrong width when the container is already visible,
        # so compute the width explicitly from tile count and layout geometry.
        m = self._row.contentsMargins()
        container_w = m.left() + n * TILE_W + max(0, n - 1) * self._row.spacing() + m.right()
        self._container.setFixedSize(container_w, TILE_H + 12)
        self._scroll.setMinimumHeight(TILE_H + 16)

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
            tile.hide()
            tile.setParent(None)
            tile.deleteLater()
        self._tiles.clear()
        self._active_idx = -1

    def _on_tile_clicked(self, idx: int) -> None:
        self.set_active(idx)
        self.step_clicked.emit(idx)
