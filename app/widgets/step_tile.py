from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, QRect, QSize, Signal
from PySide6.QtGui import (
    QColor, QFont, QPainter, QPainterPath, QPen, QPixmap,
)
from PySide6.QtWidgets import QSizePolicy, QWidget

TILE_W = 158
TILE_H = 195

_COLOR_ACTIVE   = QColor("#00c8c8")
_COLOR_BORDER   = QColor("#2a2a2a")
_COLOR_BG       = QColor("#181818")
_COLOR_BG_HOVER = QColor("#1e2020")
_COLOR_TEXT     = QColor("#c0c0c0")
_COLOR_SUBTEXT  = QColor("#606060")
_COLOR_BADGE_A  = QColor("#00c8c8")
_COLOR_BADGE_B  = QColor("#c87800")
_COLOR_LOADING  = QColor("#303030")

_SPEC_MARGIN  = 8    # px from tile edge to spectrogram
_SPEC_TOP     = 26   # px from top to spectrogram start
_BAR_H        = 3    # active indicator bar height


class StepTile(QWidget):
    """A single morph-step tile: step number, spectrogram thumbnail, duration."""

    clicked = Signal(int)  # step_index (0-based)

    def __init__(
        self,
        step_index: int,
        audio: np.ndarray,
        sample_rate: int,
        is_first: bool = False,
        is_last: bool = False,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._index = step_index
        self._duration = len(audio) / sample_rate if sample_rate > 0 else 0.0
        self._is_first = is_first
        self._is_last = is_last
        self._active = False
        self._hovered = False
        self._pixmap: QPixmap | None = None

        self.setFixedSize(TILE_W, TILE_H)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.setAttribute(Qt.WidgetAttribute.WA_Hover)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    # ── Public API ─────────────────────────────────────────────────────

    def set_spectrogram(self, pixmap: QPixmap) -> None:
        self._pixmap = pixmap
        self.update()

    def set_active(self, active: bool) -> None:
        if self._active != active:
            self._active = active
            self.update()

    # ── Events ─────────────────────────────────────────────────────────

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self._index)

    def enterEvent(self, event) -> None:
        self._hovered = True
        self.update()

    def leaveEvent(self, event) -> None:
        self._hovered = False
        self.update()

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()

        # ── Background ──
        bg = _COLOR_BG_HOVER if self._hovered else _COLOR_BG
        path = QPainterPath()
        path.addRoundedRect(1, 1, w - 2, h - 2, 5, 5)
        p.fillPath(path, bg)

        # ── Border ──
        border_color = _COLOR_ACTIVE if self._active else (
            QColor("#383838") if self._hovered else _COLOR_BORDER
        )
        pen = QPen(border_color, 1.5 if self._active else 1.0)
        p.setPen(pen)
        p.drawPath(path)

        # ── Active top bar ──
        if self._active:
            p.fillRect(QRect(2, 1, w - 4, _BAR_H), _COLOR_ACTIVE)

        # ── Step number ──
        p.setPen(_COLOR_ACTIVE if self._active else _COLOR_TEXT)
        font = QFont()
        font.setPointSize(9)
        font.setWeight(QFont.Weight.DemiBold)
        p.setFont(font)
        p.drawText(QRect(8, 5, 60, 20), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                   str(self._index + 1))

        # ── A / B badge ──
        if self._is_first or self._is_last:
            badge_label = "A" if self._is_first else "B"
            badge_color = _COLOR_BADGE_A if self._is_first else _COLOR_BADGE_B
            badge_rect = QRect(w - 26, 4, 20, 18)
            badge_path = QPainterPath()
            badge_path.addRoundedRect(badge_rect.x(), badge_rect.y(),
                                      badge_rect.width(), badge_rect.height(), 3, 3)
            p.fillPath(badge_path, badge_color)
            p.setPen(QColor("#0a0a0a"))
            font_b = QFont()
            font_b.setPointSize(8)
            font_b.setWeight(QFont.Weight.Bold)
            p.setFont(font_b)
            p.drawText(badge_rect, Qt.AlignmentFlag.AlignCenter, badge_label)

        # ── Spectrogram area ──
        spec_rect = QRect(
            _SPEC_MARGIN,
            _SPEC_TOP,
            w - _SPEC_MARGIN * 2,
            h - _SPEC_TOP - 22,
        )
        if self._pixmap is not None:
            p.drawPixmap(spec_rect, self._pixmap)
        else:
            # Loading placeholder
            p.fillRect(spec_rect, _COLOR_LOADING)
            p.setPen(_COLOR_SUBTEXT)
            font_s = QFont()
            font_s.setPointSize(8)
            p.setFont(font_s)
            p.drawText(spec_rect, Qt.AlignmentFlag.AlignCenter, "computing…")

        # ── Duration label ──
        p.setPen(_COLOR_SUBTEXT)
        font_d = QFont()
        font_d.setPointSize(8)
        p.setFont(font_d)
        dur_rect = QRect(8, h - 18, w - 16, 14)
        p.drawText(dur_rect,
                   Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                   f"{self._duration:.2f} s")
