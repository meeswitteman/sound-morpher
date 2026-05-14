from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QBrush, QColor, QLinearGradient, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget


class WaveformWidget(QWidget):
    """Renders an audio waveform as a filled min/max envelope using QPainter."""

    def __init__(self, color: str = "#00c8c8", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._color = QColor(color)
        self._audio: np.ndarray | None = None
        self._envelope: np.ndarray | None = None  # shape (N, 2): [min, max] per column
        self._last_width = 0
        self._trim_start: float | None = None
        self._trim_end: float | None = None

        self.setMinimumHeight(80)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent)

    def set_audio(self, audio: np.ndarray | None) -> None:
        self._audio = audio
        self._envelope = None
        self._trim_start: float | None = None
        self._trim_end: float | None = None
        self.update()

    def set_trim_region(self, start_frac: float | None, end_frac: float | None) -> None:
        """Overlay a trim selection. Fractions are in [0.0, 1.0] of audio length."""
        self._trim_start = start_frac
        self._trim_end = end_frac
        self.update()

    def resizeEvent(self, event) -> None:
        self._envelope = None
        super().resizeEvent(event)

    def _compute_envelope(self) -> np.ndarray:
        audio = self._audio
        mono = audio.mean(axis=1) if audio.ndim == 2 else audio.flatten()

        w = max(self.width(), 1)
        total = len(mono)
        chunk = max(total // w, 1)
        cols = total // chunk
        if cols == 0:
            return np.zeros((1, 2), dtype=np.float32)

        trimmed = mono[: cols * chunk].reshape(cols, chunk)
        env = np.stack([trimmed.min(axis=1), trimmed.max(axis=1)], axis=1)
        return env.astype(np.float32)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w, h = self.width(), self.height()
        cy = h / 2.0

        painter.fillRect(0, 0, w, h, QColor("#111111"))

        if self._audio is None or len(self._audio) == 0:
            painter.setPen(QColor("#303030"))
            painter.drawText(
                0, 0, w, h,
                Qt.AlignmentFlag.AlignCenter,
                "No audio loaded — drag a WAV here",
            )
            return

        if self._envelope is None or self._last_width != w:
            self._envelope = self._compute_envelope()
            self._last_width = w

        env = self._envelope
        n = len(env)
        if n == 0:
            return

        x_scale = w / n

        # Build filled polygon: top (maxima) left→right, bottom (minima) right→left
        path = QPainterPath()
        path.moveTo(0.0, cy - float(env[0, 1]) * cy)
        for i in range(1, n):
            path.lineTo(i * x_scale, cy - float(env[i, 1]) * cy)
        for i in range(n - 1, -1, -1):
            path.lineTo(i * x_scale, cy - float(env[i, 0]) * cy)
        path.closeSubpath()

        # Gradient fill: accent color at edge, darker in center
        grad = QLinearGradient(0, 0, 0, h)
        top = QColor(self._color)
        top.setAlpha(200)
        mid = QColor(self._color)
        mid.setAlpha(80)
        grad.setColorAt(0.0, top)
        grad.setColorAt(0.5, mid)
        grad.setColorAt(1.0, top)
        painter.fillPath(path, QBrush(grad))

        # Crisp top-edge stroke
        painter.setPen(QPen(self._color, 1.2))
        top_path = QPainterPath()
        top_path.moveTo(0.0, cy - float(env[0, 1]) * cy)
        for i in range(1, n):
            top_path.lineTo(i * x_scale, cy - float(env[i, 1]) * cy)
        painter.drawPath(top_path)

        # Centre line
        painter.setPen(QPen(QColor("#1e1e1e"), 1))
        painter.drawLine(0, int(cy), w, int(cy))

        # Trim region overlay
        if self._trim_start is not None and self._trim_end is not None:
            sx = int(self._trim_start * w)
            ex = int(self._trim_end * w)
            dim = QColor(0, 0, 0, 160)
            if sx > 0:
                painter.fillRect(0, 0, sx, h, dim)
            if ex < w:
                painter.fillRect(ex, 0, w - ex, h, dim)
            marker = QPen(QColor("#ffffff"), 1)
            marker.setStyle(Qt.PenStyle.DashLine)
            painter.setPen(marker)
            painter.drawLine(sx, 0, sx, h)
            painter.drawLine(ex, 0, ex, h)
