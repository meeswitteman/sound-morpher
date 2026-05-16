from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import (
    QBrush, QColor, QLinearGradient, QPainter, QPainterPath, QPen
)
from PySide6.QtWidgets import (
    QDialog, QHBoxLayout, QLabel, QPushButton, QSizePolicy,
    QVBoxLayout, QWidget,
)


# ── Canvas ─────────────────────────────────────────────────────────────────────

class _OverlapCanvas(QWidget):
    """Both waveforms drawn at the same timescale; B can be dragged left/right."""

    offset_changed = Signal(int)  # offset in samples (positive = B starts later)

    _COLOR_A = QColor("#00c8c8")
    _COLOR_B = QColor("#c87800")

    def __init__(
        self,
        mono_a: np.ndarray,
        mono_b: np.ndarray,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._mono_a = mono_a
        self._mono_b = mono_b
        self._offset = 0  # samples; positive = B starts later than A
        self._drag_x: float | None = None
        self._drag_off0 = 0

        self.setMinimumHeight(130)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setCursor(Qt.CursorShape.SizeHorCursor)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent)

    # ── Public ─────────────────────────────────────────────────────────────────

    @property
    def offset(self) -> int:
        return self._offset

    @offset.setter
    def offset(self, value: int) -> None:
        self._offset = value
        self.update()

    # ── Geometry helpers ───────────────────────────────────────────────────────

    def _total_samples(self) -> int:
        a_start = max(0, -self._offset)
        b_start = max(0, self._offset)
        return max(a_start + len(self._mono_a), b_start + len(self._mono_b))

    def _px_per_sample(self) -> float:
        total = self._total_samples()
        return self.width() / max(total, 1)

    def _a_start_px(self) -> float:
        return max(0, -self._offset) * self._px_per_sample()

    def _b_start_px(self) -> float:
        return max(0, self._offset) * self._px_per_sample()

    # ── Envelope ───────────────────────────────────────────────────────────────

    @staticmethod
    def _envelope(mono: np.ndarray, n_cols: int) -> np.ndarray:
        n = len(mono)
        cols = max(n_cols, 1)
        chunk = max(n // cols, 1)
        actual = n // chunk
        actual = min(actual, cols)
        if actual == 0:
            return np.zeros((1, 2), dtype=np.float32)
        r = mono[: actual * chunk].reshape(actual, chunk)
        return np.stack([r.min(axis=1), r.max(axis=1)], axis=1).astype(np.float32)

    # ── Drawing ────────────────────────────────────────────────────────────────

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        cy = h / 2.0
        scale = self._px_per_sample()

        painter.fillRect(0, 0, w, h, QColor("#111111"))

        # Draw each waveform
        for mono, color, start_px in (
            (self._mono_a, self._COLOR_A, self._a_start_px()),
            (self._mono_b, self._COLOR_B, self._b_start_px()),
        ):
            px_width = max(int(len(mono) * scale), 1)
            env = self._envelope(mono, px_width)
            n = len(env)
            if n == 0:
                continue
            x_step = px_width / n

            path = QPainterPath()
            path.moveTo(start_px, cy - float(env[0, 1]) * cy)
            for i in range(1, n):
                path.lineTo(start_px + i * x_step, cy - float(env[i, 1]) * cy)
            for i in range(n - 1, -1, -1):
                path.lineTo(start_px + i * x_step, cy - float(env[i, 0]) * cy)
            path.closeSubpath()

            grad = QLinearGradient(0, 0, 0, h)
            top_c = QColor(color)
            top_c.setAlpha(150)
            mid_c = QColor(color)
            mid_c.setAlpha(55)
            grad.setColorAt(0.0, top_c)
            grad.setColorAt(0.5, mid_c)
            grad.setColorAt(1.0, top_c)
            painter.fillPath(path, QBrush(grad))

            painter.setPen(QPen(color, 1.1))
            edge = QPainterPath()
            edge.moveTo(start_px, cy - float(env[0, 1]) * cy)
            for i in range(1, n):
                edge.lineTo(start_px + i * x_step, cy - float(env[i, 1]) * cy)
            painter.drawPath(edge)

        # Centre line
        painter.setPen(QPen(QColor("#222222"), 1))
        painter.drawLine(0, int(cy), w, int(cy))

        # B-start marker
        b_px = int(self._b_start_px())
        if 0 <= b_px < w:
            pen = QPen(self._COLOR_B, 1)
            pen.setStyle(Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.drawLine(b_px, 0, b_px, h)

        # A-start marker (only visible when A is shifted right)
        a_px = int(self._a_start_px())
        if a_px > 0:
            pen = QPen(self._COLOR_A, 1)
            pen.setStyle(Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.drawLine(a_px, 0, a_px, h)

        # Legend labels
        painter.setFont(self.font())
        painter.setPen(self._COLOR_A)
        painter.drawText(4, 14, "A")
        painter.setPen(self._COLOR_B)
        painter.drawText(4, 28, "B ← drag →")

    # ── Interaction ────────────────────────────────────────────────────────────

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_x = event.position().x()
            self._drag_off0 = self._offset

    def mouseMoveEvent(self, event) -> None:
        if self._drag_x is None:
            return
        dx = event.position().x() - self._drag_x
        total = max(self._total_samples(), 1)
        delta = int(dx * total / max(self.width(), 1))
        self._offset = self._drag_off0 + delta
        self.offset_changed.emit(self._offset)
        self.update()

    def mouseReleaseEvent(self, _event) -> None:
        self._drag_x = None

    def resizeEvent(self, event) -> None:
        self.update()
        super().resizeEvent(event)


# ── Dialog ─────────────────────────────────────────────────────────────────────

class OverlapPanel(QDialog):
    """Visual overlap tool: shows A and B at the same timescale; drag B to align.

    After accept(), read `shifted_a` and `shifted_b` (one may have silence
    prepended; the other is the original array).
    """

    def __init__(
        self,
        audio_a: np.ndarray,
        audio_b: np.ndarray,
        sample_rate: int,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._audio_a = audio_a
        self._audio_b = audio_b
        self._sr = sample_rate

        self.shifted_a: np.ndarray = audio_a
        self.shifted_b: np.ndarray = audio_b

        self.setWindowTitle("Overlap View — Align B relative to A")
        self.setModal(True)
        self.setMinimumWidth(620)
        self.setMinimumHeight(280)

        self._build()

    # ── Build ──────────────────────────────────────────────────────────────────

    def _build(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        # Instruction
        info = QLabel(
            "Drag the orange waveform (B) left or right to set its start offset "
            "relative to A (cyan).  Click Apply to prepend silence to the shifted clip."
        )
        info.setStyleSheet("color: #707070; font-size: 11px;")
        info.setWordWrap(True)
        layout.addWidget(info)

        # Duration info row
        self._lbl_info = QLabel()
        self._lbl_info.setStyleSheet("color: #909090; font-size: 11px;")
        layout.addWidget(self._lbl_info)

        # Canvas
        mono_a = (
            self._audio_a.mean(axis=1)
            if self._audio_a.ndim == 2
            else self._audio_a.ravel()
        ).astype(np.float32)
        mono_b = (
            self._audio_b.mean(axis=1)
            if self._audio_b.ndim == 2
            else self._audio_b.ravel()
        ).astype(np.float32)

        self._canvas = _OverlapCanvas(mono_a, mono_b, parent=self)
        self._canvas.offset_changed.connect(self._on_offset_changed)
        layout.addWidget(self._canvas, stretch=1)

        # Offset label
        self._lbl_offset = QLabel()
        self._lbl_offset.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._lbl_offset.setStyleSheet("color: #c0c0c0; font-size: 12px;")
        layout.addWidget(self._lbl_offset)

        # Buttons
        btn_row = QHBoxLayout()
        btn_reset = QPushButton("Reset")
        btn_reset.setToolTip("Clear offset — B starts at the same time as A")
        btn_reset.clicked.connect(self._on_reset)

        self._btn_apply = QPushButton("Apply")
        self._btn_apply.setDefault(True)
        self._btn_apply.clicked.connect(self._on_apply)

        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(self.reject)

        btn_row.addWidget(btn_reset)
        btn_row.addStretch()
        btn_row.addWidget(btn_cancel)
        btn_row.addWidget(self._btn_apply)
        layout.addLayout(btn_row)

        self._update_labels()

    # ── Slots ──────────────────────────────────────────────────────────────────

    def _on_offset_changed(self, _offset: int) -> None:
        self._update_labels()

    def _on_reset(self) -> None:
        self._canvas.offset = 0
        self._update_labels()

    def _on_apply(self) -> None:
        offset = self._canvas.offset
        ch = self._audio_a.shape[1] if self._audio_a.ndim == 2 else 1

        def _silence(n_samples: int) -> np.ndarray:
            pad = np.zeros((n_samples, ch), dtype=np.float32)
            return pad if ch > 1 else pad.ravel()

        if offset > 0:
            # B starts later: prepend silence to B
            self.shifted_a = self._audio_a
            self.shifted_b = np.concatenate([_silence(offset), self._audio_b], axis=0)
        elif offset < 0:
            # A starts later: prepend silence to A
            self.shifted_a = np.concatenate([_silence(-offset), self._audio_a], axis=0)
            self.shifted_b = self._audio_b
        else:
            self.shifted_a = self._audio_a
            self.shifted_b = self._audio_b

        self.accept()

    # ── Labels ─────────────────────────────────────────────────────────────────

    def _update_labels(self) -> None:
        sr = self._sr
        len_a = len(self._audio_a)
        len_b = len(self._audio_b)
        offset = self._canvas.offset
        ms = offset * 1000 / sr

        self._lbl_info.setText(
            f"A: {len_a / sr:.2f}s  |  B: {len_b / sr:.2f}s"
        )

        sign = "+" if offset >= 0 else ""
        self._lbl_offset.setText(
            f"B offset: {sign}{ms:.0f} ms  ({sign}{offset} samples)"
        )
