from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, QSize, Signal
from PySide6.QtGui import (
    QBrush, QColor, QLinearGradient, QPainter, QPainterPath, QPen
)
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from app.audio_engine import AudioEngine


class _WarpWidget(QWidget):
    """Waveform widget with click-and-drag time-warp interaction."""

    anchor_changed = Signal(float, float)   # (anchor_frac, drag_frac)

    def __init__(self, color: str = "#00c8c8", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._color = QColor(color)
        self._audio: np.ndarray | None = None
        self._envelope: np.ndarray | None = None
        self._last_width = 0
        self._anchor: float | None = None
        self._drag: float | None = None
        self._pressed = False

        self.setMinimumHeight(140)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent)
        self.setCursor(Qt.CursorShape.SizeHorCursor)
        self.setToolTip(
            "Click to set anchor point · "
            "drag left = compress left / stretch right · "
            "drag right = stretch left / compress right"
        )

    def set_audio(self, audio: np.ndarray) -> None:
        self._audio = audio
        self._envelope = None
        self.update()

    def reset(self) -> None:
        self._anchor = None
        self._drag = None
        self._pressed = False
        self.update()

    def resizeEvent(self, event) -> None:
        self._envelope = None
        super().resizeEvent(event)

    def mousePressEvent(self, event) -> None:
        if self._audio is None or event.button() != Qt.MouseButton.LeftButton:
            return
        frac = max(0.0, min(1.0, event.position().x() / self.width()))
        self._anchor = frac
        self._drag = frac
        self._pressed = True
        self.anchor_changed.emit(self._anchor, self._drag)
        self.update()

    def mouseMoveEvent(self, event) -> None:
        if not self._pressed or self._anchor is None:
            return
        frac = max(0.0, min(1.0, event.position().x() / self.width()))
        self._drag = frac
        self.anchor_changed.emit(self._anchor, self._drag)
        self.update()

    def mouseReleaseEvent(self, event) -> None:
        self._pressed = False

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
        return np.stack([trimmed.min(axis=1), trimmed.max(axis=1)], axis=1).astype(np.float32)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        cy = h / 2.0

        painter.fillRect(0, 0, w, h, QColor("#111111"))

        if self._audio is None or len(self._audio) == 0:
            painter.setPen(QColor("#303030"))
            painter.drawText(0, 0, w, h, Qt.AlignmentFlag.AlignCenter,
                             "Click and drag to time-warp")
            return

        if self._envelope is None or self._last_width != w:
            self._envelope = self._compute_envelope()
            self._last_width = w

        env = self._envelope
        n = len(env)
        if n == 0:
            return
        x_scale = w / n

        # Waveform fill
        path = QPainterPath()
        path.moveTo(0.0, cy - float(env[0, 1]) * cy)
        for i in range(1, n):
            path.lineTo(i * x_scale, cy - float(env[i, 1]) * cy)
        for i in range(n - 1, -1, -1):
            path.lineTo(i * x_scale, cy - float(env[i, 0]) * cy)
        path.closeSubpath()

        grad = QLinearGradient(0, 0, 0, h)
        top = QColor(self._color); top.setAlpha(200)
        mid = QColor(self._color); mid.setAlpha(80)
        grad.setColorAt(0.0, top)
        grad.setColorAt(0.5, mid)
        grad.setColorAt(1.0, top)
        painter.fillPath(path, QBrush(grad))

        painter.setPen(QPen(self._color, 1.2))
        top_path = QPainterPath()
        top_path.moveTo(0.0, cy - float(env[0, 1]) * cy)
        for i in range(1, n):
            top_path.lineTo(i * x_scale, cy - float(env[i, 1]) * cy)
        painter.drawPath(top_path)

        painter.setPen(QPen(QColor("#1e1e1e"), 1))
        painter.drawLine(0, int(cy), w, int(cy))

        if self._anchor is None or self._drag is None:
            return

        ax = int(self._anchor * w)
        dx = int(self._drag * w)

        # Region tints: red = compressed, blue = stretched
        if dx < ax:
            left_col  = QColor(220,  80,  80, 55)   # compress
            right_col = QColor( 80, 150, 220, 55)   # stretch
        elif dx > ax:
            left_col  = QColor( 80, 150, 220, 55)   # stretch
            right_col = QColor(220,  80,  80, 55)   # compress
        else:
            left_col = right_col = QColor(0, 0, 0, 0)

        painter.fillRect(0,  0, ax,     h, left_col)
        painter.fillRect(ax, 0, w - ax, h, right_col)

        # Anchor line — white dashed
        pen = QPen(QColor("#ffffff"), 1, Qt.PenStyle.DashLine)
        painter.setPen(pen)
        painter.drawLine(ax, 0, ax, h)

        # Drag line — yellow solid
        painter.setPen(QPen(QColor("#ffdd00"), 2))
        painter.drawLine(dx, 0, dx, h)


class TimeWarpPanel(QDialog):
    """Modal dialog: drag an anchor point to time-stretch or compress audio."""

    def __init__(
        self,
        audio: np.ndarray,
        sample_rate: int,
        audio_engine: AudioEngine,
        accent_color: str = "#00c8c8",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._audio = audio
        self._sr = sample_rate
        self._engine = audio_engine
        self._duration = len(audio) / sample_rate
        self._anchor: float | None = None
        self._drag: float | None = None

        self.warped_audio: np.ndarray | None = None

        self.setWindowTitle("Time Warp")
        self.setModal(True)
        self.setMinimumWidth(600)
        self.setMinimumHeight(340)
        self._build(accent_color)

    # ── Build ──────────────────────────────────────────────────────────

    def _build(self, color: str) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        instr = QLabel(
            "Click to set anchor point, then drag left/right to warp timing around it."
        )
        instr.setStyleSheet("color: #707070; font-size: 11px;")
        layout.addWidget(instr)

        self._waveform = _WarpWidget(color=color)
        self._waveform.set_audio(self._audio)
        self._waveform.anchor_changed.connect(self._on_anchor_changed)
        layout.addWidget(self._waveform)

        # Info row
        info_box = QGroupBox("Warp Info")
        info_row = QHBoxLayout(info_box)
        self._lbl_anchor = QLabel("Anchor: —")
        self._lbl_left   = QLabel("Left: —")
        self._lbl_right  = QLabel("Right: —")
        self._lbl_anchor.setStyleSheet("color: #c0c0c0; font-size: 11px;")
        self._lbl_left.setStyleSheet("color: #5090dd; font-size: 11px;")
        self._lbl_right.setStyleSheet("color: #dd5050; font-size: 11px;")
        info_row.addWidget(self._lbl_anchor)
        info_row.addStretch()
        info_row.addWidget(self._lbl_left)
        info_row.addStretch()
        info_row.addWidget(self._lbl_right)
        layout.addWidget(info_box)

        # Action buttons
        btn_row = QHBoxLayout()
        self._btn_preview = QPushButton("▶  Preview")
        self._btn_preview.setEnabled(False)
        self._btn_preview.setToolTip("Preview the warped audio")
        self._btn_preview.clicked.connect(self._on_preview)

        btn_reset = QPushButton("Reset")
        btn_reset.setToolTip("Clear the warp and start over")
        btn_reset.clicked.connect(self._on_reset)

        btn_row.addWidget(self._btn_preview)
        btn_row.addStretch()
        btn_row.addWidget(btn_reset)
        layout.addLayout(btn_row)

        # OK / Cancel
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        btn_ok = buttons.button(QDialogButtonBox.StandardButton.Ok)
        btn_ok.setText("Apply Warp")
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    # ── Slots ──────────────────────────────────────────────────────────

    def _on_anchor_changed(self, anchor: float, drag: float) -> None:
        self._anchor = anchor
        self._drag = drag

        anchor_t = anchor * self._duration
        drag_t   = drag   * self._duration

        self._lbl_anchor.setText(f"Anchor: {anchor_t:.3f} s → {drag_t:.3f} s")

        left_orig = anchor_t
        left_new  = drag_t
        if left_orig > 0:
            self._lbl_left.setText(f"Left: ×{left_new / left_orig:.2f}")
        else:
            self._lbl_left.setText("Left: —")

        right_orig = self._duration - anchor_t
        right_new  = self._duration - drag_t
        if right_orig > 0:
            self._lbl_right.setText(f"Right: ×{right_new / right_orig:.2f}")
        else:
            self._lbl_right.setText("Right: —")

        self._btn_preview.setEnabled(abs(anchor - drag) > 0.005)

    def _on_preview(self) -> None:
        warped = _compute_warp(self._audio, self._anchor, self._drag)
        if warped is not None:
            self._engine.play(warped, self._sr)

    def _on_reset(self) -> None:
        self._anchor = None
        self._drag = None
        self._waveform.reset()
        self._lbl_anchor.setText("Anchor: —")
        self._lbl_left.setText("Left: —")
        self._lbl_right.setText("Right: —")
        self._btn_preview.setEnabled(False)

    def _on_accept(self) -> None:
        if self._anchor is not None and abs(self._anchor - self._drag) > 0.005:
            warped = _compute_warp(self._audio, self._anchor, self._drag)
            if warped is not None:
                self.warped_audio = warped
        self._engine.stop()
        self.accept()

    def closeEvent(self, event) -> None:
        self._engine.stop()
        super().closeEvent(event)


# ── Time-warp algorithm ───────────────────────────────────────────────────────

def _stretch_segment(seg: np.ndarray, target_len: int) -> np.ndarray:
    """Pitch-preserving stretch of `seg` to exactly `target_len` samples."""
    if target_len <= 0:
        return np.zeros(0, dtype=np.float32)
    if len(seg) == 0:
        return np.zeros(target_len, dtype=np.float32)
    if len(seg) == target_len:
        return seg.astype(np.float32)

    rate = float(len(seg)) / float(target_len)
    rate = float(np.clip(rate, 0.1, 10.0))

    if len(seg) < 512:
        # Segment too short for the phase-vocoder; fall back to basic resample
        from scipy.signal import resample as sp_resample
        out = sp_resample(seg.astype(np.float32), target_len)
    else:
        try:
            import librosa
            out = librosa.effects.time_stretch(seg.astype(np.float32), rate=rate)
        except Exception:
            from scipy.signal import resample as sp_resample
            out = sp_resample(seg.astype(np.float32), target_len)

    out = out.astype(np.float32)
    if len(out) >= target_len:
        return out[:target_len]
    return np.pad(out, (0, target_len - len(out)))


def _compute_warp(
    audio: np.ndarray,
    anchor_frac: float | None,
    drag_frac: float | None,
) -> np.ndarray | None:
    if anchor_frac is None or drag_frac is None:
        return None

    n = len(audio)
    anchor  = max(1, min(int(anchor_frac * n), n - 1))
    new_anc = max(1, min(int(drag_frac   * n), n - 1))

    is_2d    = audio.ndim == 2
    channels = audio.shape[1] if is_2d else 1

    out_channels: list[np.ndarray] = []
    for ch in range(channels):
        sig = audio[:, ch] if is_2d else audio.ravel()

        left_out  = _stretch_segment(sig[:anchor],  new_anc)
        right_out = _stretch_segment(sig[anchor:],  n - new_anc)

        out_channels.append(np.concatenate([left_out, right_out]))

    if is_2d:
        return np.stack(out_channels, axis=1).astype(np.float32)
    return out_channels[0].astype(np.float32)
