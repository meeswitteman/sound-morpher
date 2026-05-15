from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import (
    QBrush, QColor, QLinearGradient, QPainter, QPainterPath, QPen
)
from PySide6.QtWidgets import (
    QButtonGroup,
    QDialog,
    QDialogButtonBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QRadioButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from app.audio_engine import AudioEngine


class _WarpWidget(QWidget):
    """Waveform with drag-to-warp interaction. Optionally shows a ghost reference."""

    anchor_changed = Signal(float, float)  # (anchor_frac, drag_frac)

    def __init__(self, color: str = "#00c8c8", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._color = QColor(color)
        self._audio: np.ndarray | None = None
        self._envelope: np.ndarray | None = None
        self._last_width = 0
        self._anchor: float | None = None
        self._drag: float | None = None
        self._pressed = False

        self._ref_audio: np.ndarray | None = None
        self._ref_color = QColor("#c87800")
        self._ref_envelope: np.ndarray | None = None
        self._ref_last_width = 0

        self.setMinimumHeight(160)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent)
        self.setCursor(Qt.CursorShape.SizeHorCursor)
        self.setToolTip(
            "Click to set anchor · drag left = compress left / stretch right"
        )

    def set_audio(self, audio: np.ndarray) -> None:
        self._audio = audio
        self._envelope = None
        self.update()

    def set_color(self, color: str) -> None:
        self._color = QColor(color)
        self._envelope = None
        self.update()

    def set_reference(self, audio: np.ndarray, color: str) -> None:
        self._ref_audio = audio
        self._ref_color = QColor(color)
        self._ref_envelope = None
        self._ref_last_width = 0
        self.update()

    def clear_reference(self) -> None:
        self._ref_audio = None
        self._ref_envelope = None
        self.update()

    def reset(self) -> None:
        self._anchor = None
        self._drag = None
        self._pressed = False
        self.update()

    def resizeEvent(self, event) -> None:
        self._envelope = None
        self._ref_envelope = None
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

    def _compute_envelope(self, audio: np.ndarray) -> np.ndarray:
        mono = audio.mean(axis=1) if audio.ndim == 2 else audio.flatten()
        w = max(self.width(), 1)
        total = len(mono)
        chunk = max(total // w, 1)
        cols = total // chunk
        if cols == 0:
            return np.zeros((1, 2), dtype=np.float32)
        trimmed = mono[: cols * chunk].reshape(cols, chunk)
        return np.stack([trimmed.min(axis=1), trimmed.max(axis=1)], axis=1).astype(np.float32)

    def _draw_waveform(
        self,
        painter: QPainter,
        env: np.ndarray,
        color: QColor,
        fill_alpha: int,
        h: int,
        cy: float,
    ) -> None:
        w = self.width()
        n = len(env)
        if n == 0:
            return
        x_scale = w / n

        path = QPainterPath()
        path.moveTo(0.0, cy - float(env[0, 1]) * cy)
        for i in range(1, n):
            path.lineTo(i * x_scale, cy - float(env[i, 1]) * cy)
        for i in range(n - 1, -1, -1):
            path.lineTo(i * x_scale, cy - float(env[i, 0]) * cy)
        path.closeSubpath()

        top = QColor(color); top.setAlpha(fill_alpha)
        mid = QColor(color); mid.setAlpha(fill_alpha // 2)
        grad = QLinearGradient(0, 0, 0, h)
        grad.setColorAt(0.0, top)
        grad.setColorAt(0.5, mid)
        grad.setColorAt(1.0, top)
        painter.fillPath(path, QBrush(grad))

        line_c = QColor(color); line_c.setAlpha(fill_alpha)
        painter.setPen(QPen(line_c, 1.2))
        top_path = QPainterPath()
        top_path.moveTo(0.0, cy - float(env[0, 1]) * cy)
        for i in range(1, n):
            top_path.lineTo(i * x_scale, cy - float(env[i, 1]) * cy)
        painter.drawPath(top_path)

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

        # Ghost reference waveform (drawn first, behind)
        if self._ref_audio is not None and len(self._ref_audio) > 0:
            if self._ref_envelope is None or self._ref_last_width != w:
                self._ref_envelope = self._compute_envelope(self._ref_audio)
                self._ref_last_width = w
            self._draw_waveform(painter, self._ref_envelope, self._ref_color, 65, h, cy)

        # Active waveform (drawn on top)
        if self._envelope is None or self._last_width != w:
            self._envelope = self._compute_envelope(self._audio)
            self._last_width = w
        self._draw_waveform(painter, self._envelope, self._color, 200, h, cy)

        painter.setPen(QPen(QColor("#1e1e1e"), 1))
        painter.drawLine(0, int(cy), w, int(cy))

        if self._anchor is None or self._drag is None:
            return

        ax = int(self._anchor * w)
        dx = int(self._drag * w)

        if dx < ax:
            left_col  = QColor(220,  80,  80, 55)
            right_col = QColor( 80, 150, 220, 55)
        elif dx > ax:
            left_col  = QColor( 80, 150, 220, 55)
            right_col = QColor(220,  80,  80, 55)
        else:
            left_col = right_col = QColor(0, 0, 0, 0)

        painter.fillRect(0,  0, ax,     h, left_col)
        painter.fillRect(ax, 0, w - ax, h, right_col)

        pen = QPen(QColor("#ffffff"), 1, Qt.PenStyle.DashLine)
        painter.setPen(pen)
        painter.drawLine(ax, 0, ax, h)

        painter.setPen(QPen(QColor("#ffdd00"), 2))
        painter.drawLine(dx, 0, dx, h)


# ── Single-audio warp dialog (fallback when only one sound is loaded) ─────────

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
        self.setMinimumHeight(360)
        self._build(accent_color)

    def _build(self, color: str) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        instr = QLabel("Click to set anchor point, then drag left/right to warp timing.")
        instr.setStyleSheet("color: #707070; font-size: 11px;")
        layout.addWidget(instr)

        self._waveform = _WarpWidget(color=color)
        self._waveform.set_audio(self._audio)
        self._waveform.anchor_changed.connect(self._on_anchor_changed)
        layout.addWidget(self._waveform)

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

        btn_row = QHBoxLayout()
        self._btn_preview = QPushButton("▶  Preview")
        self._btn_preview.setEnabled(False)
        self._btn_preview.clicked.connect(self._on_preview)
        btn_reset = QPushButton("Reset")
        btn_reset.clicked.connect(self._on_reset)
        btn_row.addWidget(self._btn_preview)
        btn_row.addStretch()
        btn_row.addWidget(btn_reset)
        layout.addLayout(btn_row)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Apply Warp")
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _on_anchor_changed(self, anchor: float, drag: float) -> None:
        self._anchor = anchor
        self._drag = drag
        anchor_t = anchor * self._duration
        drag_t   = drag   * self._duration
        self._lbl_anchor.setText(f"Anchor: {anchor_t:.3f} s → {drag_t:.3f} s")
        left_orig = anchor_t
        left_new  = drag_t
        self._lbl_left.setText(
            f"Left: ×{left_new / left_orig:.2f}" if left_orig > 0 else "Left: —"
        )
        right_orig = self._duration - anchor_t
        right_new  = self._duration - drag_t
        self._lbl_right.setText(
            f"Right: ×{right_new / right_orig:.2f}" if right_orig > 0 else "Right: —"
        )
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


# ── Dual-audio align dialog ───────────────────────────────────────────────────

class AlignPanel(QDialog):
    """Show A and B waveforms overlaid at the same scale; warp one to align them."""

    def __init__(
        self,
        audio_a: np.ndarray,
        audio_b: np.ndarray,
        sample_rate: int,
        audio_engine: AudioEngine,
        warp_target: str = "A",
        color_a: str = "#00c8c8",
        color_b: str = "#c87800",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._audio_a = audio_a
        self._audio_b = audio_b
        self._sr = sample_rate
        self._engine = audio_engine
        self._color_a = color_a
        self._color_b = color_b
        self._anchor: float | None = None
        self._drag: float | None = None
        self._duration: float = 0.0

        self.warped_a: np.ndarray | None = None
        self.warped_b: np.ndarray | None = None

        self.setWindowTitle("Align Sounds")
        self.setModal(True)
        self.setMinimumWidth(680)
        self.setMinimumHeight(420)
        self._build(warp_target)

    def _build(self, initial_target: str) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        instr = QLabel(
            "Both sounds are shown at the same visual scale (full width). "
            "The solid waveform is being warped; the ghost is the reference. "
            "Click and drag left/right to stretch or compress timing."
        )
        instr.setStyleSheet("color: #707070; font-size: 11px;")
        instr.setWordWrap(True)
        layout.addWidget(instr)

        # Warp-target selector + action buttons on one row
        ctrl_row = QHBoxLayout()

        lbl = QLabel("Warp:")
        lbl.setStyleSheet("color: #909090; font-size: 11px;")
        self._radio_a = QRadioButton("Sound A  (solid)")
        self._radio_b = QRadioButton("Sound B  (solid)")
        self._radio_a.setStyleSheet(f"color: {self._color_a}; font-size: 11px;")
        self._radio_b.setStyleSheet(f"color: {self._color_b}; font-size: 11px;")

        self._btn_group = QButtonGroup(self)
        self._btn_group.addButton(self._radio_a, 0)
        self._btn_group.addButton(self._radio_b, 1)

        # Set initial selection without triggering the signal yet
        if initial_target == "A":
            self._radio_a.setChecked(True)
        else:
            self._radio_b.setChecked(True)

        self._btn_preview = QPushButton("▶  Preview")
        self._btn_preview.setEnabled(False)
        self._btn_preview.setToolTip("Preview the warped audio")
        self._btn_preview.clicked.connect(self._on_preview)

        btn_reset = QPushButton("Reset")
        btn_reset.setToolTip("Clear the current warp")
        btn_reset.clicked.connect(self._on_reset)

        ctrl_row.addWidget(lbl)
        ctrl_row.addWidget(self._radio_a)
        ctrl_row.addWidget(self._radio_b)
        ctrl_row.addStretch()
        ctrl_row.addWidget(self._btn_preview)
        ctrl_row.addWidget(btn_reset)
        layout.addLayout(ctrl_row)

        # Waveform view
        self._waveform = _WarpWidget()
        self._waveform.anchor_changed.connect(self._on_anchor_changed)
        layout.addWidget(self._waveform)

        # Duration legend
        dur_a = len(self._audio_a) / self._sr
        dur_b = len(self._audio_b) / self._sr
        legend = QLabel(
            f"A: {dur_a:.2f} s    B: {dur_b:.2f} s    "
            f"(both shown at full width — visual scale reflects shape, not time)"
        )
        legend.setStyleSheet("color: #505050; font-size: 10px;")
        layout.addWidget(legend)

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

        # OK / Cancel
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Apply Warp")
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        # Wire radio buttons and initialise the waveform view
        self._btn_group.buttonClicked.connect(lambda _: self._update_warp_target())
        self._update_warp_target()

    # ── Helpers ────────────────────────────────────────────────────────

    def _active_target(self) -> str:
        return "A" if self._radio_a.isChecked() else "B"

    def _update_warp_target(self) -> None:
        target = self._active_target()
        self._on_reset()

        if target == "A":
            active, active_color = self._audio_a, self._color_a
            ref,    ref_color    = self._audio_b, self._color_b
        else:
            active, active_color = self._audio_b, self._color_b
            ref,    ref_color    = self._audio_a, self._color_a

        self._duration = len(active) / self._sr
        self._waveform.set_color(active_color)
        self._waveform.set_audio(active)
        self._waveform.set_reference(ref, ref_color)

    # ── Slots ──────────────────────────────────────────────────────────

    def _on_anchor_changed(self, anchor: float, drag: float) -> None:
        self._anchor = anchor
        self._drag = drag
        anchor_t = anchor * self._duration
        drag_t   = drag   * self._duration
        self._lbl_anchor.setText(f"Anchor: {anchor_t:.3f} s → {drag_t:.3f} s")
        self._lbl_left.setText(
            f"Left: ×{drag_t / anchor_t:.2f}" if anchor_t > 0 else "Left: —"
        )
        right_orig = self._duration - anchor_t
        right_new  = self._duration - drag_t
        self._lbl_right.setText(
            f"Right: ×{right_new / right_orig:.2f}" if right_orig > 0 else "Right: —"
        )
        self._btn_preview.setEnabled(abs(anchor - drag) > 0.005)

    def _on_preview(self) -> None:
        target = self._active_target()
        audio = self._audio_a if target == "A" else self._audio_b
        warped = _compute_warp(audio, self._anchor, self._drag)
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
            target = self._active_target()
            audio = self._audio_a if target == "A" else self._audio_b
            warped = _compute_warp(audio, self._anchor, self._drag)
            if warped is not None:
                if target == "A":
                    self.warped_a = warped
                else:
                    self.warped_b = warped
        self._engine.stop()
        self.accept()

    def closeEvent(self, event) -> None:
        self._engine.stop()
        super().closeEvent(event)


# ── Time-warp algorithm ───────────────────────────────────────────────────────

def _stretch_segment(seg: np.ndarray, target_len: int) -> np.ndarray:
    if target_len <= 0:
        return np.zeros(0, dtype=np.float32)
    if len(seg) == 0:
        return np.zeros(target_len, dtype=np.float32)
    if len(seg) == target_len:
        return seg.astype(np.float32)

    rate = float(np.clip(float(len(seg)) / float(target_len), 0.1, 10.0))

    if len(seg) < 512:
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
