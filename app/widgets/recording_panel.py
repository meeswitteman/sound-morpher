from __future__ import annotations

import numpy as np
import sounddevice as sd
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QBrush, QColor, QLinearGradient, QPainter, QPen
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from app.audio_engine import AudioEngine


class VuMeterWidget(QWidget):
    """Horizontal VU meter with green → yellow → red gradient."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._level = 0.0
        self.setFixedHeight(18)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def set_level(self, level: float) -> None:
        level = max(0.0, min(1.0, level))
        if abs(level - self._level) > 0.01:
            self._level = level
            self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        w, h = self.width(), self.height()

        painter.fillRect(0, 0, w, h, QColor("#111111"))

        if self._level <= 0:
            return

        fill_w = int(w * self._level)

        grad = QLinearGradient(0, 0, w, 0)
        grad.setColorAt(0.00, QColor("#00c040"))
        grad.setColorAt(0.65, QColor("#c8c000"))
        grad.setColorAt(0.85, QColor("#c04800"))
        grad.setColorAt(1.00, QColor("#ff0000"))
        painter.fillRect(0, 2, fill_w, h - 4, QBrush(grad))

        # Tick marks at -18, -12, -6, -3 dB (approximate linear fractions)
        painter.setPen(QPen(QColor("#1e1e1e"), 1))
        for frac in (0.25, 0.50, 0.75, 0.90):
            painter.drawLine(int(w * frac), 0, int(w * frac), h)


class RecordingPanel(QDialog):
    """Modal dialog: select input device, record, preview, and accept audio."""

    def __init__(self, audio_engine: AudioEngine, parent=None) -> None:
        super().__init__(parent)
        self._engine = audio_engine
        self._stream: sd.InputStream | None = None
        self._buffer: list[np.ndarray] = []
        self._recording = False
        self._level = 0.0

        self.recorded_audio: np.ndarray | None = None
        self.recorded_sr: int = 44100

        self.setWindowTitle("Record Audio")
        self.setModal(True)
        self.setMinimumWidth(440)
        self._build()

        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(40)
        self._poll_timer.timeout.connect(self._poll_vu)

    # ── Build ──────────────────────────────────────────────────────────

    def _build(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # Device selector
        dev_row = QHBoxLayout()
        dev_row.addWidget(QLabel("Input device:"))
        self._combo_device = QComboBox()
        self._combo_device.setMinimumWidth(230)
        self._device_indices: list[int] = []
        devices = self._engine.list_input_devices()
        default_idx = self._engine.default_input_device()
        for d in devices:
            self._combo_device.addItem(d["name"])
            self._device_indices.append(d["index"])
            if d["index"] == default_idx:
                self._combo_device.setCurrentIndex(self._combo_device.count() - 1)
        if not devices:
            self._combo_device.addItem("(no input devices found)")
        dev_row.addWidget(self._combo_device)
        layout.addLayout(dev_row)

        # VU meter
        layout.addWidget(QLabel("Input level:"))
        self._vu = VuMeterWidget()
        layout.addWidget(self._vu)

        # Duration counter
        self._lbl_duration = QLabel("00:00.000")
        self._lbl_duration.setStyleSheet(
            "font-size: 26px; font-weight: 600; color: #e0e0e0; padding: 6px 0;"
        )
        self._lbl_duration.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._lbl_duration)

        # Record toggle
        self._btn_record = QPushButton("● Start Recording")
        self._btn_record.setProperty("accent", "true")
        self._btn_record.setMinimumHeight(36)
        self._btn_record.clicked.connect(self._toggle_recording)
        layout.addWidget(self._btn_record)

        # Preview
        self._btn_preview = QPushButton("▶  Preview")
        self._btn_preview.setEnabled(False)
        self._btn_preview.clicked.connect(self._on_preview)
        layout.addWidget(self._btn_preview)

        # Status line
        self._lbl_status = QLabel("Ready to record")
        self._lbl_status.setStyleSheet("color: #606060; font-size: 11px;")
        self._lbl_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._lbl_status)

        # Accept / Cancel
        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self._btn_ok = btns.button(QDialogButtonBox.StandardButton.Ok)
        self._btn_ok.setText("Use Recording")
        self._btn_ok.setEnabled(False)
        btns.accepted.connect(self._on_accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    # ── Recording control ──────────────────────────────────────────────

    def _toggle_recording(self) -> None:
        if self._recording:
            self._stop_recording()
        else:
            self._start_recording()

    def _start_recording(self) -> None:
        self._buffer.clear()
        self._recording = True
        self._btn_record.setText("■  Stop Recording")
        self._btn_preview.setEnabled(False)
        self._btn_ok.setEnabled(False)
        self._lbl_status.setText("Recording…")

        idx = self._combo_device.currentIndex()
        device = self._device_indices[idx] if self._device_indices else None

        try:
            self._stream = sd.InputStream(
                device=device,
                channels=1,
                samplerate=44100,
                dtype="float32",
                callback=self._audio_callback,
            )
            self._stream.start()
        except Exception as exc:
            self._recording = False
            self._btn_record.setText("● Start Recording")
            self._lbl_status.setText(f"Device error: {exc}")
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(
                self,
                "Recording Device Error",
                f"Could not open the selected input device:\n\n{exc}\n\n"
                "Troubleshooting tips:\n"
                "• Check that your microphone or line-in is connected.\n"
                "• Make sure no other application is using the device.\n"
                "• Try selecting a different device from the dropdown.\n"
                "• On Windows, check Privacy Settings → Microphone access.",
            )
            return

        self._poll_timer.start()

    def _stop_recording(self) -> None:
        self._recording = False
        self._poll_timer.stop()

        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        self._vu.set_level(0.0)
        self._btn_record.setText("● Start Recording")

        if self._buffer:
            self.recorded_audio = np.concatenate(self._buffer, axis=0)
            self.recorded_sr = 44100
            dur = len(self.recorded_audio) / 44100
            self._lbl_status.setText(
                f"Captured {dur:.2f} s  —  press 'Use Recording' to accept"
            )
            self._btn_preview.setEnabled(True)
            self._btn_ok.setEnabled(True)
        else:
            self._lbl_status.setText("Nothing recorded")

    # ── Callbacks ──────────────────────────────────────────────────────

    def _audio_callback(
        self, indata: np.ndarray, frames: int, time, status
    ) -> None:
        # Runs in a C thread — only safe to write plain Python objects
        self._buffer.append(indata.copy())
        rms = float(np.sqrt(np.mean(indata**2)))
        self._level = min(rms * 5.0, 1.0)

    def _poll_vu(self) -> None:
        self._vu.set_level(self._level)
        if self._recording and self._buffer:
            frames = sum(len(b) for b in self._buffer)
            dur = frames / 44100
            m, s = divmod(dur, 60)
            self._lbl_duration.setText(f"{int(m):02d}:{s:06.3f}")

    # ── Slots ──────────────────────────────────────────────────────────

    def _on_preview(self) -> None:
        if self.recorded_audio is not None:
            self._engine.play(self.recorded_audio, self.recorded_sr)

    def _on_accept(self) -> None:
        if self._recording:
            self._stop_recording()
        self.accept()

    def closeEvent(self, event) -> None:
        if self._recording:
            self._stop_recording()
        self._engine.stop()
        super().closeEvent(event)
