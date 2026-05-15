from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QButtonGroup,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

from app.audio_engine import AudioEngine
from app.widgets.waveform_widget import WaveformWidget


class FadePanel(QDialog):
    """Modal dialog: apply a fade-in or fade-out envelope to a selected region."""

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

        self.faded_audio: np.ndarray | None = None

        self.setWindowTitle("Fade In / Fade Out")
        self.setModal(True)
        self.setMinimumWidth(560)
        self.setMinimumHeight(340)
        self._build(accent_color)
        self._sync_overlay()

    # ── Build ──────────────────────────────────────────────────────────

    def _build(self, color: str) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        self._waveform = WaveformWidget(color=color)
        self._waveform.setMinimumHeight(120)
        self._waveform.set_audio(self._audio)
        self._waveform.setCursor(Qt.CursorShape.CrossCursor)
        self._waveform.setToolTip("Left-click: set start  |  Right-click: set end")
        self._waveform.left_clicked.connect(self._on_waveform_left)
        self._waveform.right_clicked.connect(self._on_waveform_right)
        layout.addWidget(self._waveform)

        region_box = QGroupBox("Fade Region")
        form = QFormLayout(region_box)
        form.setSpacing(6)

        self._spin_start = QDoubleSpinBox()
        self._spin_start.setDecimals(3)
        self._spin_start.setRange(0.0, self._duration)
        self._spin_start.setSingleStep(0.01)
        self._spin_start.setSuffix(" s")
        self._spin_start.setValue(0.0)
        self._spin_start.setToolTip("Start of fade region in seconds")

        self._spin_end = QDoubleSpinBox()
        self._spin_end.setDecimals(3)
        self._spin_end.setRange(0.0, self._duration)
        self._spin_end.setSingleStep(0.01)
        self._spin_end.setSuffix(" s")
        self._spin_end.setValue(self._duration)
        self._spin_end.setToolTip("End of fade region in seconds")

        self._lbl_region = QLabel()
        self._lbl_region.setStyleSheet("color: #606060; font-size: 11px;")

        form.addRow("Start:", self._spin_start)
        form.addRow("End:", self._spin_end)
        form.addRow("", self._lbl_region)
        layout.addWidget(region_box)

        options_row = QHBoxLayout()

        type_box = QGroupBox("Type")
        type_layout = QHBoxLayout(type_box)
        self._radio_in = QRadioButton("Fade In")
        self._radio_in.setChecked(True)
        self._radio_out = QRadioButton("Fade Out")
        self._type_group = QButtonGroup(self)
        self._type_group.addButton(self._radio_in)
        self._type_group.addButton(self._radio_out)
        type_layout.addWidget(self._radio_in)
        type_layout.addWidget(self._radio_out)

        curve_box = QGroupBox("Curve")
        curve_layout = QHBoxLayout(curve_box)
        self._radio_linear = QRadioButton("Linear")
        self._radio_linear.setChecked(True)
        self._radio_smooth = QRadioButton("Smooth")
        self._curve_group = QButtonGroup(self)
        self._curve_group.addButton(self._radio_linear)
        self._curve_group.addButton(self._radio_smooth)
        curve_layout.addWidget(self._radio_linear)
        curve_layout.addWidget(self._radio_smooth)

        options_row.addWidget(type_box)
        options_row.addWidget(curve_box)
        layout.addLayout(options_row)

        btn_row = QHBoxLayout()
        self._btn_preview = QPushButton("▶  Preview")
        self._btn_preview.setToolTip("Play the audio with fade applied")
        self._btn_preview.clicked.connect(self._on_preview)

        btn_play_orig = QPushButton("▶  Original")
        btn_play_orig.setToolTip("Play the original audio without changes")
        btn_play_orig.clicked.connect(lambda: self._engine.play(self._audio, self._sr))

        btn_reset = QPushButton("Reset")
        btn_reset.setToolTip("Reset region to full audio length")
        btn_reset.clicked.connect(self._on_reset)

        btn_row.addWidget(self._btn_preview)
        btn_row.addWidget(btn_play_orig)
        btn_row.addStretch()
        btn_row.addWidget(btn_reset)
        layout.addLayout(btn_row)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Apply Fade")
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self._spin_start.valueChanged.connect(self._on_start_changed)
        self._spin_end.valueChanged.connect(self._on_end_changed)

    # ── Slots ──────────────────────────────────────────────────────────

    def _on_waveform_left(self, frac: float) -> None:
        self._spin_start.setValue(frac * self._duration)

    def _on_waveform_right(self, frac: float) -> None:
        self._spin_end.setValue(frac * self._duration)

    def _on_start_changed(self, value: float) -> None:
        if value > self._spin_end.value():
            self._spin_end.setValue(value)
        self._sync_overlay()

    def _on_end_changed(self, value: float) -> None:
        if value < self._spin_start.value():
            self._spin_start.setValue(value)
        self._sync_overlay()

    def _on_preview(self) -> None:
        self._engine.play(self._get_faded(), self._sr)

    def _on_reset(self) -> None:
        self._spin_start.setValue(0.0)
        self._spin_end.setValue(self._duration)

    def _on_accept(self) -> None:
        self._engine.stop()
        self.faded_audio = self._get_faded()
        self.accept()

    # ── Helpers ────────────────────────────────────────────────────────

    def _get_faded(self) -> np.ndarray:
        start_f = int(self._spin_start.value() * self._sr)
        end_f = int(self._spin_end.value() * self._sr)
        start_f = max(0, min(start_f, len(self._audio)))
        end_f = max(start_f, min(end_f, len(self._audio)))
        n = end_f - start_f
        if n <= 0:
            return self._audio.copy()

        if self._radio_smooth.isChecked():
            env = ((1.0 - np.cos(np.linspace(0.0, np.pi, n))) / 2.0).astype(np.float32)
        else:
            env = np.linspace(0.0, 1.0, n, dtype=np.float32)

        if self._radio_out.isChecked():
            env = env[::-1].copy()

        result = self._audio.copy()
        if result.ndim == 2:
            result[start_f:end_f] = (result[start_f:end_f] * env[:, np.newaxis]).astype(np.float32)
        else:
            result[start_f:end_f] = (result[start_f:end_f] * env).astype(np.float32)
        return result

    def _sync_overlay(self) -> None:
        start_frac = self._spin_start.value() / self._duration if self._duration else 0.0
        end_frac = self._spin_end.value() / self._duration if self._duration else 1.0
        self._waveform.set_trim_region(start_frac, end_frac)
        region_dur = self._spin_end.value() - self._spin_start.value()
        self._lbl_region.setText(f"Region length: {region_dur:.3f} s")

    def closeEvent(self, event) -> None:
        self._engine.stop()
        super().closeEvent(event)