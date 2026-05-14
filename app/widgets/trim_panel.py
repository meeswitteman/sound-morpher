from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from app.audio_engine import AudioEngine
from app.widgets.waveform_widget import WaveformWidget


class TrimPanel(QDialog):
    """Modal dialog: set start/end trim points on a source audio clip."""

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

        self.trimmed_audio: np.ndarray | None = None

        self.setWindowTitle("Trim Audio")
        self.setModal(True)
        self.setMinimumWidth(560)
        self.setMinimumHeight(280)
        self._build(accent_color)
        self._sync_overlay()

    # ── Build ──────────────────────────────────────────────────────────

    def _build(self, color: str) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # Waveform
        self._waveform = WaveformWidget(color=color)
        self._waveform.setMinimumHeight(120)
        self._waveform.set_audio(self._audio)
        layout.addWidget(self._waveform)

        # Time inputs
        form_box = QGroupBox("Trim Region")
        form = QFormLayout(form_box)
        form.setSpacing(6)

        self._spin_start = QDoubleSpinBox()
        self._spin_start.setDecimals(3)
        self._spin_start.setRange(0.0, self._duration)
        self._spin_start.setSingleStep(0.01)
        self._spin_start.setSuffix(" s")
        self._spin_start.setValue(0.0)
        self._spin_start.setToolTip("Start of the trim region in seconds")

        self._spin_end = QDoubleSpinBox()
        self._spin_end.setDecimals(3)
        self._spin_end.setRange(0.0, self._duration)
        self._spin_end.setSingleStep(0.01)
        self._spin_end.setSuffix(" s")
        self._spin_end.setValue(self._duration)
        self._spin_end.setToolTip("End of the trim region in seconds")

        self._lbl_trimmed = QLabel()
        self._lbl_trimmed.setStyleSheet("color: #606060; font-size: 11px;")

        form.addRow("Start:", self._spin_start)
        form.addRow("End:", self._spin_end)
        form.addRow("", self._lbl_trimmed)
        layout.addWidget(form_box)

        # Buttons row
        btn_row = QHBoxLayout()

        self._btn_preview = QPushButton("▶  Preview Trimmed")
        self._btn_preview.setToolTip("Play the trimmed region")
        self._btn_preview.clicked.connect(self._on_preview)

        self._btn_play_full = QPushButton("▶  Play Full")
        self._btn_play_full.setToolTip("Play the entire original audio")
        self._btn_play_full.clicked.connect(self._on_play_full)

        btn_reset = QPushButton("Reset")
        btn_reset.setToolTip("Reset trim to the full audio length")
        btn_reset.clicked.connect(self._on_reset)

        btn_row.addWidget(self._btn_preview)
        btn_row.addWidget(self._btn_play_full)
        btn_row.addStretch()
        btn_row.addWidget(btn_reset)
        layout.addLayout(btn_row)

        # Accept / Cancel
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        btn_ok = buttons.button(QDialogButtonBox.StandardButton.Ok)
        btn_ok.setText("Apply Trim")
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        # Connections
        self._spin_start.valueChanged.connect(self._on_start_changed)
        self._spin_end.valueChanged.connect(self._on_end_changed)

    # ── Slots ──────────────────────────────────────────────────────────

    def _on_start_changed(self, value: float) -> None:
        if value > self._spin_end.value():
            self._spin_end.setValue(value)
        self._sync_overlay()

    def _on_end_changed(self, value: float) -> None:
        if value < self._spin_start.value():
            self._spin_start.setValue(value)
        self._sync_overlay()

    def _on_preview(self) -> None:
        trimmed = self._get_trimmed()
        if len(trimmed) > 0:
            self._engine.play(trimmed, self._sr)

    def _on_play_full(self) -> None:
        self._engine.play(self._audio, self._sr)

    def _on_reset(self) -> None:
        self._spin_start.setValue(0.0)
        self._spin_end.setValue(self._duration)

    def _on_accept(self) -> None:
        trimmed = self._get_trimmed()
        if len(trimmed) == 0:
            return
        self._engine.stop()
        self.trimmed_audio = trimmed
        self.accept()

    # ── Helpers ────────────────────────────────────────────────────────

    def _get_trimmed(self) -> np.ndarray:
        start_frame = int(self._spin_start.value() * self._sr)
        end_frame = int(self._spin_end.value() * self._sr)
        start_frame = max(0, min(start_frame, len(self._audio)))
        end_frame = max(start_frame, min(end_frame, len(self._audio)))
        return self._audio[start_frame:end_frame]

    def _sync_overlay(self) -> None:
        start_frac = self._spin_start.value() / self._duration if self._duration else 0.0
        end_frac = self._spin_end.value() / self._duration if self._duration else 1.0
        self._waveform.set_trim_region(start_frac, end_frac)

        trimmed_dur = self._spin_end.value() - self._spin_start.value()
        self._lbl_trimmed.setText(f"Trimmed length: {trimmed_dur:.3f} s")

    def closeEvent(self, event) -> None:
        self._engine.stop()
        super().closeEvent(event)
