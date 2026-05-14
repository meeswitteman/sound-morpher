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
    QSlider,
    QVBoxLayout,
    QWidget,
)

from app.audio_engine import AudioEngine

_TARGET_PEAK_DBFS = -0.1   # headroom after peak normalize
_TARGET_RMS_DBFS  = -18.0  # broadcast-style RMS target


def _db_to_linear(db: float) -> float:
    return 10.0 ** (db / 20.0)


def _linear_to_db(linear: float) -> float:
    if linear <= 0.0:
        return -96.0
    return 20.0 * np.log10(linear)


def _peak_dbfs(audio: np.ndarray) -> float:
    peak = float(np.abs(audio).max())
    return _linear_to_db(peak)


def _rms_dbfs(audio: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
    return _linear_to_db(rms)


class VolumePanel(QDialog):
    """Modal dialog: adjust gain or normalize a source audio clip."""

    def __init__(
        self,
        audio: np.ndarray,
        sample_rate: int,
        audio_engine: AudioEngine,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._audio = audio
        self._sr = sample_rate
        self._engine = audio_engine
        self._gain_db = 0.0

        self.adjusted_audio: np.ndarray | None = None

        self.setWindowTitle("Adjust Volume")
        self.setModal(True)
        self.setMinimumWidth(420)
        self._build()
        self._refresh_levels()

    # ── Build ──────────────────────────────────────────────────────────

    def _build(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # Current levels
        levels_box = QGroupBox("Current Levels")
        levels_form = QFormLayout(levels_box)
        self._lbl_peak = QLabel()
        self._lbl_rms  = QLabel()
        self._lbl_peak.setStyleSheet("font-family: monospace;")
        self._lbl_rms.setStyleSheet("font-family: monospace;")
        levels_form.addRow("Peak:", self._lbl_peak)
        levels_form.addRow("RMS:", self._lbl_rms)
        layout.addWidget(levels_box)

        # Gain control
        gain_box = QGroupBox("Gain Adjustment")
        gain_layout = QVBoxLayout(gain_box)

        slider_row = QHBoxLayout()
        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(-240, 120)   # tenths of dB: -24.0 … +12.0
        self._slider.setValue(0)
        self._slider.setTickInterval(60)
        self._slider.setTickPosition(QSlider.TickPosition.TicksBelow)

        self._spin = QDoubleSpinBox()
        self._spin.setRange(-24.0, 12.0)
        self._spin.setDecimals(1)
        self._spin.setSingleStep(0.1)
        self._spin.setSuffix(" dB")
        self._spin.setValue(0.0)
        self._spin.setFixedWidth(90)

        slider_row.addWidget(QLabel("-24 dB"))
        slider_row.addWidget(self._slider, stretch=1)
        slider_row.addWidget(QLabel("+12 dB"))
        slider_row.addWidget(self._spin)
        gain_layout.addLayout(slider_row)

        # After-gain level preview
        self._lbl_after = QLabel()
        self._lbl_after.setStyleSheet("color: #606060; font-size: 11px;")
        self._lbl_after.setAlignment(Qt.AlignmentFlag.AlignCenter)
        gain_layout.addWidget(self._lbl_after)

        layout.addWidget(gain_box)

        # Normalize buttons
        norm_row = QHBoxLayout()
        btn_norm_peak = QPushButton("Normalize Peak")
        btn_norm_peak.setToolTip(
            f"Scale so the loudest sample reaches {_TARGET_PEAK_DBFS} dBFS"
        )
        btn_norm_rms = QPushButton("Normalize RMS")
        btn_norm_rms.setToolTip(
            f"Scale so the average RMS level reaches {_TARGET_RMS_DBFS} dBFS"
        )
        norm_row.addWidget(btn_norm_peak)
        norm_row.addWidget(btn_norm_rms)
        layout.addLayout(norm_row)

        # Preview + reset row
        pr_row = QHBoxLayout()
        btn_preview = QPushButton("▶  Preview")
        btn_preview.setToolTip("Play with the current gain applied")
        btn_reset = QPushButton("Reset")
        btn_reset.setToolTip("Set gain back to 0 dB")
        pr_row.addWidget(btn_preview)
        pr_row.addStretch()
        pr_row.addWidget(btn_reset)
        layout.addLayout(pr_row)

        # Accept / Cancel
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        btn_ok = buttons.button(QDialogButtonBox.StandardButton.Ok)
        btn_ok.setText("Apply")
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        # Wiring
        self._slider.valueChanged.connect(self._on_slider_changed)
        self._spin.valueChanged.connect(self._on_spin_changed)
        btn_norm_peak.clicked.connect(self._on_normalize_peak)
        btn_norm_rms.clicked.connect(self._on_normalize_rms)
        btn_preview.clicked.connect(self._on_preview)
        btn_reset.clicked.connect(self._on_reset)

    # ── Slots ──────────────────────────────────────────────────────────

    def _on_slider_changed(self, value: int) -> None:
        db = value / 10.0
        self._spin.blockSignals(True)
        self._spin.setValue(db)
        self._spin.blockSignals(False)
        self._gain_db = db
        self._refresh_after()

    def _on_spin_changed(self, value: float) -> None:
        self._slider.blockSignals(True)
        self._slider.setValue(int(round(value * 10)))
        self._slider.blockSignals(False)
        self._gain_db = value
        self._refresh_after()

    def _on_normalize_peak(self) -> None:
        peak = float(np.abs(self._audio).max())
        if peak < 1e-9:
            return
        target_linear = _db_to_linear(_TARGET_PEAK_DBFS)
        gain_linear = target_linear / peak
        self._set_gain_db(_linear_to_db(gain_linear))

    def _on_normalize_rms(self) -> None:
        rms = float(np.sqrt(np.mean(self._audio.astype(np.float64) ** 2)))
        if rms < 1e-9:
            return
        target_linear = _db_to_linear(_TARGET_RMS_DBFS)
        gain_linear = target_linear / rms
        db = _linear_to_db(gain_linear)
        # Cap at +12 dB to avoid wild amplification of near-silent audio
        self._set_gain_db(min(db, 12.0))

    def _on_preview(self) -> None:
        self._engine.play(self._get_adjusted(), self._sr)

    def _on_reset(self) -> None:
        self._set_gain_db(0.0)

    def _on_accept(self) -> None:
        self._engine.stop()
        self.adjusted_audio = self._get_adjusted()
        self.accept()

    # ── Helpers ────────────────────────────────────────────────────────

    def _set_gain_db(self, db: float) -> None:
        db = max(-24.0, min(12.0, db))
        self._gain_db = db
        self._spin.blockSignals(True)
        self._slider.blockSignals(True)
        self._spin.setValue(round(db, 1))
        self._slider.setValue(int(round(db * 10)))
        self._spin.blockSignals(False)
        self._slider.blockSignals(False)
        self._refresh_after()

    def _get_adjusted(self) -> np.ndarray:
        gain = _db_to_linear(self._gain_db)
        result = (self._audio * gain).astype(np.float32)
        return np.clip(result, -1.0, 1.0)

    def _refresh_levels(self) -> None:
        peak = _peak_dbfs(self._audio)
        rms  = _rms_dbfs(self._audio)
        self._lbl_peak.setText(f"{peak:+.1f} dBFS")
        self._lbl_rms.setText(f"{rms:+.1f} dBFS")
        self._refresh_after()

    def _refresh_after(self) -> None:
        adjusted = self._get_adjusted()
        new_peak = _peak_dbfs(adjusted)
        new_rms  = _rms_dbfs(adjusted)
        clipping = new_peak > 0.0
        color = "#c84040" if clipping else "#606060"
        clip_note = "  ⚠ clipping" if clipping else ""
        self._lbl_after.setStyleSheet(f"color: {color}; font-size: 11px;")
        self._lbl_after.setText(
            f"After: peak {new_peak:+.1f} dBFS  ·  RMS {new_rms:+.1f} dBFS{clip_note}"
        )

    def closeEvent(self, event) -> None:
        self._engine.stop()
        super().closeEvent(event)
