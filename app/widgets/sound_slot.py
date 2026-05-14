from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

from app.audio_engine import AudioEngine
from app.widgets.waveform_widget import WaveformWidget


class SoundSlot(QGroupBox):
    """Displays one sound source (A or B) with waveform, metadata, and controls."""

    audio_changed = Signal(object, int)  # (np.ndarray | None, sample_rate)
    status_message = Signal(str)        # informational notices (resampling, float WAV, …)

    def __init__(
        self,
        label: str,
        accent_color: str,
        audio_engine: AudioEngine,
        target_sr: int = 44100,
        parent=None,
    ) -> None:
        super().__init__(f"Sound {label}", parent)
        self._label = label
        self._engine = audio_engine
        self._target_sr = target_sr
        self._accent_color = accent_color
        self._audio: np.ndarray | None = None
        self._display_name: str = ""

        self.setMinimumHeight(200)
        self._build(accent_color)

    # ── Public API ─────────────────────────────────────────────────────

    @property
    def audio(self) -> np.ndarray | None:
        return self._audio

    @property
    def sample_rate(self) -> int:
        return self._target_sr

    @property
    def name(self) -> str:
        return self._display_name

    def set_audio(
        self,
        audio: np.ndarray,
        src_sr: int,
        display_name: str = "",
    ) -> None:
        normalized = self._engine.normalize_audio(
            audio, src_sr=src_sr, target_sr=self._target_sr
        )
        self._audio = normalized
        self._display_name = display_name
        self._waveform.set_audio(normalized)
        self._btn_play.setEnabled(True)
        self._btn_trim.setEnabled(True)
        self._btn_volume.setEnabled(True)
        self._update_meta(display_name, normalized)
        self.audio_changed.emit(normalized, self._target_sr)

    def clear(self) -> None:
        self._audio = None
        self._waveform.set_audio(None)
        self._btn_play.setEnabled(False)
        self._btn_trim.setEnabled(False)
        self._btn_volume.setEnabled(False)
        self._lbl_meta.setText("—")
        self.audio_changed.emit(None, self._target_sr)

    # ── Build ──────────────────────────────────────────────────────────

    def _build(self, color: str) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(7)

        self._waveform = WaveformWidget(color=color)
        self._waveform.setMinimumHeight(110)
        layout.addWidget(self._waveform)

        btn_row = QHBoxLayout()
        self._btn_load = QPushButton("Load WAV")
        self._btn_load.setToolTip(f"Select a WAV file for Sound {self._label}")
        self._btn_record = QPushButton("● Rec")
        self._btn_record.setToolTip(f"Record from microphone as Sound {self._label}")
        self._btn_play = QPushButton("▶")
        self._btn_play.setFixedWidth(34)
        self._btn_play.setEnabled(False)
        self._btn_play.setToolTip(f"Preview Sound {self._label}")

        self._btn_trim = QPushButton("✂")
        self._btn_trim.setFixedWidth(34)
        self._btn_trim.setEnabled(False)
        self._btn_trim.setToolTip(f"Trim Sound {self._label} to a shorter region")

        self._btn_volume = QPushButton("⊿")
        self._btn_volume.setFixedWidth(34)
        self._btn_volume.setEnabled(False)
        self._btn_volume.setToolTip(f"Adjust volume or normalize Sound {self._label}")

        self._btn_load.clicked.connect(self._on_load)
        self._btn_record.clicked.connect(self._on_record)
        self._btn_play.clicked.connect(self._on_play)
        self._btn_trim.clicked.connect(self._on_trim)
        self._btn_volume.clicked.connect(self._on_volume)

        btn_row.addWidget(self._btn_load)
        btn_row.addWidget(self._btn_record)
        btn_row.addWidget(self._btn_play)
        btn_row.addWidget(self._btn_trim)
        btn_row.addWidget(self._btn_volume)
        layout.addLayout(btn_row)

        self._lbl_meta = QLabel("—")
        self._lbl_meta.setStyleSheet("color: #505050; font-size: 10px; padding: 0 2px;")
        layout.addWidget(self._lbl_meta)

    # ── Slots ──────────────────────────────────────────────────────────

    def _on_load(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            f"Load WAV for Sound {self._label}",
            "",
            "WAV files (*.wav);;All files (*)",
        )
        if not path:
            return
        try:
            info = self._engine.get_wav_info(path)
            audio, sr = self._engine.load_wav(path)
        except Exception as exc:
            QMessageBox.critical(self, "Load Error", f"Could not read file:\n{exc}")
            return

        notices: list[str] = []
        if sr != self._target_sr:
            notices.append(
                f"Sound {self._label}: resampled {sr} Hz → {self._target_sr} Hz"
            )
        if "FLOAT" in info.get("subtype", "").upper():
            notices.append(
                f"Sound {self._label}: 32-bit float WAV converted to project format"
            )
        if notices:
            self.status_message.emit("  |  ".join(notices))

        self.set_audio(audio, sr, Path(path).name)

    def _on_record(self) -> None:
        from app.widgets.recording_panel import RecordingPanel

        dlg = RecordingPanel(self._engine, parent=self)
        if dlg.exec() and dlg.recorded_audio is not None:
            self.set_audio(dlg.recorded_audio, dlg.recorded_sr, "recording")

    def _on_play(self) -> None:
        if self._audio is not None:
            self._engine.play(self._audio, self._target_sr)

    def _on_trim(self) -> None:
        if self._audio is None:
            return
        from app.widgets.trim_panel import TrimPanel

        dlg = TrimPanel(
            self._audio,
            self._target_sr,
            self._engine,
            accent_color=self._accent_color,
            parent=self,
        )
        if dlg.exec() and dlg.trimmed_audio is not None:
            self.set_audio(dlg.trimmed_audio, self._target_sr, self._display_name)

    def _on_volume(self) -> None:
        if self._audio is None:
            return
        from app.widgets.volume_panel import VolumePanel

        dlg = VolumePanel(self._audio, self._target_sr, self._engine, parent=self)
        if dlg.exec() and dlg.adjusted_audio is not None:
            self.set_audio(dlg.adjusted_audio, self._target_sr, self._display_name)

    # ── Helpers ────────────────────────────────────────────────────────

    def _update_meta(self, name: str, audio: np.ndarray) -> None:
        dur = len(audio) / self._target_sr
        ch = audio.shape[1] if audio.ndim == 2 else 1
        ch_label = "stereo" if ch == 2 else "mono"
        text = f"{name}  ·  {dur:.2f} s  ·  {self._target_sr} Hz  ·  {ch_label}"
        self._lbl_meta.setText(text)
