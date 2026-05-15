from __future__ import annotations

import time
from collections import deque

from pathlib import Path

from PySide6.QtCore import Qt, QSize, QSettings
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from app.audio_engine import AudioEngine
from app.bpm_engine import BpmEngine
from app.export import ExportEngine
from app.morph_engine import MorphEngine
from app.project_file import ProjectFile, ProjectFileError, SMORPH_FILTER
from app.project_state import ProjectState
from app.widgets.plugin_param_panel import PluginParamPanel
from app.widgets.sound_slot import SoundSlot
from app.widgets.step_grid import StepGrid
from plugins.registry import build_default_registry


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.audio_engine = AudioEngine()
        self.project = ProjectState()
        self.morph_engine = MorphEngine(self)
        self.bpm_engine = BpmEngine(self)
        self.export_engine = ExportEngine(self)
        self.plugin_registry = build_default_registry()
        self._tap_times: deque[float] = deque(maxlen=8)
        self._playing = False
        self._unsaved_changes = False
        self._settings = QSettings("SoundMorpher", "SoundMorpher")

        self.setWindowTitle("Sound Morpher")
        self.setMinimumSize(960, 600)
        self.resize(1280, 740)

        self._build_menu()
        self._build_central()
        self._build_bottom_toolbar()
        self._build_statusbar()
        self._connect_shortcuts()
        self._connect_morph_engine()
        self._connect_bpm_engine()
        self._connect_export_engine()
        self._populate_plugin_dropdown()
        self._rebuild_recent_menu()

    # ── Menu ──────────────────────────────────────────────────────────

    def _build_menu(self) -> None:
        mb = self.menuBar()

        file_menu = mb.addMenu("File")
        file_menu.addAction("New Project", self._on_new, QKeySequence.StandardKey.New)
        file_menu.addAction("Open Project…", self._on_open, QKeySequence.StandardKey.Open)
        file_menu.addSeparator()
        self._act_save = file_menu.addAction("Save", self._on_save, QKeySequence.StandardKey.Save)
        self._act_save.setEnabled(False)
        file_menu.addAction("Save As…", self._on_save_as, QKeySequence("Ctrl+Shift+S"))
        file_menu.addSeparator()
        self._recent_menu = file_menu.addMenu("Recent Projects")
        self._recent_menu.addAction("(none)").setEnabled(False)
        file_menu.addSeparator()
        file_menu.addAction("Quit", self.close, QKeySequence.StandardKey.Quit)

        project_menu = mb.addMenu("Project")
        project_menu.addAction("Project Settings…")

        self._plugins_menu = mb.addMenu("Plugins")
        self._plugins_menu.addAction("Manage Plugins…")

        help_menu = mb.addMenu("Help")
        help_menu.addAction("About Sound Morpher", self._on_about)

    # ── Central widget ─────────────────────────────────────────────────

    def _build_central(self) -> None:
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(2)
        splitter.setChildrenCollapsible(False)

        splitter.addWidget(self._build_sound_panel())
        splitter.addWidget(self._build_morph_panel())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([290, 990])

        self.setCentralWidget(splitter)

    def _build_sound_panel(self) -> QWidget:
        panel = QWidget()
        panel.setFixedWidth(290)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 10, 6, 10)
        layout.setSpacing(10)

        self.slot_a = SoundSlot(
            "A", "#00c8c8", self.audio_engine, self.project.sample_rate
        )
        self.slot_b = SoundSlot(
            "B", "#c87800", self.audio_engine, self.project.sample_rate
        )
        self.slot_a.audio_changed.connect(self._on_audio_a_changed)
        self.slot_b.audio_changed.connect(self._on_audio_b_changed)
        self.slot_a.status_message.connect(self.statusBar().showMessage)
        self.slot_b.status_message.connect(self.statusBar().showMessage)
        self.slot_a.align_requested.connect(self._on_align_requested)
        self.slot_b.align_requested.connect(self._on_align_requested)

        layout.addWidget(self.slot_a)
        layout.addWidget(self.slot_b)
        layout.addStretch()
        return panel

    def _build_morph_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(6, 10, 10, 10)
        layout.setSpacing(8)

        layout.addWidget(self._build_morph_settings())
        layout.addWidget(self._build_step_grid_area(), stretch=1)
        return panel

    def _build_morph_settings(self) -> QGroupBox:
        box = QGroupBox("Morph Settings")
        outer = QVBoxLayout(box)
        outer.setSpacing(6)

        row = QHBoxLayout()
        row.setSpacing(12)

        # Algorithm
        row.addWidget(QLabel("Algorithm:"))
        self.combo_algorithm = QComboBox()
        self.combo_algorithm.setToolTip("Morphing algorithm (plugins loaded at startup)")
        self.combo_algorithm.addItem("Crossfade")
        row.addWidget(self.combo_algorithm)

        row.addWidget(_vline())

        # Steps
        row.addWidget(QLabel("Steps:"))
        self.spin_steps = QSpinBox()
        self.spin_steps.setRange(2, 32)
        self.spin_steps.setValue(8)
        self.spin_steps.setToolTip("Number of morph steps (2 = A and B only, 32 = max)")
        row.addWidget(self.spin_steps)

        row.addWidget(_vline())

        # BPM
        row.addWidget(QLabel("BPM:"))
        self.spin_bpm = QSpinBox()
        self.spin_bpm.setRange(20, 300)
        self.spin_bpm.setValue(120)
        self.spin_bpm.setToolTip("Playback tempo in beats per minute")
        row.addWidget(self.spin_bpm)

        # Beats per step
        row.addWidget(QLabel("Beats/step:"))
        self.spin_beats = QSpinBox()
        self.spin_beats.setRange(1, 64)
        self.spin_beats.setValue(4)
        self.spin_beats.setToolTip("Number of beats before advancing to the next morph step")
        row.addWidget(self.spin_beats)

        self.btn_tap = QPushButton("Tap")
        self.btn_tap.setFixedWidth(46)
        self.btn_tap.setToolTip("Tap to measure BPM")
        row.addWidget(self.btn_tap)

        row.addWidget(_vline())

        self.chk_loop = QCheckBox("Loop")
        self.chk_loop.setToolTip("Loop playback back to step 1 after last step")
        row.addWidget(self.chk_loop)

        row.addWidget(_vline())

        self.chk_dtw = QCheckBox("DTW Align")
        self.chk_dtw.setToolTip(
            "Dynamic Time Warping: time-align A and B by their MFCC similarity "
            "before morphing. Helps when A and B have different tempos or timing."
        )
        row.addWidget(self.chk_dtw)

        row.addStretch()
        outer.addLayout(row)

        # Algorithm parameter panel (auto-generated per plugin)
        self._param_panel = PluginParamPanel()
        outer.addWidget(self._param_panel)

        self.combo_algorithm.currentIndexChanged.connect(self._on_algorithm_changed)

        return box

    def _build_step_grid_area(self) -> QGroupBox:
        box = QGroupBox("Step Grid")
        self._step_grid_layout = QVBoxLayout(box)

        self._step_grid_placeholder = QLabel(
            "Load Sound A and Sound B, then click  ⟳ Recompute  to generate morph steps"
        )
        self._step_grid_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._step_grid_placeholder.setStyleSheet(
            "color: #404040; font-size: 12px; padding: 40px;"
        )
        self._step_grid_layout.addWidget(self._step_grid_placeholder)

        self._step_grid: StepGrid | None = None
        return box

    # ── Bottom toolbar ─────────────────────────────────────────────────

    def _build_bottom_toolbar(self) -> None:
        tb = QToolBar("Playback")
        tb.setMovable(False)
        tb.setFloatable(False)
        tb.setIconSize(QSize(14, 14))
        tb.setAllowedAreas(Qt.ToolBarArea.BottomToolBarArea)
        self.addToolBar(Qt.ToolBarArea.BottomToolBarArea, tb)

        self.btn_play_all = QPushButton("▶  Play All")
        self.btn_play_all.setProperty("accent", "true")
        self.btn_play_all.setEnabled(False)
        self.btn_play_all.setToolTip("Play all morph steps in sequence (Ctrl+Space)")

        self.btn_stop = QPushButton("■  Stop")
        self.btn_stop.setEnabled(False)
        self.btn_stop.setToolTip("Stop playback (Escape)")

        self.btn_recompute = QPushButton("⟳  Recompute")
        self.btn_recompute.setEnabled(False)
        self.btn_recompute.setToolTip("Recalculate all morph steps with current settings")

        self.btn_export = QPushButton("↓  Export Steps")
        self.btn_export.setEnabled(False)
        self.btn_export.setToolTip("Export all morph steps as WAV files")

        # Beat indicator container (dots rebuilt when beats_per_step changes)
        self._beat_dots: list[QLabel] = []
        self._beat_container = QWidget()
        self._beat_layout = QHBoxLayout(self._beat_container)
        self._beat_layout.setContentsMargins(8, 0, 0, 0)
        self._beat_layout.setSpacing(5)
        self._beat_layout.addWidget(QLabel("Beat:"))
        self._rebuild_beat_dots(4)

        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)

        tb.addWidget(self.btn_play_all)
        tb.addWidget(self.btn_stop)
        tb.addSeparator()
        tb.addWidget(self.btn_recompute)
        tb.addSeparator()
        tb.addWidget(self.btn_export)
        tb.addSeparator()
        tb.addWidget(self._beat_container)
        tb.addWidget(spacer)

    # ── Status bar ─────────────────────────────────────────────────────

    def _build_statusbar(self) -> None:
        sb = self.statusBar()
        sb.showMessage("Ready — no project loaded")

        self.lbl_status_sr = QLabel("44100 Hz · 16-bit")
        self.lbl_status_sr.setStyleSheet("color: #454545; padding: 0 10px;")
        sb.addPermanentWidget(self.lbl_status_sr)

    # ── Keyboard shortcuts ─────────────────────────────────────────────

    def _connect_shortcuts(self) -> None:
        QShortcut(QKeySequence("Escape"), self, self._on_stop)
        QShortcut(QKeySequence("Ctrl+Space"), self, self._on_play_all)
        QShortcut(QKeySequence("F5"), self, self._on_recompute)

    # ── Audio slot callbacks ───────────────────────────────────────────

    def _on_audio_a_changed(self, audio, sr: int) -> None:
        import numpy as np
        self.project.audio_a = audio if isinstance(audio, np.ndarray) else None
        self.project.name_a = self.slot_a.name
        self._refresh_recompute_button()
        self._set_unsaved(True)

    def _on_audio_b_changed(self, audio, sr: int) -> None:
        import numpy as np
        self.project.audio_b = audio if isinstance(audio, np.ndarray) else None
        self.project.name_b = self.slot_b.name
        self._refresh_recompute_button()
        self._set_unsaved(True)

    def _on_align_requested(self, label: str) -> None:
        audio_a = self.project.audio_a
        audio_b = self.project.audio_b

        from app.widgets.timewarp_panel import AlignPanel, TimeWarpPanel

        if audio_a is not None and audio_b is not None:
            dlg = AlignPanel(
                audio_a, audio_b,
                self.project.sample_rate,
                self.audio_engine,
                warp_target=label,
                parent=self,
            )
            if dlg.exec():
                if dlg.warped_a is not None:
                    self.slot_a.set_audio(
                        dlg.warped_a, self.project.sample_rate,
                        self.project.name_a or "source_a",
                    )
                if dlg.warped_b is not None:
                    self.slot_b.set_audio(
                        dlg.warped_b, self.project.sample_rate,
                        self.project.name_b or "source_b",
                    )
        else:
            # Only one sound loaded: single-waveform fallback
            slot = self.slot_a if label == "A" else self.slot_b
            audio = audio_a if label == "A" else audio_b
            if audio is None:
                return
            color = "#00c8c8" if label == "A" else "#c87800"
            name = (self.project.name_a if label == "A" else self.project.name_b) or f"source_{label.lower()}"
            dlg = TimeWarpPanel(audio, self.project.sample_rate, self.audio_engine,
                                accent_color=color, parent=self)
            if dlg.exec() and dlg.warped_audio is not None:
                slot.set_audio(dlg.warped_audio, self.project.sample_rate, name)

    def _refresh_recompute_button(self) -> None:
        ready = self.project.ready_to_morph and not self.morph_engine.is_running
        self.btn_recompute.setEnabled(ready)
        if ready and not self.project.has_morph_steps:
            self.statusBar().showMessage(
                "Both sounds loaded — click ⟳ Recompute to generate morph steps"
            )

    # ── Plugin helpers ─────────────────────────────────────────────────

    def _populate_plugin_dropdown(self) -> None:
        self.combo_algorithm.clear()
        for name in self.plugin_registry.names():
            plugin = self.plugin_registry.get(name)
            self.combo_algorithm.addItem(name)
            self.combo_algorithm.setItemData(
                self.combo_algorithm.count() - 1,
                plugin.description,
                256,  # Qt.ItemDataRole.ToolTipRole
            )
        self.combo_algorithm.setEnabled(self.combo_algorithm.count() > 0)
        # Load params for the default (first) plugin
        if self.plugin_registry.names():
            first = self.plugin_registry.get(self.plugin_registry.names()[0])
            self._param_panel.load_plugin(first)

    def _on_algorithm_changed(self, _index: int) -> None:
        name = self.combo_algorithm.currentText()
        if name in self.plugin_registry:
            plugin = self.plugin_registry.get(name)
            self._param_panel.load_plugin(plugin)

    # ── MorphEngine wiring ─────────────────────────────────────────────

    def _connect_morph_engine(self) -> None:
        self.morph_engine.progress.connect(self._on_morph_progress)
        self.morph_engine.finished.connect(self._on_morph_finished)
        self.morph_engine.error.connect(self._on_morph_error)
        self.btn_recompute.clicked.connect(self._on_recompute)

    def _on_recompute(self) -> None:
        if not self.project.ready_to_morph or self.morph_engine.is_running:
            return

        plugin_name = self.combo_algorithm.currentText()
        plugin = self.plugin_registry.get(plugin_name)
        steps = self.spin_steps.value()

        # Notify if A and B have different durations (match_lengths will pad)
        len_a = len(self.project.audio_a)
        len_b = len(self.project.audio_b)
        if len_a != len_b:
            sr = self.project.sample_rate
            dur_a = len_a / sr
            dur_b = len_b / sr
            self.statusBar().showMessage(
                f"Note: A ({dur_a:.2f}s) ≠ B ({dur_b:.2f}s) — shorter will be zero-padded. "
                f"Computing {steps} steps…"
            )
        else:
            self.statusBar().showMessage(
                f"Computing {steps} morph steps using {plugin_name}…"
            )

        self.btn_recompute.setEnabled(False)
        self.btn_recompute.setText("⟳  Computing…")

        self.morph_engine.compute(
            plugin=plugin,
            audio_a=self.project.audio_a,
            audio_b=self.project.audio_b,
            steps=steps,
            sample_rate=self.project.sample_rate,
            params=self._param_panel.get_params(),
            dtw=self.chk_dtw.isChecked(),
        )

    def _on_morph_progress(self, value: int) -> None:
        self.statusBar().showMessage(f"Computing… {value}%")

    def _on_morph_finished(self, steps: list) -> None:
        self.project.morph_steps = steps
        self.project.steps = len(steps)
        self.btn_recompute.setText("⟳  Recompute")
        self._refresh_recompute_button()
        self.btn_play_all.setEnabled(True)
        self.btn_stop.setEnabled(True)
        self.btn_export.setEnabled(True)
        self._act_save.setEnabled(True)
        self.statusBar().showMessage(
            f"Ready — {len(steps)} morph steps computed  "
            f"({self.combo_algorithm.currentText()})"
        )
        self._set_unsaved(True)
        self._show_step_grid(steps)

    def _show_step_grid(self, steps: list) -> None:
        """Inject (or refresh) the StepGrid into the step grid area."""
        self._step_grid_placeholder.hide()

        if self._step_grid is None:
            self._step_grid = StepGrid()
            self._step_grid.step_clicked.connect(self._on_step_clicked)
            self._step_grid_layout.addWidget(self._step_grid)

        self._step_grid.load_steps(steps, self.project.sample_rate)

    def _on_step_clicked(self, idx: int) -> None:
        steps = self.project.morph_steps
        if 0 <= idx < len(steps):
            self.audio_engine.play(steps[idx], self.project.sample_rate)

    def _on_morph_error(self, message: str) -> None:
        from PySide6.QtWidgets import QMessageBox
        self.btn_recompute.setText("⟳  Recompute")
        self._refresh_recompute_button()
        self.statusBar().showMessage("Morph computation failed")
        QMessageBox.critical(self, "Morph Error", f"Could not compute morph steps:\n\n{message}")

    # ── BPM engine wiring ──────────────────────────────────────────────

    def _connect_bpm_engine(self) -> None:
        self.bpm_engine.step_advance.connect(self._on_step_advance)
        self.bpm_engine.beat_tick.connect(self._on_beat_tick)
        self.bpm_engine.playback_stopped.connect(self._on_playback_stopped)
        self.btn_play_all.clicked.connect(self._on_play_all)
        self.btn_stop.clicked.connect(self._on_stop)
        self.btn_tap.clicked.connect(self._on_tap)
        self.spin_beats.valueChanged.connect(self._on_beats_changed)

    def _on_play_all(self) -> None:
        if not self.project.has_morph_steps:
            return
        if self._playing:
            self._on_stop()
            return
        self._playing = True
        self.btn_play_all.setText("■  Stop")
        self.btn_play_all.setProperty("accent", "false")
        self.btn_play_all.style().unpolish(self.btn_play_all)
        self.btn_play_all.style().polish(self.btn_play_all)
        self._rebuild_beat_dots(self.spin_beats.value())
        self.bpm_engine.configure(
            bpm=self.spin_bpm.value(),
            beats_per_step=self.spin_beats.value(),
            total_steps=len(self.project.morph_steps),
            loop=self.chk_loop.isChecked(),
        )
        self.bpm_engine.start_playback()

    def _on_stop(self) -> None:
        self._playing = False
        self.bpm_engine.stop_playback()
        self.audio_engine.stop()
        self._reset_playback_ui()

    def _on_step_advance(self, idx: int) -> None:
        steps = self.project.morph_steps
        if 0 <= idx < len(steps):
            self.audio_engine.play(steps[idx], self.project.sample_rate)
        if self._step_grid is not None:
            self._step_grid.set_active(idx)

    def _on_beat_tick(self, beat: int) -> None:
        n = len(self._beat_dots)
        if n == 0:
            return
        pos = beat % n
        for i, dot in enumerate(self._beat_dots):
            if i == pos:
                dot.setStyleSheet("color: #00c8c8; font-size: 13px;")
            else:
                dot.setStyleSheet("color: #252525; font-size: 11px;")

    def _on_playback_stopped(self) -> None:
        self._playing = False
        self._reset_playback_ui()
        if self._step_grid is not None:
            self._step_grid.clear_active()

    def _reset_playback_ui(self) -> None:
        self.btn_play_all.setText("▶  Play All")
        self.btn_play_all.setProperty("accent", "true")
        self.btn_play_all.style().unpolish(self.btn_play_all)
        self.btn_play_all.style().polish(self.btn_play_all)
        for dot in self._beat_dots:
            dot.setStyleSheet("color: #252525; font-size: 11px;")

    def _on_tap(self) -> None:
        now = time.perf_counter()
        self._tap_times.append(now)
        if len(self._tap_times) >= 2:
            intervals = [
                self._tap_times[i] - self._tap_times[i - 1]
                for i in range(1, len(self._tap_times))
            ]
            avg = sum(intervals) / len(intervals)
            bpm = round(60.0 / avg)
            bpm = max(20, min(300, bpm))
            self.spin_bpm.setValue(bpm)

    def _on_beats_changed(self, value: int) -> None:
        if not self._playing:
            self._rebuild_beat_dots(value)

    def _rebuild_beat_dots(self, count: int) -> None:
        """Replace beat indicator dots with `count` new dots (max 16)."""
        for dot in self._beat_dots:
            self._beat_layout.removeWidget(dot)
            dot.deleteLater()
        self._beat_dots.clear()
        n = max(1, min(count, 16))
        for _ in range(n):
            dot = QLabel("●")
            dot.setStyleSheet("color: #252525; font-size: 11px;")
            self._beat_dots.append(dot)
            self._beat_layout.addWidget(dot)

    # ── Step tile click (stops BPM playback first) ─────────────────────

    def _on_step_clicked(self, idx: int) -> None:
        if self._playing:
            self._on_stop()
        steps = self.project.morph_steps
        if 0 <= idx < len(steps):
            self.audio_engine.play(steps[idx], self.project.sample_rate)

    # ── Export ────────────────────────────────────────────────────────────

    def _connect_export_engine(self) -> None:
        self.export_engine.progress.connect(self._on_export_progress)
        self.export_engine.finished.connect(self._on_export_finished)
        self.export_engine.error.connect(self._on_export_error)
        self.btn_export.clicked.connect(self._on_export)

    def _on_export(self) -> None:
        if not self.project.has_morph_steps or self.export_engine.is_running:
            return

        folder = QFileDialog.getExistingDirectory(
            self, "Export Steps — Choose Output Folder", ""
        )
        if not folder:
            return

        from PySide6.QtWidgets import QProgressDialog
        self._export_dialog = QProgressDialog(
            "Exporting morph steps…", "Cancel", 0, 100, self
        )
        self._export_dialog.setWindowTitle("Exporting")
        self._export_dialog.setMinimumDuration(0)
        self._export_dialog.setValue(0)
        self._export_dialog.canceled.connect(self._on_export_cancel)

        self.btn_export.setEnabled(False)
        self.export_engine.export(
            steps=self.project.morph_steps,
            output_dir=folder,
            sample_rate=self.project.sample_rate,
            bit_depth=self.project.bit_depth,
        )

    def _on_export_progress(self, value: int) -> None:
        if hasattr(self, "_export_dialog"):
            self._export_dialog.setValue(value)

    def _on_export_finished(self, output_dir: str) -> None:
        if hasattr(self, "_export_dialog"):
            self._export_dialog.close()
        self.btn_export.setEnabled(True)
        n = len(self.project.morph_steps)
        self.statusBar().showMessage(f"Exported {n} steps to {output_dir}")

        from PySide6.QtCore import QUrl
        from PySide6.QtGui import QDesktopServices
        from PySide6.QtWidgets import QMessageBox
        msg = QMessageBox(self)
        msg.setWindowTitle("Export Complete")
        msg.setText(f"Exported {n} morph step{'s' if n != 1 else ''} to:")
        msg.setInformativeText(output_dir)
        msg.setIcon(QMessageBox.Icon.Information)
        open_btn = msg.addButton("Open Folder", QMessageBox.ButtonRole.ActionRole)
        msg.addButton(QMessageBox.StandardButton.Ok)
        msg.exec()
        if msg.clickedButton() is open_btn:
            QDesktopServices.openUrl(QUrl.fromLocalFile(output_dir))

    def _on_export_error(self, message: str) -> None:
        if hasattr(self, "_export_dialog"):
            self._export_dialog.close()
        self.btn_export.setEnabled(True)
        self.statusBar().showMessage("Export failed")
        from PySide6.QtWidgets import QMessageBox
        QMessageBox.critical(self, "Export Error", f"Could not export steps:\n\n{message}")

    def _on_export_cancel(self) -> None:
        self.btn_export.setEnabled(True)

    # ── File menu ──────────────────────────────────────────────────────

    def _on_new(self) -> None:
        if not self._confirm_discard():
            return
        self._on_stop()
        self.project = ProjectState()
        self.slot_a.clear()
        self.slot_b.clear()
        if self._step_grid is not None:
            self._step_grid.load_steps([], self.project.sample_rate)
            self._step_grid_placeholder.show()
        self.btn_play_all.setEnabled(False)
        self.btn_stop.setEnabled(False)
        self.btn_export.setEnabled(False)
        self._act_save.setEnabled(False)
        self._set_unsaved(False)
        self.statusBar().showMessage("New project")

    def _on_open(self) -> None:
        if not self._confirm_discard():
            return
        path, _ = QFileDialog.getOpenFileName(self, "Open Project", "", SMORPH_FILTER)
        if not path:
            return
        try:
            state = ProjectFile.load(path)
        except ProjectFileError as exc:
            QMessageBox.critical(self, "Open Error", str(exc))
            return
        self._on_stop()
        self._load_project_into_ui(state)
        self._add_to_recent(path)
        self._set_unsaved(False)

    def _on_save(self) -> None:
        if self.project.file_path:
            self._save_to(self.project.file_path)
        else:
            self._on_save_as()

    def _on_save_as(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save Project As", "", SMORPH_FILTER)
        if not path:
            return
        self._save_to(path)

    def _save_to(self, path: str) -> None:
        self.project.algorithm = self.combo_algorithm.currentText()
        self.project.algorithm_params = self._param_panel.get_params()
        self.project.bpm = self.spin_bpm.value()
        self.project.beats_per_step = self.spin_beats.value()
        self.project.loop = self.chk_loop.isChecked()
        try:
            ProjectFile.save(path, self.project)
        except ProjectFileError as exc:
            QMessageBox.critical(self, "Save Error", str(exc))
            return
        self._add_to_recent(path)
        self._set_unsaved(False)
        self.statusBar().showMessage(f"Saved — {Path(path).name}")

    # ── Project loading ────────────────────────────────────────────────

    def _load_project_into_ui(self, state: ProjectState) -> None:
        self.project = state

        # Audio slots
        if state.audio_a is not None:
            self.slot_a.set_audio(state.audio_a, state.sample_rate, state.name_a or "source_a")
        else:
            self.slot_a.clear()
        if state.audio_b is not None:
            self.slot_b.set_audio(state.audio_b, state.sample_rate, state.name_b or "source_b")
        else:
            self.slot_b.clear()

        # Settings controls
        self.spin_steps.setValue(state.steps)
        self.spin_bpm.setValue(state.bpm)
        self.spin_beats.setValue(state.beats_per_step)
        self.chk_loop.setChecked(state.loop)

        idx = self.combo_algorithm.findText(state.algorithm)
        if idx >= 0:
            self.combo_algorithm.setCurrentIndex(idx)
        if state.algorithm_params:
            self._param_panel.set_params(state.algorithm_params)

        # Step grid
        has_steps = len(state.morph_steps) > 0
        if has_steps:
            self._show_step_grid(state.morph_steps)
            self.btn_play_all.setEnabled(True)
            self.btn_stop.setEnabled(True)
            self.btn_export.setEnabled(True)
            self._act_save.setEnabled(True)
        else:
            self._step_grid_placeholder.show()
            if self._step_grid is not None:
                self._step_grid.hide()

        self._refresh_recompute_button()
        self.statusBar().showMessage(
            f"Loaded — {Path(state.file_path).name if state.file_path else 'project'}"
        )

    # ── Unsaved-changes helpers ────────────────────────────────────────

    def _set_unsaved(self, dirty: bool) -> None:
        self._unsaved_changes = dirty
        self._act_save.setEnabled(dirty or bool(self.project.file_path))
        self._update_title()

    def _update_title(self) -> None:
        base = "Sound Morpher"
        if self.project.file_path:
            name = Path(self.project.file_path).stem
            title = f"{base} — {name}"
        else:
            title = base
        if self._unsaved_changes:
            title += " *"
        self.setWindowTitle(title)

    def _confirm_discard(self) -> bool:
        """Return True if it's safe to discard the current project."""
        if not self._unsaved_changes:
            return True
        reply = QMessageBox.question(
            self,
            "Unsaved Changes",
            "The current project has unsaved changes.\nSave before continuing?",
            QMessageBox.StandardButton.Save
            | QMessageBox.StandardButton.Discard
            | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Save,
        )
        if reply == QMessageBox.StandardButton.Save:
            self._on_save()
            return not self._unsaved_changes  # False if save was cancelled
        return reply == QMessageBox.StandardButton.Discard

    # ── Recent files ───────────────────────────────────────────────────

    def _add_to_recent(self, path: str) -> None:
        recent: list[str] = self._settings.value("recent_files", [])
        if isinstance(recent, str):
            recent = [recent]
        path = str(Path(path).resolve())
        if path in recent:
            recent.remove(path)
        recent.insert(0, path)
        recent = recent[:10]
        self._settings.setValue("recent_files", recent)
        self._rebuild_recent_menu()

    def _rebuild_recent_menu(self) -> None:
        self._recent_menu.clear()
        recent: list[str] = self._settings.value("recent_files", [])
        if isinstance(recent, str):
            recent = [recent]
        if not recent:
            self._recent_menu.addAction("(none)").setEnabled(False)
            return
        for path in recent:
            label = Path(path).name
            action = self._recent_menu.addAction(label)
            action.setData(path)
            action.triggered.connect(lambda checked, p=path: self._open_recent(p))
        self._recent_menu.addSeparator()
        self._recent_menu.addAction("Clear Recent", self._clear_recent)

    def _open_recent(self, path: str) -> None:
        if not self._confirm_discard():
            return
        if not Path(path).exists():
            QMessageBox.warning(self, "File Not Found", f"Cannot find:\n{path}")
            return
        try:
            state = ProjectFile.load(path)
        except ProjectFileError as exc:
            QMessageBox.critical(self, "Open Error", str(exc))
            return
        self._on_stop()
        self._load_project_into_ui(state)
        self._add_to_recent(path)
        self._set_unsaved(False)

    def _clear_recent(self) -> None:
        self._settings.remove("recent_files")
        self._rebuild_recent_menu()

    # ── Close event ────────────────────────────────────────────────────

    def closeEvent(self, event) -> None:
        if self._unsaved_changes and not self._confirm_discard():
            event.ignore()
            return
        self._on_stop()
        self.bpm_engine.wait(500)
        event.accept()

    def _on_about(self) -> None:
        from PySide6.QtWidgets import QMessageBox
        QMessageBox.about(
            self,
            "About Sound Morpher",
            "<b>Sound Morpher</b><br>"
            "Version 1.0 (development)<br><br>"
            "Morph between two audio samples using pluggable algorithms.",
        )


# ── Helpers ────────────────────────────────────────────────────────────

def _vline() -> QFrame:
    line = QFrame()
    line.setFrameShape(QFrame.Shape.VLine)
    line.setFixedHeight(20)
    return line
