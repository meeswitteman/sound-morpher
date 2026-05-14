from __future__ import annotations

import time

from PySide6.QtCore import QThread, Signal


class BpmEngine(QThread):
    """Fires beat and step-advance signals at BPM-accurate intervals.

    Uses perf_counter for timing accuracy:
      - sleeps to ~2 ms before the next beat, then busy-waits the remainder.

    All signals are emitted from the worker thread; Qt delivers them to the
    main thread via queued connections automatically.
    """

    beat_tick    = Signal(int)   # beat index within current step  (0-based)
    step_advance = Signal(int)   # new step index to play          (0-based)
    playback_stopped = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._bpm:            float = 120.0
        self._beats_per_step: int   = 4
        self._total_steps:    int   = 8
        self._loop:           bool  = False
        self._start_step:     int   = 0
        self._stop_flag:      bool  = False

    # ── Configuration ──────────────────────────────────────────────────

    def configure(
        self,
        bpm:            float,
        beats_per_step: int,
        total_steps:    int,
        loop:           bool,
        start_step:     int = 0,
    ) -> None:
        self._bpm            = max(1.0, float(bpm))
        self._beats_per_step = max(1, int(beats_per_step))
        self._total_steps    = max(1, int(total_steps))
        self._loop           = bool(loop)
        self._start_step     = max(0, min(start_step, total_steps - 1))

    # ── Control ────────────────────────────────────────────────────────

    def start_playback(self) -> None:
        if self.isRunning():
            self.stop_playback()
            self.wait(500)
        self._stop_flag = False
        self.start()

    def stop_playback(self) -> None:
        self._stop_flag = True

    # ── Thread body ────────────────────────────────────────────────────

    def run(self) -> None:
        step = self._start_step
        beat = 0
        beat_interval = 60.0 / self._bpm

        # Emit first step + first beat immediately
        self.step_advance.emit(step)
        self.beat_tick.emit(beat)

        next_beat = time.perf_counter() + beat_interval

        while not self._stop_flag:
            # Sleep until ~2 ms before the next beat
            remaining = next_beat - time.perf_counter()
            if remaining > 0.003:
                self.msleep(int((remaining - 0.002) * 1000))

            # Busy-wait the last milliseconds for accuracy
            while time.perf_counter() < next_beat:
                if self._stop_flag:
                    self.playback_stopped.emit()
                    return

            next_beat += beat_interval
            beat = (beat + 1) % self._beats_per_step

            if beat == 0:
                # Advance to next step
                step += 1
                if step >= self._total_steps:
                    if self._loop:
                        step = 0
                    else:
                        self.step_advance.emit(step - 1)   # ensure last step lit
                        self.playback_stopped.emit()
                        return
                self.step_advance.emit(step)

            self.beat_tick.emit(beat)

        self.playback_stopped.emit()
