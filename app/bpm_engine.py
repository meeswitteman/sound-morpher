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
        self._loop_mode:      str   = "off"   # "off" | "loop" | "pingpong"
        self._reverse:        bool  = False
        self._stop_flag:      bool  = False

    # ── Configuration ──────────────────────────────────────────────────

    def configure(
        self,
        bpm:            float,
        beats_per_step: int,
        total_steps:    int,
        loop_mode:      str  = "off",
        reverse:        bool = False,
    ) -> None:
        self._bpm            = max(1.0, float(bpm))
        self._beats_per_step = max(1, int(beats_per_step))
        self._total_steps    = max(1, int(total_steps))
        self._loop_mode      = loop_mode
        self._reverse        = bool(reverse)

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
        n = self._total_steps
        direction = -1 if self._reverse else 1
        step = (n - 1) if self._reverse else 0
        beat = 0
        beat_interval = 60.0 / self._bpm

        self.step_advance.emit(step)
        self.beat_tick.emit(beat)

        next_beat = time.perf_counter() + beat_interval

        while not self._stop_flag:
            remaining = next_beat - time.perf_counter()
            if remaining > 0.003:
                self.msleep(int((remaining - 0.002) * 1000))

            while time.perf_counter() < next_beat:
                if self._stop_flag:
                    self.playback_stopped.emit()
                    return

            next_beat += beat_interval
            beat = (beat + 1) % self._beats_per_step

            if beat == 0:
                step += direction

                if step >= n or step < 0:
                    if self._loop_mode == "pingpong":
                        direction = -direction
                        step += 2 * direction   # undo overshoot + one step back
                    elif self._loop_mode == "loop":
                        step = 0 if direction > 0 else n - 1
                    else:
                        # Ensure the final step lights up before stopping
                        self.step_advance.emit(max(0, min(step - direction, n - 1)))
                        self.playback_stopped.emit()
                        return

                self.step_advance.emit(step)

            self.beat_tick.emit(beat)

        self.playback_stopped.emit()
