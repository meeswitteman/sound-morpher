from __future__ import annotations

from typing import Any

import numpy as np
from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal

from plugins.base import (
    MorphPlugin,
    dtw_align,
    limit_peaks,
    match_lengths,
    match_step_loudness,
)


def with_original_endpoints(
    steps: list[np.ndarray],
    audio_a: np.ndarray,
    audio_b: np.ndarray,
) -> list[np.ndarray]:
    """Replace the first and last step with the untouched sources.

    The first step of a morph is sound A and the last is sound B, exactly as
    the user loaded and edited them, and they must be heard in full. Nothing
    downstream of the source slots may change them: not a plugin that
    resynthesises its endpoints (the LPC and WORLD vocoders did, and LPC lost
    B's tail entirely when A was shorter), not the zero-padding that equalises
    lengths, not Stretch to Fit or DTW Align, and not level matching or the
    limiter. Enforcing it here, after all of those, makes it hold for every
    plugin, including ones not written yet.

    The endpoints therefore keep their own length. Every consumer (tiles,
    playback, export, project files) handles steps of different lengths.
    """
    if not steps:
        return steps
    out = list(steps)
    out[0] = np.array(audio_a, dtype=np.float32, copy=True)
    if len(out) > 1:
        out[-1] = np.array(audio_b, dtype=np.float32, copy=True)
    return out


class _Signals(QObject):
    progress = Signal(int)        # 0–100
    finished = Signal(list)       # list[np.ndarray]
    error = Signal(str)
    cancelled = Signal()


class _Worker(QRunnable):
    def __init__(
        self,
        plugin: MorphPlugin,
        audio_a: np.ndarray,
        audio_b: np.ndarray,
        steps: int,
        sample_rate: int,
        params: dict[str, Any],
        dtw: bool = False,
        level_match: bool = True,
        stretch_to_fit: bool = False,
    ) -> None:
        super().__init__()
        self.signals = _Signals()
        self._plugin = plugin
        self._audio_a = audio_a
        self._audio_b = audio_b
        self._steps = steps
        self._sample_rate = sample_rate
        self._params = params
        self._dtw = dtw
        self._level_match = level_match
        self._stretch_to_fit = stretch_to_fit
        self.setAutoDelete(True)

    def run(self) -> None:
        try:
            self.signals.progress.emit(0)
            a, b = self._audio_a, self._audio_b
            # Length matching happens here rather than inside each plugin so the
            # user's choice applies uniformly. Plugins still call match_lengths()
            # themselves; by then it is a no-op.
            if self._stretch_to_fit:
                a, b = match_lengths(a, b, mode="stretch")
            if self._dtw:
                a, b = dtw_align(a, b, self._sample_rate)

            steps = self._steps

            def _progress(completed: int) -> None:
                self.signals.progress.emit(int(completed / steps * 100))

            result = self._plugin.morph(
                a,
                b,
                steps,
                self._sample_rate,
                progress_cb=_progress,
                **self._params,
            )
            if len(result) != steps:
                raise ValueError(
                    f"Plugin '{self._plugin.name}' returned {len(result)} steps, "
                    f"expected {steps}"
                )
            if self._level_match:
                # No shared trim: the endpoints are fixed at their original
                # level (below), so trimming only the steps in between would
                # put a dip of up to 3 dB into the loudness line next to them.
                # Overshoot in those steps goes to the limiter instead.
                result = match_step_loudness(
                    result, a, b, max_trim_db=0.0, sample_rate=self._sample_rate
                )
            else:
                # Without level matching nothing else guards the ceiling, and a
                # float export would carry overs straight into the file. The
                # limiter returns steps that are already under it untouched.
                result = [limit_peaks(s, self._sample_rate) for s in result]
            self.signals.finished.emit(
                with_original_endpoints(result, self._audio_a, self._audio_b)
            )
        except Exception as exc:
            self.signals.error.emit(str(exc))


class MorphEngine(QObject):
    """Runs morph computation on a QThreadPool worker thread."""

    progress  = Signal(int)
    finished  = Signal(list)   # list[np.ndarray]
    error     = Signal(str)
    cancelled = Signal()

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._pool = QThreadPool.globalInstance()
        self._active = False
        self._generation = 0   # incremented each compute() and cancel()
        self._worker: _Worker | None = None  # keep Python ref alive during run

    @property
    def is_running(self) -> bool:
        return self._active

    def compute(
        self,
        plugin: MorphPlugin,
        audio_a: np.ndarray,
        audio_b: np.ndarray,
        steps: int,
        sample_rate: int,
        params: dict[str, Any] | None = None,
        dtw: bool = False,
        level_match: bool = True,
        stretch_to_fit: bool = False,
    ) -> None:
        """Start async morph computation. Emits finished() or error() when done."""
        if self._active:
            return

        self._active = True
        self._generation += 1
        gen = self._generation

        worker = _Worker(
            plugin,
            audio_a,
            audio_b,
            steps,
            sample_rate,
            params or {},
            dtw=dtw,
            level_match=level_match,
            stretch_to_fit=stretch_to_fit,
        )
        worker.signals.progress.connect(self.progress)
        worker.signals.finished.connect(lambda result, g=gen: self._on_finished(result, g))
        worker.signals.error.connect(lambda msg, g=gen: self._on_error(msg, g))
        self._worker = worker  # prevent GC of worker+signals while thread runs
        self._pool.start(worker)

    def cancel(self) -> None:
        """Immediately free the engine for new work; stale result is discarded on arrival."""
        self._active = False
        self._generation += 1   # invalidate the running worker's generation

    def _on_finished(self, result: list, gen: int) -> None:
        self._worker = None
        if gen != self._generation:
            self.cancelled.emit()
            return
        self._active = False
        self.finished.emit(result)

    def _on_error(self, msg: str, gen: int) -> None:
        self._worker = None
        if gen != self._generation:
            return
        self._active = False
        self.error.emit(msg)
