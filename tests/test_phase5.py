import time
import pytest
from PySide6.QtCore import QCoreApplication


def test_bpm_engine_import():
    from app.bpm_engine import BpmEngine
    assert BpmEngine is not None


def test_bpm_engine_step_signals(qt_app):
    """BpmEngine emits correct step_advance sequence."""
    from app.bpm_engine import BpmEngine

    engine = BpmEngine()
    engine.configure(bpm=600.0, beats_per_step=1, total_steps=3, loop=False)

    steps_received = []
    engine.step_advance.connect(steps_received.append)

    stopped = []
    engine.playback_stopped.connect(lambda: stopped.append(True))

    engine.start_playback()

    # Wait up to 2 seconds for playback to finish
    deadline = time.perf_counter() + 2.0
    while not stopped and time.perf_counter() < deadline:
        QCoreApplication.processEvents()
        time.sleep(0.01)

    engine.wait(500)

    assert len(steps_received) >= 3, f"Expected >=3 step signals, got {steps_received}"
    assert steps_received[0] == 0
    assert stopped, "playback_stopped never emitted"


def test_bpm_engine_loop(qt_app):
    """BpmEngine with loop=True keeps advancing beyond total_steps."""
    from app.bpm_engine import BpmEngine

    engine = BpmEngine()
    engine.configure(bpm=1200.0, beats_per_step=1, total_steps=2, loop=True)

    steps_received = []
    engine.step_advance.connect(steps_received.append)

    engine.start_playback()
    deadline = time.perf_counter() + 0.5
    while time.perf_counter() < deadline:
        QCoreApplication.processEvents()
        time.sleep(0.01)
    engine.stop_playback()
    engine.wait(500)

    assert len(steps_received) > 2, "Loop should advance past total_steps"


def test_bpm_engine_stop(qt_app):
    """Stopping BpmEngine emits playback_stopped."""
    from app.bpm_engine import BpmEngine

    # Use high BPM so the busy-wait interval is short (~100ms) and wait(500) is sufficient
    engine = BpmEngine()
    engine.configure(bpm=600.0, beats_per_step=4, total_steps=8, loop=False)

    stopped = []
    engine.playback_stopped.connect(lambda: stopped.append(True))

    engine.start_playback()
    time.sleep(0.05)
    engine.stop_playback()
    engine.wait(1000)
    QCoreApplication.processEvents()

    assert stopped, "playback_stopped not emitted after stop_playback()"


def test_tap_tempo_logic():
    """Tap intervals correctly map to BPM."""
    import time as _time
    from collections import deque

    taps: deque[float] = deque(maxlen=8)
    # Simulate 4 taps at 500 ms intervals → 120 BPM
    base = 0.0
    for i in range(4):
        taps.append(base + i * 0.5)

    intervals = [taps[i] - taps[i - 1] for i in range(1, len(taps))]
    avg = sum(intervals) / len(intervals)
    bpm = round(60.0 / avg)
    assert bpm == 120
