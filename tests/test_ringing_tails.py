"""Play All lets each step ring out under the next instead of cutting it off.

Playback used sd.play(), one stream per sound: starting the next step stopped
the previous one mid-tail. It now goes through one shared output stream and a
mixer that sums any number of voices.
"""
from __future__ import annotations

import time

import numpy as np
import pytest
from PySide6.QtCore import QCoreApplication

from app.audio_engine import AudioEngine, VoiceMixer


# ── The mixer ─────────────────────────────────────────────────────────────────

def test_overlapping_voices_are_summed():
    mixer = VoiceMixer(2)
    mixer.add(np.full((100, 2), 0.25, dtype=np.float32))
    mixer.add(np.full((50, 2), 0.5, dtype=np.float32))
    block = mixer.render(80)
    np.testing.assert_allclose(block[:50], 0.75)
    np.testing.assert_allclose(block[50:], 0.25)


def test_a_voice_plays_to_its_end_and_then_drops_out():
    mixer = VoiceMixer(2)
    tail = np.linspace(1.0, 0.0, 250, dtype=np.float32).reshape(-1, 1)
    mixer.add(tail)
    heard = np.concatenate([mixer.render(64) for _ in range(5)])
    np.testing.assert_allclose(heard[:250, 0], tail.ravel())
    np.testing.assert_array_equal(heard[250:], 0.0)
    assert not mixer.active


def test_mono_goes_to_both_channels():
    mixer = VoiceMixer(2)
    mixer.add(np.full(10, 0.3, dtype=np.float32))
    block = mixer.render(10)
    np.testing.assert_allclose(block[:, 0], 0.3)
    np.testing.assert_allclose(block[:, 1], 0.3)


def test_sum_is_clipped_not_wrapped():
    mixer = VoiceMixer(2)
    mixer.add(np.full((10, 2), 0.8, dtype=np.float32))
    mixer.add(np.full((10, 2), 0.8, dtype=np.float32))
    assert float(np.max(mixer.render(10))) == pytest.approx(1.0)


def test_clear_silences_everything():
    mixer = VoiceMixer(2)
    mixer.add(np.ones((100, 2), dtype=np.float32) * 0.5)
    mixer.add(np.ones((100, 2), dtype=np.float32) * 0.5)
    mixer.clear()
    assert not mixer.active
    np.testing.assert_array_equal(mixer.render(10), 0.0)


# ── The engine, with a fake output stream ─────────────────────────────────────

class _FakeStream:
    opened: list["_FakeStream"] = []

    def __init__(self, samplerate, channels, dtype, callback):
        self.samplerate = samplerate
        self.callback = callback
        self.closed = False
        _FakeStream.opened.append(self)

    def start(self):
        pass

    def stop(self):
        pass

    def close(self):
        self.closed = True


@pytest.fixture
def engine(monkeypatch):
    import app.audio_engine as ae

    _FakeStream.opened = []
    monkeypatch.setattr(ae.sd, "OutputStream", _FakeStream)
    eng = AudioEngine()
    yield eng
    eng.close()


def _pull(stream: _FakeStream, frames: int) -> np.ndarray:
    out = np.zeros((frames, 2), dtype=np.float32)
    stream.callback(out, frames, None, None)
    return out


def test_play_overlapping_keeps_the_previous_sound(engine):
    engine.play_overlapping(np.full((100, 2), 0.25, dtype=np.float32), 44100)
    engine.play_overlapping(np.full((100, 2), 0.5, dtype=np.float32), 44100)
    np.testing.assert_allclose(_pull(_FakeStream.opened[-1], 10), 0.75)


def test_play_still_replaces_what_is_playing(engine):
    """Clicking a tile or a slot's play button is a standalone preview."""
    engine.play(np.full((100, 2), 0.25, dtype=np.float32), 44100)
    engine.play(np.full((100, 2), 0.5, dtype=np.float32), 44100)
    np.testing.assert_allclose(_pull(_FakeStream.opened[-1], 10), 0.5)


def test_stop_silences_ringing_tails(engine):
    engine.play_overlapping(np.ones((100, 2), dtype=np.float32) * 0.3, 44100)
    engine.play_overlapping(np.ones((100, 2), dtype=np.float32) * 0.3, 44100)
    engine.stop()
    assert not engine.is_playing()
    np.testing.assert_array_equal(_pull(_FakeStream.opened[-1], 10), 0.0)


def test_one_stream_is_shared_until_the_rate_changes(engine):
    engine.play(np.zeros((10, 2), dtype=np.float32), 44100)
    engine.play_overlapping(np.zeros((10, 2), dtype=np.float32), 44100)
    assert len(_FakeStream.opened) == 1
    engine.play(np.zeros((10, 2), dtype=np.float32), 48000)
    assert len(_FakeStream.opened) == 2
    assert _FakeStream.opened[0].closed
    assert _FakeStream.opened[1].samplerate == 48000


def test_close_releases_the_device(engine):
    engine.play(np.zeros((10, 2), dtype=np.float32), 44100)
    engine.close()
    assert _FakeStream.opened[0].closed


# ── Play All wiring ───────────────────────────────────────────────────────────

def test_play_all_steps_ring_out(qt_app, tmp_path, monkeypatch):
    from PySide6.QtCore import QSettings

    import app.main_window as mw

    ini = str(tmp_path / "settings.ini")
    monkeypatch.setattr(mw, "QSettings", lambda *_: QSettings(ini, QSettings.Format.IniFormat))
    win = mw.MainWindow()
    calls = []
    monkeypatch.setattr(win.audio_engine, "play_overlapping", lambda a, sr: calls.append(("overlap", len(a))))
    monkeypatch.setattr(win.audio_engine, "play", lambda a, sr: calls.append(("replace", len(a))))
    win.project.morph_steps = [np.zeros((100 + i, 2), dtype=np.float32) for i in range(3)]
    win._playing = True              # as during Play All

    for idx in range(3):
        win._on_step_advance(idx)

    assert calls == [("overlap", 100), ("overlap", 101), ("overlap", 102)]
    win.close()
    win.deleteLater()


def test_last_step_is_started_only_once(qt_app):
    """The sequencer used to re-announce the last step as it stopped, which
    restarted its audio; with ringing tails it would sound twice."""
    from app.bpm_engine import BpmEngine

    engine = BpmEngine()
    engine.configure(bpm=1200.0, beats_per_step=1, total_steps=3, loop_mode="off")
    received, stopped = [], []
    engine.step_advance.connect(received.append)
    engine.playback_stopped.connect(lambda: stopped.append(True))

    engine.start_playback()
    deadline = time.perf_counter() + 2.0
    while not stopped and time.perf_counter() < deadline:
        QCoreApplication.processEvents()
        time.sleep(0.01)
    engine.wait(500)

    assert stopped
    assert received == [0, 1, 2]


# ── One Stop button ───────────────────────────────────────────────────────────

@pytest.fixture
def playing_window(qt_app, tmp_path, monkeypatch):
    """MainWindow with three short steps, fast tempo, and no real audio."""
    from PySide6.QtCore import QSettings

    import app.main_window as mw

    ini = str(tmp_path / "settings.ini")
    monkeypatch.setattr(mw, "QSettings", lambda *_: QSettings(ini, QSettings.Format.IniFormat))
    win = mw.MainWindow()
    started = []
    monkeypatch.setattr(win.audio_engine, "play_overlapping", lambda a, sr: started.append(len(a)))
    monkeypatch.setattr(win.audio_engine, "stop", lambda: None)
    win.project.morph_steps = [np.zeros((100 + i, 2), dtype=np.float32) for i in range(3)]
    win.btn_play_all.setEnabled(True)
    win.spin_bpm.setValue(300)
    win.spin_beats.setValue(1)
    win.started = started
    yield win
    win._on_stop()
    win.bpm_engine.wait(1000)
    win.close()
    win.deleteLater()


def _pump(seconds: float) -> None:
    deadline = time.perf_counter() + seconds
    while time.perf_counter() < deadline:
        QCoreApplication.processEvents()
        time.sleep(0.005)


def _stop_buttons(win) -> list:
    from PySide6.QtWidgets import QPushButton

    return [b for b in win.findChildren(QPushButton) if "Stop" in b.text() and b.isVisible()]


def test_play_all_lights_up_instead_of_turning_into_a_stop_button(playing_window):
    win = playing_window
    win.show()
    label = win.btn_play_all.text()

    win._on_play_all()
    _pump(0.05)
    assert win.btn_play_all.text() == label
    assert win.btn_play_all.property("playing") == "true"
    assert len(_stop_buttons(win)) == 1

    win._on_stop()
    win.bpm_engine.wait(1000)
    _pump(0.05)
    assert win.btn_play_all.property("playing") == "false"


def test_play_all_while_playing_restarts_from_the_first_step(playing_window):
    win = playing_window
    win._on_play_all()
    deadline = time.perf_counter() + 5.0
    while len(win.started) < 2 and time.perf_counter() < deadline:
        _pump(0.01)
    assert win.started[:2] == [100, 101]

    win.started.clear()
    win._on_play_all()
    _pump(0.05)
    # Restarted at step 0, and the old run's late stop signal did not reset it.
    assert win.started[0] == 100
    assert win._playing
    assert win.btn_play_all.property("playing") == "true"


def test_sequence_end_turns_the_light_off(playing_window):
    win = playing_window
    win._on_play_all()
    deadline = time.perf_counter() + 5.0    # 3 steps at 0.2 s, slack for load
    while win._playing and time.perf_counter() < deadline:
        _pump(0.02)
    assert not win._playing
    assert win.btn_play_all.property("playing") == "false"
    assert win.started == [100, 101, 102]


def test_a_step_announced_just_before_stop_is_not_played(playing_window):
    """The sequencer thread can post a step right before it sees the stop."""
    win = playing_window
    win._on_play_all()
    _pump(0.02)
    win._on_stop()
    win.started.clear()
    win._on_step_advance(2)          # the late delivery
    assert win.started == []
