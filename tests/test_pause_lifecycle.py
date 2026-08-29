from types import SimpleNamespace
from unittest.mock import patch

import pytest

import core.window as game_window

from main import (
    GameWindowUnavailableWhilePaused,
    ensure_bound_game_foreground,
    refresh_game_window,
    shutdown_bot,
    wait_for_web_resume,
)


class RecordingReporter:
    def __init__(self):
        self.updates = []

    def update(self, state, message, **values):
        self.updates.append((state, message, values))


class PausedLimits:
    def __init__(self, pause_values=None):
        self.pause_values = iter(pause_values) if pause_values is not None else None

    def pause_requested(self):
        if self.pause_values is None:
            return True
        return next(self.pause_values)

    @staticmethod
    def stop_requested():
        return False


def test_missing_game_window_ends_paused_worker_without_input_or_focus():
    reporter = RecordingReporter()
    controller = SimpleNamespace(commands=[])
    bot = SimpleNamespace(hwnd=1234, gamepad=controller)

    with (
        patch("main.is_game_window_alive", return_value=False),
        patch("main.time.monotonic", side_effect=[100.0, 105.1]),
        patch("main.time.sleep", return_value=None),
        patch("main.activate_window") as activate,
        pytest.raises(GameWindowUnavailableWhilePaused),
    ):
        wait_for_web_resume(PausedLimits(), reporter, bot)

    activate.assert_not_called()
    assert controller.commands == []
    state, _message, values = reporter.updates[-1]
    assert state == "failed"
    assert values["error"] == "game_window_unavailable_while_paused"
    assert values["movement_mode"] == "idle"


def test_transient_window_check_failure_resets_after_handle_recovers():
    reporter = RecordingReporter()
    bot = SimpleNamespace(hwnd=1234, gamepad=SimpleNamespace())

    with (
        patch("main.is_game_window_alive", side_effect=[False, True]),
        patch("main.time.monotonic", side_effect=[100.0, 104.9]),
        patch("main.time.sleep", return_value=None),
    ):
        resumed = wait_for_web_resume(
            PausedLimits([True, True, True, False]),
            reporter,
            bot,
        )

    assert resumed
    assert reporter.updates[-1][0] == "preparing"


def test_shutdown_with_stale_handle_closes_resources_without_game_input():
    class RecordingBot:
        hwnd = 1234

        def __init__(self):
            self.stop_calls = []

        def stop(self, *, release_input=True):
            self.stop_calls.append(release_input)

    bot = RecordingBot()
    with (
        patch("main.is_game_window_alive", return_value=False),
        patch("main.activate_window") as activate,
    ):
        shutdown_bot(bot)

    activate.assert_not_called()
    assert bot.stop_calls == [False]


def test_common_foreground_gate_never_focuses_or_maximizes_while_paused():
    intervention = SimpleNamespace(
        poll=lambda _controller, _now: True,
    )
    bot = SimpleNamespace(
        hwnd=1234,
        gamepad=SimpleNamespace(),
        intervention=intervention,
        mark_manual_pause=lambda: None,
    )
    with (
        patch("main.ensure_game_window_foreground") as focus,
        patch("main.maximize_game_window") as maximize,
        patch("main.find_game_window") as find_window,
    ):
        assert not ensure_bound_game_foreground(bot)
        assert not refresh_game_window(bot)

    focus.assert_not_called()
    maximize.assert_not_called()
    find_window.assert_not_called()


def test_shutdown_during_pause_never_focuses_game_window():
    stop_calls = []
    bot = SimpleNamespace(
        hwnd=1234,
        gamepad=SimpleNamespace(),
        intervention=SimpleNamespace(poll=lambda _controller, _now: True),
        mark_manual_pause=lambda: None,
        stop=lambda *, release_input=True: stop_calls.append(release_input),
    )
    with patch("main.ensure_game_window_foreground") as focus:
        shutdown_bot(bot)

    focus.assert_not_called()
    assert stop_calls == [False]


def test_low_level_pause_guard_blocks_every_window_side_effect():
    game_window.set_interaction_pause_guard(lambda: True)
    try:
        with (
            patch("core.window.is_game_window") as is_game,
            patch("core.window.activate_window") as activate,
            patch("core.window.ctypes.windll.user32.ShowWindow") as show,
            patch("core.window.ctypes.windll.user32.GetCursorPos") as cursor,
        ):
            assert not game_window.ensure_game_window_foreground(1234)
            assert not game_window.maximize_game_window(1234)
            assert not game_window.physical_click(100, 100, hwnd=1234)
            assert not game_window.physical_scroll(100, 100, 1, hwnd=1234)
            assert not game_window.window_message_click(1234, 100, 100)

        is_game.assert_not_called()
        activate.assert_not_called()
        show.assert_not_called()
        cursor.assert_not_called()
    finally:
        game_window.set_interaction_pause_guard(None)


def test_foreground_retry_stops_if_keyboard_pause_arrives_mid_operation():
    checks = iter([False, True])
    game_window.set_interaction_pause_guard(lambda: next(checks))
    try:
        with (
            patch("core.window.is_game_window", return_value=True),
            patch("core.window.activate_window") as activate,
        ):
            assert not game_window.ensure_game_window_foreground(1234)
        activate.assert_not_called()
    finally:
        game_window.set_interaction_pause_guard(None)
