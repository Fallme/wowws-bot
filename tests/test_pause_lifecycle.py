from types import SimpleNamespace
from unittest.mock import patch

import core.window as game_window

from main import (
    ensure_bound_game_foreground,
    refresh_game_window,
    restore_game_foreground_after_pause,
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


def test_missing_game_window_during_pause_remains_recoverable():
    reporter = RecordingReporter()
    controller = SimpleNamespace(commands=[])
    intervention = SimpleNamespace(
        poll=lambda _controller, _now=None: False,
        latched=False,
        last_trigger="window_switch",
        resumed_from_web=False,
    )
    bot = SimpleNamespace(
        hwnd=1234,
        gamepad=controller,
        intervention=intervention,
    )

    with (
        patch("main.is_game_window_alive", return_value=False),
        patch("main.time.sleep", return_value=None),
        patch("main.restore_game_foreground_after_pause", return_value=True) as restore,
    ):
        resumed = wait_for_web_resume(
            PausedLimits([True, True, False]),
            reporter,
            bot,
        )

    assert resumed
    assert resumed.resumed
    restore.assert_called_once()
    assert controller.commands == []
    assert not any(state == "failed" for state, _message, _values in reporter.updates)
    assert any("任务已保留" in message for _state, message, _values in reporter.updates)


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
    assert resumed.resumed
    assert reporter.updates[-1][0] == "preparing"


def test_clear_pause_gate_does_not_request_scene_recovery():
    reporter = RecordingReporter()
    bot = SimpleNamespace(hwnd=1234, gamepad=SimpleNamespace())

    resumed = wait_for_web_resume(
        PausedLimits([False]),
        reporter,
        bot,
    )

    assert resumed
    assert not resumed.resumed
    assert reporter.updates == []


def test_new_activity_during_focus_restore_keeps_pause_gate_closed():
    reporter = RecordingReporter()
    pause_values = [False, False, False, False]
    polls = iter([True, False, True, False])
    intervention = SimpleNamespace(
        poll=lambda _controller, _now=None: next(polls),
        latched=False,
        last_trigger="window_switch",
        resumed_from_web=False,
    )
    bot = SimpleNamespace(
        hwnd=1234,
        gamepad=SimpleNamespace(),
        intervention=intervention,
    )

    with (
        patch("main.restore_game_foreground_after_pause", side_effect=[False, True]) as restore,
        patch("main.time.sleep", return_value=None),
    ):
        resumed = wait_for_web_resume(
            PausedLimits(pause_values),
            reporter,
            bot,
        )

    assert resumed
    assert resumed.resumed
    assert restore.call_count == 2
    assert any("继续等待5秒静默" in message for _state, message, _values in reporter.updates)


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


def test_pause_release_restores_foreground_only_after_gate_is_clear():
    paused = [True, False, False]
    bot = SimpleNamespace(
        hwnd=1234,
        gamepad=SimpleNamespace(),
        intervention=SimpleNamespace(
            poll=lambda _controller, _now: paused.pop(0)
        ),
        mark_manual_pause=lambda: None,
    )

    with patch("main.ensure_game_window_foreground", return_value=True) as focus:
        assert not restore_game_foreground_after_pause(bot, "自动暂停已解除")
        focus.assert_not_called()
        assert restore_game_foreground_after_pause(bot, "自动暂停已解除")

    focus.assert_called_once_with(1234)
