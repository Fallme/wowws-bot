from types import SimpleNamespace
from unittest.mock import patch

import pytest

from main import (
    GameWindowUnavailableWhilePaused,
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
