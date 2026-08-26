from types import SimpleNamespace
from unittest.mock import patch

from core.frame_guard import CaptureFault
from core.ui import ScreenState
from main import recover_after_battle_fault, recover_current_scene


class SequenceVision:
    def __init__(self, states):
        self.states = iter(states)

    def grab(self, _hwnd, *, allow_stale=False):
        value = next(self.states)
        if isinstance(value, Exception):
            raise value
        return value

    @staticmethod
    def classify_screen(image):
        return image


def make_bot(states):
    return SimpleNamespace(hwnd=1, vision=SequenceVision(states))


def test_recovery_requires_two_matching_frames_and_does_not_click():
    bot = make_bot(
        [ScreenState.RESULTS, ScreenState.BATTLE, ScreenState.BATTLE]
    )

    with (
        patch("main.time.sleep", return_value=None),
        patch("main.return_to_port") as return_to_port,
    ):
        state = recover_current_scene(bot, attempts=3)

    assert state == ScreenState.BATTLE
    return_to_port.assert_not_called()


def test_recovery_ignores_capture_fault_and_then_confirms_results():
    bot = make_bot(
        [
            CaptureFault("temporary"),
            ScreenState.UNKNOWN,
            ScreenState.RESULTS,
            ScreenState.RESULTS,
        ]
    )

    with patch("main.time.sleep", return_value=None):
        state = recover_current_scene(bot, attempts=4)

    assert state == ScreenState.RESULTS


def test_recovery_returns_unknown_for_unstable_observations():
    bot = make_bot(
        [ScreenState.PORT, ScreenState.BATTLE, ScreenState.RESULTS]
    )

    with patch("main.time.sleep", return_value=None):
        state = recover_current_scene(bot, attempts=3)

    assert state == ScreenState.UNKNOWN


def test_battle_fault_retries_unknown_until_battle_is_stable():
    bot = SimpleNamespace(hwnd=1, vision=SimpleNamespace())

    with (
        patch(
            "main.recover_current_scene",
            side_effect=[
                ScreenState.UNKNOWN,
                ScreenState.UNKNOWN,
                ScreenState.BATTLE,
            ],
        ) as classify,
        patch("main.time.sleep", return_value=None),
    ):
        state = recover_after_battle_fault(bot, attempts=3)

    assert state == ScreenState.BATTLE
    assert classify.call_count == 3


def test_battle_fault_waits_through_loading_and_resumes_battle():
    bot = SimpleNamespace(hwnd=1, vision=SimpleNamespace())

    with (
        patch(
            "main.recover_current_scene",
            return_value=ScreenState.LOADING,
        ),
        patch("main.wait_for_battle", return_value=True) as wait_for_battle,
    ):
        state = recover_after_battle_fault(bot, attempts=3)

    assert state == ScreenState.BATTLE
    wait_for_battle.assert_called_once()


def test_battle_fault_uses_global_return_to_port_after_unknown_retry_limit():
    bot = SimpleNamespace(hwnd=1, vision=SimpleNamespace())

    with (
        patch(
            "main.recover_current_scene",
            return_value=ScreenState.UNKNOWN,
        ) as classify,
        patch("main.time.sleep", return_value=None),
        patch("main.return_to_port", return_value=True) as return_to_port,
    ):
        state = recover_after_battle_fault(bot, attempts=3)

    assert state == ScreenState.PORT
    assert classify.call_count == 3
    return_to_port.assert_called_once_with(bot, attempts=3)
