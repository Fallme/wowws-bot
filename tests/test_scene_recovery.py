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


def test_battle_fault_never_returns_to_port_before_confirmed_results():
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

    assert state == ScreenState.UNKNOWN
    assert classify.call_count == 3
    return_to_port.assert_not_called()


def test_active_round_rejects_stable_port_until_results_are_seen():
    bot = make_bot([ScreenState.PORT, ScreenState.PORT, ScreenState.PORT])

    with patch("main.time.sleep", return_value=None):
        state = recover_current_scene(
            bot,
            attempts=3,
            round_in_progress=True,
        )

    assert state == ScreenState.UNKNOWN


def test_active_round_accepts_real_port_after_result_was_missed():
    frame = object()

    class ConfirmedPortVision:
        @staticmethod
        def grab(_hwnd, *, allow_stale=False):
            return frame

        @staticmethod
        def classify_screen(_image):
            return ScreenState.PORT

    bot = SimpleNamespace(
        hwnd=1,
        vision=ConfirmedPortVision(),
        distance_reader=SimpleNamespace(backend=object()),
    )
    with (
        patch("main.time.sleep", return_value=None),
        patch("main.port_battle_action_point", return_value=(100, 100)),
    ):
        state = recover_current_scene(
            bot,
            attempts=2,
            round_in_progress=True,
        )

    assert state == ScreenState.PORT


def test_active_round_battle_hud_overrides_false_port_classification():
    class ConflictingVision:
        @staticmethod
        def grab(_hwnd, *, allow_stale=False):
            return object()

        @staticmethod
        def classify_screen(_image):
            return ScreenState.PORT

        @staticmethod
        def _has_battle_hud(_image):
            return True

    bot = SimpleNamespace(hwnd=1, vision=ConflictingVision())
    with patch("main.time.sleep", return_value=None):
        state = recover_current_scene(
            bot,
            attempts=2,
            round_in_progress=True,
        )

    assert state == ScreenState.BATTLE


def test_active_round_returns_survey_instead_of_collapsing_it_to_unknown():
    frame = object()

    class SurveyVision:
        @staticmethod
        def grab(_hwnd, *, allow_stale=False):
            return frame

        @staticmethod
        def classify_screen(_image):
            return ScreenState.PORT

        @staticmethod
        def _has_battle_hud(_image):
            return False

    bot = SimpleNamespace(
        hwnd=1,
        vision=SurveyVision(),
        distance_reader=SimpleNamespace(backend=object()),
    )
    with (
        patch("main.time.sleep", return_value=None),
        patch("main.is_battle_survey_page", return_value=True),
    ):
        state = recover_current_scene(
            bot,
            attempts=2,
            round_in_progress=True,
        )

    assert state == ScreenState.SURVEY
