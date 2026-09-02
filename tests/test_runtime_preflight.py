import math
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from core.calibration import AUTOMATIC_PREFLIGHT_KEY, CalibrationStore
from core.frame_guard import CaptureFault
from core.ocr import OcrToken
from core.ui import ScreenState
from main import (
    automatic_input_preflight,
    classify_runtime_screen,
    configure_opening_autopilot,
    dismiss_battle_overlay,
    normalize_tactical_map_overlay,
    opening_autopilot_target,
    prepare_battle,
    refresh_game_window,
    run_battle,
    tactical_map_is_open,
    tactical_map_local_point,
    wait_for_battle,
    wait_while_loading,
    wait_for_recognized_screen,
)
from core.vision import CaptureZone, PlayerPose


class FakeVision:
    def __init__(self):
        self.screen_capture = SimpleNamespace(last_backend="print_window")
        self.allow_stale_requests = []

    def grab(self, _hwnd, *, allow_stale=False):
        self.allow_stale_requests.append(allow_stale)
        return np.full((90, 160, 3), 80, dtype=np.uint8)

    def classify_screen(self, _image):
        return ScreenState.PORT


def test_loading_capture_fault_returns_to_lifecycle_instead_of_stopping_run():
    vision = SimpleNamespace(
        grab=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            CaptureFault("capture unavailable")
        )
    )
    bot = SimpleNamespace(hwnd=1, vision=vision)

    assert wait_while_loading(bot, timeout=1) is None


class FakeController:
    def __init__(self):
        self.stop_calls = 0

    def stop(self):
        self.stop_calls += 1


def test_runtime_exit_colour_requires_exact_ocr_before_it_is_actionable():
    frame = np.zeros((1000, 1600, 3), dtype=np.uint8)

    class EmptyBackend:
        @staticmethod
        def recognize(_image):
            return []

    bot = SimpleNamespace(
        vision=SimpleNamespace(classify_screen=lambda _image: ScreenState.EXIT_CONFIRMATION),
        distance_reader=SimpleNamespace(backend=EmptyBackend()),
    )

    assert classify_runtime_screen(bot, frame) == ScreenState.UNKNOWN


def test_runtime_promotes_exact_port_exit_dialog_from_unknown():
    frame = np.zeros((1000, 1600, 3), dtype=np.uint8)

    class PortExitBackend:
        @staticmethod
        def recognize(_image):
            return [
                OcrToken("确认", 0.99),
                OcrToken("退出游戏？", 0.99),
                OcrToken(
                    "否",
                    0.99,
                    ((850, 470), (890, 470), (890, 500), (850, 500)),
                ),
            ]

    bot = SimpleNamespace(
        vision=SimpleNamespace(
            classify_screen=lambda _image: ScreenState.UNKNOWN,
            is_daily_reward_page=lambda _image, _backend: False,
        ),
        distance_reader=SimpleNamespace(backend=PortExitBackend()),
    )

    assert (
        classify_runtime_screen(bot, frame)
        == ScreenState.PORT_EXIT_CONFIRMATION
    )


def test_startup_recovers_stable_mode_selector_before_port_workflow():
    frame = np.full((90, 160, 3), 80, dtype=np.uint8)
    backend = object()

    class SelectorVision:
        @staticmethod
        def grab(_hwnd, *, allow_stale=False):
            return frame

    bot = SimpleNamespace(
        hwnd=1,
        vision=SelectorVision(),
        gamepad=SimpleNamespace(),
        intervention=None,
        distance_reader=SimpleNamespace(backend=backend),
    )
    with (
        patch("main.ensure_capture_foreground", return_value=True),
        patch("main.time.sleep", return_value=None),
        patch(
            "main.classify_runtime_screen",
            side_effect=[
                ScreenState.UNKNOWN,
                ScreenState.UNKNOWN,
                ScreenState.PORT,
                ScreenState.PORT,
            ],
        ),
        patch(
            "main.in_battle_type_selector",
            side_effect=[True, True, False, False],
        ),
        patch("main.select_mode_from_screen", return_value=True) as select_mode,
        patch.dict("main.os.environ", {"WOWS_MODE": "cooperative"}),
    ):
        image, state = wait_for_recognized_screen(bot, timeout=2)

    assert image is frame
    assert state == ScreenState.PORT
    select_mode.assert_called_once_with(
        1,
        "cooperative",
        image=frame,
        backend=backend,
    )


def test_startup_exits_mode_selector_when_target_card_ocr_fails():
    frame = np.full((90, 160, 3), 80, dtype=np.uint8)
    escape_calls = []

    class SelectorVision:
        @staticmethod
        def grab(_hwnd, *, allow_stale=False):
            return frame

    bot = SimpleNamespace(
        hwnd=1,
        vision=SelectorVision(),
        gamepad=SimpleNamespace(escape=lambda: escape_calls.append("escape")),
        intervention=None,
        distance_reader=SimpleNamespace(backend=object()),
    )
    with (
        patch("main.ensure_capture_foreground", return_value=True),
        patch("main.time.sleep", return_value=None),
        patch(
            "main.classify_runtime_screen",
            side_effect=[
                ScreenState.UNKNOWN,
                ScreenState.UNKNOWN,
                ScreenState.PORT,
                ScreenState.PORT,
            ],
        ),
        patch(
            "main.in_battle_type_selector",
            side_effect=[True, True, False, False],
        ),
        patch("main.select_mode_from_screen", return_value=False),
    ):
        _image, state = wait_for_recognized_screen(bot, timeout=2)

    assert state == ScreenState.PORT
    assert escape_calls == ["escape"]


def test_startup_dismisses_stable_battle_survey_then_rechecks_scene():
    frame = np.full((90, 160, 3), 80, dtype=np.uint8)
    escape_calls = []

    class SurveyVision:
        @staticmethod
        def grab(_hwnd, *, allow_stale=False):
            return frame

    bot = SimpleNamespace(
        hwnd=1,
        vision=SurveyVision(),
        gamepad=SimpleNamespace(escape=lambda: escape_calls.append("escape")),
        intervention=None,
        distance_reader=SimpleNamespace(backend=object()),
    )
    with (
        patch("main.ensure_capture_foreground", return_value=True),
        patch("main.time.sleep", return_value=None),
        patch(
            "main.classify_runtime_screen",
            side_effect=[
                ScreenState.UNKNOWN,
                ScreenState.UNKNOWN,
                ScreenState.PORT,
                ScreenState.PORT,
            ],
        ),
        patch(
            "main.is_battle_survey_page",
            side_effect=[True, True, False, False],
        ),
        patch("main.in_battle_type_selector", return_value=False),
    ):
        _image, state = wait_for_recognized_screen(bot, timeout=2)

    assert state == ScreenState.PORT
    assert escape_calls == ["escape"]


def test_automatic_port_preflight_releases_input_and_saves_status(tmp_path):
    controller = FakeController()
    bot = SimpleNamespace(hwnd=1, gamepad=controller, vision=FakeVision())
    store = CalibrationStore(tmp_path / "input_calibration.json")

    with (
        patch("main.ensure_game_window_foreground", return_value=True),
        patch("main.time.sleep", return_value=None),
    ):
        status = automatic_input_preflight(
            bot,
            "World of Warships",
            (0, 0, 2560, 1600),
            ScreenState.PORT,
            store=store,
        )

    assert status.valid
    assert controller.stop_calls == 1
    assert status.resolution == (2560, 1600)
    assert bot.vision.allow_stale_requests == [True]


def test_automatic_results_preflight_accepts_a_legitimately_static_frame(tmp_path):
    class ResultsVision(FakeVision):
        def classify_screen(self, _image):
            return ScreenState.RESULTS

    controller = FakeController()
    bot = SimpleNamespace(hwnd=1, gamepad=controller, vision=ResultsVision())
    store = CalibrationStore(tmp_path / "input_calibration.json")

    with (
        patch("main.ensure_game_window_foreground", return_value=True),
        patch("main.time.sleep", return_value=None),
    ):
        status = automatic_input_preflight(
            bot,
            "World of Warships",
            (0, 0, 2560, 1600),
            ScreenState.RESULTS,
            store=store,
        )

    assert status.valid
    assert controller.stop_calls == 1
    assert bot.vision.allow_stale_requests == [True]


def test_automatic_battle_preflight_preserves_existing_ship_controls(tmp_path):
    class BattleVision(FakeVision):
        def classify_screen(self, _image):
            return ScreenState.BATTLE

    controller = FakeController()
    bot = SimpleNamespace(hwnd=1, gamepad=controller, vision=BattleVision())
    store = CalibrationStore(tmp_path / "input_calibration.json")

    with (
        patch("main.ensure_game_window_foreground", return_value=True),
        patch("main.time.sleep", return_value=None),
    ):
        status = automatic_input_preflight(
            bot,
            "World of Warships",
            (0, 0, 2560, 1600),
            ScreenState.BATTLE,
            store=store,
        )

    assert status.valid
    assert controller.stop_calls == 0
    assert bot.vision.allow_stale_requests == [False]
    record = store.load()
    assert record is not None
    assert (
        record.observations[AUTOMATIC_PREFLIGHT_KEY]["input_check"]
        == "battle_controls_preserved"
    )


def test_battle_hud_is_checked_before_commander_dialog_detector():
    class BattleVision:
        def grab(self, _hwnd, *, allow_stale=False):
            return np.full((90, 160, 3), 80, dtype=np.uint8)

        def classify_screen(self, _image):
            return ScreenState.BATTLE

    bot = SimpleNamespace(hwnd=1, vision=BattleVision())
    with patch("main.time.sleep", return_value=None):
        assert wait_for_battle(bot, timeout=2)


def test_wait_for_battle_fails_closed_on_no_commander_warning():
    class NoCommanderVision:
        @staticmethod
        def grab(_hwnd, *, allow_stale=False):
            return np.full((90, 160, 3), 80, dtype=np.uint8)

        @staticmethod
        def classify_screen(_image):
            return ScreenState.UNKNOWN

        @staticmethod
        def in_no_commander_confirmation(_image):
            return True

    bot = SimpleNamespace(hwnd=1, vision=NoCommanderVision())
    with (
        patch("main.time.sleep", return_value=None),
        patch("main.classify_runtime_screen", return_value=ScreenState.UNKNOWN),
    ):
        assert not wait_for_battle(bot, timeout=2)


def test_new_round_waits_for_loading_then_fresh_upper_right_clock():
    class NewRoundVision:
        def __init__(self):
            self.states = [
                ScreenState.BATTLE,
                ScreenState.LOADING,
                ScreenState.BATTLE,
                ScreenState.BATTLE,
            ]
            self.index = 0

        def grab(self, _hwnd, *, allow_stale=False):
            return np.full((90, 160, 3), 80, dtype=np.uint8)

        def classify_screen(self, _image):
            state = self.states[min(self.index, len(self.states) - 1)]
            self.index += 1
            return state

        @staticmethod
        def read_battle_clock_seconds(_image, _backend):
            return 19 * 60 + 49

    vision = NewRoundVision()
    bot = SimpleNamespace(hwnd=1, vision=vision)
    with patch("main.time.sleep", return_value=None):
        assert wait_for_battle(bot, timeout=2, require_new_round=True)

    # The first battle-looking frame is rejected because no loading transition
    # had been observed; the new round is accepted only after the full chain.
    assert vision.index == 4


def test_first_battle_hud_frame_starts_engine_before_clock_and_autopilot_finish():
    class OpeningVision:
        def __init__(self):
            self.states = [
                ScreenState.LOADING,
                ScreenState.BATTLE,
                ScreenState.BATTLE,
            ]
            self.index = 0

        def grab(self, _hwnd, *, allow_stale=False):
            return np.full((90, 160, 3), 80, dtype=np.uint8)

        def classify_screen(self, _image):
            state = self.states[min(self.index, len(self.states) - 1)]
            self.index += 1
            return state

        @staticmethod
        def read_battle_clock_seconds(_image, _backend):
            return 19 * 60 + 55

    class OpeningController:
        def __init__(self):
            self.reassertions = 0

        def reassert_full_speed(self):
            self.reassertions += 1

    controller = OpeningController()
    bot = SimpleNamespace(
        hwnd=1,
        vision=OpeningVision(),
        gamepad=controller,
        intervention=None,
    )
    with (
        patch("main.time.sleep", return_value=None),
        patch("main.ensure_capture_foreground", return_value=True),
        patch("main.configure_opening_autopilot", return_value=False),
    ):
        assert wait_for_battle(bot, timeout=2, require_new_round=True)

    assert controller.reassertions == 1
    assert bot._opening_motion_prestarted


def test_confirmed_loading_boundary_accepts_stable_hud_when_clock_ocr_is_missing():
    class ClocklessBattleVision:
        @staticmethod
        def grab(_hwnd, *, allow_stale=False):
            return np.full((90, 160, 3), 80, dtype=np.uint8)

        @staticmethod
        def classify_screen(_image):
            return ScreenState.BATTLE

        @staticmethod
        def read_battle_clock_seconds(_image, _backend):
            return None

    controller = SimpleNamespace(reassert_full_speed=lambda: None)
    bot = SimpleNamespace(
        hwnd=1,
        vision=ClocklessBattleVision(),
        gamepad=controller,
        intervention=None,
    )
    with (
        patch("main.time.sleep", return_value=None),
        patch("main.ensure_capture_foreground", return_value=True),
        patch("main.configure_opening_autopilot", return_value=False),
    ):
        assert wait_for_battle(
            bot,
            timeout=2,
            require_new_round=True,
            loading_already_seen=True,
        )


def test_run_battle_dispatches_full_speed_before_first_analysis():
    events = []

    class ImmediateController:
        def full_speed(self):
            events.append("full_speed")

    class ImmediateBot:
        hwnd = 1
        gamepad = ImmediateController()
        last_movement_reason = ""

        def reset(self):
            events.append("reset")

        def combat_tick(self):
            events.append("analyze")
            return "ended"

    with (
        patch("main.ensure_game_window_foreground", return_value=True),
        patch("main.time.sleep", return_value=None),
    ):
        assert run_battle(ImmediateBot())

    assert events == ["reset", "full_speed", "analyze"]


def test_run_battle_scene_interlock_rejects_port_before_any_command():
    events = []

    class PortVision:
        @staticmethod
        def grab(_hwnd, *, allow_stale=False):
            return np.zeros((90, 160, 3), dtype=np.uint8)

        @staticmethod
        def classify_screen(_image):
            return ScreenState.PORT

    class InterlockedBot:
        hwnd = 1
        vision = PortVision()
        gamepad = SimpleNamespace()
        intervention = None

        @staticmethod
        def reset(*_args, **_kwargs):
            events.append("reset")

    with patch("main.ensure_bound_game_foreground", return_value=True):
        assert run_battle(InterlockedBot(), resume_existing=True) == "resume_state"

    assert events == []


def test_resumed_battle_reasserts_full_speed_before_autopilot_setup():
    events = []

    class BattleVision:
        @staticmethod
        def grab(_hwnd, *, allow_stale=False):
            return np.zeros((90, 160, 3), dtype=np.uint8)

        @staticmethod
        def classify_screen(_image):
            return ScreenState.BATTLE

        @staticmethod
        def is_autopilot_enabled(_image):
            return False

    class Controller:
        @staticmethod
        def reassert_full_speed():
            events.append("full_speed")

    class ResumedBot:
        hwnd = 1
        vision = BattleVision()
        gamepad = Controller()
        intervention = None
        last_movement_reason = ""

        @staticmethod
        def reset(*_args, **_kwargs):
            events.append("reset")

        @staticmethod
        def enable_generic_center_route(_reason):
            events.append("center_route")

        @staticmethod
        def combat_tick():
            events.append("analyze")
            return "ended"

    with (
        patch("main.ensure_bound_game_foreground", return_value=True),
        patch(
            "main.configure_opening_autopilot",
            side_effect=lambda _bot: events.append("autopilot") or False,
        ),
        patch("main.time.sleep", return_value=None),
    ):
        assert run_battle(ResumedBot(), resume_existing=True)

    assert events == [
        "full_speed",
        "autopilot",
        "center_route",
        "analyze",
    ]


def test_resumed_stalled_route_ignores_lingering_green_autopilot_hud():
    events = []

    class BattleVision:
        @staticmethod
        def grab(_hwnd, *, allow_stale=False):
            return np.zeros((90, 160, 3), dtype=np.uint8)

        @staticmethod
        def classify_screen(_image):
            return ScreenState.BATTLE

        @staticmethod
        def is_autopilot_enabled(_image):
            return True

    class StalledBot:
        hwnd = 1
        vision = BattleVision()
        intervention = None
        gamepad = SimpleNamespace(
            resynchronize_forward_controls=lambda: events.append("resync")
        )
        last_movement_reason = ""
        native_autopilot_abandoned = True

        def reset(self, *_args, **_kwargs):
            events.append("reset")
            self.native_autopilot_abandoned = False

        @staticmethod
        def enable_generic_center_route(_reason):
            events.append("center_route")

        @staticmethod
        def combat_tick():
            events.append("analyze")
            return "ended"

    bot = StalledBot()
    with (
        patch("main.ensure_bound_game_foreground", return_value=True),
        patch("main.configure_opening_autopilot") as configure,
        patch("main.time.sleep", return_value=None),
    ):
        assert run_battle(bot, resume_existing=True)

    configure.assert_not_called()
    assert bot.native_autopilot_abandoned
    assert events == ["resync", "center_route", "analyze"]


def test_wait_for_battle_prefers_positive_hud_over_false_port_state():
    class ConflictingVision:
        def __init__(self):
            self.frames = 0

        def grab(self, _hwnd, *, allow_stale=False):
            self.frames += 1
            return np.zeros((90, 160, 3), dtype=np.uint8)

        @staticmethod
        def classify_screen(_image):
            return ScreenState.PORT

        @staticmethod
        def _has_battle_hud(_image):
            return True

    vision = ConflictingVision()
    bot = SimpleNamespace(hwnd=1, vision=vision)
    with patch("main.time.sleep", return_value=None):
        assert wait_for_battle(bot, timeout=2)

    assert vision.frames == 2


def test_fresh_battle_reasserts_full_speed_even_when_autopilot_succeeds():
    events = []

    class BattleVision:
        @staticmethod
        def grab(_hwnd, *, allow_stale=False):
            return np.zeros((90, 160, 3), dtype=np.uint8)

        @staticmethod
        def classify_screen(_image):
            return ScreenState.BATTLE

        @staticmethod
        def is_autopilot_enabled(_image):
            return False

    class FreshBot:
        hwnd = 1
        vision = BattleVision()
        intervention = None
        gamepad = SimpleNamespace(
            reassert_full_speed=lambda: events.append("full_speed")
        )
        last_movement_reason = ""

        @staticmethod
        def reset(*_args, **_kwargs):
            events.append("reset")

        @staticmethod
        def combat_tick():
            events.append("analyze")
            return "ended"

    with (
        patch("main.ensure_bound_game_foreground", return_value=True),
        patch(
            "main.configure_opening_autopilot",
            side_effect=lambda _bot: events.append("autopilot") or True,
        ),
        patch("main.time.sleep", return_value=None),
    ):
        assert run_battle(FreshBot())

    assert events == ["full_speed", "reset", "autopilot", "analyze"]


def test_run_battle_leaves_false_battle_after_three_non_battle_frames():
    class BattleVision:
        @staticmethod
        def grab(_hwnd, *, allow_stale=False):
            return np.zeros((90, 160, 3), dtype=np.uint8)

        @staticmethod
        def classify_screen(_image):
            return ScreenState.BATTLE

    class WaitingBot:
        hwnd = 1
        vision = BattleVision()
        gamepad = SimpleNamespace()
        intervention = None
        last_analysis = None

        @staticmethod
        def reset(*_args, **_kwargs):
            return None

        def combat_tick(self):
            self.last_analysis = SimpleNamespace(
                in_battle=False,
                health=1.0,
            )
            return "waiting"

    with (
        patch("main.ensure_bound_game_foreground", return_value=True),
        patch("main.configure_opening_autopilot", return_value=True),
        patch("main.time.sleep", return_value=None),
    ):
        assert run_battle(WaitingBot()) == "resume_state"


def test_run_battle_never_reopens_map_after_lost_autopilot():
    retry_flags = []
    events = []

    class BattleVision:
        @staticmethod
        def grab(_hwnd, *, allow_stale=False):
            return np.zeros((90, 160, 3), dtype=np.uint8)

        @staticmethod
        def classify_screen(_image):
            return ScreenState.BATTLE

    class RetryBot:
        hwnd = 1
        vision = BattleVision()
        gamepad = SimpleNamespace()
        intervention = None
        autopilot_retry_pending = False
        last_analysis = SimpleNamespace(in_battle=True, health=1.0)
        ticks = 0

        @staticmethod
        def reset(*_args, **_kwargs):
            return None

        def combat_tick(self):
            self.ticks += 1
            if self.ticks == 1:
                self.autopilot_retry_pending = True
                return "waiting"
            return "ended"

        def enable_generic_center_route(self, reason):
            self.autopilot_retry_pending = False
            events.append(reason)

    def configure(_bot, *, retrying=False):
        retry_flags.append(retrying)
        return not retrying

    with (
        patch("main.ensure_bound_game_foreground", return_value=True),
        patch("main.configure_opening_autopilot", side_effect=configure),
        patch("main.time.sleep", return_value=None),
    ):
        assert run_battle(RetryBot())

    assert retry_flags == [False]
    assert events == [
        "原生自动航行结束，小地图闭环驾驶接管；本局不再打开M地图"
    ]


def test_quick_battle_timeout_is_counted_only_inside_confirmed_battle():
    class BattleVision:
        @staticmethod
        def grab(_hwnd, *, allow_stale=False):
            return np.zeros((90, 160, 3), dtype=np.uint8)

        @staticmethod
        def classify_screen(_image):
            return ScreenState.BATTLE

    class QuickBot:
        hwnd = 1
        vision = BattleVision()
        gamepad = SimpleNamespace()
        intervention = None

        @staticmethod
        def reset(*_args, **_kwargs):
            return None

        @staticmethod
        def combat_tick():
            raise AssertionError("五分钟已到，不应再发送战斗指令")

    with (
        patch("main.ensure_bound_game_foreground", return_value=True),
        patch("main.configure_opening_autopilot", return_value=True),
        patch("main.time.monotonic", side_effect=[0.0, 301.0]),
    ):
        assert run_battle(QuickBot(), quick_battle=True) == "quick_timeout"


def test_quick_battle_death_immediately_requests_next_round():
    class BattleVision:
        @staticmethod
        def grab(_hwnd, *, allow_stale=False):
            return np.zeros((90, 160, 3), dtype=np.uint8)

        @staticmethod
        def classify_screen(_image):
            return ScreenState.BATTLE

    class SunkBot:
        hwnd = 1
        vision = BattleVision()
        gamepad = SimpleNamespace()
        intervention = None
        last_analysis = None

        @staticmethod
        def reset(*_args, **_kwargs):
            return None

        def combat_tick(self):
            self.last_analysis = SimpleNamespace(in_battle=True, health=0.0)
            return "waiting"

    with (
        patch("main.ensure_bound_game_foreground", return_value=True),
        patch("main.configure_opening_autopilot", return_value=True),
        patch("main.time.monotonic", side_effect=[0.0, 1.0]),
    ):
        assert run_battle(SunkBot(), quick_battle=True) == "quick_death"


def test_keyboard_pause_skips_capture_focus_and_all_followup_commands():
    class PausedIntervention:
        @staticmethod
        def command_generation_paused():
            return False

        def poll(self, _controller, _now):
            return True

    class PausedBot:
        hwnd = 1
        gamepad = SimpleNamespace()
        intervention = PausedIntervention()

        @staticmethod
        def mark_manual_pause():
            return None

        @staticmethod
        def combat_tick():
            raise AssertionError("暂停期间不得截图或下发战斗指令")

    checks = iter([False, True])
    with (
        patch("main.ensure_game_window_foreground") as focus,
        patch("main.time.sleep", return_value=None),
    ):
        assert not run_battle(PausedBot(), should_stop=lambda: next(checks))

    focus.assert_not_called()


def test_battle_resume_returns_to_scene_router_before_another_combat_tick():
    class PauseThenResume:
        def __init__(self):
            self.states = iter([False, True, False])

        def poll(self, _controller, _now):
            return next(self.states)

        @staticmethod
        def command_generation_paused():
            return False

    class ResumingBot:
        hwnd = 1
        gamepad = SimpleNamespace()
        intervention = PauseThenResume()

        @staticmethod
        def mark_manual_pause():
            return None

        @staticmethod
        def reset(*_args, **_kwargs):
            return None

        @staticmethod
        def combat_tick():
            raise AssertionError("继续后必须先重新识别场景，不能沿用旧战斗循环")

    with (
        patch("main.ensure_bound_game_foreground", return_value=True),
        patch("main.configure_opening_autopilot", return_value=True),
        patch("main.restore_game_foreground_after_pause", return_value=True),
        patch("main.time.sleep", return_value=None),
    ):
        assert run_battle(ResumingBot()) == "resume_state"


def test_tactical_map_point_maps_minimap_center_to_screen_center():
    assert tactical_map_local_point(2560, 1600, (0.5, 0.5)) == (1280, 800)

    left_top = tactical_map_local_point(2560, 1600, (0.0, 0.0))
    right_bottom = tactical_map_local_point(2560, 1600, (1.0, 1.0))
    assert left_top[0] < 1280 < right_bottom[0]
    assert left_top[1] < 800 < right_bottom[1]


def test_tactical_map_text_is_positive_evidence_for_open_overlay():
    class StaticBackend:
        @staticmethod
        def recognize(_image):
            return [
                OcrToken("自动驾驶控制", 0.99),
                OcrToken("按下 M 来离开战术地图模式。", 0.98),
            ]

    bot = SimpleNamespace(
        distance_reader=SimpleNamespace(backend=StaticBackend())
    )
    image = np.zeros((1000, 1600, 3), dtype=np.uint8)

    assert tactical_map_is_open(bot, image)


def test_opening_waypoint_is_near_center_on_enemy_side():
    target = opening_autopilot_target((0.22, 0.89))

    assert target[0] == pytest.approx(0.599, abs=0.01)
    assert target[1] == pytest.approx(0.362, abs=0.01)
    assert math.dist(target, (0.5, 0.5)) == pytest.approx(0.17)

    retries = [
        opening_autopilot_target((0.22, 0.89), attempt_index=index)
        for index in range(3)
    ]
    assert len(set(retries)) == 3
    assert [round(math.dist(point, (0.5, 0.5)), 3) for point in retries] == [
        0.17,
        0.203,
        0.233,
    ]


def test_interrupted_tactical_map_is_closed_once_before_battle_resume():
    toggles = []

    class StaticBackend:
        @staticmethod
        def recognize(_image):
            return [OcrToken("自动驾驶控制", 0.99)]

    bot = SimpleNamespace(
        hwnd=1,
        _tactical_map_left_open=True,
        intervention=None,
        gamepad=SimpleNamespace(
            toggle_tactical_map=lambda: toggles.append("m")
        ),
        vision=SimpleNamespace(
            grab=lambda _hwnd, allow_stale=False: np.zeros(
                (1000, 1600, 3), dtype=np.uint8
            )
        ),
        distance_reader=SimpleNamespace(backend=StaticBackend()),
    )

    with patch("main.time.sleep", return_value=None):
        assert normalize_tactical_map_overlay(bot)

    assert toggles == ["m"]
    assert not bot._tactical_map_left_open


def test_stale_tactical_map_flag_never_opens_map_on_normal_battle_screen():
    toggles = []

    class StaticBackend:
        @staticmethod
        def recognize(_image):
            return [OcrToken("战斗进行中", 0.99)]

    bot = SimpleNamespace(
        hwnd=1,
        _tactical_map_left_open=True,
        intervention=None,
        gamepad=SimpleNamespace(
            toggle_tactical_map=lambda: toggles.append("m")
        ),
        vision=SimpleNamespace(
            grab=lambda _hwnd, allow_stale=False: np.zeros(
                (1000, 1600, 3), dtype=np.uint8
            )
        ),
        distance_reader=SimpleNamespace(backend=StaticBackend()),
    )

    assert normalize_tactical_map_overlay(bot)
    assert toggles == []
    assert not bot._tactical_map_left_open


def test_escape_and_exit_overlays_use_esc_until_battle_is_restored():
    states = iter([ScreenState.ESCAPE_MENU, ScreenState.BATTLE])
    escapes = []

    class OverlayVision:
        @staticmethod
        def grab(_hwnd, *, allow_stale=False):
            return np.zeros((100, 160, 3), dtype=np.uint8)

        @staticmethod
        def classify_screen(_image):
            return next(states)

    bot = SimpleNamespace(
        hwnd=1,
        intervention=None,
        gamepad=SimpleNamespace(escape=lambda: escapes.append("esc")),
        vision=OverlayVision(),
    )
    with (
        patch("main.ensure_capture_foreground", return_value=True),
        patch("main.time.sleep", return_value=None),
    ):
        restored = dismiss_battle_overlay(
            bot,
            ScreenState.EXIT_CONFIRMATION,
        )

    assert restored == ScreenState.BATTLE
    assert escapes == ["esc", "esc"]


def test_prepare_battle_cancels_port_actions_when_second_frame_is_battle():
    port_frame = np.zeros((90, 160, 3), dtype=np.uint8)
    battle_frame = np.ones((90, 160, 3), dtype=np.uint8)

    class TransitionVision:
        def __init__(self):
            self.frames = [port_frame, battle_frame]

        def grab(self, _hwnd, *, allow_stale=False):
            return self.frames.pop(0) if len(self.frames) > 1 else self.frames[0]

        @staticmethod
        def classify_screen(image):
            return ScreenState.BATTLE if image[0, 0, 0] else ScreenState.PORT

    bot = SimpleNamespace(hwnd=1, vision=TransitionVision())
    with (
        patch("main.time.sleep", return_value=None),
        patch("main.select_requested_ship") as select_ship,
        patch("main.ensure_requested_mode") as select_mode,
        patch("main.enter_battle") as enter,
    ):
        assert prepare_battle(bot)

    select_ship.assert_not_called()
    select_mode.assert_not_called()
    enter.assert_not_called()


def test_opening_autopilot_crosses_center_not_unstable_capture_circle():
    minimap = np.zeros((200, 200, 3), dtype=np.uint8)
    events = []

    class AutopilotVision:
        @staticmethod
        def grab(_hwnd, *, allow_stale=False):
            return np.zeros((1000, 1600, 3), dtype=np.uint8)

        @staticmethod
        def classify_screen(_image):
            return ScreenState.BATTLE

        @staticmethod
        def find_minimap(_image):
            return minimap

        @staticmethod
        def find_player_pose_on_minimap(_minimap):
            return PlayerPose(position=(50, 100), heading=(1.0, 0.0))

        @staticmethod
        def find_nearest_capture_zone(_minimap, _player):
            return CaptureZone(center=(100, 100), radius=20)

    class AutopilotController:
        @staticmethod
        def toggle_tactical_map():
            events.append("toggle")

    bot = SimpleNamespace(
        hwnd=1,
        vision=AutopilotVision(),
        gamepad=AutopilotController(),
        enable_opening_autopilot=lambda target: events.append(target),
    )
    clicks = []
    with (
        patch("main.time.sleep", return_value=None),
        patch(
            "main.get_client_rect",
            return_value={"left": 10, "top": 20, "right": 1610, "bottom": 1020},
        ),
        patch(
            "main.physical_click",
            side_effect=lambda x, y, **_kwargs: clicks.append((x, y)) or True,
        ),
        patch("main.tactical_map_is_open", side_effect=[True] * 5 + [False]),
    ):
        assert configure_opening_autopilot(bot)
        assert not configure_opening_autopilot(bot)

    assert events[:2] == ["toggle", "toggle"]
    assert "第1次" in events[2]
    assert len(clicks) == 1
    # Capture-circle OCR is telemetry only. The first target already crosses
    # the centre on the spawn-to-centre ray; later retries advance farther
    # into the enemy half.
    assert 920 < clicks[0][0] <= 970


def test_opening_autopilot_captures_and_freezes_five_tactical_map_frames():
    minimap = np.zeros((200, 200, 3), dtype=np.uint8)
    static_samples = []
    lifecycle = []

    class TacticalVision:
        @staticmethod
        def grab(_hwnd, *, allow_stale=False):
            return np.full((1000, 1600, 3), 80, dtype=np.uint8)

        @staticmethod
        def classify_screen(_image):
            return ScreenState.BATTLE

        @staticmethod
        def find_minimap(_image):
            return minimap

        @staticmethod
        def find_player_pose_on_minimap(_minimap):
            return PlayerPose(position=(50, 100), heading=(1.0, 0.0))

    class TacticalTextBackend:
        calls = 0

        @classmethod
        def recognize(cls, _image):
            cls.calls += 1
            return [OcrToken("自动驾驶控制", 0.99)] if cls.calls == 1 else []

    bot = SimpleNamespace(
        hwnd=1,
        vision=TacticalVision(),
        distance_reader=SimpleNamespace(backend=TacticalTextBackend()),
        gamepad=SimpleNamespace(toggle_tactical_map=lambda: None),
        intervention=None,
        begin_tactical_map_static_capture=lambda: lifecycle.append("begin"),
        capture_tactical_map_static_layer=(
            lambda image: static_samples.append(image.copy())
            or len(static_samples) >= 3
        ),
        enable_opening_autopilot=lambda *_args, **_kwargs: None,
    )
    with (
        patch("main.time.sleep", return_value=None),
        patch(
            "main.get_client_rect",
            return_value={"left": 0, "top": 0, "right": 1600, "bottom": 1000},
        ),
        patch("main.physical_click", return_value=True),
        patch("main.tactical_map_is_open", side_effect=[True] * 5 + [False]),
    ):
        assert configure_opening_autopilot(bot)

    assert lifecycle == ["begin"]
    assert len(static_samples) == 5
    assert all(sample.shape == (1000, 1600, 3) for sample in static_samples)


def test_opening_autopilot_refuses_short_center_click_without_player_arrow():
    minimap = np.zeros((200, 200, 3), dtype=np.uint8)

    class MissingPlayerVision:
        @staticmethod
        def grab(_hwnd, *, allow_stale=False):
            return np.zeros((1000, 1600, 3), dtype=np.uint8)

        @staticmethod
        def classify_screen(_image):
            return ScreenState.BATTLE

        @staticmethod
        def find_minimap(_image):
            return minimap

        @staticmethod
        def find_player_pose_on_minimap(_minimap):
            return None

    bot = SimpleNamespace(
        hwnd=1,
        vision=MissingPlayerVision(),
        gamepad=SimpleNamespace(toggle_tactical_map=lambda: None),
        enable_opening_autopilot=lambda *_args, **_kwargs: None,
    )
    with (
        patch("main.time.sleep", return_value=None),
        patch("main.physical_click") as click,
    ):
        assert not configure_opening_autopilot(bot)

    click.assert_not_called()


def test_rejected_map_key_does_not_consume_the_battle_attempt():
    minimap = np.zeros((200, 200, 3), dtype=np.uint8)

    class ReadyVision:
        @staticmethod
        def grab(_hwnd, *, allow_stale=False):
            return np.zeros((1000, 1600, 3), dtype=np.uint8)

        @staticmethod
        def classify_screen(_image):
            return ScreenState.BATTLE

        @staticmethod
        def find_minimap(_image):
            return minimap

        @staticmethod
        def find_player_pose_on_minimap(_minimap):
            return PlayerPose(position=(50, 100), heading=(1.0, 0.0))

    def reject_map_key():
        raise RuntimeError("游戏窗口不在前台，拒绝发送键盘操作")

    bot = SimpleNamespace(
        hwnd=1,
        vision=ReadyVision(),
        gamepad=SimpleNamespace(toggle_tactical_map=reject_map_key),
        intervention=None,
        enable_opening_autopilot=lambda *_args, **_kwargs: None,
    )
    with (
        patch("main.time.sleep", return_value=None),
        patch(
            "main.get_client_rect",
            return_value={"left": 0, "top": 0, "right": 1600, "bottom": 1000},
        ),
    ):
        assert not configure_opening_autopilot(bot)

    assert not getattr(bot, "_tactical_map_attempted_this_battle", False)


def test_opening_autopilot_uses_only_one_enemy_biased_destination():
    minimap = np.zeros((200, 200, 3), dtype=np.uint8)
    clicks = []

    class RetryVision:
        @staticmethod
        def grab(_hwnd, *, allow_stale=False):
            return np.zeros((1000, 1600, 3), dtype=np.uint8)

        @staticmethod
        def classify_screen(_image):
            return ScreenState.BATTLE

        @staticmethod
        def find_minimap(_image):
            return minimap

        @staticmethod
        def find_player_pose_on_minimap(_minimap):
            return PlayerPose(position=(40, 180), heading=(0.0, -1.0))

        @staticmethod
        def analyze_minimap(_minimap):
            return [(160, 40)], False

        @staticmethod
        def is_autopilot_enabled(_image):
            return False

    bot = SimpleNamespace(
        hwnd=1,
        vision=RetryVision(),
        gamepad=SimpleNamespace(toggle_tactical_map=lambda: None),
        intervention=None,
        enable_opening_autopilot=lambda *_args, **_kwargs: None,
    )
    with (
        patch("main.time.sleep", return_value=None),
        patch(
            "main.get_client_rect",
            return_value={"left": 0, "top": 0, "right": 1600, "bottom": 1000},
        ),
        patch(
            "main.physical_click",
            side_effect=lambda x, y, **_kwargs: clicks.append((x, y)) or True,
        ),
        patch("main.tactical_map_is_open", side_effect=[True] * 5 + [False]),
    ):
        assert configure_opening_autopilot(bot, retrying=True)

    assert len(clicks) == 1


def test_opening_autopilot_never_clicks_until_map_is_stably_confirmed():
    minimap = np.zeros((200, 200, 3), dtype=np.uint8)

    class VisionWithoutMapEvidence:
        @staticmethod
        def grab(_hwnd, *, allow_stale=False):
            return np.zeros((1000, 1600, 3), dtype=np.uint8)

        @staticmethod
        def classify_screen(_image):
            return ScreenState.BATTLE

        @staticmethod
        def find_minimap(_image):
            return minimap

        @staticmethod
        def find_player_pose_on_minimap(_minimap):
            return PlayerPose(position=(40, 180), heading=(0.0, -1.0))

    bot = SimpleNamespace(
        hwnd=1,
        vision=VisionWithoutMapEvidence(),
        gamepad=SimpleNamespace(toggle_tactical_map=lambda: None),
        intervention=None,
        enable_opening_autopilot=lambda *_args, **_kwargs: None,
    )
    with (
        patch("main.time.sleep", return_value=None),
        patch("main.get_client_rect", return_value={"left": 0, "top": 0}),
        patch("main.tactical_map_is_open", return_value=False),
        patch("main.physical_click") as click,
        patch("main.window_message_click") as message_click,
    ):
        assert not configure_opening_autopilot(bot)

    click.assert_not_called()
    message_click.assert_not_called()
    assert bot._tactical_map_attempted_this_battle
    assert not bot._tactical_map_left_open


def test_opening_autopilot_requires_lower_left_game_confirmation():
    minimap = np.zeros((200, 200, 3), dtype=np.uint8)
    enabled = []
    toggles = []
    clicks = []

    class UnconfirmedAutopilotVision:
        @staticmethod
        def grab(_hwnd, *, allow_stale=False):
            return np.zeros((1000, 1600, 3), dtype=np.uint8)

        @staticmethod
        def classify_screen(_image):
            return ScreenState.BATTLE

        @staticmethod
        def find_minimap(_image):
            return minimap

        @staticmethod
        def find_player_pose_on_minimap(_minimap):
            return PlayerPose(position=(40, 180), heading=(0.0, -1.0))

        @staticmethod
        def read_autopilot_enabled_text(_image, _backend):
            return False

    bot = SimpleNamespace(
        hwnd=1,
        vision=UnconfirmedAutopilotVision(),
        distance_reader=SimpleNamespace(backend=object()),
        gamepad=SimpleNamespace(
            toggle_tactical_map=lambda: toggles.append("m")
        ),
        intervention=None,
        enable_opening_autopilot=lambda *_args, **_kwargs: enabled.append(True),
    )
    with (
        patch("main.time.sleep", return_value=None),
        patch("main.get_client_rect", return_value={"left": 0, "top": 0}),
        patch(
            "main.tactical_map_is_open",
            side_effect=([True] * 5 + [False]) * 3,
        ),
        patch(
            "main.physical_click",
            side_effect=lambda x, y, **_kwargs: clicks.append((x, y)) or True,
        ),
    ):
        assert not configure_opening_autopilot(bot)

    assert enabled == []
    assert toggles == ["m"] * 6
    assert len(clicks) == 3
    assert len(set(clicks)) == 3


def test_refresh_game_window_rebinds_recreated_hwnd_and_maximizes():
    rebound = []
    bot = SimpleNamespace(
        hwnd=11,
        rebind_window=lambda hwnd: rebound.append(hwnd) or True,
    )
    with (
        patch("main.is_game_window", return_value=False),
        patch(
            "main.find_game_window",
            return_value=[(22, "World of Warships", (0, 0, 2560, 1440))],
        ),
        patch("main.maximize_game_window", return_value=True) as maximize,
    ):
        assert refresh_game_window(bot)

    assert rebound == [22]
    maximize.assert_called_once_with(22)
