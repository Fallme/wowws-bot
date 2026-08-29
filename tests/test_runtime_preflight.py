from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from core.calibration import AUTOMATIC_PREFLIGHT_KEY, CalibrationStore
from core.ui import ScreenState
from main import (
    automatic_input_preflight,
    configure_opening_autopilot,
    prepare_battle,
    refresh_game_window,
    run_battle,
    tactical_map_local_point,
    wait_for_battle,
    wait_for_recognized_screen,
)
from core.vision import CaptureZone, PlayerPose


class FakeVision:
    def __init__(self):
        self.screen_capture = SimpleNamespace(last_backend="print_window")

    def grab(self, _hwnd):
        return np.full((90, 160, 3), 80, dtype=np.uint8)

    def classify_screen(self, _image):
        return ScreenState.PORT


class FakeController:
    def __init__(self):
        self.stop_calls = 0

    def stop(self):
        self.stop_calls += 1


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
        "reset",
        "autopilot",
        "center_route",
        "analyze",
    ]


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


def test_run_battle_retries_lost_autopilot_before_enabling_generic_qe():
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

    assert retry_flags == [False, True]
    assert events == ["原生自动航行三次敌方偏移重试均失败，Q/E小地图驾驶接管"]


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


def test_tactical_map_point_maps_minimap_center_to_screen_center():
    assert tactical_map_local_point(2560, 1600, (0.5, 0.5)) == (1280, 800)

    left_top = tactical_map_local_point(2560, 1600, (0.0, 0.0))
    right_bottom = tactical_map_local_point(2560, 1600, (1.0, 1.0))
    assert left_top[0] < 1280 < right_bottom[0]
    assert left_top[1] < 800 < right_bottom[1]


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
    ):
        assert configure_opening_autopilot(bot)

    assert events == ["toggle", "toggle", "地图中心敌方远端"]
    assert len(clicks) == 1
    # Capture-circle OCR is telemetry only. The first target already crosses
    # the centre on the spawn-to-centre ray; later retries advance farther
    # into the enemy half.
    assert 810 < clicks[0][0] <= 930


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


def test_lost_autopilot_retries_three_progressive_enemy_biased_destinations():
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
    ):
        assert not configure_opening_autopilot(bot, retrying=True)

    assert len(clicks) == 3
    assert clicks[0][0] < clicks[1][0] < clicks[2][0]
    assert clicks[0][1] > clicks[1][1] > clicks[2][1]


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
