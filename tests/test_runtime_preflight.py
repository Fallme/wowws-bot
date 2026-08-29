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


def test_opening_autopilot_uses_map_center_not_unstable_capture_circle():
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

    assert events == ["toggle", "toggle", "地图中心"]
    assert len(clicks) == 1
    # Capture-circle OCR is telemetry only. The first target is on the
    # player-to-centre approach ray; later retries advance farther to centre.
    assert 710 <= clicks[0][0] < 810


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
