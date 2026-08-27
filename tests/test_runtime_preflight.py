from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from core.calibration import AUTOMATIC_PREFLIGHT_KEY, CalibrationStore
from core.ui import ScreenState
from main import (
    automatic_input_preflight,
    configure_opening_autopilot,
    prepare_battle,
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
    with (
        patch("main.time.sleep", return_value=None),
        patch("main.confirm_no_commander") as confirmation,
    ):
        assert wait_for_battle(bot, timeout=2)

    confirmation.assert_not_called()


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
