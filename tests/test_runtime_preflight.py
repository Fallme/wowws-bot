from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from core.calibration import CalibrationStore
from core.ui import ScreenState
from main import (
    automatic_input_preflight,
    run_battle,
    tactical_map_local_point,
    wait_for_battle,
)


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
        patch("main.activate_window", return_value=True),
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
        patch("main.activate_window", return_value=True),
        patch("main.time.sleep", return_value=None),
    ):
        assert run_battle(ImmediateBot())

    assert events == ["reset", "full_speed", "analyze"]


def test_tactical_map_point_maps_minimap_center_to_screen_center():
    assert tactical_map_local_point(2560, 1600, (0.5, 0.5)) == (1280, 800)

    left_top = tactical_map_local_point(2560, 1600, (0.0, 0.0))
    right_bottom = tactical_map_local_point(2560, 1600, (1.0, 1.0))
    assert left_top[0] < 1280 < right_bottom[0]
    assert left_top[1] < 800 < right_bottom[1]
