import importlib
import sys
import types
import unittest
from unittest.mock import Mock, patch

class FakeGamepad:
    def __init__(self):
        self.calls = []

    def left_joystick(self, **values):
        self.calls.append(("left_joystick", values))

    def right_joystick(self, **values):
        self.calls.append(("right_joystick", values))

    def right_trigger(self, **values):
        self.calls.append(("right_trigger", values))

    def press_button(self, **values):
        self.calls.append(("press_button", values))

    def release_button(self, **values):
        self.calls.append(("release_button", values))

    def update(self):
        self.calls.append(("update", {}))


def load_controller_module(constructor):
    buttons = types.SimpleNamespace(
        XUSB_GAMEPAD_A="a",
        XUSB_GAMEPAD_B="b",
        XUSB_GAMEPAD_X="x",
        XUSB_GAMEPAD_Y="y",
    )
    fake_vgamepad = types.SimpleNamespace(
        VX360Gamepad=constructor,
        XUSB_BUTTON=buttons,
    )
    with patch.dict(sys.modules, {"vgamepad": fake_vgamepad}):
        sys.modules.pop("core.gamepad", None)
        return importlib.import_module("core.gamepad")


class GamepadControllerTests(unittest.TestCase):
    def test_injected_device_is_reused(self):
        constructor = Mock(side_effect=AssertionError("must not create another device"))
        module = load_controller_module(constructor)
        device = FakeGamepad()

        controller = module.GamepadController(device, pulse_seconds=0)
        controller.set_movement(0.5, -0.25)
        controller.fire()

        constructor.assert_not_called()
        self.assertIn(
            ("left_joystick", {"x_value": int(-0.25 * 32767), "y_value": int(0.5 * 32767)}),
            device.calls,
        )
        self.assertIn(("right_trigger", {"value": 255}), device.calls)

    def test_stop_neutralizes_continuous_controls(self):
        module = load_controller_module(lambda: FakeGamepad())
        device = FakeGamepad()
        controller = module.GamepadController(device, pulse_seconds=0)

        controller.steer_right(1.0)
        controller.full_speed()
        controller.stop()

        self.assertIn(("left_joystick", {"x_value": 0, "y_value": 0}), device.calls)
        self.assertIn(("right_joystick", {"x_value": 0, "y_value": 0}), device.calls)
        self.assertIn(("right_trigger", {"value": 0}), device.calls)

    def test_set_movement_clamps_both_axes_in_one_update(self):
        module = load_controller_module(lambda: FakeGamepad())
        device = FakeGamepad()
        controller = module.GamepadController(device, pulse_seconds=0)

        controller.set_movement(1.5, -2)

        self.assertIn(
            ("left_joystick", {"x_value": -32767, "y_value": 32767}),
            device.calls,
        )


if __name__ == "__main__":
    unittest.main()
