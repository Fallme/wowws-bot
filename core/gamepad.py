"""Virtual gamepad output with explicit dispatch telemetry.

Dispatching a driver update only proves that Python called vgamepad. It does
not prove that the game accepted the command; calibration and visual feedback
provide that proof.
"""

from __future__ import annotations

from dataclasses import dataclass
import time

import vgamepad as vg


@dataclass(frozen=True)
class CommandDispatch:
    action: str
    dispatched_at: float
    throttle: float = 0.0
    rudder: float = 0.0


class GamepadController:
    """Own one virtual Xbox controller and always provide a neutral stop."""

    def __init__(self, gamepad=None, pulse_seconds: float = 0.06):
        self.device = gamepad or vg.VX360Gamepad()
        self.pulse_seconds = max(0.02, float(pulse_seconds))
        self._left_x = 0.0
        self._left_y = 0.0
        self.last_dispatch: CommandDispatch | None = None
        self.dispatch_count = 0

    @property
    def raw_device(self):
        return self.device

    def _record(self, action: str):
        self.dispatch_count += 1
        self.last_dispatch = CommandDispatch(
            action,
            time.time(),
            throttle=self._left_y,
            rudder=self._left_x,
        )

    def _update_left_stick(self, action="movement"):
        self.device.left_joystick(
            x_value=int(self._left_x * 32767),
            y_value=int(self._left_y * 32767),
        )
        self.device.update()
        self._record(action)

    def _pulse_button(self, action, button):
        self.device.press_button(button=button)
        self.device.update()
        time.sleep(self.pulse_seconds)
        self.device.release_button(button=button)
        self.device.update()
        self._record(action)

    def steer_left(self, amount: float = 0.7):
        self._left_x = -max(0.0, min(float(amount), 1.0))
        self._update_left_stick("rudder_left")

    def steer_right(self, amount: float = 0.7):
        self._left_x = max(0.0, min(float(amount), 1.0))
        self._update_left_stick("rudder_right")

    def straight(self):
        self._left_x = 0.0
        self._update_left_stick("rudder_center")

    def set_movement(self, throttle: float, rudder: float):
        # Keep the optional legacy backend under the same forward-only safety
        # contract as the native keyboard controller.
        self._left_y = max(0.0, min(float(throttle), 1.0))
        self._left_x = max(-1.0, min(float(rudder), 1.0))
        self._update_left_stick("movement")

    def full_speed(self):
        self.set_movement(1.0, self._left_x)

    def reassert_full_speed(self):
        self.full_speed()

    def resynchronize_forward_controls(self):
        self._left_x = 0.0
        self._left_y = 1.0
        self._update_left_stick("forward_controls_resynchronized")

    def takeover_from_autopilot(self):
        self.resynchronize_forward_controls()

    def fire(self):
        self.device.right_trigger(value=255)
        self.device.update()
        time.sleep(self.pulse_seconds)
        self.device.right_trigger(value=0)
        self.device.update()
        self._record("main_fire")

    def lock(self):
        self._pulse_button("secondary_lock", vg.XUSB_BUTTON.XUSB_GAMEPAD_A)

    def torpedo(self):
        self._pulse_button("torpedo", vg.XUSB_BUTTON.XUSB_GAMEPAD_X)

    def smoke(self):
        self._pulse_button("smoke", vg.XUSB_BUTTON.XUSB_GAMEPAD_B)

    def pause_automation(self):
        """Centre steering but leave the existing engine command untouched."""
        self._left_x = 0.0
        self._update_left_stick("manual_intervention_pause")

    def note_automation_activity(self):
        pass

    def stop(self):
        """Return all continuous controls to neutral, even after a failure."""
        self._left_x = 0.0
        self._left_y = 0.0
        self.device.left_joystick(x_value=0, y_value=0)
        self.device.right_joystick(x_value=0, y_value=0)
        self.device.right_trigger(value=0)
        self.device.update()
        self._record("neutral_stop")
