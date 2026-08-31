"""Low-level keyboard and mouse control for the game.

World of Warships uses a latched engine telegraph: W/S change a discrete
speed notch while A/D are continuous rudder controls.  Treating the left
stick as two absolute axes therefore does not work unless the game has an
explicit controller mapping.  This controller models the native keyboard
semantics and only emits input when a requested state changes.
"""

from __future__ import annotations

import ctypes
import ctypes.wintypes
from dataclasses import dataclass
import logging
import time


KEYEVENTF_KEYUP = 0x0002
INPUT_MOUSE = 0
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004

VK = {
    "a": 0x41,
    "d": 0x44,
    "e": 0x45,
    "m": 0x4D,
    "q": 0x51,
    "s": 0x53,
    "w": 0x57,
    "x": 0x58,
    "3": 0x33,
    "4": 0x34,
    "t": 0x54,
    "r": 0x52,
    "u": 0x55,
    "y": 0x59,
    "esc": 0x1B,
}

logger = logging.getLogger("input")

ULONG_PTR = ctypes.wintypes.WPARAM


class _MOUSEINPUT(ctypes.Structure):
    _fields_ = (
        ("dx", ctypes.wintypes.LONG),
        ("dy", ctypes.wintypes.LONG),
        ("mouseData", ctypes.wintypes.DWORD),
        ("dwFlags", ctypes.wintypes.DWORD),
        ("time", ctypes.wintypes.DWORD),
        ("dwExtraInfo", ULONG_PTR),
    )


class _INPUTUNION(ctypes.Union):
    _fields_ = (("mi", _MOUSEINPUT),)


class _INPUT(ctypes.Structure):
    _anonymous_ = ("union",)
    _fields_ = (("type", ctypes.wintypes.DWORD), ("union", _INPUTUNION))


class SendInputBackend:
    """Inject the virtual-key events accepted by this game installation.

    Live testing showed that scan-code ``SendInput`` calls are acknowledged by
    Windows but ignored by the game. ``keybd_event`` virtual-key input changes
    the engine telegraph immediately, so keyboard events deliberately use that
    path. Mouse clicks still use the atomic ``SendInput`` structure below.
    """

    def __init__(self, tap_seconds: float = 0.035):
        self.tap_seconds = max(0.015, float(tap_seconds))
        self.user32 = ctypes.windll.user32
        self.kernel32 = ctypes.windll.kernel32
        self.last_injected_tick_ms: int | None = None

    def _mark_injected(self):
        self.last_injected_tick_ms = int(self.kernel32.GetTickCount())

    def _send_key(self, key: str, *, key_up: bool):
        virtual_key = VK[key]
        flags = KEYEVENTF_KEYUP if key_up else 0
        self.user32.keybd_event(virtual_key, 0, flags, 0)
        self._mark_injected()

    def key_down(self, key: str):
        self._send_key(key, key_up=False)

    def key_up(self, key: str):
        self._send_key(key, key_up=True)

    def tap(self, key: str):
        self.key_down(key)
        time.sleep(self.tap_seconds)
        self.key_up(key)

    def left_click(self):
        down = _INPUT(type=INPUT_MOUSE, mi=_MOUSEINPUT(0, 0, 0, MOUSEEVENTF_LEFTDOWN, 0, 0))
        up = _INPUT(type=INPUT_MOUSE, mi=_MOUSEINPUT(0, 0, 0, MOUSEEVENTF_LEFTUP, 0, 0))
        events = (_INPUT * 2)(down, up)
        sent = self.user32.SendInput(2, events, ctypes.sizeof(_INPUT))
        if sent != 2:
            raise ctypes.WinError()
        self._mark_injected()


@dataclass(frozen=True)
class KeyboardDispatch:
    action: str
    dispatched_at: float
    throttle: float = 0.0
    rudder: float = 0.0
    throttle_notch: int = 0
    rudder_notch: int = 0


class KeyboardController:
    """Native W/S engine-telegraph and A/D rudder controller."""

    backend_name = "windows_native_keyboard"
    MAX_NOTCH = 4
    MAX_RUDDER_NOTCH = 2

    def __init__(self, backend=None, *, focus_guard=None):
        self.device = backend or SendInputBackend()
        # Native key injection is global in Windows.  The caller supplies a
        # guard bound to the verified game HWND so no W/Q/E/R/T/M input can be
        # delivered to a browser or another monitor's active application.
        self._focus_guard = focus_guard
        self._throttle_notch = 0
        self._rudder_notch = 0
        self.last_dispatch: KeyboardDispatch | None = None
        self.dispatch_count = 0
        self._automation_observer = None
        self._last_observed_injected_tick_ms: int | None = None

    @staticmethod
    def _clamp(value: float) -> float:
        return max(-1.0, min(float(value), 1.0))

    @classmethod
    def _notch_for(cls, throttle: float) -> int:
        value = cls._clamp(throttle)
        if abs(value) < 0.10:
            return 0
        # Battle automation is forward-only.  A visual navigation fault must
        # never be able to request reverse telegraph through this controller.
        if value < 0:
            return 0
        magnitude = max(1, min(cls.MAX_NOTCH, round(abs(value) * cls.MAX_NOTCH)))
        return magnitude

    def _record(self, action: str, throttle: float, rudder: float):
        self.dispatch_count += 1
        self.last_dispatch = KeyboardDispatch(
            action=action,
            dispatched_at=time.time(),
            throttle=throttle,
            rudder=rudder,
            throttle_notch=self._throttle_notch,
            rudder_notch=self._rudder_notch,
        )
        injected_tick = self.last_injected_tick_ms
        if (
            injected_tick is not None
            and injected_tick != self._last_observed_injected_tick_ms
        ):
            self._last_observed_injected_tick_ms = injected_tick
            observer = self._automation_observer
            if observer is not None:
                try:
                    observer(self)
                except Exception:
                    # The input has already been dispatched. Telemetry cleanup
                    # must never turn that accepted command into a retry/fault.
                    logger.exception("自动输入监听回执失败；保留已派发的游戏指令")

    def set_automation_observer(self, observer=None):
        """Observe completed native input so it cannot become user activity."""
        self._automation_observer = observer

    def _ensure_target_focus(self):
        if self._focus_guard is not None and not self._focus_guard():
            raise RuntimeError("游戏窗口不在前台，拒绝发送键盘操作")

    def _set_throttle_notch(self, target: int):
        target = max(0, min(int(target), self.MAX_NOTCH))
        delta = target - self._throttle_notch
        if delta:
            # The native telegraph can be changed by game autopilot or manual
            # takeover while our cache is frozen. This applies to both cached
            # upshifts and downshifts: four W taps from an actual FULL-reverse
            # state only reach STOP. Eight W taps first establish FULL ahead
            # from every possible state, then bounded S taps select a
            # non-negative target notch.
            for _ in range(self.MAX_NOTCH * 2):
                self.device.tap("w")
            for _ in range(self.MAX_NOTCH - target):
                self.device.tap("s")
            self._throttle_notch = target

    def _set_rudder(self, rudder: float):
        value = self._clamp(rudder)
        if abs(value) < 0.10:
            target = 0
        else:
            magnitude = 1 if abs(value) < 0.68 else self.MAX_RUDDER_NOTCH
            target = magnitude if value > 0 else -magnitude
        delta = target - self._rudder_notch
        key = "e" if delta > 0 else "q"
        for _ in range(abs(delta)):
            self.device.tap(key)
        self._rudder_notch = target

    def set_movement(self, throttle: float, rudder: float):
        self._ensure_target_focus()
        throttle = self._clamp(throttle)
        rudder = self._clamp(rudder)
        self._set_throttle_notch(self._notch_for(throttle))
        self._set_rudder(rudder)
        self._record("movement", throttle, rudder)

    def steer_left(self, amount: float = 0.7):
        self.set_movement(self._throttle_notch / self.MAX_NOTCH, -abs(amount))

    def steer_right(self, amount: float = 0.7):
        self.set_movement(self._throttle_notch / self.MAX_NOTCH, abs(amount))

    def straight(self):
        self.set_movement(self._throttle_notch / self.MAX_NOTCH, 0.0)

    def full_speed(self):
        self.set_movement(1.0, 0.0)

    def reassert_full_speed(self):
        """Resend FULL ahead even when the cached telegraph already says FULL.

        The battle HUD can become visible a fraction of a second before it
        starts accepting movement keys. In that case the initial W taps are
        dropped while the controller cache still advances to notch four.
        Reasserting is harmless because the in-game telegraph clamps at FULL.
        """
        self._ensure_target_focus()
        self.device.key_up("s")
        # From FULL reverse to FULL ahead spans eight telegraph steps.
        for _ in range(self.MAX_NOTCH * 2):
            self.device.tap("w")
        self._throttle_notch = self.MAX_NOTCH
        self._record("full_speed_reassert", 1.0, 0.0)

    def toggle_tactical_map(self):
        self._ensure_target_focus()
        self.device.tap("m")
        self._record(
            "toggle_tactical_map",
            self._throttle_notch / self.MAX_NOTCH,
            self._rudder_notch / self.MAX_RUDDER_NOTCH,
        )

    def resynchronize_forward_controls(self):
        """Cancel external steering and establish FULL-ahead plus neutral rudder."""
        self._ensure_target_focus()
        self.device.key_up("q")
        self.device.key_up("e")
        # Four Q taps reach hard-left from any real rudder state; two E taps
        # then land exactly at neutral. This repairs cache drift left by native
        # autopilot or a manual keyboard takeover without guessing direction.
        for _ in range(self.MAX_RUDDER_NOTCH * 2):
            self.device.tap("q")
        for _ in range(self.MAX_RUDDER_NOTCH):
            self.device.tap("e")
        self._rudder_notch = 0
        self.reassert_full_speed()

    def takeover_from_autopilot(self):
        """Backward-compatible alias for deterministic control hand-off."""
        self.resynchronize_forward_controls()

    def fire(self):
        self._ensure_target_focus()
        self.device.left_click()
        self._record("main_fire", self._throttle_notch / self.MAX_NOTCH, 0.0)

    def lock(self):
        self._ensure_target_focus()
        # X is the native target-lock command.  It is deterministic and avoids
        # clicking an unverified screen coordinate.
        self.device.tap("x")
        self._record("target_lock", self._throttle_notch / self.MAX_NOTCH, 0.0)

    def torpedo(self):
        self._ensure_target_focus()
        self.device.tap("3")
        self._record("torpedo", self._throttle_notch / self.MAX_NOTCH, 0.0)

    def smoke(self):
        self._ensure_target_focus()
        self.device.tap("4")
        self._record("smoke", self._throttle_notch / self.MAX_NOTCH, 0.0)

    def escape(self):
        """Open/back out of a game menu inside the already-focused client."""
        self._ensure_target_focus()
        self.device.tap("esc")
        self._record("escape", self._throttle_notch / self.MAX_NOTCH, 0.0)

    def damage_control(self):
        self._ensure_target_focus()
        self.device.tap("r")
        self._record("damage_control", self._throttle_notch / self.MAX_NOTCH, 0.0)

    def heal(self):
        self._ensure_target_focus()
        self.device.tap("t")
        self._record("repair_party", self._throttle_notch / self.MAX_NOTCH, 0.0)

    def use_consumable_cycle(self):
        """Try every common consumable slot once for ship-agnostic recovery.

        Consumable assignments vary by ship (for example Naples uses T for
        smoke rather than Repair Party).  The health-threshold workflow uses
        this bounded cycle so it does not need to infer each ship's dynamic
        loadout from icons.  The caller owns the cooldown and 20% HP gating.
        """
        self._ensure_target_focus()
        for key in ("r", "t", "u", "y"):
            try:
                self.device.tap(key)
            except (KeyError, OSError, ValueError) as error:
                # One unavailable/unsupported slot must not abort an entire
                # multi-battle run. Continue probing the remaining slots and
                # leave an auditable warning in the live console.
                logger.warning("消耗品按键 %s 派发失败，继续尝试下一槽位: %s", key, error)
        self._record(
            "consumable_cycle",
            self._throttle_notch / self.MAX_NOTCH,
            0.0,
        )

    def use_other_consumables(self):
        """Try non-damage-control consumables without touching the R slot.

        R has its own fire/flood trigger and cooldown.  Keeping T/U/Y in a
        separate bounded cycle prevents ordinary HP loss from wasting damage
        control before a fire or flooding event is actually visible.
        """
        self._ensure_target_focus()
        for key in ("t", "u", "y"):
            try:
                self.device.tap(key)
            except (KeyError, OSError, ValueError) as error:
                logger.warning("其他消耗品按键 %s 派发失败，继续尝试下一槽位: %s", key, error)
        self._record(
            "other_consumables",
            self._throttle_notch / self.MAX_NOTCH,
            0.0,
        )

    @property
    def last_injected_tick_ms(self):
        return getattr(self.device, "last_injected_tick_ms", None)

    def pause_automation(self):
        """Keep the current telegraph and Q/E rudder notch unchanged."""
        self._record(
            "manual_intervention_pause",
            self._throttle_notch / self.MAX_NOTCH,
            0.0,
        )

    def note_automation_activity(self):
        """Mark focus-management key events as bot-generated input."""
        marker = getattr(self.device, "_mark_injected", None)
        if marker is not None:
            marker()

    def stop(self):
        """Deterministically establish STOP and neutral rudder from any state."""
        self._ensure_target_focus()
        # Cached notches can be stale after native autopilot or manual input.
        # Saturate each real game control to one endpoint, then step back to
        # neutral. This is bounded and correct for every engine/rudder state.
        self.device.key_up("q")
        self.device.key_up("e")
        for _ in range(self.MAX_RUDDER_NOTCH * 2):
            self.device.tap("q")
        for _ in range(self.MAX_RUDDER_NOTCH):
            self.device.tap("e")
        self._rudder_notch = 0
        self.device.key_up("s")
        for _ in range(self.MAX_NOTCH * 2):
            self.device.tap("w")
        for _ in range(self.MAX_NOTCH):
            self.device.tap("s")
        self._throttle_notch = 0
        # Release all movement keys even if an earlier injection was interrupted.
        for key in ("a", "d", "q", "e", "w", "s"):
            self.device.key_up(key)
        self._record("neutral_stop", 0.0, 0.0)
