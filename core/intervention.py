"""Pause automated battle input briefly when the player uses the keyboard."""

from __future__ import annotations

import ctypes
import ctypes.wintypes
import logging
import os
from pathlib import Path
import time


# Native SendInput events reach GetLastInputInfo almost immediately.  Keep a
# very small grace window for our own key injection, but never mask a real
# player keypress for nearly half a second: doing so made manual intervention
# occasionally miss its first key and let the bot steal focus back.
AUTOMATION_TICK_TOLERANCE_MS = 120
# GetAsyncKeyState's transition bit can become visible a little after the
# keybd_event call that produced it.  Limit the late-drain window to the game
# foreground and to a recently acknowledged automation batch.  This prevents
# the bot's own W/Q/E/M edges from being paired with a later mouse event and
# mislabeled as a human keypress.
AUTOMATION_KEY_SETTLE_SECONDS = 0.75
logger = logging.getLogger("intervention")


class _LASTINPUTINFO(ctypes.Structure):
    _fields_ = (
        ("cbSize", ctypes.wintypes.UINT),
        ("dwTime", ctypes.wintypes.DWORD),
    )


def _last_input_tick() -> int:
    info = _LASTINPUTINFO(cbSize=ctypes.sizeof(_LASTINPUTINFO), dwTime=0)
    if not ctypes.windll.user32.GetLastInputInfo(ctypes.byref(info)):
        return 0
    return int(info.dwTime)


def _foreground_window() -> int:
    return int(ctypes.windll.user32.GetForegroundWindow() or 0)


def _keyboard_activity() -> bool:
    """Return whether any real/injected keyboard key changed since last poll.

    ``GetLastInputInfo`` reports mouse and keyboard together.  The low bit of
    ``GetAsyncKeyState`` records a key transition since our previous query, so
    scanning keyboard virtual-key codes lets ordinary mouse movement/clicks be
    ignored without installing a global hook.
    """
    user32 = ctypes.windll.user32
    for virtual_key in range(0x08, 0xFF):
        # Bit 0 means the key transitioned since our previous query. Bit 15
        # only means that it is currently down. Treating bit 15 as activity
        # made a held/stale key look new whenever an automated cursor move
        # changed GetLastInputInfo, so the bot cancelled its own UI click.
        if int(user32.GetAsyncKeyState(virtual_key)) & 0x0001:
            return True
    return False


def _tick_distance(left: int, right: int) -> int:
    """Return the unsigned 32-bit distance between Windows tick counters."""
    return min((left - right) & 0xFFFFFFFF, (right - left) & 0xFFFFFFFF)


class UserInterventionMonitor:
    """Observe real keyboard input without mistaking our own injections.

    ``GetLastInputInfo`` is system-wide. Keyboard activity therefore pauses
    automation even after the user has switched to another window. An explicit
    foreground switch also starts one pause window, while mouse-only activity
    is ignored. Native injected ticks are ignored.
    """

    def __init__(
        self,
        hwnd,
        *,
        pause_seconds: float = 4.0,
        latch_seconds: float = 20.0,
        resume_path=None,
        pause_path=None,
        input_tick_reader=None,
        keyboard_activity_reader=None,
        foreground_reader=None,
    ):
        self.hwnd = int(hwnd or 0)
        self.pause_seconds = max(1.0, float(pause_seconds))
        self.latch_seconds = max(self.pause_seconds, float(latch_seconds))
        configured_resume_path = resume_path or os.environ.get("WOWS_RESUME_FILE")
        self.resume_path = Path(configured_resume_path) if configured_resume_path else None
        configured_pause_path = pause_path or os.environ.get("WOWS_PAUSE_FILE")
        self.pause_path = Path(configured_pause_path) if configured_pause_path else None
        self._input_tick_reader = input_tick_reader or _last_input_tick
        self._keyboard_activity_reader = keyboard_activity_reader or (
            _keyboard_activity if input_tick_reader is None else lambda: True
        )
        self._foreground_reader = foreground_reader or _foreground_window
        self._last_seen_tick: int | None = None
        self.pause_until = 0.0
        self.intervention_started_at: float | None = None
        self.last_user_input_at: float | None = None
        self.latched = False
        self.web_paused = False
        self.resumed_from_web = False
        self._last_foreground: int | None = None
        self._automation_keyboard_quiet_until = 0.0
        self.last_trigger = ""

    def reset(self):
        self._last_seen_tick = self._input_tick_reader()
        self._keyboard_activity_reader()
        self._last_foreground = self._foreground_reader()
        self._automation_keyboard_quiet_until = 0.0
        self.pause_until = 0.0
        self.intervention_started_at = None
        self.last_user_input_at = None
        self.latched = False
        self.web_paused = False
        self.resumed_from_web = False
        self.last_trigger = ""

    def _consume_web_resume(self) -> bool:
        if self.resume_path is None or not self.resume_path.exists():
            return False
        try:
            self.resume_path.unlink()
        except OSError:
            return False
        self._last_seen_tick = self._input_tick_reader()
        self.pause_until = 0.0
        self.intervention_started_at = None
        self.last_user_input_at = None
        self.latched = False
        self.web_paused = False
        self.resumed_from_web = True
        self.last_trigger = "web_resume"
        # The Continue button is normally clicked from the browser.  Seed the
        # foreground baseline there so that this intentional Web action does
        # not immediately look like a fresh switch away from the game.
        self._last_foreground = self._foreground_reader()
        return True

    def _note_user_intervention(self, now: float, *, trigger: str) -> None:
        """Extend the quiet-period pause for one verified user action."""
        if (
            self.intervention_started_at is None
            or self.last_user_input_at is None
            or now - self.last_user_input_at > self.pause_seconds
        ):
            self.intervention_started_at = now
        self.last_trigger = trigger
        self.last_user_input_at = now
        self.pause_until = now + self.pause_seconds
        was_latched = self.latched
        if now - self.intervention_started_at >= self.latch_seconds:
            self.latched = True
        if self.latched and not was_latched:
            logger.warning(
                "[USER] 连续操作达到 %.0f 秒，已锁定暂停；仅网页“继续”可恢复",
                self.latch_seconds,
            )

    @staticmethod
    def _automation_tick(controller) -> int | None:
        value = getattr(controller, "last_injected_tick_ms", None)
        return None if value is None else int(value)

    @staticmethod
    def _automation_ticks(controller) -> tuple[int, ...]:
        values = getattr(controller, "recent_injected_key_ticks_ms", None)
        if not values:
            tick = UserInterventionMonitor._automation_tick(controller)
            return () if tick is None else (tick,)
        try:
            return tuple(int(value) for value in values)
        except (TypeError, ValueError):
            return ()

    @classmethod
    def _matches_automation_tick(cls, controller, current_tick: int) -> bool:
        return any(
            _tick_distance(current_tick, injected_tick)
            <= AUTOMATION_TICK_TOLERANCE_MS
            for injected_tick in cls._automation_ticks(controller)
        )

    def acknowledge_automation(self, controller=None) -> None:
        """Consume state left by a known automation keyboard dispatch.

        ``GetAsyncKeyState`` keeps a transition bit until somebody reads it.
        A tactical-map M press followed by an automated cursor move therefore
        used to look like one late *human* keyboard event: LASTINPUTINFO pointed
        at the newer mouse move while the old M transition was still pending.
        Explicitly acknowledging our key dispatch drains that transition and
        establishes a new baseline.  Real keyboard input after this call and
        foreground switches are still observed normally; mouse-only activity
        never enters the pause state.
        """
        current_tick = self._input_tick_reader()
        injected_ticks = self._automation_ticks(controller)
        # Do not drain a real key that happened after the bot dispatch. The
        # controller observer calls this immediately, so a wider distance is
        # evidence that LASTINPUTINFO now belongs to another input source.
        if (
            injected_ticks
            and not self._matches_automation_tick(controller, current_tick)
        ):
            return
        self._last_seen_tick = current_tick
        self._keyboard_activity_reader()
        if injected_ticks:
            self._automation_keyboard_quiet_until = (
                time.monotonic() + AUTOMATION_KEY_SETTLE_SECONDS
            )
        current_foreground = self._foreground_reader()
        if current_foreground:
            self._last_foreground = current_foreground

    def poll(self, controller, now: float | None = None) -> bool:
        now = time.monotonic() if now is None else float(now)
        self.resumed_from_web = False
        if self.pause_path is not None and self.pause_path.exists():
            self.latched = True
            self.web_paused = True
            return True
        if self._consume_web_resume():
            return False
        # Losing foreground after the game was active is itself a user
        # takeover signal.  This closes the focus-stealing race even when the
        # Alt+Tab/Win key transition occurs while OCR is busy, or when the
        # user switches applications with the taskbar.  Ordinary mouse use
        # inside the current window still does not pause automation.
        current_foreground = self._foreground_reader()
        previous_foreground = self._last_foreground
        if current_foreground:
            self._last_foreground = current_foreground
        if (
            previous_foreground == self.hwnd
            and current_foreground not in (0, self.hwnd)
        ):
            logger.info(
                "[USER] 检测到切屏: 游戏窗口=%s，当前前台=%s；至少静默 %.0f 秒前不切回游戏",
                self.hwnd,
                current_foreground,
                self.pause_seconds,
            )
            self._note_user_intervention(now, trigger="window_switch")
        current_tick = self._input_tick_reader()
        previous_tick = self._last_seen_tick
        self._last_seen_tick = current_tick

        if previous_tick is None or current_tick == previous_tick:
            if self.latched:
                return True
            if now >= self.pause_until and self.intervention_started_at is not None:
                self.intervention_started_at = None
                self.last_user_input_at = None
            return now < self.pause_until
        # Mouse movement/clicks are never user-intervention signals.  Window
        # switching is handled independently above, and only a subsequent
        # keyboard event may extend that pause or make it latch.
        if not self._keyboard_activity_reader():
            return self.latched or now < self.pause_until

        is_automation = self._matches_automation_tick(controller, current_tick)
        # The async transition bit for a known injected key can arrive after
        # its exact LASTINPUTINFO tick has already been replaced by mouse
        # motion.  Drain that delayed edge only while the game remains in the
        # foreground; a genuine Alt+Tab is still caught by window_switch.
        is_late_automation = (
            current_foreground == self.hwnd
            and time.monotonic() <= self._automation_keyboard_quiet_until
        )
        if not is_automation and not is_late_automation:
            self._note_user_intervention(now, trigger="keyboard")
        return self.latched or now < self.pause_until

    @property
    def remaining_seconds(self) -> float:
        if self.latched:
            return 0.0
        return max(0.0, self.pause_until - time.monotonic())

    def continuous_seconds(self, now: float | None = None) -> float:
        if self.intervention_started_at is None:
            return 0.0
        current = time.monotonic() if now is None else float(now)
        return max(0.0, current - self.intervention_started_at)

    def command_generation_paused(self, now: float | None = None) -> bool:
        """Inspect pause state without consuming resume files or input events."""
        if self.resume_path is not None and self.resume_path.exists():
            return False
        if self.pause_path is not None and self.pause_path.exists():
            return True
        current = time.monotonic() if now is None else float(now)
        return bool(self.latched or current < self.pause_until)
