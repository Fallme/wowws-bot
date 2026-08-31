"""Reliable minimap-motion based collision recovery."""

from collections import deque
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class RecoveryCommand:
    throttle: float
    rudder: float
    phase: str


class StuckRecoveryController:
    """Detect a stationary ship and perform a bounded forward-only escape."""

    def __init__(
        self,
        *,
        preferred_side=1,
        stationary_seconds=18.0,
        stationary_pixels=5.0,
        low_speed_seconds=8.0,
        low_speed_knots=1.5,
        escape_turn_seconds=8.0,
        forward_seconds=6.0,
        cooldown_seconds=12.0,
        reverse_seconds=None,
    ):
        self.preferred_side = 1 if preferred_side >= 0 else -1
        self.stationary_seconds = max(4.0, float(stationary_seconds))
        self.stationary_pixels = max(1.0, float(stationary_pixels))
        self.low_speed_seconds = max(3.0, float(low_speed_seconds))
        self.low_speed_knots = max(0.4, float(low_speed_knots))
        # ``reverse_seconds`` is accepted only for old integrations. It now
        # controls the first forward-turn phase and can never enable reverse.
        self.escape_turn_seconds = float(
            escape_turn_seconds if reverse_seconds is None else reverse_seconds
        )
        self.forward_seconds = max(2.0, float(forward_seconds))
        self.cooldown_seconds = max(3.0, float(cooldown_seconds))
        self.samples = deque()
        self.recovery_started = None
        self.cooldown_until = 0.0
        self.low_speed_started = None
        self.active_side = self.preferred_side
        self._next_fallback_side = self.preferred_side

    def reset(self):
        self.samples.clear()
        self.recovery_started = None
        self.cooldown_until = 0.0
        self.low_speed_started = None
        self.active_side = self.preferred_side
        self._next_fallback_side = self.preferred_side

    def cancel(self):
        """Yield immediately to a higher-priority live safety manoeuvre."""
        self.samples.clear()
        self.recovery_started = None
        self.low_speed_started = None

    def _record(self, now, position):
        if position is None:
            self.samples.clear()
            return
        self.samples.append((now, position))
        oldest = now - self.stationary_seconds
        while self.samples and self.samples[0][0] < oldest:
            self.samples.popleft()

    def _stationary(self, now):
        if len(self.samples) < 8:
            return False
        if now - self.samples[0][0] < self.stationary_seconds * 0.92:
            return False
        points = [position for _, position in self.samples]
        origin = points[0]
        return max(math.dist(origin, point) for point in points) <= self.stationary_pixels

    def _low_speed_stalled(self, now, speed_knots, intended_throttle):
        """Detect a beached ship even when its marker jitters or drifts.

        A ship touching terrain can slide several minimap pixels while making
        only 0.4-0.6 kt. Position-only detection therefore never called it
        stationary. Numeric speed plus a forward command is authoritative for
        this case and deliberately independent from the collision text OCR.
        """
        if intended_throttle < 0.25 or speed_knots is None:
            self.low_speed_started = None
            return False
        try:
            speed = abs(float(speed_knots))
        except (TypeError, ValueError):
            self.low_speed_started = None
            return False
        if not math.isfinite(speed) or speed > self.low_speed_knots:
            self.low_speed_started = None
            return False
        if self.low_speed_started is None:
            self.low_speed_started = now
            return False
        return now - self.low_speed_started >= self.low_speed_seconds

    def _begin_recovery(self, now, escape_rudder):
        self.recovery_started = now
        self.low_speed_started = None
        if escape_rudder is not None and abs(float(escape_rudder)) >= 0.2:
            self.active_side = 1 if float(escape_rudder) > 0 else -1
        else:
            # With no reliable clearance signal, alternate recovery sides so
            # a failed contact cannot repeat the same circular trap forever.
            self.active_side = self._next_fallback_side
            self._next_fallback_side *= -1
        self.samples.clear()
        return RecoveryCommand(1.0, self.active_side, "forward_escape_turn")

    def update(
        self,
        now,
        position,
        intended_throttle,
        escape_rudder=None,
        speed_knots=None,
    ):
        self._record(now, position)
        if self.recovery_started is not None:
            elapsed = now - self.recovery_started
            if elapsed < self.escape_turn_seconds:
                # Reverse is never a legal automation command: a stale position
                # detector must not make the ship back across the map. Full
                # ahead is intentional: the former 0.32 command could not pull
                # a battleship free from terrain contact.
                return RecoveryCommand(1.0, self.active_side, "forward_escape_turn")
            if elapsed < self.escape_turn_seconds + self.forward_seconds:
                return RecoveryCommand(0.82, -self.active_side, "forward_clear")
            self.recovery_started = None
            self.cooldown_until = now + self.cooldown_seconds
            self.samples.clear()
            self.low_speed_started = None
            return None

        low_speed_stalled = self._low_speed_stalled(
            now,
            speed_knots,
            intended_throttle,
        )
        if now < self.cooldown_until or intended_throttle < 0.25:
            return None
        if low_speed_stalled or self._stationary(now):
            return self._begin_recovery(now, escape_rudder)
        return None
