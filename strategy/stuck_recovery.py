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
    """Detect a stationary ship and perform a bounded reverse-turn maneuver."""

    def __init__(
        self,
        *,
        preferred_side=1,
        stationary_seconds=24.0,
        stationary_pixels=5.0,
        reverse_seconds=4.5,
        forward_seconds=4.5,
        cooldown_seconds=45.0,
    ):
        self.preferred_side = 1 if preferred_side >= 0 else -1
        self.stationary_seconds = stationary_seconds
        self.stationary_pixels = stationary_pixels
        self.reverse_seconds = reverse_seconds
        self.forward_seconds = forward_seconds
        self.cooldown_seconds = cooldown_seconds
        self.samples = deque()
        self.recovery_started = None
        self.cooldown_until = 0.0
        self.active_side = self.preferred_side
        self._next_fallback_side = self.preferred_side

    def reset(self):
        self.samples.clear()
        self.recovery_started = None
        self.cooldown_until = 0.0
        self.active_side = self.preferred_side
        self._next_fallback_side = self.preferred_side

    def cancel(self):
        """Yield immediately to a higher-priority live safety manoeuvre."""
        self.samples.clear()
        self.recovery_started = None

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

    def update(self, now, position, intended_throttle, escape_rudder=None):
        self._record(now, position)
        if self.recovery_started is not None:
            elapsed = now - self.recovery_started
            if elapsed < self.reverse_seconds:
                return RecoveryCommand(-1.0, self.active_side, "reverse")
            if elapsed < self.reverse_seconds + self.forward_seconds:
                return RecoveryCommand(0.82, -self.active_side, "forward_turn")
            self.recovery_started = None
            self.cooldown_until = now + self.cooldown_seconds
            self.samples.clear()
            return None

        if now < self.cooldown_until or intended_throttle < 0.65:
            return None
        if self._stationary(now):
            self.recovery_started = now
            if escape_rudder is not None and abs(float(escape_rudder)) >= 0.2:
                self.active_side = 1 if float(escape_rudder) > 0 else -1
            else:
                # With no reliable clearance signal, alternate recovery sides
                # so repeated contacts cannot produce the same circular trap.
                self.active_side = self._next_fallback_side
                self._next_fallback_side *= -1
            self.samples.clear()
            return RecoveryCommand(-1.0, self.active_side, "reverse")
        return None
