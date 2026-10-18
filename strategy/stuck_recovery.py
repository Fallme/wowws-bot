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
    """Back clear of a collision before requesting a fresh route."""

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
        # Keep legacy constructor arguments compatible with existing configs.
        self.reverse_seconds = max(2.0, min(float(
            12.0 if reverse_seconds is None else reverse_seconds
        ), 20.0))
        self.forward_seconds = max(2.0, float(forward_seconds))
        self.cooldown_seconds = max(3.0, float(cooldown_seconds))
        self.escape_turn_seconds = max(2.0, float(escape_turn_seconds))
        # A recovery that ends where the previous one did is a replay of the
        # same collision.  Remember the last replan point and force an
        # opposite rudder hold so the next attempt leaves that spot.
        self.loop_window_seconds = 45.0
        self.loop_distance = 0.012
        self.samples = deque()
        self.recovery_started = None
        self.cooldown_until = 0.0
        self.low_speed_started = None
        self.active_side = self.preferred_side
        self._next_fallback_side = self.preferred_side
        self.reverse_until = None
        self._last_observation_at = None
        self._last_position = None
        self._last_cycle_end = None
        self._forced_side = 0
        self._forced_until = 0.0
        self.recovery_loops = 0

    def reset(self):
        self._last_observation_at = None
        self.reverse_until = None
        self.samples.clear()
        self.recovery_started = None
        self.cooldown_until = 0.0
        self.low_speed_started = None
        self.active_side = self.preferred_side
        self._next_fallback_side = self.preferred_side
        self._last_position = None
        self._last_cycle_end = None
        self._forced_side = 0
        self._forced_until = 0.0
        self.recovery_loops = 0

    def cancel(self):
        """Yield immediately to a higher-priority live safety manoeuvre."""
        self._last_observation_at = None
        self.samples.clear()
        self.recovery_started = None
        self.low_speed_started = None
        self.reverse_until = None

    def forced_turn(self, now):
        """Rudder side to hold after the same recovery spot repeated."""
        if now < self._forced_until:
            return self._forced_side
        return 0

    def begin_reverse(self, now, escape_rudder=None, seconds=None):
        """Arm bounded reverse for either native or local navigation."""
        command = self._begin_recovery(now, escape_rudder)
        if seconds is not None:
            self.reverse_until = now + max(2.0, min(float(seconds), 20.0))
        return command

    def _record(self, now, position):
        if self._last_observation_at is not None and not 0 < now - self._last_observation_at <= 5.0:
            self.samples.clear()
            self.low_speed_started = None
        self._last_observation_at = now
        if position is None:
            self.samples.clear()
            return
        self._last_position = tuple(position)
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
        looping = False
        if self._last_cycle_end is not None and self._last_position is not None:
            ended_at, ended_position = self._last_cycle_end
            looping = (
                0 < now - ended_at <= self.loop_window_seconds
                and math.dist(ended_position, self._last_position) <= self.loop_distance
            )
        self.recovery_started = now
        self.low_speed_started = None
        if escape_rudder is not None and abs(float(escape_rudder)) >= 0.2:
            self.active_side = 1 if float(escape_rudder) > 0 else -1
        else:
            # Legacy diagnostic side; actual steering is replanned after reverse.
            self.active_side = self._next_fallback_side
            self._next_fallback_side *= -1
        if looping:
            # The previous replan drove straight back into the same obstacle.
            # Force the opposite turn so the next route starts elsewhere.
            self.recovery_loops += 1
            self._forced_side = -self.active_side if self.active_side else -self.preferred_side
            self._forced_until = now + self.escape_turn_seconds
        else:
            self.recovery_loops = 0
            self._forced_side = 0
            self._forced_until = 0.0
        self.samples.clear()
        self.reverse_until = now + self.reverse_seconds
        return RecoveryCommand(-1.0, 0.0, "reverse_clear")

    def update(
        self,
        now,
        position,
        intended_throttle,
        escape_rudder=None,
        speed_knots=None,
    ):
        self._record(now, position)
        if self.reverse_until is not None:
            if now < self.reverse_until:
                return RecoveryCommand(-1.0, 0.0, "reverse_clear")
            self._last_cycle_end = (now, self._last_position)
            self.cancel()
            self.cooldown_until = now + self.cooldown_seconds
            return RecoveryCommand(0.0, 0.0, "replan")

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
