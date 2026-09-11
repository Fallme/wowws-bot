"""Estimated physical helm and fresh minimap motion, independent of key IO."""

import math


class YawMotion:
    """Fresh clockwise angular velocity from the minimap bow, in rad/s."""

    def __init__(self):
        self.anchor = None
        self.rate = None
        self.observed_at = None

    def observe(self, heading, now):
        angle = math.atan2(heading[1], heading[0])
        if self.anchor is None:
            self.anchor = (angle, now)
            return
        previous, stamp = self.anchor
        dt = now - stamp
        if dt <= 0 or dt > 5:
            self.__init__()
            self.anchor = (angle, now)
            return
        if dt < 1:
            return
        delta = math.atan2(math.sin(angle - previous), math.cos(angle - previous))
        self.anchor = (angle, now)
        rate = delta / dt
        if abs(rate) > math.radians(15):
            self.rate = self.observed_at = None
            return
        self.rate = rate if self.rate is None else .5 * (self.rate + rate)
        self.observed_at = now

    def fresh_rate(self, now):
        return self.rate if self.observed_at is not None and 0 <= now - self.observed_at <= 3 else None


class HelmModel:
    def __init__(self, shift_seconds=15.0):
        self.shift_seconds = max(1.0, float(shift_seconds))
        self.actual = 0.0
        self.order = 0.0
        self.updated_at = None

    def advance(self, now):
        if self.updated_at is not None:
            if now < self.updated_at:
                return self.actual
            travel = max(0.0, now - self.updated_at) / self.shift_seconds
            self.actual += max(-travel, min(self.order - self.actual, travel))
        self.updated_at = now
        return self.actual

    def command(self, value, now):
        self.advance(now)
        # Match KeyboardController's actual five Q/E notches.
        magnitude = 0.0 if abs(value) < 0.10 else (0.5 if abs(value) < 0.68 else 1.0)
        self.order = math.copysign(magnitude, value)


class MapMotion:
    """Estimate map km/s across multi-second windows; reject pose jumps."""

    def __init__(self):
        self.anchor = None
        self.speed = None
        self.observed_at = None

    def observe(self, position, now, map_span_km):
        if self.anchor is None:
            self.anchor = (position, now)
            return
        old_position, old_time = self.anchor
        dt = now - old_time
        if dt > 12 or dt <= 0:
            self.anchor = (position, now)
            self.speed = None
            self.observed_at = None
            return
        if dt < 3:
            return
        speed = math.dist(position, old_position) * map_span_km / dt
        self.anchor = (position, now)
        if not 0 <= speed <= 0.20:
            self.speed = None
            self.observed_at = None
            return
        self.speed = speed if self.speed is None else self.speed * 0.5 + speed * 0.5
        self.observed_at = now

    def fresh_speed(self, now):
        if self.observed_at is None or not 0 <= now - self.observed_at <= 5:
            return None
        return self.speed
