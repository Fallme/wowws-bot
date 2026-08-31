"""Small temporal filters for noisy visual point detections."""

from __future__ import annotations

import math
from collections import deque


class ConsecutivePointFilter:
    """Return only points that persist across consecutive observations."""

    def __init__(self, match_radius: float):
        self.match_radius = max(1.0, float(match_radius))
        self._previous = []

    def reset(self):
        self._previous = []

    def update(self, points):
        current = [(float(point[0]), float(point[1])) for point in points]
        confirmed = [
            point
            for point in current
            if any(math.dist(point, old) <= self.match_radius for old in self._previous)
        ]
        self._previous = current
        return [(round(point[0]), round(point[1])) for point in confirmed]


class ArrowHeadingFilter:
    """Smooth the live white-arrow vector without deriving heading from travel.

    The arrow contour can briefly expose its stern as the sharpest vertex when
    range rings, labels or an island overlap it.  A single large reversal is
    therefore held for one confirmation frame.  Ordinary turns are accepted
    immediately and blended on the unit circle, so the reported vector always
    remains the visible bow direction rather than a lagging displacement path.
    """

    def __init__(self, *, blend=0.72, reversal_dot=-0.15):
        self.blend = max(0.5, min(float(blend), 1.0))
        self.reversal_dot = max(-0.95, min(float(reversal_dot), 0.25))
        self.heading = None
        self._pending = None

    def reset(self):
        self.heading = None
        self._pending = None

    @staticmethod
    def _normalized(vector):
        if vector is None or len(vector) < 2:
            return None
        x, y = float(vector[0]), float(vector[1])
        magnitude = math.hypot(x, y)
        if not math.isfinite(magnitude) or magnitude <= 1e-6:
            return None
        return (x / magnitude, y / magnitude)

    def update(self, vector):
        raw = self._normalized(vector)
        if raw is None:
            return self.heading
        if self.heading is None:
            self.heading = raw
            return self.heading

        dot = self.heading[0] * raw[0] + self.heading[1] * raw[1]
        if dot < self.reversal_dot:
            pending = self._pending
            if pending is None or pending[0] * raw[0] + pending[1] * raw[1] < 0.82:
                self._pending = raw
                return self.heading
            # The same opposite bow direction persisted in two independent
            # captures.  Treat it as a completed hard turn, not contour noise.
            self.heading = self._normalized((pending[0] + raw[0], pending[1] + raw[1]))
            self._pending = None
            return self.heading

        self._pending = None
        old_weight = 1.0 - self.blend
        blended = (
            self.heading[0] * old_weight + raw[0] * self.blend,
            self.heading[1] * old_weight + raw[1] * self.blend,
        )
        normalized = self._normalized(blended)
        if normalized is not None:
            self.heading = normalized
        return self.heading


class CourseHeadingFilter:
    """Estimate the ship's real heading from minimap position history.

    The white player-arrow contour changes shape with zoom, range rings and
    overlays.  Its inferred tip can flip by 90-180 degrees between frames.
    Position displacement is slower to become available, but is authoritative
    once the ship has travelled several pixels.
    """

    def __init__(self, *, history=18, minimum_travel=5.0, maximum_jump=45.0):
        self.samples = deque(maxlen=max(6, int(history)))
        self.minimum_travel = max(2.0, float(minimum_travel))
        self.maximum_jump = max(self.minimum_travel * 2, float(maximum_jump))
        self.heading = None
        self._pending_reversal = None
        self._pending_reversal_samples = 0

    def reset(self):
        self.samples.clear()
        self.heading = None
        self._pending_reversal = None
        self._pending_reversal_samples = 0

    def update(self, point):
        if point is None:
            return self.heading
        current = (float(point[0]), float(point[1]))
        if self.samples and math.dist(current, self.samples[-1]) > self.maximum_jump:
            self.reset()
        self.samples.append(current)

        origin = None
        # Use the nearest *recent* sample with enough travel.  Using the oldest
        # point inside the history window makes a turning ship keep the heading
        # it had before the turn and is especially harmful after a broad
        # battleship U-turn: the controller then keeps adding the same rudder
        # and circles indefinitely.
        for candidate in reversed(tuple(self.samples)[:-1]):
            travel = math.dist(candidate, current)
            if self.minimum_travel <= travel <= self.maximum_jump:
                origin = candidate
                break
        if origin is None:
            return self.heading

        dx = current[0] - origin[0]
        dy = current[1] - origin[1]
        length = math.hypot(dx, dy)
        raw = (dx / length, dy / length)
        if self.heading is None:
            self.heading = raw
            return self.heading

        dot = self.heading[0] * raw[0] + self.heading[1] * raw[1]
        # One opposite displacement can be player-marker jitter.  A persistent
        # opposite course, however, is a real completed turn and must become
        # authoritative; otherwise every subsequent bearing is inverted.
        if dot < -0.20:
            pending = self._pending_reversal
            pending_dot = (
                -1.0
                if pending is None
                else pending[0] * raw[0] + pending[1] * raw[1]
            )
            if pending is not None and pending_dot >= 0.72:
                self._pending_reversal_samples += 1
                combined = (pending[0] + raw[0], pending[1] + raw[1])
                combined_length = math.hypot(*combined)
                if combined_length > 0:
                    self._pending_reversal = (
                        combined[0] / combined_length,
                        combined[1] / combined_length,
                    )
            else:
                self._pending_reversal = raw
                self._pending_reversal_samples = 1
            if self._pending_reversal_samples < 3:
                return self.heading
            self.heading = self._pending_reversal
            self._pending_reversal = None
            self._pending_reversal_samples = 0
            return self.heading
        self._pending_reversal = None
        self._pending_reversal_samples = 0
        blended = (
            self.heading[0] * 0.35 + raw[0] * 0.65,
            self.heading[1] * 0.35 + raw[1] * 0.65,
        )
        magnitude = math.hypot(*blended)
        if magnitude > 0:
            self.heading = (blended[0] / magnitude, blended[1] / magnitude)
        return self.heading
