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
