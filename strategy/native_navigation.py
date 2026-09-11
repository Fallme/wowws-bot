"""Conservative stalled-route detection independent of a green HUD label."""

import math


class NativeRouteWatchdog:
    def __init__(self):
        self.reset()

    def reset(self):
        self.anchor = None
        self.since = None
        self.last = None
        self.samples = 0

    def stalled(self, now, position, speed, *, cached=False, grace=False,
                timeout=20.0, low_speed=1.5):
        valid = (not grace and not cached and position is not None and
                 speed is not None and math.isfinite(speed) and abs(speed) <= low_speed and
                 all(math.isfinite(p) and 0 <= p <= 1 for p in position))
        if not valid:
            self.reset()
            return False
        if (self.last is None or not 0 < now - self.last <= 5 or
                math.dist(position, self.anchor) > .0015):
            self.anchor = position
            self.since = now
            self.samples = 0
        self.last = now
        self.samples += 1
        return self.samples >= 3 and now - self.since >= max(6.0, timeout)
