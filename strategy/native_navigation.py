"""Conservative stalled-route detection independent of a green HUD label."""

import math


class NativeRouteWatchdog:
    def __init__(self):
        self.reset()

    def reset(self):
        self._reset_stall()
        self._reset_loss()

    def _reset_stall(self):
        self.anchor = None
        self.since = None
        self.last = None
        self.samples = 0

    def _reset_loss(self):
        self.loss_since = None
        self.loss_last = None
        self.loss_samples = 0

    def lost_indicator(self, now, visible, speed, *, grace=False, timeout=12.0):
        # Only fresh negative OCR samples count; missing reads do not vote.
        if (grace or speed is None or not math.isfinite(speed) or abs(speed) > 1.5
                or visible is True):
            self._reset_loss()
            return False
        if self.loss_last is not None and not 0 < now - self.loss_last <= 12.0:
            self._reset_loss()
        if visible is not False:
            return False
        if self.loss_since is None:
            self.loss_since = now
        self.loss_last = now
        self.loss_samples += 1
        return self.loss_samples >= 3 and now - self.loss_since >= max(6.0, timeout)

    def stalled(self, now, position, speed, *, cached=False, grace=False,
                timeout=20.0, low_speed=1.5):
        valid = (not grace and not cached and position is not None and
                 speed is not None and math.isfinite(speed) and abs(speed) <= low_speed and
                 all(math.isfinite(p) and 0 <= p <= 1 for p in position))
        if not valid:
            self._reset_stall()
            return False
        if (self.last is None or not 0 < now - self.last <= 5 or
                math.dist(position, self.anchor) > .0015):
            self.anchor = position
            self.since = now
            self.samples = 0
        self.last = now
        self.samples += 1
        return self.samples >= 3 and now - self.since >= max(6.0, timeout)
