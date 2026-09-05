"""Fresh numeric HP evidence for the damage-control fallback."""

from collections import deque
import math


class HealthLossTracker:
    def __init__(self):
        self.samples = deque(maxlen=4)

    def reset(self):
        self.samples.clear()

    def observe(self, health, now):
        if health is None or not math.isfinite(health) or not 0 < health <= 1:
            self.reset()
            return
        if self.samples:
            previous_time, previous_health = self.samples[-1]
            if now <= previous_time:
                return
            if now - previous_time > 8 or previous_health - health < 0.0005:
                self.reset()
        self.samples.append((now, health))

    def sustained_loss(self, now):
        return (
            len(self.samples) == 4
            and 0 <= now - self.samples[-1][0] <= 8
            and self.samples[0][1] - self.samples[-1][1] >= 0.003
        )
