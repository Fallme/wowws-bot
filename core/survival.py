"""Damage patterns from independent numeric HP observations."""

from collections import deque
import math


class HealthLossTracker:
    def __init__(self):
        self.samples = deque(maxlen=16)

    def reset(self):
        self.samples.clear()

    def observe(self, health, now):
        # A missed OCR frame is not a recovery. Keep the last independent
        # readings for their bounded freshness window so one unreadable frame
        # cannot repeatedly erase a real damage streak.
        if health is None or not math.isfinite(health):
            return
        if not 0 < health <= 1:
            self.reset()
            return
        if self.samples:
            previous_time, previous_health = self.samples[-1]
            if now <= previous_time:
                return
            if now - previous_time > 8 or health - previous_health > 0.0005:
                self.reset()
        self.samples.append((now, health))

    def _recent_samples(self, now, window_seconds):
        if not self.samples or not 0 <= now - self.samples[-1][0] <= 8:
            return []
        return [
            sample for sample in self.samples
            if 0 <= now - sample[0] <= window_seconds
        ]

    def burst_loss(self, now, *, minimum_loss=0.08, window_seconds=8.0):
        """Confirm a large drop with a second independent lower-HP reading.

        The confirming read can be stable. A single OCR spike that recovers
        on the next read cannot spend damage control.
        """
        samples = self._recent_samples(now, window_seconds)
        if len(samples) < 3:
            return False
        baseline = max(health for _, health in samples[:-2])
        return (
            baseline - samples[-2][1] >= minimum_loss - 1e-9
            and samples[-1][1] <= samples[-2][1] + 0.0005
        )

    def sustained_loss(self, now, *, minimum_loss=0.003, window_seconds=8.0,
                       minimum_seconds=2.0):
        samples = self._recent_samples(now, window_seconds)
        if len(samples) < 3:
            return False
        decreases = [
            current_time
            for (_, previous_health), (current_time, current_health)
            in zip(samples, samples[1:])
            if previous_health - current_health >= 0.0005 - 1e-9
        ]
        return (
            len(decreases) >= 2
            and samples[-1][0] - samples[0][0] >= minimum_seconds
            and now - decreases[-1] <= 8
            and samples[0][1] - samples[-1][1] >= minimum_loss - 1e-9
        )
