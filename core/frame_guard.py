"""Quality and freshness checks for captured game frames."""

from __future__ import annotations

from dataclasses import dataclass
import time

import cv2
import numpy as np


class CaptureFault(RuntimeError):
    """Raised when the runtime cannot prove it has a usable live frame."""


@dataclass(frozen=True)
class FrameQuality:
    valid: bool
    reason: str
    mean: float
    contrast: float
    delta: float
    stale_seconds: float


class FrameGuard:
    def __init__(
        self,
        *,
        stale_after: float = 8.0,
        minimum_mean: float = 3.0,
        minimum_contrast: float = 4.0,
        change_threshold: float = 0.35,
    ):
        self.stale_after = max(1.0, float(stale_after))
        self.minimum_mean = float(minimum_mean)
        self.minimum_contrast = float(minimum_contrast)
        self.change_threshold = max(0.0, float(change_threshold))
        self._previous = None
        self._last_changed_at = None

    @staticmethod
    def _signature(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.resize(gray, (64, 36), interpolation=cv2.INTER_AREA)

    def reset(self):
        self._previous = None
        self._last_changed_at = None

    def inspect(self, image, now: float | None = None) -> FrameQuality:
        current = time.monotonic() if now is None else float(now)
        if image is None or not isinstance(image, np.ndarray) or image.size == 0:
            return FrameQuality(False, "capture_missing", 0.0, 0.0, 0.0, 0.0)
        if image.ndim != 3 or image.shape[2] < 3:
            return FrameQuality(False, "capture_format_invalid", 0.0, 0.0, 0.0, 0.0)

        mean = float(image.mean())
        contrast = float(image.std())
        if mean < self.minimum_mean or contrast < self.minimum_contrast:
            return FrameQuality(False, "capture_black_or_blank", mean, contrast, 0.0, 0.0)

        signature = self._signature(image)
        if self._previous is None:
            delta = 255.0
            self._last_changed_at = current
        else:
            delta = float(cv2.absdiff(signature, self._previous).mean())
            if delta >= self.change_threshold:
                self._last_changed_at = current
        self._previous = signature
        last_changed = current if self._last_changed_at is None else self._last_changed_at
        stale_seconds = max(0.0, current - last_changed)
        if stale_seconds >= self.stale_after:
            return FrameQuality(False, "capture_stale", mean, contrast, delta, stale_seconds)
        return FrameQuality(True, "ok", mean, contrast, delta, stale_seconds)
