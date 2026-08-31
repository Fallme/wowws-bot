"""Closed-loop movement feedback and safety faults."""

from __future__ import annotations

from dataclasses import dataclass
import math


class SafetyFault(RuntimeError):
    """Raised when an issued command cannot be verified from the game view."""


@dataclass(frozen=True)
class MovementFeedback:
    verified: bool
    pending: bool
    displacement: float
    reason: str


class MovementFeedbackMonitor:
    def __init__(
        self,
        *,
        timeout_seconds: float = 18.0,
        missing_timeout_seconds: float = 10.0,
        movement_pixels: float = 4.0,
    ):
        self.timeout_seconds = max(2.0, float(timeout_seconds))
        self.missing_timeout_seconds = max(2.0, float(missing_timeout_seconds))
        self.movement_pixels = max(1.0, float(movement_pixels))
        self.reset()

    def reset(self):
        self.started_at = None
        self.origin = None
        self.last_seen_at = None
        self.verified = False

    def update(self, now: float, position, intended_throttle: float) -> MovementFeedback:
        now = float(now)
        # The automation contract is forward-only. Negative input is treated
        # as no verifiable movement request rather than normalizing reverse as
        # a supported operating mode.
        if intended_throttle < 0.55:
            return MovementFeedback(self.verified, False, 0.0, "throttle_below_check_threshold")
        if self.started_at is None:
            self.started_at = now

        if position is not None:
            point = (float(position[0]), float(position[1]))
            self.last_seen_at = now
            if self.origin is None:
                self.origin = point
            displacement = math.dist(self.origin, point)
            if displacement >= self.movement_pixels:
                self.verified = True
                return MovementFeedback(True, False, displacement, "movement_observed")
        else:
            displacement = 0.0

        if self.verified:
            return MovementFeedback(True, False, displacement, "movement_previously_verified")
        if self.last_seen_at is None and now - self.started_at >= self.missing_timeout_seconds:
            raise SafetyFault("无法识别玩家小地图位置，不能验证控制反馈")
        if now - self.started_at >= self.timeout_seconds:
            raise SafetyFault("已发送航行指令，但未观察到舰船位置变化")
        return MovementFeedback(False, True, displacement, "waiting_for_movement")
