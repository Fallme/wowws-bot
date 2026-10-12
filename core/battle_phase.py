"""Temporal battle-phase evidence, separate from visual HUD presence."""
from enum import Enum

from core.ui import ScreenState


class BattlePhase(str, Enum):
    LOADING = "loading"
    PREPARING = "preparing"
    ACTIVE = "active"
    UNKNOWN = "unknown"


class BattlePhaseTracker:
    def __init__(self, duration_seconds=1200):
        self.duration_seconds = duration_seconds
        self.previous = None
        self.active = False

    def observe(self, screen, clock, now, *, active_evidence=False):
        if screen != ScreenState.BATTLE:
            if screen in {ScreenState.LOADING, ScreenState.PORT, ScreenState.RESULTS}:
                self.previous = None
                self.active = False
            return BattlePhase.LOADING if screen == ScreenState.LOADING else BattlePhase.UNKNOWN
        if clock is None or not 0 <= clock <= self.duration_seconds:
            self.previous = None
            return BattlePhase.UNKNOWN
        if clock == self.duration_seconds:
            self.previous = (clock, now)
            self.active = False
            return BattlePhase.PREPARING
        if self.active:
            self.previous = (clock, now)
            return BattlePhase.ACTIVE
        previous = self.previous
        self.previous = (clock, now)
        # Short preparation countdowns are not the match timer. Once active,
        # however, the final minute is still a live battle.
        if clock <= 60 and not active_evidence:
            return BattlePhase.PREPARING
        if previous is not None:
            value, observed_at = previous
            elapsed = now - observed_at
            if 0 < elapsed <= 10 and 0 < value - clock <= elapsed + 2:
                self.active = True
                return BattlePhase.ACTIVE
        return BattlePhase.UNKNOWN
