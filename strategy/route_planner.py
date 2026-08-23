"""Persistent coarse route planning for station battles.

The visual detector can rediscover a different capture circle as UI colours
change.  This planner locks the central objective once per battle, creates a
simple entry waypoint, and keeps the arrival state after the ship reaches the
zone.  Local island and torpedo avoidance remain short-lived overrides around
this persistent route.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class RouteSnapshot:
    phase: str = "unplanned"
    progress: float = 0.0
    waypoint_index: int = 0
    waypoint_count: int = 2
    arrived: bool = False
    target: tuple[float, float] | None = None


class CoarseRoutePlanner:
    """Lock one central cap and expose departure/entry/station phases."""

    def __init__(self, *, zone_match_ratio: float = 0.08):
        self.zone_match_ratio = max(0.03, min(float(zone_match_ratio), 0.15))
        self.reset()

    def reset(self):
        self.zone = None
        self.start_position = None
        self.initial_distance = None
        self.entry_waypoint = None
        self.arrived = False
        self._max_progress = 0.0
        self.snapshot = RouteSnapshot()

    def observe_zone(self, zone, minimap_shape) -> bool:
        """Accept the first central zone and reject later target switching."""
        if zone is None:
            return False
        if self.zone is None:
            self.zone = zone
            return True
        scale = max(1.0, float(min(minimap_shape[:2])))
        separation = math.dist(self.zone.center, zone.center)
        if separation > scale * self.zone_match_ratio:
            return False
        # Smooth only matching observations so capture animation jitter cannot
        # make the route bearing jump.
        center = (
            int(round(self.zone.center[0] * 0.85 + zone.center[0] * 0.15)),
            int(round(self.zone.center[1] * 0.85 + zone.center[1] * 0.15)),
        )
        radius = self.zone.radius * 0.85 + zone.radius * 0.15
        self.zone = type(zone)(center=center, radius=radius)
        return True

    def update(self, player, *, inside_zone: bool = False) -> RouteSnapshot:
        if self.zone is None or player is None:
            self.snapshot = RouteSnapshot(arrived=self.arrived)
            return self.snapshot
        if self.start_position is None:
            self.start_position = tuple(player)
            self.initial_distance = max(math.dist(player, self.zone.center), 1.0)
            dx = self.start_position[0] - self.zone.center[0]
            dy = self.start_position[1] - self.zone.center[1]
            length = max(math.hypot(dx, dy), 1.0)
            # First target is the near edge of the capture circle.  The second
            # waypoint is its centre and is used for station keeping.
            entry_radius = self.zone.radius * 0.62
            self.entry_waypoint = (
                self.zone.center[0] + dx / length * entry_radius,
                self.zone.center[1] + dy / length * entry_radius,
            )

        remaining = math.dist(player, self.zone.center)
        raw_progress = 1.0 - remaining / max(self.initial_distance, 1.0)
        self._max_progress = max(self._max_progress, max(0.0, min(raw_progress, 1.0)))
        if inside_zone:
            self.arrived = True
            self._max_progress = 1.0

        if self.arrived:
            phase = "station"
            waypoint_index = 2
            target = self.zone.center
        elif remaining <= self.zone.radius * 1.55:
            phase = "final_approach"
            waypoint_index = 1
            target = self.zone.center
        elif self._max_progress < 0.08:
            phase = "departure"
            waypoint_index = 0
            target = self.entry_waypoint
        else:
            phase = "transit"
            waypoint_index = 0
            target = self.entry_waypoint

        self.snapshot = RouteSnapshot(
            phase=phase,
            progress=self._max_progress,
            waypoint_index=waypoint_index,
            arrived=self.arrived,
            target=target,
        )
        return self.snapshot
