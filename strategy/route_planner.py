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


def far_side_waypoint(player, zone, *, fraction: float = 0.70):
    """Return a point beyond a capture-zone centre from the player's approach.

    Clicking the near edge or exact centre often makes game-native autopilot
    stop short.  A target on the far side carries the ship fully into the cap.
    """
    if player is None or zone is None:
        return None
    dx = float(zone.center[0]) - float(player[0])
    dy = float(zone.center[1]) - float(player[1])
    length = math.hypot(dx, dy)
    if length < 1.0:
        return tuple(float(value) for value in zone.center)
    offset = float(zone.radius) * max(0.35, min(float(fraction), 0.85))
    return (
        float(zone.center[0]) + dx / length * offset,
        float(zone.center[1]) + dy / length * offset,
    )


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
        self._retarget_candidate = None
        self._retarget_samples = 0
        self.snapshot = RouteSnapshot()

    def observe_zone(self, zone, minimap_shape, *, allow_retarget: bool = False) -> bool:
        """Accept the first central zone and reject later target switching."""
        if zone is None:
            return False
        if self.zone is None:
            self.zone = zone
            return True
        scale = max(1.0, float(min(minimap_shape[:2])))
        separation = math.dist(self.zone.center, zone.center)
        if separation > scale * self.zone_match_ratio:
            if not allow_retarget:
                return False
            # A Hough circle can briefly jump from a real cap to a range ring
            # or island bay.  Keep displaying/following the locked point
            # until the alternate point is independently seen three times.
            candidate = (
                getattr(zone, "label", ""),
                round(zone.center[0] / scale, 2),
                round(zone.center[1] / scale, 2),
            )
            if candidate == self._retarget_candidate:
                self._retarget_samples += 1
            else:
                self._retarget_candidate = candidate
                self._retarget_samples = 1
            if self._retarget_samples < 3:
                return False
            # The selected point was captured by our team or disappeared and
            # a live neutral/red A/B/C/D point is now nearer. Start a fresh
            # route from the current player pose rather than following an old
            # map-specific objective.
            self.zone = zone
            self.start_position = None
            self.initial_distance = None
            self.entry_waypoint = None
            self.arrived = False
            self._max_progress = 0.0
            self._retarget_candidate = None
            self._retarget_samples = 0
            return True
        # Smooth only matching observations so capture animation jitter cannot
        # make the route bearing jump.
        center = (
            int(round(self.zone.center[0] * 0.85 + zone.center[0] * 0.15)),
            int(round(self.zone.center[1] * 0.85 + zone.center[1] * 0.15)),
        )
        radius = self.zone.radius * 0.85 + zone.radius * 0.15
        self.zone = type(zone)(
            center=center,
            radius=radius,
            label=getattr(zone, "label", ""),
            state=getattr(zone, "state", "unknown"),
        )
        self._retarget_candidate = None
        self._retarget_samples = 0
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
            # Aim through the cap centre to its far side.  A near-edge target
            # causes both native autopilot and generic steering to stop short.
            self.entry_waypoint = far_side_waypoint(player, self.zone)

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
            target = self.entry_waypoint
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
