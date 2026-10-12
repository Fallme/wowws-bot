"""Offline constant-velocity intercept solver; distances km, times seconds.

Use measured game-space speed, not uncalibrated displayed ship knots.
This calculation does not establish launcher readiness or a clear firing arc.
"""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class TorpedoSolution:
    time_seconds: float
    aim_position_km: tuple[float, float]
    travel_km: float


def solve_intercept(relative_position_km, target_velocity_km_s,
                    torpedo_speed_km_s, range_km):
    """Return earliest reachable intercept, or None for invalid/unreachable data.

    Position is relative to launch location. Velocity is target world velocity
    in the same axes, not target velocity relative to the moving own ship.
    """
    try:
        x, y = map(float, relative_position_km)
        vx, vy = map(float, target_velocity_km_s)
        speed, reach = float(torpedo_speed_km_s), float(range_km)
    except (TypeError, ValueError, OverflowError):
        return None
    if not all(map(math.isfinite, (x, y, vx, vy, speed, reach))):
        return None
    if speed <= 0 or reach <= 0 or math.hypot(x, y) == 0:
        return None
    a = vx * vx + vy * vy - speed * speed
    b = 2 * (x * vx + y * vy)
    c = x * x + y * y
    if abs(a) <= 1e-12 * max(speed * speed, vx * vx + vy * vy):
        roots = [-c / b] if b else []
    else:
        discriminant = b * b - 4 * a * c
        if discriminant < 0:
            return None
        q = -0.5 * (b + math.copysign(math.sqrt(discriminant), b))
        roots = [q / a, c / q] if q else []
    candidates = [t for t in roots if math.isfinite(t) and t > 0
                  and speed * t <= reach + 1e-9]
    if not candidates:
        return None
    t = min(candidates)
    return TorpedoSolution(t, (x + vx * t, y + vy * t), speed * t)
