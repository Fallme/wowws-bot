"""Offline summaries and safety checks for recorded battle event streams."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

from core.events import read_events


@dataclass(frozen=True)
class ReplayReport:
    ticks: int
    distance_ticks: int
    distance_coverage: float
    verified_ticks: int
    closest_distance_km: float | None
    average_distance_confidence: float
    movement_modes: dict[str, int]
    safety_violations: list[str]

    def to_dict(self):
        return asdict(self)


def analyze_event_stream(
    source: Path,
    *,
    too_close_km: float = 4.5,
    island_emergency_distance: float = 0.020,
) -> ReplayReport:
    ticks = [event for event in read_events(source) if event.topic == "battle.tick"]
    distances = [
        float(
            event.payload.get("target_distance_km")
            if event.payload.get("target_distance_km") is not None
            else event.payload["minimap_distance_km"]
        )
        for event in ticks
        if event.payload.get("target_distance_km") is not None
        or event.payload.get("minimap_distance_km") is not None
    ]
    confidences = [
        float(event.payload.get("distance_confidence") or 0)
        for event in ticks
        if event.payload.get("target_distance_km") is not None
    ]
    modes = Counter(str(event.payload.get("movement_mode") or "unknown") for event in ticks)
    violations = []
    for event in ticks:
        payload = event.payload
        throttle = payload.get("throttle")
        island = payload.get("island_distance")
        inside_capture = bool(payload.get("inside_capture_point"))
        movement_mode = str(payload.get("movement_mode") or "")
        if throttle is None:
            continue
        if inside_capture and float(throttle) > 0.65:
            violations.append(
                f"tick {payload.get('tick')}: full speed while inside capture point"
            )
        if (
            not inside_capture
            and movement_mode not in {"avoid_island", "recovery"}
            and 0 <= float(throttle) < 0.8
        ):
            violations.append(
                f"tick {payload.get('tick')}: reduced speed outside capture point"
            )
        if island is not None and float(island) <= island_emergency_distance and float(throttle) > 0:
            violations.append(
                f"tick {payload.get('tick')}: emergency island range while advancing"
            )

    total = len(ticks)
    return ReplayReport(
        ticks=total,
        distance_ticks=len(distances),
        distance_coverage=(len(distances) / total) if total else 0.0,
        verified_ticks=sum(
            bool(event.payload.get("movement_verified")) for event in ticks
        ),
        closest_distance_km=min(distances) if distances else None,
        average_distance_confidence=(sum(confidences) / len(confidences))
        if confidences
        else 0.0,
        movement_modes=dict(sorted(modes.items())),
        safety_violations=violations,
    )
