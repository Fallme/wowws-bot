from core.vision import CaptureZone
from strategy.route_planner import CoarseRoutePlanner


def test_route_locks_first_central_zone_and_rejects_target_switch():
    planner = CoarseRoutePlanner(zone_match_ratio=0.08)
    first = CaptureZone(center=(200, 200), radius=40)
    other = CaptureZone(center=(320, 200), radius=40)

    assert planner.observe_zone(first, (500, 500, 3))
    assert not planner.observe_zone(other, (500, 500, 3))
    assert planner.zone.center == first.center


def test_route_progresses_to_station_and_keeps_arrival_state():
    planner = CoarseRoutePlanner()
    planner.observe_zone(CaptureZone(center=(250, 200), radius=40), (500, 500, 3))

    departure = planner.update((100, 200))
    transit = planner.update((170, 200))
    station = planner.update((240, 200), inside_zone=True)
    after_avoidance = planner.update((180, 200), inside_zone=False)

    assert departure.phase == "departure"
    assert transit.phase == "transit"
    assert station.phase == "station"
    assert station.progress == 1.0
    assert after_avoidance.phase == "station"
    assert after_avoidance.arrived
