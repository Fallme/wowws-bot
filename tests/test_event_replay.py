import json

from core.events import EventBus, JsonlEventRecorder, read_events
from core.replay import analyze_event_stream


def test_event_stream_round_trip_and_replay_report(tmp_path):
    destination = tmp_path / "events.jsonl"
    bus = EventBus()
    recorder = JsonlEventRecorder(destination)
    bus.subscribe(recorder)
    bus.publish(
        "battle.tick",
        tick=1,
        target_distance_km=9.4,
        inside_capture_point=False,
        distance_confidence=0.93,
        island_distance=None,
        movement_mode="hold_range",
        throttle=1.0,
        movement_verified=True,
    )
    bus.publish(
        "battle.tick",
        tick=2,
        target_distance_km=3.8,
        inside_capture_point=False,
        distance_confidence=0.91,
        island_distance=None,
        movement_mode="hold_range",
        throttle=1.0,
        movement_verified=True,
    )
    recorder.close()

    events = read_events(destination)
    assert [event.sequence for event in events] == [1, 2]
    report = analyze_event_stream(destination)
    assert report.ticks == 2
    assert report.distance_coverage == 1.0
    assert report.closest_distance_km == 3.8
    assert report.movement_modes == {"hold_range": 2}
    assert report.safety_violations == []


def test_replay_detects_advancing_inside_safety_boundaries(tmp_path):
    destination = tmp_path / "unsafe.jsonl"
    destination.write_text(
        json.dumps(
            {
                "sequence": 1,
                "topic": "battle.tick",
                "emitted_at": 1.0,
                "payload": {
                    "tick": 7,
                    "target_distance_km": 4.0,
                    "inside_capture_point": True,
                    "island_distance": 0.015,
                    "movement_mode": "approach",
                    "throttle": 1.0,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    report = analyze_event_stream(destination)
    assert len(report.safety_violations) == 2
