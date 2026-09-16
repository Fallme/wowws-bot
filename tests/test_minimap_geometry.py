from collections import deque

from core.minimap_geometry import confirm_capture_zones
from core.vision import CaptureZone


def layer(offset=0):
    return [
        CaptureZone((100 + offset, 200), 40, "A", "neutral"),
        CaptureZone((300 + offset, 205), 41, "B", "friendly"),
        CaptureZone((500 + offset, 200), 40, "C", "hostile"),
    ]


def test_point_layer_requires_three_complete_consecutive_frames():
    samples = deque(maxlen=5)
    assert confirm_capture_zones(samples, layer(), (600, 600, 3)) is None
    assert confirm_capture_zones(samples, layer(2), (600, 600, 3)) is None
    confirmed = confirm_capture_zones(samples, layer(1), (600, 600, 3))
    assert confirmed is not None
    assert [zone.label for zone in confirmed] == ["A", "B", "C"]
    assert confirmed[0].center[0] == 101


def test_a_missing_point_frame_invalidates_old_vote():
    samples = deque(maxlen=5)
    for offset in (0, 1):
        assert confirm_capture_zones(samples, layer(offset), (600, 600, 3)) is None
    assert confirm_capture_zones(samples, layer()[:2], (600, 600, 3)) is None
    assert confirm_capture_zones(samples, layer(2), (600, 600, 3)) is None
    assert confirm_capture_zones(samples, layer(1), (600, 600, 3)) is None
    assert confirm_capture_zones(samples, layer(), (600, 600, 3)) is not None
