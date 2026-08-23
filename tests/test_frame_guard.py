import numpy as np

from core.frame_guard import FrameGuard


def textured_frame(offset=0):
    yy, xx = np.indices((120, 180))
    base = ((xx * 3 + yy * 5 + offset) % 255).astype(np.uint8)
    return np.dstack((base, np.roll(base, 3, axis=1), np.roll(base, 5, axis=0)))


def test_black_frame_is_rejected():
    guard = FrameGuard()
    quality = guard.inspect(np.zeros((100, 100, 3), dtype=np.uint8), now=0)
    assert not quality.valid
    assert quality.reason == "capture_black_or_blank"


def test_changing_frames_remain_valid():
    guard = FrameGuard(stale_after=2, change_threshold=0.1)
    assert guard.inspect(textured_frame(0), now=0).valid
    assert guard.inspect(textured_frame(10), now=1).valid
    assert guard.inspect(textured_frame(20), now=2.5).valid


def test_identical_frames_trip_stale_guard():
    guard = FrameGuard(stale_after=2, change_threshold=0.1)
    frame = textured_frame()
    assert guard.inspect(frame, now=0).valid
    quality = guard.inspect(frame.copy(), now=2.1)
    assert not quality.valid
    assert quality.reason == "capture_stale"
