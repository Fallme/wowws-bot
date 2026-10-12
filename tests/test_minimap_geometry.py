from collections import deque
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from core.minimap_geometry import confirm_capture_zones, minimap_grid_bounds
from core.vision import CaptureZone, Vision


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


def grid_frame(width, height, size, inset=8):
    frame = np.full((height, width, 3), 30, np.uint8)
    left, top = width - inset - size, height - inset - size
    for offset in np.linspace(0, size, 11).round().astype(int):
        cv2.line(frame, (left + offset, top), (left + offset, top + size), (110, 110, 110), 1)
        cv2.line(frame, (left, top + offset), (left + size, top + offset), (110, 110, 110), 1)
    return frame, (left, top, left + size, top + size)


@pytest.mark.parametrize("width,height,size", [
    (1366, 768, 280), (1920, 1080, 320), (2560, 1440, 670),
    (3840, 2160, 400), (3840, 2160, 900), (1600, 1200, 850), (3440, 1440, 500),
])
def test_grid_follows_ui_size_independent_of_resolution(width, height, size):
    frame, bounds = grid_frame(width, height, size)
    assert minimap_grid_bounds(frame) == pytest.approx(bounds, abs=4)


def test_blank_and_port_pages_are_not_grid_anchors():
    assert minimap_grid_bounds(np.zeros((1080, 1920, 3), np.uint8)) is None
    assert minimap_grid_bounds(cv2.imread(str(Path("tests/fixtures/port_ship_selected.png")))) is None


def test_same_resolution_ui_resize_refreshes_bounds_and_missing_anchor_expires(monkeypatch):
    now = [0.0]
    monkeypatch.setattr("core.vision.time.monotonic", lambda: now[0])
    vision = Vision(screen_capture=object())
    initial_bounds = None
    for stamp, size in ((0, 320), (1, 450), (2, 450), (3, 450)):
        now[0] = stamp
        frame, bounds = grid_frame(1920, 1080, size)
        vision.find_minimap(frame)
        if stamp == 0:
            initial_bounds = vision.minimap_bounds
        expected = initial_bounds if stamp in (1, 2) else bounds
        assert vision.minimap_bounds == pytest.approx(expected, abs=4)
        assert vision.minimap_bounds_source == "grid"
    now[0] = 9
    vision.find_minimap(np.zeros_like(frame))
    assert vision.minimap_bounds_source == "ratio_fallback"


def test_same_size_crop_origin_change_clears_navigation_and_pose():
    from bot import BattleBot
    vision = SimpleNamespace(minimap_bounds=(100, 100, 500, 500), minimap_bounds_source="grid")
    bot = BattleBot(1, {"strategy": {}}, vision=vision, gamepad=object())
    bot._synchronize_minimap_geometry((400, 400, 3))
    bot._battle_map_islands = [{"points": [[.2, .2], [.3, .2], [.3, .3]]}]
    bot._cached_player_pose_normalized = (.2, .2)
    bot.island_route_planner.path = [(10, 10)]
    bot._synchronize_minimap_geometry((400, 400, 3))
    assert bot._battle_map_islands
    vision.minimap_bounds = (105, 105, 505, 505)
    bot._synchronize_minimap_geometry((400, 400, 3))
    assert not bot._battle_map_islands
    assert bot._cached_player_pose_normalized is None
    assert not bot.island_route_planner.path


@pytest.mark.parametrize("factor", [.75, 1., 1.5])
def test_live_capture_rings_survive_scaled_ui(factor):
    image = cv2.imread("tests/fixtures/live_battle.png")
    image = cv2.resize(image, None, fx=factor, fy=factor)
    vision = Vision(screen_capture=object())
    minimap = vision.find_minimap(image)
    zones = vision.find_capture_zones(minimap)
    assert [zone.state for zone in zones] == ["hostile", "friendly", "neutral"]
    assert [zone.center[0] / minimap.shape[1] for zone in zones] == pytest.approx([.178, .5, .822], abs=.015)
