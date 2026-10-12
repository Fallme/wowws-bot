import cv2
import numpy as np
import pytest

from core.minimap_geometry import minimap_grid_bounds
from core.vision import Vision


def moved_grid(width, height, size, offset_x, offset_y):
    frame = np.full((height, width, 3), 30, dtype=np.uint8)
    left, top = width - size - offset_x, height - size - offset_y
    for offset in np.linspace(0, size, 11).round().astype(int):
        cv2.line(frame, (left + offset, top), (left + offset, top + size), (110, 110, 110), 1)
        cv2.line(frame, (left, top + offset), (left + size, top + offset), (110, 110, 110), 1)
    return frame, (left, top, left + size, top + size)


@pytest.mark.parametrize("width,height,size,offset_x,offset_y", [
    (1280, 720, 260, 70, 65),
    (1920, 1080, 400, 170, 120),
    (2560, 1440, 620, 150, 110),
    (3440, 1440, 440, 200, 160),
    (3840, 2160, 540, 280, 200),
    (1600, 1200, 480, 160, 90),
])
def test_grid_follows_position_independently_of_framebuffer(width, height, size, offset_x, offset_y):
    image, bounds = moved_grid(width, height, size, offset_x, offset_y)
    assert minimap_grid_bounds(image) == pytest.approx(bounds, abs=4)
    vision = Vision(screen_capture=object())
    minimap = vision.find_minimap(image)
    assert vision.minimap_bounds_source == "grid"
    assert vision.minimap_bounds == pytest.approx(bounds, abs=4)
    assert minimap.shape[:2] == pytest.approx((size, size), abs=4)


def test_one_axis_or_irregular_decorations_do_not_authorize_moved_map():
    image = np.full((1080, 1920, 3), 30, dtype=np.uint8)
    for offset in range(0, 401, 40):
        cv2.line(image, (1320 + offset, 560), (1320 + offset, 960), (110, 110, 110), 1)
    assert minimap_grid_bounds(image) is None
