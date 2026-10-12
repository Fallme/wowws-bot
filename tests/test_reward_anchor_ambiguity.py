import numpy as np

from core.results import ResultRewardReader
from test_ui_scale_navigation import GeometryBackend


def test_unrelated_horizontal_numbers_far_from_earned_caption_are_rejected():
    image = np.zeros((1200, 2200, 3), dtype=np.uint8)
    backend = GeometryBackend()
    backend.paint(image, "在战斗中获得", (120, 650, 300, 680))
    for text, x in (("12345", 1350), ("999", 1550), ("200", 1710)):
        backend.paint(image, text, (x, 700, x + 100, 730))
    result = ResultRewardReader(backend).read(image)
    assert not result.recognized
    assert result.resource_values() == {"credits": 0, "ship_xp": 0, "free_xp": 0}


def test_earned_caption_does_not_authorize_clipped_values_in_old_fixed_crops():
    image = np.zeros((1000, 1600, 3), dtype=np.uint8)
    backend = GeometryBackend()
    backend.paint(image, "在战斗中获得", (280, 325, 460, 355))
    # Numeric-only fragments fit the previous fixed profiles but do not form
    # the actual complete, aligned resource row.
    backend.paint(image, "4567", (285, 375, 350, 400))
    backend.paint(image, "222", (462, 390, 515, 415))
    backend.paint(image, "333", (595, 418, 640, 440))
    result = ResultRewardReader(backend).read(image)
    assert not result.recognized
