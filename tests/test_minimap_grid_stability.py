"""Grid jitter and a one-cell alias must not repeatedly reset map layers."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from core.ui import MINIMAP_REGION
from core.vision import Vision
from test_minimap_position_adaptation import moved_grid


CANONICAL = (1881, 921, 2557, 1597)
ALIAS_X = (1814, 921, 2487, 1596)
ALIAS_Y = (1881, 854, 2555, 1528)
MOVED = (1801, 871, 2477, 1547)


def frame(shape=(1600, 2560)):
    image = np.full((*shape, 3), 65, np.uint8)
    image[::4, ::4] = 190
    return image


def layout(bounds):
    return SimpleNamespace(source=lambda _name: "anchor" if bounds is not None else "fallback",
                           rect=lambda _name: bounds)


def replay(bounds, times, images=None):
    vision = Vision(screen_capture=object())
    selected, sources = [], []
    with patch("core.vision.detect_hud_layout", side_effect=[layout(value) for value in bounds]):
        for index, now in enumerate(times):
            image = frame() if images is None else images[index]
            with patch("core.vision.time.monotonic", return_value=now):
                vision.find_minimap(image)
            selected.append(vision.minimap_bounds)
            sources.append(vision.minimap_bounds_source)
    return vision, selected, sources


def test_four_pixel_jitter_and_single_cell_aliases_keep_the_canonical_crop():
    bounds = [CANONICAL, (1884, 924, 2559, 1594), ALIAS_X, ALIAS_Y,
              (1878, 919, 2555, 1595), CANONICAL]
    _, selected, sources = replay(bounds, (100, 101, 102, 103, 104, 105))

    assert selected == [CANONICAL] * 6
    assert sources == ["grid"] * 6


def test_stable_ui_movement_requires_three_fresh_frames_and_accepts_the_median():
    _, selected, sources = replay(
        [CANONICAL, MOVED, (1803, 872, 2479, 1548), (1799, 870, 2475, 1546)],
        (100, 101, 102, 103),
    )

    assert selected == [CANONICAL, CANONICAL, CANONICAL, MOVED]
    assert sources == ["grid"] * 4


def test_multiple_readers_of_one_ndarray_cannot_confirm_a_grid_jump():
    moved = frame()
    _, selected, _ = replay(
        [CANONICAL, MOVED, MOVED, MOVED, MOVED, MOVED],
        (100, 101, 101.1, 101.2, 102, 103),
        [frame(), moved, moved, moved, moved.copy(), moved.copy()],
    )

    assert selected == [CANONICAL] * 5 + [MOVED]


def test_weak_grid_expires_at_four_seconds_without_the_old_two_second_gate():
    vision, selected, sources = replay([CANONICAL, None, None], (100, 102.2, 104.01))

    assert sources == ["grid", "grid", "ratio_fallback"]
    assert selected[-1] == MINIMAP_REGION.pixels(2560, 1600)
    assert vision._minimap_grid_cache is None


def test_unconfirmed_alias_candidates_do_not_renew_the_old_grid_lifetime():
    vision, _, sources = replay([CANONICAL, ALIAS_X, ALIAS_Y, ALIAS_X], (100, 101, 103, 104.1))

    assert sources == ["grid", "grid", "grid", "ratio_fallback"]
    assert vision._minimap_grid_cache is None


def test_missing_grid_interrupts_moved_ui_confirmation():
    _, selected, sources = replay([CANONICAL, MOVED, None, MOVED, MOVED, MOVED],
                                   (100, 101, 102, 103, 104, 105))

    assert selected == [CANONICAL] * 5 + [MOVED]
    assert sources == ["grid"] * 6


def test_long_capture_gap_cannot_join_old_and_new_jump_votes():
    vision, _, sources = replay([CANONICAL, MOVED, MOVED], (100, 101, 106))

    assert sources == ["grid", "grid", "ratio_fallback"]
    assert len(vision._minimap_grid_pending) == 1


def test_new_framebuffer_dimensions_accept_the_new_grid_immediately():
    smaller = (940, 452, 1276, 796)
    vision, selected, sources = replay([CANONICAL, smaller], (100, 101),
                                        [frame(), frame((800, 1280))])

    assert selected == [CANONICAL, smaller]
    assert sources == ["grid", "grid"]
    assert vision._minimap_grid_cache[0] == (1280, 800)
    assert vision._minimap_grid_pending == []


@pytest.mark.parametrize("size,offset_x,offset_y", [(260, 120, 100), (310, 110, 50)])
def test_real_grid_detector_adapts_to_stable_movement_and_ui_resize(size, offset_x, offset_y):
    original, old_bounds = moved_grid(1280, 720, 260, 70, 65)
    shifted, expected = moved_grid(1280, 720, size, offset_x, offset_y)
    vision = Vision(screen_capture=object())
    selected = []
    for index, image in enumerate([original, shifted, shifted.copy(), shifted.copy()]):
        with patch("core.vision.time.monotonic", return_value=100 + index):
            assert vision.find_minimap(image) is not None
        selected.append(vision.minimap_bounds)

    assert selected[0] == pytest.approx(old_bounds, abs=4)
    assert selected[:3] == [selected[0]] * 3
    assert selected[-1] == pytest.approx(expected, abs=4)


def test_real_live_frames_keep_map_geometry_through_jitter_and_a_cell_alias():
    root = Path("E:/aimemo/docs/screenshots/wows_driving_trials/run_20261005_141526_1791180926725332600")
    paths = [root / f"full_{tick:04d}.png" for tick in range(0, 101, 20)]
    if not all(path.exists() for path in paths):
        pytest.skip("Optional saved live minimap trial unavailable")
    vision = Vision(screen_capture=object())
    selected = []
    for index, path in enumerate(paths):
        image = cv2.imread(str(path))
        assert image is not None
        with patch("core.vision.time.monotonic", return_value=100 + index):
            assert vision.find_minimap(image) is not None
        selected.append(vision.minimap_bounds)
        assert vision.minimap_bounds_source == "grid"

    assert selected[0] == pytest.approx(CANONICAL, abs=3)
    assert selected == [selected[0]] * len(selected)
