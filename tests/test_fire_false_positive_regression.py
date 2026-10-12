"""Sunset-lit Rupprecht decks and turret bearings are not fire evidence."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from core.hud_layout import detect_hud_layout
from core.vision import Vision
from test_hud_scale_adaptation import hud_frame


TRIAL = Path("E:/aimemo/docs/screenshots/wows_driving_trials/run_20261005_134040_1791178840243077000")


@pytest.mark.parametrize("name", [f"full_{tick:04d}.png" for tick in range(0, 101, 20)])
def test_full_health_sunset_trial_without_fire_never_reports_fire(name):
    path = TRIAL / name
    if not path.exists():
        pytest.skip("Optional live Rupprecht trial image unavailable")
    image = cv2.imread(str(path))
    assert image is not None
    vision = Vision()

    assert not vision.fire_status_icon_visible(image)
    assert not vision.is_on_fire(image)


@pytest.mark.parametrize("name", [
    "fire_dual_anchor.jpg", "fire_live_siberia.jpg", "fire_live_1080p.jpg", "fire_live_2k.jpg",
])
def test_real_burning_ship_fixtures_keep_both_flame_anchors(name):
    image = cv2.imread(str(Path(__file__).parent / "fixtures" / name))
    assert image is not None
    vision = Vision()

    assert vision.fire_anchor_bits(image) == (True, True)
    assert vision.is_on_fire(image)


@pytest.mark.parametrize("width,height,scale", [
    (1280, 720, .8), (1920, 1080, 1.25), (3440, 1440, .75), (3840, 2160, .6),
])
def test_complete_damage_card_keeps_independent_ui_scale_and_position(width, height, scale):
    image, _, (cx, cy) = hud_frame(width, height, scale, center_shift=(45, -45))
    x, y = round(cx - 85 * scale), round(cy - 285 * scale)
    side = round(35 * scale)
    cv2.rectangle(image, (x, y), (x + side, y + side), (0, 55, 190), -1)
    # An internal dark glyph does not invalidate a complete red card outline.
    cv2.circle(image, (x + side // 2, y + side // 2), max(2, round(5 * scale)), (15, 15, 15), -1)

    assert Vision().fire_status_icon_visible(image)
    assert Vision().is_on_fire(image)


@pytest.mark.parametrize("kind", ["irregular_deck", "oversized_artwork", "clipped_blob"])
def test_red_artwork_in_measured_condition_strip_needs_a_complete_compact_card(kind):
    image, _, _ = hud_frame(1920, 1080, 1.0)
    layout = detect_hud_layout(image)
    x1, y1, x2, y2 = layout.rect("fire_status")
    if kind == "irregular_deck":
        # A 49x62 red deck component used to pass the fill>=.15 condition.
        points = np.array([(x1 + 45, y1 + 25), (x1 + 94, y1 + 87),
                           (x1 + 45, y1 + 87)], np.int32)
        cv2.fillConvexPoly(image, points, (0, 55, 190))
    elif kind == "oversized_artwork":
        cv2.rectangle(image, (x1 + 25, y1 + 15), (x1 + 105, y1 + 95), (0, 55, 190), -1)
    else:
        cv2.rectangle(image, (x1 - 10, y1 + 25), (x1 + 35, y1 + 65), (0, 55, 190), -1)

    assert not Vision().fire_status_icon_visible(image)


def test_low_contrast_orange_bearing_markers_cannot_corroborate_each_other():
    image, (cx, cy, _), _ = hud_frame(1920, 1080, 1.0)
    for x, y in ((round(cx), round(cy)), (864, 756)):
        cv2.rectangle(image, (x - 4, y - 16), (x + 4, y - 4), (0, 110, 220), -1)
        cv2.circle(image, (x, y + 2), 9, (0, 130, 255), -1)

    assert Vision().fire_anchor_bits(image) == (False, False)
    assert not Vision().is_on_fire(image)
