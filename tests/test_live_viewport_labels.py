"""Target labels need positive ship/HP evidence, including live sunset frames."""

from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from core.ocr import OcrToken, TargetDistanceReader
from core.vision import Vision


FIXTURES = Path(__file__).parent / "fixtures"
TRIALS = Path("E:/aimemo/docs/screenshots/wows_driving_trials")
HISTORICAL = TRIALS / "run_20261004_195900_1791115140712338300" / "full_0060.png"
CURRENT = TRIALS / "run_20261005_134040_1791178840243077000"


def load(path):
    if not path.exists():
        pytest.skip("Optional retained real battle diagnostic is unavailable")
    image = cv2.imread(str(path))
    assert image is not None
    return image


def vision():
    return Vision(screen_capture=object())


def draw_label(image, center, scale=1, *, health=True, marker=True, digits=True):
    x, y = center
    red = (42, 80, 254)
    if marker:
        polygon = np.array([(x - 10 * scale, y + 5 * scale),
                            (x + 10 * scale, y + 5 * scale),
                            (x + 10 * scale, y - 5 * scale),
                            (x, y - 5 * scale)], np.int32)
        cv2.fillConvexPoly(image, polygon, red)
    bx, by = round(x - 56 * scale), round(y + 35 * scale)
    if health:
        cv2.rectangle(image, (bx, by), (round(bx + 112 * scale), round(by + 14 * scale)), red, -1)
    if digits:
        for offset in range(8, 104, 12):
            cv2.rectangle(image, (round(bx + offset * scale), round(by + 3 * scale)),
                          (round(bx + (offset + 4) * scale), round(by + 11 * scale)), (235, 235, 235), -1)


@pytest.mark.parametrize("width,height", [(1280, 720), (1920, 1080), (3440, 1440), (3840, 2160)])
def test_real_target_survives_global_resolution_and_aspect_changes(width, height):
    source = load(FIXTURES / "live_battle.png")
    image = cv2.resize(source, (width, height), interpolation=cv2.INTER_AREA)
    points = vision().find_enemies_in_viewport(image)
    assert len(points) == 1
    assert points[0] == pytest.approx((1080 * width / 2560, 604 * height / 1600), abs=4)


@pytest.mark.parametrize("scale,position", [(.6, (.45, .35)), (1.4, (.62, .43))])
def test_real_label_local_scale_and_position_are_independent_of_own_ship_hud(scale, position):
    source = load(FIXTURES / "live_battle.png")
    image = source.copy()
    # Move only this real label. Other HUD groups keep their original sizes.
    label = source[568:675, 1015:1145]
    image[568:675, 1015:1145] = (110, 95, 95)
    resized = cv2.resize(label, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    x, y = round(image.shape[1] * position[0]), round(image.shape[0] * position[1])
    image[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    points = vision().find_enemies_in_viewport(image)
    assert len(points) == 1
    assert points[0] == pytest.approx((x + 65 * scale, y + 36 * scale), abs=4)


@pytest.mark.parametrize("health,marker,digits", [(True, False, True), (True, True, False), (False, True, True)])
def test_an_unpaired_bar_marker_or_red_strokes_cannot_establish_a_target(health, marker, digits):
    image = np.full((1080, 1920, 3), (100, 105, 115), np.uint8)
    draw_label(image, (900, 450), health=health, marker=marker, digits=digits)
    assert vision().find_enemies_in_viewport(image) == []


def test_red_glyph_row_and_sunset_coloured_fragments_are_rejected():
    image = np.full((1080, 1920, 3), (110, 160, 220), np.uint8)
    for x in range(230, 1700, 75):
        cv2.rectangle(image, (x, 430), (x + 7, 438), (0, 50, 235), -1)
        cv2.rectangle(image, (x, 580), (x + 55, 590), (25, 130, 250), -1)
    assert vision().find_enemies_in_viewport(image) == []


def test_detected_moved_minimap_excludes_even_a_complete_label_lookalike():
    image = np.full((1080, 1920, 3), (100, 105, 115), np.uint8)
    draw_label(image, (1300, 550))
    layout = type("Layout", (), {"ui_scale": 1, "rect": lambda _, name: (1100, 470, 1500, 850)})()
    with patch("core.vision.detect_hud_layout", return_value=layout):
        assert vision().find_enemies_in_viewport(image) == []


def test_over_eight_positive_targets_remain_visible_to_the_bot_safety_guard():
    image = np.full((1080, 1920, 3), (100, 105, 115), np.uint8)
    for x in range(220, 1600, 150):
        draw_label(image, (x, 350))
    assert len(vision().find_enemies_in_viewport(image)) == 10


def test_live_damaged_target_anchor_keeps_health_and_distance_inside_ocr_crop():
    image = load(FIXTURES / "live_battle.png")
    point, = vision().find_enemies_in_viewport(image)
    roi = TargetDistanceReader.target_roi(image, point)
    # Measured source text boxes: HP y638..653 and distance y659..676.
    assert roi.x <= 1024 and roi.x + roi.width >= 1140
    assert roi.y <= 638 and roi.y + roi.height >= 676

    class Backend:
        def recognize(self, crop):
            return [OcrToken("9598 / 60950", .99), OcrToken("23.7 公里", .99)]

    result = TargetDistanceReader(backend=Backend()).read(image, point, "live-target")
    assert result.accepted and result.value_km == 23.7


def test_retained_sunset_frame_returns_five_real_label_anchors_without_horizon_noise():
    points = vision().find_enemies_in_viewport(load(HISTORICAL))
    assert len(points) == 5
    expected = [(143, 569), (568, 553), (608, 553), (1333, 555), (1595, 585)]
    for actual, target in zip(points, expected):
        assert actual == pytest.approx(target, abs=4)


@pytest.mark.parametrize("name,expected", [
    ("full_0000.png", []), ("full_0020.png", [(935, 606)]),
    ("full_0040.png", [(485, 610), (695, 604), (2265, 600)]),
    ("full_0060.png", [(2410, 606)]), ("full_0080.png", []), ("full_0100.png", []),
])
def test_retained_current_sunset_frames_do_not_turn_sun_water_or_capture_letters_into_targets(name, expected):
    points = vision().find_enemies_in_viewport(load(CURRENT / name))
    assert len(points) == len(expected)
    for actual, target in zip(points, expected):
        assert actual == pytest.approx(target, abs=4)
