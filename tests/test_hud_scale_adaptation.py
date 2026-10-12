"""Resolution and HUD size/position vary independently in these frames."""

from pathlib import Path
from unittest.mock import patch
import json

import cv2
import numpy as np
import pytest

from core.hud_layout import detect_hud_layout
from core.ocr import OcrToken
from core.vision import Vision


def hud_frame(width, height, scale, left_shift=(45, -35), center_shift=(70, -65)):
    image = np.full((height, width, 3), (65, 50, 35), dtype=np.uint8)
    radius = 125 * scale
    cx, cy = 150 * scale + left_shift[0], height - 148 * scale + left_shift[1]
    cv2.circle(image, (round(cx), round(cy)), round(radius), (210, 210, 210),
               max(1, round(2 * scale)))
    cv2.circle(image, (round(cx), round(cy)), round(radius * .79), (185, 185, 185), -1)
    # The own-ship status silhouette is above the dial. A circle alone is weak.
    points = np.array([(cx - .86 * radius, cy - 1.4 * radius),
                       (cx + .82 * radius, cy - 1.4 * radius),
                       (cx + .68 * radius, cy - 1.57 * radius),
                       (cx - .15 * radius, cy - 1.57 * radius)], np.int32)
    cv2.fillConvexPoly(image, points, (40, 170, 90))
    center_x, center_y = width / 2 + center_shift[0], height - 17 * scale + center_shift[1]
    for offset in (-280, -180, -80, 80, 180, 280):
        x = center_x + offset * scale
        half = 11.5 * scale
        cv2.rectangle(image, (round(x - half), round(center_y - half)),
                      (round(x + half), round(center_y + half)), (230, 230, 230),
                      max(1, round(scale)))
    return image, (cx, cy, radius), (center_x, center_y)


CASES = [(1280, 720, .8), (1920, 1080, 1.25), (3440, 1440, .75), (3840, 2160, .6)]


@pytest.mark.parametrize("width,height,scale", CASES)
def test_independent_dial_and_key_row_scale_and_translation(width, height, scale):
    image, (cx, cy, radius), _ = hud_frame(width, height, scale)
    layout = detect_hud_layout(image)
    assert layout.source("health_text") == "anchor"
    assert layout.source("consumables") == "anchor"
    assert layout.scale("health_text") == pytest.approx(scale, rel=.10)
    assert layout.scale("consumables") == pytest.approx(scale, rel=.12)
    x1, y1, x2, y2 = layout.rect("fire_dial")
    assert (x1 + x2) / 2 == pytest.approx(cx, abs=radius * .1)
    assert (y1 + y2) / 2 == pytest.approx(cy, abs=radius * .1)
    assert Vision.hud_layout(image) is layout


@pytest.mark.parametrize("width,height,scale", CASES)
def test_health_and_speed_read_the_moved_own_ship_hud(width, height, scale):
    image, (cx, cy, radius), _ = hud_frame(width, height, scale)
    hp_x, hp_y = round(cx + radius * .5), round(cy - radius * 1.14)
    speed_x, speed_y = round(cx + radius * 1.6), round(cy)
    image[hp_y:hp_y + 4, hp_x:hp_x + 12] = (235, 0, 235)
    image[speed_y:speed_y + 4, speed_x:speed_x + 12] = (240, 230, 0)

    class Backend:
        def recognize(self, crop):
            if np.any(np.all(crop == (235, 0, 235), axis=2)):
                return [OcrToken("40 000 / 80 000", .99)]
            if np.any(np.all(crop == (240, 230, 0), axis=2)):
                return [OcrToken("23.4 kts", .99)]
            return []

    assert Vision.read_health_fraction(image, Backend()) == .5
    assert Vision.read_speed_knots(image, Backend()) == 23.4


def droplet(image, point, scale):
    x, y = point
    cv2.rectangle(image, (round(x - 4 * scale), round(y - 16 * scale)),
                  (round(x + 4 * scale), round(y - 4 * scale)), (0, 90, 160), -1)
    cv2.circle(image, (round(x), round(y + 2 * scale)), max(2, round(9 * scale)),
               (0, 128, 255), -1)


@pytest.mark.parametrize("width,height,scale", CASES)
def test_fire_keeps_two_independent_anchors_with_scaled_hud(width, height, scale):
    image, (cx, cy, _), _ = hud_frame(width, height, scale)
    droplet(image, (cx, cy), scale)
    # A viewport ship marker has the same UI glyph size, but is outside the dial.
    droplet(image, (width * .45, height * .7), scale)
    assert Vision().fire_anchor_bits(image) == (True, True)
    assert Vision().is_on_fire(image)


@pytest.mark.parametrize("width,height,scale", CASES)
def test_fire_card_and_rudder_follow_moved_bottom_group(width, height, scale):
    image, _, (cx, cy) = hud_frame(width, height, scale)
    # Reference positions relative to the key row, with independent translation.
    fire_x, fire_y = round(cx - 85 * scale), round(cy - 285 * scale)
    cv2.rectangle(image, (fire_x, fire_y),
                  (fire_x + round(35 * scale), fire_y + round(35 * scale)), (0, 55, 190), -1)
    q_x, q_y = round(cx - 165 * scale), round(cy - 320 * scale)
    cv2.rectangle(image, (q_x, q_y),
                  (q_x + round(18 * scale), q_y + round(18 * scale)), (40, 210, 40), -1)
    vision = Vision()
    assert vision.fire_status_icon_visible(image)
    assert vision.detect_rudder_indicator(image) == "Q"


@pytest.mark.parametrize("width,height,scale", CASES)
def test_flooding_requires_both_moved_hud_groups_and_countdown(width, height, scale):
    image, (cx, cy, radius), (center_x, center_y) = hud_frame(width, height, scale)
    lx, ly = round(cx + radius * .3), round(cy - radius * .6)
    bx, by = round(center_x - 10 * scale), round(center_y - 200 * scale)
    side = round(24 * scale)
    cv2.rectangle(image, (lx, ly), (lx + round(18 * scale), ly + round(18 * scale)), (255, 120, 0), -1)
    cv2.rectangle(image, (bx, by), (bx + side, by + side), (255, 120, 0), -1)
    for dx in (side + round(5 * scale), side + round(15 * scale)):
        cv2.rectangle(image, (bx + dx, by + round(3 * scale)),
                      (bx + dx + round(4 * scale), by + round(13 * scale)), (220, 220, 220), -1)
    vision = Vision()
    assert vision.is_flooding(image)
    assert vision.flooding_status_icon_visible(image)
    missing_left = image.copy()
    missing_left[ly:ly + round(19 * scale) + 2, lx:lx + round(19 * scale) + 2] = (65, 50, 35)
    assert not vision.is_flooding(missing_left)
    assert not vision.flooding_status_icon_visible(missing_left)


def test_sea_circles_other_ships_and_minimap_flames_do_not_become_own_ship_hud():
    image = np.full((1080, 1920, 3), (155, 105, 65), np.uint8)
    cv2.circle(image, (145, 940), 90, (210, 210, 210), 2)
    droplet(image, (145, 940), 1)
    droplet(image, (1500, 900), 1)
    assert detect_hud_layout(image).source("fire_dial") == "fallback"
    assert not Vision().is_on_fire(image)
    assert not Vision().is_flooding(image)


def test_one_frame_layout_is_computed_once_and_no_previous_frame_is_reused():
    image, _, _ = hud_frame(1920, 1080, .9)
    import core.hud_layout as module
    with patch.object(module, "_left_anchor", wraps=module._left_anchor) as left:
        first = detect_hud_layout(image)
        assert detect_hud_layout(image) is first
        Vision.read_health_fraction(image, type("B", (), {"recognize": lambda *_: []})())
        Vision().is_on_fire(image)
        Vision().is_flooding(image)
        assert left.call_count == 1
        assert detect_hud_layout(image.copy()) is not first
        assert left.call_count == 2
    json.dumps(first.diagnostics())


def test_inferred_clock_stays_right_of_capture_timers_and_rejects_conflicts():
    image, _, _ = hud_frame(1920, 1080, 1.25)
    layout = detect_hud_layout(image)
    assert layout.source("clock") == "inferred"
    assert layout.rect("clock")[0] >= 1920 * .67
    box = ((20, 5), (90, 5), (90, 25), (20, 25))

    class Backend:
        def __init__(self, tokens):
            self.tokens = tokens

        def recognize(self, crop):
            return self.tokens

    assert Vision.read_battle_clock_seconds(image, Backend([OcrToken("19:58", .99, box)])) == 1198
    assert Vision.read_battle_clock_seconds(image, Backend([
        OcrToken("19:58", .99, box), OcrToken("1:15", .99, box)])) is None
    assert Vision.read_battle_clock_seconds(image, Backend([OcrToken("19:58", .40, box)])) is None


@pytest.mark.parametrize("name", ["fire_dual_anchor.jpg", "fire_live_siberia.jpg", "fire_live_2k.jpg"])
def test_real_fire_fixtures_measure_dial_independent_of_frame_aspect(name):
    image = cv2.imread(str(Path(__file__).parent / "fixtures" / name))
    layout = detect_hud_layout(image)
    assert layout.source("fire_dial") == "anchor"
    assert Vision().fire_anchor_bits(image) == (True, True)


@pytest.mark.parametrize("width,height,scale", CASES)
def test_viewport_label_strokes_scale_with_hud_not_framebuffer(width, height, scale):
    image, _, _ = hud_frame(width, height, scale)
    x, y = round(width * .5), round(height * .4)
    # A ship-type glyph sits above a rectangular HP label with white digits.
    # Three unanchored red strokes can equally be sunset/water fragments.
    points = np.array([(x - 10 * scale, y), (x + 10 * scale, y),
                       (x + 10 * scale, y - 10 * scale), (x, y - 10 * scale)], np.int32)
    cv2.fillConvexPoly(image, points, (0, 50, 235))
    bar_x, bar_y = round(x - 56 * scale), round(y + 30 * scale)
    cv2.rectangle(image, (bar_x, bar_y),
                  (round(bar_x + 112 * scale), round(bar_y + 14 * scale)), (0, 50, 235), -1)
    for offset in range(8, 104, 12):
        cv2.rectangle(image, (round(bar_x + offset * scale), round(bar_y + 3 * scale)),
                      (round(bar_x + (offset + 4) * scale), round(bar_y + 11 * scale)),
                      (235, 235, 235), -1)
    enemies = Vision().find_enemies_in_viewport(image)
    assert len(enemies) == 1
    assert enemies[0][0] == pytest.approx(x, abs=12 * scale)


def test_loading_action_returns_actual_shifted_rectangle_and_rejects_thin_hud_bar():
    image = np.full((1000, 1600, 3), 40, np.uint8)
    cv2.rectangle(image, (600, 800), (800, 850), (55, 110, 55), -1)
    vision = Vision()
    assert vision.loading_start_action_box(image) == (600, 800, 801, 851)
    assert vision._has_loading_start_action(image)
    thin = np.full_like(image, 40)
    cv2.rectangle(thin, (600, 800), (800, 804), (55, 110, 55), -1)
    assert vision.loading_start_action_box(thin) is None
    assert not vision._has_loading_start_action(thin)


def test_shifted_result_buttons_need_both_colours_on_the_same_row():
    image = np.full((1000, 1600, 3), 40, np.uint8)
    cv2.rectangle(image, (400, 800), (580, 840), (150, 150, 50), -1)
    cv2.rectangle(image, (1000, 800), (1160, 840), (0, 120, 230), -1)
    assert Vision().in_results(image)
    missing = image.copy()
    missing[800:841, 1000:1161] = 40
    assert not Vision().in_results(missing)
    wrong_row = missing.copy()
    cv2.rectangle(wrong_row, (1000, 700), (1160, 740), (0, 120, 230), -1)
    assert not Vision().in_results(wrong_row)


def test_shifted_modal_action_colours_keep_page_darkness_and_rectangle_guards():
    image = np.full((1000, 1600, 3), 70, np.uint8)
    cv2.rectangle(image, (520, 340), (740, 390), (55, 110, 55), -1)
    assert Vision().in_escape_menu(image)
    exit_page = np.full_like(image, 70)
    cv2.rectangle(exit_page, (650, 600), (830, 640), (180, 90, 35), -1)
    assert Vision().in_exit_confirmation(exit_page)
    round_icon = np.full_like(image, 70)
    cv2.circle(round_icon, (600, 380), 30, (55, 110, 55), -1)
    assert not Vision().in_escape_menu(round_icon)


@pytest.mark.parametrize("factor,offset_x,offset_y", [(.75, 40, 30), (1.25, 35, 15)])
def test_real_status_panel_local_resize_keeps_the_complete_speed_glyph(factor, offset_x, offset_y):
    source = cv2.imread(str(Path(__file__).parent / "fixtures" / "live_battle.png"))
    height, width = source.shape[:2]
    old_top = round(height * .68)
    panel = source[old_top:, :round(width * .255)]
    image = source.copy()
    image[old_top:, :round(width * .32)] = (70, 75, 80)
    resized = cv2.resize(panel, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)
    new_top = height - offset_y - resized.shape[0]
    image[new_top:new_top + resized.shape[0], offset_x:offset_x + resized.shape[1]] = resized
    layout = detect_hud_layout(image)
    assert layout.source("speed") == "anchor"
    # Actual RapidOCR's source glyph quad is x371..441,y1401..1422. The test
    # checks the complete transformed glyph, including the speed unit suffix.
    expected = (371 * factor + offset_x, (1401 - old_top) * factor + new_top,
                441 * factor + offset_x, (1422 - old_top) * factor + new_top)
    x1, y1, x2, y2 = layout.rect("speed")
    assert x1 <= expected[0] and y1 <= expected[1]
    assert x2 >= expected[2] and y2 >= expected[3]
    assert layout.scale("speed") == pytest.approx(factor * 1.057, rel=.15)
