"""The selected torpedo launcher must actually point along its prediction."""

from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from core.ocr import OcrToken
from core.torpedo_hud import (
    TorpedoHud, _SlotEvidence, _cooldown_seconds, detect_torpedo_weapon,
    inspect_torpedo_hud,
)


def cone_frame(width=1000, height=600, ui_scale=.7, *, white_offset=0,
               broad=(20, 205, 20), small=True, white=True, overlap=False):
    image = np.zeros((height, width, 3), np.uint8)
    for x in (-280, -180, -80, 80, 180, 280):
        cx, cy = width / 2 + x * ui_scale, height - 17 * ui_scale
        half = 11.5 * ui_scale
        cv2.rectangle(image, (round(cx - half), round(cy - half)),
                      (round(cx + half), round(cy + half)), (225, 225, 225), 1)
    apex = (round(width / 2), round(height * .87))
    top_y = round(height * .35)
    if broad is not None:
        # A perspective arc has measurable curvature; a health bar does not.
        cv2.ellipse(image, (round(width / 2), round(height * .9)),
                    (round(width * .30), round(height * .18)), 0, 210, 330,
                    broad, max(2, round(4 * ui_scale)))
    half = height * (.04 if overlap else .09)
    if small:
        if overlap:
            cv2.fillConvexPoly(image, np.array([
                [round(width / 2 - half), top_y], [round(width / 2 + half), top_y], apex]), (20, 180, 20))
        for sign in (-1, 1):
            cv2.line(image, (round(width / 2 + sign * half), top_y), apex,
                      (20, 220, 20), max(2, round(3 * ui_scale)))
    cv2.circle(image, (round(width / 2), round(height * .5)),
               max(7, round(18 * ui_scale)), (20, 220, 20), max(1, round(2 * ui_scale)))
    if white:
        for sign in (-1, 1):
            cv2.line(image, (round(width / 2 + white_offset + sign * height * .04), top_y), apex,
                      (235, 235, 235), max(1, round(2 * ui_scale)))
    return image, (round(width / 2 + white_offset), round(height * .5))


def inspect_selected(image, target, before=None, **kwargs):
    with patch("core.torpedo_hud._weapon_slot_evidence", return_value=_SlotEvidence(True, True)):
        return inspect_torpedo_hud(image, image if before is None else before, target, **kwargs)


@pytest.mark.parametrize("width,height,ui_scale", [
    (1280, 720, .8), (1920, 1080, 1.25), (3440, 1440, .75), (3840, 2160, .6),
])
def test_independent_ui_size_and_resolution_require_actual_sector_alignment(width, height, ui_scale):
    image, target = cone_frame(width, height, ui_scale)
    result = inspect_selected(image, target)
    assert result.loaded_arc and result.ready_to_align and result.safe_to_fire
    assert result.aim_point[1] == result.lead_point[1]
    assert result.alignment_error_px <= result.alignment_tolerance_px
    assert result.alignment_error_degrees <= 2


def test_unaligned_white_prediction_is_ready_for_mouse_movement_but_cannot_fire():
    image, target = cone_frame(white_offset=55)
    result = inspect_selected(image, target)
    assert result.loaded_arc and result.ready_to_align
    assert not result.safe_to_fire
    assert result.lead_point[0] > result.aim_point[0]
    assert result.alignment_error_px > result.alignment_tolerance_px


def test_before_already_selected_and_identical_pixels_still_allow_geometric_confirmation():
    image, target = cone_frame()
    result = inspect_selected(image, target, before=image.copy())
    assert result.safe_to_fire


def test_white_edges_can_cover_green_edges_when_highlighted_green_interior_remains():
    image, target = cone_frame(overlap=True)
    result = inspect_selected(image, target)
    assert result.safe_to_fire


@pytest.mark.parametrize("broad", [None, (20, 205, 205)])
def test_green_small_fan_and_reticle_do_not_prove_large_arc_loaded(broad):
    image, target = cone_frame(broad=broad)
    result = inspect_selected(image, target)
    assert result.aim_point is not None
    assert result.lead_point is not None
    assert not result.loaded_arc
    assert not result.ready_to_align and not result.safe_to_fire


def test_large_green_arc_with_white_prediction_and_reticle_needs_small_current_fan():
    image, target = cone_frame(small=False)
    result = inspect_selected(image, target)
    assert result.loaded_arc
    assert result.reticle_point is not None
    assert result.aim_point is None and not result.safe_to_fire


def test_white_mouse_pointer_and_filled_explosion_never_become_prediction_sector():
    image, target = cone_frame(white=False)
    cv2.fillConvexPoly(image, np.array([[500, 290], [517, 311], [505, 309], [499, 326]]),
                       (235, 235, 235))
    cv2.circle(image, (650, 340), 45, (240, 240, 240), -1)
    result = inspect_selected(image, target)
    assert result.aim_point is not None
    assert result.lead_point is None and not result.safe_to_fire


def test_current_small_fan_axis_survives_missing_white_line_for_one_target_lock_attempt():
    image, target = cone_frame(white=False)
    result = inspect_selected(image, target)
    assert result.aim_point is not None
    assert result.lead_point is None and not result.safe_to_fire


def test_yellow_arc_plus_green_ship_health_bar_above_crosshair_is_not_loaded():
    image, target = cone_frame(broad=(20, 205, 205))
    cv2.line(image, (400, 150), (600, 150), (20, 220, 20), 5)
    result = inspect_selected(image, target)
    assert result.aim_point is not None and result.lead_point is not None
    assert not result.loaded_arc and not result.safe_to_fire


@pytest.mark.parametrize("endpoints", [((300, 430), (700, 430)), ((300, 460), (700, 390))])
def test_long_straight_green_lines_below_crosshair_are_not_loaded_arcs(endpoints):
    image, target = cone_frame(broad=None)
    cv2.line(image, *endpoints, (20, 220, 20), 3)
    result = inspect_selected(image, target)
    assert not result.loaded_arc and not result.safe_to_fire


@pytest.mark.parametrize("weapon,selected", [(None, True), (False, True), (True, False)])
def test_geometrically_aligned_cones_cannot_override_unknown_weapon_or_unselected_slot(weapon, selected):
    image, target = cone_frame()
    with patch("core.torpedo_hud._weapon_slot_evidence", return_value=_SlotEvidence(weapon, selected)):
        result = inspect_torpedo_hud(image, image, target)
    assert not result.ready_to_align and not result.safe_to_fire


def test_restore_inspection_keeps_weapon_and_selection_without_target_or_before():
    image, _ = cone_frame()
    with patch("core.torpedo_hud._weapon_slot_evidence", return_value=_SlotEvidence(True, False)):
        result = inspect_torpedo_hud(image, None, None)
    assert result.weapon_is_torpedo is True and result.selected is False
    assert not result.safe_to_fire


def test_positive_countdown_blocks_even_aligned_loaded_torpedoes():
    image, target = cone_frame()
    with patch("core.torpedo_hud._cooldown_seconds", return_value=30):
        result = inspect_selected(image, target)
    assert result.loaded_arc and result.aligned
    assert not result.safe_to_fire


@pytest.mark.parametrize("field,value", [
    ("alignment_error_px", -1), ("alignment_error_px", float("nan")),
    ("alignment_error_px", float("inf")), ("alignment_tolerance_px", -1),
    ("alignment_tolerance_px", float("inf")), ("alignment_error_degrees", -1),
    ("alignment_error_degrees", float("nan")),
])
def test_alignment_requires_finite_nonnegative_measurements(field, value):
    result = TorpedoHud(None, True, (500, 300), weapon_is_torpedo=True,
                       selected=True, aim_point=(500, 300), alignment_error_px=0,
                       alignment_tolerance_px=5, alignment_error_degrees=0)
    assert result.safe_to_fire
    assert not replace(result, **{field: value}).safe_to_fire


def test_legacy_three_field_constructor_remains_valid_but_has_no_alignment_authority():
    assert not TorpedoHud(None, True, (500, 300)).safe_to_fire


def test_digit_three_alone_is_unknown_weapon_and_no_backend_fails_closed():
    image, _ = cone_frame()

    class Backend:
        def recognize(self, crop):
            return [OcrToken("3", .99, ((10, 10), (18, 10), (18, 22), (10, 22)))]

    assert detect_torpedo_weapon(image, Backend()) is None
    assert detect_torpedo_weapon(image, None) is None


def test_cooldown_follows_verified_slot_geometry_and_cannot_read_an_unrelated_old_roi():
    image = np.zeros((1080, 1920, 3), np.uint8)
    slot = _SlotEvidence(True, True, (600, 930, 650, 980), .8)
    image[915:919, 620:632] = (112, 222, 244)

    class Backend:
        def recognize(self, crop):
            assert np.any(np.all(crop == (112, 222, 244), axis=2))
            return [OcrToken("42", .99, ((15, 10), (30, 10), (30, 22), (15, 22)))]

    assert _cooldown_seconds(image, Backend(), slot) == 42
    assert _cooldown_seconds(image, Backend(), _SlotEvidence(None, False)) is None


def test_real_selected_slot_identity_without_white_prediction_remains_unsafe():
    # The actual image match is independent of OCR's mocked key-label output.
    image = cv2.imread(str(Path(__file__).parent / "fixtures" / "live_battle.png"))

    class KeyBackend:
        def recognize(self, crop):
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, np.array([35, 100, 80]), np.array([95, 255, 255]))
            # Remove the thin selected-slot outline adjoining the filled [3].
            mask = cv2.erode(mask, np.ones((5, 5), np.uint8))
            _, _, stats, _ = cv2.connectedComponentsWithStats(mask)
            tokens = []
            for x, y, w, h, area in stats[1:]:
                if 12 <= w <= 28 and 12 <= h <= 28 and area >= 100:
                    tokens.append(OcrToken("3", .99,
                                  ((x + 4, y + 3), (x + w - 3, y + 3),
                                   (x + w - 3, y + h - 3), (x + 4, y + h - 3))))
            return tokens

    backend = KeyBackend()
    assert detect_torpedo_weapon(image, backend) is True
    result = inspect_torpedo_hud(image, image, (1280, 800), backend)
    assert result.weapon_is_torpedo is True and result.selected
    assert result.loaded_arc
    assert result.aim_point is not None
    assert result.lead_point is None and not result.safe_to_fire


@pytest.mark.parametrize("ui_scale", [.75, 1.25])
def test_actual_slot3_template_moved_with_ui_keeps_its_own_countdown(ui_scale):
    source = cv2.imread(str(Path(__file__).parent / "fixtures" / "live_battle.png"))
    image = source.copy()
    image[1390:, 820:1780] = (70, 75, 80)
    row = cv2.resize(source[1440:1600, 850:1700], None, fx=ui_scale, fy=ui_scale,
                     interpolation=cv2.INTER_CUBIC)
    left, top = round(1280 - row.shape[1] / 2 + 30), 1570 - row.shape[0]
    image[top:top + row.shape[0], left:left + row.shape[1]] = row
    body_x = left + round((1181 - 850 + 33) * ui_scale)
    body_y = top + round((1493 - 1440) * ui_scale)
    # Distinct evidence above this moved tube, rather than the old fixed ROI.
    image[body_y - 15:body_y - 11, body_x - 4:body_x + 4] = (112, 222, 244)

    class Backend:
        def recognize(self, crop):
            marker = np.all(crop == (112, 222, 244), axis=2)
            yy, xx = np.nonzero(marker)
            if len(xx):
                x1, x2, y1, y2 = int(xx.min()), int(xx.max() + 1), int(yy.min()), int(yy.max() + 1)
                return [OcrToken("42", .99, ((x1, y1), (x2, y1), (x2, y2), (x1, y2)))]
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            green = cv2.inRange(hsv, np.array([35, 100, 80]), np.array([95, 255, 255]))
            green = cv2.erode(green, np.ones((5, 5), np.uint8))
            _, _, stats, _ = cv2.connectedComponentsWithStats(green)
            return [OcrToken("3", .99, ((x + 2, y + 2), (x + w - 2, y + 2),
                                       (x + w - 2, y + h - 2), (x + 2, y + h - 2)))
                    for x, y, w, h, area in stats[1:]
                    if 10 <= w <= 32 and 10 <= h <= 32 and area >= 60]

    result = inspect_torpedo_hud(image, image, (1280, 800), Backend())
    assert result.weapon_is_torpedo is True and result.selected
    assert result.loaded_arc and result.aim_point is not None
    assert result.cooldown_seconds == 42
    assert not result.ready_to_align and not result.safe_to_fire
