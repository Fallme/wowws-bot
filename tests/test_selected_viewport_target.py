"""A locked, damaged enemy keeps its target identity when its glyph turns peach."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from core.ocr import OcrToken, TargetDistanceReader
from core.vision import Vision


CAPTURE = Path("E:/aimemo/docs/screenshots/wows_selected_enemy_20261005.png")
DAYLIGHT_LOCKED = Path(
    "E:/aimemo/docs/screenshots/wows_torpedoes/"
    "torpedo_20261006_105044_1791255044703436500/frame_0007_result.png"
)


def source_frame():
    if not CAPTURE.is_file():
        pytest.skip("Optional retained selected-enemy capture is unavailable")
    image = cv2.imread(str(CAPTURE))
    assert image is not None
    return image


def target_points(image):
    return Vision(screen_capture=object()).find_enemies_in_viewport(image)


@pytest.mark.parametrize("scale", [.5, .75, 1, 1.25])
def test_actual_locked_low_hp_enemy_survives_resolution_changes(scale):
    image = cv2.resize(source_frame(), None, fx=scale, fy=scale,
                       interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
    point, = target_points(image)
    assert point == pytest.approx((1229 * scale, 724 * scale), abs=3)

    class Backend:
        def recognize(self, crop):
            return [OcrToken("14 715 / 69 000", .99), OcrToken("2.5 公里", .99)]

    reader = TargetDistanceReader(Backend())
    roi = reader.target_roi(image, point)
    # The paired anchor must retain the actual distance y781..797, even at
    # tiny UI size, instead of moving to the gun reticle/sea in front of it.
    assert roi.y <= round(781 * scale)
    assert roi.y + roi.height >= round(797 * scale)
    result = reader.read(image, point, "locked-enemy")
    assert result is not None and result.accepted and result.value_km == 2.5


@pytest.mark.parametrize("scale", [.5, .75, 1, 1.25])
def test_actual_daylight_locked_enemy_with_hue_below_ten_keeps_its_hp_pair(scale):
    if not DAYLIGHT_LOCKED.exists():
        pytest.skip("Retained native daylight lock evidence unavailable")
    image = cv2.imread(str(DAYLIGHT_LOCKED))
    image = cv2.resize(image, None, fx=scale, fy=scale,
                       interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)

    point, = target_points(image)

    assert point == pytest.approx((1097 * scale, 704 * scale), abs=4)


@pytest.mark.parametrize("missing", ["glyph", "hp", "digits"])
@pytest.mark.parametrize("scale", [.5, .75, 1, 1.25])
def test_actual_daylight_glyph_still_needs_its_independent_hp_and_digits(missing, scale):
    if not DAYLIGHT_LOCKED.exists():
        pytest.skip("Retained native daylight lock evidence unavailable")
    image = cv2.imread(str(DAYLIGHT_LOCKED))
    if missing == "glyph":
        image[690:719, 1077:1119] = (140, 130, 120)
    elif missing == "hp":
        image[738:757, 1037:1162] = (140, 130, 120)
    else:
        stripe = image[739:754, 1037:1162]
        hsv = cv2.cvtColor(stripe, cv2.COLOR_BGR2HSV)
        # Remove white cores and the antialias blends before resampling,
        # while preserving the independent saturated red HP fill.
        stripe[(hsv[:, :, 1] < 120) & (hsv[:, :, 2] > 175)] = (130, 130, 130)
    image = cv2.resize(image, None, fx=scale, fy=scale,
                       interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)

    assert target_points(image) == []


@pytest.mark.parametrize("scale,position", [
    (.5, (.40, .36)), (.75, (.60, .40)), (1, (.45, .50)), (1.25, (.60, .34)),
])
def test_actual_locked_label_can_move_and_scale_independently_of_own_ship_hud(scale, position):
    image = source_frame()
    label = image[665:815, 1150:1320].copy()
    image[665:815, 1150:1320] = (145, 130, 115)
    label = cv2.resize(label, None, fx=scale, fy=scale,
                       interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
    x, y = round(image.shape[1] * position[0]), round(image.shape[0] * position[1])
    image[y:y + label.shape[0], x:x + label.shape[1]] = label
    point, = target_points(image)
    assert point == pytest.approx((x + 79 * scale, y + 59 * scale), abs=3)


@pytest.mark.parametrize("missing", ["glyph", "hp", "digits"])
def test_actual_selected_reticle_without_complete_enemy_label_is_rejected(missing):
    image = source_frame()
    if missing == "glyph":
        image[710:737, 1204:1251] = (145, 130, 115)
    elif missing == "hp":
        image[756:780, 1165:1296] = (145, 130, 115)
    else:
        strip = image[759:779, 1168:1294]
        hsv = cv2.cvtColor(strip, cv2.COLOR_BGR2HSV)
        white = (hsv[:, :, 1] < 65) & (hsv[:, :, 2] > 175)
        strip[white] = (130, 130, 130)
    assert target_points(image) == []


@pytest.mark.parametrize("replacement", [(80, 220, 80), (189, 225, 255)])
def test_actual_friendly_or_peach_hp_fill_cannot_support_selected_enemy(replacement):
    image = source_frame()
    strip = image[759:779, 1168:1294]
    hsv = cv2.cvtColor(strip, cv2.COLOR_BGR2HSV)
    red = ((hsv[:, :, 0] <= 12) | (hsv[:, :, 0] >= 165)) & (hsv[:, :, 1] >= 145)
    strip[red] = replacement
    assert target_points(image) == []


def synthetic_label(scale=1, *, hp=True, glyph=True, digits=True, glyph_color=None):
    image = np.full((1080, 1920, 3), (110, 120, 130), np.uint8)
    x, y = 900, 430
    peach = (189, 225, 255) if glyph_color is None else glyph_color
    if glyph:
        # Three compact diagonal strokes reproduce the selected ship glyph.
        for offset in (-12, -3, 6):
            points = np.asarray([(x + (offset - 3) * scale, y + 5 * scale),
                                 (x + (offset + 3) * scale, y + 5 * scale),
                                 (x + (offset + 8) * scale, y - 5 * scale),
                                 (x + (offset + 2) * scale, y - 5 * scale)], np.int32)
            cv2.fillConvexPoly(image, points, peach)
    bx, by = round(x - 56 * scale), round(y + 35 * scale)
    if hp:
        # Only 25% remains; the entire numeric strip cannot be inferred from
        # the remaining red width. It has to be paired with the marker above.
        cv2.rectangle(image, (bx, by), (round(bx + 28 * scale), round(by + 14 * scale)),
                      (42, 80, 254), -1)
    if digits:
        for offset in range(8, 104, 12):
            cv2.rectangle(image, (round(bx + offset * scale), round(by + 3 * scale)),
                          (round(bx + (offset + 4) * scale), round(by + 11 * scale)),
                          (235, 235, 235), -1)
    return image


@pytest.mark.parametrize("scale", [.5, .75, 1, 1.25])
def test_selected_glyph_is_paired_with_small_remaining_hp_fill(scale):
    point, = target_points(synthetic_label(scale))
    assert point == pytest.approx((900, 430), abs=4)


@pytest.mark.parametrize("color", [(110, 132, 247), (116, 151, 254), (118, 153, 255)])
@pytest.mark.parametrize("scale", [.5, .75, 1, 1.25])
def test_measured_daylight_glyph_colors_require_a_true_enemy_label(color, scale):
    point, = target_points(synthetic_label(scale, glyph_color=color))
    assert point == pytest.approx((900, 430), abs=4)
    assert target_points(synthetic_label(scale, glyph_color=color, hp=False)) == []
    assert target_points(synthetic_label(scale, glyph_color=color, digits=False)) == []


@pytest.mark.parametrize("hp,glyph,digits", [(False, True, True), (True, False, True), (True, True, False)])
def test_peach_fragments_without_hp_marker_and_numeric_pair_cannot_form_a_target(hp, glyph, digits):
    assert target_points(synthetic_label(hp=hp, glyph=glyph, digits=digits)) == []


@pytest.mark.parametrize("glyph_color", [(80, 220, 80), (0, 170, 255), (210, 220, 225)])
def test_green_or_saturated_orange_or_grey_fragment_is_not_a_selected_ship(glyph_color):
    assert target_points(synthetic_label(glyph_color=glyph_color)) == []


def test_large_peach_cloud_with_red_foam_and_white_strokes_is_not_a_ship_glyph():
    image = synthetic_label(glyph=False)
    cv2.ellipse(image, (900, 430), (60, 20), 0, 0, 360, (189, 225, 255), -1)
    assert target_points(image) == []
