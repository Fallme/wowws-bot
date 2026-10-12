"""Faint filled prediction geometry retains background; paint does not."""

from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from core.hud_layout import detect_hud_layout
from core.torpedo_hud import (
    _Sector, _SlotEvidence, _translucent_prediction_sectors,
    _translucent_sector_background,
    inspect_torpedo_hud,
)


def translucent_frame(width=1000, height=600, ui_scale=.7, *, shift=0,
                      alpha=.32, broad=True, orange=False, opaque=False,
                      white=True, flat=False):
    yy, xx = np.mgrid[:height, :width]
    base = 95 + yy * .015 + 12 * np.sin(yy / 13) + 8 * np.sin(xx / 29)
    if flat:
        base = np.full_like(base, 110)
    image = np.clip(np.stack([base + 8, base, base - 6], axis=-1), 0, 255).astype(np.uint8)
    for offset in (-280, -180, -80, 80, 180, 280):
        cx, cy = width / 2 + offset * ui_scale, height - 17 * ui_scale
        half = 11.5 * ui_scale
        cv2.rectangle(image, (round(cx - half), round(cy - half)),
                      (round(cx + half), round(cy + half)), (225, 225, 225), 1)
    apex = (round(width / 2), round(height * .87))
    top_y = round(height * .35)
    launch_colour = (20, 205, 205) if orange else (20, 220, 20)
    if broad:
        cv2.ellipse(image, (round(width / 2), round(height * .9)),
                    (round(width * .30), round(height * .18)), 0, 210, 330,
                    launch_colour, max(2, round(4 * ui_scale)))
    for sign in (-1, 1):
        cv2.line(image, (round(width / 2 + sign * height * .09), top_y), apex,
                 launch_colour, max(2, round(3 * ui_scale)))
    cv2.circle(image, (round(width / 2), round(height * .5)),
               max(7, round(18 * ui_scale)), launch_colour, max(1, round(2 * ui_scale)))
    if white:
        mask = np.zeros((height, width), np.uint8)
        polygon = np.array([[round(width / 2 + shift - height * .04), top_y],
                            [round(width / 2 + shift + height * .04), top_y], apex], np.int32)
        cv2.fillConvexPoly(mask, polygon, 255)
        if opaque:
            image[mask > 0] = (185, 185, 185)
        else:
            image[mask > 0] = np.round(image[mask > 0] * (1 - alpha) + 255 * alpha).astype(np.uint8)
    return image, (round(width / 2 + shift), round(height * .5))


def inspect(image, target, *, cooldown=None):
    with patch("core.torpedo_hud._weapon_slot_evidence", return_value=_SlotEvidence(True, True)), \
         patch("core.torpedo_hud._cooldown_seconds", return_value=cooldown):
        return inspect_torpedo_hud(image, image.copy(), target)


@pytest.mark.parametrize("width,height,ui_scale", [
    (1280, 720, .8), (1920, 1080, 1.25), (3440, 1440, .75), (3840, 2160, .6),
])
def test_translucent_band_aligns_only_with_actual_green_loaded_sector(width, height, ui_scale):
    image, target = translucent_frame(width, height, ui_scale)
    result = inspect(image, target)
    assert result.loaded_arc and result.ready_to_align and result.safe_to_fire
    assert result.alignment_error_px <= result.alignment_tolerance_px
    assert result.alignment_error_degrees <= 2


def test_shifted_translucent_prediction_requires_relative_alignment_before_fire():
    image, target = translucent_frame(shift=65)
    result = inspect(image, target)
    assert result.loaded_arc and result.ready_to_align
    assert result.lead_point[0] > result.aim_point[0]
    assert not result.safe_to_fire


@pytest.mark.parametrize("kwargs", [{"orange": True}, {"broad": False}])
def test_translucent_white_band_cannot_supply_green_or_loaded_authority(kwargs):
    image, target = translucent_frame(**kwargs)
    with patch("core.torpedo_hud._translucent_prediction_sectors", wraps=_translucent_prediction_sectors) as fallback:
        result = inspect(image, target)
    assert not result.loaded_arc and not result.ready_to_align and not result.safe_to_fire
    fallback.assert_not_called()


def test_positive_countdown_cannot_use_the_translucent_fallback():
    image, target = translucent_frame()
    with patch("core.torpedo_hud._translucent_prediction_sectors", wraps=_translucent_prediction_sectors) as fallback:
        result = inspect(image, target, cooldown=30)
    assert result.loaded_arc and not result.ready_to_align and not result.safe_to_fire
    fallback.assert_not_called()


@pytest.mark.parametrize("kwargs", [{"opaque": True}, {"flat": True}, {"alpha": .9}])
def test_opaque_or_textureless_wedge_is_not_translucent_prediction(kwargs):
    image, target = translucent_frame(**kwargs)
    result = inspect(image, target)
    assert result.loaded_arc
    assert result.lead_point is None and not result.safe_to_fire


@pytest.mark.parametrize("kind", ["cursor", "explosion", "cloud"])
def test_white_pointer_explosion_and_cloud_with_a_ready_launcher_do_not_supply_lead(kind):
    image, target = translucent_frame(white=False)
    width, height = image.shape[1], image.shape[0]
    if kind == "cursor":
        cv2.fillConvexPoly(image, np.array([[500, 285], [517, 311], [504, 308], [499, 326]]), (235, 235, 235))
    elif kind == "explosion":
        cv2.circle(image, (500, 300), 60, (240, 240, 240), -1)
    else:
        cv2.ellipse(image, (500, 270), (110, 65), 0, 0, 360, (180, 180, 180), -1)
        cv2.circle(image, (450, 260), 65, (185, 185, 185), -1)
    result = inspect(image, target)
    assert result.loaded_arc and result.lead_point is None and not result.safe_to_fire


def test_no_prediction_in_real_selected_ice_scene_stays_unsafe():
    image = cv2.imread(str(Path(__file__).parent / "fixtures" / "live_battle.png"))
    assert image is not None
    result = inspect(image, (1280, 800))
    assert result.loaded_arc and result.lead_point is None and not result.safe_to_fire


def test_bright_thin_prediction_has_priority_over_faint_filled_fallback():
    image, target = translucent_frame(white=False)
    apex = (500, 522)
    for x in (476, 524):
        cv2.line(image, (x, 210), apex, (235, 235, 235), 2)
    with patch("core.torpedo_hud._translucent_prediction_sectors", side_effect=AssertionError("bright geometry already exists")):
        assert inspect(image, target).safe_to_fire


def test_two_separate_translucent_bands_cannot_form_a_middle_prediction():
    image, target = translucent_frame(white=False)
    apex = (500, 522)
    for center in (450, 550):
        mask = np.zeros(image.shape[:2], np.uint8)
        cv2.fillConvexPoly(mask, np.array([[center - 14, 210],
                                         [center + 14, 210], apex]), 255)
        image[mask > 0] = np.round(image[mask > 0] * .68 + 255 * .32).astype(np.uint8)
    slopes = ((500 - 436) / 312, (500 - 564) / 312)
    intercepts = tuple(x - slope * 210 for x, slope in zip((436, 564), slopes))
    combined = _Sector(apex, slopes, intercepts, 210, 490, 1000)
    assert not _translucent_sector_background(image, combined, .7)
    assert not inspect(image, target).safe_to_fire


def test_retained_real_translucent_band_is_geometry_only_without_green_loaded_authority():
    path = Path("E:/aimemo/docs/screenshots/wows_driving_trials/run_20261004_191531_1791112531200493900/full_0140.png")
    if not path.exists():
        pytest.skip("Optional real translucent prediction capture is unavailable")
    image = cv2.imread(str(path))
    assert image is not None
    layout = detect_hud_layout(image)
    height, width = image.shape[:2]
    bounds = (round(width * .1), round(height * .18), round(width * .9), round(height * .9))
    # The measured orange perspective apex is only a geometry reference here.
    # Production never calls this branch for that orange/empty launcher.
    reference = _Sector((1309, 1024), (.20, .30), (1104.2, 1001.8), 550, 1040, 1000)
    sectors = _translucent_prediction_sectors(image, bounds, layout, reference, 750)
    assert sectors
    assert min(abs(sector.axis_x(750) - 1172) for sector in sectors) < 20
    result = inspect(image, (1331, 449))
    assert not result.loaded_arc and not result.safe_to_fire


@pytest.mark.parametrize("scale", [.75, 1.0])
def test_live_sunlit_prediction_recovers_geometry_despite_shell_trail_samples(scale):
    image = cv2.imread(str(Path(__file__).parent / "fixtures" / "torpedo_prediction_live_2560.png"))
    assert image is not None
    if scale != 1:
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    # This regression verifies sea-overlay geometry independently of OCR slot
    # identity; the live worker still confirms the actual weapon and cooldown.
    result = inspect(image, (round(1427 * scale), round(664 * scale)))
    assert result.loaded_arc and result.ready_to_align
    assert abs(result.lead_point[0] - 1441 * scale) < 8 * scale
    assert abs(result.aim_point[0] - 1279 * scale) < 8 * scale
    assert result.alignment_error_px > result.alignment_tolerance_px
    assert not result.safe_to_fire
