"""Loaded side launchers need camera movement out of an orange bow bearing."""

from pathlib import Path
from dataclasses import replace
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from core.ocr import OcrToken, RapidOcrBackend
from core.torpedo_hud import (
    _Sector, _SlotEvidence, _cooldown_seconds, _translucent_sector_background,
    inspect_torpedo_hud,
)
from test_torpedo_alignment_vision import cone_frame


TRIAL = Path("E:/aimemo/docs/screenshots/wows_torpedoes/torpedo_20261005_221219_1791209539606363600")


@pytest.fixture(scope="module")
def backend():
    return RapidOcrBackend()


@pytest.mark.parametrize("frame,target", [(2, (769, 642)), (14, (346, 642))])
@pytest.mark.parametrize("scale", [.5, .75, 1, 1.25])
def test_actual_loaded_arc_and_orange_bearing_allow_only_camera_movement(frame, target, scale, backend):
    path = TRIAL / f"frame_{frame:04d}_observation.png"
    before_path = TRIAL / f"frame_{frame - 1:04d}_before.png"
    if not path.exists() or not before_path.exists():
        pytest.skip("Retained native orange-bearing trial is unavailable")
    image, before = cv2.imread(str(path)), cv2.imread(str(before_path))
    image, before = [cv2.resize(value, None, fx=scale, fy=scale,
                                interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
                     for value in (image, before)]
    point = tuple(round(value * scale) for value in target)
    result = inspect_torpedo_hud(image, before, point, backend)
    assert result.weapon_is_torpedo is True and result.selected
    assert result.cooldown_seconds is None  # Grey main-gun mount "2" is not reload.
    assert result.loaded_arc
    assert result.reticle_point == pytest.approx((1280 * scale, 800 * scale), abs=4)
    assert result.bearing_available is False
    assert not result.safe_to_fire


def orange_cone(*, broad=(20, 205, 20), small=True, reticle="ring"):
    image, target = cone_frame(broad=broad, small=small)
    image[276:325, 476:525] = 0
    if reticle == "ring":
        cv2.circle(image, (500, 300), 13, (0, 170, 220), 2)
    elif reticle == "filled":
        cv2.circle(image, (500, 300), 13, (0, 170, 220), -1)
    elif reticle == "warning":
        cv2.fillConvexPoly(image, np.asarray([(500, 285), (487, 313), (513, 313)]), (0, 170, 220))
    return image, target


def inspect_selected(image, target, *, weapon=True, selected=True):
    with patch("core.torpedo_hud._weapon_slot_evidence", return_value=_SlotEvidence(weapon, selected)):
        return inspect_torpedo_hud(image, image, target)


def test_orange_circle_cannot_authorize_a_shot_even_with_other_aligned_green_and_white_fans():
    image, target = orange_cone()
    result = inspect_selected(image, target)
    assert result.loaded_arc and result.reticle_point is not None
    assert result.aim_point is not None and result.lead_point is not None
    assert result.ready_to_align and result.aligned
    assert result.bearing_available is False
    assert not result.safe_to_fire
    # A fresh green bearing would permit the same independently measured axes.
    assert replace(result, bearing_available=True).safe_to_fire


def test_equally_central_green_ring_cannot_hide_orange_blocking_evidence():
    image, target = cone_frame()
    cv2.circle(image, (500, 300), 18, (0, 170, 220), 2)

    result = inspect_selected(image, target)

    assert result.ready_to_align and result.aligned
    assert result.bearing_available is False
    assert not result.safe_to_fire


def test_orange_bearing_retains_an_unaligned_white_prediction_for_mouse_movement():
    image, target = cone_frame(white_offset=55)
    # Recolour only the circular green pixels: the offset white edge crosses
    # this area and must remain independently observable.
    ring = np.zeros(image.shape[:2], np.uint8)
    cv2.circle(ring, (500, 300), 13, 255, 5)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    image[(ring > 0) & (hsv[:, :, 1] > 100) & (hsv[:, :, 0] >= 37)
          & (hsv[:, :, 0] <= 94)] = 0
    cv2.circle(image, (500, 300), 13, (0, 170, 220), 2)

    result = inspect_selected(image, target)

    assert result.ready_to_align and result.bearing_available is False
    assert result.lead_point[0] - result.aim_point[0] > result.alignment_tolerance_px
    assert not result.safe_to_fire


@pytest.mark.parametrize("scale", [.5, .75, 1, 1.25])
def test_real_close_enemy_with_orange_circle_keeps_white_line_alignment(scale, backend):
    trial = TRIAL.parent / "torpedo_20261005_221948_1791209988694974000"
    path, before_path = trial / "frame_0003_observation.png", trial / "frame_0001_before.png"
    if not path.exists() or not before_path.exists():
        pytest.skip("Retained native close-range alignment evidence unavailable")
    image, before = [cv2.resize(cv2.imread(str(value)), None, fx=scale, fy=scale)
                     for value in (path, before_path)]
    result = inspect_torpedo_hud(
        image, before, (round(1291 * scale), round(686 * scale)), backend,
    )

    assert result.weapon_is_torpedo is True and result.selected and result.loaded_arc
    assert result.reticle_point == pytest.approx((1280 * scale, 800 * scale), abs=4)
    assert result.bearing_available is False and result.ready_to_align
    assert result.aim_point is not None and result.lead_point is not None
    assert result.lead_point[0] < result.aim_point[0] - result.alignment_tolerance_px
    assert not result.safe_to_fire


def test_real_orange_only_movement_geometry_must_be_verified_again_when_bearing_turns_green():
    path = TRIAL.parent / "torpedo_20261005_221948_1791209988694974000/frame_0003_observation.png"
    if not path.exists():
        pytest.skip("Retained native close-range alignment evidence unavailable")
    image = cv2.imread(str(path))
    # Only recolour the real central orange circle, leaving sea/white pixels
    # unchanged. The new frame must return to strict white-edge verification.
    crop = image[775:825, 1255:1305]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    orange = cv2.inRange(hsv, np.array([9, 65, 90]), np.array([34, 255, 255]))
    crop[orange > 0] = (20, 220, 20)

    with patch("core.torpedo_hud._translucent_sector_background",
               wraps=_translucent_sector_background) as verification:
        result = inspect_selected(image, (1291, 686))

    assert result.loaded_arc and result.bearing_available is True
    assert verification.call_count > 0
    assert all(call.kwargs.get("movement_only") is False for call in verification.call_args_list)
    assert not result.safe_to_fire


def test_real_neutral_white_pair_with_unequal_alpha_is_movement_evidence_only():
    path = TRIAL.parent / "torpedo_20261005_221948_1791209988694974000/frame_0003_observation.png"
    if not path.exists():
        pytest.skip("Retained native close-range alignment evidence unavailable")
    image = cv2.imread(str(path))
    # Measured native white edges: the left/right alpha estimates differ by
    # 0.143. This pair must fail the ordinary firing-frame proof even though
    # both edges independently whiten and retain the underlying sea texture.
    sector = _Sector(
        (1278.1633, 1262.9890), (.30555556, .22459893),
        (892.25, 994.49733), 873, 1240, 797.39,
    )

    assert not _translucent_sector_background(image, sector, 1)
    assert _translucent_sector_background(image, sector, 1, movement_only=True)


@pytest.mark.parametrize("small,white", [(False, True), (True, False), (False, False)])
def test_orange_reticle_does_not_replace_missing_green_sector_or_white_prediction(small, white):
    image, target = cone_frame(small=small, white=white)
    image[276:325, 476:525] = 0
    cv2.circle(image, (500, 300), 13, (0, 170, 220), 2)

    result = inspect_selected(image, target)

    assert result.bearing_available is False
    assert not result.ready_to_align and not result.safe_to_fire


@pytest.mark.parametrize("broad", [None, (20, 205, 205)])
def test_orange_circle_never_supplies_missing_green_readiness(broad):
    image, target = orange_cone(broad=broad)
    result = inspect_selected(image, target)
    assert result.reticle_point is not None
    assert not result.loaded_arc and not result.safe_to_fire


@pytest.mark.parametrize("weapon,selected", [(None, True), (False, True), (True, False)])
def test_unknown_or_unselected_weapon_cannot_use_orange_circle_for_camera_alignment(weapon, selected):
    image, target = orange_cone(small=False)
    result = inspect_selected(image, target, weapon=weapon, selected=selected)
    assert result.reticle_point is None
    assert not result.safe_to_fire


@pytest.mark.parametrize("kind", ["filled", "warning"])
def test_filled_orange_blob_or_warning_triangle_is_not_a_camera_reticle(kind):
    image, target = orange_cone(small=False, reticle=kind)
    result = inspect_selected(image, target)
    assert result.reticle_point is None
    assert not result.safe_to_fire


@pytest.mark.parametrize("color,expected", [((170, 170, 170), None), ((112, 222, 244), 2)])
def test_upper_main_gun_number_needs_actual_yellow_counter_pixels(color, expected):
    image = np.zeros((1080, 1920, 3), np.uint8)
    slot = _SlotEvidence(True, True, (600, 930, 650, 980), .8)
    # slot crop is x594..656, y901..935; this box stays above the keycap.
    image[905:916, 610:620] = color

    class Backend:
        def recognize(self, crop):
            if crop.shape[:2] == (34, 62):
                return [OcrToken("2", .999, ((16, 4), (26, 4), (26, 15), (16, 15)))]
            return []

    assert _cooldown_seconds(image, Backend(), slot) == expected
