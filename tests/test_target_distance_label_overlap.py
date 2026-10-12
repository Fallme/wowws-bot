from pathlib import Path

import cv2
import numpy as np
import pytest

from core.ocr import OcrToken, TargetDistanceReader


class LabelBackend:
    def recognize(self, image):
        return [OcrToken("62 927 / 63 900", .99), OcrToken("5.7 km", .99)]


def label_crop(scale, *, numeric=True):
    image = np.zeros((round(100 * scale), round(250 * scale), 3), np.uint8)
    x, y, width, height = (round(value * scale) for value in (50, 45, 120, 14))
    image[y:y + height, x:x + width] = (160, 210, 25)
    if numeric:
        for offset in (10, 25, 40, 65, 80, 95):
            gx = x + round(offset * scale)
            image[y + round(3 * scale):y + round(11 * scale),
                  gx:gx + max(1, round(3 * scale))] = 255
    return image


@pytest.mark.parametrize("scale", [.5, .75, 1, 1.25, 1.5])
def test_numeric_green_hp_strip_is_ambiguous_at_independent_ui_sizes(scale):
    assert TargetDistanceReader._friendly_health_label_overlaps(label_crop(scale))


@pytest.mark.parametrize("scale", [.5, 1, 1.5])
def test_plain_green_shape_cannot_establish_a_friendly_hp_label(scale):
    assert not TargetDistanceReader._friendly_health_label_overlaps(label_crop(scale, numeric=False))


def test_overlapping_friend_distance_cannot_become_accepted_enemy_evidence():
    image = np.zeros((720, 1280, 3), np.uint8)
    anchor = (640, 360)
    reader = TargetDistanceReader(LabelBackend())
    roi = reader.target_roi(image, anchor)
    crop = label_crop(.5)
    image[roi.y:roi.y + crop.shape[0], roi.x:roi.x + crop.shape[1]] = crop
    result = reader.read(image, anchor, "enemy-1")
    assert result is not None and result.value_km == 5.7
    assert not result.accepted
    assert result.reject_reason == "friendly_label_overlap"


@pytest.mark.parametrize("scale", [.5, .75, 1, 1.25])
def test_actual_mixed_enemy_friend_capture_rejects_friend_distance(scale):
    source = Path("E:/aimemo/docs/screenshots/wows_torpedoes/torpedo_20261005_203649_1791203809232784600/frame_0001_observation.png")
    if not source.is_file():
        pytest.skip("Optional live capture is unavailable")
    image = cv2.imread(str(source))
    image = cv2.resize(image, None, fx=scale, fy=scale,
                       interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
    reader = TargetDistanceReader(LabelBackend())
    anchor = tuple(round(value * scale) for value in (1419, 606))
    roi = reader.target_roi(image, anchor)
    crop = image[roi.y:roi.y + roi.height, roi.x:roi.x + roi.width]
    assert reader._friendly_health_label_overlaps(crop)
    result = reader.read(image, anchor, "enemy-1")
    assert result is not None and not result.accepted
    assert result.reject_reason == "friendly_label_overlap"


@pytest.mark.parametrize("scale", [.5, .75, 1, 1.25])
def test_actual_single_enemy_label_remains_accepted_at_small_ui_sizes(scale):
    source = Path(__file__).parent / "fixtures" / "live_battle.png"
    image = cv2.imread(str(source))
    assert image is not None
    image = cv2.resize(image, None, fx=scale, fy=scale,
                       interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
    reader = TargetDistanceReader(LabelBackend())
    anchor = tuple(round(value * scale) for value in (1080, 604))
    result = reader.read(image, anchor, "enemy-1")
    assert result is not None and result.accepted
    assert result.reject_reason is None


@pytest.mark.parametrize("kind", ["ring", "textured_sea", "solid_green"])
def test_green_reticle_and_sea_do_not_invalidate_otherwise_valid_distance(kind):
    image = np.full((720, 1280, 3), (110, 110, 75), np.uint8)
    anchor = (640, 360)
    reader = TargetDistanceReader(LabelBackend())
    roi = reader.target_roi(image, anchor)
    crop = image[roi.y:roi.y + roi.height, roi.x:roi.x + roi.width]
    if kind == "ring":
        cv2.circle(crop, (crop.shape[1] // 2, crop.shape[0] // 2), 20, (30, 220, 30), 2)
    elif kind == "solid_green":
        crop[20:34, 20:100] = (160, 210, 25)
    else:
        _, xx = np.mgrid[:crop.shape[0], :crop.shape[1]]
        crop[:] = np.stack([110 + 10 * np.sin(xx / 15),
                            150 + 10 * np.sin(xx / 15),
                            55 + 10 * np.sin(xx / 15)], axis=-1).astype(np.uint8)
    result = reader.read(image, anchor, "enemy-1")
    assert result is not None and result.accepted
    assert result.reject_reason is None
