import cv2
import numpy as np
import pytest

from core.hud_layout import detect_hud_layout
from core.ocr import OcrToken, TargetDistanceReader
from core.torpedo_hud import _cooldown_seconds


def key_row(width, height, scale, offset_x=0, offset_y=0):
    image = np.zeros((height, width, 3), dtype=np.uint8)
    center_x, center_y = width / 2 + offset_x, height - 18 * scale + offset_y
    for x in (1000, 1100, 1200, 1360, 1460, 1560):
        cx = center_x + (x - 1280) * scale
        half = 11.5 * scale
        cv2.rectangle(image, (round(cx - half), round(center_y - half)),
                      (round(cx + half), round(center_y + half)), (230, 230, 230), 1)
    return image


@pytest.mark.parametrize("width,height,scale", [
    (1280, 720, .8), (1920, 1080, 1.25), (3440, 1440, .75), (3840, 2160, 1.1),
])
def test_target_roi_uses_actual_ui_pixels_and_live_target_position(width, height, scale):
    image = key_row(width, height, scale, offset_x=35, offset_y=-25)
    layout = detect_hud_layout(image)
    assert layout.source("consumables") == "anchor"
    anchor = (round(width * .63), round(height * .37))
    roi = TargetDistanceReader.target_roi(image, anchor)
    assert roi.width == pytest.approx(320 * scale, rel=.12)
    assert roi.height == pytest.approx(108 * scale, rel=.12)
    assert roi.x + roi.width / 2 == pytest.approx(anchor[0], abs=1)
    assert roi.y < anchor[1] < roi.y + roi.height


def test_enlarged_target_label_is_read_below_the_old_resolution_crop():
    image = key_row(1920, 1080, 1.25)
    anchor = (800, 400)
    cv2.rectangle(image, (780, 490), (820, 494), (255, 0, 0), -1)

    class PixelBackend:
        def recognize(self, crop):
            if np.count_nonzero((crop[:, :, 0] > 180) & (crop[:, :, 1] < 20)):
                return [OcrToken("40000 / 60000", .99), OcrToken("7.2 km", .99)]
            return []

    result = TargetDistanceReader(backend=PixelBackend()).read(image, anchor, "enemy")
    assert result is not None and result.accepted
    assert result.value_km == pytest.approx(7.2)


def test_key_row_alone_cannot_associate_a_countdown_with_torpedoes():
    image = key_row(1920, 1080, .75, offset_x=-160, offset_y=-115)
    layout = detect_hud_layout(image)
    assert layout.source("consumables") == "anchor"
    class Backend:
        def recognize(self, crop):
            return [OcrToken("42", .98, ((10, 10), (24, 10), (24, 22), (10, 22)))]

    assert _cooldown_seconds(image, Backend()) is None
