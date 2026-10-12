"""A decorative slash must not hide an observed overlapping HP prefix."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from core.ocr import OcrToken, RapidOcrBackend
from core.vision import Vision
from test_live_health_recovery import read_tokens, token


EVIDENCE = Path("E:/aimemo/docs/screenshots/wows_blockages/"
                "health_reading_20261006_002614_1791217574369893200")
TRIAL = Path("E:/aimemo/docs/screenshots/wows_driving_trials/"
             "run_20261006_002605_1791217565978733000")


@pytest.mark.parametrize("ornament", ["/", "|", "｜"])
def test_observed_ship_name_prefix_with_decorative_slash_cannot_leave_200_hp(ornament):
    # The actual enhanced OCR has an overlapping "59 2" prefix; its final 2
    # is duplicated by the following "200". Joining only /59 + 200 is wrong.
    assert read_tokens([
        token(f"{ornament} IX 鲁普雷希特亲王 59 2", 10, 236, 66, 86),
        token("200/59", 227, 280, 69, 83),
        token("200", 287, 316, 68, 84),
    ]) is None


@pytest.mark.parametrize("hp", [0, 200, 623, 47906, 58623])
def test_real_zero_low_and_damaged_hp_are_kept_after_a_non_numeric_ship_name(hp):
    assert read_tokens([
        token("/ IX 鲁普雷希特亲王", 10, 220),
        token(f"{hp}/59200", 224, 330),
    ]) == pytest.approx(hp / 59200)


def test_nearby_actual_fraction_does_not_become_a_numerator_prefix():
    assert read_tokens([token("1/2", 190, 220), token("200/59200", 224, 330)]) == pytest.approx(200 / 59200)


@pytest.mark.parametrize("current", [0, 200, 47906, 59200])
def test_original_and_enlarged_ambiguity_retries_for_the_complete_fraction(current):
    class Backend(RapidOcrBackend):
        def __init__(self):
            self.calls = 0

        def recognize(self, _image):
            self.calls += 1
            if self.calls == 1:
                # Actual original crop: overlapping denominator groups.
                return [token("IX鲁普雷希特亲王59200/59", 10, 283, 66, 86),
                        token("9200", 274, 316, 68, 84)]
            if self.calls == 2:
                # Actual 2.4x enlarged crop: decorative slash plus overlap.
                return [token("/ IX 鲁普雷希特亲王 59 2", 23, 567, 160, 206),
                        token("200/59", 545, 672, 167, 200),
                        token("200", 689, 757, 165, 202)]
            return [OcrToken(f"{current}/59200", .95)]

    backend = Backend()
    image = np.zeros((1600, 2560, 3), np.uint8)
    crop = np.zeros((110, 467, 3), np.uint8)
    with patch("core.vision.detect_hud_layout",
               return_value=SimpleNamespace(crop=lambda *_: crop)):
        assert Vision.read_health_fraction(image, backend) == pytest.approx(current / 59200)
    assert backend.calls == 3


@pytest.fixture(scope="module")
def cpu_backend():
    return RapidOcrBackend(prefer_gpu=False)


@pytest.mark.skipif(not (EVIDENCE / "frame_0001_before.png").exists(),
                    reason="Locally retained lossless HP evidence")
@pytest.mark.parametrize("number", [1, 3, 5, 7, 9, 11])
def test_original_live_health_change_png_preserves_observed_full_fraction(number, cpu_backend):
    image = cv2.imread(str(EVIDENCE / f"frame_{number:04d}_before.png"))
    assert image is not None
    assert Vision.read_health_fraction(image, cpu_backend) == 1


@pytest.mark.skipif(not (TRIAL / "full_0040.png").exists(),
                    reason="Locally retained lossless driving evidence")
def test_original_live_trial_frame_preserves_observed_full_fraction(cpu_backend):
    image = cv2.imread(str(TRIAL / "full_0040.png"))
    assert image is not None
    assert Vision.read_health_fraction(image, cpu_backend) == 1
