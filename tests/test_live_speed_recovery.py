"""Speed recovery is anchored to the own-ship engine telegraph, not sea circles."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from core.ocr import OcrToken, RapidOcrBackend
from core.vision import Vision


LIVE_ROOT = Path("E:/aimemo/docs/screenshots/wows_driving_trials/run_20261005_134040_1791178840243077000")


def token(text, x, y, width=44, height=20, confidence=.99):
    return OcrToken(text, confidence, ((x, y), (x + width, y),
                                     (x + width, y + height), (x, y + height)))


def telegraph_tokens():
    return [token("FULL", 200, 40), token("3/4", 210, 70, 24),
            token("1/2", 210, 100, 24), token("STOP", 200, 160)]


def fallback_speed(tokens):
    image = np.zeros((720, 1280, 3), np.uint8)
    wrong_crop = np.zeros((40, 60, 3), np.uint8)

    class Backend:
        def recognize(self, crop):
            return tokens if crop.shape[0] > 40 else []

    with patch("core.vision.detect_hud_layout", return_value=SimpleNamespace(crop=lambda *_: wrong_crop)):
        return Vision.read_speed_knots(image, Backend())


def test_recovery_uses_speed_next_to_verified_engine_column():
    tokens = telegraph_tokens() + [token("32.0 kts", 270, 43, 70),
                                    token("8.2 kts", 20, 43, 70)]
    assert fallback_speed(tokens) == 32.0


def test_split_decimal_overlapping_the_units_last_digit_is_not_two_knots():
    tokens = telegraph_tokens() + [token("24.2", 270, 73, 47, confidence=.91),
                                    token("2kts", 309, 73, 39, confidence=.99)]
    assert fallback_speed(tokens) == 24.2


@pytest.mark.parametrize("tokens", [
    [token("32.0 kts", 270, 43, 70)],
    telegraph_tokens()[:2] + [token("32.0 kts", 270, 43, 70)],
    telegraph_tokens() + [token("32.0", 270, 43, 70)],
    telegraph_tokens() + [token("12.5 km", 270, 43, 70)],
    telegraph_tokens() + [token("32.0 kts", 270, 43, 70, confidence=.69)],
    telegraph_tokens() + [token("32.0 kts", 270, 43, 70), token("31.0 kts", 270, 100, 70)],
    telegraph_tokens() + [OcrToken("32.0 kts", .99, ((float("nan"), 0),) * 4)],
])
def test_wider_search_rejects_missing_column_unrelated_units_and_uncertain_numbers(tokens):
    assert fallback_speed(tokens) is None


def test_crop_failure_remains_unknown_without_breaking_the_control_loop():
    class Backend:
        def recognize(self, _):
            raise RuntimeError("temporary OCR failure")

    assert Vision.read_speed_knots(np.zeros((720, 1280, 3), np.uint8), Backend()) is None


@pytest.fixture(scope="module")
def actual_backend():
    return RapidOcrBackend(prefer_gpu=False)


@pytest.mark.skipif(not (LIVE_ROOT / "full_0020.png").exists(), reason="Locally retained live trial evidence")
@pytest.mark.parametrize("name,expected", [("full_0020.png", 32.0), ("full_0040.png", 32.0),
                                           ("full_0060.png", 24.2)])
def test_real_trial_speed_with_actual_cpu_ocr(name, expected, actual_backend):
    image = cv2.imread(str(LIVE_ROOT / name))
    assert Vision.read_speed_knots(image, actual_backend) == expected


@pytest.mark.skipif(not (LIVE_ROOT / "full_0020.png").exists(), reason="Locally retained live trial evidence")
@pytest.mark.parametrize("resolution_scale,ui_scale,offset_x,offset_y", [
    (.5, 1, 0, 0), (.75, 1, 0, 0), (1, .65, 55, 30), (1, 1.25, 35, 15),
])
def test_real_speed_readout_survives_independent_ui_scaling_and_movement(
        resolution_scale, ui_scale, offset_x, offset_y, actual_backend):
    source = cv2.imread(str(LIVE_ROOT / "full_0020.png"))
    if resolution_scale != 1:
        image = cv2.resize(source, None, fx=resolution_scale, fy=resolution_scale,
                           interpolation=cv2.INTER_AREA)
    else:
        h, w = source.shape[:2]
        old_top = round(h * .68)
        panel = cv2.resize(source[old_top:, :round(w * .255)], None,
                           fx=ui_scale, fy=ui_scale, interpolation=cv2.INTER_CUBIC)
        image = source.copy()
        image[old_top:, :round(w * .32)] = (70, 75, 80)
        top = h - offset_y - panel.shape[0]
        image[top:top + panel.shape[0], offset_x:offset_x + panel.shape[1]] = panel
    assert Vision.read_speed_knots(image, actual_backend) == 32.0
