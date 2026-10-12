"""Live HP fractions cannot absorb nearby digits or overlapping OCR boxes."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from core.ocr import OcrToken, RapidOcrBackend
from core.vision import Vision


LIVE_ROOT = Path("E:/aimemo/docs/screenshots/wows_driving_trials/run_20261005_134743_1791179263892492500")
FULL_HP_ROOT = Path("E:/aimemo/docs/screenshots/wows_driving_trials/run_20261005_141526_1791180926725332600")


def token(text, left, right, top=68, bottom=88, confidence=.99):
    return OcrToken(text, confidence, ((left, top), (right, top), (right, bottom), (left, bottom)))


def read_tokens(tokens):
    image = np.zeros((600, 1000, 3), np.uint8)
    crop = np.zeros((112, 474, 3), np.uint8)

    class Backend:
        calls = 0
        def recognize(self, _):
            self.calls += 1
            return tokens

    backend = Backend()
    with patch("core.vision.detect_hud_layout", return_value=SimpleNamespace(crop=lambda *_: crop)):
        result = Vision.read_health_fraction(image, backend)
    assert backend.calls == 1
    return result


@pytest.mark.parametrize("loose", [token("7", 12, 45, 69, 82), token("7", 322, 335),
                                    token("123", 322, 350)])
def test_complete_fraction_wins_without_absorbing_independent_digits(loose):
    assert read_tokens([loose, token("IX鲁普雷希特亲王58623/59200", 36, 317)]) == pytest.approx(58623 / 59200)


@pytest.mark.parametrize("value,expected", [("52 344 / 86 000", 52344 / 86000),
                                           ("58,623/59,200", 58623 / 59200),
                                           ("0 / 59 200", 0)])
def test_complete_thousands_groups_and_zero_preserve_numeric_contract(value, expected):
    assert read_tokens([OcrToken(value, .99)]) == pytest.approx(expected)


def test_adjacent_three_digit_group_can_complete_the_same_baseline_fraction():
    assert read_tokens([token("IX鲁普雷希特亲王58623/59", 36, 283),
                        token("200", 287, 317, 69, 88)]) == pytest.approx(58623 / 59200)


def test_current_and_maximum_groups_are_joined_completely_not_as_623_hp():
    assert read_tokens([token("58", 200, 220), token("623/59", 224, 284),
                        token("200", 288, 318)]) == pytest.approx(58623 / 59200)


def test_three_digit_numerator_with_a_clear_adjacent_prefix_is_not_complete_hp():
    assert read_tokens([token("58", 200, 220), token("623/59200", 224, 330)]) == pytest.approx(58623 / 59200)


@pytest.mark.parametrize("prefix", [token("IX鲁普雷希特亲王592", 37, 236, 65, 85),
                                     token("IX鲁普雷希特亲王59", 37, 230, 66, 86),
                                     token("鲁普雷希特亲王 59", 62, 223, 65, 85.4),
                                     token("58", 208, 235, 68, 83)])
def test_overlapping_numerator_prefix_cannot_join_only_the_maximum(prefix):
    assert read_tokens([prefix, token("200/59", 227, 280, 68, 82),
                        token("200", 288, 315, 68, 83)]) is None


def test_ship_name_numeric_prefix_cannot_be_dropped_from_complete_suffix():
    assert read_tokens([token("鲁普雷希特亲王 59", 36, 220),
                        token("200/59200", 224, 330)]) is None


@pytest.mark.parametrize("current", ["0", "200", "623"])
def test_genuine_low_or_zero_hp_after_a_ship_name_without_digits_remains_valid(current):
    assert read_tokens([token("鲁普雷希特亲王", 36, 220),
                        token(f"{current}/59200", 224, 330)]) == pytest.approx(int(current) / 59200)


def test_overlapping_full_hp_groups_retry_until_the_fraction_is_complete():
    class Backend(RapidOcrBackend):
        def __init__(self):
            self.calls = 0
        def recognize(self, _):
            self.calls += 1
            if self.calls == 1:
                return [token("IX鲁普雷希特亲王592", 37, 236, 65, 85),
                        token("200/59", 227, 280, 68, 82), token("200", 288, 315, 68, 83)]
            if self.calls == 2:
                return [token("鲁普雷希特亲王 59", 36, 220),
                        token("200/59", 224, 284), token("200", 288, 318)]
            return [OcrToken("IX鲁普雷希特亲王59200/59200", .99)]

    backend = Backend()
    image = np.zeros((600, 1000, 3), np.uint8)
    assert Vision.read_health_fraction(image, backend) == 1
    assert backend.calls == 3


@pytest.mark.parametrize("prefix", [token("58", 200, 220, 96, 111), token("58", 170, 190),
                                     token("7", 190, 223, 69, 82)])
def test_three_digit_real_low_hp_does_not_absorb_an_unrelated_prefix(prefix):
    assert read_tokens([prefix, token("623/59200", 224, 330)]) == pytest.approx(623 / 59200)


@pytest.mark.parametrize("suffix", [token("9200", 273, 315, 70, 86),
                                      token("9200", 287, 329), token("7", 287, 299),
                                      token("200", 287, 317, 95, 111),
                                      token("200", 350, 380), token("200", 287, 317, 73, 82)])
def test_overlapping_partial_non_three_digit_other_row_and_wrong_size_are_rejected(suffix):
    assert read_tokens([token("IX鲁普雷希特亲王58623/59", 36, 283), suffix]) is None


def test_single_box_with_an_extra_invalid_digit_group_is_not_ten_percent_hp():
    assert read_tokens([token("58623/59200 7", 200, 340)]) is None


def test_ambiguous_split_is_retried_on_an_enhanced_crop():
    class Backend(RapidOcrBackend):
        def __init__(self):
            self.calls = 0
        def recognize(self, _):
            self.calls += 1
            if self.calls == 1:
                return [token("IX鲁普雷希特亲王58623/59", 36, 283), token("9200", 273, 315, 70, 86)]
            return [OcrToken("58623/59200", .99)]

    backend = Backend()
    image = np.zeros((600, 1000, 3), np.uint8)
    assert Vision.read_health_fraction(image, backend) == pytest.approx(58623 / 59200)
    assert backend.calls == 2


@pytest.fixture(scope="module")
def actual_backend():
    return RapidOcrBackend(prefer_gpu=False)


@pytest.mark.skipif(not (LIVE_ROOT / "full_0120.png").exists(), reason="Locally retained live trial evidence")
@pytest.mark.parametrize("name", ["full_0120.png", "full_0140.png", "full_0160.png"])
def test_real_live_fraction_with_actual_cpu_ocr(name, actual_backend):
    image = cv2.imread(str(LIVE_ROOT / name))
    assert Vision.read_health_fraction(image, actual_backend) == pytest.approx(58623 / 59200)


@pytest.mark.skipif(not (FULL_HP_ROOT / "full_0000.png").exists(), reason="Locally retained live trial evidence")
@pytest.mark.parametrize("name", ["full_0000.png", "full_0020.png", "full_0040.png", "full_0060.png"])
def test_real_full_hp_split_current_with_actual_cpu_ocr(name, actual_backend):
    image = cv2.imread(str(FULL_HP_ROOT / name))
    assert Vision.read_health_fraction(image, actual_backend) == 1


@pytest.mark.skipif(not (LIVE_ROOT / "full_0120.png").exists(), reason="Locally retained live trial evidence")
@pytest.mark.parametrize("resolution_scale,ui_scale,offset_x,offset_y", [
    (.5, 1, 0, 0), (.75, 1, 0, 0), (1, .75, 40, 30), (1, 1.25, 35, 15),
])
def test_real_hp_fraction_at_changed_resolution_and_moved_ui(
        resolution_scale, ui_scale, offset_x, offset_y, actual_backend):
    source = cv2.imread(str(LIVE_ROOT / "full_0160.png"))
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
    assert Vision.read_health_fraction(image, actual_backend) == pytest.approx(58623 / 59200)
