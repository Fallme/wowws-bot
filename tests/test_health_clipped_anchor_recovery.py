"""Lossless live HP evidence: clipped rows, punctuation, and weak prefixes."""

from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from core.hud_layout import HudLayout
from core.ocr import RapidOcrBackend
from core.vision import Vision
from test_live_health_recovery import read_tokens, token


EVIDENCE = Path("E:/aimemo/docs/screenshots/wows_blockages/"
                "health_reading_20261006_105057_1791255057586619500")


@pytest.mark.parametrize("separator", [".", "·", ":", "："])
@pytest.mark.parametrize("current", [0, 200, 52056, 59200])
def test_actual_integer_thousands_punctuation_preserves_full_fraction(separator, current):
    value = f"{current:,}".replace(",", separator)
    assert read_tokens([token(f"IX鲁普雷希特亲王{value}/59{separator}200", 10, 317)]) == pytest.approx(current / 59200)


@pytest.mark.parametrize("text", ["59200/59.20", "59200/59:2007", "59200/59·200 7"])
def test_punctuation_does_not_complete_missing_or_extra_digits(text):
    assert read_tokens([token(text, 10, 317)]) is None


@pytest.mark.parametrize("prefix", ["IY色普重希特亲王50", "50"])
def test_low_confidence_overlapping_prefix_vetoes_a_confident_three_digit_suffix(prefix):
    # Exact first false-zero screenshot: the clipped prefix is .67575, while
    # the independent-looking tail is .91395. Dropping the former yields 0%.
    assert read_tokens([token(prefix, 40 if len(prefix) > 2 else 212, 228,
                              74, 86, confidence=.67575),
                        token("200/50-206", 223, 314, 77, 86, confidence=.91395)]) is None


def test_low_confidence_prefix_cannot_authorize_a_joined_hp_value():
    assert read_tokens([token("58", 200, 220, confidence=.65),
                        token("623/59200", 224, 330)]) is None


@pytest.mark.parametrize("current", [0, 200, 52056])
def test_low_confidence_non_numeric_ship_name_keeps_true_damage_or_zero(current):
    assert read_tokens([token("鲁普雷希特亲王", 36, 220, confidence=.65),
                        token(f"{current}/59200", 224, 330)]) == pytest.approx(current / 59200)


def test_measured_hp_row_near_lower_edge_is_read_with_its_complete_glyphs():
    image = np.zeros((160, 500, 3), np.uint8)
    # This is actual image content below the predicted crop, rather than a
    # cached OCR token. The backend only reads the full fraction when visible.
    image[98:119, 210:315] = (17, 132, 225)
    layout = HudLayout(500, 160, overrides={"health_text": (0, 0, 474, 108)},
                       scales={"left": 1.125})

    class Backend:
        calls = 0
        def recognize(self, candidate):
            self.calls += 1
            full_glyphs = np.count_nonzero(candidate[:, 210:315, 2] == 225) == 21 * 105
            if full_glyphs:
                return [token("IX鲁普雷希特亲王59200/59200", 10, 317, 98, 119)]
            return [token("IY色普重希特亲王50", 40, 228, 98, 108, confidence=.67575),
                    token("200/50-206", 223, 314, 100, 108, confidence=.91395)]

    backend = Backend()
    with patch("core.vision.detect_hud_layout", return_value=layout):
        assert Vision.read_health_fraction(image, backend) == 1
    assert backend.calls == 1


class LimitedCpuBackend(RapidOcrBackend):
    def __init__(self):
        super().__init__(prefer_gpu=False)

    def _params(self, model_root, *, use_cuda):
        params = super()._params(model_root, use_cuda=use_cuda)
        params["EngineConfig.onnxruntime.intra_op_num_threads"] = 1
        params["EngineConfig.onnxruntime.inter_op_num_threads"] = 1
        return params


@pytest.fixture(scope="module")
def actual_cpu_backend():
    return LimitedCpuBackend()


@pytest.mark.skipif(not (EVIDENCE / "frame_0001_before.png").exists(),
                    reason="Locally retained lossless live evidence")
@pytest.mark.parametrize("number,phase,expected", [
    (1, "before", 1), (2, "after", 1), (3, "before", 1),
    (5, "before", 52056 / 59200), (15, "after", 0),
])
def test_retained_original_live_png_recovers_full_damage_and_true_zero(
        number, phase, expected, actual_cpu_backend):
    image = cv2.imread(str(EVIDENCE / f"frame_{number:04d}_{phase}.png"))
    assert image is not None
    assert Vision.read_health_fraction(image, actual_cpu_backend) == pytest.approx(expected)
