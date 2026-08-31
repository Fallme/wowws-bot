from pathlib import Path

import cv2
import numpy as np

from core.ocr import OcrToken
from core.vision import Vision
from port_navigator import (
    _verified_action_point,
    find_ship_card,
    is_requested_ship_selected,
    port_battle_action_point,
    results_requeue_action_point,
    results_return_to_port_action_point,
)


FIXTURE_ROOT = Path("tests") / "fixtures"


def token(text, left, top, right, bottom, confidence=0.98):
    return OcrToken(
        text,
        confidence,
        ((left, top), (right, top), (right, bottom), (left, bottom)),
    )


class StaticBackend:
    def __init__(self, tokens):
        self.tokens = list(tokens)

    def recognize(self, _image):
        return list(self.tokens)


def test_text_button_merges_adjacent_ocr_tokens_and_returns_their_union_center():
    image = np.zeros((900, 1600, 3), dtype=np.uint8)
    backend = StaticBackend(
        [
            token("加入", 40, 20, 90, 50),
            token("战斗", 96, 20, 146, 50),
        ]
    )

    # The backend coordinates are local to the wide top-header search crop.
    assert port_battle_action_point(image, backend) == (701, 35)


def test_text_button_enlarged_retry_maps_coordinates_back_to_original_image():
    image = np.zeros((100, 300, 3), dtype=np.uint8)

    class SmallTextBackend:
        def recognize(self, sample):
            if sample.shape[1] == 300:
                return [token("回列港曰", 100, 20, 220, 50)]
            return [token("回到港口", 145, 29, 319, 72)]

    assert _verified_action_point(
        image,
        SmallTextBackend(),
        ("回到港口",),
    ) == (160, 35)


def test_result_actions_accept_conservative_low_contrast_ocr_aliases():
    image = np.zeros((1000, 1600, 3), dtype=np.uint8)
    backend = StaticBackend(
        [
            token("刻港口", 40, 60, 130, 90, 0.78),
            token("避续战斗!", 100, 60, 220, 90, 0.82),
        ]
    )

    assert results_return_to_port_action_point(image, backend) == (485, 855)
    assert results_requeue_action_point(image, backend) == (1088, 855)


def test_ship_templates_survive_resolution_changes_in_real_port_fixture():
    source = cv2.imread(str(FIXTURE_ROOT / "port_ship_selected.png"))
    assert source is not None

    for scale in (0.50, 0.67, 0.75, 1.25):
        resized = cv2.resize(
            source,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC,
        )
        assert find_ship_card(resized, "napoli") is not None
        assert find_ship_card(resized, "pommern") is not None
        assert is_requested_ship_selected(resized, "napoli")
        assert not is_requested_ship_selected(resized, "pommern")


def test_compact_autopilot_indicator_remains_visible_on_4k_framebuffer():
    compact = np.zeros((2160, 3840, 3), dtype=np.uint8)
    noise = compact.copy()
    compact[1730:1740, 450:510] = (40, 190, 40)
    noise[1730:1732, 450:458] = (40, 190, 40)

    assert Vision.is_autopilot_enabled(compact)
    assert not Vision.is_autopilot_enabled(noise)
