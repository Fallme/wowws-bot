import numpy as np

from core.ocr import OcrToken
from port_navigator import (
    _is_early_exit_confirmation,
    quick_exit_menu_action_point,
)


def token(text, left, top, right, bottom, confidence=0.98):
    return OcrToken(
        text,
        confidence,
        ((left, top), (right, top), (right, bottom), (left, bottom)),
    )


class StaticBackend:
    def __init__(self, tokens):
        self.tokens = tokens

    def recognize(self, _image):
        return list(self.tokens)


class BrokenBackend:
    @staticmethod
    def recognize(_image):
        raise RuntimeError("OCR unavailable")


def test_quick_exit_finds_leave_battle_and_rejects_exit_game():
    image = np.zeros((1494, 2560, 3), dtype=np.uint8)
    backend = StaticBackend(
        [
            token("退出游戏", 1150, 520, 1390, 570),
            token("离开战斗", 1160, 590, 1400, 640),
            token("回到战斗", 1160, 700, 1400, 750),
        ]
    )

    assert quick_exit_menu_action_point(image, backend) == (1280, 615)
    assert (
        quick_exit_menu_action_point(
            image,
            StaticBackend([token("退出游戏", 1150, 520, 1390, 570)]),
        )
        is None
    )


def test_early_exit_second_page_requires_both_text_anchors():
    image = np.zeros((1494, 2560, 3), dtype=np.uint8)
    confirmed = StaticBackend(
        [
            token("提前退出战斗", 1170, 390, 1390, 440),
            token("离开战斗?", 1190, 830, 1370, 880),
            token("是", 1160, 900, 1210, 945),
            token("否", 1350, 900, 1400, 945),
        ]
    )
    missing_heading = StaticBackend(
        [token("离开战斗?", 1190, 830, 1370, 880)]
    )

    assert _is_early_exit_confirmation(image, confirmed)
    assert not _is_early_exit_confirmation(image, missing_heading)


def test_early_exit_ocr_failure_never_authorizes_confirmation_click():
    image = np.zeros((1494, 2560, 3), dtype=np.uint8)

    assert not _is_early_exit_confirmation(image, BrokenBackend())
