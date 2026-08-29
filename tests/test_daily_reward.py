from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import cv2
import numpy as np

from core.ocr import OcrToken, RapidOcrBackend
from core.ui import ScreenState
from main import classify_runtime_screen, return_to_port
from port_navigator import claim_daily_reward, daily_reward_claim_point


def token(text, confidence, left, top, right, bottom):
    return OcrToken(
        text=text,
        confidence=confidence,
        box=((left, top), (right, top), (right, bottom), (left, bottom)),
    )


class StaticBackend:
    def __init__(self, tokens):
        self.tokens = tokens

    def recognize(self, _image):
        return list(self.tokens)


def test_daily_reward_requires_heading_and_returns_claim_text_center():
    image = np.zeros((1000, 1600, 3), dtype=np.uint8)
    backend = StaticBackend(
        [
            token("每日奖励", 0.96, 650, 70, 950, 130),
            token("领取", 0.94, 720, 780, 880, 850),
        ]
    )

    assert daily_reward_claim_point(image, backend) == (800, 815)

    no_heading = StaticBackend([token("领取", 0.99, 720, 780, 880, 850)])
    assert daily_reward_claim_point(image, no_heading) is None

    obtained_overlay = StaticBackend(
        [
            token("已获得", 0.98, 650, 70, 950, 130),
            token("已开启2个补给箱", 0.95, 570, 720, 1030, 790),
        ]
    )
    assert daily_reward_claim_point(image, obtained_overlay) is None


def test_daily_reward_ignores_explanation_and_selects_collect_button():
    image = np.zeros((1494, 2560, 3), dtype=np.uint8)
    backend = StaticBackend(
        [
            token("每日奖励", 0.98, 1170, 25, 1390, 75),
            token("您有3天时间来领取奖励", 0.96, 1050, 185, 1500, 235),
            token("收集您的奖励", 0.94, 930, 920, 1070, 965),
        ]
    )

    assert daily_reward_claim_point(image, backend) == (1000, 942)


def test_real_daily_reward_screenshot_selects_collect_button():
    fixture = Path(__file__).parent / "fixtures" / "daily_reward_collect.png"
    image = cv2.imread(str(fixture))

    point = daily_reward_claim_point(image, RapidOcrBackend())

    assert point is not None
    assert 900 <= point[0] <= 1110
    assert 900 <= point[1] <= 980


def test_daily_reward_click_is_dispatched_only_after_ocr_confirmation():
    image = np.zeros((1000, 1600, 3), dtype=np.uint8)
    backend = StaticBackend(
        [
            token("每日登录奖励", 0.95, 620, 60, 980, 125),
            token("领取", 0.93, 720, 780, 880, 850),
        ]
    )

    with patch("port_navigator._click_local", return_value=True) as click:
        assert claim_daily_reward(1, image, backend=backend)

    click.assert_called_once_with(1, (800, 815))


def test_runtime_classifier_promotes_only_confirmed_daily_reward_page():
    image = np.zeros((1000, 1600, 3), dtype=np.uint8)
    backend = StaticBackend(
        [
            token("每日奖励", 0.96, 650, 70, 950, 130),
            token("领取", 0.94, 720, 780, 880, 850),
        ]
    )
    bot = SimpleNamespace(
        vision=SimpleNamespace(
            classify_screen=lambda _image: ScreenState.UNKNOWN,
        ),
        distance_reader=SimpleNamespace(backend=backend),
    )

    assert classify_runtime_screen(bot, image) == ScreenState.DAILY_REWARD


def test_runtime_classifier_keeps_unknown_when_reward_ocr_fails():
    image = np.zeros((1000, 1600, 3), dtype=np.uint8)
    detector = Mock(side_effect=RuntimeError("OCR unavailable"))
    bot = SimpleNamespace(
        vision=SimpleNamespace(
            classify_screen=lambda _image: ScreenState.UNKNOWN,
            is_daily_reward_page=detector,
        ),
        distance_reader=SimpleNamespace(backend=Mock()),
    )

    assert classify_runtime_screen(bot, image) == ScreenState.UNKNOWN


def test_return_to_port_claims_daily_reward_then_rechecks_port():
    image = np.zeros((100, 160, 3), dtype=np.uint8)
    bot = SimpleNamespace(
        hwnd=1,
        vision=SimpleNamespace(grab=lambda *_args, **_kwargs: image),
        gamepad=SimpleNamespace(escape=Mock()),
        distance_reader=SimpleNamespace(backend=Mock()),
    )

    with (
        patch("main.ensure_capture_foreground", return_value=True),
        patch(
            "main.classify_runtime_screen",
            side_effect=[ScreenState.DAILY_REWARD, ScreenState.PORT],
        ),
        patch("main.claim_daily_reward", return_value=True) as claim,
        patch("main.time.sleep", return_value=None),
    ):
        assert return_to_port(bot, attempts=2)

    claim.assert_called_once()
    bot.gamepad.escape.assert_not_called()
