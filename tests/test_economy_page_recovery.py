"""Economy pages use the same guarded Esc-to-port recovery as containers."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import cv2
import pytest

import main
from core.ocr import OcrToken
from core.overlay_hints import port_subpage_title
from core.ui import ScreenState
from test_container_page_recovery import container_frame, replay


def economy_reader(shape=(600, 1000), *, back="返回", title="经济加成",
                   confidence=.99, close=True):
    height, width = shape

    def recognize(crop):
        h, w = crop.shape[:2]
        if (h, w) == (round(height * .10), round(width * .26)):
            text = back
        elif (h, w) == (round(height * .12), round(width * .65) - round(width * .35)):
            text = title
        elif h == int(height * .14) and w > width * .27:
            text = "ESC" if close else ""
        else:
            return []
        x1, x2 = (w * .85, w * .95) if text == "ESC" else (w * .05, w * .40)
        y1, y2 = h * .10, h * .25
        return [OcrToken(text, confidence, ((x1, y1), (x2, y1), (x2, y2), (x1, y2)))] if text else []

    return SimpleNamespace(recognize=Mock(side_effect=recognize))


@pytest.mark.parametrize("scale", [.5, 1, 1.5])
def test_economy_page_requires_its_back_link_and_centered_title_at_each_size(scale):
    frame = cv2.resize(container_frame(), None, fx=scale, fy=scale)
    assert port_subpage_title(frame, economy_reader(frame.shape[:2])) == "经济加成"


@pytest.mark.parametrize("settings", [
    {"back": ""}, {"title": "战斗任务"}, {"confidence": .74},
])
def test_incomplete_economy_identity_is_not_a_known_port_subpage(settings):
    assert port_subpage_title(container_frame(), economy_reader(**settings)) is None


@pytest.fixture
def economy(replay):
    bot, page, port, _ = replay
    bot.distance_reader.backend = economy_reader()
    return bot, page, port


@pytest.mark.parametrize("coarse", [ScreenState.BATTLE, ScreenState.LOADING, ScreenState.PORT])
def test_economy_identity_prevents_all_coarse_navigation_conflicts(economy, coarse):
    bot, page, _ = economy
    bot.vision.classify_screen.return_value = coarse
    bot.vision.classify_screen.side_effect = None
    assert main.classify_runtime_screen(bot, page) == ScreenState.UNKNOWN
    assert main.classify_battle_continuity_screen(bot, page) == ScreenState.UNKNOWN


def test_economy_returns_to_port_with_one_escape_and_before_after_evidence(economy, tmp_path):
    bot, page, port = economy
    bot.vision.grab.side_effect = [page, port]

    assert main.return_to_port(bot, attempts=2)

    bot.gamepad.escape.assert_called_once()
    rows = [json.loads(line) for path in tmp_path.rglob("observations.jsonl")
            for line in path.read_text(encoding="utf-8").splitlines()]
    assert {row["phase"] for row in rows if row["pair_id"] == "escape_unknown:1"} == {"before", "after"}
    assert rows[-1]["state"] == "port" and rows[-1]["phase"] == "result"


def test_startup_closes_the_confirmed_economy_page_without_waiting_for_timeout(economy):
    bot, page, port = economy
    bot.vision.grab.side_effect = [page] * 3 + [port] * 3
    image, state = main.wait_for_recognized_screen(bot, timeout=300)
    assert image is port and state == ScreenState.PORT
    bot.gamepad.escape.assert_called_once()
    assert bot.vision.grab.call_count == 6


@pytest.mark.parametrize("guard", ["no_close_hint", "numeric_battle", "paused"])
def test_economy_page_cannot_bypass_escape_guards(economy, guard):
    bot, page, _ = economy
    bot.vision.grab.return_value = page
    if guard == "no_close_hint":
        bot.distance_reader.backend = economy_reader(close=False)
    elif guard == "numeric_battle":
        bot.vision.has_numeric_battle_hud.return_value = True
    else:
        main.operation_paused.return_value = True
    assert not main.return_to_port(bot, attempts=1)
    bot.gamepad.escape.assert_not_called()
