"""Shop pages close through guarded Esc recovery without touching purchases."""

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


def shop_reader(shape=(600, 1000), *, back="返回", search="按名称搜索",
                commerce="优惠券 钱包", confidence=.99, close=True):
    height, width = shape

    def recognize(crop):
        h, w = crop.shape[:2]
        if (h, w) == (round(height * .10), round(width * .26)):
            text = back
        elif (h, w) == (round(height * .12), round(width * .86) - round(width * .38)):
            text = search + commerce
        elif h == int(height * .14) and w > width * .27:
            text = "ESC" if close else ""
        else:
            return []
        x1, x2 = (w * .85, w * .95) if text == "ESC" else (w * .05, w * .80)
        y1, y2 = h * .10, h * .25
        return [OcrToken(text, confidence, ((x1, y1), (x2, y1), (x2, y2), (x1, y2)))] if text else []

    return SimpleNamespace(recognize=Mock(side_effect=recognize))


@pytest.mark.parametrize("scale", [.5, 1, 1.5])
def test_shop_shell_is_identified_without_product_names_at_each_size(scale):
    frame = cv2.resize(container_frame(), None, fx=scale, fy=scale)
    assert port_subpage_title(frame, shop_reader(frame.shape[:2])) == "商店"


@pytest.mark.parametrize("settings", [
    {"back": ""}, {"search": ""}, {"commerce": ""}, {"confidence": .74},
])
def test_partial_shop_shell_does_not_override_runtime_state(settings):
    assert port_subpage_title(container_frame(), shop_reader(**settings)) is None


@pytest.fixture
def shop(replay):
    bot, page, port, _ = replay
    bot.distance_reader.backend = shop_reader()
    return bot, page, port


@pytest.mark.parametrize("coarse", [
    ScreenState.PORT, ScreenState.BATTLE, ScreenState.LOADING,
    ScreenState.EXIT_CONFIRMATION, ScreenState.RESULTS,
])
def test_shop_never_acquires_battle_or_exit_confirmation_actions(shop, coarse):
    bot, page, _ = shop
    bot.vision.classify_screen.side_effect = None
    bot.vision.classify_screen.return_value = coarse
    assert main.classify_runtime_screen(bot, page) == ScreenState.UNKNOWN
    assert main.classify_battle_continuity_screen(bot, page) == ScreenState.UNKNOWN


def test_shop_detail_and_shop_list_close_to_port_with_two_escape_pairs(shop, tmp_path):
    bot, page, port = shop
    bot.vision.grab.side_effect = [page, page, port]
    assert main.return_to_port(bot, attempts=3)
    assert bot.gamepad.escape.call_count == 2
    rows = [json.loads(line) for path in tmp_path.rglob("observations.jsonl")
            for line in path.read_text(encoding="utf-8").splitlines()]
    for attempt in (1, 2):
        assert {row["phase"] for row in rows
                if row["pair_id"] == f"escape_unknown:{attempt}"} == {"before", "after"}
    assert rows[-1]["state"] == "port" and rows[-1]["phase"] == "result"


def test_startup_recovers_confirmed_shop_without_waiting_for_timeout(shop):
    bot, page, port = shop
    bot.vision.grab.side_effect = [page] * 3 + [port] * 3
    image, state = main.wait_for_recognized_screen(bot, timeout=300)
    assert image is port and state == ScreenState.PORT
    bot.gamepad.escape.assert_called_once()


@pytest.mark.parametrize("guard", ["no_close_hint", "numeric_battle", "paused"])
def test_shop_identity_does_not_bypass_escape_guards(shop, guard):
    bot, page, _ = shop
    bot.vision.grab.return_value = page
    if guard == "no_close_hint":
        bot.distance_reader.backend = shop_reader(close=False)
    elif guard == "numeric_battle":
        bot.vision.has_numeric_battle_hud.return_value = True
    else:
        main.operation_paused.return_value = True
    assert not main.return_to_port(bot, attempts=1)
    bot.gamepad.escape.assert_not_called()


def test_frozen_shop_cannot_rearm_escape_budget(shop):
    bot, page, _ = shop
    bot.vision.grab.return_value = page
    for _ in range(2):
        assert not main.return_to_port(bot, attempts=3)
    assert bot.gamepad.escape.call_count == 3
