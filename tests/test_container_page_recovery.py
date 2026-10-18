"""Container-deck artwork must close to port instead of acquiring battle control."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import cv2
import numpy as np
import pytest

import main
from core.ocr import OcrToken
from core.overlay_hints import is_port_container_page
from core.ui import ScreenState


def container_frame():
    frame = np.full((600, 1000, 3), 90, np.uint8)
    cv2.rectangle(frame, (950, 12), (981, 30), (215, 215, 215), -1)
    return frame


def reader(*, back="返回港口", title="补给箱", confidence=.99, close=True):
    def recognize(crop):
        height, width = crop.shape[:2]
        if height / width < .25:
            text = back
        elif height / width > .36:
            text = title
        else:
            text = "ESC" if close and width > 270 else ""
        x1, x2 = ((width * .85, width * .95) if text == "ESC"
                  else (width * .05, width * .40))
        y1, y2 = height * .10, height * .25
        return [OcrToken(text, confidence, ((x1, y1), (x2, y1), (x2, y2), (x1, y2)))] if text else []
    return SimpleNamespace(recognize=Mock(side_effect=recognize))


@pytest.mark.parametrize("scale", [.5, 1, 1.5])
def test_container_headings_are_confirmed_at_different_frame_sizes(scale):
    frame = cv2.resize(container_frame(), None, fx=scale, fy=scale)
    assert is_port_container_page(frame, reader())


@pytest.mark.parametrize("settings", [
    {"back": "加入战斗"}, {"title": "每日奖励"}, {"confidence": .74},
])
def test_one_heading_or_low_confidence_cannot_override_battle(settings):
    assert not is_port_container_page(container_frame(), reader(**settings))


def test_no_keycap_means_no_additional_ocr_on_the_battle_path():
    backend = reader()
    assert not is_port_container_page(np.full((600, 1000, 3), 90, np.uint8), backend)
    backend.recognize.assert_not_called()


@pytest.fixture
def replay(tmp_path, monkeypatch):
    page = container_frame()
    port = np.full_like(page, 110)
    reward = np.full_like(page, 30)
    bot = SimpleNamespace(
        hwnd=1, blockage_diagnostic_root=tmp_path,
        vision=SimpleNamespace(
            grab=Mock(),
            classify_screen=Mock(side_effect=lambda frame:
                ScreenState.BATTLE if frame is page else
                ScreenState.PORT if frame is port else ScreenState.UNKNOWN),
            has_numeric_battle_hud=Mock(return_value=False),
            _has_loading_start_action=Mock(return_value=False),
        ),
        gamepad=SimpleNamespace(escape=Mock()),
        distance_reader=SimpleNamespace(backend=reader()),
    )
    for name in ("operation_paused", "operation_stopped", "is_battle_survey_page",
                 "in_battle_type_selector", "battle_overlay_text_confirmed",
                 "is_port_exit_confirmation_page", "is_daily_reward_page"):
        monkeypatch.setattr(main, name, Mock(return_value=False))
    monkeypatch.setattr(main, "ensure_capture_foreground", Mock(return_value=True))
    monkeypatch.setattr(main.time, "sleep", Mock())
    return bot, page, port, reward


def test_container_subpage_stays_unknown_during_battle_continuity(replay):
    bot, page, _, _ = replay
    assert main.classify_runtime_screen(bot, page) == ScreenState.UNKNOWN
    assert main.classify_battle_continuity_screen(bot, page) == ScreenState.UNKNOWN


def test_reward_then_container_then_port_has_two_escape_pairs(replay, tmp_path):
    bot, page, port, reward = replay
    bot.vision.grab.side_effect = [reward, page, port]

    assert main.return_to_port(bot, attempts=3)

    assert bot.gamepad.escape.call_count == 2
    assert bot.vision.grab.call_count == 3
    rows = [json.loads(line) for path in tmp_path.rglob("observations.jsonl")
            for line in path.read_text(encoding="utf-8").splitlines()]
    for attempt in (1, 2):
        pair = [row for row in rows if row["pair_id"] == f"escape_unknown:{attempt}"]
        assert {row["phase"] for row in pair} == {"before", "after"}
    assert rows[-1]["phase"] == "result" and rows[-1]["state"] == "port"


@pytest.mark.parametrize("guard", ["no_close_hint", "numeric_battle", "paused"])
def test_container_identity_does_not_bypass_escape_guards(replay, guard):
    bot, page, _, _ = replay
    bot.vision.grab.return_value = page
    if guard == "no_close_hint":
        bot.distance_reader.backend = reader(close=False)
    elif guard == "numeric_battle":
        bot.vision.has_numeric_battle_hud.return_value = True
    else:
        main.operation_paused.return_value = True

    assert not main.return_to_port(bot, attempts=1)
    bot.gamepad.escape.assert_not_called()


def test_frozen_container_page_does_not_rearm_the_escape_budget(replay):
    bot, page, _, _ = replay
    bot.vision.grab.return_value = page
    for _ in range(2):
        assert not main.return_to_port(bot, attempts=3)
    assert bot.gamepad.escape.call_count == 3


def test_startup_closes_confirmed_container_page_without_waiting_for_timeout(replay):
    bot, page, port, _ = replay
    bot.vision.grab.side_effect = [page] * 3 + [port] * 3

    image, state = main.wait_for_recognized_screen(bot, timeout=300)

    assert image is port and state == ScreenState.PORT
    bot.gamepad.escape.assert_called_once()
    assert bot.vision.grab.call_count == 6
