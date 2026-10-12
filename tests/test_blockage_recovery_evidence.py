"""Offline recovery regressions for the reported overlays and missing evidence."""

import json
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest

from core.ui import ScreenState
from main import dismiss_unknown_overlay, recover_current_scene, return_to_port


def make_bot(root=None):
    return SimpleNamespace(
        hwnd=1,
        vision=SimpleNamespace(grab=Mock(return_value=np.zeros((90, 160, 3), np.uint8))),
        gamepad=SimpleNamespace(escape=Mock()),
        blockage_diagnostic_root=root,
        distance_reader=SimpleNamespace(backend=object()),
    )


def observations(root):
    files = list(root.rglob("observations.jsonl"))
    assert files
    return [json.loads(line) for path in files for line in path.read_text(encoding="utf-8").splitlines()]


def test_last_allowed_port_escape_is_observed_and_has_before_after_evidence(tmp_path):
    bot = make_bot(tmp_path)
    with (
        patch("main.operation_paused", return_value=False),
        patch("main.ensure_capture_foreground", return_value=True),
        patch("main.classify_runtime_screen", side_effect=[ScreenState.UNKNOWN] * 3 + [ScreenState.PORT]),
        patch("main.is_battle_survey_page", return_value=False),
        patch("main.in_battle_type_selector", return_value=False),
        patch("main.escape_close_hint", return_value=(10, 10)),
        patch("main.time.sleep"),
    ):
        assert return_to_port(bot, attempts=3)
    assert bot.gamepad.escape.call_count == 3
    assert bot._unknown_escape_attempts == 0
    assert not bot._recovery_needs_user
    rows = observations(tmp_path)
    for attempt in (1, 2, 3):
        pair = [row for row in rows if row["pair_id"] == f"escape_unknown:{attempt}"]
        assert {row["phase"] for row in pair} == {"before", "after"}
    assert rows[-1]["phase"] == "result" and rows[-1]["state"] == "port"


def test_active_unknown_overlay_esc_restores_scene_without_returning_to_port(tmp_path):
    bot = make_bot(tmp_path)
    with (
        patch("main.operation_paused", return_value=False),
        patch("main.ensure_capture_foreground", return_value=True),
        patch("main.classify_runtime_screen", side_effect=[ScreenState.UNKNOWN, ScreenState.BATTLE]),
        patch("main.escape_close_hint", return_value=(10, 10)),
        patch("main.time.sleep"),
        patch("main.return_to_port") as navigate,
    ):
        assert dismiss_unknown_overlay(bot) == ScreenState.BATTLE
    bot.gamepad.escape.assert_called_once()
    navigate.assert_not_called()
    assert {row["phase"] for row in observations(tmp_path)} >= {"before", "after", "result"}


@pytest.mark.parametrize("state", [ScreenState.PORT, ScreenState.BATTLE, ScreenState.RESULTS, ScreenState.LOADING])
def test_overlay_fallback_preserves_recognized_scenes(state):
    bot = make_bot()
    with (
        patch("main.operation_paused", return_value=False),
        patch("main.ensure_capture_foreground", return_value=True),
        patch("main.classify_runtime_screen", return_value=state),
    ):
        assert dismiss_unknown_overlay(bot) == state
    bot.gamepad.escape.assert_not_called()


def test_unknown_overlay_budget_is_shared_over_repeated_calls(tmp_path):
    bot = make_bot(tmp_path)
    with (
        patch("main.operation_paused", return_value=False),
        patch("main.ensure_capture_foreground", return_value=True),
        patch("main.classify_runtime_screen", return_value=ScreenState.UNKNOWN),
        patch("main.escape_close_hint", return_value=(10, 10)),
        patch("main.time.sleep"),
    ):
        for _ in range(4):
            assert dismiss_unknown_overlay(bot, attempts=2) == ScreenState.UNKNOWN
    assert bot.gamepad.escape.call_count == 3
    assert observations(tmp_path)[-1]["action"] == "escape_budget_exhausted"


def test_pause_after_escape_prevents_after_capture_and_more_input(tmp_path):
    bot = make_bot(tmp_path)
    paused = [False]
    bot.gamepad.escape.side_effect = lambda: paused.__setitem__(0, True)
    with (
        patch("main.operation_paused", side_effect=lambda _: paused[0]),
        patch("main.ensure_capture_foreground", return_value=True),
        patch("main.classify_runtime_screen", return_value=ScreenState.UNKNOWN),
        patch("main.escape_close_hint", return_value=(10, 10)),
        patch("main.time.sleep"),
    ):
        assert dismiss_unknown_overlay(bot) == ScreenState.UNKNOWN
    bot.vision.grab.assert_called_once()
    bot.gamepad.escape.assert_called_once()
    assert not any(row["phase"] == "after" for row in observations(tmp_path))


def test_post_battle_join_text_corrects_port_frame_misread_as_battle(tmp_path):
    bot = make_bot(tmp_path)
    with (
        patch("main.operation_paused", return_value=False),
        patch("main.ensure_capture_foreground", return_value=True),
        patch("main.classify_runtime_screen", return_value=ScreenState.BATTLE),
        patch("main.port_battle_action_point", return_value=(80, 5)),
        patch("main.is_battle_survey_page", return_value=False),
    ):
        assert return_to_port(bot, attempts=1, post_battle_context=True)
    bot.gamepad.escape.assert_not_called()
    assert observations(tmp_path)[-1]["state"] == "port"


def test_post_battle_real_hud_without_join_text_never_receives_escape(tmp_path):
    bot = make_bot(tmp_path)
    with (
        patch("main.operation_paused", return_value=False),
        patch("main.ensure_capture_foreground", return_value=True),
        patch("main.classify_runtime_screen", return_value=ScreenState.BATTLE),
        patch("main.port_battle_action_point", return_value=None),
        patch("main.is_battle_survey_page", return_value=False),
    ):
        assert not return_to_port(bot, attempts=1, post_battle_context=True)
    bot.gamepad.escape.assert_not_called()
    assert observations(tmp_path)[-1]["action"] == "live_battle"


def test_scene_recovery_keeps_context_and_final_recovered_frame(tmp_path):
    bot = make_bot(tmp_path)
    with (
        patch("main.operation_paused", return_value=False),
        patch("main.ensure_capture_foreground", return_value=True),
        patch("main.classify_runtime_screen", side_effect=[ScreenState.BATTLE] * 2 + [ScreenState.UNKNOWN] * 2 + [ScreenState.RESULTS] * 2),
        patch("main.is_battle_survey_page", return_value=False),
        patch("main.time.sleep"),
    ):
        assert recover_current_scene(bot, attempts=2) == ScreenState.BATTLE
        assert recover_current_scene(bot, attempts=2) == ScreenState.UNKNOWN
        assert recover_current_scene(bot, attempts=2) == ScreenState.RESULTS
    rows = observations(tmp_path)
    assert rows[0]["phase"] == "context" and rows[0]["state"] == "battle"
    assert rows[-1]["phase"] == "result" and rows[-1]["state"] == "results"
    bot.gamepad.escape.assert_not_called()
