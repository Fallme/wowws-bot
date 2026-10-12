"""Unknown pages authorize Esc only from fresh, explicitly closable evidence."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest

from core.ui import ScreenState
from main import dismiss_unknown_overlay, return_to_port


@pytest.fixture
def bot(tmp_path):
    frames = [np.full((90, 160, 3), value, np.uint8) for value in (20, 40, 60, 80)]
    return SimpleNamespace(
        hwnd=1,
        vision=SimpleNamespace(
            grab=Mock(side_effect=frames),
            # The background HUD can remain behind a real closable overlay.
            _has_battle_hud=Mock(return_value=True),
            has_numeric_battle_hud=Mock(return_value=False),
            _has_loading_start_action=Mock(return_value=False),
        ),
        gamepad=SimpleNamespace(escape=Mock()),
        distance_reader=SimpleNamespace(backend=object()),
        blockage_diagnostic_root=tmp_path,
        frames=frames,
        _unknown_overlay_escape_attempts=0,
        _unknown_escape_attempts=0,
    )


def recover(bot, path, *, state=None):
    if state is None:
        state = ScreenState.LOADING if path == "post_battle_loading" else ScreenState.UNKNOWN
    with (
        patch("main.operation_paused", return_value=False),
        patch("main.ensure_capture_foreground", return_value=True),
        patch("main.classify_runtime_screen", return_value=state),
        patch("main.is_battle_survey_page", return_value=False),
        patch("main.in_battle_type_selector", return_value=path == "selector"),
        patch("main.time.sleep"),
    ):
        if path == "overlay":
            return dismiss_unknown_overlay(bot, attempts=3)
        return return_to_port(bot, attempts=2, post_battle_context=path == "post_battle_loading")


def escape_budget(bot, path):
    return bot._unknown_overlay_escape_attempts if path == "overlay" else bot._unknown_escape_attempts


@pytest.mark.parametrize("path", ["overlay", "unknown_port", "selector", "post_battle_loading"])
def test_unknown_page_without_close_hint_does_not_press_escape_or_consume_budget(bot, path):
    bot._unknown_overlay_escape_attempts = bot._unknown_escape_attempts = 2
    with patch("main.escape_close_hint", return_value=None):
        recover(bot, path)

    bot.gamepad.escape.assert_not_called()
    assert escape_budget(bot, path) == 2
    assert not getattr(bot, "_recovery_needs_user", False)


@pytest.mark.parametrize("path", ["overlay", "unknown_port"])
def test_corner_hint_does_not_authorize_escape_over_fresh_numeric_live_hud(bot, path):
    bot.vision.has_numeric_battle_hud.return_value = True
    with patch("main.escape_close_hint", return_value=(10, 10)):
        recover(bot, path)

    bot.gamepad.escape.assert_not_called()
    assert escape_budget(bot, path) == 0
    assert bot.vision.has_numeric_battle_hud.called
    args = bot.vision.has_numeric_battle_hud.call_args_list[0].args
    assert args[0] is bot.frames[0]
    assert args[1] is bot.distance_reader.backend


@pytest.mark.parametrize("path", ["overlay", "unknown_port"])
def test_hint_must_still_exist_on_fresh_frame_before_a_second_escape(bot, path):
    with patch("main.escape_close_hint", side_effect=[(10, 10), None, None, None]) as hints:
        recover(bot, path)

    bot.gamepad.escape.assert_called_once()
    assert escape_budget(bot, path) == 1
    assert hints.call_count >= 2
    assert hints.call_args_list[0].args[0] is bot.frames[0]
    assert hints.call_args_list[1].args[0] is bot.frames[1]
    # A broad HUD-shaped background alone must not reject a closable overlay.
    assert bot.vision._has_battle_hud.return_value is True
    assert bot.vision.has_numeric_battle_hud.return_value is False


@pytest.mark.parametrize("path", ["overlay", "unknown_port"])
def test_close_hint_ocr_failure_preserves_unknown_page_without_escape(bot, path):
    with patch("main.escape_close_hint", side_effect=RuntimeError("OCR unavailable")):
        recover(bot, path)

    bot.gamepad.escape.assert_not_called()
    assert escape_budget(bot, path) == 0


def test_post_battle_context_cannot_escape_a_live_loading_start_page(bot):
    bot.vision._has_loading_start_action.return_value = True
    bot.distance_reader.backend = SimpleNamespace(
        recognize=Mock(return_value=[SimpleNamespace(text="开始战斗")]),
    )
    with patch("main.escape_close_hint", return_value=(10, 10)):
        recover(bot, "post_battle_loading")

    bot.gamepad.escape.assert_not_called()
    assert bot._unknown_escape_attempts == 0


@pytest.mark.parametrize("path", ["overlay", "unknown_port"])
def test_known_live_battle_returns_before_close_hint_recovery(bot, path):
    with patch("main.escape_close_hint", return_value=(10, 10)) as hints:
        recover(bot, path, state=ScreenState.BATTLE)

    bot.gamepad.escape.assert_not_called()
    hints.assert_not_called()
