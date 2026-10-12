"""Fresh central dialog text, rather than a broad HUD colour, permits Esc."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from core.ocr import OcrToken
from core.ui import ScreenState
from main import battle_overlay_text_confirmed, classify_runtime_screen, dismiss_battle_overlay


EXIT_WORDS = ("确认", "离开战斗?", "是", "否", "继续战斗!")


def frame(marker=20):
    return np.full((100, 200, 3), marker, np.uint8)


def backend(words, confidence=.95):
    return SimpleNamespace(
        recognize=Mock(return_value=[OcrToken(word, confidence) for word in words]),
    )


def make_bot(images, words=()):
    return SimpleNamespace(
        hwnd=1,
        vision=SimpleNamespace(grab=Mock(side_effect=images)),
        gamepad=SimpleNamespace(escape=Mock()),
        distance_reader=SimpleNamespace(backend=backend(words)),
        blockage_diagnostic_root=None,
        last_analysis=SimpleNamespace(image=frame(250)),
    )


def test_stale_initial_exit_dialog_cannot_escape_fresh_battle_without_diagnostics():
    fresh = frame(20)
    bot = make_bot([fresh], EXIT_WORDS)
    with (
        patch("main.operation_paused", return_value=False),
        patch("main.ensure_capture_foreground", return_value=True),
        patch("main.classify_runtime_screen", return_value=ScreenState.BATTLE) as classify,
    ):
        assert dismiss_battle_overlay(bot, ScreenState.EXIT_CONFIRMATION) == ScreenState.BATTLE

    bot.vision.grab.assert_called_once_with(1, allow_stale=True)
    assert classify.call_args.args[1] is fresh
    bot.gamepad.escape.assert_not_called()
    bot.distance_reader.backend.recognize.assert_not_called()


def test_coarse_menu_without_central_menu_words_cannot_press_escape():
    bot = make_bot([frame()], ("Esc", "离开战斗"))
    with (
        patch("main.operation_paused", return_value=False),
        patch("main.ensure_capture_foreground", return_value=True),
        patch("main.classify_runtime_screen", return_value=ScreenState.ESCAPE_MENU),
    ):
        assert dismiss_battle_overlay(bot, ScreenState.ESCAPE_MENU) == ScreenState.ESCAPE_MENU

    bot.vision.grab.assert_called_once()
    bot.gamepad.escape.assert_not_called()


def test_confirmed_menu_returns_to_game_after_one_escape_and_a_fresh_battle_frame():
    before, after = frame(20), frame(80)
    bot = make_bot([before, after], ("菜单", "回到游戏"))
    with (
        patch("main.operation_paused", return_value=False),
        patch("main.ensure_capture_foreground", return_value=True),
        patch("main.classify_runtime_screen", side_effect=[ScreenState.ESCAPE_MENU, ScreenState.BATTLE]) as classify,
        patch("main.time.sleep"),
    ):
        assert dismiss_battle_overlay(bot, ScreenState.ESCAPE_MENU) == ScreenState.BATTLE

    bot.gamepad.escape.assert_called_once()
    assert bot.vision.grab.call_count == 2
    assert classify.call_args_list[0].args[1] is before
    assert classify.call_args_list[1].args[1] is after
    central_crop = bot.distance_reader.backend.recognize.call_args.args[0]
    assert central_crop.shape == (60, 80, 3)
    assert np.all(central_crop == 20)


def test_sunk_exit_dialog_requires_all_central_words_and_accepts_punctuation():
    image = frame()
    reader = backend(EXIT_WORDS)

    assert battle_overlay_text_confirmed(image, reader, ScreenState.EXIT_CONFIRMATION)
    central_crop = reader.recognize.call_args.args[0]
    assert central_crop.shape == (60, 80, 3)
    assert np.shares_memory(central_crop, image)


def test_sunk_exit_dialog_rejects_each_missing_required_word():
    for missing in EXIT_WORDS:
        reader = backend([word for word in EXIT_WORDS if word != missing])
        assert not battle_overlay_text_confirmed(frame(), reader, ScreenState.EXIT_CONFIRMATION), missing


def test_sunk_exit_dialog_rejects_each_low_confidence_required_word():
    for uncertain in EXIT_WORDS:
        reader = SimpleNamespace(recognize=Mock(return_value=[
            OcrToken(word, .5 if word == uncertain else .95) for word in EXIT_WORDS
        ]))
        assert not battle_overlay_text_confirmed(frame(), reader, ScreenState.EXIT_CONFIRMATION), uncertain


def test_coarse_exit_colours_with_death_hud_remain_live_battle():
    image = frame()
    reader = backend(())
    bot = SimpleNamespace(
        vision=SimpleNamespace(classify_screen=Mock(return_value=ScreenState.EXIT_CONFIRMATION)),
        distance_reader=SimpleNamespace(backend=reader),
    )
    with (
        patch("main.is_port_exit_confirmation_page", return_value=False),
        patch("main.is_early_exit_confirmation_page", return_value=False),
        patch("main.quick_death_leave_hint_visible", return_value=True) as death_hud,
    ):
        assert classify_runtime_screen(bot, image) == ScreenState.BATTLE

    death_hud.assert_called_once_with(image, reader)


def test_menu_text_ocr_failure_does_not_send_escape():
    bot = make_bot([frame()])
    bot.distance_reader.backend.recognize.side_effect = RuntimeError("OCR unavailable")
    with (
        patch("main.operation_paused", return_value=False),
        patch("main.ensure_capture_foreground", return_value=True),
        patch("main.classify_runtime_screen", return_value=ScreenState.ESCAPE_MENU),
    ):
        assert dismiss_battle_overlay(bot, ScreenState.ESCAPE_MENU) == ScreenState.ESCAPE_MENU

    bot.gamepad.escape.assert_not_called()


def test_sunk_hud_overrides_false_loading_colour_without_sending_inputs():
    image = frame()
    reader = backend(())
    bot = SimpleNamespace(
        vision=SimpleNamespace(classify_screen=lambda _: ScreenState.LOADING),
        distance_reader=SimpleNamespace(backend=reader),
    )
    with (
        patch("main.loading_start_confirmed", return_value=False),
        patch("main.quick_death_leave_hint_visible", return_value=True),
    ):
        assert classify_runtime_screen(bot, image) == ScreenState.BATTLE


def test_confirmed_new_match_start_retains_loading_even_with_old_death_hint():
    image = frame()
    bot = SimpleNamespace(
        vision=SimpleNamespace(classify_screen=lambda _: ScreenState.LOADING),
        distance_reader=SimpleNamespace(backend=backend(())),
    )
    with (
        patch("main.loading_start_confirmed", return_value=True),
        patch("main.quick_death_leave_hint_visible", return_value=True) as death_hud,
    ):
        assert classify_runtime_screen(bot, image) == ScreenState.LOADING
    death_hud.assert_not_called()
