"""Offline regressions for cancellation, unread settlement and stuck overlays."""
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from core.battle_phase import BattlePhase, BattlePhaseTracker
from core.results import BattleRewards
from core.ui import ScreenState
from main import (
    collect_battle_rewards, collect_rewards_before_navigation, operation_paused,
    return_to_port, wait_for_game_window, wait_for_recognized_screen, wait_for_web_resume,
)
from test_result_lifecycle import FixedRewardReader, make_bot


@pytest.mark.parametrize("raw,confidence,recognized", [("0", .98, True), ("", .98, False), ("0", .4, False), ("10", .98, False)])
def test_zero_resource_requires_explicit_confident_ocr(raw, confidence, recognized):
    reward = BattleRewards(credits=10000, ship_xp=250, free_xp=0,
                           raw_text={"free_xp": raw}, confidence={"free_xp": confidence})
    reader = SimpleNamespace(backend=None, read=lambda _: reward)
    with patch("main.time.sleep"):
        confirmed, result, _ = collect_battle_rewards(make_bot([ScreenState.RESULTS] * 4), reader, attempts=4)
    assert confirmed
    assert result.recognized == recognized
    assert result.free_xp == 0


def test_unread_result_retry_is_bounded_and_returns_unrecognized_without_navigation():
    bot = make_bot([])
    retry = Mock()
    with patch("main.collect_battle_rewards", return_value=(True, BattleRewards(), ScreenState.RESULTS)) as collect, \
         patch("main.time.sleep"), patch("main.handle_post_battle") as navigate:
        confirmed, rewards, state = collect_rewards_before_navigation(
            bot, None, resume=lambda: True, on_retry=retry, max_retries=3,
        )
    assert collect.call_count == retry.call_count == 3
    assert confirmed and state == ScreenState.RESULTS
    assert not rewards.recognized
    assert rewards.resource_values() == {"credits": 0, "ship_xp": 0, "free_xp": 0}
    navigate.assert_not_called()


@pytest.mark.parametrize("overlay", [ScreenState.SURVEY, ScreenState.ESCAPE_MENU, ScreenState.EXIT_CONFIRMATION])
def test_known_overlay_is_dismissed_without_discarding_pending_rewards(overlay):
    success = (True, FixedRewardReader().read(None), ScreenState.RESULTS)
    readings = [(True, BattleRewards(), ScreenState.RESULTS), (False, BattleRewards(), overlay), success]
    with patch("main.collect_battle_rewards", side_effect=readings), patch("main.time.sleep"), \
         patch("main.dismiss_current_battle_survey") as survey, patch("main.dismiss_battle_overlay") as menu:
        assert collect_rewards_before_navigation(make_bot([]), None, resume=lambda: True, on_retry=Mock()) == success
    assert survey.call_count == int(overlay == ScreenState.SURVEY)
    assert menu.call_count == int(overlay != ScreenState.SURVEY)


def test_survey_ocr_failure_preserves_unknown_page():
    with patch("main.is_battle_survey_page", side_effect=RuntimeError("OCR unavailable")), patch("main.time.sleep"):
        confirmed, rewards, state = collect_battle_rewards(make_bot([ScreenState.UNKNOWN] * 2), FixedRewardReader(), attempts=2)
    assert not confirmed and not rewards.recognized
    assert state == ScreenState.UNKNOWN


def test_unknown_escape_budget_is_shared_across_recovery_calls():
    bot = make_bot([])
    bot.gamepad = SimpleNamespace(escape=Mock())
    bot.distance_reader = SimpleNamespace(backend=object())
    with patch("main.ensure_capture_foreground", return_value=True), \
         patch("main.classify_runtime_screen", return_value=ScreenState.UNKNOWN) as classify, \
         patch("main.is_battle_survey_page", return_value=False), \
         patch("main.escape_close_hint", return_value=(10, 10)), \
         patch("main.in_battle_type_selector", return_value=False), patch("main.time.sleep"):
        for _ in range(3):
            assert not return_to_port(bot, attempts=3)
        assert bot.gamepad.escape.call_count == 3
        assert bot._recovery_needs_user
        classify.return_value = ScreenState.PORT
        assert return_to_port(bot, attempts=1)
        assert bot._unknown_escape_attempts == 0
        assert not bot._recovery_needs_user


def test_normal_loading_does_not_consume_post_battle_escape_budget():
    bot = make_bot([])
    bot.gamepad = SimpleNamespace(escape=Mock())
    with patch("main.ensure_capture_foreground", return_value=True), \
         patch("main.classify_runtime_screen", return_value=ScreenState.LOADING), \
         patch("main.is_battle_survey_page", return_value=False), \
         patch("main.in_battle_type_selector", return_value=False), patch("main.time.sleep"):
        assert not return_to_port(bot, attempts=3)
    bot.gamepad.escape.assert_not_called()
    assert not getattr(bot, "_recovery_needs_user", False)


def test_post_battle_loading_ad_shares_escape_budget():
    bot = make_bot([])
    bot.gamepad = SimpleNamespace(escape=Mock())
    bot.distance_reader = SimpleNamespace(backend=object())
    with patch("main.ensure_capture_foreground", return_value=True), \
         patch("main.classify_runtime_screen", return_value=ScreenState.LOADING), \
         patch("main.is_battle_survey_page", return_value=False), \
         patch("main.escape_close_hint", return_value=(10, 10)), \
         patch("main.in_battle_type_selector", return_value=False), patch("main.time.sleep"):
        for _ in range(3):
            assert not return_to_port(bot, attempts=3, post_battle_context=True)
    assert bot.gamepad.escape.call_count == 3
    assert bot._recovery_needs_user


def test_stop_is_observed_even_when_web_pause_is_clear():
    limits = SimpleNamespace(stop_requested=lambda: True, pause_requested=Mock(return_value=False))
    assert not wait_for_web_resume(limits, Mock(), make_bot([])).allowed
    limits.pause_requested.assert_not_called()


def test_stop_interrupts_startup_and_inner_actions_before_capture():
    bot = make_bot([])
    bot.run_limits = SimpleNamespace(stop_requested=lambda: True)
    bot.vision.grab = Mock()
    with patch("main.find_game_window") as find, patch("main.ensure_capture_foreground") as focus:
        assert wait_for_game_window(should_stop=bot.run_limits.stop_requested) is None
        assert wait_for_recognized_screen(bot) == (None, ScreenState.UNKNOWN)
        assert operation_paused(bot)
        assert not return_to_port(bot)
    bot.vision.grab.assert_not_called()
    find.assert_not_called()
    focus.assert_not_called()


def test_last_minute_takeover_requires_live_motion_or_damage_evidence():
    tracker = BattlePhaseTracker()
    assert tracker.observe(ScreenState.BATTLE, 59, 0, active_evidence=True) == BattlePhase.UNKNOWN
    assert tracker.observe(ScreenState.BATTLE, 58, 1, active_evidence=True) == BattlePhase.ACTIVE
    tracker.observe(ScreenState.LOADING, None, 2)
    assert tracker.observe(ScreenState.BATTLE, 30, 3) == BattlePhase.PREPARING
    assert tracker.observe(ScreenState.BATTLE, 29, 4) == BattlePhase.PREPARING
