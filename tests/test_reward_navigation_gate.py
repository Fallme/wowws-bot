from unittest.mock import Mock, patch

from core.results import BattleRewards
from core.ui import ScreenState
from main import collect_battle_rewards, collect_rewards_before_navigation
from test_result_lifecycle import FixedRewardReader, make_bot


def test_battle_invalidates_previously_confirmed_result_frames():
    with patch("main.time.sleep"):
        confirmed, rewards, state = collect_battle_rewards(
            make_bot([ScreenState.RESULTS] * 2 + [ScreenState.BATTLE]),
            FixedRewardReader(), attempts=3,
        )
    assert not confirmed
    assert not rewards.recognized
    assert state == ScreenState.BATTLE


def test_single_numeric_read_is_not_consensus():
    with patch("main.time.sleep"):
        confirmed, rewards, _ = collect_battle_rewards(
            make_bot([ScreenState.RESULTS] * 2), FixedRewardReader(), attempts=2,
        )
    assert confirmed
    assert not rewards.recognized


def test_failed_ocr_and_unknown_page_hold_until_recognized():
    failed = (True, BattleRewards(), ScreenState.RESULTS)
    unknown = (False, BattleRewards(), ScreenState.UNKNOWN)
    success = (True, FixedRewardReader().read(None), ScreenState.RESULTS)
    retry = Mock()
    bot = make_bot([])
    with patch("main.collect_battle_rewards", side_effect=[failed, unknown, success]), patch("main.time.sleep"):
        result = collect_rewards_before_navigation(bot, None, resume=lambda: True, on_retry=retry)
    assert result == success
    assert retry.call_count == 2
    assert bot.last_analysis is None


def test_user_stop_leaves_unread_settlement_without_navigation():
    with patch("main.collect_battle_rewards", return_value=(True, BattleRewards(), ScreenState.RESULTS)), patch("main.time.sleep"):
        result = collect_rewards_before_navigation(
            make_bot([]), None, resume=Mock(side_effect=[True, False]), on_retry=Mock(),
        )
    assert result is None


def test_retry_returns_to_same_live_battle():
    battle = (False, BattleRewards(), ScreenState.BATTLE)
    with patch("main.collect_battle_rewards", side_effect=[(True, BattleRewards(), ScreenState.RESULTS), battle]), patch("main.time.sleep"):
        result = collect_rewards_before_navigation(make_bot([]), None, resume=lambda: True, on_retry=Mock())
    assert result == battle
