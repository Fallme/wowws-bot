"""An unread reward can close a confirmed settlement without fabricating income."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from core.results import BattleRewards
from core.ui import ScreenState
from main import collect_rewards_before_navigation


def bot():
    return SimpleNamespace(last_analysis=SimpleNamespace(image=object()),
                           _settlement_recovery_pending=False)


def partial_rewards():
    return BattleRewards(
        credits=42_105, ship_xp=700, free_xp=0, recognized=False,
        provider="CUDAExecutionProvider", raw_text={"credits": "42 105", "ship_xp": "700", "free_xp": "?"},
        confidence={"credits": .97, "ship_xp": .91, "free_xp": .2}, outcome="victory",
    )


def assert_no_income(rewards):
    assert not rewards.recognized
    assert rewards.resource_values() == {"credits": 0, "ship_xp": 0, "free_xp": 0}


def test_exhausted_confirmed_settlement_has_no_partial_income_but_keeps_diagnostics():
    state = ScreenState.RESULTS
    active_bot = bot()
    partial = partial_rewards()
    retry = Mock()
    with (
        patch("main.collect_battle_rewards", return_value=(True, partial, state)) as collect,
        patch("main.operation_paused", return_value=False),
        patch("main.time.sleep"),
        patch("main.handle_post_battle") as navigate,
    ):
        confirmed, rewards, observed = collect_rewards_before_navigation(
            active_bot, None, resume=lambda: True, on_retry=retry,
        )
    assert confirmed and observed == state
    assert_no_income(rewards)
    assert rewards.provider == partial.provider
    assert rewards.raw_text == partial.raw_text
    assert rewards.confidence == partial.confidence
    assert rewards.outcome == "victory"
    assert not active_bot._settlement_recovery_pending
    assert collect.call_count == retry.call_count == 3
    assert all(call.kwargs["attempts"] == 6 for call in collect.call_args_list)
    navigate.assert_not_called()


@pytest.mark.parametrize("state", [ScreenState.UNKNOWN, ScreenState.LOADING, ScreenState.PORT, ScreenState.RESULTS])
def test_old_confirmed_pending_result_cannot_authorize_an_unconfirmed_latest_scene(state):
    active_bot = bot()
    retry = Mock()
    partial = partial_rewards()
    readings = [(True, partial, ScreenState.RESULTS), (True, partial, ScreenState.RESULTS),
                (False, partial, state)]
    with (
        patch("main.collect_battle_rewards", side_effect=readings),
        patch("main.operation_paused", return_value=False),
        patch("main.time.sleep"),
        patch("main.return_to_port") as port,
        patch("main.handle_post_battle") as navigate,
    ):
        confirmed, rewards, observed = collect_rewards_before_navigation(
            active_bot, None, resume=lambda: True, on_retry=retry,
        )
    assert not confirmed and observed == state
    assert_no_income(rewards)
    assert active_bot._settlement_recovery_pending
    port.assert_not_called()
    navigate.assert_not_called()


@pytest.mark.parametrize("state", [ScreenState.UNKNOWN, ScreenState.LOADING, ScreenState.PORT])
def test_confirmation_flag_without_a_current_reward_surface_is_insufficient(state):
    active_bot = bot()
    with (
        patch("main.collect_battle_rewards", return_value=(True, partial_rewards(), state)),
        patch("main.operation_paused", return_value=False),
        patch("main.time.sleep"),
    ):
        confirmed, rewards, observed = collect_rewards_before_navigation(
            active_bot, None, resume=lambda: True, on_retry=Mock(), max_retries=1,
        )
    assert not confirmed and observed == state
    assert_no_income(rewards)
    assert active_bot._settlement_recovery_pending


def test_consensus_before_budget_preserves_real_values_and_clears_recovery_pending():
    active_bot = bot()
    active_bot._settlement_recovery_pending = True
    recognized = BattleRewards(credits=42_105, ship_xp=700, free_xp=47, recognized=True, outcome="victory")
    success = (True, recognized, ScreenState.RESULTS)
    retry = Mock()
    readings = [(True, partial_rewards(), ScreenState.RESULTS),
                (False, BattleRewards(), ScreenState.UNKNOWN), success]
    with (
        patch("main.collect_battle_rewards", side_effect=readings) as collect,
        patch("main.operation_paused", return_value=False),
        patch("main.time.sleep"),
    ):
        result = collect_rewards_before_navigation(active_bot, None, resume=lambda: True, on_retry=retry)
    assert result == success
    assert collect.call_count == 3
    assert retry.call_count == 2
    assert not active_bot._settlement_recovery_pending


def test_new_live_battle_preserves_the_previously_confirmed_settlement_boundary():
    active_bot = bot()
    active_bot._settlement_recovery_pending = True
    live_battle = (False, BattleRewards(), ScreenState.BATTLE)
    with (
        patch("main.collect_battle_rewards", side_effect=[(True, partial_rewards(), ScreenState.RESULTS), live_battle]) as collect,
        patch("main.operation_paused", return_value=False),
        patch("main.time.sleep"),
    ):
        result = collect_rewards_before_navigation(active_bot, None, resume=lambda: True, on_retry=Mock())
    assert result == live_battle
    assert collect.call_count == 2
    assert active_bot._settlement_recovery_pending


def test_live_battle_without_a_confirmed_settlement_has_no_pending_round_boundary():
    active_bot = bot()
    with patch("main.collect_battle_rewards", return_value=(False, BattleRewards(), ScreenState.BATTLE)):
        result = collect_rewards_before_navigation(active_bot, None, resume=lambda: True, on_retry=Mock())
    assert result[0] is False and result[2] == ScreenState.BATTLE
    assert not active_bot._settlement_recovery_pending


def test_stop_gate_returns_none_without_navigating_or_exhausting_budget():
    retry = Mock()
    with (
        patch("main.collect_battle_rewards", return_value=(True, partial_rewards(), ScreenState.RESULTS)) as collect,
        patch("main.operation_paused", return_value=False),
        patch("main.time.sleep"),
        patch("main.handle_post_battle") as navigate,
    ):
        result = collect_rewards_before_navigation(bot(), None, resume=Mock(side_effect=[True, False]), on_retry=retry)
    assert result is None
    collect.assert_called_once()
    retry.assert_called_once()
    navigate.assert_not_called()


def test_takeover_during_capture_does_not_consume_the_only_retry():
    active_bot = bot()
    paused = [False]
    calls = [0]

    def collect(*_args, **_kwargs):
        calls[0] += 1
        if calls[0] == 1:
            paused[0] = True
            return False, BattleRewards(), ScreenState.UNKNOWN
        return True, partial_rewards(), ScreenState.RESULTS

    def resume():
        paused[0] = False
        return True

    retry = Mock()
    with (
        patch("main.collect_battle_rewards", side_effect=collect) as collector,
        patch("main.operation_paused", side_effect=lambda _bot: paused[0]),
        patch("main.time.sleep"),
    ):
        confirmed, rewards, observed = collect_rewards_before_navigation(
            active_bot, None, resume=resume, on_retry=retry, max_retries=1, attempts_per_retry=4,
        )
    assert confirmed and observed == ScreenState.RESULTS
    assert_no_income(rewards)
    assert collector.call_count == 2
    assert retry.call_count == 1
    assert all(call.kwargs["attempts"] == 4 for call in collector.call_args_list)
