"""A confirmed settlement closes the plan even when resource OCR fails."""

from unittest.mock import patch

from core.results import BattleRewards
from core.ui import ScreenState
from test_lifecycle_closure import GATE, drive_run


def drive_unrecognized_rewards(max_rounds):
    # A partial numeric result must not become the reported resource totals.
    settlement = (
        True,
        BattleRewards(credits=12_345, recognized=False, outcome="victory"),
        ScreenState.RESULTS,
    )
    with (
        patch("main.collect_rewards_before_navigation", return_value=settlement) as collect,
        patch("main.wait_for_battle", return_value=True) as wait_for_battle,
        patch("main.finalize_round_diagnostics") as finalize,
    ):
        exit_code, bot, battle, port_return, _claim, _quick = drive_run(
            max_rounds=max_rounds,
            use_real_stop=True,
            recover=[ScreenState.BATTLE] * max_rounds,
            battle_finished=True,
            port_return=True,
            queue_return=True,
            gates=[GATE] * 20,
            monotonic=[0.0] * 50,
        )
    updates = bot.runtime_reporter.update.call_args_list
    unrecognized = [item.kwargs for item in updates
                    if item.kwargs.get("rewards_status") == "unrecognized"]
    completed = [item.kwargs for item in updates if item.args[0] == "completed"]
    return exit_code, bot, battle, port_return, collect, wait_for_battle, finalize, unrecognized, completed


def test_one_unrecognized_settlement_reports_empty_rewards_and_completes_in_port():
    (exit_code, bot, battle, port_return, collect, wait_for_battle,
     finalize, unrecognized, completed) = drive_unrecognized_rewards(1)

    assert exit_code == 0
    assert len(unrecognized) == 1
    assert unrecognized[0]["current_round"] == unrecognized[0]["rewards_round"] == 1
    assert unrecognized[0]["completed_rounds"] == 0
    assert unrecognized[0]["last_rewards"] == {}
    assert unrecognized[0]["last_outcome"] == "victory"
    assert len(completed) == 1 and completed[0]["completed_rounds"] == 1
    battle.assert_called_once()
    collect.assert_called_once()
    port_return.assert_called_once_with(bot, attempts=5, post_battle_context=True)
    finalize.assert_called_once_with(bot, 1, outcome="victory")
    bot.queue_next_battle_mock.assert_not_called()
    wait_for_battle.assert_not_called()


def test_two_unrecognized_settlements_requeue_once_and_count_each_round_once():
    (exit_code, bot, battle, port_return, collect, wait_for_battle,
     finalize, unrecognized, completed) = drive_unrecognized_rewards(2)

    assert exit_code == 0
    assert [item["current_round"] for item in unrecognized] == [1, 2]
    assert [item["rewards_round"] for item in unrecognized] == [1, 2]
    assert [item["completed_rounds"] for item in unrecognized] == [0, 1]
    assert all(item["last_rewards"] == {} for item in unrecognized)
    assert len(completed) == 1 and completed[0]["completed_rounds"] == 2
    assert battle.call_count == collect.call_count == 2
    assert [item.args[1] for item in finalize.call_args_list] == [1, 2]
    bot.queue_next_battle_mock.assert_called_once()
    wait_for_battle.assert_called_once()
    assert wait_for_battle.call_args.kwargs["require_new_round"] is True
    port_return.assert_called_once_with(bot, attempts=5, post_battle_context=True)


def test_changed_page_after_reward_retries_is_observed_again_before_round_closure():
    calls = []

    def settlement(bot, *_args, **_kwargs):
        calls.append(True)
        bot._settlement_recovery_pending = len(calls) == 1
        return (
            len(calls) > 1, BattleRewards(),
            ScreenState.UNKNOWN if len(calls) == 1 else ScreenState.RESULTS,
        )

    with (
        patch("main.collect_rewards_before_navigation", side_effect=settlement),
        patch("main.finalize_round_diagnostics") as finalize,
    ):
        exit_code, bot, battle, port_return, _claim, _quick = drive_run(
            max_rounds=1, use_real_stop=True,
            recover=[ScreenState.BATTLE, ScreenState.RESULTS],
            battle_finished=True, port_return=True,
            gates=[GATE] * 20, monotonic=[0.0] * 50,
        )

    assert exit_code == 0 and len(calls) == 2
    battle.assert_called_once()
    # The changed page must not trigger the unconfirmed-settlement navigation
    # branch; only the final, verified plan completion returns to port.
    port_return.assert_called_once_with(bot, attempts=5, post_battle_context=True)
    finalize.assert_called_once_with(bot, 1, outcome="unknown")
    updates = bot.runtime_reporter.update.call_args_list
    recovering = [item for item in updates if item.kwargs.get("rewards_status") == "pending"]
    assert recovering and recovering[-1].kwargs["completed_rounds"] == 0
    completed = [item for item in updates if item.args[0] == "completed"]
    assert len(completed) == 1 and completed[0].kwargs["completed_rounds"] == 1
    bot.queue_next_battle_mock.assert_not_called()
