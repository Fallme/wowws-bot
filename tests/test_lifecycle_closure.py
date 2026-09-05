"""Closed-loop regression tests for the main lifecycle.

These tests drive main.run() with a fully stubbed environment to prove that
the failure counters introduced for loop closure actually terminate or
recover instead of spinning forever:

- battle_timeout twice without a confirmed port return raises (task ends
  with a controlled failure instead of looping the same stuck match);
- quick-battle closure failing three times raises the same way;
- a daily-reward page that cannot be claimed three times is force-closed
  with Esc and the plan keeps running.
"""

from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from core.ui import ScreenState
from main import ResumeGateResult, run
from runtime_control import RunLimits

IMAGE = np.zeros((90, 160, 3), dtype=np.uint8)
GATE = ResumeGateResult(allowed=True, resumed=False)
SHIP_CONFIG = {
    "name": "napoli",
    "display_name": "Napoli",
    "secondary": {"range": 8.0},
}


def drive_run(
    *,
    quick_battle=False,
    recover,
    battle_finished="battle_timeout",
    port_return=False,
    claim_reward=False,
    quick_closure=ScreenState.BATTLE,
    gates,
    monotonic,
):
    """Run main.run() with stubs, returning (exit_code, fake_bot, mocks)."""
    fake_bot = SimpleNamespace(
        hwnd=1,
        vision=SimpleNamespace(
            screen_capture=SimpleNamespace(last_backend="print_window"),
        ),
        gamepad=SimpleNamespace(escape=Mock(), confirm=Mock()),
        intervention=None,
        distance_reader=SimpleNamespace(backend=Mock()),
    )
    limits = RunLimits(
        max_rounds=999,
        duration_minutes=99999,
        quick_battle=quick_battle,
    )
    with ExitStack() as stack:
        stack.enter_context(patch("main.configure_dpi_awareness"))
        stack.enter_context(patch("main.configure_logging"))
        stack.enter_context(patch("main.ship_key_from_env", return_value="napoli"))
        stack.enter_context(patch("main.load_ship_config", return_value=SHIP_CONFIG))
        stack.enter_context(patch("main.RunLimits.from_env", return_value=limits))
        stack.enter_context(patch("main.RuntimeReporter", return_value=Mock()))
        stack.enter_context(
            patch(
                "main.wait_for_game_window",
                return_value=(1, "World of Warships", (0, 0, 10, 10)),
            )
        )
        stack.enter_context(patch("main.BattleBot", return_value=fake_bot))
        stack.enter_context(patch("main.set_interaction_pause_guard"))
        stack.enter_context(patch("main.set_automation_input_observer"))
        stack.enter_context(patch("main.ResultRewardReader", return_value=Mock()))
        stack.enter_context(patch("main.ensure_bound_game_foreground", return_value=True))
        stack.enter_context(
            patch(
                "main.wait_for_recognized_screen",
                return_value=(IMAGE, ScreenState.PORT),
            )
        )
        stack.enter_context(
            patch(
                "main.automatic_input_preflight",
                return_value=SimpleNamespace(reason="ok"),
            )
        )
        stack.enter_context(patch("main.wait_for_web_resume", side_effect=gates))
        stack.enter_context(patch("main.refresh_game_window", return_value=True))
        stack.enter_context(patch("main.recover_current_scene", side_effect=recover))
        run_battle = stack.enter_context(
            patch("main.run_battle", return_value=battle_finished)
        )
        port_return_mock = stack.enter_context(
            patch("main.return_to_port", return_value=port_return)
        )
        claim = stack.enter_context(
            patch("main.claim_daily_reward", return_value=claim_reward)
        )
        quick_closure_mock = stack.enter_context(
            patch(
                "main.force_quick_battle_return_to_port",
                return_value=quick_closure,
            )
        )
        stack.enter_context(patch("main.prepare_battle", return_value=False))
        stack.enter_context(patch("main.operation_paused", return_value=False))
        stack.enter_context(patch("main.lifecycle_stop_requested", return_value=False))
        stack.enter_context(patch("main.shutdown_bot"))
        stack.enter_context(patch("main.time.monotonic", side_effect=monotonic))
        stack.enter_context(patch("main.time.sleep", return_value=None))
        exit_code = run()
    return exit_code, fake_bot, run_battle, port_return_mock, claim, quick_closure_mock


def test_repeated_battle_timeout_without_port_return_terminates_task():
    """Two watchdog timeouts with an unconfirmed port return must raise a
    controlled failure instead of looping the same stuck match forever."""
    exit_code, _bot, run_battle, port_return, _claim, _quick = drive_run(
        recover=[ScreenState.BATTLE, ScreenState.BATTLE],
        battle_finished="battle_timeout",
        port_return=False,
        gates=[GATE, GATE, GATE, GATE],
        monotonic=[0.0],
    )

    assert exit_code == 1
    assert run_battle.call_count == 2
    assert port_return.call_count == 2


def test_battle_timeout_with_confirmed_port_return_keeps_plan_running():
    """A single timeout that returns to port successfully is not counted
    against the run: the plan continues instead of stopping."""
    exit_code, _bot, run_battle, port_return, _claim, _quick = drive_run(
        recover=[ScreenState.BATTLE],
        battle_finished="battle_timeout",
        port_return=True,
        gates=[GATE, GATE, GATE, None],
        monotonic=[0.0, 999999.0],
    )

    assert exit_code == 0
    assert run_battle.call_count == 1
    assert port_return.call_count == 1


def test_quick_battle_closure_failing_three_times_terminates_task():
    """Quick battles must count a round only after a confirmed exit; three
    consecutive unconfirmed closures stop the task instead of spinning."""
    exit_code, _bot, run_battle, port_return, _claim, quick_closure = drive_run(
        quick_battle=True,
        recover=[ScreenState.BATTLE, ScreenState.BATTLE, ScreenState.BATTLE],
        battle_finished="quick_timeout",
        quick_closure=ScreenState.BATTLE,
        gates=[GATE, GATE, GATE, GATE, GATE],
        monotonic=[0.0],
    )

    assert exit_code == 1
    assert run_battle.call_count == 3
    assert quick_closure.call_count == 3
    assert port_return.call_count == 0


def test_daily_reward_three_failures_force_close_and_keep_plan_running():
    """A reward page that cannot be claimed must not stall the plan: after
    three failures the page is closed with Esc and the task keeps going."""
    exit_code, bot, _run_battle, port_return, claim, _quick = drive_run(
        recover=[
            ScreenState.DAILY_REWARD,
            ScreenState.DAILY_REWARD,
            ScreenState.DAILY_REWARD,
            ScreenState.PORT,
            ScreenState.PORT,
        ],
        claim_reward=False,
        port_return=True,
        gates=[GATE, GATE, GATE, GATE, GATE, GATE, None],
        monotonic=[0.0, 999999.0],
    )

    assert exit_code == 0
    assert claim.call_count == 3
    assert port_return.call_count == 3
    assert bot.gamepad.escape.call_count == 1
