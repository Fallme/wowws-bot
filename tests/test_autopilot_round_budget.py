"""Transient loading before control handoff cannot create a new match budget."""

from unittest.mock import Mock

import pytest

import main
from core.ui import ScreenState
from test_autopilot_interrupted_setup import setup  # Reuse mock Windows/capture adapters.


def exhaust_opening_budget(harness):
    bot, state = harness.bot, harness.state
    state.interruption, state.enable_after_click = None, None
    bot.vision.read_battle_clock_seconds = lambda *_: 1198
    assert main.wait_for_battle(bot, timeout=2, require_new_round=True,
                                loading_already_seen=True)
    assert len(state.clicks) == 3
    assert bot._tactical_map_attempts_used == 3
    assert not getattr(bot, "_opening_autopilot_retry_pending", False)
    assert not getattr(bot, "_round_control_initialized", False)


def test_transient_loading_before_control_handoff_never_grants_fourth_map_click(
        setup, monkeypatch):
    bot, state = setup.bot, setup.state
    exhaust_opening_budget(setup)
    actions = list(state.actions)
    # A real fresh scene gate cancels entry before reset when the live HUD
    # briefly disappears after the loading-stage waypoint attempts.
    monkeypatch.setattr(main, "classify_battle_continuity_screen",
                        lambda *_: ScreenState.LOADING)
    assert main.run_battle(bot, resume_existing=False) == "resume_state"
    bot.reset.assert_not_called()
    assert len(state.clicks) == 3
    monkeypatch.setattr(main, "classify_battle_continuity_screen",
                        lambda *_: ScreenState.BATTLE)
    for _ in range(2):
        assert main.wait_for_battle(bot, timeout=2, require_new_round=True,
                                    loading_already_seen=True)
    assert len(state.clicks) == 3
    assert state.actions == actions
    assert bot._tactical_map_attempts_used == 3


@pytest.mark.parametrize("diagnostic_failure", [False, True])
def test_confirmed_closure_gives_next_match_three_fresh_attempts(
        setup, diagnostic_failure):
    bot, state = setup.bot, setup.state
    exhaust_opening_budget(setup)
    bot.complete_round_diagnostics = Mock(
        side_effect=OSError("diagnostic storage unavailable") if diagnostic_failure else None)
    main.finalize_round_diagnostics(bot, 1, outcome="victory")
    bot.complete_round_diagnostics.assert_called_once_with(1, outcome="victory")
    state.clicks.clear()
    state.actions.clear()
    assert main.wait_for_battle(bot, timeout=2, require_new_round=True,
                                loading_already_seen=True)
    assert len(state.clicks) == 3
    assert [backend for backend, *_ in state.clicks] == ["physical", "message", "message"]
    assert state.actions.count("M_open") == 3
    assert bot._tactical_map_attempts_used == 3


def test_fresh_control_handoff_preserves_budget_consumed_during_loading(setup):
    bot, state = setup.bot, setup.state
    exhaust_opening_budget(setup)
    assert main.run_battle(bot, resume_existing=False) is True
    assert bot.reset.call_args.kwargs["preserve_static_map"] is True
    assert len(state.clicks) == 3
    assert bot._tactical_map_attempts_used == 3
    bot.enable_generic_center_route.assert_called_once()
    bot.combat_tick.assert_called_once()


def test_confirmed_closure_then_direct_new_hud_cannot_reuse_old_opening_attempted_flag(setup):
    bot, state = setup.bot, setup.state
    exhaust_opening_budget(setup)
    bot._opening_motion_prestarted = True
    main.finalize_round_diagnostics(bot, 1, outcome="victory")
    assert not bot._opening_motion_prestarted
    # Model the reset adapter's existing contract with the production budget
    # initializer, rather than mocking away configure_opening_autopilot.
    reset_status = bot.reset.side_effect

    def reset(**kwargs):
        reset_status(**kwargs)
        if not kwargs.get("preserve_static_map", False):
            bot._opening_autopilot_budget_active = False
            main.reset_opening_autopilot_attempts(bot)

    bot.reset.side_effect = reset
    state.clicks.clear()
    state.actions.clear()
    state.enable_after_click = 1
    # The user can continue to a new HUD while paused. This entry therefore
    # goes straight to run_battle without another wait_for_battle call.
    assert main.run_battle(bot, resume_existing=False) is True
    assert bot.reset.call_args.kwargs["preserve_static_map"] is False
    assert len(state.clicks) == 1 and state.clicks[0][0] == "physical"
    assert bot.opening_autopilot_active
    bot.enable_generic_center_route.assert_not_called()
