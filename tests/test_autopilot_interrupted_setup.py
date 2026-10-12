"""A paused waypoint confirmation is resumable, with one bounded budget.

Only screenshot/Windows adapters are mocked. Both the configuration and
the lifecycle handoff run their production control flow.
"""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

import main
from core.frame_guard import CaptureFault
from core.tactical_geometry import TacticalMapGeometry
from core.ui import ScreenState
from core.vision import PlayerPose


@pytest.fixture
def setup(monkeypatch):
    state = SimpleNamespace(paused=False, map_open=False, enabled=False,
                            fail_after_click=2, interruption="capture", injected=False,
                            enable_after_click=3, hud_reads={}, captures=0, actions=[], clicks=[])
    geometry = TacticalMapGeometry((1000, 1600),
                                  ((395, 95), (1205, 95), (1205, 905), (395, 905)), 810)
    bot = SimpleNamespace(hwnd=7, intervention=None, strategy={}, opening_autopilot_active=False,
                          native_autopilot_abandoned=False, generic_center_route_active=False,
                          _opening_autopilot_attempted=True, battle_start_time=1000.,
                          combat_tick=Mock(return_value="ended"), enable_generic_center_route=Mock())

    def grab(*_args, **_kwargs):
        count = len(state.clicks)
        if (not state.injected and state.interruption == "capture"
                and count == state.fail_after_click and not state.map_open
                and state.hud_reads.get(count, 0)):
            state.injected = True
            raise CaptureFault("切屏期间游戏画面连续3次截取失败")
        state.captures += 1
        frame = np.zeros((1000, 1600, 3), np.uint8)
        frame[0, 0] = (int(state.map_open), int(state.enabled), 0)
        return frame

    def hud_enabled(frame, _backend):
        count = len(state.clicks)
        state.hud_reads[count] = state.hud_reads.get(count, 0) + 1
        return bool(frame[0, 0, 1])

    def operation_paused(_bot):
        if (not state.injected and state.interruption == "pause"
                and len(state.clicks) == state.fail_after_click and not state.map_open):
            state.injected, state.paused = True, True
        return state.paused

    def toggle():
        state.map_open = not state.map_open
        state.actions.append("M_open" if state.map_open else "M_close")

    def normalize(_bot):
        if state.paused:
            return False
        if state.map_open:
            toggle()
        bot._tactical_map_left_open = False
        return True

    def click(backend, x, y):
        assert state.map_open and not state.paused
        state.clicks.append((backend, x, y))
        if len(state.clicks) == state.enable_after_click:
            state.enabled = True
        return True

    def enable(label, *, target_normalized=None):
        bot.opening_autopilot_active = True
        bot.opening_autopilot_target = label
        bot.opening_autopilot_target_normalized = target_normalized

    def reset(**_kwargs):
        # Real reset clears local native status; fresh HUD must re-establish
        # it while the preserve_static_map contract retains the setup budget.
        bot.opening_autopilot_active = False
        bot._native_autopilot_confirmed = False

    def forward_controls():
        # A W reassertion cancels an existing native waypoint in the game.
        state.enabled = False

    bot.vision = SimpleNamespace(grab=Mock(side_effect=grab),
                                read_health_fraction=lambda *_args: 1.,
                                read_autopilot_enabled_text=Mock(side_effect=hud_enabled),
                                find_minimap=lambda _: np.zeros((200, 200, 3), np.uint8),
                                find_player_pose_on_minimap=lambda _: PlayerPose((40, 180), (0., -1.)))
    bot.distance_reader = SimpleNamespace(backend=object())
    bot.gamepad = SimpleNamespace(toggle_tactical_map=Mock(side_effect=toggle),
                                  resynchronize_forward_controls=Mock(side_effect=forward_controls),
                                  full_speed=Mock())
    bot._resynchronize_forward_controls = bot.gamepad.resynchronize_forward_controls
    bot.enable_opening_autopilot = Mock(side_effect=enable)
    bot.reset = Mock(side_effect=reset)
    monkeypatch.setattr(main.time, "sleep", lambda _: None)
    monkeypatch.setattr(main.time, "monotonic", lambda: 1000.)
    monkeypatch.setattr(main, "operation_paused", operation_paused)
    monkeypatch.setattr(main, "normalize_tactical_map_overlay", normalize)
    monkeypatch.setattr(main, "ensure_bound_game_foreground", lambda _: True)
    monkeypatch.setattr(main, "ensure_capture_foreground", lambda _: True)
    monkeypatch.setattr(main, "observe_battle_phase", lambda *_args: main.BattlePhase.ACTIVE)
    monkeypatch.setattr(main, "classify_battle_continuity_screen", lambda *_args: ScreenState.BATTLE)
    monkeypatch.setattr(main, "loading_start_confirmed", lambda *_args: False)
    monkeypatch.setattr(main, "tactical_map_is_open", lambda _, frame: bool(frame[0, 0, 0]))
    monkeypatch.setattr(main, "detect_tactical_map_geometry", lambda _: geometry)
    monkeypatch.setattr(main, "get_client_rect", lambda _: {"left": 0, "top": 0, "width": 1600, "height": 1000})
    monkeypatch.setattr(main, "physical_click", lambda x, y, **_kwargs: click("physical", x, y))
    monkeypatch.setattr(main, "window_message_click", lambda _, x, y, **_kwargs: click("message", x, y))
    monkeypatch.setattr(main, "_save_navigation_debug", Mock())
    monkeypatch.setattr(main.ctypes.windll.user32, "GetForegroundWindow", lambda: 7)
    return SimpleNamespace(bot=bot, state=state)


@pytest.mark.parametrize("interruption", ["capture", "pause"])
def test_interrupted_second_click_resumes_third_and_never_restarts_its_budget(setup, interruption):
    bot, state = setup.bot, setup.state
    state.interruption = interruption
    assert main.configure_opening_autopilot(bot) is False
    assert len(state.clicks) == 2 and bot._opening_autopilot_retry_pending
    bot.enable_opening_autopilot.assert_not_called()
    state.paused = False
    assert main.configure_opening_autopilot(bot) is True
    assert [backend for backend, *_ in state.clicks] == ["physical", "message", "message"]
    assert len({(x, y) for _, x, y in state.clicks}) == 3
    assert state.actions.count("M_open") == 3
    assert bot._tactical_map_attempts_used == 3
    assert not bot._opening_autopilot_retry_pending
    bot.enable_opening_autopilot.assert_called_once()
    # A subsequent lifecycle visit cannot grant three additional clicks.
    assert main.configure_opening_autopilot(bot) is False
    assert len(state.clicks) == 3 and state.actions.count("M_open") == 3


def test_interrupted_click_that_already_enabled_native_route_is_adopted_without_new_m(setup):
    bot, state = setup.bot, setup.state
    assert main.configure_opening_autopilot(bot) is False
    target = bot._opening_autopilot_pending_target
    state.enabled = True  # The second dispatch took effect during the pause.
    actions, clicks = list(state.actions), list(state.clicks)
    assert main.configure_opening_autopilot(bot) is True
    assert state.actions == actions and state.clicks == clicks
    assert bot.opening_autopilot_active and bot._native_autopilot_confirmed
    assert bot.opening_autopilot_target_normalized == target
    assert bot._tactical_map_attempts_used == 2


def test_three_actual_unconfirmed_attempts_remain_exhausted_on_later_visits(setup):
    bot, state = setup.bot, setup.state
    state.interruption, state.enable_after_click = None, None
    assert main.configure_opening_autopilot(bot) is False
    assert len(state.clicks) == 3 and not getattr(bot, "_opening_autopilot_retry_pending", False)
    before = list(state.actions)
    for _ in range(3):
        assert main.configure_opening_autopilot(bot) is False
    assert state.actions == before and len(state.clicks) == 3
    bot.enable_opening_autopilot.assert_not_called()


@pytest.mark.parametrize("actually_enabled", [False, True])
def test_capture_fault_after_last_click_can_verify_but_never_grants_a_fourth(setup, actually_enabled):
    bot, state = setup.bot, setup.state
    state.fail_after_click, state.enable_after_click = 3, None
    assert main.configure_opening_autopilot(bot) is False
    assert len(state.clicks) == 3 and bot._opening_autopilot_retry_pending
    state.enabled = actually_enabled
    assert main.configure_opening_autopilot(bot) is actually_enabled
    assert len(state.clicks) == 3 and state.actions.count("M_open") == 3
    assert not bot._opening_autopilot_retry_pending


def test_pause_before_first_input_does_not_spend_any_map_attempt(setup):
    bot, state = setup.bot, setup.state
    state.paused = True
    assert main.configure_opening_autopilot(bot) is False
    assert not state.actions and not state.clicks and state.captures == 0
    state.paused, state.interruption, state.enable_after_click = False, None, 1
    assert main.configure_opening_autopilot(bot) is True
    assert len(state.clicks) == 1


def test_lifecycle_handoff_preserves_pending_budget_and_resumes_only_remaining_click(setup):
    bot, state = setup.bot, setup.state
    assert main.configure_opening_autopilot(bot) is False
    assert main.run_battle(bot, resume_existing=False) is True
    assert len(state.clicks) == 3 and bot._tactical_map_attempts_used == 3
    assert bot.reset.call_args.kwargs["preserve_static_map"] is True
    bot.enable_generic_center_route.assert_not_called()
    bot.combat_tick.assert_called_once()


def test_fresh_initialization_checks_pending_native_route_before_w_can_cancel_it(setup):
    bot, state = setup.bot, setup.state
    assert main.configure_opening_autopilot(bot) is False
    state.enabled, state.enable_after_click = True, None
    clicks = list(state.clicks)
    assert main.run_battle(bot, resume_existing=False) is True
    assert state.enabled and bot.opening_autopilot_active
    assert state.clicks == clicks
    bot._resynchronize_forward_controls.assert_not_called()
    bot.gamepad.full_speed.assert_not_called()
    bot.enable_generic_center_route.assert_not_called()


@pytest.mark.parametrize("already_enabled", [False, True])
def test_loading_handoff_verifies_pending_route_before_full_speed_and_keeps_budget(setup, already_enabled):
    bot, state = setup.bot, setup.state
    assert main.configure_opening_autopilot(bot) is False
    state.enabled = already_enabled
    if already_enabled:
        state.enable_after_click = None
    bot.vision.read_battle_clock_seconds = lambda *_args: 1198
    bot.gamepad.reassert_full_speed = Mock(side_effect=bot._resynchronize_forward_controls.side_effect)
    actions = list(state.actions)
    assert main.wait_for_battle(bot, timeout=2, require_new_round=True,
                                loading_already_seen=True, should_stop=lambda: state.captures > 60)
    assert bot._opening_autopilot_preconfigured
    assert not bot._opening_autopilot_retry_pending
    assert len(state.clicks) == (2 if already_enabled else 3)
    if already_enabled:
        assert state.actions == actions
    bot.gamepad.reassert_full_speed.assert_not_called()
    bot.gamepad.full_speed.assert_not_called()
    bot._resynchronize_forward_controls.assert_not_called()


def test_same_battle_ordinary_resume_keeps_generic_navigation_without_reopening_m(setup):
    bot, state = setup.bot, setup.state
    bot.generic_center_route_active = True
    bot.route_plan = object()
    route = bot.route_plan
    assert main.run_battle(bot, resume_existing=True) is True
    assert not state.actions and not state.clicks
    bot.reset.assert_not_called()
    bot.enable_generic_center_route.assert_not_called()
    assert bot.route_plan is route


def test_confirmed_next_match_gets_three_new_attempts_starting_with_physical_click(setup):
    bot, state = setup.bot, setup.state
    bot._tactical_map_attempted_this_battle = True
    bot._tactical_map_attempts_used = 3
    bot._tactical_map_prefer_message_click = True
    bot._opening_autopilot_retry_pending = False
    state.interruption, state.enable_after_click = None, 1
    bot.vision.read_battle_clock_seconds = lambda *_args: 1198
    assert main.wait_for_battle(bot, timeout=2, require_new_round=True,
                                loading_already_seen=True, should_stop=lambda: state.captures > 60)
    assert len(state.clicks) == 1 and state.clicks[0][0] == "physical"
    assert bot._tactical_map_attempts_used == 1
    assert bot._opening_autopilot_preconfigured


def test_settlement_closes_deferred_setup_before_next_match_gets_its_own_budget(setup):
    bot, state = setup.bot, setup.state
    assert main.configure_opening_autopilot(bot) is False
    assert bot._opening_autopilot_retry_pending
    main.finalize_round_diagnostics(bot, 1, outcome="victory")
    state.clicks.clear()
    state.enabled, state.enable_after_click = False, 1
    bot.vision.read_battle_clock_seconds = lambda *_args: 1198
    assert main.wait_for_battle(bot, timeout=2, require_new_round=True,
                                loading_already_seen=True, should_stop=lambda: state.captures > 80)
    assert len(state.clicks) == 1 and state.clicks[0][0] == "physical"
    assert bot._tactical_map_attempts_used == 1
