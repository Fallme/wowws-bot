"""Regression coverage for live HUD loss and same-round continuation."""

from types import SimpleNamespace
from unittest.mock import Mock

import cv2
import numpy as np
import pytest

import main
from core import window
from core.intervention import UserInterventionMonitor, _keyboard_activity
from core.ui import ScreenState
from core.vision import Vision
from strategy.native_navigation import NativeRouteWatchdog
from tests.test_contact_trajectory_regression import native_bot, analysis


def test_recognition_pause_publishes_continue_and_resume_without_input():
    status = SimpleNamespace(manual_intervention_active=False,
                             manual_intervention_latched=False)
    updates = []
    def update(state, message, **values):
        updates.append(state)
        for key, value in values.items():
            setattr(status, key, value)
    intervention = SimpleNamespace(poll=Mock(return_value=True), latched=False)
    controller = Mock()
    bot = SimpleNamespace(intervention=intervention, gamepad=controller,
        mark_manual_pause=Mock(),
        runtime_reporter=SimpleNamespace(status=status, update=update))
    assert main.operation_paused(bot)
    assert status.manual_intervention_active
    assert main.operation_paused(bot)
    assert updates == ["paused"]
    intervention.latched = True
    assert main.operation_paused(bot)
    assert status.manual_intervention_latched
    intervention.poll.return_value = False
    intervention.latched = False
    assert not main.operation_paused(bot)
    assert not status.manual_intervention_active
    assert not status.manual_intervention_latched
    assert updates == ["paused", "paused", "recovering"]
    assert not controller.mock_calls


@pytest.mark.parametrize("state", [ScreenState.LOADING, ScreenState.PORT, ScreenState.BATTLE])
def test_ocr_rejection_of_loading_survives_every_entry_gate(monkeypatch, state):
    frame = np.zeros((90, 160, 3), np.uint8)
    vision = SimpleNamespace(
        grab=lambda *a, **kw: frame,
        classify_screen=lambda _: state,
        _has_battle_hud=lambda _: True,
        _has_loading_start_action=lambda _: True,
    )
    backend = SimpleNamespace(recognize=lambda _: [SimpleNamespace(text="FULL")])
    bot = SimpleNamespace(hwnd=1, vision=vision, distance_reader=SimpleNamespace(backend=backend))
    monkeypatch.setattr(main.time, "sleep", lambda _: None)
    assert main.classify_battle_continuity_screen(bot, frame) == ScreenState.BATTLE
    assert main.wait_for_battle(bot, timeout=1)


@pytest.mark.parametrize("tokens", [[], [SimpleNamespace(text="\u5f00\u59cb\u6218\u6597")]])
def test_real_or_unreadable_loading_button_still_blocks_commands(tokens):
    vision = SimpleNamespace(classify_screen=lambda _: ScreenState.LOADING,
        _has_battle_hud=lambda _: True, _has_loading_start_action=lambda _: True)
    bot = SimpleNamespace(vision=vision, distance_reader=SimpleNamespace(
        backend=SimpleNamespace(recognize=lambda _: tokens)))
    assert main.classify_battle_continuity_screen(bot, np.zeros((90, 160, 3), np.uint8)) == ScreenState.LOADING


def test_actual_low_contrast_match_can_be_recovered():
    frame = cv2.imread("tests/fixtures/battle_low_contrast_1440.png")
    assert frame is not None
    vision = Vision()
    bot = SimpleNamespace(vision=vision)
    assert vision._has_battle_hud(frame)
    assert main.classify_battle_continuity_screen(bot, frame) == ScreenState.BATTLE


def test_low_contrast_hud_fallback_requires_player_marker(monkeypatch):
    frame = cv2.imread("tests/fixtures/battle_low_contrast_1440.png")
    vision = Vision()
    monkeypatch.setattr(vision, "find_player_pose_on_minimap", lambda _: None)
    assert not vision._has_battle_hud(frame)


def test_keyboard_poll_drains_all_alt_tab_edges(monkeypatch):
    pending = {0x09, 0x12, 0xA4}
    def read(key):
        if key in pending:
            pending.remove(key)
            return 1
        return 0
    monkeypatch.setattr("core.intervention.ctypes.windll.user32", SimpleNamespace(GetAsyncKeyState=read))
    assert _keyboard_activity()
    assert not pending
    assert not _keyboard_activity()


def test_web_continue_drains_old_keys_but_detects_new_activity(tmp_path):
    tick = [100]
    keys = [False]
    def read_keys():
        old, keys[0] = keys[0], False
        return old
    request = tmp_path / "resume"
    monitor = UserInterventionMonitor(7, input_tick_reader=lambda: tick[0],
        keyboard_activity_reader=read_keys, foreground_reader=lambda: 99,
        foreground_matcher=lambda target, foreground: target == foreground,
        resume_path=request)
    monitor.reset()
    monitor.latched = True
    keys[0] = True
    request.write_text("resume")
    assert not monitor.poll(None, now=10)
    tick[0] += 10
    assert not monitor.poll(None, now=11)
    keys[0] = True
    tick[0] += 10
    assert monitor.poll(None, now=12)


def test_third_foreground_attempt_reports_success(monkeypatch):
    state = {"focused": False, "attempts": 0}
    def activate(_):
        state["attempts"] += 1
        state["focused"] = state["attempts"] == 3
    monkeypatch.setattr(window, "_interaction_paused", lambda: False)
    monkeypatch.setattr(window, "is_game_window", lambda _: True)
    monkeypatch.setattr(window, "_foreground_matches", lambda _: state["focused"])
    monkeypatch.setattr(window, "activate_window", activate)
    monkeypatch.setattr(window.time, "sleep", lambda _: None)
    assert window.ensure_game_window_foreground(7)
    assert state["attempts"] == 3


def test_missing_navigation_text_and_low_speed_recover_without_player_pose():
    bot, backend = native_bot()
    frame = analysis(autopilot_hud_visible=False, speed_knots=0,
        player_position=None, minimap_player_normalized=None)
    for now in (30., 34., 38.):
        bot._execute_rules(frame, now)
        assert bot.opening_autopilot_active
    bot._execute_rules(frame, 42.)
    assert not bot.opening_autopilot_active
    assert bot.native_autopilot_abandoned
    assert bot.autopilot_retry_pending
    assert backend.events


@pytest.mark.parametrize("speed,visible", [(25, False), (0, None), (None, False), (0, True)])
def test_navigation_loss_requires_fresh_absence_and_low_speed(speed, visible):
    watchdog = NativeRouteWatchdog()
    for now in range(60):
        assert not watchdog.lost_indicator(float(now), visible, speed)


def test_navigation_loss_does_not_accumulate_across_pause_or_ocr_gap():
    watchdog = NativeRouteWatchdog()
    for now in (0, 4, 8):
        assert not watchdog.lost_indicator(now, False, 0)
    watchdog.reset()
    assert not watchdog.lost_indicator(12, False, 0)
    assert not watchdog.lost_indicator(60, False, 0)


@pytest.mark.parametrize("dead", [False, True])
def test_same_round_resume_preserves_route_and_does_not_restart_opening(monkeypatch, dead):
    frame = np.zeros((90, 160, 3), np.uint8)
    controls = SimpleNamespace(resynchronize_forward_controls=Mock(), full_speed=Mock())
    bot = SimpleNamespace(hwnd=1, intervention=None, gamepad=controls,
        vision=SimpleNamespace(grab=lambda *a, **kw: frame,
            read_health_fraction=lambda *a: 0.0 if dead else 1.0),
        generic_center_route_active=True, native_autopilot_abandoned=False,
        _resynchronize_forward_controls=Mock(), enable_generic_center_route=Mock(),
        reset=Mock(), combat_tick=lambda: "ended", battle_start_time=100.,
        last_heal=90., route_plan=object())
    route = bot.route_plan
    monkeypatch.setattr(main, "ensure_bound_game_foreground", lambda _: True)
    monkeypatch.setattr(main, "normalize_tactical_map_overlay", lambda _: True)
    monkeypatch.setattr(main, "classify_battle_continuity_screen", lambda *a: ScreenState.BATTLE)
    configure = Mock()
    monkeypatch.setattr(main, "configure_opening_autopilot", configure)
    monkeypatch.setattr(main.time, "monotonic", lambda: 101.)
    assert main.run_battle(bot, resume_existing=True)
    configure.assert_not_called()
    bot.reset.assert_not_called()
    bot.enable_generic_center_route.assert_not_called()
    assert bot._resynchronize_forward_controls.call_count == (0 if dead else 1)
    controls.resynchronize_forward_controls.assert_not_called()
    controls.full_speed.assert_not_called()
    assert bot.route_plan is route
    assert bot.battle_start_time == 100.
    assert bot.last_heal == 90.


def test_unreadable_render_window_is_not_reported_as_restored(monkeypatch):
    def unavailable(_):
        raise OSError("render surface is being recreated")
    monkeypatch.setattr(window, "get_window_rect", unavailable)
    monkeypatch.setattr(window.win32gui, "GetForegroundWindow", lambda: 7)
    assert not window.is_usable_game_window(7)
    assert not window._foreground_matches(7)


def test_automatic_resume_retries_until_replacement_window_is_foreground(monkeypatch):
    now = [10.]
    foreground = [7]
    attempts = []
    monitor = UserInterventionMonitor(7, pause_seconds=5,
        input_tick_reader=lambda: 100, keyboard_activity_reader=lambda: False,
        foreground_reader=lambda: foreground[0],
        foreground_matcher=lambda target, current: target == current)
    monitor.reset()
    foreground[0] = 99
    bot = SimpleNamespace(hwnd=7, intervention=monitor, gamepad=SimpleNamespace())
    reporter = SimpleNamespace(update=Mock())
    limits = SimpleNamespace(pause_requested=lambda: False, stop_requested=lambda: False)
    def ensure(_bot):
        attempts.append(now[0])
        if len(attempts) < 3:
            return False
        bot.hwnd = monitor.hwnd = foreground[0] = 8
        return True
    monkeypatch.setattr(main.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(main.time, "sleep", lambda delay: now.__setitem__(0, now[0] + max(delay, 1)))
    monkeypatch.setattr(main, "ensure_bound_game_foreground", ensure)
    monkeypatch.setattr(main, "is_game_window_alive", lambda _: True)
    result = main.wait_for_web_resume(limits, reporter, bot)
    assert result.allowed and result.resumed
    assert attempts[0] >= 15
    assert len(attempts) == 3
    assert bot.hwnd == foreground[0] == 8
    assert not monitor.latched
