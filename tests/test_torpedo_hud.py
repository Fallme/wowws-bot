"""Replay torpedo policy with real bot methods, fresh mocked frames and no input."""
from types import SimpleNamespace
from unittest.mock import Mock
import cv2
import numpy as np
import pytest
import bot as bot_module
from bot import BattleAnalysis, BattleBot
from core.torpedo_hud import TorpedoHud, inspect_torpedo_hud
from core.ui import ScreenState


def analysis():
    return BattleAnalysis(image=np.zeros((600, 1000, 3), np.uint8),
        width=1000, height=600, in_battle=True, health=.8, health_recognized=True,
        visible_target=True, target_track_id="enemy-1", target_viewport_point=(500, 300),
        target_distance_km=2.0, distance_confidence=.95,
        minimap_distance_km=2.0, enemies=[(500, 300)])


def hud(*, aligned=True, selected=True, weapon=True, loaded=True,
        lead=(510, 305), aim=(509, 305), cooldown=None, reticle=(500, 300)):
    return TorpedoHud(cooldown, loaded, lead, weapon_is_torpedo=weapon,
        selected=selected, aim_point=aim, alignment_error_px=1 if aligned else 90,
        alignment_tolerance_px=8, alignment_error_degrees=.1 if aligned else 10,
        reticle_point=reticle)


@pytest.fixture
def policy(monkeypatch):
    instance = object.__new__(BattleBot)
    instance.ship = {"has_torpedoes": True, "torpedo": {"range": 6, "reload": 90}}
    instance.strategy = {"distance_ocr_min_confidence": .78}
    calls = []

    def action(name):
        method = Mock(return_value=True)
        def dispatch(*args):
            calls.append((name, args))
            return method.return_value
        method.side_effect = dispatch
        return method

    instance.gamepad = SimpleNamespace(**{name: action(name) for name in
        ("torpedo", "main_gun", "aim_relative", "fire_torpedo", "lock")})
    paused = {"value": False}
    instance.intervention = SimpleNamespace(poll=Mock(return_value=False),
        command_generation_paused=lambda: paused["value"])
    frames = []
    def grab(*_args, **_kwargs):
        frame = np.zeros((600, 1000, 3), np.uint8)
        frames.append(frame)
        return frame
    instance.vision = SimpleNamespace(grab=Mock(side_effect=grab),
        classify_screen=Mock(return_value=ScreenState.BATTLE),
        read_health_fraction=Mock(return_value=.8),
        find_enemies_in_viewport=Mock(return_value=[(500, 300)]))
    observation = SimpleNamespace(accepted=True, value_km=2.0, confidence=.95)
    instance.distance_reader = SimpleNamespace(backend=object(), read=Mock(return_value=observation))
    instance.hwnd = 1
    instance._last_torpedo_attempt = instance._last_torpedo_release = instance.last_torpedo = 0.0
    instance._torpedo_restore_pending = False
    instance._torpedo_trace = None
    instance._record_torpedo_frame = Mock()
    instance.mark_manual_pause = Mock()
    instance.events = SimpleNamespace(publish=Mock())
    context = SimpleNamespace(bot=instance, calls=calls, frames=frames, paused=paused,
        observation=observation, states=[hud(), hud(), hud(loaded=False, cooldown=90)],
        restore_status=hud(selected=False, loaded=False, lead=None, aim=None))

    def inspect(_frame, before, _target, _backend=None):
        if before is None:
            return context.restore_status
        return context.states.pop(0) if len(context.states) > 1 else context.states[0]

    context.inspect = Mock(side_effect=inspect)
    context.detect = Mock(side_effect=[True, True])
    context.rect = Mock(return_value={"left": 40, "top": 30, "width": 1000, "height": 600})
    context.absolute_click = Mock()
    monkeypatch.setattr(bot_module, "inspect_torpedo_hud", context.inspect)
    monkeypatch.setattr(bot_module, "detect_torpedo_weapon", context.detect)
    monkeypatch.setattr(bot_module, "detect_main_gun_selected", lambda *_:
                        context.restore_status.weapon_is_torpedo is True and not context.restore_status.selected)
    monkeypatch.setattr(bot_module, "get_client_rect", context.rect)
    monkeypatch.setattr(bot_module, "physical_click", context.absolute_click, raising=False)
    monkeypatch.setattr(bot_module.time, "monotonic", lambda: 1000.0)
    monkeypatch.setattr(bot_module.time, "sleep", Mock())
    return context


def run(policy):
    policy.bot._try_torpedo_attack(analysis(), 1000.0)


def test_weak_green_line_and_white_triangle_do_not_prove_safe_torpedo_hud():
    before = np.zeros((600, 1000, 3), np.uint8)
    selected = before.copy()
    cv2.line(selected, (380, 350), (620, 365), (20, 205, 20), 5)
    cv2.fillConvexPoly(selected, np.array([[480, 280], [520, 280], [500, 330]]), (225, 225, 225))
    assert not inspect_torpedo_hud(selected, before, (500, 300)).safe_to_fire
    assert not inspect_torpedo_hud(before, before, (500, 300)).safe_to_fire


def test_centered_target_without_white_band_proves_alignment_on_two_fresh_frames(policy):
    # The game tints the white lead band green where it overlaps the loaded
    # fan, so an aligned shot can have no detectable lead_point at all.
    policy.states = [hud(lead=None, aim=(509, 305)),
                     hud(lead=None, aim=(509, 305)),
                     hud(loaded=False, cooldown=90)]
    run(policy)
    policy.bot.gamepad.fire_torpedo.assert_called_once()
    policy.bot.gamepad.aim_relative.assert_not_called()


def test_offset_target_without_white_band_does_not_fire(policy):
    policy.states = [hud(lead=None, aim=(560, 305), reticle=(560, 300)),
                     hud(lead=None, aim=(560, 305), reticle=(560, 300))]
    run(policy)
    policy.bot.gamepad.fire_torpedo.assert_not_called()


def test_centered_target_cannot_replace_a_visible_but_unaligned_lead(policy):
    policy.states = [hud(aligned=False, lead=(560, 305), aim=(509, 305)),
                     hud(aligned=False, lead=(560, 305), aim=(509, 305))]
    run(policy)
    policy.bot.gamepad.fire_torpedo.assert_not_called()


def test_unaligned_hud_requires_relative_aim_and_two_fresh_aligned_frames(policy):
    policy.states = [hud(aligned=False, aim=(490, 305), lead=(580, 305)),
                     hud(), hud(), hud(loaded=False, cooldown=90)]
    original = analysis()
    policy.bot._try_torpedo_attack(original, 1000.0)
    assert policy.calls == [("torpedo", ()), ("aim_relative", (24, 0)),
                            ("fire_torpedo", ()), ("main_gun", ())]
    assert policy.detect.call_count == 2
    assert policy.inspect.call_count == 4
    assert policy.bot._last_torpedo_release == 1000.0
    assert not policy.bot._torpedo_restore_pending
    assert all(frame is not original.image for frame in policy.frames)
    assert len({id(frame) for frame in policy.frames}) == len(policy.frames)
    policy.absolute_click.assert_not_called()
    assert [call.args[0] for call in policy.bot.events.publish.call_args_list] == [
        "torpedo.launch_attempt", "torpedo.launch_confirmed"]


def test_already_aligned_requires_two_frames_before_fire(policy):
    run(policy)
    assert policy.calls == [("torpedo", ()), ("fire_torpedo", ()), ("main_gun", ())]
    assert policy.inspect.call_count == 3  # 2 alignment and reload; slot 2 confirms restoration.
    policy.bot.gamepad.aim_relative.assert_not_called()
    policy.absolute_click.assert_not_called()


@pytest.mark.parametrize("capability", [False, None])
def test_absent_or_unknown_weapon_never_selects_slot_three(policy, capability):
    policy.detect.side_effect = None
    policy.detect.return_value = capability
    run(policy)
    assert policy.calls == []
    assert not policy.bot._torpedo_restore_pending


def test_custom_false_config_can_use_visually_confirmed_weapon(policy):
    policy.bot.ship = {"has_torpedoes": False, "type": "CUSTOM"}
    run(policy)
    policy.bot.gamepad.torpedo.assert_called_once()
    policy.bot.gamepad.fire_torpedo.assert_called_once()


@pytest.mark.parametrize("status", [hud(cooldown=35), hud(loaded=False),
    hud(weapon=None), hud(selected=False), hud(lead=None, aim=None, reticle=None)])
def test_unsafe_selected_hud_does_not_fire(policy, status):
    policy.states = [status]
    run(policy)
    policy.bot.gamepad.fire_torpedo.assert_not_called()
    policy.bot.gamepad.aim_relative.assert_not_called()
    policy.bot.gamepad.main_gun.assert_called_once()
    assert policy.bot._last_torpedo_release == 0


@pytest.mark.parametrize("distance", [2.8, 5.0])
def test_changed_fresh_distance_cancels_old_target(policy, distance):
    changed = SimpleNamespace(accepted=True, value_km=distance, confidence=.95)
    policy.bot.distance_reader.read.side_effect = [policy.observation, policy.observation, changed]
    run(policy)
    policy.bot.gamepad.torpedo.assert_called_once()
    policy.bot.gamepad.fire_torpedo.assert_not_called()
    policy.bot.gamepad.main_gun.assert_called_once()


def test_missing_target_label_cancels_before_alignment(policy):
    policy.bot.vision.find_enemies_in_viewport.side_effect = [[(500, 300)], [(500, 300)], []]
    run(policy)
    policy.bot.gamepad.torpedo.assert_called_once()
    policy.bot.gamepad.fire_torpedo.assert_not_called()


@pytest.mark.parametrize("reason", ["port", "results", "menu", "sunk", "capture_size", "client_size"])
def test_invalid_fresh_frame_cancels_without_restore_keys(policy, reason):
    counts = {"value": 0}
    scenes = {"port": ScreenState.PORT, "results": ScreenState.RESULTS, "menu": ScreenState.ESCAPE_MENU}
    def next_value(first, later):
        counts["value"] += 1
        return first if counts["value"] <= 2 else later
    if reason in scenes:
        policy.bot.vision.classify_screen.side_effect = lambda _: next_value(ScreenState.BATTLE, scenes[reason])
    elif reason == "sunk":
        policy.bot.vision.read_health_fraction.side_effect = lambda *_: next_value(.8, 0.0)
    elif reason == "capture_size":
        policy.bot.vision.grab.side_effect = lambda *a, **k: np.zeros(
            next_value((600, 1000, 3), (620, 1000, 3)), np.uint8)
    else:
        policy.rect.side_effect = lambda _: {"left": 40, "top": 30, "height": 600,
                                             "width": next_value(1000, 1020)}
    run(policy)
    policy.bot.gamepad.torpedo.assert_called_once()
    policy.bot.gamepad.fire_torpedo.assert_not_called()
    policy.bot.gamepad.main_gun.assert_not_called()
    assert policy.bot._torpedo_restore_pending


def test_user_pause_after_selection_stops_aim_fire_and_restore(policy):
    def select():
        policy.calls.append(("torpedo", ()))
        policy.paused["value"] = True
        return True
    policy.bot.gamepad.torpedo.side_effect = select
    run(policy)
    assert policy.calls == [("torpedo", ())]
    assert policy.bot._torpedo_restore_pending


def test_missing_white_line_allows_only_one_lock_for_unique_central_enemy(policy):
    policy.states = [hud(lead=None, aim=None)]
    run(policy)
    assert policy.calls == [("torpedo", ()), ("lock", ()), ("main_gun", ())]
    policy.bot.gamepad.fire_torpedo.assert_not_called()


def test_two_central_enemies_cannot_authorize_lock(policy):
    policy.states = [hud(lead=None, aim=None)]
    counts = {"value": 0}
    def enemies(_frame):
        counts["value"] += 1
        return [(500, 300)] if counts["value"] <= 2 else [(500, 300), (560, 300)]
    policy.bot.vision.find_enemies_in_viewport.side_effect = enemies
    run(policy)
    policy.bot.gamepad.lock.assert_not_called()
    policy.bot.gamepad.fire_torpedo.assert_not_called()


@pytest.mark.parametrize("method", ["torpedo", "aim_relative", "fire_torpedo"])
def test_failed_input_does_not_record_release(policy, method):
    if method == "aim_relative":
        policy.states = [hud(aligned=False, lead=(580, 305), aim=(490, 305))]
    getattr(policy.bot.gamepad, method).return_value = False
    run(policy)
    assert policy.bot._last_torpedo_release == 0
    assert policy.bot.last_torpedo == 0
    published = [call.args[0] for call in policy.bot.events.publish.call_args_list]
    assert "torpedo.launch_attempt" not in published
    assert "torpedo.launch_confirmed" not in published


@pytest.mark.parametrize("failure", ["input", "selected_feedback"])
def test_failed_restore_retries_only_restore_next_tick(policy, monkeypatch, failure):
    clock = {"value": 1000.0}
    monkeypatch.setattr(bot_module.time, "monotonic", lambda: clock["value"])
    policy.states = [hud(cooldown=35)]
    if failure == "input":
        policy.bot.gamepad.main_gun.return_value = False
    else:
        policy.restore_status = hud(selected=True)
    run(policy)
    assert policy.bot._torpedo_restore_pending
    assert policy.bot.gamepad.main_gun.call_count == 1
    assert policy.bot.gamepad.torpedo.call_count == 1
    policy.bot._try_torpedo_attack(analysis(), 1000.5)
    assert policy.bot._torpedo_restore_pending
    assert policy.bot.gamepad.main_gun.call_count == 1
    assert policy.bot.gamepad.torpedo.call_count == 1
    policy.bot.gamepad.main_gun.return_value = True
    policy.restore_status = hud(selected=False, loaded=False, lead=None, aim=None)
    clock["value"] = 1001.0
    policy.bot._try_torpedo_attack(analysis(), 1001.0)
    assert not policy.bot._torpedo_restore_pending
    assert policy.bot.gamepad.main_gun.call_count == 2
    assert policy.bot.gamepad.torpedo.call_count == 1
    policy.bot.gamepad.fire_torpedo.assert_not_called()


def test_unaligned_attempt_is_bounded_with_fixed_monotonic_clock(policy):
    policy.states = [hud(aligned=False, lead=(580, 305), aim=(490, 305))]
    run(policy)
    assert 0 < policy.bot.gamepad.aim_relative.call_count <= 8
    policy.bot.gamepad.fire_torpedo.assert_not_called()
    assert policy.inspect.call_count <= 13


def test_second_capability_observation_must_also_confirm_weapon(policy):
    policy.detect.side_effect = [True, None]
    run(policy)
    assert policy.detect.call_count == 2
    policy.bot.gamepad.torpedo.assert_not_called()


def test_pause_at_final_firing_boundary_blocks_launch_and_restore(policy):
    def record(_frame, _phase, **details):
        if details.get("firing"):
            policy.paused["value"] = True
    policy.bot._record_torpedo_frame.side_effect = record
    run(policy)
    policy.bot.gamepad.torpedo.assert_called_once()
    policy.bot.gamepad.fire_torpedo.assert_not_called()
    policy.bot.gamepad.main_gun.assert_not_called()
    assert policy.bot._torpedo_restore_pending
    assert policy.bot._last_torpedo_release == 0


@pytest.mark.parametrize("slow_step", ["distance", "hud", "final_record"])
def test_recognition_or_evidence_passing_deadline_cannot_fire(policy, monkeypatch, slow_step):
    clock = {"value": 1000.0}
    monkeypatch.setattr(bot_module.time, "monotonic", lambda: clock["value"])
    if slow_step == "distance":
        calls = {"count": 0}
        def read(*_args):
            calls["count"] += 1
            if calls["count"] == 3:
                clock["value"] = 1015.0
            return policy.observation
        policy.bot.distance_reader.read.side_effect = read
    elif slow_step == "hud":
        normal_inspect = policy.inspect.side_effect
        calls = {"count": 0}
        def inspect(*args):
            status = normal_inspect(*args)
            if args[1] is not None:
                calls["count"] += 1
                if calls["count"] == 2:
                    clock["value"] = 1015.0
            return status
        policy.inspect.side_effect = inspect
    else:
        def record(_frame, _phase, **details):
            if details.get("firing"):
                clock["value"] = 1015.0
        policy.bot._record_torpedo_frame.side_effect = record
    run(policy)
    policy.bot.gamepad.fire_torpedo.assert_not_called()
    policy.bot.gamepad.main_gun.assert_called_once()
    assert policy.bot._last_torpedo_release == 0


def test_after_shot_results_frame_cannot_confirm_launch_or_receive_restore_key(policy):
    policy.bot.vision.classify_screen.side_effect = lambda _: (
        ScreenState.RESULTS if policy.bot.gamepad.fire_torpedo.call_count else ScreenState.BATTLE
    )
    run(policy)
    policy.bot.gamepad.fire_torpedo.assert_called_once()
    policy.bot.gamepad.main_gun.assert_not_called()
    assert policy.bot._torpedo_restore_pending
    assert policy.inspect.call_count == 2
    published = [call.args[0] for call in policy.bot.events.publish.call_args_list]
    assert published == ["torpedo.launch_attempt"]


def test_window_resize_at_final_firing_boundary_cancels_launch(policy):
    def record(_frame, _phase, **details):
        if details.get("firing"):
            policy.rect.return_value = {"left": 40, "top": 30, "width": 1100, "height": 600}
    policy.bot._record_torpedo_frame.side_effect = record
    run(policy)
    policy.bot.gamepad.fire_torpedo.assert_not_called()
    policy.bot.gamepad.main_gun.assert_not_called()
    assert policy.bot._last_torpedo_release == 0


def test_point_blank_target_is_not_excluded_by_old_six_hundred_meter_cutoff(policy):
    close = analysis()
    close.target_distance_km = close.minimap_distance_km = .4
    policy.observation.value_km = .4
    policy.bot._try_torpedo_attack(close, 1000.0)
    policy.bot.gamepad.fire_torpedo.assert_called_once()


def test_new_ready_launcher_does_not_wait_for_nominal_whole_ship_reload(policy, monkeypatch):
    run(policy)
    assert policy.bot.gamepad.fire_torpedo.call_count == 1
    monkeypatch.setattr(bot_module.time, "monotonic", lambda: 1009.0)
    policy.detect.side_effect = [True, True]
    policy.states = [hud(), hud(), hud(loaded=False, cooldown=90)]
    policy.bot._try_torpedo_attack(analysis(), 1009.0)
    assert policy.bot.gamepad.fire_torpedo.call_count == 2


def test_slow_aiming_defers_navigation_until_next_fresh_analysis(policy, monkeypatch):
    clock = {"value": 1000.0}
    monkeypatch.setattr(bot_module.time, "monotonic", lambda: clock["value"])
    policy.bot._manual_intervention_active = False
    policy.bot._execute_survival_consumables = Mock(return_value=True)
    def slow_aim(_analysis, _now):
        clock["value"] += 1.0
    policy.bot._try_torpedo_attack = Mock(side_effect=slow_aim)
    # No navigation/controller state was copied from before this camera move.
    # The old analysis must not reach the steering branch on this tick.
    policy.bot._execute_rules(analysis(), 1000.0)
    policy.bot._execute_survival_consumables.assert_called_once()
    policy.bot._try_torpedo_attack.assert_called_once()
    assert policy.calls == []


@pytest.mark.parametrize("action", ["aim_relative", "lock"])
def test_observation_evidence_timeout_blocks_aim_and_lock_input(policy, monkeypatch, action):
    clock = {"value": 1000.0}
    monkeypatch.setattr(bot_module.time, "monotonic", lambda: clock["value"])
    policy.states = [hud(aligned=False, lead=(580, 305), aim=(490, 305))
                     if action == "aim_relative" else hud(lead=None, aim=None)]
    def record(_frame, phase, **_details):
        if phase == "observation":
            clock["value"] = 1015.0
    policy.bot._record_torpedo_frame.side_effect = record
    run(policy)
    getattr(policy.bot.gamepad, action).assert_not_called()
    policy.bot.gamepad.fire_torpedo.assert_not_called()
    policy.bot.gamepad.main_gun.assert_called_once()
    assert policy.bot._last_torpedo_release == 0


def test_live_speed_recognition_can_align_over_multiple_fresh_cycles(policy, monkeypatch):
    clock = {"value": 1000.0}
    monkeypatch.setattr(bot_module.time, "monotonic", lambda: clock["value"])
    normal_inspect = policy.inspect.side_effect
    def inspect(*args):
        clock["value"] += 3.1
        return normal_inspect(*args)
    policy.inspect.side_effect = inspect
    policy.states = [hud(aligned=False, aim=(490, 305), lead=(580, 305)),
                     hud(), hud(), hud(loaded=False, cooldown=90)]
    run(policy)
    assert clock["value"] > 1006.0
    policy.bot.gamepad.aim_relative.assert_called_once()
    policy.bot.gamepad.fire_torpedo.assert_called_once()


def test_extended_attempt_budget_does_not_allow_a_stale_aligned_frame(policy, monkeypatch):
    clock = {"value": 1000.0}
    monkeypatch.setattr(bot_module.time, "monotonic", lambda: clock["value"])
    normal_inspect = policy.inspect.side_effect
    def inspect(*args):
        clock["value"] += 4.1
        return normal_inspect(*args)
    policy.inspect.side_effect = inspect
    run(policy)
    policy.bot.gamepad.fire_torpedo.assert_not_called()
    assert policy.bot._last_torpedo_release == 0


def test_side_target_is_centered_without_firing_until_two_green_aligned_frames(policy):
    policy.states = [hud(lead=None, aim=None, reticle=(850, 300)),
                     hud(), hud(), hud(loaded=False, cooldown=90)]
    run(policy)
    assert policy.calls == [("torpedo", ()), ("aim_relative", (-40, 0)),
                            ("fire_torpedo", ()), ("main_gun", ())]
    assert policy.inspect.call_count == 4
