from types import SimpleNamespace

import pytest

from core.keyboard import KeyboardController
from core.vision import CaptureZone
from strategy.secondary_movement import SecondaryMovementController, SecondaryMovementInput, MovementMode
from tests.test_contact_trajectory_regression import native_bot, analysis
from tests.test_keyboard_controller import TelegraphBackend


def contact(distance=10.0, **kwargs):
    values = dict(elapsed=100, health=1.0, visible_target=True,
                  minimap_target_bearing=0.2, minimap_distance_km=distance,
                  map_center_bearing=0.1, map_center_distance_km=15)
    values.update(kwargs)
    return SecondaryMovementInput(**values)


@pytest.mark.parametrize("inside", [False, True])
@pytest.mark.parametrize("pursue", [False, True])
@pytest.mark.parametrize("distance", [10.0, 9.8, 7.0])
def test_close_contact_selects_actual_second_notch_inside_and_outside_cap(inside, pursue, distance):
    movement = SecondaryMovementController(pursue_enemies=pursue)
    backend = TelegraphBackend(actual_notch=4)
    controller = KeyboardController(backend)
    command = movement.plan(contact(distance, inside_capture_point=inside))
    controller.set_movement(command.throttle, command.rudder)
    assert command.throttle == 0.5
    assert backend.actual_notch == 2


def test_contact_speed_hysteresis_reset_and_missing_measurement():
    movement = SecondaryMovementController()
    for distance, expected in [(10.2, 1.0), (10, .5), (10.4, .5), (9.9, .5), (11.1, 1)]:
        assert movement.plan(contact(distance)).throttle == expected
    assert movement.plan(contact(9)).throttle == .5
    assert movement.plan(contact(None)).throttle == 1
    movement.plan(contact(9))
    movement.reset()
    assert movement.plan(contact(10.4)).throttle == 1


@pytest.mark.parametrize("distance", [None, 0, -1, float("nan"), float("inf")])
def test_bad_minimap_distance_or_viewport_ocr_does_not_enable_slowdown(distance):
    movement = SecondaryMovementController()
    assert movement.plan(contact(distance, target_distance_km=5)).throttle == 1
    assert movement.plan(contact(5, minimap_target_bearing=None)).throttle == 1


def test_emergency_evasion_retains_speed_priority():
    movement = SecondaryMovementController()
    command = movement.plan(contact(torpedoes_incoming=True))
    assert command.mode == MovementMode.EVADE
    assert command.throttle == 1
    command = movement.plan(contact(island_distance=.01, island_avoidance_rudder=-1))
    assert command.mode == MovementMode.AVOID_ISLAND
    assert command.throttle < .5


def test_native_close_contact_slows_in_own_half_and_does_not_reassert_full_speed():
    bot, backend = native_bot()
    frame = analysis(minimap_player_normalized=(.2, .5),
                     nearest_enemy_normalized=(.4, .5), minimap_target_bearing=.2,
                     minimap_distance_km=10, minimap_enemy_count=1)
    for t in (40, 41):
        bot._execute_rules(frame, t)
        assert bot.opening_autopilot_active
    bot._execute_rules(frame, 42)
    assert bot.opening_autopilot_active
    assert backend.actual_notch == 2
    count = len(backend.events)
    for t in range(43, 48):
        frame.player_position = (500 + t, 500)
        bot._execute_rules(frame, t)
        assert backend.actual_notch == 2
    assert ("tap", "w") not in backend.events[count:]
    assert not any(key in {"q", "e"} for _, key in backend.events)


def test_native_priority_cancels_old_recovery_without_steering():
    bot, backend = native_bot()
    bot.stuck_recovery.begin_reverse(40)
    bot._execute_rules(analysis(navigation_blocked=True, torpedoes_incoming=True), 41)
    assert bot.opening_autopilot_active
    assert bot.stuck_recovery.reverse_until is None
    assert backend.events == []


def test_throttle_only_preserves_actual_and_cached_rudder():
    backend = TelegraphBackend(actual_notch=4, actual_rudder=-2)
    controller = KeyboardController(backend)
    controller._rudder_notch = 2
    controller.set_throttle(.5)
    assert backend.actual_notch == 2
    assert backend.actual_rudder == -2
    assert controller._rudder_notch == 2
    assert all(key in {"w", "s"} for _, key in backend.events)


@pytest.mark.parametrize("changes", [dict(), dict(player_position=None), dict(player_pose_cached=True)])
def test_armed_reverse_prevents_forward_planning_even_when_pose_is_missing(monkeypatch, changes):
    bot, backend = native_bot()
    bot.opening_autopilot_active = False
    bot.stuck_recovery.begin_reverse(100)
    monkeypatch.setattr(bot.movement, "plan", lambda _: pytest.fail("planned during reverse"))
    for t in (100, 101, 110):
        bot._execute_rules(analysis(**changes), t)
        assert backend.actual_notch == -4
    assert ("tap", "w") not in backend.events


def test_reverse_stops_before_replanning_and_waits_for_live_pose():
    bot, backend = native_bot()
    bot.opening_autopilot_active = False
    zone = CaptureZone(center=(500, 500), radius=50)
    bot.route_planner.zone = zone
    bot.route_planner.update((100, 100))
    old_waypoint = bot.route_planner.entry_waypoint
    bot._battle_map_islands = ["retained terrain"]
    bot.stuck_recovery.begin_reverse(100)
    bot._execute_rules(analysis(), 100)
    backend.notch_history.clear()
    bot._execute_rules(analysis(), 112)
    assert backend.actual_notch == 0
    assert max(backend.notch_history) == 0
    assert bot.route_planner.zone is zone
    assert bot.route_planner.entry_waypoint is None
    assert bot.route_planner.start_position is None
    assert bot._battle_map_islands == ["retained terrain"]
    for t, changes in [(113, dict(player_position=None)), (114, dict(player_pose_cached=True))]:
        bot._execute_rules(analysis(**changes), t)
        assert backend.actual_notch == 0
    bot.route_planner.update((700, 500))
    assert bot.route_planner.start_position == (700, 500)
    assert bot.route_planner.entry_waypoint != old_waypoint
    bot._execute_rules(analysis(player_position=(700, 500)), 115)
    assert backend.actual_notch > 0


def test_user_pause_prevents_even_reverse_input():
    bot, backend = native_bot()
    bot.stuck_recovery.begin_reverse(100)
    bot.intervention = SimpleNamespace(poll=lambda *_: True)
    bot._execute_rules(analysis(), 101)
    assert backend.events == []
    assert bot.stuck_recovery.reverse_until is None


def test_failed_stop_retries_replan_without_replaying_forward_route(monkeypatch):
    bot, backend = native_bot()
    bot.opening_autopilot_active = False
    bot.stuck_recovery.begin_reverse(100)
    bot._execute_rules(analysis(), 100)
    send = bot.gamepad.set_movement

    def fail(*args):
        raise RuntimeError("focus unavailable")

    monkeypatch.setattr(bot.gamepad, "set_movement", fail)
    with pytest.raises(RuntimeError, match="focus unavailable"):
        bot._execute_rules(analysis(), 112)
    assert backend.actual_notch == -4
    assert bot.stuck_recovery.reverse_until is not None
    monkeypatch.setattr(bot.gamepad, "set_movement", send)
    monkeypatch.setattr(bot.movement, "plan", lambda _: pytest.fail("planned before stopping"))
    bot._execute_rules(analysis(), 113)
    assert backend.actual_notch == 0
    assert bot._last_movement_mode == "recovery:replan"


@pytest.mark.parametrize("rudder", [-2, -1, 0, 1, 2])
def test_native_reverse_neutralizes_unknown_real_rudder_without_forward(rudder):
    backend = TelegraphBackend(actual_notch=4, actual_rudder=rudder)
    controller = KeyboardController(backend)
    controller.reverse_escape()
    assert backend.actual_rudder == 0
    assert backend.actual_notch == -4
    assert ("tap", "w") not in backend.events
    count = len(backend.events)
    controller.reverse_escape()
    assert len(backend.events) == count
