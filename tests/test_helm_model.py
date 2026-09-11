import cv2
import numpy as np
import pytest
import math
from types import SimpleNamespace

from bot import BattleBot, BattleAnalysis
from core.vision import CaptureZone, PlayerPose, Vision
from strategy.helm_model import HelmModel, MapMotion
from strategy.secondary_movement import SecondaryMovementController, SecondaryMovementInput, MovementMode
from tools.calibrate_navigation import calibrate


def test_rudder_reversal_takes_time_and_uses_real_notch():
    helm = HelmModel(15)
    helm.command(0.4, 0)
    assert helm.advance(3) == pytest.approx(0.2)
    assert helm.advance(15) == 0.5
    helm.command(-1, 15)
    assert helm.advance(22.5) == pytest.approx(0)
    assert helm.advance(37.5) == -1


def test_motion_rejects_jump_and_expires():
    motion = MapMotion()
    motion.observe((0.5, 0.5), 0, 40)
    motion.observe((0.506, 0.5), 3, 40)
    assert motion.fresh_speed(3) == pytest.approx(0.08)
    assert motion.fresh_speed(9) is None
    motion.observe((0.9, 0.1), 6, 40)
    assert motion.fresh_speed(6) is None


def test_cap_interior_land_is_not_erased(monkeypatch):
    vision = Vision()
    frame = np.full((420, 420, 3), (72, 48, 30), np.uint8)
    cv2.circle(frame, (210, 210), 85, (220, 220, 220), 2)
    cv2.rectangle(frame, (230, 180), (262, 226), (195, 195, 195), -1)
    monkeypatch.setattr(vision, 'find_capture_zones', lambda *a, **k: [CaptureZone((210, 210), 85)])
    monkeypatch.setattr(vision, 'find_player_pose_on_minimap', lambda *a: None)
    outlines = vision.find_minimap_island_outlines(frame)
    assert any(cv2.pointPolygonTest(np.asarray(o['points'], np.float32), (245/420, 200/420), False) >= 0 for o in outlines)


def test_game_scale_changes_forecast_travel_distance():
    kwargs = dict(horizon_seconds=30, safety_clearance_km=0.2)
    real = Vision.plan_kinematic_rudder((500,500), PlayerPose((250,250),(1,0)), (450,250), [], **kwargs)
    game = Vision.plan_kinematic_rudder((500,500), PlayerPose((250,250),(1,0)), (450,250), [], speed_scale=5.22, **kwargs)
    assert (game.predicted_endpoint[0] - 250/499) / (real.predicted_endpoint[0] - 250/499) == pytest.approx(5.22)


def test_map_span_changes_physical_distance():
    bot = BattleBot(1, {'strategy': {'map_span_km': 42}}, vision=Vision(), gamepad=object())
    assert bot._minimap_pixels_to_km(np.zeros((420,420,3)), 42) == pytest.approx(4.2)


def test_enemy_pursuit_does_not_require_central_cap():
    controller = SecondaryMovementController(pursue_enemies=True)
    command = controller.plan(SecondaryMovementInput(elapsed=60, health=1, visible_target=True,
        minimap_target_bearing=-0.3, minimap_distance_km=20, capture_point_bearing=0.8,
        kinematic_rudder=-0.5))
    assert command.mode == MovementMode.APPROACH
    assert command.throttle == 1
    assert command.rudder == -0.5


def test_enemy_pursuit_still_yields_to_island():
    controller = SecondaryMovementController(pursue_enemies=True)
    command = controller.plan(SecondaryMovementInput(elapsed=1, health=1, visible_target=True,
        minimap_target_bearing=0.3, kinematic_rudder=-1, kinematic_avoidance_required=True,
        kinematic_collision_time_seconds=8))
    assert command.mode == MovementMode.AVOID_ISLAND
    assert command.rudder == -1


def test_enemy_target_is_not_blended_back_toward_cap():
    bot = BattleBot(1, {'strategy': {'pursue_enemies': True}}, vision=object(), gamepad=object())
    target, _ = bot._kinematic_target_normalized(BattleAnalysis(
        image=None, width=500, height=500,
        nearest_enemy_normalized=(0.2,0.4), navigation_target_normalized=(0.8,0.9),
        minimap_target_bearing=-0.6))
    assert target == (0.2,0.4)


def test_calibration_recovers_known_steady_circle():
    rows = [dict(time_seconds=t, x=0.5+math.cos(t*0.04)/50,
                 y=0.5+math.sin(t*0.04)/50, heading_degrees=90+math.degrees(t*0.04),
                 speed_knots=30) for t in range(0, 21, 5)]
    result = calibrate(rows, 50)
    assert result['steady_turn_radius_km'] == pytest.approx(1)
    assert result['qe_speed_scale'] == pytest.approx(0.04/(30*1.852/3600), abs=0.0001)


def test_native_route_keeps_ownership_when_only_terrain_prediction_is_unsafe(monkeypatch):
    bot = BattleBot(1, {'strategy': {'pursue_enemies': True}}, vision=object(), gamepad=object())
    bot.intervention = SimpleNamespace(poll=lambda *args: False)
    bot.opening_autopilot_active = True
    bot._battle_map_islands = [{'points': [[0.1,0.1],[0.2,0.1],[0.2,0.2]]}]
    monkeypatch.setattr(bot, '_execute_survival_consumables', lambda *args: True)
    called = []
    monkeypatch.setattr(bot, 'request_autopilot_retry', called.append)
    bot._execute_rules(BattleAnalysis(image=None, width=500, height=500,
        kinematic_avoidance_required=True), 50)
    assert called == []
