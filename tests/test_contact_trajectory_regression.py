import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from bot import BattleAnalysis, BattleBot
from core.events import RuntimeEvent
from core.keyboard import KeyboardController
from core.trajectory import TrajectoryRecorder
from core.vision import Vision
from tests.test_keyboard_controller import TelegraphBackend


def native_bot():
    backend = TelegraphBackend(actual_notch=4)
    controls = KeyboardController(backend)
    bot = BattleBot(1, {"secondary": {"range": 11.4}, "strategy": {}}, vision=object(), gamepad=controls)
    bot.intervention = SimpleNamespace(poll=lambda *_: False)
    bot.enable_opening_autopilot("test", target_normalized=(.5, .5))
    bot._native_autopilot_started_at = 0.0
    return bot, backend


def analysis(**kwargs):
    values = dict(image=None, width=1000, height=1000, in_battle=True,
                  player_position=(500, 500), minimap_player_normalized=(.5, .5),
                  minimap_heading=(1., 0.), speed_knots=30)
    values.update(kwargs)
    return BattleAnalysis(**values)


@pytest.mark.parametrize("changes", [dict(), dict(kinematic_avoidance_required=True),
    dict(nearest_enemy_normalized=(.7, .5), minimap_target_bearing=.2, minimap_distance_km=28),
    dict(nearest_enemy_normalized=(.7, .5), minimap_target_bearing=.2, minimap_distance_km=10, player_pose_cached=True)])
def test_native_route_never_yields_for_arrival_terrain_far_or_cached_contact(changes):
    bot, backend = native_bot()
    for t in range(40, 45):
        bot._execute_rules(analysis(**changes), float(t))
    assert bot.opening_autopilot_active
    assert backend.events == []


def test_close_contact_requires_three_fresh_consecutive_frames_and_one_handoff():
    bot, backend = native_bot()
    bot._native_spawn_position = (.2, .5)
    frame = analysis(minimap_player_normalized=(.55, .5), nearest_enemy_normalized=(.7, .5), minimap_target_bearing=.2, minimap_distance_km=10)
    for t in (40, 41):
        bot._execute_rules(frame, t)
        assert bot.opening_autopilot_active
    bot._execute_rules(frame, 42)
    assert not bot.opening_autopilot_active
    assert bot.autopilot_retry_pending
    assert "10.0km" in bot.last_movement_reason
    assert bot.stuck_recovery.reverse_until is None
    count = len(backend.events)
    bot.request_autopilot_retry("duplicate")
    assert len(backend.events) == count


def test_contact_sampling_gap_does_not_count_as_continuous_contact():
    bot, _ = native_bot()
    bot._native_spawn_position = (.2, .5)
    frame = analysis(minimap_player_normalized=(.55, .5), nearest_enemy_normalized=(.7, .5), minimap_target_bearing=.2, minimap_distance_km=10)
    for t in (40, 41, 50):
        bot._execute_rules(frame, t)
    assert bot.opening_autopilot_active


@pytest.mark.parametrize("spawn,near_mid,across", [
    ((.2, .5), (.51, .5), (.55, .5)),
    ((.8, .5), (.49, .5), (.45, .5)),
    ((.5, .2), (.5, .51), (.5, .55)),
    ((.5, .8), (.5, .49), (.5, .45)),
])
def test_native_route_keeps_ownership_until_enemy_half(spawn, near_mid, across):
    bot, backend = native_bot()
    for t, pos in enumerate([spawn] * 3 + [near_mid] * 4, 100):
        bot._execute_rules(analysis(minimap_player_normalized=pos,
            nearest_enemy_normalized=(.6, .6), minimap_target_bearing=.2,
            minimap_distance_km=5), float(t))
        assert bot.opening_autopilot_active
    assert backend.events == []
    for t in (107., 108., 109.):
        bot._execute_rules(analysis(minimap_player_normalized=across,
            nearest_enemy_normalized=(.6, .6), minimap_target_bearing=.2,
            minimap_distance_km=5), t)
    assert not bot.opening_autopilot_active


def test_native_stall_arms_bounded_reverse_then_returns_to_forward():
    bot, backend = native_bot()
    frame = analysis(speed_knots=.4)
    for t in range(30, 50):
        bot._execute_rules(frame, float(t))
        assert bot.opening_autopilot_active
    bot._execute_rules(frame, 50.)
    assert not bot.opening_autopilot_active
    assert bot.stuck_recovery.reverse_until == 62.
    for t in (51., 60.):
        bot._execute_rules(frame, t)
        assert backend.actual_notch == -4
        assert backend.actual_rudder == 0
    bot._execute_rules(frame, 62.)
    assert backend.actual_notch > 0
    assert bot.stuck_recovery.reverse_until is None


@pytest.mark.parametrize("enemy,bearing", [((.4, .5), 1.), ((.5, .6), .5), ((.51, .5), 0.)])
def test_in_range_target_does_not_hold_course_away_or_overshoot(enemy, bearing):
    bot, _ = native_bot()
    bot.strategy["pursue_enemies"] = True
    target, reason = bot._kinematic_target_normalized(analysis(
        nearest_enemy_normalized=enemy, minimap_distance_km=5, minimap_target_bearing=bearing))
    assert target == enemy
    assert reason == "敌舰"


def test_in_range_closing_course_can_hold_steady():
    bot, _ = native_bot()
    target, reason = bot._kinematic_target_normalized(analysis(
        nearest_enemy_normalized=(.6, .5), minimap_distance_km=5, minimap_target_bearing=0.))
    assert target[0] > .5
    assert target[1] == .5
    assert reason == "接战稳舵"


def test_navigation_target_lock_survives_nearest_contact_changes_and_expires():
    from core.tracking import NavigationTargetLock
    lock = NavigationTargetLock()
    assert lock.update([(.4, .5), (.6, .5)], (.4, .5), 1) == (.4, .5)
    assert lock.update([(.401, .5), (.59, .5)], (.59, .5), 2) == (.401, .5)
    assert lock.update([(.59, .5)], (.59, .5), 3) is None
    assert lock.update([(.59, .5)], (.59, .5), 4) is None
    assert lock.update([(.59, .5)], (.59, .5), 5) == (.59, .5)
    assert lock.update([], None, 10) is None


def test_after_island_override_aft_enemy_requires_turn_even_in_secondary_range():
    from strategy.secondary_movement import SecondaryMovementController, SecondaryMovementInput, MovementMode
    controller = SecondaryMovementController(pursue_enemies=True)
    values = dict(elapsed=180, health=1., visible_target=True,
                  minimap_distance_km=5., minimap_target_bearing=-.6)
    avoidance = controller.plan(SecondaryMovementInput(**values,
        kinematic_avoidance_required=True, kinematic_rudder=1.,
        kinematic_collision_time_seconds=5.))
    assert avoidance.mode == MovementMode.AVOID_ISLAND
    assert avoidance.rudder > 0
    pursuit = controller.plan(SecondaryMovementInput(**values))
    assert pursuit.rudder < 0
    assert pursuit.throttle > 0


def test_capture_hold_does_not_project_outside_zone():
    bot, _ = native_bot()
    target, reason = bot._kinematic_target_normalized(analysis(
        inside_capture_point=True, minimap_player_normalized=(.59, .5),
        capture_zone_center_normalized=(.5, .5), capture_zone_radius_normalized=.1,
        navigation_target_normalized=(.5, .5)))
    assert target == (.5, .5)
    assert reason != "点内稳舵"


def test_trajectory_flushes_records_and_does_not_connect_pose_gaps(tmp_path):
    recorder = TrajectoryRecorder(tmp_path)
    for index, (stamp, position, cached) in enumerate([(1, [.2, .3], False), (2, [.21, .3], False),
        (3, [.21, .3], True), (4, [.22, .3], False), (20, [.23, .3], False), (21, [.8, .3], False)]):
        recorder(RuntimeEvent(index, "battle.tick", stamp, dict(minimap_player=position,
            player_pose_cached=cached, autopilot_enabled=index < 2, movement_mode="qe")))
    assert recorder.connected(recorder.points[0], recorder.points[1])
    for i in range(2, 6):
        assert not recorder.connected(recorder.points[i-1], recorder.points[i])
    recorder.close()
    records = [json.loads(line) for line in (tmp_path / "trajectory.jsonl").read_text().splitlines()]
    assert len(records) == 6 and records[2]["position"] is None
    assert cv2.imread(str(tmp_path / "trajectory.png")).shape == (850, 800, 3)


SCREENSHOT = Path("E:/aimemo/tmp/codex-clipboard-060d7519-9835-450e-b17a-b36f849f9264.png")


@pytest.mark.skipif(not SCREENSHOT.exists(), reason="User screenshot is local regression evidence")
def test_user_four_caps_are_not_enemies_or_a_filled_island():
    minimap = cv2.imread(str(SCREENSHOT))[747:1423, 900:1575]
    vision = Vision()
    zones = vision.find_capture_zones(minimap, (517, 408))
    assert len(zones) == 4
    for zone, expected in zip(zones, [(210, 219), (283, 391), (390, 284), (453, 463)]):
        assert zone.center == pytest.approx(expected, abs=4)
    assert [z.state for z in zones] == ["hostile", "neutral", "neutral", "friendly"]
    assert vision.analyze_minimap(minimap)[0] == []
    mask = np.zeros(minimap.shape[:2], dtype=np.uint8)
    for island in vision.find_minimap_island_outlines(minimap):
        polygon = np.array([[round(x*674), round(y*675)] for x, y in island["points"]], np.int32)
        cv2.fillPoly(mask, [polygon], 255)
    assert mask[200, 200] == 0  # tinted open water inside A
    assert np.count_nonzero(mask[175:189, 222:237]) > 0  # real island inside A survives


@pytest.mark.skipif(not SCREENSHOT.exists(), reason="User screenshot is local regression evidence")
def test_existing_ocr_reads_all_four_located_letters_without_detector_loss():
    from core.ocr import RapidOcrBackend
    minimap = cv2.imread(str(SCREENSHOT))[747:1423, 900:1575]
    zones = Vision().find_capture_zones(minimap, (517, 408), RapidOcrBackend(prefer_gpu=False))
    assert [z.label for z in zones] == ["A", "B", "C", "D"]


def test_resume_focus_race_returns_to_scene_router_instead_of_failing_run(monkeypatch):
    import main
    from core.ui import ScreenState
    def lost_focus():
        raise RuntimeError("游戏窗口不在前台，拒绝发送键盘操作")
    bot = SimpleNamespace(hwnd=1, intervention=None,
        vision=SimpleNamespace(grab=lambda *a, **kw: np.zeros((90, 160, 3), np.uint8)),
        gamepad=SimpleNamespace(resynchronize_forward_controls=lost_focus))
    monkeypatch.setattr(main, "ensure_bound_game_foreground", lambda *_: True)
    monkeypatch.setattr(main, "normalize_tactical_map_overlay", lambda *_: True)
    monkeypatch.setattr(main, "classify_battle_continuity_screen", lambda *_: ScreenState.BATTLE)
    assert main.run_battle(bot, resume_existing=True) == "resume_state"
