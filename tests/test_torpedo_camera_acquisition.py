"""Camera search uses current visual proof and never borrows steering/fire authority."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

import bot as bot_module
import main
from bot import BattleAnalysis, BattleBot
from core.ui import ScreenState
from core.vision import PlayerPose


@pytest.fixture
def camera(monkeypatch):
    instance = object.__new__(BattleBot)
    instance.strategy = {}
    instance.ship = {"has_torpedoes": True, "torpedo": {"range": 6}}
    instance.map_span_km = 50
    instance.hwnd = 7
    instance._torpedo_restore_pending = False
    instance._battle_capture_zone_layout = []
    instance._reset_torpedo_acquisition()
    clock = {"value": 1000.}
    controls = ("aim_relative", "torpedo", "main_gun", "lock", "fire_torpedo",
                "full_speed", "rudder", "pause_automation", "stop")
    instance.gamepad = SimpleNamespace(**{name: Mock(return_value=True) for name in controls})
    instance.intervention = SimpleNamespace(poll=Mock(return_value=False),
                                           command_generation_paused=Mock(return_value=False))
    instance.mark_manual_pause = Mock()
    instance.events = SimpleNamespace(publish=Mock())
    instance._record_target_acquisition_frame = Mock()
    frames = []
    def grab(*_args, **_kwargs):
        frame = np.zeros((600, 1000, 3), np.uint8)
        frames.append(frame)
        return frame
    minimap = np.zeros((500, 500, 3), np.uint8)
    instance.vision = SimpleNamespace(grab=Mock(side_effect=grab),
        classify_screen=Mock(return_value=ScreenState.BATTLE),
        read_health_fraction=Mock(return_value=.8),
        find_enemies_in_viewport=Mock(return_value=[]),
        find_minimap=Mock(return_value=minimap),
        find_player_pose_on_minimap=Mock(return_value=PlayerPose((250, 250), (0., -1.))),
        analyze_minimap=Mock(return_value=([(250, 200)], False)),
        minimap_pixels_to_km=lambda _map, distance: distance / 10)
    instance.distance_reader = SimpleNamespace(backend=object())
    current = BattleAnalysis(image=np.zeros((600, 1000, 3), np.uint8), width=1000, height=600,
        in_battle=True, health=.8, health_recognized=True, minimap_distance_km=5,
        minimap_player_normalized=(.5, .5), minimap_heading=(0., -1.),
        nearest_enemy_normalized=(.5, .4),
        minimap_contacts=[{"position": (.5, .4), "kind": "enemy"}])
    rect = Mock(return_value={"width": 1000, "height": 600})
    detector = Mock(return_value=True)
    monkeypatch.setattr(bot_module, "get_client_rect", rect)
    monkeypatch.setattr(bot_module, "detect_torpedo_weapon", detector)
    monkeypatch.setattr(bot_module.time, "monotonic", lambda: clock["value"])
    def tick():
        result = instance._acquire_torpedo_target(current, clock["value"])
        clock["value"] += 2
        return result
    return SimpleNamespace(bot=instance, analysis=current, clock=clock, rect=rect,
                           detector=detector, frames=frames, tick=tick)


def ready(camera):
    assert not camera.tick()
    assert not camera.tick()


def assert_camera_only(camera, moves=0):
    assert camera.bot.gamepad.aim_relative.call_count == moves
    for name in ("torpedo", "main_gun", "lock", "fire_torpedo", "full_speed", "rudder",
                 "pause_automation", "stop"):
        getattr(camera.bot.gamepad, name).assert_not_called()
    assert camera.analysis.target_distance_km is None
    assert camera.analysis.target_track_id is None


def test_three_current_contacts_take_one_camera_step_and_keep_fire_range_unknown(camera):
    ready(camera)
    assert camera.tick()
    assert_camera_only(camera, 1)
    camera.bot.gamepad.aim_relative.assert_called_once_with(100, 0)
    assert len(camera.frames) == 2
    assert all(frame is not camera.analysis.image for frame in camera.frames)
    assert [call.args[1] for call in camera.bot._record_target_acquisition_frame.call_args_list] == ["before", "after"]


@pytest.mark.parametrize("field,value", [
    ("in_battle", False), ("ended", True), ("health", 0), ("health", None),
    ("health_recognized", False), ("player_pose_cached", True),
    ("minimap_player_normalized", None), ("nearest_enemy_normalized", None),
    ("minimap_distance_km", 6.01), ("minimap_distance_km", float("nan")),
    ("minimap_contacts", []), ("visible_target", True),
    ("enemies", [(700, 300)]), ("target_viewport_point", (700, 300)),
])
def test_unknown_dead_cached_distant_or_already_visible_target_never_moves(camera, field, value):
    setattr(camera.analysis, field, value)
    if field == "enemies":
        camera.analysis.enemy_source = "viewport"
    for _ in range(4):
        assert not camera.tick()
    assert_camera_only(camera)
    camera.bot.vision.grab.assert_not_called()


@pytest.mark.parametrize("setting", ["auto_torpedo", "torpedo_camera_acquisition"])
def test_disabled_feature_never_grabs_or_moves(camera, setting):
    camera.bot.strategy[setting] = False
    for _ in range(3):
        camera.tick()
    camera.bot.vision.grab.assert_not_called()
    assert_camera_only(camera)


def test_stable_minimap_exact_six_boundary_still_needs_fresh_contact_range(camera):
    camera.analysis.minimap_distance_km = 6
    camera.analysis.nearest_enemy_normalized = (.5, .38)
    camera.analysis.minimap_contacts[0]["position"] = (.5, .38)
    camera.bot.vision.analyze_minimap.return_value = ([(250, 190)], False)
    ready(camera)
    assert camera.tick()
    assert_camera_only(camera, 1)


@pytest.mark.parametrize("failure,reason", [
    ("menu", "battle_frame_unconfirmed"), ("dead", "alive_health_unconfirmed"),
    ("unknown_hp", "alive_health_unconfirmed"), ("resize", "client_size_changed"),
    ("pose_missing", "fresh_contact_unconfirmed"), ("pose_jumped", "fresh_contact_unconfirmed"),
    ("contact_missing", "fresh_contact_unconfirmed"), ("contact_changed", "fresh_contact_unconfirmed"),
    ("contact_ambiguous", "fresh_contact_unconfirmed"), ("now_distant", "fresh_contact_unconfirmed"),
    ("wrong_weapon", "capability_not_confirmed"),
])
def test_fresh_frame_rejection_cancels_search_without_any_control(camera, failure, reason):
    ready(camera)
    bot = camera.bot
    if failure == "menu":
        bot.vision.classify_screen.return_value = ScreenState.RESULTS
    elif failure in {"dead", "unknown_hp"}:
        bot.vision.read_health_fraction.return_value = 0 if failure == "dead" else None
    elif failure == "resize":
        camera.rect.return_value = {"width": 1001, "height": 600}
    elif failure in {"pose_missing", "pose_jumped"}:
        bot.vision.find_player_pose_on_minimap.return_value = (
            None if failure == "pose_missing" else PlayerPose((300, 250), (0., -1.)))
    elif failure in {"contact_missing", "contact_changed", "contact_ambiguous", "now_distant"}:
        points = {"contact_missing": [], "contact_changed": [(350, 200)],
                  "contact_ambiguous": [(250, 200), (252, 200)], "now_distant": [(250, 185)]}[failure]
        bot.vision.analyze_minimap.return_value = (points, False)
    else:
        camera.detector.return_value = None
    assert not camera.tick()
    assert_camera_only(camera)
    assert bot._record_target_acquisition_frame.call_args.kwargs["cancel_reason"] == reason


def test_known_capture_point_cannot_become_a_fresh_close_ship_contact(camera):
    ready(camera)
    camera.bot._battle_capture_zone_layout = [{"position": (.5, .4)}]
    assert not camera.tick()
    assert_camera_only(camera)


def test_real_target_in_fresh_frame_stops_search_even_without_any_distance_ocr(camera):
    ready(camera)
    camera.bot.vision.find_enemies_in_viewport.return_value = [(770, 300)]
    assert camera.tick()  # Old empty viewport cannot drive this tick's controls.
    assert_camera_only(camera)
    camera.detector.assert_not_called()
    assert camera.bot._record_target_acquisition_frame.call_args.kwargs["cancel_reason"] == "viewport_target_visible"


def test_target_found_after_move_prevents_the_next_step_while_ocr_catches_up(camera):
    ready(camera)
    camera.bot.vision.find_enemies_in_viewport.side_effect = [[], [(770, 300)]]
    assert camera.tick()
    camera.analysis.enemies = [(770, 300)]
    camera.analysis.enemy_source = "viewport"
    for _ in range(5):
        assert not camera.tick()
    assert_camera_only(camera, 1)


def test_production_minimap_fallback_enemies_do_not_masquerade_as_visible_viewport(camera):
    # analyze() exports red minimap points through enemies when the main view
    # is empty. Only its source distinguishes these from real viewport labels.
    camera.analysis.enemies = [(250, 200)]
    camera.analysis.enemy_source = "minimap"
    ready(camera)
    assert camera.tick()
    assert_camera_only(camera, 1)


@pytest.mark.parametrize("slow", ["hp", "weapon", "evidence"])
def test_recognition_and_evidence_latency_cannot_send_a_stale_camera_step(camera, slow):
    ready(camera)
    def delayed(*_args, **_kwargs):
        camera.clock["value"] += 4.01
        return .8 if slow == "hp" else True
    if slow == "hp":
        camera.bot.vision.read_health_fraction.side_effect = delayed
    elif slow == "weapon":
        camera.detector.side_effect = delayed
    else:
        camera.bot._record_target_acquisition_frame.side_effect = delayed
    assert not camera.tick()
    assert_camera_only(camera)


@pytest.mark.parametrize("failure", ["pause", "resize"])
def test_last_input_boundary_honors_intervention_and_resize(camera, failure):
    ready(camera)
    def record(_frame, phase, **_kwargs):
        if phase == "before":
            if failure == "pause":
                camera.bot.intervention.command_generation_paused.return_value = True
            else:
                camera.rect.return_value = {"width": 999, "height": 600}
    camera.bot._record_target_acquisition_frame.side_effect = record
    assert not camera.tick()
    assert_camera_only(camera)


def test_contact_change_or_long_gap_restarts_confirmation_without_buying_budget(camera):
    ready(camera)
    camera.clock["value"] += 9
    assert not camera.tick()
    assert not camera.tick()
    camera.analysis.nearest_enemy_normalized = (.6, .4)
    camera.analysis.minimap_contacts[0]["position"] = (.6, .4)
    assert not camera.tick()
    assert_camera_only(camera)


def test_each_episode_and_whole_match_are_bounded_with_retry_and_resume(camera):
    resumed = False
    for _ in range(70):
        camera.tick()
        if camera.bot._torpedo_acquisition_moves == 10 and not resumed:
            camera.bot._reset_torpedo_acquisition(preserve_budget=True)
            resumed = True
    assert_camera_only(camera, 24)
    dxs = [call.args[0] for call in camera.bot.gamepad.aim_relative.call_args_list]
    assert dxs == [100] * 8 + [-100] * 16
    camera.bot._reset_torpedo_acquisition(preserve_budget=True)
    for _ in range(5):
        camera.tick()
    assert_camera_only(camera, 24)


def test_confirmed_round_closure_grants_new_budget_before_diagnostic_storage_failure(camera):
    camera.bot._torpedo_acquisition_moves = 24
    camera.bot.complete_round_diagnostics = Mock(side_effect=OSError("storage"))
    main.finalize_round_diagnostics(camera.bot, 1, outcome="victory")
    assert camera.bot._torpedo_acquisition_moves == 0
    assert camera.bot._torpedo_acquisition_started_at is None
    ready(camera)
    assert camera.tick()
    assert_camera_only(camera, 1)


def test_even_fast_camera_input_defers_weapons_and_native_helm_until_fresh_analysis(camera):
    ready(camera)
    camera.bot._manual_intervention_active = False
    camera.bot._execute_survival_consumables = Mock(return_value=True)
    camera.bot._try_torpedo_attack = Mock()
    camera.bot.opening_autopilot_active = True
    camera.bot._execute_rules(camera.analysis, camera.clock["value"])
    assert_camera_only(camera, 1)
    camera.bot._try_torpedo_attack.assert_not_called()


def test_uncertain_mouse_ack_uses_a_move_budget_and_cooldown_without_any_other_input(camera):
    ready(camera)
    camera.bot.gamepad.aim_relative.return_value = False
    assert camera.tick()
    for _ in range(4):
        assert not camera.tick()
    assert_camera_only(camera, 1)
    assert camera.bot._torpedo_acquisition_moves == 1
    assert camera.bot._record_target_acquisition_frame.call_args.kwargs["cancel_reason"] == "camera_input_rejected"
