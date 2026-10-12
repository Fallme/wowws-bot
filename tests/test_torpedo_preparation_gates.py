"""Close-range preparation and main-battery restoration use their own evidence."""

from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

import bot as bot_module
from core.ocr import OcrToken, RapidOcrBackend
from core.torpedo_hud import detect_main_gun_selected, detect_torpedo_weapon
from core.ui import ScreenState
from test_torpedo_hud import analysis, policy


ROOT = Path(__file__).resolve().parent
LIVE_MAIN2 = Path("E:/aimemo/docs/screenshots/wows_driving_trials/run_20261005_142256_1791181376552983600/full_0100.png")


def preparation_reason(policy):
    events = [call for call in policy.bot.events.publish.call_args_list
              if call.args[0] == "torpedo.preparation_skipped"]
    assert len(events) == 1
    reason = events[0].kwargs["reason"]
    evidence = policy.bot._record_torpedo_frame.call_args_list[-1]
    assert evidence.args[1] == "result"
    assert evidence.kwargs["preparing"]
    assert evidence.kwargs["cancel_reason"] == reason
    assert evidence.args[0] is not None
    return reason


def test_unassociated_navigation_contact_cannot_veto_a_custom_viewport_attack(policy):
    policy.bot.ship = {"type": "CUSTOM", "has_torpedoes": False}
    current = analysis()
    current.minimap_distance_km = 18
    policy.bot._try_torpedo_attack(current, 1000)
    assert policy.detect.call_count == 2
    policy.bot.gamepad.fire_torpedo.assert_called_once()


def test_six_kilometre_boundary_still_requires_both_fresh_same_track_readings(policy):
    current = analysis()
    current.target_distance_km = 6
    current.minimap_distance_km = 19
    policy.observation.value_km = 6
    policy.observation.target_track_id = current.target_track_id
    policy.bot._try_torpedo_attack(current, 1000)
    assert policy.detect.call_count == 2
    assert all(call.args[2] == current.target_track_id for call in policy.bot.distance_reader.read.call_args_list)
    policy.bot.gamepad.fire_torpedo.assert_called_once()


def test_edge_label_outside_supported_geometry_is_explicitly_skipped(policy):
    current = analysis()
    current.target_viewport_point = (944, 300)
    policy.bot._try_torpedo_attack(current, 1000)
    assert preparation_reason(policy) == "target_outside_aim_region"
    policy.detect.assert_not_called()
    policy.bot.gamepad.torpedo.assert_not_called()
    policy.bot.gamepad.fire_torpedo.assert_not_called()


@pytest.mark.parametrize("second", [False, None])
def test_second_capability_failure_keeps_a_preparation_reason_and_its_actual_frame(policy, second):
    policy.detect.side_effect = [True, second]
    policy.bot._try_torpedo_attack(analysis(), 1000)
    assert preparation_reason(policy) == "capability_not_confirmed"
    assert policy.bot._record_torpedo_frame.call_args.args[0] is policy.frames[-1]
    assert policy.bot._record_torpedo_frame.call_args.kwargs["weapon"] is second
    policy.bot.gamepad.torpedo.assert_not_called()


def test_friendly_label_overlap_rejection_is_preserved_and_sends_no_weapon_input(policy):
    policy.observation.accepted = False
    policy.observation.reject_reason = "friendly_label_overlap"
    policy.bot._try_torpedo_attack(analysis(), 1000)
    assert preparation_reason(policy) == "friendly_label_overlap"
    policy.bot.gamepad.torpedo.assert_not_called()
    policy.bot.gamepad.fire_torpedo.assert_not_called()


def test_normal_approach_uses_consecutive_current_ranges_instead_of_old_async_median(policy):
    current = analysis()
    current.target_distance_km = 4.65
    policy.observation.value_km = 4.0
    following = SimpleNamespace(accepted=True, value_km=3.9, confidence=.95)
    policy.bot.distance_reader.read.side_effect = [policy.observation, following, following, following]
    policy.bot._try_torpedo_attack(current, 1000)
    policy.bot.gamepad.fire_torpedo.assert_called_once()


def test_old_close_candidate_cannot_authorize_a_fresh_out_of_range_target(policy):
    current = analysis()
    current.target_distance_km = 5.8
    policy.observation.value_km = 6.01
    policy.bot._try_torpedo_attack(current, 1000)
    assert preparation_reason(policy) == "target_outside_torpedo_range"
    policy.bot.gamepad.torpedo.assert_not_called()


@pytest.mark.parametrize("failure,reason", [("track", "target_track_changed"),
                                           ("distance", "target_distance_changed"),
                                           ("range", "target_outside_torpedo_range")])
def test_second_fresh_target_must_keep_its_track_distance_and_range(policy, failure, reason):
    changed = SimpleNamespace(accepted=True, value_km=2, confidence=.95, target_track_id="enemy-1")
    if failure == "track":
        changed.target_track_id = "other-enemy"
    elif failure == "distance":
        changed.value_km = 2.8
    else:
        changed.value_km = 6.1
    policy.bot.distance_reader.read.side_effect = [policy.observation, changed]
    policy.bot._try_torpedo_attack(analysis(), 1000)
    assert preparation_reason(policy) == reason
    policy.bot.gamepad.torpedo.assert_not_called()


def test_nearby_competing_enemy_labels_are_diagnosed_before_slot_three(policy):
    policy.bot.vision.find_enemies_in_viewport.return_value = [(500, 300), (506, 300)]
    policy.bot._try_torpedo_attack(analysis(), 1000)
    assert preparation_reason(policy) == "viewport_target_ambiguous"
    policy.bot.gamepad.torpedo.assert_not_called()


@pytest.mark.parametrize("failure,reason", [("page", "battle_frame_unconfirmed"),
                                           ("health", "alive_health_unconfirmed"),
                                           ("client", "client_size_changed")])
def test_invalid_preparation_frame_is_saved_with_its_exact_gate_reason(policy, failure, reason):
    if failure == "page":
        policy.bot.vision.classify_screen.return_value = ScreenState.RESULTS
    elif failure == "health":
        policy.bot.vision.read_health_fraction.return_value = None
    else:
        policy.rect.return_value["width"] = 999
    policy.bot._try_torpedo_attack(analysis(), 1000)
    assert preparation_reason(policy) == reason
    assert policy.bot._record_torpedo_frame.call_args.args[0] is policy.frames[-1]
    policy.bot.gamepad.torpedo.assert_not_called()


def test_preparation_rejection_diagnostics_obey_the_attempt_retry_interval(policy):
    current = analysis()
    current.target_viewport_point = (944, 300)
    policy.bot._try_torpedo_attack(current, 1000)
    policy.bot._try_torpedo_attack(current, 1001)
    assert preparation_reason(policy) == "target_outside_aim_region"
    assert policy.bot._record_torpedo_frame.call_count == 1


class BadgeBackend:
    def __init__(self, label="2", confidence=.99):
        self.label = label
        self.confidence = confidence

    def recognize(self, image):
        ys, xs = np.nonzero(np.min(image, axis=2) > 230)
        if not len(xs):
            return []
        x1, x2, y1, y2 = float(xs.min()), float(xs.max() + 1), float(ys.min()), float(ys.max() + 1)
        return [OcrToken(self.label, self.confidence, ((x1, y1), (x2, y1), (x2, y2), (x1, y2)))]


def selected_frame(*, solid=False, badge=True, frame=True, count_inside=False):
    image = np.full((1080, 1920, 3), 30, np.uint8)
    if frame:
        cv2.rectangle(image, (890, 945), (956, 1007), (20, 205, 20), -1 if solid else 2)
    if badge:
        cv2.rectangle(image, (911, 1007), (934, 1031), (20, 205, 20), -1)
        cv2.putText(image, "2", (917, 1026), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)
    if count_inside:
        cv2.putText(image, "2", (917, 980), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)
    return image


def test_main_battery_requires_the_selected_frame_and_its_own_green_two_badge():
    assert detect_main_gun_selected(selected_frame(), BadgeBackend()) is True


@pytest.mark.parametrize("image", [selected_frame(solid=True), selected_frame(frame=False),
                                   selected_frame(badge=False, count_inside=True)])
def test_solid_green_box_isolated_badge_and_interior_count_are_not_selected_main_guns(image):
    assert detect_main_gun_selected(image, BadgeBackend()) is not True


def test_selected_three_or_low_confidence_two_cannot_confirm_main_battery_restore():
    assert detect_main_gun_selected(selected_frame(), BadgeBackend("3")) is False
    assert detect_main_gun_selected(selected_frame(), BadgeBackend(confidence=.6)) is None
    assert detect_main_gun_selected(selected_frame(), None) is None


@pytest.fixture(scope="module")
def actual_backend():
    return RapidOcrBackend()


@pytest.mark.parametrize("name,expected", [("battle_low_contrast_1440.png", True),
                                          ("battle_long_run.png", True),
                                          ("live_battle.png", False)])
def test_actual_selected_main_battery_and_selected_torpedo_pngs(name, expected, actual_backend):
    image = cv2.imread(str(ROOT / "fixtures" / name))
    assert detect_main_gun_selected(image, actual_backend) is expected


@pytest.mark.skipif(not LIVE_MAIN2.exists(), reason="Retained historical live PNG")
@pytest.mark.parametrize("resolution", [.5, .75, 1])
def test_actual_main_battery_selection_across_resolutions(resolution, actual_backend):
    image = cv2.imread(str(LIVE_MAIN2))
    if resolution != 1:
        image = cv2.resize(image, None, fx=resolution, fy=resolution, interpolation=cv2.INTER_AREA)
    assert detect_main_gun_selected(image, actual_backend) is True


@pytest.mark.skipif(not LIVE_MAIN2.exists(), reason="Retained historical live PNG")
@pytest.mark.parametrize("scale", [.75, 1.25])
def test_actual_main_battery_selection_follows_independent_moved_weapon_ui(scale, actual_backend):
    source = cv2.imread(str(LIVE_MAIN2))
    image = source.copy()
    image[1390:, 820:1780] = (70, 75, 80)
    row = cv2.resize(source[1440:1600, 850:1700], None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    left, top = round(1280 - row.shape[1] / 2 + 30), 1570 - row.shape[0]
    image[top:top + row.shape[0], left:left + row.shape[1]] = row
    assert detect_main_gun_selected(image, actual_backend) is True


def test_actual_slot_two_clears_restore_pending_even_when_torpedo_reload_hides_slot_three(policy, monkeypatch, actual_backend):
    image = cv2.imread(str(ROOT / "fixtures" / "battle_long_run.png"))
    assert detect_torpedo_weapon(image, actual_backend) is None
    monkeypatch.setattr(bot_module, "detect_main_gun_selected", detect_main_gun_selected)
    policy.bot.distance_reader.backend = actual_backend
    policy.bot.vision.grab.side_effect = lambda *_args, **_kwargs: image.copy()
    policy.rect.return_value.update(width=image.shape[1], height=image.shape[0])
    policy.bot._torpedo_restore_pending = True
    assert policy.bot._restore_torpedo_guns()
    assert not policy.bot._torpedo_restore_pending
    assert policy.bot._record_torpedo_frame.call_args.kwargs["main_gun_selected"] is True


def test_unconfirmed_main_battery_keeps_restore_pending_without_a_new_torpedo_attempt(policy, monkeypatch):
    monkeypatch.setattr(bot_module, "detect_main_gun_selected", lambda *_: None)
    policy.bot._torpedo_restore_pending = True
    policy.bot._try_torpedo_attack(analysis(), 1000)
    assert policy.bot._torpedo_restore_pending
    policy.bot.gamepad.torpedo.assert_not_called()
    policy.bot.gamepad.fire_torpedo.assert_not_called()
