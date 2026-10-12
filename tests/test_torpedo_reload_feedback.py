"""Reload feedback can cover the icon, without authorizing another shot.

Historical PNGs prove the counter layout, not that a torpedo was fired. The
positive replays place their untouched counter pixels inside the selected
slot of a separate real PNG; alignment is supplied only to model the caller's
already completed two-frame check.
"""

from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from core.ocr import RapidOcrBackend
from core.torpedo_hud import (
    TorpedoHud, _SlotEvidence, _cooldown_seconds, _parse_reload_time, confirm_torpedo_reload, detect_torpedo_weapon,
    inspect_torpedo_hud,
)
from test_torpedo_hud import analysis, policy


FIXTURES = Path(__file__).parent / "fixtures"
LIVE_RELOAD = Path("E:/aimemo/docs/screenshots/wows_torpedoes/torpedo_20261005_203649_1791203809232784600/frame_0001_observation.png")
SELECTED_ENEMY = Path("E:/aimemo/docs/screenshots/wows_selected_enemy_20261005.png")


@pytest.fixture(scope="module")
def backend():
    return RapidOcrBackend()


@pytest.fixture(scope="module")
def selected_image():
    return cv2.imread(str(FIXTURES / "live_battle.png"))


def prefire_context(image, backend):
    actual = inspect_torpedo_hud(image, image, (image.shape[1] // 2, image.shape[0] // 2), backend)
    assert actual.weapon_is_torpedo is True and actual.selected and actual.loaded_arc
    assert actual.slot_rect is not None and actual.aim_point is not None
    assert actual.lead_point is None and not actual.safe_to_fire
    return replace(actual, lead_point=actual.aim_point, alignment_error_px=0.,
                   alignment_tolerance_px=8., alignment_error_degrees=0.)


def with_actual_counter(selected, ready, source):
    after = selected.copy()
    x1, y1, x2, y2 = ready.slot_rect
    dy = selected.shape[0] - source.shape[0]
    after[y1:y2, x1:x2] = source[y1 - dy:y2 - dy, x1:x2]
    return after


@pytest.mark.parametrize("source_path,expected", [(FIXTURES / "battle_long_run.png", 28), (LIVE_RELOAD, 16)])
def test_actual_occluded_counters_are_read_only_with_a_verified_selected_slot(source_path, expected, selected_image, backend):
    if not source_path.exists():
        pytest.skip("Retained live reload PNG")
    source = cv2.imread(str(source_path))
    assert detect_torpedo_weapon(source, backend) is None
    assert _cooldown_seconds(source, backend) is None
    ready = prefire_context(selected_image, backend)
    after = with_actual_counter(selected_image, ready, source)
    assert detect_torpedo_weapon(after, backend) is None
    assert confirm_torpedo_reload(after, ready, backend) == expected
    assert not inspect_torpedo_hud(after, selected_image, (1280, 800), backend).safe_to_fire
    # Original captures selected slot 2: counter pixels alone are no feedback
    # for a shot attributed to the selected torpedo slot.
    dy = selected_image.shape[0] - source.shape[0]
    shifted = replace(ready, slot_rect=tuple(value - dy if index % 2 else value
                                           for index, value in enumerate(ready.slot_rect)),
                      frame_shape=tuple(source.shape[:2]))
    assert confirm_torpedo_reload(source, shifted, backend) is None


@pytest.mark.parametrize("resolution", [.5, .75])
def test_feedback_tracks_resolution_and_current_selected_key(resolution, selected_image, backend):
    ready = prefire_context(selected_image, backend)
    source = cv2.imread(str(FIXTURES / "battle_long_run.png"))
    after = with_actual_counter(selected_image, ready, source)
    scaled_before = cv2.resize(selected_image, None, fx=resolution, fy=resolution, interpolation=cv2.INTER_AREA)
    scaled_after = cv2.resize(after, None, fx=resolution, fy=resolution, interpolation=cv2.INTER_AREA)
    scaled_ready = prefire_context(scaled_before, backend)
    assert confirm_torpedo_reload(scaled_after, scaled_ready, backend) == 28


def move_weapon_row(image, scale):
    moved = image.copy()
    moved[1390:, 820:1780] = (70, 75, 80)
    row = cv2.resize(image[1440:1600, 850:1700], None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    left, top = round(1280 - row.shape[1] / 2 + 30), 1570 - row.shape[0]
    moved[top:top + row.shape[0], left:left + row.shape[1]] = row
    return moved


@pytest.mark.parametrize("ui_scale", [.75, 1.25])
def test_feedback_tracks_independent_moved_weapon_ui(ui_scale, selected_image, backend):
    ready = prefire_context(selected_image, backend)
    source = cv2.imread(str(FIXTURES / "battle_long_run.png"))
    after = with_actual_counter(selected_image, ready, source)
    moved_before, moved_after = move_weapon_row(selected_image, ui_scale), move_weapon_row(after, ui_scale)
    moved_ready = prefire_context(moved_before, backend)
    assert confirm_torpedo_reload(moved_after, moved_ready, backend) == 28
    # A cached position from before a UI move does not locate this counter.
    assert confirm_torpedo_reload(moved_after, ready, backend) is None


@pytest.mark.parametrize("change", ["no_geometry", "not_loaded", "no_identity", "not_selected", "unaligned", "bad_scale", "wrong_frame"])
def test_unverified_or_stale_prefire_context_is_not_reload_feedback(change, selected_image, backend):
    ready = prefire_context(selected_image, backend)
    after = with_actual_counter(selected_image, ready, cv2.imread(str(FIXTURES / "battle_long_run.png")))
    updates = {
        "no_geometry": {"slot_rect": None}, "not_loaded": {"loaded_arc": False},
        "no_identity": {"weapon_is_torpedo": None}, "not_selected": {"selected": False},
        "unaligned": {"alignment_error_degrees": 20}, "bad_scale": {"slot_scale": float("nan")},
        "wrong_frame": {"frame_shape": (1494, 2560)},
    }
    assert confirm_torpedo_reload(after, replace(ready, **updates[change]), backend) is None


def test_no_backend_or_plain_legacy_hud_has_no_reload_feedback(selected_image):
    assert confirm_torpedo_reload(selected_image, TorpedoHud(None, True, (500, 300)), None) is None


@pytest.mark.parametrize("text,expected", [("1:39", 99), ("0:52", 52), ("4:00", 240),
                                          ("16", 16), ("28秒", 28), ("1 : 39 s", 99)])
def test_reload_time_parses_seconds_and_strict_minutes_seconds(text, expected):
    assert _parse_reload_time(text) == expected


@pytest.mark.parametrize("text", ["1:60", "4:01", "5:00", "1:9", "1:39/52", "2.5", "-16", "99:99"])
def test_invalid_reload_time_is_not_feedback(text):
    assert _parse_reload_time(text) is None


def test_minutes_seconds_counter_uses_actual_ocr_and_selected_slot_geometry(selected_image, backend):
    ready = prefire_context(selected_image, backend)
    after = selected_image.copy()
    x1, y1, x2, y2 = ready.slot_rect
    body = after[y1:y2, x1:x2]
    body[:] = (70, 85, 90)
    font_scale = .77
    size, _ = cv2.getTextSize("1:39", cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
    cv2.putText(body, "1:39", ((body.shape[1] - size[0]) // 2, (body.shape[0] + size[1]) // 2),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (112, 222, 244), 2, cv2.LINE_AA)
    assert confirm_torpedo_reload(after, ready, backend) == 99


def test_existing_body_reload_counter_blocks_prefire_even_with_a_green_loaded_arc(selected_image, backend):
    ready = prefire_context(selected_image, backend)
    after = with_actual_counter(selected_image, ready, cv2.imread(str(FIXTURES / "battle_long_run.png")))
    # Model the harder partial-occlusion case where enough of the body still
    # proves the icon identity. The counter pixels and OCR remain real.
    visible_identity = _SlotEvidence(True, True, ready.slot_rect, ready.slot_scale)
    with patch("core.torpedo_hud._weapon_slot_evidence", return_value=visible_identity):
        current = inspect_torpedo_hud(after, after, (1280, 800), backend)
    assert current.weapon_is_torpedo is True and current.selected and current.loaded_arc
    assert current.cooldown_seconds == 28
    aligned = replace(current, lead_point=current.aim_point, alignment_error_px=0.,
                      alignment_tolerance_px=8., alignment_error_degrees=0.)
    assert not aligned.ready_to_align and not aligned.safe_to_fire
    assert confirm_torpedo_reload(after, aligned, backend) is None


@pytest.mark.skipif(not SELECTED_ENEMY.exists(), reason="Retained user-selected nearby enemy PNG")
@pytest.mark.parametrize("resolution", [.5, .75, 1, 1.25])
def test_actual_nearby_enemy_png_keeps_torpedo_identity_at_small_key_sizes(resolution, backend):
    image = cv2.imread(str(SELECTED_ENEMY))
    if resolution != 1:
        image = cv2.resize(image, None, fx=resolution, fy=resolution,
                           interpolation=cv2.INTER_AREA if resolution < 1 else cv2.INTER_CUBIC)
    assert detect_torpedo_weapon(image, backend) is True


def test_postfire_policy_confirms_real_occluded_counter_without_reusing_it_to_fire(policy, backend, selected_image):
    ready = prefire_context(selected_image, backend)
    after = with_actual_counter(selected_image, ready, cv2.imread(str(FIXTURES / "battle_long_run.png")))
    postfire = inspect_torpedo_hud(after, None, None, backend)
    assert postfire.weapon_is_torpedo is None and postfire.cooldown_seconds is None
    policy.states = [ready, ready, postfire]
    policy.bot.distance_reader.backend = backend
    policy.rect.return_value.update(width=2560, height=1600)
    policy.bot.vision.find_enemies_in_viewport.return_value = [(1280, 800)]
    grabs = {"count": 0}

    def fresh_frame(*_args, **_kwargs):
        grabs["count"] += 1
        return (selected_image if grabs["count"] <= 4 else after).copy()

    policy.bot.vision.grab.side_effect = fresh_frame
    current = analysis()
    current.image, current.width, current.height = selected_image, 2560, 1600
    current.target_viewport_point, current.enemies = (1280, 800), [(1280, 800)]
    policy.bot._try_torpedo_attack(current, 1000)
    policy.bot.gamepad.fire_torpedo.assert_called_once()
    confirmations = [call for call in policy.bot.events.publish.call_args_list
                     if call.args[0] == "torpedo.launch_confirmed"]
    assert len(confirmations) == 1 and confirmations[0].kwargs["cooldown_seconds"] == 28
