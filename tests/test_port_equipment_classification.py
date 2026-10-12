"""Real equipment-port evidence must beat its battle-like weapon panels."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from core.ui import ScreenState
from core.vision import Vision


FIXTURES = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture(scope="module")
def equipment_port():
    # Lossless real startup capture from 2026-10-05, windowed 1707x1067.
    image = cv2.imread(str(FIXTURES / "port_equipment_live_1707.png"))
    assert image is not None
    return image


@pytest.mark.parametrize("scale", [.5, .75, 1, 1.25, 1.5, 2])
def test_real_equipment_page_is_port_at_different_window_sizes(equipment_port, scale):
    frame = cv2.resize(equipment_port, None, fx=scale, fy=scale,
                       interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    vision = Vision(screen_capture=object())
    votes = vision._port_anchor_votes(frame)
    assert votes["top_port_tabs"]
    assert sum(bool(vote) for vote in votes.values()) >= 3
    assert vision.in_port(frame)
    assert not vision._has_battle_hud(frame)
    assert vision.classify_screen(frame) == ScreenState.PORT


@pytest.mark.parametrize("scale,offset_x,offset_y", [(.65, 65, 8), (1.25, -65, 0)])
def test_equipment_tab_strip_tracks_independent_header_ui_scale_and_shift(
        equipment_port, scale, offset_x, offset_y):
    frame = equipment_port.copy()
    source = equipment_port[:80]
    header = cv2.resize(source, None, fx=scale, fy=scale,
                        interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    h, w = header.shape[:2]
    left = round((frame.shape[1] - w) / 2) + offset_x
    # Relocate just the actual header UI; ship data and carousel stay native.
    frame[:max(80, offset_y + h)] = 0
    dst_x1, dst_x2 = max(0, left), min(frame.shape[1], left + w)
    frame[offset_y:offset_y + h, dst_x1:dst_x2] = header[:, dst_x1 - left:dst_x2 - left]
    vision = Vision(screen_capture=object())
    assert vision._has_port_tab_strip(frame)
    assert vision.classify_screen(frame) == ScreenState.PORT


@pytest.mark.parametrize("filename,baseline_state,baseline_port", [
    ("live_battle.png", ScreenState.BATTLE, False),
    ("battle_reference.png", ScreenState.BATTLE, False),
    ("battle_long_run.png", ScreenState.BATTLE, False),
    # These existing generic PORT conflicts are resolved by main's separate
    # numeric battle-continuity gate; this fix must not change that contract.
    ("battle_low_contrast_1440.png", ScreenState.PORT, False),
    ("battle_port_color_conflict.png", ScreenState.PORT, True),
])
def test_actual_battle_score_and_hotbar_do_not_form_a_port_tab_strip(
        filename, baseline_state, baseline_port):
    frame = cv2.imread(str(FIXTURES / filename))
    assert frame is not None
    vision = Vision(screen_capture=object())
    assert not vision._has_port_tab_strip(frame)
    assert bool(vision.in_port(frame)) == baseline_port
    assert vision.classify_screen(frame) == baseline_state


def test_equipment_page_requires_the_real_battle_action(equipment_port):
    frame = equipment_port.copy()
    frame[:46, 780:930] = 0
    vision = Vision(screen_capture=object())
    assert not vision._has_port_tab_strip(frame)
    assert not vision.in_port(frame)


def test_equipment_page_requires_the_actual_multi_tab_row(equipment_port):
    frame = equipment_port.copy()
    frame[48:76, 480:1225] = 0
    vision = Vision(screen_capture=object())
    assert not vision._has_port_tab_strip(frame)
    assert not vision.in_port(frame)


def test_regular_vertical_text_strokes_without_two_rails_are_not_tabs(equipment_port):
    frame = equipment_port.copy()
    frame[45:92] = 0
    for x in (488, 610, 732, 854, 976, 1098, 1220):
        cv2.line(frame, (x, 51), (x, 72), (95, 95, 95), 1)
    vision = Vision(screen_capture=object())
    assert not vision._has_port_tab_strip(frame)
    assert not vision.in_port(frame)


def test_battle_action_and_tabs_without_independent_port_panels_do_not_authorize_port(
        equipment_port):
    frame = np.zeros_like(equipment_port)
    frame[:80] = equipment_port[:80]
    vision = Vision(screen_capture=object())
    assert vision._has_port_tab_strip(frame)
    assert not vision.in_port(frame)
