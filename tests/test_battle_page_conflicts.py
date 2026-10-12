"""September 21: live battle mistaken for loading/port blocks all driving."""
from types import SimpleNamespace
from unittest.mock import Mock

import cv2
import numpy as np
import pytest

import main
from core.battle_phase import BattlePhase, BattlePhaseTracker
from core.ui import ScreenState
from core.vision import Vision


@pytest.mark.parametrize("tokens", [[], None, [SimpleNamespace(text="")]])
@pytest.mark.parametrize("hud", [True, False])
def test_empty_ocr_never_confirms_start_action(tokens, hud):
    bot = SimpleNamespace(
        vision=SimpleNamespace(_has_loading_start_action=lambda _: True,
                               _has_battle_hud=lambda _: hud),
        distance_reader=SimpleNamespace(backend=SimpleNamespace(recognize=lambda _: tokens)),
    )
    assert not main.loading_start_confirmed(bot, np.zeros((90, 160, 3), np.uint8))


@pytest.mark.parametrize("backend", [None, SimpleNamespace(recognize=Mock(side_effect=RuntimeError("OCR unavailable")))])
def test_unavailable_ocr_never_confirms_start_action(backend):
    bot = SimpleNamespace(vision=SimpleNamespace(_has_loading_start_action=lambda _: True),
                          distance_reader=SimpleNamespace(backend=backend))
    assert not main.loading_start_confirmed(bot, np.zeros((90, 160, 3), np.uint8))


def test_unreadable_false_button_does_not_reset_active_phase():
    bot = SimpleNamespace(vision=SimpleNamespace(
        classify_screen=lambda _: ScreenState.LOADING,
        _has_loading_start_action=lambda _: True, _has_battle_hud=lambda _: True,
        read_battle_clock_seconds=lambda *_: 1198,
    ), distance_reader=SimpleNamespace(backend=SimpleNamespace(recognize=lambda _: [])))
    tracker = bot._battle_phase_tracker = BattlePhaseTracker()
    tracker.observe(ScreenState.BATTLE, 1200, 0)
    tracker.observe(ScreenState.BATTLE, 1199, 1)
    frame = np.zeros((90, 160, 3), np.uint8)
    state = main.classify_battle_continuity_screen(bot, frame)
    assert state == ScreenState.BATTLE
    assert main.observe_battle_phase(bot, frame, state) == BattlePhase.ACTIVE


def test_port_veto_is_overridden_only_by_independent_hud_proof(monkeypatch):
    image = cv2.imread("tests/fixtures/battle_port_color_conflict.png")
    vision = Vision()
    assert vision.classify_screen(image) == ScreenState.PORT
    assert not vision._has_battle_hud(image)
    assert vision._has_battle_hud_layout(image)
    # Keep this regression deterministic: exercise actual geometry, control
    # flow and separate OCR values without initializing an inference provider.
    monkeypatch.setattr(vision, "read_battle_clock_seconds", lambda *_: 883)
    monkeypatch.setattr(vision, "read_health_fraction", lambda *_: 0.98)
    bot = SimpleNamespace(vision=vision,
        distance_reader=SimpleNamespace(backend=SimpleNamespace(recognize=lambda _: [])))
    assert main.classify_battle_continuity_screen(bot, image) == ScreenState.BATTLE
    monkeypatch.setattr(vision, "read_health_fraction", lambda *_: None)
    assert main.classify_battle_continuity_screen(bot, image) == ScreenState.PORT


@pytest.mark.parametrize("clock,health", [(None, 1.0), (1199, None)])
def test_port_recovery_requires_both_numeric_anchors(monkeypatch, clock, health):
    image = cv2.imread("tests/fixtures/battle_port_color_conflict.png")
    vision = Vision()
    monkeypatch.setattr(vision, "read_battle_clock_seconds", lambda *_: clock)
    monkeypatch.setattr(vision, "read_health_fraction", lambda *_: health)
    assert not vision.has_numeric_battle_hud(image, object())
