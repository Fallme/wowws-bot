"""Replay the actual loading/economy failures without dispatching game input."""
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from core.results import ResultRewardReader
from core.ui import ScreenState
from core.vision import Vision
from main import classify_battle_continuity_screen, configure_opening_autopilot


@pytest.fixture(scope="module")
def reader():
    return ResultRewardReader()


@pytest.mark.parametrize("vertical_shift", [0, 80])
def test_real_reward_row_and_moved_panel(reader, vertical_shift):
    image = cv2.imread("tests/fixtures/results_live_1494.png")
    if vertical_shift:
        image = cv2.warpAffine(
            image, np.float32([[1, 0, 0], [0, 1, vertical_shift]]),
            (image.shape[1], image.shape[0]),
        )
    reward = reader.read(image)
    assert reward.recognized
    assert reward.resource_values() == {
        "credits": 67769, "ship_xp": 1747, "free_xp": 250,
    }


@pytest.mark.parametrize("filename,expected", [
    ("loading_tips_live.png", ScreenState.LOADING),
    ("port_economy_live.png", ScreenState.UNKNOWN),
])
def test_noncombat_pages_never_dispatch_opening_map(monkeypatch, reader, filename, expected):
    image = cv2.imread("tests/fixtures/" + filename)
    vision = Vision()
    monkeypatch.setattr(vision, "grab", lambda *a, **k: image)
    def forbidden(*args, **kwargs):
        pytest.fail("A noncombat page must not receive an autopilot command")
    bot = SimpleNamespace(
        vision=vision, hwnd=1,
        _opening_autopilot_attempted=True,
        distance_reader=SimpleNamespace(backend=reader.backend),
        gamepad=SimpleNamespace(toggle_tactical_map=forbidden),
        enable_opening_autopilot=forbidden,
    )
    monkeypatch.setattr("main.operation_paused", lambda bot: False)
    monkeypatch.setattr("main.normalize_tactical_map_overlay", lambda bot: True)
    assert classify_battle_continuity_screen(bot, image) == expected
    assert configure_opening_autopilot(bot) is False
    assert not getattr(bot, "_tactical_map_attempted_this_battle", False)
    assert bot._opening_autopilot_attempted is False
