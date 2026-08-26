from unittest.mock import patch

import cv2
import numpy as np

import port_navigator
from core.ui import PORT_BATTLE_BUTTON, ScreenState
from port_navigator import enter_battle


def _port_frame():
    frame = np.zeros((1000, 1600, 3), dtype=np.uint8)
    x1, y1, x2, y2 = PORT_BATTLE_BUTTON.pixels(1600, 1000)
    frame[y1:y2, x1:x2] = (0, 145, 255)
    return frame


class EntryVision:
    def __init__(self, states):
        self.states = states

    def classify_screen(self, image):
        return self.states[id(image)]

    @staticmethod
    def in_no_commander_confirmation(_image):
        return False


def _capture_sequence(frames):
    remaining = list(frames)
    last = frames[-1]

    def capture(_hwnd=None):
        nonlocal last
        if remaining:
            last = remaining.pop(0)
        return last

    return capture


def test_enter_battle_clicks_once_and_accepts_loading_transition():
    before = _port_frame()
    button = before.copy()
    loading = np.full_like(before, 20)
    vision = EntryVision(
        {
            id(before): ScreenState.PORT,
            id(button): ScreenState.PORT,
            id(loading): ScreenState.LOADING,
        }
    )

    with (
        patch(
            "port_navigator._capture",
            side_effect=_capture_sequence([before, button, loading]),
        ),
        patch("port_navigator._click_local", return_value=True) as click,
        patch("port_navigator.time.sleep", return_value=None),
    ):
        assert enter_battle(1, vision=vision, configure_port=False)

    assert click.call_count == 1


def test_enter_battle_allows_retry_only_after_stable_actionable_port():
    port = _port_frame()
    vision = EntryVision({id(port): ScreenState.PORT})

    with (
        patch("port_navigator._capture", return_value=port),
        patch("port_navigator._click_local", return_value=True) as click,
        patch("port_navigator.time.sleep", return_value=None),
    ):
        assert not enter_battle(1, vision=vision, configure_port=False)

    # Observation never duplicates the original Join Battle click.
    assert click.call_count == 1


def test_enter_battle_does_not_observe_or_retry_when_click_was_not_dispatched():
    port = _port_frame()
    vision = EntryVision({id(port): ScreenState.PORT})

    with (
        patch("port_navigator._capture", return_value=port) as capture,
        patch("port_navigator._click_local", return_value=False) as click,
        patch("port_navigator.time.sleep", return_value=None),
    ):
        assert not enter_battle(1, vision=vision, configure_port=False)

    assert click.call_count == 1
    assert capture.call_count == 2


def test_local_click_uses_window_message_only_after_physical_click_fails():
    with (
        patch("port_navigator.ensure_game_window_foreground", return_value=True),
        patch(
            "port_navigator.get_client_rect",
            return_value={"left": 100, "top": 200, "width": 2560, "height": 1600},
        ),
        patch("port_navigator.physical_click", return_value=False) as physical,
        patch("port_navigator.window_message_click", return_value=True) as message,
        patch("port_navigator.time.sleep", return_value=None),
    ):
        assert port_navigator._click_local(7, (1280, 36))

    physical.assert_called_once_with(1380, 236, hwnd=7)
    message.assert_called_once_with(7, 1380, 236)
