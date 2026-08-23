from unittest.mock import Mock, patch

import numpy as np

import port_navigator
from core.ui import RESULTS_REQUEUE_BUTTON, ScreenState


def test_queue_next_battle_clicks_only_verified_requeue_region():
    image = np.zeros((1063, 1707, 3), dtype=np.uint8)
    vision = Mock()
    vision.classify_screen.side_effect = [
        ScreenState.RESULTS,
        ScreenState.LOADING,
    ]

    with (
        patch.object(port_navigator, "_capture", return_value=image),
        patch.object(port_navigator, "_click_region", return_value=True) as click,
        patch.object(port_navigator, "confirm_no_commander", return_value=False),
        patch.object(port_navigator.time, "sleep"),
    ):
        assert port_navigator.queue_next_battle(vision=vision)

    click.assert_called_once_with(None, image, RESULTS_REQUEUE_BUTTON)


def test_queue_next_battle_refuses_unknown_screen():
    image = np.zeros((1063, 1707, 3), dtype=np.uint8)
    vision = Mock()
    vision.classify_screen.return_value = ScreenState.UNKNOWN

    with (
        patch.object(port_navigator, "_capture", return_value=image),
        patch.object(port_navigator, "_click_region") as click,
    ):
        assert not port_navigator.queue_next_battle(vision=vision)

    click.assert_not_called()
