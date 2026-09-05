from unittest.mock import Mock, patch

import numpy as np

import port_navigator
from core.ocr import OcrToken
from core.frame_guard import CaptureFault
from core.ui import (
    RESULTS_REQUEUE_BUTTON,
    RESULTS_RETURN_TO_PORT_BUTTON,
    ScreenState,
)


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


def test_queue_next_battle_never_bypasses_no_commander_warning():
    image = np.zeros((1063, 1707, 3), dtype=np.uint8)
    vision = Mock()
    vision.classify_screen.side_effect = [ScreenState.RESULTS, ScreenState.UNKNOWN, ScreenState.UNKNOWN]
    vision.in_no_commander_confirmation.return_value = True
    backend = Mock()
    backend.recognize.return_value = [OcrToken("没有指挥官", .99)]

    with (
        patch.object(port_navigator, "_capture", return_value=image),
        patch.object(port_navigator, "_click_region", return_value=True) as click,
        patch.object(port_navigator, "results_requeue_action_point", return_value=(500, 700)),
        patch.object(port_navigator, "_click_local", return_value=True) as local_click,
        patch.object(port_navigator.time, "sleep"),
    ):
        assert not port_navigator.queue_next_battle(vision=vision, backend=backend)

    local_click.assert_called_once()
    click.assert_not_called()


def test_false_commander_shape_during_transition_keeps_queue():
    image = np.zeros((900, 1600, 3), dtype=np.uint8)
    vision = Mock()
    vision.classify_screen.side_effect = [ScreenState.RESULTS, ScreenState.UNKNOWN, ScreenState.LOADING]
    vision.in_no_commander_confirmation.return_value = True
    backend = Mock()
    backend.recognize.return_value = [OcrToken("正在加载", .99)]
    with (
        patch.object(port_navigator, "_capture", return_value=image),
        patch.object(port_navigator, "results_requeue_action_point", return_value=(500, 700)),
        patch.object(port_navigator, "_click_local", return_value=True) as click,
        patch.object(port_navigator.time, "sleep"),
    ):
        assert port_navigator.queue_next_battle(vision=vision, backend=backend)
    click.assert_called_once()


def test_capture_failure_after_requeue_click_preserves_entry_lock():
    image = np.zeros((900, 1600, 3), dtype=np.uint8)
    vision = Mock()
    vision.classify_screen.return_value = ScreenState.RESULTS
    with (
        patch.object(port_navigator, "_capture", side_effect=[image, CaptureFault("transition")]),
        patch.object(port_navigator, "_click_region", return_value=True),
        patch.object(port_navigator.time, "sleep"),
    ):
        assert port_navigator.queue_next_battle(vision=vision)


def test_queue_next_battle_honors_user_pause_before_capture_or_click():
    vision = Mock()
    with (
        patch("port_navigator._capture") as capture,
        patch("port_navigator._click_region") as click,
    ):
        assert not port_navigator.queue_next_battle(
            vision=vision,
            should_abort=lambda: True,
        )

    capture.assert_not_called()
    click.assert_not_called()


def test_queue_next_battle_capture_fault_preserves_result_page():
    vision = Mock()
    with (
        patch(
            "port_navigator._capture",
            side_effect=CaptureFault("capture unavailable"),
        ),
        patch("port_navigator._click_region") as click,
    ):
        assert not port_navigator.queue_next_battle(vision=vision)

    click.assert_not_called()


def test_post_battle_navigation_honors_pause_before_foreground_capture():
    vision = Mock()
    with (
        patch("port_navigator._capture") as capture,
        patch("port_navigator._click_region") as click,
    ):
        assert not port_navigator.handle_post_battle(
            vision=vision,
            should_abort=lambda: True,
        )

    capture.assert_not_called()
    click.assert_not_called()


def test_post_battle_capture_fault_preserves_current_page():
    with (
        patch(
            "port_navigator._capture",
            side_effect=CaptureFault("capture unavailable"),
        ),
        patch("port_navigator._click_region") as click,
    ):
        assert not port_navigator.handle_post_battle(vision=Mock())

    click.assert_not_called()


def test_post_battle_uses_verified_relative_return_when_ocr_misses():
    image = np.zeros((1063, 1707, 3), dtype=np.uint8)
    vision = Mock()
    vision.classify_screen.side_effect = [ScreenState.RESULTS, ScreenState.PORT]

    with (
        patch.object(port_navigator, "_capture", return_value=image),
        patch.object(
            port_navigator,
            "results_return_to_port_action_point",
            return_value=None,
        ),
        patch.object(port_navigator, "_click_region", return_value=True) as click,
        patch.object(port_navigator.time, "sleep"),
    ):
        assert port_navigator.handle_post_battle(
            vision=vision,
            backend=Mock(),
        )

    click.assert_called_once_with(None, image, RESULTS_RETURN_TO_PORT_BUTTON)


def test_post_battle_does_not_click_transient_escape_menu_guess():
    image = np.zeros((1063, 1707, 3), dtype=np.uint8)
    vision = Mock()
    vision.classify_screen.side_effect = [
        ScreenState.ESCAPE_MENU,
        ScreenState.RESULTS,
        ScreenState.PORT,
    ]

    with (
        patch.object(port_navigator, "_capture", return_value=image),
        patch.object(port_navigator, "_click_region", return_value=True) as click,
        patch.object(port_navigator.time, "sleep"),
    ):
        assert port_navigator.handle_post_battle(vision=vision)

    click.assert_called_once_with(None, image, RESULTS_RETURN_TO_PORT_BUTTON)


def test_post_battle_exit_confirmation_requires_ocr_recheck():
    image = np.zeros((1063, 1707, 3), dtype=np.uint8)
    vision = Mock()
    vision.classify_screen.return_value = ScreenState.EXIT_CONFIRMATION

    with (
        patch.object(port_navigator, "_capture", return_value=image),
        patch.object(
            port_navigator,
            "is_early_exit_confirmation_page",
            return_value=False,
        ),
        patch.object(port_navigator, "_click_region") as click,
        patch.object(port_navigator.time, "sleep"),
    ):
        assert not port_navigator.handle_post_battle(
            vision=vision,
            backend=Mock(),
        )

    click.assert_not_called()
