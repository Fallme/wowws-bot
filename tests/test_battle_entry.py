from unittest.mock import patch

import cv2
import numpy as np

import port_navigator
from core.ocr import OcrToken
from core.ui import PORT_BATTLE_BUTTON, ScreenState
from port_navigator import enter_battle


class StaticOcrBackend:
    def __init__(self, text):
        self.text = text

    def recognize(self, _image):
        return [
            OcrToken(
                self.text,
                0.98,
                ((100, 100), (220, 100), (220, 130), (100, 130)),
            )
        ]


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


def test_battle_entry_never_bypasses_no_commander_warning():
    warning = np.full((1000, 1600, 3), 35, dtype=np.uint8)

    class NoCommanderVision:
        @staticmethod
        def in_no_commander_confirmation(image):
            return image is warning

        @staticmethod
        def classify_screen(_image):
            return ScreenState.UNKNOWN

    with (
        patch("port_navigator._capture", return_value=warning),
        patch("port_navigator._click_region") as click_region,
        patch("port_navigator.time.sleep", return_value=None),
    ):
        assert not port_navigator._observe_battle_entry(
            7,
            NoCommanderVision(),
            samples=2,
            interval=0,
        )

    click_region.assert_not_called()


def test_loading_state_wins_over_false_no_commander_shape():
    loading = np.full((1000, 1600, 3), 20, dtype=np.uint8)

    class LoadingVision:
        @staticmethod
        def in_no_commander_confirmation(_image):
            return True

        @staticmethod
        def classify_screen(_image):
            return ScreenState.LOADING

    with (
        patch("port_navigator._capture", return_value=loading),
        patch("port_navigator.time.sleep", return_value=None),
    ):
        assert port_navigator._observe_battle_entry(
            7,
            LoadingVision(),
            samples=1,
            interval=0,
        )


def test_confirm_no_commander_is_a_read_only_fail_closed_guard():
    warning = np.full((1000, 1600, 3), 35, dtype=np.uint8)
    vision = EntryVision({id(warning): ScreenState.UNKNOWN})
    vision.in_no_commander_confirmation = lambda _image: True

    with patch("port_navigator._click_region") as click_region:
        assert not port_navigator.confirm_no_commander(7, warning, vision)

    click_region.assert_not_called()


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


def test_selected_ship_no_commander_is_read_only_from_detail_panel():
    frame = np.zeros((1000, 1600, 3), dtype=np.uint8)

    assert port_navigator.is_selected_ship_without_commander(
        frame, StaticOcrBackend("没有指挥官")
    )
    assert not port_navigator.is_selected_ship_without_commander(
        frame, StaticOcrBackend("利托里奥 战舰指挥官")
    )


def test_commander_ocr_failure_is_unknown_and_blocks_matchmaking():
    class BrokenBackend:
        @staticmethod
        def recognize(_image):
            raise RuntimeError("OCR unavailable")

    frame = np.zeros((1000, 1600, 3), dtype=np.uint8)
    assert port_navigator.is_selected_ship_without_commander(
        frame, BrokenBackend()
    ) is None

    with (
        patch("port_navigator._capture", return_value=frame),
        patch("port_navigator.is_requested_ship_selected", return_value=True),
        patch("port_navigator._right_click_local") as right_click,
        patch("port_navigator._click_local") as click,
    ):
        assert not port_navigator.ensure_selected_ship_commander(
            7, "pommern", backend=BrokenBackend()
        )

    right_click.assert_not_called()
    click.assert_not_called()


def test_no_commander_recall_requires_card_and_menu_text_before_click():
    frame = np.zeros((1000, 1600, 3), dtype=np.uint8)
    with (
        patch("port_navigator._capture", return_value=frame),
        patch("port_navigator.is_requested_ship_selected", return_value=True),
        patch(
            "port_navigator.is_selected_ship_without_commander",
            side_effect=[True, False],
        ),
        patch("port_navigator.find_ship_card_without_commander", return_value=(400, 850)),
        patch("port_navigator._right_click_local", return_value=True) as right_click,
        patch(
            "port_navigator._find_recall_commander_action",
            side_effect=[None, (360, 760)],
        ),
        patch("port_navigator._click_local", return_value=True) as click,
        patch("port_navigator.time.sleep", return_value=None),
    ):
        assert port_navigator.ensure_selected_ship_commander(
            7, "pommern", backend=StaticOcrBackend("召回指挥官")
        )

    right_click.assert_called_once_with(7, (400, 850))
    click.assert_called_once_with(7, (360, 760))


def test_no_commander_recall_rejects_wrong_selected_ship():
    frame = np.zeros((1000, 1600, 3), dtype=np.uint8)
    with (
        patch("port_navigator._capture", return_value=frame),
        patch("port_navigator.is_requested_ship_selected", return_value=False),
        patch(
            "port_navigator.is_selected_ship_without_commander",
            return_value=True,
        ) as no_commander,
        patch("port_navigator._right_click_local") as right_click,
        patch("port_navigator._click_local") as click,
    ):
        assert not port_navigator.ensure_selected_ship_commander(
            7,
            "napoli",
            backend=StaticOcrBackend("没有指挥官"),
        )

    no_commander.assert_not_called()
    right_click.assert_not_called()
    click.assert_not_called()


def test_commander_recall_requires_both_no_commander_indicators_even_with_open_menu():
    frame = np.zeros((1000, 1600, 3), dtype=np.uint8)
    with (
        patch("port_navigator._capture", return_value=frame),
        patch("port_navigator.is_requested_ship_selected", return_value=True),
        patch("port_navigator.is_selected_ship_without_commander", return_value=True),
        patch("port_navigator.find_ship_card_without_commander", return_value=None),
        patch("port_navigator._find_recall_commander_action", return_value=(360, 760)),
        patch("port_navigator._right_click_local") as right_click,
        patch("port_navigator._click_local") as click,
    ):
        assert not port_navigator.ensure_selected_ship_commander(7, "napoli")
    right_click.assert_not_called()
    click.assert_not_called()


def test_ship_with_commander_does_not_inspect_other_carousel_warnings():
    with (
        patch("port_navigator._capture", return_value=np.zeros((1000, 1600, 3), dtype=np.uint8)),
        patch("port_navigator.is_requested_ship_selected", return_value=True),
        patch("port_navigator.is_selected_ship_without_commander", return_value=False),
        patch("port_navigator.find_ship_card_without_commander") as card,
        patch("port_navigator._right_click_local") as right_click,
    ):
        assert port_navigator.ensure_selected_ship_commander(7, "napoli")
    card.assert_not_called()
    right_click.assert_not_called()


def test_carousel_warning_belongs_to_named_card_at_multiple_ui_scales():
    for scale in (0.75, 1.0, 1.5):
        def token(text, x, y, width, confidence=0.98):
            box = tuple((int(a * scale), int(b * scale)) for a, b in (
                (x, y), (x + width, y), (x + width, y + 14), (x, y + 14)
            ))
            return OcrToken(text, confidence, box)

        class Backend:
            tokens = [
                token("没有指挥官", 40, 70, 85),
                token("波美拉尼亚", 120, 87, 85),
                token("那不勒斯", 350, 87, 65),
            ]

            def recognize(self, image):
                return self.tokens

        frame = np.zeros((int(1000 * scale), int(1600 * scale), 3), dtype=np.uint8)
        backend = Backend()
        assert port_navigator.find_ship_card_without_commander(frame, "pommern", None, backend)
        assert port_navigator.find_ship_card_without_commander(frame, "napoli", None, backend) is None
        backend.tokens.append(token("无指挥官", 265, 70, 70))
        assert port_navigator.find_ship_card_without_commander(frame, "napoli", None, backend)
        assert port_navigator.find_ship_card_without_commander(frame, "custom", "那不勒斯", backend)
        backend.tokens[-1] = token("无指挥官", 265, 70, 70, confidence=0.40)
        assert port_navigator.find_ship_card_without_commander(frame, "napoli", None, backend) is None
