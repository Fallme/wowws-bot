import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    import cv2
    import numpy as np
except ImportError:  # The production environment installs requirements.txt.
    cv2 = np = None


@unittest.skipUnless(cv2 is not None, "OpenCV runtime is not installed")
class PortSelectionTests(unittest.TestCase):
    FIXTURE_ROOT = Path("tests") / "fixtures"
    @classmethod
    def setUpClass(cls):
        # Keep the pure screenshot tests independent of Windows capture drivers.
        sys.modules.setdefault("mss", types.SimpleNamespace(MSS=object))
        sys.modules.setdefault(
            "core.window",
            types.SimpleNamespace(
                activate_window=lambda _hwnd: True,
                ensure_game_window_foreground=lambda _hwnd: True,
                get_client_rect=lambda _hwnd: {
                    "left": 0,
                    "top": 0,
                    "width": 2560,
                    "height": 1440,
                },
                get_window_rect=lambda _hwnd: {
                    "left": 0,
                    "top": 0,
                    "width": 2560,
                    "height": 1440,
                },
                physical_click=lambda _x, _y, **_kwargs: True,
                physical_scroll=lambda _x, _y, _delta, **_kwargs: True,
                window_message_click=lambda _hwnd, _x, _y: True,
            ),
        )
        from core.ocr import OcrToken
        from port_navigator import (
            _find_mode_card_from_ocr,
            ShipSelectionError,
            detect_port_mode,
            ensure_requested_mode,
            find_custom_ship_card,
            find_ship_card,
            is_custom_ship_selected,
            is_requested_ship_selected,
            in_battle_type_selector,
            is_battle_survey_page,
            select_requested_ship,
            selected_ship_scores,
        )

        cls.OcrToken = OcrToken
        cls.find_mode_card_from_ocr = staticmethod(_find_mode_card_from_ocr)
        cls.ShipSelectionError = ShipSelectionError
        cls.detect_port_mode = staticmethod(detect_port_mode)
        cls.ensure_requested_mode = staticmethod(ensure_requested_mode)
        cls.find_custom_ship_card = staticmethod(find_custom_ship_card)
        cls.find_ship_card = staticmethod(find_ship_card)
        cls.is_custom_ship_selected = staticmethod(is_custom_ship_selected)
        cls.is_requested_ship_selected = staticmethod(is_requested_ship_selected)
        cls.in_battle_type_selector = staticmethod(in_battle_type_selector)
        cls.is_battle_survey_page = staticmethod(is_battle_survey_page)
        cls.select_requested_ship = staticmethod(select_requested_ship)
        cls.selected_ship_scores = staticmethod(selected_ship_scores)

    def backend(self, *tokens):
        class Backend:
            def recognize(_self, _image):
                return list(tokens)

        return Backend()

    def test_detects_asymmetric_from_reference_port(self):
        image = cv2.imread(str(self.FIXTURE_ROOT / "port_ship_selected.png"))
        self.assertEqual(self.detect_port_mode(image), "asymmetric")

    def test_detects_cooperative_from_reference_port(self):
        image = cv2.imread(str(self.FIXTURE_ROOT / "port_mode_selector.png"))
        self.assertEqual(self.detect_port_mode(image), "cooperative")

    def test_exact_mode_ocr_rejects_random_as_cooperative(self):
        image = cv2.imread(str(self.FIXTURE_ROOT / "port_mode_selector.png"))
        backend = self.backend(self.OcrToken("随机战", 0.99))

        self.assertEqual(
            self.detect_port_mode(image, backend=backend),
            "random",
        )

    def test_exact_mode_ocr_confirms_cooperative(self):
        image = cv2.imread(str(self.FIXTURE_ROOT / "port_mode_selector.png"))
        backend = self.backend(self.OcrToken("联合作战", 0.99))

        self.assertEqual(
            self.detect_port_mode(image, backend=backend),
            "cooperative",
        )

    def test_normal_port_is_not_confused_with_battle_type_selector(self):
        image = cv2.imread(str(self.FIXTURE_ROOT / "port_mode_selector.png"))

        self.assertFalse(self.in_battle_type_selector(image))

    def test_open_battle_type_page_is_detected_from_asymmetric_card(self):
        image = np.full((1600, 2560, 3), 70, dtype=np.uint8)
        cv2.circle(image, (1580, 875), 60, (180, 30, 180), -1)

        self.assertTrue(self.in_battle_type_selector(image))

    def test_current_battle_type_page_fixture_is_detected(self):
        image = cv2.imread(
            str(self.FIXTURE_ROOT / "battle_type_selector_current.jpg")
        )

        self.assertTrue(self.in_battle_type_selector(image))

    def test_battle_survey_requires_specific_question_and_close_action(self):
        image = np.zeros((1000, 1600, 3), dtype=np.uint8)
        survey = self.backend(
            self.OcrToken("您对刚刚进行的这场战斗满意度如何？", 0.99),
            self.OcrToken("非常不满意 不满意 一般 满意 非常满意", 0.98),
            self.OcrToken("关闭", 0.99),
        )
        unrelated = self.backend(
            self.OcrToken("返回港口", 0.99),
            self.OcrToken("关闭", 0.99),
        )

        self.assertTrue(self.is_battle_survey_page(image, backend=survey))
        self.assertFalse(self.is_battle_survey_page(image, backend=unrelated))

    def test_mode_card_click_point_comes_from_ocr_box(self):
        image = np.zeros((1000, 1600, 3), dtype=np.uint8)
        backend = self.backend(
            self.OcrToken(
                "联合作战",
                0.99,
                ((100, 200), (220, 200), (220, 240), (100, 240)),
            )
        )

        point = self.find_mode_card_from_ocr(
            image,
            "cooperative",
            backend,
        )

        # Search crop begins at (288, 200); click follows the OCR box centre,
        # not a 2560x1440-specific hard-coded card coordinate.
        self.assertEqual(point, (448, 420))

    def test_mode_selection_page_must_close_before_coop_is_verified(self):
        selector = np.full((1600, 2560, 3), 70, dtype=np.uint8)
        cv2.circle(selector, (1580, 875), 60, (180, 30, 180), -1)
        cooperative_port = cv2.imread(
            str(self.FIXTURE_ROOT / "port_mode_selector.png")
        )

        class PortVision:
            @staticmethod
            def classify_screen(_image):
                from core.ui import ScreenState

                return ScreenState.PORT

        with (
            patch("port_navigator._capture", side_effect=[selector, cooperative_port]),
            patch("port_navigator._click_local", return_value=True) as click,
            patch("port_navigator.time.sleep", return_value=None),
        ):
            selected = self.ensure_requested_mode(
                1,
                "cooperative",
                vision=PortVision(),
            )

        self.assertTrue(selected)
        # Starting on the selector must click only the requested mode card; it
        # must not click the port header again and accidentally close the page.
        self.assertEqual(click.call_count, 1)

    def test_finds_both_supported_ship_cards(self):
        image = cv2.imread(str(self.FIXTURE_ROOT / "port_ship_selected.png"))
        for ship in ("pommern", "napoli"):
            match = self.find_ship_card(image, ship)
            self.assertIsNotNone(match, ship)
            (x, y), score = match
            self.assertGreater(score, 0.68)
            self.assertTrue(0 <= x < image.shape[1])
            self.assertTrue(int(image.shape[0] * 0.73) <= y < image.shape[0])

    def test_rejects_unknown_ship(self):
        image = np.zeros((1440, 2560, 3), dtype=np.uint8)
        with self.assertRaises(ValueError):
            self.find_ship_card(image, "unknown")

    def test_verifies_selected_ship_from_detail_panel(self):
        image = cv2.imread(str(self.FIXTURE_ROOT / "port_ship_selected.png"))
        scores = self.selected_ship_scores(image)

        self.assertGreater(scores["napoli"], 0.95)
        self.assertLess(scores["pommern"], 0.55)
        self.assertTrue(self.is_requested_ship_selected(image, "napoli"))
        self.assertFalse(self.is_requested_ship_selected(image, "pommern"))

    def test_finds_exact_custom_ship_name_from_split_ocr_tokens(self):
        image = np.zeros((1440, 2560, 3), dtype=np.uint8)
        backend = self.backend(
            self.OcrToken(
                "大选",
                0.94,
                ((800, 120), (850, 120), (850, 150), (800, 150)),
            ),
            self.OcrToken(
                "帝侯",
                0.92,
                ((858, 120), (910, 120), (910, 150), (858, 150)),
            ),
        )
        match = self.find_custom_ship_card(image, "大选帝侯", backend)
        self.assertIsNotNone(match)
        (x, y), confidence = match
        self.assertTrue(800 <= x <= 910)
        self.assertGreaterEqual(y, int(image.shape[0] * 0.735))
        self.assertGreater(confidence, 0.9)

    def test_custom_ship_name_requires_full_exact_match(self):
        image = np.zeros((1440, 2560, 3), dtype=np.uint8)
        backend = self.backend(
            self.OcrToken(
                "大选帝侯 B",
                0.96,
                ((800, 120), (940, 120), (940, 150), (800, 150)),
            )
        )
        self.assertIsNone(self.find_custom_ship_card(image, "大选帝侯", backend))

    def test_selected_custom_ship_accepts_tier_prefix_only(self):
        image = np.zeros((1440, 2560, 3), dtype=np.uint8)
        tiered = self.backend(
            self.OcrToken(
                "IX 石见",
                0.96,
                ((100, 40), (210, 40), (210, 70), (100, 70)),
            )
        )
        wrong_ship = self.backend(
            self.OcrToken(
                "IX 石见 B",
                0.96,
                ((100, 40), (240, 40), (240, 70), (100, 70)),
            )
        )
        self.assertTrue(self.is_custom_ship_selected(image, "石见", tiered))
        self.assertFalse(self.is_custom_ship_selected(image, "石见", wrong_ship))

    def test_selected_latin_ship_starting_with_roman_letter_is_exact(self):
        image = np.zeros((1440, 2560, 3), dtype=np.uint8)
        backend = self.backend(
            self.OcrToken(
                "Vermont",
                0.97,
                ((100, 40), (220, 40), (220, 70), (100, 70)),
            )
        )
        self.assertTrue(self.is_custom_ship_selected(image, "Vermont", backend))

    def test_missing_custom_ship_raises_actionable_error(self):
        image = np.zeros((1440, 2560, 3), dtype=np.uint8)

        class PortVision:
            @staticmethod
            def classify_screen(_image):
                from core.ui import ScreenState

                return ScreenState.PORT

        with patch("port_navigator._capture", return_value=image):
            with self.assertRaisesRegex(self.ShipSelectionError, "重新选择"):
                self.select_requested_ship(
                    None,
                    "custom",
                    PortVision(),
                    custom_name="不存在的舰船",
                    ocr_backend=self.backend(),
                    custom_max_scrolls=0,
                )

    def test_strict_ship_selection_rejects_false_port_with_battle_hud(self):
        image = np.zeros((1440, 2560, 3), dtype=np.uint8)

        class ConflictingVision:
            @staticmethod
            def classify_screen(_image):
                from core.ui import ScreenState

                return ScreenState.PORT

            @staticmethod
            def _has_battle_hud(_image):
                return True

        with (
            patch("port_navigator._capture", return_value=image),
            patch("port_navigator._scroll_ship_carousel_down") as scroll,
            patch("port_navigator._click_local") as click,
        ):
            selected = self.select_requested_ship(
                1,
                "pommern",
                ConflictingVision(),
                ocr_backend=self.backend(),
                require_port_action=True,
            )

        self.assertFalse(selected)
        scroll.assert_not_called()
        click.assert_not_called()


if __name__ == "__main__":
    unittest.main()
