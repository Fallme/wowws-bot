import sys
import types
import unittest
from pathlib import Path

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
                get_window_rect=lambda _hwnd: {
                    "left": 0,
                    "top": 0,
                    "width": 2560,
                    "height": 1440,
                },
                physical_click=lambda _x, _y: True,
            ),
        )
        from port_navigator import (
            detect_port_mode,
            find_ship_card,
            is_requested_ship_selected,
            selected_ship_scores,
        )

        cls.detect_port_mode = staticmethod(detect_port_mode)
        cls.find_ship_card = staticmethod(find_ship_card)
        cls.is_requested_ship_selected = staticmethod(is_requested_ship_selected)
        cls.selected_ship_scores = staticmethod(selected_ship_scores)

    def test_detects_asymmetric_from_reference_port(self):
        image = cv2.imread(str(self.FIXTURE_ROOT / "port_ship_selected.png"))
        self.assertEqual(self.detect_port_mode(image), "asymmetric")

    def test_detects_cooperative_from_reference_port(self):
        image = cv2.imread(str(self.FIXTURE_ROOT / "port_mode_selector.png"))
        self.assertEqual(self.detect_port_mode(image), "cooperative")

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


if __name__ == "__main__":
    unittest.main()
