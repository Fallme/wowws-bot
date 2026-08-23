import unittest
from pathlib import Path

try:
    import cv2
except ImportError:  # The production environment installs requirements.txt.
    cv2 = None

from core.ui import ScreenState


@unittest.skipUnless(cv2 is not None, "OpenCV runtime is not installed")
class ScreenStateRegressionTests(unittest.TestCase):
    FIXTURE_ROOT = Path("tests") / "fixtures"
    @classmethod
    def setUpClass(cls):
        from core.vision import Vision

        cls.vision = Vision()

    def classify(self, relative_path):
        image = cv2.imread(str(Path(relative_path)))
        self.assertIsNotNone(image, relative_path)
        return self.vision.classify_screen(image)

    def test_loading_cinematic_wins_over_port_colour_overlap(self):
        self.assertEqual(
            self.classify(self.FIXTURE_ROOT / "loading.png"),
            ScreenState.LOADING,
        )

    def test_reference_port_and_battle_frames_remain_distinct(self):
        self.assertEqual(
            self.classify(self.FIXTURE_ROOT / "port_ship_selected.png"),
            ScreenState.PORT,
        )
        self.assertEqual(
            self.classify(self.FIXTURE_ROOT / "battle_reference.png"),
            ScreenState.BATTLE,
        )

    def test_escape_menu_overrides_broad_loading_signal(self):
        self.assertEqual(
            self.classify(self.FIXTURE_ROOT / "escape_menu.png"),
            ScreenState.ESCAPE_MENU,
        )

    def test_live_battle_hud_wins_over_loading_heuristic(self):
        self.assertEqual(
            self.classify(self.FIXTURE_ROOT / "live_battle.png"),
            ScreenState.BATTLE,
        )

    def test_live_result_screen_is_not_exit_confirmation(self):
        self.assertEqual(
            self.classify(self.FIXTURE_ROOT / "results.png"),
            ScreenState.RESULTS,
        )

if __name__ == "__main__":
    unittest.main()
