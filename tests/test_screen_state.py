import unittest
from pathlib import Path

import numpy as np

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

    def test_login_startup_artwork_is_loading_never_battle_or_modal(self):
        image = cv2.imread(str(self.FIXTURE_ROOT / "login_startup.png"))
        self.assertIsNotNone(image)
        self.assertTrue(self.vision._is_login_splash(image))
        self.assertFalse(self.vision._has_battle_hud(image))
        self.assertEqual(self.vision.classify_screen(image), ScreenState.LOADING)

    def test_reference_port_and_battle_frames_remain_distinct(self):
        self.assertEqual(
            self.classify(self.FIXTURE_ROOT / "port_ship_selected.png"),
            ScreenState.PORT,
        )
        self.assertEqual(
            self.classify(self.FIXTURE_ROOT / "battle_reference.png"),
            ScreenState.BATTLE,
        )

    def test_battle_bottom_hud_is_not_treated_as_port_carousel(self):
        for filename in ("battle_reference.png", "live_battle.png"):
            image = cv2.imread(str(self.FIXTURE_ROOT / filename))
            self.assertTrue(self.vision._has_battle_hud(image), filename)
            self.assertFalse(self.vision._is_port_ship_bar(image), filename)

        for filename in ("port_ship_selected.png", "port_mode_selector.png"):
            image = cv2.imread(str(self.FIXTURE_ROOT / filename))
            self.assertTrue(self.vision._is_port_ship_bar(image), filename)
            self.assertFalse(self.vision._has_battle_hud(image), filename)

    def test_positive_port_controls_take_precedence_over_battle_candidate(self):
        """A port must never become battle merely because its lower UI is textured."""
        image = cv2.imread(str(self.FIXTURE_ROOT / "port_ship_selected.png"))
        self.assertIsNotNone(image)
        self.assertTrue(self.vision.in_port(image))
        self.assertEqual(self.vision.classify_screen(image), ScreenState.PORT)

    def test_port_with_ship_tooltip_still_requires_and_passes_port_anchors(self):
        image = cv2.imread(str(self.FIXTURE_ROOT / "port_with_ship_tooltip.png"))
        self.assertIsNotNone(image)
        votes = self.vision._port_anchor_votes(image)
        self.assertGreaterEqual(sum(bool(value) for value in votes.values()), 3)
        self.assertEqual(self.vision.classify_screen(image), ScreenState.PORT)

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

    def test_prebattle_start_action_wins_over_false_battle_hud_anchors(self):
        class ConflictingPrebattleVision(type(self).vision.__class__):
            @staticmethod
            def in_loading(_image):
                return True

            @staticmethod
            def _has_loading_start_action(_image):
                return True

            @staticmethod
            def _has_battle_hud(_image):
                return True

            @staticmethod
            def in_results(_image):
                return False

            @staticmethod
            def in_port(_image):
                return False

            @staticmethod
            def _is_login_splash(_image):
                return False

            @staticmethod
            def _is_port_reward_overlay(_image):
                return False

        image = np.zeros((1000, 1600, 3), dtype=np.uint8)

        self.assertEqual(
            ConflictingPrebattleVision().classify_screen(image),
            ScreenState.LOADING,
        )

    def test_translucent_prebattle_start_button_is_detected_at_2k(self):
        """The real roster button is only about 14% of the tight ROI."""
        from core.ui import LOADING_START_BUTTON

        image = np.zeros((1494, 2560, 3), dtype=np.uint8)
        x1, y1, x2, y2 = LOADING_START_BUTTON.pixels(
            image.shape[1], image.shape[0]
        )
        # Match the partially transparent green fill measured from the latest
        # 2K roster capture: wide enough to be a button, but below the old
        # 0.16 area threshold.
        cv2.rectangle(
            image,
            (x1 + 15, y1 + 53),
            (x1 + 15 + 200, y1 + 53 + 18),
            (95, 115, 31),
            -1,
        )
        self.assertTrue(self.vision._has_loading_start_action(image))

    def test_live_consumables_are_not_a_loading_start_action(self):
        image = cv2.imread(str(self.FIXTURE_ROOT / "live_battle.png"))
        self.assertIsNotNone(image)

        self.assertFalse(self.vision._has_loading_start_action(image))

    def test_bright_live_battle_from_walkthrough_is_battle(self):
        self.assertEqual(
            self.classify(
                self.FIXTURE_ROOT
                / "live_run_20260829"
                / "08_battle_start.png"
            ),
            ScreenState.BATTLE,
        )

    def test_muted_ocean_battle_wins_over_false_blue_exit_button(self):
        image = cv2.imread(
            str(self.FIXTURE_ROOT / "live_battle_muted_hud.jpg")
        )
        self.assertIsNotNone(image)
        # This real frame previously matched the blue confirmation-button ROI
        # because open sea filled that whole crop. Its independent HUD anchors
        # must keep the active match in BATTLE.
        self.assertTrue(self.vision.in_exit_confirmation(image))
        self.assertTrue(self.vision._has_battle_hud(image))
        self.assertEqual(self.vision.classify_screen(image), ScreenState.BATTLE)

    def test_battle_remains_actionable_when_player_hud_is_temporarily_obscured(self):
        image = cv2.imread(
            str(
                self.FIXTURE_ROOT
                / "live_run_20260829"
                / "08_battle_start.png"
            )
        )
        self.assertIsNotNone(image)
        height, width = image.shape[:2]
        image[int(height * 0.70) :, : int(width * 0.25)] = 0

        self.assertTrue(self.vision._has_battle_hud(image))
        self.assertEqual(self.vision.classify_screen(image), ScreenState.BATTLE)

    def test_unlabelled_matchmaking_artwork_is_loading(self):
        self.assertEqual(
            self.classify(
                self.FIXTURE_ROOT / "live_run_20260829" / "07_loading.png"
            ),
            ScreenState.LOADING,
        )

    def test_live_result_screen_is_not_exit_confirmation(self):
        self.assertEqual(
            self.classify(self.FIXTURE_ROOT / "results.png"),
            ScreenState.RESULTS,
        )

    def test_scattered_ocean_blue_is_not_an_exit_confirmation_button(self):
        image = np.full((1000, 1600, 3), 120, dtype=np.uint8)
        # Several disconnected cyan patches can exceed the old colour ratio,
        # but they are not one rectangular modal action.
        for x in range(860, 980, 24):
            cv2.circle(image, (x, 485), 9, (200, 120, 40), -1)
        self.assertFalse(self.vision.in_exit_confirmation(image))

    def test_mode_card_colour_is_not_an_escape_resume_button(self):
        image = np.full((1000, 1600, 3), 78, dtype=np.uint8)
        # A small green/olive emblem in the same ROI must not stand in for the
        # large solid resume bar used by the actual escape menu.
        cv2.circle(image, (800, 490), 24, (70, 105, 70), -1)
        self.assertFalse(self.vision.in_escape_menu(image))

    def test_port_reward_overlay_can_never_enter_battle_state(self):
        image = cv2.imread(str(self.FIXTURE_ROOT / "port_reward_overlay.png"))
        self.assertIsNotNone(image)
        self.assertTrue(self.vision._is_port_reward_overlay(image))
        self.assertEqual(self.vision.classify_screen(image), ScreenState.UNKNOWN)

        # The negative guard must not mask either reference battle frame.
        for filename in ("battle_reference.png", "live_battle.png"):
            battle = cv2.imread(str(self.FIXTURE_ROOT / filename))
            self.assertFalse(self.vision._is_port_reward_overlay(battle), filename)
            self.assertEqual(self.vision.classify_screen(battle), ScreenState.BATTLE)

    def test_port_reward_card_is_port_not_live_battle(self):
        from core.results import ResultRewardReader

        normal_port = cv2.imread(str(self.FIXTURE_ROOT / "port_ship_selected.png"))
        self.assertFalse(ResultRewardReader._looks_like_port_reward_card(normal_port))
        candidates = sorted(
            Path("runtime/screenshots/runs").rglob(
                "result_unrecognized_*.png"
            )
        )
        port_cards = [
            image
            for image in (cv2.imread(str(path)) for path in candidates)
            if image is not None
            and ResultRewardReader._looks_like_port_reward_card(image)
        ]
        self.assertGreaterEqual(len(port_cards), 1)
        for image in port_cards:
            self.assertEqual(self.vision.classify_screen(image), ScreenState.PORT)

if __name__ == "__main__":
    unittest.main()
