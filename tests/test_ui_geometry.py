import unittest

from core.ui import (
    BATTLE_TYPE_COOPERATIVE_CARD,
    MINIMAP_REGION,
    PORT_BATTLE_BUTTON,
    PORT_MODE_SELECTOR,
    SHIP_NAME_TEMPLATES,
)


class UiGeometryTests(unittest.TestCase):
    def test_port_button_matches_reference_resolution(self):
        x, y = PORT_BATTLE_BUTTON.center(2560, 1600)

        self.assertTrue(1260 <= x <= 1300)
        self.assertTrue(20 <= y <= 40)

    def test_minimap_is_large_and_bottom_right_anchored(self):
        x1, y1, x2, y2 = MINIMAP_REGION.pixels(2560, 1600)

        self.assertEqual((x2, y2), (2560, 1600))
        self.assertLessEqual(x1, 1880)
        self.assertLessEqual(y1, 920)
        self.assertGreaterEqual(x2 - x1, 680)
        self.assertGreaterEqual(y2 - y1, 680)

    def test_cooperative_card_is_upper_left_mode_card(self):
        x, y = BATTLE_TYPE_COOPERATIVE_CARD.center(2560, 1600)

        self.assertTrue(950 <= x <= 1000)
        self.assertTrue(530 <= y <= 570)

    def test_mode_selector_is_immediately_right_of_battle_button(self):
        battle_x, _ = PORT_BATTLE_BUTTON.center(2560, 1600)
        mode_x, mode_y = PORT_MODE_SELECTOR.center(2560, 1600)

        self.assertTrue(1450 <= mode_x <= 1530)
        self.assertGreater(mode_x, battle_x)
        self.assertTrue(20 <= mode_y <= 45)

    def test_only_supported_ship_templates_are_exposed(self):
        self.assertEqual(set(SHIP_NAME_TEMPLATES), {"napoli", "pommern"})


if __name__ == "__main__":
    unittest.main()
