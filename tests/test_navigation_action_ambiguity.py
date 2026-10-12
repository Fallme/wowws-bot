import numpy as np

import port_navigator as navigation
from test_ui_scale_navigation import GeometryBackend


def test_moved_duplicate_battle_buttons_do_not_authorize_an_arbitrary_click():
    image = np.zeros((1080, 1920, 3), dtype=np.uint8)
    backend = GeometryBackend()
    backend.paint(image, "加入战斗", (280, 150, 450, 180))
    backend.paint(image, "加入战斗", (1330, 150, 1500, 180))
    assert navigation.port_battle_action_point(image, backend) is None


def test_a_single_moved_button_still_returns_its_actual_coordinate():
    image = np.zeros((1080, 1920, 3), dtype=np.uint8)
    backend = GeometryBackend()
    expected = backend.paint(image, "加入战斗", (280, 150, 450, 180))
    assert navigation.port_battle_action_point(image, backend) == expected
