"""Offline UI movement/zoom tests using OCR boxes from actual pixel markers."""

from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
import pytest

import port_navigator as navigation
from core.ocr import OcrToken
from core.results import ResultRewardReader


class GeometryBackend:
    """Return OCR geometry from visible markers, independent of crop origin."""

    supports_ui_anchors = True
    execution_provider = "offline_geometry"

    def __init__(self):
        self.markers = {}
        self.calls = 0

    def paint(self, image, text, bounds):
        index = len(self.markers)
        # Saturated red/purple markers do not resemble resource-icon colours.
        color = (60 + index * 11, 25, 145 + index * 9)
        x1, y1, x2, y2 = map(int, bounds)
        image[y1:y2, x1:x2] = color
        self.markers[color] = (text, (x2 - x1) * (y2 - y1))
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    def recognize(self, image):
        self.calls += 1
        tokens = []
        for color, (text, full_area) in self.markers.items():
            ys, xs = np.where(np.all(image == color, axis=2))
            if len(xs) < full_area * .90:
                continue
            x1, x2, y1, y2 = int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1
            tokens.append(OcrToken(text, .99, ((x1, y1), (x2, y1), (x2, y2), (x1, y2))))
        return tokens


def star(image, center, radius, color):
    angles = np.arange(10) * np.pi / 5 - np.pi / 2
    radii = np.where(np.arange(10) % 2, radius * .42, radius)
    points = np.column_stack((np.cos(angles) * radii + center[0],
                              np.sin(angles) * radii + center[1])).astype(np.int32)
    cv2.fillPoly(image, [points], color)


def reward_panel(shape, position, unit):
    image = np.zeros((*shape, 3), np.uint8)
    backend = GeometryBackend()
    x, y = position
    backend.paint(image, "在战斗中获得", (x, y, x + 6 * unit, y + unit))
    baseline = y + 2.2 * unit
    backend.paint(image, "67 769", (x, baseline - .5 * unit, x + 4.4 * unit, baseline + .5 * unit))
    backend.paint(image, "1 747", (x + 6.2 * unit, baseline - .5 * unit,
                                  x + 9.6 * unit, baseline + .5 * unit))
    backend.paint(image, "250", (x + 11.4 * unit, baseline - .5 * unit,
                                x + 14 * unit, baseline + .5 * unit))
    cv2.circle(image, (int(x + 5.2 * unit), int(baseline)), int(.6 * unit), (240, 240, 240), -1)
    star(image, (x + 10.5 * unit, baseline), .65 * unit, (240, 240, 240))
    star(image, (x + 14.9 * unit, baseline), .65 * unit, (30, 230, 30))
    return image, backend


@pytest.mark.parametrize("shape,position,unit", [
    ((768, 1024), (120, 270), 18),
    ((900, 2400), (1560, 610), 28),
    ((1600, 1200), (240, 990), 36),
    # 4K framebuffer with compact UI: no global-resize equivalence.
    ((2160, 3840), (2300, 1400), 22),
])
def test_earned_rewards_follow_local_caption_and_icon_geometry(shape, position, unit):
    image, backend = reward_panel(shape, position, unit)
    result = ResultRewardReader(backend).read(image)
    assert result.recognized
    assert result.resource_values() == {"credits": 67769, "ship_xp": 1747, "free_xp": 250}


def test_account_resources_in_old_reward_rois_cannot_become_earned_rewards():
    image = np.zeros((1000, 1600, 3), np.uint8)
    backend = GeometryBackend()
    backend.paint(image, "67 769", (280, 370, 370, 405))
    backend.paint(image, "1 747", (460, 370, 545, 405))
    backend.paint(image, "250", (595, 370, 650, 405))
    backend.paint(image, "账户总资源", (1230, 35, 1440, 65))
    result = ResultRewardReader(backend).read(image)
    assert not result.recognized
    assert result.resource_values() == {"credits": 0, "ship_xp": 0, "free_xp": 0}


@pytest.mark.parametrize("shape,label,bounds,finder", [
    ((900, 2400), "加入战斗", (250, 145, 450, 173), navigation.port_battle_action_point),
    ((1600, 1200), "联合作战", (750, 270, 950, 302), navigation.port_mode_selector_action_point),
    ((2160, 3840), "返回港口", (400, 1570, 610, 1600), navigation.results_return_to_port_action_point),
    ((768, 1024), "继续战斗", (360, 450, 485, 478), navigation.results_requeue_action_point),
])
def test_actions_map_moved_text_to_actual_frame_coordinates(shape, label, bounds, finder):
    image = np.zeros((*shape, 3), np.uint8)
    backend = GeometryBackend()
    expected = backend.paint(image, label, bounds)
    assert finder(image, backend) == expected


def test_original_button_roi_is_the_fast_path():
    image = np.zeros((1000, 1600, 3), np.uint8)
    backend = GeometryBackend()
    expected = backend.paint(image, "加入战斗", (675, 20, 805, 45))
    assert navigation.port_battle_action_point(image, backend) == expected
    assert backend.calls == 1


def test_separate_numeric_boxes_can_anchor_a_moved_reward_row_without_icons():
    image = np.zeros((1200, 1800, 3), np.uint8)
    backend = GeometryBackend()
    backend.paint(image, "在战斗中获得", (950, 670, 1160, 700))
    backend.paint(image, "67 769", (950, 725, 1090, 755))
    backend.paint(image, "1 747", (1160, 725, 1280, 755))
    backend.paint(image, "250", (1350, 725, 1430, 755))
    reward = ResultRewardReader(backend).read(image)
    assert reward.recognized
    assert reward.resource_values() == {"credits": 67769, "ship_xp": 1747, "free_xp": 250}


def test_mode_detection_and_selector_card_follow_moved_labels():
    image = np.zeros((1600, 1200, 3), np.uint8)
    backend = GeometryBackend()
    backend.paint(image, "联合作战", (750, 265, 950, 297))
    assert navigation.detect_port_mode(image, backend) == "cooperative"

    image[:] = 0
    backend = GeometryBackend()
    backend.paint(image, "选择一种战斗模式", (60, 290, 380, 326))
    expected = backend.paint(image, "非对称战斗", (940, 1210, 1140, 1244))
    assert navigation.in_battle_type_selector(image, backend)
    assert navigation._find_mode_card_from_ocr(image, "asymmetric", backend) == expected


def test_survey_can_move_outside_the_original_dialog_crop():
    image = np.zeros((900, 2400, 3), np.uint8)
    backend = GeometryBackend()
    backend.paint(image, "战斗评价", (90, 260, 260, 289))
    expected = backend.paint(image, "跳过", (140, 410, 205, 439))
    assert navigation.is_battle_survey_page(image, backend)
    assert navigation.battle_survey_dismiss_point(image, backend) == expected


def test_moved_quit_dialog_associates_cancel_with_its_own_title():
    image = np.zeros((900, 2400, 3), np.uint8)
    backend = GeometryBackend()
    backend.paint(image, "确认", (280, 275, 360, 299))
    backend.paint(image, "退出游戏？", (220, 320, 420, 345))
    expected = backend.paint(image, "否", (360, 420, 400, 448))
    backend.paint(image, "否", (2010, 790, 2050, 818))
    assert navigation.port_exit_confirmation_no_action_point(image, backend) == expected


def test_loading_start_click_uses_moved_ocr_box_and_rechecks_pause():
    image = np.zeros((1600, 1200, 3), np.uint8)
    backend = GeometryBackend()
    expected = backend.paint(image, "开始战斗", (230, 1030, 380, 1060))
    vision = SimpleNamespace(_has_loading_start_action=lambda _: True)
    with patch.object(navigation, "_click_local", return_value=True) as click:
        assert navigation.click_loading_start_action(image=image, vision=vision, backend=backend)
    click.assert_called_once_with(None, expected)
    with patch.object(navigation, "_click_local") as click:
        assert not navigation.click_loading_start_action(
            image=image, vision=vision, backend=backend, should_abort=lambda: True,
        )
    click.assert_not_called()


def test_loading_start_fallback_uses_the_detected_rectangle_center():
    image = np.zeros((1200, 2400, 3), np.uint8)
    vision = SimpleNamespace(
        _has_loading_start_action=lambda _: True,
        loading_start_action_box=lambda _: (940, 940, 1220, 1010),
    )
    with patch.object(navigation, "_click_local", return_value=True) as click:
        assert navigation.click_loading_start_action(image=image, vision=vision, backend=GeometryBackend())
    click.assert_called_once_with(None, (1080, 975))


def test_moved_carousel_name_uses_its_local_glyph_scale():
    image = np.zeros((1600, 1200, 3), np.uint8)
    backend = GeometryBackend()
    backend.paint(image, "测试战舰", (190, 925, 380, 961))
    point, confidence = navigation.find_custom_ship_card(image, "测试战舰", backend)
    assert point == (285, 925)
    assert confidence == .99


@pytest.mark.parametrize("shape,ui_scale,position", [
    ((900, 2400), .64, (350, 540)),
    ((1600, 1200), 1.35, (440, 1090)),
    ((2160, 3840), .72, (1800, 1590)),
])
def test_ship_templates_match_independent_ui_zoom(shape, ui_scale, position):
    # Nonrepeating glyph strokes provide a meaningful template correlation.
    template = np.zeros((28, 110, 3), np.uint8)
    cv2.putText(template, "SHIP", (2, 23), cv2.FONT_HERSHEY_SIMPLEX, .8, (20, 180, 230), 2)
    height, width = shape
    scale = height / navigation.SHIP_REFERENCE_SIZE[1] * ui_scale
    rendered = cv2.resize(template, None, fx=scale, fy=scale,
                          interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
    image = np.zeros((*shape, 3), np.uint8)
    x, y = position
    image[y:y + rendered.shape[0], x:x + rendered.shape[1]] = rendered
    with patch.object(navigation, "_load_ship_name_template", return_value=template):
        found = navigation.find_ship_card(image, "pommern", minimum_score=.84)
    assert found is not None
    point, score = found
    assert x <= point[0] <= x + rendered.shape[1]
    assert abs(point[1] - y) <= rendered.shape[0]
    assert score >= .84
