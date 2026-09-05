import cv2
import numpy as np
import pytest
from pathlib import Path

from core.ocr import OcrToken, RapidOcrBackend
from core.ui import (
    FIRE_DIAL_ROI,
    FIRE_SHIP_ROI,
    FIRE_STATUS_ROI,
    NO_COMMANDER_CONFIRM_BUTTON,
)
from core.vision import PlayerPose, Vision


@pytest.fixture(scope="module")
def live_frame():
    image = cv2.imread(str(Path("tests") / "fixtures" / "live_battle.png"))
    assert image is not None
    return image


def test_live_viewport_groups_target_strokes(live_frame):
    enemies = Vision().find_enemies_in_viewport(live_frame)
    assert len(enemies) == 1


def test_live_health_uses_ship_status_gauge(live_frame):
    vision = Vision()
    health = vision.health_pct(vision.find_health_bar(live_frame))
    assert health == pytest.approx(52344 / 86000, abs=0.04)


def test_health_ocr_keeps_grouped_digits_as_one_fraction(live_frame):
    class Backend:
        @staticmethod
        def recognize(_image):
            return [
                OcrToken("波美拉尼亚", 0.99),
                OcrToken("52 344 / 86 000", 0.96),
            ]

    assert Vision.read_health_fraction(
        live_frame, Backend()
    ) == pytest.approx(52344 / 86000)


def test_health_ocr_retries_an_enhanced_crop_after_original_failure(live_frame):
    class RetryBackend(RapidOcrBackend):
        def __init__(self):
            self.calls = 0

        def recognize(self, _image):
            self.calls += 1
            if self.calls == 1:
                return []
            return [OcrToken("52 344 / 86 000", 0.92)]

    backend = RetryBackend()
    assert Vision.read_health_fraction(live_frame, backend) == pytest.approx(
        52344 / 86000
    )
    assert backend.calls == 2


def test_live_player_arrow_uses_range_ring(live_frame):
    vision = Vision()
    minimap = vision.find_minimap(live_frame)
    player = vision.find_player_on_minimap(minimap)
    assert player is not None
    assert player[0] == pytest.approx(157, abs=4)
    assert player[1] == pytest.approx(153, abs=4)


def test_live_player_pose_exposes_a_normalized_heading(live_frame):
    vision = Vision()
    minimap = vision.find_minimap(live_frame)
    pose = vision.find_player_pose_on_minimap(minimap)
    assert pose is not None
    assert np.hypot(*pose.heading) == pytest.approx(1.0)
    # The white bow arrow and its heading line point toward the lower-left.
    assert pose.heading[0] < -0.5
    assert pose.heading[1] > 0.5


def test_tactical_map_crop_matches_native_autopilot_grid_geometry():
    frame = np.zeros((1000, 1600, 3), dtype=np.uint8)
    expected_size = 810
    left = (1600 - expected_size) // 2
    top = (1000 - expected_size) // 2
    grid = np.zeros((expected_size, expected_size, 3), dtype=np.uint8)
    grid[:, ::2] = 220
    frame[top : top + expected_size, left : left + expected_size] = grid

    tactical_map = Vision.find_tactical_map(frame)

    assert tactical_map is not None
    assert tactical_map.shape == (expected_size, expected_size, 3)
    assert np.array_equal(tactical_map, grid)


def test_real_tactical_grid_rectifies_capture_points_to_minimap_coordinates():
    path = (
        Path("training_assets")
        / "user_captures"
        / "codex-clipboard-68c91356-07a4-4753-a25a-0824b8b62c6f.png"
    )
    frame = cv2.imread(str(path))
    assert frame is not None
    vision = Vision(screen_capture=object())

    tactical_map = vision.find_tactical_map(frame)
    minimap = vision.find_minimap(frame)

    assert tactical_map is not None
    assert minimap is not None
    assert 1240 <= tactical_map.shape[0] <= 1340
    assert tactical_map.shape[0] == tactical_map.shape[1]
    tactical_zones = vision.find_capture_zones(tactical_map)
    minimap_zones = vision.find_capture_zones(minimap)
    assert [zone.label for zone in tactical_zones] == ["A", "B", "C"]
    assert [zone.label for zone in minimap_zones] == ["A", "B", "C"]
    for tactical_zone, minimap_zone in zip(tactical_zones, minimap_zones):
        tactical_position = (
            tactical_zone.center[0] / tactical_map.shape[1],
            tactical_zone.center[1] / tactical_map.shape[0],
        )
        minimap_position = (
            minimap_zone.center[0] / minimap.shape[1],
            minimap_zone.center[1] / minimap.shape[0],
        )
        assert tactical_position[0] == pytest.approx(
            minimap_position[0],
            abs=0.025,
        )
        assert tactical_position[1] == pytest.approx(
            minimap_position[1],
            abs=0.025,
        )
    # The translucent M plane is poor terrain evidence; the concurrently
    # visible small minimap retains the complete, high-contrast island layer.
    assert len(vision.find_minimap_island_outlines(minimap)) >= 15


@pytest.mark.parametrize(
    ("points", "expected"),
    [
        ([[602, 60], [610, 77], [616, 60]], (610, 77)),
        ([[616, 83], [616, 98], [634, 91]], (634, 91)),
        ([[657, 60], [642, 73], [655, 79]], (657, 60)),
        ([[210, 86], [203, 104], [217, 104]], (210, 86)),
    ],
)
def test_player_arrow_tip_is_opposite_its_short_stern_edge(points, expected):
    """Regression polygons sampled from live 2K minimap arrows."""
    tip = Vision._arrow_tip(np.asarray(points, dtype=np.float64))

    assert tuple(tip) == expected


def test_player_arrow_near_minimap_edge_is_still_detected():
    minimap = np.full((320, 320, 3), 35, dtype=np.uint8)
    center = (165, 303)
    cv2.circle(minimap, center, 42, (0, 180, 210), 2)
    cv2.fillConvexPoly(
        minimap,
        np.array([[165, 296], [160, 311], [170, 311]], dtype=np.int32),
        (245, 245, 245),
    )

    pose = Vision().find_player_pose_on_minimap(minimap)

    assert pose is not None
    assert pose.position[0] == pytest.approx(center[0], abs=3)
    assert pose.position[1] == pytest.approx(303, abs=4)
    assert pose.heading[1] < -0.8


def test_minimap_grid_converts_one_cell_to_five_kilometres():
    minimap = np.full((690, 690, 3), 35, dtype=np.uint8)

    assert Vision().minimap_pixels_to_km(minimap, 69) == pytest.approx(5.0)


def test_speed_ocr_uses_kts_value_not_engine_notch_label(live_frame):
    class SpeedBackend:
        @staticmethod
        def recognize(_image):
            return [
                OcrToken("3/4", 0.99, ((10, 0),)),
                OcrToken("23.6kts", 0.92, ((90, 0),)),
            ]

    assert Vision.read_speed_knots(live_frame, SpeedBackend()) == pytest.approx(23.6)


def test_speed_ocr_retries_an_enhanced_crop_after_original_failure(live_frame):
    class RetryBackend(RapidOcrBackend):
        def __init__(self):
            self.calls = 0

        def recognize(self, _image):
            self.calls += 1
            if self.calls == 1:
                return []
            return [OcrToken("29.2 kts", 0.91)]

    backend = RetryBackend()
    assert Vision.read_speed_knots(live_frame, backend) == pytest.approx(29.2)
    assert backend.calls == 2


def test_central_capture_circle_is_selected():
    minimap = np.full((420, 420, 3), (80, 105, 120), dtype=np.uint8)
    cv2.circle(minimap, (210, 210), 45, (220, 220, 220), 2)
    cv2.rectangle(minimap, (202, 202), (218, 218), (220, 220, 220), 2)
    cv2.circle(minimap, (90, 320), 45, (220, 220, 220), 2)

    zone = Vision().find_central_capture_zone(minimap)

    assert zone is not None
    assert zone.center[0] == pytest.approx(210, abs=4)
    assert zone.center[1] == pytest.approx(210, abs=4)
    assert zone.radius == pytest.approx(45, abs=5)


def test_nearest_capture_circle_is_selected_from_player_position():
    minimap = np.full((420, 420, 3), (80, 105, 120), dtype=np.uint8)
    cv2.circle(minimap, (95, 100), 40, (220, 220, 220), 2)
    cv2.circle(minimap, (315, 300), 40, (220, 220, 220), 2)

    zone = Vision().find_nearest_capture_zone(minimap, (70, 210))

    assert zone is not None
    assert zone.center[0] == pytest.approx(95, abs=4)
    assert zone.center[1] == pytest.approx(100, abs=4)


def test_three_capture_points_are_recovered_from_alignment_amid_range_rings():
    minimap = np.full((690, 690, 3), (45, 58, 64), dtype=np.uint8)
    player = (355, 165)
    # Player range rings and an offset circular decoy must not become points.
    cv2.circle(minimap, player, 70, (210, 210, 210), 2)
    cv2.circle(minimap, (430, 235), 62, (205, 205, 205), 2)
    for center in ((180, 350), (325, 350), (470, 350)):
        cv2.circle(minimap, center, 48, (230, 230, 230), 2)

    zones = Vision().find_capture_zones(minimap, player)

    assert [zone.label for zone in zones] == ["A", "B", "C"]
    assert [zone.center[0] for zone in zones] == pytest.approx(
        [180, 325, 470], abs=5
    )
    assert Vision().find_nearest_capture_zone(minimap, player).label == "B"


def test_obscured_middle_capture_point_is_inferred_and_player_is_inside():
    minimap = np.full((684, 684, 3), (45, 58, 64), dtype=np.uint8)
    player = (335, 327)
    cv2.circle(minimap, (177, 347), 48, (230, 230, 230), 2)
    cv2.circle(minimap, (320, 346), 64, (210, 210, 210), 2)
    cv2.circle(minimap, (463, 346), 48, (230, 230, 230), 2)

    zone = Vision().find_nearest_capture_zone(minimap, player)

    assert zone.label == "B"
    assert zone.center == pytest.approx((320, 346), abs=5)
    assert np.hypot(player[0] - zone.center[0], player[1] - zone.center[1]) < zone.radius


def test_capture_formation_is_detected_at_map_specific_diagonal_angle():
    minimap = np.full((600, 600, 3), (45, 58, 64), dtype=np.uint8)
    centers = ((150, 180), (300, 300), (450, 420))
    for center in centers:
        cv2.circle(minimap, center, 44, (230, 230, 230), 2)

    zones = Vision().find_capture_zones(minimap, (520, 100))

    assert len(zones) == 3
    for zone, expected in zip(zones, centers):
        assert zone.center == pytest.approx(expected, abs=5)


def test_live_nearest_capture_circle_rejects_player_range_ring(live_frame):
    vision = Vision()
    minimap = vision.find_minimap(live_frame)
    player = vision.find_player_on_minimap(minimap)

    zone = vision.find_nearest_capture_zone(minimap, player)

    assert zone is not None
    assert zone.center[0] == pytest.approx(133, abs=6)
    assert zone.center[1] == pytest.approx(353, abs=6)


def test_live_capture_letters_keep_their_visible_ownership_state(live_frame):
    vision = Vision()
    minimap = vision.find_minimap(live_frame)
    player = vision.find_player_on_minimap(minimap)

    zones = vision.find_capture_zones(minimap, player)

    assert [(zone.label, zone.state) for zone in zones] == [
        ("A", "hostile"),
        ("B", "friendly"),
        ("C", "neutral"),
    ]


def test_player_range_ring_beats_dense_yellow_clutter():
    minimap = np.full((420, 420, 3), 35, dtype=np.uint8)
    player = (210, 95)
    cv2.circle(minimap, player, 50, (0, 180, 210), 2)
    cv2.fillConvexPoly(
        minimap,
        np.array([[210, 86], [203, 104], [217, 104]], dtype=np.int32),
        (245, 245, 245),
    )

    # A second white triangle has more nearby yellow pixels, but those pixels
    # are concentrated in one direction rather than forming a centred ring.
    decoy = (95, 300)
    cv2.fillConvexPoly(
        minimap,
        np.array([[95, 291], [88, 309], [102, 309]], dtype=np.int32),
        (245, 245, 245),
    )
    minimap[250:285, 110:175] = (0, 180, 210)

    pose = Vision().find_player_pose_on_minimap(minimap)

    assert pose is not None
    assert pose.position[0] == pytest.approx(player[0], abs=3)
    assert pose.position[1] == pytest.approx(player[1], abs=3)


def test_player_arrow_silhouette_beats_capture_glyph_in_stronger_ring():
    minimap = np.full((650, 690, 3), 35, dtype=np.uint8)
    player = (350, 69)
    # The opening player ring is partly obscured by labels at spawn.
    cv2.ellipse(minimap, player, (72, 72), 0, 20, 330, (0, 180, 210), 2)
    cv2.fillConvexPoly(
        minimap,
        np.array([[350, 60], [343, 78], [357, 78]], dtype=np.int32),
        (245, 245, 245),
    )

    # Reproduce the stronger circular evidence around a non-arrow map glyph.
    decoy = (383, 381)
    cv2.circle(minimap, decoy, 72, (0, 180, 210), 3)
    cv2.rectangle(minimap, (379, 374), (387, 388), (245, 245, 245), -1)

    pose = Vision().find_player_pose_on_minimap(minimap)

    assert pose is not None
    assert pose.position[0] == pytest.approx(player[0], abs=3)
    assert pose.position[1] == pytest.approx(player[1], abs=3)


@pytest.mark.parametrize("width", [1920, 2560, 3840])
def test_live_arrow_without_yellow_range_ring(width):
    image = cv2.imread(str(Path(__file__).parent / "fixtures" / "autopilot_no_yellow_ring.png"))
    assert image is not None
    image = cv2.resize(image, (width, round(image.shape[0] * width / image.shape[1])))
    vision = Vision(screen_capture=object())
    minimap = vision.find_minimap(image)
    pose = vision.find_player_pose_on_minimap(minimap)
    assert pose is not None
    assert pose.position[0] / minimap.shape[1] == pytest.approx(177 / 691, abs=0.008)
    assert pose.position[1] / minimap.shape[0] == pytest.approx(88 / 650, abs=0.008)
    assert pose.heading[1] > 0.9


def test_two_white_arrows_without_range_evidence_are_ambiguous():
    minimap = np.full((650, 690, 3), 35, dtype=np.uint8)
    for x in (200, 400):
        cv2.fillConvexPoly(minimap, np.array([[x, 100], [x - 7, 118], [x + 7, 118]]), (245, 245, 245))
    assert Vision(screen_capture=object()).find_player_pose_on_minimap(minimap) is None


@pytest.mark.parametrize("color", [(140, 210, 140), (245, 245, 245)])
def test_no_ring_fallback_rejects_colored_arrow_and_white_rectangle(color):
    minimap = np.full((650, 690, 3), 35, dtype=np.uint8)
    if color == (245, 245, 245):
        cv2.rectangle(minimap, (200, 100), (214, 118), color, -1)
    else:
        cv2.fillConvexPoly(minimap, np.array([[207, 100], [200, 118], [214, 118]]), color)
    assert Vision(screen_capture=object()).find_player_pose_on_minimap(minimap) is None


def test_live_island_signal_detects_terrain_in_the_corrected_bow_direction(live_frame):
    vision = Vision()
    minimap = vision.find_minimap(live_frame)
    pose = vision.find_player_pose_on_minimap(minimap)
    risk = vision.find_island_risk(minimap, pose)
    assert risk is not None
    assert risk.distance < 0.10


def test_island_detector_selects_the_clearer_turn_side():
    minimap = np.full((320, 320, 3), 35, dtype=np.uint8)
    # A solid neutral island overlaps the forward-left corridor.
    minimap[65:120, 118:154] = (210, 210, 210)
    pose = PlayerPose(position=(160, 190), heading=(0.0, -1.0))

    risk = Vision().find_island_risk(minimap, pose)

    assert risk is not None
    assert risk.distance < 0.30
    assert risk.avoidance_rudder > 0


def test_thin_range_ring_is_not_treated_as_an_island():
    minimap = np.full((320, 320, 3), 35, dtype=np.uint8)
    cv2.circle(minimap, (160, 190), 85, (220, 220, 220), 1)
    pose = PlayerPose(position=(160, 190), heading=(0.0, -1.0))

    assert Vision().find_island_risk(minimap, pose) is None


def test_connected_grid_and_range_rings_are_not_islands():
    minimap = np.full((320, 320, 3), 35, dtype=np.uint8)
    for coordinate in range(0, 321, 32):
        cv2.line(minimap, (coordinate, 0), (coordinate, 319), (190, 190, 190), 1)
        cv2.line(minimap, (0, coordinate), (319, coordinate), (190, 190, 190), 1)
    cv2.circle(minimap, (160, 190), 80, (200, 200, 200), 1)
    pose = PlayerPose(position=(160, 190), heading=(0.0, -1.0))

    assert Vision().find_island_risk(minimap, pose) is None


def test_kinematic_rudder_turns_around_a_predicted_collision():
    pose = PlayerPose((160, 180), (0.0, -1.0))
    island = {
        "points": [
            (0.495, 0.52),
            (0.505, 0.52),
            (0.505, 0.53),
            (0.495, 0.53),
        ]
    }

    plan = Vision.plan_kinematic_rudder(
        (320, 320, 3),
        pose,
        (160, 30),
        [island],
        speed_knots=30,
        rudder_shift_seconds=15,
        turning_radius_km=1,
        preferred_side=1,
    )

    assert plan.avoidance_required
    assert plan.collision_time_seconds == pytest.approx(87, abs=2)
    assert plan.rudder > 0
    assert plan.minimum_clearance_km > 0.45


def test_kinematic_rudder_steers_toward_a_clear_forward_enemy():
    plan = Vision.plan_kinematic_rudder(
        (320, 320, 3),
        PlayerPose((160, 180), (0.0, -1.0)),
        (260, 180),
        [],
        speed_knots=30,
        rudder_shift_seconds=15,
        turning_radius_km=1,
    )

    assert not plan.avoidance_required
    assert plan.rudder > 0


def test_kinematic_rudder_may_hold_straight_when_enemy_turn_crosses_land():
    island = {
        "points": [
            (0.515, 0.53),
            (0.54, 0.53),
            (0.54, 0.55),
            (0.515, 0.55),
        ]
    }

    plan = Vision.plan_kinematic_rudder(
        (320, 320, 3),
        PlayerPose((160, 180), (0.0, -1.0)),
        (260, 180),
        [island],
    )

    assert plan.avoidance_required
    assert plan.rudder == 0.0
    assert plan.minimum_clearance_km > 0.45


def test_snow_islands_produce_browser_outlines_while_grid_lines_are_filtered():
    minimap = np.full((420, 420, 3), (72, 48, 30), dtype=np.uint8)
    for coordinate in range(0, 421, 42):
        cv2.line(minimap, (coordinate, 0), (coordinate, 419), (105, 82, 63), 1)
        cv2.line(minimap, (0, coordinate), (419, coordinate), (105, 82, 63), 1)
    cv2.fillConvexPoly(
        minimap,
        np.array([[70, 90], [125, 65], [155, 108], [112, 145], [62, 132]], dtype=np.int32),
        (238, 232, 222),
    )
    cv2.fillConvexPoly(
        minimap,
        np.array([[265, 245], [330, 230], [357, 278], [312, 327], [252, 300]], dtype=np.int32),
        (245, 240, 232),
    )

    outlines = Vision().find_minimap_island_outlines(minimap)

    assert len(outlines) == 2
    assert all(len(item["points"]) >= 3 for item in outlines)
    assert all(item["area"] > 0.005 for item in outlines)


def test_compact_range_ring_fits_are_not_promoted_to_capture_points():
    minimap = np.full((650, 691, 3), (60, 45, 35), dtype=np.uint8)
    player = (609, 69)
    for center in ((559, 86), (554, 180), (548, 274)):
        cv2.circle(minimap, center, 37, (220, 220, 220), 2)

    assert Vision().find_capture_zones(minimap, player) == []


def test_autopilot_hud_indicator_requires_green_enabled_text():
    enabled = np.zeros((1600, 2560, 3), dtype=np.uint8)
    disabled = enabled.copy()
    enabled[1280:1380, 300:520] = (40, 190, 40)
    disabled[1280:1380, 300:520] = (150, 150, 150)

    assert Vision().is_autopilot_enabled(enabled)
    assert not Vision().is_autopilot_enabled(disabled)


def test_minimap_enemy_mask_rejects_brown_islands_and_clusters_red_glyphs():
    hsv = np.zeros((320, 320, 3), dtype=np.uint8)
    hsv[:] = (100, 55, 75)
    # Saturated brown/orange terrain used to be counted as nearby enemies.
    cv2.circle(hsv, (85, 85), 35, (20, 80, 170), -1)
    cv2.rectangle(hsv, (180, 55), (250, 105), (16, 70, 155), -1)
    # Two red ship markers, each with a nearby label stroke.
    for center_x, center_y in ((120, 210), (245, 180)):
        cv2.fillConvexPoly(
            hsv,
            np.array(
                [
                    [center_x, center_y - 8],
                    [center_x - 7, center_y + 7],
                    [center_x + 7, center_y + 7],
                ],
                dtype=np.int32,
            ),
            (4, 230, 245),
        )
        cv2.rectangle(
            hsv,
            (center_x + 14, center_y - 3),
            (center_x + 22, center_y + 3),
            (4, 220, 235),
            -1,
        )

    enemies = Vision().find_enemies_from_hsv(hsv)

    assert len(enemies) == 2
    assert all(point[1] > 150 for point in enemies)


def test_rudder_indicator_reads_q_and_e_without_viewport_navigation():
    neutral = np.zeros((1600, 2560, 3), dtype=np.uint8)
    q_frame = neutral.copy()
    e_frame = neutral.copy()
    q_frame[1210:1260, 1100:1140] = (40, 210, 40)
    e_frame[1210:1260, 1390:1430] = (40, 210, 40)

    assert Vision().detect_rudder_indicator(neutral) == "neutral"
    assert Vision().detect_rudder_indicator(q_frame) == "Q"
    assert Vision().detect_rudder_indicator(e_frame) == "E"


def test_autopilot_status_uses_ocr_text_box_colour_not_broad_green_hud():
    height, width = 1600, 2560
    image = np.zeros((height, width, 3), dtype=np.uint8)
    crop_top = int(height * 0.68)
    box = ((100.0, 100.0), (240.0, 100.0), (240.0, 140.0), (100.0, 140.0))

    class Backend:
        @staticmethod
        def recognize(_image):
            return [OcrToken("自动驾驶", 0.99, box)]

    # Unrelated green content outside the OCR glyph box is ignored.
    image[1200:1300, 500:650] = (0, 255, 0)
    assert not Vision.read_autopilot_enabled_text(image, Backend())

    image[crop_top + 100 : crop_top + 141, 100:241] = (80, 180, 80)
    assert Vision.read_autopilot_enabled_text(image, Backend())


def test_autopilot_status_reports_unreadable_separately_from_absent():
    image = np.zeros((1600, 2560, 3), dtype=np.uint8)

    class FailingBackend:
        @staticmethod
        def recognize(_image):
            raise RuntimeError("ocr unavailable")

    class EmptyBackend:
        @staticmethod
        def recognize(_image):
            return []

    assert Vision.read_autopilot_enabled_text(image, None) is None
    assert Vision.read_autopilot_enabled_text(image, FailingBackend()) is None
    assert Vision.read_autopilot_enabled_text(image, EmptyBackend()) is False


def test_no_commander_detector_requires_confirm_button():
    image = np.full((1000, 1600, 3), 75, dtype=np.uint8)
    image[360:660, 480:1120] = 40
    x1, y1, x2, y2 = NO_COMMANDER_CONFIRM_BUTTON.pixels(1600, 1000)
    # Teal BGR button matching the live warning's affirmative action.
    image[y1:y2, x1:x2] = (105, 100, 35)
    assert Vision().in_no_commander_confirmation(image)

    image[y1:y2, x1:x2] = 45
    assert not Vision().in_no_commander_confirmation(image)


FIRE_FIXTURES = [
    "fire_dual_anchor.jpg",
    "fire_live_siberia.jpg",
    "fire_live_1080p.jpg",
    "fire_live_2k.jpg",
]


def _draw_fire_droplet(image, cx, cy, stem_color=(0, 90, 160), blob_color=(0, 128, 255)):
    """Draw the HUD flame marker: pin stem + round head with a darker core.

    Both tones stay inside the orange mask (H < 40, S/V >= 110), so the
    component develops a value gradient instead of the flat profile of a
    solid dot, and the fill/wh geometry matches the droplet gates.
    """
    cv2.rectangle(image, (cx - 4, cy - 16), (cx + 4, cy - 4), stem_color, -1)
    cv2.circle(image, (cx, cy + 2), 9, blob_color, -1)


@pytest.mark.parametrize("name", FIRE_FIXTURES)
def test_fire_detected_across_resolutions(name):
    """Real burning-ship frames: dial marker and viewport droplets must both
    confirm, and is_on_fire must follow the dial anchor."""
    fire = cv2.imread(str(Path("tests") / "fixtures" / name))
    assert fire is not None
    vision = Vision()
    assert vision.fire_anchor_bits(fire) == (True, True)
    assert vision.is_on_fire(fire)


def test_fire_rejects_centre_screen_orange_fragments():
    image = np.zeros((1600, 2560, 3), dtype=np.uint8)
    # Two orange target/tracer fragments near the middle of the viewport.
    image[790:810, 1260:1280] = (0, 145, 255)
    image[820:840, 1300:1320] = (0, 145, 255)
    assert not Vision().is_on_fire(image)


def test_fire_dial_marker_alone_does_not_confirm_fire():
    """A dial-only droplet is diagnostic evidence, not enough to spend R."""
    image = np.full((1600, 2560, 3), 200, dtype=np.uint8)
    cx = int(2560 * (FIRE_DIAL_ROI.left + FIRE_DIAL_ROI.right) / 2)
    cy = int(1600 * (FIRE_DIAL_ROI.top + FIRE_DIAL_ROI.bottom) / 2)
    _draw_fire_droplet(image, cx, cy)
    dial, ship = Vision().fire_anchor_bits(image)
    assert dial
    assert not ship
    assert not Vision().is_on_fire(image)


def test_fire_status_countdown_card_confirms_fire_without_dial_anchor():
    image = np.zeros((1494, 2560, 3), dtype=np.uint8)
    x1, y1, _x2, _y2 = FIRE_STATUS_ROI.pixels(2560, 1494)
    # Representative red condition card above the consumable row.
    cv2.rectangle(image, (x1 + 35, y1 + 35), (x1 + 82, y1 + 82), (0, 55, 190), -1)
    cv2.circle(image, (x1 + 50, y1 + 58), 7, (15, 15, 15), -1)
    cv2.circle(image, (x1 + 68, y1 + 58), 7, (15, 15, 15), -1)

    vision = Vision()
    assert vision.fire_anchor_bits(image) == (False, False)
    assert vision.fire_status_icon_visible(image)
    assert vision.is_on_fire(image)


def test_red_tracer_through_status_strip_does_not_confirm_fire():
    image = np.zeros((1494, 2560, 3), dtype=np.uint8)
    x1, y1, x2, y2 = FIRE_STATUS_ROI.pixels(2560, 1494)
    cv2.line(image, (x1, y2 - 5), (x2, y1 + 5), (0, 40, 220), 3)

    assert not Vision().fire_status_icon_visible(image)
    assert not Vision().is_on_fire(image)


def test_fire_dial_anchor_rejects_solid_round_marker():
    """A solid orange dot on the dial plate is a module dot, not a flame
    droplet: bounding-box fill ~pi/4 and zero value gradient fail the
    droplet gates even on a bright plate (so the ring check is not what
    rejects it)."""
    image = np.full((1600, 2560, 3), 200, dtype=np.uint8)
    cx = int(2560 * (FIRE_DIAL_ROI.left + FIRE_DIAL_ROI.right) / 2)
    cy = int(1600 * (FIRE_DIAL_ROI.top + FIRE_DIAL_ROI.bottom) / 2)
    cv2.circle(image, (cx, cy), 12, (0, 165, 255), -1)
    dial, _ship = Vision().fire_anchor_bits(image)
    assert not dial


def test_fire_dial_anchor_rejects_consumable_diamond_badge():
    """Diamond consumable-slot badges are dull orange (mean S ~125): they
    must not pass the vivid-droplet saturation gate.  The diamond geometry
    (fill 0.5) and the two-tone value gradient are deliberately within the
    droplet envelope so rejection comes from S_MIN alone."""
    image = np.full((1600, 2560, 3), 200, dtype=np.uint8)
    cx = int(2560 * (FIRE_DIAL_ROI.left + FIRE_DIAL_ROI.right) / 2)
    cy = int(1600 * (FIRE_DIAL_ROI.top + FIRE_DIAL_ROI.bottom) / 2)
    cv2.fillPoly(
        image,
        [np.array([[cx, cy - 16], [cx + 16, cy], [cx, cy + 16], [cx - 16, cy]])],
        (100, 150, 200),
    )
    cv2.fillPoly(
        image,
        [np.array([[cx, cy], [cx + 16, cy], [cx, cy + 16], [cx - 16, cy]])],
        (55, 82, 110),
    )
    dial, _ship = Vision().fire_anchor_bits(image)
    assert not dial


def test_fire_dial_anchor_rejects_text_glyph_row():
    """Port scoreboard/carousel digits can pass every single-blob gate but
    line up in a uniform-pitch row: has_text_row must reject all of them."""
    image = np.full((1600, 2560, 3), 200, dtype=np.uint8)
    y = int(1600 * FIRE_DIAL_ROI.top) + 8
    for i, cx in enumerate(range(60, 60 + 3 * 40, 40)):
        _draw_fire_droplet(image, cx, y + 16)
    dial, _ship = Vision().fire_anchor_bits(image)
    assert not dial


def test_fire_ship_anchor_ignores_minimap_and_feed_markers():
    """Minimap fire markers, capture-point icons and damage-feed crosses
    live right of the ship ROI edge (x >= 0.75); even a perfect droplet
    there must not confirm the viewport anchor."""
    image = np.zeros((1600, 2560, 3), dtype=np.uint8)
    _draw_fire_droplet(image, int(2560 * 0.79), int(1600 * 0.75))
    dial, ship = Vision().fire_anchor_bits(image)
    assert not dial
    assert not ship


def test_fire_anchor_bits_report_which_anchor_is_missing():
    """fire_anchor_bits must tell callers which HUD anchor failed, so live
    diagnostics can distinguish a missing dial marker from missing viewport
    flames."""
    fire = cv2.imread(str(Path("tests") / "fixtures" / "fire_dual_anchor.jpg"))
    assert fire is not None
    vision = Vision()
    height, width = fire.shape[:2]

    assert vision.fire_anchor_bits(fire) == (True, True)

    no_dial = fire.copy()
    no_dial[
        int(height * FIRE_DIAL_ROI.top) : int(height * FIRE_DIAL_ROI.bottom),
        int(width * FIRE_DIAL_ROI.left) : int(width * FIRE_DIAL_ROI.right),
    ] = 0
    assert vision.fire_anchor_bits(no_dial) == (False, True)
    assert not vision.is_on_fire(no_dial)

    no_ship = fire.copy()
    no_ship[
        int(height * FIRE_SHIP_ROI.top) : int(height * FIRE_SHIP_ROI.bottom),
        int(width * FIRE_SHIP_ROI.left) : int(width * FIRE_SHIP_ROI.right),
    ] = 0
    assert vision.fire_anchor_bits(no_ship) == (True, False)

    assert vision.fire_anchor_bits(np.zeros((90, 160, 3), np.uint8)) == (
        False,
        False,
    )


def test_flooding_requires_blue_icons_below_health_and_above_consumables():
    flood = np.zeros((1600, 2560, 3), dtype=np.uint8)
    # One compact blue icon below the numeric HP block.
    flood[1300:1318, 150:172] = (255, 120, 0)
    # The matching central condition icon above the consumable bar.
    flood[1340:1364, 1270:1296] = (255, 120, 0)
    assert Vision().is_flooding(flood)

    left_only = flood.copy()
    left_only[1304:1400, 1203:1382] = 0
    assert not Vision().is_flooding(left_only)

    center_only = flood.copy()
    center_only[1264:1400, 64:307] = 0
    assert not Vision().is_flooding(center_only)


def test_island_aware_waypoint_bends_only_a_blocked_minimap_route():
    island = {
        "points": [
            [0.43, 0.38],
            [0.57, 0.38],
            [0.57, 0.62],
            [0.43, 0.62],
        ]
    }
    player = (40, 160)
    target = (280, 160)
    waypoint = Vision.plan_island_aware_waypoint(
        (320, 320, 3), player, target, [island]
    )
    assert waypoint != target
    assert abs(waypoint[1] - 160) > 20

    clear_target = (40, 30)
    assert Vision.plan_island_aware_waypoint(
        (320, 320, 3), player, clear_target, [island]
    ) == clear_target
