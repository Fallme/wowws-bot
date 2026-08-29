import cv2
import numpy as np
import pytest
from pathlib import Path

from core.ocr import OcrToken, RapidOcrBackend
from core.ui import NO_COMMANDER_CONFIRM_BUTTON
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
    # The ship in this fixture is sailing broadly toward the right side.
    assert pose.heading[0] > 0.5


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


def test_live_island_signal_is_not_an_immediate_collision(live_frame):
    vision = Vision()
    minimap = vision.find_minimap(live_frame)
    pose = vision.find_player_pose_on_minimap(minimap)
    risk = vision.find_island_risk(minimap, pose)
    assert risk is None or risk.distance > 0.10


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


def test_no_commander_detector_requires_confirm_button():
    image = np.full((1000, 1600, 3), 75, dtype=np.uint8)
    image[360:660, 480:1120] = 40
    x1, y1, x2, y2 = NO_COMMANDER_CONFIRM_BUTTON.pixels(1600, 1000)
    # Teal BGR button matching the live warning's affirmative action.
    image[y1:y2, x1:x2] = (105, 100, 35)
    assert Vision().in_no_commander_confirmation(image)

    image[y1:y2, x1:x2] = 45
    assert not Vision().in_no_commander_confirmation(image)


def test_hazard_icons_are_read_from_lower_left_ship_status_only():
    image = np.zeros((1600, 2560, 3), dtype=np.uint8)
    # Old centre-screen false positive: two orange target/tracer fragments.
    image[790:810, 1260:1280] = (0, 145, 255)
    image[820:840, 1300:1320] = (0, 145, 255)
    assert not Vision().is_on_fire(image)

    # Verified fire marker position beside the numeric HP/ship silhouette.
    image[1220:1232, 160:174] = (0, 145, 255)
    image[1240:1252, 178:192] = (0, 145, 255)
    assert Vision().is_on_fire(image)

    flood = np.zeros_like(image)
    flood[1320:1332, 158:172] = (255, 120, 0)
    flood[1340:1352, 178:192] = (255, 120, 0)
    assert Vision().is_flooding(flood)


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
