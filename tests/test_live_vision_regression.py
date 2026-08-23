import cv2
import numpy as np
import pytest
from pathlib import Path

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


def test_no_commander_detector_requires_confirm_button():
    image = np.full((1000, 1600, 3), 75, dtype=np.uint8)
    image[360:660, 480:1120] = 40
    x1, y1, x2, y2 = NO_COMMANDER_CONFIRM_BUTTON.pixels(1600, 1000)
    # Teal BGR button matching the live warning's affirmative action.
    image[y1:y2, x1:x2] = (105, 100, 35)
    assert Vision().in_no_commander_confirmation(image)

    image[y1:y2, x1:x2] = 45
    assert not Vision().in_no_commander_confirmation(image)
