from collections import deque

import cv2
import numpy as np

from core.terrain import (
    confirm_terrain,
    exclude_player_land,
    extend_template,
    player_land_depth,
    rasterize,
    repair_overlay_gaps,
)
from core.vision import PlayerPose, Vision
from strategy.island_route import IslandRoutePlanner


def rectangle(x1, y1, x2, y2):
    return {"points": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
            "area": (x2 - x1) * (y2 - y1)}


def test_pixel_votes_reject_moving_shapes_even_with_matching_centroids():
    samples = deque(maxlen=5)
    # Old centroid/area matching promoted the whole last rectangle.
    for x in (0.20, 0.23, 0.26):
        confirmed = confirm_terrain(samples, [rectangle(x, .2, x + .05, .25)])
    assert confirmed is None


def test_template_recovers_occluded_coast_without_erasing_confirmed_land():
    old = [rectangle(.2, .2, .3, .4)]
    complete = [rectangle(.2, .2, .4, .4)]
    samples = deque(maxlen=5)
    for frame in (complete, old, complete, [], complete):
        confirmed = confirm_terrain(samples, frame)
    template = extend_template(old, confirmed)
    assert rasterize(template)[150, 190] == 1
    assert np.all(rasterize(template) >= rasterize(old))
    assert extend_template(template, []) == template


def test_incompatible_map_cannot_contaminate_template():
    old = [rectangle(.1, .1, .3, .3)]
    assert extend_template(old, [rectangle(.6, .6, .8, .8)]) == old


def test_small_coincidental_overlap_does_not_merge_another_map():
    old = [rectangle(.1, .1, .3, .3)]
    candidate = [rectangle(.29, .29, .49, .49), rectangle(.65, .65, .8, .8)]
    assert extend_template(old, candidate) == old


def test_arrow_on_actual_land_cannot_carve_escape_route():
    planner = IslandRoutePlanner()
    assert planner.waypoint((400, 400), (200, 200), (350, 200),
                            [rectangle(.49, .49, .515, .515)]) is None
    assert not planner.path


def test_route_smoothing_cannot_cut_diagonal_land_corners():
    mask = np.zeros((4, 4), np.uint8)
    mask[0, 1] = 1
    assert not IslandRoutePlanner._clear(mask, (0, 0), (3, 3))
    assert not IslandRoutePlanner._clear(mask, (3, 3), (0, 0))
    assert IslandRoutePlanner._clear(mask, (0, 1), (3, 1))


def test_overlay_repair_preserves_visible_water_channel():
    land = np.zeros((80, 80), np.uint8)
    land[15:65, 15:65] = 255
    land[:, 38:43] = 0
    unknown = np.zeros_like(land, dtype=bool)
    unknown[20:35, 38:43] = True
    repaired = repair_overlay_gaps(land, unknown, 5)
    assert repaired[27, 40] == 255
    assert repaired[50, 40] == 0
    assert np.all(repaired[~unknown] == land[~unknown])


def test_thin_ring_alone_does_not_become_an_island():
    frame = np.full((420, 420, 3), (72, 48, 30), np.uint8)
    cv2.circle(frame, (210, 210), 75, (230, 230, 230), 2)
    assert Vision().find_minimap_island_outlines(frame) == []


def test_confirmed_shore_next_to_ship_is_not_erased_as_player_icon():
    frame = np.zeros((400, 400, 3), np.uint8)
    risk = Vision().find_island_risk(
        frame, PlayerPose((200, 200), (1., 0.)),
        island_outlines=[rectangle(.51, .48, .525, .52)],
    )
    assert risk is not None
    assert risk.distance < .02


def test_island_route_planner_never_returns_a_waypoint_inside_land():
    island = rectangle(.38, .25, .62, .75)
    planner = IslandRoutePlanner()
    waypoint = planner.waypoint(
        (320, 320, 3), (30, 160), (290, 160), [island]
    )
    assert waypoint is not None
    assert not (.38 * 320 <= waypoint[0] <= .62 * 320
                and .25 * 320 <= waypoint[1] <= .75 * 320)
    assert planner.waypoint(
        (320, 320, 3), (30, 30), (290, 290), [island]
    ) is not None


def test_island_route_planner_snaps_a_coarse_objective_to_nearby_water():
    # The central objective is allowed to fall on an island in the noisy
    # fallback layer.  The planner must choose the adjacent shoreline water,
    # not report a permanently blocked route.
    island = rectangle(.38, .25, .62, .75)
    planner = IslandRoutePlanner()
    waypoint = planner.waypoint(
        (320, 320, 3), (160, 300), (160, 160), [island]
    )

    assert waypoint is not None
    assert not (.38 * 320 <= waypoint[0] <= .62 * 320
                and .25 * 320 <= waypoint[1] <= .75 * 320)


def test_land_outline_enclosing_the_live_player_is_rejected():
    false_island = rectangle(.3, .5, .7, .9)

    assert player_land_depth(false_island, (.5, .77)) > .02
    assert exclude_player_land([false_island], (.5, .77)) == []
    # A marker beside the coast, or a ship near the shoreline, stays valid.
    assert exclude_player_land([false_island], (.9, .2)) == [false_island]
    assert exclude_player_land([false_island], (.305, .77)) == [false_island]


def test_route_planner_recovers_from_a_false_island_around_the_spawn():
    false_island = rectangle(.3, .5, .7, .9)
    planner = IslandRoutePlanner()

    waypoint = planner.waypoint(
        (400, 400), (200, 308), (100, 280), [false_island],
        clearance_ratio=.01,
    )

    assert waypoint is not None


def test_dark_minimap_grey_islands_survive_the_adaptive_value_floor():
    image = np.zeros((400, 400, 3), np.uint8)
    image[:, :] = (45, 10, 5)  # dark blue sea; V stays far below the old 125
    cv2.rectangle(image, (120, 120), (220, 190), (100, 100, 100), -1)
    cv2.rectangle(image, (260, 240), (300, 300), (85, 85, 85), -1)

    outlines = Vision().find_minimap_island_outlines(image)

    assert len(outlines) >= 2
