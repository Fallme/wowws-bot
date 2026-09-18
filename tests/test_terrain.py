from collections import deque

import cv2
import numpy as np

from core.terrain import confirm_terrain, extend_template, rasterize, repair_overlay_gaps
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
