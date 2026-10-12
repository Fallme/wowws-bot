"""The navigation grid must terminate at its real tenth row, not a HUD line."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from core.tactical_geometry import detect_tactical_map_geometry


LIVE_TRIAL = Path(
    "E:/aimemo/docs/screenshots/wows_driving_trials/"
    "run_20261005_215228_1791208348042238000"
)
REAL_CORNERS = np.asarray(((655, 110), (1903, 110), (1953, 1402), (605, 1402)))


@pytest.mark.parametrize("name", [
    "autopilot_map_before_click_1_215755.png",
    "autopilot_map_before_click_2_215815.png",
])
@pytest.mark.parametrize("scale", [.5, .75, 1, 1.25])
def test_real_map_ends_above_weapon_hud_at_all_frame_scales(name, scale):
    path = LIVE_TRIAL / name
    if not path.exists():
        pytest.skip("Optional live navigation evidence unavailable")
    frame = cv2.imread(str(path))
    frame = cv2.resize(frame, None, fx=scale, fy=scale)

    geometry = detect_tactical_map_geometry(frame)

    assert geometry is not None
    assert np.asarray(geometry.corners) / scale == pytest.approx(REAL_CORNERS, abs=9)
    assert geometry.matched_rows >= 10 and geometry.matched_columns >= 10
    # A one-row shift also corrupts the waypoint inside the grid, not just its crop.
    assert np.asarray(geometry.normalized_to_pixel((.5, .5))) / scale == pytest.approx(
        (1279, 731), abs=9,
    )


def trapezoid_with_extra_hud_row(extra_y):
    frame = np.full((850, 1100, 3), 65, np.uint8)
    corners = np.float32(((180, 80), (780, 80), (795, 695), (165, 695)))
    transform = cv2.getPerspectiveTransform(
        np.float32(((0, 0), (1, 0), (1, 1), (0, 1))), corners,
    )
    for fraction in np.linspace(0, 1, 11):
        for line in (((0, fraction), (1, fraction)), ((fraction, 0), (fraction, 1))):
            endpoints = cv2.perspectiveTransform(np.float32([line]), transform)[0]
            cv2.line(frame, tuple(np.rint(endpoints[0]).astype(int)),
                     tuple(np.rint(endpoints[1]).astype(int)), (145, 145, 145), 2)
    cv2.line(frame, (160, extra_y), (800, extra_y), (145, 145, 145), 2)
    return frame, corners


@pytest.mark.parametrize("extra_y", [755, 757, 760, 765])
def test_periodic_hud_row_cannot_extend_the_grid_by_one_cell(extra_y):
    frame, corners = trapezoid_with_extra_hud_row(extra_y)

    geometry = detect_tactical_map_geometry(frame)

    assert geometry is not None
    assert np.asarray(geometry.corners) == pytest.approx(corners, abs=4)


@pytest.mark.parametrize("vertical_bottom", [640, 660])
def test_repeated_lines_without_a_measured_bottom_border_cannot_authorize_input(vertical_bottom):
    frame = np.full((850, 1100, 3), 65, np.uint8)
    for y in np.linspace(80, 695, 11):
        cv2.line(frame, (180, round(y)), (780, round(y)), (145, 145, 145), 2)
    for x in np.linspace(180, 780, 11):
        cv2.line(frame, (round(x), 80), (round(x), vertical_bottom), (145, 145, 145), 2)

    assert detect_tactical_map_geometry(frame) is None


def test_a_coast_continuing_above_the_border_does_not_move_the_top_row():
    frame = np.full((850, 1100, 3), 65, np.uint8)
    for y in np.linspace(80, 695, 11):
        cv2.line(frame, (180, round(y)), (780, round(y)), (145, 145, 145), 2)
    for x in np.linspace(180, 780, 11):
        cv2.line(frame, (round(x), 0), (round(x), 695), (145, 145, 145), 2)

    geometry = detect_tactical_map_geometry(frame)

    assert geometry is not None
    assert np.asarray(geometry.corners) == pytest.approx(
        np.asarray(((180, 80), (780, 80), (780, 695), (180, 695))), abs=3,
    )
