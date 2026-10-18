"""Bounded recovery of snow/HUD line noise, retaining all geometry gates."""

from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pytest

import core.tactical_geometry as module
from core.tactical_geometry import TacticalMapGeometry, detect_tactical_map_geometry


LIVE_MAPS = Path("E:/aimemo/docs/screenshots/wows_driving_trials/run_20261018_152724_1792308444438167100")
REAL_CORNERS = ((697, 110), (1861, 110), (1910, 1312), (649, 1312))


@pytest.mark.parametrize("name,measurable", [
    ("tactical_grid_unconfirmed_1_153414.png", False),
    ("tactical_grid_unconfirmed_2_153423.png", True),
    ("tactical_grid_unconfirmed_3_153431.png", False),
])
def test_saved_snow_maps_recover_only_when_a_complete_border_is_measured(name, measurable):
    path = LIVE_MAPS / name
    if not path.exists():
        pytest.skip("Optional live snow-map evidence unavailable")
    image = cv2.imread(str(path))
    with patch.object(module.cv2, "HoughLinesP", wraps=cv2.HoughLinesP) as hough:
        geometry = detect_tactical_map_geometry(image)
    assert hough.call_count <= 2
    if not measurable:
        # The snow and resident minimap obscure opposite outer edges. More
        # detected rows do not authorize an extrapolated waypoint.
        assert geometry is None
        return
    assert geometry is not None
    assert np.asarray(geometry.corners) == pytest.approx(np.asarray(REAL_CORNERS), abs=12)
    expected = TacticalMapGeometry(image.shape[:2], REAL_CORNERS, 1200)
    for point in ((.5, .5), (.72, .83), (.25, .4)):
        assert geometry.normalized_to_pixel(point) == pytest.approx(expected.normalized_to_pixel(point), abs=12)
    assert geometry.matched_rows >= 8 and geometry.matched_columns >= 8


def test_snow_recovery_cannot_bypass_the_axis_work_budget():
    path = LIVE_MAPS / "tactical_grid_unconfirmed_2_153423.png"
    if not path.exists():
        pytest.skip("Optional live snow-map evidence unavailable")
    image = cv2.imread(str(path))
    with patch.object(module, "MAX_AXIS_EVALUATIONS", 1):
        assert detect_tactical_map_geometry(image) is None


def test_unrelated_short_rows_do_not_authorize_a_column_family():
    frame = np.full((850, 1100, 3), 65, np.uint8)
    # Rows and repeated verticals occupy disjoint parts of the viewport.
    for y in range(20, 830, 6):
        cv2.line(frame, (10, y), (490, y), (145, 145, 145), 1)
    for x in np.linspace(590, 1080, 11):
        cv2.line(frame, (round(x), 30), (round(x), 800), (145, 145, 145), 2)
    assert detect_tactical_map_geometry(frame) is None


def test_real_grid_survives_more_than_64_unrelated_row_models():
    frame = np.full((1000, 1600, 3), 65, np.uint8)
    corners = np.float32(((190, 180), (790, 180), (805, 795), (175, 795)))
    transform = cv2.getPerspectiveTransform(
        np.float32(((0, 0), (1, 0), (1, 1), (0, 1))), corners,
    )
    for fraction in np.linspace(0, 1, 11):
        for line in (((0, fraction), (1, fraction)), ((fraction, 0), (fraction, 1))):
            endpoints = cv2.perspectiveTransform(np.float32([line]), transform)[0]
            cv2.line(frame, tuple(np.rint(endpoints[0]).astype(int)),
                     tuple(np.rint(endpoints[1]).astype(int)), (145, 145, 145), 2)
    for y in range(12, 985, 6):
        cv2.line(frame, (1070, y), (1550, y), (190, 190, 190), 1)

    with patch.object(module, "_supported_row_groups", wraps=module._supported_row_groups) as support:
        geometry = detect_tactical_map_geometry(frame)

    assert support.call_count == 1
    assert len(support.call_args.args[0]) > module.MAX_AXIS_MODELS
    assert geometry is not None
    assert np.asarray(geometry.corners) == pytest.approx(corners, abs=5)
    assert geometry.matched_rows >= 10 and geometry.matched_columns >= 10
