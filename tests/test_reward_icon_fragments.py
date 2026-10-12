"""An emblem split from one resource glyph must not become a column."""
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from core.results import ResultRewardReader


def star_contour(cx, cy, radius):
    angles = np.arange(10) * np.pi / 5 - np.pi / 2
    radii = np.where(np.arange(10) % 2, radius * .42, radius)
    return np.column_stack((np.cos(angles) * radii + cx,
                            np.sin(angles) * radii + cy)).astype(np.int32).reshape(-1, 1, 2)


@pytest.mark.parametrize("scale", [.5, 1, 1.5])
def test_fragment_overlapping_coin_does_not_split_reward_icon_pairing(scale):
    image = np.zeros((round(600 * scale), round(1000 * scale), 3), np.uint8)
    label = tuple(value * scale for value in (100, 100, 280, 130))
    # Coordinates are local to the reward strip. The interrupted coin can
    # yield a round outer contour and a separate inner emblem protruding at
    # its edge, as observed in the 2026-10-05 personal-results screenshot.
    circle = cv2.ellipse2Poly((200, 35), (18, 18), 0, 0, 360, 8).reshape(-1, 1, 2)
    contours = [circle, star_contour(210, 44, 15),
                star_contour(390, 35, 20), star_contour(530, 35, 20)]
    contours = [np.round(contour * scale).astype(np.int32) for contour in contours]
    left, top = round(70 * scale), int(131.5 * scale)
    free = contours[-1] + np.array([[[left, top]]])
    cv2.fillPoly(image, [free], (30, 240, 30))
    with patch("core.results.cv2.findContours", return_value=(contours, None)):
        regions = ResultRewardReader._reward_icon_regions(image, label)
    assert regions is not None
    assert regions["credits"].right == pytest.approx((left + 182 * scale) / image.shape[1], abs=.003)
    assert regions["ship_xp"].left > regions["credits"].right
    assert regions["free_xp"].left > regions["ship_xp"].right


def test_saved_live_result_with_split_coin_reads_exact_rewards():
    source = Path("E:/aimemo/docs/screenshots/wows_blockages/result_rewards_20261005_134549_1791179149527690500/frame_0001_observation.jpg")
    if not source.exists():
        pytest.skip("Optional retained local live-test screenshot unavailable")
    from core.ocr import RapidOcrBackend
    rewards = ResultRewardReader(RapidOcrBackend()).read(cv2.imread(str(source)))
    assert rewards.recognized
    assert rewards.resource_values() == {"credits": 41010, "ship_xp": 1174, "free_xp": 78}
    assert rewards.outcome == "victory"


@pytest.mark.parametrize("scale", [1, .5, .75])
def test_lossless_live_coin_with_touching_emblem_keeps_all_three_rewards(scale):
    source = Path("E:/aimemo/docs/screenshots/wows_driving_trials/run_20261005_142256_1791181376552983600/result_unrecognized_143015.png")
    if not source.exists():
        pytest.skip("Optional retained lossless live-test screenshot unavailable")
    from core.ocr import RapidOcrBackend
    image = cv2.imread(str(source))
    if scale != 1:
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    rewards = ResultRewardReader(RapidOcrBackend()).read(image)
    assert rewards.recognized
    assert rewards.resource_values() == {"credits": 47691, "ship_xp": 1471, "free_xp": 93}
    assert rewards.outcome == "victory"
