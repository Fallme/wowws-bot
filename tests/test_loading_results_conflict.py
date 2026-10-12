"""A live pre-battle start action cannot complete a round as RESULTS."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import cv2
import numpy as np
import pytest
import main
from core.ocr import OcrToken
from core.ui import ScreenState
from core.vision import Vision
from test_health_clipped_anchor_recovery import LimitedCpuBackend


def test_exact_start_action_overrides_result_artwork_but_not_other_action_words():
    frame = np.zeros((600, 1000, 3), np.uint8)
    vision = SimpleNamespace(classify_screen=lambda _: ScreenState.RESULTS,
                             _has_loading_start_action=lambda _: True)
    for text, expected in [("开始战斗", ScreenState.LOADING),
                           ("继续战斗", ScreenState.RESULTS),
                           ("返回港口", ScreenState.RESULTS),
                           ("", ScreenState.RESULTS)]:
        backend = SimpleNamespace(recognize=Mock(return_value=[OcrToken(text, .99)]))
        bot = SimpleNamespace(vision=vision, distance_reader=SimpleNamespace(backend=backend))
        assert main.classify_runtime_screen(bot, frame) == expected


def test_missing_action_shape_cannot_reclassify_result_from_ocr_words():
    bot = SimpleNamespace(
        vision=SimpleNamespace(classify_screen=lambda _: ScreenState.RESULTS,
                               _has_loading_start_action=lambda _: False),
        distance_reader=SimpleNamespace(backend=SimpleNamespace(
            recognize=Mock(return_value=[OcrToken("开始战斗", .99)]))))
    assert main.classify_runtime_screen(bot, np.zeros((600, 1000, 3), np.uint8)) == ScreenState.RESULTS
    bot.distance_reader.backend.recognize.assert_not_called()


def test_live_loading_tip_conflict_keeps_round_unfinished():
    path = Path("E:/aimemo/docs/screenshots/wows_blockages/result_rewards_20261006_111334_1791256414327032100/frame_0001_observation.png")
    if not path.exists():
        pytest.skip("external live diagnostic is unavailable")
    frame = cv2.imread(str(path))
    vision = Vision()
    assert vision.classify_screen(frame) == ScreenState.RESULTS
    backend = SimpleNamespace(recognize=Mock(return_value=[OcrToken("开始战斗", .99)]))
    bot = SimpleNamespace(vision=vision, distance_reader=SimpleNamespace(backend=backend))
    assert main.classify_runtime_screen(bot, frame) == ScreenState.LOADING
    assert main.classify_battle_continuity_screen(bot, frame) == ScreenState.LOADING


def test_live_start_action_is_confirmed_by_actual_bounded_cpu_ocr():
    path = Path("E:/aimemo/docs/screenshots/wows_blockages/result_rewards_20261006_111334_1791256414327032100/frame_0001_observation.png")
    if not path.exists():
        pytest.skip("external live diagnostic is unavailable")
    image = cv2.imread(str(path))
    bot = SimpleNamespace(vision=Vision(), distance_reader=SimpleNamespace(backend=LimitedCpuBackend()))
    previous_threads = cv2.getNumThreads()
    cv2.setNumThreads(1)
    try:
        assert main.loading_start_confirmed(bot, image)
        assert main.classify_runtime_screen(bot, image) == ScreenState.LOADING
    finally:
        cv2.setNumThreads(previous_threads)
