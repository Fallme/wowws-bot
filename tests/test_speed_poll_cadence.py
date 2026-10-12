"""Slow combat frames must refresh speed before its freshness limit."""
from pathlib import Path
from unittest.mock import Mock, patch

import cv2

from bot import BattleBot
from test_battle_navigation import ConflictingHealthVision, FixtureDistanceReader, RecordingGamepad


def test_speed_refresh_follows_elapsed_time_and_unreadable_values_expire(tmp_path):
    image = cv2.imread(str(Path(__file__).parent / "fixtures/live_battle.png"))
    vision = ConflictingHealthVision(image)
    vision.read_speed_knots = Mock(side_effect=[22.0, 24.0, None, None])
    bot = BattleBot(1, {"strategy": {}}, vision=vision,
                    gamepad=RecordingGamepad(), distance_reader=FixtureDistanceReader())
    bot._debug_dir = tmp_path
    results = []
    for tick, now in enumerate((100, 103, 103.5, 106, 108), 1):
        bot.tick = tick
        with patch("bot.time.monotonic", return_value=now):
            results.append(bot.analyze().speed_knots)
    assert results == [22.0, 24.0, 24.0, 24.0, None]
    assert vision.read_speed_knots.call_count == 4


def test_round_reset_forces_a_new_speed_read():
    image = cv2.imread(str(Path(__file__).parent / "fixtures/live_battle.png"))
    bot = BattleBot(1, {"strategy": {}}, vision=ConflictingHealthVision(image),
                    gamepad=RecordingGamepad(), distance_reader=FixtureDistanceReader())
    bot._last_speed_ocr_attempt_at = 100
    bot.reset()
    assert bot._last_speed_ocr_attempt_at == float("-inf")
