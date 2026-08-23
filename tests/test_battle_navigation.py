import time
from pathlib import Path

import cv2

from bot import BattleAnalysis, BattleBot
from core.ocr import DistanceObservation, Rect
from core.vision import Vision


class FixtureVision(Vision):
    def __init__(self, image):
        super().__init__()
        self.image = image

    def grab(self, hwnd):
        return self.image.copy()


class RecordingGamepad:
    def __init__(self):
        self.movements = []

    def set_movement(self, throttle, rudder):
        self.movements.append((throttle, rudder))

    def stop(self):
        self.movements.append((0.0, 0.0))


class FixtureDistanceReader:
    def read(self, image, anchor, target_track_id, *, captured_at):
        return DistanceObservation(
            value_km=23.7,
            raw_text="23.7 公里",
            confidence=0.95,
            roi=Rect(925, 620, 256, 81),
            target_track_id=target_track_id,
            captured_at=captured_at,
            accepted=True,
        )


def test_live_analysis_produces_heading_distance_and_island_clearance():
    image = cv2.imread(str(Path("tests") / "fixtures" / "live_battle.png"))
    gamepad = RecordingGamepad()
    bot = BattleBot(
        1,
        {"strategy": {}},
        vision=FixtureVision(image),
        gamepad=gamepad,
        distance_reader=FixtureDistanceReader(),
    )
    bot.tick = 1

    bot.analyze()  # First frame seeds the consecutive-target filter.
    bot.analyze()  # Second frame creates the target track and first OCR sample.
    analysis = bot.analyze()

    assert analysis.player_position is not None
    assert analysis.minimap_distance is not None
    assert analysis.minimap_distance > 0.18
    assert analysis.minimap_distance_km is not None
    assert analysis.minimap_distance_km > 10
    assert analysis.minimap_target_bearing is not None
    assert analysis.capture_point_distance_km is not None
    assert analysis.map_center_distance_km is not None
    assert analysis.island_distance is None or analysis.island_distance > 0.10
    assert analysis.target_distance_km == 23.7


def test_emergency_island_command_never_reverses_from_vision_alone():
    gamepad = RecordingGamepad()
    bot = BattleBot(
        1,
        {"strategy": {}},
        vision=object(),
        gamepad=gamepad,
    )
    now = time.monotonic()
    bot.battle_start_time = now - 120
    analysis = BattleAnalysis(
        image=None,
        width=2560,
        height=1600,
        minimap_distance=0.25,
        minimap_enemy_count=1,
        player_position=(160, 190),
        island_distance=0.015,
        island_avoidance_rudder=-1,
    )

    bot._execute_rules(analysis, now)

    throttle, rudder = gamepad.movements[-1]
    assert throttle > 0
    assert rudder < -0.7


def test_island_manoeuvre_requires_four_consistent_observations():
    bot = BattleBot(
        1,
        {"strategy": {}},
        vision=object(),
        gamepad=RecordingGamepad(),
    )

    for _ in range(3):
        bot._island_samples.append((0.04, 1.0))
        assert bot._stable_island_risk() is None
    bot._island_samples.append((0.04, 1.0))
    distance, side = bot._stable_island_risk()
    assert distance == 0.04
    assert side == 1.0


def test_island_manoeuvre_rejects_ambiguous_turn_side():
    bot = BattleBot(
        1,
        {"strategy": {}},
        vision=object(),
        gamepad=RecordingGamepad(),
    )
    for side in (0.0, 0.0, -1.0, 1.0):
        bot._island_samples.append((0.015, side))

    assert bot._stable_island_risk() is None
