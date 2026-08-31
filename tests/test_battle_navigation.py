import math
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from bot import BattleAnalysis, BattleBot, select_forward_navigation_enemy
from core.ocr import DistanceObservation, Rect
from core.vision import CaptureZone, PlayerPose, Vision
from core.feedback import SafetyFault
from core.ui import ScreenState


class FixtureVision(Vision):
    def __init__(self, image):
        super().__init__()
        self.image = image

    def grab(self, hwnd):
        return self.image.copy()


class ConflictingHealthVision(FixtureVision):
    @staticmethod
    def read_health_fraction(_image, _backend):
        return 1.0

    @staticmethod
    def health_pct(_area):
        return 0.888


class MissingHealthOcrVision(FixtureVision):
    @staticmethod
    def read_health_fraction(_image, _backend):
        return None

    @staticmethod
    def health_pct(_area):
        raise AssertionError("生命值不得回退到血条估算")


class HazardVision(ConflictingHealthVision):
    @staticmethod
    def is_on_fire(_image):
        return True

    @staticmethod
    def is_flooding(_image):
        return True


class ZeroHealthHazardVision(HazardVision):
    @staticmethod
    def read_health_fraction(_image, _backend):
        return 0.0


class RecordingGamepad:
    def __init__(self):
        self.movements = []

    def set_movement(self, throttle, rudder):
        self.movements.append((throttle, rudder))

    def stop(self):
        self.movements.append((0.0, 0.0))


class DamageControlGamepad(RecordingGamepad):
    def __init__(self):
        super().__init__()
        self.damage_control_uses = 0

    def damage_control(self):
        self.damage_control_uses += 1


class SurvivalGamepad(DamageControlGamepad):
    def __init__(self):
        super().__init__()
        self.heal_uses = 0
        self.consumable_cycle_uses = 0
        self.other_consumable_uses = 0

    def heal(self):
        self.heal_uses += 1

    def use_consumable_cycle(self):
        self.consumable_cycle_uses += 1

    def use_other_consumables(self):
        self.other_consumable_uses += 1


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


class SequenceScreenVision(Vision):
    def __init__(self, states):
        super().__init__()
        self.states = list(states)
        self.image = cv2.imread(str(Path("tests") / "fixtures" / "live_battle.png"))

    def grab(self, _hwnd):
        return self.image.copy()

    def classify_screen(self, _image):
        if len(self.states) > 1:
            return self.states.pop(0)
        return self.states[0]


def test_navigation_enemy_prefers_forward_contact_over_nearer_rear_contact():
    pose = SimpleNamespace(position=(100.0, 100.0), heading=(0.0, -1.0))
    rear_near = (100.0, 106.0)
    forward_farther = (102.0, 78.0)

    selected = select_forward_navigation_enemy(
        Vision(), pose, [rear_near, forward_farther]
    )

    assert selected is not None
    enemy, distance, bearing = selected
    assert enemy == forward_farther
    assert distance == pytest.approx(math.hypot(2.0, 22.0))
    assert abs(bearing) < 0.55


def test_navigation_enemy_rejects_all_contacts_behind_ship():
    pose = SimpleNamespace(position=(100.0, 100.0), heading=(0.0, -1.0))

    assert select_forward_navigation_enemy(
        Vision(), pose, [(100.0, 106.0), (90.0, 110.0)]
    ) is None


def test_battle_end_requires_two_consecutive_result_frames_not_port_guess():
    vision = SequenceScreenVision(
        [ScreenState.UNKNOWN, ScreenState.RESULTS, ScreenState.UNKNOWN,
         ScreenState.PORT, ScreenState.PORT]
    )
    bot = BattleBot(
        1,
        {"strategy": {}},
        vision=vision,
        gamepad=RecordingGamepad(),
        distance_reader=FixtureDistanceReader(),
    )

    assert not bot.analyze().ended
    assert not bot.analyze().ended
    assert not bot.analyze().ended
    assert not bot.analyze().ended
    assert not bot.analyze().ended


def test_battle_end_accepts_two_consecutive_result_frames():
    vision = SequenceScreenVision([ScreenState.RESULTS, ScreenState.RESULTS])
    bot = BattleBot(
        1,
        {"strategy": {}},
        vision=vision,
        gamepad=RecordingGamepad(),
        distance_reader=FixtureDistanceReader(),
    )

    assert not bot.analyze().ended
    assert bot.analyze().ended


def test_reset_clears_pending_battle_end_confirmation():
    vision = SequenceScreenVision([ScreenState.RESULTS, ScreenState.RESULTS])
    bot = BattleBot(
        1,
        {"strategy": {}},
        vision=vision,
        gamepad=RecordingGamepad(),
        distance_reader=FixtureDistanceReader(),
    )

    assert not bot.analyze().ended
    bot.reset()
    assert not bot.analyze().ended


def test_reset_preserving_engine_still_clears_previous_battle_navigation_state():
    gamepad = RecordingGamepad()
    bot = BattleBot(
        1,
        {"strategy": {}},
        vision=object(),
        gamepad=gamepad,
        distance_reader=FixtureDistanceReader(),
    )
    bot._battle_map_islands = [(10, 20)]
    bot._battle_capture_zones = [("A", 30, 40)]
    bot._island_layer_candidates.append(((10, 20),))
    bot._zone_layer_candidates.append((("A", 30, 40),))
    bot.opening_autopilot_active = True
    bot.opening_autopilot_target = "上一局航点"
    bot.opening_autopilot_target_normalized = (0.8, 0.2)
    bot._tactical_map_left_open = True
    bot.generic_center_route_active = True

    bot.reset(preserve_movement=True)

    assert gamepad.movements == []
    assert bot._battle_map_islands == []
    assert bot._battle_capture_zones == []
    assert list(bot._island_layer_candidates) == []
    assert list(bot._zone_layer_candidates) == []
    assert not bot.opening_autopilot_active
    assert bot.opening_autopilot_target == ""
    assert bot.opening_autopilot_target_normalized is None
    assert not bot._tactical_map_left_open
    assert not bot.generic_center_route_active


def test_reset_preserves_normalized_static_map_for_same_battle_resume():
    bot = BattleBot(
        1,
        {"strategy": {}},
        vision=object(),
        gamepad=RecordingGamepad(),
        distance_reader=FixtureDistanceReader(),
    )
    islands = [
        {
            "points": [(0.1, 0.2), (0.2, 0.2), (0.2, 0.3)],
            "area": 0.005,
        }
    ]
    layout = [
        {
            "label": "A",
            "state": "neutral",
            "position": (0.25, 0.5),
            "radius": 0.1,
        }
    ]
    bot._battle_map_islands = islands
    bot._battle_capture_zone_layout = layout
    bot._battle_capture_zones = [
        CaptureZone(center=(50, 100), radius=20, label="A", state="neutral")
    ]
    bot._static_map_source = "tactical_map"

    bot.reset(preserve_movement=True, preserve_static_map=True)

    assert bot._battle_map_islands == islands
    assert bot._battle_capture_zone_layout == layout
    assert bot._battle_capture_zones == []
    assert bot._static_map_source == "tactical_map"
    rematerialized = bot._capture_zones_for_shape((300, 400, 3))
    assert rematerialized == [
        CaptureZone(center=(100, 150), radius=30, label="A", state="neutral")
    ]


def test_tactical_map_static_layers_lock_after_three_frames_and_stop_updating():
    class TacticalStaticVision:
        def __init__(self):
            self.island_calls = 0
            self.zone_calls = 0

        @staticmethod
        def find_tactical_map(image):
            return image

        @staticmethod
        def find_player_pose_on_minimap(_image):
            return PlayerPose(position=(200, 300), heading=(0.0, -1.0))

        def find_minimap_island_outlines(self, _image):
            self.island_calls += 1
            return [
                {
                    "points": [
                        (0.10, 0.20),
                        (0.25, 0.20),
                        (0.25, 0.35),
                        (0.10, 0.35),
                    ],
                    "area": 0.0225,
                }
            ]

        def find_capture_zones(self, _image, *, player=None):
            assert player == (200, 300)
            self.zone_calls += 1
            return [
                CaptureZone(
                    center=(100, 200),
                    radius=40,
                    label="A",
                    state="neutral",
                )
            ]

    vision = TacticalStaticVision()
    bot = BattleBot(
        1,
        {"strategy": {}},
        vision=vision,
        gamepad=RecordingGamepad(),
        distance_reader=FixtureDistanceReader(),
    )
    tactical_frame = np.full((400, 400, 3), 100, dtype=np.uint8)

    assert not bot.capture_tactical_map_static_layer(tactical_frame, begin=True)
    assert not bot.capture_tactical_map_static_layer(tactical_frame)
    assert bot.capture_tactical_map_static_layer(tactical_frame)
    assert bot._static_map_source == "tactical_map"
    assert len(bot._battle_map_islands) == 1
    assert bot._capture_zones_for_shape((200, 300, 3)) == [
        CaptureZone(center=(75, 100), radius=20, label="A", state="neutral")
    ]

    # Once both layers are locked, later calls cannot re-run or mutate static
    # recognition. Player/heading/enemy detection remains in analyze().
    assert bot.capture_tactical_map_static_layer(tactical_frame)
    assert vision.island_calls == 3
    assert vision.zone_calls == 3


def test_locked_static_map_keeps_player_and_enemy_layers_live_each_frame():
    image = cv2.imread(str(Path("tests") / "fixtures" / "live_battle.png"))

    class DynamicLayerVision(FixtureVision):
        def __init__(self, frame):
            super().__init__(frame)
            self.dynamic_calls = 0
            self.pose_calls = 0

        def analyze_minimap(self, _minimap):
            self.dynamic_calls += 1
            return [(100 + self.dynamic_calls * 4, 80)], False

        def find_player_pose_on_minimap(self, _minimap):
            self.pose_calls += 1
            return PlayerPose(
                position=(120 + self.pose_calls * 3, 160),
                heading=(1.0, 0.0),
            )

        @staticmethod
        def find_minimap_island_outlines(_minimap):
            raise AssertionError("静态山体锁定后不得重新识别")

        @staticmethod
        def find_capture_zones(_minimap, player=None):
            raise AssertionError("静态点位锁定后不得重新识别")

    vision = DynamicLayerVision(image)
    bot = BattleBot(
        1,
        {"strategy": {}},
        vision=vision,
        gamepad=RecordingGamepad(),
        distance_reader=FixtureDistanceReader(),
    )
    bot._battle_map_islands = [
        {
            "points": [(0.70, 0.70), (0.75, 0.70), (0.75, 0.75)],
            "area": 0.00125,
        }
    ]
    bot._battle_capture_zone_layout = [
        {
            "label": "A",
            "state": "neutral",
            "position": (0.5, 0.5),
            "radius": 0.08,
        }
    ]
    bot._static_map_source = "tactical_map"

    first = bot.analyze()
    second = bot.analyze()

    assert vision.dynamic_calls == 2
    assert vision.pose_calls == 2
    assert first.player_position == (123, 160)
    assert second.player_position == (126, 160)
    assert first.minimap_enemy_count == 0
    assert second.minimap_enemy_count == 1
    assert second.capture_zones[0]["position"] == pytest.approx(
        [0.5, 0.5],
        abs=0.002,
    )


def test_live_analysis_produces_heading_distance_and_island_risk():
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
    # The corrected bow vector points toward the lower-left island in this
    # captured frame, so the terrain detector must report an immediate risk.
    assert analysis.island_distance is not None
    assert analysis.island_distance < 0.10
    assert analysis.target_distance_km == 23.7
    assert len(analysis.capture_zones) == 3
    assert analysis.capture_zones[0]["state"] == "hostile"
    assert analysis.capture_zones[1]["state"] == "friendly"
    assert analysis.capture_zones[2]["state"] == "neutral"
    assert analysis.minimap_contacts


def test_exact_health_ocr_is_not_overwritten_by_padded_bar_fallback():
    image = cv2.imread(str(Path("tests") / "fixtures" / "live_battle.png"))
    bot = BattleBot(
        1,
        {"strategy": {}},
        vision=ConflictingHealthVision(image),
        gamepad=RecordingGamepad(),
        distance_reader=FixtureDistanceReader(),
    )

    for tick in range(3):
        bot.tick = tick
        analysis = bot.analyze()

    assert analysis.health == 1.0
    assert analysis.health_recognized


def test_missing_health_digits_never_calls_bar_estimator():
    image = cv2.imread(str(Path("tests") / "fixtures" / "live_battle.png"))
    bot = BattleBot(
        1,
        {"strategy": {}},
        vision=MissingHealthOcrVision(image),
        gamepad=RecordingGamepad(),
        distance_reader=FixtureDistanceReader(),
    )

    analysis = bot.analyze()

    assert not analysis.health_recognized


def test_fire_and_flooding_require_two_consecutive_frames():
    image = cv2.imread(str(Path("tests") / "fixtures" / "live_battle.png"))
    bot = BattleBot(
        1,
        {"strategy": {}},
        vision=HazardVision(image),
        gamepad=RecordingGamepad(),
        distance_reader=FixtureDistanceReader(),
    )

    first = bot.analyze()
    bot.tick = 1
    second = bot.analyze()

    assert not first.on_fire
    assert not first.flooding
    assert second.on_fire
    assert second.flooding


def test_death_suppresses_stale_fire_and_flooding_icons():
    image = cv2.imread(str(Path("tests") / "fixtures" / "live_battle.png"))
    bot = BattleBot(
        1,
        {"strategy": {}},
        vision=ZeroHealthHazardVision(image),
        gamepad=RecordingGamepad(),
        distance_reader=FixtureDistanceReader(),
    )

    bot.analyze()
    bot.tick = 1
    analysis = bot.analyze()

    assert analysis.health == 0.0
    assert not analysis.on_fire
    assert not analysis.flooding


def test_confirmed_hazard_uses_damage_control_without_waiting_for_hp_drop():
    gamepad = DamageControlGamepad()
    bot = BattleBot(
        1,
        {"strategy": {"damage_control_cooldown_seconds": 80}},
        vision=object(),
        gamepad=gamepad,
    )
    bot.intervention = SimpleNamespace(poll=lambda *_args: False)
    now = time.monotonic()
    bot.battle_start_time = now - 100
    bot.last_damage_control = now - 100
    analysis = BattleAnalysis(
        image=None,
        width=2560,
        height=1600,
        health=1.0,
        health_recognized=True,
        on_fire=True,
        player_position=(100, 100),
        map_center_bearing=0.0,
    )

    bot._execute_rules(analysis, now)
    bot._execute_rules(analysis, now + 1)

    assert gamepad.damage_control_uses == 1


def test_survival_consumables_still_run_while_native_autopilot_owns_steering():
    gamepad = SurvivalGamepad()
    bot = BattleBot(
        1,
        {
            "strategy": {
                "damage_control_cooldown_seconds": 80,
                "other_consumable_cooldown_seconds": 30,
            }
        },
        vision=object(),
        gamepad=gamepad,
    )
    bot.intervention = SimpleNamespace(poll=lambda *_args: False)
    now = time.monotonic()
    bot.battle_start_time = now - 100
    bot.last_damage_control = now - 100
    bot.last_heal = now - 100
    bot.opening_autopilot_active = True
    analysis = BattleAnalysis(
        image=None,
        width=2560,
        height=1600,
        health=0.79,
        health_recognized=True,
        on_fire=True,
        autopilot_enabled=True,
        player_position=(100, 100),
    )

    bot._execute_rules(analysis, now)

    assert gamepad.damage_control_uses == 1
    assert gamepad.other_consumable_uses == 1
    assert gamepad.consumable_cycle_uses == 0
    assert gamepad.movements == []


def test_other_consumables_retry_every_thirty_seconds_while_health_is_missing():
    gamepad = SurvivalGamepad()
    bot = BattleBot(
        1,
        {
            "strategy": {
                "other_consumable_cooldown_seconds": 30,
            }
        },
        vision=object(),
        gamepad=gamepad,
    )
    bot.intervention = SimpleNamespace(poll=lambda *_args: False)
    now = time.monotonic()
    bot.last_heal = now - 100

    def analysis_at(health):
        return BattleAnalysis(
            image=None,
            width=2560,
            height=1600,
            health=health,
            health_recognized=True,
        )

    assert bot._execute_survival_consumables(analysis_at(1.0), now)
    assert gamepad.other_consumable_uses == 0
    assert bot._execute_survival_consumables(analysis_at(0.99), now + 1)
    assert gamepad.other_consumable_uses == 1
    assert bot._execute_survival_consumables(analysis_at(0.70), now + 30)
    assert gamepad.other_consumable_uses == 1
    assert bot._execute_survival_consumables(analysis_at(0.70), now + 31)
    assert gamepad.other_consumable_uses == 2
    assert bot._execute_survival_consumables(analysis_at(0.39), now + 61)
    assert gamepad.other_consumable_uses == 3


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


def test_island_manoeuvre_accepts_three_consistent_recent_observations():
    bot = BattleBot(
        1,
        {"strategy": {}},
        vision=object(),
        gamepad=RecordingGamepad(),
    )

    for _ in range(2):
        bot._island_samples.append((0.04, 1.0))
        assert bot._stable_island_risk() is None
    bot._island_samples.append((0.04, 1.0))
    distance, side = bot._stable_island_risk()
    assert distance == 0.04
    assert side == 1.0


def test_normal_rudder_direction_waits_for_ship_response_before_reversing():
    bot = BattleBot(
        1,
        {"strategy": {"rudder_minimum_hold_seconds": 4.0}},
        vision=object(),
        gamepad=RecordingGamepad(),
    )

    assert bot._latency_compensated_rudder(0.4, 10.0) == 0.4
    assert bot._latency_compensated_rudder(-0.4, 12.0) == 0.4
    assert bot._latency_compensated_rudder(-0.4, 14.1) == 0.0
    assert bot._latency_compensated_rudder(-0.4, 14.8) == -0.4


def test_island_override_releases_to_neutral_before_opposite_rudder():
    bot = BattleBot(
        1,
        {"strategy": {"rudder_minimum_hold_seconds": 4.0}},
        vision=object(),
        gamepad=RecordingGamepad(),
    )

    assert bot._latency_compensated_rudder(0.4, 10.0) == 0.4
    assert bot._latency_compensated_rudder(
        -0.8, 10.5, safety_override=True
    ) == 0.0
    assert bot._latency_compensated_rudder(
        -0.8, 11.2, safety_override=True
    ) == -0.8


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


def test_low_speed_collision_escalates_island_avoidance_to_full_power_escape():
    gamepad = RecordingGamepad()
    bot = BattleBot(
        1,
        {
            "strategy": {
                "straight_opening_seconds": 2,
                "stuck_seconds": 30,
                "stuck_stationary_pixels": 2,
                "stuck_low_speed_seconds": 8,
                "stuck_low_speed_knots": 1.5,
                "stuck_escape_turn_seconds": 8,
            }
        },
        vision=object(),
        gamepad=gamepad,
        distance_reader=FixtureDistanceReader(),
    )
    bot.intervention = SimpleNamespace(poll=lambda *_args: False)
    bot.battle_start_time = 0.0

    for second in range(9):
        analysis = BattleAnalysis(
            image=None,
            width=2560,
            height=1494,
            in_battle=True,
            player_position=(100 + second, 200),
            speed_knots=0.6,
            map_center_bearing=-0.2,
            map_center_distance_km=12.0,
            capture_point_bearing=-0.2,
            capture_point_distance_km=12.0,
            island_distance=0.03,
            island_avoidance_rudder=-1.0,
        )
        bot._execute_rules(analysis, 100.0 + second)

    assert gamepad.movements[-1] == (1.0, -1)
    assert bot.last_movement_reason == "舰船位置长时间未变化，执行自动脱困"
    assert bot._last_movement_mode == "recovery:forward_escape_turn"


def test_movement_feedback_retries_before_requesting_human_intervention():
    class RetryGamepad(RecordingGamepad):
        def __init__(self):
            super().__init__()
            self.reassertions = 0

        def reassert_full_speed(self):
            self.reassertions += 1

    class AlwaysFailingFeedback:
        @staticmethod
        def update(_now, _position, _throttle):
            raise SafetyFault("没有位移")

        @staticmethod
        def reset():
            pass

    gamepad = RetryGamepad()
    bot = BattleBot(
        1,
        {"strategy": {"movement_feedback_retries": 3}},
        vision=object(),
        gamepad=gamepad,
    )
    bot.feedback = AlwaysFailingFeedback()

    for attempt in range(3):
        assert bot._movement_feedback_update(attempt, (100, 100), 1.0) is None

    assert bot.generic_center_route_active
    assert gamepad.reassertions == 3
    with pytest.raises(SafetyFault, match="连续反馈失败"):
        bot._movement_feedback_update(4, (100, 100), 1.0)


def test_missing_minimap_pose_keeps_safe_course_without_ending_battle():
    class MissingPoseFeedback:
        @staticmethod
        def update(_now, _position, _throttle):
            raise SafetyFault("无法识别玩家小地图位置，不能验证控制反馈")

        @staticmethod
        def reset():
            pass

    class SafeCourseGamepad(RecordingGamepad):
        def __init__(self):
            super().__init__()
            self.reassertions = 0

        def reassert_full_speed(self):
            self.reassertions += 1

    gamepad = SafeCourseGamepad()
    bot = BattleBot(1, {"strategy": {}}, vision=object(), gamepad=gamepad)
    bot.feedback = MissingPoseFeedback()

    for tick in range(5):
        assert bot._movement_feedback_update(tick, None, 1.0) is None

    assert bot.generic_center_route_active
    assert bot.movement_feedback_failures == 0
    assert gamepad.reassertions == 5


def test_native_autopilot_crawling_speed_requests_route_retry_after_six_seconds():
    bot = BattleBot(
        1,
        {"strategy": {"autopilot_zero_speed_retry_seconds": 6.0}},
        vision=object(),
        gamepad=RecordingGamepad(),
    )
    bot.intervention = SimpleNamespace(poll=lambda *_args: False)
    bot.enable_opening_autopilot("地图中心敌方远端")
    analysis = BattleAnalysis(
        image=None,
        width=2560,
        height=1494,
        in_battle=True,
        autopilot_enabled=True,
        speed_knots=0.6,
        player_position=(100, 100),
    )

    bot._execute_rules(analysis, 100.0)
    assert not bot.autopilot_retry_pending

    bot._execute_rules(analysis, 106.1)

    assert bot.autopilot_retry_pending
    assert not bot.opening_autopilot_active
    assert bot.native_autopilot_abandoned
    assert "持续低速" in bot.last_movement_reason

    # The game's green label can linger after native control is abandoned.
    # It must not re-lock Q/E on the next frame.
    bot._execute_rules(analysis, 106.5)
    assert not bot.opening_autopilot_active
    assert bot.autopilot_retry_pending


def test_native_autopilot_stalled_position_requests_route_retry_immediately():
    class StalledFeedback:
        @staticmethod
        def update(_now, _position, _throttle):
            raise SafetyFault("已发送航行指令，但未观察到舰船位置变化")

        @staticmethod
        def reset():
            pass

    bot = BattleBot(
        1,
        {"strategy": {}},
        vision=object(),
        gamepad=RecordingGamepad(),
    )
    bot.intervention = SimpleNamespace(poll=lambda *_args: False)
    bot.feedback = StalledFeedback()
    bot.enable_opening_autopilot("地图中心敌方远端")
    analysis = BattleAnalysis(
        image=None,
        width=2560,
        height=1494,
        in_battle=True,
        autopilot_enabled=True,
        speed_knots=12.0,
        player_position=(100, 100),
    )

    bot._execute_rules(analysis, 100.0)

    assert bot.autopilot_retry_pending
    assert not bot.opening_autopilot_active
    assert "位移闭环" in bot.last_movement_reason


def test_battle_feedback_accepts_slow_battleship_minimap_progress():
    bot = BattleBot(
        1,
        {"strategy": {}},
        vision=object(),
        gamepad=RecordingGamepad(),
    )

    assert bot.feedback.update(0, (100, 100), 1.0).pending
    assert bot.feedback.update(12, (102, 100), 1.0).verified


def test_lost_native_autopilot_requests_retry_before_generic_qe_route():
    class FallbackGamepad(RecordingGamepad):
        def __init__(self):
            super().__init__()
            self.takeovers = 0
            self.reassertions = 0

        def takeover_from_autopilot(self):
            self.takeovers += 1

        def reassert_full_speed(self):
            self.reassertions += 1

    gamepad = FallbackGamepad()
    bot = BattleBot(
        1,
        {"strategy": {}},
        vision=object(),
        gamepad=gamepad,
    )
    bot.intervention = SimpleNamespace(poll=lambda *_args: False)
    bot.enable_opening_autopilot("最近占领点远端")
    now = time.monotonic()
    bot.battle_start_time = now - 30
    analysis = BattleAnalysis(
        image=None,
        width=2560,
        height=1600,
        player_position=(100, 100),
        map_center_bearing=0.15,
        map_center_distance_km=12.0,
        capture_point_bearing=-0.4,
        capture_point_distance_km=8.0,
    )

    # A single missed HUD sample must not cancel native navigation or fall
    # through into ordinary Q/E steering.  Only three consecutive misses are
    # accepted as a real native-autopilot loss.
    bot._execute_rules(analysis, now)
    bot._execute_rules(analysis, now + 0.1)

    assert gamepad.takeovers == 0
    assert gamepad.movements == []
    assert bot.opening_autopilot_active

    bot._execute_rules(analysis, now + 0.2)

    # The route disappeared before a confirmed arrival. Q/E remains blocked
    # while the lifecycle receives a request to retry the tactical-map route.
    assert gamepad.takeovers == 1
    assert gamepad.movements == []
    assert not bot.opening_autopilot_active
    assert bot.autopilot_retry_pending
    assert not bot.generic_center_route_active


def test_native_autopilot_arrival_allows_qe_on_following_frame():
    gamepad = RecordingGamepad()
    bot = BattleBot(1, {"strategy": {}}, vision=object(), gamepad=gamepad)
    bot.intervention = SimpleNamespace(poll=lambda *_args: False)
    bot.enable_opening_autopilot(
        "地图中心敌方远端",
        target_normalized=(0.50, 0.25),
    )
    now = time.monotonic()
    analysis = BattleAnalysis(
        image=None,
        width=2560,
        height=1600,
        player_position=(100, 100),
        minimap_player_normalized=(0.52, 0.29),
        map_center_bearing=0.15,
        map_center_distance_km=12.0,
    )

    for offset in (0.0, 0.1, 0.2):
        bot._execute_rules(analysis, now + offset)

    assert not bot.autopilot_retry_pending
    assert bot.generic_center_route_active
    assert gamepad.movements == []

    bot._execute_rules(analysis, now + 0.3)
    assert len(gamepad.movements) == 1


def test_live_autopilot_hud_hard_interlock_blocks_even_emergency_rudder():
    class InterlockedGamepad(RecordingGamepad):
        def __init__(self):
            super().__init__()
            self.takeovers = 0

        def takeover_from_autopilot(self):
            self.takeovers += 1

    gamepad = InterlockedGamepad()
    bot = BattleBot(
        1,
        {"strategy": {"opening_autopilot_minimum_seconds": 35}},
        vision=object(),
        gamepad=gamepad,
    )
    bot.intervention = SimpleNamespace(poll=lambda *_args: False)
    bot.enable_opening_autopilot("最近占领点远端")
    now = time.monotonic()
    bot.battle_start_time = now - 5
    analysis = BattleAnalysis(
        image=None,
        width=2560,
        height=1600,
        autopilot_enabled=True,
        player_position=(100, 100),
        minimap_enemy_count=1,
        minimap_distance_km=5.0,
        island_distance=0.01,
        island_avoidance_rudder=-1.0,
        capture_point_bearing=0.8,
        capture_point_distance_km=8.0,
    )

    bot._execute_rules(analysis, now)

    assert gamepad.takeovers == 0
    assert gamepad.movements == []
    assert bot.opening_autopilot_active
    assert bot.last_movement_reason.startswith("游戏自动航行")


def test_enemy_contact_never_cancels_native_autopilot():
    class ContactGamepad(RecordingGamepad):
        def __init__(self):
            super().__init__()
            self.takeovers = 0

        def takeover_from_autopilot(self):
            self.takeovers += 1

    gamepad = ContactGamepad()
    bot = BattleBot(
        1,
        {"strategy": {"opening_autopilot_minimum_seconds": 0}},
        vision=object(),
        gamepad=gamepad,
    )
    bot.intervention = SimpleNamespace(poll=lambda *_args: False)
    bot.enable_opening_autopilot("最近占领点远端")
    now = time.monotonic()
    bot.battle_start_time = now - 60
    analysis = BattleAnalysis(
        image=None,
        width=2560,
        height=1600,
        autopilot_enabled=True,
        player_position=(100, 100),
        minimap_enemy_count=1,
        minimap_distance_km=8.0,
        capture_point_bearing=0.3,
        capture_point_distance_km=7.0,
    )

    for frame in range(5):
        bot._execute_rules(analysis, now + frame * 0.1)

    # Enemy distance is telemetry only while the green native-autopilot HUD is
    # still present.  Q/E can resume only after that HUD naturally disappears.
    assert gamepad.takeovers == 0
    assert gamepad.movements == []
    assert bot.opening_autopilot_active
    assert not bot.generic_center_route_active

    # A continuing green HUD always remains authoritative and cannot allow a
    # Q/E command, even after enemies remain inside secondary range.
    bot._execute_rules(analysis, now + 0.5)
    assert gamepad.movements == []
    assert bot.opening_autopilot_active
