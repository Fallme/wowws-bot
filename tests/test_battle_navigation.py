import math
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import pytest

from bot import BattleAnalysis, BattleBot, select_forward_navigation_enemy
from core.ocr import DistanceObservation, Rect
from core.vision import Vision
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


def test_battle_feedback_accepts_slow_battleship_minimap_progress():
    bot = BattleBot(
        1,
        {"strategy": {}},
        vision=object(),
        gamepad=RecordingGamepad(),
    )

    assert bot.feedback.update(0, (100, 100), 1.0).pending
    assert bot.feedback.update(12, (102, 100), 1.0).verified


def test_failed_native_autopilot_falls_through_to_generic_center_route():
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

    # Native navigation ended by itself; the controller does not send a
    # synthetic cancellation/full-speed command on this transition frame.
    assert gamepad.takeovers == 0
    assert gamepad.movements == []
    assert not bot.opening_autopilot_active
    assert bot.generic_center_route_active
    assert analysis.capture_point_bearing == analysis.map_center_bearing
    assert analysis.capture_point_distance_km == analysis.map_center_distance_km

    # The cancellation frame is deliberately steering-free.  Minimap driving
    # is allowed only on the following control frame.
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
