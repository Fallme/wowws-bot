"""Local battle analysis and rule-based orchestration."""

import logging
import math
import os
import time
from collections import deque
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from core.events import EventBus, JsonlEventRecorder
from core.input import create_input_controller
from core.feedback import MovementFeedbackMonitor, SafetyFault
from core.intervention import UserInterventionMonitor
from core.ocr import (
    DistanceOcrService,
    DistanceTrackFilter,
    TargetDistanceReader,
    ViewportTargetTracker,
)
from core.tracking import ConsecutivePointFilter, CourseHeadingFilter
from core.ui import ScreenState
from core.vision import Vision
from strategy.secondary_movement import (
    MovementMode,
    SecondaryMovementController,
    SecondaryMovementInput,
)
from strategy.route_planner import CoarseRoutePlanner
from strategy.stuck_recovery import StuckRecoveryController

logger = logging.getLogger("bot")
DEFAULT_DEBUG_ROOT = Path(
    os.environ.get(
        "WOWS_DEBUG_DIR",
        r"E:\aimemo\docs\screenshots\wowws_bot",
    )
)


@dataclass
class BattleAnalysis:
    image: Any
    width: int
    height: int
    enemies: list[tuple[int, int]] = field(default_factory=list)
    torpedoes_incoming: bool = False
    nearby_enemy_ship_types: set[str] = field(default_factory=set)
    reload_ready: bool = False
    health: float = 1.0
    ended: bool = False
    in_battle: bool = False
    enemy_source: str = "none"
    visible_target: bool = False
    target_offset_x: float | None = None
    target_track_id: str | None = None
    target_distance_km: float | None = None
    minimap_distance_km: float | None = None
    distance_source: str = "unknown"
    distance_confidence: float = 0.0
    distance_ocr_raw: str = ""
    minimap_distance: float | None = None
    minimap_target_bearing: float | None = None
    minimap_enemy_count: int = 0
    player_position: tuple[int, int] | None = None
    map_center_bearing: float | None = None
    map_center_distance_km: float | None = None
    capture_point_bearing: float | None = None
    capture_point_distance_km: float | None = None
    inside_capture_point: bool = False
    route_phase: str = "unplanned"
    route_progress: float = 0.0
    route_waypoint: int = 0
    route_arrived: bool = False
    island_distance: float | None = None
    island_avoidance_rudder: float | None = None
    movement_verified: bool = False
    on_fire: bool = False


def torpedo_evasion_threat(
    torpedoes_confirmed: bool,
    nearest_enemy_distance_km: float | None,
    nearby_enemy_ship_types,
) -> bool:
    """Allow evasion only for a confirmed nearby torpedo-capable ship."""
    normalized_types = {
        str(value).strip().lower() for value in (nearby_enemy_ship_types or ())
    }
    return bool(
        torpedoes_confirmed
        and nearest_enemy_distance_km is not None
        and nearest_enemy_distance_km <= 10.0
        and {"destroyer", "cruiser"} & normalized_types
    )


class BattleBot:
    """Coordinate local visual analysis and deterministic battle rules."""

    def __init__(
        self,
        hwnd,
        ship_config: dict,
        *,
        vision=None,
        gamepad=None,
        distance_reader=None,
    ):
        self.hwnd = hwnd
        self.ship = ship_config
        self.strategy = ship_config.get("strategy", {})
        self.vision = vision or Vision()
        # Keep the attribute name for compatibility with existing strategy and
        # tests; the default implementation is now native keyboard SendInput.
        self.gamepad = gamepad or create_input_controller()
        self._distance_ocr_async = distance_reader is None
        self.distance_reader = distance_reader or TargetDistanceReader(
            minimum_confidence=float(
                self.strategy.get("distance_ocr_min_confidence", 0.78)
            )
        )
        self.distance_ocr_service = (
            DistanceOcrService(self.distance_reader)
            if self._distance_ocr_async
            else None
        )

        self.movement = SecondaryMovementController(
            preferred_side=int(self.strategy.get("preferred_side", 1)),
            opening_seconds=float(self.strategy.get("opening_seconds", 55)),
            disengage_health=float(self.strategy.get("disengage_health", 0.28)),
            secondary_enter_distance=float(
                self.strategy.get(
                    "secondary_enter_distance",
                    self.strategy.get("brawl_map_distance", 0.18),
                )
            ),
            ideal_outer_distance=float(
                self.strategy.get("ideal_outer_distance", 0.15)
            ),
            ideal_inner_distance=float(
                self.strategy.get("ideal_inner_distance", 0.10)
            ),
            too_close_distance=float(
                self.strategy.get("too_close_distance", 0.07)
            ),
            island_warning_distance=float(
                self.strategy.get("island_warning_distance", 0.10)
            ),
            island_emergency_distance=float(
                self.strategy.get("island_emergency_distance", 0.055)
            ),
            secondary_range_km=float(
                self.ship.get("secondary", {}).get("range", 11.4)
            ),
            brake_start_km=float(
                self.strategy.get("brake_start_distance_km", 13.0)
            ),
            ideal_outer_km=float(
                self.strategy.get(
                    "ideal_outer_distance_km",
                    self.ship.get("secondary", {}).get("range", 11.4),
                )
            ),
            ideal_inner_km=float(
                self.strategy.get("ideal_inner_distance_km", 7.0)
            ),
            too_close_km=float(
                self.strategy.get("too_close_distance_km", 4.5)
            ),
            capture_throttle=float(
                self.strategy.get("capture_throttle", 0.36)
            ),
            enemy_steering_weight=float(
                self.strategy.get("enemy_steering_weight", 0.28)
            ),
            center_steering_gain=float(
                self.strategy.get("center_steering_gain", 1.35)
            ),
            straight_opening_seconds=float(
                self.strategy.get("straight_opening_seconds", 12.0)
            ),
            secondary_target_km=float(
                self.strategy.get("secondary_target_distance_km", 10.0)
            ),
            secondary_inner_km=float(
                self.strategy.get("secondary_inner_distance_km", 7.0)
            ),
        )
        self.stuck_recovery = StuckRecoveryController(
            preferred_side=int(self.strategy.get("preferred_side", 1)),
            stationary_seconds=float(self.strategy.get("stuck_seconds", 24)),
        )
        self.feedback = MovementFeedbackMonitor(
            timeout_seconds=float(self.strategy.get("feedback_timeout_seconds", 18)),
            missing_timeout_seconds=float(
                self.strategy.get("position_missing_timeout_seconds", 10)
            ),
        )
        self.minimap_target_filter = ConsecutivePointFilter(match_radius=18)
        self.course_heading_filter = CourseHeadingFilter(
            minimum_travel=float(
                self.strategy.get("course_minimum_travel_pixels", 5.0)
            )
        )
        self.route_planner = CoarseRoutePlanner(
            zone_match_ratio=float(
                self.strategy.get("capture_zone_match_ratio", 0.08)
            )
        )
        self.viewport_target_filter = ConsecutivePointFilter(match_radius=90)
        self.viewport_target_tracker = ViewportTargetTracker(match_radius=140)
        self.distance_filter = DistanceTrackFilter(
            stable_samples=int(self.strategy.get("distance_stable_frames", 2)),
            maximum_spread_km=float(
                self.strategy.get("distance_maximum_spread_km", 0.45)
            ),
            stale_seconds=float(
                self.strategy.get("distance_stale_seconds", 1.2)
            ),
        )
        self.smoke_used = False
        self.last_fire = 0.0
        self.last_lock = 0.0
        self.last_torpedo = 0.0
        self.last_damage_control = 0.0
        self.last_heal = 0.0
        self.heal_used = 0
        self.tick = 0
        self.battle_start_time = 0.0
        self._cached_health = 1.0
        self._cached_reload = False
        self._health_samples = deque(maxlen=5)
        self._torpedo_samples = deque(maxlen=3)
        self._fire_samples = deque(maxlen=2)
        self._secondary_contact_samples = deque(maxlen=5)
        self._island_samples = deque(maxlen=5)
        self._capture_zone = None
        self._unknown_since = None
        self._post_battle_grace_seconds = float(
            self.strategy.get("post_battle_grace_seconds", 900.0)
        )
        self.intervention = UserInterventionMonitor(
            hwnd,
            pause_seconds=float(
                self.strategy.get("manual_intervention_pause_seconds", 5.0)
            ),
            latch_seconds=float(
                self.strategy.get("manual_intervention_latch_seconds", 15.0)
            ),
        )
        self.movement_verified = False
        self._last_full_speed_reassert = 0.0
        self._manual_intervention_active = False
        self._manual_intervention_latched = False
        self.opening_autopilot_active = False
        self.opening_autopilot_target = ""
        self._last_movement_mode = None
        self._last_ocr_at = 0.0
        self._ocr_failures = 0
        self._debug_dir = DEFAULT_DEBUG_ROOT / "pending"
        self.events = EventBus()
        self._event_recorder = None
        self.last_analysis: BattleAnalysis | None = None
        self.last_movement_command = None
        self.last_movement_reason = ""

    def reset(self):
        now = time.monotonic()
        if self._event_recorder is not None:
            self._event_recorder.close()
        self.smoke_used = False
        self.last_fire = now
        self.last_lock = now
        self.last_torpedo = now
        damage_control_cooldown = float(
            self.strategy.get("damage_control_cooldown_seconds", 80.0)
        )
        heal_cooldown = float(
            self.strategy.get("heal_cooldown_seconds", 80.0)
        )
        self.last_damage_control = now - damage_control_cooldown
        self.last_heal = now - heal_cooldown
        self.heal_used = 0
        self.tick = 0
        self.battle_start_time = now
        self._cached_health = 1.0
        self._cached_reload = False
        self._health_samples.clear()
        self._torpedo_samples.clear()
        self._fire_samples.clear()
        self._secondary_contact_samples.clear()
        self._island_samples.clear()
        self._capture_zone = None
        self._unknown_since = None
        self.movement_verified = False
        self._last_full_speed_reassert = 0.0
        self._manual_intervention_active = False
        self._manual_intervention_latched = False
        self.opening_autopilot_active = False
        self.opening_autopilot_target = ""
        self._last_movement_mode = None
        self._last_ocr_at = 0.0
        self._ocr_failures = 0
        self._debug_dir = DEFAULT_DEBUG_ROOT / time.strftime("run_%Y%m%d_%H%M%S")
        self.events = EventBus()
        self._event_recorder = JsonlEventRecorder(self._debug_dir / "events.jsonl")
        self.events.subscribe(self._event_recorder)
        self.events.publish(
            "battle.started",
            ship=self.ship.get("name", "unknown"),
            secondary_range_km=self.ship.get("secondary", {}).get("range"),
        )
        self.last_analysis = None
        self.last_movement_command = None
        self.last_movement_reason = ""
        self.movement.reset()
        self.stuck_recovery.reset()
        self.feedback.reset()
        self.minimap_target_filter.reset()
        self.course_heading_filter.reset()
        self.route_planner.reset()
        self.viewport_target_filter.reset()
        self.viewport_target_tracker.reset()
        self.distance_filter.reset()
        self.intervention.reset()
        if self._distance_ocr_async:
            self.distance_ocr_service.close()
            self.distance_ocr_service = DistanceOcrService(self.distance_reader)
        self.gamepad.stop()

    def enable_opening_autopilot(self, target: str):
        self.opening_autopilot_active = True
        self.opening_autopilot_target = str(target or "地图中心")
        self.last_movement_command = None
        self.last_movement_reason = (
            f"游戏自动航行已设定至{self.opening_autopilot_target}"
        )
        self._last_movement_mode = "autopilot_route"

    def analyze(self) -> BattleAnalysis:
        observed_at = time.monotonic()
        try:
            # A stationary ship or tactical-map transition can legitimately
            # produce identical frames. Movement feedback, not frame novelty,
            # is the authoritative safety check during battle.
            image = self.vision.grab(self.hwnd, allow_stale=True)
        except TypeError:
            # Lightweight fixture/custom vision implementations may expose the
            # older one-argument interface.
            image = self.vision.grab(self.hwnd)
        height, width = image.shape[:2]
        analysis = BattleAnalysis(image=image, width=width, height=height)
        state = self.vision.classify_screen(image)
        analysis.ended = state in {ScreenState.RESULTS, ScreenState.PORT}
        analysis.in_battle = state == ScreenState.BATTLE

        if not analysis.in_battle:
            self._island_samples.clear()
            self.viewport_target_tracker.reset()
            self.distance_filter.reset()
        if analysis.ended:
            return analysis

        if analysis.in_battle:
            minimap = self.vision.find_minimap(image)
            minimap_enemies = []
            island_sample = None
            if minimap is not None:
                minimap_enemies, torpedoes_seen = (
                    self.vision.analyze_minimap(minimap)
                )
                if len(minimap_enemies) > 16:
                    logger.warning(
                        "拒绝异常小地图检测: %s 个敌舰候选", len(minimap_enemies)
                    )
                    minimap_enemies = []
                minimap_enemies = self.minimap_target_filter.update(minimap_enemies)
                # The minimap contains large yellow capture/range overlays that
                # are visually indistinguishable from the old broad torpedo
                # mask. False positives here caused an immediate hard turn away
                # from the capture point. Keep this emergency override disabled
                # until a local, player-relative torpedo detector is available.
                self._torpedo_samples.append(bool(torpedoes_seen))
                analysis.minimap_enemy_count = len(minimap_enemies)
                pose = self.vision.find_player_pose_on_minimap(minimap)
                player = None if pose is None else pose.position
                analysis.player_position = player
                if pose is not None:
                    course_heading = self.course_heading_filter.update(player)
                    navigation_pose = (
                        pose
                        if course_heading is None
                        else replace(pose, heading=course_heading)
                    )
                    map_center = (
                        minimap.shape[1] / 2.0,
                        minimap.shape[0] / 2.0,
                    )
                    center_pixels = math.dist(player, map_center)
                    analysis.map_center_distance_km = (
                        self.vision.minimap_pixels_to_km(
                            minimap, center_pixels
                        )
                    )
                    analysis.map_center_bearing = self.vision.relative_bearing(
                        navigation_pose, map_center
                    )
                    if self.route_planner.zone is None or self.tick % 15 == 0:
                        detected_zone = self.vision.find_central_capture_zone(
                            minimap
                        )
                        if detected_zone is not None:
                            self.route_planner.observe_zone(
                                detected_zone,
                                minimap.shape,
                            )
                    self._capture_zone = self.route_planner.zone
                    if self._capture_zone is not None:
                        capture_pixels = math.dist(
                            player, self._capture_zone.center
                        )
                        analysis.capture_point_distance_km = (
                            self.vision.minimap_pixels_to_km(
                                minimap, capture_pixels
                            )
                        )
                        arrival_distance_km = float(
                            self.strategy.get(
                                "capture_arrival_distance_km", 4.5
                            )
                        )
                        analysis.inside_capture_point = (
                            capture_pixels <= self._capture_zone.radius * 0.95
                            or analysis.capture_point_distance_km
                            <= arrival_distance_km
                        )
                        route = self.route_planner.update(
                            player,
                            inside_zone=analysis.inside_capture_point,
                        )
                        route_target = route.target or self._capture_zone.center
                        analysis.capture_point_bearing = (
                            self.vision.relative_bearing(
                                navigation_pose, route_target
                            )
                        )
                        analysis.route_phase = route.phase
                        analysis.route_progress = route.progress
                        analysis.route_waypoint = route.waypoint_index
                        analysis.route_arrived = route.arrived
                    else:
                        route = self.route_planner.update(player)
                        analysis.route_phase = route.phase
                        analysis.route_progress = route.progress
                        analysis.route_waypoint = route.waypoint_index
                        analysis.route_arrived = route.arrived
                island_risk = (
                    None
                    if pose is None or course_heading is None
                    else self.vision.find_island_risk(minimap, navigation_pose)
                )
                if island_risk is not None:
                    island_sample = (
                        island_risk.distance,
                        island_risk.avoidance_rudder,
                    )
                if pose is not None and minimap_enemies:
                    nearest_map_enemy = min(
                        minimap_enemies,
                        key=lambda enemy: (enemy[0] - player[0]) ** 2
                        + (enemy[1] - player[1]) ** 2,
                    )
                    diagonal = math.hypot(minimap.shape[1], minimap.shape[0])
                    minimap_pixels = math.hypot(
                        nearest_map_enemy[0] - player[0],
                        nearest_map_enemy[1] - player[1],
                    )
                    analysis.minimap_distance = minimap_pixels / max(
                        diagonal, 1
                    )
                    analysis.minimap_distance_km = (
                        self.vision.minimap_pixels_to_km(
                            minimap, minimap_pixels
                        )
                    )
                    analysis.minimap_target_bearing = (
                        self.vision.relative_bearing(
                            navigation_pose, nearest_map_enemy
                        )
                    )

                type_detector = getattr(
                    self.vision,
                    "classify_nearby_enemy_ship_types",
                    None,
                )
                if type_detector is not None:
                    detected_types = type_detector(image, minimap)
                    analysis.nearby_enemy_ship_types = {
                        str(value).strip().lower() for value in detected_types
                    }
                analysis.torpedoes_incoming = bool(
                    bool(self.strategy.get("enable_minimap_torpedo_evasion", True))
                    and torpedo_evasion_threat(
                        sum(self._torpedo_samples) >= 2,
                        analysis.minimap_distance_km,
                        analysis.nearby_enemy_ship_types,
                    )
                )

            self._island_samples.append(island_sample)
            stable_island = self._stable_island_risk()
            if stable_island is not None:
                (
                    analysis.island_distance,
                    analysis.island_avoidance_rudder,
                ) = stable_island

            viewport_enemies = self.vision.find_enemies_in_viewport(image)
            if len(viewport_enemies) > 8:
                logger.warning(
                    "拒绝异常主视野检测: %s 个敌舰候选", len(viewport_enemies)
                )
                viewport_enemies = []
            viewport_enemies = self.viewport_target_filter.update(viewport_enemies)
            if viewport_enemies:
                analysis.enemies = viewport_enemies
                analysis.enemy_source = "viewport"
                analysis.visible_target = True
                center_x = width / 2
                (
                    target_track,
                    nearest,
                    observation,
                    evidence,
                    stable_distance,
                ) = self._update_target_distance(
                    image,
                    viewport_enemies,
                    observed_at,
                    center_x,
                )
                analysis.target_offset_x = max(
                    -1.0,
                    min((nearest[0] - center_x) / max(center_x, 1), 1.0),
                )
                if target_track is not None:
                    analysis.target_track_id = target_track.track_id
                    if stable_distance is not None:
                        analysis.target_distance_km = stable_distance.value_km
                        analysis.distance_confidence = stable_distance.confidence
                displayed_evidence = observation or evidence
                if displayed_evidence is not None:
                    analysis.distance_ocr_raw = displayed_evidence.raw_text
                    if analysis.target_distance_km is None:
                        analysis.distance_confidence = displayed_evidence.confidence
            elif minimap_enemies:
                analysis.enemies = minimap_enemies
                analysis.enemy_source = "minimap"
                self.viewport_target_tracker.update([], preferred_x=width / 2)
                self.distance_filter.update(observed_at, None, None)
            else:
                self.viewport_target_tracker.update([], preferred_x=width / 2)
                self.distance_filter.update(observed_at, None, None)

        health = self.vision.health_pct(
            self.vision.find_health_bar(image)
        )
        if health is not None:
            # Sample every control tick so repair and low-health decisions do
            # not inherit the old multi-second delay.
            health = max(0.0, min(float(health), 1.0))
            self._health_samples.append(health)
            if len(self._health_samples) >= 3:
                ordered = sorted(self._health_samples)
                spread = ordered[-1] - ordered[0]
                if spread <= 0.18:
                    median_health = ordered[len(ordered) // 2]
                    if median_health <= self._cached_health + 0.05:
                        self._cached_health = median_health
        self._fire_samples.append(bool(self.vision.is_on_fire(image)))
        analysis.on_fire = (
            len(self._fire_samples) == self._fire_samples.maxlen
            and all(self._fire_samples)
        )
        if self.tick % 3 == 0:
            self._cached_reload = self.vision.reload_ready(
                self.vision.find_reload_bar(image)
            )
        analysis.health = self._cached_health
        analysis.reload_ready = self._cached_reload
        analysis.movement_verified = self.movement_verified

        if self.tick % 20 == 0:
            self.vision.save_debug_frame(
                self._debug_dir / f"full_{self.tick:04d}.png",
                image,
            )
            logger.info(
                "视觉: 敌人=%s 来源=%s 目标距离=%s 小地图网格=%s 中央点=%s 点内=%s 前方岛距=%s",
                len(analysis.enemies),
                analysis.enemy_source,
                "未知"
                if analysis.target_distance_km is None
                else f"{analysis.target_distance_km:.1f}km",
                "未知"
                if analysis.minimap_distance_km is None
                else f"{analysis.minimap_distance_km:.1f}km",
                "未知"
                if analysis.capture_point_distance_km is None
                else f"{analysis.capture_point_distance_km:.1f}km",
                "是" if analysis.inside_capture_point else "否",
                "安全"
                if analysis.island_distance is None
                else f"{analysis.island_distance:.2f}",
            )

        return analysis

    def _stable_island_risk(self):
        """Require four consistent terrain observations before manoeuvring."""
        required = 4
        if len(self._island_samples) < required:
            return None
        samples = list(self._island_samples)[-required:]
        if any(sample is None for sample in samples):
            return None
        distances = [sample[0] for sample in samples]
        if max(distances) - min(distances) > 0.035:
            return None
        sides = [sample[1] for sample in samples if abs(sample[1]) >= 0.5]
        positive = sum(side > 0 for side in sides)
        negative = sum(side < 0 for side in sides)
        if max(positive, negative) < 3:
            return None
        side = 1.0 if positive > negative else -1.0
        distance = sum(distances) / len(distances)
        return distance, side

    def _distance_candidate_ids(self, candidates):
        identified = []
        active = self.viewport_target_tracker.active
        for index, candidate in enumerate(candidates[:3]):
            provisional_id = f"candidate-{index}"
            if active is not None and math.dist(candidate, active.point) <= (
                self.viewport_target_tracker.match_radius
            ):
                provisional_id = active.track_id
            identified.append((candidate, provisional_id))
        return identified

    def _record_ocr_error(self, error):
        self._ocr_failures += 1
        if self._ocr_failures <= 3 or self._ocr_failures % 20 == 0:
            logger.warning("目标距离 OCR 失败: %s", error)

    def _update_target_distance(
        self,
        image,
        viewport_enemies,
        observed_at,
        center_x,
    ):
        candidates = self.viewport_target_tracker.ordered_candidates(
            viewport_enemies,
            preferred_x=center_x,
        )
        observation = None
        evidence = None
        target_track = None
        ocr_interval = float(
            self.strategy.get("distance_ocr_interval_seconds", 0.18)
        )
        should_read_ocr = (
            self.distance_filter.stable is None
            or observed_at - self._last_ocr_at >= max(ocr_interval, 0.05)
        )

        if self._distance_ocr_async:
            result = self.distance_ocr_service.poll()
            if result is not None:
                self.events.publish(
                    "distance.observed",
                    execution_provider=self.ocr_provider,
                    point=result.point,
                    value_km=None
                    if result.observation is None
                    else result.observation.value_km,
                    confidence=0.0
                    if result.observation is None
                    else result.observation.confidence,
                    accepted=bool(
                        result.observation is not None
                        and result.observation.accepted
                    ),
                    raw_text=""
                    if result.observation is None
                    else result.observation.raw_text,
                    error=result.error,
                )
                if result.error:
                    self._record_ocr_error(result.error)
                else:
                    self._ocr_failures = 0
                evidence = result.evidence
                if result.point is not None and result.observation is not None:
                    current_point = min(
                        viewport_enemies,
                        key=lambda point: math.dist(point, result.point),
                    )
                    if math.dist(current_point, result.point) <= (
                        self.viewport_target_tracker.match_radius * 1.5
                    ):
                        target_track = self.viewport_target_tracker.adopt(current_point)
                        observation = replace(
                            result.observation,
                            target_track_id=target_track.track_id,
                        )
            if target_track is None:
                target_track = self.viewport_target_tracker.match_active(
                    viewport_enemies
                )
            if should_read_ocr and not self.distance_ocr_service.pending:
                submitted = self.distance_ocr_service.submit(
                    image,
                    self._distance_candidate_ids(candidates),
                    captured_at=observed_at,
                )
                if submitted:
                    self._last_ocr_at = observed_at
        elif should_read_ocr:
            for candidate, provisional_id in self._distance_candidate_ids(candidates):
                try:
                    candidate_observation = self.distance_reader.read(
                        image,
                        candidate,
                        provisional_id,
                        captured_at=observed_at,
                    )
                    self._ocr_failures = 0
                except Exception as error:
                    self._record_ocr_error(error)
                    self.events.publish(
                        "distance.observed",
                        point=candidate,
                        accepted=False,
                        error=str(error),
                    )
                    continue
                self.events.publish(
                    "distance.observed",
                    execution_provider=self.ocr_provider,
                    point=candidate,
                    value_km=None
                    if candidate_observation is None
                    else candidate_observation.value_km,
                    confidence=0.0
                    if candidate_observation is None
                    else candidate_observation.confidence,
                    accepted=bool(
                        candidate_observation is not None
                        and candidate_observation.accepted
                    ),
                    raw_text=""
                    if candidate_observation is None
                    else candidate_observation.raw_text,
                )
                if candidate_observation is not None and (
                    evidence is None
                    or candidate_observation.confidence > evidence.confidence
                ):
                    evidence = candidate_observation
                if candidate_observation is not None and candidate_observation.accepted:
                    target_track = self.viewport_target_tracker.adopt(candidate)
                    observation = replace(
                        candidate_observation,
                        target_track_id=target_track.track_id,
                    )
                    break
            self._last_ocr_at = observed_at
        if target_track is None:
            target_track = self.viewport_target_tracker.match_active(viewport_enemies)

        if target_track is not None:
            stable_distance = self.distance_filter.update(
                observed_at,
                target_track.track_id,
                observation,
            )
            nearest = target_track.point
        else:
            stable_distance = self.distance_filter.update(observed_at, None, None)
            nearest = candidates[0]
        return target_track, nearest, observation, evidence, stable_distance

    def combat_tick(self):
        analysis = self.analyze()
        self.last_analysis = analysis
        if analysis.ended:
            self.gamepad.stop()
            self.events.publish("battle.ended", tick=self.tick)
            return "ended"

        if not analysis.in_battle:
            # A single missed/transition frame must not pull the engine from
            # FULL to STOP. Release steering only and wait for positive result
            # evidence; this also covers the spectator period after death.
            pause = getattr(self.gamepad, "pause_automation", None)
            if pause is not None:
                pause()
            else:
                self.gamepad.stop()
            now = time.monotonic()
            if self._unknown_since is None:
                self._unknown_since = now
            if now - self._unknown_since >= self._post_battle_grace_seconds:
                self.gamepad.stop()
                raise SafetyFault("战斗中连续多帧无法确认战斗 HUD，已停止控制")
            self._finish_tick(analysis)
            return "waiting"

        self._unknown_since = None

        now = time.monotonic()
        self._execute_rules(analysis, now)
        self._finish_tick(analysis)
        return "combat"

    def _execute_rules(self, analysis: BattleAnalysis, now: float):
        if self.intervention.poll(self.gamepad, now):
            # Freeze command generation only. Do not release rudder or change
            # the latched engine telegraph: the ship keeps executing the last
            # accepted command while the player interacts.
            pause_source = (
                "网页手动暂停"
                if self.intervention.web_paused
                else "用户键盘介入"
            )
            self.last_movement_reason = (
                f"{pause_source}：保持既有油门与舵，暂停下发新指令"
            )
            if not self._manual_intervention_active:
                logger.info(
                    "[USER] %s，冻结新系统指令%s；保持既有航行状态",
                    pause_source,
                    "直到网页继续"
                    if self.intervention.web_paused
                    else f" {self.intervention.pause_seconds:.1f} 秒",
                )
            if self.intervention.latched and not self._manual_intervention_latched:
                logger.warning(
                    "[USER] 持续介入达到 %.1f 秒，进入手动保持；等待网页继续",
                    self.intervention.latch_seconds,
                )
            self._manual_intervention_active = True
            self._manual_intervention_latched = self.intervention.latched
            self._last_movement_mode = "manual_pause"
            return

        if self._manual_intervention_active:
            source = "网页确认" if self.intervention.resumed_from_web else "5秒内无用户输入"
            logger.info("[SYSTEM] %s，下一控制帧恢复进点主航线", source)
            if not self.movement_verified:
                self.feedback.reset()
            self._manual_intervention_active = False
            self._manual_intervention_latched = False
            self._last_movement_mode = None

        if self.opening_autopilot_active:
            map_distance = analysis.minimap_distance_km
            # Autopilot takeover is based only on the white player marker and
            # red enemy markers on the minimap. Viewport OCR can lock aircraft
            # or friendly labels and must never cancel game-native navigation.
            distance_in_range = bool(
                map_distance is not None
                and map_distance <= self.movement.secondary_range_km
            )
            self._secondary_contact_samples.append(
                distance_in_range
            )
            confirmed_contact = bool(
                len(self._secondary_contact_samples) >= 3
                and all(list(self._secondary_contact_samples)[-3:])
            )
            if not confirmed_contact:
                self.last_movement_command = None
                self.last_movement_reason = (
                    f"游戏自动航行至{self.opening_autopilot_target}；"
                    f"按小地图5km网格等待敌舰进入{self.movement.secondary_range_km:.1f}km副炮射程"
                )
                if self._last_movement_mode != "autopilot_route":
                    logger.info("[SYSTEM] %s", self.last_movement_reason)
                self._last_movement_mode = "autopilot_route"
                feedback = self.feedback.update(
                    now,
                    analysis.player_position,
                    1.0,
                )
                self.movement_verified = feedback.verified
                return

            takeover = getattr(self.gamepad, "takeover_from_autopilot", None)
            if takeover is not None:
                takeover()
            else:
                self.gamepad.full_speed()
            self.opening_autopilot_active = False
            self._secondary_contact_samples.clear()
            self._last_movement_mode = None
            logger.info(
                "[SYSTEM] 敌舰距离%.1fkm进入副炮射程，取消游戏自动航行并切回系统驾驶",
                map_distance,
            )

        enemy_count = max(len(analysis.enemies), analysis.minimap_enemy_count)
        command = self.movement.plan(
            SecondaryMovementInput(
                elapsed=max(0.0, now - self.battle_start_time),
                health=analysis.health,
                visible_target=analysis.visible_target,
                target_offset_x=analysis.target_offset_x,
                target_distance_km=analysis.target_distance_km,
                minimap_distance=analysis.minimap_distance,
                minimap_distance_km=analysis.minimap_distance_km,
                minimap_target_bearing=analysis.minimap_target_bearing,
                map_center_bearing=analysis.map_center_bearing,
                map_center_distance_km=analysis.map_center_distance_km,
                capture_point_bearing=analysis.capture_point_bearing,
                capture_point_distance_km=analysis.capture_point_distance_km,
                inside_capture_point=analysis.inside_capture_point,
                route_phase=analysis.route_phase,
                route_arrived=analysis.route_arrived,
                enemy_count=enemy_count,
                torpedoes_incoming=analysis.torpedoes_incoming,
                island_distance=analysis.island_distance,
                island_avoidance_rudder=analysis.island_avoidance_rudder,
            )
        )
        self.last_movement_command = command
        self.last_movement_reason = command.reason
        safety_override = command.mode in {
            MovementMode.AVOID_ISLAND,
            MovementMode.EVADE,
            MovementMode.SEPARATE,
        }
        if safety_override:
            self.stuck_recovery.cancel()
            recovery = None
        else:
            recovery = self.stuck_recovery.update(
                now,
                analysis.player_position,
                command.throttle,
            )
        if recovery is not None:
            self.gamepad.set_movement(recovery.throttle, recovery.rudder)
            feedback = self.feedback.update(
                now, analysis.player_position, recovery.throttle
            )
            self.movement_verified = feedback.verified
            recovery_mode = f"recovery:{recovery.phase}"
            self.last_movement_command = None
            self.last_movement_reason = "舰船位置长时间未变化，执行自动脱困"
            if recovery_mode != self._last_movement_mode:
                logger.warning(
                    "检测到舰船位置长时间不变，自动脱困: %s | 推力=%.2f 舵角=%.2f",
                    recovery.phase,
                    recovery.throttle,
                    recovery.rudder,
                )
                self._last_movement_mode = recovery_mode
            return

        self.gamepad.set_movement(command.throttle, command.rudder)
        if (
            command.throttle >= 0.95
            and not self.movement_verified
            and now - self._last_full_speed_reassert >= 1.5
        ):
            reassert = getattr(self.gamepad, "reassert_full_speed", None)
            if reassert is not None:
                reassert()
                self._last_full_speed_reassert = now
        feedback = self.feedback.update(now, analysis.player_position, command.throttle)
        self.movement_verified = feedback.verified
        if command.mode != self._last_movement_mode:
            logger.info(
                "[SYSTEM] 移动状态: %s | 推力=%.2f 舵角=%.2f | %s",
                command.mode.value,
                command.throttle,
                command.rudder,
                command.reason,
            )
            self._last_movement_mode = command.mode

        # Target-lock input is intentionally deferred until movement is proven;
        # the opening priority is to get underway and reach the capture point.
        if analysis.visible_target and self.movement_verified:
            self._engage_visible_enemy(analysis, now)

        damage_control_cooldown = float(
            self.strategy.get("damage_control_cooldown_seconds", 80.0)
        )
        if (
            analysis.on_fire
            and analysis.health < 0.99
            and now - self.last_damage_control >= damage_control_cooldown
            and hasattr(self.gamepad, "damage_control")
        ):
            self.gamepad.damage_control()
            self.last_damage_control = now
            logger.info("检测到持续着火，使用 R 损害管制")

        heal_threshold = float(self.strategy.get("heal_threshold", 0.55))
        heal_cooldown = float(
            self.strategy.get("heal_cooldown_seconds", 80.0)
        )
        heal_max_uses = int(self.strategy.get("heal_max_uses", 3))
        if (
            0 < analysis.health <= heal_threshold
            and self.heal_used < heal_max_uses
            and now - self.last_heal >= heal_cooldown
            and hasattr(self.gamepad, "heal")
        ):
            self.gamepad.heal()
            self.last_heal = now
            self.heal_used += 1
            logger.info(
                "血量 %.0f%%，使用 T 维修小组 (%s/%s)",
                analysis.health * 100,
                self.heal_used,
                heal_max_uses,
            )

        smoke_threshold = self.strategy.get("smoke_threshold", 0.5)
        if (
            self.ship.get("has_smoke")
            and self.strategy.get("auto_smoke", False)
            and not self.smoke_used
            and 0 < analysis.health < smoke_threshold
        ):
            self.gamepad.smoke()
            self.smoke_used = True
            logger.info("那不勒斯使用 4 号位烟雾 (HP: %.0f%%)", analysis.health * 100)

        torpedo_reload = self.ship.get("torpedo", {}).get("reload", 90)
        if (
            self.ship.get("has_torpedoes")
            and analysis.visible_target
            and command.mode in {MovementMode.BRAWL, MovementMode.CAPTURE}
            and now - self.last_torpedo > torpedo_reload
        ):
            self.gamepad.torpedo()
            self.last_torpedo = now
            logger.info("[%03d] 鱼雷指令已派发", self.tick)

    def _engage_visible_enemy(self, analysis: BattleAnalysis, now: float):
        if now - self.last_lock > 3:
            self.gamepad.lock()
            self.last_lock = now
            logger.info("[%03d] 副炮锁定指令已派发", self.tick)

        # Main-gun fire is intentionally disabled while the movement loop is
        # being stabilised. X target lock remains active for secondary guns.

    def _finish_tick(self, analysis: BattleAnalysis):
        command = self.last_movement_command
        self.events.publish(
            "battle.tick",
            tick=self.tick,
            in_battle=analysis.in_battle,
            health=analysis.health,
            enemies=max(len(analysis.enemies), analysis.minimap_enemy_count),
            target_track_id=analysis.target_track_id,
            # Keep the public/control distance aligned with movement logic.
            # Viewport OCR remains available separately for diagnostics only.
            target_distance_km=analysis.minimap_distance_km,
            viewport_distance_km=analysis.target_distance_km,
            minimap_distance_km=analysis.minimap_distance_km,
            distance_source=(
                "minimap_grid"
                if analysis.minimap_distance_km is not None
                else "unknown"
            ),
            distance_confidence=analysis.distance_confidence,
            distance_ocr_raw=analysis.distance_ocr_raw,
            minimap_distance=analysis.minimap_distance,
            capture_point_distance_km=analysis.capture_point_distance_km,
            inside_capture_point=analysis.inside_capture_point,
            route_phase=analysis.route_phase,
            route_progress=analysis.route_progress,
            route_waypoint=analysis.route_waypoint,
            route_arrived=analysis.route_arrived,
            island_distance=analysis.island_distance,
            movement_mode=self.current_movement_mode,
            throttle=None if command is None else command.throttle,
            rudder=None if command is None else command.rudder,
            movement_reason=self.last_movement_reason,
            movement_verified=self.movement_verified,
        )
        if self.tick % 4 == 0:
            logger.info(
                "[%03d] HP:%.0f%% 装填:%s 敌人:%s 距离:%s 鱼雷:%s 反馈:%s",
                self.tick,
                analysis.health * 100,
                "完成" if analysis.reload_ready else "等待",
                max(len(analysis.enemies), analysis.minimap_enemy_count),
                "未知"
                if analysis.minimap_distance_km is None
                else f"{analysis.minimap_distance_km:.1f}km(小地图)",
                "有" if analysis.torpedoes_incoming else "无",
                "已确认" if self.movement_verified else "等待",
            )
        self.tick += 1

    @property
    def current_movement_mode(self) -> str:
        mode = self._last_movement_mode
        if isinstance(mode, MovementMode):
            return mode.value
        return str(mode or "idle")

    @property
    def ocr_status(self) -> str:
        analysis = self.last_analysis
        if analysis is not None and analysis.target_distance_km is not None:
            return "stable"
        if self.distance_ocr_service is not None and self.distance_ocr_service.pending:
            return "reading"
        if analysis is not None and analysis.visible_target:
            return "searching"
        return "no_target"

    @property
    def ocr_provider(self) -> str:
        return str(
            getattr(self.distance_reader, "execution_provider", "custom")
            or "custom"
        )

    @property
    def manual_intervention_latched(self) -> bool:
        return bool(self.intervention.latched)

    @property
    def manual_intervention_seconds(self) -> float:
        return self.intervention.continuous_seconds()

    def stop(self):
        self.gamepad.stop()
        self.events.publish("runtime.stopped", tick=self.tick)
        if self.distance_ocr_service is not None:
            self.distance_ocr_service.close()
        if self._event_recorder is not None:
            self._event_recorder.close()
            self._event_recorder = None
