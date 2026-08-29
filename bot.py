"""Local battle analysis and rule-based orchestration."""

import logging
import math
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
DEFAULT_DEBUG_ROOT = (
    Path(__file__).resolve().parent / "runtime" / "screenshots" / "runs"
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
    health_recognized: bool = False
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
    minimap_player_normalized: tuple[float, float] | None = None
    minimap_heading: tuple[float, float] | None = None
    nearest_enemy_normalized: tuple[float, float] | None = None
    minimap_contacts: list[dict[str, Any]] = field(default_factory=list)
    capture_zones: list[dict[str, Any]] = field(default_factory=list)
    minimap_islands: list[dict[str, Any]] = field(default_factory=list)
    minimap_snapshot: str = ""
    capture_zone_center_normalized: tuple[float, float] | None = None
    capture_zone_radius_normalized: float | None = None
    capture_zone_label: str = ""
    navigation_target_normalized: tuple[float, float] | None = None
    navigation_source: str = "unknown"
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
    flooding: bool = False
    speed_knots: float | None = None
    autopilot_enabled: bool = False
    rudder_indicator: str = "neutral"


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


def select_forward_navigation_enemy(
    vision,
    pose,
    enemies,
    *,
    max_abs_bearing: float = 0.55,
):
    """Choose a red minimap contact that does not require an about-turn.

    A strictly-nearest selector is unstable after ships pass one another: a
    contact just behind the stern hides every useful contact ahead and makes
    the controller fall back to circling the objective. Keep navigation
    forward-only and mildly prefer contacts close to the current heading.
    """
    candidates = []
    for enemy in enemies or ():
        bearing = float(vision.relative_bearing(pose, enemy))
        if abs(bearing) > max_abs_bearing:
            continue
        distance = math.dist(pose.position, enemy)
        score = distance * (1.0 + 0.35 * abs(bearing))
        candidates.append((score, distance, abs(bearing), enemy, bearing))
    if not candidates:
        return None
    _, distance, _, enemy, bearing = min(
        candidates, key=lambda value: value[:3]
    )
    return enemy, distance, bearing


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
        self._owns_gamepad = gamepad is None
        self.gamepad = gamepad or create_input_controller(hwnd=hwnd)
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
            reverse_seconds=min(
                4.8, float(self.strategy.get("stuck_reverse_seconds", 4.5))
            ),
            forward_seconds=min(
                4.8, float(self.strategy.get("stuck_forward_seconds", 4.5))
            ),
        )
        self.feedback = MovementFeedbackMonitor(
            timeout_seconds=float(self.strategy.get("feedback_timeout_seconds", 18)),
            missing_timeout_seconds=float(
                self.strategy.get("position_missing_timeout_seconds", 10)
            ),
            movement_pixels=float(
                self.strategy.get("feedback_movement_pixels", 2.0)
            ),
        )
        self.max_movement_feedback_retries = max(
            1, int(self.strategy.get("movement_feedback_retries", 3))
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
        self._health_ocr_valid = False
        self._cached_speed_knots: float | None = None
        self._cached_reload = False
        self._torpedo_samples = deque(maxlen=3)
        self._fire_samples = deque(maxlen=2)
        self._flood_samples = deque(maxlen=2)
        self._autopilot_hud_samples = deque(maxlen=3)
        self._island_samples = deque(maxlen=5)
        self._island_avoidance_until = 0.0
        self._island_avoidance_rudder = 0.0
        self._capture_zone = None
        # Terrain and capture circles do not move during a match.  Freeze the
        # first reliable minimap interpretation and use it as a separate
        # static layer; only ship/contact markers are allowed to update later.
        self._battle_map_islands: list[dict[str, Any]] = []
        self._battle_capture_zones = []
        self._island_layer_candidates = deque(maxlen=5)
        self._zone_layer_candidates = deque(maxlen=5)
        self._unknown_since = None
        self._ended_state_streak = 0
        self._post_battle_grace_seconds = float(
            self.strategy.get("post_battle_grace_seconds", 900.0)
        )
        self.intervention = UserInterventionMonitor(
            hwnd,
            pause_seconds=float(
                self.strategy.get("manual_intervention_pause_seconds", 5.0)
            ),
            latch_seconds=float(
                self.strategy.get("manual_intervention_latch_seconds", 20.0)
            ),
        )
        self.intervention.reset()
        self.movement_verified = False
        self._last_full_speed_reassert = 0.0
        self._last_applied_rudder = 0.0
        self._rudder_commanded_at = 0.0
        self._rudder_release_until = 0.0
        self._manual_intervention_active = False
        self._manual_intervention_latched = False
        self.opening_autopilot_active = False
        self.opening_autopilot_target = ""
        self.opening_autopilot_target_normalized = None
        self.autopilot_retry_pending = False
        self._tactical_map_left_open = False
        self.generic_center_route_active = False
        self.movement_feedback_failures = 0
        self._last_movement_mode = None
        self._last_ocr_at = 0.0
        self._ocr_failures = 0
        self._debug_dir = DEFAULT_DEBUG_ROOT / "pending"
        self.events = EventBus()
        self._event_recorder = None
        self.last_analysis: BattleAnalysis | None = None
        self.last_movement_command = None
        self.last_movement_reason = ""

    def rebind_window(self, hwnd) -> bool:
        """Bind capture, intervention and native input to a replacement HWND.

        World of Warships can recreate its top-level window while changing
        display mode or recovering from a launcher/login transition.  Keeping
        the old handle makes both OCR and the keyboard focus guard target a
        dead window.  Custom/test controllers are intentionally preserved;
        only the controller owned by this bot is rebuilt.
        """
        new_hwnd = int(hwnd or 0)
        if not new_hwnd:
            return False
        old_hwnd = int(self.hwnd or 0)
        self.hwnd = new_hwnd
        if self._owns_gamepad and new_hwnd != old_hwnd:
            self.gamepad = create_input_controller(hwnd=new_hwnd)
        intervention = getattr(self, "intervention", None)
        if intervention is not None:
            intervention.hwnd = new_hwnd
            intervention.reset()
        if new_hwnd != old_hwnd:
            self.feedback.reset()
            self.movement_verified = False
            logger.info("游戏窗口已重新绑定: %s -> %s", old_hwnd, new_hwnd)
        return True

    def reset(self, *, preserve_movement=False):
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
        self._health_ocr_valid = False
        self._cached_speed_knots = None
        self._cached_reload = False
        self._torpedo_samples.clear()
        self._fire_samples.clear()
        self._flood_samples.clear()
        self._autopilot_hud_samples.clear()
        self._island_samples.clear()
        self._island_avoidance_until = 0.0
        self._island_avoidance_rudder = 0.0
        self._capture_zone = None
        self._unknown_since = None
        self._ended_state_streak = 0
        self.movement_verified = False
        self._last_full_speed_reassert = 0.0
        self._last_applied_rudder = 0.0
        self._rudder_commanded_at = 0.0
        self._rudder_release_until = 0.0
        self._manual_intervention_active = False
        self._manual_intervention_latched = False
        self.autopilot_retry_pending = False
        if not preserve_movement:
            self._battle_map_islands = []
            self._battle_capture_zones = []
            self._island_layer_candidates.clear()
            self._zone_layer_candidates.clear()
            self.opening_autopilot_active = False
            self.opening_autopilot_target = ""
            self.opening_autopilot_target_normalized = None
            self._tactical_map_left_open = False
            self.generic_center_route_active = False
        self.movement_feedback_failures = 0
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
        # Do not clear a real keyboard pause at a lifecycle boundary. The
        # monitor is seeded in __init__ and injected input is filtered by the
        # controller's timestamp; reset here used to erase intervention that
        # arrived while a new battle was being recognized.
        if self._distance_ocr_async:
            self.distance_ocr_service.close()
            self.distance_ocr_service = DistanceOcrService(self.distance_reader)
        if not preserve_movement:
            self.gamepad.stop()

    @staticmethod
    def _island_layer_signature(islands) -> tuple:
        shapes = []
        for island in islands or ():
            points = island.get("points", ()) if isinstance(island, dict) else ()
            if len(points) < 3:
                continue
            xs = [float(point[0]) for point in points]
            ys = [float(point[1]) for point in points]
            shapes.append(
                (
                    round(sum(xs) / len(xs) / 0.035),
                    round(sum(ys) / len(ys) / 0.035),
                    round((max(xs) - min(xs)) / 0.04),
                    round((max(ys) - min(ys)) / 0.04),
                )
            )
        return tuple(sorted(shapes))

    @staticmethod
    def _zone_layer_signature(zones, minimap_shape) -> tuple:
        height, width = minimap_shape[:2]
        scale = max(min(height, width), 1)
        return tuple(
            sorted(
                (
                    str(getattr(zone, "label", "") or "").upper(),
                    round((zone.center[0] / max(width, 1)) / 0.025),
                    round((zone.center[1] / max(height, 1)) / 0.025),
                    round((float(zone.radius) / scale) / 0.02),
                )
                for zone in (zones or ())
            )
        )

    @staticmethod
    def _confirmed_static_layer(samples, signature, items, *, required=3):
        if not items or not signature:
            samples.clear()
            return None
        samples.append((signature, list(items)))
        if len(samples) < required:
            return None
        recent = list(samples)[-required:]
        if all(sample_signature == signature for sample_signature, _ in recent):
            return recent[-1][1]
        return None

    @staticmethod
    def _refresh_zone_states(fixed_zones, observed_zones):
        """Keep fixed point geometry while allowing ownership colour to update."""
        refreshed = []
        for fixed in fixed_zones:
            compatible = [
                candidate
                for candidate in observed_zones
                if (
                    not getattr(fixed, "label", "")
                    or not getattr(candidate, "label", "")
                    or str(candidate.label).upper() == str(fixed.label).upper()
                )
                and math.dist(candidate.center, fixed.center)
                <= max(float(fixed.radius) * 0.35, 12.0)
            ]
            if compatible:
                nearest = min(
                    compatible,
                    key=lambda candidate: math.dist(candidate.center, fixed.center),
                )
                refreshed.append(
                    replace(fixed, state=getattr(nearest, "state", "unknown"))
                )
            else:
                refreshed.append(fixed)
        return refreshed

    def enable_opening_autopilot(
        self,
        target: str,
        *,
        target_normalized: tuple[float, float] | None = None,
    ):
        self.opening_autopilot_active = True
        self.generic_center_route_active = False
        self.opening_autopilot_target = str(target or "地图中心")
        self.opening_autopilot_target_normalized = target_normalized
        self.autopilot_retry_pending = False
        self._autopilot_hud_samples.clear()
        self.last_movement_command = None
        self.last_movement_reason = (
            f"游戏自动航行已设定至{self.opening_autopilot_target}"
        )
        self._last_movement_mode = "autopilot_route"

    def enable_generic_center_route(self, reason: str = ""):
        """Fall back from unreliable tactical-map navigation without stopping."""
        self.opening_autopilot_active = False
        self.opening_autopilot_target_normalized = None
        self.generic_center_route_active = True
        self._last_movement_mode = None
        self.last_movement_reason = reason or "通用驾驶接管，驶向地图中央"

    def request_autopilot_retry(self, reason: str = ""):
        """Hold Q/E while the lifecycle retries the native map route."""
        self.opening_autopilot_active = False
        self.generic_center_route_active = False
        self.autopilot_retry_pending = True
        self.last_movement_command = None
        self._last_movement_mode = "autopilot_retry"
        self.last_movement_reason = reason or "原生自动航行失效，正在重新设置航点"

    def _apply_map_center_objective(self, analysis: BattleAnalysis):
        analysis.capture_point_bearing = analysis.map_center_bearing
        analysis.capture_point_distance_km = analysis.map_center_distance_km
        arrival_distance = float(
            self.strategy.get("capture_arrival_distance_km", 4.5)
        )
        arrived = bool(
            analysis.map_center_distance_km is not None
            and analysis.map_center_distance_km <= arrival_distance
        )
        analysis.inside_capture_point = arrived
        analysis.route_phase = "station" if arrived else "transit"
        analysis.route_progress = 1.0 if arrived else analysis.route_progress
        analysis.route_waypoint = 2 if arrived else 0
        analysis.route_arrived = arrived
        analysis.navigation_target_normalized = (0.5, 0.5)
        analysis.navigation_source = "minimap_center"

    def _apply_generic_objective(self, analysis: BattleAnalysis):
        """Use map centre as the stable default after native navigation.

        Capture circles are useful telemetry, but their OCR/Hough result is
        not reliable enough to own the helm.  They remain visible in the Web
        radar while the fallback controller always heads toward map centre.
        """
        self._apply_map_center_objective(analysis)

    def _reassert_full_speed(self):
        reassert = getattr(self.gamepad, "reassert_full_speed", None)
        if reassert is not None:
            reassert()
            return
        full_speed = getattr(self.gamepad, "full_speed", None)
        if full_speed is not None:
            full_speed()

    def _movement_feedback_update(self, now, position, throttle):
        """Retry feedback faults locally before escalating to human takeover."""
        try:
            feedback = self.feedback.update(now, position, throttle)
        except SafetyFault as error:
            # The white player arrow can disappear for a few captures while
            # the tactical overlay fades, an island label overlaps it, or a
            # frame arrives mid-transition.  That is a vision gap, not proof
            # that the ship stopped.  Keep the last safe straight/full-speed
            # command and wait for the next minimap frame; crucially, do not
            # turn without a current minimap pose and never end a whole round
            # just because one detector is temporarily unavailable.
            if "无法识别玩家小地图位置" in str(error):
                self.feedback.reset()
                self.movement_verified = False
                self.enable_generic_center_route(
                    "小地图暂未定位舰船，保持当前安全航向与全速，等待实时小地图恢复"
                )
                self._reassert_full_speed()
                logger.warning(
                    "[SYSTEM] 小地图暂未定位白箭头；保持当前航向，不打舵，等待下一帧识别"
                )
                return None
            self.movement_feedback_failures += 1
            if self.movement_feedback_failures > self.max_movement_feedback_retries:
                raise SafetyFault(
                    "通用航行连续反馈失败，重试后仍无法确认舰船移动: "
                    f"{error}"
                ) from error
            self.feedback.reset()
            self.movement_verified = False
            self.enable_generic_center_route(
                "航行反馈失败，通用驾驶向地图中央接管；"
                f"正在重试 {self.movement_feedback_failures}/"
                f"{self.max_movement_feedback_retries}"
            )
            self._reassert_full_speed()
            logger.warning(
                "[SYSTEM] 航行反馈失败 (%s/%s): %s；重新识别并向地图中央接管",
                self.movement_feedback_failures,
                self.max_movement_feedback_retries,
                error,
            )
            return None
        self.movement_verified = feedback.verified
        if feedback.verified:
            self.movement_feedback_failures = 0
        return feedback

    def _latency_compensated_rudder(
        self,
        desired: float,
        now: float,
        *,
        safety_override: bool = False,
    ) -> float:
        """Keep Q/E decisions long enough for a battleship's rudder to react."""
        desired = max(-1.0, min(float(desired), 1.0))

        def direction(value):
            return 0 if abs(value) < 0.10 else (1 if value > 0 else -1)

        current_direction = direction(self._last_applied_rudder)
        desired_direction = direction(desired)
        hold_seconds = float(
            self.strategy.get("rudder_minimum_hold_seconds", 1.1)
        )
        maximum_hold_seconds = max(
            hold_seconds,
            min(
                float(self.strategy.get("rudder_maximum_hold_seconds", 4.5)),
                4.8,
            ),
        )
        # A single Q/E order may not stay active long enough to make the
        # ship circle.  Always release to neutral before another correction;
        # the next minimap frame then has to justify a new steering order.
        if (
            current_direction
            and self._rudder_commanded_at > 0
            and now - self._rudder_commanded_at >= maximum_hold_seconds
        ):
            self._last_applied_rudder = 0.0
            self._rudder_commanded_at = 0.0
            self._rudder_release_until = now + 0.65
            return 0.0
        if now < self._rudder_release_until:
            return 0.0
        changing_direction = desired_direction != current_direction
        if (
            changing_direction
            and not safety_override
            and self._rudder_commanded_at > 0
            and now - self._rudder_commanded_at < hold_seconds
        ):
            return self._last_applied_rudder
        # Never switch directly from Q to E (or vice versa), including island
        # avoidance. Battleships react late; an immediate opposite command
        # makes successive minimap frames alternate and eventually circle.
        if (
            current_direction
            and desired_direction
            and desired_direction != current_direction
        ):
            self._last_applied_rudder = 0.0
            self._rudder_commanded_at = 0.0
            self._rudder_release_until = now + 0.65
            return 0.0
        if changing_direction:
            self._rudder_commanded_at = now
        self._last_applied_rudder = desired
        return desired

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
        # While a battle controller is active, the broad port detector is not
        # authoritative. A live HUD can resemble the port carousel when the
        # minimap/consumables are dense. Require a positive result surface to
        # end the battle so a false PORT frame can never launch ship selection.
        battle_hud_detector = getattr(self.vision, "_has_battle_hud", None)
        battle_hud_visible = bool(
            callable(battle_hud_detector) and battle_hud_detector(image)
        )
        if state == ScreenState.PORT and battle_hud_visible:
            state = ScreenState.BATTLE
        raw_ended = state == ScreenState.RESULTS
        if raw_ended:
            self._ended_state_streak += 1
        else:
            self._ended_state_streak = 0
        # A single result/port-shaped frame can be a transition animation or
        # a broad colour match.  End combat only after the lifecycle evidence
        # is stable on two consecutive captures; until then the existing
        # throttle/rudder telegraph is left untouched.
        analysis.ended = self._ended_state_streak >= 2
        analysis.in_battle = state == ScreenState.BATTLE

        if not analysis.in_battle:
            self._island_samples.clear()
            self.viewport_target_tracker.reset()
            self.distance_filter.reset()
        if analysis.ended:
            return analysis

        if analysis.in_battle:
            autopilot_detector = getattr(self.vision, "is_autopilot_enabled", None)
            analysis.autopilot_enabled = bool(
                autopilot_detector is not None and autopilot_detector(image)
            )
            rudder_detector = getattr(self.vision, "detect_rudder_indicator", None)
            if rudder_detector is not None:
                analysis.rudder_indicator = str(
                    rudder_detector(image) or "neutral"
                )
            minimap = self.vision.find_minimap(image)
            minimap_enemies = []
            minimap_zones = []
            island_sample = None
            if minimap is not None:
                minimap_enemies, torpedoes_seen = (
                    self.vision.analyze_minimap(minimap)
                )
                island_outline_finder = getattr(
                    self.vision, "find_minimap_island_outlines", None
                )
                if island_outline_finder is not None and not self._battle_map_islands:
                    detected_islands = island_outline_finder(minimap)
                    confirmed_islands = self._confirmed_static_layer(
                        self._island_layer_candidates,
                        self._island_layer_signature(detected_islands),
                        detected_islands,
                    )
                    if confirmed_islands:
                        self._battle_map_islands = list(confirmed_islands)
                        logger.info(
                            "连续三帧一致，已锁定本局小地图山体图层: %s 个轮廓",
                            len(self._battle_map_islands),
                        )
                analysis.minimap_islands = list(self._battle_map_islands)
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
                    analysis.minimap_player_normalized = (
                        player[0] / max(minimap.shape[1], 1),
                        player[1] / max(minimap.shape[0], 1),
                    )
                    course_heading = self.course_heading_filter.update(player)
                    navigation_pose = (
                        pose
                        if course_heading is None
                        else replace(pose, heading=course_heading)
                    )
                    analysis.minimap_heading = tuple(navigation_pose.heading)
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
                    # Every control frame derives the objective from the
                    # visible A/B/C/D circles, not from the ship's concentric
                    # gun/concealment rings or the main camera.  A neutral
                    # white or hostile red point wins over a friendly green
                    # point, then the nearest eligible point is selected.
                    zone_finder = getattr(self.vision, "find_capture_zones", None)
                    if zone_finder is not None:
                        detected_zones = zone_finder(minimap, player=player)
                    else:
                        detected_zones = []
                    if self._battle_capture_zones:
                        self._battle_capture_zones = self._refresh_zone_states(
                            self._battle_capture_zones,
                            detected_zones,
                        )
                        minimap_zones = list(self._battle_capture_zones)
                    elif detected_zones:
                        confirmed_zones = self._confirmed_static_layer(
                            self._zone_layer_candidates,
                            self._zone_layer_signature(
                                detected_zones, minimap.shape
                            ),
                            detected_zones,
                        )
                        if confirmed_zones:
                            self._battle_capture_zones = list(confirmed_zones)
                            logger.info(
                                "连续三帧一致，已锁定本局占领点图层: %s 个点位",
                                len(self._battle_capture_zones),
                            )
                        minimap_zones = list(self._battle_capture_zones)
                    else:
                        self._zone_layer_candidates.clear()
                        minimap_zones = []
                    analysis.capture_zones = [
                        {
                            "label": zone.label,
                            "state": getattr(zone, "state", "unknown"),
                            "position": [
                                zone.center[0] / max(minimap.shape[1], 1),
                                zone.center[1] / max(minimap.shape[0], 1),
                            ],
                            "radius": zone.radius
                            / max(min(minimap.shape[:2]), 1),
                        }
                        for zone in minimap_zones
                    ]
                    select_zone = getattr(
                        self.vision,
                        "select_navigation_capture_zone",
                        None,
                    )
                    if select_zone is not None:
                        detected_zone = select_zone(minimap_zones, player)
                    else:
                        find_nearest = getattr(
                            self.vision, "find_nearest_capture_zone", None
                        )
                        detected_zone = (
                            find_nearest(minimap, player)
                            if find_nearest is not None
                            else self.vision.find_central_capture_zone(minimap)
                        )
                    if detected_zone is not None:
                        self.route_planner.observe_zone(
                            detected_zone,
                            minimap.shape,
                            # The point geometry is frozen for this match.
                            # Ownership can change visually, but must not make
                            # the route jump between circles every frame.
                            allow_retarget=False,
                        )
                    self._capture_zone = self.route_planner.zone
                    if self._capture_zone is not None:
                        analysis.capture_zone_center_normalized = (
                            self._capture_zone.center[0]
                            / max(minimap.shape[1], 1),
                            self._capture_zone.center[1]
                            / max(minimap.shape[0], 1),
                        )
                        analysis.capture_zone_radius_normalized = (
                            self._capture_zone.radius
                            / max(min(minimap.shape[:2]), 1)
                        )
                        analysis.capture_zone_label = getattr(
                            self._capture_zone, "label", ""
                        )
                        capture_pixels = math.dist(
                            player, self._capture_zone.center
                        )
                        analysis.capture_point_distance_km = (
                            self.vision.minimap_pixels_to_km(
                                minimap, capture_pixels
                            )
                        )
                        analysis.inside_capture_point = (
                            capture_pixels <= self._capture_zone.radius * 0.96
                        )
                        route = self.route_planner.update(
                            player,
                            inside_zone=analysis.inside_capture_point,
                        )
                        route_target = route.target or self._capture_zone.center
                        analysis.navigation_target_normalized = (
                            route_target[0] / max(minimap.shape[1], 1),
                            route_target[1] / max(minimap.shape[0], 1),
                        )
                        analysis.navigation_source = "minimap_capture_zone"
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
                    if self.generic_center_route_active:
                        self._apply_generic_objective(analysis)
                    elif (
                        self.opening_autopilot_active
                        and self.opening_autopilot_target_normalized is not None
                    ):
                        analysis.navigation_target_normalized = (
                            self.opening_autopilot_target_normalized
                        )
                        analysis.navigation_source = "native_autopilot"
                    if (
                        not self.opening_autopilot_active
                        and analysis.navigation_target_normalized is not None
                        and self._battle_map_islands
                    ):
                        final_target = (
                            analysis.navigation_target_normalized[0]
                            * minimap.shape[1],
                            analysis.navigation_target_normalized[1]
                            * minimap.shape[0],
                        )
                        waypoint_planner = getattr(
                            self.vision, "plan_island_aware_waypoint", None
                        )
                        if waypoint_planner is not None:
                            safe_target = waypoint_planner(
                                minimap.shape,
                                player,
                                final_target,
                                self._battle_map_islands,
                            )
                            if safe_target is not None:
                                analysis.capture_point_bearing = (
                                    self.vision.relative_bearing(
                                        navigation_pose, safe_target
                                    )
                                )
                                if math.dist(safe_target, final_target) > 2.0:
                                    analysis.navigation_target_normalized = (
                                        safe_target[0]
                                        / max(minimap.shape[1], 1),
                                        safe_target[1]
                                        / max(minimap.shape[0], 1),
                                    )
                                    analysis.navigation_source = (
                                        "minimap_island_waypoint"
                                    )
                island_risk = None
                if pose is not None and course_heading is not None:
                    try:
                        island_risk = self.vision.find_island_risk(
                            minimap,
                            navigation_pose,
                            island_outlines=self._battle_map_islands,
                        )
                    except TypeError:
                        # Compatibility with test/custom Vision adapters.
                        island_risk = self.vision.find_island_risk(
                            minimap, navigation_pose
                        )
                if island_risk is not None:
                    island_sample = (
                        island_risk.distance,
                        island_risk.avoidance_rudder,
                    )
                if pose is not None and minimap_enemies:
                    navigation_enemy = select_forward_navigation_enemy(
                        self.vision,
                        navigation_pose,
                        minimap_enemies,
                    )
                    if navigation_enemy is not None:
                        selected_enemy, minimap_pixels, enemy_bearing = (
                            navigation_enemy
                        )
                        diagonal = math.hypot(
                            minimap.shape[1], minimap.shape[0]
                        )
                        analysis.minimap_distance = minimap_pixels / max(
                            diagonal, 1
                        )
                        analysis.minimap_distance_km = (
                            self.vision.minimap_pixels_to_km(
                                minimap, minimap_pixels
                            )
                        )
                        analysis.minimap_target_bearing = enemy_bearing
                        analysis.nearest_enemy_normalized = (
                            selected_enemy[0] / max(minimap.shape[1], 1),
                            selected_enemy[1] / max(minimap.shape[0], 1),
                        )
                analysis.minimap_contacts = [
                    {
                        "position": [
                            enemy[0] / max(minimap.shape[1], 1),
                            enemy[1] / max(minimap.shape[0], 1),
                        ],
                        "kind": "enemy",
                    }
                    for enemy in minimap_enemies
                ]

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
            if (
                island_sample is not None
                and island_sample[0] <= self.movement.island_emergency_distance
                and abs(island_sample[1]) >= 0.2
            ):
                stable_island = island_sample
            if stable_island is not None:
                (
                    analysis.island_distance,
                    analysis.island_avoidance_rudder,
                ) = stable_island
                self._island_avoidance_rudder = stable_island[1]
                self._island_avoidance_until = observed_at + float(
                    # Never carry a one-time island observation into a long
                    # turn.  Every Q/E correction is capped below five
                    # seconds and must be renewed from fresh minimap terrain.
                    self.strategy.get("island_turn_commit_seconds", 3.8)
                )
            elif observed_at < self._island_avoidance_until:
                # A battleship reacts several seconds after Q/E input. Keep
                # the chosen escape turn through intermittent vision gaps so
                # route steering cannot immediately cancel it.
                analysis.island_distance = (
                    self.movement.island_warning_distance * 0.92
                )
                analysis.island_avoidance_rudder = (
                    self._island_avoidance_rudder
                )

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

        health = None
        health_reader = getattr(self.vision, "read_health_fraction", None)
        if health_reader is not None and self.tick % 3 == 0:
            try:
                health = health_reader(
                    image,
                    getattr(self.distance_reader, "backend", None),
                )
            except Exception:
                logger.debug("生命值 OCR 本帧不可用", exc_info=True)
        # Health is calculated exclusively from the numeric ``current/max``
        # readout.  Never estimate it from the padded colour bar: on this HUD a
        # visually full bar occupies only about 88.8% of the old crop.
        if health is not None:
            health = max(0.0, min(float(health), 1.0))
            self._cached_health = health
            self._health_ocr_valid = True
        self._fire_samples.append(bool(self.vision.is_on_fire(image)))
        analysis.on_fire = (
            len(self._fire_samples) == self._fire_samples.maxlen
            and all(self._fire_samples)
        )
        flooding_detector = getattr(self.vision, "is_flooding", None)
        self._flood_samples.append(
            bool(flooding_detector is not None and flooding_detector(image))
        )
        analysis.flooding = (
            len(self._flood_samples) == self._flood_samples.maxlen
            and all(self._flood_samples)
        )
        if self._health_ocr_valid and self._cached_health <= 0.0:
            # Sunk ships can retain orange wreck/HP decoration in the status
            # ROIs. Death is authoritative and suppresses hazard reporting as
            # well as every input command.
            analysis.on_fire = False
            analysis.flooding = False
        if self.tick % 3 == 0:
            self._cached_reload = self.vision.reload_ready(
                self.vision.find_reload_bar(image)
            )
        analysis.health = self._cached_health
        analysis.health_recognized = self._health_ocr_valid
        speed_reader = getattr(self.vision, "read_speed_knots", None)
        if speed_reader is not None and self.tick % 3 == 0:
            try:
                speed_knots = speed_reader(
                    image,
                    getattr(self.distance_reader, "backend", None),
                )
                if speed_knots is not None:
                    self._cached_speed_knots = float(speed_knots)
            except Exception:
                logger.debug("航速 OCR 本帧不可用", exc_info=True)
        analysis.speed_knots = self._cached_speed_knots
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
        """Accept three matching terrain observations from the latest five."""
        required = 3
        if len(self._island_samples) < required:
            return None
        samples = [sample for sample in self._island_samples if sample is not None]
        if len(samples) < required:
            return None
        samples = samples[-required:]
        distances = [sample[0] for sample in samples]
        if max(distances) - min(distances) > 0.060:
            return None
        sides = [sample[1] for sample in samples if abs(sample[1]) >= 0.5]
        positive = sum(side > 0 for side in sides)
        negative = sum(side < 0 for side in sides)
        if max(positive, negative) < 2:
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
        if analysis.health_recognized and analysis.health <= 0.0:
            # The ship is already destroyed. Do not release, reassert or alter
            # any control: simply observe until a genuine result page appears.
            self.last_movement_command = None
            self.last_movement_reason = "舰船生命值为0，停止下发指令并等待结算画面"
            self._last_movement_mode = "awaiting_results"
            self._finish_tick(analysis)
            return "waiting"
        if analysis.ended:
            # Screen classification is intentionally tentative here.  Keep
            # the existing telegraph/rudder until the lifecycle confirms the
            # result page on consecutive frames; a single false result frame
            # must not stop an otherwise active battle.
            pause = getattr(self.gamepad, "pause_automation", None)
            if pause is not None:
                pause()
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

    def mark_manual_pause(self):
        """Expose an already-observed keyboard/Web pause without input."""
        was_active = self._manual_intervention_active
        was_latched = self._manual_intervention_latched
        web_paused = bool(getattr(self.intervention, "web_paused", False))
        latched = bool(getattr(self.intervention, "latched", False))
        pause_source = (
            "网页手动暂停" if web_paused else "用户键盘介入"
        )
        self.last_movement_reason = (
            f"{pause_source}：停止截图、切窗和全部游戏指令；"
            + (
                "等待网页点击继续"
                if latched
                else (
                    f"静默 {float(getattr(self.intervention, 'pause_seconds', 5.0)):.0f} "
                    "秒后自动恢复"
                )
            )
        )
        self._manual_intervention_active = True
        self._manual_intervention_latched = latched
        self._last_movement_mode = "manual_pause"
        if not was_active:
            logger.info(
                "[USER] %s，立即冻结截图、切窗及全部游戏指令；%s",
                pause_source,
                "等待网页继续"
                if web_paused
                else (
                    f"连续 {float(getattr(self.intervention, 'pause_seconds', 5.0)):.0f} "
                    "秒无新键盘操作后自动恢复，"
                    f"持续操作满 {float(getattr(self.intervention, 'latch_seconds', 20.0)):.0f} "
                    "秒将锁定暂停"
                ),
            )
        if latched and not was_latched and not web_paused:
            logger.warning(
                "[USER] 持续键盘介入达到 %.0f 秒，已转为永久暂停；等待网页点击继续",
                float(getattr(self.intervention, "latch_seconds", 20.0)),
            )

    def _execute_rules(self, analysis: BattleAnalysis, now: float):
        if self.intervention.poll(self.gamepad, now):
            # Freeze command generation only. Do not release rudder or change
            # the latched engine telegraph: the ship keeps executing the last
            # accepted command while the player interacts.
            self.mark_manual_pause()
            return

        if self._manual_intervention_active:
            source = (
                "网页点击继续"
                if getattr(self.intervention, "resumed_from_web", False)
                else (
                    f"连续 {float(getattr(self.intervention, 'pause_seconds', 5.0)):.0f} "
                    "秒无新的键盘操作"
                )
            )
            logger.info("[SYSTEM] %s，重新识别当前战斗状态并恢复控制", source)
            if not self.movement_verified:
                self.feedback.reset()
            self._manual_intervention_active = False
            self._manual_intervention_latched = False
            self._last_movement_mode = None

        if analysis.autopilot_enabled and not self.opening_autopilot_active:
            # Recover from a stale internal flag without touching the keyboard.
            # The live green HUD is authoritative over any cached workflow state.
            self.opening_autopilot_active = True
            self.generic_center_route_active = False
            self.opening_autopilot_target = (
                self.opening_autopilot_target or "当前游戏航点"
            )
            self._autopilot_hud_samples.clear()
            logger.info("[SYSTEM] 检测到游戏自动航行仍开启，恢复原生航线互锁")

        # Survival consumables are independent from steering.  They must be
        # evaluated before the native-autopilot branch returns; otherwise R/T
        # could remain unused for the whole opening route.
        if not self._execute_survival_consumables(analysis, now):
            return

        if self.opening_autopilot_active:
            self._autopilot_hud_samples.append(bool(analysis.autopilot_enabled))
            if (
                not analysis.autopilot_enabled
                and len(self._autopilot_hud_samples) >= 3
                and not any(self._autopilot_hud_samples)
            ):
                self.feedback.reset()
                self.movement_verified = False
                arrived = bool(
                    self.opening_autopilot_target_normalized is not None
                    and analysis.minimap_player_normalized is not None
                    and math.dist(
                        self.opening_autopilot_target_normalized,
                        analysis.minimap_player_normalized,
                    )
                    <= 0.085
                )
                if arrived:
                    self.enable_generic_center_route(
                        "游戏自动航行已抵达敌方远端航点，通用驾驶按小地图接管"
                    )
                    self._apply_generic_objective(analysis)
                    logger.info(
                        "[SYSTEM] 自动航行已抵达航点；本帧不打舵，下一帧由小地图Q/E接管"
                    )
                else:
                    self.request_autopilot_retry(
                        "原生自动航行在抵达前失效，保持当前操舵并重试地图航点"
                    )
                    logger.warning(
                        "[SYSTEM] 自动航行在抵达前失效；暂停Q/E，申请三次敌方偏移航点重试"
                    )
                return
            # Native autopilot is exclusive.  Enemy distance, island risk and
            # all other combat rules are observation-only until the game HUD
            # itself confirms that native navigation has finished.
            self.last_movement_command = None
            self.last_movement_reason = (
                f"游戏自动航行至{self.opening_autopilot_target}；"
                "绿色自动航行互锁中，禁止Q/E，等待自动航行自然结束"
            )
            if self._last_movement_mode != "autopilot_route":
                logger.info("[SYSTEM] %s", self.last_movement_reason)
            self._last_movement_mode = "autopilot_route"
            if not analysis.autopilot_enabled:
                return
            try:
                feedback = self.feedback.update(
                    now,
                    analysis.player_position,
                    1.0,
                )
            except SafetyFault as error:
                # Lack of a pose is not authority to cancel native navigation.
                # Keep the game route and wait for the next reliable minimap
                # frame; no Q/E or throttle key is sent here.
                self.feedback.reset()
                self.movement_verified = False
                logger.warning(
                    "[SYSTEM] 自动航行反馈暂不可用: %s；保持自动航行互锁，不打舵",
                    error,
                )
                return
            self.movement_verified = feedback.verified
            return

        enemy_count = analysis.minimap_enemy_count
        command = self.movement.plan(
            SecondaryMovementInput(
                elapsed=max(0.0, now - self.battle_start_time),
                health=analysis.health,
                visible_target=bool(analysis.minimap_enemy_count),
                target_offset_x=None,
                target_distance_km=None,
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
        safety_override = command.mode in {
            MovementMode.AVOID_ISLAND,
            MovementMode.EVADE,
            MovementMode.SEPARATE,
        }
        compensated_rudder = self._latency_compensated_rudder(
            command.rudder,
            now,
            safety_override=safety_override,
        )
        if compensated_rudder != command.rudder:
            command = replace(command, rudder=compensated_rudder)
        self.last_movement_command = command
        self.last_movement_reason = command.reason
        if safety_override:
            self.stuck_recovery.cancel()
            recovery = None
        else:
            recovery = self.stuck_recovery.update(
                now,
                analysis.player_position,
                command.throttle,
                escape_rudder=analysis.island_avoidance_rudder,
            )
        if recovery is not None:
            # A keyboard event can arrive after the frame-level pause check but
            # before movement planning finishes.  Re-check at the exact input
            # boundary so even the recovery branch cannot leak one command.
            if self.intervention.poll(self.gamepad, time.monotonic()):
                self.mark_manual_pause()
                return
            self.gamepad.set_movement(recovery.throttle, recovery.rudder)
            self._movement_feedback_update(
                now, analysis.player_position, recovery.throttle
            )
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

        if self.intervention.poll(self.gamepad, time.monotonic()):
            self.mark_manual_pause()
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
        self._movement_feedback_update(
            now, analysis.player_position, command.throttle
        )
        if command.mode != self._last_movement_mode:
            logger.info(
                "[SYSTEM] 移动状态: %s | 推力=%.2f 舵角=%.2f | %s",
                command.mode.value,
                command.throttle,
                command.rudder,
                command.reason,
            )
            self._last_movement_mode = command.mode

        # Check again after the movement dispatch. A keyboard press that
        # happened during this frame must block lock/fire/R/T/smoke commands;
        # it must not wait for the next 300 ms control frame.
        if self.intervention.poll(self.gamepad, time.monotonic()):
            self.mark_manual_pause()
            return

        # Target-lock input is intentionally deferred until movement is proven;
        # the opening priority is to get underway and reach the capture point.
        if analysis.visible_target and self.movement_verified:
            self._engage_visible_enemy(analysis, now)

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

    def _execute_survival_consumables(
        self, analysis: BattleAnalysis, now: float
    ) -> bool:
        """Dispatch R/T/smoke even while native autopilot owns steering."""
        damage_control_cooldown = float(
            self.strategy.get("damage_control_cooldown_seconds", 80.0)
        )
        if (
            (analysis.on_fire or analysis.flooding)
            and now - self.last_damage_control >= damage_control_cooldown
            and hasattr(self.gamepad, "damage_control")
        ):
            if self.intervention.poll(self.gamepad, time.monotonic()):
                self.mark_manual_pause()
                return False
            self.gamepad.damage_control()
            self.last_damage_control = now
            logger.info(
                "检测到%s，使用 R 损害管制",
                "着火/漏水"
                if analysis.on_fire and analysis.flooding
                else "着火"
                if analysis.on_fire
                else "漏水",
            )

        heal_cooldown = float(
            self.strategy.get("heal_cooldown_seconds", 80.0)
        )
        heal_max_uses = int(self.strategy.get("heal_max_uses", 3))
        heal_loss_step = max(
            0.05,
            min(0.50, float(self.strategy.get("heal_loss_step", 0.20))),
        )
        next_heal_threshold = max(
            0.0, 1.0 - heal_loss_step * (self.heal_used + 1)
        )
        if (
            analysis.health_recognized
            and 0 < analysis.health <= next_heal_threshold + 0.005
            and self.heal_used < heal_max_uses
            and now - self.last_heal >= heal_cooldown
            and hasattr(self.gamepad, "heal")
        ):
            if self.intervention.poll(self.gamepad, time.monotonic()):
                self.mark_manual_pause()
                return False
            self.gamepad.heal()
            self.last_heal = now
            self.heal_used += 1
            logger.info(
                "数值血量 %.0f%% 已跨过 %.0f%% 档位，使用 T 维修小组 (%s/%s)",
                analysis.health * 100,
                next_heal_threshold * 100,
                self.heal_used,
                heal_max_uses,
            )

        smoke_threshold = float(self.strategy.get("smoke_threshold", 0.5))
        if (
            self.ship.get("has_smoke")
            and self.strategy.get("auto_smoke", False)
            and not self.smoke_used
            and analysis.health_recognized
            and 0 < analysis.health < smoke_threshold
            and hasattr(self.gamepad, "smoke")
        ):
            if self.intervention.poll(self.gamepad, time.monotonic()):
                self.mark_manual_pause()
                return False
            self.gamepad.smoke()
            self.smoke_used = True
            logger.info(
                "那不勒斯使用 4 号位烟雾 (HP: %.0f%%)",
                analysis.health * 100,
            )
        return True

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
            health_recognized=analysis.health_recognized,
            on_fire=analysis.on_fire,
            flooding=analysis.flooding,
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
            minimap_heading=analysis.minimap_heading,
            autopilot_enabled=analysis.autopilot_enabled,
            rudder_indicator=analysis.rudder_indicator,
            movement_mode=self.current_movement_mode,
            throttle=None if command is None else command.throttle,
            rudder=None if command is None else command.rudder,
            movement_reason=self.last_movement_reason,
            movement_verified=self.movement_verified,
        )
        if self.tick % 4 == 0:
            logger.info(
                "[%03d] HP:%s 装填:%s 敌人:%s 距离:%s 鱼雷:%s 反馈:%s",
                self.tick,
                f"{analysis.health * 100:.0f}%"
                if analysis.health_recognized
                else "识别中",
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
    def manual_intervention_active(self) -> bool:
        return bool(
            self._manual_intervention_active
            or self.intervention.command_generation_paused()
        )

    @property
    def manual_intervention_seconds(self) -> float:
        return self.intervention.continuous_seconds()

    @property
    def manual_intervention_remaining_seconds(self) -> float:
        return float(getattr(self.intervention, "remaining_seconds", 0.0))

    @property
    def damage_control_ready(self) -> bool:
        cooldown = float(
            self.strategy.get("damage_control_cooldown_seconds", 80.0)
        )
        return time.monotonic() - self.last_damage_control >= cooldown

    @property
    def heal_ready(self) -> bool:
        cooldown = float(self.strategy.get("heal_cooldown_seconds", 80.0))
        max_uses = int(self.strategy.get("heal_max_uses", 3))
        return (
            self.heal_used < max_uses
            and time.monotonic() - self.last_heal >= cooldown
        )

    def stop(self, *, release_input: bool = True):
        """Close runtime resources, optionally releasing game controls.

        ``release_input=False`` is reserved for a destroyed game HWND.  It
        avoids sending key-up events into whichever application owns focus
        after the game has closed while still flushing OCR/event resources.
        """
        if release_input:
            self.gamepad.stop()
        self.events.publish("runtime.stopped", tick=self.tick)
        if self.distance_ocr_service is not None:
            self.distance_ocr_service.close()
        if self._event_recorder is not None:
            self._event_recorder.close()
            self._event_recorder = None
