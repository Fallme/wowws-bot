"""Station-first navigation for secondary-battery ships.

The ship keeps flank speed outside a capture zone and treats the central cap
as its persistent navigation objective. Enemy bearings only bias that route
while the nearest enemy is outside secondary range; they never cause an early
about-turn.  Navigation distance is derived from the white player marker and
red enemy markers on the minimap's 5 km grid.  Viewport OCR is deliberately
excluded because aircraft and friendly labels can be mistaken for the target.
"""

from dataclasses import dataclass
from enum import Enum


class MovementMode(str, Enum):
    ROUTE_PLANNING = "route_planning"
    ROUTE_TRANSIT = "route_transit"
    OPENING = "opening"
    SEARCH = "search"
    APPROACH = "approach"
    BRAWL = "hold_range"
    CAPTURE = "hold_capture"
    SEPARATE = "separate"
    AVOID_ISLAND = "avoid_island"
    DISENGAGE = "disengage"
    EVADE = "evade"


@dataclass(frozen=True)
class SecondaryMovementInput:
    elapsed: float
    health: float
    visible_target: bool
    target_offset_x: float | None = None
    target_distance_km: float | None = None
    minimap_distance: float | None = None
    minimap_distance_km: float | None = None
    minimap_target_bearing: float | None = None
    map_center_bearing: float | None = None
    map_center_distance_km: float | None = None
    capture_point_bearing: float | None = None
    capture_point_distance_km: float | None = None
    inside_capture_point: bool = False
    route_phase: str = "unplanned"
    route_arrived: bool = False
    enemy_count: int = 0
    torpedoes_incoming: bool = False
    island_distance: float | None = None
    island_avoidance_rudder: float | None = None
    kinematic_rudder: float | None = None
    kinematic_avoidance_required: bool = False
    kinematic_collision_time_seconds: float | None = None
    kinematic_clearance_km: float | None = None
    kinematic_target: str = ""


@dataclass(frozen=True)
class MovementCommand:
    mode: MovementMode
    throttle: float
    rudder: float
    reason: str


class SecondaryMovementController:
    """Drive into the central objective and keep secondary contact en route."""

    def __init__(
        self,
        *,
        preferred_side: int = 1,
        opening_seconds: float = 55.0,
        disengage_health: float = 0.28,
        secondary_enter_distance: float = 0.18,
        ideal_outer_distance: float = 0.15,
        ideal_inner_distance: float = 0.10,
        too_close_distance: float = 0.07,
        island_warning_distance: float = 0.04,
        island_emergency_distance: float = 0.02,
        secondary_range_km: float = 11.4,
        brake_start_km: float = 13.0,
        ideal_outer_km: float = 11.4,
        ideal_inner_km: float = 7.0,
        too_close_km: float = 4.5,
        brawl_map_distance: float | None = None,
        capture_throttle: float = 0.72,
        enemy_steering_weight: float = 0.84,
        center_steering_gain: float = 1.35,
        straight_opening_seconds: float = 12.0,
        secondary_target_km: float = 10.0,
        secondary_inner_km: float = 7.0,
        qe_speed_knots: float = 30.0,
        qe_rudder_shift_seconds: float = 15.0,
        qe_turning_radius_km: float = 1.0,
    ):
        # Legacy distance-band arguments remain accepted so existing ship
        # configs load cleanly. Station-first control deliberately does not
        # turn away merely because an enemy is close or health is low.
        self.preferred_side = 1 if preferred_side >= 0 else -1
        self.opening_seconds = max(0.0, float(opening_seconds))
        self.secondary_range_km = self._bounded(secondary_range_km, 2.0, 30.0)
        self.island_warning_distance = self._bounded(
            island_warning_distance, 0.025, 0.18
        )
        self.island_emergency_distance = self._bounded(
            island_emergency_distance, 0.012, self.island_warning_distance
        )
        self.capture_throttle = self._bounded(capture_throttle, 0.50, 0.85)
        self.enemy_steering_weight = self._bounded(
            enemy_steering_weight, 0.55, 0.90
        )
        self.center_steering_gain = self._bounded(
            center_steering_gain, 0.6, 2.0
        )
        self.straight_opening_seconds = self._bounded(
            straight_opening_seconds, 2.0, 12.0
        )
        self.secondary_target_km = self._bounded(
            min(secondary_target_km, self.secondary_range_km - 0.4), 4.0, 20.0
        )
        self.secondary_inner_km = self._bounded(
            min(secondary_inner_km, self.secondary_target_km - 0.5), 3.0, 15.0
        )
        self.qe_speed_knots = self._bounded(qe_speed_knots, 5.0, 60.0)
        self.qe_rudder_shift_seconds = self._bounded(
            qe_rudder_shift_seconds, 3.0, 45.0
        )
        self.qe_turning_radius_km = self._bounded(
            qe_turning_radius_km, 0.25, 3.0
        )
        # When the objective lies behind the current course, atan2 can jump
        # between +pi and -pi on adjacent minimap frames.  Keep one turn side
        # until the target is clearly ahead so Q/E cannot alternate into a
        # broad rear-half circle.
        self._objective_recovery_direction = 0

    @staticmethod
    def _bounded(value: float, minimum: float, maximum: float) -> float:
        return max(minimum, min(float(value), maximum))

    @staticmethod
    def _clamp(value: float, limit: float = 1.0) -> float:
        return max(-limit, min(float(value), limit))

    def reset(self):
        self._objective_recovery_direction = 0

    def _island_rudder(self, state: SecondaryMovementInput) -> float:
        if (
            state.kinematic_avoidance_required
            and state.kinematic_rudder is not None
        ):
            return self._clamp(state.kinematic_rudder)
        rudder = state.island_avoidance_rudder
        if rudder is None or abs(rudder) < 0.2:
            return 0.76 * self.preferred_side
        return 0.76 if rudder > 0 else -0.76

    def _evasion_rudder(self, state: SecondaryMovementInput) -> float:
        rudder = self._island_rudder(state)
        if abs(rudder) < 0.2:
            rudder = 0.76 * self.preferred_side
        return rudder

    @staticmethod
    def _enemy_bearing(state: SecondaryMovementInput) -> float | None:
        # Steering has exactly one source of truth: player/enemy markers on
        # the minimap. Viewport labels can belong to aircraft or allies.
        return state.minimap_target_bearing

    @classmethod
    def _forward_enemy_bearing(
        cls, state: SecondaryMovementInput
    ) -> float | None:
        """Return only a red contact that can be pursued without a U-turn."""
        bearing = cls._enemy_bearing(state)
        if bearing is None or abs(bearing) > 0.50:
            return None
        return bearing

    @staticmethod
    def _objective_bearing(state: SecondaryMovementInput) -> float | None:
        if state.capture_point_bearing is not None:
            return state.capture_point_bearing
        return state.map_center_bearing

    def _effective_distance(
        self, state: SecondaryMovementInput
    ) -> tuple[float | None, str]:
        if state.minimap_distance_km is not None:
            return state.minimap_distance_km, "小地图5km网格"
        # ``target_distance_km`` is retained in the input model for diagnostic
        # compatibility, but it is viewport OCR and must not steer the ship.
        return None, "未知"

    def _route_rudder(
        self,
        state: SecondaryMovementInput,
        *,
        pursue_enemy: bool,
        inside_capture: bool,
    ) -> float:
        objective = self._objective_bearing(state)
        objective_is_behind = bool(
            objective is not None and abs(objective) >= 0.52
        )
        if objective_is_behind and not inside_capture:
            if not self._objective_recovery_direction:
                if (
                    state.kinematic_rudder is not None
                    and abs(state.kinematic_rudder) >= 0.2
                ):
                    self._objective_recovery_direction = (
                        1 if state.kinematic_rudder > 0 else -1
                    )
                else:
                    self._objective_recovery_direction = (
                        1 if objective > 0 else -1
                    )
            # A half-rudder correction cannot recover a battleship already
            # steaming away from the objective.  Enemy markers are ignored
            # until this fixed-side U-turn brings the target ahead again.
            return 0.86 * self._objective_recovery_direction
        if (
            self._objective_recovery_direction
            and not inside_capture
            and objective is not None
            and abs(objective) > 0.18
        ):
            return 0.76 * self._objective_recovery_direction
        if objective is None or abs(objective) <= 0.18 or inside_capture:
            self._objective_recovery_direction = 0

        if state.kinematic_rudder is not None:
            return self._clamp(state.kinematic_rudder)

        # Outside the objective, a red contact may bias the helm only when the
        # fixed map objective is already in the forward cone.  This prevents a
        # contact in the spawn half from pulling the ship farther backward.
        enemy_allowed = bool(
            pursue_enemy
            and (
                inside_capture
                or objective is None
                or abs(objective) <= 0.28
            )
        )
        enemy = self._forward_enemy_bearing(state) if enemy_allowed else None
        if objective is None:
            bearing = enemy or 0.0
        elif enemy is None:
            bearing = objective
        else:
            # Once native autopilot has finished, a forward red contact owns
            # most of the course.  The central objective remains a weak anchor
            # against map-edge drift, but no rear contact may cause an about-turn.
            # A forward red contact should own the course once it is safe to
            # pursue.  The objective remains only as a guardrail against map
            # edge drift; the old 68% blend left secondary ships circling in
            # the rear half of the map instead of closing for damage.
            enemy_weight = min(
                0.90,
                self.enemy_steering_weight + (0.06 if inside_capture else 0.0),
            )
            bearing = objective * (1.0 - enemy_weight) + enemy * enemy_weight
        if abs(bearing) < 0.05:
            return 0.0
        # Normal navigation uses a broad, gradual arc. Hard rudder is reserved
        # for island/torpedo overrides so the ship does not look as if it has
        # decided to turn around immediately after spawning.
        limit = 0.38 if inside_capture else 0.46
        return self._clamp(bearing * self.center_steering_gain, limit)

    @staticmethod
    def _objective_text(state: SecondaryMovementInput) -> str:
        if state.capture_point_distance_km is not None:
            return f"中央占领点约{state.capture_point_distance_km:.1f}km"
        if state.map_center_distance_km is not None:
            return f"地图中心约{state.map_center_distance_km:.1f}km"
        return "当前航向"

    def _route_reason_prefix(self, state: SecondaryMovementInput) -> str:
        objective = self._objective_bearing(state)
        if (
            not state.inside_capture_point
            and objective is not None
            and (
                abs(objective) >= 0.52
                or self._objective_recovery_direction
            )
        ):
            return "检测到舰首背离中央点，锁定单侧硬舵回正"
        return {
            "departure": "离开出生点",
            "transit": "沿预设航线驶向中央点",
            "final_approach": "进入中央占领点",
        }.get(state.route_phase, "驶向中央占领点")

    def _kinematic_avoidance_command(
        self, state: SecondaryMovementInput
    ) -> MovementCommand:
        collision_seconds = state.kinematic_collision_time_seconds
        emergency = bool(
            collision_seconds is None or collision_seconds <= 30.0
        )
        clearance_text = (
            f"，预测净空约{state.kinematic_clearance_km:.1f}km"
            if state.kinematic_clearance_km is not None
            else ""
        )
        collision_text = (
            f"约{collision_seconds:.0f}秒后触岸"
            if collision_seconds is not None
            else "候选航迹均不安全"
        )
        return MovementCommand(
            MovementMode.AVOID_ISLAND,
            throttle=(
                0.55
                if emergency
                else (0.82 if state.inside_capture_point else 1.0)
            ),
            rudder=self._island_rudder(state),
            reason=(
                f"运动学预测{collision_text}，按{self.qe_speed_knots:.0f}节/"
                f"{self.qe_rudder_shift_seconds:.0f}秒舵效/"
                f"{self.qe_turning_radius_km:.1f}km转弯半径"
                f"选择安全航迹{clearance_text}"
                + (
                    f"；避山优先，安全后继续朝{state.kinematic_target}推进"
                    if state.kinematic_target
                    else ""
                )
            ),
        )

    def plan(self, state: SecondaryMovementInput) -> MovementCommand:
        if state.elapsed < self.straight_opening_seconds:
            return MovementCommand(
                MovementMode.ROUTE_PLANNING,
                throttle=1.0,
                rudder=0.0,
                reason="已锁定中央点航线，开局直航建立真实航迹",
            )

        kinematic_emergency = bool(
            state.kinematic_avoidance_required
            and (
                state.kinematic_collision_time_seconds is None
                or state.kinematic_collision_time_seconds <= 30.0
            )
        )
        if kinematic_emergency:
            return self._kinematic_avoidance_command(state)

        if (
            state.kinematic_rudder is None
            and state.island_distance is not None
            and state.island_distance <= self.island_emergency_distance
        ):
            return MovementCommand(
                MovementMode.AVOID_ISLAND,
                throttle=0.32,
                rudder=self._island_rudder(state),
                reason="连续确认前方近岛，低速向净空侧绕行；不凭视觉单帧倒车",
            )

        if state.torpedoes_incoming:
            return MovementCommand(
                MovementMode.EVADE,
                throttle=1.0,
                rudder=self._evasion_rudder(state),
                reason="发现来袭鱼雷，保持全速向可通航一侧规避",
            )

        if state.kinematic_avoidance_required:
            return self._kinematic_avoidance_command(state)

        if (
            state.kinematic_rudder is None
            and state.island_distance is not None
            and state.island_distance <= self.island_warning_distance
        ):
            return MovementCommand(
                MovementMode.AVOID_ISLAND,
                throttle=0.82 if state.inside_capture_point else 1.0,
                rudder=self._island_rudder(state),
                reason="预测当前航路会撞岛，保持推进并向净空水域绕行",
            )

        distance, distance_source = self._effective_distance(state)
        # Route-planner arrival is historical and can remain true after the
        # ship drifts out of a point.  Current minimap position is authoritative:
        # after the contacted enemy dies, resume full-speed travel until the
        # white player marker is actually back inside the objective.
        arrived = state.inside_capture_point

        # Until the central objective is reached, the route owns steering.
        # Enemies are observed for later contact but cannot pull the ship away
        # from its opening plan. Safety overrides above remain immediate.
        if not arrived:
            rudder = self._route_rudder(
                state,
                pursue_enemy=True,
                inside_capture=False,
            )
            objective = self._objective_text(state)
            phase_text = self._route_reason_prefix(state)
            target_suffix = (
                f"；运动学航迹朝{state.kinematic_target}收敛"
                if state.kinematic_rudder is not None
                and state.kinematic_target
                else ""
            )
            return MovementCommand(
                MovementMode.ROUTE_TRANSIT,
                throttle=1.0,
                rudder=rudder,
                reason=(
                    f"{phase_text}，四档全速推进；当前目标{objective}"
                    f"{target_suffix}"
                ),
            )

        enemy_outside = bool(
            distance is not None and distance > self.secondary_target_km + 0.8
        )
        rudder = self._route_rudder(
            state,
            pursue_enemy=True,
            inside_capture=state.inside_capture_point,
        )

        if distance is not None and distance < self.secondary_inner_km:
            return MovementCommand(
                MovementMode.BRAWL,
                throttle=0.62,
                rudder=rudder,
                reason=(
                    f"已到达点位，{distance_source}{distance:.1f}km过近；"
                    "保持向前低速通过，不倒船、不追逐身后目标"
                ),
            )

        if enemy_outside:
            return MovementCommand(
                MovementMode.APPROACH,
                throttle=1.0,
                rudder=rudder,
                reason=(
                    f"已到达点位，{distance_source}{distance:.1f}km超出理想副炮距离；"
                    f"向敌舰接近至约{self.secondary_target_km:.0f}km"
                ),
            )

        if distance is not None:
            return MovementCommand(
                MovementMode.BRAWL,
                throttle=0.82 if state.inside_capture_point else 0.88,
                rudder=rudder,
                reason=(
                    f"已到达点位，{distance_source}{distance:.1f}km在副炮有效区；"
                    f"维持约{self.secondary_target_km:.0f}km并兼顾留点"
                ),
            )

        if self._forward_enemy_bearing(state) is not None:
            return MovementCommand(
                MovementMode.APPROACH,
                throttle=1.0,
                rudder=rudder,
                reason="发现前方红点，敌距未确认，优先向敌舰方向推进接敌",
            )

        return MovementCommand(
            MovementMode.CAPTURE,
            throttle=self.capture_throttle,
            rudder=rudder,
            reason="已到达中央点，敌距未确认，保持推进搜索前方红点",
        )
