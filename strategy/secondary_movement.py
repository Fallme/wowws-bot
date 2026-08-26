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
    REVERSE_RANGE = "reverse_range"
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
        capture_throttle: float = 0.36,
        enemy_steering_weight: float = 0.28,
        center_steering_gain: float = 1.35,
        straight_opening_seconds: float = 12.0,
        secondary_target_km: float = 10.0,
        secondary_inner_km: float = 7.0,
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
        self.capture_throttle = self._bounded(capture_throttle, 0.20, 0.60)
        self.enemy_steering_weight = self._bounded(
            enemy_steering_weight, 0.0, 0.45
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

    @staticmethod
    def _bounded(value: float, minimum: float, maximum: float) -> float:
        return max(minimum, min(float(value), maximum))

    @staticmethod
    def _clamp(value: float, limit: float = 1.0) -> float:
        return max(-limit, min(float(value), limit))

    def reset(self):
        """The station-first controller is intentionally stateless."""

    def _island_rudder(self, state: SecondaryMovementInput) -> float:
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
        enemy_outside_secondary: bool,
        inside_capture: bool,
    ) -> float:
        objective = self._objective_bearing(state)
        enemy = self._enemy_bearing(state) if enemy_outside_secondary else None
        if objective is None:
            bearing = enemy or 0.0
        elif enemy is None:
            bearing = objective
        else:
            enemy_weight = (
                min(self.enemy_steering_weight, 0.12)
                if inside_capture
                else self.enemy_steering_weight
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

    def plan(self, state: SecondaryMovementInput) -> MovementCommand:
        if state.elapsed < self.straight_opening_seconds:
            return MovementCommand(
                MovementMode.ROUTE_PLANNING,
                throttle=1.0,
                rudder=0.0,
                reason="已锁定中央点航线，开局直航建立真实航迹",
            )

        if (
            state.island_distance is not None
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

        if (
            state.island_distance is not None
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
                enemy_outside_secondary=False,
                inside_capture=False,
            )
            objective = self._objective_text(state)
            phase_text = {
                "departure": "离开出生点",
                "transit": "沿预设航线驶向中央点",
                "final_approach": "进入中央占领点",
            }.get(state.route_phase, "驶向中央占领点")
            return MovementCommand(
                MovementMode.ROUTE_TRANSIT,
                throttle=1.0,
                rudder=rudder,
                reason=f"{phase_text}，四档全速推进；当前目标{objective}",
            )

        enemy_outside = bool(
            distance is not None and distance > self.secondary_target_km + 0.8
        )
        rudder = self._route_rudder(
            state,
            enemy_outside_secondary=enemy_outside,
            inside_capture=state.inside_capture_point,
        )

        if distance is not None and distance < self.secondary_inner_km:
            enemy_bearing = self._enemy_bearing(state) or 0.0
            return MovementCommand(
                MovementMode.REVERSE_RANGE,
                throttle=-0.45,
                rudder=self._clamp(-enemy_bearing * 0.65, 0.42),
                reason=(
                    f"已到达点位，{distance_source}{distance:.1f}km过近；"
                    f"倒船将敌舰拉回约{self.secondary_target_km:.0f}km副炮距离"
                ),
            )

        if enemy_outside:
            return MovementCommand(
                MovementMode.APPROACH,
                throttle=0.72 if state.inside_capture_point else 1.0,
                rudder=rudder,
                reason=(
                    f"已到达点位，{distance_source}{distance:.1f}km超出理想副炮距离；"
                    f"向敌舰接近至约{self.secondary_target_km:.0f}km"
                ),
            )

        if distance is not None:
            return MovementCommand(
                MovementMode.BRAWL,
                throttle=self.capture_throttle if state.inside_capture_point else 0.62,
                rudder=rudder,
                reason=(
                    f"已到达点位，{distance_source}{distance:.1f}km在副炮有效区；"
                    f"维持约{self.secondary_target_km:.0f}km并兼顾留点"
                ),
            )

        return MovementCommand(
            MovementMode.CAPTURE,
            throttle=self.capture_throttle,
            rudder=rudder,
            reason="已到达中央点，敌距未确认，低速留点等待接敌",
        )
