"""Shared run limits and status reporting for CLI and local control panel."""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any


logger = logging.getLogger("runtime")


@dataclass(frozen=True)
class RunLimits:
    max_rounds: int = 0
    duration_minutes: float = 0
    run_id: str = "standalone"
    state_file: Path | None = None
    stop_file: Path | None = None
    pause_file: Path | None = None
    quick_battle: bool = False
    close_game_when_done: bool = False

    @classmethod
    def from_env(cls) -> "RunLimits":
        max_rounds = max(0, int(os.environ.get("WOWS_MAX_ROUNDS", "0")))
        duration = max(0.0, float(os.environ.get("WOWS_DURATION_MINUTES", "0")))
        state = os.environ.get("WOWS_STATE_FILE", "").strip()
        stop = os.environ.get("WOWS_STOP_FILE", "").strip()
        pause = os.environ.get("WOWS_PAUSE_FILE", "").strip()
        quick = os.environ.get("WOWS_QUICK_BATTLE", "").strip().lower()
        close_game = os.environ.get(
            "WOWS_CLOSE_GAME_WHEN_DONE", ""
        ).strip().lower()
        return cls(
            max_rounds=max_rounds,
            duration_minutes=duration,
            run_id=os.environ.get("WOWS_RUN_ID", "standalone").strip()
            or "standalone",
            state_file=Path(state) if state else None,
            stop_file=Path(stop) if stop else None,
            pause_file=Path(pause) if pause else None,
            quick_battle=quick in {"1", "true", "yes", "on"},
            close_game_when_done=close_game in {"1", "true", "yes", "on"},
        )

    @property
    def duration_seconds(self) -> float:
        return self.duration_minutes * 60

    def reached(self, completed_rounds: int, started_at: float) -> bool:
        return self.schedule_reached(completed_rounds, started_at) or self.stop_requested()

    def schedule_reached(self, completed_rounds: int, started_at: float) -> bool:
        """Return whether the configured plan is complete.

        This is intentionally separate from an explicit stop request so a time
        limit can finish the active battle instead of terminating it halfway.
        """
        if self.max_rounds and completed_rounds >= self.max_rounds:
            return True
        if self.duration_seconds and time.monotonic() - started_at >= self.duration_seconds:
            return True
        return False

    def stop_requested(self) -> bool:
        return bool(self.stop_file and self.stop_file.exists())

    def pause_requested(self) -> bool:
        return bool(self.pause_file and self.pause_file.exists())


@dataclass
class RuntimeStatus:
    run_id: str
    state: str = "idle"
    message: str = "等待启动"
    ship: str = ""
    ship_display_name: str = ""
    mode: str = "asymmetric"
    completed_rounds: int = 0
    current_round: int = 0
    max_rounds: int = 0
    duration_minutes: float = 0
    quick_battle: bool = False
    close_game_when_done: bool = False
    started_at: float = 0
    updated_at: float = field(default_factory=time.time)
    error: str = ""
    safety_state: str = "idle"
    calibration_valid: bool = False
    movement_verified: bool = False
    frame_status: str = "unknown"
    capture_backend: str = "uninitialized"
    target_distance_km: float | None = None
    distance_source: str = "unknown"
    minimap_distance_km: float | None = None
    distance_confidence: float = 0.0
    target_track_id: str = ""
    ocr_status: str = "no_target"
    ocr_provider: str = "uninitialized"
    movement_mode: str = "idle"
    movement_reason: str = ""
    capture_point_distance_km: float | None = None
    inside_capture_point: bool = False
    route_phase: str = "unplanned"
    route_progress: float = 0.0
    route_waypoint: int = 0
    route_arrived: bool = False
    minimap_player: tuple[float, float] | None = None
    minimap_heading: tuple[float, float] | None = None
    navigation_target: tuple[float, float] | None = None
    capture_zone_center: tuple[float, float] | None = None
    capture_zone_radius: float | None = None
    capture_zone_label: str = ""
    nearest_enemy: tuple[float, float] | None = None
    minimap_enemy_count: int = 0
    minimap_contacts: list[dict[str, Any]] = field(default_factory=list)
    capture_zones: list[dict[str, Any]] = field(default_factory=list)
    minimap_islands: list[dict[str, Any]] = field(default_factory=list)
    minimap_snapshot: str = ""
    navigation_source: str = "unknown"
    autopilot_enabled: bool = False
    rudder_indicator: str = "neutral"
    commanded_rudder: float | None = None
    island_distance: float | None = None
    health_percent: float | None = None
    speed_knots: float | None = None
    on_fire: bool = False
    flooding: bool = False
    damage_control_ready: bool = False
    heal_ready: bool = False
    other_consumables_ready: bool = False
    elapsed_seconds: float = 0.0
    stop_after_current: bool = False
    rewards_status: str = "pending"
    rewards_round: int = 0
    last_rewards: dict[str, Any] = field(default_factory=dict)
    last_outcome: str = "unknown"
    manual_intervention_latched: bool = False
    manual_intervention_active: bool = False
    manual_intervention_seconds: float = 0.0
    manual_intervention_remaining_seconds: float = 0.0
    paused_by_user: bool = False


class RuntimeReporter:
    def __init__(
        self,
        limits: RunLimits,
        *,
        ship: str,
        mode: str,
        ship_display_name: str = "",
    ):
        self.limits = limits
        self.started_monotonic = time.monotonic()
        self.status = RuntimeStatus(
            run_id=limits.run_id,
            ship=ship,
            ship_display_name=ship_display_name,
            mode=mode,
            max_rounds=limits.max_rounds,
            duration_minutes=limits.duration_minutes,
            quick_battle=limits.quick_battle,
            close_game_when_done=limits.close_game_when_done,
            started_at=time.time(),
        )

    def update(self, state: str, message: str, **values: Any):
        self.status.state = state
        self.status.message = message
        self.status.updated_at = time.time()
        for key, value in values.items():
            if hasattr(self.status, key):
                setattr(self.status, key, value)
        self._write()

    def _write(self):
        self.status.elapsed_seconds = max(
            0.0, time.monotonic() - self.started_monotonic
        )
        destination = self.limits.state_file
        if destination is None:
            return
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(asdict(self.status), ensure_ascii=False, indent=2)
        with NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            stream.write(payload)
            temporary = Path(stream.name)
        for attempt in range(12):
            try:
                temporary.replace(destination)
                return
            except PermissionError:
                # Antivirus/indexers and a simultaneous control-panel read can
                # briefly hold the destination without delete sharing on
                # Windows. Status reporting must never terminate a battle.
                time.sleep(0.025 * (attempt + 1))
        logger.warning("运行状态文件暂时被占用，本次状态更新将在下一帧重试")
        temporary.unlink(missing_ok=True)
