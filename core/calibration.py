"""Persistent, fail-closed input calibration records."""

from __future__ import annotations

import json
import platform
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from tempfile import NamedTemporaryFile

from core.input import DEFAULT_INPUT_BACKEND, configured_input_backend


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CALIBRATION_PATH = PROJECT_ROOT / "data" / "input_calibration.json"
CALIBRATION_VERSION = 2
REQUIRED_ACTIONS = (
    "throttle_forward",
    "rudder_left",
    "rudder_right",
    "main_fire",
    "secondary_lock",
)
AUTOMATIC_PREFLIGHT_KEY = "automatic_preflight"


@dataclass(frozen=True)
class CalibrationStatus:
    valid: bool
    reason: str
    confirmed_actions: tuple[str, ...] = ()
    created_at: float = 0.0
    game_title: str = ""
    resolution: tuple[int, int] = (0, 0)
    backend: str = DEFAULT_INPUT_BACKEND

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["confirmed_actions"] = list(self.confirmed_actions)
        payload["resolution"] = list(self.resolution)
        return payload


@dataclass
class InputCalibration:
    version: int = CALIBRATION_VERSION
    created_at: float = field(default_factory=time.time)
    machine: str = field(default_factory=platform.node)
    backend: str = DEFAULT_INPUT_BACKEND
    game_title: str = ""
    resolution: list[int] = field(default_factory=lambda: [0, 0])
    confirmed_actions: list[str] = field(default_factory=list)
    observations: dict[str, dict] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict) -> "InputCalibration":
        return cls(
            version=int(payload.get("version", 0)),
            created_at=float(payload.get("created_at", 0)),
            machine=str(payload.get("machine", "")),
            backend=str(payload.get("backend", "")),
            game_title=str(payload.get("game_title", "")),
            resolution=[int(value) for value in payload.get("resolution", [0, 0])[:2]],
            confirmed_actions=[str(value) for value in payload.get("confirmed_actions", [])],
            observations=dict(payload.get("observations", {})),
        )


class CalibrationStore:
    def __init__(self, path: Path = DEFAULT_CALIBRATION_PATH, max_age_days: float = 30):
        self.path = Path(path)
        self.max_age_days = max(0.0, float(max_age_days))

    def load(self) -> InputCalibration | None:
        if not self.path.exists():
            return None
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                return None
            return InputCalibration.from_dict(payload)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return None

    def status(self, now: float | None = None) -> CalibrationStatus:
        record = self.load()
        if record is None:
            return CalibrationStatus(False, "尚未完成实机输入校准")
        if record.version != CALIBRATION_VERSION:
            return CalibrationStatus(False, "校准版本已过期，需要重新校准")
        if record.machine and record.machine != platform.node():
            return CalibrationStatus(False, "校准来自另一台电脑")
        if record.backend != configured_input_backend():
            return CalibrationStatus(False, "校准输入后端与当前版本不一致")
        if not record.game_title.strip():
            return CalibrationStatus(False, "校准记录缺少游戏窗口信息")
        automatic_preflight = record.observations.get(AUTOMATIC_PREFLIGHT_KEY, {})
        automatic_valid = bool(
            isinstance(automatic_preflight, dict)
            and automatic_preflight.get("passed") is True
        )
        if not automatic_valid:
            missing = sorted(set(REQUIRED_ACTIONS) - set(record.confirmed_actions))
            if missing:
                return CalibrationStatus(False, f"校准缺少动作: {', '.join(missing)}")
        current = time.time() if now is None else float(now)
        if self.max_age_days and current - record.created_at > self.max_age_days * 86400:
            return CalibrationStatus(False, "校准已超过有效期，需要重新校准")
        resolution = tuple((record.resolution + [0, 0])[:2])
        if resolution[0] <= 0 or resolution[1] <= 0:
            return CalibrationStatus(False, "校准记录缺少有效分辨率")
        return CalibrationStatus(
            True,
            "自动港口自检已通过" if automatic_valid else "实机输入校准已通过",
            tuple(record.confirmed_actions),
            record.created_at,
            record.game_title,
            resolution,
            record.backend,
        )

    def save(self, record: InputCalibration) -> CalibrationStatus:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(asdict(record), ensure_ascii=False, indent=2)
        with NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=self.path.parent,
            prefix=f".{self.path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            stream.write(payload)
            temporary = Path(stream.name)
        temporary.replace(self.path)
        return self.status()
