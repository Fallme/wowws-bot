"""Application configuration loading and validation."""

import os
from copy import deepcopy
from pathlib import Path

import yaml

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_SHIP_CONFIG = BASE_DIR / "config" / "ship.yaml"
CUSTOM_SHIP_KEY = "custom"


def _custom_ship_config(data: dict) -> dict:
    """Build a conservative runtime profile from the two Web form fields."""
    name = os.environ.get("WOWS_CUSTOM_SHIP_NAME", "").strip()
    if not name or len(name) > 64 or any(character in name for character in "\r\n\t"):
        raise ValueError("自定义舰船必须填写完整名称（最多 64 个字符）")
    try:
        secondary_range = float(
            os.environ.get("WOWS_CUSTOM_SECONDARY_RANGE", "")
        )
    except ValueError as error:
        raise ValueError("自定义舰船副炮射程必须是数字") from error
    if not 1.0 <= secondary_range <= 30.0:
        raise ValueError("自定义舰船副炮射程必须在 1.0 到 30.0 km 之间")

    # Unknown ships deliberately inherit the generic battleship controls and
    # do not use ship-specific torpedo/smoke commands. Only distance-dependent
    # movement is adjusted from the user-provided secondary range.
    base = data.get("pommern")
    if not isinstance(base, dict):
        raise ValueError("自定义舰船缺少基础控制配置")
    ship = deepcopy(base)
    ship.update(
        {
            "name": name,
            "display_name": name,
            "type": "CUSTOM",
            "nation": "custom",
            "has_torpedoes": False,
            "has_smoke": False,
        }
    )
    ship["secondary"]["range"] = secondary_range
    strategy = ship["strategy"]
    inner = min(max(secondary_range * 0.62, 3.0), secondary_range - 0.8)
    strategy.update(
        {
            "brake_start_distance_km": secondary_range + 1.6,
            "ideal_outer_distance_km": max(1.0, secondary_range - 0.2),
            "ideal_inner_distance_km": max(1.0, inner),
            "too_close_distance_km": max(1.0, secondary_range * 0.4),
            "secondary_target_distance_km": max(1.0, secondary_range - 0.5),
            "secondary_inner_distance_km": max(1.0, inner),
        }
    )
    return ship


def ship_key_from_env() -> str:
    key = os.environ.get("WOWS_SHIP", "pommern").strip().lower()
    if not key:
        raise ValueError("WOWS_SHIP must not be empty")
    return key


def load_ship_config(ship_key: str, path: Path | None = None) -> dict:
    config_path = path or DEFAULT_SHIP_CONFIG
    with config_path.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)

    if not isinstance(data, dict):
        raise ValueError(f"Ship configuration must be a mapping: {config_path}")
    if ship_key == CUSTOM_SHIP_KEY:
        return _custom_ship_config(data)
    if ship_key not in data:
        choices = ", ".join(sorted(data))
        raise KeyError(f"Unknown ship {ship_key!r}; available ships: {choices}")

    ship = data[ship_key]
    if not isinstance(ship, dict) or not ship.get("name"):
        raise ValueError(f"Invalid ship configuration for {ship_key!r}")
    secondary = ship.get("secondary")
    if not isinstance(secondary, dict) or secondary.get("range", 0) <= 0:
        raise ValueError(f"Ship {ship_key!r} requires a positive secondary range")
    return ship
