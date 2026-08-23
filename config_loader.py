"""Application configuration loading and validation."""

import os
from pathlib import Path

import yaml

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_SHIP_CONFIG = BASE_DIR / "config" / "ship.yaml"


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
