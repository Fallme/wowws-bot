"""Launch World of Warships through the installed Steam client."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import subprocess
import winreg


STEAM_APP_ID = "552990"


@dataclass(frozen=True)
class LaunchResult:
    started: bool
    method: str
    detail: str


def _registry_value(root, path: str, name: str) -> str:
    try:
        with winreg.OpenKey(root, path) as key:
            value, _ = winreg.QueryValueEx(key, name)
        return str(value or "").strip()
    except OSError:
        return ""


def find_steam_executable() -> Path | None:
    """Locate Steam from an explicit override or standard registry keys."""
    override = os.environ.get("WOWS_STEAM_EXE", "").strip()
    candidates = [Path(override)] if override else []
    user_exe = _registry_value(
        winreg.HKEY_CURRENT_USER,
        r"Software\Valve\Steam",
        "SteamExe",
    )
    if user_exe:
        candidates.append(Path(user_exe))
    for root, key in (
        (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\WOW6432Node\Valve\Steam"),
        (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Valve\Steam"),
    ):
        install = _registry_value(root, key, "InstallPath")
        if install:
            candidates.append(Path(install) / "Steam.exe")
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def launch_game(*, popen=subprocess.Popen, startfile=os.startfile) -> LaunchResult:
    """Request a game launch without assuming that Steam is on ``PATH``."""
    explicit_game = os.environ.get("WOWS_GAME_EXE", "").strip()
    if explicit_game:
        executable = Path(explicit_game)
        if not executable.is_file():
            return LaunchResult(False, "game_exe", f"游戏路径不存在: {executable}")
        popen(
            [str(executable)],
            cwd=str(executable.parent),
            creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
        )
        return LaunchResult(True, "game_exe", str(executable))

    steam = find_steam_executable()
    if steam is not None:
        popen(
            [str(steam), "-applaunch", STEAM_APP_ID],
            cwd=str(steam.parent),
            creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
        )
        return LaunchResult(True, "steam", str(steam))

    uri = f"steam://rungameid/{STEAM_APP_ID}"
    try:
        startfile(uri)
    except OSError as error:
        return LaunchResult(False, "steam_uri", str(error))
    return LaunchResult(True, "steam_uri", uri)
