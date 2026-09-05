"""Locate and launch World of Warships from Steam or WG Game Center installs."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
import os
from pathlib import Path
import re
import subprocess
import winreg


STEAM_APP_ID = "552990"
LAUNCH_CLIENTS = ("steam", "wgc")


@dataclass(frozen=True)
class LaunchResult:
    started: bool
    method: str
    detail: str


def normalize_launch_client(value: str | None) -> str:
    client = str(value or "steam").strip().lower()
    if client not in LAUNCH_CLIENTS:
        raise ValueError("游戏客户端必须是 Steam 或 WG Game Center")
    return client


def _registry_value(root, path: str, name: str) -> str:
    try:
        with winreg.OpenKey(root, path) as key:
            value, _ = winreg.QueryValueEx(key, name)
        return str(value or "").strip()
    except OSError:
        return ""


def _path_from_registry(value: str) -> Path | None:
    """Turn a registry path or DisplayIcon value into an ordinary Path."""
    cleaned = str(value or "").strip().strip('"')
    if not cleaned:
        return None
    cleaned = re.sub(r",\s*-?\d+\s*$", "", cleaned).strip().strip('"')
    return Path(os.path.expandvars(cleaned))


def _first_file(candidates) -> Path | None:
    seen: set[str] = set()
    for candidate in candidates:
        if candidate is None:
            continue
        path = Path(candidate).expanduser()
        key = str(path).casefold()
        if key in seen:
            continue
        seen.add(key)
        if path.is_file():
            return path.resolve()
    return None


def _uninstall_entries():
    paths = (
        r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall",
        r"SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall",
    )
    for root in (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE):
        for path in paths:
            try:
                with winreg.OpenKey(root, path) as key:
                    count = winreg.QueryInfoKey(key)[0]
                    subkeys = [winreg.EnumKey(key, index) for index in range(count)]
            except OSError:
                continue
            for subkey in subkeys:
                full_path = f"{path}\\{subkey}"
                yield {
                    "name": _registry_value(root, full_path, "DisplayName"),
                    "location": _registry_value(root, full_path, "InstallLocation"),
                    "icon": _registry_value(root, full_path, "DisplayIcon"),
                }


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
    return _first_file(candidates)


def _steam_library_roots(steam: Path):
    root = steam.parent
    yield root
    library_file = root / "steamapps" / "libraryfolders.vdf"
    try:
        text = library_file.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return
    for value in re.findall(r'"path"\s+"([^"]+)"', text, flags=re.IGNORECASE):
        yield Path(value.replace("\\\\", "\\"))


def find_steam_game_executable() -> Path | None:
    """Locate the installed Steam game for status display and diagnostics."""
    override = os.environ.get("WOWS_STEAM_GAME_EXE", "").strip()
    if override:
        found = _first_file([Path(override)])
        if found is not None:
            return found
    steam = find_steam_executable()
    if steam is None:
        return None
    candidates: list[Path] = []
    for library in _steam_library_roots(steam):
        steamapps = library / "steamapps"
        manifest = steamapps / f"appmanifest_{STEAM_APP_ID}.acf"
        try:
            text = manifest.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        match = re.search(r'"installdir"\s+"([^"]+)"', text, flags=re.IGNORECASE)
        if not match:
            continue
        game_root = steamapps / "common" / match.group(1)
        candidates.extend(
            (game_root / "WorldOfWarships.exe", game_root / "WorldOfWarships64.exe")
        )
    return _first_file(candidates)


def find_wgc_executable() -> Path | None:
    """Locate the WG Game Center executable without scanning whole drives."""
    override = os.environ.get("WOWS_WGC_EXE", "").strip()
    candidates: list[Path] = [Path(override)] if override else []
    registry_keys = (
        r"SOFTWARE\Wargaming.net\GameCenter",
        r"SOFTWARE\WOW6432Node\Wargaming.net\GameCenter",
    )
    for root in (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE):
        for key in registry_keys:
            for name in ("path", "Path", "InstallPath", "install_path", "RootDir"):
                value = _registry_value(root, key, name)
                if not value:
                    continue
                path = _path_from_registry(value)
                if path is not None:
                    candidates.append(path if path.suffix else path / "wgc.exe")
    for entry in _uninstall_entries():
        if "wargaming" not in entry["name"].casefold() or "game center" not in entry[
            "name"
        ].casefold():
            continue
        location = _path_from_registry(entry["location"])
        icon = _path_from_registry(entry["icon"])
        if location is not None:
            candidates.append(location / "wgc.exe")
        candidates.append(icon)
    for variable in ("ProgramData", "ProgramFiles", "ProgramFiles(x86)", "LOCALAPPDATA"):
        base = os.environ.get(variable, "").strip()
        if base:
            candidates.append(Path(base) / "Wargaming.net" / "GameCenter" / "wgc.exe")
    return _first_file(candidates)


def _fixed_drive_roots():
    """Yield mounted local fixed drives (C:\, D:\ ...).

    Probing every letter with Path.exists() can block forever when a stale
    device letter is present (e.g. a disconnected USB drive that still has a
    mount point), so enumerate only drives Windows reports as mounted.
    """
    DRIVE_FIXED = 3
    bits = ctypes.windll.kernel32.GetLogicalDrives()
    for offset in range(26):
        if not bits & (1 << offset):
            continue
        letter = f"{chr(ord('A') + offset)}:\\"
        if ctypes.windll.kernel32.GetDriveTypeW(letter) == DRIVE_FIXED:
            yield Path(letter)


def _game_executable_candidates(game_root: Path):
    yield game_root / "WorldOfWarships.exe"
    yield game_root / "WorldOfWarships64.exe"
    bin_root = game_root / "bin"
    try:
        versions = sorted(
            (item for item in bin_root.iterdir() if item.is_dir()),
            key=lambda item: int(item.name) if item.name.isdigit() else -1,
            reverse=True,
        )
    except OSError:
        versions = []
    for version in versions:
        yield version / "bin64" / "WorldOfWarships64.exe"


def find_wgc_game_executable(*, search_roots=None) -> Path | None:
    """Locate a WG-installed game from overrides, uninstall data and common roots."""
    for variable in ("WOWS_WGC_GAME_EXE", "WOWS_GAME_EXE"):
        override = os.environ.get(variable, "").strip()
        if override:
            found = _first_file([Path(override)])
            if found is not None:
                return found

    game_roots: list[Path] = []
    for entry in _uninstall_entries():
        name = entry["name"].casefold()
        if "world of warships" not in name and "战舰世界" not in name:
            continue
        location = _path_from_registry(entry["location"])
        icon = _path_from_registry(entry["icon"])
        if location is not None:
            game_roots.append(location)
        if icon is not None:
            game_roots.append(icon.parent)

    roots = list(search_roots) if search_roots is not None else list(_fixed_drive_roots())
    relative_roots = (
        Path("Games") / "World_of_Warships",
        Path("Games") / "World of Warships",
        Path("World_of_Warships"),
        Path("Wargaming.net") / "World_of_Warships",
        Path("Program Files") / "Wargaming.net" / "World_of_Warships",
        Path("Program Files (x86)") / "Wargaming.net" / "World_of_Warships",
    )
    for root in roots:
        game_roots.extend(Path(root) / relative for relative in relative_roots)

    candidates = (
        executable
        for game_root in game_roots
        for executable in _game_executable_candidates(game_root)
    )
    return _first_file(candidates)


def launcher_statuses() -> dict:
    """Return paths found for each UI choice. This function never starts anything."""
    steam = find_steam_executable()
    steam_game = find_steam_game_executable()
    wgc = find_wgc_executable()
    wgc_game = find_wgc_game_executable()
    return {
        "steam": {
            "name": "Steam",
            "available": steam is not None,
            "launcher_path": str(steam or ""),
            "game_path": str(steam_game or ""),
            "detail": "已自动找到 Steam" if steam else "未检测到 Steam",
        },
        "wgc": {
            "name": "WG Game Center",
            "available": wgc_game is not None,
            "launcher_path": str(wgc or ""),
            "game_path": str(wgc_game or ""),
            "detail": (
                "已自动找到游戏路径"
                if wgc_game
                else "已找到 WG Center，但未找到游戏路径"
                if wgc
                else "未检测到 WG Center 游戏路径"
            ),
        },
    }


def launch_game(
    *, client: str | None = None, popen=subprocess.Popen, startfile=os.startfile
) -> LaunchResult:
    """Launch the game with the selected client and never cross-fallback clients."""
    try:
        selected = normalize_launch_client(
            client if client is not None else os.environ.get("WOWS_LAUNCHER_CLIENT")
        )
    except ValueError as error:
        return LaunchResult(False, "invalid_client", str(error))

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
        return LaunchResult(True, "game_exe", str(executable.resolve()))

    if selected == "wgc":
        executable = find_wgc_game_executable()
        if executable is None:
            center = find_wgc_executable()
            detail = (
                f"已找到 WG Game Center（{center}），但未找到《战舰世界》安装路径"
                if center
                else "未找到 WG Game Center 或《战舰世界》安装路径"
            )
            return LaunchResult(False, "wgc_game", detail)
        popen(
            [str(executable)],
            cwd=str(executable.parent),
            creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
        )
        return LaunchResult(True, "wgc_game", str(executable))

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
