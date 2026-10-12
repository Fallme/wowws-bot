"""Locate and launch World of Warships from Steam or WG Game Center installs."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
import os
import logging
from pathlib import Path
import re
import subprocess
import winreg
import xml.etree.ElementTree as ET

from core.wgc_version import resolve_wgc_version
from core.wgc_readiness import wait_for_wgc_backend

logger = logging.getLogger("launcher")


STEAM_APP_ID = "552990"
LAUNCH_CLIENTS = ("steam", "wgc")
LAST_CLIENT_PATH = Path(__file__).resolve().parents[1] / "data" / "last_launcher.txt"


def last_launch_client() -> str:
    try:
        return normalize_launch_client(LAST_CLIENT_PATH.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return "steam"


def remember_launch_client(client: str) -> None:
    selected = normalize_launch_client(client)
    LAST_CLIENT_PATH.parent.mkdir(parents=True, exist_ok=True)
    LAST_CLIENT_PATH.write_text(selected, encoding="utf-8")


@dataclass(frozen=True)
class LaunchResult:
    # "started" means the launch request was dispatched, not that the game
    # is ready. The lifecycle must verify a real game-process window.
    started: bool
    method: str
    detail: str
    game_directory: str = ""
    launcher_path: str = ""


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
            path = Path(override)
            found = _first_file([path])
            if found is not None:
                return found
            # Updates can remove the versioned executable an old override
            # named. The verified installation root remains useful; its
            # current metadata decides the executable at launch time.
            directory = wgc_game_directory(path)
            if directory is not None:
                found = _first_file(_game_executable_candidates(directory))
                if found is not None:
                    return found

    game_roots: list[Path] = []
    # WGC records installations on arbitrary drives under apps/<game id>/.
    # Prefer its own registration over guessed common installation folders.
    center = find_wgc_executable()
    if center is not None:
        for record in (center.parent / "apps").glob("wows.*/*"):
            try:
                if record.is_file() and record.stat().st_size <= 4096:
                    game_roots.append(Path(record.read_text(encoding="utf-8-sig").strip()))
            except OSError:
                continue
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


def wgc_game_directory(executable: Path) -> Path | None:
    """Resolve the installation, including overrides pointing into bin/bin64."""
    for directory in tuple(executable.parents)[:4]:
        try:
            root = ET.parse(directory / "game_info.xml").getroot()
            if (root.findtext("game/id") or "").upper().startswith("WOWS."):
                return directory.resolve()
        except (OSError, ET.ParseError):
            continue
    return None


def find_wgc_direct_game_executable(game_directory: Path) -> Path | None:
    """Return the last known server engine without networking during UI reads."""
    return resolve_wgc_version(game_directory).executable


def _early_process_exit(process) -> int | None:
    """Catch the WG SDK's immediate restart request, before a window exists."""
    wait = getattr(process, "wait", None)
    if not callable(wait):
        return None
    try:
        code = wait(timeout=4)
    except subprocess.TimeoutExpired:
        return None
    return code if isinstance(code, int) else None


def wait_for_wgc_ready(directory: Path, *, timeout: float = 30.0, should_stop=None) -> bool:
    center = find_wgc_executable()
    return center is not None and wait_for_wgc_backend(
        center, directory, timeout=timeout, should_stop=should_stop)


def launcher_statuses() -> dict:
    """Return paths found for each UI choice. This function never starts anything."""
    steam = find_steam_executable()
    steam_game = find_steam_game_executable()
    wgc = find_wgc_executable()
    wgc_game = find_wgc_game_executable()
    wgc_directory = wgc_game_directory(wgc_game) if wgc_game else None
    selection = resolve_wgc_version(wgc_directory) if wgc_directory else None
    direct_game = selection.executable if selection else None
    return {
        "steam": {
            "name": "Steam",
            "available": steam is not None,
            "launcher_path": str(steam or ""),
            "game_path": str(steam_game or ""),
            "detail": "已自动找到 Steam" if steam else "未检测到 Steam",
        },
        "wgc": {
            "name": "WG 版（直接启动）",
            "available": direct_game is not None,
            "launcher_path": str(wgc or ""),
            "game_path": str(direct_game or ""),
            "version": selection.version if selection else "",
            "realm": selection.realm if selection else "",
            "version_source": selection.source if selection else "",
            "detail": (
                selection.detail + "；启动时自动同步服务器版本并直接运行游戏"
                if selection
                else "未检测到 WG 版游戏安装路径"
            ),
        },
    }


def launch_game(
    *, client: str | None = None, popen=subprocess.Popen, startfile=os.startfile,
    should_stop=None
) -> LaunchResult:
    """Launch the game with the selected client and never cross-fallback clients."""
    try:
        selected = normalize_launch_client(
            client if client is not None else os.environ.get("WOWS_LAUNCHER_CLIENT", last_launch_client())
        )
    except ValueError as error:
        return LaunchResult(False, "invalid_client", str(error))

    explicit_game = os.environ.get("WOWS_GAME_EXE", "").strip()
    if explicit_game and selected != "wgc":
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
        directory = wgc_game_directory(executable) if executable else None
        if directory is None:
            return LaunchResult(False, "wgc_game", "无法确认 WG 版战舰世界安装及 game_info.xml")
        # WGC downloads several builds. Its runner selects the one enabled
        # on the selected realm, which may differ from client.installed.
        selection = resolve_wgc_version(directory, refresh=True)
        native = selection.executable
        if native is None:
            return LaunchResult(False, "wgc_direct",
                                selection.detail,
                                str(directory))
        command = [str(native)]
        try:
            app_type = (ET.parse(directory / "app_type.xml").getroot().findtext("app_type") or "").strip()
        except (OSError, ET.ParseError):
            app_type = ""
        if app_type and re.fullmatch(r"[a-zA-Z0-9_]+", app_type):
            command.append(f"--apptype={app_type}")
        if should_stop is not None and should_stop():
            return LaunchResult(False, "wgc_direct", "已按用户要求停止启动", str(directory))
        retry_detail = ""
        try:
            # Mirror WG's native command and keep the user's shared profile,
            # graphics/UI settings and selected realm at the installation root.
            options = {"cwd": str(directory),
                       "creationflags": getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)}
            process = popen(command, **options)
            code = _early_process_exit(process)
            if code is not None:
                # Cold WG SDK initialization can start its background service
                # and exit the engine. Retry that same engine once; never
                # invoke a launcher command or wait for a launcher button.
                logger.info("游戏本体提前退出(%s)，等待 WG 后台初始化完成后重试本体: %s", code, native)
                ready = wait_for_wgc_ready(directory, should_stop=should_stop)
                if should_stop is not None and should_stop():
                    return LaunchResult(False, "wgc_direct", "已按用户要求停止启动", str(directory))
                if not ready:
                    return LaunchResult(False, "wgc_direct",
                                        f"{selection.detail}；本体提前退出({code})，WG 后台在30秒内未就绪，未重复启动",
                                        str(directory))
                retry_detail = f"；本体提前退出({code})，WG 后台就绪后已重试一次"
                logger.info("WG 后台已就绪，直接重试游戏本体: %s", native)
                process = popen(command, **options)
                code = _early_process_exit(process)
                if code is not None:
                    return LaunchResult(False, "wgc_direct",
                                        f"{selection.detail}；游戏本体连续提前退出({code})，已停止重试",
                                        str(directory))
        except OSError as error:
            return LaunchResult(False, "wgc_direct", f"无法直接启动游戏本体: {error}", str(directory))
        remember_launch_client(selected)
        return LaunchResult(True, "wgc_direct", f"{selection.detail}{retry_detail}；已直接启动游戏本体: {native}；等待真实游戏窗口", str(directory))

    steam = find_steam_executable()
    if steam is not None:
        popen(
            [str(steam), "-applaunch", STEAM_APP_ID],
            cwd=str(steam.parent),
            creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
        )
        remember_launch_client(selected)
        return LaunchResult(True, "steam", str(steam))

    uri = f"steam://rungameid/{STEAM_APP_ID}"
    try:
        startfile(uri)
    except OSError as error:
        return LaunchResult(False, "steam_uri", str(error))
    remember_launch_client(selected)
    return LaunchResult(True, "steam_uri", uri)
