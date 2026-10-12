"""Read WG backend readiness without opening or controlling its launcher UI."""

from __future__ import annotations

import ctypes
from ctypes import wintypes
from datetime import datetime
from pathlib import Path
import re
import time

import win32process


def running_wgc_process_ids() -> set[int]:
    kernel = ctypes.windll.kernel32
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.QueryFullProcessImageNameW.argtypes = [wintypes.HANDLE, wintypes.DWORD,
                                                wintypes.LPWSTR, ctypes.POINTER(wintypes.DWORD)]
    found = set()
    for pid in win32process.EnumProcesses():
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            continue
        try:
            size = wintypes.DWORD(32768)
            buffer = ctypes.create_unicode_buffer(size.value)
            if kernel.QueryFullProcessImageNameW(handle, 0, buffer, ctypes.byref(size)):
                if Path(buffer.value).name.casefold() == "wgc.exe":
                    found.add(pid)
        finally:
            kernel.CloseHandle(handle)
    return found


def _path_key(value: str) -> str:
    return value.strip().replace("\\", "/").rstrip("/").casefold()


def _session_matches_process(pid: int, center: Path, line: str) -> bool:
    # Windows can recycle a PID. Match both the executable and the process
    # creation time to this log session, not merely its integer identifier.
    match = re.search(r"\b(\d{2}:\d{2}:\d{2}\.\d+)\s*\|.*Starting log session \d+ at "
                      r"(\d{4}-\d{2}-\d{2}), Time Zone = GMT([+-]\d{4})", line)
    if not match:
        return False
    try:
        session_time = datetime.strptime(
            f"{match[2]} {match[1]} {match[3]}", "%Y-%m-%d %H:%M:%S.%f %z").timestamp()
    except ValueError:
        return False
    kernel = ctypes.windll.kernel32
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.QueryFullProcessImageNameW.argtypes = [wintypes.HANDLE, wintypes.DWORD,
                                                wintypes.LPWSTR, ctypes.POINTER(wintypes.DWORD)]
    kernel.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)] * 4
    handle = kernel.OpenProcess(0x1000, False, pid)
    if not handle:
        return False
    try:
        size = wintypes.DWORD(32768)
        buffer = ctypes.create_unicode_buffer(size.value)
        if not kernel.QueryFullProcessImageNameW(handle, 0, buffer, ctypes.byref(size)):
            return False
        if _path_key(buffer.value) != _path_key(str(Path(center).resolve())):
            return False
        times = [wintypes.FILETIME() for _ in range(4)]
        if not kernel.GetProcessTimes(handle, *(ctypes.byref(value) for value in times)):
            return False
        created = ((times[0].dwHighDateTime << 32) | times[0].dwLowDateTime) / 10_000_000 - 11644473600
        return -1 <= session_time - created <= 15
    finally:
        kernel.CloseHandle(handle)


def wgc_backend_ready(center: Path, directory: Path) -> bool:
    """Require initialization and this installation's Ready state in a live session.

    A running process alone is insufficient: the native engine exits while
    WG is still initializing. Old sessions and other games cannot authorize
    a retry. Only readiness markers are inspected; account data is not logged.
    """
    pids = running_wgc_process_ids()
    if not pids:
        return False
    records = []
    for path in (Path(center).parent / "logs").glob("wgc_*.log"):
        try:
            records.append((path.stat().st_mtime_ns, path))
        except OSError:
            continue
    seen = set()
    for _, path in sorted(records, reverse=True)[:64]:
        try:
            if path.stat().st_size > 64 * 1024 * 1024:
                continue
            with path.open("r", encoding="utf-8", errors="replace") as stream:
                head = stream.read(4096)
        except OSError:
            continue
        session_line = next((line for line in head.splitlines() if "Starting log session " in line), "")
        session = re.search(r"Starting log session (\d+) at", session_line)
        if not session:
            continue
        pid = int(session.group(1))
        if pid not in pids or pid in seen:
            continue
        if not _session_matches_process(pid, center, session_line):
            continue
        seen.add(pid)
        initialized = False
        closed = False
        latest = ""
        try:
            with path.open("r", encoding="utf-8", errors="replace") as stream:
                for line in stream:
                    if "GameCenter initialization finished" in line:
                        initialized = True
                    if f"Log session {pid} finished" in line:
                        closed = True
                    if "[GameCtl] Game state detected as [" not in line or "] for [" not in line:
                        continue
                    target = line.rsplit("|", 1)[-1].strip().rstrip("]")
                    if _path_key(target) == _path_key(str(directory)):
                        latest = line
        except OSError:
            continue
        if initialized and not closed and latest:
            if ("canStart = true" in latest and "launchForbidden = false" in latest
                    and "condition = Ready(0)" in latest
                    and "running = false" in latest and "processStarting = none" in latest):
                return True
    return False


def wait_for_wgc_backend(center: Path, directory: Path, *, timeout: float = 30.0,
                         should_stop=None) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if should_stop is not None and should_stop():
            return False
        if wgc_backend_ready(center, directory):
            return True
        time.sleep(min(0.25, max(0.0, deadline - time.monotonic())))
    return False
