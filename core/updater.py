"""Repository update checks and optional anonymous activity heartbeat."""

from __future__ import annotations

import base64
import json
import logging
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any

logger = logging.getLogger("updater")

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
VERSION_PATH = BASE_DIR / "VERSION"
STATUS_PATH = DATA_DIR / "update_status.json"
INSTALL_ID_PATH = DATA_DIR / "install_id"
HEARTBEAT_STAMP_PATH = DATA_DIR / "heartbeat.json"
DEFAULT_REPO = "Fallme/wowws-bot"
USER_AGENT = "wowws-bot-updater/1.0"

_LOCK = threading.Lock()
_STOP = threading.Event()
_THREAD: threading.Thread | None = None
_STARTUP_CHECKED = False


def repo_slug() -> str:
    return (os.environ.get("WOWS_UPDATE_REPO") or DEFAULT_REPO).strip() or DEFAULT_REPO


def repo_url() -> str:
    return f"https://github.com/{repo_slug()}"


def local_version() -> str:
    try:
        value = VERSION_PATH.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return "unknown"
    return value.splitlines()[0].strip() if value else "unknown"


def _version_key(value: str) -> tuple[int, ...]:
    return tuple(int(part) for part in re.findall(r"\d+", str(value or "")))


def version_state(current: str, latest: str) -> tuple[bool, bool]:
    """Return (update_available, local_ahead) for two version strings."""
    if not latest or latest == "unknown":
        return False, False
    current_key = _version_key(current)
    latest_key = _version_key(latest)
    if current_key and latest_key and current_key != latest_key:
        return latest_key > current_key, latest_key < current_key
    if current == "unknown":
        return False, False
    return latest != current, False


def install_id() -> str:
    try:
        value = INSTALL_ID_PATH.read_text(encoding="utf-8").strip()
    except OSError:
        value = ""
    if not re.fullmatch(r"[0-9a-f]{32}", value or ""):
        value = uuid.uuid4().hex
        try:
            INSTALL_ID_PATH.parent.mkdir(parents=True, exist_ok=True)
            INSTALL_ID_PATH.write_text(value, encoding="utf-8")
        except OSError:
            logger.debug("无法保存匿名安装编号", exc_info=True)
    return value


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary.replace(path)
    except OSError:
        logger.debug("无法保存更新检查结果", exc_info=True)


def _http(url: str, *, method: str = "GET", data: bytes | None = None,
          headers: dict[str, str] | None = None, timeout: float = 5.0) -> bytes:
    request = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={"User-Agent": USER_AGENT, **(headers or {})},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def _remote_version() -> tuple[str, str]:
    repo = repo_slug()
    errors = []
    token = _github_token()
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    # The contents API is the most reliable source on networks where
    # raw.githubusercontent.com or jsDelivr is throttled.
    for url in (
        f"https://api.github.com/repos/{repo}/contents/VERSION?ref=main",
        f"https://ghproxy.net/https://api.github.com/repos/{repo}/contents/VERSION?ref=main",
    ):
        try:
            payload = json.loads(_http(url, headers=headers, timeout=5.0))
            content = payload.get("content") or ""
            value = base64.b64decode(content).decode("utf-8", "replace").strip()
        except (
            OSError,
            urllib.error.URLError,
            json.JSONDecodeError,
            ValueError,
        ) as error:
            errors.append(f"github-api: {error}")
            continue
        if value:
            return value.splitlines()[0].strip(), "github-api"
        errors.append("github-api: empty")
    attempts = (
        (f"https://cdn.jsdelivr.net/gh/{repo}@main/VERSION", "jsdelivr"),
        (f"https://raw.githubusercontent.com/{repo}/main/VERSION", "github-raw"),
        (
            f"https://ghproxy.net/https://raw.githubusercontent.com/{repo}/main/VERSION",
            "ghproxy",
        ),
    )
    for url, source in attempts:
        try:
            value = _http(url, timeout=5.0).decode("utf-8", "replace").strip()
        except (OSError, urllib.error.URLError) as error:
            errors.append(f"{source}: {error}")
            continue
        if value:
            return value.splitlines()[0].strip(), source
        errors.append(f"{source}: empty")
    raise RuntimeError("; ".join(errors) or "no update source")


def _latest_commit() -> dict[str, Any]:
    repo = repo_slug()
    token = _github_token()
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    attempts = (
        f"https://api.github.com/repos/{repo}/commits/main",
        f"https://ghproxy.net/https://api.github.com/repos/{repo}/commits/main",
    )
    for url in attempts:
        try:
            payload = json.loads(_http(url, headers=headers, timeout=5.0))
        except (OSError, urllib.error.URLError, json.JSONDecodeError) as error:
            logger.debug("提交信息读取失败: %s", error)
            continue
        commit = payload.get("commit") or {}
        author = commit.get("author") or {}
        return {
            "sha": payload.get("sha", ""),
            "message": (commit.get("message") or "").splitlines()[0][:160],
            "date": author.get("date", ""),
            "url": payload.get("html_url", repo_url()),
        }
    return {}


def _github_token() -> str:
    token = os.environ.get("WOWS_GITHUB_TOKEN", "").strip()
    if token:
        return token
    try:
        return (DATA_DIR / "github_token").read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def _github_traffic() -> dict[str, Any]:
    token = _github_token()
    if not token:
        return {}
    repo = repo_slug()
    headers = {"Accept": "application/vnd.github+json", "Authorization": f"Bearer {token}"}
    result: dict[str, Any] = {}
    for key, path in (("clones", "traffic/clones"), ("views", "traffic/views")):
        try:
            payload = json.loads(_http(
                f"https://api.github.com/repos/{repo}/{path}",
                headers=headers,
                timeout=5.0,
            ))
        except (OSError, urllib.error.URLError, json.JSONDecodeError):
            continue
        result[f"{key}_count_14d"] = int(payload.get("count") or 0)
        result[f"{key}_unique_14d"] = int(payload.get("uniques") or 0)
    return result


def _telemetry_stats() -> dict[str, Any]:
    url = os.environ.get("WOWS_TELEMETRY_STATS_URL", "").strip()
    if not url:
        return {}
    try:
        payload = json.loads(_http(url, timeout=5.0))
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as error:
        logger.debug("活跃统计读取失败: %s", error)
        return {}
    result = {}
    for key in ("active_7d", "active_30d", "total"):
        try:
            result[key] = int(payload.get(key) or 0)
        except (TypeError, ValueError):
            pass
    return result


def _send_heartbeat(version: str) -> bool | None:
    url = os.environ.get("WOWS_HEARTBEAT_URL", "").strip()
    if not url:
        return None
    interval = float(os.environ.get("WOWS_HEARTBEAT_HOURS", "12"))
    stamp = _read_json(HEARTBEAT_STAMP_PATH)
    now = time.time()
    if now - float(stamp.get("sent_at") or 0) < max(3600.0, interval * 3600):
        return bool(stamp.get("sent_at"))
    payload = json.dumps(
        {
            "install_id": install_id(),
            "version": version,
            "platform": sys.platform,
            "event": "update_check",
            "sent_at": int(now),
        },
        ensure_ascii=False,
    ).encode("utf-8")
    try:
        _http(
            url,
            method="POST",
            data=payload,
            headers={"Content-Type": "application/json"},
            timeout=4.0,
        )
    except (OSError, urllib.error.URLError) as error:
        logger.debug("匿名活跃心跳发送失败: %s", error)
        return False
    _write_json(HEARTBEAT_STAMP_PATH, {"sent_at": now, "version": version})
    return True


def _collect(force: bool) -> dict[str, Any]:
    current = local_version()
    latest = ""
    source = ""
    error = ""
    try:
        latest, source = _remote_version()
    except RuntimeError as exc:
        error = str(exc)
    update_available, local_ahead = version_state(current, latest)
    commit = _latest_commit()
    github = _github_traffic()
    stats = _telemetry_stats()
    heartbeat = _send_heartbeat(current)
    payload = {
        "current_version": current,
        "latest_version": latest or "unknown",
        "update_available": update_available,
        "local_ahead": local_ahead,
        "source": source,
        "repo_url": repo_url(),
        "commit": commit,
        "github": github,
        "stats": stats,
        "heartbeat": heartbeat,
        "error": error,
        "checked_at": time.time(),
        "cached": False,
    }
    _write_json(STATUS_PATH, {**payload, "cached": False})
    return payload


def check_for_updates(*, force: bool = False) -> dict[str, Any]:
    """Return cached or fresh update information.

    Normal deployments check at most once per WOWS_UPDATE_CHECK_HOURS (6h).
    Failures retry after WOWS_UPDATE_RETRY_MINUTES (15m) instead of hammering
    the repository.
    """
    global _STARTUP_CHECKED
    with _LOCK:
        cached = _read_json(STATUS_PATH)
        now = time.time()
        checked_at = float(cached.get("checked_at") or 0)
        age = now - checked_at if checked_at else float("inf")
        ttl = max(0.25, float(os.environ.get("WOWS_UPDATE_CHECK_HOURS", "6"))) * 3600
        startup = not _STARTUP_CHECKED
        if cached and not force and not startup:
            retry = age < max(300.0, float(
                os.environ.get("WOWS_UPDATE_RETRY_MINUTES", "15")
            ) * 60)
            if age < ttl and (not cached.get("error") or retry):
                return {**cached, "cached": True}
        result = _collect(force)
        _STARTUP_CHECKED = True
        return result


def _auto_loop() -> None:
    if _STOP.wait(1.0):
        return
    while not _STOP.is_set():
        try:
            check_for_updates(force=False)
        except Exception:
            logger.debug("后台更新检查失败", exc_info=True)
        ttl = max(0.25, float(os.environ.get("WOWS_UPDATE_CHECK_HOURS", "6"))) * 3600
        if _STOP.wait(ttl):
            return


def start_auto() -> None:
    global _THREAD
    if _THREAD is not None and _THREAD.is_alive():
        return
    _STOP.clear()
    _THREAD = threading.Thread(target=_auto_loop, name="update-check", daemon=True)
    _THREAD.start()


def stop_auto() -> None:
    _STOP.set()
