from __future__ import annotations

import json
import time

import pytest

from core import updater


@pytest.fixture(autouse=True)
def isolated_state(tmp_path, monkeypatch):
    monkeypatch.setattr(updater, "VERSION_PATH", tmp_path / "VERSION")
    monkeypatch.setattr(updater, "STATUS_PATH", tmp_path / "update_status.json")
    monkeypatch.setattr(updater, "INSTALL_ID_PATH", tmp_path / "install_id")
    monkeypatch.setattr(updater, "HEARTBEAT_STAMP_PATH", tmp_path / "heartbeat.json")
    monkeypatch.setattr(updater, "_STARTUP_CHECKED", False)
    monkeypatch.delenv("WOWS_HEARTBEAT_URL", raising=False)
    monkeypatch.delenv("WOWS_TELEMETRY_STATS_URL", raising=False)
    monkeypatch.delenv("WOWS_GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("WOWS_UPDATE_CHECK_HOURS", raising=False)
    monkeypatch.delenv("WOWS_UPDATE_RETRY_MINUTES", raising=False)


def _payload(latest: str = "2026.11.01.1", **extra):
    return {
        "current_version": "2026.11.01.1",
        "latest_version": latest,
        "update_available": latest != "2026.11.01.1",
        "local_ahead": False,
        "source": "test",
        "repo_url": "https://github.com/Fallme/wowws-bot",
        "commit": {},
        "github": {},
        "stats": {},
        "heartbeat": None,
        "error": "",
        "checked_at": time.time(),
        "cached": False,
        **extra,
    }


def test_version_state_compares_numeric_parts():
    assert updater.version_state("2026.10.31.1", "2026.11.01.1") == (True, False)
    assert updater.version_state("2026.11.02", "2026.11.01.1") == (False, True)
    assert updater.version_state("2026.11.01.1", "2026.11.01.1") == (False, False)
    assert updater.version_state("unknown", "2026.11.01.1") == (False, False)
    assert updater.version_state("2026.11.01.1", "unknown") == (False, False)


def test_install_id_is_stable_and_anonymous(tmp_path):
    first = updater.install_id()
    assert len(first) == 32
    assert all(char in "0123456789abcdef" for char in first)
    assert updater.install_id() == first
    assert (tmp_path / "install_id").read_text(encoding="utf-8") == first


def test_first_check_in_each_process_ignores_fresh_cache(monkeypatch):
    calls = []

    def fake_collect(force):
        calls.append(force)
        return _payload()

    monkeypatch.setattr(updater, "_collect", fake_collect)
    fresh = _payload(checked_at=time.time())
    (updater.STATUS_PATH).write_text(json.dumps(fresh), encoding="utf-8")

    first = updater.check_for_updates()
    assert calls == [False]
    assert first["cached"] is False

    second = updater.check_for_updates()
    assert calls == [False]
    assert second["cached"] is True

    monkeypatch.setattr(updater, "_STARTUP_CHECKED", False)
    third = updater.check_for_updates()
    assert calls == [False, False]
    assert third["cached"] is False


def test_force_bypasses_cache_even_after_startup(monkeypatch):
    calls = []
    monkeypatch.setattr(
        updater,
        "_collect",
        lambda force: calls.append(force) or _payload(cached=False),
    )
    updater.STATUS_PATH.write_text(json.dumps(_payload()), encoding="utf-8")
    updater.check_for_updates()
    result = updater.check_for_updates(force=True)
    assert calls == [False, True]
    assert result["cached"] is False


def test_collect_reports_update_and_persists_status(monkeypatch):
    updater.VERSION_PATH.write_text("2026.10.31.1\n", encoding="utf-8")
    monkeypatch.setattr(updater, "_remote_version", lambda: ("2026.11.01.1", "test"))
    monkeypatch.setattr(updater, "_latest_commit", lambda: {"message": "release"})
    monkeypatch.setattr(updater, "_github_traffic", lambda: {})
    monkeypatch.setattr(updater, "_telemetry_stats", lambda: {})
    monkeypatch.setattr(updater, "_send_heartbeat", lambda version: None)

    result = updater.check_for_updates()

    assert result["update_available"] is True
    assert result["local_ahead"] is False
    assert result["latest_version"] == "2026.11.01.1"
    assert result["error"] == ""
    saved = json.loads(updater.STATUS_PATH.read_text(encoding="utf-8"))
    assert saved["update_available"] is True
    assert saved["checked_at"] == result["checked_at"]


def test_network_failure_is_cached_briefly_then_retried(monkeypatch):
    calls = []
    monkeypatch.setattr(
        updater,
        "_collect",
        lambda force: calls.append(force) or _payload(
            latest="unknown", update_available=False, error="offline"
        ),
    )
    updater.STATUS_PATH.write_text(
        json.dumps(_payload(latest="unknown", update_available=False, error="offline")),
        encoding="utf-8",
    )
    updater.check_for_updates()
    cached = updater.check_for_updates()
    assert cached["cached"] is True
    assert cached["error"] == "offline"
    assert calls == [False]

    stale = _payload(latest="unknown", update_available=False, error="offline")
    stale["checked_at"] = time.time() - 20 * 60
    updater.STATUS_PATH.write_text(json.dumps(stale), encoding="utf-8")
    updater.check_for_updates()
    assert calls == [False, False]
