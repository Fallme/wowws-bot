"""A failed task can continue with its original progress and reward ledger."""

import json
import sqlite3
from unittest.mock import Mock

import pytest

import control_server
from control_server import ControlStore, RunnerManager
from runtime_control import RunLimits, RuntimeReporter


def isolated_manager(tmp_path, monkeypatch):
    store = ControlStore(tmp_path / "panel.db")
    manager = RunnerManager(store)
    for key, name in (
        ("STATE_PATH", "state.json"),
        ("STOP_PATH", "stop.request"),
        ("PAUSE_PATH", "pause.request"),
        ("RESUME_PATH", "resume.request"),
        ("LOG_PATH", "runtime.log"),
    ):
        monkeypatch.setattr(control_server, key, tmp_path / name)
    monkeypatch.setattr(control_server, "DATA_DIR", tmp_path)
    monkeypatch.setattr(control_server, "running_game_requires_elevation", lambda: False)
    monkeypatch.setattr(control_server, "remember_launch_client", lambda _client: None)
    monkeypatch.setattr(control_server, "last_launch_client", lambda: "steam")
    process = Mock(pid=1234, returncode=1)
    process.poll.return_value = None
    launched = Mock(return_value=process)
    monkeypatch.setattr(control_server.subprocess, "Popen", launched)
    return store, manager, process, launched


def test_failed_run_resumes_original_id_and_progress_without_duplicate_history(tmp_path, monkeypatch):
    store, manager, process, launched = isolated_manager(tmp_path, monkeypatch)
    profile = {
        "ship": "pommern", "mode": "cooperative",
        "limit_type": "rounds", "limit_value": 30,
        "quick_battle": False, "ocr_device": "nvidia",
        "launcher_client": "steam",
    }
    try:
        store.create_run("old-run", "pommern", "cooperative", "rounds", 30,
                         resume_config=profile)
        store.update_run_progress("old-run", 12, 5400)
        store.upsert_battle_result("old-run", 12, "victory",
                                   {"credits": 53203, "ship_xp": 1665, "free_xp": 105},
                                   rewards_recognized=True)
        store.finish_run("old-run", "failed", error="transient capture fault")

        result = manager.resume_finished("old-run")
        assert result == {"run_id": "old-run", "resumed": True}
        environment = launched.call_args.kwargs["env"]
        assert environment["WOWS_RUN_ID"] == "old-run"
        assert environment["WOWS_INITIAL_COMPLETED_ROUNDS"] == "12"
        assert environment["WOWS_INITIAL_ELAPSED_SECONDS"] == "5400.0"
        assert environment["WOWS_RESUME_EXISTING"] == "1"
        row = store.get_run("old-run")
        assert row["status"] == "running" and row["completed_rounds"] == 12
        assert len(store.dashboard()["runs"]) == 1
        assert len(store.dashboard()["history"]) == 1
        with pytest.raises(RuntimeError, match="已有任务"):
            manager.resume_finished("old-run")

        # A failed restart must not erase the twelve settled rounds. Preserve
        # the new error and archive the runner log before a later start clears it.
        manager.log_stream.write("runner failed after restart\n")
        manager.log_stream.flush()
        control_server.STATE_PATH.write_text(json.dumps({
            "run_id": "old-run", "state": "failed", "completed_rounds": 0,
            "elapsed_seconds": 1, "error": "restart fault",
        }), encoding="utf-8")
        process.poll.return_value = 1
        manager._reconcile()
        row = store.get_run("old-run")
        assert row["status"] == "failed"
        assert row["completed_rounds"] == 12
        assert row["duration_seconds"] == 5400
        assert row["last_error"] == "restart fault"
        assert "runner failed" in (tmp_path / "run_logs" / "old-run.log").read_text(encoding="utf-8")
    finally:
        if manager.log_stream and not manager.log_stream.closed:
            manager.log_stream.close()
        store.close()


def test_legacy_custom_ship_run_requires_matching_saved_ship(tmp_path, monkeypatch):
    store, manager, _process, launched = isolated_manager(tmp_path, monkeypatch)
    try:
        store.create_run("legacy", "鲁普雷希特亲王", "cooperative", "rounds", 30)
        store.update_run_progress("legacy", 12, 4000)
        store.finish_run("legacy", "failed")
        monkeypatch.setattr(control_server, "load_custom_ship", lambda: {
            "name": "另一艘船", "secondary_range": 11.4,
        })
        with pytest.raises(ValueError, match="配置已变化"):
            manager.resume_finished("legacy")
        launched.assert_not_called()
        monkeypatch.setattr(control_server, "load_custom_ship", lambda: {
            "name": "鲁普雷希特亲王", "secondary_range": 11.4,
        })
        monkeypatch.setattr(control_server, "save_custom_ship", lambda *_args: None)
        assert manager.resume_finished("legacy")["resumed"]
        environment = launched.call_args.kwargs["env"]
        assert environment["WOWS_CUSTOM_SHIP_NAME"] == "鲁普雷希特亲王"
        assert environment["WOWS_INITIAL_COMPLETED_ROUNDS"] == "12"
        assert environment["WOWS_RESUME_EXISTING"] == "1"
    finally:
        if manager.log_stream and not manager.log_stream.closed:
            manager.log_stream.close()
        store.close()


def test_interrupted_run_requires_continue_or_explicit_terminate_before_new_start(tmp_path, monkeypatch):
    store, manager, _process, launched = isolated_manager(tmp_path, monkeypatch)
    profile = {
        "ship": "pommern", "mode": "cooperative",
        "limit_type": "rounds", "limit_value": 30,
        "quick_battle": False, "ocr_device": "nvidia",
        "launcher_client": "steam",
    }
    try:
        store.create_run("interrupted", "pommern", "cooperative", "rounds", 30,
                         resume_config=profile)
        store.update_run_progress("interrupted", 12, 5400)
        store.finish_run("interrupted", "failed", error="capture lost")
        assert store.get_resume_candidate()["id"] == "interrupted"
        monkeypatch.setattr(
            control_server, "CalibrationStore",
            lambda: Mock(status=lambda: Mock(to_dict=lambda: {})),
        )
        monkeypatch.setattr(control_server, "tail_log", lambda: [])
        status = manager.status()
        assert status["resume_candidate"] == {
            "run_id": "interrupted", "completed_rounds": 12,
        }
        assert status["completed_rounds"] == 12
        assert status["current_round"] == 13
        with pytest.raises(ValueError, match="请点击继续"):
            manager.start(profile)
        launched.assert_not_called()

        # Terminate an already-exited worker to explicitly abandon this plan.
        assert manager.stop()
        assert store.get_resume_candidate() is None
        assert store.get_run("interrupted")["abandoned"] == 1
        with pytest.raises(ValueError, match="仅能接续"):
            manager.resume_finished("interrupted")

        fresh = manager.start(profile)
        assert not fresh["resumed"] and fresh["run_id"] != "interrupted"
        assert launched.call_args.kwargs["env"]["WOWS_RESUME_EXISTING"] == "0"
        assert store.get_run(fresh["run_id"])["completed_rounds"] == 0
    finally:
        if manager.log_stream and not manager.log_stream.closed:
            manager.log_stream.close()
        store.close()


def test_explicit_termination_during_live_run_disables_automatic_continue(tmp_path, monkeypatch):
    store, manager, process, _launched = isolated_manager(tmp_path, monkeypatch)
    monkeypatch.setattr(manager, "_watch_stop_request", lambda _process: None)
    try:
        manager.start({
            "ship": "pommern", "mode": "cooperative",
            "limit_type": "rounds", "limit_value": 30,
        })
        run_id = manager.run_id
        assert manager.stop()
        assert store.get_run(run_id)["abandoned"] == 1
        process.poll.return_value = 1
        manager._reconcile()
        assert store.get_resume_candidate() is None
    finally:
        if manager.log_stream and not manager.log_stream.closed:
            manager.log_stream.close()
        store.close()


def test_reporter_seeds_resumed_progress_and_active_duration(tmp_path):
    limits = RunLimits(run_id="same-run", max_rounds=30,
                       initial_completed_rounds=12, initial_elapsed_seconds=5400,
                       state_file=tmp_path / "state.json")
    reporter = RuntimeReporter(limits, ship="pommern", mode="cooperative")
    reporter.update("starting", "接续任务")
    state = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))
    assert state["completed_rounds"] == 12
    assert state["current_round"] == 13
    assert state["elapsed_seconds"] >= 5400


def test_legacy_explicitly_stopped_run_is_not_primary_resume_candidate(tmp_path):
    database = tmp_path / "legacy.db"
    with sqlite3.connect(database) as connection:
        connection.execute(
            """CREATE TABLE runs (id TEXT PRIMARY KEY, ship TEXT NOT NULL,
               mode TEXT NOT NULL, limit_type TEXT NOT NULL,
               limit_value REAL NOT NULL, status TEXT NOT NULL,
               started_at REAL NOT NULL, ended_at REAL)"""
        )
        connection.execute(
            """INSERT INTO runs VALUES
               ('legacy-stop', 'pommern', 'cooperative', 'rounds', 30,
                'stopped', 1, 2)"""
        )
    store = ControlStore(database)
    try:
        assert store.get_run("legacy-stop")["abandoned"] == 1
        assert store.get_resume_candidate() is None
    finally:
        store.close()
