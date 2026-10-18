"""Offline cold-start checks: only a ready WG backend permits one native retry."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import os
import subprocess

import pytest

import core.launcher as launcher
import core.wgc_readiness as readiness


@pytest.fixture
def direct_install(tmp_path, monkeypatch):
    directory = tmp_path / "game"
    directory.mkdir()
    native = directory / "bin" / "13357625" / "bin64" / "WorldOfWarships64.exe"
    native.parent.mkdir(parents=True)
    native.touch()
    (directory / "app_type.xml").write_text(
        "<protocol><app_type>high</app_type></protocol>", encoding="utf-8")
    for name in ("WOWS_GAME_EXE", "WOWS_WGC_GAME_EXE", "WOWS_LAUNCHER_CLIENT"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(launcher, "LAST_CLIENT_PATH", tmp_path / "last_launcher.txt")
    monkeypatch.setattr(launcher, "find_wgc_game_executable", lambda: native)
    monkeypatch.setattr(launcher, "wgc_game_directory", lambda _: directory)
    # Version resolution and all process dispatches stay offline.
    monkeypatch.setattr(launcher, "resolve_wgc_version", lambda *_args, **_kwargs:
                        SimpleNamespace(executable=native, detail="verified server version"))
    return directory, native


def running_engine(native):
    engine = Mock()
    engine.wait.side_effect = subprocess.TimeoutExpired(str(native), 4)
    return engine


def exited_engine(code=0):
    engine = Mock()
    engine.wait.return_value = code
    return engine


def test_running_native_does_not_wait_for_or_start_wg_backend(direct_install, monkeypatch):
    _directory, native = direct_install
    wait_ready = Mock(side_effect=AssertionError("warm game must not enter cold wait"))
    monkeypatch.setattr(launcher, "wait_for_wgc_ready", wait_ready)
    popen, startfile = Mock(return_value=running_engine(native)), Mock()
    result = launcher.launch_game(client="wgc", popen=popen, startfile=startfile)
    assert result.started
    assert popen.call_count == 1
    wait_ready.assert_not_called()
    startfile.assert_not_called()


def test_cold_exit_waits_until_ready_before_retrying_identical_native_command(
        direct_install, monkeypatch):
    directory, native = direct_install
    events = []
    processes = iter((exited_engine(), running_engine(native)))

    def dispatch(*_args, **_kwargs):
        events.append("native")
        return next(processes)

    def become_ready(*_args, **_kwargs):
        events.append("backend_ready")
        return True

    wait_ready = Mock(side_effect=become_ready)
    monkeypatch.setattr(launcher, "wait_for_wgc_ready", wait_ready)
    popen, startfile = Mock(side_effect=dispatch), Mock()
    should_stop = lambda: False
    result = launcher.launch_game(client="wgc", popen=popen, startfile=startfile,
                                  should_stop=should_stop)
    assert result.started
    assert events == ["native", "backend_ready", "native"]
    wait_ready.assert_called_once()
    assert wait_ready.call_args.args == (directory,)
    assert wait_ready.call_args.kwargs["should_stop"] is should_stop
    assert 0 < wait_ready.call_args.kwargs["timeout"] <= 30
    assert popen.call_args_list[0] == popen.call_args_list[1]
    assert popen.call_args.args[0] == [str(native), "--apptype=high"]
    assert Path(popen.call_args.kwargs["cwd"]) == directory
    assert "后台稳定后已重试1次" in result.detail
    startfile.assert_not_called()


def test_backend_timeout_is_explicit_failure_without_second_dispatch(
        direct_install, monkeypatch):
    wait_ready = Mock(return_value=False)
    monkeypatch.setattr(launcher, "wait_for_wgc_ready", wait_ready)
    popen, startfile = Mock(return_value=exited_engine()), Mock()
    result = launcher.launch_game(client="wgc", popen=popen, startfile=startfile)
    assert not result.started
    assert "未稳定就绪" in result.detail
    assert "已停止重试" in result.detail
    assert popen.call_count == 1
    assert not launcher.LAST_CLIENT_PATH.exists()
    startfile.assert_not_called()


@pytest.mark.parametrize("ready_result", [False, True])
def test_stop_during_backend_wait_cancels_even_if_backend_becomes_ready(
        direct_install, monkeypatch, ready_result):
    stop = {"requested": False}

    def wait_then_cancel(*_args, **_kwargs):
        stop["requested"] = True
        return ready_result

    monkeypatch.setattr(launcher, "wait_for_wgc_ready", wait_then_cancel)
    popen, startfile = Mock(return_value=exited_engine()), Mock()
    result = launcher.launch_game(client="wgc", popen=popen, startfile=startfile,
                                  should_stop=lambda: stop["requested"])
    assert not result.started
    assert "停止启动" in result.detail
    assert popen.call_count == 1
    assert not launcher.LAST_CLIENT_PATH.exists()
    startfile.assert_not_called()


def test_stop_before_first_dispatch_never_opens_a_process(direct_install, monkeypatch):
    wait_ready = Mock()
    monkeypatch.setattr(launcher, "wait_for_wgc_ready", wait_ready)
    popen, startfile = Mock(), Mock()
    result = launcher.launch_game(client="wgc", popen=popen, startfile=startfile,
                                  should_stop=lambda: True)
    assert not result.started
    assert "停止启动" in result.detail
    popen.assert_not_called()
    startfile.assert_not_called()
    wait_ready.assert_not_called()


def test_second_early_exit_stops_after_only_one_ready_wait(direct_install, monkeypatch):
    wait_ready = Mock(return_value=True)
    monkeypatch.setattr(launcher, "wait_for_wgc_ready", wait_ready)
    popen, startfile = Mock(side_effect=[exited_engine(), exited_engine(17)]), Mock()
    result = launcher.launch_game(client="wgc", popen=popen, startfile=startfile)
    assert not result.started
    assert "连续提前退出(17)" in result.detail
    assert popen.call_count == 2
    wait_ready.assert_called_once()
    assert not launcher.LAST_CLIENT_PATH.exists()
    startfile.assert_not_called()


def state_line(directory, *, can_start="true", forbidden="false", running="false",
               starting="none", condition="Ready(0)"):
    return ("[GameCtl] Game state detected as ["
            f"canStart = {can_start}, launchForbidden = {forbidden}, "
            f"running = {running}, processStarting = {starting}, condition = {condition}"
            f"] for [WOWS.WW.PRODUCTION|{directory}]\n")


@pytest.fixture
def backend_log(tmp_path, monkeypatch):
    center = tmp_path / "GameCenter" / "wgc.exe"
    logs = center.parent / "logs"
    logs.mkdir(parents=True)
    directory = tmp_path / "game"
    directory.mkdir()
    monkeypatch.setattr(readiness, "running_wgc_process_ids", lambda: {4242})
    monkeypatch.setattr(readiness, "_session_matches_process",
                        lambda *_args: True, raising=False)

    def write_log(body, *, pid=4242, name="wgc_current.log", modified=None):
        path = logs / name
        path.write_text(f"Starting log session {pid} at 2026-10-11\n" + body,
                        encoding="utf-8")
        if modified is not None:
            os.utime(path, (modified, modified))
        return path

    return center, directory, write_log


def test_backend_readiness_requires_initialization_of_live_session_and_target_game(backend_log):
    center, directory, write_log = backend_log
    write_log("GameCenter initialization finished\nPostload initialization\n"
              + state_line(directory))
    assert readiness.wgc_backend_ready(center, directory)


def test_initialization_and_ready_without_postload_cannot_authorize_retry(backend_log):
    center, directory, write_log = backend_log
    write_log("GameCenter initialization finished\n" + state_line(directory))
    assert not readiness.wgc_backend_ready(center, directory)


def test_ready_game_state_without_initialization_cannot_authorize_retry(backend_log):
    center, directory, write_log = backend_log
    write_log("Postload initialization\n" + state_line(directory))
    assert not readiness.wgc_backend_ready(center, directory)


def test_stale_pid_ready_log_cannot_authorize_retry(backend_log):
    center, directory, write_log = backend_log
    write_log("GameCenter initialization finished\nPostload initialization\n"
              + state_line(directory), pid=1234)
    assert not readiness.wgc_backend_ready(center, directory)


def test_no_running_backend_cannot_authorize_retry(backend_log, monkeypatch):
    center, directory, write_log = backend_log
    write_log("GameCenter initialization finished\nPostload initialization\n"
              + state_line(directory))
    monkeypatch.setattr(readiness, "running_wgc_process_ids", lambda: set())
    assert not readiness.wgc_backend_ready(center, directory)


def test_finished_log_session_cannot_authorize_retry(backend_log):
    center, directory, write_log = backend_log
    write_log("GameCenter initialization finished\nPostload initialization\n" + state_line(directory)
              + "Log session 4242 finished\n")
    assert not readiness.wgc_backend_ready(center, directory)


def test_other_installation_ready_state_cannot_authorize_retry(backend_log):
    center, directory, write_log = backend_log
    write_log("GameCenter initialization finished\nPostload initialization\n"
              + state_line(directory / "other"))
    assert not readiness.wgc_backend_ready(center, directory)


@pytest.mark.parametrize("blocked_state", [
    {"can_start": "false"}, {"forbidden": "true"}, {"running": "true"},
    {"starting": "9999"}, {"condition": "Updating(1)"},
])
def test_latest_blocked_game_state_overrides_previous_ready_state(backend_log, blocked_state):
    center, directory, write_log = backend_log
    write_log("GameCenter initialization finished\nPostload initialization\n" + state_line(directory)
              + state_line(directory, **blocked_state))
    assert not readiness.wgc_backend_ready(center, directory)


def test_newer_finished_session_overrides_older_ready_log_with_same_pid(backend_log):
    center, directory, write_log = backend_log
    body = "GameCenter initialization finished\nPostload initialization\n" + state_line(directory)
    write_log(body, name="wgc_old.log", modified=100)
    write_log(body + "Log session 4242 finished\n", name="wgc_new.log", modified=200)
    assert not readiness.wgc_backend_ready(center, directory)


def test_readiness_can_find_latest_target_state_in_large_log_tail(backend_log):
    center, directory, write_log = backend_log
    write_log("GameCenter initialization finished\nPostload initialization\n"
              + state_line(directory, can_start="false") + ("unrelated log entry\n" * 18000)
              + state_line(directory))
    assert readiness.wgc_backend_ready(center, directory)


def test_blocked_state_in_middle_of_large_log_is_not_hidden_by_noisy_tail(backend_log):
    center, directory, write_log = backend_log
    write_log("GameCenter initialization finished\nPostload initialization\n" + state_line(directory)
              + ("unrelated log entry\n" * 18000)
              + state_line(directory, can_start="false")
              + ("unrelated log entry\n" * 12000))
    assert not readiness.wgc_backend_ready(center, directory)


def test_reused_live_pid_does_not_authorize_an_old_log_session(backend_log, monkeypatch):
    center, directory, write_log = backend_log
    write_log("GameCenter initialization finished\nPostload initialization\n"
              + state_line(directory))
    matching = Mock(return_value=False)
    monkeypatch.setattr(readiness, "_session_matches_process", matching, raising=False)
    assert not readiness.wgc_backend_ready(center, directory)
    matching.assert_called_once()
    assert matching.call_args.args[0] == 4242
    assert matching.call_args.args[1] == center


def test_log_directory_comparison_accepts_case_and_separator_differences(backend_log):
    center, directory, write_log = backend_log
    recorded = str(directory).upper().replace("\\", "/") + "/"
    write_log("GameCenter initialization finished\nPostload initialization\n" + state_line(recorded))
    assert readiness.wgc_backend_ready(center, directory)


@pytest.fixture
def fake_clock(monkeypatch):
    now = {"time": 0.0}
    monkeypatch.setattr(readiness.time, "monotonic", lambda: now["time"])
    monkeypatch.setattr(readiness.time, "sleep", lambda amount:
                        now.__setitem__("time", now["time"] + amount))
    monkeypatch.setattr(readiness, "running_wgc_process_ids", lambda: {4242})
    return now


def test_wait_returns_only_after_ready_marker_without_real_sleep(fake_clock, monkeypatch):
    ready = Mock(side_effect=lambda *_args: fake_clock["time"] >= 0.5)
    monkeypatch.setattr(readiness, "wgc_backend_ready", ready)
    center, directory = Path("center/wgc.exe"), Path("game")
    assert readiness.wait_for_wgc_backend(center, directory, timeout=3)
    assert fake_clock["time"] == 2.5
    assert ready.call_count == 11


def test_wait_requires_postload_before_starting_settle_interval(
        backend_log, fake_clock, monkeypatch):
    center, directory, write_log = backend_log
    path = write_log("GameCenter initialization finished\n" + state_line(directory))
    read_ready = readiness.wgc_backend_ready
    observed = []

    def staged_ready(*args):
        if fake_clock["time"] == 0.5:
            with path.open("a", encoding="utf-8") as stream:
                stream.write("Postload initialization\n")
        result = read_ready(*args)
        observed.append((fake_clock["time"], result))
        return result

    monkeypatch.setattr(readiness, "wgc_backend_ready", staged_ready)
    assert readiness.wait_for_wgc_backend(center, directory, timeout=3)
    assert observed[:3] == [(0.0, False), (0.25, False), (0.5, True)]
    assert fake_clock["time"] == 2.5


def test_wait_settle_interval_can_be_injected_offline(fake_clock, monkeypatch):
    monkeypatch.setattr(readiness, "wgc_backend_ready", Mock(return_value=True))
    assert readiness.wait_for_wgc_backend(
        Path("center/wgc.exe"), Path("game"), timeout=1, stable_seconds=0.5)
    assert fake_clock["time"] == 0.5


def test_wait_ready_flap_restarts_the_entire_settle_interval(fake_clock, monkeypatch):
    ready = Mock(side_effect=lambda *_args: fake_clock["time"] != 0.75)
    monkeypatch.setattr(readiness, "wgc_backend_ready", ready)
    assert readiness.wait_for_wgc_backend(Path("center/wgc.exe"), Path("game"), timeout=4)
    assert fake_clock["time"] == 3.0


def test_wait_pid_set_change_restarts_the_entire_settle_interval(fake_clock, monkeypatch):
    monkeypatch.setattr(readiness, "wgc_backend_ready", Mock(return_value=True))
    monkeypatch.setattr(readiness, "running_wgc_process_ids", lambda:
                        {4242} if fake_clock["time"] < 1.0 else {5555})
    assert readiness.wait_for_wgc_backend(Path("center/wgc.exe"), Path("game"), timeout=4)
    assert fake_clock["time"] == 3.0


def test_wait_pid_change_during_read_cannot_start_settle_interval(fake_clock, monkeypatch):
    backend = {"pid": 4242}
    monkeypatch.setattr(readiness, "running_wgc_process_ids", lambda: {backend["pid"]})

    def ready_after_restart(*_args):
        backend["pid"] = 5555
        return True

    monkeypatch.setattr(readiness, "wgc_backend_ready", ready_after_restart)
    assert readiness.wait_for_wgc_backend(Path("center/wgc.exe"), Path("game"), timeout=3)
    assert fake_clock["time"] == 2.25


def test_wait_backend_disappearance_restarts_settle_interval(fake_clock, monkeypatch):
    monkeypatch.setattr(readiness, "wgc_backend_ready", Mock(return_value=True))
    monkeypatch.setattr(readiness, "running_wgc_process_ids", lambda:
                        set() if fake_clock["time"] == 0.75 else {4242})
    assert readiness.wait_for_wgc_backend(Path("center/wgc.exe"), Path("game"), timeout=4)
    assert fake_clock["time"] == 3.0


def test_wait_stable_readiness_cannot_extend_timeout_budget(fake_clock, monkeypatch):
    monkeypatch.setattr(readiness, "wgc_backend_ready", Mock(return_value=True))
    assert not readiness.wait_for_wgc_backend(
        Path("center/wgc.exe"), Path("game"), timeout=1)
    assert fake_clock["time"] == 1.0


def test_wait_timeout_is_bounded_without_real_sleep(fake_clock, monkeypatch):
    ready = Mock(return_value=False)
    monkeypatch.setattr(readiness, "wgc_backend_ready", ready)
    assert not readiness.wait_for_wgc_backend(Path("center/wgc.exe"), Path("game"), timeout=0.6)
    assert fake_clock["time"] == pytest.approx(0.6)
    assert ready.call_count == 3


def test_wait_checks_stop_before_reading_backend_again(fake_clock, monkeypatch):
    ready = Mock(return_value=False)
    monkeypatch.setattr(readiness, "wgc_backend_ready", ready)
    assert not readiness.wait_for_wgc_backend(Path("center/wgc.exe"), Path("game"), timeout=30,
                                             should_stop=lambda: fake_clock["time"] >= 0.25)
    assert fake_clock["time"] == 0.25
    assert ready.call_count == 1


def test_wait_cancellation_during_settling_stops_without_retry(fake_clock, monkeypatch):
    ready = Mock(return_value=True)
    monkeypatch.setattr(readiness, "wgc_backend_ready", ready)
    assert not readiness.wait_for_wgc_backend(
        Path("center/wgc.exe"), Path("game"), timeout=30,
        should_stop=lambda: fake_clock["time"] >= 0.75)
    assert fake_clock["time"] == 0.75
    assert ready.call_count == 3


def test_wait_cancellation_during_log_read_wins_over_ready(fake_clock, monkeypatch):
    stop = {"requested": False}

    def cancel_and_ready(*_args):
        stop["requested"] = True
        return True

    monkeypatch.setattr(readiness, "wgc_backend_ready", cancel_and_ready)
    assert not readiness.wait_for_wgc_backend(
        Path("center/wgc.exe"), Path("game"), timeout=30, stable_seconds=0,
        should_stop=lambda: stop["requested"])
    assert fake_clock["time"] == 0.0


def test_missing_center_path_does_not_start_or_wait_for_backend(monkeypatch):
    monkeypatch.setattr(launcher, "find_wgc_executable", lambda: None)
    wait = Mock()
    monkeypatch.setattr(launcher, "wait_for_wgc_backend", wait)
    assert not launcher.wait_for_wgc_ready(Path("game"))
    wait.assert_not_called()
