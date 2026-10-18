"""Startup observes the dispatched engine without launching any real process."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import main
from core.launcher import LaunchResult


GAME_WINDOW = (42, "World of Warships", (0, 0, 1920, 1080))


@pytest.fixture
def clock(monkeypatch):
    clock = SimpleNamespace(now=1000.0, sleeps=[])

    def sleep(seconds):
        assert seconds >= 0
        clock.sleeps.append(seconds)
        clock.now += seconds

    monkeypatch.setattr(main.time, "monotonic", lambda: clock.now)
    monkeypatch.setattr(main.time, "sleep", sleep)
    return clock


@pytest.fixture(autouse=True)
def forbid_real_startup(monkeypatch):
    # Each test must supply its own window observations and launch result.
    monkeypatch.setattr(main, "find_game_window", Mock(return_value=[]))
    launch = Mock(side_effect=AssertionError("Real startup is forbidden in this test"))
    monkeypatch.setattr(main, "launch_game", launch)
    monkeypatch.setattr(main, "BattleBot", Mock(side_effect=AssertionError("No game control")))
    return launch


@pytest.mark.parametrize("exit_code", [0, 1, -1, 3221225477])
def test_engine_exit_after_four_seconds_ends_three_hundred_second_wait(
    clock, monkeypatch, caplog, exit_code
):
    process = Mock()
    process.poll.side_effect = lambda: exit_code if clock.now >= 1004.0 else None
    find = Mock(return_value=[])
    monkeypatch.setattr(main, "find_game_window", find)

    assert main.wait_for_game_window(timeout=300, launch_process=process) is None

    assert clock.now == 1004.0
    assert clock.sleeps == [1.0] * 4
    # Once exited, the wait does not inspect an unrelated or stale window.
    assert find.call_count == 4
    assert str(exit_code) in caplog.text
    assert any(record.levelname == "ERROR" for record in caplog.records)


def test_already_exited_engine_is_rejected_before_window_search(clock, monkeypatch):
    process = Mock()
    process.poll.return_value = 0
    find = Mock(return_value=[GAME_WINDOW])
    monkeypatch.setattr(main, "find_game_window", find)

    assert main.wait_for_game_window(timeout=300, launch_process=process) is None

    find.assert_not_called()
    assert clock.sleeps == []


def test_engine_exiting_during_window_search_does_not_accept_stale_window(
    clock, monkeypatch, caplog
):
    process = Mock()
    process.poll.side_effect = [None, 1]
    find = Mock(return_value=[GAME_WINDOW])
    monkeypatch.setattr(main, "find_game_window", find)

    assert main.wait_for_game_window(timeout=300, launch_process=process) is None

    find.assert_called_once_with()
    assert process.poll.call_count == 2
    assert clock.sleeps == []
    assert any(record.levelname == "ERROR" for record in caplog.records)


def test_live_engine_returns_window_only_after_it_appears(clock, monkeypatch):
    process = Mock()
    process.poll.return_value = None
    find = Mock(side_effect=lambda: [GAME_WINDOW] if clock.now >= 1003.0 else [])
    monkeypatch.setattr(main, "find_game_window", find)

    assert main.wait_for_game_window(timeout=300, launch_process=process) == GAME_WINDOW

    assert clock.now == 1003.0
    assert find.call_count == 4
    # The fourth observation gets an additional liveness check after enumeration.
    assert process.poll.call_count == 5


def test_without_process_preserves_existing_window_lookup(clock, monkeypatch):
    find = Mock(return_value=[GAME_WINDOW])
    monkeypatch.setattr(main, "find_game_window", find)

    assert main.wait_for_game_window() == GAME_WINDOW

    find.assert_called_once_with()
    assert clock.sleeps == []


def test_without_process_preserves_delayed_steam_window_lookup(clock, monkeypatch):
    find = Mock(side_effect=lambda: [GAME_WINDOW] if clock.now >= 1002.0 else [])
    monkeypatch.setattr(main, "find_game_window", find)

    assert main.wait_for_game_window(timeout=60) == GAME_WINDOW

    assert clock.now == 1002.0
    assert find.call_count == 3


def test_live_engine_without_window_observes_bounded_timeout(clock):
    process = Mock()
    process.poll.return_value = None

    assert main.wait_for_game_window(
        timeout=3.5, poll_interval=1.0, launch_process=process
    ) is None

    assert clock.now == 1003.5
    assert clock.sleeps == [1.0, 1.0, 1.0, 0.5]


def test_stop_before_observation_never_dispatches_or_queries_process(
    clock, monkeypatch, forbid_real_startup
):
    process = Mock()
    find = Mock(return_value=[GAME_WINDOW])
    monkeypatch.setattr(main, "find_game_window", find)

    assert main.wait_for_game_window(
        timeout=300, should_stop=lambda: True, launch_process=process
    ) is None

    forbid_real_startup.assert_not_called()
    find.assert_not_called()
    process.poll.assert_not_called()
    assert clock.sleeps == []


def test_stop_during_wait_does_not_poll_or_dispatch_again(
    clock, monkeypatch, forbid_real_startup
):
    process = Mock()
    process.poll.return_value = None
    find = Mock(return_value=[])
    monkeypatch.setattr(main, "find_game_window", find)

    assert main.wait_for_game_window(
        timeout=300,
        should_stop=lambda: clock.now >= 1002.0,
        launch_process=process,
    ) is None

    assert clock.now == 1002.0
    assert process.poll.call_count == 2
    assert find.call_count == 2
    forbid_real_startup.assert_not_called()


def test_launch_result_retains_process_without_changing_value_or_repr():
    process = Mock()
    expected = LaunchResult(True, "wgc_direct", "dispatched")
    result = LaunchResult(True, "wgc_direct", "dispatched", process=process)

    assert expected.process is None
    assert result.process is process
    assert result == expected
    assert repr(result) == repr(expected)


@pytest.fixture
def startup_entry(monkeypatch):
    limits = SimpleNamespace(quick_battle=False, stop_requested=lambda: False)
    reporter = Mock()
    monkeypatch.setattr(main, "configure_dpi_awareness", Mock())
    monkeypatch.setattr(main, "configure_logging", Mock())
    monkeypatch.setattr(main, "ship_key_from_env", lambda: "test_ship")
    monkeypatch.setattr(
        main,
        "load_ship_config",
        lambda _: {"name": "test_ship", "secondary": {"range": 6}},
    )
    monkeypatch.setattr(main.RunLimits, "from_env", lambda: limits)
    monkeypatch.setattr(main, "RuntimeReporter", Mock(return_value=reporter))
    monkeypatch.setattr(main, "last_launch_client", lambda: "wgc")
    monkeypatch.setenv("WOWS_MODE", "cooperative")
    monkeypatch.setenv("WOWS_LAUNCHER_CLIENT", "wgc")
    monkeypatch.setenv("WOWS_GAME_LAUNCH_TIMEOUT", "300")
    return SimpleNamespace(limits=limits, reporter=reporter)


def test_runtime_tracks_dispatched_engine_and_reports_exit_instead_of_window_timeout(
    clock, monkeypatch, startup_entry
):
    process = Mock()
    process.poll.side_effect = lambda: 1 if clock.now >= 1006.0 else None
    result = LaunchResult(True, "wgc_direct", "dispatched", process=process)
    launch = Mock(return_value=result)
    monkeypatch.setattr(main, "launch_game", launch)

    assert main.run() == 1

    # Two seconds to look for an already running game, then four to engine exit.
    assert clock.now == 1006.0
    launch.assert_called_once_with(
        client="wgc", should_stop=startup_entry.limits.stop_requested
    )
    final = startup_entry.reporter.update.call_args
    assert final.args[0] == "failed"
    assert "1" in final.args[1]
    assert final.kwargs == {"error": "game_process_exited"}
    main.BattleBot.assert_not_called()


def test_runtime_stop_before_dispatch_does_not_start_new_process(
    clock, startup_entry, forbid_real_startup
):
    startup_entry.limits.stop_requested = lambda: True

    assert main.run() == 0

    forbid_real_startup.assert_not_called()
    assert startup_entry.reporter.update.call_args.args[0] == "stopped"
    assert clock.sleeps == []


def test_runtime_stop_during_engine_wait_never_retries_dispatch(
    clock, monkeypatch, startup_entry
):
    process = Mock()
    process.poll.return_value = None
    launch = Mock(return_value=LaunchResult(True, "wgc_direct", "dispatched", process=process))
    monkeypatch.setattr(main, "launch_game", launch)
    startup_entry.limits.stop_requested = lambda: clock.now >= 1004.0

    assert main.run() == 0

    assert clock.now == 1004.0
    assert launch.call_count == 1
    assert startup_entry.reporter.update.call_args.args[0] == "stopped"
    main.BattleBot.assert_not_called()


def test_runtime_steam_without_process_keeps_window_timeout_failure(
    clock, monkeypatch, startup_entry
):
    monkeypatch.setenv("WOWS_LAUNCHER_CLIENT", "steam")
    monkeypatch.setenv("WOWS_GAME_LAUNCH_TIMEOUT", "3")
    launch = Mock(return_value=LaunchResult(True, "steam", "dispatched"))
    monkeypatch.setattr(main, "launch_game", launch)

    assert main.run() == 1

    assert clock.now == 1005.0
    final = startup_entry.reporter.update.call_args
    assert final.args[0] == "failed"
    assert final.kwargs == {"error": "game_window_not_found"}
    launch.assert_called_once_with(
        client="steam", should_stop=startup_entry.limits.stop_requested
    )
    main.BattleBot.assert_not_called()
