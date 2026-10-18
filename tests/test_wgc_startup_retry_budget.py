"""Bounded native-engine recovery, with no real processes or launcher calls."""

from types import SimpleNamespace
from unittest.mock import Mock
import subprocess

import pytest

from core import launcher


class FakeClock:
    def __init__(self):
        self.now = 0.0
        self.sleeps = []

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        assert seconds >= 0
        self.sleeps.append(seconds)
        self.now += seconds


class FakeEngine:
    def __init__(self, clock, *, code=None, exit_after=0.0):
        self.clock = clock
        self.code = code
        self.exit_after = exit_after
        self.wait_timeouts = []
        self.poll = Mock(side_effect=AssertionError("No process polling is needed"))

    def wait(self, *, timeout):
        self.wait_timeouts.append(timeout)
        if self.code is None or self.exit_after > timeout:
            self.clock.sleep(timeout)
            raise subprocess.TimeoutExpired("fake_native_engine", timeout)
        self.clock.sleep(self.exit_after)
        return self.code


@pytest.fixture
def startup(tmp_path, monkeypatch):
    clock = FakeClock()
    directory = tmp_path.resolve()
    native = directory / "bin" / "13357625" / "bin64" / "WorldOfWarships64.exe"
    (directory / "app_type.xml").write_text(
        "<protocol><app_type>high</app_type></protocol>", encoding="utf-8")
    monkeypatch.setattr(launcher.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(launcher.time, "sleep", clock.sleep)
    monkeypatch.setattr(launcher, "find_wgc_game_executable", Mock(return_value=native))
    monkeypatch.setattr(launcher, "wgc_game_directory", Mock(return_value=directory))
    monkeypatch.setattr(launcher, "resolve_wgc_version", Mock(return_value=SimpleNamespace(
        executable=native, detail="测试区服当前版本")))
    # Fail if recovery ever tries to discover or dispatch a launcher instead.
    monkeypatch.setattr(launcher, "find_wgc_executable", Mock(
        side_effect=AssertionError("Must use the mocked readiness check")))
    monkeypatch.setattr(launcher, "find_steam_executable", Mock(
        side_effect=AssertionError("WG recovery must not fall back to Steam")))
    monkeypatch.setattr(launcher.subprocess, "Popen", Mock(
        side_effect=AssertionError("Real process dispatch is forbidden")))
    remember = Mock()
    ready = Mock(return_value=True)
    startfile = Mock(side_effect=AssertionError("Must never open a launcher URI"))
    monkeypatch.setattr(launcher, "remember_launch_client", remember)
    monkeypatch.setattr(launcher, "wait_for_wgc_ready", ready)

    def run(engines, *, should_stop=None):
        popen = Mock(side_effect=engines)
        result = launcher.launch_game(
            client="wgc", popen=popen, startfile=startfile, should_stop=should_stop)
        startfile.assert_not_called()
        for call in popen.call_args_list:
            assert call.args[0] == [str(native), "--apptype=high"]
            assert call.kwargs["cwd"] == str(directory)
        return result, popen

    return SimpleNamespace(
        clock=clock, directory=directory, native=native,
        ready=ready, remember=remember, run=run)


@pytest.mark.parametrize("second_code", [-1, 0, 0xFFFFFFFF])
def test_cold_handshake_can_recover_on_third_native_dispatch(startup, second_code):
    first = FakeEngine(startup.clock, code=-1)
    second = FakeEngine(startup.clock, code=second_code)
    running = FakeEngine(startup.clock)

    result, popen = startup.run([first, second, running])

    assert result.started
    assert result.process is running
    assert result.method == "wgc_direct"
    assert "已重试2次" in result.detail
    assert popen.call_count == 3
    assert startup.ready.call_count == 2
    assert startup.clock.now == pytest.approx(7.0)  # 3 s retry delay + 4 s live check.
    assert max(startup.clock.sleeps) <= 4.0
    startup.remember.assert_called_once_with("wgc")


def test_continuous_minus_one_exits_stop_after_three_native_dispatches(startup):
    engines = [FakeEngine(startup.clock, code=-1) for _ in range(3)]

    result, popen = startup.run(engines)

    assert not result.started
    assert result.process is None
    assert "连续提前退出(-1)" in result.detail
    assert popen.call_count == 3
    assert startup.ready.call_count == 2
    assert startup.clock.now == pytest.approx(3.0)
    startup.remember.assert_not_called()


def test_all_early_checks_and_backend_waits_share_the_45_second_budget(startup):
    first = FakeEngine(startup.clock, code=-1, exit_after=4.0)
    second = FakeEngine(startup.clock, code=-1, exit_after=4.0)
    running = FakeEngine(startup.clock)
    backend_timeouts = []

    def consume_remaining_backend_budget(directory, *, timeout, should_stop):
        assert directory == startup.directory
        backend_timeouts.append(timeout)
        startup.clock.sleep(timeout)
        return True

    startup.ready.side_effect = consume_remaining_backend_budget
    result, popen = startup.run([first, second, running])

    assert not result.started
    assert "45秒" in result.detail
    assert popen.call_count == 2
    assert backend_timeouts == pytest.approx([30.0, 7.0])
    assert first.wait_timeouts == [4.0]
    assert second.wait_timeouts == [4.0]
    assert running.wait_timeouts == []
    assert startup.clock.now == pytest.approx(45.0)
    startup.remember.assert_not_called()


def test_retry_delay_cannot_dispatch_after_consuming_the_remaining_budget(startup):
    first = FakeEngine(startup.clock, code=-1, exit_after=4.0)
    second = FakeEngine(startup.clock, code=-1, exit_after=4.0)
    running = FakeEngine(startup.clock)
    backend_durations = iter([30.0, 5.0])

    def settle_backend(directory, *, timeout, should_stop):
        elapsed = next(backend_durations)
        assert elapsed <= timeout
        startup.clock.sleep(elapsed)
        return True

    startup.ready.side_effect = settle_backend
    result, popen = startup.run([first, second, running])

    assert not result.started
    assert "45秒" in result.detail
    assert popen.call_count == 2
    assert startup.clock.now == pytest.approx(45.0)
    assert sum(startup.clock.sleeps[4:]) == pytest.approx(2.0)
    assert running.wait_timeouts == []


def test_cancel_during_retry_delay_prevents_third_native_dispatch(startup):
    engines = [FakeEngine(startup.clock, code=-1) for _ in range(2)]
    running = FakeEngine(startup.clock)
    should_stop = lambda: startup.clock.now >= 0.5

    result, popen = startup.run([*engines, running], should_stop=should_stop)

    assert not result.started
    assert "按用户要求停止启动" in result.detail
    assert popen.call_count == 2
    assert startup.ready.call_count == 2
    assert startup.clock.now == pytest.approx(0.5)
    assert running.wait_timeouts == []
    startup.remember.assert_not_called()


@pytest.mark.parametrize("first_code", [-1, 0, 17])
def test_second_fatal_exit_stops_without_third_dispatch_or_delay(startup, first_code):
    first = FakeEngine(startup.clock, code=first_code)
    fatal = FakeEngine(startup.clock, code=17)
    running = FakeEngine(startup.clock)

    result, popen = startup.run([first, fatal, running])

    assert not result.started
    assert "连续提前退出(17)" in result.detail
    assert popen.call_count == 2
    startup.ready.assert_called_once()
    assert startup.clock.now == 0.0
    assert running.wait_timeouts == []
    startup.remember.assert_not_called()


@pytest.mark.parametrize("early_exits", [0, 1, 2])
def test_live_native_handle_is_returned_without_further_dispatch(startup, early_exits):
    exited = [FakeEngine(startup.clock, code=-1) for _ in range(early_exits)]
    running = FakeEngine(startup.clock)

    result, popen = startup.run([*exited, running])

    assert result.started
    assert result.process is running
    assert popen.call_count == early_exits + 1
    assert startup.ready.call_count == early_exits
    assert running.wait_timeouts == [4.0]
    for engine in [*exited, running]:
        engine.poll.assert_not_called()


def test_cancellation_before_first_dispatch_starts_no_process(startup):
    result, popen = startup.run([FakeEngine(startup.clock)], should_stop=lambda: True)

    assert not result.started
    assert "按用户要求停止启动" in result.detail
    popen.assert_not_called()
    startup.ready.assert_not_called()
    startup.remember.assert_not_called()


def test_backend_not_ready_stops_before_another_native_dispatch(startup):
    startup.ready.return_value = False
    result, popen = startup.run([
        FakeEngine(startup.clock, code=-1), FakeEngine(startup.clock)])

    assert not result.started
    assert "未稳定就绪" in result.detail
    assert popen.call_count == 1
    startup.ready.assert_called_once()
    startup.remember.assert_not_called()
