from __future__ import annotations

import time

from core import perf


def setup_function():
    perf.reset()


def test_records_timed_call_and_returns_original_value():
    @perf.timed("unit.step", slow_ms=0)
    def work(value):
        return value * 2

    assert work(3) == 6
    summary = perf.maybe_summary(force=True)
    assert summary is not None
    assert "unit.step n=1" in summary
    assert "last=" in summary


def test_summary_is_throttled_until_forced():
    perf.record("unit.throttle", 0.01, slow_ms=1000)
    assert perf.maybe_summary() is not None
    perf.record("unit.throttle", 0.01, slow_ms=1000)
    assert perf.maybe_summary() is None
    assert perf.maybe_summary(force=True) is not None


def test_slow_call_is_recorded_even_without_summary():
    started = time.perf_counter()
    @perf.timed("unit.slow", slow_ms=0)
    def work():
        return "ok"

    assert work() == "ok"
    summary = perf.maybe_summary(force=True)
    assert "unit.slow n=1" in summary
    assert time.perf_counter() >= started
