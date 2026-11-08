"""Per-step OCR/capture latency tracking for live troubleshooting."""

from __future__ import annotations

import functools
import logging
import threading
import time

logger = logging.getLogger("perf")

_LOCK = threading.Lock()
_STATS: dict[str, dict[str, float]] = {}
_LAST_SUMMARY = 0.0
_SUMMARY_INTERVAL = 30.0
_SLOW_SAMPLE_MS = 250.0


def record(label: str, seconds: float, *, slow_ms: float = _SLOW_SAMPLE_MS) -> None:
    """Record one step duration and log immediately when it is slow."""
    elapsed_ms = max(0.0, float(seconds) * 1000.0)
    with _LOCK:
        stat = _STATS.setdefault(
            label,
            {"count": 0.0, "total_ms": 0.0, "max_ms": 0.0, "last_ms": 0.0},
        )
        stat["count"] += 1
        stat["total_ms"] += elapsed_ms
        stat["max_ms"] = max(stat["max_ms"], elapsed_ms)
        stat["last_ms"] = elapsed_ms
    if elapsed_ms >= slow_ms:
        logger.info("[OCR延迟] %s %.0fms", label, elapsed_ms)


def timed(label: str, *, slow_ms: float = _SLOW_SAMPLE_MS):
    """Measure a callable without changing its signature or return value."""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            started = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                record(label, time.perf_counter() - started, slow_ms=slow_ms)

        return wrapper

    return decorator


def maybe_summary(*, force: bool = False, reset: bool = True) -> str | None:
    """Emit one compact latency summary at most every 30 seconds."""
    global _LAST_SUMMARY
    now = time.monotonic()
    with _LOCK:
        if not force and now - _LAST_SUMMARY < _SUMMARY_INTERVAL:
            return None
        if not _STATS:
            _LAST_SUMMARY = now
            return None
        items = sorted(
            _STATS.items(),
            key=lambda item: (item[1]["max_ms"], item[1]["total_ms"]),
            reverse=True,
        )[:10]
        parts = []
        for label, stat in items:
            count = max(1.0, stat["count"])
            parts.append(
                f"{label} n={int(stat['count'])} "
                f"avg={stat['total_ms'] / count:.0f}ms "
                f"max={stat['max_ms']:.0f}ms "
                f"last={stat['last_ms']:.0f}ms"
            )
        if reset:
            _STATS.clear()
        _LAST_SUMMARY = now
    text = " | ".join(parts)
    logger.info("[OCR延迟汇总] %s", text)
    return text


def reset() -> None:
    """Clear accumulated samples (used by tests and task boundaries)."""
    global _LAST_SUMMARY
    with _LOCK:
        _STATS.clear()
        _LAST_SUMMARY = 0.0
