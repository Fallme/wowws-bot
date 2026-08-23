"""Structured runtime events for live diagnostics and offline replay.

The event stream is deliberately independent from Python logging: logs are for
people, while these JSON Lines records are a stable machine-readable callback
surface modelled after MAA's task/message separation.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable


@dataclass(frozen=True)
class RuntimeEvent:
    sequence: int
    topic: str
    emitted_at: float
    payload: dict[str, Any]


def _json_value(value: Any) -> Any:
    if is_dataclass(value):
        return _json_value(asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


class EventBus:
    """Publish typed topics to local callbacks without coupling producers."""

    def __init__(self):
        self._subscribers: list[Callable[[RuntimeEvent], None]] = []
        self._sequence = 0
        self._lock = threading.Lock()

    def subscribe(self, callback: Callable[[RuntimeEvent], None]):
        with self._lock:
            self._subscribers.append(callback)

    def publish(self, topic: str, **payload: Any) -> RuntimeEvent:
        with self._lock:
            self._sequence += 1
            event = RuntimeEvent(
                sequence=self._sequence,
                topic=topic,
                emitted_at=time.time(),
                payload=_json_value(payload),
            )
            subscribers = tuple(self._subscribers)
        for callback in subscribers:
            callback(event)
        return event


class JsonlEventRecorder:
    """Append events to a replayable JSONL file and flush every record."""

    def __init__(self, destination: Path):
        self.destination = Path(destination)
        self.destination.parent.mkdir(parents=True, exist_ok=True)
        self._stream = self.destination.open("a", encoding="utf-8", buffering=1)
        self._lock = threading.Lock()
        self._closed = False

    def __call__(self, event: RuntimeEvent):
        record = {
            "sequence": event.sequence,
            "topic": event.topic,
            "emitted_at": event.emitted_at,
            "payload": event.payload,
        }
        line = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
        with self._lock:
            if not self._closed:
                self._stream.write(line + "\n")

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._stream.close()


def read_events(source: Path) -> list[RuntimeEvent]:
    events = []
    with Path(source).open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            raw = json.loads(line)
            events.append(
                RuntimeEvent(
                    sequence=int(raw["sequence"]),
                    topic=str(raw["topic"]),
                    emitted_at=float(raw["emitted_at"]),
                    payload=dict(raw.get("payload") or {}),
                )
            )
    return events
