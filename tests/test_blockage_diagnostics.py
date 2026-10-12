import json
from unittest.mock import patch

import numpy as np

from core.blockage_diagnostics import BlockageTrace


def frame(value=0):
    return np.full((24, 32, 3), value, dtype=np.uint8)


def observations(trace):
    return [json.loads(line) for line in
            (trace.folder / "observations.jsonl").read_text(encoding="utf-8").splitlines()]


def test_critical_pair_and_result_survive_sampling_budget_and_throttle(tmp_path):
    trace = BlockageTrace(tmp_path, "blocked", maximum=3, interval=100)
    with patch("core.blockage_diagnostics.time.monotonic", return_value=10):
        assert trace.record(frame(), state="unknown", action="waiting", attempt=0)
        assert not trace.record(frame(), state="unknown", action="waiting", attempt=1)
        for phase, state in (("before", "popup"), ("after", "port")):
            assert trace.record(frame(), state=state, action="escape", attempt=2, phase=phase)
        assert trace.finish(frame(), state="port", attempt=2, outcome="recovered")
    rows = observations(trace)
    assert [row["phase"] for row in rows] == ["before", "after", "result"]
    assert rows[0]["pair_id"] == rows[1]["pair_id"] == "escape:2"
    assert rows[-1]["details"]["outcome"] == "recovered"
    assert trace.count == len(list(trace.folder.glob("frame_*.jpg"))) == 3


def test_old_pairs_are_evicted_together_to_retain_latest_attempt(tmp_path):
    trace = BlockageTrace(tmp_path, "blocked", maximum=3, interval=100)
    for attempt in range(6):
        for phase in ("before", "after"):
            assert trace.record(frame(), state="popup", action="escape", attempt=attempt, phase=phase)
        assert trace.finish(frame(), state="popup", attempt=attempt, outcome="still_blocked")
        rows = observations(trace)
        assert [row["phase"] for row in rows] == ["before", "after", "result"]
        assert all(row["attempt"] == attempt for row in rows)
        assert trace.count == len(list(trace.folder.glob("frame_*.jpg"))) == 3


def test_observation_defaults_retain_more_than_previous_nine_frame_limit(tmp_path):
    trace = BlockageTrace(tmp_path, "blocked", interval=0)
    for attempt in range(12):
        assert trace.record(frame(), state="unknown", action="waiting", attempt=attempt)
    assert not trace.record(frame(), state="unknown", action="waiting", attempt=12)
    assert trace.count == 12
    assert trace.maximum == 60


def test_context_keeps_latest_frames_and_original_capture_metadata(tmp_path):
    trace = BlockageTrace(tmp_path, "blocked", interval=0.5, context_frames=2)
    with patch("core.blockage_diagnostics.time.monotonic", side_effect=[1, 1.2, 2, 3, 4]):
        assert trace.remember(frame(1), state="port", attempt=1)
        assert not trace.remember(frame(2), state="port", attempt=2)
        assert trace.remember(frame(3), state="port", attempt=3)
        assert trace.remember(frame(4), state="popup", attempt=4)
        assert trace.folder is None
        assert trace.record(frame(5), state="popup", action="escape", attempt=5, phase="before")
    rows = observations(trace)
    assert [row["phase"] for row in rows] == ["context", "context", "before"]
    assert [row["attempt"] for row in rows] == [3, 4, 5]
    assert [row["elapsed"] for row in rows] == [0, 1, 2]


def test_disabled_trace_does_not_copy_or_create_files(tmp_path):
    trace = BlockageTrace(None, "blocked")
    assert not trace.remember(frame(), state="port")
    assert not trace.record(frame(), state="popup", action="escape", attempt=1, phase="before")
    assert not trace.finish(frame(), state="port", outcome="recovered")
    assert trace.folder is None
    assert not list(tmp_path.iterdir())


def test_screenshot_failure_does_not_break_recovery(tmp_path):
    trace = BlockageTrace(tmp_path, "blocked")
    with patch("core.blockage_diagnostics.cv2.imwrite", return_value=False):
        assert not trace.record(frame(), state="popup", action="escape", attempt=1, phase="before")
    assert trace.count == 0
    assert trace.record(frame(), state="port", action="escape", attempt=1, phase="after")


def test_failed_write_keeps_previous_pair_at_capacity(tmp_path):
    trace = BlockageTrace(tmp_path, "blocked", maximum=3)
    for phase in ("before", "after"):
        assert trace.record(frame(), state="popup", action="escape", attempt=1, phase=phase)
    assert trace.finish(frame(), state="popup", attempt=1, outcome="blocked")
    before_failure = observations(trace)
    with patch("core.blockage_diagnostics.cv2.imwrite", return_value=False):
        assert not trace.record(frame(), state="popup", action="escape", attempt=2, phase="before")
    assert observations(trace) == before_failure
    assert trace.count == len(list(trace.folder.glob("frame_*.jpg"))) == 3


def test_label_cannot_escape_evidence_root(tmp_path):
    trace = BlockageTrace(tmp_path, "../../blocked:/escape")
    assert trace.record(frame(), state="popup", action="escape", attempt=1, phase="before")
    assert trace.folder.parent == tmp_path
    assert trace.folder.name.startswith("blocked__escape_")
