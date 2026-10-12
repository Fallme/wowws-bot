"""Colour-sensitive diagnostics retain the captured pixels and bounded pairs."""

import json
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from bot import BattleBot
from core.blockage_diagnostics import BlockageTrace
from main import blockage_trace


def pixels(seed):
    return np.random.default_rng(seed).integers(0, 256, (31, 47, 3), dtype=np.uint8)


def rows(trace):
    return [json.loads(line) for line in
            (trace.folder / "observations.jsonl").read_text(encoding="utf-8").splitlines()]


def assert_pixels(trace, row, expected):
    assert row["frame"].endswith(".png")
    restored = cv2.imread(str(trace.folder / row["frame"]), cv2.IMREAD_UNCHANGED)
    np.testing.assert_array_equal(restored, expected)


def test_lossless_evidence_preserves_every_colour_channel_and_latest_pair_at_capacity(tmp_path):
    trace = BlockageTrace(tmp_path, "colour_threshold", maximum=3, interval=100, lossless=True)
    for attempt in range(5):
        frames = [pixels(attempt * 3 + index) for index in range(3)]
        for phase, image in zip(("before", "after"), frames):
            assert trace.record(image, state="results", action="ocr_retry", attempt=attempt, phase=phase)
        assert trace.finish(frames[2], state="results", attempt=attempt, outcome="unrecognized")
        retained = rows(trace)
        assert [row["phase"] for row in retained] == ["before", "after", "result"]
        assert [row["attempt"] for row in retained] == [attempt] * 3
        assert retained[0]["pair_id"] == retained[1]["pair_id"] == f"ocr_retry:{attempt}"
        for row, expected in zip(retained, frames):
            assert_pixels(trace, row, expected)
        assert trace.count == len(list(trace.folder.glob("frame_*.png"))) == 3
        assert not list(trace.folder.glob("*.jpg"))
        assert not (trace.folder / "pending_frame.png").exists()


def test_lossless_context_retains_original_pixels_when_the_capture_buffer_changes(tmp_path):
    trace = BlockageTrace(tmp_path, "context", context_frames=1, interval=0, lossless=True)
    before = pixels(100)
    expected_before = before.copy()
    assert trace.remember(before, state="battle", attempt=1)
    before[:] = 0
    after = pixels(101)
    assert trace.record(after, state="results", action="capture", attempt=2, phase="before")
    retained = rows(trace)
    assert [row["phase"] for row in retained] == ["context", "before"]
    assert_pixels(trace, retained[0], expected_before)
    assert_pixels(trace, retained[1], after)


def test_failed_lossless_write_preserves_the_previous_pair_and_cleans_pending(tmp_path):
    trace = BlockageTrace(tmp_path, "failed_write", maximum=3, lossless=True)
    for phase in ("before", "after"):
        assert trace.record(pixels(102), state="results", action="ocr", attempt=1, phase=phase)
    assert trace.finish(pixels(103), state="results", attempt=1)
    retained = rows(trace)
    files = {row["frame"]: (trace.folder / row["frame"]).read_bytes() for row in retained}

    def failed_write(path, image, options):
        assert path.endswith("pending_frame.png")
        assert options == [int(cv2.IMWRITE_PNG_COMPRESSION), 1]
        # Simulate a partial write as well as the OpenCV failure result.
        with open(path, "wb") as pending:
            pending.write(b"incomplete")
        return False

    with patch("core.blockage_diagnostics.cv2.imwrite", side_effect=failed_write):
        assert not trace.record(pixels(104), state="results", action="ocr", attempt=2, phase="before")
    assert rows(trace) == retained
    assert files == {name: (trace.folder / name).read_bytes() for name in files}
    assert not (trace.folder / "pending_frame.png").exists()


def test_default_trace_keeps_jpeg_encoding_for_existing_callers(tmp_path):
    trace = BlockageTrace(tmp_path, "default")
    original = pixels(105)
    assert trace.record(original, state="unknown", action="capture", attempt=0)
    saved = rows(trace)[0]
    path = trace.folder / saved["frame"]
    assert path.suffix == ".jpg"
    assert path.read_bytes().startswith(b"\xff\xd8")
    assert not np.array_equal(cv2.imread(str(path)), original)


def test_only_rewards_choose_lossless_in_the_general_recovery_factory(tmp_path):
    bot = SimpleNamespace(blockage_diagnostic_root=tmp_path)
    rewards = blockage_trace(bot, "result_rewards")
    unknown = blockage_trace(bot, "unknown_overlay")
    assert rewards.lossless
    assert not unknown.lossless
    assert blockage_trace(bot, "result_rewards") is rewards
    original = pixels(106)
    assert rewards.record(original, state="results", action="ocr", attempt=1, phase="before")
    assert_pixels(rewards, rows(rewards)[0], original)


def test_health_change_diagnostics_save_both_original_colour_frames(tmp_path):
    before, after = pixels(107), pixels(108)
    bot = SimpleNamespace(blockage_diagnostic_root=tmp_path, _health_reading_trace=None,
                          _cached_health=1.0, _health_observed_at=100, tick=3,
                          last_analysis=SimpleNamespace(image=before, in_battle=True, hud_layout={}))
    BattleBot._record_health_transition(bot, after, .003, 104, {})
    trace = bot._health_reading_trace
    retained = rows(trace)
    assert [row["phase"] for row in retained] == ["before", "after"]
    for row, expected in zip(retained, (before, after)):
        assert_pixels(trace, row, expected)


def test_torpedo_diagnostics_use_lossless_pixels_for_fan_threshold_replay(tmp_path):
    bot = SimpleNamespace(_torpedo_trace=None, _last_torpedo_attempt=2.1)

    def redirected_trace(root, label, **options):
        return BlockageTrace(tmp_path, label, **options)

    original = pixels(109)
    with patch("bot.BlockageTrace", side_effect=redirected_trace):
        BattleBot._record_torpedo_frame(bot, original, "before", selected=True)
    trace = bot._torpedo_trace
    assert_pixels(trace, rows(trace)[0], original)
    assert rows(trace)[0]["details"]["selected"] is True
