import json
from types import SimpleNamespace

import cv2
import numpy as np

from bot import BattleBot
from unittest.mock import Mock


def test_large_numeric_health_change_keeps_prior_and_current_frame_with_crops(tmp_path):
    before = np.full((40, 64, 3), 30, np.uint8)
    after = np.full((40, 64, 3), 60, np.uint8)
    previous_roi = {"rects": {"health_text": (0, 30, 20, 40)}}
    current_roi = {"rects": {"health_text": (1, 29, 21, 39)}}
    bot = SimpleNamespace(blockage_diagnostic_root=tmp_path,
                          _health_reading_trace=None, _cached_health=1.0,
                          _health_observed_at=100, tick=3,
                          last_analysis=SimpleNamespace(image=before, in_battle=True,
                                                        hud_layout=previous_roi))
    BattleBot._record_health_transition(bot, after, .003, 104, current_roi)
    folder = bot._health_reading_trace.folder
    rows = [json.loads(line) for line in (folder / "observations.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [row["phase"] for row in rows] == ["before", "after"]
    assert rows[0]["pair_id"] == rows[1]["pair_id"]
    assert cv2.imread(str(folder / rows[0]["frame"]))[0, 0, 0] == 30
    assert cv2.imread(str(folder / rows[1]["frame"]))[0, 0, 0] == 60
    assert rows[0]["details"]["hud_layout"]["rects"]["health_text"] == [0, 30, 20, 40]
    assert rows[1]["details"]["hud_layout"]["rects"]["health_text"] == [1, 29, 21, 39]
    assert rows[1]["details"]["previous_health"] == 1.0
    assert rows[1]["details"]["health"] == .003
    assert rows[1]["details"]["previous_read_age_seconds"] == 4


def test_deferred_health_evidence_preserves_original_metadata_and_bounded_frame_pairs(tmp_path):
    before = np.full((40, 64, 3), 30, np.uint8)
    after = np.full((40, 64, 3), 60, np.uint8)
    previous_roi = {"rects": {"health_text": [0, 30, 20, 40]}}
    current_roi = {"rects": {"health_text": [1, 29, 21, 39]}}
    writer = Mock(return_value=True)
    bot = SimpleNamespace(
        blockage_diagnostic_root=tmp_path, _health_reading_trace=SimpleNamespace(record=writer),
        _cached_health=1.0, _health_observed_at=100, tick=3,
        last_analysis=SimpleNamespace(image=before, in_battle=True, hud_layout=previous_roi),
    )
    BattleBot._record_health_transition(bot, after, .8, 104, current_roi, defer=True)
    writer.assert_not_called()
    # State and layout can already belong to another frame when disk writing
    # runs. Evidence must still describe the change at observation time.
    bot._cached_health = .5
    bot._health_observed_at = 110
    bot.tick = 20
    bot.last_analysis = None
    previous_roi["rects"]["health_text"][0] = 99
    current_roi["rects"]["health_text"][0] = 99
    BattleBot._flush_health_transition_evidence(bot)
    prior, current = writer.call_args_list
    assert prior.args[0] is before
    assert current.args[0] is after
    assert prior.kwargs["attempt"] == current.kwargs["attempt"] == 4
    assert prior.kwargs["details"]["hud_layout"]["rects"]["health_text"][0] == 0
    assert current.kwargs["details"]["hud_layout"]["rects"]["health_text"][0] == 1
    assert current.kwargs["details"]["previous_health"] == 1.0
    assert current.kwargs["details"]["health"] == .8
    assert current.kwargs["details"]["previous_read_age_seconds"] == 4
    assert not bot._pending_health_evidence

    for step in range(10):
        BattleBot._record_health_transition(bot, after, .4, 110 + step, {}, defer=True)
    assert len(bot._pending_health_evidence) == 4
