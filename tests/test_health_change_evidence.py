import json
from types import SimpleNamespace

import cv2
import numpy as np

from bot import BattleBot


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
