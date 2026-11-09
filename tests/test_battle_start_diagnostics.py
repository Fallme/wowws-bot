import json
from types import SimpleNamespace

import numpy as np

import main
from core.battle_phase import BattlePhase


def test_stalled_battle_saves_frame_and_structured_ocr(tmp_path, monkeypatch):
    token = SimpleNamespace(text="19:42", confidence=0.93,
                            box=[(1, 2), (10, 2), (10, 8), (1, 8)])
    backend = SimpleNamespace(recognize=lambda crop: [token],
                              execution_provider="test")
    bot = SimpleNamespace(blockage_diagnostic_root=tmp_path,
                          distance_reader=SimpleNamespace(backend=backend))
    monkeypatch.setattr(main.time, "monotonic", lambda: 100.0)
    image = np.zeros((200, 400, 3), dtype=np.uint8)

    assert main.record_battle_start_diagnostic(bot, image, BattlePhase.UNKNOWN, 12)
    assert not main.record_battle_start_diagnostic(bot, image, BattlePhase.UNKNOWN, 13)
    path = next(tmp_path.rglob("observations.jsonl"))
    row = json.loads(path.read_text(encoding="utf-8"))
    assert (path.parent / row["frame"]).exists()
    assert row["details"]["ocr_tokens"][0]["text"] == "19:42"
    assert row["details"]["ocr_tokens"][0]["box"][0] == [265.0, 2.0]
    assert row["details"]["ocr_elapsed_ms"] >= 0


def test_ocr_exception_still_keeps_frame(tmp_path):
    def fail(_crop):
        raise RuntimeError("ocr unavailable")

    bot = SimpleNamespace(blockage_diagnostic_root=tmp_path,
                          distance_reader=SimpleNamespace(backend=SimpleNamespace(recognize=fail)))
    assert main.record_battle_start_diagnostic(bot, np.zeros((100, 100, 3), np.uint8),
                                               BattlePhase.UNKNOWN, 20)
    path = next(tmp_path.rglob("observations.jsonl"))
    row = json.loads(path.read_text(encoding="utf-8"))
    assert "ocr unavailable" in row["details"]["ocr_error"]
    assert (path.parent / row["frame"]).exists()
