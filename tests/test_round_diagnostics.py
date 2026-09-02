from types import SimpleNamespace

import bot as bot_module
from bot import BattleBot


def test_completed_round_keeps_matching_frames_events_and_log_then_prunes_old(tmp_path, monkeypatch):
    root = tmp_path / "runs"
    old = root / "run_20000101_000000"
    old.mkdir(parents=True)
    (old / "full_0000.png").write_bytes(b"old")
    monkeypatch.setattr(bot_module, "DEFAULT_DEBUG_ROOT", root)

    gamepad = SimpleNamespace(stop=lambda: None)
    bot = BattleBot(1, {"strategy": {}}, vision=object(), gamepad=gamepad)
    bot.reset(preserve_movement=True)
    current = bot._debug_dir
    (current / "full_0000.png").write_bytes(b"current")
    bot.events.publish("battle.tick", tick=0)

    completed = bot.complete_round_diagnostics(3, outcome="victory")

    assert completed == current
    assert (current / "full_0000.png").read_bytes() == b"current"
    assert (current / "events.jsonl").exists()
    assert (current / "round.log").exists()
    assert (current / "completed.json").exists()
    assert not old.exists()
    assert bot.complete_round_diagnostics(3) == current
    bot.stop(release_input=False)
