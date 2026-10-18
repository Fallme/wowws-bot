import os

import bot as bot_module


def test_prune_keeps_newest_completed_runs_and_pending(tmp_path):
    for index in range(5):
        run = tmp_path / f"run_{index}"
        run.mkdir()
        (run / "completed.json").write_text("{}", encoding="utf-8")
        stamp = 1000 + index
        os.utime(run, (stamp, stamp))
    pending = tmp_path / "pending"
    pending.mkdir()
    (pending / "completed.json").write_text("{}", encoding="utf-8")
    incomplete = tmp_path / "run_live"
    incomplete.mkdir()

    removed = bot_module.prune_debug_runs(tmp_path, 2)

    assert removed == 3
    assert sorted(entry.name for entry in tmp_path.iterdir()) == [
        "pending", "run_3", "run_4", "run_live",
    ]


def test_prune_is_a_no_op_within_the_retention_cap(tmp_path):
    run = tmp_path / "run_0"
    run.mkdir()
    (run / "completed.json").write_text("{}", encoding="utf-8")

    assert bot_module.prune_debug_runs(tmp_path, 5) == 0
    assert run.exists()


def test_retention_limit_follows_the_environment(monkeypatch):
    monkeypatch.setenv("WOWS_DEBUG_KEEP_RUNS", "7")
    assert bot_module.debug_retention_limit() == 7
    monkeypatch.setenv("WOWS_DEBUG_KEEP_RUNS", "not-a-number")
    assert bot_module.debug_retention_limit() == bot_module.DEFAULT_KEEP_DEBUG_RUNS
