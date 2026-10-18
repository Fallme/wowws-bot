import json
from unittest.mock import patch

from control_server import ControlStore, RunnerManager
from runtime_control import RunLimits, RuntimeReporter


def test_rewards_survive_missed_polls_and_replay_without_duplicates(tmp_path):
    path = tmp_path / "state.json"
    reporter = RuntimeReporter(
        RunLimits(run_id="ledger", state_file=path),
        ship="pommern", mode="cooperative",
    )
    for round_no, credits in [(1, 41013), (2, 204484)]:
        reporter.update(
            "returning", "settled", rewards_status="recognized",
            rewards_round=round_no, last_rewards={"credits": credits},
            last_outcome="victory",
        )
        reporter.update("battle", "next", rewards_status="pending", last_rewards={})
    snapshot = json.loads(path.read_text(encoding="utf-8"))
    store = ControlStore(tmp_path / "test.db")
    store.create_run("ledger", "pommern", "cooperative", "rounds", 2)
    manager = RunnerManager(store)
    with patch.object(manager, "_read_runtime_state", return_value=snapshot):
        manager.status()
        manager.status()
    assert store.dashboard()["totals"]["credits"] == 245497
    assert store.dashboard()["totals"]["entries"] == 2


def test_replayed_ledger_keeps_original_timestamp_and_does_not_inflate_ids(tmp_path):
    store = ControlStore(tmp_path / "test.db")
    store.create_run("replay", "pommern", "cooperative", "rounds", 5)
    entry = store.upsert_battle_result(
        "replay", 1, "victory",
        {"credits": 100, "ship_xp": 10, "free_xp": 1},
        rewards_recognized=True,
    )
    store.upsert_auto_rewards(
        "replay", 1, {"credits": 100, "ship_xp": 10, "free_xp": 1},
    )
    with store.lock:
        original = store.connection.execute(
            "SELECT created_at FROM battle_history WHERE id=?", (entry,),
        ).fetchone()[0]

    for _ in range(50):
        store.upsert_battle_result(
            "replay", 1, "victory",
            {"credits": 100, "ship_xp": 10, "free_xp": 1},
            rewards_recognized=True,
        )
        store.upsert_auto_rewards(
            "replay", 1, {"credits": 100, "ship_xp": 10, "free_xp": 1},
        )

    with store.lock:
        current = store.connection.execute(
            "SELECT created_at FROM battle_history WHERE id=?", (entry,),
        ).fetchone()[0]
        rows = store.connection.execute(
            "SELECT COUNT(*) FROM battle_history WHERE run_id='replay'",
        ).fetchone()[0]
        sequence = store.connection.execute(
            "SELECT seq FROM sqlite_sequence WHERE name='battle_history'",
        ).fetchone()[0]
    assert current == original
    assert rows == 1
    assert sequence == entry


def test_startup_repair_reconstructs_the_round_count_from_settlements(tmp_path):
    database = tmp_path / "test.db"
    store = ControlStore(database)
    store.create_run("repair", "pommern", "cooperative", "rounds", 10)
    store.upsert_battle_result(
        "repair", 1, "victory", {"credits": 1, "ship_xp": 1},
        rewards_recognized=True,
    )
    store.upsert_battle_result(
        "repair", 2, "defeat", {"credits": 2, "ship_xp": 2},
        rewards_recognized=True,
    )
    with store.lock, store.connection:
        store.connection.execute(
            "UPDATE runs SET completed_rounds=0 WHERE id='repair'",
        )
    store.close()

    reopened = ControlStore(database)
    with reopened.lock:
        completed = reopened.connection.execute(
            "SELECT completed_rounds FROM runs WHERE id='repair'",
        ).fetchone()[0]
    assert completed == 2
