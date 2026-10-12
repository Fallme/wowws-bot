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
