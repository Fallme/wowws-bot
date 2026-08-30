import tempfile
import unittest
from pathlib import Path

from control_server import ControlStore


class ControlStoreTests(unittest.TestCase):
    def test_resource_entries_are_aggregated_per_run_and_globally(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ControlStore(Path(directory) / "panel.db")
            try:
                store.create_run("run-1", "pommern", "asymmetric", "rounds", 2)
                store.add_resources(
                    "run-1",
                    1,
                    {"credits": 100000, "coal": 200, "free_xp": 50},
                    "first",
                )
                store.add_resources("run-1", 2, {"credits": 25000}, "second")

                dashboard = store.dashboard()
                self.assertEqual(dashboard["totals"]["credits"], 125000)
                self.assertEqual(dashboard["totals"]["coal"], 200)
                self.assertEqual(dashboard["runs"][0]["free_xp"], 50)

                store.upsert_auto_rewards(
                    "run-1",
                    1,
                    {"credits": 258088, "ship_xp": 897, "free_xp": 540},
                )
                # Repeated dashboard polling updates the same automatic row.
                store.upsert_auto_rewards(
                    "run-1",
                    1,
                    {"credits": 258088, "ship_xp": 897, "free_xp": 540},
                )
                dashboard = store.dashboard()
                self.assertEqual(dashboard["totals"]["ship_xp"], 897)
                self.assertEqual(dashboard["totals"]["free_xp"], 590)
                self.assertEqual(dashboard["totals"]["credits"], 383088)

                store.upsert_battle_result(
                    "run-1",
                    1,
                    "victory",
                    {"credits": 258088, "ship_xp": 897, "free_xp": 540},
                    rewards_recognized=True,
                )
                # A later poll for the same result updates it rather than
                # creating a duplicate battle in the group's history.
                store.upsert_battle_result(
                    "run-1",
                    1,
                    "victory",
                    {"credits": 258088, "ship_xp": 897, "free_xp": 540},
                    rewards_recognized=True,
                )
                dashboard = store.dashboard()
                self.assertEqual(len(dashboard["history"]), 1)
                self.assertEqual(dashboard["history"][0]["outcome"], "victory")
                self.assertTrue(dashboard["history"][0]["rewards_recognized"])

                store.create_run(
                    "run-quick",
                    "pommern",
                    "cooperative",
                    "rounds",
                    5,
                    quick_battle=True,
                )
                store.update_run_progress("run-quick", 3, 900)
                quick_run = next(
                    item
                    for item in store.dashboard()["runs"]
                    if item["id"] == "run-quick"
                )
                self.assertTrue(quick_run["quick_battle"])
                self.assertEqual(quick_run["completed_rounds"], 3)
                self.assertEqual(quick_run["credits"], 0)
            finally:
                store.close()


if __name__ == "__main__":
    unittest.main()
