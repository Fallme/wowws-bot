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
            finally:
                store.close()


if __name__ == "__main__":
    unittest.main()
