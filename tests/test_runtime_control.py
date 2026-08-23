import tempfile
import time
import unittest
from pathlib import Path

from runtime_control import RunLimits, RuntimeReporter


class RunLimitsTests(unittest.TestCase):
    def test_round_limit_stops_after_completed_rounds(self):
        limits = RunLimits(max_rounds=3)
        self.assertFalse(limits.reached(2, time.monotonic()))
        self.assertTrue(limits.reached(3, time.monotonic()))

    def test_duration_limit_uses_elapsed_monotonic_time(self):
        limits = RunLimits(duration_minutes=1)
        self.assertTrue(limits.reached(0, time.monotonic() - 61))
        self.assertTrue(limits.schedule_reached(0, time.monotonic() - 61))
        self.assertFalse(limits.stop_requested())

    def test_stop_file_requests_shutdown(self):
        with tempfile.TemporaryDirectory() as directory:
            stop = Path(directory) / "stop.request"
            limits = RunLimits(stop_file=stop)
            self.assertFalse(limits.reached(0, time.monotonic()))
            stop.touch()
            self.assertTrue(limits.reached(0, time.monotonic()))

    def test_pause_file_is_independent_from_stop(self):
        with tempfile.TemporaryDirectory() as directory:
            pause = Path(directory) / "pause.request"
            limits = RunLimits(pause_file=pause)
            self.assertFalse(limits.pause_requested())
            pause.write_text("pause", encoding="utf-8")
            self.assertTrue(limits.pause_requested())
            self.assertFalse(limits.stop_requested())

    def test_reporter_persists_status_atomically(self):
        with tempfile.TemporaryDirectory() as directory:
            state_path = Path(directory) / "state.json"
            limits = RunLimits(run_id="run-1", state_file=state_path)
            reporter = RuntimeReporter(limits, ship="pommern", mode="asymmetric")
            reporter.update("battle", "战斗进行中", current_round=2)

            payload = state_path.read_text(encoding="utf-8")
            self.assertIn('"state": "battle"', payload)
            self.assertIn('"current_round": 2', payload)


if __name__ == "__main__":
    unittest.main()
