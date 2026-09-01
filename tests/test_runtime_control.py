import tempfile
import time
import unittest
from unittest.mock import patch
from pathlib import Path

from main import (
    count_quick_battle_for_plan,
    count_settled_battle_for_plan,
    lifecycle_stop_requested,
)
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

    def test_quick_battle_completion_closes_round_limited_plan_without_results(self):
        limits = RunLimits(max_rounds=2, quick_battle=True)
        started_at = time.monotonic()
        completed = 0

        completed = count_quick_battle_for_plan(
            completed,
            "quick_timeout",
            closure_confirmed=True,
        )
        self.assertEqual(completed, 1)
        self.assertFalse(limits.schedule_reached(completed, started_at))

        completed = count_quick_battle_for_plan(
            completed,
            "quick_death",
            closure_confirmed=True,
        )
        self.assertEqual(completed, 2)
        self.assertTrue(limits.schedule_reached(completed, started_at))

    def test_non_quick_completion_signal_does_not_advance_quick_counter(self):
        self.assertEqual(
            count_quick_battle_for_plan(
                3,
                "results",
                closure_confirmed=True,
            ),
            3,
        )

    def test_normal_round_counts_only_after_settlement_confirmation(self):
        self.assertEqual(
            count_settled_battle_for_plan(
                2,
                settlement_confirmed=False,
            ),
            2,
        )
        self.assertEqual(
            count_settled_battle_for_plan(
                2,
                settlement_confirmed=True,
            ),
            3,
        )

    def test_quick_exit_signal_does_not_count_until_port_closure_is_confirmed(self):
        self.assertEqual(
            count_quick_battle_for_plan(
                2,
                "quick_timeout",
                closure_confirmed=False,
            ),
            2,
        )

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

    def test_schedule_limit_waits_for_active_round_to_reach_settlement(self):
        limits = RunLimits(duration_minutes=1)
        started_at = time.monotonic() - 61

        self.assertFalse(
            lifecycle_stop_requested(
                limits,
                0,
                started_at,
                round_active=True,
            )
        )
        self.assertTrue(
            lifecycle_stop_requested(
                limits,
                0,
                started_at,
                round_active=False,
            )
        )

    def test_explicit_stop_remains_hard_interrupt_during_active_round(self):
        with tempfile.TemporaryDirectory() as directory:
            stop = Path(directory) / "stop.request"
            stop.touch()
            limits = RunLimits(stop_file=stop)

            self.assertTrue(
                lifecycle_stop_requested(
                    limits,
                    0,
                    time.monotonic(),
                    round_active=True,
                )
            )

    def test_close_game_option_is_read_from_environment(self):
        with patch.dict(
            "runtime_control.os.environ",
            {"WOWS_CLOSE_GAME_WHEN_DONE": "1"},
            clear=True,
        ):
            limits = RunLimits.from_env()

        self.assertTrue(limits.close_game_when_done)

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
