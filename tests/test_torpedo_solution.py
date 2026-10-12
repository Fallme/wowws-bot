import math
import unittest

from strategy.torpedo_solution import solve_intercept


class InterceptTests(unittest.TestCase):
    def test_stationary_at_range_boundary(self):
        result = solve_intercept((3, 4), (0, 0), .1, 5)
        self.assertAlmostEqual(result.time_seconds, 50)

    def test_crossing_leads_target(self):
        result = solve_intercept((4, 0), (0, .03), .1, 6)
        self.assertGreater(result.aim_position_km[1], 0)
        self.assertAlmostEqual(math.hypot(*result.aim_position_km), result.travel_km)

    def test_receding_intercept_exceeds_range(self):
        self.assertIsNone(solve_intercept((5, 0), (.05, 0), .1, 6))

    def test_faster_receding_target(self):
        self.assertIsNone(solve_intercept((3, 0), (.2, 0), .1, 6))

    def test_equal_speed_approaching(self):
        result = solve_intercept((4, 0), (-.1, 0), .1, 6)
        self.assertAlmostEqual(result.time_seconds, 20)

    def test_invalid_observations(self):
        for position in [(float('nan'), 0), (float('inf'), 0), (0, 0), None]:
            self.assertIsNone(solve_intercept(position, (0, 0), .1, 6))
        self.assertIsNone(solve_intercept((2, 0), (0, 0), 0, 6))


if __name__ == '__main__':
    unittest.main()
