import unittest

from strategy.secondary_movement import (
    MovementMode,
    SecondaryMovementController,
    SecondaryMovementInput,
)


class SecondaryMovementTests(unittest.TestCase):
    def setUp(self):
        self.controller = SecondaryMovementController(preferred_side=1)

    def plan(self, **overrides):
        values = {
            "elapsed": 80,
            "health": 1.0,
            "visible_target": False,
            "enemy_count": 0,
            "map_center_bearing": 0.0,
            "map_center_distance_km": 20.0,
            "capture_point_bearing": 0.0,
            "capture_point_distance_km": 18.0,
        }
        values.update(overrides)
        return self.controller.plan(SecondaryMovementInput(**values))

    def test_outside_capture_point_always_uses_full_speed(self):
        command = self.plan(elapsed=5, capture_point_bearing=0.22)

        self.assertEqual(command.mode, MovementMode.ROUTE_PLANNING)
        self.assertEqual(command.throttle, 1.0)
        self.assertEqual(command.rudder, 0.0)

    def test_inside_capture_point_keeps_searching_instead_of_loitering(self):
        command = self.plan(
            inside_capture_point=True,
            capture_point_distance_km=1.0,
            capture_point_bearing=-0.2,
        )

        self.assertEqual(command.mode, MovementMode.CAPTURE)
        self.assertGreaterEqual(command.throttle, 0.7)
        self.assertLess(command.rudder, 0)

    def test_enemy_inside_secondary_range_does_not_cause_turn_away(self):
        command = self.plan(
            target_distance_km=8.0,
            minimap_distance_km=8.0,
            minimap_target_bearing=0.9,
            capture_point_bearing=-0.3,
            health=0.12,
            route_arrived=True,
            inside_capture_point=True,
        )

        self.assertEqual(command.mode, MovementMode.BRAWL)
        self.assertGreater(command.throttle, 0)
        self.assertLess(command.rudder, 0)

    def test_very_close_enemy_does_not_reverse_outside_the_point(self):
        command = self.plan(
            target_distance_km=2.5,
            minimap_target_bearing=0.8,
            capture_point_bearing=0.1,
        )

        self.assertEqual(command.throttle, 1.0)
        self.assertGreaterEqual(command.rudder, 0)

    def test_forward_far_enemy_is_prioritized_over_central_route(self):
        command = self.plan(
            target_distance_km=18.0,
            minimap_distance_km=18.0,
            minimap_target_bearing=0.35,
            capture_point_bearing=-0.4,
            enemy_count=1,
            route_arrived=True,
            inside_capture_point=True,
        )

        self.assertEqual(command.mode, MovementMode.APPROACH)
        self.assertEqual(command.throttle, 1.0)
        self.assertGreater(command.rudder, 0)

    def test_minimap_five_kilometre_grid_is_distance_fallback(self):
        command = self.plan(
            target_distance_km=None,
            minimap_distance_km=15.0,
            minimap_target_bearing=0.2,
            capture_point_bearing=0.0,
            route_arrived=True,
            inside_capture_point=True,
        )

        self.assertEqual(command.mode, MovementMode.APPROACH)
        self.assertEqual(command.throttle, 1.0)
        self.assertIn("小地图5km网格15.0km", command.reason)

    def test_minimap_grid_overrides_untrusted_viewport_ocr(self):
        command = self.plan(
            target_distance_km=9.0,
            minimap_distance_km=20.0,
            minimap_target_bearing=0.35,
            capture_point_bearing=-0.2,
            route_arrived=True,
            inside_capture_point=True,
        )

        self.assertEqual(command.mode, MovementMode.APPROACH)
        self.assertIn("小地图5km网格20.0km", command.reason)
        self.assertGreater(command.rudder, 0)

    def test_viewport_ocr_alone_never_controls_distance(self):
        command = self.plan(
            target_distance_km=4.0,
            minimap_distance_km=None,
            capture_point_bearing=0.0,
            route_arrived=True,
            inside_capture_point=True,
        )

        self.assertEqual(command.mode, MovementMode.CAPTURE)
        self.assertNotIn("4.0km", command.reason)

    def test_viewport_bearing_cannot_change_rudder_without_minimap_enemy(self):
        command = self.plan(
            target_offset_x=1.0,
            minimap_target_bearing=None,
            minimap_distance_km=16.0,
            capture_point_bearing=-0.25,
            route_arrived=True,
            inside_capture_point=True,
        )

        self.assertEqual(command.mode, MovementMode.APPROACH)
        self.assertLess(command.rudder, 0)

    def test_enemy_death_resumes_objective_when_ship_is_outside_point(self):
        command = self.plan(
            minimap_distance_km=None,
            capture_point_bearing=0.25,
            capture_point_distance_km=6.0,
            route_arrived=True,
            inside_capture_point=False,
            enemy_count=0,
        )

        self.assertEqual(command.mode, MovementMode.ROUTE_TRANSIT)
        self.assertEqual(command.throttle, 1.0)
        self.assertGreater(command.rudder, 0)

    def test_low_health_no_longer_triggers_disengage(self):
        command = self.plan(
            health=0.05,
            target_distance_km=7.0,
            capture_point_bearing=0.25,
        )

        self.assertNotEqual(command.mode, MovementMode.DISENGAGE)
        self.assertEqual(command.throttle, 1.0)
        self.assertGreater(command.rudder, 0)

    def test_island_warning_steers_but_keeps_full_speed_outside_point(self):
        command = self.plan(
            island_distance=0.035,
            island_avoidance_rudder=-1,
        )

        self.assertEqual(command.mode, MovementMode.AVOID_ISLAND)
        self.assertEqual(command.throttle, 1.0)
        self.assertLess(command.rudder, -0.7)

    def test_emergency_island_clearance_stays_forward(self):
        command = self.plan(
            torpedoes_incoming=True,
            island_distance=0.015,
            island_avoidance_rudder=1,
        )

        self.assertEqual(command.mode, MovementMode.AVOID_ISLAND)
        self.assertGreater(command.throttle, 0)

    def test_opening_first_establishes_straight_course(self):
        command = self.plan(
            elapsed=3,
            capture_point_bearing=0.9,
            island_distance=0.01,
            island_avoidance_rudder=-1,
        )
        self.assertEqual(command.mode, MovementMode.ROUTE_PLANNING)
        self.assertEqual(command.throttle, 1.0)
        self.assertEqual(command.rudder, 0.0)

    def test_forward_enemy_biases_route_before_arrival_without_uturn(self):
        command = self.plan(
            route_phase="transit",
            route_arrived=False,
            target_distance_km=4.0,
            minimap_target_bearing=0.35,
            capture_point_bearing=-0.25,
        )

        self.assertEqual(command.mode, MovementMode.ROUTE_TRANSIT)
        self.assertEqual(command.throttle, 1.0)
        self.assertGreater(command.rudder, 0)

    def test_rearward_objective_ignores_forward_contact_and_forces_uturn(self):
        command = self.plan(
            route_phase="transit",
            route_arrived=False,
            minimap_target_bearing=-0.10,
            capture_point_bearing=0.90,
        )

        self.assertEqual(command.mode, MovementMode.ROUTE_TRANSIT)
        self.assertEqual(command.throttle, 1.0)
        self.assertGreater(command.rudder, 0.8)
        self.assertIn("舰首背离中央点", command.reason)

    def test_uturn_side_stays_fixed_across_pi_sign_flip(self):
        first = self.plan(
            route_phase="transit",
            capture_point_bearing=0.98,
        )
        second = self.plan(
            route_phase="transit",
            capture_point_bearing=-0.98,
        )

        self.assertGreater(first.rudder, 0.8)
        self.assertGreater(second.rudder, 0.7)

    def test_enemy_bias_returns_only_after_objective_is_ahead(self):
        self.plan(route_phase="transit", capture_point_bearing=0.90)
        command = self.plan(
            route_phase="transit",
            capture_point_bearing=0.10,
            minimap_target_bearing=-0.30,
        )

        self.assertLess(command.rudder, 0)
        self.assertNotIn("舰首背离中央点", command.reason)

    def test_rear_enemy_never_causes_about_turn(self):
        command = self.plan(
            route_phase="transit",
            route_arrived=False,
            minimap_distance_km=18.0,
            minimap_target_bearing=0.82,
            capture_point_bearing=-0.25,
            enemy_count=1,
        )

        self.assertEqual(command.mode, MovementMode.ROUTE_TRANSIT)
        self.assertLess(command.rudder, 0)

    def test_close_enemy_never_triggers_reverse_after_reaching_point(self):
        command = self.plan(
            route_arrived=True,
            inside_capture_point=True,
            target_distance_km=5.5,
            minimap_distance_km=5.5,
            minimap_target_bearing=0.4,
        )

        self.assertEqual(command.mode, MovementMode.BRAWL)
        self.assertGreaterEqual(command.throttle, 0.6)

    def test_torpedo_evasion_keeps_full_speed(self):
        command = self.plan(torpedoes_incoming=True)

        self.assertEqual(command.mode, MovementMode.EVADE)
        self.assertEqual(command.throttle, 1.0)
        self.assertGreater(abs(command.rudder), 0.7)


if __name__ == "__main__":
    unittest.main()
