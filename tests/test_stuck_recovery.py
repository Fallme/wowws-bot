from strategy.stuck_recovery import StuckRecoveryController


def test_moving_ship_does_not_trigger_recovery():
    controller = StuckRecoveryController(stationary_seconds=10)
    command = None
    for second in range(13):
        command = controller.update(second, (100 + second, 200), 1.0)
    assert command is None


def test_stationary_ship_reverses_then_requests_replan_and_cooldown():
    controller = StuckRecoveryController(
        stationary_seconds=10,
        reverse_seconds=4,
    )
    command = None
    for second in range(12):
        command = controller.update(second, (100, 200), 1.0)
    assert command is not None
    assert command.phase == "reverse_clear"
    assert command.throttle == -1.0

    command = controller.update(14, (100, 200), 1.0)
    assert command.phase == "replan"
    assert command.throttle == 0
    assert controller.update(15, (100, 200), 1.0) is None


def test_recovery_never_generates_blind_forward_throttle():
    controller = StuckRecoveryController(
        stationary_seconds=8,
        reverse_seconds=4,
    )
    commands = []
    for second in range(20):
        command = controller.update(second, (100, 200), 1.0)
        if command is not None:
            commands.append(command)

    assert commands
    assert commands[0].phase == "reverse_clear"
    assert commands[-1].phase == "replan"
    assert all(command.throttle <= 0 for command in commands)


def test_missing_position_clears_stationary_evidence():
    controller = StuckRecoveryController(stationary_seconds=10)
    for second in range(9):
        controller.update(second, (100, 200), 1.0)
    controller.update(9, None, 1.0)
    assert controller.update(11, (100, 200), 1.0) is None


def test_observation_gap_does_not_count_as_sustained_low_speed():
    controller = StuckRecoveryController(low_speed_seconds=8)
    for second in (0, 1, 2, 20, 21, 22):
        assert controller.update(second, (100, 200), 1.0, speed_knots=.4) is None


def test_slow_but_continuous_capture_still_detects_collision():
    controller = StuckRecoveryController(low_speed_seconds=8)
    assert controller.update(0, (100, 200), 1.0, speed_knots=.4) is None
    assert controller.update(4, (100, 200), 1.0, speed_knots=.4) is None
    assert controller.update(8, (100, 200), 1.0, speed_knots=.4).phase == "reverse_clear"


def test_reverse_uses_neutral_rudder_even_when_previous_route_wanted_a_turn():
    controller = StuckRecoveryController(stationary_seconds=10)
    command = None
    for second in range(12):
        command = controller.update(
            second,
            (100, 200),
            1.0,
            escape_rudder=-0.8,
        )
    assert command is not None
    assert command.rudder == 0
    assert command.throttle == -1


def test_sustained_low_speed_triggers_even_when_marker_slowly_drifts():
    controller = StuckRecoveryController(
        stationary_seconds=30,
        stationary_pixels=2,
        low_speed_seconds=8,
        low_speed_knots=1.5,
    )
    command = None
    for second in range(10):
        # More than the stationary-pixel budget, matching a ship that slides
        # along terrain at the observed 0.4-0.6 kt.
        command = controller.update(
            second,
            (100 + second, 200),
            1.0,
            speed_knots=0.6,
        )

    assert command is not None
    assert command.phase == "reverse_clear"
    assert command.throttle == -1.0


def test_normal_acceleration_clears_low_speed_stall_timer():
    controller = StuckRecoveryController(
        stationary_seconds=30,
        low_speed_seconds=8,
        low_speed_knots=1.5,
    )
    commands = []
    speeds = [0.0, 0.4, 0.9, 1.6, 2.4, 4.0, 6.0, 8.0, 10.0]
    for second, speed in enumerate(speeds):
        commands.append(
            controller.update(
                second,
                (100 + second * 2, 200),
                1.0,
                speed_knots=speed,
            )
        )

    assert all(command is None for command in commands)


def test_low_throttle_or_missing_speed_cannot_create_false_low_speed_stall():
    controller = StuckRecoveryController(
        stationary_seconds=30,
        low_speed_seconds=5,
    )
    for second in range(8):
        assert (
            controller.update(
                second,
                (100 + second, 200),
                0.0,
                speed_knots=0.0,
            )
            is None
        )
    for second in range(8, 16):
        assert (
            controller.update(
                second,
                (100 + second, 200),
                1.0,
                speed_knots=None,
            )
            is None
        )
