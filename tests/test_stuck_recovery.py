from strategy.stuck_recovery import StuckRecoveryController


def test_moving_ship_does_not_trigger_recovery():
    controller = StuckRecoveryController(stationary_seconds=10)
    command = None
    for second in range(13):
        command = controller.update(second, (100 + second, 200), 1.0)
    assert command is None


def test_stationary_ship_uses_two_bounded_forward_escape_phases():
    controller = StuckRecoveryController(
        stationary_seconds=10,
        escape_turn_seconds=4,
        forward_seconds=3,
    )
    command = None
    for second in range(12):
        command = controller.update(second, (100, 200), 1.0)
    assert command is not None
    assert command.phase == "forward_escape_turn"
    assert command.throttle == 1.0

    command = controller.update(16, (100, 200), 1.0)
    assert command.phase == "forward_clear"
    assert command.throttle > 0


def test_recovery_never_generates_reverse_throttle():
    controller = StuckRecoveryController(
        stationary_seconds=8,
        escape_turn_seconds=4,
        forward_seconds=3,
    )
    commands = []
    for second in range(20):
        command = controller.update(second, (100, 200), 1.0)
        if command is not None:
            commands.append(command)

    assert commands
    assert all(command.throttle >= 0 for command in commands)


def test_missing_position_clears_stationary_evidence():
    controller = StuckRecoveryController(stationary_seconds=10)
    for second in range(9):
        controller.update(second, (100, 200), 1.0)
    controller.update(9, None, 1.0)
    assert controller.update(11, (100, 200), 1.0) is None


def test_recovery_uses_live_clearance_side_when_available():
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
    assert command.rudder == -1


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
    assert command.phase == "forward_escape_turn"
    assert command.throttle == 1.0


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
