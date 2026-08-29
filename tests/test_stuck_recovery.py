from strategy.stuck_recovery import StuckRecoveryController


def test_moving_ship_does_not_trigger_recovery():
    controller = StuckRecoveryController(stationary_seconds=10)
    command = None
    for second in range(13):
        command = controller.update(second, (100 + second, 200), 1.0)
    assert command is None


def test_stationary_ship_reverses_then_turns_forward():
    controller = StuckRecoveryController(
        stationary_seconds=10,
        reverse_seconds=4,
        forward_seconds=3,
    )
    command = None
    for second in range(12):
        command = controller.update(second, (100, 200), 1.0)
    assert command is not None
    assert command.phase == "reverse"
    assert command.throttle == -1.0

    command = controller.update(16, (100, 200), 1.0)
    assert command.phase == "forward_turn"
    assert command.throttle > 0


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
