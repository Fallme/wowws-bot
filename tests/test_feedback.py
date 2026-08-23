import pytest

from core.feedback import MovementFeedbackMonitor, SafetyFault


def test_observed_displacement_verifies_control():
    monitor = MovementFeedbackMonitor(timeout_seconds=10, movement_pixels=4)
    assert monitor.update(0, (100, 100), 1.0).pending
    feedback = monitor.update(2, (105, 100), 1.0)
    assert feedback.verified
    assert not feedback.pending


def test_no_displacement_trips_safety_fault():
    monitor = MovementFeedbackMonitor(timeout_seconds=5, movement_pixels=4)
    monitor.update(0, (100, 100), 1.0)
    with pytest.raises(SafetyFault, match="未观察到"):
        monitor.update(5.1, (101, 100), 1.0)


def test_missing_player_position_trips_safety_fault():
    monitor = MovementFeedbackMonitor(
        timeout_seconds=20,
        missing_timeout_seconds=3,
    )
    monitor.update(0, None, 1.0)
    with pytest.raises(SafetyFault, match="无法识别"):
        monitor.update(3.1, None, 1.0)


def test_reverse_movement_is_also_verified():
    monitor = MovementFeedbackMonitor(timeout_seconds=10, movement_pixels=4)
    assert monitor.update(0, (100, 100), -0.7).pending
    assert monitor.update(2, (95, 100), -0.7).verified
