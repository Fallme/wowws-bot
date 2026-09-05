import pytest

from core.survival import HealthLossTracker


def test_requires_three_decreases_in_four_fresh_readings():
    tracker = HealthLossTracker()
    for i, hp in enumerate([1.0, .99, .98]):
        tracker.observe(hp, i * 3)
        assert not tracker.sustained_loss(i * 3)
    tracker.observe(.97, 9)
    assert tracker.sustained_loss(9)
    assert not tracker.sustained_loss(18)


@pytest.mark.parametrize('value', [None, 0, float('nan'), .99, 1.0])
def test_missing_dead_stable_or_healing_sample_breaks_streak(value):
    tracker = HealthLossTracker()
    tracker.observe(1, 0)
    tracker.observe(.99, 3)
    tracker.observe(value, 6)
    tracker.observe(.97, 9)
    assert not tracker.sustained_loss(9)


def test_stale_gap_duplicate_timestamp_and_reset_do_not_trigger():
    tracker = HealthLossTracker()
    for hp in [1, .99, .98, .97]:
        tracker.observe(hp, 1)
    assert not tracker.sustained_loss(1)
    tracker.observe(.96, 20)
    assert len(tracker.samples) == 1
    tracker.reset()
    assert not tracker.sustained_loss(20)
