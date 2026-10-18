import pytest

from core.survival import HealthLossTracker


def test_two_confirmed_decreases_over_two_seconds_are_sustained_damage():
    tracker = HealthLossTracker()
    tracker.observe(1.0, 0)
    tracker.observe(.998, 1)
    assert not tracker.sustained_loss(1)
    tracker.observe(.996, 2)
    assert tracker.sustained_loss(2)
    assert not tracker.sustained_loss(11)


def test_missing_ocr_does_not_clear_independent_damage_samples_or_extend_freshness():
    tracker = HealthLossTracker()
    tracker.observe(1.0, 0)
    tracker.observe(.998, 1)
    tracker.observe(None, 2)
    tracker.observe(.996, 3)
    assert tracker.sustained_loss(3)
    tracker.observe(None, 12)
    assert not tracker.sustained_loss(12)


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
