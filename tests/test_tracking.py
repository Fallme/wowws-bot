import pytest

from core.tracking import ConsecutivePointFilter, CourseHeadingFilter


def test_first_detection_is_not_immediately_confirmed():
    tracker = ConsecutivePointFilter(match_radius=10)
    assert tracker.update([(100, 100)]) == []


def test_nearby_consecutive_detection_is_confirmed():
    tracker = ConsecutivePointFilter(match_radius=10)
    tracker.update([(100, 100)])
    assert tracker.update([(106, 103)]) == [(106, 103)]


def test_large_jump_requires_reconfirmation():
    tracker = ConsecutivePointFilter(match_radius=10)
    tracker.update([(100, 100)])
    assert tracker.update([(150, 150)]) == []


def test_course_heading_uses_real_position_displacement():
    tracker = CourseHeadingFilter(minimum_travel=5)
    assert tracker.update((100, 180)) is None
    assert tracker.update((101, 177)) is None
    heading = tracker.update((102, 173))

    assert heading[0] == pytest.approx(2 / (53 ** 0.5), abs=0.05)
    assert heading[1] == pytest.approx(-7 / (53 ** 0.5), abs=0.05)


def test_course_heading_rejects_one_frame_reversal():
    tracker = CourseHeadingFilter(minimum_travel=3)
    tracker.update((100, 100))
    forward = tracker.update((100, 94))
    reversed_sample = tracker.update((100, 102))

    assert reversed_sample == forward
