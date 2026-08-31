import pytest

from core.tracking import ArrowHeadingFilter, ConsecutivePointFilter, CourseHeadingFilter


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


def test_arrow_heading_uses_visible_vector_without_position_history():
    tracker = ArrowHeadingFilter(blend=1.0)

    assert tracker.update((3, -4)) == pytest.approx((0.6, -0.8))


def test_arrow_heading_rejects_one_frame_stern_flip():
    tracker = ArrowHeadingFilter(blend=1.0)
    forward = tracker.update((0, -1))

    assert tracker.update((0, 1)) == forward
    assert tracker.update((0, 1))[1] > 0.99


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


def test_course_heading_accepts_a_persistent_completed_turn():
    tracker = CourseHeadingFilter(minimum_travel=3)
    tracker.update((100, 120))
    forward = tracker.update((100, 114))

    # Three consistent observations in the opposite direction represent a
    # genuine turn, not a one-frame player-marker glitch.
    tracker.update((100, 121))
    tracker.update((100, 128))
    turned = tracker.update((100, 135))

    assert forward[1] < 0
    assert turned[1] > 0.9


def test_course_heading_uses_recent_motion_instead_of_old_loop_origin():
    tracker = CourseHeadingFilter(minimum_travel=4)
    for point in ((100, 100), (108, 96), (116, 98), (119, 106), (115, 114)):
        heading = tracker.update(point)

    # The latest leg is down-left.  An oldest-origin estimator would still
    # report right/down and keep a slow ship circling.
    assert heading[0] < 0
    assert heading[1] > 0
