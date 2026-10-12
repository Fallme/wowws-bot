from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from core.tactical_geometry import TacticalMapGeometry, detect_tactical_map_geometry
import core.tactical_geometry as tactical_geometry
from core.frame_guard import FrameGuard
from core.ui import ScreenState
from core.vision import PlayerPose, Vision
from main import configure_opening_autopilot, opening_autopilot_target, tactical_map_is_open


def grid_frame(shape, corners):
    frame = np.full((*shape, 3), 65, dtype=np.uint8)
    transform = cv2.getPerspectiveTransform(
        np.float32([[0, 0], [1, 0], [1, 1], [0, 1]]), np.float32(corners),
    )
    for fraction in np.linspace(0, 1, 11):
        for line in (((0, fraction), (1, fraction)), ((fraction, 0), (fraction, 1))):
            endpoints = cv2.perspectiveTransform(np.float32([line]), transform)[0]
            cv2.line(frame, tuple(np.rint(endpoints[0]).astype(int)),
                     tuple(np.rint(endpoints[1]).astype(int)), (145, 145, 145), 2)
    return frame


@pytest.mark.parametrize("shape,corners", [
    ((768, 1024), ((180, 80), (780, 80), (795, 695), (165, 695))),
    ((1000, 1600), ((220, 190), (610, 190), (628, 610), (202, 610))),
    ((1440, 2560), ((1300, 170), (1830, 170), (1848, 730), (1282, 730))),
    ((2160, 3840), ((580, 130), (2400, 130), (2460, 2000), (520, 2000))),
])
def test_shifted_independently_scaled_grid_has_measured_perspective(shape, corners):
    frame = grid_frame(shape, corners)
    geometry = detect_tactical_map_geometry(frame)

    assert geometry is not None
    assert geometry.frame_shape == shape
    assert np.asarray(geometry.corners) == pytest.approx(np.asarray(corners), abs=5)
    expected = TacticalMapGeometry(shape, corners, 500)
    assert geometry.normalized_to_pixel((.72, .83)) == pytest.approx(
        expected.normalized_to_pixel((.72, .83)), abs=4,
    )
    assert geometry.rectify(frame).shape == (geometry.size, geometry.size, 3)
    assert tactical_map_is_open(SimpleNamespace(), frame)


def test_native_live_grid_matches_visible_outer_corners():
    fixture = Path(__file__).parents[1] / "training_assets/user_captures/codex-clipboard-68c91356-07a4-4753-a25a-0824b8b62c6f.png"
    if not fixture.exists():
        pytest.skip("Optional tactical-map training image unavailable")
    frame = cv2.imread(str(fixture))
    geometry = detect_tactical_map_geometry(frame)

    assert geometry is not None
    assert 1240 <= geometry.size <= 1340
    assert np.asarray(geometry.corners) == pytest.approx(np.asarray(
        ((715, 112), (1840, 112), (1880, 1402), (673, 1402)),
    ), abs=10)
    assert geometry.matched_rows >= 9 and geometry.matched_columns >= 9


def test_coast_edges_or_only_a_resident_minimap_do_not_authorize_m_map():
    empty = np.zeros((1000, 1600, 3), dtype=np.uint8)
    for value in (120, 211, 297, 402, 578, 701, 849):
        cv2.line(empty, (100, value), (1300, value), (200, 200, 200), 2)
    resident = grid_frame((1000, 1600), ((1110, 510), (1590, 510), (1590, 990), (1110, 990)))

    assert detect_tactical_map_geometry(empty) is None
    assert detect_tactical_map_geometry(resident) is None
    assert not tactical_map_is_open(SimpleNamespace(), resident)


def make_bot(frames, shape=(1000, 1600)):
    pending = iter(frames)
    blank = np.zeros((*shape, 3), dtype=np.uint8)
    calls = []
    toggles = []
    enabled = []

    def grab(_hwnd, *, allow_stale=False):
        calls.append(allow_stale)
        return next(pending, blank)

    vision = SimpleNamespace(
        grab=grab,
        classify_screen=lambda _image: ScreenState.BATTLE,
        find_minimap=lambda _image: np.zeros((200, 200, 3), dtype=np.uint8),
        find_player_pose_on_minimap=lambda _image: PlayerPose((50, 100), (1, 0)),
    )
    bot = SimpleNamespace(
        hwnd=1, vision=vision, intervention=None,
        gamepad=SimpleNamespace(toggle_tactical_map=lambda: toggles.append("m")),
        enable_opening_autopilot=lambda *_args, **_kwargs: enabled.append(True),
    )
    return bot, blank, calls, toggles, enabled


def test_waypoint_uses_last_fresh_geometry_and_refreshed_client_origin():
    first = grid_frame((1000, 1600), ((370, 100), (1150, 100), (1180, 900), (340, 900)))
    latest = grid_frame((1000, 1600), ((190, 200), (590, 200), (610, 640), (170, 640)))
    blank = np.zeros((1000, 1600, 3), dtype=np.uint8)
    bot, _, calls, toggles, enabled = make_bot([blank, *[first] * 4, latest, blank])
    expected = detect_tactical_map_geometry(latest).normalized_to_pixel(
        opening_autopilot_target((.25, .5)),
    )
    with (
        patch("main.time.sleep"),
        patch("main.get_client_rect", side_effect=[
            {"left": 10, "top": 20, "width": 1600, "height": 1000},
            {"left": 130, "top": 70, "width": 1600, "height": 1000},
        ]),
        patch("main.physical_click", return_value=True) as click,
    ):
        assert configure_opening_autopilot(bot)
        assert not configure_opening_autopilot(bot)

    assert click.call_args.args == (expected[0] + 130, expected[1] + 70)
    assert calls[1:6] == [True] * 5
    assert toggles == ["m", "m"] and enabled == [True]


@pytest.mark.parametrize("reason", ["last_grid_missing", "frame_resized", "client_resized"])
def test_unmeasurable_or_resized_latest_map_never_dispatches_a_click(reason):
    frame = grid_frame((1000, 1600), ((370, 100), (1150, 100), (1180, 900), (340, 900)))
    blank = np.zeros_like(frame)
    latest = (blank if reason == "last_grid_missing" else
              cv2.resize(frame, (1280, 800)) if reason == "frame_resized" else frame)
    bot, _, _, _, enabled = make_bot([blank, *[frame] * 4, latest, blank])
    rect = {"left": 0, "top": 0, "width": 1280 if reason == "client_resized" else 1600,
            "height": 800 if reason == "client_resized" else 1000}
    with (
        patch("main.time.sleep"),
        patch("main.get_client_rect", return_value=rect),
        patch("main.physical_click") as physical,
        patch("main.window_message_click") as message,
    ):
        assert not configure_opening_autopilot(bot)

    physical.assert_not_called()
    message.assert_not_called()
    assert enabled == []
    assert bot._tactical_map_attempted_this_battle


def test_grid_geometry_rejects_reuse_after_capture_resize():
    geometry = TacticalMapGeometry((1000, 1600), ((100, 100), (800, 100), (800, 800), (100, 800)), 700)
    with pytest.raises(ValueError, match="different frame size"):
        geometry.rectify(np.zeros((800, 1280, 3), dtype=np.uint8))


def test_static_m_samples_are_five_new_captures_with_real_frame_guard():
    tactical = grid_frame((1000, 1600), ((370, 100), (1150, 100), (1180, 900), (340, 900)))
    battle = np.full_like(tactical, 65)
    battle[100:400, 100:400] = 120
    frames = iter([battle, *[tactical] * 5, battle])
    captures = []

    def capture(_hwnd):
        frame = next(frames).copy()
        captures.append(frame)
        return frame

    guard = FrameGuard(stale_after=1)
    original_inspect = guard.inspect
    # Advance enough time that the repeated M contents are definitely stale.
    guard.inspect = lambda image: original_inspect(image, now=len(captures) * 10)
    vision = Vision(screen_capture=SimpleNamespace(capture_window=capture), frame_guard=guard)
    vision.classify_screen = lambda _image: ScreenState.BATTLE
    vision.find_minimap = lambda _image: np.zeros((200, 200, 3), dtype=np.uint8)
    vision.find_player_pose_on_minimap = lambda _image: PlayerPose((50, 100), (1, 0))
    vision.read_battle_clock_seconds = None
    vision.read_autopilot_enabled_text = None
    bot = SimpleNamespace(
        hwnd=1, vision=vision, intervention=None,
        gamepad=SimpleNamespace(toggle_tactical_map=lambda: None),
        enable_opening_autopilot=lambda *_args, **_kwargs: None,
    )
    with (
        patch("main.time.sleep"),
        patch("main.get_client_rect", return_value={"left": 0, "top": 0, "width": 1600, "height": 1000}),
        patch("main.physical_click", return_value=True),
    ):
        assert configure_opening_autopilot(bot)

    assert len(captures) == 7
    assert len({id(image) for image in captures[1:6]}) == 5
    assert np.array_equal(captures[1], captures[5])


def test_repeated_geometry_checks_share_hough_but_mutated_frame_is_remeasured():
    frame = grid_frame((1000, 1600), ((370, 100), (1150, 100), (1180, 900), (340, 900)))
    original_hough = cv2.HoughLinesP
    with patch("core.tactical_geometry.cv2.HoughLinesP", wraps=original_hough) as hough:
        first = detect_tactical_map_geometry(frame)
        assert first is not None
        assert detect_tactical_map_geometry(frame) is first
        assert hough.call_count == 1

        frame[:] = 65
        assert detect_tactical_map_geometry(frame) is None
        assert hough.call_count == 2


def test_dense_hough_axis_noise_is_rejected_before_combinatorial_grid_search():
    frame = np.full((1400, 1400, 3), 60, np.uint8)
    for y in range(12, 1388, 5):
        cv2.line(frame, (10, y), (1100, y), (190, 190, 190), 1)
    for x in range(1190, 1350, 20):
        cv2.line(frame, (x, 12), (x, 1388), (190, 190, 190), 1)
    with patch("core.tactical_geometry._axes", wraps=tactical_geometry._axes) as axes:
        assert detect_tactical_map_geometry(frame) is None

    axes.assert_not_called()


def test_per_axis_evaluation_budget_rejects_dense_models_before_distance_fitting():
    models = [(0, float(value), float(value), 0, 1400)
              for value in np.linspace(0, 1000, tactical_geometry.MAX_AXIS_MODELS)]
    with patch("core.tactical_geometry.np.take_along_axis") as distances:
        assert tactical_geometry._axes(models, 0, 1001, perspective=True) == []

    distances.assert_not_called()


def test_dense_unrelated_lines_fail_closed_before_the_expensive_grid_fit():
    import core.tactical_geometry as module
    frame = np.full((1400, 1400, 3), 65, dtype=np.uint8)
    for y in range(12, 1388, 5):
        cv2.line(frame, (10, y), (1100, y), (140, 140, 140), 1)
    for x in range(1190, 1350, 20):
        cv2.line(frame, (x, 10), (x, 1380), (140, 140, 140), 1)
    with patch.object(module, "_axes", wraps=module._axes) as fitting:
        assert module.detect_tactical_map_geometry(frame) is None
        fitting.assert_not_called()


def test_axis_work_budget_rejects_ambiguous_model_sets():
    import core.tactical_geometry as module
    models = [(0, index * 10, index * 10, 0, 700) for index in range(65)]
    assert module._axes(models, 100, 700, perspective=True) == []
    with patch.object(module, "MAX_AXIS_EVALUATIONS", 1):
        assert module._axes(models[:11], 50, 100, perspective=True) == []
